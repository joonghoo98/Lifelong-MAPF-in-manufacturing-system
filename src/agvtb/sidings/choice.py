"""Siding variants as short texts, and the candidate sidings each one picks on a loaded map.

The texts: grid-<mode>-s<dr><dc> (the grid rule, mode in GRID_MODES, origin shift dr, dc in {0, 1}), strict (the
strict rule), strict-drop:<id>,<id>,... (the strict rule less those ids), recorded (the map file's info.sidings)
and list:<id>,<id>,... (those ids). The first two are the variant names of M2's table. User decision G3
(2026-10-21) puts variant (ga) into the research maps' configs under the key "sidings".
"""
import re
from dataclasses import dataclass

from agvtb.sidings.grid_rule import GRID_MODES
from agvtb.sidings.lattice import grid_sidings_of_world
from agvtb.sidings.strict_rule import strict_sidings_of_world
from agvtb.world.loader import LoadedWorld

_FORMS = "grid-<mode>-s<dr><dc>, strict, strict-drop:<id>,..., recorded or list:<id>,..."
_GRID = re.compile(r"grid-([a-z]+)-s([01])([01])")


@dataclass(frozen=True)
class SidingChoice:
    """One siding variant; rule is "grid", "strict", "recorded" or "list"."""

    rule: str
    mode: str | None = None  # grid: one of GRID_MODES
    shift: tuple[int, int] | None = None  # grid: the origin shift (dr, dc)
    drop: tuple[str, ...] = ()  # strict: ids left out
    nodes: tuple[str, ...] = ()  # list: the ids, in the order given

    @property
    def name(self) -> str:
        """The text of this variant: parse_siding_choice(choice.name) == choice."""
        if self.rule == "grid":
            return f"grid-{self.mode}-s{self.shift[0]}{self.shift[1]}"
        if self.rule == "strict" and self.drop:
            return "strict-drop:" + ",".join(self.drop)
        if self.rule == "list":
            return "list:" + ",".join(self.nodes)
        return self.rule


def parse_siding_choice(text: str) -> SidingChoice:
    """The variant a text names. Raises ValueError, quoting the text and the forms it takes, if it names none."""
    if text in ("strict", "recorded"):
        return SidingChoice(text)
    grid = _GRID.fullmatch(text)
    if grid and grid.group(1) in GRID_MODES:
        return SidingChoice("grid", mode=grid.group(1), shift=(int(grid.group(2)), int(grid.group(3))))
    for prefix, rule in (("strict-drop:", "strict"), ("list:", "list")):
        if text.startswith(prefix):
            ids = tuple(text[len(prefix):].split(","))
            if all(ids):
                return SidingChoice(rule, drop=ids) if rule == "strict" else SidingChoice(rule, nodes=ids)
    raise ValueError(f"unknown siding choice {text!r}; use {_FORMS}")


def candidates_of(loaded: LoadedWorld, choice: SidingChoice) -> list[str]:
    """The candidate sidings choice picks on the loaded map, as sorted distinct node ids.

    grid raises LatticeError on a map that is no lattice. strict-drop and recorded raise ValueError, naming the
    map, for ids the strict rule does not pick and for a map file that records no sidings. A list keeps ids that
    are no node: condition M reports them as not_in_graph.
    """
    world = loaded.world
    if choice.rule == "grid":
        return sorted(grid_sidings_of_world(world, choice.mode, choice.shift))
    if choice.rule == "strict":
        strict = strict_sidings_of_world(world)
        missing = sorted(set(choice.drop) - set(strict))
        if missing:
            raise ValueError(f"{world.name}: strict-drop ids not among the strict sidings: {missing}")
        return [s for s in strict if s not in choice.drop]
    if choice.rule == "recorded":  # read as M2's sidings_table reads it
        recorded = loaded.map_info.get("sidings")
        if recorded is None:
            raise ValueError(f"{world.name}: the map file records no sidings")
        if not isinstance(recorded, list):
            raise ValueError(f"{world.name}: info.sidings must be a list of node ids, got {recorded!r}")
        return sorted({str(node) for node in recorded})
    return sorted(set(choice.nodes))
