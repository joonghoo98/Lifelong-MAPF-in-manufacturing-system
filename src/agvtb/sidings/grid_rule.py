"""The grid rule of the sidings: which cells of a cell grid can hold waiting vehicles.

It re-implements DAIM's auto_parks (mapgen/grid_from_regions.py) on a plain cell grid.
A cell is (row, col); row grows with y and col with x.
"""
import json
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path

Cell = tuple[int, int]  # (row, col)

GRID_MODES = ("bridge", "legacy")  # how grid_sidings restores sidings to the corridor; "bridge" is the default
_LEGACY_MAX_ROUNDS = 64


@dataclass(frozen=True)
class CellGrid:
    """A rows x cols block of cells, some of them blocked, and the cells of its ports in a fixed order."""

    rows: int
    cols: int
    blocked: frozenset[Cell]
    ports: tuple[Cell, ...]

    def free_cells(self) -> frozenset[Cell]:
        """The cells of the rows x cols block that are not blocked."""
        return frozenset(
            (r, c) for r in range(self.rows) for c in range(self.cols) if (r, c) not in self.blocked
        )


def load_daim_grid(path: str | Path) -> CellGrid:
    """Read the rows, cols, blocked cells and ports of a DAIM <name>.grid.json; other keys are ignored.

    The ports keep the order of the file.
    """
    data = json.loads(Path(path).read_bytes())
    return CellGrid(
        rows=data["rows"],
        cols=data["cols"],
        blocked=frozenset((r, c) for r, c in data["blocked"]),
        ports=tuple((r, c) for r, c in data["ports"]),
    )


def grid_candidates(grid: CellGrid) -> list[Cell]:
    """The free cells on an odd row and an odd column that are not ports, sorted."""
    ports = set(grid.ports)
    return sorted((r, c) for r, c in grid.free_cells() if r % 2 == 1 and c % 2 == 1 and (r, c) not in ports)


def grid_sidings(grid: CellGrid, mode: str = "bridge") -> list[Cell]:
    """The cells that can all hold a waiting vehicle at once while the ports stay in one corridor, sorted.

    Start from the candidates: no two touch, so on an open grid the even rows and columns stay a corridor
    when all of them are taken. Blocked cells can still cut the corridor (the free cells that are no siding)
    into components. Until it is one, the main component stays and sidings are restored to the corridor
    so that the stray components (the others) join it:
    - bridge: the sidings that touch both the main component and a stray one; if there are none,
      all the sidings that touch a stray one;
    - legacy: all the sidings that touch a stray one.
    Last, a siding with no corridor cell next to it is dropped.

    The main component has the most ports, then the most cells, then holds the smallest cell; DAIM leaves
    that last tie to the iteration order of a Python set. Ports that blocked cells cut off from each other
    cannot be joined by any sidings; the rounds still end.
    Raises ValueError when mode is not in GRID_MODES.
    """
    if mode not in GRID_MODES:
        raise ValueError(f"grid rule mode must be one of {', '.join(GRID_MODES)}, got {mode!r}")
    free = grid.free_cells()
    ports = set(grid.ports)  # a port listed twice counts once
    sidings = set(grid_candidates(grid))
    max_rounds = _LEGACY_MAX_ROUNDS if mode == "legacy" else len(sidings) + 1
    for _ in range(max_rounds):
        corridor = free - sidings
        comps = _components(corridor)
        if not comps:
            break
        # the key is a total order (components share no cell): the result does not depend on set iteration order
        main = min(comps, key=lambda comp: (-len(comp & ports), -len(comp), min(comp)))
        stray = corridor - main
        if not stray:
            break
        restore: set[Cell] = set()
        if mode == "bridge":
            restore = {s for s in sidings if _touches(s, main) and _touches(s, stray)}
        if not restore:
            restore = {s for s in sidings if _touches(s, stray)}
        if not restore:
            break
        sidings -= restore
    corridor = free - sidings
    return sorted(s for s in sidings if _touches(s, corridor))


def _touches(cell: Cell, cells: AbstractSet[Cell]) -> bool:
    """Whether one of the four neighbours of cell is in cells."""
    r, c = cell
    return any(nb in cells for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)))


def _components(cells: AbstractSet[Cell]) -> list[frozenset[Cell]]:
    """The 4-neighbour connected components of cells, in no particular order."""
    seen: set[Cell] = set()
    comps = []
    for start in cells:
        if start in seen:
            continue
        seen.add(start)
        comp, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    comp.add(nb)
                    stack.append(nb)
        comps.append(frozenset(comp))
    return comps
