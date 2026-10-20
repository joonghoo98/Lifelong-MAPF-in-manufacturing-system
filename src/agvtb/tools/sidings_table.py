"""Table, lists and figures of a map's sidings: python -m agvtb.tools.sidings_table CONFIG [CONFIG ...] --out DIR.

For each map every rule is run: the grid rule (each mode, each origin shift), the strict rule and, when the
map file records its sidings (a DAIM map), those. Each result is checked with condition M. DIR gets
sidings_table.csv and sidings_table.md (one row per map and variant, the maps in the order given),
DIR/<map>/<variant>.json (the sidings of the variant and what condition M found) and, with --plot,
DIR/<map>/<variant>.png for the variants in PLOTTED_VARIANTS.
"""
import argparse
import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from agvtb.sidings.condition_m import ConditionMReport, check_condition_m
from agvtb.sidings.grid_rule import GRID_MODES, grid_candidates, grid_sidings
from agvtb.sidings.lattice import GRID_SHIFTS, LatticeError, lattice_grid
from agvtb.sidings.strict_rule import strict_sidings_of_world
from agvtb.world.loader import LoadedWorld, load_world
from agvtb.world.model import World
from agvtb.world.plot import plot_world

TABLE_COLUMNS = (
    "map", "collision", "variant", "candidates", "sidings", "usable", "port_access", "pocket", "port",
    "not_in_graph", "aisle", "violations", "max_vehicles", "n_vehicles", "ok",
)
PLOTTED_VARIANTS = ("grid-bridge-s00", "strict")  # the variants --plot draws, when the map has them


@dataclass(frozen=True)
class Variant:
    """One way to choose the sidings of a map."""

    rule: str  # "grid", "strict" or "recorded"
    mode: str | None = None  # the mode of the grid rule, one of GRID_MODES
    shift: tuple[int, int] | None = None  # the origin shift (dr, dc) of the grid rule, one of GRID_SHIFTS

    @property
    def name(self) -> str:
        """The name of its row, list and figure: grid-<mode>-s<dr><dc> (grid-bridge-s00), strict or recorded."""
        if self.rule == "grid":
            dr, dc = self.shift
            return f"grid-{self.mode}-s{dr}{dc}"
        return self.rule


@dataclass(frozen=True)
class MapResult:
    """What map_rows finds for one map."""

    rows: list[dict]  # one per variant, keyed by TABLE_COLUMNS
    details: dict[str, dict]  # variant name -> its list (the content of its JSON file), in the order of rows
    notes: list[str]  # why some rows are missing


def map_rows(loaded: LoadedWorld, n_vehicles: int | None = None) -> MapResult:
    """Run every siding rule on the map, check each result with condition M and tabulate it.

    The variants come in this order: the grid rule, with the modes of GRID_MODES as the outer ring and the
    shifts of GRID_SHIFTS inside; the strict rule; and, if the map file has info["sidings"], those as
    "recorded" (the ids read as distinct strings). A map that is no lattice (LatticeError) gets no grid rows,
    only the note "grid rule skipped: ...".

    Condition M is checked for n_vehicles vehicles: the argument, else info["fleet"]["AMR"], else 0. Raises
    ValueError if n_vehicles is negative. A row has the exclusions of condition M by reason, its violations as
    a count, and max_vehicles: the number of usable sidings when there is no violation, else 0.

    details[name] is what goes into the JSON file of a variant: its rule, mode and shift (a list [dr, dc]), its
    sidings and the usable ones (sorted), the exclusions, the violations in the order condition M found them,
    and the summary of condition M.
    """
    world = loaded.world
    if n_vehicles is None:
        n_vehicles = _fleet_size(loaded.map_info)
    chosen, notes = _chosen_sidings(loaded)
    rows: list[dict] = []
    details: dict[str, dict] = {}
    for variant, candidates, sidings in chosen:
        report = check_condition_m(loaded.graph, loaded.conflicts, world.port_nodes, sidings, n_vehicles)
        rows.append(_row(world, variant, candidates, sidings, report))
        details[variant.name] = _detail(variant, sidings, report)
    return MapResult(rows, details, notes)


def main(argv: list[str] | None = None) -> int:
    """Run every siding rule on the maps the CONFIGs name, check them with condition M, and write the results.

    The table of all the maps, a list for each map and variant and, with --plot, figures go into DIR.
    Prints a line for each map, and the notes.
    """
    parser = argparse.ArgumentParser(prog="python -m agvtb.tools.sidings_table", description=main.__doc__)
    parser.add_argument("config", nargs="+", help="map config JSON, e.g. configs/maps/mini_deadend.json")
    parser.add_argument("--out", required=True, metavar="DIR", help="folder for the table, the lists and the figures")
    parser.add_argument(
        "--vehicles", type=int, metavar="N",
        help="number of vehicles condition M is checked for (default: info.fleet.AMR of the map file, else 0)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help=f"also save a figure of the variants {', '.join(PLOTTED_VARIANTS)} of each map",
    )
    args = parser.parse_args(argv)
    if args.vehicles is not None and args.vehicles < 0:
        parser.error(f"--vehicles must be >= 0, got {args.vehicles}")

    out = Path(args.out)
    table: list[dict] = []
    for config in args.config:
        loaded = load_world(config)
        result = map_rows(loaded, args.vehicles)
        table += result.rows
        _write_map(out / loaded.world.name, loaded.world, result, args.plot)
        print(f"{loaded.world.name}: {len(result.rows)} variants", flush=True)  # a long run is watched in a log
        for note in result.notes:
            print(note, flush=True)
    _write_text(out / "sidings_table.csv", _csv_text(table))
    _write_text(out / "sidings_table.md", _markdown_text(table))
    return 0


def _fleet_size(info: dict) -> int:
    """The number of vehicles a map file records, info["fleet"]["AMR"], or 0 if it records none."""
    fleet = info.get("fleet")
    return int(fleet["AMR"]) if isinstance(fleet, dict) and "AMR" in fleet else 0


def _chosen_sidings(loaded: LoadedWorld) -> tuple[list[tuple[Variant, int, list[str]]], list[str]]:
    """Each variant of the map with the number of its candidates and its sidings (sorted node ids), and the notes."""
    world = loaded.world
    chosen: list[tuple[Variant, int, list[str]]] = []
    notes: list[str] = []
    try:
        # the shift only moves the cell numbers, no check depends on it: if (0, 0) fails, so do the others
        lattices = {shift: lattice_grid(world, shift) for shift in GRID_SHIFTS}
    except LatticeError as error:
        notes.append(f"grid rule skipped: {error}")
    else:
        candidate_counts = {shift: len(grid_candidates(lattice.grid)) for shift, lattice in lattices.items()}
        for mode in GRID_MODES:
            for shift in GRID_SHIFTS:
                lattice = lattices[shift]
                sidings = sorted(lattice.node_at[cell] for cell in grid_sidings(lattice.grid, mode))
                chosen.append((Variant("grid", mode, shift), candidate_counts[shift], sidings))
    # the candidates of the strict rule are the nodes that are no port: all nodes less the ports (each is a node)
    chosen.append((Variant("strict"), len(world.nodes) - len(set(world.port_nodes)), strict_sidings_of_world(world)))
    if "sidings" in loaded.map_info:
        recorded = sorted({str(node) for node in loaded.map_info["sidings"]})  # a node id is a string
        chosen.append((Variant("recorded"), len(recorded), recorded))
    return chosen, notes


def _row(world: World, variant: Variant, candidates: int, sidings: list[str], report: ConditionMReport) -> dict:
    """The table row of a variant, keyed by TABLE_COLUMNS in their order."""
    reasons = Counter(report.excluded.values())
    usable = len(report.sidings)
    return {
        "map": world.name,
        "collision": world.collision,
        "variant": variant.name,
        "candidates": candidates,
        "sidings": len(sidings),
        "usable": usable,
        "port_access": reasons["port_access"],
        "pocket": reasons["pocket"],
        "port": reasons["port"],
        "not_in_graph": reasons["not_in_graph"],
        "aisle": report.aisle_size,
        "violations": len(report.violations),
        "max_vehicles": 0 if report.violations else usable,
        "n_vehicles": report.n_vehicles,
        "ok": report.ok,
    }


def _detail(variant: Variant, sidings: list[str], report: ConditionMReport) -> dict:
    """The list of a variant as plain data: its sidings and what condition M made of them."""
    return {
        "rule": variant.rule,
        "mode": variant.mode,
        "shift": None if variant.shift is None else list(variant.shift),
        "sidings": sidings,
        "usable": list(report.sidings),
        "excluded": dict(sorted(report.excluded.items())),
        "violations": list(report.violations),  # in the order condition M found them
        "condition_m": report.summary(),
    }


def _write_map(folder: Path, world: World, result: MapResult, plot: bool) -> None:
    """Write the JSON list of each variant of a map into folder, and with plot the figures of PLOTTED_VARIANTS."""
    for name, detail in result.details.items():
        _write_text(folder / f"{name}.json", json.dumps(detail, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    if plot:
        for name in PLOTTED_VARIANTS:
            if name in result.details:
                detail = result.details[name]
                plot_world(
                    world, folder / f"{name}.png", sidings=detail["usable"], excluded=detail["excluded"],
                    title=f"{world.name}: {name} (blue square: usable siding, black x: excluded candidate)",
                )


def _csv_text(rows: list[dict]) -> str:
    """The rows as CSV: the header line TABLE_COLUMNS, \\n at the end of each line, booleans as True and False."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=TABLE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _markdown_text(rows: list[dict]) -> str:
    """The rows as a Markdown table: the header row, the separator row, then one row for each of rows."""
    lines = ["| " + " | ".join(TABLE_COLUMNS) + " |", "|" + "---|" * len(TABLE_COLUMNS)]
    lines += ["| " + " | ".join(str(row[column]) for column in TABLE_COLUMNS) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    """Write text to path in UTF-8, \\n at the end of each line on any platform, making its folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
