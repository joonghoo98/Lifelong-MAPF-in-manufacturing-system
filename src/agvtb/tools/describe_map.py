"""Summarise a map: python -m agvtb.tools.describe_map CONFIG [--plot PATH]."""
import argparse
from collections import Counter
from pathlib import Path

from agvtb.world.loader import LoadedWorld, load_world
from agvtb.world.plot import plot_world


def main(argv: list[str] | None = None) -> int:
    """Print the seven summary lines of the map CONFIG names; with --plot also save a figure of it."""
    parser = argparse.ArgumentParser(prog="python -m agvtb.tools.describe_map", description=main.__doc__)
    parser.add_argument("config", help="map config JSON, e.g. configs/maps/mini_deadend.json")
    parser.add_argument("--plot", metavar="PATH", help="also save a PNG figure of the map at PATH")
    args = parser.parse_args(argv)

    loaded = load_world(args.config)
    print("\n".join(_summary_lines(loaded)))
    if args.plot is not None:
        plot_world(loaded.world, Path(args.plot), title=loaded.world.name)
    return 0


def _summary_lines(loaded: LoadedWorld) -> list[str]:
    """The seven lines of the summary."""
    world = loaded.world
    kinds = Counter(state.kind for state in loaded.graph.values())
    pairs = sum(len(others) for others in loaded.conflicts.values()) // 2  # each pair is in both sets
    speeds = [segment.speed_mps for segment in world.segments.values()]
    return [
        f"map: {world.name}",
        f"collision: {world.collision}",
        f"nodes: {len(world.nodes)}  segments: {len(world.segments)}  ports: {len(world.port_nodes)}",
        f"states: S={kinds['S']} M={kinds['M']} R={kinds['R']} L={kinds['L']}",
        f"conflict_pairs: {pairs}",
        f"footprint_m: {world.footprint.length_m:g} x {world.footprint.width_m:g}",
        f"speed_mps: {min(speeds):g}-{max(speeds):g}",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
