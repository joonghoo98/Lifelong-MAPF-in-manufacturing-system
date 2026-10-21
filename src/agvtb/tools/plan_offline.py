"""Plan tasks of many vehicles offline on one map: python -m agvtb.tools.plan_offline CONFIG --vehicles N --tasks K.

The siding variant is the config's "sidings" key unless --sidings names another. The gate (condition M, a round
trip from every siding to every node, at least N usable sidings) is checked first; if it refuses, nothing is
planned or written and the exit code is 3. Otherwise the plans and the validator's findings go to --out as JSON.
Exit codes: 0 every task planned and no finding, 1 a task unplanned or a finding, 2 a usage error, 3 the gate
refused. The vehicle count is set here, for each run (user decision G4, 2026-10-21).
"""
import argparse
import json
import math
import time
from pathlib import Path

from agvtb.planning.gate import GateError
from agvtb.planning.offline import plan_offline, result_json
from agvtb.planning.safe_tail import SafeTailPlanner
from agvtb.sidings.choice import candidates_of, parse_siding_choice
from agvtb.world.config import load_map_config
from agvtb.world.loader import load_world


def main(argv: list[str] | None = None) -> int:
    """Check the gate, plan K tasks of N vehicles on the map CONFIG names, validate the plans, write them."""
    parser = argparse.ArgumentParser(prog="python -m agvtb.tools.plan_offline", description=main.__doc__)
    parser.add_argument("config", help="map config JSON, e.g. configs/maps/mini_deadend.json")
    parser.add_argument("--vehicles", type=int, required=True, metavar="N", help="number of vehicles (>= 1)")
    parser.add_argument("--tasks", type=int, required=True, metavar="K", help="number of tasks (>= 0)")
    parser.add_argument("--interval", type=float, default=0.0, metavar="S",
                        help="seconds between two dispatches (default 0)")
    parser.add_argument("--seed", type=int, default=0, metavar="X", help="seed of the start sidings and tasks")
    parser.add_argument("--budget", type=int, metavar="B",
                        help=f"search expansions per task (default {SafeTailPlanner.DEFAULT_BUDGET})")
    parser.add_argument("--sidings", metavar="CHOICE",
                        help="siding variant, e.g. grid-bridge-s00, strict, list:a,b (default: the config's)")
    parser.add_argument("--out", metavar="FILE", help="JSON file for the plans and the findings")
    args = parser.parse_args(argv)
    if args.vehicles < 1:
        parser.error(f"--vehicles must be >= 1, got {args.vehicles}")
    if args.tasks < 0:
        parser.error(f"--tasks must be >= 0, got {args.tasks}")
    if args.budget is not None and args.budget < 1:
        parser.error(f"--budget must be >= 1, got {args.budget}")
    if not (math.isfinite(args.interval) and args.interval >= 0):
        parser.error(f"--interval must be finite and >= 0, got {args.interval}")
    if not math.isfinite(max(args.tasks - 1, 0) * args.interval):
        parser.error(f"--interval {args.interval} makes the last dispatch time of {args.tasks} tasks overflow")
    if args.out is not None and not Path(args.out).parent.is_dir():
        parser.error(f"--out: no folder {Path(args.out).parent}")
    if args.out is not None and Path(args.out).is_dir():
        parser.error(f"--out: {args.out} is a folder, not a file")

    try:
        cfg = load_map_config(args.config)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    text = args.sidings if args.sidings is not None else cfg.sidings
    if text is None:
        parser.error(f"{args.config}: no siding variant; give --sidings or a \"sidings\" key in the config")
    try:
        choice = parse_siding_choice(text)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        lw = load_world(args.config)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    try:
        candidates = candidates_of(lw, choice)
    except ValueError as exc:
        parser.error(str(exc))
    ports = sorted(set(lw.world.port_nodes))
    if args.tasks > 0 and len(ports) < 2:
        parser.error(f"{lw.world.name}: a task needs two ports, the map has {ports}")

    name = lw.world.name
    started = time.perf_counter()
    try:
        result = plan_offline(lw.graph, lw.conflicts, lw.world.port_nodes, candidates, args.vehicles, args.tasks,
                              seed=args.seed, interval_s=args.interval, budget=args.budget)
    except GateError as err:
        _print_paper(err.report)
        print("gate refused:")
        for violation in err.report.violations:
            print(f"  {violation}")
        return 3
    _print_paper(result.gate)
    if args.out is not None:
        data = {
            "config": str(args.config), "map": name, "map_sha256": lw.map_sha256, "sidings_choice": choice.name,
            "vehicles": args.vehicles, "tasks": args.tasks, "seed": args.seed, "interval_s": args.interval,
            "budget": args.budget if args.budget is not None else SafeTailPlanner.DEFAULT_BUDGET,
        } | result_json(result)
        Path(args.out).write_text(json.dumps(data), encoding="utf-8")
    planned = [p for p in result.plans if p is not None]
    fallbacks = sum(p.fallback for p in planned)
    findings = sum(len(found) for found in result.checks.values())
    print(f"{name}: {len(planned)} of {args.tasks} tasks planned ({fallbacks} by fallback), {findings} check findings")
    print(f"time: {time.perf_counter() - started:.2f} s")  # on screen only, never in the result
    return 0 if result.ok else 1


def _print_paper(report) -> None:
    outside = list(report.paper_outside)
    print(f"paper condition (not part of the gate): {len(outside)} nodes outside {outside}")


if __name__ == "__main__":
    raise SystemExit(main())
