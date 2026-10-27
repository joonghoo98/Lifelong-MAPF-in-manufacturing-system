"""One lifelong simulation run on one map: python -m agvtb.tools.run_sim CONFIG --vehicles N --rate R.

The siding variant is the config's "sidings" key unless --sidings names another; the vehicle count is set here,
for each run (user decision G4). Demand is a uniform from-to table over the map's ports, or --fromto CSV scaled to
one job a minute; --rate is jobs per minute. The gate is checked first; if it refuses, nothing is run or written
and the exit code is 3. Exit codes: 0 no deadlock and no check finding, 1 a deadlock or a finding, 2 a usage
error, 3 the gate refused.

With --planner m6 (M6 design §5.1) the vehicles rest on the cells of --rest-rule (default no-cut) and each job is
planned by the fast path first (M6b; --no-fast-path: every job a fallback bundle, as M6a); the M6 gate (Theorem C)
is checked instead, no siding variant is read and A2 is off.
"""
import argparse
import json
import math
import time
from pathlib import Path

from agvtb.planning.gate import GateError
from agvtb.planning.m6_gate import RULES
from agvtb.planning.safe_tail import SafeTailPlanner
from agvtb.sidings.choice import candidates_of, parse_siding_choice
from agvtb.sim.demand import normalize, read_fromto
from agvtb.sim.simulator import SimConfig, simulate
from agvtb.world.config import load_map_config
from agvtb.world.loader import load_world


def main(argv: list[str] | None = None) -> int:
    """Run the lifelong simulation for warmup + measure seconds and report KPIs, deadlocks and check findings."""
    parser = argparse.ArgumentParser(prog="python -m agvtb.tools.run_sim", description=main.__doc__)
    parser.add_argument("config", help="map config JSON, e.g. configs/maps/mini_deadend.json")
    parser.add_argument("--vehicles", type=int, required=True, metavar="N", help="number of vehicles (>= 1)")
    parser.add_argument("--rate", type=float, required=True, metavar="R", help="jobs per minute (>= 0)")
    parser.add_argument("--seed", type=int, default=0, metavar="X", help="seed of the start sidings and the demand")
    parser.add_argument("--warmup", type=float, default=3600.0, metavar="S", help="warm-up seconds (default 3600)")
    parser.add_argument("--measure", type=float, default=14400.0, metavar="S",
                        help="measured seconds after the warm-up (default 14400)")
    parser.add_argument("--dt", type=float, default=0.5, metavar="S", help="clock step in seconds (default 0.5)")
    parser.add_argument("--budget", type=int, metavar="B",
                        help=f"search expansions per task (default {SafeTailPlanner.DEFAULT_BUDGET})")
    parser.add_argument("--sidings", metavar="CHOICE",
                        help="siding variant, e.g. grid-bridge-s00, strict, list:a,b (default: the config's)")
    parser.add_argument("--fromto", metavar="CSV", help="from-to rate table (default: uniform over the ports)")
    parser.add_argument("--tail-cut", action=argparse.BooleanOptionalAction, default=None,
                        help="tail cut A2 (default on; --no-tail-cut turns it off)")
    parser.add_argument("--cut-margin", type=float, default=0.0, metavar="S",
                        help="A2: the old plan must leave the splice at now + S or later (default 0)")
    parser.add_argument("--planner", choices=("sl", "m6"), default="sl",
                        help="sl: sidings, A1 + B + A2 (default); m6: rest cells, the fast path first, then the fallback")
    parser.add_argument("--rest-rule", choices=RULES, help="m6 only: the rest cells (default no-cut)")
    parser.add_argument("--fast-path", action=argparse.BooleanOptionalAction, default=None,
                        help="m6 only: the fast path first (default on; --no-fast-path: only the fallback, as M6a)")
    parser.add_argument("--out", metavar="FILE", help="JSON file for the result")
    args = parser.parse_args(argv)
    m6 = args.planner == "m6"
    if m6 and args.tail_cut is True:
        parser.error("--tail-cut: m6 does not use A2")
    if m6 and args.sidings is not None:
        parser.error("--sidings: m6 rests on the cells of its rest rule, not on sidings")
    if not m6 and args.rest_rule is not None:
        parser.error("--rest-rule: only --planner m6 has a rest rule")
    if not m6 and args.fast_path is not None:
        parser.error("--fast-path: only --planner m6 has a fast path")
    tail_cut = False if m6 else args.tail_cut is not False
    rest_rule = (args.rest_rule or "no-cut") if m6 else None
    fast_path = (args.fast_path is not False) if m6 else None
    if args.vehicles < 1:
        parser.error(f"--vehicles must be >= 1, got {args.vehicles}")
    for name, value, ok in (("--rate", args.rate, args.rate >= 0), ("--warmup", args.warmup, args.warmup >= 0),
                            ("--measure", args.measure, args.measure > 0), ("--dt", args.dt, args.dt > 0),
                            ("--cut-margin", args.cut_margin, args.cut_margin >= 0)):
        if not (math.isfinite(value) and ok):
            parser.error(f"{name} is out of range, got {value}")
    if args.budget is not None and args.budget < 1:
        parser.error(f"--budget must be >= 1, got {args.budget}")
    if args.out is not None and not Path(args.out).parent.is_dir():
        parser.error(f"--out: no folder {Path(args.out).parent}")
    if args.out is not None and Path(args.out).is_dir():
        parser.error(f"--out: {args.out} is a folder, not a file")

    try:
        cfg = load_map_config(args.config)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    choice, candidates = None, ()
    if not m6:
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
    if not m6:
        try:
            candidates = candidates_of(lw, choice)
        except ValueError as exc:
            parser.error(str(exc))
    if len(set(lw.world.port_nodes)) < 2:
        parser.error(f"{lw.world.name}: a job needs two ports, the map has {sorted(set(lw.world.port_nodes))}")
    fromto = None
    if args.fromto is not None:
        try:
            fromto = normalize(read_fromto(args.fromto, lw.world.port_nodes))
        except (OSError, ValueError) as exc:
            parser.error(f"--fromto: {exc}")

    sim_cfg = SimConfig(args.vehicles, args.rate, args.seed, args.warmup, args.measure, args.dt, args.budget, fromto,
                        tail_cut, args.cut_margin, args.planner, rest_rule or "no-cut", fast_path is not False)
    started = time.perf_counter()
    try:
        result = simulate(lw, candidates, sim_cfg)
    except GateError as err:
        print("gate refused:")
        for violation in err.report.violations:
            print(f"  {violation}")
        return 3
    wall_s = time.perf_counter() - started
    if args.out is not None:
        spans = [j.t_completed - j.t_assigned for j in result.jobs if j.t_completed is not None]
        data = {
            "config": str(args.config), "map": lw.world.name, "map_sha256": lw.map_sha256,
            "sidings_choice": choice.name if choice else None, "vehicles": args.vehicles, "rate_per_min": args.rate, "seed": args.seed,
            "warmup_s": args.warmup, "measure_s": args.measure, "dt": args.dt,
            "budget": args.budget if args.budget is not None else SafeTailPlanner.DEFAULT_BUDGET,
            "fromto": args.fromto, "tail_cut": tail_cut, "cut_margin": args.cut_margin,
            "planner": args.planner, "rest_rule": rest_rule, "fast_path": fast_path,
            "ok": result.ok, "gate": result.gate.summary(), "kpi": result.kpi,
            "deadlock": result.deadlock, "checks": result.checks, "stats": result.stats,
            "events_sha256": result.events_sha256, "n_events": result.n_events, "t_end": result.t_end,
            # for the DAIM st24 comparison (M4c): wall time of the run (a measurement, never an input), the longest
            # assignment-to-completion of any job in the whole run (st24's job_assign_to_done_max_s), jobs created
            "wall_s": wall_s, "job_max_s": max(spans) if spans else None, "jobs_created": len(result.jobs),
        }
        if m6:
            data["m6"] = result.m6
        Path(args.out).write_text(json.dumps(data), encoding="utf-8")
    findings = sum(len(found) for found in result.checks.values())
    d = result.deadlock
    print(f"{lw.world.name}: {result.kpi['completed']} jobs completed, global deadlock {d['global']}, "
          f"partial {d['partial_jobs'] + d['held_at_end']}, wait cycles {len(d['wait_cycles'])}, "
          f"{findings} check findings")
    print(f"time: {wall_s:.2f} s")  # a measurement: in the result JSON as wall_s, never an input of the run
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
