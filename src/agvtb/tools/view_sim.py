"""Watch a lifelong simulation run live: python -m agvtb.tools.view_sim CONFIG --vehicles N --rate R.

A pygame-ce window draws the run as it goes (user decision V1). The run has no end: it goes on until the window is
closed. In the window (user, 2026-10-22): S speed, W warm-up, T sim time (type a number, Enter), F fast run (event to
event up to warm-up + sim time, then on at the speed), Space pause, + / - speed step, . one dt step while paused, Esc
quit; the KPI lines are those of [warm-up end, now]. --warmup, --sim-time and --speed only set the starting values.
If the run freezes the last frame stays until Esc. With
--screenshot FILE --at T it advances to simulated time T, draws one frame, saves it as PNG and ends (works without
a display with SDL_VIDEODRIVER=dummy). The viewer does not judge the run: it prints the same summary as run_sim and
returns 0; 2 is a usage error and 3 a gate refusal, as in run_sim. With --planner m6 (user, 2026-10-27) the vehicles
rest on the cells of --rest-rule (default no-cut, as run_sim) and each job is planned by the fast path first
(--no-fast-path: every plan a fallback bundle, as M6a); it needs no siding variant.
"""
import argparse
import math
import os
from pathlib import Path

from agvtb.planning.gate import GateError
from agvtb.planning.m6_gate import RULES
from agvtb.sidings.choice import candidates_of, parse_siding_choice
from agvtb.sim.demand import normalize, read_fromto
from agvtb.sim.simulator import SimConfig, Simulation
from agvtb.world.config import load_map_config
from agvtb.world.loader import load_world

INSTALL_HINT = 'the viewer needs pygame-ce: .venv/Scripts/python.exe -m pip install "pygame-ce>=2.5,<2.6"'


def _size(text: str) -> tuple[int, int] | None:
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    w, h = int(parts[0]), int(parts[1])
    return (w, h) if w >= 200 and h >= 200 else None


def main(argv: list[str] | None = None) -> int:
    """Run the simulation in a window, or save one frame at a simulated time."""
    parser = argparse.ArgumentParser(prog="python -m agvtb.tools.view_sim", description=main.__doc__)
    parser.add_argument("config", help="map config JSON, e.g. configs/maps/mini_deadend.json")
    parser.add_argument("--vehicles", type=int, required=True, metavar="N", help="number of vehicles (>= 1)")
    parser.add_argument("--rate", type=float, required=True, metavar="R", help="jobs per minute (>= 0)")
    parser.add_argument("--seed", type=int, default=0, metavar="X", help="seed of the start sidings and the demand")
    parser.add_argument("--warmup", type=float, default=3600.0, metavar="S", help="starting warm-up seconds (default 3600)")
    parser.add_argument("--sim-time", type=float, default=86400.0, metavar="S",
                        help="starting sim time of a fast run after the warm-up (default 86400)")
    parser.add_argument("--dt", type=float, default=0.5, metavar="S", help="clock step in seconds (default 0.5)")
    parser.add_argument("--speed", type=float, default=10.0, metavar="V",
                        help="starting simulated seconds per wall second (default 10)")
    parser.add_argument("--size", default="1280x800", metavar="WxH", help="window size, each >= 200 (default 1280x800)")
    parser.add_argument("--sidings", metavar="CHOICE", help="siding variant (default: the config's)")
    parser.add_argument("--fromto", metavar="CSV", help="from-to rate table (default: uniform over the ports)")
    parser.add_argument("--screenshot", metavar="FILE", help="save one frame at --at as PNG and end")
    parser.add_argument("--at", type=float, metavar="T", help="simulated time of the screenshot")
    parser.add_argument("--planner", choices=("sl", "m6"), default="sl",
                        help="sl: sidings, A1 + B + A2 (default); m6: rest cells, the fast path first, then the fallback")
    parser.add_argument("--rest-rule", choices=RULES, help="m6 only: the rest cells (default no-cut)")
    parser.add_argument("--fast-path", action=argparse.BooleanOptionalAction, default=None,
                        help="m6 only: the fast path first (default on; --no-fast-path: only the fallback, as M6a)")
    args = parser.parse_args(argv)
    m6 = args.planner == "m6"
    if m6 and args.sidings is not None:
        parser.error("--sidings: m6 rests on the cells of its rest rule, not on sidings")
    if not m6 and args.rest_rule is not None:
        parser.error("--rest-rule: only --planner m6 has a rest rule")
    if not m6 and args.fast_path is not None:
        parser.error("--fast-path: only --planner m6 has a fast path")
    if args.vehicles < 1:
        parser.error(f"--vehicles must be >= 1, got {args.vehicles}")
    for name, value, ok in (("--rate", args.rate, args.rate >= 0), ("--warmup", args.warmup, args.warmup >= 0),
                            ("--sim-time", args.sim_time, args.sim_time > 0), ("--dt", args.dt, args.dt > 0),
                            ("--speed", args.speed, args.speed > 0)):
        if not (math.isfinite(value) and ok):
            parser.error(f"{name} is out of range, got {value}")
    size = _size(args.size)
    if size is None:
        parser.error(f"--size must be WxH with both >= 200, got {args.size}")
    if (args.screenshot is None) != (args.at is None):
        parser.error("--screenshot and --at go together")
    if args.at is not None and not (math.isfinite(args.at) and args.at >= 0):
        parser.error(f"--at must be finite and >= 0, got {args.at}")
    if args.screenshot is not None and not Path(args.screenshot).parent.is_dir():
        parser.error(f"--screenshot: no folder {Path(args.screenshot).parent}")
    if args.screenshot is not None and Path(args.screenshot).is_dir():
        parser.error(f"--screenshot: {args.screenshot} is a folder, not a file")
    if args.screenshot is None and os.environ.get("SDL_VIDEODRIVER", "").lower() == "dummy":
        parser.error("SDL_VIDEODRIVER=dummy shows no window: a live run could not be seen or closed; use --screenshot")
    try:
        from agvtb.viz.viewer import Viewer
    except ModuleNotFoundError:
        parser.error(INSTALL_HINT)

    try:
        cfg = load_map_config(args.config)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    choice = None
    if not m6:
        text = args.sidings if args.sidings is not None else cfg.sidings
        if text is None:
            parser.error(f"{args.config}: no siding variant; give --sidings or a \"sidings\" key in the config")
        try:
            choice = parse_siding_choice(text)
        except ValueError as exc:
            parser.error(str(exc))
    print(f"loading {args.config} ...", flush=True)
    try:
        lw = load_world(args.config)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    try:
        candidates = candidates_of(lw, choice) if choice is not None else ()
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

    sim_cfg = SimConfig(args.vehicles, args.rate, args.seed, args.warmup, math.inf, args.dt, None, fromto,
                        planner=args.planner, rest_rule=args.rest_rule or "no-cut", fast_path=args.fast_path is not False)
    try:
        sim = Simulation(lw, candidates, sim_cfg)
    except GateError as err:
        print("gate refused:")
        for violation in err.report.violations:
            print(f"  {violation}")
        return 3
    import pygame  # only after the import of the viewer worked

    try:
        viewer = Viewer(sim, size, args.speed, args.warmup, args.sim_time)
    except pygame.error as exc:
        parser.error(f"cannot open a window ({exc}); try SDL_VIDEODRIVER=dummy with --screenshot")
    if args.screenshot is not None:
        sim.advance_to(args.at)
        viewer.draw()
        viewer.save(args.screenshot)
        print(f"screenshot at t {sim.now:.1f} s: {args.screenshot}")
    else:
        if viewer.run() in ("end", "frozen"):
            viewer.hold()
    pygame.quit()  # close the window first: the overlap check of a long run takes a while (final review I3)
    print(f"checking the run up to t {sim.now:.1f} s ...", flush=True)
    result = sim.result(viewer.warmup)
    d = result.deadlock
    findings = sum(len(found) for found in result.checks.values())
    print(f"{lw.world.name}: {result.kpi['completed']} jobs completed in [{viewer.warmup:g}, {result.t_end:.1f}] s, "
          f"global deadlock {d['global']}, partial {d['partial_jobs'] + d['held_at_end']}, wait cycles "
          f"{len(d['wait_cycles'])}, {findings} check findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
