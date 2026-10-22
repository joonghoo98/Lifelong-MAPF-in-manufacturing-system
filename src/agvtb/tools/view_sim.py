"""Watch a lifelong simulation run live: python -m agvtb.tools.view_sim CONFIG --vehicles N --rate R.

A pygame-ce window draws the run as it goes (user decision V1). Keys: Space pause, + / - speed (x1, 2, 5, 10, 30,
100), . one dt step while paused, Esc quit. When the run ends or freezes the last frame stays until Esc. With
--screenshot FILE --at T it advances to simulated time T, draws one frame, saves it as PNG and ends (works without
a display with SDL_VIDEODRIVER=dummy). The viewer does not judge the run: it prints the same summary as run_sim and
returns 0; 2 is a usage error and 3 a gate refusal, as in run_sim.
"""
import argparse
import math
from pathlib import Path

from agvtb.planning.gate import GateError
from agvtb.sidings.choice import candidates_of, parse_siding_choice
from agvtb.sim.demand import normalize, read_fromto
from agvtb.sim.simulator import SimConfig, Simulation
from agvtb.world.config import load_map_config
from agvtb.world.loader import load_world

SPEEDS = (1.0, 2.0, 5.0, 10.0, 30.0, 100.0)
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
    parser.add_argument("--hours", type=float, default=24.0, metavar="H", help="length of the run (default 24)")
    parser.add_argument("--dt", type=float, default=0.5, metavar="S", help="clock step in seconds (default 0.5)")
    parser.add_argument("--speed", type=float, default=10.0, metavar="V",
                        help="simulated seconds per wall second: 1, 2, 5, 10, 30 or 100 (default 10)")
    parser.add_argument("--size", default="1280x800", metavar="WxH", help="window size, each >= 200 (default 1280x800)")
    parser.add_argument("--sidings", metavar="CHOICE", help="siding variant (default: the config's)")
    parser.add_argument("--fromto", metavar="CSV", help="from-to rate table (default: uniform over the ports)")
    parser.add_argument("--screenshot", metavar="FILE", help="save one frame at --at as PNG and end")
    parser.add_argument("--at", type=float, metavar="T", help="simulated time of the screenshot")
    args = parser.parse_args(argv)
    if args.vehicles < 1:
        parser.error(f"--vehicles must be >= 1, got {args.vehicles}")
    for name, value, ok in (("--rate", args.rate, args.rate >= 0), ("--hours", args.hours, args.hours > 0),
                            ("--dt", args.dt, args.dt > 0)):
        if not (math.isfinite(value) and ok):
            parser.error(f"{name} is out of range, got {value}")
    if args.speed not in SPEEDS:
        parser.error(f"--speed must be one of {', '.join(f'{s:g}' for s in SPEEDS)}, got {args.speed:g}")
    size = _size(args.size)
    if size is None:
        parser.error(f"--size must be WxH with both >= 200, got {args.size}")
    if (args.screenshot is None) != (args.at is None):
        parser.error("--screenshot and --at go together")
    if args.at is not None and not (math.isfinite(args.at) and args.at >= 0):
        parser.error(f"--at must be finite and >= 0, got {args.at}")
    if args.screenshot is not None and not Path(args.screenshot).parent.is_dir():
        parser.error(f"--screenshot: no folder {Path(args.screenshot).parent}")
    try:
        from agvtb.viz.viewer import Viewer
    except ModuleNotFoundError:
        parser.error(INSTALL_HINT)

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
    print(f"loading {args.config} ...", flush=True)
    try:
        lw = load_world(args.config)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
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

    sim_cfg = SimConfig(args.vehicles, args.rate, args.seed, 0.0, args.hours * 3600.0, args.dt, None, fromto)
    try:
        sim = Simulation(lw, candidates, sim_cfg)
    except GateError as err:
        print("gate refused:")
        for violation in err.report.violations:
            print(f"  {violation}")
        return 3
    import pygame  # only after the import of the viewer worked

    try:
        viewer = Viewer(sim, size, args.speed)
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
    result = sim.result()
    d = result.deadlock
    findings = sum(len(found) for found in result.checks.values())
    print(f"{lw.world.name}: {result.kpi['completed']} jobs completed by t {result.t_end:.1f} s, global deadlock "
          f"{d['global']}, partial {d['partial_jobs'] + d['held_at_end']}, wait cycles {len(d['wait_cycles'])}, "
          f"{findings} check findings")
    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
