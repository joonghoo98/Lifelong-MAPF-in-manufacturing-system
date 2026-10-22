"""Ask a few questions in the console, then open the live viewer: the program behind view_sim.bat.

Each question shows its default in brackets; an empty answer takes it, a bad one is asked again. The maps offered
are the configs that name a siding variant (key "sidings"). The run itself is agvtb.tools.view_sim.
"""
import math
from collections.abc import Callable
from pathlib import Path

from agvtb.world.config import load_map_config

DEFAULT_MAP = "mini_deadend"


def runnable_configs(root: Path) -> list[Path]:
    """The configs under root that name a siding variant, sorted by name (case ignored)."""
    found = []
    for path in root.rglob("*.json"):
        try:
            if load_map_config(path).sidings is not None:
                found.append(path)
        except (OSError, ValueError, KeyError):
            continue
    return sorted(found, key=lambda p: (p.stem.lower(), str(p)))


def _ask(question: str, default, parse: Callable[[str], object], input_fn, print_fn):
    while True:
        shown = f"{default:g}" if isinstance(default, float) else default
        text = input_fn(f"{question} [{shown}]: ").strip()
        if not text:
            return default
        try:
            return parse(text)
        except ValueError as exc:
            print_fn(f"  {exc}; try again")


def _number(text: str, kind: type) -> float:
    try:
        return kind(text)
    except ValueError:
        raise ValueError("needs a whole number" if kind is int else "needs a number") from None


def _int_at_least(low: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        value = _number(text, int)
        if value < low:
            raise ValueError(f"needs a whole number >= {low}")
        return value
    return parse


def _rate(text: str) -> float:
    value = _number(text, float)
    if not (math.isfinite(value) and value >= 0):
        raise ValueError("needs a number >= 0")
    return value


def build_argv(input_fn=input, print_fn=print, root: Path = Path("configs") / "maps") -> list[str]:
    """Ask for the map, vehicles, jobs per minute and seed; the arguments of view_sim. Warm-up, sim time and speed
    are set in the viewer itself (user, 2026-10-22)."""
    configs = runnable_configs(root)
    if not configs:
        raise SystemExit(f"no map config with a siding variant under {root}")
    print_fn("Maps:")
    for i, path in enumerate(configs, 1):
        print_fn(f"  {i}) {path.stem}")
    default_map = next((i for i, p in enumerate(configs, 1) if p.stem == DEFAULT_MAP), 1)

    def pick(text: str) -> int:
        value = _number(text, int)
        if not 1 <= value <= len(configs):
            raise ValueError(f"needs a number from 1 to {len(configs)}")
        return value

    chosen = configs[_ask("Map number", default_map, pick, input_fn, print_fn) - 1]
    vehicles = _ask("Vehicles", 8, _int_at_least(1), input_fn, print_fn)
    rate = _ask("Jobs per minute", 2.0, _rate, input_fn, print_fn)
    seed = _ask("Seed", 0, _int_at_least(0), input_fn, print_fn)
    return [str(chosen), "--vehicles", str(vehicles), "--rate", f"{rate:g}", "--seed", str(seed)]


def main(argv: list[str] | None = None, input_fn=input, run: Callable[[list[str]], int] | None = None,
         root: Path = Path("configs") / "maps") -> int:
    """Ask, then run the live viewer; its exit code. A gate refusal (too many vehicles for the map) asks again;
    Ctrl+C or the end of input during the questions ends quietly with 1."""
    if run is None:
        from agvtb.tools.view_sim import main as run
    while True:
        try:
            args = build_argv(input_fn, print, root)
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        print("In the window: S speed, W warm-up, T sim time (type a number, Enter), F fast run, Space pause, "
              "E stop/resume, Esc quit.", flush=True)
        code = run(args)
        if code != 3:
            return code
        print("The map cannot take that many vehicles: try fewer vehicles or another map.", flush=True)
        print(flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
