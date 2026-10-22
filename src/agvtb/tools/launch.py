"""Ask a few questions in the console, then open the live viewer: the program behind view_sim.bat.

Each question shows its default in brackets; an empty answer takes it, a bad one is asked again. The maps offered
are the configs that name a siding variant (key "sidings"). The run itself is agvtb.tools.view_sim.
"""
import math
from collections.abc import Callable
from pathlib import Path

from agvtb.world.config import load_map_config

SPEEDS = (1.0, 2.0, 5.0, 10.0, 30.0, 100.0)
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


def _int_at_least(low: int) -> Callable[[str], int]:
    def parse(text: str) -> int:
        value = int(text)
        if value < low:
            raise ValueError(f"needs a whole number >= {low}")
        return value
    return parse


def _rate(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value >= 0):
        raise ValueError("needs a number >= 0")
    return value


def _speed(text: str) -> float:
    value = float(text)
    if value not in SPEEDS:
        raise ValueError(f"needs one of {', '.join(f'{s:g}' for s in SPEEDS)}")
    return value


def build_argv(input_fn=input, print_fn=print, root: Path = Path("configs") / "maps") -> list[str]:
    """Ask for the map, vehicles, jobs per minute, seed and speed; the arguments of view_sim."""
    configs = runnable_configs(root)
    if not configs:
        raise SystemExit(f"no map config with a siding variant under {root}")
    print_fn("Maps:")
    for i, path in enumerate(configs, 1):
        print_fn(f"  {i}) {path.stem}")
    default_map = next((i for i, p in enumerate(configs, 1) if p.stem == DEFAULT_MAP), 1)

    def pick(text: str) -> int:
        value = int(text)
        if not 1 <= value <= len(configs):
            raise ValueError(f"needs a number from 1 to {len(configs)}")
        return value

    chosen = configs[_ask("Map number", default_map, pick, input_fn, print_fn) - 1]
    vehicles = _ask("Vehicles", 8, _int_at_least(1), input_fn, print_fn)
    rate = _ask("Jobs per minute", 2.0, _rate, input_fn, print_fn)
    seed = _ask("Seed", 0, _int_at_least(0), input_fn, print_fn)
    speed = _ask("Speed (sim seconds per second)", 10.0, _speed, input_fn, print_fn)
    return [str(chosen), "--vehicles", str(vehicles), "--rate", f"{rate:g}", "--seed", str(seed),
            "--speed", f"{speed:g}"]


def main(argv: list[str] | None = None, input_fn=input, run: Callable[[list[str]], int] | None = None,
         root: Path = Path("configs") / "maps") -> int:
    """Ask, then run the live viewer; its exit code."""
    args = build_argv(input_fn, print, root)
    print("Keys in the window: Space pause, + / - speed, . one step while paused, Esc quit.", flush=True)
    if run is None:
        from agvtb.tools.view_sim import main as run
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
