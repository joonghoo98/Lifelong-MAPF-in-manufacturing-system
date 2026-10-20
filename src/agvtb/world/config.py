"""Read a per-map config JSON: map file, scale, motion values, collision mode."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

COLLISION_MODES = ("grid", "polygon", "profile")
_REQUIRED_KEYS = (
    "name", "map", "units_per_meter", "accel_mps2", "decel_mps2",
    "rotation_s_per_90deg", "dwell_s", "collision",
)


class ConfigError(ValueError):
    """The map config file is incomplete or has a bad value."""


@dataclass(frozen=True)
class MapConfig:
    name: str
    map_path: Path
    units_per_meter: float  # map units per metre
    accel_mps2: float
    decel_mps2: float
    rotation_s_per_90deg: float
    dwell_s: float
    collision: str  # one of COLLISION_MODES
    profile_path: Path | None  # set only when collision == "profile"


def load_map_config(path: str | Path) -> MapConfig:
    """Load a map config; "map" and "profile" are relative to the config's folder."""
    path = Path(path)
    where = path.name
    data = json.loads(path.read_bytes())
    for key in _REQUIRED_KEYS:
        if key not in data:
            raise ConfigError(f"{where}: missing key {key!r}")
    collision = data["collision"]
    if collision not in COLLISION_MODES:
        raise ConfigError(
            f"{where}: collision must be one of {', '.join(COLLISION_MODES)}, got {collision!r}"
        )
    profile_path = None
    if collision == "profile":
        if "profile" not in data:
            raise ConfigError(f"{where}: collision 'profile' needs a 'profile' key")
        profile_path = path.parent / _text(data, "profile", where)
    return MapConfig(
        name=_text(data, "name", where),
        map_path=path.parent / _text(data, "map", where),
        units_per_meter=_number(data, "units_per_meter", where),
        accel_mps2=_number(data, "accel_mps2", where),
        decel_mps2=_number(data, "decel_mps2", where),
        rotation_s_per_90deg=_number(data, "rotation_s_per_90deg", where),
        dwell_s=_number(data, "dwell_s", where, zero_ok=True),
        collision=collision,
        profile_path=profile_path,
    )


def _text(data: dict, key: str, where: str) -> str:
    value = data[key]
    if not isinstance(value, str):
        raise ConfigError(f"{where}: {key} must be a string, got {value!r}")
    return value


def _number(data: dict, key: str, where: str, *, zero_ok: bool = False) -> float:
    """data[key] as a float, finite and > 0 (>= 0 with zero_ok)."""
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}: {key} must be a number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:  # an int too big for a float
        number = math.inf
    if not math.isfinite(number):  # json reads Infinity and 1e999 as inf, and NaN as nan
        raise ConfigError(f"{where}: {key} must be finite, got {value!r}")
    if not (number >= 0 if zero_ok else number > 0):
        raise ConfigError(f"{where}: {key} must be {'>= 0' if zero_ok else '> 0'}, got {value!r}")
    return number
