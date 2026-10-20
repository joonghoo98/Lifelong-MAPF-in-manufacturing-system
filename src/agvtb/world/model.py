"""The world in SI units (m, s): map file + map config -> World, and the heading rule."""
import math
from dataclasses import dataclass

from agvtb.world.config import MapConfig
from agvtb.world.mapfile import MapFile


def heading_of(dx: float, dy: float, offset_deg: float = 0.0) -> int:
    """Heading in whole degrees [0, 360) of a map-unit direction (dx, dy) plus an offset.

    The rule of the DAIM map generator; y is not flipped.
    """
    return int(round(math.degrees(math.atan2(dy, dx)) + offset_deg)) % 360


@dataclass(frozen=True)
class Segment:
    u: str
    v: str
    length_m: float
    speed_mps: float
    heading_deg: int  # whole degrees in [0, 360), see heading_of


@dataclass(frozen=True)
class Footprint:
    length_m: float  # along the heading, safety margin included
    width_m: float


@dataclass(frozen=True)
class World:
    """A map in metres and seconds: what every later stage reads."""

    name: str
    nodes: dict[str, tuple[float, float]]  # id -> (x, y) in metres
    segments: dict[tuple[str, str], Segment]  # (u, v) -> segment
    port_nodes: tuple[str, ...]  # distinct node ids of the ports, sorted
    footprint: Footprint
    accel_mps2: float
    decel_mps2: float
    rotation_s_per_90deg: float
    dwell_s: float
    collision: str  # "grid", "polygon" or "profile"


def build_world(cfg: MapConfig, mf: MapFile) -> World:
    """Convert a map file from map units to metres with cfg.units_per_meter."""
    scale = cfg.units_per_meter
    segments = {}
    for raw in mf.segments:
        (x0, y0), (x1, y1) = mf.nodes[raw.start], mf.nodes[raw.end]
        dx, dy = x1 - x0, y1 - y0  # map units: the heading does not depend on the scale
        segments[(raw.start, raw.end)] = Segment(
            u=raw.start,
            v=raw.end,
            length_m=math.hypot(dx, dy) / scale,
            speed_mps=raw.speed / scale,
            heading_deg=heading_of(dx, dy, raw.heading_offset),
        )
    vehicle = mf.vehicle
    return World(
        name=cfg.name,
        nodes={node_id: (x / scale, y / scale) for node_id, (x, y) in mf.nodes.items()},
        segments=segments,
        port_nodes=tuple(sorted({port.node_id for port in mf.ports})),
        footprint=Footprint(
            length_m=(vehicle.length + vehicle.margin_front + vehicle.margin_rear) / scale,
            width_m=(vehicle.width + vehicle.margin_left + vehicle.margin_right) / scale,
        ),
        accel_mps2=cfg.accel_mps2,
        decel_mps2=cfg.decel_mps2,
        rotation_s_per_90deg=cfg.rotation_s_per_90deg,
        dwell_s=cfg.dwell_s,
        collision=cfg.collision,
    )
