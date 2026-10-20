"""Read an xMSmap v4.0 JSON file into plain records, in map units.

Only the fields the testbed uses are read. Other keys (passOnly, forbidAvoidAt,
homes, direction, parameters, ...) are ignored (D15).
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class MapFormatError(ValueError):
    """The map file cannot be used as a routing map."""


@dataclass(frozen=True)
class RawSegment:
    seg_id: int
    start: str
    end: str
    speed: float
    heading_offset: float  # degrees, from the segment's "heading"; 0 when absent


@dataclass(frozen=True)
class RawPort:
    port_id: str
    node_id: str


@dataclass(frozen=True)
class RawVehicle:
    length: float
    width: float
    margin_front: float
    margin_rear: float
    margin_left: float
    margin_right: float


@dataclass(frozen=True)
class MapFile:
    path: Path
    sha256: str  # of the file bytes
    nodes: dict[str, tuple[float, float]]  # id -> (x, y)
    segments: tuple[RawSegment, ...]
    ports: tuple[RawPort, ...]
    vehicle: RawVehicle
    info: dict


def load_xmsmap(path: str | Path) -> MapFile:
    """Load an xMSmap file and check it; every number stays in map units."""
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    nodes = {n["id"]: (float(n["x"]), float(n["y"])) for n in data["nodes"]}
    return MapFile(
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        nodes=nodes,
        segments=_read_segments(data["segments"], nodes),
        ports=_read_ports(data["ports"], nodes),
        vehicle=_read_vehicle(data["vehicleModels"][0]),
        info=data.get("info", {}),
    )


def _read_segments(
    items: list[dict], nodes: dict[str, tuple[float, float]]
) -> tuple[RawSegment, ...]:
    segments = []
    seen: dict[tuple[str, str], int] = {}  # (start, end) -> id of the first segment
    for item in items:
        seg_id, start, end = item["id"], item["startNodeId"], item["endNodeId"]
        for node_id in (start, end):
            if node_id not in nodes:
                raise MapFormatError(f"segment {seg_id}: unknown node {node_id!r}")
        if nodes[start] == nodes[end]:
            raise MapFormatError(f"segment {seg_id}: zero-length ({start} -> {end})")
        if any(part.get("kind") == "Arc" for part in item.get("parts") or ()):
            raise MapFormatError(f"segment {seg_id}: arc parts are not supported")
        if (start, end) in seen:
            raise MapFormatError(
                f"segment {seg_id}: duplicate of segment {seen[(start, end)]} ({start} -> {end})"
            )
        seen[(start, end)] = seg_id
        heading = item.get("heading")
        offset = heading[0]["degree"] if heading else 0
        segments.append(RawSegment(seg_id, start, end, float(item["speed"]), float(offset)))
    return tuple(segments)


def _read_ports(
    items: list[dict], nodes: dict[str, tuple[float, float]]
) -> tuple[RawPort, ...]:
    ports = []
    for item in items:
        if item["nodeId"] not in nodes:
            raise MapFormatError(f"port {item['id']}: unknown node {item['nodeId']!r}")
        ports.append(RawPort(item["id"], item["nodeId"]))
    return tuple(ports)


def _read_vehicle(model: dict) -> RawVehicle:
    size, margin = model["dimension"], model["safetyMargin"]
    return RawVehicle(
        length=float(size["length"]),
        width=float(size["width"]),
        margin_front=float(margin["front"]),
        margin_rear=float(margin["rear"]),
        margin_left=float(margin["left"]),
        margin_right=float(margin["right"]),
    )
