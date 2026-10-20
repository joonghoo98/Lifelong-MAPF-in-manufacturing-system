"""Shared test helpers: repo paths, the tiny hand-made map, link maps, and World builders."""
import json
from pathlib import Path

from agvtb.world.config import load_map_config
from agvtb.world.mapfile import load_xmsmap
from agvtb.world.model import World, build_world

REPO = Path(__file__).resolve().parents[1]
GRID_MAP = REPO.parent / "grid_map"
CONFIGS = REPO / "configs" / "maps"
DAIM_CONFIGS = CONFIGS / "daim"  # configs of the nine DAIM songdo maps; their map files stay in DAIM_ENGINES
DAIM = REPO.parent / "DAIM_simulation" / "DAIM_simulation"
DAIM_ENGINES = DAIM / "experiments" / "results" / "songdo_placement_engines"


def tiny_map_data(vehicle_m: float = 0.5) -> dict:
    """xMSmap-v4.0 dict of the tiny map: nodes A(0,0) B(1,0) C(1,1), port P1 at C.

    Segments A<->B and B<->C (ids 1-4), all speed 0.5. Values are in metres.
    """
    def segment(seg_id, start, end):
        return {"id": seg_id, "startNodeId": start, "endNodeId": end, "speed": 0.5}

    return {
        "schema": "xmsmap-v4.0",
        "nodes": [
            {"id": "A", "x": 0.0, "y": 0.0},
            {"id": "B", "x": 1.0, "y": 0.0},
            {"id": "C", "x": 1.0, "y": 1.0},
        ],
        "segments": [
            segment(1, "A", "B"),
            segment(2, "B", "A"),
            segment(3, "B", "C"),
            segment(4, "C", "B"),
        ],
        "ports": [{"id": "P1", "nodeId": "C"}],
        "vehicleModels": [{
            "dimension": {"length": vehicle_m, "width": vehicle_m},
            "safetyMargin": {"front": 0.0, "rear": 0.0, "left": 0.0, "right": 0.0},
        }],
    }


def tiny_config(tmp_path, vehicle_m: float = 0.5, collision: str = "grid") -> Path:
    """Write the tiny map and a config for it into tmp_path; return the config path."""
    (tmp_path / "tiny_map.json").write_text(
        json.dumps(tiny_map_data(vehicle_m)), encoding="utf-8"
    )
    config = {
        "name": "tiny",
        "map": "tiny_map.json",  # relative to the config's folder
        "units_per_meter": 1.0,
        "accel_mps2": 0.5,
        "decel_mps2": 0.5,
        "rotation_s_per_90deg": 2.5,
        "dwell_s": 10.0,
        "collision": collision,
    }
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def link_config(
    tmp_path,
    nodes: dict[str, tuple[float, float]],
    links: list[tuple[str, str]],
    ports: list[str],
    name: str = "links",
    info: dict | None = None,
) -> Path:
    """Write a map of two-way links and a grid config for it into tmp_path; return the config path.

    nodes maps id -> (x, y) in metres. Each link (a, b) becomes the segments a -> b and b -> a, numbered
    1, 2, 3, ... in link order, all at speed 1.0. The vehicle is 0.5 m square with no margin. info,
    when given, becomes the map's "info" block.
    """
    segments = []
    for a, b in links:
        for start, end in ((a, b), (b, a)):
            segments.append({"id": len(segments) + 1, "startNodeId": start, "endNodeId": end, "speed": 1.0})
    data = {
        "schema": "xmsmap-v4.0",
        "nodes": [{"id": node, "x": x, "y": y} for node, (x, y) in nodes.items()],
        "segments": segments,
        "ports": [{"id": f"P{i}", "nodeId": port} for i, port in enumerate(ports)],
        "vehicleModels": [{
            "dimension": {"length": 0.5, "width": 0.5},
            "safetyMargin": {"front": 0.0, "rear": 0.0, "left": 0.0, "right": 0.0},
        }],
    }
    if info is not None:
        data["info"] = info
    (tmp_path / f"{name}_map.json").write_text(json.dumps(data), encoding="utf-8")
    config = {
        "name": name,
        "map": f"{name}_map.json",  # relative to the config's folder
        "units_per_meter": 1.0,
        "accel_mps2": 0.5,
        "decel_mps2": 0.5,
        "rotation_s_per_90deg": 2.5,
        "dwell_s": 10.0,
        "collision": "grid",
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def tiny_world(tmp_path, vehicle_m: float = 0.5, collision: str = "grid") -> World:
    """World of the tiny map (see tiny_config)."""
    return _world_from_config(tiny_config(tmp_path, vehicle_m, collision))


def world_of(name: str) -> World:
    """World of one of the four real maps, from configs/maps/<name>.json."""
    return _world_from_config(CONFIGS / f"{name}.json")


def daim_world(name: str) -> World:
    """World of one DAIM songdo map, from configs/maps/daim/<name>.json."""
    return world_of_config(DAIM_CONFIGS / f"{name}.json")


def daim_map_info(name: str) -> dict:
    """The "info" block of the map file that configs/maps/daim/<name>.json points at."""
    return load_xmsmap(load_map_config(DAIM_CONFIGS / f"{name}.json").map_path).info


def world_of_config(config_path: Path) -> World:
    """World of one config. No conflicts are built, so a map the cell rule would refuse still loads."""
    cfg = load_map_config(config_path)
    return build_world(cfg, load_xmsmap(cfg.map_path))


def _world_from_config(config_path: Path) -> World:
    """M1's name for world_of_config, kept for the callers that use it."""
    return world_of_config(config_path)
