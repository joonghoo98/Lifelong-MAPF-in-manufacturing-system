"""Shared test helpers: repo paths, the tiny hand-made map, and World builders."""
import json
from pathlib import Path

from agvtb.world.config import load_map_config
from agvtb.world.mapfile import load_xmsmap
from agvtb.world.model import World, build_world

REPO = Path(__file__).resolve().parents[1]
GRID_MAP = REPO.parent / "grid_map"
CONFIGS = REPO / "configs" / "maps"


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


def tiny_world(tmp_path, vehicle_m: float = 0.5, collision: str = "grid") -> World:
    """World of the tiny map (see tiny_config)."""
    return _world_from_config(tiny_config(tmp_path, vehicle_m, collision))


def world_of(name: str) -> World:
    """World of one of the four real maps, from configs/maps/<name>.json."""
    return _world_from_config(CONFIGS / f"{name}.json")


def _world_from_config(config_path: Path) -> World:
    cfg = load_map_config(config_path)
    return build_world(cfg, load_xmsmap(cfg.map_path))
