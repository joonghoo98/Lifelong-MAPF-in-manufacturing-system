"""Shared test helpers: repo paths and the tiny hand-made map."""
from pathlib import Path

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
