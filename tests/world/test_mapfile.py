import json

import pytest

from agvtb.world.mapfile import MapFormatError, load_xmsmap
from helpers import GRID_MAP, tiny_map_data


def test_songdo_counts():
    mf = load_xmsmap(GRID_MAP / "songdo_grid_bi_xMSmap.json")
    assert (len(mf.nodes), len(mf.segments), len(mf.ports)) == (1386, 4838, 10)
    assert (mf.vehicle.length, mf.vehicle.width) == (10, 10)
    assert mf.info == {}

def test_mini_and_random_counts():
    mini = load_xmsmap(GRID_MAP / "mini_deadend" / "mini_deadend_xMSmap.json")
    rnd = load_xmsmap(GRID_MAP / "random_32_32_20" / "random_32_32_20_xMSmap.json")
    assert (len(mini.nodes), len(mini.segments), len(mini.ports)) == (166, 530, 16)
    assert (len(rnd.nodes), len(rnd.segments), len(rnd.ports)) == (846, 2594, 47)

def test_kairos_offsets_and_hash():
    mf = load_xmsmap(GRID_MAP / "KAIROS_AMR" / "KAIROS_AMR.map.json")
    assert (len(mf.nodes), len(mf.segments), len(mf.ports)) == (61, 130, 14)
    assert sum(1 for s in mf.segments if s.heading_offset == 180) == 29
    assert next(s for s in mf.segments if (s.start, s.end) == ("na.12", "na.13")).heading_offset == 180
    assert mf.sha256 == "90217ef76b552827df2e123499245a81416f9a6c2d5965ad64cbec0b248bcc47"
    v = mf.vehicle
    assert (v.margin_front, v.margin_rear, v.margin_left, v.margin_right) == (70, 80, 65, 65)

def _write_map(tmp_path, mutation):
    # tiny 맵(3 노드 A · B · C)에 mutation 하나를 넣어 JSON 으로 쓰고 경로를 돌려준다.
    data = tiny_map_data()
    nodes = {n["id"]: n for n in data["nodes"]}
    if mutation == "unknown_node":
        data["segments"][0]["endNodeId"] = "Z"
    elif mutation == "zero_length":
        nodes["C"]["x"], nodes["C"]["y"] = nodes["B"]["x"], nodes["B"]["y"]
    elif mutation == "duplicate":
        data["segments"].append({"id": 5, "startNodeId": "A", "endNodeId": "B", "speed": 0.5})
    elif mutation == "arc":
        data["segments"][0]["parts"] = [{"kind": "Arc"}]
    elif mutation == "bad_port":
        data["ports"][0]["nodeId"] = "Z"
    elif mutation == "duplicate_node":  # a second B, far from the first
        data["nodes"].append({"id": "B", "x": 7.0, "y": 7.0})
    elif mutation == "comma_in_id":  # C is renamed "C,D" everywhere, so nothing else is wrong
        nodes["C"]["id"] = "C,D"
        data["segments"][2]["endNodeId"] = data["segments"][3]["startNodeId"] = "C,D"
        data["ports"][0]["nodeId"] = "C,D"
    elif mutation in ("infinite_x", "nan_y", "huge_int_x"):  # huge_int_x: an int no float can hold
        key, value = {"infinite_x": ("x", float("inf")), "nan_y": ("y", float("nan")),
                      "huge_int_x": ("x", 10 ** 400)}[mutation]
        nodes["C"][key] = value
    elif mutation in ("speed_zero", "speed_negative", "speed_infinite", "speed_nan"):
        data["segments"][1]["speed"] = {"speed_zero": 0, "speed_negative": -0.5,
                                        "speed_infinite": float("inf"), "speed_nan": float("nan")}[mutation]
    elif mutation == "no_segments":
        data["segments"] = []
    elif mutation == "port_on_untouched_node":  # D lies on no segment, and port P2 sits on D
        data["nodes"].append({"id": "D", "x": 5.0, "y": 5.0})
        data["ports"].append({"id": "P2", "nodeId": "D"})
    else:
        raise ValueError(f"unknown mutation {mutation!r}")
    path = tmp_path / "bad_map.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path

@pytest.mark.parametrize("mutation, message", [
    ("unknown_node", r"segment 1: unknown node 'Z'"),
    ("zero_length", r"segment 3: zero-length \(B -> C\)"),
    ("duplicate", r"segment 5: duplicate of segment 1 \(A -> B\)"),
    ("arc", r"segment 1: arc parts are not supported"),
    ("bad_port", r"port P1: unknown node 'Z'"),
    ("duplicate_node", r"node B: duplicate node id"),
    ("comma_in_id", r"node C,D: id must not contain ','"),
    ("infinite_x", r"node C: coordinates must be finite, got \(inf, 1.0\)"),
    ("nan_y", r"node C: coordinates must be finite, got \(1.0, nan\)"),
    ("huge_int_x", r"node C: coordinates must be finite"),
    ("speed_zero", r"segment 2: speed must be finite and > 0, got 0$"),
    ("speed_negative", r"segment 2: speed must be finite and > 0, got -0\.5$"),
    ("speed_infinite", r"segment 2: speed must be finite and > 0, got inf$"),
    ("speed_nan", r"segment 2: speed must be finite and > 0, got nan$"),
    ("no_segments", r"bad_map\.json: the map has no segments"),
    ("port_on_untouched_node", r"port P2: node 'D' is touched by no segment"),
])
def test_rejects_bad_maps(tmp_path, mutation, message):
    # 3 노드 A(0,0) · B(1,0) · C(1,1), A↔B · B↔C 양방향, 포트 C 인 맵을 쓰고 mutation 하나를 넣는다:
    # unknown_node: 세그먼트 끝을 "Z" 로 · zero_length: C 를 B 와 같은 좌표로 · duplicate: A→B 를 한 번 더 ·
    # arc: A→B 에 "parts": [{"kind": "Arc"}] · bad_port: 포트 nodeId 를 "Z" 로
    # (the first five; the others are listed in _write_map). Each message names the id of the problem.
    path = _write_map(tmp_path, mutation)
    with pytest.raises(MapFormatError, match=message):
        load_xmsmap(path)
