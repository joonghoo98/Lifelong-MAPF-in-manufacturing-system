import json
import math

import pytest

from agvtb.world.config import ConfigError, load_map_config
from agvtb.world.mapfile import load_xmsmap
from agvtb.world.model import build_world, heading_of
from helpers import tiny_config, tiny_world, world_of

def test_heading_rule():
    assert [heading_of(1, 0), heading_of(0, 1), heading_of(-1, 0), heading_of(0, -1)] == [0, 90, 180, 270]
    assert heading_of(1, 0, 180) == 180
    a = math.radians(182.6)
    assert heading_of(math.cos(a), math.sin(a)) == 183

def test_songdo_si_units():
    w = world_of("songdo_grid_bi")
    assert {s.speed_mps for s in w.segments.values()} == {0.5}
    assert (w.footprint.length_m, w.footprint.width_m) == (0.5, 0.5)
    s = w.segments[("G100_17", "G100_18")]
    assert (s.length_m, s.heading_deg) == (pytest.approx(1.0), 90)
    p7 = w.segments[("Port7", "G83_40")]
    assert p7.length_m == pytest.approx(0.47838, abs=1e-4) and p7.heading_deg == 183
    assert (w.accel_mps2, w.rotation_s_per_90deg, w.dwell_s, w.collision) == (0.5, 2.5, 10.0, "polygon")

def test_mini_speed_from_json():
    w = world_of("mini_deadend")
    assert {round(s.speed_mps, 6) for s in w.segments.values()} == {round(14.37 / 16.14, 6)}
    assert w.footprint.length_m == pytest.approx(10 / 16.14)

def test_kairos_world():
    w = world_of("KAIROS_AMR")
    assert (w.footprint.length_m, w.footprint.width_m) == (pytest.approx(0.93), pytest.approx(0.83))
    assert {round(s.speed_mps, 6) for s in w.segments.values()} == {1.0, 0.5, 0.4, 0.3, 0.2, 0.1}
    assert len(w.port_nodes) == 14

@pytest.mark.parametrize("patch", [{"collision": "profile"}, {"collision": "magic"}, {"units_per_meter": 0}])
def test_config_validation(tmp_path, patch):
    path = tiny_config(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8")) | patch
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_map_config(path)

# --- tests beyond the plan's: details the tests above do not pin down ---

def test_heading_rounds_to_nearest():
    def heading_at(degrees):
        a = math.radians(degrees)
        return heading_of(math.cos(a), math.sin(a))
    assert {d: heading_at(d) for d in (2.4, 2.6, 182.4, 359.6)} == {2.4: 2, 2.6: 3, 182.4: 182, 359.6: 0}

def test_kairos_headings():
    # Arrival headings written in the bundled profile (next of M,u,v). na.12 -> na.13 has
    # heading offset 180 (drives backwards); na.39 <-> na.40 runs 271.35 deg, off the axis.
    w = world_of("KAIROS_AMR")
    keys = [("na.12", "na.13"), ("na.13", "na.12"), ("na.39", "na.40"), ("na.40", "na.39")]
    assert [w.segments[key].heading_deg for key in keys] == [90, 90, 271, 271]

def test_tiny_world(tmp_path):
    w = tiny_world(tmp_path)
    assert (w.name, w.collision) == ("tiny", "grid")
    assert w.nodes == {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0)}
    assert all((s.u, s.v) == key for key, s in w.segments.items())
    assert {key: (s.length_m, s.speed_mps, s.heading_deg) for key, s in w.segments.items()} == {
        ("A", "B"): (1.0, 0.5, 0), ("B", "A"): (1.0, 0.5, 180),
        ("B", "C"): (1.0, 0.5, 90), ("C", "B"): (1.0, 0.5, 270),
    }
    assert w.port_nodes == ("C",)
    assert (w.footprint.length_m, w.footprint.width_m) == (0.5, 0.5)
    assert (w.accel_mps2, w.decel_mps2, w.rotation_s_per_90deg, w.dwell_s) == (0.5, 0.5, 2.5, 10.0)
    big = tiny_world(tmp_path, vehicle_m=0.8, collision="polygon")
    assert (big.footprint.length_m, big.footprint.width_m, big.collision) == (0.8, 0.8, "polygon")

def test_world_follows_the_config(tmp_path):
    # 2 map units = 1 m: coordinates, lengths, speeds and the footprint halve, headings stay.
    # The motion values are all different here, so a mix-up between them shows.
    path = tiny_config(tmp_path)
    patch = {"name": "other", "units_per_meter": 2.0, "accel_mps2": 0.4, "decel_mps2": 0.6,
             "rotation_s_per_90deg": 3.0, "dwell_s": 0}
    path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8")) | patch), encoding="utf-8")
    cfg = load_map_config(path)
    w = build_world(cfg, load_xmsmap(cfg.map_path))
    assert w.name == "other"
    assert (w.accel_mps2, w.decel_mps2, w.rotation_s_per_90deg, w.dwell_s) == (0.4, 0.6, 3.0, 0)
    assert w.nodes == {"A": (0.0, 0.0), "B": (0.5, 0.0), "C": (0.5, 0.5)}
    s = w.segments[("B", "C")]
    assert (s.length_m, s.speed_mps, s.heading_deg) == (0.5, 0.25, 90)
    assert (w.footprint.length_m, w.footprint.width_m) == (0.25, 0.25)

def test_port_nodes_are_sorted_and_distinct(tmp_path):
    # Ports P1 and P3 share node C, and the file lists C before A.
    cfg = load_map_config(tiny_config(tmp_path))
    data = json.loads(cfg.map_path.read_text(encoding="utf-8"))
    data["ports"] = [{"id": "P1", "nodeId": "C"}, {"id": "P2", "nodeId": "A"}, {"id": "P3", "nodeId": "C"}]
    cfg.map_path.write_text(json.dumps(data), encoding="utf-8")
    assert build_world(cfg, load_xmsmap(cfg.map_path)).port_nodes == ("A", "C")
