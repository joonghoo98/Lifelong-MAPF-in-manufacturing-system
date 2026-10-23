import json

import pytest

from agvtb.world.config import ConfigError, load_map_config
from helpers import CONFIGS, tiny_config, world_of

REQUIRED_KEYS = [
    "name", "map", "units_per_meter", "accel_mps2", "decel_mps2",
    "rotation_s_per_90deg", "dwell_s", "collision",
]

def _edit(path, drop=(), **patch):
    # Rewrite the config file: remove the keys in drop, then set the keys in patch.
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in drop:
        del data[key]
    path.write_text(json.dumps(data | patch), encoding="utf-8")
    return path

def test_tiny_config_fields(tmp_path):
    cfg = load_map_config(tiny_config(tmp_path))
    assert (cfg.name, cfg.units_per_meter, cfg.collision, cfg.profile_path) == ("tiny", 1.0, "grid", None)
    assert (cfg.accel_mps2, cfg.decel_mps2, cfg.rotation_s_per_90deg, cfg.dwell_s) == (0.5, 0.5, 2.5, 10.0)
    assert cfg.map_path == tmp_path / "tiny_map.json"  # relative to the config's folder, not the cwd

@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_missing_key(tmp_path, key):
    path = _edit(tiny_config(tmp_path), drop=[key])
    with pytest.raises(ConfigError, match=key):
        load_map_config(path)

@pytest.mark.parametrize("key, value", [
    ("units_per_meter", -20), ("accel_mps2", 0), ("decel_mps2", 0), ("rotation_s_per_90deg", -2.5),
    ("accel_mps2", float("nan")), ("dwell_s", -1),
    ("units_per_meter", "20"), ("accel_mps2", True), ("name", 5),
])
def test_rejects_bad_value(tmp_path, key, value):
    path = _edit(tiny_config(tmp_path), **{key: value})
    with pytest.raises(ConfigError, match=key):
        load_map_config(path)

@pytest.mark.parametrize("key", ["units_per_meter", "accel_mps2", "decel_mps2", "rotation_s_per_90deg", "dwell_s"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), 10 ** 400], ids=["inf", "-inf", "int beyond float"])
def test_rejects_non_finite_number(tmp_path, key, value):
    # json parses Infinity, and an int too big for a float is as unusable as one
    path = _edit(tiny_config(tmp_path), **{key: value})
    with pytest.raises(ConfigError, match=f"{key} must be finite"):
        load_map_config(path)

def test_rejects_the_json_literal_1e999(tmp_path):
    # 1e999 is a plain number to the json module, and it comes out as infinity
    path = tiny_config(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace('"units_per_meter": 1.0', '"units_per_meter": 1e999'),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="units_per_meter must be finite, got inf"):
        load_map_config(path)

def test_dwell_zero_is_legal(tmp_path):
    # The design spec (section 15) anticipates a later run without dwell.
    cfg = load_map_config(_edit(tiny_config(tmp_path), dwell_s=0))
    assert cfg.dwell_s == 0

@pytest.mark.parametrize("patch, message", [
    ({"collision": "magic"}, "collision must be one of"),
    ({"collision": "profile"}, "needs a 'profile' key"),
])
def test_collision_rules(tmp_path, patch, message):
    path = _edit(tiny_config(tmp_path), **patch)
    with pytest.raises(ConfigError, match=message):
        load_map_config(path)

def test_profile_path(tmp_path):
    path = _edit(tiny_config(tmp_path), collision="profile", profile="profiles/tiny_collision.json")
    assert load_map_config(path).profile_path == tmp_path / "profiles" / "tiny_collision.json"
    # A "profile" key means nothing unless collision is "profile".
    assert load_map_config(_edit(path, collision="polygon")).profile_path is None

def test_map_and_profile_paths_drop_the_parent_hops(tmp_path):
    # Windows refuses paths over 260 characters, and the '..' hops alone can push a deep clone over (R23)
    (tmp_path / "configs").mkdir()
    path = tiny_config(tmp_path).rename(tmp_path / "configs" / "tiny.json")
    cfg = load_map_config(_edit(path, map="../maps/tiny_map.json", collision="profile", profile="../maps/p.json"))
    assert cfg.map_path == tmp_path / "maps" / "tiny_map.json"
    assert cfg.profile_path == tmp_path / "maps" / "p.json"

@pytest.mark.parametrize("name, units, collision, counts", [
    ("songdo_grid_bi", 20, "polygon", (1386, 4838, 10)),
    ("mini_deadend", 16.14, "grid", (166, 530, 16)),
    ("random_32_32_20", 20, "grid", (846, 2594, 47)),
    ("KAIROS_AMR", 1000, "profile", (61, 130, 14)),
])
def test_four_configs_build_worlds(name, units, collision, counts):
    # counts = nodes, segments, port nodes (the same numbers test_mapfile.py reads from the files)
    cfg = load_map_config(CONFIGS / f"{name}.json")
    assert (cfg.name, cfg.units_per_meter, cfg.collision) == (name, units, collision)
    assert (cfg.accel_mps2, cfg.decel_mps2, cfg.rotation_s_per_90deg, cfg.dwell_s) == (0.5, 0.5, 2.5, 10.0)
    assert cfg.map_path.is_file()
    if collision == "profile":
        assert cfg.profile_path.is_file()
    else:
        assert cfg.profile_path is None
    w = world_of(name)
    assert (len(w.nodes), len(w.segments), len(w.port_nodes)) == counts


@pytest.mark.parametrize("config, choice", [
    ("mini_deadend.json", "grid-bridge-s00"), ("random_32_32_20.json", "grid-bridge-s00"),
    ("songdo_grid_bi.json", "grid-bridge-s00"),
    ("KAIROS_AMR.json", "list:na.17,na.196,na.197,na.198,na.202,na.203,na.204,na.40,na.41,na.42,na.59,na.60"),
    ("daim/grid_songdo_manual.json", "grid-bridge-s00"), ("daim/grid_songdo_travelbr.json", None),
])
def test_research_configs_carry_the_users_variant(config, choice):  # user decision G3, 2026-10-21
    assert load_map_config(CONFIGS / config).sidings == choice


def test_sidings_is_optional_and_must_be_a_string(tmp_path):
    assert load_map_config(tiny_config(tmp_path)).sidings is None
    with pytest.raises(ConfigError, match="sidings"):
        load_map_config(_edit(tiny_config(tmp_path), sidings=3))
