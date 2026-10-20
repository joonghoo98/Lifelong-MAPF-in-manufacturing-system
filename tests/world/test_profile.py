import json
import math
from collections import Counter
from dataclasses import replace

import pytest

from agvtb.world.config import load_map_config
from agvtb.world.loader import load_world
from agvtb.world.mapfile import load_xmsmap
from agvtb.world.model import build_world
from agvtb.world.profile import ProfileMismatchError, check_profile_durations, load_collision_profile
from agvtb.world.states import State, build_state_graph, geom_key
from helpers import CONFIGS

@pytest.fixture(scope="module")
def kairos():
    cfg = load_map_config(CONFIGS / "KAIROS_AMR.json"); mf = load_xmsmap(cfg.map_path)
    w = build_world(cfg, mf)
    return w, mf, cfg, *load_collision_profile(w, mf, cfg.profile_path)

def test_counts_and_pairs(kairos):
    w, mf, cfg, g, c = kairos
    kinds = Counter(st.kind for st in g.values())
    assert (kinds["S"], kinds["M"], kinds["R"]) == (169, 130, 510)
    assert kinds["L"] == sum(1 for st in g.values() if st.kind == "S" and st.nodes[0] in w.port_nodes) > 0
    pairs = {tuple(sorted((a, b))) for a, bs in c.items() for b in bs if a[0] != "L" and b[0] != "L"}
    assert len(pairs) == 33000

def test_durations_match_d4(kairos):
    w, mf, cfg, g, c = kairos
    assert check_profile_durations(w, g) == []
    assert g["M,na.1,na.2"].heading == 0 and g["M,na.1,na.2"].duration_s == 15.01

def test_l_conflicts_follow_s(kairos):
    w, mf, cfg, g, c = kairos
    for sid, st in g.items():
        if st.kind == "L":
            s = "S," + sid[2:]
            assert c[s] - {sid} <= c[sid] and s in c[sid]
    assert all(a in c[b] for a, bs in c.items() for b in bs)

def test_sha_mismatch_raises(kairos, tmp_path):
    w, mf, cfg, g, c = kairos
    data = json.loads(cfg.profile_path.read_text(encoding="utf-8")); data["map_sha256"] = "0" * 64
    bad = tmp_path / "bad_profile.json"; bad.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ProfileMismatchError):
        load_collision_profile(w, mf, bad)

# --- further checks: details the four tests above do not pin down ---

def _gap(a, b):
    """|a - b| of two durations: 0 for two None, infinity for one None."""
    if a is None or b is None:
        return 0.0 if a is b else math.inf
    return abs(a - b)

def _edited_profile(cfg, tmp_path, edit):
    """A copy of the profile in tmp_path, changed by edit(data); returns its path."""
    data = json.loads(cfg.profile_path.read_text(encoding="utf-8")); edit(data)
    path = tmp_path / "edited_profile.json"; path.write_text(json.dumps(data), encoding="utf-8")
    return path

def test_profile_states_are_read_as_given(kairos):
    # with no port nodes there is no L and only the profile is left: the ids in sorted order (the loader
    # sorts them, whatever the file's order), next, durations (an S has none) and conflicts as in the file
    w, mf, cfg, g, c = kairos
    g0, c0 = load_collision_profile(replace(w, port_nodes=()), mf, cfg.profile_path)
    raw = json.loads(cfg.profile_path.read_text(encoding="utf-8"))["states"]
    assert list(g0) == sorted(raw) and list(c0) == sorted(raw)
    assert {sid: list(st.next) for sid, st in g0.items()} == {sid: r["next"] for sid, r in raw.items()}
    assert {sid: st.duration_s for sid, st in g0.items()} == {sid: r["duration"] for sid, r in raw.items()}
    assert {sid: set(bs) for sid, bs in c0.items()} == {sid: set(r["conflicts"]) for sid, r in raw.items()}

def test_state_fields_by_kind(kairos):
    # S, M, R and L filled as build_state_graph fills them: heading None for R only, from_heading and
    # to_heading set for R only, duration_s None for S only. na.5 is a port node: its S has the L in next
    w, mf, cfg, g, c = kairos
    assert g["S,na.5,90"] == State(
        sid="S,na.5,90", kind="S", nodes=("na.5",), heading=90, from_heading=None, to_heading=None,
        duration_s=None, next=("L,na.5,90", "M,na.5,na.6", "R,na.5,90,270"))
    assert g["M,na.1,na.2"] == State(
        sid="M,na.1,na.2", kind="M", nodes=("na.1", "na.2"), heading=0, from_heading=None, to_heading=None,
        duration_s=15.01, next=("S,na.2,0",))
    assert g["R,na.10,0,180"] == State(
        sid="R,na.10,0,180", kind="R", nodes=("na.10",), heading=None, from_heading=0, to_heading=180,
        duration_s=5.0, next=("S,na.10,180",))
    assert g["L,na.5,90"] == State(
        sid="L,na.5,90", kind="L", nodes=("na.5",), heading=90, from_heading=None, to_heading=None,
        duration_s=10.0, next=("S,na.5,90",))

def test_graph_equals_the_rule_built_graph(kairos):
    # the profile and the rules of build_state_graph give one graph: the same ids in the same order and,
    # for each state, the same fields and next (L included); durations within 1e-9
    w, mf, cfg, g, c = kairos
    rule = build_state_graph(w)
    assert list(g) == list(rule)
    assert [sid for sid in g if replace(g[sid], duration_s=None) != replace(rule[sid], duration_s=None)] == []
    assert [sid for sid in g if _gap(g[sid].duration_s, rule[sid].duration_s) > 1e-9] == []

def test_one_l_per_port_s(kairos):
    # every S at a port node has one L of its pose, and that L is in the next of this S and of no other;
    # na.5 is a port node with two poses, so it has two L
    w, mf, cfg, g, c = kairos
    at_port = {sid for sid, st in g.items() if st.kind == "S" and st.nodes[0] in w.port_nodes}
    loads = {sid for sid, st in g.items() if st.kind == "L"}
    assert loads == {"L" + sid[1:] for sid in at_port} and len(loads) == 15
    assert {"L,na.5,90", "L,na.5,270"} <= loads
    assert [sid for sid in at_port if "L" + sid[1:] not in g[sid].next] == []
    assert {sid for sid, st in g.items() if loads & set(st.next)} == at_port

def test_conflicts_put_each_l_in_the_space_of_its_s(kairos):
    # states conflict when the spaces they take do, and an L takes the space of its S. So an L conflicts
    # with its S, with what that S conflicts with in the file, and with the L of each of those
    w, mf, cfg, g, c = kairos
    raw = json.loads(cfg.profile_path.read_text(encoding="utf-8"))["states"]
    in_file = {sid: set(r["conflicts"]) for sid, r in raw.items()}
    space = {sid: geom_key(sid) for sid in g}
    expected = {a: {b for b in g if b != a and (space[b] == space[a] or space[b] in in_file[space[a]])} for a in g}
    assert [a for a in g if c[a] != expected[a]] == []

def test_graph_and_conflict_structure(kairos):
    # sorted ids; one frozenset of conflicts per state in the graph's order, never the state itself;
    # every next and every conflict names a state of the graph; every next tuple is sorted
    w, mf, cfg, g, c = kairos
    assert list(g) == sorted(g) and list(c) == list(g)
    assert all(isinstance(bs, frozenset) for bs in c.values())
    assert [a for a, bs in c.items() if a in bs] == []
    assert [a for a, bs in c.items() if not bs <= set(g)] == []
    assert [sid for sid, st in g.items() if st.next != tuple(sorted(st.next)) or not set(st.next) <= set(g)] == []

def test_l_states_follow_the_world(kairos):
    # the port nodes and the dwell time are the world's: other values, other L states
    w, mf, cfg, g, c = kairos
    g2, _ = load_collision_profile(replace(w, port_nodes=("na.5",), dwell_s=4.0), mf, cfg.profile_path)
    loads = {sid: st for sid, st in g2.items() if st.kind == "L"}
    assert set(loads) == {"L,na.5,90", "L,na.5,270"} and {st.duration_s for st in loads.values()} == {4.0}

def test_check_profile_durations_reports_mismatches(kairos):
    # an M or R more than 1e-6 s off is reported and a closer one is not; an M without a segment in the
    # world is a line, not a KeyError; S and L are not looked at
    w, mf, cfg, g, c = kairos
    m, r, near = g["M,na.1,na.2"], g["R,na.10,0,180"], g["M,na.11,na.35"]
    edited = {
        **g,
        m.sid: replace(m, duration_s=m.duration_s + 2e-6),
        r.sid: replace(r, duration_s=r.duration_s - 2e-6),
        near.sid: replace(near, duration_s=near.duration_s + 9e-7),
        "M,na.1,nowhere": replace(m, sid="M,na.1,nowhere", nodes=("na.1", "nowhere")),
        "S,na.1,0": replace(g["S,na.1,0"], duration_s=99.0),
        "L,na.5,90": replace(g["L,na.5,90"], duration_s=99.0),
    }
    lines = check_profile_durations(w, edited)
    assert [line.split(":")[0] for line in lines] == ["M,na.1,na.2", "R,na.10,0,180", "M,na.1,nowhere"]
    assert lines[0] == "M,na.1,na.2: duration 15.010002 s, the world gives 15.01 s"
    assert lines[2] == "M,na.1,nowhere: the world has no segment na.1 -> nowhere"

def test_check_profile_durations_uses_the_worlds_values(kairos):
    # the world's rotation time and speeds decide, not constants: a slower turn puts all 510 R in the
    # list, faster segments all 130 M
    w, mf, cfg, g, c = kairos
    turn_lines = check_profile_durations(replace(w, rotation_s_per_90deg=3.0), g)
    assert len(turn_lines) == 510 and all(line.startswith("R,") for line in turn_lines)
    faster = {key: replace(seg, speed_mps=seg.speed_mps * 2) for key, seg in w.segments.items()}
    move_lines = check_profile_durations(replace(w, segments=faster), g)
    assert len(move_lines) == 130 and all(line.startswith("M,") for line in move_lines)

def test_changed_map_file_raises(kairos, tmp_path):
    # the map was edited after the profile was made: one byte more is enough to change the sha256
    w, mf, cfg, g, c = kairos
    edited = tmp_path / "KAIROS_AMR.map.json"; edited.write_bytes(cfg.map_path.read_bytes() + b"\n")
    changed = load_xmsmap(edited)
    assert changed.sha256 != mf.sha256
    with pytest.raises(ProfileMismatchError) as err:
        load_collision_profile(w, changed, cfg.profile_path)
    assert isinstance(err.value, ValueError)
    assert mf.sha256 in str(err.value) and changed.sha256 in str(err.value)  # the digest in the profile, and the map's

def test_output_order_does_not_depend_on_the_file_order(kairos, tmp_path):
    # the file lists its states, next and conflicts the other way round: the result is the same,
    # with ids, next tuples and graph keys in sorted order
    w, mf, cfg, g, c = kairos
    def backwards(data):
        data["states"] = {sid: dict(r, next=r["next"][::-1], conflicts=r["conflicts"][::-1])
                          for sid, r in reversed(data["states"].items())}
    g2, c2 = load_collision_profile(w, mf, _edited_profile(cfg, tmp_path, backwards))
    assert list(g2) == list(g) and g2 == g and list(c2) == list(c) and c2 == c

@pytest.mark.parametrize("edit, message", [
    (lambda data: data.pop("map_sha256"), "map_sha256 None"),
    (lambda data: data.update(schema="collision-v2"), "schema must be 'collision-v1', got 'collision-v2'"),
    (lambda data: data["states"]["S,na.1,0"].update(type="X"), "state S,na.1,0: unknown type 'X'"),
    (lambda data: data["states"]["M,na.1,na.2"].update(next=["S,na.1,0"]), "state M,na.1,na.2: next 'S,na.1,0' is not a stop at na.2"),
    # the contract: ids that name nothing, a state against itself, one-sided conflicts, nodes the world lacks
    (lambda data: data["states"]["S,na.1,0"]["next"].append("M,na.1,nowhere"),
     "state S,na.1,0: next 'M,na.1,nowhere' is not a state of the profile"),
    (lambda data: data["states"]["S,na.1,0"]["conflicts"].append("S,nowhere,0"),
     "state S,na.1,0: conflict 'S,nowhere,0' is not a state of the profile"),
    (lambda data: data["states"]["S,na.1,0"]["conflicts"].append("S,na.1,0"),
     "state S,na.1,0: conflicts with itself"),
    (lambda data: data["states"]["M,na.1,na.2"]["conflicts"].remove("S,na.1,0"),  # S,na.1,0 still lists the M
     "conflicts are not symmetric: S,na.1,0 lists 'M,na.1,na.2', which does not list S,na.1,0"),
    (lambda data: data["states"]["S,na.1,0"].update(node="na.nowhere"),
     "state S,na.1,0: node 'na.nowhere' is not in the world"),
    (lambda data: data["states"]["M,na.1,na.2"].update({"from": "na.nowhere"}),
     "state M,na.1,na.2: node 'na.nowhere' is not in the world"),
], ids=["no map_sha256", "other schema", "unknown state type", "M next is not a stop at its end node",
        "next names no state", "conflict names no state", "state conflicts with itself",
        "conflicts are one-sided", "S node is not in the world", "M node is not in the world"])
def test_unusable_profile_raises(kairos, tmp_path, edit, message):
    w, mf, cfg, g, c = kairos
    with pytest.raises(ProfileMismatchError, match=message):
        load_collision_profile(w, mf, _edited_profile(cfg, tmp_path, edit))

def test_rotation_time_of_the_config_must_match_the_profile(kairos, tmp_path):
    # the reviewer's probe: the config says 1.0 s per 90 degrees but the profile's 510 R durations are made for
    # 2.5 s. The graph would contradict its own World, so it is refused. The message gives the count and the first lines.
    w, mf, cfg, g, c = kairos
    config = json.loads((CONFIGS / "KAIROS_AMR.json").read_text(encoding="utf-8"))
    config.update(map=str(cfg.map_path.resolve()), profile=str(cfg.profile_path.resolve()), rotation_s_per_90deg=1.0)
    path = tmp_path / "KAIROS_turns_fast.json"; path.write_text(json.dumps(config), encoding="utf-8")
    lines = check_profile_durations(replace(w, rotation_s_per_90deg=1.0), g)
    assert len(lines) == 510
    with pytest.raises(ProfileMismatchError, match="510 durations disagree with the world") as err:
        load_world(path)
    message = str(err.value)
    assert lines[0] in message and lines[2] in message and lines[3] not in message  # only the first few are shown

def test_segment_speeds_must_match_the_profile(kairos):
    # the same for M: speeds twice the profile's make all 130 M durations wrong
    w, mf, cfg, g, c = kairos
    faster = {key: replace(seg, speed_mps=seg.speed_mps * 2) for key, seg in w.segments.items()}
    with pytest.raises(ProfileMismatchError, match="130 durations disagree with the world.*M,na.1,na.2: duration 15.01 s"):
        load_collision_profile(replace(w, segments=faster), mf, cfg.profile_path)

def test_check_profile_durations_reports_a_missing_duration(kairos):
    # an M or R without a duration is a line, not a TypeError
    w, mf, cfg, g, c = kairos
    m, r = g["M,na.1,na.2"], g["R,na.10,0,180"]
    lines = check_profile_durations(w, {**g, m.sid: replace(m, duration_s=None), r.sid: replace(r, duration_s=None)})
    assert lines == ["M,na.1,na.2: no duration, the world gives 15.01 s", "R,na.10,0,180: no duration, the world gives 5 s"]
