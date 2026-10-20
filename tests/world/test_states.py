import pytest

from agvtb.world.states import State, build_state_graph, geom_key, shortest_turn_deg
from helpers import tiny_world, world_of

def test_shortest_turn():
    assert [shortest_turn_deg(0, 90), shortest_turn_deg(0, 270), shortest_turn_deg(10, 350), shortest_turn_deg(0, 180)] == [90, 90, 20, 180]

def test_tiny_state_ids(tmp_path):
    g = build_state_graph(tiny_world(tmp_path))
    by = lambda k: {s for s, st in g.items() if st.kind == k}
    assert by("S") == {"S,A,0", "S,A,180", "S,B,0", "S,B,90", "S,B,180", "S,B,270", "S,C,90", "S,C,270"}
    assert by("M") == {"M,A,B", "M,B,A", "M,B,C", "M,C,B"}
    assert len(by("R")) == 16 and by("L") == {"L,C,90", "L,C,270"}

def test_tiny_durations_and_next(tmp_path):
    g = build_state_graph(tiny_world(tmp_path))
    assert [g[s].duration_s for s in ("M,A,B", "R,B,0,90", "R,B,0,180", "R,B,270,0", "L,C,90")] == [2.0, 2.5, 5.0, 2.5, 10.0]
    assert g["S,A,0"].duration_s is None
    assert g["S,A,0"].next == ("M,A,B", "R,A,0,180")
    assert g["S,C,270"].next == ("L,C,270", "M,C,B", "R,C,270,90")
    assert g["M,A,B"].next == ("S,B,0",) and g["L,C,90"].next == ("S,C,90",)
    assert geom_key("L,C,90") == "S,C,90" and geom_key("M,A,B") == "M,A,B"

def test_songdo_port_headings():
    g = build_state_graph(world_of("songdo_grid_bi"))
    assert {"S,Port7,3", "S,Port7,183", "L,Port7,3", "L,Port7,183"} <= g.keys()
    assert g["R,Port7,3,183"].duration_s == 5.0
    assert g["R,G83_40,0,3"].duration_s == pytest.approx(3 / 90 * 2.5)

# --- tests beyond the plan's: details the tests above do not pin down ---

REAL_MAPS = ["mini_deadend", "random_32_32_20", "songdo_grid_bi", "KAIROS_AMR"]

def test_shortest_turn_matches_brute_force():
    # every pair of whole-degree headings: the smallest of the ways round, from either side
    brute = lambda a, b: min(abs(a - b + 360 * k) for k in (-1, 0, 1))
    assert all(shortest_turn_deg(a, b) == brute(a, b) for a in range(360) for b in range(360))

def test_state_fields_by_kind(tmp_path):
    # heading is None for R only; from/to_heading are set for R only; duration_s is None for S only
    g = build_state_graph(tiny_world(tmp_path))
    assert g["S,A,0"] == State(
        sid="S,A,0", kind="S", nodes=("A",), heading=0, from_heading=None, to_heading=None,
        duration_s=None, next=("M,A,B", "R,A,0,180"))
    assert g["M,A,B"] == State(
        sid="M,A,B", kind="M", nodes=("A", "B"), heading=0, from_heading=None, to_heading=None,
        duration_s=2.0, next=("S,B,0",))
    assert g["R,B,270,0"] == State(
        sid="R,B,270,0", kind="R", nodes=("B",), heading=None, from_heading=270, to_heading=0,
        duration_s=2.5, next=("S,B,0",))
    assert g["L,C,90"] == State(
        sid="L,C,90", kind="L", nodes=("C",), heading=90, from_heading=None, to_heading=None,
        duration_s=10.0, next=("S,C,90",))

@pytest.mark.parametrize("name, counts", [
    ("mini_deadend", {"S": 660, "M": 530, "R": 1972, "L": 64}),
    ("random_32_32_20", {"S": 3156, "M": 2594, "R": 9012, "L": 94}),
    ("songdo_grid_bi", {"S": 5530, "M": 4838, "R": 16610, "L": 20}),
    ("KAIROS_AMR", {"S": 169, "M": 130, "R": 510, "L": 15}),  # S, M, R as in its bundled profile, which has no L
])
def test_real_map_state_counts(name, counts):
    # counts from a separate computation of the same four rules
    g = build_state_graph(world_of(name))
    assert {k: sum(st.kind == k for st in g.values()) for k in "SMRL"} == counts

@pytest.mark.parametrize("name", REAL_MAPS)
def test_real_map_ids_next_and_order(name):
    g = build_state_graph(world_of(name))
    assert list(g) == sorted(g)  # the graph itself comes in sorted order
    ident = {  # the id that each state's own fields spell
        "S": lambda s: f"S,{s.nodes[0]},{s.heading}",
        "M": lambda s: f"M,{s.nodes[0]},{s.nodes[1]}",
        "R": lambda s: f"R,{s.nodes[0]},{s.from_heading},{s.to_heading}",
        "L": lambda s: f"L,{s.nodes[0]},{s.heading}",
    }
    arrival = {  # the S that an M, R or L leads to
        "M": lambda s: f"S,{s.nodes[1]},{s.heading}",
        "R": lambda s: f"S,{s.nodes[0]},{s.to_heading}",
        "L": lambda s: f"S,{s.nodes[0]},{s.heading}",
    }
    for sid, st in g.items():
        assert ident[st.kind](st) == sid
        assert list(st.next) == sorted(st.next) and set(st.next) <= g.keys()
        if st.kind == "S":
            assert st.duration_s is None
        else:
            assert st.duration_s > 0 and st.next == (arrival[st.kind](st),)

@pytest.mark.parametrize("name", REAL_MAPS)
def test_real_map_s_next_and_l_states(name):
    world = world_of(name)
    g = build_state_graph(world)
    leaving = {}  # S id -> ids of the M, R and L that start from it, by their own fields
    for sid, st in g.items():
        if st.kind != "S":
            start_heading = st.from_heading if st.kind == "R" else st.heading
            leaving.setdefault(f"S,{st.nodes[0]},{start_heading}", []).append(sid)
    assert all(st.next == tuple(sorted(leaving.get(sid, ()))) for sid, st in g.items() if st.kind == "S")
    # one L for each S at a port node; its space is that of the S
    at_port = {sid for sid, st in g.items() if st.kind == "S" and st.nodes[0] in world.port_nodes}
    assert {geom_key(sid) for sid, st in g.items() if st.kind == "L"} == at_port
    assert all(geom_key(sid) == sid for sid, st in g.items() if st.kind != "L")
