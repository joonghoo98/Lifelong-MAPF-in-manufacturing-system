import dataclasses
import json

import pytest

from agvtb.world.collision_grid import GridRuleError, check_grid_rule, grid_conflicts, grid_pitch
from agvtb.world.collision_poly import poly_conflicts
from agvtb.world.config import load_map_config
from agvtb.world.loader import load_world
from agvtb.world.mapfile import load_xmsmap
from agvtb.world.model import Footprint, build_world
from agvtb.world.states import build_state_graph, geom_key
from helpers import tiny_config, tiny_map_data, tiny_world, world_of

def test_tiny_conflicts(tmp_path):
    w = tiny_world(tmp_path); c = grid_conflicts(w, build_state_graph(w))
    assert c["S,A,0"] == {"S,A,180", "M,A,B", "M,B,A", "R,A,0,180", "R,A,180,0"}
    assert len(c["M,A,B"]) == 23
    assert all(a in c[b] for a, bs in c.items() for b in bs)          # 대칭
    assert "L,C,90" in c["S,C,270"] and "M,A,B" not in c["L,C,90"]

def test_check_grid_rule(tmp_path):
    check_grid_rule(world_of("mini_deadend"))                           # 대각선 0.876 m ≤ 간격 1.0 m
    check_grid_rule(world_of("random_32_32_20"))
    with pytest.raises(GridRuleError):
        check_grid_rule(tiny_world(tmp_path, vehicle_m=0.8))             # 대각선 1.131 m > 1.0 m

# --- further checks: details the two tests above do not pin down ---

def test_grid_pitch(tmp_path):
    # the shortest segment, not a typical one: songdo's Port7 link is 0.478 m among 1 m segments
    assert grid_pitch(tiny_world(tmp_path)) == 1.0
    assert grid_pitch(world_of("mini_deadend")) == pytest.approx(1.0)
    assert grid_pitch(world_of("random_32_32_20")) == pytest.approx(1.0)
    assert grid_pitch(world_of("songdo_grid_bi")) == pytest.approx(0.478383, abs=1e-6)

def test_grid_rule_error_names_map_diagonal_and_pitch(tmp_path):
    with pytest.raises(GridRuleError) as err:
        check_grid_rule(tiny_world(tmp_path, vehicle_m=0.8))
    message = str(err.value)
    assert "tiny" in message and "1.131" in message and "1.000" in message

def test_songdo_fails_the_grid_rule():
    # why songdo is a polygon map: its Port7 link (0.478 m) is shorter than the vehicle diagonal (0.707 m)
    with pytest.raises(GridRuleError, match=r"songdo_grid_bi.*0\.707.*0\.478"):
        check_grid_rule(world_of("songdo_grid_bi"))

def test_check_grid_rule_tolerance(tmp_path):
    # A diagonal equal to the pitch fits (0.6 x 0.8 -> 1.0 m on a 1.0 m grid), and float noise up to
    # 1e-9 m is forgiven; a diagonal more than that over does not fit.
    w = tiny_world(tmp_path)
    def with_footprint(length_m, width_m):
        return dataclasses.replace(w, footprint=Footprint(length_m, width_m))
    assert check_grid_rule(with_footprint(0.6, 0.8)) is None
    check_grid_rule(with_footprint(0.6, 0.8 + 5e-10))    # diagonal 1 m + 4e-10
    with pytest.raises(GridRuleError):
        check_grid_rule(with_footprint(0.6, 0.8 + 3e-9))  # diagonal 1 m + 2.4e-9

def test_conflicts_cover_every_state_of_the_graph(tmp_path):
    # only the graph given counts: S,C,90 has nothing to collide with here, yet it is still a key
    w = tiny_world(tmp_path); full = build_state_graph(w)
    part = {sid: full[sid] for sid in ("S,A,0", "M,A,B", "S,C,90")}
    c = grid_conflicts(w, part)
    assert c == {"S,A,0": {"M,A,B"}, "M,A,B": {"S,A,0"}, "S,C,90": set()}
    assert all(type(v) is frozenset for v in c.values())

def test_grid_conflicts_do_not_check_the_rule(tmp_path):
    # the check is the loader's job: a vehicle too big for the grid changes nothing here and raises nothing
    small, big = tiny_world(tmp_path), tiny_world(tmp_path, vehicle_m=0.8)
    graph = build_state_graph(small)
    assert grid_conflicts(big, graph) == grid_conflicts(small, graph)

def test_tiny_matches_the_definition(tmp_path):
    # two different states collide exactly when they share a node, checked over all pairs of the 30 states
    w = tiny_world(tmp_path); g = build_state_graph(w)
    expected = {a: frozenset(b for b in g if b != a and set(g[a].nodes) & set(g[b].nodes)) for a in g}
    assert grid_conflicts(w, g) == expected

# songdo_grid_bi is run with the cell rule too, though it is a polygon map: its polygon conflicts
# are compared with the cell rule.
@pytest.mark.parametrize("name, pairs", [
    ("mini_deadend", 40749),
    ("random_32_32_20", 181080),
    ("songdo_grid_bi", 350943),
])
def test_real_map_pair_counts(name, pairs):
    # unordered pairs of colliding states; the numbers come from a separate computation of the same rule
    w = world_of(name); c = grid_conflicts(w, build_state_graph(w))
    assert sum(len(v) for v in c.values()) == 2 * pairs

@pytest.mark.parametrize("name", ["mini_deadend", "random_32_32_20", "songdo_grid_bi"])
def test_real_map_conflict_structure(name):
    w = world_of(name); g = build_state_graph(w); c = grid_conflicts(w, g)
    assert list(c) == list(g)                                   # a key for every state, in graph order
    assert all(type(v) is frozenset for v in c.values())
    assert [s for s, v in c.items() if s in v] == []            # never itself
    assert [(a, b) for a, bs in c.items() for b in bs if a not in c[b]] == []   # symmetric
    # an L takes the space of the S with the same pose: apart from the pair itself, the same conflicts
    assert [s for s in g if s.startswith("L,") and c[s] | {s} != c[geom_key(s)] | {geom_key(s)}] == []

# --- the lattice premise: the cell rule is right only on a lattice, so check_grid_rule checks one ---

def _edited_tiny_config(tmp_path, edit):
    """Config of the tiny map (grid mode, 0.5 m vehicle, 1 m segments) after edit(data) changed its xMSmap dict."""
    tmp_path.mkdir(exist_ok=True)
    config = tiny_config(tmp_path)
    data = tiny_map_data()
    edit(data)
    (tmp_path / "tiny_map.json").write_text(json.dumps(data), encoding="utf-8")
    return config

def _edited_tiny_world(tmp_path, edit):
    cfg = load_map_config(_edited_tiny_config(tmp_path, edit))
    return build_world(cfg, load_xmsmap(cfg.map_path))

def _add_segments(data, *pairs):
    # speed 0.5, ids go on from the last one
    for start, end in pairs:
        data["segments"].append({"id": len(data["segments"]) + 1, "startNodeId": start, "endNodeId": end, "speed": 0.5})

def _node_0_6_m_from_another(data):
    # D lies 0.6 m from C with no segment between them; D-E is a plain 1 m link of its own
    data["nodes"] += [{"id": "D", "x": 1.6, "y": 1.0}, {"id": "E", "x": 2.6, "y": 1.0}]
    _add_segments(data, ("D", "E"), ("E", "D"))

def _unit_square_with_both_diagonals(data):
    # the square A B C D; its diagonals A-C and B-D cross in the middle and share no node
    data["nodes"].append({"id": "D", "x": 0.0, "y": 1.0})
    _add_segments(data, ("C", "D"), ("D", "C"), ("D", "A"), ("A", "D"), ("A", "C"), ("C", "A"), ("B", "D"), ("D", "B"))

def _two_node_ids_at_one_point(data):
    # A2 sits on A and links to B as A does
    data["nodes"].append({"id": "A2", "x": 0.0, "y": 0.0})
    _add_segments(data, ("A2", "B"), ("B", "A2"))

PROBES = [  # the final review's three probes, each with a 0.5 m vehicle and 1.0 m segments
    pytest.param(_node_0_6_m_from_another, r"node D is off the 1\.000 m lattice of node A", id="node 0.6 m from another"),
    pytest.param(_unit_square_with_both_diagonals, r"segment A -> C is not axis-aligned", id="unit square with both diagonals"),
    pytest.param(_two_node_ids_at_one_point, r"nodes A and A2 are at the same lattice point", id="two node ids at one point"),
]

@pytest.mark.parametrize("edit, message", PROBES)
def test_check_grid_rule_needs_a_lattice(tmp_path, edit, message):
    # the diagonal rule alone lets each probe through (0.707 m <= 1.0 m); the map is no lattice, so GridRuleError
    # names the node or segment and points to polygon mode
    world = _edited_tiny_world(tmp_path, edit)
    assert grid_pitch(world) == 1.0
    with pytest.raises(GridRuleError, match=message) as err:
        check_grid_rule(world)
    assert "collision: polygon" in str(err.value)

@pytest.mark.parametrize("edit", [probe.values[0] for probe in PROBES], ids=[probe.id for probe in PROBES])
def test_the_probes_are_real_misses_of_the_cell_rule(tmp_path, edit):
    # why the lattice is checked: on each probe polygon mode finds colliding pairs that the cell rule does not
    world = dataclasses.replace(_edited_tiny_world(tmp_path, edit), collision="polygon")
    graph = build_state_graph(world)
    poly, grid = poly_conflicts(world, graph), grid_conflicts(world, graph)
    assert all(grid[sid] <= poly[sid] for sid in graph)
    assert sum(len(poly[sid] - grid[sid]) for sid in graph) > 0

def test_load_world_refuses_a_map_that_is_no_lattice(tmp_path):
    with pytest.raises(GridRuleError, match="node D is off the"):
        load_world(_edited_tiny_config(tmp_path, _node_0_6_m_from_another))

def test_check_grid_rule_rejects_a_segment_longer_than_the_pitch(tmp_path):
    # B -> E is 2 m long between two nodes of the 1 m lattice: it runs through (2, 0), which has no node of its own
    def edit(data):
        data["nodes"].append({"id": "E", "x": 3.0, "y": 0.0})
        _add_segments(data, ("B", "E"), ("E", "B"))
    with pytest.raises(GridRuleError, match=r"segment B -> E is 2\.000 m long, not the pitch 1\.000 m") as err:
        check_grid_rule(_edited_tiny_world(tmp_path, edit))
    assert "collision: polygon" in str(err.value)

def test_axis_alignment_comes_from_the_coordinates_not_the_heading(tmp_path):
    # a heading offset of -45 degrees gives the diagonal A -> C the axis heading 0; it is still a diagonal
    def edit(data):
        data["segments"].append({"id": 5, "startNodeId": "A", "endNodeId": "C", "speed": 0.5, "heading": [{"degree": -45}]})
    world = _edited_tiny_world(tmp_path, edit)
    assert world.segments[("A", "C")].heading_deg == 0
    with pytest.raises(GridRuleError, match=r"segment A -> C is not axis-aligned"):
        check_grid_rule(world)

def test_check_grid_rule_lattice_tolerance(tmp_path):
    # a coordinate off by 1e-7 m is float noise and fits; one off by 1e-5 m is a real shift
    def move_c(shift):
        return lambda data: data["nodes"][2].update(x=data["nodes"][2]["x"] + shift)
    assert check_grid_rule(_edited_tiny_world(tmp_path / "noise", move_c(1e-7))) is None
    with pytest.raises(GridRuleError, match=r"segment B -> C is not axis-aligned"):
        check_grid_rule(_edited_tiny_world(tmp_path / "shifted", move_c(1e-5)))
