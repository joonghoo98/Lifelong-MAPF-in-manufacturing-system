import dataclasses

import pytest

from agvtb.world.collision_grid import GridRuleError, check_grid_rule, grid_conflicts, grid_pitch
from agvtb.world.model import Footprint
from agvtb.world.states import build_state_graph, geom_key
from helpers import tiny_world, world_of

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
