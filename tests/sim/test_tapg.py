import pytest

from agvtb.sim.tapg import Tapg
from helpers import fake_room_with_spurs

P1 = [("S,c10,0", 0.0), ("M,c10,c11", 0.0), ("S,c11,0", 2.0)]
P2 = [("S,c20,0", 0.0), ("M,c20,c10", 3.0), ("S,c10,0", 5.0), ("M,c10,c00", 5.0), ("S,c00,0", 7.0)]


def two(p1, p2):
    graph, conflicts, _ports = fake_room_with_spurs()
    g = Tapg(graph, conflicts)
    g.add_vehicle(1, p1[0][0], p1[0][1])
    g.add_vehicle(2, p2[0][0], p2[0][1])
    g.append(1, p1)
    g.append(2, p2)
    return g


def test_one_cross_edge_from_the_move_that_frees_the_cell():
    g = two(P1, P2)
    assert g.cross_edges() == [(("M,c10,c11", 1, 0.0), ("M,c20,c10", 2, 3.0))]
    assert g.paths[2] == P2 and g.duration[("S,c20,0", 2, 0.0)] == 3.0 and not g.has_cycle()


def test_an_s_target_also_gets_an_edge_to_the_move_before_it():
    p1 = [("S,c10,0", 0.0), ("M,c10,c11", 0.0), ("S,c11,0", 2.0), ("M,c11,c12", 2.0), ("S,c12,0", 4.0)]
    p2 = [("S,c20,0", 0.0), ("M,c20,c10", 1.0), ("S,c10,0", 3.0)]
    assert two(p1, p2).cross_edges() == [
        (("M,c10,c11", 1, 0.0), ("M,c20,c10", 2, 1.0)),
        (("M,c10,c11", 1, 0.0), ("S,c10,0", 2, 3.0)),
        (("S,c10,0", 1, 0.0), ("M,c20,c10", 2, 1.0)),
    ]


def test_two_zero_length_stops_at_one_time_get_one_edge_from_the_lower_id():  # env:1697-1721
    p1 = [("S,c01,0", 0.0), ("M,c01,c11", 0.0), ("S,c11,0", 2.0), ("M,c11,c21", 2.0), ("S,c21,0", 4.0)]
    p2 = [("S,c12,0", 0.0), ("M,c12,c11", 0.0), ("S,c11,0", 2.0), ("M,c11,c10", 2.0), ("S,c10,0", 4.0)]
    edges = two(p1, p2).cross_edges()
    assert (("S,c11,0", 1, 2.0), ("S,c11,0", 2, 2.0)) in edges
    assert (("S,c11,0", 2, 2.0), ("S,c11,0", 1, 2.0)) not in edges


def test_claim_remove_and_wake():
    g = two(P1, P2)
    assert not g.claimable(g.key(2, 1), 2) and g.claimable(g.key(1, 1), 1)
    assert g.remove(g.key(1, 1)) == [2]
    assert g.claimable(g.key(2, 1), 2) and g.claimable(("M,zz,zz", 2, 0.0), 2)
    assert g.prune(1, 1) == [] and g.paths[1] == [("M,c10,c11", 0.0), ("S,c11,0", 2.0)]


def test_related_is_ancestors_and_descendants():
    g = two(P1, P2)
    assert g.key(1, 1) in g.related(g.key(2, 1)) and g.key(2, 4) in g.related(g.key(2, 1))
    assert g.key(1, 0) in g.related(g.key(2, 1))  # through vehicle 1's own chain
    assert g.key(1, 2) not in g.related(g.key(2, 1))


def test_recompute_pushes_waiters_after_the_move_they_wait_for():  # research §B5
    g = two(P1, P2)
    g.recompute(0.0, {})
    assert g.paths[2] == [("S,c20,0", 0.0), ("M,c20,c10", 2.0), ("S,c10,0", 4.0), ("M,c10,c00", 4.0),
                          ("S,c00,0", 6.0)]
    assert g.cross_edges() == [(("M,c10,c11", 1, 0.0), ("M,c20,c10", 2, 2.0))]
    g.recompute(10.0, {})
    assert g.paths[1][0] == ("S,c10,0", 10.0) and g.paths[2][1] == ("M,c20,c10", 12.0)


def test_recompute_keeps_the_real_end_of_a_move():  # Review Focus 4
    g = two(P1, P2)
    g.recompute(1.0, {g.key(1, 1): 4.5})  # vehicle 1's move really ends at 4.5
    assert g.paths[1] == [("S,c10,0", 1.0), ("M,c10,c11", 1.0), ("S,c11,0", 4.5)]
    assert g.paths[2][1] == ("M,c20,c10", 4.5)


def test_a_dwell_stop_keeps_its_length():
    graph, conflicts, _ports = fake_room_with_spurs()
    g = Tapg(graph, conflicts)
    g.add_vehicle(1, "S,c10,0", 0.0)
    path = [("S,c10,0", 0.0), ("M,c10,c00", 0.0), ("S,c00,0", 2.0), ("L,c00,0", 5.0), ("S,c00,0", 15.0)]
    g.append(1, path)
    g.mark_dwell(1, 2)  # the S before L, as DAIM's commit marks it (research §A6-3)
    g.recompute(0.0, {})
    assert g.paths[1] == path


def test_append_must_continue_the_path():
    graph, conflicts, _ports = fake_room_with_spurs()
    g = Tapg(graph, conflicts)
    g.add_vehicle(1, "S,c10,0", 0.0)
    with pytest.raises(ValueError, match="S,c20,0"):
        g.append(1, [("S,c20,0", 0.0), ("M,c20,c21", 0.0), ("S,c21,0", 2.0)])
