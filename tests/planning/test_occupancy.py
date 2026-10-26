import pytest

from agvtb.planning.intervals import INF
from agvtb.planning.reservations import affects, path_constraints
from helpers import cells_of, evict_world, facing, occupancy_of, rest_at


def moving_path(lw, mv: str, t: float) -> list[tuple[str, float]]:
    """A stop, the move mv from t, and the stop after it."""
    g = lw.graph
    s0 = next(s for s in sorted(g) if g[s].kind == "S" and mv in g[s].next)
    return [(s0, 0.0), (mv, t), (g[mv].next[0], t + g[mv].duration_s)]


def test_soft_entries_are_the_last_element(tmp_path):
    lw, cm = evict_world(tmp_path, "loop")
    path = moving_path(lw, "M,T,B", 1.0)
    occ, cons, _, _ = occupancy_of(lw, cm, cells_of(cm, "A", "B"), {2: path})
    n_last = 1 + len(affects(lw.conflicts, path[-1][0]))
    assert [e for e in cons[2] if e[2] == INF] == path_constraints(path, lw.graph, lw.conflicts)[-n_last:]
    assert occ.rests[2] == path[-1] and occ.homes[2] == cm.cell_of["B"] and occ.owner(cm.cell_of["B"]) == 2


def raw(lw, cm, now, entries):
    """An Occupancy of raw entries {v: [(key, lo, hi)]} only."""
    return occupancy_of(lw, cm, cells_of(cm, "A"), {}, now, entries)[0]


H7 = {7: [("K", 0.0, 5.0)]}  # vehicle 7 holds key K over [0, 5)


@pytest.mark.parametrize("now,soft,locked,with_split,hard_only", [
    (0.0, {8: 8.0}, (), [(5.0, 8.0, ()), (8.0, INF, (8,))], [(5.0, 8.0, ())]),
    (10.0, {8: 8.0}, (), [(5.0, 8.0, ()), (10.0, INF, (8,))], [(5.0, 8.0, ())]),
    (0.0, {8: 3.0}, (), [(5.0, INF, (8,))], []),
    (0.0, {8: 6.0, 9: 7.0}, (), [(5.0, 6.0, ()), (7.0, INF, (8, 9))], [(5.0, 6.0, ())]),
    (0.0, {8: 6.0, 9: 7.0}, (8,), [(5.0, 6.0, ())], [(5.0, 6.0, ())]),
])
def test_the_last_interval_splits_at_the_soft_entries(tmp_path, now, soft, locked, with_split, hard_only):
    lw, cm = evict_world(tmp_path, "loop")
    occ = raw(lw, cm, now, H7 | {v: [("K", lo, INF)] for v, lo in soft.items()})
    assert occ.intervals("K", 1, frozenset(locked), True) == with_split
    assert occ.intervals("K", 1, frozenset(locked), False) == hard_only


def test_a_soft_entry_covers_the_gaps_before_a_later_hard_entry(tmp_path):  # final review C1
    lw, cm = evict_world(tmp_path, "loop")
    occ = raw(lw, cm, 0.0, {7: [("K", 0.0, 5.0), ("K", 10.0, 12.0)], 8: [("K", 3.0, INF)]})
    assert occ.intervals("K", 1, frozenset(), True) == [(12.0, INF, (8,))]  # (5, 10) is not free: 8 rests from 3
    assert occ.intervals("K", 1, frozenset(), False) == []


def test_the_planning_vehicle_does_not_see_itself(tmp_path):
    lw, cm = evict_world(tmp_path, "loop")
    occ = raw(lw, cm, 0.0, H7 | {8: [("K", 6.0, INF)], 9: [("K", 7.0, INF)]})
    assert occ.intervals("K", 8, frozenset(), True) == [(5.0, 7.0, ()), (7.0, INF, (9,))]
    assert occ.intervals("K", 7, frozenset(), True) == [(0.0, 6.0, ()), (7.0, INF, (8, 9))]


def test_a_copy_is_its_own(tmp_path):
    lw, cm = evict_world(tmp_path, "loop")
    occ, _, _, _ = occupancy_of(lw, cm, cells_of(cm, "A", "B"), {1: rest_at(lw, "A"), 2: rest_at(lw, "B")})
    work = occ.copy()
    work.lift(2)
    key = rest_at(lw, "B")[0][0]
    assert occ.rests[2] is not None and occ.owner(cm.cell_of["B"]) == 2
    assert occ.intervals(key, 1, frozenset(), False) == []
    assert work.rests[2] is None and work.owner(cm.cell_of["B"]) is None
    assert work.intervals(key, 1, frozenset(), False) == [(0.0, INF, ())]


def test_commit_and_lift(tmp_path):
    lw, cm = evict_world(tmp_path, "loop")
    g, mv = lw.graph, "M,B,A"
    s_b, s_a = facing(lw, "B", "A"), g[mv].next[0]
    occ, _, _, _ = occupancy_of(lw, cm, cells_of(cm, "A", "B"), {2: [(s_b, 0.0)]})
    occ.lift(2)
    path = [(s_b, 0.0), (mv, 1.0), (s_a, 1.0 + g[mv].duration_s)]
    occ.commit(2, path, rest=False)
    assert occ.rests[2] is None and occ.homes[2] is None and occ.added[2] == path and occ.order == [2]
    assert occ.intervals(s_a, 1, frozenset(), False)[-1][1] == INF  # nobody holds A's stop forever
    occ.commit(2, [path[-1]], rest=True)
    assert occ.rests[2] == path[-1] and occ.homes[2] == cm.cell_of["A"] and occ.owner(cm.cell_of["A"]) == 2
    assert occ.added[2] == path and occ.intervals(s_a, 1, frozenset(), False)[-1][1] != INF


def test_a_second_commit_joins_the_paths(tmp_path):  # Review Focus 4
    lw, cm = evict_world(tmp_path, "loop")
    occ, _, ends, _ = occupancy_of(lw, cm, cells_of(cm, "A", "B", "T"), {2: rest_at(lw, "B")})
    stop = ends[2][0]
    occ.lift(2)
    p1 = [ends[2], (stop, 1.0)]
    occ.commit(2, p1, rest=True)
    occ.lift(2)
    p2 = [(stop, 1.0), (stop, 2.0)]
    occ.commit(2, p2, rest=True)
    assert occ.added[2] == p1 + p2[1:] and occ.order == [2] and occ.rests[2] == p2[-1]
    with pytest.raises(ValueError):
        occ.commit(2, [(rest_at(lw, "T")[0][0], 3.0)], rest=True)  # it must start where the vehicle stands


def test_parking_looks_at_hard_entries_only(tmp_path):  # decision 5
    lw, cm = evict_world(tmp_path, "loop")
    sid = rest_at(lw, "A")[0][0]
    x = affects(lw.conflicts, sid)[0]
    occ = raw(lw, cm, 0.0, {7: [(x, 0.0, 9.0)], 8: [(x, 0.0, INF)]})
    assert not occ.parking_ok(sid, 5.0, 1) and occ.parking_ok(sid, 9.0, 1) and occ.parking_ok(sid, 5.0, 7)
