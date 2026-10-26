from collections import Counter

from agvtb.planning.intervals import EPS, INF
from agvtb.planning.split_sipp import SplitSipp, target_locks
from helpers import cells_of, evict_world, facing, occupancy_of, rest_at


def load_time(found):
    return next(t for sid, t in found.path if sid.startswith("L,"))


def test_the_eviction_branch_keeps_its_early_arrival(tmp_path):  # CE-1, spec test 1 (spec line 256)
    lw, cm = evict_world(tmp_path, "ce1")
    rest, sipp = cells_of(cm, "A", "X"), SplitSipp(lw.graph, lw.conflicts, cm, Counter())
    start, me, dwell = rest_at(lw, "A")[0][0], frozenset({1}), sipp.dwell_s
    alone, _, _, _ = occupancy_of(lw, cm, rest, {1: rest_at(lw, "A")})
    a_x = load_time(sipp.staged(alone, 1, start, 0.0, ["N"], me, False))
    held, _, _, _ = occupancy_of(lw, cm, rest, {1: rest_at(lw, "A"), 2: rest_at(lw, "X")})
    a_d = load_time(sipp.staged(held, 1, start, 0.0, ["N"], me, False))
    assert a_d > a_x + 1.0  # the detour is later
    window = [(sid, lo, hi) for sid in sorted(lw.graph) if lw.graph[sid].kind == "S" and lw.graph[sid].nodes[0] == "N"
              for lo, hi in ((0.0, a_x), (a_x + dwell + 1.0, 1e6))]
    occ, _, _, _ = occupancy_of(lw, cm, rest, {1: rest_at(lw, "A"), 2: rest_at(lw, "X")}, 0.0, {9: window})
    found = sipp.staged(occ, 1, start, 0.0, ["N"], me, True)
    assert found is not None and found.blocking == ((2, occ.rests[2]),)
    assert a_x - EPS <= load_time(found) and load_time(found) + dwell < a_x + dwell + 1.0


def test_hard_only_goes_around_a_resting_vehicle(tmp_path):
    lw, cm = evict_world(tmp_path, "ce1")
    sipp = SplitSipp(lw.graph, lw.conflicts, cm, Counter())
    occ, _, _, _ = occupancy_of(lw, cm, cells_of(cm, "A", "X"), {1: rest_at(lw, "A"), 2: rest_at(lw, "X")})
    found = sipp.staged(occ, 1, rest_at(lw, "A")[0][0], 0.0, ["N"], frozenset({1}), False)
    assert found is not None and found.blocking == ()
    assert all("X" not in lw.graph[sid].nodes for sid, _ in found.path)


def test_blocking_lists_the_last_met_first(tmp_path):
    lw, cm = evict_world(tmp_path, "chain")
    sipp = SplitSipp(lw.graph, lw.conflicts, cm, Counter())
    at = {0: "A", 1: "B", 2: "C", 3: "D"}
    occ, _, _, _ = occupancy_of(lw, cm, cells_of(cm, *at.values()), {v: rest_at(lw, n) for v, n in at.items()})
    found = sipp.staged(occ, 0, rest_at(lw, "A")[0][0], 0.0, ["E"], frozenset({0}), True)
    assert [v for v, _ in found.blocking] == [3, 2, 1] and all(seen == occ.rests[v] for v, seen in found.blocking)


def test_nearest_rest_prefers_a_low_degree_cell(tmp_path):  # Φ (design §8)
    lw, cm = evict_world(tmp_path, "fork")
    sipp = SplitSipp(lw.graph, lw.conflicts, cm, Counter())
    occ, _, _, _ = occupancy_of(lw, cm, cells_of(cm, "F", "G"), {})
    found = sipp.nearest_rest(occ, 1, facing(lw, "J", "E"), 0.0, {}, frozenset({1}), True)
    assert lw.graph[found.path[-1][0]].nodes[0] == "F"


def test_nearest_rest_keeps_out_of_owned_and_locked_cells(tmp_path):
    lw, cm = evict_world(tmp_path, "fork")
    sipp = SplitSipp(lw.graph, lw.conflicts, cm, Counter())
    occ, _, _, _ = occupancy_of(lw, cm, cells_of(cm, "F", "G", "G2"), {2: rest_at(lw, "F")})
    for split in (False, True):
        found = sipp.nearest_rest(occ, 1, facing(lw, "J", "E"), 0.0, {cm.cell_of["G"]: 100.0}, frozenset({1}), split)
        assert lw.graph[found.path[-1][0]].nodes[0] == "G2" and found.blocking == ()


def test_a_vehicle_cannot_rest_where_it_stood_through_the_lock(tmp_path):  # spec line 173 (c) on cells
    lw, cm = evict_world(tmp_path, "fork")
    sipp = SplitSipp(lw.graph, lw.conflicts, cm, Counter())
    occ, _, _, _ = occupancy_of(lw, cm, cells_of(cm, "E", "G2"), {})
    found = sipp.nearest_rest(occ, 1, facing(lw, "G2", "G"), 0.0, {cm.cell_of["G2"]: 4.0}, frozenset({1}), False)
    # turning round on G2 (5 s) would end after the lock, and be cheaper than E; but it never left G2
    assert lw.graph[found.path[-1][0]].nodes[0] == "E"


def test_a_split_search_does_not_pass_a_resting_vehicle_unseen(tmp_path):  # final review C1
    lw, cm = evict_world(tmp_path, "loop")
    sipp = SplitSipp(lw.graph, lw.conflicts, cm, Counter())
    occ, _, _, _ = occupancy_of(lw, cm, cells_of(cm, "A", "B"), {1: rest_at(lw, "A")}, 0.0,
                                {7: [("M,A,B", 20.0, 25.0)], 8: [("M,A,B", 0.0, INF)]})
    found = sipp.staged(occ, 1, rest_at(lw, "A")[0][0], 0.0, ["P"], frozenset({1}), True)
    used = [t for sid, t in found.path if sid == "M,A,B"]
    assert not used or 8 in [v for v, _ in found.blocking]  # 8 holds M,A,B from 0: pass it only by evicting 8


def test_a_search_over_budget_fails(tmp_path):
    lw, cm = evict_world(tmp_path, "ce1")
    stats = Counter()
    sipp = SplitSipp(lw.graph, lw.conflicts, cm, stats, budget=1)
    occ, _, _, _ = occupancy_of(lw, cm, cells_of(cm, "A"), {1: rest_at(lw, "A")})
    assert sipp.staged(occ, 1, rest_at(lw, "A")[0][0], 0.0, ["N"], frozenset({1}), True) is None
    assert stats["m6_search_budget"] == 1


def test_target_locks_are_the_last_visits(tmp_path):
    lw, cm = evict_world(tmp_path, "fork")
    g = lw.graph
    path = [(facing(lw, "J", "E"), 0.0), ("M,J,E", 1.0), (g["M,J,E"].next[0], 4.0)]
    assert target_locks(path, cm, g) == {cm.cell_of["J"]: 4.0, cm.cell_of["E"]: INF}


def test_the_exit_time_lets_the_vehicle_leave_its_cell_any_way(tmp_path):  # decision 2
    lw, cm = evict_world(tmp_path, "fork")
    g, sipp = lw.graph, SplitSipp(lw.graph, lw.conflicts, cm, Counter())
    turn = g[next(x for x in g[facing(lw, "J", "E")].next if x.startswith("R,") and "M,J,F" in g[g[x].next[0]].next)]
    assert sipp.exit_s(facing(lw, "G2", "G")) == 2 * g["M,G2,G"].duration_s  # one way out: the move, then as long
    # E is straight ahead; F and G take a quarter turn and a move: cleared once the turn has ended and lasted again
    assert sipp.exit_s(facing(lw, "J", "E")) == max(2 * turn.duration_s, turn.duration_s + 2 * g["M,J,F"].duration_s)
