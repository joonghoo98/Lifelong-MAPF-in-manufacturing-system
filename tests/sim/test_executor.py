import math

import pytest

from agvtb.sim.executor import Executor, wait_cycles, world_geometry
from agvtb.sim.kernel import Kernel
from agvtb.sim.simulator import agent_constraints, live_constraints
from agvtb.sim.tapg import Tapg
from helpers import fake_room_with_spurs, spur_room_world


def run(paths, graph_conflicts=None, geometry=None):
    graph, conflicts = graph_conflicts or fake_room_with_spurs()[:2]
    k, g, advanced = Kernel(), Tapg(graph, conflicts), []
    for vid, p in paths.items():
        g.add_vehicle(vid, p[0][0], p[0][1])
    ex = Executor(graph, g, k, geometry or (lambda sid: (1.0, 0.5)), 0.5, 0.5,
                  lambda vid, i, t: advanced.append((vid, i, t)))
    for vid, p in paths.items():
        g.append(vid, p)
        ex.add(vid)
    k.run_until(60.0, dt=0.5)
    return ex, advanced


def times(ex, vid):
    return [t for _, t in ex.vehicles[vid].executed]


def test_one_cell_then_stop():
    ex, _ = run({1: [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 2.0)]})
    assert [s for s, _ in ex.vehicles[1].executed] == ["S,c00,0", "M,c00,c10", "S,c10,0"]
    assert times(ex, 1) == pytest.approx([0.0, 0.0, 3.0]) and ex.vehicles[1].state == "DONE"


def test_three_straight_cells_pass_through_in_seven_seconds():
    p = [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 2.0), ("M,c10,c20", 2.0), ("S,c20,0", 4.0),
         ("M,c20,c30", 4.0), ("S,c30,0", 6.0)]
    ex, _ = run({1: p})
    assert times(ex, 1) == pytest.approx([0, 0, 2.5, 2.5, 4.5, 4.5, 7.0])


def test_a_load_dwells_its_duration():
    p = [("S,c10,0", 0.0), ("M,c10,c00", 0.0), ("S,c00,0", 2.0), ("L,c00,0", 2.0), ("S,c00,0", 12.0),
         ("M,c00,c10", 12.0), ("S,c10,0", 14.0)]
    ex, advanced = run({1: p})
    assert times(ex, 1) == pytest.approx([0, 0, 3, 3, 13, 13, 16])
    assert (1, 2, pytest.approx(3.0)) in advanced  # on_advance at the arrival before the load


def test_a_turn_stops_rotates_and_starts_again(tmp_path):
    lw = spur_room_world(tmp_path)
    p = [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 1.0), ("R,c10,0,90", 1.0), ("S,c10,90", 3.5),
         ("M,c10,c11", 3.5), ("S,c11,90", 4.5)]
    tri = 4 * math.sqrt(0.5)  # 1 m at 1.0 m/s with 0.5 m/s^2 never cruises
    ex, _ = run({1: p}, (lw.graph, lw.conflicts), world_geometry(lw.world))
    assert times(ex, 1) == pytest.approx([0, 0, tri, tri, tri + 2.5, tri + 2.5, 2 * tri + 2.5])


def test_a_waiting_vehicle_wakes_when_the_blocker_moves_on():  # Review Focus 5
    p1 = [("S,c10,0", 0.0), ("M,c10,c11", 0.0), ("S,c11,0", 2.0)]
    p2 = [("S,c20,0", 0.0), ("M,c20,c10", 3.0), ("S,c10,0", 5.0), ("M,c10,c00", 5.0), ("S,c00,0", 7.0)]
    ex, _ = run({1: p1, 2: p2})
    start = 3.0 + 1e-9  # woken just after vehicle 1 arrives and its nodes go
    assert times(ex, 2) == pytest.approx([0, start, start + 2.5, start + 2.5, start + 5.0], abs=1e-12)


def test_shift_after_prune_keeps_the_last_state():  # DAIM vis:6271-6297
    ex, _ = run({1: [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 2.0)]})
    v = ex.vehicles[1]
    n = min(v.path_idx, len(ex.tapg.paths[1]) - 1)
    ex.tapg.prune(1, n)
    ex.shift(1, n)
    assert ex.tapg.paths[1] == [("S,c10,0", 2.0)] and (v.path_idx, v.claim_idx) == (1, 1)


def test_wait_cycles():
    assert wait_cycles({1: {2}, 2: {1}}) == [(1, 2)]
    assert wait_cycles({3: {1}, 1: {2}, 2: {3}, 4: {1}}) == [(1, 2, 3)]
    assert wait_cycles({1: {2}, 2: set()}) == []


def test_two_recomputes_during_one_move_keep_its_real_end():  # final review I1, Review Focus 4
    graph, conflicts = fake_room_with_spurs()[:2]
    k, g = Kernel(), Tapg(graph, conflicts)
    g.add_vehicle(1, "S,c00,0", 0.0)
    ex = Executor(graph, g, k, lambda sid: (1.0, 0.5), 0.5, 0.5, lambda *a: None)
    g.append(1, [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 2.0)])
    ex.add(1)
    for t in (0.5, 2.0):
        k.run_until(t, dt=0.5)
        g.recompute(t, ex.inprog_end())
    assert g.paths[1][-1] == ("S,c10,0", 3.0)


def test_wait_cycles_follow_every_blocker():  # final review I2
    assert wait_cycles({1: {2, 3}, 3: {1}}) == [(1, 3)]
    assert wait_cycles({1: {2}, 2: {1, 3}, 3: {4}, 4: {3}}) == [(1, 2), (3, 4)]


def test_revocable_only_beyond_the_entered_and_claimed_states():  # DAIM env:1561-1565
    ex, _ = run({1: [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 2.0)]})
    v = ex.vehicles[1]
    v.path_idx, v.claim_idx = 2, 4
    assert [ex.revocable(1, k) for k in (2, 3, 4, 5)] == [False, False, True, True]


def test_retry_only_for_a_vehicle_blocked_on_a_stop():  # DAIM vis:5583-5586 (A2 final review C1)
    ex, _ = run({1: [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 2.0)]})
    v, n = ex.vehicles[1], len(ex.tapg.paths[1])
    assert v.state == "DONE" and not ex.retry(1, 60.0)
    v.state, v.dwelling, v.path_idx = "WAITING", True, n - 1
    assert not ex.retry(1, 60.0)                 # dwelling at a port: its load end wakes it
    v.dwelling, v.path_idx = False, n
    assert not ex.retry(1, 60.0)                 # nothing left to enter
    v.path_idx = n - 1
    assert ex.retry(1, 60.0)                     # blocked on a stop: it looks again now


def blocked_on_c10():
    """Vehicle 2's planned pass through c11 (not executed: vehicle 2 is not in the executor) comes before vehicle 1's
    move c10 -> c11, so vehicle 1 waits on S,c10 for good (plan review 1, important 2)."""
    graph, conflicts = fake_room_with_spurs()[:2]
    k, g = Kernel(), Tapg(graph, conflicts)
    g.add_vehicle(1, "S,c10,0", 0.0)
    g.add_vehicle(2, "S,c21,0", 0.0)
    ex = Executor(graph, g, k, lambda sid: (1.0, 0.5), 0.5, 0.5, lambda *a: None)
    g.append(1, [("S,c10,0", 0.0), ("M,c10,c11", 10.0), ("S,c11,0", 12.0), ("M,c11,c12", 12.0), ("S,c12,0", 14.0)])
    g.append(2, [("S,c21,0", 0.0), ("M,c21,c11", 0.0), ("S,c11,0", 2.0), ("M,c11,c01", 2.0), ("S,c01,0", 4.0)])
    ex.add(1)
    k.run_until(5.0, dt=0.5)
    return k, g, ex


def test_a_vehicle_blocked_on_its_splice_moves_only_if_retried():  # DAIM final review C1; Review Focus 1
    for retried in (True, False):
        k, g, ex = blocked_on_c10()
        v = ex.vehicles[1]
        assert v.state == "WAITING" and v.path_idx == 0 and not v.dwelling
        g.revoke_beyond(1, 1)                     # the cut at the splice S,c10, another way on
        g.append(1, [("S,c10,0", 0.0), ("M,c10,c00", 5.0), ("S,c00,0", 7.0)])
        if retried:
            assert ex.retry(1, 5.0)
        k.run_until(60.0, dt=0.5)
        assert (v.state == "DONE") is retried


def test_live_constraints_are_each_vehicles_own():  # the A2 batch update rebuilds one vehicle (DAIM vis:5437-5438)
    ex, _ = run({1: [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 2.0)],
                 2: [("S,c20,0", 0.0), ("M,c20,c21", 0.0), ("S,c21,0", 2.0)]})
    assert live_constraints(ex.tapg, ex, 3.0) == {vid: agent_constraints(ex.tapg, ex, vid, 3.0) for vid in (1, 2)}
