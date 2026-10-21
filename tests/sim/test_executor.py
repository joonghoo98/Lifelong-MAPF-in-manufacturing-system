import math

import pytest

from agvtb.sim.executor import Executor, wait_cycles, world_geometry
from agvtb.sim.kernel import Kernel
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
