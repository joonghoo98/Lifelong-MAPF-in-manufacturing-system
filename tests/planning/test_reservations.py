import math

import pytest

from agvtb.planning.intervals import EPS, INF, interval_at, merge_intervals, safe_intervals
from agvtb.planning.reservations import ReservationTable, affects, path_constraints
from helpers import fake_room_with_spurs

# reservations_expected.out.txt: DAIM _build_constraints + _make_constraint_table (sipp_planner.py:296-335)
MOVE = [("S,c40,0", 0.0), ("M,c40,c41", 3.0), ("S,c41,0", 5.0)]
MOVE_TABLE = {
    "M,c30,c40": [(0.0, 3.0), (1.0, 5.0)], "M,c31,c41": [(1.0, 5.0), (3.0, INF)],
    "M,c40,c30": [(0.0, 3.0), (1.0, 5.0)], "M,c40,c41": [(0.0, 3.0), (3.0, 5.0), (3.0, INF)],
    "M,c40,c50": [(0.0, 3.0), (1.0, 5.0)], "M,c41,c31": [(1.0, 5.0), (3.0, INF)],
    "M,c41,c40": [(0.0, 3.0), (1.0, 5.0), (3.0, INF)], "M,c41,c42": [(1.0, 5.0), (3.0, INF)],
    "M,c41,c51": [(1.0, 5.0), (3.0, INF)], "M,c42,c41": [(1.0, 5.0), (3.0, INF)],
    "M,c50,c40": [(0.0, 3.0), (1.0, 5.0)], "M,c51,c41": [(1.0, 5.0), (3.0, INF)],
    "S,c40,0": [(0.0, 3.0), (3.0, 5.0)], "S,c41,0": [(3.0, 5.0), (5.0, INF)],
}
LOAD = [("S,c00,0", 0.0), ("L,c00,0", 0.0), ("S,c00,0", 10.0)]
LOAD_TABLE = {
    "M,c00,c01": [(0.0, 10.0), (8.0, INF)], "M,c00,c10": [(0.0, 10.0), (8.0, INF)],
    "M,c01,c00": [(0.0, 10.0), (8.0, INF)], "M,c10,c00": [(0.0, 10.0), (8.0, INF)],
    "S,c00,0": [(0.0, 10.0), (10.0, INF)],
}
ZERO = [("S,c40,0", 0.0), ("M,c40,c41", 0.0), ("S,c41,0", 2.0)]
ZERO_TABLE = {
    "M,c30,c40": [(0.0, 2.0)], "M,c31,c41": [(0.0, 2.0), (0.0, INF)], "M,c40,c30": [(0.0, 2.0)],
    "M,c40,c41": [(0.0, 2.0), (0.0, INF)], "M,c40,c50": [(0.0, 2.0)], "M,c41,c31": [(0.0, 2.0), (0.0, INF)],
    "M,c41,c40": [(0.0, 2.0), (0.0, INF)], "M,c41,c42": [(0.0, 2.0), (0.0, INF)],
    "M,c41,c51": [(0.0, 2.0), (0.0, INF)], "M,c42,c41": [(0.0, 2.0), (0.0, INF)], "M,c50,c40": [(0.0, 2.0)],
    "M,c51,c41": [(0.0, 2.0), (0.0, INF)], "S,c40,0": [(0.0, 2.0)], "S,c41,0": [(0.0, 2.0), (2.0, INF)],
}
EARLY = [("S,c10,0", 0.5), ("M,c10,c20", 1.0), ("S,c20,0", 3.0)]


def table_of(path):
    graph, conflicts, _ports = fake_room_with_spurs()
    table = ReservationTable(graph, conflicts)
    table.set_path(1, path)
    return table.table()


def test_merge_joins_overlapping_and_touching_intervals():  # DAIM sipp_planner.py:86-96
    assert merge_intervals([(5.0, 7.0), (1.0, 3.0), (3.0, 4.0), (6.0, 9.0)]) == [(1.0, 4.0), (5.0, 9.0)]
    assert merge_intervals([]) == []


def test_safe_intervals_are_the_gaps():  # DAIM sipp_planner.py:99-109
    assert safe_intervals([]) == [(0.0, INF)]
    assert safe_intervals([(2.0, 3.0), (5.0, INF)]) == [(0.0, 2.0), (3.0, 5.0)]
    assert safe_intervals([(0.0, 1.0)]) == [(1.0, INF)]
    assert safe_intervals([(1.0, 2.0)], lo=1.5, hi=4.0) == [(2.0, 4.0)]


def test_interval_at_is_half_open_with_eps():  # DAIM safe_tail_planner.py:269-274
    ivs = [(0.0, 2.0), (3.0, 5.0)]
    assert interval_at(ivs, 1.5) == (0.0, 2.0)
    assert interval_at(ivs, 2.0 - EPS / 10) is None
    assert interval_at(ivs, 3.0 - EPS / 10) == (3.0, 5.0)
    assert interval_at(ivs, 5.0) is None and EPS == 1e-9 and INF == math.inf


def test_a_path_reserves_its_keys_and_blocks_its_affects_their_length_earlier():
    assert table_of(MOVE) == MOVE_TABLE


def test_a_load_takes_the_place_of_its_stop_and_a_stop_of_no_length_is_skipped():
    assert table_of(LOAD) == LOAD_TABLE
    assert table_of(ZERO) == ZERO_TABLE


def test_the_table_leaves_out_the_planning_vehicle_and_starts_no_earlier_than_zero():
    graph, conflicts, _ports = fake_room_with_spurs()
    table = ReservationTable(graph, conflicts)
    table.set_path(1, MOVE)
    table.set_path(2, EARLY)
    only2 = ReservationTable(graph, conflicts)
    only2.set_path(2, EARLY)
    assert table.table(exclude=1) == only2.table()
    assert table.table(exclude=1)["S,c10,0"] == [(0.5, 1.0), (1.0, 3.0)]
    assert table.table(exclude=1)["M,c00,c10"] == [(0.0, 1.0), (0.0, 3.0)]  # max(0, 0.5 - 2.0)
    table.set_path(1, EARLY)  # a vehicle's new path replaces its old reservations
    assert table.vehicles == (1, 2) and table.table(exclude=2) == only2.table()


def test_affects_leave_out_l_and_an_l_blocks_what_its_stop_blocks():
    _graph, conflicts, _ports = fake_room_with_spurs()
    stop = ("M,c00,c01", "M,c00,c10", "M,c01,c00", "M,c10,c00")
    assert affects(conflicts, "L,c00,0") == affects(conflicts, "S,c00,0") == stop


def test_path_constraints_and_unknown_states():
    graph, conflicts, _ports = fake_room_with_spurs()
    assert path_constraints([("S,c03,0", 0.0)], graph, conflicts) == [
        ("S,c03,0", 0.0, INF), ("M,c02,c03", 0.0, INF), ("M,c03,c02", 0.0, INF)]
    with pytest.raises(KeyError):
        path_constraints([("S,zz,0", 0.0)], graph, conflicts)
