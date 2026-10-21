import ast
from pathlib import Path

import agvtb.validate.plans as plans_module
from agvtb.planning.reservations import ReservationTable
from agvtb.planning.safe_tail import SafeTailPlanner
from agvtb.validate.plans import (conflicting, early_fallback, live_entries, occupancy, pairwise_overlaps,
                                  path_vs_entries, tail_violations, task_order_violation, unprotected)
from helpers import fake_room_with_spurs

INF = float("inf")
A = [("S,c00,0", 0.0), ("M,c00,c10", 5.0), ("S,c10,0", 7.0)]
B = [("S,c20,0", 0.0), ("M,c20,c10", 6.0), ("S,c10,0", 8.0)]


# DAIM tests/test_plan_validator.py line numbers; exact outputs from validator_expected.out.txt
def test_conflict_rules():  # :25
    _graph, conflicts, _ports = fake_room_with_spurs()
    assert conflicting(conflicts, "S,c00,0", "S,c00,0")
    assert conflicting(conflicts, "L,c00,0", "S,c00,0")
    assert conflicting(conflicts, "S,c00,0", "M,c10,c00")
    assert not conflicting(conflicts, "S,c00,0", "S,c20,0")
    assert conflicting(conflicts, "S,c00,0", "S,c00,90")  # two stops at one node, whatever their poses


def test_occupancy():
    assert occupancy(A) == [("S,c00,0", 0.0, 5.0), ("M,c00,c10", 5.0, 7.0), ("S,c10,0", 7.0, INF)]
    assert occupancy(A, 6.0) == [("M,c00,c10", 5.0, 7.0), ("S,c10,0", 7.0, INF)]


def test_pairwise_overlaps_finds_shared_cell_and_ignores_disjoint():  # :34
    _graph, conflicts, _ports = fake_room_with_spurs()
    assert pairwise_overlaps({1: A, 2: B}, conflicts) == [
        {"a": 1, "b": 2, "sa": "M,c00,c10", "sb": "M,c20,c10", "t": 6.0},
        {"a": 1, "b": 2, "sa": "S,c10,0", "sb": "M,c20,c10", "t": 7.0},
        {"a": 1, "b": 2, "sa": "S,c10,0", "sb": "S,c10,0", "t": 8.0},
    ]
    assert pairwise_overlaps({1: A, 2: [("S,c40,0", 0.0)]}, conflicts) == []
    two_parked = {1: [("S,c03,0", 0.0)], 2: [("S,c02,0", 0.0), ("M,c02,c03", 3.0), ("S,c03,0", 5.0)]}
    assert pairwise_overlaps(two_parked, conflicts) == [
        {"a": 1, "b": 2, "sa": "S,c03,0", "sb": "M,c02,c03", "t": 3.0},
        {"a": 1, "b": 2, "sa": "S,c03,0", "sb": "S,c03,0", "t": 5.0},
    ]


def test_tail_violations():  # :55
    sidings = ["c03", "c23", "c43"]
    v = tail_violations({1: [("S,c03,0", 0.0)], 2: [("S,c50,0", 0.0)]}, {1: "c03", 2: "c23"}, sidings)
    assert v == ["vehicle 2: its path ends at c50, which is no siding"]
    v2 = tail_violations({1: [("S,c03,0", 0.0)]}, {1: "c03", 2: "c03"}, ["c03"])
    assert v2 == ["siding c03 is owned by more than one vehicle"]
    v3 = tail_violations({1: [("S,c43,0", 0.0)]}, {1: "c03"}, ["c03", "c43"])
    assert v3 == ["vehicle 1: its path ends at c43, not at its own siding c03"]
    v4 = tail_violations({1: [("M,c03,c02", 0.0)]}, {1: "c03"}, ["c03"])
    assert v4 == ["vehicle 1: its path ends on M,c03,c02, which is no stop state"]


def test_task_order():  # :64
    ok = [("S,c03,0", 0.0), ("L,c00,0", 6.0), ("S,c00,0", 16.0), ("L,c50,0", 26.0), ("S,c50,0", 36.0),
          ("S,c43,0", 44.0)]
    assert task_order_violation(ok, ["c00", "c50"], ["c43"]) is None
    bad = [("S,c03,0", 0.0), ("L,c50,0", 6.0), ("S,c50,0", 16.0), ("L,c00,0", 26.0), ("S,c00,0", 36.0),
           ("S,c43,0", 44.0)]
    assert task_order_violation(bad, ["c00", "c50"], ["c43"]) is not None
    assert task_order_violation(ok[:-1], ["c00", "c50"], ["c43"]) is not None  # does not end on a siding


def test_early_fallback():  # DAIM safe-tail design §9.1 (5): a fallback leaves no earlier than T*
    path = [("S,c03,0", 0.0), ("M,c03,c02", 50.0), ("S,c02,0", 52.0)]
    assert not early_fallback(path, 50.0) and early_fallback(path, 50.5)
    assert not early_fallback([("S,c03,0", 0.0)], 50.0)


def test_a_planned_path_does_not_overlap_the_reservations_it_avoided():  # DAIM tests/test_safe_tail_planner.py:61
    graph, conflicts, _ports = fake_room_with_spurs()
    other = [("S,c50,0", 0.0), ("M,c50,c40", 40.0), ("S,c40,0", 42.0), ("M,c40,c41", 42.0), ("S,c41,0", 44.0),
             ("M,c41,c42", 44.0), ("S,c42,0", 46.0), ("M,c42,c43", 46.0), ("S,c43,0", 48.0)]
    table = ReservationTable(graph, conflicts)
    table.set_path(7, other)
    res = SafeTailPlanner(graph, conflicts, ["c03", "c23", "c43"]).search(
        "S,c03,0", 0.0, ["c00", "c50"], table.table(), ["c03", "c23"])
    assert pairwise_overlaps({1: res.path, 7: other}, conflicts) == []


def test_the_validator_does_not_import_the_planner():  # design §11
    tree = ast.parse(Path(plans_module.__file__).read_text(encoding="utf-8"))
    names = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any(name.startswith("agvtb.planning") for name in names)


RAW = [("S,c00,0", 0.0), ("M,c00,c10", 0.0), ("S,c10,0", 2.0), ("M,c10,c20", 2.0), ("S,c20,0", 4.0)]


def test_live_entries_hold_the_current_and_claimed_states_from_now():  # DAIM plan_validator.py:68-83
    assert live_entries(RAW, 1, 3, 1.5) == [("M,c00,c10", 1.5, 2.0), ("S,c10,0", 1.5, 2.0),
                                            ("M,c10,c20", 2.0, 4.0), ("S,c20,0", 4.0, INF)]
    assert live_entries(RAW, 9, 9, 6.0) == [("S,c20,0", 6.0, INF)]
    assert live_entries(RAW, 0, 0, 3.0)[0] == ("S,c00,0", 3.0, 3.0) and live_entries([], 0, 0, 1.0) == []


def test_path_vs_entries_is_one_side_of_pairwise_overlaps():  # validator_expected.out.txt (M3)
    _graph, conflicts, _ports = fake_room_with_spurs()
    got = path_vs_entries(A, {2: occupancy(B)}, conflicts)
    assert got == [{k: v for k, v in o.items() if k != "a"} for o in pairwise_overlaps({1: A, 2: B}, conflicts)]
    assert len(got) == 3


def test_unprotected_finds_conflicting_states_with_no_order():  # DAIM plan_validator.py:125-148
    _graph, conflicts, _ports = fake_room_with_spurs()
    mine = [(0, "S,c10,0", "a")]
    others = {2: [(0, "M,c20,c10", "b"), (1, "S,c40,0", "c")]}
    assert unprotected(mine, others, lambda n: set(), conflicts) == [
        {"i": 0, "b": 2, "j": 0, "sa": "S,c10,0", "sb": "M,c20,c10"}]
    assert unprotected(mine, others, lambda n: {"b"}, conflicts) == []
