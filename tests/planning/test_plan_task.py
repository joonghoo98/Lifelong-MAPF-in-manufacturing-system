import pytest

from agvtb.planning.intervals import INF
from agvtb.planning.ledger import SidingLedger
from agvtb.planning.reservations import ReservationTable
from agvtb.planning.safe_tail import SafeTailPlanner
from agvtb.sidings.lattice import grid_sidings_of_world
from agvtb.world.loader import load_world
from helpers import CONFIGS, fake_room_with_spurs, path_digest

SIDINGS = ["c03", "c23", "c43"]


def planner(budget=None):
    graph, conflicts, _ports = fake_room_with_spurs()
    return SafeTailPlanner(graph, conflicts, SIDINGS, budget=budget)


def loads(path):
    return [(s, t) for s, t in path if s.startswith("L,")]


# DAIM tests/test_safe_tail_planner.py line numbers; values and digests from a1_digests.out.txt (A == B)
def test_plan_task_counts_a_found_plan():  # :42-50
    pl = planner()
    res = pl.plan_task("S,c03,0", 0.0, ["c00", "c50"], {}, SIDINGS, "c03")
    assert path_digest(res.path) == "a2bd5a2b821d643b2c05121e901cc946a6fff61142e086de9f3e28b7bb41698e"
    assert dict(pl.stats) == {"search_ok": 1} and pl.expansions_ok == [17]
    assert pl.path_conflicts(res.path, {}) == []


def test_budget_exhaustion_falls_back_to_a_wait_then_static_route_home():  # :73
    pl = planner(budget=1)
    table = {"S,c22,0": [(10.0, 50.0)]}
    res = pl.plan_task("S,c03,0", 0.0, ["c00", "c50"], table, SIDINGS, "c03")
    assert (res.fallback, res.siding, res.t_star, res.expansions) == (True, "c03", 50.0, 0)
    assert res.path[0] == ("S,c03,0", 0.0) and res.path[1] == ("M,c03,c02", 50.0)
    assert res.path[-1] == ("S,c03,0", 102.0) and loads(res.path) == [("L,c00,0", 56.0), ("L,c50,0", 76.0)]
    assert pl.path_conflicts(res.path, table) == []
    assert dict(pl.stats) == {"budget_exceeded": 1, "fallback_used": 1}
    assert path_digest(res.path) == "6078b0f3304d9a6e3541e2624ac55aa9d6951c83ae77b1e944ca8aa7cbbdf6e5"


def test_port_blocked_forever_makes_both_search_and_fallback_fail():  # :86
    pl = planner()
    assert pl.plan_task("S,c03,0", 0.0, ["c00", "c50"], {"S,c00,0": [(0.0, INF)]}, SIDINGS, "c03") is None
    assert dict(pl.stats) == {"search_exhausted": 1, "fallback_fail": 1}


def test_start_blocked_makes_the_fallback_invalid_too():  # :92, Review Focus 2
    pl = planner()
    assert pl.plan_task("S,c03,0", 0.0, ["c00", "c50"], {"S,c03,0": [(0.0, 30.0)]}, SIDINGS, "c03") is None
    assert dict(pl.stats) == {"start_blocked": 1, "fallback_invalid": 1, "fallback_fail": 1}


def test_fallback_needs_a_home_and_a_stop_to_start_from():
    pl = planner()
    assert pl.fallback("S,c03,0", 0.0, ["c00"], {}, None) is None
    assert pl.fallback("M,c03,c02", 0.0, ["c00"], {}, "c03") is None and dict(pl.stats) == {}


def test_path_conflicts_look_at_the_affects_of_the_last_state_only():
    pl = planner()
    table = {"M,c41,c40": [(10.0, 12.0)]}
    assert pl.path_conflicts([("S,c41,0", 0.0)], table) == [(0, "M,c41,c40")]
    assert pl.path_conflicts([("S,c41,0", 0.0), ("M,c41,c31", 1.0), ("S,c31,0", 3.0)], table) == []


def test_mini_fallback_matches_daim():  # a1_digests.out.txt C, mini fallback
    lw = load_world(CONFIGS / "mini_deadend.json")
    sidings = sorted(grid_sidings_of_world(lw.world, "bridge", (0, 0)))
    pl = SafeTailPlanner(lw.graph, lw.conflicts, sidings, budget=1)
    table = {"S,G11_10,0": [(5.0, 40.0)]}
    res = pl.plan_task("S,G10_10,0", 0.0, ["G11_10", "G9_1"], table, sidings, "G10_10")
    assert (res.fallback, res.t_star, res.path[1]) == (True, 40.0, ("M,G10_10,G11_10", 40.0))
    assert res.path[-1] == ("S,G10_10,180", 106.44885177453028)
    assert path_digest(res.path) == "8cad1b073db3cea3490c5dab121e51548e7db996466e7e184541d0c3db2b9cc0"


def test_batch_of_two_tasks_uses_distinct_sidings():  # :114
    graph, conflicts, _ports = fake_room_with_spurs()
    pl = SafeTailPlanner(graph, conflicts, SIDINGS)
    led = SidingLedger(SIDINGS)
    led.assign(1, "c03")
    led.assign(2, "c23")
    r1 = pl.plan_task("S,c03,0", 0.0, ["c00", "c50"], {}, led.candidates(1), led.home(1))
    led.transfer(1, r1.siding)
    table = ReservationTable(graph, conflicts)
    table.set_path(1, r1.path)
    r2 = pl.plan_task("S,c23,0", 0.0, ["c50", "c00"], table.table(), led.candidates(2), led.home(2))
    led.transfer(2, r2.siding)
    assert (r1.siding, r2.siding) == ("c43", "c03") and led.snapshot() == {1: "c43", 2: "c03"}
    assert loads(r2.path) == [("L,c50,0", 40.0), ("L,c00,0", 60.0)] and r2.path[-1] == ("S,c03,0", 76.0)
    assert path_digest(r2.path) == "4843386babce17d1456c1a70c162fed77960b5a61cbe9f0d21dafbdc0d86dbc2"


def test_ledger_keeps_one_owner_per_siding_and_one_siding_per_vehicle():  # :127
    led = SidingLedger(SIDINGS)
    led.assign(1, "c03")
    led.assign(2, "c23")
    assert led.candidates(1) == ["c03", "c43"] and led.candidates(2) == ["c23", "c43"]
    with pytest.raises(ValueError, match="c03"):
        led.assign(2, "c03")
    with pytest.raises(ValueError, match="zz"):
        led.assign(3, "zz")
    led.transfer(1, "c43")
    assert led.home(1) == "c43" and led.owner("c03") is None and led.owner("c43") == 1
    assert led.candidates(2) == ["c03", "c23"] and led.snapshot() == {1: "c43", 2: "c23"}
    assert led.sidings == SIDINGS
