import dataclasses
import os
import subprocess
import sys

import pytest

from agvtb.planning.intervals import INF
from agvtb.planning.reservations import ReservationTable
from agvtb.planning.safe_tail import SafeTailPlanner, reverse_times
from agvtb.sidings.lattice import grid_sidings_of_world
from agvtb.world.loader import load_world
from helpers import CONFIGS, REPO, fake_room_with_spurs, path_digest

SIDINGS = ["c03", "c23", "c43"]
OTHER = [("S,c50,0", 0.0), ("M,c50,c40", 40.0), ("S,c40,0", 42.0), ("M,c40,c41", 42.0), ("S,c41,0", 44.0),
         ("M,c41,c42", 44.0), ("S,c42,0", 46.0), ("M,c42,c43", 46.0), ("S,c43,0", 48.0)]  # DAIM test :55


def planner(budget=None):
    graph, conflicts, _ports = fake_room_with_spurs()
    return SafeTailPlanner(graph, conflicts, SIDINGS, budget=budget)


def loads(path):
    return [(s, t) for s, t in path if s.startswith("L,")]


# DAIM tests/test_safe_tail_planner.py line numbers; values and digests from a1_digests.out.txt (A == B)
def test_reverse_times_count_the_time_of_every_state_on_the_way():
    graph, _conflicts, _ports = fake_room_with_spurs()
    h = reverse_times(graph, ["c00"])
    assert (h["S,c00,0"], h["M,c01,c00"], h["L,c00,0"], h["S,c03,0"]) == (0.0, 2.0, 10.0, 6.0)
    assert reverse_times(graph, ["zz"]) == {}


def test_task_visits_src_then_dst_and_parks_on_the_nearest_free_siding():  # :42
    pl = planner()
    res = pl.search("S,c03,0", 0.0, ["c00", "c50"], {}, SIDINGS)
    assert (res.fallback, res.expansions, res.siding, res.t_star) == (False, 17, "c43", 0.0)
    assert res.path[0] == ("S,c03,0", 0.0) and res.path[-1] == ("S,c43,0", 44.0)
    assert loads(res.path) == [("L,c00,0", 6.0), ("L,c50,0", 26.0)]
    assert path_digest(res.path) == "a2bd5a2b821d643b2c05121e901cc946a6fff61142e086de9f3e28b7bb41698e"
    assert dict(pl.stats) == {}  # search counts only its failures; plan_task counts search_ok (Task 5)


def test_task_respects_reservations_and_owned_sidings():  # :53
    graph, conflicts, _ports = fake_room_with_spurs()
    table = ReservationTable(graph, conflicts)
    table.set_path(7, OTHER)
    res = SafeTailPlanner(graph, conflicts, SIDINGS).search("S,c03,0", 0.0, ["c00", "c50"], table.table(),
                                                            ["c03", "c23"])
    assert (res.siding, res.path[-1], res.expansions) == ("c23", ("S,c23,0", 66.0), 55)
    assert loads(res.path) == [("L,c00,0", 6.0), ("L,c50,0", 44.0)]
    assert path_digest(res.path) == "f1d9aede896d7f2e7a1abb0958ee4707ef71d6488f21cc756ef72ebaba67128c"


def test_nearest_siding_with_a_later_reservation_is_skipped_for_parking():  # :67
    res = planner().search("S,c03,0", 0.0, ["c00", "c50"], {"S,c43,0": [(100.0, 110.0)]}, SIDINGS)
    assert (res.siding, res.path[-1], res.expansions) == ("c23", ("S,c23,0", 48.0), 33)
    assert path_digest(res.path) == "cf481b6d4cb467e034a856a057453da98ce93f9f3698ad1650f318c3d78eefa2"


def test_same_inputs_give_the_same_path():  # :98
    table = {"S,c20,0": [(10.0, 20.0)]}
    r1 = planner().search("S,c23,0", 5.0, ["c50", "c00"], table, SIDINGS)
    r2 = planner().search("S,c23,0", 5.0, ["c50", "c00"], table, SIDINGS)
    assert r1 == r2 and (r1.siding, r1.path[-1], r1.expansions) == ("c03", ("S,c03,0", 53.0), 22)
    assert loads(r1.path) == [("L,c50,0", 17.0), ("L,c00,0", 37.0)]
    assert path_digest(r1.path) == "3766728e3429ed1f162019d8653f5eac18356c013e9b7124974cd72b0491d74c"


def test_single_leg_and_same_port_legs():  # :105
    pl = planner()
    one = pl.search("S,c03,0", 0.0, ["c50"], {}, SIDINGS)
    assert (loads(one.path), one.path[-1], one.expansions) == ([("L,c50,0", 16.0)], ("S,c43,0", 34.0), 26)
    assert path_digest(one.path) == "710a5302e80116d46181467a405a2063fd1ad5cf869130d5a85b920c5f7d9ae8"
    two = pl.search("S,c03,0", 0.0, ["c50", "c50"], {}, SIDINGS)
    assert (loads(two.path), two.expansions) == ([("L,c50,0", 16.0), ("L,c50,0", 26.0)], 27)
    assert path_digest(two.path) == "ea6d1b9e6d1f2eeec24a169b1303c249c9be2ff0168be872c9928ba238e290ec"


def test_start_state_inside_someone_elses_reservation_is_refused():  # :92, Review Focus 2
    pl = planner()
    assert pl.search("S,c03,0", 0.0, ["c00", "c50"], {"S,c03,0": [(0.0, 30.0)]}, SIDINGS) is None
    assert dict(pl.stats) == {"start_blocked": 1}


@pytest.mark.parametrize("start, legs, table, budget, stat", [
    ("S,c03,0", ["c00", "c50"], {"S,c22,0": [(10.0, 50.0)]}, 1, "budget_exceeded"),  # :73
    ("S,c03,0", ["c00", "c50"], {"S,c00,0": [(0.0, INF)]}, None, "search_exhausted"),  # :86
    ("S,c03,0", ["zz"], {}, None, "unreachable"),
    ("M,c03,c02", ["c00"], {}, None, "start_invalid"),
    ("S,zz,0", ["c00"], {}, None, "start_invalid"),
])
def test_failures_return_none_and_are_counted(start, legs, table, budget, stat):  # Review Focus 3
    pl = planner(budget)
    assert pl.search(start, 0.0, legs, table, SIDINGS) is None and dict(pl.stats) == {stat: 1}


MINI_CASES = [  # a1_digests.out.txt C: (start, legs, siding, last state, expansions, digest)
    ("S,G10_10,0", ["G11_10", "G9_1"], "G8_2", ("S,G8_2,90", 45.47077244258873), 22,
     "7d9a64b1d611bcd17a0ff9062eac227bd8366044f3fc90a0d655ec4fe2cf6b29"),
    ("S,G8_6,0", ["G9_1", "G11_10"], "G10_10", ("S,G10_10,180", 64.20250521920669), 91,
     "6957445ba62daa65fb14f0c7067e35f0418614dcda56c12e085248300fd7793d"),
    ("S,G18_6,0", ["G14_10", "G1_4"], "G2_4", ("S,G2_4,0", 80.94154488517748), 168,
     "5541061448a83d3c57aa3e2dd4d27b9521752d6a5e95a2583d94fb6b94257f5c"),
]


@pytest.fixture(scope="module")
def mini():
    lw = load_world(CONFIGS / "mini_deadend.json")
    return lw, sorted(grid_sidings_of_world(lw.world, "bridge", (0, 0)))


@pytest.mark.parametrize("start, legs, siding, last, expansions, digest", MINI_CASES)
def test_mini_tasks_match_daim(mini, start, legs, siding, last, expansions, digest):
    lw, sidings = mini
    res = SafeTailPlanner(lw.graph, lw.conflicts, sidings).search(start, 0.0, legs, {}, sidings)
    assert (res.siding, res.path[-1], res.expansions, path_digest(res.path)) == (siding, last, expansions, digest)


def test_search_ignores_hash_seed():  # Review Focus 4
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO / 'tests')!r})\n"
        "from helpers import CONFIGS, path_digest\n"
        "from agvtb.planning.safe_tail import SafeTailPlanner\n"
        "from agvtb.sidings.lattice import grid_sidings_of_world\n"
        "from agvtb.world.loader import load_world\n"
        "lw = load_world(CONFIGS / 'mini_deadend.json')\n"
        "s = sorted(grid_sidings_of_world(lw.world, 'bridge', (0, 0)))\n"
        "r = SafeTailPlanner(lw.graph, lw.conflicts, s).search('S,G18_6,0', 0.0, ['G14_10', 'G1_4'], {}, s)\n"
        "print(path_digest(r.path))\n"
    )
    outs = {subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed},
                           capture_output=True, text=True, check=True, timeout=300).stdout.strip()
            for seed in ("1", "2")}
    assert outs == {"5541061448a83d3c57aa3e2dd4d27b9521752d6a5e95a2583d94fb6b94257f5c"}


def test_budget_default_and_bad_inputs():  # Review Focus 3
    graph, conflicts, _ports = fake_room_with_spurs()
    assert SafeTailPlanner(graph, conflicts, SIDINGS).budget == SafeTailPlanner.DEFAULT_BUDGET == 200_000
    with pytest.raises(ValueError, match=r"\b0\b"):
        SafeTailPlanner(graph, conflicts, SIDINGS, budget=0)
    odd = dict(graph)
    odd["L,c50,0"] = dataclasses.replace(graph["L,c50,0"], duration_s=5.0)
    with pytest.raises(ValueError, match=r"5\.0"):
        SafeTailPlanner(odd, conflicts, SIDINGS)
