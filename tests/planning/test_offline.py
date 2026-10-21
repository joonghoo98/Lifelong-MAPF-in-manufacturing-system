import os
import random
import subprocess
import sys

import pytest

from agvtb.planning.gate import GateError
from agvtb.planning.offline import initial_stop, make_tasks, plan_offline
from agvtb.sidings.choice import candidates_of, parse_siding_choice
from agvtb.world.loader import load_world
from helpers import CONFIGS, REPO, fake_room_with_spurs, spur_room_world, timelines_digest

SIDINGS = ["c03", "c23", "c43"]


def fake_run(n, k, seed, interval=0.0):
    graph, conflicts, ports = fake_room_with_spurs()
    return plan_offline(graph, conflicts, ports, SIDINGS, n, k, seed=seed, interval_s=interval)


@pytest.fixture(scope="module")
def mini():
    lw = load_world(CONFIGS / "mini_deadend.json")
    return lw, candidates_of(lw, parse_siding_choice("grid-bridge-s00"))


def mini_run(mini, n, k, seed, interval=0.0, budget=None):
    lw, cands = mini
    return plan_offline(lw.graph, lw.conflicts, lw.world.port_nodes, cands, n, k, seed=seed, interval_s=interval,
                        budget=budget)


@pytest.fixture(scope="module")
def mini15(mini):  # the longest M3 run (DAIM's code: 3.9 s); two tests read it
    return mini_run(mini, 15, 60, 3, interval=4.0)


def ends(result):
    return {v: tl[-1][1] for v, tl in result.timelines.items()}


# offline_expected.out.txt and offline_more.out.txt: these rules driven with DAIM's planner, ledger, table, validator
def test_fake_room_two_vehicles():
    r = fake_run(2, 6, 0)
    assert r.ok and r.start_sidings == {0: "c23", 1: "c43"} and r.homes == {0: "c23", 1: "c43"}
    assert [(t.index, t.dispatch_s, t.vehicle, t.legs) for t in r.tasks[:3]] == [
        (0, 0.0, 0, ("c00", "c50")), (1, 0.0, 1, ("c50", "c00")), (2, 0.0, 0, ("c50", "c00"))]
    assert [p.expansions for p in r.plans] == [39, 60, 23, 52, 32, 17]
    assert ends(r) == {0: 148.0, 1: 180.0} and r.stats == {"search_ok": 6}
    assert timelines_digest(r.timelines) == "31172dbc8bf5b4bcc140e3bfd1c313709e90351ab020db133f6100e2507a5ab4"


def test_fake_room_three_vehicles_spaced_tasks():
    r = fake_run(3, 9, 1, interval=5.0)
    assert r.ok and r.homes == {0: "c03", 1: "c43", 2: "c23"} and ends(r) == {0: 190.0, 1: 226.0, 2: 236.0}
    assert [p.expansions for p in r.plans] == [47, 61, 51, 68, 40, 55, 30, 41, 45]
    assert timelines_digest(r.timelines) == "e610728dfa7720975776bebc7c08e5e0fc22ea022a41fb088193bbd85f3028f6"


def test_mini_eight_vehicles(mini):
    r = mini_run(mini, 8, 24, 0)
    assert r.ok and r.stats == {"search_ok": 24}
    assert r.start_sidings == dict(enumerate(["G2_2", "G2_6", "G10_6", "G22_6", "G4_8", "G4_6", "G20_6", "G4_2"]))
    assert r.homes == dict(enumerate(["G10_10", "G24_4", "G2_10", "G24_10", "G12_4", "G22_4", "G10_6", "G14_6"]))
    assert r.tasks[0].legs == ("G1_9", "G1_4") and r.tasks[5].legs == ("G14_10", "G1_9")
    assert [p.expansions for p in r.plans][:8] == [88, 143, 562, 539, 391, 522, 1190, 1798]
    assert timelines_digest(r.timelines) == "c95422ec0f0559160f193023a06867badce2e4010b5b0b9a05ac489d39ed1b4f"


def test_mini_budget_one_falls_back_every_task_and_stays_valid(mini):  # design §12: budget exceeded -> fallback
    r = mini_run(mini, 8, 24, 0, budget=1)
    assert r.ok and r.stats == {"budget_exceeded": 24, "fallback_used": 24}
    assert all(p.fallback for p in r.plans) and r.homes == r.start_sidings
    assert timelines_digest(r.timelines) == "7be71572b9c9128bb9e3a48593be96d426ea4bcd698481935c541278266198a6"


def test_mini_fifteen_vehicles_sixty_tasks(mini15):
    r = mini15
    assert r.ok and r.stats == {"search_ok": 60}
    assert timelines_digest(r.timelines) == "6cf610e2a47848e4ab3de6410af7b783a8e6ea7cc69cd3dcd086cb1a3ff77ed7"


def test_a_vehicle_still_on_its_way_starts_when_its_path_ends(mini15):  # Review Focus 1; offline_more.out.txt 2. and 5.
    r = mini15
    last = {v: tl[0] for v, tl in r.timelines.items()}
    kinds = {"at dispatch": 0, "later": 0}
    for task in sorted(r.tasks, key=lambda t: (t.dispatch_s, t.vehicle, t.index)):
        plan = r.plans[task.index]
        sid, t_last = last[task.vehicle]
        assert plan.path[0] == (sid, max(task.dispatch_s, t_last))
        kinds["later" if plan.path[0][1] > task.dispatch_s else "at dispatch"] += 1
        last[task.vehicle] = plan.path[-1]
    assert kinds == {"at dispatch": 16, "later": 44}


def test_the_gate_refuses_before_any_planning(mini, tmp_path):  # user decisions G1 and G2, Review Focus 5
    with pytest.raises(GateError) as err:
        mini_run(mini, 38, 4, 0)
    assert err.value.report.condition_m.fatal
    lw = spur_room_world(tmp_path)
    with pytest.raises(GateError, match="c13"):
        plan_offline(lw.graph, lw.conflicts, lw.world.port_nodes, ["c12", "c32"], 2, 4)


@pytest.mark.parametrize("n, k, interval, match", [
    (0, 1, 0.0, "n_vehicles"), (1, -1, 0.0, "n_tasks"), (1, 1, -1.0, "interval"), (1, 1, float("inf"), "interval"),
])
def test_rejects_bad_counts(n, k, interval, match):  # Review Focus 3
    with pytest.raises(ValueError, match=match):
        fake_run(n, k, 0, interval)


def test_tasks_need_two_ports_and_start_poses_prefer_heading_zero():
    with pytest.raises(ValueError, match="c00"):
        make_tasks(["c00"], 1, 1, 0.0, random.Random(0))
    assert make_tasks(["c00"], 1, 0, 0.0, random.Random(0)) == []
    graph, _conflicts, _ports = fake_room_with_spurs()
    assert initial_stop(graph, "c03") == "S,c03,0"
    kairos = load_world(CONFIGS / "KAIROS_AMR.json")  # offline_more.out.txt 4.
    assert (initial_stop(kairos.graph, "na.198"), initial_stop(kairos.graph, "na.40")) == ("S,na.198,0", "S,na.40,271")
    with pytest.raises(ValueError, match="zz"):
        initial_stop(graph, "zz")


def test_offline_ignores_hash_seed():  # Review Focus 4
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO / 'tests')!r})\n"
        "from helpers import fake_room_with_spurs, timelines_digest\n"
        "from agvtb.planning.offline import plan_offline\n"
        "g, c, p = fake_room_with_spurs()\n"
        "r = plan_offline(g, c, p, ['c03', 'c23', 'c43'], 3, 9, seed=1, interval_s=5.0)\n"
        "print(timelines_digest(r.timelines))\n"
    )
    outs = {subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed},
                           capture_output=True, text=True, check=True, timeout=300).stdout.strip()
            for seed in ("1", "2")}
    assert outs == {"e610728dfa7720975776bebc7c08e5e0fc22ea022a41fb088193bbd85f3028f6"}
