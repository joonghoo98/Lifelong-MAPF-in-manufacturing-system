import pytest

from agvtb.planning.gate import GateError
from agvtb.planning.m6_gate import RULES
from agvtb.sim import simulator
from agvtb.sim.simulator import SimConfig, Simulation, simulate
from agvtb.world.loader import load_world
from helpers import CONFIGS, one_way_tiny_config, split_config


@pytest.fixture(scope="module")
def kairos():
    return load_world(CONFIGS / "KAIROS_AMR.json")


def m6(lw, **kw):  # M6a's mode: every plan a fallback bundle
    cfg = dict(n_vehicles=26, rate_per_min=1.0, seed=0, warmup_s=0.0, measure_s=1800.0, planner="m6",
               rest_rule="branch", fast_path=False) | kw
    return simulate(lw, (), SimConfig(**cfg))


def clean(r):
    d = r.deadlock
    assert not d["global"] and d["wait_cycles"] == [] and all(not found for found in r.checks.values())
    assert r.stats["plan_fail"] == r.stats["inline_miss"] == r.stats["stray_arrivals"] == 0


@pytest.fixture(scope="module")
def run26(kairos):
    return m6(kairos)


def test_the_m6_gate_refuses_cells_that_break_theorem_b(kairos, monkeypatch):  # R32 I-2, user decision 2026-10-27
    seen = []
    monkeypatch.setattr(simulator, "cell_facts", lambda cm, graph, conflicts, ports: seen.append(cm) or ("cells: x",))
    with pytest.raises(GateError) as err:
        Simulation(kairos, (), SimConfig(1, 1.0, planner="m6", rest_rule="branch"))
    assert len(seen) == 1 and len(seen[0].cells) == 44
    assert not err.value.report.ok and err.value.report.violations == ("cells: x",)


def test_the_sl_defaults_stay():
    c = SimConfig(4, 1.0)
    assert (c.planner, c.rest_rule, c.tail_cut, c.fast_path) == ("sl", "no-cut", True, True)


def test_the_m6_gate_refuses_one_vehicle_too_many(kairos):
    with pytest.raises(GateError) as err:
        Simulation(kairos, (), SimConfig(27, 1.0, planner="m6", rest_rule="branch"))
    assert err.value.report.failed == ("C1",)


def test_a_short_kairos_m6_run_is_clean(run26):  # Review Focus 3, 4
    clean(run26)
    assert run26.kpi["completed"] >= 5 and run26.stats["commits"] == run26.stats["m6_fallback_used"]
    assert all(c.fallback for c in run26.commits) and any(c.evict and c.moving for c in run26.commits)
    assert any(c.moving and not c.evict for c in run26.commits)


def test_m6_runs_are_deterministic(kairos):  # Review Focus 5
    assert m6(kairos, measure_s=600.0).events_sha256 == m6(kairos, measure_s=600.0).events_sha256


@pytest.mark.parametrize("rule", RULES)
def test_short_mini_m6_runs_are_clean(rule):
    r = simulate(load_world(CONFIGS / "mini_deadend.json"), (),
                 SimConfig(10, 1.0, warmup_s=0.0, measure_s=1800.0, planner="m6", rest_rule=rule, fast_path=False))
    clean(r)
    assert r.kpi["completed"] >= 5


def test_a_cycle_after_a_bundle_is_a_finding(kairos, monkeypatch):  # M6a gate B (plan decision 1)
    real, calls = simulator.first_cycle, []

    def fake(graph):
        calls.append(1)
        return [("fake",)] if len(calls) == 1 else real(graph)
    monkeypatch.setattr(simulator, "first_cycle", fake)
    r = m6(kairos, measure_s=300.0)
    assert r.stats["m6_bundle_cycles"] == 1 and r.checks["commit_gate"][0]["why"] == "m6_bundle" and not r.ok


def test_slow_jobs_are_reported(run26):
    m, d = run26.m6, run26.deadlock
    assert m["rest_rule"] == "branch" and m["slow_jobs"] == len(m["slow"]) == d["partial_jobs"] + d["held_at_end"]
    assert all(s["fallback_wait_s"] >= 0.0 for s in m["slow"])


def test_the_m6_gate_refuses_maps_the_theorems_do_not_cover(tmp_path):  # final review I1, M1
    for sub, make in (("a", one_way_tiny_config), ("b", split_config)):
        (tmp_path / sub).mkdir()
        with pytest.raises(GateError) as err:
            Simulation(load_world(make(tmp_path / sub)), (), SimConfig(1, 1.0, planner="m6", rest_rule="all"))
        assert not err.value.report.ok and err.value.report.violations[0].startswith("map: ")
