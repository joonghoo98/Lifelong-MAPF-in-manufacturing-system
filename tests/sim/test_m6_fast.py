import pytest

from agvtb.planning.eviction import EvictionPlanner
from agvtb.planning.m6_gate import RULES
from agvtb.sidings.choice import candidates_of, parse_siding_choice
from agvtb.sim import simulator
from agvtb.sim.executor import Executor
from agvtb.sim.simulator import SimConfig, simulate
from agvtb.world.loader import load_world
from helpers import CONFIGS


@pytest.fixture(scope="module")
def kairos():
    return load_world(CONFIGS / "KAIROS_AMR.json")


def m6(lw, **kw):
    cfg = dict(n_vehicles=26, rate_per_min=1.0, seed=0, warmup_s=0.0, measure_s=1800.0, planner="m6",
               rest_rule="branch") | kw
    return simulate(lw, (), SimConfig(**cfg))


def clean(r):
    d = r.deadlock
    assert not d["global"] and d["wait_cycles"] == [] and all(not found for found in r.checks.values())
    assert r.stats["plan_fail"] == r.stats["inline_miss"] == r.stats["stray_arrivals"] == 0


@pytest.fixture(scope="module")
def run26(kairos):
    return m6(kairos)


def test_a_short_kairos_fast_run_is_clean(run26):
    s = run26.stats
    clean(run26)
    assert s["m6_fast_ok"] >= 1 and s["m6_fast_evicted"] >= 1
    assert s["commits"] == s["m6_fast_ok"] - s["m6_fast_overlap"] - s["m6_gate_revoked"] + s["m6_fallback_used"]
    assert s["m6_fast_overlap"] == 0
    assert any(not c.fallback and c.evict for c in run26.commits)


@pytest.mark.parametrize("rule", RULES)
def test_short_mini_fast_runs_are_clean(rule):
    r = simulate(load_world(CONFIGS / "mini_deadend.json"), (),
                 SimConfig(10, 1.0, warmup_s=0.0, measure_s=1800.0, planner="m6", rest_rule=rule))
    clean(r)
    assert r.kpi["completed"] >= 5 and r.stats["m6_fast_ok"] >= 1


def test_fast_runs_are_deterministic(kairos):
    assert m6(kairos, measure_s=600.0).events_sha256 == m6(kairos, measure_s=600.0).events_sha256


def arm(monkeypatch):
    """A fake event-graph cycle right after the first fast bundle goes on."""
    real_cycle, real_plan, state = simulator.first_cycle, EvictionPlanner.plan, {"armed": False, "fired": 0}

    def plan(self, *a, **kw):
        b = real_plan(self, *a, **kw)
        if b is not None and not state["fired"]:
            state["armed"] = True
        return b

    def fake(graph):
        if state["armed"]:
            state["armed"], state["fired"] = False, 1
            return [("fake",)]
        return real_cycle(graph)
    monkeypatch.setattr(EvictionPlanner, "plan", plan)
    monkeypatch.setattr(simulator, "first_cycle", fake)
    return state


def test_a_cycle_after_a_fast_bundle_is_revoked(kairos, monkeypatch):  # design §5.7
    state = arm(monkeypatch)
    r = m6(kairos, n_vehicles=12, measure_s=300.0)  # 12: the first plan is a fast bundle (26: none in 300 s)
    assert state["fired"] == 1 and r.stats["m6_gate_revoked"] == 1 and r.checks["commit_gate"] == []
    assert r.stats["m6_bundle_cycles"] == 0 and r.commits[0].fallback
    clean(r)  # the TAPG after the revoke runs on without findings


def test_a_fast_bundle_that_cannot_be_revoked_is_a_finding(kairos, monkeypatch):  # Review Focus 5
    state = arm(monkeypatch)
    monkeypatch.setattr(Executor, "revocable", lambda self, vid, k: False)
    r = m6(kairos, n_vehicles=12, measure_s=300.0)
    assert state["fired"] == 1 and r.stats["m6_gate_revoke_refused"] == 1
    assert r.checks["commit_gate"][0]["why"] == "revoke_refused" and not r.ok


def test_an_overlapping_fast_bundle_goes_to_the_fallback(kairos, monkeypatch):  # final review I2
    from agvtb.sim import m6_mode
    real_check, real_plan, state = m6_mode.path_vs_entries, EvictionPlanner.plan, {"armed": False, "fired": 0}

    def plan(self, *a, **kw):
        b = real_plan(self, *a, **kw)
        if b is not None and not state["fired"]:
            state["armed"] = True
        return b

    def check(path, others, conflicts):  # the first look at the first fast bundle finds a (fake) overlap
        if state["armed"]:
            state["armed"], state["fired"] = False, 1
            return [{"b": -1, "sa": path[0][0], "sb": path[0][0], "t": path[0][1]}]
        return real_check(path, others, conflicts)
    monkeypatch.setattr(EvictionPlanner, "plan", plan)
    monkeypatch.setattr(m6_mode, "path_vs_entries", check)
    r = m6(kairos, n_vehicles=12, measure_s=300.0)
    assert state["fired"] == 1 and r.stats["m6_fast_overlap"] == 1 and r.commits[0].fallback
    clean(r)  # the overlap never went on the TAPG


def test_a_bundle_with_a_finished_member_is_not_sent_back():  # R30 acceptance: I2 looked before kick
    # a finished member's index is one past its stop until kick moves it back (executor.py:82); read before that, its
    # first new state was held from now on, and clean bundles went to the fallback (the first here at t = 687 s)
    r = simulate(load_world(CONFIGS / "mini_deadend.json"), (),
                 SimConfig(40, 1.0, warmup_s=0.0, measure_s=1800.0, planner="m6", rest_rule="no-cut"))
    clean(r)
    assert r.stats["m6_fast_ok"] >= 1 and r.stats["m6_fast_overlap"] == 0


def test_the_sl_mode_has_no_fast_path_counts():  # G4 (passes before M6b too: it guards the default mode)
    lw = load_world(CONFIGS / "mini_deadend.json")
    r = simulate(lw, candidates_of(lw, parse_siding_choice("grid-bridge-s00")),
                 SimConfig(4, 2.0, 0, 0.0, 300.0, 0.5, tail_cut=False))
    assert r.kpi["completed"] >= 1 and not any(k.startswith("m6_") for k in r.stats)
