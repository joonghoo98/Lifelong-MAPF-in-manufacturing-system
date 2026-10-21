import os
import subprocess
import sys

import pytest

from agvtb.planning.safe_tail import SafeTailPlanner
from agvtb.sidings.choice import candidates_of, parse_siding_choice
from agvtb.sim.simulator import SimConfig, simulate
from agvtb.world.loader import load_world
from helpers import CONFIGS, REPO


@pytest.fixture(scope="module")
def mini():
    lw = load_world(CONFIGS / "mini_deadend.json")
    return lw, candidates_of(lw, parse_siding_choice("grid-bridge-s00"))


def short(mini, **kw):
    lw, cands = mini
    cfg = dict(n_vehicles=4, rate_per_min=2.0, seed=0, warmup_s=0.0, measure_s=900.0, dt=0.5) | kw
    return simulate(lw, cands, SimConfig(**cfg))


@pytest.fixture(scope="module")
def base(mini):
    return short(mini)


def test_a_short_mini_run_is_clean(base):
    assert base.ok and base.kpi["completed"] >= 5 and not base.deadlock["global"]
    assert all(not found for found in base.checks.values()) and base.deadlock["wait_cycles"] == []


def test_same_input_same_event_log(mini, base):
    assert short(mini).events_sha256 == base.events_sha256


@pytest.mark.parametrize("dt", [0.1, 2.0])
def test_dt_does_not_change_the_event_log(mini, base, dt):  # design D2, Review Focus 1
    assert short(mini, dt=dt).events_sha256 == base.events_sha256


def test_event_log_ignores_hash_seed(base):
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO / 'tests')!r})\n"
        "from helpers import CONFIGS\n"
        "from agvtb.sidings.choice import candidates_of, parse_siding_choice\n"
        "from agvtb.sim.simulator import SimConfig, simulate\n"
        "from agvtb.world.loader import load_world\n"
        "lw = load_world(CONFIGS / 'mini_deadend.json')\n"
        "c = candidates_of(lw, parse_siding_choice('grid-bridge-s00'))\n"
        "print(simulate(lw, c, SimConfig(4, 2.0, 0, 0.0, 900.0, 0.5)).events_sha256)\n"
    )
    outs = {subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": s},
                           capture_output=True, text=True, check=True, timeout=600).stdout.strip() for s in ("1", "2")}
    assert outs == {base.events_sha256}


def test_a_vehicle_on_its_tail_plans_from_its_siding(mini):  # Review Focus 2, research §A3
    r = short(mini, n_vehicles=2, rate_per_min=4.0)
    assert r.ok and all(c.start_t >= c.t - 1e-6 for c in r.commits)  # path times are kept to 6 decimals (env:1942)
    assert any(c.moving and c.start_t > c.t for c in r.commits)  # a tail still running when the job came


def test_failed_plan_is_retried_at_the_next_dispatch(mini, monkeypatch):  # Review Focus 3, user S6
    real, calls = SafeTailPlanner.plan_task, []

    def flaky(self, *args, **kw):
        calls.append(args[1])
        return None if len(calls) == 1 else real(self, *args, **kw)

    monkeypatch.setattr(SafeTailPlanner, "plan_task", flaky)
    r = short(mini, n_vehicles=2, rate_per_min=2.0, measure_s=1800.0)
    first = r.jobs[0]
    retry = next(c for c in r.commits if c.vid == first.vehicle)
    assert r.stats["plan_fail"] == 1 and r.ok and r.kpi["completed"] >= 1
    assert retry.t > first.t_assigned  # not planned when it failed
    assert any(c.vid != first.vehicle and c.t == retry.t for c in r.commits)  # but with the next dispatch's replan


def test_no_completion_for_7200_s_with_backlog_stops_the_run(mini, monkeypatch):  # design §11
    monkeypatch.setattr(SafeTailPlanner, "plan_task", lambda self, *a, **k: None)
    r = short(mini, measure_s=20000.0)
    assert r.deadlock["global"] and r.deadlock["frozen_at"] == 0.0 and r.t_end == pytest.approx(7200.0)
    assert not r.ok


def test_kpi_matches_the_job_records(base):  # research §F2 definitions
    done = [j for j in base.jobs if j.t_completed is not None and j.t_completed >= 0.0]
    window = base.t_end - 0.0
    assert base.kpi["completed"] == len(done)
    assert base.kpi["throughput_per_h"] == pytest.approx(len(done) / window * 3600)
    assert base.kpi["cycle_s"] == pytest.approx(sum(j.t_completed - j.t_created for j in done) / len(done))
