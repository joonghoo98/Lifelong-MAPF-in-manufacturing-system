import pytest

from agvtb.planning.safe_tail import SafeTailPlanner
from agvtb.sidings.choice import candidates_of, parse_siding_choice
from agvtb.sim import simulator
from agvtb.sim.executor import Executor
from agvtb.sim.simulator import SimConfig, Simulation, simulate
from agvtb.sim.tapg import Tapg
from agvtb.world.loader import load_world
from helpers import CONFIGS


@pytest.fixture(scope="module")
def mini():
    lw = load_world(CONFIGS / "mini_deadend.json")
    return lw, candidates_of(lw, parse_siding_choice("grid-bridge-s00"))


PRE_B = {  # events_sha256 at master 47ab4ff, before B (plans/2026-10-22-m5b-gate-b_files/pre_b_sha.out.txt)
    (4, 2.0, 0.0, 900.0): "bd8076e9ed7b0915a52f7b2d947b739d41d0b386b819545c1cba8e9de1305758",
    (2, 4.0, 0.0, 900.0): "df9c58b4a8e6be902a9fbdb40aa53d6300d1a6ebb5acad3bd1de173a382ed1f9",
    (8, 2.0, 3600.0, 3600.0): "df1db586910aafbbf8305065e54018c3838e4c86ee0a3bea3375063d518b888a",
}


def cfg(n, rate, warm=0.0, meas=900.0):
    return SimConfig(n, rate, 0, warm, meas, 0.5, tail_cut=False)  # B on the A1 path (M5c: A2 is on by default)


def force_cycles(monkeypatch, which):
    """first_cycle answers a fake cycle on the calls numbered (from 1) in which; None = every call."""
    real, calls = simulator.first_cycle, []

    def fake(graph):
        calls.append(1)
        return [("fake",)] if which is None or len(calls) in which else real(graph)
    monkeypatch.setattr(simulator, "first_cycle", fake)
    return calls


def no_stale_nodes(sim):
    return all({key for key in sim.tapg.duration if key[1] == vid} <= {sim.tapg.key(vid, i) for i in range(len(p))}
               for vid, p in sim.tapg.paths.items())


def gate_log(monkeypatch):
    """Per gate call, what it did in order: ("revoke", waiters returned), ("wake", vehicles given), ("check", path)."""
    log, inside = [], []
    real_gate, real_revoke, real_wake, real_check = (Simulation._commit_gate, Tapg.revoke_beyond, Executor.wake,
                                                     Simulation._check_plan)

    def note(what, value):
        if inside:
            log[-1].append((what, value))
        return value

    def gate(self, *args):
        log.append([])
        inside.append(1)
        try:
            return real_gate(self, *args)
        finally:
            inside.pop()

    def wake(self, vids, t):
        vids = note("wake", list(vids))
        return real_wake(self, vids, t)

    def check(self, t, vid, res, legs):
        note("check", res.path)
        return real_check(self, t, vid, res, legs)
    monkeypatch.setattr(Simulation, "_commit_gate", gate)
    monkeypatch.setattr(Tapg, "revoke_beyond", lambda self, vid, k: note("revoke", real_revoke(self, vid, k)))
    monkeypatch.setattr(Executor, "wake", wake)
    monkeypatch.setattr(Simulation, "_check_plan", check)
    return log


@pytest.mark.parametrize("key", sorted(PRE_B))
def test_the_event_log_is_the_one_before_b_when_the_gate_never_fires(mini, key):
    r = simulate(*mini, cfg(*key))
    assert r.events_sha256 == PRE_B[key]
    assert r.stats["commit_gate_checks"] == r.stats["commits"] and r.stats["commit_gate_rejected"] == 0
    assert r.stats["event_cycles"] == r.stats["event_forever_edges_max"] == 0
    assert r.checks["event_cycles"] == r.checks["commit_gate"] == r.checks["tapg_cycles"] == []


def test_a_cycle_making_commit_is_replaced_by_the_fallback(mini, monkeypatch):  # spec §6.2-2, DAIM B I-1, Review Focus 5
    base = simulate(*mini, cfg(4, 2.0))
    c0 = base.commits[0]
    assert not c0.moving and sum(c.t == c0.t for c in base.commits) == 1  # a DONE vehicle, alone in its replan (review 1 m-6)
    calls, got, real_fb = force_cycles(monkeypatch, {1}), [], SafeTailPlanner.fallback
    monkeypatch.setattr(SafeTailPlanner, "fallback", lambda self, *a, **kw: got.append(real_fb(self, *a, **kw)) or got[-1])
    log = gate_log(monkeypatch)
    sim = Simulation(*mini, cfg(4, 2.0))
    sim.advance_to(c0.t)
    assert len(calls) == 4  # the gate: after the append, the revoke, the fallback's append; then the replan's post-check
    assert sim.counts["commit_gate_rejected"] == 1 and sim.commits[0].vid == c0.vid and sim.commits[0].fallback
    assert len(got) == 1 and sim.tapg.paths[c0.vid][1:] == got[0].path[1:] and no_stale_nodes(sim)
    w = log[0][0][1]
    assert log[0] == [("revoke", w), ("wake", w), ("check", got[0].path)]  # waiters woken, fallback checked (final review I-1)
    sim.advance_to(sim.end)
    assert sim.result().ok


def test_a_cycle_that_stays_after_the_revoke_keeps_the_plan_and_fails_the_run(mini, monkeypatch):  # Review Focus 3
    force_cycles(monkeypatch, None)
    r = simulate(*mini, cfg(4, 2.0))
    assert r.stats["commit_gate_preexisting_cycle"] == r.stats["commits"] > 0 and r.stats["commit_gate_rejected"] == 0
    assert not any(c.fallback for c in r.commits) and {e["why"] for e in r.checks["commit_gate"]} == {"preexisting"}
    assert 0 < r.stats["event_cycles"] == len(r.checks["event_cycles"]) <= r.stats["commits"]  # once per replan, after its commits
    assert not r.ok


def test_the_post_check_runs_once_per_replan_that_commits(mini, monkeypatch):  # user decision (plan review 1 m-1)
    per_replan, posts, plans = [], [], []
    real_replan, real_post, real_plan = Simulation._replan, Simulation._post_check, SafeTailPlanner.plan_task

    def plan_task(self, *args):  # the first plan fails: that vehicle stays pending and goes with the next one
        plans.append(1)
        return None if len(plans) == 1 else real_plan(self, *args)

    def replan(self, t, force=False):
        n = self.counts["commits"]
        real_replan(self, t, force)
        if self.counts["commits"] > n:
            per_replan.append(self.counts["commits"] - n)

    def post(self, t):
        posts.append(t)
        return real_post(self, t)
    monkeypatch.setattr(SafeTailPlanner, "plan_task", plan_task)
    monkeypatch.setattr(Simulation, "_replan", replan)
    monkeypatch.setattr(Simulation, "_post_check", post)
    r = simulate(*mini, cfg(4, 2.0))
    assert len(posts) == len(per_replan) and sum(per_replan) == r.stats["commits"]
    assert max(per_replan) > 1  # a replan committed several vehicles: measuring per commit would count more


def gate_entries(mini, config, monkeypatch):
    """A plain run's gate calls in order: (number of the call's first first_cycle call, vid, state, claim_idx, old_len)."""
    rows, real = [], Simulation._commit_gate
    with monkeypatch.context() as m:
        calls = force_cycles(m, set())  # counts the calls, fakes none

        def gate(self, t, vid, res, legs, table, old_len):
            v = self.executor.vehicles[vid]
            rows.append((len(calls) + 1, vid, v.state, v.claim_idx, old_len))
            return real(self, t, vid, res, legs, table, old_len)
        m.setattr(Simulation, "_commit_gate", gate)
        simulate(*mini, config)
    return rows


def test_a_moving_vehicle_loses_only_the_new_segment(mini, monkeypatch):  # Review Focus 1, plan review 1 I-1
    config = cfg(8, 2.0, 3600.0, 3600.0)
    i, (n, vid, _state, _claim, old_len) = next((i, e) for i, e in enumerate(gate_entries(mini, config, monkeypatch))
                                                 if e[2] == "MOVING" and e[3] == e[4])  # on its way, claimed up to the cut
    force_cycles(monkeypatch, {n})
    cuts, real_revoke = [], Tapg.revoke_beyond

    def revoke(self, v, k):
        head = list(self.paths[v][:k])
        waiters = real_revoke(self, v, k)
        cuts.append((v, k, list(self.paths[v]) == head))
        return waiters
    monkeypatch.setattr(Tapg, "revoke_beyond", revoke)
    r = simulate(*mini, config)
    assert cuts == [(vid, old_len, True)]  # only the new segment came off; what the vehicle had claimed stayed
    assert r.commits[i].vid == vid and r.commits[i].moving and r.commits[i].fallback
    assert r.stats["commit_gate_rejected"] == 1 and r.ok


def test_no_fallback_after_the_revoke_leaves_the_vehicle_pending(mini, monkeypatch):  # Review Focus 2
    t0 = simulate(*mini, cfg(2, 2.0, meas=1800.0)).jobs[0].t_assigned  # the first commit's tick, before any gate fires
    force_cycles(monkeypatch, {1})
    monkeypatch.setattr(SafeTailPlanner, "fallback", lambda self, *a, **kw: None)
    sim = Simulation(*mini, cfg(2, 2.0, meas=1800.0))
    sim.advance_to(t0)
    vid = sim.mcs.jobs[0].vehicle
    v = sim.executor.vehicles[vid]
    assert vid in sim.pending and v.state == "DONE" and v.path_idx == len(sim.tapg.paths[vid])  # final review I-1
    sim.advance_to(sim.end)
    r = sim.result()
    first = r.jobs[0]
    retry = next(c for c in r.commits if c.vid == first.vehicle)
    assert r.stats["commit_gate_fallback_fail"] == 1 and r.stats["plan_fail"] == 1 and r.ok and no_stale_nodes(sim)
    assert retry.t > first.t_assigned and not retry.fallback and first.t_completed is not None


def test_a_refused_revoke_keeps_the_plan_and_fails_the_run(mini, monkeypatch):  # vis:5508-5510 (final review M-1)
    force_cycles(monkeypatch, {1})
    revoked = []
    monkeypatch.setattr(Executor, "revocable", lambda self, vid, k: False)  # not reachable in B (plan review 1): forced
    monkeypatch.setattr(Tapg, "revoke_beyond", lambda self, vid, k: revoked.append(vid) or [])
    r = simulate(*mini, cfg(4, 2.0))
    assert r.stats["commit_gate_revoke_refused"] == 1 and r.stats["commit_gate_rejected"] == 0 and revoked == []
    assert [e["why"] for e in r.checks["commit_gate"]] == ["revoke_refused"] and not r.ok


def test_a_cycle_after_the_fallback_is_a_finding(mini, monkeypatch):  # vis:5525-5527 (final review M-1)
    force_cycles(monkeypatch, {1, 3})  # the commit, then the fallback that replaced it
    r = simulate(*mini, cfg(4, 2.0))
    assert r.stats["commit_gate_rejected"] == r.stats["commit_gate_fallback_cycle"] == 1 and r.commits[0].fallback
    assert [e["why"] for e in r.checks["commit_gate"]] == ["fallback"] and not r.ok


def test_the_gate_sees_the_live_cross_edges(mini, monkeypatch):  # final review M-2: the detector gets the real TAPG
    cross, real = [], simulator.event_graph

    def spy(paths, claims, nodes, cross_edges):
        graph, forever = real(paths, claims, nodes, cross_edges)
        cross.append(sum(b[1] != a[1] for a, bs in graph.items() for b in bs))
        return graph, forever
    monkeypatch.setattr(simulator, "event_graph", spy)
    r = simulate(*mini, cfg(4, 2.0))
    assert len(cross) > r.stats["commit_gate_checks"] > 0 and max(cross) > 0


def test_a_tapg_node_cycle_is_still_a_finding(mini, monkeypatch):  # 정정 가: the old check keeps its own name
    monkeypatch.setattr(Tapg, "has_cycle", lambda self: True)
    r = simulate(*mini, cfg(4, 2.0))
    assert len(r.checks["tapg_cycles"]) == r.stats["tapg_cycles"] == r.stats["commits"] and not r.ok
