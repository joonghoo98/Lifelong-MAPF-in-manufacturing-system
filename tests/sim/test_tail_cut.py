import pytest

from agvtb.planning.intervals import INF
from agvtb.planning.safe_tail import SafeTailPlanner, reverse_times, tail_cut_index
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


PRE_B_8 = "df1db586910aafbbf8305065e54018c3838e4c86ee0a3bea3375063d518b888a"  # (8, 2.0, 3600, 3600) before B and A2


def cfg(cut=True, margin=0.0, n=8, rate=2.0):
    return SimConfig(n, rate, 0, 3600.0, 3600.0, 0.5, tail_cut=cut, cut_margin=margin)


@pytest.fixture(scope="module")
def on(mini):
    return simulate(*mini, cfg())


def snapshot(sim):
    return {v: list(p) for v, p in sim.tapg.paths.items()}, set(sim.tapg.duration), sim.tapg.cross_edges()


def force_cut_cycles(monkeypatch, which):
    """first_cycle answers a fake cycle on its calls inside _try_cut numbered (from 1) in which (1 after the cut's
    attach, 2 after the gate's revoke, 3 after the old tail is back). Per _try_cut call that made such calls:
    (t, vid, (first, last) call numbers, the paths and nodes as before it when it returned None)."""
    real_fc, real_try, calls, tries, inside = simulator.first_cycle, Simulation._try_cut, [], [], []

    def fake(graph):
        if not inside:
            return real_fc(graph)
        calls.append(1)
        return [("fake",)] if len(calls) in which else real_fc(graph)

    def try_cut(self, t, vid, legs, table):
        n, before = len(calls), snapshot(self)
        inside.append(1)
        try:
            out = real_try(self, t, vid, legs, table)
        finally:
            inside.pop()
        if len(calls) > n:
            tries.append((t, vid, (n + 1, len(calls)), out is None and snapshot(self)[:2] == before[:2]))
        return out
    monkeypatch.setattr(simulator, "first_cycle", fake)
    monkeypatch.setattr(Simulation, "_try_cut", try_cut)
    return tries


def test_a2_off_is_the_run_before_a2(mini):  # DAIM spec §6 "꺼짐이면 비트 단위로 같다"
    r = simulate(*mini, cfg(cut=False))
    assert r.events_sha256 == PRE_B_8 and r.stats["cut_try"] == r.stats["cut_none"] == 0


def test_a2_is_on_by_default():  # user decision 2026-10-23
    assert SimConfig(1, 1.0).tail_cut is True and SimConfig(1, 1.0).cut_margin == 0.0


def test_a2_cuts_tails_at_ports_and_on_the_way_and_stays_clean(on):  # spec §8.1-2, -3, -11; Review Focus 2
    s = on.stats
    assert s["cut_ok_port"] > 0 and s["cut_ok_transit"] > 0 and s["cut_ok"] == s["cut_ok_port"] + s["cut_ok_transit"]
    assert s["cut_try"] == s["cut_ok"] + s["cut_search_fail"] + s["cut_aborted_claimed"] + s["cut_gate_rejected"]
    assert sum(c.cut for c in on.commits) == s["cut_ok"] and s["cut_tail_s"] > 0
    assert on.ok and s["plan_fail"] == 0 and on.checks["unprotected"] == [] and s["event_cycles"] == 0


def test_the_same_seed_gives_the_same_a2_run(mini, on):  # spec §8.1-12
    assert simulate(*mini, cfg()).events_sha256 == on.events_sha256


def test_every_cut_attach_restores_the_whole_kept_prefix(mini, monkeypatch):  # DAIM final review I2; RF 3
    attaches, restores = [], []
    real_attach, real_add = Simulation._cut_attach, Tapg.add_cross_edges

    def attach(self, t, vid, k, new_path):
        attaches.append((vid, min(self.executor.vehicles[vid].path_idx, k - 1), k))
        return real_attach(self, t, vid, k, new_path)

    def add(self, vid, lo, hi):
        restores.append((vid, lo, hi))
        return real_add(self, vid, lo, hi)
    monkeypatch.setattr(Simulation, "_cut_attach", attach)
    monkeypatch.setattr(Tapg, "add_cross_edges", add)
    simulate(*mini, cfg())
    assert attaches and restores == attaches and any(lo < k - 1 for _vid, lo, k in attaches)  # more than the splice


def test_a_cut_that_does_not_go_through_leaves_the_tapg_as_it_was(mini, monkeypatch):  # spec §3.4, §8.1-5
    real, n = SafeTailPlanner.search_task, []
    monkeypatch.setattr(SafeTailPlanner, "search_task",
                        lambda self, *a: n.append(1) or (None if len(n) == 1 else real(self, *a)))
    seen, real_try = [], Simulation._try_cut

    def try_cut(self, t, vid, legs, table):
        before, n = snapshot(self), self.counts["cut_none"] + self.counts["cut_search_fail"]
        out = real_try(self, t, vid, legs, table)
        if self.counts["cut_none"] + self.counts["cut_search_fail"] > n:  # not a gate rejection (edges restored there)
            seen.append(snapshot(self) == before)
        return out
    monkeypatch.setattr(Simulation, "_try_cut", try_cut)
    r = simulate(*mini, cfg())
    assert seen and all(seen) and r.stats["cut_search_fail"] >= 1 and r.ok


def test_a_cut_claimed_meanwhile_falls_back_to_a1(mini, monkeypatch):  # spec §4-2, §8.1-7
    real, n = Executor.revocable, []
    monkeypatch.setattr(Executor, "revocable", lambda self, vid, k: n.append(1) or (len(n) > 1 and real(self, vid, k)))
    r = simulate(*mini, cfg())
    assert r.stats["cut_aborted_claimed"] == 1 and r.ok


def test_a_cut_that_makes_a_cycle_gives_back_the_old_tail_then_a1(mini, monkeypatch):  # spec §4-5, §8.1-6
    tries = force_cut_cycles(monkeypatch, {1})
    r = simulate(*mini, cfg())
    t, vid, calls, restored = tries[0]
    assert calls == (1, 3) and restored  # the attach, the gate after the revoke, the old tail back as it was
    assert any(c.t == t and c.vid == vid and not c.cut for c in r.commits)  # then A1
    assert r.stats["cut_gate_rejected"] == 1 and r.stats["cut_restore_cycle"] == 0 and r.ok


def test_an_older_cycle_keeps_the_cut_and_fails_the_run(mini, monkeypatch):  # spec §4-5
    tries = force_cut_cycles(monkeypatch, {1, 2})
    r = simulate(*mini, cfg())
    t, vid, calls, _ = tries[0]
    assert calls == (1, 2) and any(c.t == t and c.vid == vid and c.cut for c in r.commits)
    assert r.stats["cut_gate_preexisting_cycle"] == 1 and [e["why"] for e in r.checks["commit_gate"]] == ["cut_preexisting"]
    assert not r.ok


def test_a_cycle_after_the_old_tail_is_back_is_a_finding(mini, monkeypatch):  # spec §4-5 (cut_restore_cycle)
    tries = force_cut_cycles(monkeypatch, {1, 3})
    r = simulate(*mini, cfg())
    assert tries[0][2] == (1, 3) and r.stats["cut_gate_rejected"] == r.stats["cut_restore_cycle"] == 1
    assert [e["why"] for e in r.checks["commit_gate"]] == ["cut_restore"] and not r.ok


def test_a_refused_revoke_at_the_cut_gate_keeps_the_cut_and_fails_the_run(mini, monkeypatch):  # spec §4-5
    real, n = Executor.revocable, []
    monkeypatch.setattr(Executor, "revocable", lambda self, vid, k: n.append(1) or (len(n) != 2 and real(self, vid, k)))
    force_cut_cycles(monkeypatch, {1})  # revocable call 1 = the cut's own, 2 = the gate's
    r = simulate(*mini, cfg())
    assert r.stats["cut_gate_revoke_refused"] == 1 and r.stats["cut_gate_rejected"] == 0
    assert [e["why"] for e in r.checks["commit_gate"]] == ["cut_revoke_refused"] and not r.ok


def test_every_cut_attach_lets_a_blocked_vehicle_look_again(mini, monkeypatch):  # spec §4-4 (C1); Review Focus 1
    seen, real = [], Executor.retry
    monkeypatch.setattr(Executor, "retry", lambda self, vid, t: seen.append(real(self, vid, t)) or seen[-1])
    r = simulate(*mini, cfg())
    s = r.stats
    assert len(seen) == s["cut_ok"] + 2 * s["cut_gate_rejected"] + s["cut_gate_preexisting_cycle"]
    assert sum(seen) == s["cut_wake"]


def test_after_a_cut_the_batch_sees_the_vehicles_whole_path(mini, monkeypatch):  # vis:5437-5438; Review Focus 4
    calls, lives = [], []
    real_one, real_live = simulator.agent_constraints, simulator.live_constraints
    monkeypatch.setattr(simulator, "agent_constraints", lambda *a: calls.append(1) or real_one(*a))
    monkeypatch.setattr(simulator, "live_constraints",
                        lambda tapg, ex, now: lives.append(len(tapg.paths)) or real_live(tapg, ex, now))
    r = simulate(*mini, cfg())
    assert len(calls) == sum(lives) + r.stats["cut_ok"]


def test_with_a2_the_dispatch_distance_starts_at_the_splice(mini):  # D5: A1 = its siding, A2 = the splice
    for cut in (True, False):
        sim, found = Simulation(*mini, cfg(cut=cut)), False
        for t in range(3600, 7200, 30):
            sim.advance_to(float(t))
            for vid, path in sim.tapg.paths.items():
                v = sim.executor.vehicles[vid]
                k = tail_cut_index(path, v.path_idx, v.claim_idx, sim.kernel.now, 0.0)
                if k > 0 and path[k - 1][0] != path[-1][0]:
                    start, src = (path[k - 1][0] if cut else path[-1][0]), sim.mcs.jobs[0].src
                    assert sim._dist_start(vid) == start  # and the dispatcher's distance is from there (review 1 m-1)
                    assert sim._dist(vid, src) == reverse_times(sim.graph, [src]).get(start, INF)
                    found = True
            if found:
                break
        assert found
