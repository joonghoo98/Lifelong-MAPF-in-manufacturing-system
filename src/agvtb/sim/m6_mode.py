"""The m6 mode of the simulator (M6 design §5.6-§5.9): every job is a fallback bundle, committed whole.

M6a has no fast path yet: each job is planned by Theorem C's procedure (M6Fallback). The bundle goes on the TAPG
member by member in its order (the evicted vehicles first, r last). Then the cross edges of every member are built
again from its old last stop: that stop was the end of its path, so it held no outgoing edge. Only then are the
members told to go on. Commit gate B looks at the event graph once all are on. A cycle is a finding and the bundle
stays: in M6a there is no other plan to put in its place (plan decision 1). The checks run after all members are on;
before that, a member not yet appended would still hold its old last stop forever.
"""
from collections import defaultdict
from collections.abc import Sequence

from agvtb.planning.m6_fallback import Bundle
from agvtb.sim.simulator import PARTIAL_S, Commit, agent_constraints
from agvtb.validate.plans import (early_fallback, live_entries, path_vs_entries, rest_tail_violations,
                                  task_order_violation, unprotected)
from agvtb.world.cells import CellMap


class M6Mode:
    """The bundle commits, checks and records of one m6 run."""

    def __init__(self, sim, cells: CellMap, rest: Sequence[str]) -> None:
        self.sim, self.cells, self.rest = sim, cells, tuple(rest)
        sim.counts["m6_bundle_cycles"] = 0
        sim.checks["m6_fallback_fail"] = []
        self.n_bundles = 0
        self.rest_nodes = sorted(n for c in self.rest for n in cells.members[c])
        self.wait_by_job: dict[int, float] = defaultdict(float)  # job -> its fallback waits, t0 - r's start

    def plan_and_commit(self, t: float, vid: int, legs: list[str], cons: dict[int, list[tuple[str, float, float]]]
                        ) -> None:
        """Plan vid's legs as a bundle and commit it; a failed plan leaves vid pending (as simulator.py:317-320)."""
        sim = self.sim
        tapg, executor = sim.tapg, sim.executor
        ends = {u: tapg.paths[u][-1] for u in sorted(tapg.paths)}
        b = sim.planner.plan(vid, legs, ends, cons)
        if b is None:
            sim.counts["plan_fail"] += 1
            sim.checks["m6_fallback_fail"].append({"t": t, "vid": vid, "legs": list(legs)})
            sim._log(t, "plan_fail", vid, tuple(legs))
            return
        old = {u: len(tapg.paths[u]) for u in b.order}
        moving = {u: executor.vehicles[u].state != "DONE" for u in b.order}
        for u in b.order:
            tapg.append(u, b.paths[u])
        for u in b.order:
            tapg.add_cross_edges(u, old[u] - 1, len(tapg.paths[u]))
        for u in b.order:
            executor.kick(u, t)
        sim.counts["commit_gate_checks"] += 1
        cyc, _ = sim._event_cycle()
        if cyc is not None:
            sim.counts["m6_bundle_cycles"] += 1
            sim.checks["commit_gate"].append({"t": t, "vid": vid, "why": "m6_bundle", "cycle": cyc[:8]})
        if tapg.has_cycle():
            sim.counts["tapg_cycles"] += 1
            sim.checks["tapg_cycles"].append({"t": t, "vid": vid})
        sim.ledger.move_all(b.homes)
        path = b.paths[vid]
        i_l = next(i for i, (sid, _) in enumerate(path) if sid.startswith("L,") and sim.graph[sid].nodes[0] == legs[0])
        sim.arrival[vid] = old[vid] + i_l - 2
        sim.notified[vid] = False
        tapg.mark_dwell(vid, sim.arrival[vid])
        job = sim.mcs.held[vid]
        sim.span[vid] = (job.jid, old[vid], old[vid] + len(path) - 1)
        self.wait_by_job[job.jid] += b.t0 - ends[vid][1]
        for u in b.order:
            cons[u] = agent_constraints(tapg, executor, u, t)
        self._check(t, vid, legs, b, old)
        for u in b.order:
            sim.commits.append(Commit(t, u, tuple(legs) if u == vid else (), ends[u][0], ends[u][1], moving[u], True,
                                      False, self.n_bundles, u != vid))
        self.n_bundles += 1
        for u in b.order[:-1]:
            sim._log(t, "evict", u, (vid, b.homes[u]))
        sim._log(t, "commit", vid, (tuple(legs), b.homes[vid], True))
        sim.counts["commits"] += 1
        del sim.pending[vid]
        sim.dispatch_t.pop(vid, None)
        sim._check_arrival(vid, t)

    def _check(self, t: float, vid: int, legs: list[str], b: Bundle, old: dict[int, int]) -> None:
        """The checks of simulator.py _check_plan and _check_tapg for every member, with m6's tails."""
        sim = self.sim
        tapg, executor, conflicts = sim.tapg, sim.executor, sim.conflicts
        for u in b.order:
            others = {w: live_entries(tapg.paths[w], executor.vehicles[w].path_idx, executor.vehicles[w].claim_idx, t)
                      for w in sorted(tapg.paths) if w != u}
            for o in path_vs_entries(b.paths[u], others, conflicts):
                sim.checks["plan_overlaps"].append({"t": t, "vid": u, **o})
            found = task_order_violation(b.paths[u], legs if u == vid else (), self.rest_nodes)
            if found is not None:
                sim.checks["order"].append({"t": t, "vid": u, "what": found})
            if early_fallback(b.paths[u], b.t0):
                sim.checks["early_fallbacks"].append({"t": t, "vid": u})
        for what in rest_tail_violations(tapg.paths, sim.ledger.snapshot(), self.cells.cell_of, self.rest):
            sim.checks["tails"].append({"t": t, "what": what})
        for u in b.order:
            pu = tapg.paths[u]
            mine = [(k, pu[k][0], tapg.key(u, k)) for k in range(old[u] - 1, len(pu)) if tapg.key(u, k) in tapg]
            others = {}
            for w in sorted(tapg.paths):
                if w != u:
                    pw = tapg.paths[w]
                    others[w] = [(j, pw[j][0], tapg.key(w, j)) for j in range(executor.vehicles[w].path_idx, len(pw))
                                 if tapg.key(w, j) in tapg]
            for o in unprotected(mine, others, tapg.related, conflicts):
                sim.checks["unprotected"].append({"t": t, "vid": u, **o})

    def report(self, w0: float, t_end: float) -> dict:
        """The rest rule and the slow jobs, counted as simulator.py _deadlock counts partial_jobs and held_at_end:
        jobs done in [w0, t_end] at least PARTIAL_S after their assignment, then jobs held that long at t_end."""
        mcs = self.sim.mcs
        slow = [{"jid": j.jid, "span_s": j.t_completed - j.t_assigned, "held_at_end": False,
                 "fallback_wait_s": self.wait_by_job.get(j.jid, 0.0)}
                for j in sorted(mcs.jobs, key=lambda j: j.jid)
                if j.t_completed is not None and w0 <= j.t_completed <= t_end and j.t_completed - j.t_assigned >= PARTIAL_S]
        slow += [{"jid": j.jid, "span_s": t_end - j.t_assigned, "held_at_end": True,
                  "fallback_wait_s": self.wait_by_job.get(j.jid, 0.0)}
                 for j in sorted(mcs.held.values(), key=lambda j: j.jid) if t_end - j.t_assigned >= PARTIAL_S]
        return {"rest_rule": self.sim.cfg.rest_rule, "slow_jobs": len(slow), "slow": slow}
