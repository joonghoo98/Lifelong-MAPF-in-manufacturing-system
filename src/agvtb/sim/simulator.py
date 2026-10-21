"""The lifelong run: demand -> assignment -> A1 plan -> TAPG execution, with KPIs, deadlock detection and checks.

The flow is DAIM's safe-tail loop (research §A): at each dispatch the pending vehicles are planned from the end of
their committed paths, after the done prefixes are pruned and the TAPG is re-timed from now; a plan is checked,
appended to the TAPG, its siding moves in the ledger, and the vehicle is told to go on. There are no periodic
timers and no timed retries (user decision S6): a failed plan stays pending until the next dispatch. Wait cycles
are only detected and recorded (S3). Every run of the same input writes the same event log, whatever dt is.
"""
import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from agvtb.planning.gate import GateError, GateReport, check_gate
from agvtb.planning.intervals import INF
from agvtb.planning.ledger import SidingLedger
from agvtb.planning.offline import initial_stop
from agvtb.planning.reservations import affects, path_constraints
from agvtb.planning.safe_tail import SafeTailPlanner, reverse_times
from agvtb.sim.demand import Demand, normalize, uniform_fromto
from agvtb.sim.executor import Executor, wait_cycles, world_geometry
from agvtb.sim.jobs import Job, Mcs
from agvtb.sim.kernel import MCS, Kernel
from agvtb.sim.tapg import Tapg
from agvtb.validate.plans import (early_fallback, live_entries, pairwise_overlaps, path_vs_entries, tail_violations,
                                  task_order_violation, unprotected)
from agvtb.world.loader import LoadedWorld
from agvtb.world.states import geom_key

FREEZE_S = 7200.0  # no completion for this long with jobs in the system: global deadlock (design §11)
PARTIAL_S = 1800.0  # a job held this long: partial deadlock (design §11)


@dataclass(frozen=True)
class SimConfig:
    n_vehicles: int
    rate_per_min: float
    seed: int = 0
    warmup_s: float = 3600.0
    measure_s: float = 14400.0
    dt: float = 0.5
    budget: int | None = None
    fromto: Mapping[tuple[str, str], float] | None = None


@dataclass(frozen=True)
class Commit:
    t: float
    vid: int
    legs: tuple[str, ...]
    start_sid: str
    start_t: float
    moving: bool  # the vehicle was still running its earlier path when this one was committed
    fallback: bool


@dataclass(frozen=True)
class SimResult:
    gate: GateReport
    kpi: dict
    deadlock: dict
    checks: dict[str, list]
    stats: dict[str, int]
    events_sha256: str
    n_events: int
    t_end: float
    jobs: list[Job]
    commits: list[Commit]

    @property
    def ok(self) -> bool:
        d = self.deadlock
        return (not d["global"] and d["partial_jobs"] == 0 and d["held_at_end"] == 0 and not d["wait_cycles"]
                and not any(self.checks.values()))


def live_constraints(tapg: Tapg, executor: Executor, now: float) -> dict[int, list[tuple[str, float, float]]]:
    """Each vehicle's reservations (key, lo, hi) as the planner sees them now (DAIM vis:6388-6462, research §A4):
    the current state from now, the claimed states from now to their end, the rest at their plan times."""
    graph, conflicts = tapg.graph, tapg.conflicts

    def cost(sid: str) -> float:
        return graph[sid].duration_s or 0.0

    out: dict[int, list[tuple[str, float, float]]] = {}
    for vid in sorted(tapg.paths):
        path, v = tapg.paths[vid], executor.vehicles[vid]
        cons: list[tuple[str, float, float]] = []
        idx = min(v.path_idx, len(path) - 1)
        cur = path[idx][0]
        end_cur = max(path[idx + 1][1], now) if idx + 1 < len(path) else INF
        cons.append((geom_key(cur), now, end_cur))
        cons += [(x, max(0.0, now - cost(x)), end_cur) for x in affects(conflicts, cur)]
        claimed = path[idx:v.claim_idx]
        for ci, (sid, _t) in enumerate(claimed):
            if ci + 1 < len(claimed):
                end = claimed[ci + 1][1]
            else:
                end = path[v.claim_idx][1] if v.claim_idx < len(path) else INF
            cons.append((geom_key(sid), now, end))
            cons += [(x, max(0.0, now - cost(x)), end) for x in affects(conflicts, sid)]
        if v.claim_idx < len(path):
            cons += path_constraints(path[v.claim_idx:], graph, conflicts)
        out[vid] = cons
    return out


def _table(cons: Mapping[int, list[tuple[str, float, float]]], exclude: int) -> dict[str, list[tuple[float, float]]]:
    table: dict[str, list[tuple[float, float]]] = {}
    for vid in sorted(cons):
        if vid != exclude:
            for key, lo, hi in cons[vid]:
                table.setdefault(key, []).append((lo, hi))
    return table


class _Run:
    def __init__(self, loaded: LoadedWorld, candidates: Iterable[str], cfg: SimConfig):
        self.cfg = cfg
        self.graph, self.conflicts, world = loaded.graph, loaded.conflicts, loaded.world
        self.gate = check_gate(self.graph, self.conflicts, world.port_nodes, candidates, cfg.n_vehicles)
        if not self.gate.ok:
            raise GateError(self.gate)
        self.sidings = list(self.gate.sidings)
        self.kernel = Kernel()
        self.tapg = Tapg(self.graph, self.conflicts)
        self.events: list[tuple] = []
        self.ex = Executor(self.graph, self.tapg, self.kernel, world_geometry(world), world.accel_mps2,
                           world.decel_mps2, self._on_advance, self._on_exec_event)
        self.planner = SafeTailPlanner(self.graph, self.conflicts, self.sidings, budget=cfg.budget)
        self.ledger = SidingLedger(self.sidings)
        vids = list(range(cfg.n_vehicles))
        self.mcs = Mcs(self.kernel, world.dwell_s, vids, self._dist, self._dispatch, self._on_complete)
        fromto = uniform_fromto(world.port_nodes) if cfg.fromto is None else normalize(cfg.fromto)
        self.demand = Demand(fromto, cfg.rate_per_min, random.Random(cfg.seed))
        self._dist_cache: dict[str, dict[str, float]] = {}
        self.pending: dict[int, list[str]] = {}
        self.dispatch_t: dict[int, float] = {}
        self.arrival: dict[int, int] = {}
        self.notified: dict[int, bool] = {}
        self.span: dict[int, tuple[int, int, int]] = {}  # vid -> (job id, lo, hi) of its committed task
        self.commits: list[Commit] = []
        self.checks: dict[str, list] = {k: [] for k in ("plan_overlaps", "order", "early_fallbacks", "tails",
                                                        "unprotected", "event_cycles", "executed_overlaps")}
        self.counts = {"plan_fail": 0, "inline_miss": 0, "event_cycles": 0, "commits": 0, "unplanned_delivery": 0}
        self.cycles_seen: set[tuple[int, ...]] = set()
        self.wait_cycles: list[tuple[float, tuple[int, ...]]] = []
        self.last_completion = 0.0
        self.n_completed = 0
        self.frozen = False
        self.snap: dict | None = None
        starts = random.Random(cfg.seed).sample(self.sidings, cfg.n_vehicles)
        for vid, siding in zip(vids, starts):
            self.ledger.assign(vid, siding)
            self.tapg.add_vehicle(vid, initial_stop(self.graph, siding), 0.0)
            self.ex.add(vid)
        self.kernel.schedule(cfg.warmup_s, MCS, self._snapshot)
        for t, src in self.demand.first_arrivals():
            self.kernel.schedule(t, MCS, self._arrival, src)
        self.kernel.schedule(FREEZE_S, MCS, self._freeze_check, 0)

    # --- logging -----------------------------------------------------------------------------------------------
    def _log(self, t: float, kind: str, who: int, data: tuple = ()) -> None:
        self.events.append((round(t, 9), kind, who, data))

    def _on_exec_event(self, t: float, kind: str, vid: int, data: tuple) -> None:
        self._log(t, kind, vid, data)
        if kind == "block":
            self._find_cycles(t)

    def _find_cycles(self, t: float) -> None:
        for cyc in wait_cycles(self.ex.blocked_by()):
            if cyc not in self.cycles_seen:
                self.cycles_seen.add(cyc)
                self.wait_cycles.append((t, cyc))
                self._log(t, "wait_cycle", cyc[0], cyc)

    # --- demand and jobs ---------------------------------------------------------------------------------------
    def _dist(self, vid: int, src: str) -> float:
        table = self._dist_cache.get(src)
        if table is None:
            table = self._dist_cache[src] = reverse_times(self.graph, [src])
        return table.get(self.tapg.paths[vid][-1][0], INF)

    def _arrival(self, t: float, src: str) -> None:
        dst, nxt = self.demand.on_arrival(t, src)
        self._log(t, "create", len(self.mcs.jobs), (src, dst))
        self.mcs.create(t, src, dst)
        self.kernel.schedule(nxt, MCS, self._arrival, src)

    def _dispatch(self, vid: int, job: Job, goal: str, t: float) -> None:
        self._log(t, "dispatch", vid, (job.jid, goal))
        self.dispatch_t.setdefault(vid, t)
        if goal == job.src:
            self.pending[vid] = [job.src, job.dst]
        else:
            idx = self._inline_arrival(vid, job)
            self.notified[vid] = False
            if idx is None:
                self.counts["inline_miss"] += 1
                self.counts["unplanned_delivery"] += 1
                self.arrival.pop(vid, None)
                self.pending[vid] = [job.dst]
            else:
                self.arrival[vid] = idx
        self._replan(t)

    def _inline_arrival(self, vid: int, job: Job) -> int | None:
        """The index of the stop before the unload in the vehicle's committed task (research §A7)."""
        span = self.span.get(vid)
        if span is None or span[0] != job.jid:
            return None
        v, path = self.ex.vehicles[vid], self.tapg.paths[vid]
        lo, hi = max(v.path_idx, span[1]), min(len(path), span[2])
        if v.dwelling and lo < len(path) and path[lo][0].startswith("L,"):
            lo += 1
        hits = [i for i in range(lo, hi) if path[i][0].startswith("L,") and self.graph[path[i][0]].nodes[0] == job.dst]
        return hits[-1] - 1 if hits else None

    def _on_complete(self, job: Job) -> None:
        t = job.t_completed
        self._log(t, "complete", job.jid, (job.vehicle,))
        self.n_completed += 1
        self.last_completion = t
        self.kernel.schedule(t + FREEZE_S, MCS, self._freeze_check, self.n_completed)

    def _freeze_check(self, t: float, seen: int) -> None:
        backlog = len(self.mcs.waiting()) + len(self.mcs.held)
        if self.n_completed == seen and backlog > 0:
            self.frozen = True
            self._log(t, "frozen", seen, ())
            self.kernel.stop()

    def _snapshot(self, t: float) -> None:
        self.snap = {"t": t, "created": len(self.mcs.jobs),
                     "busy": sum(self.mcs.busy_s(v, t) for v in self.mcs.phase)}

    # --- arrival notices ---------------------------------------------------------------------------------------
    def _on_advance(self, vid: int, path_idx: int, t: float) -> None:
        self._check_arrival(vid, t)

    def _check_arrival(self, vid: int, t: float) -> None:
        idx = self.arrival.get(vid)
        if idx is not None and not self.notified.get(vid, False) and self.ex.vehicles[vid].path_idx >= idx:
            self.notified[vid] = True
            self._log(t, "arrived", vid, (idx,))
            self.kernel.schedule(t, MCS, self.mcs.vehicle_arrived, vid)

    # --- planning ----------------------------------------------------------------------------------------------
    def _prune(self, t: float) -> None:
        for vid in sorted(self.tapg.paths):
            v = self.ex.vehicles[vid]
            n = min(v.path_idx, len(self.tapg.paths[vid]) - 1)
            if n <= 0:
                continue
            self.ex.wake(self.tapg.prune(vid, n), t)
            self.ex.shift(vid, n)
            if vid in self.arrival:
                self.arrival[vid] -= n
                if self.arrival[vid] < 0:
                    del self.arrival[vid]
            if vid in self.span:
                jid, lo, hi = self.span[vid]
                self.span[vid] = (jid, lo - n, hi - n)

    def _replan(self, t: float) -> None:
        if not self.pending:
            return
        self._prune(t)
        self.tapg.recompute(t, self.ex.inprog_end())
        self.ex.wake_ready(t)
        tasks = sorted((self.dispatch_t.get(vid, t), vid) for vid in self.pending)
        cons = live_constraints(self.tapg, self.ex, t)
        for _dt, vid in tasks:
            legs = self.pending[vid]
            path = self.tapg.paths[vid]
            start_sid, start_t = path[-1]
            res = self.planner.plan_task(start_sid, start_t, legs, _table(cons, vid), self.ledger.candidates(vid),
                                         self.ledger.home(vid))
            if res is None:
                self.counts["plan_fail"] += 1
                self._log(t, "plan_fail", vid, tuple(legs))
                continue
            self._check_plan(t, vid, res, legs)
            old_len = len(path)
            moving = self.ex.vehicles[vid].state != "DONE"
            self.tapg.append(vid, res.path)
            if self.tapg.has_cycle():
                self.counts["event_cycles"] += 1
                self.checks["event_cycles"].append({"t": t, "vid": vid})
            self.ledger.transfer(vid, res.siding)
            i_l = next(i for i, (sid, _) in enumerate(res.path) if sid.startswith("L,")
                       and self.graph[sid].nodes[0] == legs[0])
            self.arrival[vid] = old_len + i_l - 2
            self.notified[vid] = False
            self.tapg.mark_dwell(vid, self.arrival[vid])
            job = self.mcs.held[vid]
            self.span[vid] = (job.jid, old_len, old_len + len(res.path) - 1)
            cons[vid] = [c for c in cons[vid] if c[2] != INF] + path_constraints(res.path, self.graph, self.conflicts)
            self._check_tapg(t, vid, old_len)
            self.ex.kick(vid, t)
            self.commits.append(Commit(t, vid, tuple(legs), start_sid, start_t, moving, res.fallback))
            self.counts["commits"] += 1
            self._log(t, "commit", vid, (tuple(legs), res.siding, res.fallback))
            del self.pending[vid]
            self.dispatch_t.pop(vid, None)
            self._check_arrival(vid, t)
        self._find_cycles(t)

    def _check_plan(self, t: float, vid: int, res, legs) -> None:
        others = {u: live_entries(self.tapg.paths[u], self.ex.vehicles[u].path_idx, self.ex.vehicles[u].claim_idx,
                                  t) for u in sorted(self.tapg.paths) if u != vid}
        for o in path_vs_entries(res.path, others, self.conflicts):
            self.checks["plan_overlaps"].append({"t": t, "vid": vid, **o})
        found = task_order_violation(res.path, legs, self.sidings)
        if found is not None:
            self.checks["order"].append({"t": t, "vid": vid, "what": found})
        if res.fallback and early_fallback(res.path, res.t_star):
            self.checks["early_fallbacks"].append({"t": t, "vid": vid})

    def _check_tapg(self, t: float, vid: int, lo: int) -> None:
        for what in tail_violations(self.tapg.paths, self.ledger.snapshot(), self.sidings):
            self.checks["tails"].append({"t": t, "what": what})
        path = self.tapg.paths[vid]
        mine = [(k, path[k][0], self.tapg.key(vid, k)) for k in range(lo, len(path)) if self.tapg.key(vid, k) in self.tapg]
        others = {}
        for u in sorted(self.tapg.paths):
            if u == vid:
                continue
            pu = self.tapg.paths[u]
            others[u] = [(j, pu[j][0], self.tapg.key(u, j)) for j in range(self.ex.vehicles[u].path_idx, len(pu))
                         if self.tapg.key(u, j) in self.tapg]
        for o in unprotected(mine, others, self.tapg.related, self.conflicts):
            self.checks["unprotected"].append({"t": t, "vid": vid, **o})

    # --- the run -----------------------------------------------------------------------------------------------
    def run(self) -> SimResult:
        cfg = self.cfg
        self.kernel.run_until(cfg.warmup_s + cfg.measure_s, cfg.dt)
        t_end = self.kernel.now
        timelines = {vid: v.executed for vid, v in self.ex.vehicles.items()}
        self.checks["executed_overlaps"] = pairwise_overlaps(timelines, self.conflicts)
        stats = dict(sorted(self.planner.stats.items())) | self.counts | {"stray_arrivals": self.mcs.stray_arrivals}
        blob = json.dumps(self.events, separators=(",", ":")).encode()
        return SimResult(self.gate, self._kpi(t_end), self._deadlock(t_end), self.checks, stats,
                         hashlib.sha256(blob).hexdigest(), len(self.events), t_end, self.mcs.jobs, self.commits)

    def _kpi(self, t_end: float) -> dict:
        w0 = self.cfg.warmup_s
        window = t_end - w0
        done = [j for j in self.mcs.jobs if j.t_completed is not None and w0 <= j.t_completed <= t_end]
        n = len(done)

        def mean(values):
            values = list(values)
            return sum(values) / len(values) if values else None

        busy0 = self.snap["busy"] if self.snap else 0.0
        created0 = self.snap["created"] if self.snap else 0
        busy1 = sum(self.mcs.busy_s(v, t_end) for v in self.mcs.phase)
        ok_window = window > 0 and self.snap is not None
        return {
            "completed": n,
            "throughput_per_h": n / window * 3600 if ok_window else None,
            "cycle_s": mean(j.t_completed - j.t_created for j in done),
            "wait_s": mean(j.t_assigned - j.t_created for j in done),
            "retrieve_s": mean(j.t_pickup_end - j.t_assigned for j in done),
            "deliver_s": mean(j.t_completed - j.t_pickup_end for j in done),
            "utilization": (busy1 - busy0) / (window * len(self.mcs.phase)) if ok_window else None,
            "arrivals_per_min": (len(self.mcs.jobs) - created0) / window * 60 if ok_window else None,
            "backlog_end": len(self.mcs.waiting()) + len(self.mcs.held),
        }

    def _deadlock(self, t_end: float) -> dict:
        w0 = self.cfg.warmup_s
        partial = sum(1 for j in self.mcs.jobs if j.t_completed is not None and w0 <= j.t_completed <= t_end
                      and j.t_completed - j.t_assigned >= PARTIAL_S)
        held = sum(1 for j in self.mcs.held.values() if t_end - j.t_assigned >= PARTIAL_S)
        return {"global": self.frozen, "frozen_at": self.last_completion if self.frozen else None,
                "partial_jobs": partial, "held_at_end": held,
                "wait_cycles": [[t, list(c)] for t, c in self.wait_cycles]}


def simulate(loaded: LoadedWorld, candidates: Iterable[str], cfg: SimConfig) -> SimResult:
    """One lifelong run. GateError if the gate refuses the candidates for cfg.n_vehicles."""
    return _Run(loaded, candidates, cfg).run()
