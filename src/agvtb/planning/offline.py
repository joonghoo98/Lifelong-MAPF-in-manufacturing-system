"""Offline planning of many vehicles: a test scene for A1, with the independent validator run on the result.

The rules (the M3 plan, Claude's call 6; demand and dispatch come in M4):
- the gate is checked first; if it refuses, nothing is planned (user decision G1);
- each vehicle starts on a siding drawn with the seed from the usable ones, on its stop pose of heading 0
  (else the smallest stop id);
- task i goes to vehicle i mod N at time i * interval, from one port to another, both drawn with the same rng;
- the tasks are planned one by one in (dispatch time, vehicle, index) order. A vehicle starts a task where and
  when its path so far ends, or at the dispatch time if that is later. The reservation table holds the whole
  path so far of every other vehicle.
"""
import hashlib
import json
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from agvtb.planning.gate import GateError, GateReport, check_gate
from agvtb.planning.ledger import SidingLedger
from agvtb.planning.reservations import ReservationTable
from agvtb.planning.safe_tail import SafeTailPlanner, TaskPlan
from agvtb.validate.plans import early_fallback, pairwise_overlaps, tail_violations, task_order_violation
from agvtb.world.states import Conflicts, StateGraph


@dataclass(frozen=True)
class OfflineTask:
    index: int
    dispatch_s: float
    vehicle: int
    legs: tuple[str, ...]  # load port, unload port


@dataclass(frozen=True)
class OfflineResult:
    gate: GateReport
    start_sidings: dict[int, str]  # vehicle -> the siding it starts on
    tasks: list[OfflineTask]  # in index order
    plans: list[TaskPlan | None]  # plans[i] is the plan of task i
    timelines: dict[int, list[tuple[str, float]]]  # vehicle -> its whole path
    homes: dict[int, str]  # vehicle -> the siding it owns at the end
    stats: dict[str, int]  # the planner's counts, sorted by key
    checks: dict[str, list]  # the validator's findings, by check

    @property
    def ok(self) -> bool:
        """Every task planned and no check found anything."""
        return not any(self.checks.values())


def initial_stop(graph: StateGraph, node: str) -> str:
    """The stop pose a vehicle starts on at node: S,<node>,0, else the smallest S id of node."""
    zero = f"S,{node},0"
    if zero in graph:
        return zero
    stops = sorted(sid for sid, state in graph.items() if state.kind == "S" and state.nodes[0] == node)
    if not stops:
        raise ValueError(f"node {node} has no stop state")
    return stops[0]


def make_tasks(
    ports: Iterable[str], n_vehicles: int, n_tasks: int, interval_s: float, rng: random.Random
) -> list[OfflineTask]:
    """Task i: vehicle i mod n_vehicles, dispatched at i * interval_s, between two ports rng draws."""
    port_list = sorted(set(ports))
    if n_tasks > 0 and len(port_list) < 2:
        raise ValueError(f"a task needs two ports, the map has {port_list}")
    return [OfflineTask(i, i * interval_s, i % n_vehicles, tuple(rng.sample(port_list, 2))) for i in range(n_tasks)]


def plan_offline(
    graph: StateGraph,
    conflicts: Conflicts,
    ports: Iterable[str],
    candidates: Iterable[str],
    n_vehicles: int,
    n_tasks: int,
    *,
    seed: int = 0,
    interval_s: float = 0.0,
    budget: int | None = None,
) -> OfflineResult:
    """Plan n_tasks tasks of n_vehicles vehicles on the candidates that pass the gate, and validate the plans.

    Raises ValueError for n_vehicles < 1, n_tasks < 0 or an interval that is not finite and >= 0, and
    GateError (planning nothing) if the gate refuses the candidates for n_vehicles.
    """
    if n_vehicles < 1:
        raise ValueError(f"n_vehicles must be >= 1, got {n_vehicles}")
    if n_tasks < 0:
        raise ValueError(f"n_tasks must be >= 0, got {n_tasks}")
    if not (math.isfinite(interval_s) and interval_s >= 0):
        raise ValueError(f"interval_s must be finite and >= 0, got {interval_s}")
    if not math.isfinite(max(n_tasks - 1, 0) * interval_s):
        raise ValueError(f"interval_s {interval_s} makes the last dispatch time of {n_tasks} tasks overflow")
    interval_s = float(interval_s)  # an int interval would make int dispatch times and change the digests
    ports = list(ports)
    gate = check_gate(graph, conflicts, ports, candidates, n_vehicles)
    if not gate.ok:
        raise GateError(gate)
    usable = list(gate.sidings)
    rng = random.Random(seed)
    starts = rng.sample(usable, n_vehicles)
    tasks = make_tasks(ports, n_vehicles, n_tasks, interval_s, rng)

    planner = SafeTailPlanner(graph, conflicts, usable, budget=budget)
    ledger = SidingLedger(usable)
    table = ReservationTable(graph, conflicts)
    timelines: dict[int, list[tuple[str, float]]] = {}
    for v, siding in enumerate(starts):
        ledger.assign(v, siding)
        timelines[v] = [(initial_stop(graph, siding), 0.0)]
        table.set_path(v, timelines[v])
    plans: list[TaskPlan | None] = [None] * n_tasks
    for task in sorted(tasks, key=lambda t: (t.dispatch_s, t.vehicle, t.index)):
        v = task.vehicle
        start_sid, t_last = timelines[v][-1]
        t0 = max(task.dispatch_s, t_last)
        plan = planner.plan_task(start_sid, t0, list(task.legs), table.table(exclude=v), ledger.candidates(v),
                                 ledger.home(v))
        plans[task.index] = plan
        if plan is not None:
            timelines[v] = timelines[v] + plan.path[1:]
            table.set_path(v, timelines[v])
            ledger.transfer(v, plan.siding)

    homes = ledger.snapshot()
    order = []
    for task in tasks:
        plan = plans[task.index]
        if plan is not None:
            found = task_order_violation(plan.path, task.legs, usable)
            if found is not None:
                order.append(f"task {task.index}: {found}")
    checks = {
        "overlaps": pairwise_overlaps(timelines, conflicts),
        "tails": tail_violations(timelines, homes, usable),
        "order": order,
        "early_fallbacks": [t.index for t in tasks
                            if plans[t.index] is not None and plans[t.index].fallback
                            and early_fallback(plans[t.index].path, plans[t.index].t_star)],
        "unplanned": [t.index for t in tasks if plans[t.index] is None],
    }
    return OfflineResult(gate, dict(enumerate(starts)), tasks, plans, timelines, homes,
                         dict(sorted(planner.stats.items())), checks)


def timelines_digest(timelines) -> str:
    """sha256 of the JSON of {vehicle: path}, vehicles in order and named as strings (as tests/helpers.py)."""
    data = {str(v): [[s, t] for s, t in timelines[v]] for v in sorted(timelines)}
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def result_json(result: OfflineResult) -> dict:
    """The result as plain data. No wall-clock value: the same input gives the same output."""
    return {
        "ok": result.ok,
        "gate": result.gate.summary(),
        "start_sidings": {str(v): s for v, s in result.start_sidings.items()},
        "task_list": [{"index": t.index, "dispatch_s": t.dispatch_s, "vehicle": t.vehicle, "legs": list(t.legs)}
                      for t in result.tasks],
        "plans": [None if p is None else {"path": [[s, t] for s, t in p.path], "siding": p.siding,
                                          "fallback": p.fallback, "expansions": p.expansions, "t_star": p.t_star}
                  for p in result.plans],
        "timelines": {str(v): [[s, t] for s, t in tl] for v, tl in result.timelines.items()},
        "timelines_sha256": timelines_digest(result.timelines),
        "homes": {str(v): s for v, s in result.homes.items()},
        "stats": result.stats,
        "checks": result.checks,
    }
