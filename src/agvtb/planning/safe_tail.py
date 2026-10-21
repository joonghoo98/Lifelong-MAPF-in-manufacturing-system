"""A1, the safe-tail planner: one staged SIPP search per task, a fallback plan when it fails.

A re-implementation of DAIM's core/safe_tail_planner.py (decision D8: the shared document's algorithm as it
is) on M1's state graph. The expressions keep DAIM's order, down to where EPS stands, so the paths come out bit
for bit the same as DAIM's on the same input; the tests compare their sha256. "DAIM :n" is a line of that file.

A task is: from the start stop, load at legs[0], unload at legs[1] (and so on), then park on a candidate
siding nobody will need later. The search node is (stage k, stop state, safe interval of its key).
"""
import collections
import heapq
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from agvtb.planning.intervals import EPS, INF, interval_at, safe_intervals
from agvtb.planning.reservations import affects
from agvtb.world.states import Conflicts, State, StateGraph, geom_key

Table = Mapping[str, Sequence[tuple[float, float]]]  # key -> reserved intervals


def _cost(state: State) -> float:
    """How long a state lasts: an S no fixed time, the others their duration."""
    return 0.0 if state.kind == "S" else float(state.duration_s)


def reverse_times(graph: StateGraph, goal_nodes: Iterable[str]) -> dict[str, float]:
    """state -> the least time from it to any stop state of goal_nodes (DAIM :108-131). Unreachable: no key."""
    goals = set(goal_nodes)
    radj: dict[str, list[tuple[str, float]]] = collections.defaultdict(list)
    for sid, state in graph.items():
        cost = _cost(state)
        for nxt in state.next:
            radj[nxt].append((sid, cost))
    dist: dict[str, float] = {}
    heap = []
    for sid, state in graph.items():
        if state.kind == "S" and state.nodes[0] in goals:
            dist[sid] = 0.0
            heap.append((0.0, sid))
    heapq.heapify(heap)
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, INF):
            continue
        for prev, cost in radj.get(cur, ()):
            nd = d + cost
            if nd < dist.get(prev, INF):
                dist[prev] = nd
                heapq.heappush(heap, (nd, prev))
    return dist


@dataclass(frozen=True)
class TaskPlan:
    path: list[tuple[str, float]]  # first element: the start stop and the start time
    siding: str  # the node the path ends on
    fallback: bool  # made by the fallback plan
    expansions: int  # search expansions (0 for a fallback plan)
    t_star: float = 0.0  # the fallback plan's T*, the earliest it may leave; 0 for a search result


class SafeTailPlanner:
    """A1 on one state graph and siding set. stats counts outcomes with DAIM's keys."""

    DEFAULT_BUDGET = 200_000  # search expansions; over it the fallback plan takes over

    def __init__(
        self, graph: StateGraph, conflicts: Conflicts, sidings: Iterable[str], *, budget: int | None = None
    ):
        if budget is None:
            budget = self.DEFAULT_BUDGET
        elif budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        dwells = sorted({state.duration_s for state in graph.values() if state.kind == "L"})
        if len(dwells) > 1:
            raise ValueError(f"the L states must all last the same time, got {dwells}")
        self.graph = graph
        self.conflicts = conflicts
        self.sidings = sorted({str(s) for s in sidings})
        self.budget = budget
        self.dwell_s = float(dwells[0]) if dwells else 0.0
        self.stats: collections.Counter[str] = collections.Counter()
        self.expansions_ok: list[int] = []
        self._stops_at: dict[str, list[str]] = collections.defaultdict(list)
        for sid in sorted(graph):
            if graph[sid].kind == "S":
                self._stops_at[graph[sid].nodes[0]].append(sid)
        self._h_sidings = reverse_times(graph, self.sidings)
        self._h_stage: dict[str, dict[str, float]] = {}  # stage node -> reverse_times to it
        self._min_cache: dict[tuple, float] = {}

    # --- heuristic (DAIM :276-298) ---------------------------------------------------------------------------
    def _h_to(self, node: str) -> dict[str, float]:
        if node not in self._h_stage:
            self._h_stage[node] = reverse_times(self.graph, [node])
        return self._h_stage[node]

    def _min_over(self, node: str, table: Mapping[str, float], key: tuple) -> float:
        v = self._min_cache.get(key)
        if v is None:
            v = min((table.get(s, INF) for s in self._stops_at.get(node, ())), default=INF)
            self._min_cache[key] = v
        return v

    def _hfun(self, legs: Sequence[str]):
        """Admissible: time to the next stage's node + the dwells and least times still to come."""
        m = len(legs)
        H = [self._h_to(p) for p in legs]
        rest = [0.0] * m
        for k in range(m - 1, -1, -1):
            if k + 1 < m:
                after = self._min_over(legs[k], H[k + 1], ("pp", legs[k], legs[k + 1])) + rest[k + 1]
            else:
                after = self._min_over(legs[k], self._h_sidings, ("ps", legs[k]))
            rest[k] = self.dwell_s + after
        hs = self._h_sidings

        def h(k: int, sid: str) -> float:
            return H[k].get(sid, INF) + rest[k] if k < m else hs.get(sid, INF)
        return h

    # --- safe intervals and parking (DAIM :261-274, :300-303) ---------------------------------------------------
    @staticmethod
    def _ivs(key: str, table: Table, cache: dict) -> list[tuple[float, float]]:
        iv = cache.get(key)
        if iv is None:
            iv = safe_intervals(list(table.get(key, ())))
            cache[key] = iv
        return iv

    def _parking_ok(self, sid: str, t: float, table: Table) -> bool:
        """No reservation of the stop's key or its affects ends after t: nobody needs the place later."""
        keys = [geom_key(sid)] + [geom_key(x) for x in affects(self.conflicts, sid)]
        return not any(hi > t + EPS for key in keys for _, hi in table.get(key, ()))

    # --- the search (DAIM :305-385) ------------------------------------------------------------------------------
    def _succ(self, k, sid, iv, t, legs, table, cache):
        """Stop (k, sid, iv) reached at t -> [(k2, s2, iv2, arrival, path piece)]: per (move interval, arrival
        interval) pair, the earliest departure."""
        out = []
        node = self.graph[sid].nodes[0]
        for nxt in self.graph[sid].next:
            if nxt.startswith("L,"):
                if k < len(legs) and node == legs[k]:
                    a = t + self.dwell_s
                    if a < iv[1] - EPS:  # the dwell [t, a) and the stop after it lie in the same safe interval
                        out.append((k + 1, sid, iv, a, ((nxt, t), (sid, a))))
                continue
            act = self.graph[nxt]
            c = float(act.duration_s)
            for s2 in act.next:
                if self.graph[s2].kind != "S":
                    continue
                kivs = self._ivs(geom_key(s2), table, cache)
                for jlo, jhi in self._ivs(geom_key(nxt), table, cache):
                    if jlo > iv[1] + EPS:
                        break  # the stop cannot wait that long
                    d0 = max(t, jlo)
                    if d0 + c > jhi + EPS:
                        continue
                    for klo, khi in kivs:
                        d = max(d0, klo - c)
                        if d > iv[1] + EPS or d + c > jhi + EPS:
                            break
                        a = d + c
                        if a >= khi - EPS:
                            continue
                        out.append((k, s2, (klo, khi), a, ((nxt, d), (s2, a))))
        return out

    def search(
        self, start_sid: str, t0: float, legs: Sequence[str], table: Table, candidates: Iterable[str]
    ) -> TaskPlan | None:
        """The staged SIPP search. None on failure, counted in stats; a found plan is not counted here."""
        t0 = float(t0)  # DAIM's plan_task converts it (:160)
        start_state = self.graph.get(start_sid)
        if start_state is None or start_state.kind != "S":
            self.stats["start_invalid"] += 1
            return None
        cache: dict[str, list[tuple[float, float]]] = {}
        iv0 = interval_at(self._ivs(geom_key(start_sid), table, cache), t0)
        if iv0 is None:
            self.stats["start_blocked"] += 1
            return None
        m = len(legs)
        h = self._hfun(legs)
        h0 = h(0, start_sid)
        if h0 == INF:
            self.stats["unreachable"] += 1
            return None
        targets = set(candidates)
        start = (0, start_sid, iv0)
        best = {start: t0}
        parent: dict[tuple, tuple | None] = {start: None}
        heap = [(t0 + h0, t0, 0, start_sid, iv0)]
        n = 0
        while heap:
            _, t, k, sid, iv = heapq.heappop(heap)
            key = (k, sid, iv)
            if t > best.get(key, INF) + EPS:
                continue
            n += 1
            if n > self.budget:
                self.stats["budget_exceeded"] += 1
                return None
            node = self.graph[sid].nodes[0]
            if k == m and iv[1] == INF and node in targets and self._parking_ok(sid, t, table):
                return TaskPlan(self._rebuild(parent, key, start_sid, t0), node, False, n)
            for k2, s2, iv2, a, seg in self._succ(k, sid, iv, t, legs, table, cache):
                key2 = (k2, s2, iv2)
                if a < best.get(key2, INF) - EPS:
                    hv = h(k2, s2)
                    if hv == INF:
                        continue
                    best[key2] = a
                    parent[key2] = (key, seg)
                    heapq.heappush(heap, (a + hv, a, k2, s2, iv2))
        self.stats["search_exhausted"] += 1
        return None

    # --- the task: search, else the fallback plan (DAIM :155-259) ---------------------------------------------------
    def plan_task(
        self,
        start_sid: str,
        start_time: float,
        legs: Sequence[str],
        table: Table,
        candidates: Iterable[str],
        home: str | None,
    ) -> TaskPlan | None:
        """One task: the staged SIPP search, else the fallback plan. None if neither finds one (a map condition
        is broken, or a bug). Wall-clock time is not measured (results stay deterministic)."""
        legs = [str(p) for p in legs]
        res = self.search(start_sid, float(start_time), legs, table, {str(c) for c in candidates})
        if res is not None:
            self.stats["search_ok"] += 1
            self.expansions_ok.append(res.expansions)
            return res
        fb = self.fallback(start_sid, float(start_time), legs, table, home)
        if fb is None:
            self.stats["fallback_fail"] += 1
            return None
        self.stats["fallback_used"] += 1
        return fb

    def path_conflicts(self, path: Sequence[tuple[str, float]], table: Table) -> list[tuple[int, str]]:
        """Where path overlaps the table: [(path index, key)]; empty means none. The last element, if an S, is
        held forever, and its affects are looked at too."""
        bad = []
        for i, (sid, t) in enumerate(path):
            end = path[i + 1][1] if i + 1 < len(path) else INF
            keys = [geom_key(sid)]
            if i + 1 == len(path) and self.graph[sid].kind == "S":
                keys += [geom_key(x) for x in affects(self.conflicts, sid)]
            for key in keys:
                if any(lo < end - EPS and hi > t + EPS for lo, hi in table.get(key, ())):
                    bad.append((i, key))
                    break
        return bad

    def fallback(
        self, start_sid: str, t_start: float, legs: Sequence[str], table: Table, home: str | None
    ) -> TaskPlan | None:
        """Wait on the start stop until T* (the last finite reservation end), then the static shortest path
        through the stages and home on a graph where only the keys reserved forever are blocked. Checked against
        the table at the end: an overlap makes it invalid."""
        start_state = self.graph.get(start_sid)
        if start_state is None or start_state.kind != "S" or home is None:
            return None
        legs = [str(p) for p in legs]
        t_star = max((hi for ivs in table.values() for _, hi in ivs if hi != INF), default=0.0)
        t0 = max(float(t_start), t_star)
        dead = {key for key, ivs in table.items() if any(hi == INF for _, hi in ivs)}
        m = len(legs)
        dist = {(0, start_sid): 0.0}
        parent: dict[tuple, tuple | None] = {(0, start_sid): None}
        heap = [(0.0, 0, start_sid)]
        goal = None
        while heap:
            d, k, sid = heapq.heappop(heap)
            if d > dist.get((k, sid), INF) + EPS:
                continue
            node = self.graph[sid].nodes[0]
            if k == m and node == str(home):
                goal = (k, sid)
                break
            for nxt in self.graph[sid].next:
                if nxt.startswith("L,"):
                    if k < m and node == legs[k]:
                        self._relax(dist, parent, heap, (k, sid), (k + 1, sid), d + self.dwell_s,
                                    ((nxt, d), (sid, d + self.dwell_s)))
                    continue
                act = self.graph[nxt]
                if geom_key(nxt) in dead:
                    continue
                for s2 in act.next:
                    if self.graph[s2].kind == "S" and geom_key(s2) not in dead:
                        nd = d + float(act.duration_s)
                        self._relax(dist, parent, heap, (k, sid), (k, s2), nd, ((nxt, d), (s2, nd)))
        if goal is None:
            return None
        segs, key = [], goal
        while parent[key] is not None:
            prev, seg = parent[key]
            segs.append(seg)
            key = prev
        path = [(start_sid, float(t_start))]
        for seg in reversed(segs):
            path.extend((s, t0 + rel) for s, rel in seg)
        if self.path_conflicts(path, table):
            self.stats["fallback_invalid"] += 1
            return None
        return TaskPlan(path, str(home), True, 0, t_star)

    @staticmethod
    def _relax(dist, parent, heap, key, key2, nd, seg) -> None:
        if nd < dist.get(key2, INF) - EPS:
            dist[key2] = nd
            parent[key2] = (key, seg)
            heapq.heappush(heap, (nd, key2[0], key2[1]))

    @staticmethod
    def _rebuild(parent, key, start_sid, t0) -> list[tuple[str, float]]:
        segs = []
        while parent[key] is not None:
            prev, seg = parent[key]
            segs.append(seg)
            key = prev
        path = [(start_sid, t0)]
        for seg in reversed(segs):
            path.extend(seg)
        return path
