"""Split SIPP, the fast path's searches (M6 design §5.5, spec §2.3-§2.4 and §4.4) on an Occupancy.

The successor rule is SafeTailPlanner._succ's (safe_tail.py:153-185, DAIM's expressions; that file stays as it is, G4),
with safe intervals that carry the vehicles to evict: (lo, hi, owners). A node is (stage k, stop, interval). The free
and the eviction part of a key are different intervals, so neither dominates the other (spec §2.4). g is the earliest
arrival; the heap sorts by f = g + h + C_EVICT x (vehicles evicted on the branch) (CE-1, design decision 6).
Three searches share it: staged (a task's legs, spec MultigoalSIPP), nearest_rest (FindNearestIdleSpot: h = 0, the
cost PHI_S x cell degree added at the goal) and step (SingleGoalSIPP, hard only). blocking lists the evicted vehicles,
the last met first (spec line 108), each with the rest it held when met.
"""
import collections
import heapq
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from agvtb.planning.intervals import EPS, INF
from agvtb.planning.occupancy import Occupancy
from agvtb.planning.safe_tail import SafeTailPlanner, reverse_times
from agvtb.world.cells import CellMap
from agvtb.world.states import Conflicts, StateGraph, geom_key

C_EVICT = 30.0  # s per evicted vehicle, in the sort key only (design §8, spec §2.4)
PHI_S = 1.0  # s per cell-graph degree of a rest cell, added when it is taken as the goal (design §8, spec §4.4 w1)

Interval = tuple[float, float, tuple[int, ...]]  # (lo, hi, owners to evict)
Node = tuple[int, str, Interval]


@dataclass(frozen=True)
class Found:
    path: list[tuple[str, float]]  # first element: the start stop and the start time
    blocking: tuple[tuple[int, tuple[str, float]], ...]  # (vehicle, its rest when met), the last met first
    expansions: int


def target_locks(path: Sequence[tuple[str, float]], cm: CellMap, graph: StateGraph) -> dict[str, float]:
    """BuildTargetLocks (spec CE-4, design §5.5): cell -> the last time path leaves it; the last element's cells INF."""
    locks: dict[str, float] = {}
    for i, (sid, _) in enumerate(path):
        end = path[i + 1][1] if i + 1 < len(path) else INF
        for node in graph[sid].nodes:
            cell = cm.cell_of[node]
            locks[cell] = max(locks.get(cell, -INF), end)
    return locks


def merge_locks(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    """MergeTargetLocks (spec line 183): the later time per cell."""
    out = dict(a)
    for cell, t in b.items():
        out[cell] = max(out.get(cell, -INF), t)
    return out


class SplitSipp:
    """The searches on one map. stats counts searches over budget (m6_search_budget)."""

    def __init__(self, graph: StateGraph, conflicts: Conflicts, cm: CellMap, stats: collections.Counter, *,
                 budget: int | None = None):
        if budget is None:
            budget = SafeTailPlanner.DEFAULT_BUDGET
        elif budget < 1:
            raise ValueError(f"budget must be >= 1, got {budget}")
        dwells = sorted({state.duration_s for state in graph.values() if state.kind == "L"})
        if len(dwells) > 1:
            raise ValueError(f"the L states must all last the same time, got {dwells}")
        self.graph, self.conflicts, self.cm, self.stats, self.budget = graph, conflicts, cm, stats, budget
        self.dwell_s = float(dwells[0]) if dwells else 0.0
        self.c_evict, self.phi_s = C_EVICT, PHI_S
        self._stops_at: dict[str, list[str]] = collections.defaultdict(list)
        for sid in sorted(graph):
            if graph[sid].kind == "S":
                self._stops_at[graph[sid].nodes[0]].append(sid)
        self._h: dict[str, dict[str, float]] = {}  # node -> reverse_times to it
        self._exit: dict[str, float] = {}

    # --- the three searches ---------------------------------------------------------------------------------------
    def staged(self, occ: Occupancy, me: int, start: str, t0: float, legs: Sequence[str], locked: frozenset[int],
               split: bool) -> Found | None:
        """From start at t0 through the loads at legs, to just after the last one (spec MultigoalSIPP_Split, or
        _HardOnly when not split)."""
        m = len(legs)
        return self._search(occ, me, start, t0, list(legs), locked, split,
                            lambda k, sid, iv, t: frozenset() if k == m else None, self._hfun(legs))

    def nearest_rest(self, occ: Occupancy, me: int, start: str, t0: float, locks: Mapping[str, float],
                     locked: frozenset[int], split: bool) -> Found | None:
        """To a stop it may rest on (spec FindNearestIdleSpot, lines 171-174 and §4.4): its last interval, in a rest
        cell (U2 ①), after locks of that cell, no hard entry later, and the cell's owner none, me, or (split) a vehicle
        to evict that is not locked."""
        cm, graph = self.cm, self.graph

        def goal(k, sid, iv, t):
            if iv[1] != INF:
                return None
            cell = cm.cell_of[graph[sid].nodes[0]]
            if cell not in occ.rest or t <= locks.get(cell, -INF) + EPS or not occ.parking_ok(sid, t, me):
                return None
            owner = occ.owner(cell)
            if owner is None or owner == me:
                return frozenset()
            if split and owner not in locked and occ.rests.get(owner) is not None:
                return frozenset({owner})
            return None

        def phi(sid):
            return self.phi_s * len(cm.adj[cm.cell_of[graph[sid].nodes[0]]])
        return self._search(occ, me, start, t0, [], locked, split, goal, lambda k, sid: 0.0, phi)

    def step(self, occ: Occupancy, me: int, start: str, t0: float, node: str) -> Found | None:
        """To any stop at node, hard only (spec SingleGoalSIPP_HardOnly, line 206)."""
        graph = self.graph
        return self._search(occ, me, start, t0, [], frozenset(), False,
                            lambda k, sid, iv, t: frozenset() if graph[sid].nodes[0] == node else None,
                            lambda k, sid: 0.0)

    def exit_s(self, sid: str) -> float:
        """The protection time of decision 2: how long it takes to leave sid's cell for any neighbour cell. For each
        neighbour cell the quickest way out (M and R states inside the cell, then one move out), the largest of those;
        0 if the cell has no way out. Other vehicles are not looked at."""
        out = self._exit.get(sid)
        if out is None:
            inside = set(self.cm.members[self.cm.cell_of[self.graph[sid].nodes[0]]])
            dist, heap, best = {sid: 0.0}, [(0.0, sid)], {}
            while heap:
                d, cur = heapq.heappop(heap)
                if d > dist[cur]:
                    continue
                for act in self.graph[cur].next:
                    state = self.graph[act]
                    if state.kind not in ("M", "R"):
                        continue
                    if not set(state.nodes) <= inside:  # a move out of the cell
                        cell = self.cm.cell_of[state.nodes[1]]
                        best[cell] = min(best.get(cell, INF), d + state.duration_s)
                        continue
                    for s2 in state.next:
                        nd = d + state.duration_s
                        if nd < dist.get(s2, INF):
                            dist[s2] = nd
                            heapq.heappush(heap, (nd, s2))
            out = max(best.values(), default=0.0)
            self._exit[sid] = out
        return out

    # --- heuristic, successors, the search ----------------------------------------------------------------------------
    def _hfun(self, legs: Sequence[str]) -> Callable[[int, str], float]:
        """Admissible: time to the next stage's node + the dwells and least times still to come; 0 after the last."""
        m = len(legs)
        H = []
        for p in legs:
            if p not in self._h:
                self._h[p] = reverse_times(self.graph, [p])
            H.append(self._h[p])
        rest = [0.0] * m
        for k in range(m - 1, -1, -1):
            after = 0.0
            if k + 1 < m:
                after = min((H[k + 1].get(s, INF) for s in self._stops_at.get(legs[k], ())), default=INF) + rest[k + 1]
            rest[k] = self.dwell_s + after

        def h(k: int, sid: str) -> float:
            return H[k].get(sid, INF) + rest[k] if k < m else 0.0
        return h

    def _succ(self, k, sid, iv, t, legs, ivs):
        """SafeTailPlanner._succ with owners: [(k2, s2, iv2, arrival, path piece, owners of the move and arrival)]."""
        out = []
        node = self.graph[sid].nodes[0]
        for nxt in self.graph[sid].next:
            if nxt.startswith("L,"):
                if k < len(legs) and node == legs[k]:
                    a = t + self.dwell_s
                    if a < iv[1] - EPS:  # the dwell [t, a) and the stop after it lie in the same safe interval
                        out.append((k + 1, sid, iv, a, ((nxt, t), (sid, a)), ()))
                continue
            act = self.graph[nxt]
            c = float(act.duration_s)
            for s2 in act.next:
                if self.graph[s2].kind != "S":
                    continue
                kivs = ivs(geom_key(s2))
                for jlo, jhi, jown in ivs(geom_key(nxt)):
                    if jlo > iv[1] + EPS:
                        break  # the stop cannot wait that long
                    d0 = max(t, jlo)
                    if d0 + c > jhi + EPS:
                        continue
                    for klo, khi, kown in kivs:
                        d = max(d0, klo - c)
                        if d > iv[1] + EPS or d + c > jhi + EPS:
                            break
                        a = d + c
                        if a >= khi - EPS:
                            continue
                        own = tuple(sorted(set(jown) | set(kown))) if jown or kown else ()
                        out.append((k, s2, (klo, khi, kown), a, ((nxt, d), (s2, a)), own))
        return out

    def _search(self, occ: Occupancy, me: int, start: str, t0: float, legs: list[str], locked: frozenset[int],
                split: bool, goal, h, phi=None) -> Found | None:
        """goal(k, sid, iv, t) -> None (not a goal) or the vehicles to evict there. Without phi the first goal popped is
        the answer; with phi a goal goes back on the heap with phi(sid) added, and the first of those popped is."""
        cache: dict[str, list[Interval]] = {}

        def ivs(key: str) -> list[Interval]:
            if key not in cache:
                cache[key] = occ.intervals(key, me, locked, split)
            return cache[key]
        t0 = float(t0)
        iv0 = next((iv for iv in ivs(geom_key(start)) if iv[0] - EPS <= t0 < iv[1] - EPS), None)
        h0 = h(0, start)
        if iv0 is None or h0 == INF:
            return None
        node0: Node = (0, start, iv0)
        best: dict[Node, float] = {node0: t0}
        parent: dict[Node, tuple | None] = {node0: None}
        evs: dict[Node, frozenset[int]] = {node0: frozenset(iv0[2])}
        extra: dict[Node, frozenset[int]] = {}
        heap = [(t0 + h0 + self.c_evict * len(evs[node0]), t0, 0, node0)]
        n = 0
        while heap:
            _, t, term, node = heapq.heappop(heap)
            if term:
                return self._found(occ, parent, evs, node, extra[node], t0, n)
            if t > best.get(node, INF) + EPS:
                continue
            n += 1
            if n > self.budget:
                self.stats["m6_search_budget"] += 1
                return None
            k, sid, iv = node
            more = goal(k, sid, iv, t)
            if more is not None:
                if phi is None:
                    return self._found(occ, parent, evs, node, more, t0, n)
                if node not in extra:
                    extra[node] = more
                    heapq.heappush(heap, (t + self.c_evict * len(evs[node] | more) + phi(sid), t, 1, node))
            for k2, s2, iv2, a, seg, own in self._succ(k, sid, iv, t, legs, ivs):
                node2 = (k2, s2, iv2)
                if a < best.get(node2, INF) - EPS:
                    hv = h(k2, s2)
                    if hv == INF:
                        continue
                    best[node2], parent[node2], evs[node2] = a, (node, seg), evs[node] | frozenset(own)
                    heapq.heappush(heap, (a + hv + self.c_evict * len(evs[node2]), a, 0, node2))
        return None

    @staticmethod
    def _found(occ: Occupancy, parent, evs, node, more, t0: float, n: int) -> Found:
        """The path to node, and the evicted vehicles: those of the goal, then each step's new ones from the last step
        back (sorted within a step), each once."""
        segs, met, key = [], sorted(more - evs[node]), node
        while parent[key] is not None:
            prev, seg = parent[key]
            segs.append(seg)
            met += sorted(evs[key] - evs[prev])
            key = prev
        met += sorted(evs[key])
        path = [(key[1], t0)]
        for seg in reversed(segs):
            path.extend(seg)
        return Found(path, tuple((v, occ.rests[v]) for v in met), n)
