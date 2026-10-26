"""The M6 fallback bundle (design §5.6 steps 4-7): Theorem C's cell moves as state paths, one vehicle at a time.

t0 = max(r's start time, T*), T* the latest finite reservation end of the other vehicles (the same T* as in
safe_tail.py's fallback). From t0 on every vehicle stands on the last stop of its path, each in its own rest cell.
A cell move C -> D becomes the quickest way from the vehicle's stop to a stop in D through the M and R states whose
nodes all lie in C and D. By the cell rule (Theorem B) those states collide with no vehicle standing in another cell.
A load at port p goes to a stop at p inside p's cell, then to L and back to that stop. Move j starts when move j - 1
ends. Before it is returned the bundle is checked for overlaps: no two vehicles may hold conflicting states (one
place, a conflict either way, or two stops at one node) at overlapping times. Each member's path is checked against
the new paths of the other members and the last stops of the other vehicles. Their earlier states end by T*, before
any member moves. SafeTailPlanner.path_conflicts is not used: its table blocks a state's affects from their start
minus their duration and looks up a final stop's affects, so it refuses one vehicle passing a cell right after
another and two vehicles resting in neighbour cells.
"""
import collections
import heapq
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from agvtb.planning.cell_procedure import Stuck, solve
from agvtb.planning.intervals import EPS, INF
from agvtb.planning.m6_gate import station_hops
from agvtb.world.cells import CellMap
from agvtb.world.states import Conflicts, StateGraph, geom_key

Segment = tuple[list[tuple[str, float]], str]  # ([(state, time from the segment's start), ...], the stop it ends on)


def _held(path: Sequence[tuple[str, float]]) -> list[tuple[str, float, float]]:
    """[(geometric key, start, end)] of a path; the last element is held forever."""
    return [(geom_key(sid), t, path[i + 1][1] if i + 1 < len(path) else INF) for i, (sid, t) in enumerate(path)]


@dataclass(frozen=True)
class Bundle:
    """One fallback plan: the paths of r and of the vehicles it moves, each ending on a stop of its rest cell."""

    r: int
    paths: dict[int, list[tuple[str, float]]]  # member -> path; path[0] is the last element of its committed path
    homes: dict[int, str]  # member -> the rest cell its path ends in
    order: tuple[int, ...]  # the other members by their first step, then r
    t0: float  # max(r's start time, t_star): no member moves before it
    t_star: float  # the latest finite reservation end of the other vehicles
    moves: int  # cell moves


class M6Fallback:
    """Bundles of Theorem C's procedure for one map, rest rule and vehicle count."""

    def __init__(self, graph: StateGraph, conflicts: Conflicts, cm: CellMap, rest: Collection[str],
                 n_vehicles: int) -> None:
        self.graph, self.conflicts, self.cm = graph, conflicts, cm
        self.rest = tuple(sorted(rest))
        self.hops = station_hops(cm, len(cm.cells) - n_vehicles)
        self.stats: collections.Counter[str] = collections.Counter()
        self._segments: dict[tuple, Segment | None] = {}
        self._near: dict[str, set[str]] = collections.defaultdict(set)  # key -> the keys it cannot be held with
        stops_at: dict[str, set[str]] = collections.defaultdict(set)
        for sid in sorted(conflicts):
            for x in conflicts[sid]:
                self._near[geom_key(sid)].add(geom_key(x))
                self._near[geom_key(x)].add(geom_key(sid))
        for sid, state in graph.items():
            self._near[sid].add(geom_key(sid))
            if state.kind == "S":
                stops_at[state.nodes[0]].add(sid)
        for stops in stops_at.values():
            for sid in stops:
                self._near[sid] |= stops

    def _quickest(self, start: str, nodes: Collection[str], goal: Collection[str]) -> Segment | None:
        """Dijkstra from the stop start to the first stop on a goal node through the M and R states whose nodes all
        lie in nodes; ties by (time, state id). The segment alternates (action, start) and (stop, arrival)."""
        dist, parent, heap = {start: 0.0}, {start: None}, [(0.0, start)]
        while heap:
            d, sid = heapq.heappop(heap)
            if d > dist[sid]:
                continue
            if self.graph[sid].nodes[0] in goal:
                end, out = sid, []
                while parent[sid] is not None:
                    prev, act, t_act = parent[sid]
                    out += [(sid, dist[sid]), (act, t_act)]
                    sid = prev
                return out[::-1], end
            for act in self.graph[sid].next:
                state = self.graph[act]
                if state.kind not in ("M", "R") or not set(state.nodes) <= nodes:
                    continue
                for s2 in state.next:
                    nd = d + state.duration_s
                    if nd < dist.get(s2, INF):
                        dist[s2], parent[s2] = nd, (sid, act, d)
                        heapq.heappush(heap, (nd, s2))
        return None

    def _move(self, start: str, src: str, dst: str) -> Segment | None:
        key = ("move", start, src, dst)
        if key not in self._segments:
            members = self.cm.members
            self._segments[key] = self._quickest(start, set(members[src]) | set(members[dst]), set(members[dst]))
        return self._segments[key]

    def _load(self, start: str, port: str) -> Segment | None:
        key = ("load", start, port)
        if key not in self._segments:
            seg = self._quickest(start, set(self.cm.members[self.cm.cell_of[port]]), {port})
            if seg is not None:
                elements, end = seg
                lsid, t = f"L,{port},{self.graph[end].heading}", elements[-1][1] if elements else 0.0
                seg = elements + [(lsid, t), (end, t + self.graph[lsid].duration_s)], end
            self._segments[key] = seg
        return self._segments[key]

    def _fail(self, reason: str) -> None:
        self.stats[reason] += 1
        self.stats["m6_fallback_fail"] += 1
        return None

    def plan(self, r: int, legs: Sequence[str], ends: Mapping[int, tuple[str, float]],
             cons: Mapping[int, Sequence[tuple[str, float, float]]]) -> Bundle | None:
        """The bundle of r's job, or None (counted in stats) when the procedure is stuck, a cell move has no state
        path, or the bundle overlaps the reservations.

        ends: every vehicle's last path element. cons: every vehicle's reservations (agent_constraints).
        """
        t_star = max((hi for u, items in cons.items() if u != r for _, _, hi in items if hi != INF), default=0.0)
        t0 = max(ends[r][1], t_star)
        at = {u: self.cm.cell_of[self.graph[ends[u][0]].nodes[0]] for u in sorted(ends)}
        try:
            steps, final = solve(self.cm, self.rest, self.hops, at, r, legs)
        except Stuck:
            return self._fail("m6_procedure_stuck")
        clock, cur, moves = t0, {u: ends[u][0] for u in ends}, 0
        added: dict[int, list[tuple[str, float]]] = {}
        for step in steps:
            u = step[1]
            if step[0] == "move":
                seg = self._move(cur[u], step[2], step[3])
                moves += 1
            else:
                seg = self._load(cur[u], step[2])
            if seg is None:
                return self._fail("m6_cell_path_missing")
            elements, cur[u] = seg
            added.setdefault(u, []).extend((sid, clock + rel) for sid, rel in elements)
            clock += elements[-1][1] if elements else 0.0
        order = tuple(u for u in added if u != r) + (r,)
        paths = {u: [ends[u]] + added.get(u, []) for u in sorted(order)}
        homes = {u: final[u] for u in sorted(order)}
        held = {w: _held(paths[w] if w in paths else [ends[w]]) for w in sorted(ends)}
        by_key: dict[str, list[tuple[int, float, float]]] = collections.defaultdict(list)
        for w, entries in held.items():
            for key, lo, hi in entries:
                by_key[key].append((w, lo, hi))
        for u in order:
            for key, t, end in held[u]:
                if any(w != u and lo < end - EPS and t < hi - EPS
                       for near in sorted(self._near[key]) for w, lo, hi in by_key.get(near, ())):
                    return self._fail("m6_bundle_overlap")
        self.stats["m6_fallback_used"] += 1
        self.stats["m6_cell_moves"] += moves
        self.stats["m6_evicted"] += len(order) - 1
        self.stats["m6_bundle_max"] = max(self.stats["m6_bundle_max"], len(order))
        self.stats["m6_fallback_wait_s"] += t0 - ends[r][1]
        return Bundle(r, paths, homes, order, t0, t_star, moves)
