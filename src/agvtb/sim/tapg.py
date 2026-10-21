"""The TAPG: the event graph that turns committed paths into an order of execution (DAIM agv/agv_des_env.py).

A node is (state id, vehicle, plan time rounded to 6 decimals); a vehicle's nodes are chained in path order.
Cross edges between vehicles say "this state must be over before that one starts" for states that collide
(env:1676-1803, the rules in research §B2). A vehicle may start a state only when no node of another vehicle
precedes it (env:851-858); nodes leave the graph as vehicles get past them. recompute re-times every path to
the earliest times the graph allows (env:1805-1948). Plain dicts, no networkx.
"""
import heapq
from collections.abc import Mapping

from agvtb.planning.intervals import EPS, INF
from agvtb.planning.reservations import affects
from agvtb.world.states import Conflicts, StateGraph, geom_key

Key = tuple[str, int, float]
MIN_REMAIN = 1e-3  # least remaining time of a state in progress (env:83)


def _key(sid: str, vid: int, t: float) -> Key:
    return (sid, vid, round(t, 6))


class Tapg:
    def __init__(self, graph: StateGraph, conflicts: Conflicts):
        self.graph = graph
        self.conflicts = conflicts
        self.paths: dict[int, list[tuple[str, float]]] = {}
        self.duration: dict[Key, float] = {}
        self.is_dwell: set[Key] = set()
        self._pred: dict[Key, set[Key]] = {}
        self._succ: dict[Key, set[Key]] = {}
        self._affects: dict[str, frozenset[str]] = {}

    # --- structure ---------------------------------------------------------------------------------------------
    def key(self, vid: int, i: int) -> Key:
        sid, t = self.paths[vid][i]
        return _key(sid, vid, t)

    def __contains__(self, key: Key) -> bool:
        return key in self.duration

    def preds(self, key: Key) -> set[Key]:
        return set(self._pred.get(key, ()))

    def succs(self, key: Key) -> set[Key]:
        return set(self._succ.get(key, ()))

    def cross_edges(self) -> list[tuple[Key, Key]]:
        return sorted((a, b) for a, bs in self._succ.items() for b in bs if a[1] != b[1])

    def _add_node(self, key: Key, duration: float) -> None:
        self.duration[key] = duration
        self._pred.setdefault(key, set())
        self._succ.setdefault(key, set())

    def _add_edge(self, a: Key, b: Key) -> None:
        if a != b:
            self._succ[a].add(b)
            self._pred[b].add(a)

    def _affect(self, sid: str) -> frozenset[str]:
        found = self._affects.get(sid)
        if found is None:
            found = self._affects[sid] = frozenset(affects(self.conflicts, sid))
        return found

    def add_vehicle(self, vid: int, sid: str, t: float) -> None:
        self.paths[vid] = [(sid, t)]
        self._add_node(_key(sid, vid, t), INF)

    def append(self, vid: int, new_path) -> None:
        """Extend vehicle vid's path by new_path[1:] (new_path[0] is its last state), then add cross edges."""
        path = self.paths[vid]
        if new_path and new_path[0][0] != path[-1][0]:
            raise ValueError(f"vehicle {vid}: the new path starts at {new_path[0][0]}, its path ends at {path[-1][0]}")
        if len(new_path) < 2:
            return
        ext = list(new_path[1:])
        lo = len(path)
        last = self.key(vid, lo - 1)
        if last in self.duration:
            self.duration[last] = ext[0][1] - path[-1][1]
        path.extend(ext)
        hi = len(path)
        for k in range(lo, hi):
            nk = self.key(vid, k)
            self._add_node(nk, INF if k == hi - 1 else path[k + 1][1] - path[k][1])
            prev = self.key(vid, k - 1)
            if prev in self.duration:
                self._add_edge(prev, nk)
        self._cross_edges(vid, lo, hi)

    def _cross_edges(self, ai: int, lo: int, hi: int) -> None:
        """research §B2 (env:1676-1803): both directions between the new states of ai and every other path."""
        pi = self.paths[ai]
        for aj in sorted(self.paths):
            if aj == ai or not self.paths[aj]:
                continue
            pj = self.paths[aj]
            self._one_direction(ai, pi, range(hi - 1, lo - 1, -1), aj, pj, 0, len(pj), 0)
            self._one_direction(aj, pj, range(len(pj) - 1, -1, -1), ai, pi, lo, hi, lo)

    def _one_direction(self, src_v, src_p, src_range, tgt_v, tgt_p, tgt_lo, tgt_hi, prev_floor) -> None:
        transidx = tgt_hi
        for k1 in src_range:
            s1, t1 = src_p[k1]
            affect1 = self._affect(s1)
            if not affect1:
                continue
            s1_finish = _finish(src_p, k1)
            g1 = geom_key(s1)
            for k2 in range(tgt_lo, tgt_hi):
                if k2 >= transidx:
                    break
                s2, t2 = tgt_p[k2]
                if t2 < s1_finish - EPS:
                    continue
                if _tie_skip(t1, s1_finish, t2, _finish(tgt_p, k2), src_v, tgt_v):
                    continue
                g2 = geom_key(s2)
                if g2 in affect1 or g1 in self._affect(s2) or g1 == g2:
                    nk1, nk2 = _key(s1, src_v, t1), _key(s2, tgt_v, t2)
                    if nk1 in self.duration and nk2 in self.duration:
                        self._add_edge(nk1, nk2)
                    transidx = k2
                    if s2.startswith("S,") and k2 > prev_floor:
                        prev_sid, prev_t = tgt_p[k2 - 1]
                        if prev_sid.startswith(("M,", "R,")):
                            prev_nk = _key(prev_sid, tgt_v, prev_t)
                            if (prev_nk in self.duration and nk1 in self.duration
                                    and (s1_finish - t1 > EPS or not self._reaches(prev_nk, nk1))):
                                self._add_edge(nk1, prev_nk)
                    break

    def _reaches(self, a: Key, b: Key) -> bool:
        seen, stack = {a}, [a]
        while stack:
            n = stack.pop()
            if n == b:
                return True
            for m in self._succ.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        return False

    # --- claims and removal ------------------------------------------------------------------------------------
    def claimable(self, key: Key, vid: int) -> bool:
        """No node of another vehicle precedes key (a node no longer in the graph is claimable)."""
        return all(p[1] == vid for p in self._pred.get(key, ()))

    def remove(self, key: Key) -> list[int]:
        """Take key out; the other vehicles that had it as a predecessor (to wake), sorted."""
        if key not in self.duration:
            return []
        waiters = sorted({s[1] for s in self._succ[key] if s[1] != key[1]})
        for s in self._succ.pop(key):
            self._pred[s].discard(key)
        for p in self._pred.pop(key):
            self._succ[p].discard(key)
        del self.duration[key]
        self.is_dwell.discard(key)
        return waiters

    def prune(self, vid: int, upto: int) -> list[int]:
        """Drop the first upto states of vid's path and their nodes; the vehicles to wake, sorted."""
        waiters: set[int] = set()
        for i in range(upto):
            waiters.update(self.remove(self.key(vid, i)))
        self.paths[vid] = self.paths[vid][upto:]
        return sorted(waiters)

    def mark_dwell(self, vid: int, i: int) -> None:
        """The stop before a load keeps its length through recompute (research §A6-3)."""
        key = self.key(vid, i)
        if key in self.duration:
            self.is_dwell.add(key)

    # --- order -------------------------------------------------------------------------------------------------
    def _topo(self) -> list[Key] | None:
        indeg = {n: len(self._pred[n]) for n in self.duration}
        heap = [n for n, d in indeg.items() if d == 0]
        heapq.heapify(heap)
        order = []
        while heap:
            n = heapq.heappop(heap)
            order.append(n)
            for m in self._succ[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    heapq.heappush(heap, m)
        return order if len(order) == len(indeg) else None

    def has_cycle(self) -> bool:
        return self._topo() is None

    def related(self, key: Key) -> set[Key]:
        """The ancestors and descendants of key."""
        out: set[Key] = set()
        for edges in (self._pred, self._succ):
            stack = [key]
            while stack:
                for m in edges.get(stack.pop(), ()):
                    if m not in out:
                        out.add(m)
                        stack.append(m)
        out.discard(key)
        return out

    def recompute(self, now: float, inprog_end: Mapping[Key, float]) -> None:
        """Re-time every path to the earliest start the graph allows from now (research §B5, env:1805-1948).

        A state in progress ends at its real end (inprog_end, keyed by the keys before this call). A dwell stop
        keeps its length, other stops last 0, M/R/L last duration_s. Does nothing if the graph has a cycle.
        """
        order = self._topo()
        if order is None:
            return
        start: dict[Key, float] = {}

        def length(u: Key) -> float:
            end = inprog_end.get(u)
            if end is not None:
                return max(MIN_REMAIN, end - start[u])
            if u[0].startswith("S,"):
                if u in self.is_dwell:
                    d = self.duration[u]
                    return 0.0 if d == INF else d
                return 0.0
            return self.graph[u[0]].duration_s or 0.0

        for v in order:
            cross = [start[u] + length(u) for u in self._pred[v] if u[1] != v[1]]
            same = [start[u] + length(u) for u in self._pred[v] if u[1] == v[1]]
            if not same:
                start[v] = max([now] + cross)
            else:
                start[v] = max(same + cross)
        new_key = {old: _key(old[0], old[1], start[old]) for old in order}
        duration, dwell = {}, set()
        for old, new in new_key.items():
            if old[0].startswith("S,"):
                if old in self.is_dwell:
                    d = self.duration[old]
                    duration[new] = 0.0 if d == INF else d
                    dwell.add(new)
                else:
                    duration[new] = 0.0
            else:
                duration[new] = self.graph[old[0]].duration_s
        pred = {new_key[n]: {new_key[p] for p in ps if new_key[p] != new_key[n]} for n, ps in self._pred.items()}
        succ = {new_key[n]: {new_key[s] for s in ss if new_key[s] != new_key[n]} for n, ss in self._succ.items()}
        self.duration, self.is_dwell, self._pred, self._succ = duration, dwell, pred, succ
        for vid, path in self.paths.items():
            out = []
            for sid, t in path:
                new = new_key.get(_key(sid, vid, t))
                out.append((sid, new[2] if new is not None else round(t, 6)))
            self.paths[vid] = out


def _finish(path, k: int) -> float:
    return path[k + 1][1] if k + 1 < len(path) else INF


def _tie_skip(src_t, src_finish, tgt_t, tgt_finish, src_id, tgt_id) -> bool:
    """Two zero-length nodes touching at one time: only the lower vehicle id gets the edge (env:1697-1721)."""
    return (abs(tgt_t - src_finish) <= EPS and src_finish - src_t <= EPS and tgt_finish - tgt_t <= EPS
            and src_id >= tgt_id)
