"""The fast path (M6 design §5.5): the user's spec, algorithms 1-3, with U2 and the M6b plan's decisions.

PlanWithEviction_Verified (algorithm 1, _verified), ExecuteRecursiveEvict (algorithm 2, _evict) and
TrySelfClearanceAndReplan (algorithm 3, _self_clear) work on Occupancy copies. plan keeps a copy only when every step
worked (spec CE-5) and returns it as a Bundle: the evicted vehicles by their first commit, then r. U2: rests only in
rest cells (the searches' goal), any FAIL makes plan return None so that the caller runs the fallback, and the recall
of spec line 130 at most once. The plan's decisions: r's path is in the copy before Phase 4 (1); a stop that is about
to be left is protected for the time it takes to leave its cell any way (2); a vehicle is evicted only while it still
holds the rest a search met (3). The comments give the spec's line numbers.
"""
import collections
from collections.abc import Collection, Mapping, Sequence

from agvtb.planning.m6_fallback import Bundle
from agvtb.planning.occupancy import Occupancy
from agvtb.planning.split_sipp import SplitSipp, merge_locks, target_locks
from agvtb.world.cells import CellMap
from agvtb.world.states import Conflicts, StateGraph

BRANCHES = ("probe", "evict", "hard", "tail", "tail_evict", "tail_hard", "self_clear")  # design decision 8
COUNTS = ("m6_fast_ok", "m6_fast_fail", *(f"m6_fail_{b}" for b in BRANCHES), "m6_fast_evicted", "m6_fast_bundle_max",
          "m6_chain_max", "m6_sidestep_try", "m6_sidestep_ok", "m6_recall", "m6_search_budget")


class Fail(Exception):
    """A FAIL of the spec's algorithms; branch says which one (design decision 8)."""

    def __init__(self, branch: str):
        super().__init__(branch)
        self.branch = branch


class EvictionPlanner:
    """The fast path on one map and rest set. stats gets the COUNTS, each a plan's outcome or a running maximum."""

    def __init__(self, graph: StateGraph, conflicts: Conflicts, cm: CellMap, rest: Collection[str],
                 stats: collections.Counter, *, budget: int | None = None):
        self.graph, self.conflicts, self.cm = graph, conflicts, cm
        self.rest = tuple(sorted(rest))
        self.stats = stats
        self.sipp = SplitSipp(graph, conflicts, cm, stats, budget=budget)
        adj: dict[str, set[str]] = collections.defaultdict(set)
        for state in graph.values():
            if state.kind == "M":
                adj[state.nodes[0]].add(state.nodes[1])
        self._adj = {node: tuple(sorted(nbrs)) for node, nbrs in adj.items()}
        for key in COUNTS:
            stats[key] += 0

    def plan(self, r: int, legs: Sequence[str], ends: Mapping[int, tuple[str, float]],
             cons: Mapping[int, Sequence[tuple[str, float, float]]], homes: Mapping[int, str | None],
             now: float) -> Bundle | None:
        """r's job as a fast-path bundle, or None after a FAIL (counted once, under the branch that ended it).

        ends: every vehicle's last path element; cons: every vehicle's agent_constraints; homes: the rest ledger.
        """
        db = Occupancy.from_sim(self.graph, self.conflicts, self.cm, self.rest, now, cons, ends, homes)
        try:
            work = self._verified(db, r, ends[r][0], ends[r][1], list(legs), True, True)
        except Fail as f:
            self.stats[f"m6_fail_{f.branch}"] += 1
            self.stats["m6_fast_fail"] += 1
            return None
        order = tuple(u for u in work.order if u != r) + (r,)
        self.stats["m6_fast_ok"] += 1
        self.stats["m6_fast_evicted"] += len(order) - 1
        self.stats["m6_fast_bundle_max"] = max(self.stats["m6_fast_bundle_max"], len(order))
        return Bundle(r, {u: work.added[u] for u in order}, {u: work.homes[u] for u in order}, order, ends[r][1], 0.0,
                      0, True)

    def _verified(self, db: Occupancy, r: int, start: str, t0: float, legs: list[str], allow_self_clear: bool,
                  may_recall: bool) -> Occupancy:
        """Algorithm 1 on a copy of db; raises Fail."""
        work = db.copy()                                                                      # spec line 95
        work.lift(r)                                                                          # 98
        work.protect(r, start, t0, t0 + self.sipp.exit_s(start))                              # 100 (decision 2)
        cand = self.sipp.staged(work, r, start, t0, legs, frozenset({r}), True)               # 103
        if cand is None:
            raise Fail("probe")                                                               # 104-105
        path = cand.path
        if cand.blocking:                                                                     # 108-111
            locks = target_locks(cand.path, self.cm, self.graph)                              # 112
            for m, seen in cand.blocking:                                                     # 114
                if work.rests.get(m) != seen:
                    continue                                                                  # 115-116 (decision 3)
                if not self._evict(work, m, locks, frozenset({r}), 1):                        # 118
                    if allow_self_clear:
                        return self._self_clear(db, r, start, t0, legs)                       # 120-121
                    raise Fail("evict")                                                       # 122-123
            final = self.sipp.staged(work, r, start, t0, legs, frozenset({r}), False)         # 127
            if final is None:
                if not may_recall:
                    raise Fail("hard")                                                        # U2 ③
                self.stats["m6_recall"] += 1
                return self._verified(work, r, start, t0, legs, False, False)                 # 130
            path = final.path
        end, t_end = path[-1]                                                                 # 135-136
        work.commit(r, path, rest=False)                                                      # decision 1
        work.protect(r, end, t_end, t_end + self.sipp.exit_s(end))
        tail = self.sipp.nearest_rest(work, r, end, t_end, {}, frozenset({r}), True)          # 139
        if tail is None:
            raise Fail("tail")                                                                # 140
        if tail.blocking:                                                                     # 142-143
            locks_t = target_locks(tail.path, self.cm, self.graph)                            # 144
            for p, seen in tail.blocking:                                                     # 145-146
                if work.rests.get(p) == seen and not self._evict(work, p, locks_t, frozenset({r}), 1):
                    raise Fail("tail_evict")                                                  # 147
            tail = self.sipp.nearest_rest(work, r, end, t_end, {}, frozenset({r}), False)     # 148
            if tail is None:
                raise Fail("tail_hard")                                                       # 149
        work.commit(r, tail.path, rest=True)                                                  # 150, 153
        return work

    def _evict(self, work: Occupancy, m: int, locks: Mapping[str, float], locked: frozenset[int], depth: int) -> bool:
        """Algorithm 2 on work (changed even when it fails; the caller drops it then)."""
        self.stats["m6_chain_max"] = max(self.stats["m6_chain_max"], depth)
        v_m, t_idle = work.rests[m]                                                           # 163-164
        t_dep = max(work.now, t_idle)                                                         # 165
        work.lift(m)                                                                          # 168
        work.protect(m, v_m, t_idle, t_dep + self.sipp.exit_s(v_m))                           # 169 (decision 2)
        lock = locked | {m}
        found = self.sipp.nearest_rest(work, m, v_m, t_dep, locks, lock, True)                # 176
        if found is None:
            return False                                                                      # 177-178
        if found.blocking:                                                                    # 181-182
            merged = merge_locks(locks, target_locks(found.path, self.cm, self.graph))        # 183
            for p, seen in found.blocking:                                                    # 184-185
                if work.rests.get(p) == seen and not self._evict(work, p, merged, lock, depth + 1):
                    return False                                                              # 186-187
            found = self.sipp.nearest_rest(work, m, v_m, t_dep, locks, lock, False)           # 190
            if found is None:
                return False                                                                  # 191
        work.commit(m, found.path, rest=True)                                                 # 193
        return True

    def _self_clear(self, db: Occupancy, r: int, start: str, t0: float, legs: list[str]) -> Occupancy:
        """Algorithm 3 from db; raises Fail("self_clear") when no neighbour works."""
        self.stats["m6_sidestep_try"] += 1
        node = self.graph[start].nodes[0]
        nbrs = self._adj.get(node, ())
        for adj in sorted(nbrs, key=lambda n: (-len(self._adj.get(n, ())), n)):              # 203 (decision 12)
            temp = db.copy()                                                                  # 204
            step = self.sipp.step(temp, r, start, t0, adj)                                    # 206
            if step is None:
                continue                                                                      # 207
            end, t_arr = step.path[-1]                                                        # 209
            temp.lift(r)                                                                      # 211
            temp.commit(r, step.path, rest=False)                                             # 212
            try:
                work = self._verified(temp, r, end, t_arr, legs, False, True)                 # 214 (decision 7)
            except Fail:
                continue
            self.stats["m6_sidestep_ok"] += 1
            return work                                                                       # 215-217
        raise Fail("self_clear")                                                              # 218
