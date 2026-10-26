"""The occupancy of one plan (M6 design §5.4, spec §2.1-§2.3): every vehicle's reservations on a working copy.

A vehicle's reservations are what the simulator's agent_constraints gives, (key, lo, hi). Those ending at INF are its
soft occupancy: the last element of its path, a stop in its rest cell, and the keys that stop blocks. The rest are
hard. The fast path works on a copy: it lifts a vehicle's soft entries, protects a stop for a while, and commits new
paths; the caller keeps the copy only when every step worked (spec CE-5). A vehicle never sees its own entries.
The last free interval of a key is split at the soft entries of the other vehicles (spec §2.3 with no slack, design
§8): a free part before the first of them, then an eviction part whose owners must leave. A rest cell has one owner
(Theorem B), so a place to rest is judged by its cell; only hard entries can make it busy later.
"""
from collections.abc import Collection, Mapping, Sequence

from agvtb.planning.intervals import EPS, INF, safe_intervals
from agvtb.planning.reservations import affects, path_constraints
from agvtb.world.cells import CellMap
from agvtb.world.states import Conflicts, StateGraph, geom_key

Entry = tuple[str, float, float]  # (key, lo, hi); hi == INF: soft


class Occupancy:
    """Reservations by vehicle and by key, the rest of each vehicle, the owner of each rest cell, and the paths this
    plan added (added[v] starts with v's end element)."""

    def __init__(self, graph: StateGraph, conflicts: Conflicts, cm: CellMap, rest: Collection[str], now: float):
        self.graph, self.conflicts, self.cm = graph, conflicts, cm
        self.rest = frozenset(rest)
        self.now = float(now)
        self._entries: dict[int, list[Entry]] = {}
        self._by_key: dict[str, list[tuple[int, float, float]]] = {}
        self._owner: dict[str, int] = {}
        self.ends: dict[int, tuple[str, float]] = {}
        self.rests: dict[int, tuple[str, float] | None] = {}  # the stop and the time a vehicle holds soft from
        self.homes: dict[int, str | None] = {}
        self.added: dict[int, list[tuple[str, float]]] = {}
        self.order: list[int] = []  # vehicles by their first commit

    @classmethod
    def from_sim(cls, graph: StateGraph, conflicts: Conflicts, cm: CellMap, rest: Collection[str], now: float,
                 cons: Mapping[int, Sequence[Entry]], ends: Mapping[int, tuple[str, float]],
                 homes: Mapping[int, str | None]) -> "Occupancy":
        """cons: agent_constraints of every vehicle; ends: the last element of each path; homes: the rest ledger."""
        occ = cls(graph, conflicts, cm, rest, now)
        occ.ends = dict(ends)
        for v in sorted(cons):
            occ._add(v, cons[v])
            occ.rests[v] = ends.get(v) if any(hi == INF for _, _, hi in cons[v]) else None
            occ._set_home(v, homes.get(v))
        return occ

    def copy(self) -> "Occupancy":
        new = Occupancy(self.graph, self.conflicts, self.cm, self.rest, self.now)
        new._entries = {v: list(es) for v, es in self._entries.items()}
        new._by_key = {k: list(es) for k, es in self._by_key.items()}
        new._owner = dict(self._owner)
        new.ends, new.rests, new.homes = dict(self.ends), dict(self.rests), dict(self.homes)
        new.added = {v: list(p) for v, p in self.added.items()}
        new.order = list(self.order)
        return new

    def _add(self, v: int, entries: Sequence[Entry]) -> None:
        mine = self._entries.setdefault(v, [])
        for key, lo, hi in entries:
            mine.append((key, lo, hi))
            self._by_key.setdefault(key, []).append((v, lo, hi))

    def _set_home(self, v: int, cell: str | None) -> None:
        old = self.homes.get(v)
        if old is not None and self._owner.get(old) == v:
            del self._owner[old]
        self.homes[v] = cell
        if cell is not None:
            self._owner[cell] = v

    def lift(self, v: int) -> None:
        """v's soft entries go (spec lines 98, 168): it is about to leave its rest, and gives up its cell."""
        keys = sorted({key for key, _, hi in self._entries.get(v, ()) if hi == INF})
        if keys:
            self._entries[v] = [e for e in self._entries[v] if e[2] != INF]
            for key in keys:
                self._by_key[key] = [e for e in self._by_key[key] if e[0] != v or e[2] != INF]
        self.rests[v] = None
        self._set_home(v, None)

    def protect(self, v: int, sid: str, lo: float, hi: float) -> None:
        """v holds the stop sid over [lo, hi) (spec lines 100, 169), reserved as path_constraints reserves one
        element: its key, and each state it blocks from lo less that state's duration."""
        if hi <= lo + EPS:
            return
        out = [(geom_key(sid), lo, hi)]
        out += [(x, max(0.0, lo - (self.graph[x].duration_s or 0.0)), hi) for x in affects(self.conflicts, sid)]
        self._add(v, out)

    def commit(self, v: int, path: Sequence[tuple[str, float]], rest: bool = True) -> None:
        """v goes on along path from where it stands: its end element, or the end of what this plan committed for it.
        path[0] must be that stop; its time is kept. rest: v rests at the path's end (soft entries, its cell);
        otherwise the last element is not held and the next commit goes on from it (spec line 212, decision 1)."""
        cur = self.added[v][-1] if v in self.added else self.ends[v]
        if path[0][0] != cur[0]:
            raise ValueError(f"vehicle {v}: the path starts at {path[0][0]}, the vehicle stands at {cur[0]}")
        path = [cur] + [(sid, float(t)) for sid, t in path[1:]]
        entries = path_constraints(path, self.graph, self.conflicts)
        if not rest:  # the last element's entries come last: its key, then its affects
            entries = entries[:len(entries) - 1 - len(affects(self.conflicts, path[-1][0]))]
        self._add(v, entries)
        if v in self.added:
            self.added[v] += path[1:]
        else:
            self.added[v] = path
            self.order.append(v)
        self.rests[v] = path[-1] if rest else None
        self._set_home(v, self.cm.cell_of[self.graph[path[-1][0]].nodes[0]] if rest else None)

    def intervals(self, key: str, me: int, locked: frozenset[int],
                  split: bool) -> list[tuple[float, float, tuple[int, ...]]]:
        """The safe intervals (lo, hi, owners to evict) of key for vehicle me. Not split, or for a locked owner, a soft
        entry is hard. Split (design decision 4): the last free interval [t1, INF) becomes [t1, first soft start), if
        not empty, and [max(t1, now, last soft start), INF) owned by the soft entries' vehicles."""
        hard, soft = [], []
        for v, lo, hi in self._by_key.get(key, ()):
            if v == me:
                continue
            if hi != INF or not split or v in locked:
                hard.append((lo, hi))
            else:
                soft.append((v, lo))
        ivs = safe_intervals(hard)
        if not soft:
            return [(lo, hi, ()) for lo, hi in ivs]
        if not ivs or ivs[-1][1] != INF:  # no last free interval: the soft entries are as good as hard
            return [(lo, hi, ()) for lo, hi in safe_intervals(hard + [(lo, INF) for _, lo in soft])]
        t1 = ivs[-1][0]
        first, last = min(lo for _, lo in soft), max(lo for _, lo in soft)
        out = [(lo, hi, ()) for lo, hi in ivs[:-1]]
        if t1 < first - EPS:
            out.append((t1, first, ()))
        out.append((max(t1, self.now, last), INF, tuple(sorted({v for v, _ in soft}))))
        return out

    def owner(self, cell: str) -> int | None:
        return self._owner.get(cell)

    def parking_ok(self, sid: str, a: float, me: int) -> bool:
        """No hard entry of another vehicle on the stop's key or what it blocks ends after a (safe_tail.py
        _parking_ok, on hard entries only: design decision 5)."""
        keys = [geom_key(sid)] + [geom_key(x) for x in affects(self.conflicts, sid)]
        return not any(v != me and hi != INF and hi > a + EPS for k in keys for v, _, hi in self._by_key.get(k, ()))
