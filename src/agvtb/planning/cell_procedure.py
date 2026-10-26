"""Theorem C's procedure on the cell graph (M6 design §5.6 step 3, R26 §3): one job of the planned vehicle r.

r visits the cells of its legs' ports in order and then rests; the idle vehicles are pushed out of its way one move at
a time. Every move goes to a free neighbour cell. In the end every vehicle is on its own rest cell. Ported from R27
spike_rest_rules.Run, with these changes:
- between stations r takes only hops of at most room = V - N edges (the hops of C2);
- a port on a cut cell is entered from its nearest station and left back to it (C4);
- a start on a cut cell picks its way out as C3 does;
- consecutive legs in one cell are one visit;
- r rests on its station when that is a rest cell;
- ties go by cell id.
Pushing one idle vehicle along a shortest path frees its cell and fills the free end, and changes no other cell's
occupancy. So a clear needs one free cell outside the cells it clears for each vehicle on them; room and C3 give that.
"""
import heapq
from collections import deque
from collections.abc import Collection, Iterable, Mapping, Sequence

from agvtb.planning.m6_gate import components
from agvtb.world.cells import CellMap


class Stuck(RuntimeError):
    """The procedure found no way: a gate condition is broken, or a bug."""


def _bfs(cm: CellMap, src: str, allowed: Collection[str]) -> tuple[dict[str, str | None], dict[str, int]]:
    """Shortest-path parents and distances from src through the cells in allowed, neighbours in cm.adj order."""
    prev: dict[str, str | None] = {src: None}
    dist, queue = {src: 0}, deque([src])
    while queue:
        x = queue.popleft()
        for u in cm.adj[x]:
            if u in allowed and u not in prev:
                prev[u], dist[u] = x, dist[x] + 1
                queue.append(u)
    return prev, dist


def _path(prev: Mapping[str, str | None], dst: str) -> list[str]:
    """src ... dst along the parents of _bfs."""
    path = [dst]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


class _Run:
    """The cells of all vehicles during one job, and the steps taken so far."""

    def __init__(self, cm: CellMap, rest: frozenset[str], hops: Mapping[str, Mapping[str, tuple[str, ...]]],
                 at: Mapping[int, str], r: int):
        self.cm, self.rest, self.hops, self.r = cm, rest, hops, r
        self.all = frozenset(cm.members)
        self.stations = cm.stations
        self.pos = dict(at)
        self.occ: dict[str, int] = {}
        for vid in sorted(at):
            if at[vid] in self.occ or at[vid] not in self.all:
                raise Stuck(f"vehicle {vid} cannot start on {at[vid]}")
            self.occ[at[vid]] = vid
        self.room = len(self.all) - len(at)
        self.steps: list[tuple] = []

    def step(self, vid: int, dst: str) -> None:
        src = self.pos[vid]
        if dst not in self.cm.adj[src] or dst in self.occ:
            raise Stuck(f"vehicle {vid} cannot move from {src} to {dst}")
        del self.occ[src]
        self.occ[dst], self.pos[vid] = vid, dst
        self.steps.append(("move", vid, src, dst))

    def shift(self, path: Sequence[str]) -> None:
        """path[0] holds an idle vehicle and path[-1] is free: the vehicles on the path move forward, the last one
        first, until path[0] is free (R27 Run.shift)."""
        if path[0] not in self.occ or path[-1] in self.occ or any(self.occ.get(c) == self.r for c in path):
            raise Stuck(f"cannot shift along {list(path)}")
        j = len(path) - 1
        while True:
            i = max(k for k in range(j) if path[k] in self.occ)
            vid = self.occ[path[i]]
            for c in path[i + 1:j + 1]:
                self.step(vid, c)
            if i == 0:
                return
            j = i

    def clear(self, segment: Iterable[str], region: Collection[str], avoid: Collection[str]) -> None:
        """Push the idle vehicles off the cells of segment, through region; free cells outside avoid first
        (R27 Run.clear)."""
        inside = set(segment)
        for a in sorted(inside):
            if a not in self.occ:
                continue
            prev, dist = _bfs(self.cm, a, region)
            free = [c for c in prev if c not in self.occ and c not in inside]
            better = [c for c in free if c not in avoid] or free
            if not better:
                raise Stuck(f"no room to clear {sorted(inside)}")
            self.shift(_path(prev, min(better, key=lambda c: (dist[c], c not in self.rest, c))))

    def walk(self, cells: Iterable[str]) -> None:
        for c in cells:
            self.step(self.r, c)

    def route(self, target: str) -> list[tuple[str, ...]]:
        """The hops from r's station to the station target with the fewest edges (Dijkstra over the hops)."""
        start = self.pos[self.r]
        best, parent, heap = {start: 0}, {}, [(0, start)]
        while heap:
            d, w = heapq.heappop(heap)
            if w == target:
                break
            if d > best[w]:
                continue
            for z, hop in self.hops[w].items():
                if d + len(hop) < best.get(z, d + len(hop) + 1):
                    best[z], parent[z] = d + len(hop), w
                    heapq.heappush(heap, (d + len(hop), z))
        if target not in best:
            raise Stuck(f"no hops of at most {self.room} edges from {start} to {target}")
        out, z = [], target
        while z != start:
            out.append(self.hops[parent[z]][z])
            z = parent[z]
        return out[::-1]

    def goto(self, target: str, extra: Collection[str] = ()) -> None:
        """r from its station to the station target, hop by hop, each hop cleared first (design §5.6 step 3)."""
        if self.pos[self.r] not in self.stations:
            raise Stuck(f"vehicle {self.r} is on {self.pos[self.r]}, which is no station")
        hops = self.route(target)
        for k, hop in enumerate(hops):
            w = self.pos[self.r]
            self.clear(hop, self.all - {w}, {c for h in hops[k:] for c in h} | set(extra))
            self.walk(hop)

    def approach(self, cell: str) -> tuple[str, list[str]]:
        """The station nearest to a cell that is no station, and the way from it: the cells after it up to the cell.
        C4 keeps the way within room."""
        prev, dist = _bfs(self.cm, cell, self.all)
        z = min((c for c in prev if c in self.stations), key=lambda c: (dist[c], c))
        way = _path(prev, z)[::-1][1:]
        if len(way) > self.room:
            raise Stuck(f"port cell {cell} is {len(way)} edges from the nearest station, room is {self.room}")
        return z, way

    def visit(self, cell: str, ports: Sequence[str]) -> None:
        """r goes to cell and loads at each of its ports; afterwards r is on a station."""
        if cell in self.stations:
            self.goto(cell)
        else:
            z, way = self.approach(cell)
            self.goto(z, extra=way)
            self.clear(way, self.all - {z}, way)
            self.walk(way)
        for p in ports:
            self.steps.append(("load", self.r, p))
        if cell not in self.stations:
            self.walk(list(reversed(way[:-1])) + [z])

    def leave_cut(self, target: str | None) -> None:
        """r from the cut cell it starts on to a station, into a component of G - s whose idle vehicles leave it a way
        (C3); the component of target first, then the shortest way."""
        s = self.pos[self.r]
        ways = []
        for comp in components(self.cm, s):
            prev, dist = _bfs(self.cm, s, comp | {s})
            z = min((c for c in comp if c in self.stations), key=lambda c: (dist[c], c))
            way = _path(prev, z)[1:]
            if len(comp) - len(way) >= sum(1 for c in comp if c in self.occ):
                ways.append((target not in comp, len(way), z, way, comp))
        if not ways:
            raise Stuck(f"no way out of cut cell {s}")
        *_, way, comp = min(ways, key=lambda w: w[:3])
        self.clear(way, comp, way)
        self.walk(way)

    def place(self, at: str) -> None:
        """Push every idle vehicle that is off the rest cells onto the nearest free rest cell, through G - at
        (R27 Run.place)."""
        region = self.all - {at}
        for a in sorted(c for c in self.occ if c not in self.rest and c != at):
            prev, dist = _bfs(self.cm, a, region)
            free = [c for c in prev if c in self.rest and c not in self.occ]
            if not free:
                raise Stuck(f"no free rest cell for the vehicle on {a}")
            self.shift(_path(prev, min(free, key=lambda c: (dist[c], c))))

    def finish(self) -> None:
        """r ends on a rest cell, every idle vehicle on another one (R26 §3 step 6)."""
        at = self.pos[self.r]
        if at not in self.stations:
            raise Stuck(f"vehicle {self.r} ends its legs on {at}, which is no station")
        self.place(at)
        if at in self.rest:
            return
        prev, dist = _bfs(self.cm, at, {c for c in self.all if c not in self.occ})
        free = [c for c in prev if c in self.rest and c not in self.occ]
        if free:
            self.walk(_path(prev, min(free, key=lambda c: (dist[c], c)))[1:])
            return
        _, dist = _bfs(self.cm, at, self.all)
        f = min(self.rest & self.stations, key=lambda c: (c in self.occ, dist[c], c))
        self.goto(f)
        self.place(f)


def solve(cm: CellMap, rest: Collection[str], hops: Mapping[str, Mapping[str, tuple[str, ...]]],
          at: Mapping[int, str], r: int, legs: Sequence[str]) -> tuple[list[tuple], dict[int, str]]:
    """The cell moves of one job of r: (steps, final cells).

    steps, in order: ("move", vehicle, from cell, to cell) and ("load", r, port node). at: every vehicle's cell, one
    vehicle per cell. hops: station_hops(cm, len(cm.cells) - len(at)). legs: the port nodes r loads at, in order.
    final: every vehicle's rest cell, by vehicle. Raises Stuck when there is no way; on a map whose gate holds that
    means a bug.
    """
    run = _Run(cm, frozenset(rest), hops, at, r)
    groups: list[tuple[str, list[str]]] = []
    for p in legs:
        if groups and groups[-1][0] == cm.cell_of[p]:
            groups[-1][1].append(p)
        else:
            groups.append((cm.cell_of[p], [p]))
    if run.pos[r] not in cm.stations:
        run.leave_cut(groups[0][0] if groups else None)
    for cell, ports in groups:
        run.visit(cell, ports)
    run.finish()
    return run.steps, dict(sorted(run.pos.items()))
