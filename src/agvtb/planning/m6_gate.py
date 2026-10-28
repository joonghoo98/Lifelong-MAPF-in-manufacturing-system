"""The M6 gate (design §5.3): the rest rules, the station hops and Theorem C on the cell graph.

Theorem C (R26 §3; its T1..T4 are C1..C4 here) lets idle vehicles rest on the cells Q of a rest rule when
  C1  |Q| >= N, and Q holds a station;
  C2  the stations are all joined by hops of at most V - N edges (a hop: a path whose inner cells are no stations);
  C3  a planned vehicle on a cut cell s (a rest cell or a port cell) can always reach a station: some component C of
      G - s cannot be blocked (blocking it takes |C| - d + 1 idle vehicles in C, d the distance from s to the nearest
      station of C), or blocking them all takes more than N - 1 idle vehicles;
  C4  every port cell that is a cut cell lies within V - N edges of a station.
V is the number of cells and N the number of vehicles. The verdicts are those of R26 check_c.theorem_c.
The theorems also assume a map of two-way segments that join all its nodes (design §3); map_assumptions checks
that first, and state_facts the facts Lemma D (R34) takes from the states (D9, D12); a map that breaks one is
refused before its cells are built. Once they are, cell_facts checks the facts Theorem B uses besides the cell rule
(R32 I-2), and cells that break one are refused too.
"""
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass

from agvtb.planning.intervals import EPS
from agvtb.planning.reservations import affects
from agvtb.world.cells import CellMap, dead_end_cells
from agvtb.world.model import World
from agvtb.world.states import Conflicts, StateGraph, geom_key

RULES = ("no-cut", "branch", "all")


def map_assumptions(world: World) -> tuple[str, ...]:
    """The assumptions of the theorems that the map breaks (design §3), each starting "map: ": every segment has
    its reverse, and the segments join all nodes. Empty when both hold."""
    out = []
    one_way = sorted((u, v) for u, v in world.segments if (v, u) not in world.segments)
    if one_way:
        out.append(f"map: {len(one_way)} one-way segments, first {one_way[:3]}")
    nbrs: dict[str, set[str]] = {n: set() for n in world.nodes}
    for u, v in world.segments:
        nbrs[u].add(v)
        nbrs[v].add(u)
    nodes = sorted(world.nodes)
    seen, todo = set(nodes[:1]), nodes[:1]
    while todo:
        for u in sorted(nbrs[todo.pop()]):
            if u not in seen:
                seen.add(u)
                todo.append(u)
    if len(seen) < len(nodes):
        out.append(f"map: the node graph is not connected: {len(nodes) - len(seen)} of {len(nodes)} nodes are not "
                   f"joined to {nodes[0]}")
    return tuple(out)


def cell_facts(cm: CellMap, graph: StateGraph, conflicts: Conflicts, ports: Collection[str]) -> tuple[str, ...]:
    """The facts Theorem B uses besides the cell rule (R32 I-2) that the cells break, each a line starting "cells: ".

    The conflict sets are symmetric; every cell has a stop; from every stop of a cell, a stop of each neighbour cell
    is reached through states whose nodes all lie in the two cells; from every stop of a port's cell, an L state at
    the port is reached inside that cell (an L state goes back to its own stop, states.py). Empty when all hold.
    """
    out = []
    asym = sorted((a, b) for a in conflicts for b in conflicts[a] if a not in conflicts.get(b, ()))
    if asym:
        out.append(f"cells: {len(asym)} conflict entries are not symmetric, first {asym[0][0]} -> {asym[0][1]}")
    touch = {sid: frozenset(cm.cell_of[n] for n in st.nodes) for sid, st in graph.items()}
    by_cell: dict[str, list[str]] = {c: [] for c in cm.cells}
    stops: dict[str, list[str]] = {c: [] for c in cm.cells}
    for sid, cs in touch.items():
        for c in cs:
            by_cell[c].append(sid)
        if graph[sid].kind == "S":
            stops[cm.cell_of[graph[sid].nodes[0]]].append(sid)
    empty = [c for c in cm.cells if not stops[c]]
    if empty:
        out.append(f"cells: {len(empty)} cells have no stop, first {empty[0]!r}")

    def inside(cs: frozenset[str]) -> set[str]:
        return {sid for c in cs for sid in by_cell[c] if touch[sid] <= cs}

    def reaches(start: str, allowed: set[str], goal) -> bool:
        seen, todo = {start}, deque([start])
        while todo:
            x = todo.popleft()
            if goal(graph[x]):
                return True
            for y in graph[x].next:
                if y in allowed and y not in seen:
                    seen.add(y)
                    todo.append(y)
        return False

    stuck = []
    for c in cm.cells:
        for d in cm.adj[c]:
            allowed = inside(frozenset((c, d)))
            stuck += [(s, c, d) for s in stops[c]
                      if not reaches(s, allowed, lambda st: st.kind == "S" and cm.cell_of[st.nodes[0]] == d)]
    if stuck:
        s, c, d = stuck[0]
        out.append(f"cells: {len(stuck)} stops cannot reach a neighbour cell through the two cells, first {s} in "
                   f"{c!r} to {d!r}")
    unloaded = []
    for p in sorted(ports):
        c = cm.cell_of[p]
        allowed = inside(frozenset((c,)))
        unloaded += [(s, p) for s in stops[c]
                     if not reaches(s, allowed, lambda st: st.kind == "L" and st.nodes[0] == p)]
    if unloaded:
        s, p = unloaded[0]
        out.append(f"cells: {len(unloaded)} stops cannot reach the L state of their port inside its cell, first {s} "
                   f"to port {p!r}")
    return tuple(out)


def state_facts(graph: StateGraph, conflicts: Conflicts) -> tuple[str, ...]:
    """The facts Lemma D (R34) takes from the states that they break, each a line starting "states: ".

    D9: every action (M, R or L) lasts longer than EPS; the config takes a dwell of 0 s. D12 (P): every M or R into a
    stop collides with every other state that collides with the stop, by the TAPG's rule (the same geometric key, or
    a conflict either way, tapg.py). Polygon and grid conflicts give (P) by construction; a profile is a table that
    comes with the map. Empty when both hold.
    """
    out = []
    short = sorted(sid for sid, st in graph.items() if st.kind in ("M", "R", "L") and st.duration_s <= EPS)
    if short:
        out.append(f"states: {len(short)} actions last {EPS:g} s or less (R34 D9), first {short[0]} "
                   f"({graph[short[0]].duration_s:g} s)")
    aff = {sid: frozenset(affects(conflicts, sid)) for sid in graph}
    by_key: dict[str, set[str]] = {}  # geometric key -> the states that take it
    hit_by: dict[str, set[str]] = {}  # geometric key -> the states whose affects hold it
    for sid in graph:
        by_key.setdefault(geom_key(sid), set()).add(sid)
        for g in aff[sid]:
            hit_by.setdefault(g, set()).add(sid)

    def colliding(sid: str) -> set[str]:
        g = geom_key(sid)
        found = by_key.get(g, set()) | hit_by.get(g, set())
        for g2 in aff[sid]:
            found |= by_key.get(g2, set())
        found.discard(sid)
        return found

    hits: dict[str, set[str]] = {}  # stop -> the states that collide with it
    broken = []
    for sid in sorted(graph):
        if graph[sid].kind not in ("M", "R"):
            continue
        for s in graph[sid].next:
            if graph[s].kind == "S":
                if s not in hits:
                    hits[s] = colliding(s)
                broken += [(sid, s, x) for x in sorted(hits[s] - colliding(sid) - {sid})]
    if broken:
        m, s, x = broken[0]
        out.append(f"states: {len(broken)} states collide with a stop but not with a move or turn into it "
                   f"(R34 D12 (P)), first {x} with {s}, not {m}")
    return tuple(out)


def rest_cells(cm: CellMap, rule: str) -> tuple[str, ...]:
    """The rest cells Q of a rule (design §5.2), sorted. Raises ValueError for a rule not in RULES."""
    no_port = set(cm.members) - cm.port_cells
    if rule == "no-cut":
        return tuple(sorted(no_port - cm.cut))
    if rule == "branch":
        return tuple(sorted((no_port - cm.cut) | (dead_end_cells(cm) - cm.port_cells)))
    if rule == "all":
        return tuple(sorted(no_port))
    raise ValueError(f"rest rule must be one of {RULES}, got {rule!r}")


def station_hops(cm: CellMap, limit: int) -> dict[str, dict[str, tuple[str, ...]]]:
    """Station w -> {station z: the cells after w up to z on a shortest hop}, for the hops of at most limit edges.

    A breadth-first search from each station w that does not go on from the other stations (R26 distances with
    stop=stations). Stations and neighbours in sorted order; z ranges over the stations other than w.
    """
    stations = cm.stations
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for w in sorted(stations):
        prev: dict[str, str] = {}
        dist, queue, found = {w: 0}, deque([w]), {}
        while queue:
            x = queue.popleft()
            if dist[x] >= limit:
                continue
            for u in cm.adj[x]:
                if u in dist:
                    continue
                dist[u], prev[u] = dist[x] + 1, x
                if u not in stations:
                    queue.append(u)
                    continue
                path, y = [], u
                while y != w:
                    path.append(y)
                    y = prev[y]
                found[u] = tuple(reversed(path))
        out[w] = {z: found[z] for z in sorted(found)}
    return out


def _joined(stations: Collection[str], hops: list[tuple[int, str, str]]) -> tuple[bool, int]:
    """Kruskal over hops (length, w, z) in sorted order: (all stations joined, the longest hop it took)."""
    group = {w: w for w in stations}

    def find(x: str) -> str:
        while group[x] != x:
            x = group[x]
        return x

    joined, longest = 1, 0
    for d, w, z in sorted(hops):
        if joined >= len(group):
            break
        if find(w) != find(z):
            group[find(w)] = find(z)
            joined += 1
            longest = d
    return joined >= len(group), longest


def _hop_list(hops: dict[str, dict[str, tuple[str, ...]]]) -> list[tuple[int, str, str]]:
    return [(len(path), w, z) for w, found in hops.items() for z, path in found.items() if z > w]


def longest_hop(cm: CellMap) -> int:
    """The longest hop a spanning tree of the stations needs, shortest hops first (R26 maps_c.longest_hop).

    0 when there is at most one station.
    """
    return _joined(cm.stations, _hop_list(station_hops(cm, len(cm.cells))))[1]


def components(cm: CellMap, gone: str) -> list[set[str]]:
    """The components of the cell graph without the cell gone, by their smallest cell id."""
    seen, comps = {gone}, []
    for v in cm.cells:
        if v in seen:
            continue
        comp, todo = {v}, [v]
        seen.add(v)
        while todo:
            for u in cm.adj[todo.pop()]:
                if u not in seen:
                    seen.add(u)
                    comp.add(u)
                    todo.append(u)
        comps.append(comp)
    return comps


def _distances(cm: CellMap, start: str, allowed: Collection[str]) -> dict[str, int]:
    """Breadth-first distances from start through the cells in allowed."""
    dist, queue = {start: 0}, deque([start])
    while queue:
        x = queue.popleft()
        for u in cm.adj[x]:
            if u in allowed and u not in dist:
                dist[u] = dist[x] + 1
                queue.append(u)
    return dist


def theorem_c(cm: CellMap, rest: Collection[str], n_vehicles: int) -> tuple[str, ...]:
    """The conditions C1..C4 of Theorem C that fail for n_vehicles resting on rest, in order; () when it applies."""
    stations, rest, everything = cm.stations, set(rest), set(cm.members)
    room = len(everything) - n_vehicles
    failed = []
    if len(rest) < n_vehicles or not rest & stations:
        failed.append("C1")
    if not _joined(stations, _hop_list(station_hops(cm, room)))[0]:
        failed.append("C2")
    for s in sorted((rest | cm.port_cells) & cm.cut):
        needs, open_way = [], False
        for comp in components(cm, s):
            dist = _distances(cm, s, comp)
            need = len(comp) - min(dist[z] for z in comp & stations) + 1
            if need > len(rest & comp):
                open_way = True
                break
            needs.append(need)
        if not open_way and sum(needs) <= n_vehicles - 1:
            failed.append("C3")
            break
    for p in sorted(cm.port_cells & cm.cut):
        if min(d for z, d in _distances(cm, p, everything).items() if z in stations) > room:
            failed.append("C4")
            break
    return tuple(failed)


@dataclass(frozen=True)
class M6GateReport:
    """The result of check_m6_gate: the rule, the sizes it was judged on and the conditions that failed."""

    rule: str
    n_vehicles: int
    n_cells: int
    n_rest: int
    longest_hop: int
    failed: tuple[str, ...]
    # map_assumptions, else state_facts, else cell_facts; when there are some, C1..C4 were not judged
    map_violations: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.failed and not self.map_violations

    @property
    def violations(self) -> tuple[str, ...]:
        """The broken assumptions of the map, its states or its cells ("map: ...", "states: ...", "cells: ..."), then
        one line per failed condition, each starting "Cn: "."""
        room = self.n_cells - self.n_vehicles
        text = {
            "C1": f"{self.n_rest} rest cells for {self.n_vehicles} vehicles, or none of them is a station",
            "C2": f"the stations are not all joined by hops of at most {room} edges",
            "C3": f"a vehicle on a cut cell can be walled in by the other {self.n_vehicles - 1} vehicles",
            "C4": f"a port on a cut cell is more than {room} edges from every station",
        }
        return self.map_violations + tuple(f"{c}: {text[c]}" for c in self.failed)

    def summary(self) -> dict:
        """The report as plain data."""
        return {
            "ok": self.ok,
            "violations": list(self.violations),
            "rule": self.rule,
            "vehicles": self.n_vehicles,
            "cells": self.n_cells,
            "rest": self.n_rest,
            "longest_hop": self.longest_hop,
            "failed": list(self.failed),
        }


def check_m6_gate(cm: CellMap, rule: str, n_vehicles: int) -> M6GateReport:
    """Theorem C for n_vehicles resting on the cells of rule. Raises ValueError if n_vehicles < 1 or the rule is
    unknown."""
    if n_vehicles < 1:
        raise ValueError(f"n_vehicles must be >= 1, got {n_vehicles}")
    rest = rest_cells(cm, rule)
    return M6GateReport(rule, n_vehicles, len(cm.cells), len(rest), longest_hop(cm), theorem_c(cm, rest, n_vehicles))
