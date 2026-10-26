"""The M6 gate (design §5.3): the rest rules, the station hops and Theorem C on the cell graph.

Theorem C (R26 §3; its T1..T4 are C1..C4 here) lets idle vehicles rest on the cells Q of a rest rule when
  C1  |Q| >= N, and Q holds a station;
  C2  the stations are all joined by hops of at most V - N edges (a hop: a path whose inner cells are no stations);
  C3  a planned vehicle on a cut cell s (a rest cell or a port cell) can always reach a station: some component C of
      G - s cannot be blocked (blocking it takes |C| - d + 1 idle vehicles in C, d the distance from s to the nearest
      station of C), or blocking them all takes more than N - 1 idle vehicles;
  C4  every port cell that is a cut cell lies within V - N edges of a station.
V is the number of cells and N the number of vehicles. The verdicts are those of R26 check_c.theorem_c.
"""
from collections import deque
from collections.abc import Collection
from dataclasses import dataclass

from agvtb.world.cells import CellMap, dead_end_cells

RULES = ("no-cut", "branch", "all")


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


def _components(cm: CellMap, gone: str) -> list[set[str]]:
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
        for comp in _components(cm, s):
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

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def violations(self) -> tuple[str, ...]:
        """One line per failed condition, in order, each starting with "Cn: "."""
        room = self.n_cells - self.n_vehicles
        text = {
            "C1": f"{self.n_rest} rest cells for {self.n_vehicles} vehicles, or none of them is a station",
            "C2": f"the stations are not all joined by hops of at most {room} edges",
            "C3": f"a vehicle on a cut cell can be walled in by the other {self.n_vehicles - 1} vehicles",
            "C4": f"a port on a cut cell is more than {room} edges from every station",
        }
        return tuple(f"{c}: {text[c]}" for c in self.failed)

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
