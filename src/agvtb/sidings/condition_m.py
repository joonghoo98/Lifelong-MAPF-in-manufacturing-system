"""Map condition M of the sidings: can the candidates all hold a vehicle at once and the map still work?

It re-implements DAIM's condition M check (core/safe_tail_map.py) on M1's state graph and conflict sets.
First the candidates that cannot be sidings are dropped (usable_sidings). What is left must pass two tests
while a vehicle waits on every one of them:
- every port stands in the aisle, the largest component of the nodes that are no siding, and has a stop pose
  with a load state;
- from every siding a vehicle can drive to every port and back. A vehicle on a siding blocks its footprint:
  the states that take the siding's node, and the states that collide with the vehicle at one of its poses.
"""
import collections
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from agvtb.world.states import Conflicts, StateGraph

EXCLUSION_REASONS = ("not_in_graph", "port", "port_access", "pocket")  # why usable_sidings drops a candidate


@dataclass(frozen=True)
class ConditionMReport:
    """The result of check_condition_m."""

    sidings: tuple[str, ...]  # the usable sidings, sorted
    excluded: dict[str, str]  # dropped candidate -> its reason, one of EXCLUSION_REASONS
    aisle_size: int  # the number of nodes in the aisle
    n_vehicles: int
    violations: tuple[str, ...]  # one sentence each, in the order found

    @property
    def fatal(self) -> bool:
        """Fewer usable sidings than vehicles: not every vehicle can have a siding of its own."""
        return len(self.sidings) < self.n_vehicles

    @property
    def ok(self) -> bool:
        """Neither fatal nor any violation."""
        return not self.fatal and not self.violations

    def summary(self) -> dict:
        """The report as plain data, with the keys of DAIM's summary; excluded is sorted by candidate."""
        return {
            "ok": self.ok,
            "fatal": self.fatal,
            "sidings": len(self.sidings),
            "n_vehicles": self.n_vehicles,
            "excluded": dict(sorted(self.excluded.items())),
            "aisle_size": self.aisle_size,
            "violations": list(self.violations),
        }


def usable_sidings(
    graph: StateGraph, ports: Iterable[str], candidates: Iterable[str]
) -> tuple[list[str], dict[str, str], frozenset[str]]:
    """The candidates that can be sidings, why the others cannot, and the aisle.

    The nodes are the ends of every M state and the node of every S state; an M state joins its two ends.
    A candidate that is no node is dropped as not_in_graph, and one that is a port as port. The others are
    sidings, and two rules drop more of them, round after round, until a round drops none. Each round starts
    by finding the aisle, the largest component of the nodes that are no siding (the first one found, from
    the smallest id, wins a tie). Then:
    - port_access: from each port outside the aisle, in sorted order, a breadth-first search (neighbours in
      sorted order) goes to the first aisle node it meets; the sidings on that path are dropped. Every port
      is seen before the aisle is found again. If any siding was dropped, the round ends here.
    - pocket: otherwise a siding with no neighbour that is in the aisle and is no port is dropped.
    Returns (the usable sidings, sorted; {dropped candidate: reason}; the aisle of the last round).
    """
    adj = _node_adjacency(graph)
    port_set = set(ports)
    port_list = sorted(port_set)
    excluded: dict[str, str] = {}
    sidings: set[str] = set()
    for candidate in sorted(set(candidates)):
        if candidate not in adj:
            excluded[candidate] = "not_in_graph"
        elif candidate in port_set:
            excluded[candidate] = "port"
        else:
            sidings.add(candidate)
    nodes = set(adj)
    while True:
        aisle = _largest_component(nodes - sidings, adj)
        if _drop_port_access(adj, port_list, aisle, sidings, excluded):
            continue
        if not _drop_pockets(adj, port_set, aisle, sidings, excluded):
            return sorted(sidings), excluded, frozenset(aisle)


def check_condition_m(
    graph: StateGraph, conflicts: Conflicts, ports: Iterable[str], candidates: Iterable[str], n_vehicles: int
) -> ConditionMReport:
    """Check map condition M for the candidate sidings; n_vehicles only decides fatal.

    ports and candidates may be any iterables of node ids, in any order and with repeats; ids that are no
    nodes of the graph are reported, not rejected (see usable_sidings).
    The violations come in this order: each port outside the aisle; each port with no stop pose that has a
    load state (an S whose L is in the graph); each usable siding, in sorted order, that fails the round
    trip. For that, a vehicle waits on every other siding and blocks the states of its footprint: those that
    take its node and those that the conflicts of its S states list. From the siding's S states that are not
    blocked, the states reachable forward and the states reachable backward through unblocked states are
    found; a port can be served when one of its load poses is in both. A port without a load pose has its own
    violation and is not listed again. Every S state must be a key of conflicts, as in M1's.
    Raises ValueError if n_vehicles is negative.
    """
    if n_vehicles < 0:
        raise ValueError(f"n_vehicles must be >= 0, got {n_vehicles}")
    port_list = sorted(set(ports))
    sidings, excluded, aisle = usable_sidings(graph, port_list, candidates)
    violations = [
        f"port {port} is outside the aisle (largest component of {len(aisle)} nodes)"
        for port in port_list
        if port not in aisle
    ]

    stops_at: dict[str, list[str]] = {}  # node -> its S states, sorted
    through: dict[str, set[str]] = {}  # node -> the states that take it
    for sid in sorted(graph):
        state = graph[sid]
        if state.kind == "S":
            stops_at.setdefault(state.nodes[0], []).append(sid)
        for node in state.nodes:
            through.setdefault(node, set()).add(sid)
    # a load pose of a port is an S state there that has an L: the same id with its first letter L
    load_poses = {
        port: [sid for sid in stops_at.get(port, ()) if "L" + sid[1:] in graph] for port in port_list
    }
    violations += [
        f"port {port} has no stop pose with a load state" for port in port_list if not load_poses[port]
    ]

    footprint: dict[str, set[str]] = {}  # siding -> the states a vehicle parked there blocks
    holders: collections.Counter[str] = collections.Counter()  # state -> in how many footprints it is
    for siding in sidings:
        footprint[siding] = set(through.get(siding, ()))
        for stop in stops_at.get(siding, ()):
            footprint[siding] |= conflicts[stop]
        holders.update(footprint[siding])
    held = set(holders)

    successors = {sid: state.next for sid, state in graph.items()}
    predecessors: dict[str, list[str]] = {}
    for sid, following in successors.items():
        for after in following:
            predecessors.setdefault(after, []).append(sid)

    for siding in sidings:
        # blocked for this siding: held by another siding; a state only its own footprint holds is free
        blocked = held - {sid for sid in footprint[siding] if holders[sid] == 1}
        start = [sid for sid in stops_at.get(siding, ()) if sid not in blocked]
        if not start:
            violations.append(f"siding {siding}: every stop pose is blocked by the other sidings")
            continue
        forward = _reach(start, successors, graph, blocked)
        backward = _reach(start, predecessors, graph, blocked)
        missing = [
            port
            for port in port_list
            if load_poses[port] and not any(sid in forward and sid in backward for sid in load_poses[port])
        ]
        if missing:
            violations.append(f"siding {siding}: no round trip to ports {missing}")
    return ConditionMReport(tuple(sidings), excluded, len(aisle), n_vehicles, tuple(violations))


def _node_adjacency(graph: StateGraph) -> dict[str, set[str]]:
    """The undirected node graph: node -> its neighbours; a node that only has S states is in it, alone."""
    adj: dict[str, set[str]] = {}
    for state in graph.values():
        if state.kind == "M":
            u, v = state.nodes
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        elif state.kind == "S":
            adj.setdefault(state.nodes[0], set())
    return adj


def _largest_component(nodes: AbstractSet[str], adj: Mapping[str, AbstractSet[str]]) -> set[str]:
    """The largest component of the graph on nodes; the first found, from the smallest id, wins a tie."""
    seen: set[str] = set()
    best: set[str] = set()
    for start in sorted(nodes):
        if start in seen:
            continue
        seen.add(start)
        component, stack = {start}, [start]
        while stack:
            node = stack.pop()
            for neighbour in adj.get(node, ()):
                if neighbour in nodes and neighbour not in seen:
                    seen.add(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
        if len(component) > len(best):
            best = component
    return best


def _drop_port_access(
    adj: Mapping[str, AbstractSet[str]],
    ports: list[str],
    aisle: AbstractSet[str],
    sidings: set[str],
    excluded: dict[str, str],
) -> bool:
    """Drop the sidings on the way from each port outside the aisle to it; whether any was dropped.

    ports are in sorted order. A port that is no node is left alone (check_condition_m reports it).
    """
    dropped = False
    for port in ports:
        if port in aisle or port not in adj:
            continue
        for node in _path_to_aisle(adj, port, aisle):
            if node in sidings:
                sidings.discard(node)
                excluded[node] = "port_access"
                dropped = True
    return dropped


def _path_to_aisle(adj: Mapping[str, AbstractSet[str]], start: str, aisle: AbstractSet[str]) -> list[str]:
    """The path from start to the first aisle node a breadth-first search meets, listed from that node back.

    Neighbours are taken in sorted order. The list is empty if start cannot reach the aisle.
    """
    came_from: dict[str, str | None] = {start: None}
    queue = collections.deque([start])
    while queue:
        node = queue.popleft()
        for neighbour in sorted(adj.get(node, ())):
            if neighbour in came_from:
                continue
            came_from[neighbour] = node
            if neighbour in aisle:
                path: list[str] = []
                step: str | None = neighbour
                while step is not None:
                    path.append(step)
                    step = came_from[step]
                return path
            queue.append(neighbour)
    return []


def _drop_pockets(
    adj: Mapping[str, AbstractSet[str]],
    ports: AbstractSet[str],
    aisle: AbstractSet[str],
    sidings: set[str],
    excluded: dict[str, str],
) -> bool:
    """Drop the sidings with no neighbour that is in the aisle and is no port; whether any was dropped."""
    dropped = False
    for siding in sorted(sidings):
        if not any(neighbour in aisle and neighbour not in ports for neighbour in adj.get(siding, ())):
            sidings.discard(siding)
            excluded[siding] = "pocket"
            dropped = True
    return dropped


def _reach(
    start: list[str],
    neighbours: Mapping[str, Iterable[str]],
    states: StateGraph,
    blocked: AbstractSet[str],
) -> set[str]:
    """The states reachable from start through unblocked neighbours (successors or predecessors)."""
    seen = set(start)
    queue = collections.deque(start)
    while queue:
        sid = queue.popleft()
        for other in neighbours.get(sid, ()):
            if other not in seen and other in states and other not in blocked:
                seen.add(other)
                queue.append(other)
    return seen
