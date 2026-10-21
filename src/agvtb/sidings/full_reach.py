"""With every siding full, what can a vehicle still reach? The checks of the M3 gate beyond condition M.

User decision G2 (2026-10-21, MAPF/docs/PLAN.md §5): the M3 gate is condition M plus "from every siding, with
every other siding holding a vehicle, a round trip to every node" (report R07 §2, Part 1). The paper's condition
(R07 Part 4, after the PSIPP paper §4.3) is reported alongside and not judged.

A vehicle parked on a siding blocks its footprint, as condition M counts it (condition_m.py:133-163): the states
that take the siding's node and the conflicts of the siding's S states. The nodes are those the state graph
names (the ends of every M state and the node of every S state), as condition M counts nodes; R07 used the
world's nodes, which differ only by nodes on no segment. A node with no S state is always missed and outside.
"""
import collections
from collections.abc import Iterable

from agvtb.sidings.condition_m import _reach
from agvtb.world.states import Conflicts, StateGraph


class _Index:
    """What both checks need from a state graph: nodes, stop poses, footprints, and the two edge directions."""

    def __init__(self, graph: StateGraph, conflicts: Conflicts, sidings: Iterable[str]):
        self.graph = graph
        self.stops_at: dict[str, list[str]] = collections.defaultdict(list)  # node -> its S states, sorted
        through: dict[str, set[str]] = collections.defaultdict(set)  # node -> the states that take it
        nodes: set[str] = set()
        for sid in sorted(graph):
            state = graph[sid]
            if state.kind == "S":
                self.stops_at[state.nodes[0]].append(sid)
            if state.kind in ("S", "M"):
                nodes.update(state.nodes)
            for node in state.nodes:
                through[node].add(sid)
        self.sidings = sorted(set(sidings))
        unknown = [s for s in self.sidings if s not in nodes]
        if unknown:
            raise ValueError(f"sidings that are no node of the state graph: {unknown}")
        siding_set = set(self.sidings)
        self.others = [n for n in sorted(nodes) if n not in siding_set]  # the nodes that are no siding
        self.footprint = {s: set(through[s]).union(*(conflicts[x] for x in self.stops_at[s])) for s in self.sidings}
        self.succ = {sid: state.next for sid, state in graph.items()}
        self.pred: dict[str, list[str]] = collections.defaultdict(list)
        for sid, following in self.succ.items():
            for after in following:
                self.pred[after].append(sid)

    def round_trips(self, start: list[str], blocked: set[str]) -> set[str]:
        """The states reachable from start, and back to it, through unblocked states."""
        return _reach(start, self.succ, self.graph, blocked) & _reach(start, self.pred, self.graph, blocked)

    def unreached(self, both: set[str]) -> tuple[str, ...]:
        """The other nodes none of whose S states is in both."""
        return tuple(n for n in self.others if not any(sid in both for sid in self.stops_at.get(n, ())))


def round_trip_misses(graph: StateGraph, conflicts: Conflicts, sidings: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Per siding (sorted): the nodes (sorted) its vehicle cannot make a round trip to while every other siding
    holds a vehicle. A state only the siding's own footprint holds is free. R07 Part 1 (full_check).

    Raises ValueError if a siding is no node of the state graph.
    """
    ix = _Index(graph, conflicts, sidings)
    holders: collections.Counter[str] = collections.Counter()
    for states in ix.footprint.values():
        holders.update(states)
    held = set(holders)
    misses: dict[str, tuple[str, ...]] = {}
    for s in ix.sidings:
        blocked = held - {sid for sid in ix.footprint[s] if holders[sid] == 1}
        start = [sid for sid in ix.stops_at.get(s, ()) if sid not in blocked]
        misses[s] = ix.unreached(ix.round_trips(start, blocked)) if start else tuple(ix.others)
    return misses


def paper_outside(
    graph: StateGraph, conflicts: Conflicts, ports: Iterable[str], sidings: Iterable[str]
) -> tuple[str, ...]:
    """The paper's condition (R07 Part 4, paper_check): every siding full and its whole footprint blocked. From
    one reference stop pose, the first free S state of the ports that are no siding, else of the other nodes
    (each in sorted order), the nodes that are no siding and have no S state a vehicle can reach and come back
    from are outside (sorted). With no free reference pose every such node is outside.

    Raises ValueError if a siding is no node of the state graph.
    """
    ix = _Index(graph, conflicts, sidings)
    blocked: set[str] = set().union(*ix.footprint.values())
    siding_set = set(ix.sidings)
    looked = [p for p in sorted(set(ports)) if p not in siding_set] + ix.others
    ref = next((sid for n in looked for sid in ix.stops_at.get(n, ()) if sid not in blocked), None)
    if ref is None:
        return tuple(ix.others)
    return ix.unreached(ix.round_trips([ref], blocked))
