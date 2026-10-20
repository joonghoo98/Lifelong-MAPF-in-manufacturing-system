"""The strict rule of the sidings: a greedy choice of waiting nodes on a directed node/arc graph.

It re-implements DAIM's strict cut-safe sidings (mapgen/track_sidings.py, from the legacy gen_strict_sidings).
The rule: for every ordered pair (u, v), u != v, of ports and sidings, v can be reached from u while every
other siding is blocked. So a vehicle can stand on every siding at once and the ports and sidings still reach
each other. Two entry points read a graph: an xMSmap dict and an M1 World.
"""
from collections import defaultdict, deque
from collections.abc import Hashable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet

from agvtb.world.model import World

Node = Hashable


def adjacency(
    nodes: Iterable[Node], arcs: Iterable[tuple[Node, Node]]
) -> tuple[set[Node], defaultdict[Node, set[Node]], defaultdict[Node, set[Node]]]:
    """All the nodes, and the successors and predecessors of each node.

    The nodes are the listed ones and those only an arc names. succ and pred are defaultdicts, so a node
    with no arc out (or in) gives the empty set.
    """
    succ: defaultdict[Node, set[Node]] = defaultdict(set)
    pred: defaultdict[Node, set[Node]] = defaultdict(set)
    all_nodes = set(nodes)
    for start, end in arcs:
        succ[start].add(end)
        pred[end].add(start)
        all_nodes.update((start, end))
    return all_nodes, succ, pred


def pairs_ok_slow(
    succ: Mapping[Node, AbstractSet[Node]], ports: AbstractSet[Node], sidings: AbstractSet[Node]
) -> bool:
    """Whether the strict rule holds, checked pair by pair as the legacy code does.

    This is the rule itself, and the answer when there are no ports; rule_ok is its fast form.
    The arguments are only read.
    """
    occupiable = list(ports) + list(sidings)
    for u in occupiable:
        for v in occupiable:
            if u != v and v not in _reach(u, succ, sidings - {u, v}):
                return False
    return True


def rule_ok(
    succ: Mapping[Node, AbstractSet[Node]],
    pred: Mapping[Node, AbstractSet[Node]],
    ports: AbstractSet[Node],
    sidings: AbstractSet[Node],
) -> bool:
    """Whether the strict rule holds for sidings; the same answer as pairs_ok_slow, found faster.

    With ports, let C be the nodes that are no siding. The rule holds exactly when the ports are all in one
    strongly connected part of C, and every siding has a successor that reaches that part inside C and a
    predecessor that the part reaches inside C (the sidings reach each other through the part). Without
    ports it is pairs_ok_slow. No node is both a port and a siding, as strict_sidings keeps it; with such a
    node the two forms can disagree. The arguments are only read: succ and pred may be defaultdicts and stay
    as they are.
    """
    if not ports:
        return pairs_ok_slow(succ, ports, sidings)
    p0 = min(ports)  # any port gives the same answer: the rule puts every port in p0's component
    fwd = _reach(p0, succ, sidings)  # the nodes p0 reaches without stepping on a siding
    bwd = _reach(p0, pred, sidings)  # the nodes that reach p0 without stepping on a siding
    if any(p not in fwd or p not in bwd for p in ports):
        return False
    return all(
        any(w in bwd for w in succ.get(s, ())) and any(w in fwd for w in pred.get(s, ()))
        for s in sidings
    )


def strict_sidings(
    nodes: Iterable[Node],
    arcs: Iterable[tuple[Node, Node]],
    ports: Iterable[Node],
    *,
    fixed: Sequence[Node] = (),
    target: int | None = None,
) -> tuple[list[Node], bool]:
    """Greedily pick sidings that keep the strict rule; returns (the sorted sidings, whether fixed keeps it).

    The sidings start as the nodes of fixed that are in the graph and are no ports. They stay even if they
    break the rule; then the second value is False and no candidate can be added (adding a siding never
    repairs the rule). The other nodes that are no ports are the candidates, taken by (number of successors +
    number of predecessors, number of neighbours, id), ascending. Ids compare as given, so string ids sort as
    strings ('10' < '2'). A candidate stays if the rule still holds with it. With target, adding stops once
    that many sidings are in (the fixed ones count).
    """
    all_nodes, succ, pred = adjacency(nodes, arcs)
    port_set = set(ports)
    chosen = {n for n in fixed if n in all_nodes and n not in port_set}
    fixed_ok = rule_ok(succ, pred, port_set, chosen) if chosen else True
    candidates = sorted(
        all_nodes - port_set - chosen,
        key=lambda n: (len(succ[n]) + len(pred[n]), len(succ[n] | pred[n]), n),
    )
    for n in candidates:
        if target is not None and len(chosen) >= target:
            break
        chosen.add(n)
        if not rule_ok(succ, pred, port_set, chosen):
            chosen.discard(n)
    return sorted(chosen), fixed_ok


def strict_sidings_of_layout(layout: dict) -> list[str]:
    """The sidings the strict rule picks on an xMSmap dict, as sorted node ids.

    Nodes are nodes[].id, arcs are segments[] from startNodeId to endNodeId, ports are ports[].nodeId (the
    port's id if it has no nodeId).
    """
    ports = {p.get("nodeId", p.get("id")) for p in layout.get("ports", [])}
    arcs = [(seg["startNodeId"], seg["endNodeId"]) for seg in layout.get("segments", [])]
    nodes = [node["id"] for node in layout.get("nodes", [])]
    return strict_sidings(nodes, arcs, ports)[0]


def strict_sidings_of_world(world: World, *, target: int | None = None) -> list[str]:
    """The sidings the strict rule picks on world, as sorted node ids; at most target of them if given.

    Nodes are world.nodes, arcs are the keys of world.segments, ports are world.port_nodes.
    """
    return strict_sidings(list(world.nodes), list(world.segments), world.port_nodes, target=target)[0]


def _reach(
    src: Node, nb: Mapping[Node, AbstractSet[Node]], blocked: AbstractSet[Node]
) -> set[Node]:
    """The nodes reachable from src along nb (successors or predecessors) without entering a blocked node."""
    seen = {src}
    queue = deque([src])
    while queue:
        u = queue.popleft()
        for v in nb.get(u, ()):
            if v not in seen and v not in blocked:
                seen.add(v)
                queue.append(v)
    return seen
