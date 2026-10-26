"""The cell model of a map (M6 design §5.2): its nodes grouped into cells, and the cut and loop cells of the cell graph.

Cell rule (R25): when a state collides with a stop or load state at node y, y lies in a cell the state touches. A stop,
rotate or load state touches the cell of its node, a move state the cells of its two nodes. Inner-link rule: the nodes
of a cell are joined by segments inside it. Both rules are applied until nothing changes. A cell's id is its smallest
node id.
"""
import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from agvtb.world.model import World
from agvtb.world.states import Conflicts, StateGraph


@dataclass(frozen=True)
class CellMap:
    """The cells of a map and its cell graph. Every mapping is a plain dict."""

    cell_of: Mapping[str, str]  # node -> cell id (the cell's smallest node id)
    members: Mapping[str, tuple[str, ...]]  # cell id -> its nodes, sorted
    adj: Mapping[str, tuple[str, ...]]  # cell graph: cell id -> neighbour cell ids, sorted
    port_cells: frozenset[str]
    cut: frozenset[str]  # removing one splits the cell graph
    loop: frozenset[str]  # touch an edge that is no bridge
    merges: int = 0  # merges made by the inner-link rule

    @property
    def cells(self) -> tuple[str, ...]:
        return tuple(sorted(self.members))

    @property
    def stations(self) -> frozenset[str]:
        return frozenset(self.members) - self.cut


def cut_and_loop(adj: Mapping[str, Sequence[str]]) -> tuple[frozenset[str], frozenset[str]]:
    """(cut, loop) of an undirected graph: its articulation points, and the vertices on an edge that is no bridge.

    Tarjan's low-link on an explicit stack, so a big cell graph does not hit the recursion limit. Roots and
    neighbours are visited in sorted order.
    """
    order: dict[str, int] = {}
    low: dict[str, int] = {}
    cut: set[str] = set()
    bridges: set[frozenset[str]] = set()
    for root in sorted(adj):
        if root in order:
            continue
        order[root] = low[root] = len(order)
        children = 0
        stack = [(root, None, iter(sorted(adj[root])))]
        while stack:
            v, parent, nbrs = stack[-1]
            for u in nbrs:
                if u == parent:
                    continue
                if u in order:
                    low[v] = min(low[v], order[u])
                else:
                    order[u] = low[u] = len(order)
                    children += v == root
                    stack.append((u, v, iter(sorted(adj[u]))))
                    break
            else:
                stack.pop()
                if parent is not None:
                    low[parent] = min(low[parent], low[v])
                    if low[v] > order[parent]:
                        bridges.add(frozenset((parent, v)))
                    if parent != root and low[v] >= order[parent]:
                        cut.add(parent)
        if children > 1:
            cut.add(root)
    loop = frozenset(v for v in adj if any(frozenset((v, u)) not in bridges for u in adj[v]))
    return frozenset(cut), loop


def _cell_map(members: dict[str, tuple[str, ...]], links: Iterable[tuple[str, str]], ports: Iterable[str],
              merges: int) -> CellMap:
    cell_of = {n: cell for cell, nodes in members.items() for n in nodes}
    nbrs: dict[str, set[str]] = {cell: set() for cell in members}
    for a, b in links:
        if cell_of[a] != cell_of[b]:
            nbrs[cell_of[a]].add(cell_of[b])
            nbrs[cell_of[b]].add(cell_of[a])
    adj = {cell: tuple(sorted(nbrs[cell])) for cell in sorted(nbrs)}
    cut, loop = cut_and_loop(adj)
    return CellMap(cell_of, members, adj, frozenset(cell_of[p] for p in ports), cut, loop, merges)


def cell_map_from_graph(adj: Mapping[str, Iterable[str]], ports: Iterable[str]) -> CellMap:
    """The CellMap of a graph whose nodes are cells of their own (R26's node model, small test graphs)."""
    members = {v: (v,) for v in sorted(adj)}
    return _cell_map(members, [(v, u) for v in sorted(adj) for u in sorted(adj[v])], ports, 0)


def build_cell_map(world: World, graph: StateGraph, conflicts: Conflicts) -> CellMap:
    """The cells of a map by the cell rule and the inner-link rule (design §5.2), and its cell graph.

    Union-find whose root is the smaller node id, so a cell's root is its smallest node. Each round first applies
    the cell rule until a full pass changes nothing: a stop y outside the cells a colliding state touches joins the
    cell of the touched node nearest to y (Euclidean, then node id). Then the first cell (by id) that its inner
    segments do not join absorbs the nodes of a shortest node-graph path from its smallest node's part to another
    part, and the next round starts. The rounds end when every cell is joined inside.
    """
    parent = {n: n for n in sorted(world.nodes)}

    def find(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def join(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs = []
    for sid in sorted(graph):
        for other in sorted(conflicts[sid]):
            if graph[other].kind in ("S", "L"):
                pairs.append((graph[sid].nodes, graph[other].nodes[0]))
    nadj: dict[str, set[str]] = {n: set() for n in world.nodes}
    for u, v in world.segments:
        nadj[u].add(v)
        nadj[v].add(u)

    def groups() -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for n in sorted(world.nodes):
            out.setdefault(find(n), []).append(n)
        return out

    merges = 0
    while True:
        moved = True
        while moved:
            moved = False
            for touched, y in pairs:
                if find(y) not in {find(x) for x in touched}:
                    join(y, min(touched, key=lambda x: (math.dist(world.nodes[x], world.nodes[y]), x)))
                    moved = True
        split = None
        for cell, nodes in sorted(groups().items()):
            inside, part, todo = set(nodes), {nodes[0]}, [nodes[0]]
            while todo:
                x = todo.pop()
                for u in sorted(nadj[x]):
                    if u in inside and u not in part:
                        part.add(u)
                        todo.append(u)
            if len(part) < len(nodes):
                split = cell, part, inside - part
                break
        if split is None:
            break
        cell, part, others = split
        prev: dict[str, str | None] = {x: None for x in sorted(part)}
        queue, hit = deque(sorted(part)), None
        while queue and hit is None:
            x = queue.popleft()
            for u in sorted(nadj[x]):
                if u not in prev:
                    prev[u] = x
                    if u in others:
                        hit = u
                        break
                    queue.append(u)
        if hit is None:
            raise ValueError(f"cell {cell} is split and the node graph does not join its parts")
        while hit is not None:
            join(hit, cell)
            hit = prev[hit]
        merges += 1
    members = {cell: tuple(nodes) for cell, nodes in sorted(groups().items())}
    return _cell_map(members, world.segments, world.port_nodes, merges)


def dead_end_cells(cm: CellMap) -> frozenset[str]:
    """Cells of the trees that hang off the loop cells by one cell (R26 dead_end_nodes); every cell with no loop."""
    if not cm.loop:
        return frozenset(cm.members)
    out: set[str] = set()
    seen: set[str] = set()
    for v in cm.cells:
        if v in cm.loop or v in seen:
            continue
        comp, todo = {v}, [v]
        while todo:
            x = todo.pop()
            for u in cm.adj[x]:
                if u not in cm.loop and u not in comp:
                    comp.add(u)
                    todo.append(u)
        seen |= comp
        if len({u for x in comp for u in cm.adj[x] if u in cm.loop}) == 1:
            out |= comp
    return frozenset(out)
