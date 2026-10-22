"""The execution event graph of the TAPG (design §11): a cycle in it is a deadlock built into the committed plans.

A re-implementation of DAIM's core/plan_validator.py:184-255 on plain dicts; it uses none of the planner's or the
simulator's code. Each action j of a vehicle (a move M, a turn R, a load/unload L) has a start event ('s', vehicle, j)
and an end event ('e', vehicle, j); a vehicle's events run s(j) -> e(j) -> s(next action). A cross edge u -> v of
the TAPG becomes e(vehicle of u, rel(u)) -> s(vehicle of v, req(v)). The node graph alone misses two vehicles
swapping places (DAIM design §6.1); this graph does not.
"""
from collections.abc import Container, Hashable, Iterable, Mapping, Sequence

MR = ("M,", "R,")
ACT = ("M,", "R,", "L,")

Key = tuple[str, int, float]
Event = tuple[str, int, int]


def event_graph(paths: Mapping[int, Sequence[tuple[str, float]]], claims: Mapping[int, int], nodes: Container[Key],
                cross_edges: Iterable[tuple[Key, Key]]) -> tuple[dict[Event, set[Event]], int]:
    """(event graph, number of permanent blocks) of the path elements whose node (state, vehicle, time rounded to
    6 decimals) is still in nodes; for a node that appears twice in a path the later index counts.

    rel(u) = u itself if u is an L, else the first M/R at or after u (u is over when that one ends); None after the
    last M/R: an edge out of such a node never clears, a permanent block (counted, not added).
    req(v) = None if v is already claimed (index < claims[vehicle]); v itself if v is an action; the M/R just before
    v if v is a stop after one (the move and the stop are claimed together); else None (a stop after an L)."""
    idx: dict[Key, int] = {}
    rel: dict[Key, int | None] = {}
    req: dict[Key, int | None] = {}
    graph: dict[Event, set[Event]] = {}
    for vid, path in paths.items():
        n = len(path)
        nxt_mr: list[int | None] = [None] * n
        nm = None
        for k in range(n - 1, -1, -1):
            if path[k][0].startswith(MR):
                nm = k
            nxt_mr[k] = nm
        claim = claims.get(vid, 0) or 0
        prev_act, acts = None, []
        for k, (sid, t) in enumerate(path):
            key = (sid, vid, round(t, 6))
            is_act = sid.startswith(ACT)
            if key in nodes:
                idx[key] = k
                rel[key] = k if sid.startswith("L,") else nxt_mr[k]
                if k < claim:
                    req[key] = None
                elif is_act:
                    req[key] = k
                elif prev_act is not None and path[prev_act][0].startswith(MR):
                    req[key] = prev_act
                else:
                    req[key] = None
                if is_act:
                    acts.append(k)
            if is_act:  # an action counts as the previous one even when its node has left the TAPG
                prev_act = k
        for i, j in enumerate(acts):
            graph.setdefault(("s", vid, j), set()).add(("e", vid, j))
            graph.setdefault(("e", vid, j), set())
            if i + 1 < len(acts):
                graph[("e", vid, j)].add(("s", vid, acts[i + 1]))
    forever = 0
    for u, v in cross_edges:
        if u[1] == v[1]:
            continue
        q = req.get(v)
        if q is None:
            continue
        r = rel.get(u)
        if r is None:
            if u in idx:
                forever += 1
            continue
        graph.setdefault(("e", u[1], r), set()).add(("s", v[1], q))
        graph.setdefault(("s", v[1], q), set())
    return graph, forever


def first_cycle(graph: Mapping[Hashable, Iterable[Hashable]]) -> list | None:
    """One cycle [v0, v1, ...] (each vertex has an edge to the next and the last one to the first), None if none.

    Kahn's algorithm first, linear like DAIM's check (B final review I-2). Only if vertices are left, walk back from
    the smallest of them along the smallest predecessor that is left: every vertex left has one, so the walk closes
    a cycle. Whether there is a cycle is DAIM's answer; which cycle comes back may differ from networkx's."""
    succ = {u: list(vs) for u, vs in graph.items()}
    for vs in list(succ.values()):
        for v in vs:
            succ.setdefault(v, [])
    indeg = dict.fromkeys(succ, 0)
    for vs in succ.values():
        for v in vs:
            indeg[v] += 1
    stack = [u for u, d in indeg.items() if d == 0]
    while stack:
        for v in succ[stack.pop()]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    left = {u for u, d in indeg.items() if d > 0}
    if not left:
        return None
    pred: dict[Hashable, list] = {u: [] for u in left}
    for u in left:
        for v in succ[u]:
            if v in left:
                pred[v].append(u)
    walk, seen = [min(left)], {}
    while walk[-1] not in seen:
        seen[walk[-1]] = len(walk) - 1
        walk.append(min(pred[walk[-1]]))
    return walk[seen[walk[-1]]:-1][::-1]
