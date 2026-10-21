"""An independent validator of planned paths (design §11): it uses none of the planner's code.

A re-implementation of DAIM's core/plan_validator.py:28-181. The geometric key and the nodes of a state are
read from the state id here, not taken from the planner or the world model. A path is [(state, time), ...];
element i lasts until the next element's time, the last one forever.
"""
import collections
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet

from agvtb.world.states import Conflicts

EPS = 1e-9
INF = float("inf")


def _geom(sid: str) -> str:
    """L,node,h stands where S,node,h stands."""
    return "S" + sid[1:] if sid.startswith("L,") else sid


def _nodes(sid: str) -> set[str]:
    p = sid.split(",")
    if p[0] in ("S", "L", "R") and len(p) >= 2:
        return {p[1]}
    if p[0] == "M" and len(p) >= 3:
        return {p[1], p[2]}
    return set()


def _affect(conflicts: Conflicts, sid: str) -> set[str]:
    return {_geom(x) for x in conflicts.get(_geom(sid), ())}


def conflicting(conflicts: Conflicts, a: str, b: str) -> bool:
    """a and b cannot be held at once: one place, a conflict either way, or two stops at one node."""
    g1, g2 = _geom(a), _geom(b)
    if g1 == g2 or g2 in _affect(conflicts, a) or g1 in _affect(conflicts, b):
        return True
    return g1.startswith("S,") and g2.startswith("S,") and g1.split(",")[1] == g2.split(",")[1]


def occupancy(path: Sequence[tuple[str, float]], t_from: float = -INF) -> list[tuple[str, float, float]]:
    """[(state, start, end)], the last element ending at INF; those over by t_from are left out."""
    out = []
    for i, (sid, t) in enumerate(path):
        end = path[i + 1][1] if i + 1 < len(path) else INF
        if end == INF or end > t_from + EPS:
            out.append((sid, float(t), end))
    return out


def _entries_vs(conflicts: Conflicts, mine, others: Mapping[int, Sequence[tuple[str, float, float]]]) -> list[dict]:
    by_node: dict[str, list[tuple[int, str, float, float]]] = collections.defaultdict(list)
    for bid, entries in others.items():
        for sid, u0, u1 in entries:
            for n in _nodes(sid):
                by_node[n].append((bid, sid, u0, u1))
    out, seen = [], set()  # seen is per call: one vehicle a against the others
    for sid, t0, t1 in mine:
        reach = set(_nodes(sid))
        for x in _affect(conflicts, sid):
            reach |= _nodes(x)
        for n in sorted(reach):
            for bid, sid2, u0, u1 in by_node.get(n, ()):
                if not (t0 < u1 - EPS and u0 < t1 - EPS):
                    continue
                key = (bid, sid, sid2)
                if key in seen or not conflicting(conflicts, sid, sid2):
                    continue
                seen.add(key)
                out.append({"b": bid, "sa": sid, "sb": sid2, "t": max(t0, u0)})
    return out


def live_entries(raw_path: Sequence[tuple[str, float]], path_idx: int, claim_idx: int,
                 now: float) -> list[tuple[str, float, float]]:
    """What a running vehicle still holds (DAIM plan_validator.py:68-83): its current state and its claimed
    states from now until their next plan time (but not before now), the rest at their plan times, the last
    forever."""
    n = len(raw_path)
    if not n:
        return []
    idx = min(int(path_idx), n - 1)
    out = []
    for i in range(idx, n):
        sid, t = raw_path[i]
        nxt = raw_path[i + 1][1] if i + 1 < n else INF
        if i == idx or i < claim_idx:
            out.append((sid, float(now), max(nxt, float(now))))
        else:
            out.append((sid, float(t), nxt))
    return out


def path_vs_entries(path: Sequence[tuple[str, float]], others: Mapping[int, Sequence[tuple[str, float, float]]],
                    conflicts: Conflicts) -> list[dict]:
    """A new path against what the other vehicles hold: [{b, sa, sb, t}] (DAIM plan_validator.py:109-112)."""
    return _entries_vs(conflicts, occupancy(path), others)


def unprotected(mine: Sequence[tuple[int, str, Hashable]], others: Mapping[int, Sequence[tuple[int, str, Hashable]]],
                related: Callable[[Hashable], AbstractSet], conflicts: Conflicts) -> list[dict]:
    """Colliding state pairs the event graph leaves unordered (DAIM plan_validator.py:125-148): for each of my
    states (i, sid, node), each other vehicle's state (j, sid2, node2) whose node is neither an ancestor nor a
    descendant of mine, and which conflicts with it. The caller passes the event graph's relation."""
    out = []
    for i, sa, na in mine:
        ordered = related(na)
        for b in sorted(others):
            for j, sb, nb in others[b]:
                if nb not in ordered and conflicting(conflicts, sa, sb):
                    out.append({"i": i, "b": b, "j": j, "sa": sa, "sb": sb})
    return out


def pairwise_overlaps(
    paths: Mapping[int, Sequence[tuple[str, float]]], conflicts: Conflicts, t_from: float = -INF
) -> list[dict]:
    """Overlaps between vehicles [{a, b, sa, sb, t}], each pair seen one way (a < b)."""
    occ = {vehicle: occupancy(p, t_from) for vehicle, p in paths.items()}
    ids = sorted(occ)
    out = []
    for i, a in enumerate(ids):
        for v in _entries_vs(conflicts, occ[a], {b: occ[b] for b in ids[i + 1:]}):
            out.append({"a": a, **v})
    return out


def tail_violations(
    paths: Mapping[int, Sequence[tuple[str, float]]], homes: Mapping[int, str], sidings: Iterable[str]
) -> list[str]:
    """Every path ends on a stop of its vehicle's own siding, and no siding has two owners."""
    siding_set = {str(s) for s in sidings}
    out = []
    for vehicle, p in sorted(paths.items()):
        if not p:
            continue
        sid = p[-1][0]
        node = sid.split(",")[1] if "," in sid else sid
        if not sid.startswith("S,"):
            out.append(f"vehicle {vehicle}: its path ends on {sid}, which is no stop state")
        elif node not in siding_set:
            out.append(f"vehicle {vehicle}: its path ends at {node}, which is no siding")
        elif homes.get(vehicle) != node:
            out.append(f"vehicle {vehicle}: its path ends at {node}, not at its own siding {homes.get(vehicle)}")
    dup = [s for s, c in collections.Counter(homes.values()).items() if c > 1]
    out += [f"siding {s} is owned by more than one vehicle" for s in sorted(dup)]
    return out


def task_order_violation(path: Sequence[tuple[str, float]], legs: Sequence[str], sidings: Iterable[str]) -> str | None:
    """None if path loads and unloads at legs in order and ends on a siding's stop; else what is wrong."""
    j = 0
    for sid, _ in path:
        if j < len(legs) and sid.startswith("L,") and sid.split(",")[1] == str(legs[j]):
            j += 1
    if j < len(legs):
        return f"loads and unloads {j}/{len(legs)} of legs {list(legs)}"
    end = path[-1][0]
    if not end.startswith("S,") or end.split(",")[1] not in {str(s) for s in sidings}:
        return f"does not end on a siding stop: {end}"
    return None


def early_fallback(path: Sequence[tuple[str, float]], t_star: float) -> bool:
    """A fallback plan that leaves its start before T* (DAIM safe-tail design §9.1 (5))."""
    return len(path) >= 2 and path[1][1] < t_star - EPS
