"""The state graph of a World: stop (S), move (M), rotate (R) and load/unload (L) states.

State ids are "S,n,h", "M,u,v", "R,n,h1,h2" and "L,n,h": n a node, h a heading in whole degrees.
"""
from dataclasses import dataclass
from itertools import permutations

from agvtb.world.model import World


@dataclass(frozen=True)
class State:
    """One state of the graph. Which fields are set depends on its kind."""

    sid: str
    kind: str  # "S", "M", "R" or "L"
    nodes: tuple[str, ...]  # (n,) for S, R and L; (u, v) for M
    heading: int | None  # S, L: the pose; M: the segment heading; None for R
    from_heading: int | None  # R only
    to_heading: int | None  # R only
    duration_s: float | None  # None for S: it lasts as long as the vehicle waits
    next: tuple[str, ...]  # sorted ids of the states that can follow


StateGraph = dict[str, State]  # sid -> State
Conflicts = dict[str, frozenset[str]]  # sid -> sids it collides with, never itself


def shortest_turn_deg(h1: int, h2: int) -> int:
    """The shorter way round from heading h1 to h2, as an unsigned angle in [0, 180]."""
    turn = abs(h1 - h2) % 360
    return min(turn, 360 - turn)


def geom_key(sid: str) -> str:
    """The key of the space a state takes: L,n,h takes that of S,n,h; the others are their own."""
    if sid.startswith("L,"):
        return "S" + sid[1:]
    return sid


def build_state_graph(world: World) -> StateGraph:
    """The S, M, R and L states of a world. Keys and every next tuple are in sorted order."""
    states: dict[str, State] = {}
    departures: dict[tuple[str, int], list[str]] = {}  # (node, heading) of an S -> ids that leave it

    for seg in world.segments.values():  # M for each segment, and the S at both its ends
        h = seg.heading_deg
        sid = f"M,{seg.u},{seg.v}"
        states[sid] = State(
            sid=sid, kind="M", nodes=(seg.u, seg.v), heading=h, from_heading=None, to_heading=None,
            duration_s=seg.length_m / seg.speed_mps, next=(_stop_id(seg.v, h),),
        )
        departures.setdefault((seg.v, h), [])
        departures.setdefault((seg.u, h), []).append(sid)

    stops = sorted(departures)
    headings: dict[str, list[int]] = {}  # node -> headings of its S
    for node, h in stops:
        headings.setdefault(node, []).append(h)

    for node, node_headings in headings.items():  # R for every ordered pair of distinct headings
        for h1, h2 in permutations(node_headings, 2):
            sid = f"R,{node},{h1},{h2}"
            states[sid] = State(
                sid=sid, kind="R", nodes=(node,), heading=None, from_heading=h1, to_heading=h2,
                duration_s=shortest_turn_deg(h1, h2) / 90 * world.rotation_s_per_90deg,
                next=(_stop_id(node, h2),),
            )
            departures[(node, h1)].append(sid)

    ports = set(world.port_nodes)
    for node, h in stops:  # L for each S at a port, back to the same S
        if node in ports:
            sid = f"L,{node},{h}"
            states[sid] = State(
                sid=sid, kind="L", nodes=(node,), heading=h, from_heading=None, to_heading=None,
                duration_s=world.dwell_s, next=(_stop_id(node, h),),
            )
            departures[(node, h)].append(sid)

    for (node, h), leaving in departures.items():  # S last: next = the M, R and L that leave it
        sid = _stop_id(node, h)
        states[sid] = State(
            sid=sid, kind="S", nodes=(node,), heading=h, from_heading=None, to_heading=None,
            duration_s=None, next=tuple(sorted(leaving)),
        )
    return {sid: states[sid] for sid in sorted(states)}


def _stop_id(node: str, heading: int) -> str:
    return f"S,{node},{heading}"
