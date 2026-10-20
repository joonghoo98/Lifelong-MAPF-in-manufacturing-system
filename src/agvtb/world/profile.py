"""Read a collision-v1 profile: the states and conflict sets of a map, as the profile gives them.

KAIROS_AMR ships these (D14). They are used as they are, once the profile is shown to be made for
the very map file in use and to agree with the world on every duration. L states are not in the
profile; they are added here.
"""
import json
from dataclasses import replace
from pathlib import Path

from agvtb.world.mapfile import MapFile
from agvtb.world.model import World
from agvtb.world.states import Conflicts, State, StateGraph, shortest_turn_deg

_SCHEMA = "collision-v1"
_TOLERANCE_S = 1e-6  # durations closer than this count as equal
_SHOWN_MISMATCHES = 3  # duration mismatches quoted in the error message


class ProfileMismatchError(ValueError):
    """The profile does not fit this map, this world or the contract, or is not readable collision-v1."""


def load_collision_profile(
    world: World, mf: MapFile, profile_path: Path
) -> tuple[StateGraph, Conflicts]:
    """The states and conflict sets of a profile, with an L state added for each S at a port.

    Raises ProfileMismatchError if the profile is not collision-v1, if its map_sha256 is not the
    SHA-256 of the map file mf, if it breaks the contract of a graph and its conflicts (see
    _check_contract), or if an M or R duration differs from the world's (see
    check_profile_durations): the graph must not contradict its own World. The keys of both results
    and every next tuple are in sorted order.
    """
    where = profile_path.name
    data = json.loads(profile_path.read_bytes())
    if data.get("schema") != _SCHEMA:
        raise ProfileMismatchError(
            f"{where}: schema must be {_SCHEMA!r}, got {data.get('schema')!r}"
        )
    if data.get("map_sha256") != mf.sha256:
        raise ProfileMismatchError(
            f"{where}: map_sha256 {data.get('map_sha256')} is not the sha256 of "
            f"{mf.path.name} ({mf.sha256})"
        )
    states: dict[str, State] = {}
    conflicts: dict[str, set[str]] = {}
    for sid, raw in data["states"].items():
        states[sid] = _read_state(where, sid, raw)
        conflicts[sid] = set(raw["conflicts"])
    _check_contract(where, world, states, conflicts)
    _add_load_states(world, states, conflicts)
    graph = {sid: states[sid] for sid in sorted(states)}
    problems = check_profile_durations(world, graph)
    if problems:
        shown = problems[:_SHOWN_MISMATCHES]
        raise ProfileMismatchError(
            f"{where}: {len(problems)} durations disagree with the world, first {len(shown)}: "
            + "; ".join(shown)
        )
    return graph, {sid: frozenset(conflicts[sid]) for sid in graph}


def check_profile_durations(world: World, graph: StateGraph) -> list[str]:
    """One line for each M or R state whose duration is more than 1e-6 s off the world's value.

    M: segment length / speed. R: shortest turn / 90 * rotation time. S and L are not checked: an S
    lasts as long as the vehicle waits and an L as long as the dwell time. An M without a segment
    in the world, and an M or R without a duration, are reported too. An empty list means that all
    durations agree.
    """
    problems = []
    for sid, st in graph.items():
        if st.kind == "M":
            segment = world.segments.get(st.nodes)
            if segment is None:
                problems.append(f"{sid}: the world has no segment {st.nodes[0]} -> {st.nodes[1]}")
                continue
            expected = segment.length_m / segment.speed_mps
        elif st.kind == "R":
            turn = shortest_turn_deg(st.from_heading, st.to_heading)
            expected = turn / 90 * world.rotation_s_per_90deg
        else:
            continue
        if st.duration_s is None:
            problems.append(f"{sid}: no duration, the world gives {expected:.9g} s")
        elif abs(st.duration_s - expected) > _TOLERANCE_S:
            problems.append(f"{sid}: duration {st.duration_s:.9g} s, the world gives {expected:.9g} s")
    return problems


def _check_contract(
    where: str, world: World, states: dict[str, State], conflicts: dict[str, set[str]]
) -> None:
    """Raise ProfileMismatchError unless the states and conflicts read from the file hold together.

    Every node a state names is in the world; every next and conflict id is a state of the profile;
    no state conflicts with itself (that is implicit); conflicts are symmetric. Run before the L
    states are added, as those are built on this contract.
    """
    for sid, st in states.items():
        for node in st.nodes:
            if node not in world.nodes:
                raise ProfileMismatchError(f"{where}: state {sid}: node {node!r} is not in the world")
        for following in st.next:
            if following not in states:
                raise ProfileMismatchError(
                    f"{where}: state {sid}: next {following!r} is not a state of the profile"
                )
        for other in sorted(conflicts[sid]):  # a set: sorted, so the first fault found is always the same
            if other not in states:
                raise ProfileMismatchError(
                    f"{where}: state {sid}: conflict {other!r} is not a state of the profile"
                )
        if sid in conflicts[sid]:
            raise ProfileMismatchError(f"{where}: state {sid}: conflicts with itself")
    for sid in states:
        for other in sorted(conflicts[sid]):
            if sid not in conflicts[other]:
                raise ProfileMismatchError(
                    f"{where}: conflicts are not symmetric: {sid} lists {other!r}, which does not list {sid}"
                )


def _read_state(where: str, sid: str, raw: dict) -> State:
    """One profile record as a State. S, M and R have the fields set as in build_state_graph."""
    kind = raw["type"]
    successors = tuple(sorted(raw["next"]))
    if kind == "S":
        return State(
            sid=sid, kind="S", nodes=(raw["node"],), heading=raw["heading"], from_heading=None,
            to_heading=None, duration_s=None, next=successors,
        )
    if kind == "M":
        return State(
            sid=sid, kind="M", nodes=(raw["from"], raw["to"]),
            heading=_heading_of_stop(where, sid, raw["next"][0], raw["to"]),
            from_heading=None, to_heading=None, duration_s=float(raw["duration"]), next=successors,
        )
    if kind == "R":
        return State(
            sid=sid, kind="R", nodes=(raw["node"],), heading=None, from_heading=raw["from_heading"],
            to_heading=raw["to_heading"], duration_s=float(raw["duration"]), next=successors,
        )
    raise ProfileMismatchError(f"{where}: state {sid}: unknown type {kind!r}")


def _heading_of_stop(where: str, sid: str, stop_id: str, node: str) -> int:
    """The heading h of the stop id "S,<node>,<h>" that an M leads to: it is the heading of the M."""
    prefix = f"S,{node},"
    if not stop_id.startswith(prefix):
        raise ProfileMismatchError(f"{where}: state {sid}: next {stop_id!r} is not a stop at {node}")
    return int(stop_id[len(prefix):])


def _add_load_states(
    world: World, states: dict[str, State], conflicts: dict[str, set[str]]
) -> None:
    """Add an L for each S at a port node, in place: the L states, the S next and the conflict sets.

    An L is built as in build_state_graph: the pose of its S, world.dwell_s long, back to that S.
    It takes the space of its S, so it conflicts with that S, with what that S conflicts with, and
    with the L of each of those that has one. Whatever conflicts with an L has that L in its set.
    """
    ports = set(world.port_nodes)
    load_of = {  # id of an S at a port -> id of its L
        sid: "L" + sid[1:] for sid, st in states.items() if st.kind == "S" and st.nodes[0] in ports
    }
    for sid, lid in load_of.items():
        stop = states[sid]
        states[lid] = State(
            sid=lid, kind="L", nodes=stop.nodes, heading=stop.heading, from_heading=None,
            to_heading=None, duration_s=world.dwell_s, next=(sid,),
        )
        states[sid] = replace(stop, next=tuple(sorted((*stop.next, lid))))
    for sid, lid in load_of.items():  # the S sets are still the profile's: no L is in them yet
        around = conflicts[sid]
        conflicts[lid] = {sid} | around | {load_of[other] for other in around if other in load_of}
    for lid in load_of.values():  # symmetric: the other side of each pair gets the L as well
        for other in conflicts[lid]:
            conflicts[other].add(lid)
