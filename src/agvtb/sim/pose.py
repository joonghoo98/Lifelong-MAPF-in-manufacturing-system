"""Where a vehicle is at one time, and what it is doing: for drawing (the live viewer). Reads the executor only.

A moving vehicle sits on its segment at the distance its committed speed profile gives (a stopping profile until
the braking decision, the decided one after), so acceleration, braking and passing show. A rotating vehicle turns
at a constant rate the short way round, + when (to - from) % 360 <= 180 (DAIM env:741-768). Otherwise it stands on
the node of its current state, facing that state's heading (an R it waits at: its start heading).
"""
from dataclasses import dataclass

from agvtb.sim.executor import Executor
from agvtb.sim.motion import distance_at
from agvtb.world.model import World
from agvtb.world.states import StateGraph


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading_deg: float
    kind: str  # "move", "rotate", "blocked", "dwell" or "rest"


def pose_of(world: World, graph: StateGraph, executor: Executor, vid: int, t: float) -> Pose:
    v = executor.vehicles[vid]
    path = executor.tapg.paths[vid]
    sid = path[min(v.path_idx, len(path) - 1)][0]
    state = graph[sid]
    if v.state == "MOVING" and v.profile is not None and state.kind == "M":
        (x0, y0), (x1, y1) = world.nodes[state.nodes[0]], world.nodes[state.nodes[1]]
        length = v.move[2]
        f = distance_at(v.profile, t - v.t0) / length if length > 0 else 1.0
        return Pose(x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, float(state.heading), "move")
    x, y = world.nodes[state.nodes[0]]
    if v.state == "ROTATING" and state.kind == "R":
        h1, h2 = state.from_heading, state.to_heading
        diff = (h2 - h1) % 360
        signed = diff if diff <= 180 else diff - 360
        frac = min(1.0, max(0.0, (t - v.t0) / v.dur)) if v.dur > 0 else 1.0
        return Pose(x, y, (h1 + signed * frac) % 360, "rotate")
    heading = state.from_heading if state.kind == "R" else state.heading
    if v.dwelling:
        kind = "dwell"
    elif v.state == "WAITING":
        kind = "blocked"
    else:
        kind = "rest"
    return Pose(x, y, float(heading), kind)
