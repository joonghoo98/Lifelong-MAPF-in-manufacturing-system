"""The cell rule of a grid map: two states collide when the nodes they occupy overlap."""
import math

from agvtb.world.model import World
from agvtb.world.states import Conflicts, StateGraph

_TOLERANCE_M = 1e-9  # slack for float noise when the vehicle diagonal is compared with the pitch
_LATTICE_TOLERANCE_M = 1e-6  # absolute slack for matching segments and nodes to the lattice (rounded coordinates)
_LATTICE_HINT = "the cell rule needs a pure lattice map, so set collision: polygon for this map"


class GridRuleError(ValueError):
    """The map does not suit the cell rule: the vehicle diagonal is longer than the pitch, or the map is no lattice."""


def grid_pitch(world: World) -> float:
    """The grid spacing in metres: the shortest segment."""
    return min(seg.length_m for seg in world.segments.values())


def check_grid_rule(world: World) -> None:
    """Raise GridRuleError unless the cell rule holds for this world.

    The cell rule takes vehicles on different nodes as never touching. A vehicle diagonal no longer
    than the pitch is necessary for that: turns at adjacent nodes overlap when the diagonal is longer.
    It is sufficient only on a lattice, which _check_lattice checks.
    """
    diagonal = math.hypot(world.footprint.length_m, world.footprint.width_m)
    pitch = grid_pitch(world)
    if diagonal > pitch + _TOLERANCE_M:
        raise GridRuleError(
            f"{world.name}: vehicle diagonal {diagonal:.3f} m is longer than "
            f"the grid pitch {pitch:.3f} m"
        )
    _check_lattice(world, pitch)


def _check_lattice(world: World, pitch: float) -> None:
    """Raise GridRuleError unless the nodes and segments of world form one lattice of spacing pitch.

    Every segment is axis-aligned (judged by its dx and dy, not by its heading) and one pitch long;
    every node lies on the lattice anchored at the first segment's start node; no two nodes share
    a lattice point. All matches are within 1e-6 m.
    """
    for seg in world.segments.values():
        (x0, y0), (x1, y1) = world.nodes[seg.u], world.nodes[seg.v]
        dx, dy = x1 - x0, y1 - y0
        if min(abs(dx), abs(dy)) > _LATTICE_TOLERANCE_M:
            raise GridRuleError(
                f"{world.name}: segment {seg.u} -> {seg.v} is not axis-aligned "
                f"(dx {dx:.3f} m, dy {dy:.3f} m); {_LATTICE_HINT}"
            )
        if abs(seg.length_m - pitch) > _LATTICE_TOLERANCE_M:
            raise GridRuleError(
                f"{world.name}: segment {seg.u} -> {seg.v} is {seg.length_m:.3f} m long, "
                f"not the pitch {pitch:.3f} m; {_LATTICE_HINT}"
            )
    anchor = next(iter(world.segments.values())).u
    ax, ay = world.nodes[anchor]
    at_point: dict[tuple[int, int], str] = {}  # lattice point -> the first node found on it
    for node, (x, y) in world.nodes.items():
        col, row = (x - ax) / pitch, (y - ay) / pitch
        off_x, off_y = (col - round(col)) * pitch, (row - round(row)) * pitch  # metres from the nearest point
        if max(abs(off_x), abs(off_y)) > _LATTICE_TOLERANCE_M:
            raise GridRuleError(
                f"{world.name}: node {node} is off the {pitch:.3f} m lattice of node {anchor} "
                f"by ({off_x:.3f}, {off_y:.3f}) m; {_LATTICE_HINT}"
            )
        point = (round(col), round(row))
        if point in at_point:
            raise GridRuleError(
                f"{world.name}: nodes {at_point[point]} and {node} are at the same lattice point; {_LATTICE_HINT}"
            )
        at_point[point] = node


def grid_conflicts(world: World, graph: StateGraph) -> Conflicts:
    """For every state of graph, the other states that occupy one of its nodes.

    The rule needs only State.nodes: world is unused and only keeps the call the same for every
    collision mode. The vehicle size is not checked here; that is check_grid_rule's job.
    """
    at_node: dict[str, list[str]] = {}  # node -> ids of the states that occupy it
    for sid, state in graph.items():
        for node in state.nodes:
            at_node.setdefault(node, []).append(sid)
    conflicts: Conflicts = {}
    for sid, state in graph.items():
        sharing = frozenset().union(*(at_node[node] for node in state.nodes))
        conflicts[sid] = sharing - {sid}
    return conflicts
