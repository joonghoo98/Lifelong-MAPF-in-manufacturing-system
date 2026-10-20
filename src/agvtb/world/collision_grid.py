"""The cell rule of a grid map: two states collide when the nodes they occupy overlap."""
import math

from agvtb.world.model import World
from agvtb.world.states import Conflicts, StateGraph

_TOLERANCE_M = 1e-9  # slack for float noise when the vehicle diagonal is compared with the pitch


class GridRuleError(ValueError):
    """The vehicle is too big for the cell rule: its diagonal is longer than the grid pitch."""


def grid_pitch(world: World) -> float:
    """The grid spacing in metres: the shortest segment."""
    return min(seg.length_m for seg in world.segments.values())


def check_grid_rule(world: World) -> None:
    """Raise GridRuleError if the vehicle diagonal is longer than the grid pitch.

    The cell rule takes vehicles on different nodes as never touching, which is true only if
    the diagonal of the vehicle is no longer than the node spacing.
    """
    diagonal = math.hypot(world.footprint.length_m, world.footprint.width_m)
    pitch = grid_pitch(world)
    if diagonal > pitch + _TOLERANCE_M:
        raise GridRuleError(
            f"{world.name}: vehicle diagonal {diagonal:.3f} m is longer than "
            f"the grid pitch {pitch:.3f} m"
        )


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
