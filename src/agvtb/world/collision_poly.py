"""The polygon rule: two states collide when the areas they take touch or overlap."""
import math
from functools import lru_cache
from itertools import pairwise

import shapely
from shapely import affinity

from agvtb.world.model import World
from agvtb.world.states import Conflicts, State, StateGraph, shortest_turn_deg

_AXIS_COS_SIN = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))  # (cos, sin) at 0, 90, 180, 270 deg


def footprint_polygon(world: World, x: float, y: float, heading_deg: float) -> shapely.Polygon:
    """The vehicle rectangle centred on (x, y): length along heading_deg (CCW from +x), width across."""
    return shapely.Polygon(_corners(world.footprint.length_m, world.footprint.width_m, x, y, heading_deg))


def state_area(world: World, state: State) -> shapely.Polygon:
    """The area a state takes.

    S and L: the footprint at the node. M: the hull of the footprints at both ends, which is what
    a straight move sweeps. R: the area swept by the turn, see _turn_sweep.
    """
    length_m, width_m = world.footprint.length_m, world.footprint.width_m
    if state.kind in ("S", "L"):
        x, y = world.nodes[state.nodes[0]]
        return footprint_polygon(world, x, y, state.heading)
    if state.kind == "M":
        (x0, y0), (x1, y1) = (world.nodes[node] for node in state.nodes)
        ends = _corners(length_m, width_m, x0, y0, state.heading) + _corners(length_m, width_m, x1, y1, state.heading)
        return shapely.MultiPoint(ends).convex_hull
    if state.kind == "R":
        x, y = world.nodes[state.nodes[0]]
        return affinity.translate(_turn_sweep(length_m, width_m, state.from_heading, state.to_heading), x, y)
    raise ValueError(f"{state.sid}: unknown state kind {state.kind!r}")


def poly_conflicts(world: World, graph: StateGraph) -> Conflicts:
    """For every state of graph, the other states whose area touches or overlaps its own.

    Boundary contact counts. Every state is a key; the set is empty when nothing collides with it.
    """
    if not graph:  # to shapely an empty list is not an array of geometries
        return {}
    sids = sorted(graph)  # one fixed order for the areas, so tree indices map back to ids
    areas = [state_area(world, graph[sid]) for sid in sids]
    first, second = shapely.STRtree(areas).query(areas, predicate="intersects")
    colliding: dict[str, set[str]] = {sid: set() for sid in graph}
    for i, j in zip(first.tolist(), second.tolist()):
        if i != j:
            colliding[sids[i]].add(sids[j])
            colliding[sids[j]].add(sids[i])  # a hit in either direction counts: the sets stay symmetric
    return {sid: frozenset(colliding[sid]) for sid in graph}


@lru_cache(maxsize=None)  # one entry per heading pair of a map: 66 on songdo_grid_bi
def _turn_sweep(length_m: float, width_m: float, from_heading: int, to_heading: int) -> shapely.Polygon:
    """The area swept by a vehicle turning on the spot at the origin from from_heading to to_heading.

    The turn goes the short way in 1 degree steps, counter-clockwise when it is exactly 180.
    The area is the union of the hull of every two successive footprints.
    """
    steps = shortest_turn_deg(from_heading, to_heading)
    if steps == 0:
        raise ValueError(f"a turn needs two different headings, got {from_heading} and {to_heading}")
    direction = 1 if (to_heading - from_heading) % 360 <= 180 else -1
    footprints = [_corners(length_m, width_m, 0.0, 0.0, from_heading + direction * k) for k in range(steps + 1)]
    return shapely.union_all([shapely.MultiPoint(a + b).convex_hull for a, b in pairwise(footprints)])


def _corners(length_m: float, width_m: float, x: float, y: float, heading_deg: float) -> list[tuple[float, float]]:
    """Corners of the rectangle centred on (x, y), counter-clockwise from the front left."""
    cos_h, sin_h = _cos_sin(heading_deg)
    half_l, half_w = length_m / 2, width_m / 2
    return [
        (x + dx * cos_h - dy * sin_h, y + dx * sin_h + dy * cos_h)
        for dx, dy in ((half_l, half_w), (-half_l, half_w), (-half_l, -half_w), (half_l, -half_w))
    ]


def _cos_sin(angle_deg: float) -> tuple[float, float]:
    """cos and sin of an angle in degrees, exact at multiples of 90.

    math.cos(math.radians(90)) is 6e-17, not 0. That noise can open a gap between two axis-aligned
    rectangles that touch, and touching must count as a collision.
    """
    quarters, rest = divmod(angle_deg, 90)
    if rest == 0:
        return _AXIS_COS_SIN[int(quarters) % 4]
    angle = math.radians(angle_deg)
    return math.cos(angle), math.sin(angle)
