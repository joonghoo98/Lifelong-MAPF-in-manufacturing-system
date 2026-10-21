"""The reservation table: what the committed paths of the other vehicles block, and when.

DAIM's rule (core/sipp_planner.py:296-335): element i of a path, (state, t_i), lasts until the next element's
time (the last one forever). It reserves its geometric key over [t_i, t_end) and blocks each of its affects x
over [max(0, t_i - duration of x), t_end). An element that is not the last and lasts no time is skipped.
A state's affects are its conflicts less the L states (the L rule of the plan's Global Constraints): DAIM lists
no L as an affect, and an L blocks what its S blocks. Keys are geometric keys (an L takes its S's key).
"""
from collections.abc import Sequence

from agvtb.planning.intervals import INF
from agvtb.world.states import Conflicts, StateGraph, geom_key

Path = list[tuple[str, float]]


def affects(conflicts: Conflicts, sid: str) -> tuple[str, ...]:
    """The states sid blocks, sorted: the conflicts of its geometric key, less the L states."""
    return tuple(sorted(x for x in conflicts[geom_key(sid)] if not x.startswith("L,")))


def path_constraints(
    path: Sequence[tuple[str, float]], graph: StateGraph, conflicts: Conflicts
) -> list[tuple[str, float, float]]:
    """The reservations (key, lo, hi) of one path, in DAIM's order. Raises KeyError for a state not in graph."""
    out = []
    last = len(path) - 1
    for i, (sid, t_start) in enumerate(path):
        if sid not in graph:  # an error, not a gap to skip as DAIM does
            raise KeyError(sid)
        t_end = path[i + 1][1] if i < last else INF
        if t_start == t_end and i < last:
            continue
        out.append((geom_key(sid), t_start, t_end))
        for x in affects(conflicts, sid):
            cost = graph[x].duration_s or 0.0  # an S lasts no fixed time
            out.append((x, max(0.0, t_start - cost), t_end))
    return out


class ReservationTable:
    """The reservations of every vehicle's path; table(exclude) is what the planner of vehicle exclude sees."""

    def __init__(self, graph: StateGraph, conflicts: Conflicts):
        self.graph = graph
        self.conflicts = conflicts
        self._constraints: dict[int, list[tuple[str, float, float]]] = {}

    def set_path(self, vehicle: int, path: Sequence[tuple[str, float]]) -> None:
        """Replace the reservations of vehicle with those of path."""
        self._constraints[vehicle] = path_constraints(path, self.graph, self.conflicts)

    @property
    def vehicles(self) -> tuple[int, ...]:
        return tuple(sorted(self._constraints))

    def table(self, exclude: int | None = None) -> dict[str, list[tuple[float, float]]]:
        """key -> [(lo, hi), ...]: vehicles in id order, each one's reservations in path order; exclude left out."""
        table: dict[str, list[tuple[float, float]]] = {}
        for vehicle in self.vehicles:
            if vehicle == exclude:
                continue
            for key, lo, hi in self._constraints[vehicle]:
                table.setdefault(key, []).append((lo, hi))
        return table
