"""The M3 gate: planning starts only on a siding set that passes it.

User decisions of 2026-10-21 (MAPF/docs/PLAN.md §5, report R07 §9):
- G1: planning starts only on the sidings that pass the gate; otherwise it does not start and the violations are
  reported. Fewer usable sidings than vehicles is a violation too.
- G2: the gate is condition M plus a round trip from every siding to every node while every other siding holds a
  vehicle (R07 Part 1). The paper's condition (R07 Part 4) is reported alongside and not judged.
"""
from collections.abc import Iterable
from dataclasses import dataclass

from agvtb.sidings.condition_m import ConditionMReport, check_condition_m
from agvtb.sidings.full_reach import paper_outside, round_trip_misses
from agvtb.world.states import Conflicts, StateGraph


@dataclass(frozen=True)
class GateReport:
    """The result of check_gate. paper_outside is reported only: it is not part of ok or violations."""

    condition_m: ConditionMReport
    misses: dict[str, tuple[str, ...]]  # usable siding -> the nodes it has no round trip to (sorted)
    paper_outside: tuple[str, ...]  # the nodes outside the paper's condition (sorted)

    @property
    def sidings(self) -> tuple[str, ...]:
        """The usable sidings (condition M's), sorted."""
        return self.condition_m.sidings

    @property
    def n_vehicles(self) -> int:
        return self.condition_m.n_vehicles

    @property
    def violations(self) -> tuple[str, ...]:
        """Condition M's violations, then too few sidings, then each siding (sorted) that misses nodes."""
        found = list(self.condition_m.violations)
        if self.condition_m.fatal:
            found.append(f"only {len(self.sidings)} usable sidings for {self.n_vehicles} vehicles")
        found += [f"siding {s}: no round trip to nodes {list(missed)}" for s, missed in self.misses.items() if missed]
        return tuple(found)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> dict:
        """The report as plain data."""
        return {
            "ok": self.ok,
            "violations": list(self.violations),
            "condition_m": self.condition_m.summary(),
            "misses": {s: list(m) for s, m in self.misses.items() if m},
            "paper_outside": list(self.paper_outside),
        }


class GateError(RuntimeError):
    """The gate refused to start planning; report holds why."""

    def __init__(self, report: GateReport):
        super().__init__("gate refused: " + "; ".join(report.violations))
        self.report = report


def check_gate(
    graph: StateGraph, conflicts: Conflicts, ports: Iterable[str], candidates: Iterable[str], n_vehicles: int
) -> GateReport:
    """Condition M for n_vehicles, then the round trips and the paper's condition on its usable sidings.

    Raises ValueError if n_vehicles < 1.
    """
    if n_vehicles < 1:
        raise ValueError(f"n_vehicles must be >= 1, got {n_vehicles}")
    ports = list(ports)
    condition_m = check_condition_m(graph, conflicts, ports, candidates, n_vehicles)
    usable = condition_m.sidings
    return GateReport(
        condition_m,
        round_trip_misses(graph, conflicts, usable),
        paper_outside(graph, conflicts, ports, usable),
    )
