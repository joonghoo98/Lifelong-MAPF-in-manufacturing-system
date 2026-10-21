"""Time intervals of the planner: merging, the safe gaps between reservations, the gap holding a time.

The same expressions as DAIM (core/sipp_planner.py:86-109, core/safe_tail_planner.py:269-274), so that the
plans come out bit for bit the same.
"""
import math
from collections.abc import Iterable, Sequence

EPS: float = 1e-9
INF: float = math.inf


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """The intervals sorted, with those that overlap or touch joined."""
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def safe_intervals(
    blocked: Iterable[tuple[float, float]], lo: float = 0.0, hi: float = INF
) -> list[tuple[float, float]]:
    """The gaps in [lo, hi) between the blocked intervals."""
    result = []
    cur = lo
    for bs, be in merge_intervals(blocked):
        if bs > cur:
            result.append((cur, bs))
        cur = max(cur, be)
    if cur < hi:
        result.append((cur, hi))
    return result


def interval_at(intervals: Sequence[tuple[float, float]], t: float) -> tuple[float, float] | None:
    """The first interval with lo - EPS <= t < hi - EPS, or None."""
    for lo, hi in intervals:
        if lo - EPS <= t < hi - EPS:
            return (lo, hi)
    return None
