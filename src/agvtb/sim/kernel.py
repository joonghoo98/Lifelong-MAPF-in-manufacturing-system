"""The event kernel: one heap ordered by (time, rank, seq), and a clock that moves in steps of dt.

Every event runs at its own time, whatever dt is (design D2): dt only says how far the clock moves before the
events up to it are run, and when on_tick sees the clock. At one time, rank MCS runs before rank AGV, as DAIM's
unified scheduler runs the MCS source before the AGV source (research §B7); within a rank, the order of
scheduling. Handlers get the event time and read it, never the clock.
"""
import heapq
from collections.abc import Callable

MCS = 0
AGV = 1
EPS = 1e-9


class Kernel:
    def __init__(self):
        self.now = 0.0
        self.stopped = False
        self._heap: list[tuple] = []
        self._seq = 0

    def schedule(self, t: float, rank: int, handler: Callable[..., None], *args) -> None:
        """Run handler(t, *args) at time t. ValueError if t is in the past."""
        if t < self.now - EPS:
            raise ValueError(f"cannot schedule at {t}, the time is already {self.now}")
        heapq.heappush(self._heap, (t, rank, self._seq, handler, args))
        self._seq += 1

    def stop(self) -> None:
        """End run_until after the event that calls this."""
        self.stopped = True

    def run_until(self, t_end: float, dt: float, on_tick: Callable[[float], None] | None = None) -> None:
        """Move the clock by dt up to t_end, running every event due by each tick in order."""
        if not dt > 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        clock = self.now
        while clock < t_end and not self.stopped:
            clock = min(clock + dt, t_end)
            while self._heap and self._heap[0][0] <= clock and not self.stopped:
                t, _rank, _seq, handler, args = heapq.heappop(self._heap)
                self.now = t
                handler(t, *args)
            if self.stopped:
                return
            self.now = clock
            if on_tick is not None:
                on_tick(clock)
