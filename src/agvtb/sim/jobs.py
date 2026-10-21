"""Jobs and their stages on the MCS clock, and NearestPair assignment (DAIM mcs/mcs_engine.py, research §D).

A job is created at its source port, assigned to an idle vehicle, and moves through its stages on MCS events:
the vehicle's arrival at the stop before the load starts the load dwell, DWELL_DONE at arrival + dwell ends it,
and the same for the unload (user decision S2: two clocks, as DAIM). The job is complete when the unload dwell
ends; the vehicle is then idle again, even while it still drives to its siding. Every event has rank MCS.
"""
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from agvtb.sim.kernel import MCS, Kernel


@dataclass
class Job:
    jid: int
    src: str
    dst: str
    t_created: float
    vehicle: int | None = None
    t_assigned: float | None = None
    t_pickup_arr: float | None = None
    t_pickup_end: float | None = None
    t_deliver_arr: float | None = None
    t_completed: float | None = None


def nearest_pair(jobs: Sequence[Job], idle: Sequence[int], dist: Callable[[int, str], float],
                 cap: int = 400) -> list[tuple[int, Job]]:
    """DAIM's NearestPair (policies/dispatch_policies.py:263-296): the cap oldest jobs x the idle vehicles, by
    distance, greedy one to one; ties keep the older job, then the vehicle earlier in idle. INF pairs are left out."""
    oldest = sorted(jobs, key=lambda j: j.t_created)[:cap]
    pairs = []
    for job in oldest:
        for vid in idle:
            d = dist(vid, job.src)
            if d != float("inf"):
                pairs.append((d, vid, job))
    pairs.sort(key=lambda p: p[0])
    used_v: set[int] = set()
    used_j: set[int] = set()
    out = []
    for _d, vid, job in pairs:
        if vid not in used_v and job.jid not in used_j:
            used_v.add(vid)
            used_j.add(job.jid)
            out.append((vid, job))
    return out


class Mcs:
    def __init__(self, kernel: Kernel, dwell_s: float, vehicles: Iterable[int], dist: Callable[[int, str], float],
                 dispatch: Callable[[int, Job, str, float], None],
                 on_complete: Callable[[Job], None] = lambda job: None):
        self.kernel = kernel
        self.dwell_s = dwell_s
        self.dist = dist
        self.dispatch = dispatch
        self.on_complete = on_complete
        self.phase = {vid: "IDLE" for vid in sorted(vehicles)}
        self.held: dict[int, Job] = {}
        self.jobs: list[Job] = []
        self.stray_arrivals = 0
        self._waiting: dict[str, list[Job]] = {}
        self._busy_since: dict[int, float] = {}
        self._busy_total = {vid: 0.0 for vid in self.phase}

    def waiting(self) -> list[Job]:
        """The jobs not yet assigned: source ports in sorted order, each in creation order."""
        return [j for port in sorted(self._waiting) for j in self._waiting[port]]

    def busy_s(self, vid: int, t: float) -> float:
        """Time vehicle vid spent between an assignment and that job's completion, up to t."""
        since = self._busy_since.get(vid)
        return self._busy_total[vid] + (t - since if since is not None else 0.0)

    def create(self, t: float, src: str, dst: str) -> Job:
        job = Job(len(self.jobs), src, dst, t)
        self.jobs.append(job)
        self._waiting.setdefault(src, []).append(job)
        self.try_assign(t)
        return job

    def try_assign(self, t: float) -> None:
        idle = [vid for vid, ph in self.phase.items() if ph == "IDLE"]
        jobs = self.waiting()
        if not idle or not jobs:
            return
        for vid, job in nearest_pair(jobs, idle, self.dist):
            self._waiting[job.src].remove(job)
            job.vehicle, job.t_assigned = vid, t
            self.phase[vid] = "TO_PICKUP"
            self.held[vid] = job
            self._busy_since[vid] = t
            self.dispatch(vid, job, job.src, t)

    def vehicle_arrived(self, t: float, vid: int) -> None:
        """The vehicle reached the stop before the load (TO_PICKUP) or the unload (TO_DELIVERY)."""
        job = self.held.get(vid)
        phase = self.phase[vid]
        if job is None or phase not in ("TO_PICKUP", "TO_DELIVERY"):
            self.stray_arrivals += 1
            return
        if phase == "TO_PICKUP":
            job.t_pickup_arr = t
            self.phase[vid] = "LOADING"
            self.kernel.schedule(t + self.dwell_s, MCS, self._dwell_done, vid)
            self.dispatch(vid, job, job.dst, t)
        else:
            job.t_deliver_arr = t
            self.phase[vid] = "UNLOADING"
            self.kernel.schedule(t + self.dwell_s, MCS, self._dwell_done, vid)

    def _dwell_done(self, t: float, vid: int) -> None:
        job = self.held[vid]
        if self.phase[vid] == "LOADING":
            job.t_pickup_end = t
            self.phase[vid] = "TO_DELIVERY"
            return
        job.t_completed = t
        self._busy_total[vid] += t - self._busy_since.pop(vid)
        self.phase[vid] = "IDLE"
        del self.held[vid]
        self.on_complete(job)
        self.kernel.schedule(t, MCS, self._try_assign_event)

    def _try_assign_event(self, t: float) -> None:
        self.try_assign(t)
