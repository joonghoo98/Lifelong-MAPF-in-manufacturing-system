from agvtb.sim.jobs import Job, Mcs, nearest_pair
from agvtb.sim.kernel import MCS, Kernel


def job(jid, t, src):
    return Job(jid, src, "Z", t)


def test_nearest_pair_is_greedy_by_distance_then_oldest_job_then_lower_vehicle():  # DAIM dispatch_policies.py:263-296
    jobs = [job(0, 5.0, "A"), job(1, 1.0, "B"), job(2, 1.0, "A")]
    dist = {(7, "A"): 3.0, (7, "B"): 3.0, (9, "A"): 1.0, (9, "B"): float("inf")}
    got = nearest_pair(jobs, [7, 9], lambda v, s: dist[(v, s)])
    assert [(v, j.jid) for v, j in got] == [(9, 2), (7, 1)]
    assert nearest_pair(jobs, [9], lambda v, s: dist[(v, s)], cap=1) == []  # the oldest job only, from B: unreachable


def test_a_job_walks_its_stages_on_two_clocks():  # research §D1 (the st24 job log's times)
    k, events = Kernel(), []

    def dispatch(vid, jb, goal, t):
        events.append((vid, jb.jid, goal, t))
        travel = 61.0 if goal == jb.src else 10.0 + 103.0  # the fake vehicle: pickup leg, dwell + delivery leg
        k.schedule(t + travel, MCS, mcs.vehicle_arrived, vid)

    mcs = Mcs(k, 10.0, [0], lambda v, s: 1.0, dispatch)
    k.schedule(1.603, MCS, lambda t: mcs.create(t, "A", "B"))
    k.run_until(400.0, dt=1.0)
    j = mcs.jobs[0]
    times = (j.t_created, j.t_assigned, j.t_pickup_arr, j.t_pickup_end, j.t_deliver_arr, j.t_completed)
    assert [round(x, 3) for x in times] == [1.603, 1.603, 62.603, 72.603, 175.603, 185.603]
    assert events == [(0, 0, "A", 1.603), (0, 0, "B", 62.603)] and mcs.phase[0] == "IDLE"
    assert round(mcs.busy_s(0, 400.0), 3) == 184.0


def test_completion_assigns_the_next_waiting_job_at_once():
    k, got = Kernel(), []
    mcs = Mcs(k, 10.0, [0], lambda v, s: 1.0, lambda vid, jb, goal, t: got.append((jb.jid, goal, t)))
    k.schedule(1.0, MCS, lambda t: mcs.create(t, "A", "B"))
    k.schedule(2.0, MCS, lambda t: mcs.create(t, "A", "B"))
    k.schedule(3.0, MCS, mcs.vehicle_arrived, 0)
    k.schedule(20.0, MCS, mcs.vehicle_arrived, 0)
    k.run_until(100.0, dt=1.0)
    assert got == [(0, "A", 1.0), (0, "B", 3.0), (1, "A", 30.0)] and [j.jid for j in mcs.waiting()] == []
