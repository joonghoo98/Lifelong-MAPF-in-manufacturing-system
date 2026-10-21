import pytest

from agvtb.sim.kernel import AGV, MCS, Kernel


def test_events_run_in_time_rank_seq_order():
    k, log = Kernel(), []
    k.schedule(2.0, AGV, lambda t: log.append(("agv", t)))
    k.schedule(2.0, MCS, lambda t: log.append(("mcs", t)))
    k.schedule(1.0, AGV, lambda t: log.append(("early", t)))
    k.schedule(2.0, MCS, lambda t: log.append(("mcs2", t)))
    k.run_until(5.0, dt=0.5)
    assert log == [("early", 1.0), ("mcs", 2.0), ("mcs2", 2.0), ("agv", 2.0)] and k.now == 5.0


def test_same_time_order_an_mcs_event_made_by_an_agv_event_runs_before_the_next_agv_event():  # Review Focus 1
    k, log = Kernel(), []

    def arrive(t):
        log.append(("arrive", t))
        k.schedule(t, MCS, lambda u: log.append(("notified", u)))

    k.schedule(3.0, AGV, arrive)
    k.schedule(3.0, AGV, lambda t: log.append(("other agv", t)))
    k.run_until(3.0, dt=10.0)
    assert log == [("arrive", 3.0), ("notified", 3.0), ("other agv", 3.0)]


@pytest.mark.parametrize("dt", [0.1, 0.5, 2.0, 7.0])
def test_dt_does_not_change_what_runs_when(dt):  # design D2
    k, log = Kernel(), []

    def tick(t, n):
        log.append((t, n, k.now))
        if n < 30:
            k.schedule(t + 0.7, MCS if n % 2 else 1, tick, n + 1)

    k.schedule(0.0, MCS, tick, 0)
    k.run_until(25.0, dt=dt)
    times = [0.0]
    for _ in range(30):
        times.append(times[-1] + 0.7)
    assert log == [(t, n, t) for n, t in enumerate(times) if t <= 25.0]


def test_stop_and_bad_inputs():
    k, log = Kernel(), []
    k.schedule(4.0, MCS, lambda t: (log.append(t), k.stop()))
    k.schedule(6.0, MCS, lambda t: log.append(t))
    k.run_until(10.0, dt=1.0)
    assert log == [4.0] and k.stopped and k.now == 4.0
    with pytest.raises(ValueError):
        Kernel().run_until(1.0, dt=0.0)
    late = Kernel()
    late.run_until(2.0, dt=1.0)
    with pytest.raises(ValueError):
        late.schedule(1.0, MCS, lambda t: None)
