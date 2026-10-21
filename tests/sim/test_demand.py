import random

import pytest

from agvtb.sim.demand import Demand, normalize, read_fromto, uniform_fromto


def test_read_fromto_keeps_port_pairs_only(tmp_path):  # research §E, DAIM vis_mcs_unified.py:207-243
    path = tmp_path / "od.csv"
    path.write_text(",P1,P2,X\nP1,0,0.01,0.02\nP2,0.03,,0\nQ,0.5,0.5,0.5\n", encoding="utf-8")
    assert read_fromto(path, ["P1", "P2"]) == {("P1", "P2"): 0.01, ("P2", "P1"): 0.03}


def test_uniform_fromto_sums_to_one_job_a_minute():  # design §7
    od = uniform_fromto(["c", "a", "b"])
    assert sorted(od) == [("a", "b"), ("a", "c"), ("b", "a"), ("b", "c"), ("c", "a"), ("c", "b")]
    assert sum(od.values()) == pytest.approx(1 / 60) and len(set(od.values())) == 1


def test_arrivals_draw_destination_then_gap_from_one_rng():  # research §E order
    d = Demand(uniform_fromto(["a", "b", "c"]), 2.0, random.Random(7))
    assert d.rates() == pytest.approx({"a": 1 / 90, "b": 1 / 90, "c": 1 / 90})
    ref = random.Random(7)
    assert d.first_arrivals() == [(ref.expovariate(1 / 90), p) for p in ("a", "b", "c")]
    dst = ref.choices(["b", "c"], weights=[1 / 360, 1 / 360])[0]
    assert d.on_arrival(5.0, "a") == (dst, 5.0 + ref.expovariate(1 / 90))
    with pytest.raises(ValueError):
        Demand({}, -1.0, random.Random(0))


def test_normalize_makes_a_table_one_job_a_minute():  # design §7
    od = normalize({("P1", "P2"): 0.01, ("P2", "P1"): 0.03})
    assert sum(od.values()) == pytest.approx(1 / 60) and od[("P2", "P1")] == pytest.approx(3 * od[("P1", "P2")])
