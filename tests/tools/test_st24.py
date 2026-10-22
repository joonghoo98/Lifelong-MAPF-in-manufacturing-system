import json
import os
from pathlib import Path

import pytest

from agvtb.tools.st24 import compare_table, parse_condition, st24_conditions
from helpers import CONFIGS


def fake_st24(root, names, th_per_h=90.0):
    """A folder shaped like DAIM's st24 records: summary CSV (with a BOM) and one result JSON per condition."""
    st = root / "st24"
    st.mkdir()
    window = {"window_s": 86400.0, "completed": 2160, "throughput_min": 1.5, "arrival_min": 1.5, "avg_cycle_s": 120.0,
              "avg_wait_s": 0.0, "avg_retrieve_s": 50.0, "avg_deliver_s": 70.0, "utilization_agv": 0.3, "backlog_end": 2}
    lines = ["condition,TH_per_h,CT_s,util"] + [f"{n},{th_per_h!r},120.0,0.3" for n in names]
    (st / "st24_summary.csv").write_text("\ufeff" + "\n".join(lines) + "\n", encoding="utf-8")
    for n in names:
        rec = {"verdict": {"jobs_over_30min": 0, "held_over_30min": 0, "job_assign_to_done_max_s": 300.0, "wall_s": 1.0},
               "result": {"t_end": 90000.0, "frozen_at_s": None, "stopped_early": False, "window": window,
                          "kpi_all": {"completed": 2200, "waiting_loads": 0, "active_loads": 3},
                          "rule_stats": {"safe_tail_sidings": 192, "safe_tail_map": {"aisle_size": 729}},
                          "env": {"AGV_SAFE_TAIL_VALIDATE": "1"}, "agv_flags": {}}}
        (st / f"st24_{n}.json").write_text(json.dumps(rec), encoding="utf-8")
    return st


def fake_result(seed, lam):
    """The fewest keys of one of our result JSONs that passes."""
    return {"ok": True, "t_end": 90000.0, "warmup_s": 3600.0, "measure_s": 86400.0, "vehicles": 15, "seed": seed,
            "rate_per_min": lam, "sidings_choice": "recorded", "config": "configs/maps/daim/grid_songdo_dflpbr.json",
            "fromto": "x/grid_songdo_dflpbr_fromto_AGV.csv", "job_max_s": 250.0, "jobs_created": 2190, "wall_s": 5.0,
            "events_sha256": "ab" * 32,
            "kpi": {"completed": 2150, "throughput_per_h": 89.6, "arrivals_per_min": 1.49, "cycle_s": 118.0,
                    "wait_s": 0.0, "retrieve_s": 49.0, "deliver_s": 69.0, "utilization": 0.29, "backlog_end": 3},
            "deadlock": {"global": False, "frozen_at": None, "partial_jobs": 0, "held_at_end": 0, "wait_cycles": []},
            "checks": {}, "stats": {"commits": 2190}, "gate": {"condition_m": {"sidings": 192, "aisle_size": 729}}}


def test_condition_names_round_trip():  # Review Focus 3
    assert parse_condition("travelbr_s3_l2.5") == ("travelbr", 3, 2.5)
    assert parse_condition("greedybr_s1_l3") == ("greedybr", 1, 3.0)
    with pytest.raises(ValueError):
        parse_condition("travelbr_s3")


def test_st24_conditions_follow_the_summary(tmp_path):
    st = fake_st24(tmp_path, ["dflpbr_s5_l1.5", "greedybr_s1_l3"])
    conds = st24_conditions(st, Path(os.path.relpath(CONFIGS / "daim")))  # relative, as on the command line
    assert [c["name"] for c in conds] == ["dflpbr_s5_l1.5", "greedybr_s1_l3"]
    c = conds[1]
    assert (c["vehicles"], c["rate"], c["seed"], c["warmup"], c["measure"], c["sidings"]) == (15, 3.0, 1, 3600.0, 86400.0, "recorded")
    assert c["config"].endswith("grid_songdo_greedybr.json") and c["fromto"].endswith("grid_songdo_greedybr_fromto_AGV.csv")
    assert Path(c["config"]).is_absolute() and Path(c["fromto"]).is_absolute()  # final review I4: any folder works


def test_the_self_check_catches_a_wrong_daim_field(tmp_path):  # Review Focus (M4c 1)
    st = fake_st24(tmp_path, ["rand_s2_l2.5"], th_per_h=1.0)  # the CSV disagrees with result.window
    with pytest.raises(AssertionError):
        compare_table(st, tmp_path / "none")


def test_the_table_counts_passes_and_marks_missing_runs(tmp_path):
    st = fake_st24(tmp_path, ["dflpbr_s5_l1.5", "rand_s2_l2.5"])
    res = tmp_path / "res"
    res.mkdir()
    (res / "dflpbr_s5_l1.5.json").write_text(json.dumps(fake_result(seed=5, lam=1.5)), encoding="utf-8")
    (res / "rand_s2_l2.5.json").write_text("{", encoding="utf-8")  # half written: not run (final review I3)
    table = compare_table(st, res)
    assert table.splitlines()[0] == "pass 1/1 (2 conditions, 1 not run)" and "not run" in table


@pytest.mark.parametrize("wrong", [{"seed": 4}, {"config": "configs/maps/daim/grid_songdo_rand.json"},
                                   {"fromto": "x/grid_songdo_rand_fromto_AGV.csv"}])
def test_a_result_of_another_condition_is_refused(tmp_path, wrong):  # the run is the condition it claims to be
    st = fake_st24(tmp_path, ["dflpbr_s5_l1.5"])
    res = tmp_path / "res"
    res.mkdir()
    (res / "dflpbr_s5_l1.5.json").write_text(json.dumps(fake_result(seed=5, lam=1.5) | wrong), encoding="utf-8")
    with pytest.raises(AssertionError):
        compare_table(st, res)
