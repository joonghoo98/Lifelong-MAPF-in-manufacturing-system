import json
import os
from pathlib import Path

import pytest

from agvtb.tools.a2gate import gate_conditions, gate_table
from helpers import CONFIGS
from test_st24 import fake_result

CUT_STATS = {"safe_tail_cut_try": 10, "safe_tail_cut_ok": 8, "safe_tail_cut_ok_port": 1, "safe_tail_cut_ok_transit": 7,
             "safe_tail_cut_tail_s": 96.0}


def fake_gate(root, names, m1=()):
    """A folder shaped like DAIM's A2 gate records: gate_summary.csv (a BOM, one row per condition and cut; the margin
    runs are not in it) and gate_<condition>_<cut0|cut1|cut1_m1>.json. DAIM's A2 takes the cycle 120.0 -> 116.4 s."""
    gate = root / "gate"
    gate.mkdir()
    lines = ["condition,cut,TH_per_h,CT_s,util"]
    runs = [(n, s, ct, cs) for n in names for s, ct, cs in (("cut0", 120.0, {}), ("cut1", 116.4, CUT_STATS))]
    runs += [(n, "cut1_m1", 116.4, CUT_STATS) for n in m1]
    for n, suffix, ct, cs in runs:
        if suffix != "cut1_m1":
            lines.append(f"{n},{suffix[-1]},90.0,{ct!r},0.3")
        window = {"window_s": 86400.0, "completed": 2160, "throughput_min": 1.5, "arrival_min": 1.5, "avg_cycle_s": ct,
                  "avg_wait_s": 0.0, "avg_retrieve_s": 50.0, "avg_deliver_s": 70.0, "utilization_agv": 0.3,
                  "backlog_end": 2}
        rec = {"verdict": {"jobs_over_30min": 0, "held_over_30min": 0, "job_assign_to_done_max_s": 300.0, "wall_s": 1.0},
               "result": {"t_end": 90000.0, "frozen_at_s": None, "stopped_early": False, "window": window,
                          "kpi_all": {"completed": 2200, "waiting_loads": 0, "active_loads": 3},
                          "rule_stats": {"safe_tail_sidings": 192, "safe_tail_map": {"aisle_size": 729},
                                         "safe_tail_tasks_planned": 100} | cs,
                          "env": {"AGV_SAFE_TAIL_VALIDATE": "1"}, "agv_flags": {}}}
        (gate / f"gate_{n}_{suffix}.json").write_text(json.dumps(rec), encoding="utf-8")
    (gate / "gate_summary.csv").write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    return gate


def test_gate_conditions_pair_each_condition_off_and_on_and_add_the_margin_runs(tmp_path):
    gate = fake_gate(tmp_path, ["dflpbr_s5_l1.5", "rand_s2_l2.5"], m1=["rand_s2_l2.5"])
    conds = gate_conditions(gate, Path(os.path.relpath(CONFIGS / "daim")))
    assert [c["name"] for c in conds] == ["dflpbr_s5_l1.5_cut0", "dflpbr_s5_l1.5_cut1", "rand_s2_l2.5_cut0",
                                          "rand_s2_l2.5_cut1", "rand_s2_l2.5_cut1_m1"]
    assert [(c["tail_cut"], c["cut_margin"]) for c in conds] == [(False, 0.0), (True, 0.0), (False, 0.0), (True, 0.0),
                                                                 (True, 1.0)]
    assert (conds[1]["seed"], conds[1]["rate"]) == (5, 1.5) and Path(conds[1]["config"]).is_absolute()


def test_the_gate_table_puts_both_sides_a2_effect_together(tmp_path):
    gate = fake_gate(tmp_path, ["dflpbr_s5_l1.5"])  # DAIM: CT 120.0 -> 116.4, try/ok 10/8 (1·7), tail 96 s, tasks 100
    res = tmp_path / "res"
    res.mkdir()
    base = fake_result(seed=5, lam=1.5)
    off = base | {"tail_cut": False, "cut_margin": 0.0}
    on = base | {"tail_cut": True, "cut_margin": 0.0, "kpi": base["kpi"] | {"cycle_s": 114.46},
                 "stats": base["stats"] | {"cut_try": 20, "cut_ok": 16, "cut_ok_port": 2, "cut_ok_transit": 14,
                                           "cut_tail_s": 160.0}}
    (res / "dflpbr_s5_l1.5_cut0.json").write_text(json.dumps(off), encoding="utf-8")
    (res / "dflpbr_s5_l1.5_cut1.json").write_text(json.dumps(on), encoding="utf-8")
    lines = gate_table(gate, res).splitlines()
    assert lines[0] == "pass ours off 1/1 on 1/1 m1 0/0 · DAIM off 1/1 on 1/1 m1 0/0 (1 conditions, 0 not run)"
    row = next(x for x in lines if x.startswith("| dflpbr_s5_l1.5 |"))
    assert "| -3.00/-3.00 | 20/16 (2·14) | 10/8 (1·7) | 10.0/12.0 | 0/0 |" in row


def test_a_result_with_the_wrong_switch_is_refused(tmp_path):
    gate = fake_gate(tmp_path, ["dflpbr_s5_l1.5"])
    res = tmp_path / "res"
    res.mkdir()
    (res / "dflpbr_s5_l1.5_cut1.json").write_text(json.dumps(fake_result(seed=5, lam=1.5) | {"tail_cut": False}),
                                                  encoding="utf-8")
    with pytest.raises(AssertionError):
        gate_table(gate, res)
