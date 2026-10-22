"""DAIM's st24 gate records (A1, tail mode): the 30 conditions to run, and our results side by side (M5a).

The records are DAIM's 2026-10-06 runs (design §11, D12), read only: st24_summary.csv (a BOM, one row per
condition) and st24_<condition>.json. A condition is named <engine>_s<seed>_l<lambda>: 15 AMR, warm-up 3600 s,
86400 s measured, the map's recorded sidings and from-to table, lambda jobs per minute. DAIM's rule_stats is a
collections.Counter, so a missing key means 0. The pass rule is agvtb.tools.batch.passed.

  python -m agvtb.tools.st24 conditions ST24_DIR --out CONDS.json
  python -m agvtb.tools.st24 compare ST24_DIR RESULTS_DIR --out TABLE.md
"""
import argparse
import csv
import json
import re
from pathlib import Path

from agvtb.tools.batch import passed
from agvtb.world.config import load_map_config

_NAME = re.compile(r"([a-z]+)_s(\d+)_l(\d+(?:\.\d+)?)")
VEHICLES, WARMUP_S, MEASURE_S = 15, 3600.0, 86400.0


def parse_condition(name: str) -> tuple[str, int, float]:
    """("travelbr", 3, 2.5) from "travelbr_s3_l2.5"; ValueError for any other form."""
    m = _NAME.fullmatch(name)
    if m is None:
        raise ValueError(f"not an st24 condition name: {name!r}")
    return m.group(1), int(m.group(2)), float(m.group(3))


def _summary(st24_dir: Path) -> dict[str, dict]:
    with open(Path(st24_dir) / "st24_summary.csv", encoding="utf-8-sig", newline="") as fh:
        return {row["condition"]: row for row in csv.DictReader(fh)}


def st24_conditions(st24_dir: Path, configs_dir: Path = Path("configs/maps/daim")) -> list[dict]:
    """The conditions of the summary, in its order, as agvtb.tools.batch conditions."""
    out = []
    for name in _summary(st24_dir):
        engine, seed, lam = parse_condition(name)
        config = Path(configs_dir) / f"grid_songdo_{engine}.json"
        map_path = load_map_config(config).map_path
        out.append({"name": name, "config": str(config), "vehicles": VEHICLES, "rate": lam, "seed": seed,
                    "warmup": WARMUP_S, "measure": MEASURE_S, "sidings": "recorded",
                    "fromto": str(map_path.with_name(f"{map_path.stem}_fromto_AGV.csv"))})
    return out


def daim_record(st24_dir: Path, name: str, summary_row: dict) -> dict:
    """DAIM's values of one condition; asserts they are the window values the summary CSV shows."""
    d = json.loads((Path(st24_dir) / f"st24_{name}.json").read_text(encoding="utf-8"))
    v, r = d["verdict"], d["result"]
    win, rs, kall = r["window"], r["rule_stats"] or {}, r["kpi_all"]
    assert abs(float(summary_row["TH_per_h"]) - win["throughput_min"] * 60) < 1e-9, name
    assert abs(float(summary_row["CT_s"]) - win["avg_cycle_s"]) < 1e-9, name
    assert abs(float(summary_row["util"]) - win["utilization_agv"]) < 1e-9, name
    assert r["env"]["AGV_SAFE_TAIL_VALIDATE"] == "1", name  # the validator was on: its zero counts mean something

    def st(key):
        return rs.get(f"safe_tail_{key}", 0)

    return {"t_end": r["t_end"], "frozen": r.get("frozen_at_s") or ("stopped early" if r.get("stopped_early") else None),
            "partial": v["jobs_over_30min"] + v["held_over_30min"], "job_max": v["job_assign_to_done_max_s"],
            "cycles": st("exec_cycles"), "event_cycles": st("event_cycles"), "findings": st("validator_violations"),
            "plan_fail": st("fallback_fail"), "thr_h": win["throughput_min"] * 60, "cycle": win["avg_cycle_s"],
            "deliver": win["avg_deliver_s"], "util": win["utilization_agv"], "completed": win["completed"],
            "created": kall["completed"] + kall["waiting_loads"] + kall["active_loads"], "wall": v["wall_s"]}


def our_record(result: dict, name: str) -> dict:
    """Our values of one condition; asserts the result is the run of that condition."""
    engine, seed, lam = parse_condition(name)
    want = {"vehicles": VEHICLES, "warmup_s": WARMUP_S, "measure_s": MEASURE_S, "seed": seed, "rate_per_min": lam,
            "sidings_choice": "recorded"}
    got = {key: result[key] for key in want}
    assert got == want, (name, got)
    k, d, s = result["kpi"], result["deadlock"], result["stats"]
    return {"pass": passed(result), "t_end": result["t_end"], "frozen": d["frozen_at"] if d["global"] else None,
            "partial": d["partial_jobs"] + d["held_at_end"], "job_max": result["job_max_s"],
            "cycles": len(d["wait_cycles"]), "event_cycles": s.get("event_cycles", 0),
            "findings": sum(len(x) for x in result["checks"].values()), "plan_fail": s.get("plan_fail", 0),
            "thr_h": k["throughput_per_h"], "cycle": k["cycle_s"], "deliver": k["deliver_s"], "util": k["utilization"],
            "completed": k["completed"], "created": result["jobs_created"], "wall": result["wall_s"],
            "sha": result["events_sha256"]}


def _f(value, fmt="{:.1f}"):
    return "-" if value is None else fmt.format(value)


def compare_table(st24_dir: Path, results_dir: Path) -> str:
    """Markdown: "pass k/n" over the conditions that have a result, then one row per condition."""
    head = ("| condition | pass | end s | frozen | partial | longest job s ours/DAIM | cycles | findings ours/DAIM | "
            "plan fail | throughput/h ours/DAIM | cycle s ours/DAIM | deliver s ours/DAIM | utilization ours/DAIM | sha |")
    rows, n_run, n_pass = [head, "|" + "---|" * 14], 0, 0
    for name, row in _summary(st24_dir).items():
        dm = daim_record(st24_dir, name, row)
        path = Path(results_dir) / f"{name}.json"
        if not path.exists():
            rows.append(f"| {name} | not run | | | | -/{_f(dm['job_max'])} | | -/{dm['findings']} | | "
                        f"-/{_f(dm['thr_h'])} | -/{_f(dm['cycle'])} | -/{_f(dm['deliver'])} | -/{_f(dm['util'], '{:.3f}')} | |")
            continue
        o = our_record(json.loads(path.read_text(encoding="utf-8")), name)
        n_run, n_pass = n_run + 1, n_pass + o["pass"]
        rows.append(f"| {name} | {'yes' if o['pass'] else '**NO**'} | {o['t_end']:.0f} | {o['frozen'] or '-'} | "
                    f"{o['partial']} | {_f(o['job_max'])}/{_f(dm['job_max'])} | {o['cycles']}+{o['event_cycles']} | "
                    f"{o['findings']}/{dm['findings']} | {o['plan_fail']} | {_f(o['thr_h'])}/{_f(dm['thr_h'])} | "
                    f"{_f(o['cycle'])}/{_f(dm['cycle'])} | {_f(o['deliver'])}/{_f(dm['deliver'])} | "
                    f"{_f(o['util'], '{:.3f}')}/{_f(dm['util'], '{:.3f}')} | `{o['sha'][:12]}` |")
    return "\n".join([f"pass {n_pass}/{n_run}", ""] + rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agvtb.tools.st24", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_c = sub.add_parser("conditions", help="write the conditions of the st24 summary")
    p_c.add_argument("st24_dir")
    p_c.add_argument("--configs", default="configs/maps/daim")
    p_c.add_argument("--out", required=True)
    p_t = sub.add_parser("compare", help="write the table of our results against the records")
    p_t.add_argument("st24_dir")
    p_t.add_argument("results_dir")
    p_t.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "conditions":
        conds = st24_conditions(Path(args.st24_dir), Path(args.configs))
        Path(args.out).write_text(json.dumps(conds, indent=1), encoding="utf-8")
        print(f"{len(conds)} conditions: {args.out}")
    else:
        table = compare_table(Path(args.st24_dir), Path(args.results_dir))
        Path(args.out).write_text(table, encoding="utf-8")
        print(f"{table.splitlines()[0]}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
