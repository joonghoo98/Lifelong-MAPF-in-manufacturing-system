"""The experiment runner (M5a): runs a list of conditions with run_sim, several at a time, and sums them up.

A condition is a dict {"name", "config", "vehicles", "rate", "seed", "warmup", "measure", "sidings", "fromto"}
(sidings and fromto may be None). Each run is its own process (python -B -m agvtb.tools.run_sim), writing
<out>/<name>.json and <out>/<name>.log; a run whose JSON is already there and readable is skipped, so a cut batch
goes on where it stopped. Only the main thread writes <out>/progress.txt. Exit codes are those of run_sim: 0 ok,
1 result not ok, 2 usage error, 3 gate refusal.

  python -m agvtb.tools.batch run CONDS.json --out DIR [--workers N]
  python -m agvtb.tools.batch sweep CONFIG --vehicles 4,8 --rates 1,2 --seeds 0,1 [--warmup S --measure S
      --sidings C --fromto CSV] --out CONDS.json
"""
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PARTIAL_S = 1800.0  # a job held this long is a partial deadlock (design §11), over the whole run as st24 counts
SUMMARY_COLUMNS = ("name", "ok", "passed", "t_end", "global", "partial", "wait_cycles", "findings", "plan_fail",
                   "job_max_s", "completed", "throughput_per_h", "cycle_s", "deliver_s", "utilization", "backlog_end",
                   "wall_s", "events_sha256")


def passed(result: dict) -> bool:
    """The pass rule of the DAIM comparisons (M4c, R15): ok, ran to its end, no job held 30 min or more in the whole
    run (warm-up included), no plan failure."""
    job_max = result["job_max_s"]
    return bool(result["ok"] and result["t_end"] == result["warmup_s"] + result["measure_s"]
                and job_max is not None and job_max < PARTIAL_S and result["stats"].get("plan_fail", 0) == 0)


def run_argv(cond: dict, out_json: Path) -> list[str]:
    """The run_sim arguments of a condition."""
    argv = [cond["config"], "--vehicles", str(cond["vehicles"]), "--rate", f"{cond['rate']:g}", "--seed",
            str(cond["seed"]), "--warmup", f"{cond['warmup']:g}", "--measure", f"{cond['measure']:g}"]
    if cond.get("sidings") is not None:
        argv += ["--sidings", cond["sidings"]]
    if cond.get("fromto") is not None:
        argv += ["--fromto", cond["fromto"]]
    return argv + ["--out", str(out_json)]


def _done(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return True


def run_batch(conds: list[dict], out_dir: Path, workers: int, python: str = sys.executable) -> list[dict]:
    """Run every condition not done yet, at most workers at a time; [{"name", "exit", "skipped"}] in the given order."""
    names = [c["name"] for c in conds]
    if len(set(names)) != len(names):
        raise ValueError(f"condition names repeat: {sorted({n for n in names if names.count(n) > 1})}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {c["name"]: {"name": c["name"], "exit": 0, "skipped": True} for c in conds if _done(out_dir / f"{c['name']}.json")}

    def one(cond: dict) -> int:
        with open(out_dir / f"{cond['name']}.log", "w", encoding="utf-8") as log:
            return subprocess.run([python, "-B", "-m", "agvtb.tools.run_sim", *run_argv(cond, out_dir / f"{cond['name']}.json")],
                                  stdout=log, stderr=subprocess.STDOUT).returncode

    todo = [c for c in conds if c["name"] not in results]
    with ThreadPoolExecutor(max(1, workers)) as pool, open(out_dir / "progress.txt", "a", encoding="utf-8") as progress:
        futures = {pool.submit(one, c): c["name"] for c in todo}
        for future in as_completed(futures):
            name = futures[future]
            results[name] = {"name": name, "exit": future.result(), "skipped": False}
            progress.write(f"done {name} exit {results[name]['exit']} {time.strftime('%H:%M:%S')}\n")
            progress.flush()
    return [results[n] for n in names]


def summary_rows(out_dir: Path) -> list[dict]:
    """One row per readable result JSON in out_dir, sorted by name."""
    rows = []
    for path in sorted(Path(out_dir).glob("*.json")):
        if not _done(path):
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        k, d = r["kpi"], r["deadlock"]
        rows.append({"name": path.stem, "ok": r["ok"], "passed": passed(r), "t_end": r["t_end"], "global": d["global"],
                     "partial": d["partial_jobs"] + d["held_at_end"], "wait_cycles": len(d["wait_cycles"]),
                     "findings": sum(len(x) for x in r["checks"].values()), "plan_fail": r["stats"].get("plan_fail", 0),
                     "job_max_s": r["job_max_s"], "completed": k["completed"], "throughput_per_h": k["throughput_per_h"],
                     "cycle_s": k["cycle_s"], "deliver_s": k["deliver_s"], "utilization": k["utilization"],
                     "backlog_end": k["backlog_end"], "wall_s": r["wall_s"], "events_sha256": r["events_sha256"]})
    return rows


def write_summary(out_dir: Path) -> Path:
    """summary.csv of out_dir (columns SUMMARY_COLUMNS)."""
    path = Path(out_dir) / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows(out_dir))
    return path


def sweep(config: str, vehicles: list[int], rates: list[float], seeds: list[int], warmup: float, measure: float,
          sidings: str | None, fromto: str | None) -> list[dict]:
    """Every combination of vehicles x rates x seeds on one map, named <config stem>_n<v>_l<rate>_s<seed>."""
    stem = Path(config).stem
    return [{"name": f"{stem}_n{v}_l{rate:g}_s{seed}", "config": config, "vehicles": v, "rate": rate, "seed": seed,
             "warmup": warmup, "measure": measure, "sidings": sidings, "fromto": fromto}
            for v, rate, seed in itertools.product(vehicles, rates, seeds)]


def _numbers(kind):
    return lambda text: [kind(x) for x in text.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agvtb.tools.batch", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="run a conditions file")
    p_run.add_argument("conds", help="JSON list of conditions")
    p_run.add_argument("--out", required=True, help="folder for the results")
    p_run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p_sw = sub.add_parser("sweep", help="write the conditions of a vehicles x rates x seeds sweep")
    p_sw.add_argument("config")
    p_sw.add_argument("--vehicles", type=_numbers(int), required=True)
    p_sw.add_argument("--rates", type=_numbers(float), required=True)
    p_sw.add_argument("--seeds", type=_numbers(int), required=True)
    p_sw.add_argument("--warmup", type=float, default=3600.0)
    p_sw.add_argument("--measure", type=float, default=14400.0)
    p_sw.add_argument("--sidings")
    p_sw.add_argument("--fromto")
    p_sw.add_argument("--out", required=True, help="JSON file for the conditions")
    args = parser.parse_args(argv)
    if args.cmd == "sweep":
        conds = sweep(args.config, args.vehicles, args.rates, args.seeds, args.warmup, args.measure, args.sidings,
                      args.fromto)
        Path(args.out).write_text(json.dumps(conds, indent=1), encoding="utf-8")
        print(f"{len(conds)} conditions: {args.out}")
        return 0
    conds = json.loads(Path(args.conds).read_text(encoding="utf-8"))
    out = run_batch(conds, Path(args.out), args.workers)
    path = write_summary(Path(args.out))
    bad = [r for r in out if r["exit"] != 0]
    print(f"{len(out)} conditions, {sum(r['skipped'] for r in out)} skipped, {len(bad)} not ok "
          f"{[(r['name'], r['exit']) for r in bad]}; summary: {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
