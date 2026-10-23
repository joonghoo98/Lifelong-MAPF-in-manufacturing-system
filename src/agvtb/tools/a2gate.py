"""DAIM's A2 gate records (tail cut, 2026-10-17): the 66 runs to do, and our results side by side (M5c).

The records are DAIM's runs at commit 9c04deb (spec 2026-10-16 §8.3), read only: gate_summary.csv (a BOM, one row
per condition and cut; the margin runs are not in it) and gate_<condition>_<cut0|cut1|cut1_m1>.json. Every st24
condition runs with A2 off (cut0) and on (cut1), six of them also on with a 1 s margin (cut1_m1). A2's effect is the
cycle time on against off. The A2 sums are over the cut1 runs only, as DAIM's gate_summary.py (:41-42). DAIM's pass
is its gate_summary.py PASS (:18, :37); ours is agvtb.tools.batch.passed.

  python -m agvtb.tools.a2gate conditions GATE_DIR --out CONDS.json
  python -m agvtb.tools.a2gate compare GATE_DIR RESULTS_DIR --out TABLE.md
"""
import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

from agvtb.tools.st24 import _f, condition, daim_json_record, our_record

SUFFIXES = {"cut0": (False, 0.0), "cut1": (True, 0.0), "cut1_m1": (True, 1.0)}
DAIM_ZERO = ("validator_violations", "exec_cycles", "event_cycles", "fallback_fail", "inline_miss", "cut_restore_cycle",
             "cut_gate_revoke_refused", "cut_protect_violations", "recovery_rearms")  # gate_summary.py:18
BUG = ("cut_restore_cycle", "cut_gate_revoke_refused", "cut_gate_preexisting_cycle", "cut_protect_violations")
CUT = ("cut_try", "cut_ok", "cut_ok_port", "cut_ok_transit", "cut_tail_s")


def _rows(gate_dir: Path) -> dict[tuple[str, str], dict]:
    with open(Path(gate_dir) / "gate_summary.csv", encoding="utf-8-sig", newline="") as fh:
        return {(row["condition"], row["cut"]): row for row in csv.DictReader(fh)}


def _bases(gate_dir: Path) -> list[str]:
    """The conditions, in the order of gate_summary.csv."""
    return list(dict.fromkeys(name for name, _cut in _rows(gate_dir)))


def gate_conditions(gate_dir: Path, configs_dir: Path = Path("configs/maps/daim")) -> list[dict]:
    """<condition>_<suffix> for every gate record, as agvtb.tools.batch conditions with the A2 switch set."""
    out = []
    for base in _bases(gate_dir):
        for suffix, (cut, margin) in SUFFIXES.items():
            if (Path(gate_dir) / f"gate_{base}_{suffix}.json").exists():
                out.append(condition(base, configs_dir) | {"name": f"{base}_{suffix}", "tail_cut": cut,
                                                           "cut_margin": margin})
    return out


def _daim(gate_dir: Path, rows: dict, base: str, suffix: str) -> dict | None:
    path = Path(gate_dir) / f"gate_{base}_{suffix}.json"
    if not path.exists():
        return None
    rec = daim_json_record(path, rows.get((base, suffix[3])) if suffix != "cut1_m1" else None)
    rs = rec["rs"]

    def st(key):
        return rs.get(f"safe_tail_{key}", 0)

    return {"pass": rec["frozen"] is None and rec["partial"] == 0 and all(st(k) == 0 for k in DAIM_ZERO),
            "cycle": rec["cycle"], "tasks": st("tasks_planned"), "bug": sum(st(k) for k in BUG),
            **{k: st(k) for k in CUT}}


def _ours(results_dir: Path, base: str, suffix: str) -> dict | None:
    try:  # a missing or half-written result is a run not done
        result = json.loads((Path(results_dir) / f"{base}_{suffix}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cut, margin = SUFFIXES[suffix]
    o = our_record(result, base, cut, margin)
    s = result["stats"]
    return {"pass": o["pass"], "cycle": o["cycle"], "tasks": s.get("commits", 0), "bug": sum(s.get(k, 0) for k in BUG),
            **{k: s.get(k, 0) for k in CUT}}


def _yes(r: dict | None) -> str:
    return "-" if r is None else ("yes" if r["pass"] else "**NO**")


def _delta(off: dict | None, on: dict | None) -> float | None:
    if off is None or on is None or not off["cycle"] or not on["cycle"]:
        return None
    return (on["cycle"] / off["cycle"] - 1) * 100


def _cuts(r: dict | None) -> str:
    return "-" if r is None else f"{r['cut_try']}/{r['cut_ok']} ({r['cut_ok_port']}·{r['cut_ok_transit']})"


def _per_cut(r: dict | None) -> str:
    return "-" if r is None or not r["cut_ok"] else _f(r["cut_tail_s"] / r["cut_ok"])


def _sums(runs: list[dict]) -> str:
    if not runs:
        return "-"
    t = collections.Counter()
    for r in runs:
        t.update({k: r[k] for k in CUT + ("tasks",)})
    return (f"rate {t['cut_try'] / max(t['tasks'], 1):.1%} · success {t['cut_ok'] / max(t['cut_try'], 1):.1%} · "
            f"port {t['cut_ok_port'] / max(t['cut_ok'], 1):.1%} · "
            f"tail cut {t['cut_tail_s'] / max(t['cut_ok'], 1):.1f} s per cut")


def _stats(values: list[float]) -> str:
    return f"median {statistics.median(values):+.2f} · mean {statistics.mean(values):+.2f}" if values else "-"


def gate_table(gate_dir: Path, results_dir: Path) -> str:
    """Markdown: a pass line, one row per condition (off / on, ours and DAIM's), the A2 sums, then the margin runs."""
    rows = _rows(gate_dir)
    head = ("| condition | pass off/on ours | pass off/on DAIM | CT s off→on ours | CT s off→on DAIM | "
            "CT Δ% ours/DAIM | A2 try/ok (port·transit) ours | A2 try/ok (port·transit) DAIM | "
            "tail cut s per cut ours/DAIM | "
            "A2 bug ours/DAIM |")
    m1_head = "| condition | CT Δ% off→m1 ours/DAIM | port cuts m0→m1 ours/DAIM | pass m1 ours/DAIM |"
    lines, m1_lines = [head, "|" + "---|" * 10], [m1_head, "|" + "---|" * 4]
    passes = {side: {s: [0, 0] for s in SUFFIXES} for side in ("ours", "DAIM")}
    deltas, on_runs, not_run = {"ours": [], "DAIM": []}, {"ours": [], "DAIM": []}, 0
    bases = _bases(gate_dir)
    for base in bases:
        d = {s: _daim(gate_dir, rows, base, s) for s in SUFFIXES}
        o = {s: _ours(results_dir, base, s) if d[s] is not None else None for s in SUFFIXES}
        for s in SUFFIXES:
            for side, r in (("ours", o[s]), ("DAIM", d[s])):
                if r is not None:
                    passes[side][s][0] += r["pass"]
                    passes[side][s][1] += 1
            not_run += d[s] is not None and o[s] is None
        for side, r in (("ours", o), ("DAIM", d)):
            if _delta(r["cut0"], r["cut1"]) is not None:
                deltas[side].append(_delta(r["cut0"], r["cut1"]))
            if r["cut1"] is not None:
                on_runs[side].append(r["cut1"])
        lines.append(
            f"| {base} | {_yes(o['cut0'])}/{_yes(o['cut1'])} | {_yes(d['cut0'])}/{_yes(d['cut1'])} | "
            f"{_f(o['cut0'] and o['cut0']['cycle'])}→{_f(o['cut1'] and o['cut1']['cycle'])} | "
            f"{_f(d['cut0'] and d['cut0']['cycle'])}→{_f(d['cut1'] and d['cut1']['cycle'])} | "
            f"{_f(_delta(o['cut0'], o['cut1']), '{:+.2f}')}/{_f(_delta(d['cut0'], d['cut1']), '{:+.2f}')} | "
            f"{_cuts(o['cut1'])} | {_cuts(d['cut1'])} | {_per_cut(o['cut1'])}/{_per_cut(d['cut1'])} | "
            f"{_f(o['cut1'] and o['cut1']['bug'], '{}')}/{_f(d['cut1'] and d['cut1']['bug'], '{}')} |")
        if d["cut1_m1"] is not None:
            port = {side: (f"{r['cut1']['cut_ok_port']}→{r['cut1_m1']['cut_ok_port']}"
                           if r["cut1"] is not None and r["cut1_m1"] is not None else "-")
                    for side, r in (("ours", o), ("DAIM", d))}
            m1_lines.append(f"| {base} | {_f(_delta(o['cut0'], o['cut1_m1']), '{:+.2f}')}/"
                            f"{_f(_delta(d['cut0'], d['cut1_m1']), '{:+.2f}')} | {port['ours']}/{port['DAIM']} | "
                            f"{_yes(o['cut1_m1'])}/{_yes(d['cut1_m1'])} |")

    def tally(side):
        p = passes[side]
        return f"off {p['cut0'][0]}/{p['cut0'][1]} on {p['cut1'][0]}/{p['cut1'][1]} m1 {p['cut1_m1'][0]}/{p['cut1_m1'][1]}"

    top = f"pass ours {tally('ours')} · DAIM {tally('DAIM')} ({len(bases)} conditions, {not_run} runs not run)"
    sums = [f"CT Δ% on against off: ours {_stats(deltas['ours'])} · DAIM {_stats(deltas['DAIM'])}",
            f"A2 (cut1 runs): ours {_sums(on_runs['ours'])} · DAIM {_sums(on_runs['DAIM'])}"]
    return "\n".join([top, ""] + lines + [""] + sums + ([""] + m1_lines if len(m1_lines) > 2 else [])) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agvtb.tools.a2gate", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_c = sub.add_parser("conditions", help="write the conditions of the gate records")
    p_c.add_argument("gate_dir")
    p_c.add_argument("--configs", default="configs/maps/daim")
    p_c.add_argument("--out", required=True)
    p_t = sub.add_parser("compare", help="write the table of our results against the records")
    p_t.add_argument("gate_dir")
    p_t.add_argument("results_dir")
    p_t.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "conditions":
        conds = gate_conditions(Path(args.gate_dir), Path(args.configs))
        Path(args.out).write_text(json.dumps(conds, indent=1), encoding="utf-8")
        print(f"{len(conds)} conditions: {args.out}")
    else:
        table = gate_table(Path(args.gate_dir), Path(args.results_dir))
        Path(args.out).write_text(table, encoding="utf-8")
        print(f"{table.splitlines()[0]}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
