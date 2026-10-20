import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from matplotlib.figure import Figure

from agvtb.sidings.grid_rule import GRID_MODES
from agvtb.sidings.lattice import GRID_SHIFTS
from agvtb.tools import sidings_table
from agvtb.tools.sidings_table import TABLE_COLUMNS, Variant, map_rows
from agvtb.world.loader import load_world
from helpers import CONFIGS, link_config

GRID_NAMES = [f"grid-{m}-s{dr}{dc}" for m in GRID_MODES for dr, dc in GRID_SHIFTS]

def test_rows_of_mini():
    res = map_rows(load_world(CONFIGS / "mini_deadend.json"))
    assert [r["variant"] for r in res.rows] == GRID_NAMES + ["strict"]
    by = {r["variant"]: r for r in res.rows}
    assert (by["grid-bridge-s00"]["candidates"], by["grid-bridge-s00"]["sidings"]) == (39, 37)
    assert (by["grid-legacy-s10"]["candidates"], by["grid-legacy-s10"]["sidings"]) == (39, 35)
    assert by["strict"]["sidings"] == 80  # strict_expected.out.txt
    for r in res.rows:
        assert r["usable"] + r["port_access"] + r["pocket"] + r["port"] + r["not_in_graph"] == r["sidings"]
        assert r["n_vehicles"] == 0 and r["max_vehicles"] == (r["usable"] if r["violations"] == 0 else 0)

def test_kairos_has_only_the_strict_row():
    res = map_rows(load_world(CONFIGS / "KAIROS_AMR.json"))
    assert [r["variant"] for r in res.rows] == ["strict"] and res.rows[0]["sidings"] == 14
    assert any("grid rule skipped" in n and "KAIROS_AMR" in n for n in res.notes)

def test_recorded_row_and_fleet(tmp_path):
    links = [(f"c{x}{y}", f"c{x + 1}{y}") for x in range(5) for y in range(3)] + \
            [(f"c{x}{y}", f"c{x}{y + 1}") for x in range(6) for y in range(2)] + [("c02", "c03"), ("c22", "c23"), ("c42", "c43")]
    nodes = {n: (float(n[1]), float(n[2])) for link in links for n in link}
    info = {"sidings": ["c03", "c23", "c43"], "fleet": {"AMR": 2}}
    res = map_rows(load_world(link_config(tmp_path, nodes, links, ["c00", "c50"], "rec", info)))
    rec = next(r for r in res.rows if r["variant"] == "recorded")
    assert (rec["candidates"], rec["sidings"], rec["usable"], rec["aisle"], rec["n_vehicles"], rec["ok"]) == (3, 3, 3, 18, 2, True)
    assert res.rows[-1]["variant"] == "recorded"

def test_cli_writes_the_same_files_twice(tmp_path):
    outs = []
    for k in (1, 2):
        out = tmp_path / f"run{k}"
        assert sidings_table.main([str(CONFIGS / "mini_deadend.json"), "--out", str(out), "--plot"]) == 0
        outs.append(out)
    with open(outs[0] / "sidings_table.csv", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        assert tuple(next(reader)) == TABLE_COLUMNS and len(list(reader)) == 9
    md = (outs[0] / "sidings_table.md").read_text(encoding="utf-8").splitlines()
    assert len(md) == 2 + 9
    assert {p.name for p in (outs[0] / "mini_deadend").glob("*.png")} == {"grid-bridge-s00.png", "strict.png"}
    for name in ["sidings_table.csv", "sidings_table.md"] + [f"mini_deadend/{v}.json" for v in GRID_NAMES + ["strict"]]:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes()

# Beyond the plan's tests: what the four above leave open (added with the implementation). Maps are built as in
# tests/sidings/test_condition_m.py: nodes c<x><y> at (x, y) metres, a 1 m lattice, 0.5 m vehicle.

def room(w, h):  # the links of a w x h block of nodes
    return [(f"c{x}{y}", f"c{x + 1}{y}") for x in range(w - 1) for y in range(h)] + \
           [(f"c{x}{y}", f"c{x}{y + 1}") for x in range(w) for y in range(h - 1)]

def loaded(tmp_path, links, ports, name, info=None):
    nodes = {n: (float(n[1]), float(n[2])) for link in links for n in link}
    return load_world(link_config(tmp_path, nodes, links, ports, name, info))

SPURS = room(6, 3) + [("c02", "c03"), ("c22", "c23"), ("c42", "c43")]  # a room with three spurs: 3 usable sidings

def spurs(tmp_path, info=None, name="spurs"):
    return loaded(tmp_path, SPURS, ["c00", "c50"], name, {"sidings": ["c03", "c23", "c43"], **(info or {})})

def csv_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def test_variant_names():
    assert [Variant("grid", "bridge", shift).name for shift in GRID_SHIFTS] == \
           ["grid-bridge-s00", "grid-bridge-s01", "grid-bridge-s10", "grid-bridge-s11"]
    assert Variant("grid", "legacy", (1, 1)).name == "grid-legacy-s11"
    assert (Variant("strict").name, Variant("recorded").name) == ("strict", "recorded")

MINI_GRID_COUNTS = {  # (mode, shift) -> (candidates, sidings): the grid rule on the lattice of mini_deadend, as in
    # test_lattice.py COUNTS (grid_rule_expected.out.txt), here read off the table rows
    ("bridge", (0, 0)): (39, 37), ("bridge", (0, 1)): (36, 36), ("bridge", (1, 0)): (39, 37), ("bridge", (1, 1)): (36, 36),
    ("legacy", (0, 0)): (39, 36), ("legacy", (0, 1)): (36, 36), ("legacy", (1, 0)): (39, 35), ("legacy", (1, 1)): (36, 36),
}

def test_rows_of_mini_have_the_counts_of_the_rules_and_the_columns_in_order():
    res = map_rows(load_world(CONFIGS / "mini_deadend.json"))
    by = {r["variant"]: r for r in res.rows}
    for (mode, (dr, dc)), counts in MINI_GRID_COUNTS.items():
        row = by[f"grid-{mode}-s{dr}{dc}"]
        assert (row["candidates"], row["sidings"]) == counts, row["variant"]
    assert (by["strict"]["candidates"], by["strict"]["sidings"]) == (150, 80)  # 166 nodes, 16 of them ports
    assert all(tuple(r) == TABLE_COLUMNS for r in res.rows)
    assert {(r["map"], r["collision"]) for r in res.rows} == {("mini_deadend", "grid")}
    assert list(res.details) == [r["variant"] for r in res.rows] and res.notes == []

def test_every_exclusion_reason_has_its_own_column(tmp_path):
    # a 6 x 2 room with the chains c11-c12-c13-c14 (port c13: its access c12 is a siding that cuts it off, port_access;
    # c14 hangs on the port: a pocket), c31-c32-c33 (port c32 and the pocket c33 behind it) and the spur c51-c52 (usable)
    links = room(6, 2) + [("c11", "c12"), ("c12", "c13"), ("c13", "c14"), ("c31", "c32"), ("c32", "c33"), ("c51", "c52")]
    candidates = ["c00", "c12", "c13", "c14", "c32", "c33", "c52", "z1", "z2", "z3", "z4"]  # three ports, four non-nodes
    res = map_rows(loaded(tmp_path, links, ["c00", "c13", "c32"], "reasons", {"sidings": candidates}))
    rec = res.rows[-1]
    assert (rec["variant"], rec["candidates"], rec["sidings"], rec["usable"]) == ("recorded", 11, 11, 1)
    # the four counts differ, so that no two columns can be mixed up
    assert (rec["port_access"], rec["pocket"], rec["port"], rec["not_in_graph"]) == (1, 2, 3, 4)
    assert (rec["aisle"], rec["violations"], rec["max_vehicles"], rec["n_vehicles"], rec["ok"]) == (17, 0, 1, 0, True)
    detail = res.details["recorded"]
    assert detail["excluded"] == {
        "c00": "port", "c12": "port_access", "c13": "port", "c14": "pocket", "c32": "port", "c33": "pocket",
        "z1": "not_in_graph", "z2": "not_in_graph", "z3": "not_in_graph", "z4": "not_in_graph",
    }
    assert (detail["sidings"], detail["usable"]) == (sorted(candidates), ["c52"])

def test_a_violation_makes_max_vehicles_zero(tmp_path):
    # port c90 sits on a two-node island, so no vehicle can serve it from the spurs (SPURS + an island, as in
    # test_condition_m.py test_port_outside_the_aisle)
    res = map_rows(loaded(tmp_path, SPURS + [("c90", "c91")], ["c00", "c50", "c90"], "island",
                          {"sidings": ["c03", "c23", "c43"], "fleet": {"AMR": 2}}))
    rec = res.rows[-1]
    assert (rec["variant"], rec["usable"], rec["n_vehicles"]) == ("recorded", 3, 2)
    assert rec["violations"] > 0 and rec["max_vehicles"] == 0 and rec["ok"] is False
    assert len(res.details["recorded"]["violations"]) == rec["violations"]

def test_the_vehicles_argument_beats_the_fleet_and_zero_is_a_value(tmp_path):
    lw = spurs(tmp_path, {"fleet": {"AMR": 2}})
    assert {r["n_vehicles"] for r in map_rows(lw).rows} == {2}
    assert {r["n_vehicles"] for r in map_rows(lw, 0).rows} == {0}  # 0 does not fall back to the fleet
    five = map_rows(lw, 5).rows[-1]  # three usable sidings for five vehicles: fatal, with nothing violated
    assert (five["n_vehicles"], five["usable"], five["violations"], five["max_vehicles"], five["ok"]) == (5, 3, 0, 3, False)
    with pytest.raises(ValueError, match="-1"):
        map_rows(lw, -1)

@pytest.mark.parametrize("fleet", [{}, {"AGV": 3}, None, "many"])
def test_no_vehicle_count_in_the_map_means_zero(tmp_path, fleet):
    info = {} if fleet is None else {"fleet": fleet}
    assert {r["n_vehicles"] for r in map_rows(spurs(tmp_path, info)).rows} == {0}

def test_recorded_sidings_are_read_as_distinct_strings(tmp_path):
    line = {"1": (0.0, 0.0), "2": (1.0, 0.0), "3": (2.0, 0.0), "4": (3.0, 0.0)}  # node ids as DAIM names them: digits
    lw = load_world(link_config(tmp_path, line, [("1", "2"), ("2", "3"), ("3", "4")], ["1", "4"], "digits",
                                {"sidings": [2, "2", 3]}))
    rec = map_rows(lw).rows[-1]
    assert (rec["variant"], rec["candidates"], rec["sidings"], rec["not_in_graph"]) == ("recorded", 2, 2, 0)

def test_a_map_that_is_no_lattice_keeps_its_other_rows(tmp_path):
    # D1 stands diagonally from L0: no lattice (test_lattice.py test_rejects_more_bad_lattices), so polygon collision
    nodes = {"L0": (0.0, 0.0), "L1": (1.0, 0.0), "L2": (2.0, 0.0), "L3": (3.0, 0.0), "D1": (1.0, 1.0)}
    links = [("L0", "L1"), ("L1", "L2"), ("L2", "L3"), ("L0", "D1")]
    config = link_config(tmp_path, nodes, links, ["L0", "L3"], "diag", {"sidings": ["L1"]})
    data = json.loads(config.read_text(encoding="utf-8"))
    config.write_text(json.dumps({**data, "collision": "polygon"}), encoding="utf-8")
    res = map_rows(load_world(config))
    assert [r["variant"] for r in res.rows] == ["strict", "recorded"]
    assert len(res.notes) == 1 and res.notes[0].startswith("grid rule skipped: diag: ")

def test_cli_collects_the_maps_in_the_order_given_and_passes_the_vehicles_on(tmp_path, capsys):
    out = tmp_path / "out"
    spurs(tmp_path, name="spurs")  # writes spurs.json and its map into tmp_path
    config = (tmp_path / "spurs.json").rename(tmp_path / "renamed.json")  # the folder goes by the map's name, not the file's
    assert sidings_table.main([str(config), str(CONFIGS / "mini_deadend.json"), "--out", str(out), "--vehicles", "3"]) == 0
    assert capsys.readouterr().out == "spurs: 10 variants\nmini_deadend: 9 variants\n"
    rows = csv_rows(out / "sidings_table.csv")
    assert [r["map"] for r in rows] == ["spurs"] * 10 + ["mini_deadend"] * 9
    assert {r["n_vehicles"] for r in rows} == {"3"}
    assert (out / "spurs" / "recorded.json").is_file() and (out / "mini_deadend" / "strict.json").is_file()
    assert not list(out.rglob("*.png"))  # without --plot there are no figures

def test_cli_on_a_map_without_grid_rows(tmp_path, capsys):
    out = tmp_path / "out"
    assert sidings_table.main([str(CONFIGS / "KAIROS_AMR.json"), "--out", str(out), "--plot"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "KAIROS_AMR: 1 variants" and len(lines) == 2
    assert lines[1].startswith("grid rule skipped: KAIROS_AMR: ")
    assert sorted(p.name for p in (out / "KAIROS_AMR").iterdir()) == ["strict.json", "strict.png"]  # no grid-bridge-s00 figure
    detail = json.loads((out / "KAIROS_AMR" / "strict.json").read_text(encoding="utf-8"))
    assert (len(detail["usable"]), len(detail["violations"]), detail["condition_m"]["ok"]) == (14, 13, False)
    (row,) = csv_rows(out / "sidings_table.csv")
    assert (row["collision"], row["usable"], row["violations"], row["max_vehicles"], row["ok"]) == ("profile", "14", "13", "0", "False")

def test_cli_figures_mark_the_usable_sidings_and_the_excluded_candidates(tmp_path, monkeypatch):
    figures = {}  # file name -> the figure saved there
    save = Figure.savefig
    def spy(self, fname, *args, **kwargs):
        figures[Path(fname).name] = self
        return save(self, fname, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", spy)
    assert sidings_table.main([str(CONFIGS / "mini_deadend.json"), "--out", str(tmp_path), "--plot"]) == 0
    assert sorted(figures) == ["grid-bridge-s00.png", "strict.png"]
    legends = {}
    for name, figure in figures.items():
        legend = figure.axes[0].get_legend()
        legends[name] = None if legend is None else [text.get_text() for text in legend.get_texts()]
    # 37 sidings of the grid rule, none excluded, so no legend; of the 80 strict sidings 73 are usable and 7 are pockets
    assert legends == {"grid-bridge-s00.png": None, "strict.png": ["siding (73)", "pocket (7)"]}
    titles = {name: figure.axes[0].get_title().split(" (")[0] for name, figure in figures.items()}  # before the key
    assert titles == {"grid-bridge-s00.png": "mini_deadend: grid-bridge-s00", "strict.png": "mini_deadend: strict"}

def test_cli_lists_hold_the_sidings_and_what_condition_m_made_of_them(tmp_path):
    out = tmp_path / "out"
    assert sidings_table.main([str(CONFIGS / "mini_deadend.json"), "--out", str(out)]) == 0
    strict = json.loads((out / "mini_deadend" / "strict.json").read_text(encoding="utf-8"))
    assert set(strict) == {"rule", "mode", "shift", "sidings", "usable", "excluded", "violations", "condition_m"}
    assert (strict["rule"], strict["mode"], strict["shift"]) == ("strict", None, None)
    assert strict["sidings"] == sorted(strict["sidings"]) and len(strict["sidings"]) == 80
    assert strict["usable"] == sorted(strict["usable"]) and len(strict["usable"]) == 73
    assert set(strict["usable"]) | set(strict["excluded"]) == set(strict["sidings"])
    assert not set(strict["usable"]) & set(strict["excluded"])
    assert set(strict["excluded"].values()) == {"pocket"} and len(strict["excluded"]) == 7
    assert strict["condition_m"]["excluded"] == strict["excluded"] and strict["condition_m"]["sidings"] == 73
    grid = json.loads((out / "mini_deadend" / "grid-legacy-s10.json").read_text(encoding="utf-8"))
    assert (grid["rule"], grid["mode"], grid["shift"], len(grid["sidings"])) == ("grid", "legacy", [1, 0], 35)

def test_cli_files_have_unix_line_ends_and_the_json_is_indented_and_sorted(tmp_path):
    out = tmp_path / "out"
    assert sidings_table.main([str(CONFIGS / "mini_deadend.json"), "--out", str(out)]) == 0
    files = sorted(p for p in out.rglob("*") if p.is_file())
    assert len(files) == 2 + 9
    for path in files:
        data = path.read_bytes()
        assert b"\r" not in data and data.endswith(b"\n") and not data.endswith(b"\n\n"), path.name
    raw = (out / "mini_deadend" / "strict.json").read_text(encoding="utf-8")
    assert raw.startswith('{\n "condition_m": {\n  "aisle_size": ') and raw.endswith("\n}\n")  # indent of one space
    assert [line.split('"')[1] for line in raw.splitlines() if line.startswith(' "')] == \
           ["condition_m", "excluded", "mode", "rule", "shift", "sidings", "usable", "violations"]

def test_cli_rejects_negative_vehicles(tmp_path, capsys):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as stop:
        sidings_table.main([str(CONFIGS / "mini_deadend.json"), "--out", str(out), "--vehicles", "-1"])
    assert stop.value.code == 2 and "-1" in capsys.readouterr().err
    assert not out.exists()  # nothing was written

def test_cli_runs_as_a_module_and_ignores_the_hash_seed(tmp_path):
    outs = []
    for seed in ("1", "2"):
        out = tmp_path / f"seed{seed}"
        done = subprocess.run(
            [sys.executable, "-m", "agvtb.tools.sidings_table", str(CONFIGS / "mini_deadend.json"), "--out", str(out), "--plot"],
            env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True, text=True, timeout=300,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout == "mini_deadend: 9 variants\n"
        outs.append(out)
    names = [sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) for out in outs]
    assert names[0] == names[1] and len(names[0]) == 2 + 9 + 2  # the table twice, nine lists, two figures
    for name in names[0]:
        assert (outs[0] / name).read_bytes() == (outs[1] / name).read_bytes(), name
