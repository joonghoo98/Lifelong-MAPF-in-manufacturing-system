import subprocess
import sys

import matplotlib.pyplot as plt
import pytest

from agvtb.tools import describe_map
from agvtb.world.collision_grid import GridRuleError, grid_conflicts
from agvtb.world.collision_poly import poly_conflicts
from agvtb.world.loader import load_world
from agvtb.world.plot import plot_world
from helpers import CONFIGS, tiny_config

def test_load_world_dispatch():
    assert load_world(CONFIGS / "mini_deadend.json").world.collision == "grid"
    k = load_world(CONFIGS / "KAIROS_AMR.json")
    assert sum(1 for st in k.graph.values() if st.kind == "S") == 169

def test_load_world_refuses_big_vehicle(tmp_path):
    with pytest.raises(GridRuleError):
        load_world(tiny_config(tmp_path, vehicle_m=0.8))

def test_describe_cli(capsys):
    assert describe_map.main([str(CONFIGS / "mini_deadend.json")]) == 0
    out = capsys.readouterr().out
    assert "nodes: 166  segments: 530  ports: 16" in out and "collision: grid" in out

def test_plot_writes_png(tmp_path):
    lw = load_world(CONFIGS / "mini_deadend.json")
    p = plot_world(lw.world, tmp_path / "mini.png", title="mini_deadend")
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and p.stat().st_size > 1000

# --- further checks: details the four tests above do not pin down ---

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MINI_SUMMARY = (  # the counts and pairs come from a separate computation; footprint and speed are in :g format
    "map: mini_deadend\n"
    "collision: grid\n"
    "nodes: 166  segments: 530  ports: 16\n"
    "states: S=660 M=530 R=1972 L=64\n"
    "conflict_pairs: 40749\n"
    "footprint_m: 0.619579 x 0.619579\n"
    "speed_mps: 0.890335-0.890335\n"
)

def test_load_world_polygon_mode_has_no_cell_rule(tmp_path):
    # the vehicle (0.8 m on a 1 m grid) is too big for the cell rule, so grid mode refuses it; polygon mode
    # never applies that rule and its conflicts are the polygon ones: 279 pairs here against 231 by cells
    lw = load_world(tiny_config(tmp_path, vehicle_m=0.8, collision="polygon"))
    assert lw.world.collision == "polygon"
    assert lw.conflicts == poly_conflicts(lw.world, lw.graph)
    assert lw.conflicts != grid_conflicts(lw.world, lw.graph)

def test_load_world_profile_mode_takes_the_profile_conflicts():
    # the profile has 33,000 pairs among S, M and R; the L states added at the ports bring it to 33,054
    lw = load_world(CONFIGS / "KAIROS_AMR.json")
    assert lw.world.collision == "profile"
    assert sum(len(others) for others in lw.conflicts.values()) == 2 * 33054

def test_describe_prints_exactly_the_seven_lines(capsys):
    assert describe_map.main([str(CONFIGS / "mini_deadend.json")]) == 0
    assert capsys.readouterr().out == MINI_SUMMARY

def test_describe_plot_option_saves_a_figure_and_prints_the_same(capsys, tmp_path):
    figure = tmp_path / "figs" / "mini.png"  # its folder does not exist yet
    assert describe_map.main([str(CONFIGS / "mini_deadend.json"), "--plot", str(figure)]) == 0
    assert capsys.readouterr().out == MINI_SUMMARY
    assert figure.read_bytes()[:8] == PNG_SIGNATURE

def test_describe_runs_as_a_module():
    done = subprocess.run(
        [sys.executable, "-m", "agvtb.tools.describe_map", str(CONFIGS / "mini_deadend.json")],
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == MINI_SUMMARY

def test_plot_world_makes_the_folder_and_needs_no_title(tmp_path):
    world = load_world(tiny_config(tmp_path)).world
    out = tmp_path / "a" / "b" / "tiny.png"  # neither folder exists
    assert plot_world(world, out) is out
    assert out.read_bytes()[:8] == PNG_SIGNATURE

def test_plot_world_leaves_no_figure_open(tmp_path):
    world = load_world(tiny_config(tmp_path)).world
    open_before = plt.get_fignums()
    plot_world(world, tmp_path / "tiny.png", title="tiny")
    assert plt.get_fignums() == open_before
