import json
import subprocess
import sys
import warnings

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from agvtb.sidings.condition_m import EXCLUSION_REASONS
from agvtb.tools import describe_map
from agvtb.world.collision_grid import GridRuleError, grid_conflicts
from agvtb.world.collision_poly import poly_conflicts
from agvtb.world.loader import LoadedWorld, load_world
from agvtb.world.plot import plot_world
from helpers import CONFIGS, GRID_MAP, tiny_config

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

def test_loaded_world_carries_map_info_and_hash():
    import hashlib
    from helpers import DAIM_CONFIGS, GRID_MAP
    assert load_world(DAIM_CONFIGS / "grid_songdo_travelbr.json").map_info["fleet"] == {"AMR": 15}
    mini = load_world(CONFIGS / "mini_deadend.json")
    path = GRID_MAP / "mini_deadend" / "mini_deadend_xMSmap.json"
    assert mini.map_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

def test_plot_draws_sidings(tmp_path):
    lw = load_world(CONFIGS / "mini_deadend.json")
    plain = plot_world(lw.world, tmp_path / "plain.png")
    marked = plot_world(lw.world, tmp_path / "marked.png", sidings=["G2_2", "G4_2"], excluded={"G6_2": "pocket"})
    assert marked.read_bytes()[:8] == PNG_SIGNATURE and marked.read_bytes() != plain.read_bytes()

# Beyond the plan's tests: the map info and hash that LoadedWorld carries, and the overlay of plot_world.

def test_loaded_world_carries_the_info_block_as_the_file_has_it_and_keeps_its_old_form(tmp_path):
    info = json.loads((GRID_MAP / "mini_deadend" / "mini_deadend_xMSmap.json").read_bytes())["info"]
    lw = load_world(CONFIGS / "mini_deadend.json")
    assert info and lw.map_info == info
    tiny = load_world(tiny_config(tmp_path))  # the tiny map has no info block
    assert tiny.map_info == {} and len(tiny.map_sha256) == 64
    old_form = LoadedWorld(lw.world, lw.graph, lw.conflicts)  # M1 made it from these three
    assert (old_form.map_info, old_form.map_sha256) == ({}, "")

@pytest.fixture
def saved_figures(monkeypatch):
    """The figures plot_world saves, in order, so that a test can look at what was drawn."""
    figures = []
    save = Figure.savefig
    def spy(self, *args, **kwargs):
        figures.append(self)
        return save(self, *args, **kwargs)
    monkeypatch.setattr(Figure, "savefig", spy)
    return figures

def legend_texts(figure):
    legend = figure.axes[0].get_legend()
    return None if legend is None else [text.get_text() for text in legend.get_texts()]

def test_plot_with_nothing_marked_is_the_plain_picture_without_a_legend(tmp_path, saved_figures):
    world = load_world(CONFIGS / "mini_deadend.json").world
    plain = plot_world(world, tmp_path / "plain.png").read_bytes()
    nothing = [{"sidings": ()}, {"sidings": iter(())}, {"excluded": {}}, {"excluded": None},
               {"excluded": {"nowhere": "not_in_graph"}}]  # a candidate that is no node has no place on the map
    for k, kwargs in enumerate(nothing):
        assert plot_world(world, tmp_path / f"{k}.png", **kwargs).read_bytes() == plain
    assert [legend_texts(figure) for figure in saved_figures] == [None] * (1 + len(nothing))

def test_plot_legend_counts_the_sidings_and_the_crosses_of_each_reason(tmp_path, saved_figures):
    world = load_world(CONFIGS / "mini_deadend.json").world
    excluded = {"G12_2": "not_in_graph", "G10_2": "port_access", "G6_2": "pocket", "G8_2": "pocket", "nowhere": "not_in_graph"}
    plot_world(world, tmp_path / "a.png", sidings=["G2_2", "G4_2", "G2_2"], excluded=excluded)
    # the sidings, then the reasons in the order of their crosses' sizes; the sidings count once and nowhere is left out
    assert legend_texts(saved_figures[-1]) == ["siding (2)", "pocket (2)", "port_access (1)", "not_in_graph (1)"]
    plot_world(world, tmp_path / "b.png", excluded={"G6_2": "pocket"})  # crosses alone get a legend too
    assert legend_texts(saved_figures[-1]) == ["pocket (1)"]
    plot_world(world, tmp_path / "c.png", sidings=["G2_2"])  # no excluded candidate, no legend (the plan's rule)
    assert legend_texts(saved_figures[-1]) is None

def test_plot_crosses_have_a_size_for_each_reason_and_keep_it_from_figure_to_figure(tmp_path, saved_figures):
    world = load_world(CONFIGS / "mini_deadend.json").world

    def sizes(excluded):  # reason -> marker area of its crosses, read off the figure
        plot_world(world, tmp_path / "x.png", sidings=["G2_2"], excluded=excluded)
        crosses = [c for c in saved_figures[-1].axes[0].collections if not c.get_label().startswith(("_", "siding"))]
        return {c.get_label().rsplit(" (", 1)[0]: float(c.get_sizes()[0]) for c in crosses}

    reasons = [*EXCLUSION_REASONS, "other"]  # the four of condition M, and any other: a fifth size
    every = sizes({f"G{6 + 2 * k}_2": reason for k, reason in enumerate(reasons)})
    assert set(every) == set(reasons) and len(set(every.values())) == len(reasons)  # one size each, no two the same
    assert sizes({"G8_2": "port_access"}) == {"port_access": every["port_access"]}  # whichever other reasons are drawn

def test_plot_marks_lie_over_the_node_dots_and_under_the_port_labels_and_markers(tmp_path, saved_figures):
    world = load_world(CONFIGS / "mini_deadend.json").world
    plot_world(world, tmp_path / "a.png", sidings=["G2_2"], excluded={"G6_2": "pocket"})
    ax = saved_figures[-1].axes[0]
    marks = [c for c in ax.collections if not c.get_label().startswith("_")]  # the labelled ones
    _lines, nodes, ports = [c for c in ax.collections if c not in marks]
    assert len(marks) == 2 and ax.texts  # the port labels are the texts
    over, under = nodes.get_zorder(), min(ports.get_zorder(), *(text.get_zorder() for text in ax.texts))
    assert all(over < mark.get_zorder() < under for mark in marks)

def test_plot_refuses_a_siding_that_is_no_node(tmp_path):
    world = load_world(tiny_config(tmp_path)).world
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match="'nowhere'"):
        plot_world(world, out, sidings=["A", "nowhere"])
    assert not out.exists()

def test_plot_gives_no_warning_whatever_is_marked(tmp_path):
    world = load_world(CONFIGS / "mini_deadend.json").world
    cases = [{}, {"sidings": ["G2_2"]}, {"excluded": {"G6_2": "pocket"}}, {"excluded": {"nowhere": "pocket"}},
             {"sidings": [], "excluded": {}}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an empty legend would warn "No artists with labels found to put in legend"
        for k, kwargs in enumerate(cases):
            plot_world(world, tmp_path / f"{k}.png", **kwargs)
