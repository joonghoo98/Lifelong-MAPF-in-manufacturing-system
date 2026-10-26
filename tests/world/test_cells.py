import json
import os
import random
import subprocess
import sys

import pytest

from agvtb.world.cells import build_cell_map, cell_map_from_graph, cut_and_loop, dead_end_cells
from agvtb.world.loader import load_world
from helpers import CONFIGS, REPO, SMALL_GRAPHS

MAPS = {  # cells, merges, port cells, cut cells, stations (plans/2026-10-24-m6a-fallback-mode_files)
    "KAIROS_AMR": (44, 1, 13, 19, 25),
    "mini_deadend": (166, 0, 16, 4, 162),
    "songdo_grid_bi": (1384, 0, 10, 11, 1373),
}


@pytest.fixture(scope="module", params=sorted(MAPS))
def built(request):
    lw = load_world(CONFIGS / f"{request.param}.json")
    return request.param, lw, build_cell_map(lw.world, lw.graph, lw.conflicts)


def test_the_cell_counts_of_the_three_maps(built):
    name, _, cm = built
    assert (len(cm.cells), cm.merges, len(cm.port_cells), len(cm.cut), len(cm.stations)) == MAPS[name]


def test_kairos_joins_its_cell_that_was_split_inside(built):
    name, _, cm = built
    if name == "KAIROS_AMR":
        assert cm.members["na.10"] == ("na.10", "na.5", "na.53", "na.54", "na.6", "na.9")
        assert "na.10" in cm.cut and "na.10" in cm.port_cells


def test_the_cell_rule_and_the_inner_links_hold(built):
    _, lw, cm = built
    for sid in sorted(lw.graph):
        touched = {cm.cell_of[x] for x in lw.graph[sid].nodes}
        for other in sorted(lw.conflicts[sid]):
            if lw.graph[other].kind in ("S", "L"):
                assert cm.cell_of[lw.graph[other].nodes[0]] in touched, (sid, other)
    out = {}
    for u, v in lw.world.segments:
        out.setdefault(u, []).append(v)
    for cell, nodes in cm.members.items():
        inside, seen, todo = set(nodes), {nodes[0]}, [nodes[0]]
        while todo:
            for v in out.get(todo.pop(), ()):
                if v in inside and v not in seen:
                    seen.add(v)
                    todo.append(v)
        assert seen == inside and cell == nodes[0], cell
    assert set(cm.cell_of) == set(lw.world.nodes) and set(cm.cell_of.values()) == set(cm.cells)


def _brute(adj):
    def connected(gone_node=None, gone_edge=None):
        rest = [v for v in adj if v != gone_node]
        seen, todo = {rest[0]}, [rest[0]]
        while todo:
            v = todo.pop()
            for u in adj[v]:
                if u != gone_node and u not in seen and {v, u} != gone_edge:
                    seen.add(u)
                    todo.append(u)
        return len(seen) == len(rest)
    bridges = {frozenset((a, b)) for a in adj for b in adj[a] if not connected(gone_edge={a, b})}
    loop = frozenset(v for v in adj if any(frozenset((v, u)) not in bridges for u in adj[v]))
    return frozenset(v for v in adj if not connected(gone_node=v)), loop


def test_cut_and_loop_match_brute_force():
    rng = random.Random(7)
    for _ in range(300):
        nodes = [str(i) for i in range(rng.randint(2, 9))]
        edges = {(v, str(rng.randrange(int(v)))) for v in nodes[1:]}  # a spanning tree keeps it connected
        edges |= {tuple(sorted(rng.sample(nodes, 2))) for _ in range(rng.randint(0, len(nodes)))}
        adj = {v: tuple(sorted({b for a, b in edges if a == v} | {a for a, b in edges if b == v})) for v in nodes}
        assert cut_and_loop(adj) == _brute(adj)


def test_dead_end_cells_of_the_small_graphs():
    cm = cell_map_from_graph(*SMALL_GRAPHS["lollipop"])
    assert (cm.cut, cm.loop, cm.port_cells) == ({"0", "4", "5"}, {"0", "1", "2", "3"}, {"2", "6"})
    assert dead_end_cells(cm) == {"4", "5", "6"}
    spider = cell_map_from_graph(*SMALL_GRAPHS["spider"])
    assert spider.loop == frozenset() and dead_end_cells(spider) == set(spider.cells)


def test_cells_do_not_depend_on_the_hash_seed():  # Review Focus 5
    code = ("import json; from agvtb.world.loader import load_world; from agvtb.world.cells import build_cell_map; "
            f"lw = load_world(r'{CONFIGS / 'KAIROS_AMR.json'}'); "
            "print(json.dumps(build_cell_map(lw.world, lw.graph, lw.conflicts).members, sort_keys=True))")
    outs = {subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO, capture_output=True, text=True, check=True,
                           env=os.environ | {"PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO / "src")}).stdout
            for seed in ("1", "2")}
    assert len(outs) == 1
