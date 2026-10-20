import dataclasses
import math

import pytest
import shapely
from shapely import affinity

from agvtb.world.collision_grid import grid_conflicts
from agvtb.world.collision_poly import footprint_polygon, poly_conflicts, state_area
from agvtb.world.model import Footprint
from agvtb.world.states import State, build_state_graph, geom_key
from helpers import tiny_world, world_of

def test_tiny_shapes(tmp_path):
    w = tiny_world(tmp_path); g = build_state_graph(w)
    fp = footprint_polygon(w, 0, 0, 0)
    assert fp.area == pytest.approx(0.25) and fp.bounds == pytest.approx((-0.25, -0.25, 0.25, 0.25))
    m = state_area(w, g["M,A,B"])
    assert m.area == pytest.approx(0.75) and m.bounds == pytest.approx((-0.25, -0.25, 1.25, 0.25))
    r = state_area(w, g["R,B,0,90"])
    assert r.buffer(1e-9).covers(footprint_polygon(w, 1, 0, 0)) and r.buffer(1e-9).covers(footprint_polygon(w, 1, 0, 90))
    assert max(math.dist(p, (1, 0)) for p in r.exterior.coords) <= math.sqrt(0.125) + 1e-9 and r.area > 0.25

def test_tiny_poly_equals_grid(tmp_path):
    w = tiny_world(tmp_path, collision="polygon"); g = build_state_graph(w)
    assert poly_conflicts(w, g) == grid_conflicts(w, g)

@pytest.fixture(scope="module")
def songdo():
    w = world_of("songdo_grid_bi"); g = build_state_graph(w)
    return w, g, poly_conflicts(w, g), grid_conflicts(w, g)

def test_port7_overlap_needs_polygon(songdo):
    w, g, poly, grid = songdo
    assert "S,G83_40,3" in poly["S,Port7,3"]          # 중심 거리 0.478 m < 차량 0.5 m
    assert "S,G83_40,3" not in grid["S,Port7,3"]      # 칸 규칙은 이 겹침을 놓친다

def test_grid_states_match_cell_rule(songdo):
    w, g, poly, grid = songdo
    gs = {s for s, st in g.items() if all(n.startswith("G") for n in st.nodes)}
    assert {s: poly[s] & gs for s in gs} == {s: grid[s] & gs for s in gs}

def test_small_turn_area(songdo):
    w, g, poly, grid = songdo
    assert state_area(w, g["R,G83_40,0,3"]).area > 0

# --- further checks: details the tests above do not pin down ---

def _turn(from_heading, to_heading, node="B"):
    # an R state made by hand: the tiny map has no turn from 350 to 10
    return State(sid=f"R,{node},{from_heading},{to_heading}", kind="R", nodes=(node,), heading=None,
                 from_heading=from_heading, to_heading=to_heading, duration_s=None, next=())

def test_footprint_length_along_heading_width_across(tmp_path):
    # a 1.0 x 0.4 m vehicle tells length from width; headings run counter-clockwise from +x
    w = dataclasses.replace(tiny_world(tmp_path), footprint=Footprint(1.0, 0.4))
    long_box = (-0.5, -0.2, 0.5, 0.2)      # heading 0, 180: the length lies along x
    wide_box = (-0.2, -0.5, 0.2, 0.5)      # heading 90, 270: the length lies along y
    # equal to the last bit, with the centre at 0 where the noise of cos(90 deg) = 6e-17 would show:
    # at an axis heading a corner is just the centre +- a half side
    assert [footprint_polygon(w, 0, 0, h).bounds for h in (0, 90, 180, 270, 360, -90)] == [long_box, wide_box, long_box, wide_box, long_box, wide_box]
    assert footprint_polygon(w, 2, 3, 0).bounds == (2 - 0.5, 3 - 0.2, 2 + 0.5, 3 + 0.2)     # and moved to (2, 3)
    fp = footprint_polygon(w, 2, 3, 30)
    ahead = shapely.Point(2 + 0.45 * math.cos(math.radians(30)), 3 + 0.45 * math.sin(math.radians(30)))
    mirrored = shapely.Point(ahead.x, 6 - ahead.y)         # as far from the centre, towards -30 degrees
    assert fp.contains(ahead) and not fp.contains(mirrored)
    assert fp.area == pytest.approx(0.4) and fp.centroid.coords[0] == pytest.approx((2, 3))

def test_port_link_sweeps_at_the_rounded_heading(songdo):
    # the link runs at 2.62 deg but its states carry heading 3 (183): the vehicle keeps that pose all along it
    w, g, poly, grid = songdo
    for sid, heading in (("M,G83_40,Port7", 3), ("M,Port7,G83_40", 183)):
        state = g[sid]; area = state_area(w, state)
        (x0, y0), (x1, y1) = (w.nodes[n] for n in state.nodes)
        assert state.heading == heading
        assert all(area.buffer(1e-9).covers(footprint_polygon(w, x, y, heading)) for x, y in ((x0, y0), (x1, y1)))
        # a square moved along a line sweeps its own area plus (the distance x its width across the line)
        psi = math.radians(heading) - math.atan2(y1 - y0, x1 - x0)
        width_across = 0.5 * (abs(math.cos(psi)) + abs(math.sin(psi)))
        assert area.area == pytest.approx(0.25 + math.dist((x0, y0), (x1, y1)) * width_across, rel=1e-9)

def test_turn_sweep_of_a_square_is_a_360_gon(tmp_path):
    # a square turned through 90 degrees sweeps a regular 360-gon on its corner circle (radius^2 = 0.125),
    # area 180 r^2 sin(1 deg); other steps or another length of turn give another area
    w = tiny_world(tmp_path); g = build_state_graph(w)
    gon = 180 * 0.125 * math.sin(math.radians(1))
    assert state_area(w, g["R,B,0,90"]).area == pytest.approx(gon, rel=1e-9)
    assert state_area(w, g["R,B,0,180"]).area == pytest.approx(gon, rel=1e-9)      # a square looks the same every 90 degrees

def test_turn_sweeps_exactly_the_turn(songdo):
    # R,G83_40,0,3 sweeps 0 -> 1 -> 2 -> 3 degrees: the footprints at -1 and 4 degrees are not inside it
    w, g, poly, grid = songdo
    x, y = w.nodes["G83_40"]
    swept = state_area(w, g["R,G83_40,0,3"]).buffer(1e-9)
    assert [swept.covers(footprint_polygon(w, x, y, h)) for h in range(-1, 5)] == [False, True, True, True, True, False]

def test_turn_takes_the_short_way(tmp_path):
    # 350 -> 10 and 10 -> 350 both turn 20 degrees through 0, never the long way round past 180;
    # the vehicle is not square, so that a footprint at 90 differs from one at 0
    w = dataclasses.replace(tiny_world(tmp_path), footprint=Footprint(1.0, 0.4))
    for from_heading, to_heading in ((350, 10), (10, 350)):
        swept = state_area(w, _turn(from_heading, to_heading)).buffer(1e-9)
        covered = [swept.covers(footprint_polygon(w, 1, 0, h)) for h in (349, 350, 355, 0, 5, 10, 11, 90)]
        assert covered == [False, True, True, True, True, True, False, False]

def test_turn_of_exactly_180_sweeps_the_half_turn(tmp_path):
    # which way round the 180 goes cannot change the area of a rectangle; half a turn sweeps the full
    # circle of its half-diagonal
    w = dataclasses.replace(tiny_world(tmp_path), footprint=Footprint(1.0, 0.4))
    disc = math.pi * (0.5 ** 2 + 0.2 ** 2)
    assert state_area(w, _turn(0, 180)).area == pytest.approx(disc, rel=1e-4)
    assert state_area(w, _turn(180, 0)).area == pytest.approx(disc, rel=1e-4)

def test_same_turn_at_two_nodes_is_the_same_shape(tmp_path):
    # the sweep is worked out once at the origin and then moved to the node
    w = tiny_world(tmp_path); g = build_state_graph(w)
    at_a, at_b = state_area(w, g["R,A,0,180"]), state_area(w, g["R,B,0,180"])
    assert at_a.centroid.coords[0] == pytest.approx((0, 0)) and at_b.centroid.coords[0] == pytest.approx((1, 0))
    assert affinity.translate(at_a, 1, 0).equals_exact(at_b, 1e-12)

def test_state_area_rejects_bad_states(tmp_path):
    w = tiny_world(tmp_path); g = build_state_graph(w)
    with pytest.raises(ValueError, match="unknown state kind 'X'"):
        state_area(w, dataclasses.replace(g["S,A,0"], kind="X"))
    with pytest.raises(ValueError, match="two different headings"):
        state_area(w, dataclasses.replace(g["R,B,0,90"], to_heading=360))

def test_touching_counts_as_a_collision(tmp_path):
    # a 1 m vehicle on the 1 m grid: footprints at neighbouring nodes share an edge (A-B, B-C) or just a corner
    # (A-C) at every axis heading; they share no node, so the cell rule misses them
    w = tiny_world(tmp_path, vehicle_m=1.0, collision="polygon"); g = build_state_graph(w)
    c = poly_conflicts(w, g)
    stops = [s for s, st in g.items() if st.kind == "S"]
    assert all(b in c[a] for a in stops for b in stops if a != b)
    assert "S,B,0" not in grid_conflicts(w, g)["S,A,0"]

def test_conflicts_cover_every_state_of_the_graph(tmp_path):
    # only the graph given counts: S,C,90 has nothing to collide with here, yet it is still a key
    w = tiny_world(tmp_path, collision="polygon"); full = build_state_graph(w)
    part = {sid: full[sid] for sid in ("S,C,90", "M,A,B", "S,A,0")}       # not in sorted order
    c = poly_conflicts(w, part)
    assert c == {"S,A,0": {"M,A,B"}, "M,A,B": {"S,A,0"}, "S,C,90": set()}
    assert list(c) == list(part) and all(type(v) is frozenset for v in c.values())
    assert poly_conflicts(w, {}) == {}

def test_songdo_conflict_structure(songdo):
    w, g, poly, grid = songdo
    assert list(poly) == list(g)                                # a key for every state, in graph order
    assert all(type(v) is frozenset for v in poly.values())
    assert [s for s, v in poly.items() if s in v] == []         # never itself
    assert [(a, b) for a, bs in poly.items() for b in bs if a not in poly[b]] == []   # symmetric
    # an L takes the space of the S with the same pose: the same area, so apart from the pair itself the same conflicts
    ls = [s for s in g if s.startswith("L,")]
    assert len(ls) == 20 and all(state_area(w, g[s]).equals_exact(state_area(w, g[geom_key(s)]), 0) for s in ls)
    assert [s for s in ls if poly[s] | {s} != poly[geom_key(s)] | {geom_key(s)}] == []

def test_songdo_pairs_beyond_the_cell_rule(songdo):
    # the polygon rule keeps every cell-rule pair and adds 344: 351,287 against 350,943 unordered pairs. The
    # counts come from a separate computation: elementwise shapely.intersects over all bounding-box candidates
    w, g, poly, grid = songdo
    assert sum(len(v) for v in poly.values()) == 2 * 351287
    assert all(grid[s] <= poly[s] for s in g)
    extra = {(a, b) for a, bs in poly.items() for b in bs - grid[a] if a < b}
    assert len(extra) == 344
    at_port = lambda sid: {n for n in g[sid].nodes if n in w.port_nodes}
    # each is a port state against a grid state, and only Port6 and Port7 have any: they lie 0.547 m and 0.478 m
    # from their access nodes, closer than two turning vehicles reach (2 x 0.354 m); the other ports lie > 0.99 m
    assert all(bool(at_port(a)) != bool(at_port(b)) for a, b in extra)
    assert {n for a, b in extra for n in at_port(a) | at_port(b)} == {"Port6", "Port7"}
