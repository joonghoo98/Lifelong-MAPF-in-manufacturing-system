"""Shared test helpers: repo paths, the tiny hand-made map, link maps, and World builders."""
import hashlib
import json
from pathlib import Path

from agvtb.world.config import load_map_config
from agvtb.world.loader import LoadedWorld, load_world
from agvtb.world.mapfile import load_xmsmap
from agvtb.world.model import World, build_world
from agvtb.world.states import Conflicts, State, StateGraph

REPO = Path(__file__).resolve().parents[1]
GRID_MAP = REPO / "data" / "grid_map"  # copies of MAPF/grid_map files (data/sources.json)
CONFIGS = REPO / "configs" / "maps"
DAIM_CONFIGS = CONFIGS / "daim"  # configs of the nine DAIM songdo maps; their map files stay in DAIM_ENGINES
DAIM = REPO / "data" / "daim"  # copies of DAIM_simulation/DAIM_simulation files at 9c04deb (data/sources.json)
DAIM_ENGINES = DAIM / "experiments" / "results" / "songdo_placement_engines"


def tiny_map_data(vehicle_m: float = 0.5) -> dict:
    """xMSmap-v4.0 dict of the tiny map: nodes A(0,0) B(1,0) C(1,1), port P1 at C.

    Segments A<->B and B<->C (ids 1-4), all speed 0.5. Values are in metres.
    """
    def segment(seg_id, start, end):
        return {"id": seg_id, "startNodeId": start, "endNodeId": end, "speed": 0.5}

    return {
        "schema": "xmsmap-v4.0",
        "nodes": [
            {"id": "A", "x": 0.0, "y": 0.0},
            {"id": "B", "x": 1.0, "y": 0.0},
            {"id": "C", "x": 1.0, "y": 1.0},
        ],
        "segments": [
            segment(1, "A", "B"),
            segment(2, "B", "A"),
            segment(3, "B", "C"),
            segment(4, "C", "B"),
        ],
        "ports": [{"id": "P1", "nodeId": "C"}],
        "vehicleModels": [{
            "dimension": {"length": vehicle_m, "width": vehicle_m},
            "safetyMargin": {"front": 0.0, "rear": 0.0, "left": 0.0, "right": 0.0},
        }],
    }


def tiny_config(tmp_path, vehicle_m: float = 0.5, collision: str = "grid") -> Path:
    """Write the tiny map and a config for it into tmp_path; return the config path."""
    (tmp_path / "tiny_map.json").write_text(
        json.dumps(tiny_map_data(vehicle_m)), encoding="utf-8"
    )
    config = {
        "name": "tiny",
        "map": "tiny_map.json",  # relative to the config's folder
        "units_per_meter": 1.0,
        "accel_mps2": 0.5,
        "decel_mps2": 0.5,
        "rotation_s_per_90deg": 2.5,
        "dwell_s": 10.0,
        "collision": collision,
    }
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def link_config(
    tmp_path,
    nodes: dict[str, tuple[float, float]],
    links: list[tuple[str, str]],
    ports: list[str],
    name: str = "links",
    info: dict | None = None,
) -> Path:
    """Write a map of two-way links and a grid config for it into tmp_path; return the config path.

    nodes maps id -> (x, y) in metres. Each link (a, b) becomes the segments a -> b and b -> a, numbered
    1, 2, 3, ... in link order, all at speed 1.0. The vehicle is 0.5 m square with no margin. info,
    when given, becomes the map's "info" block.
    """
    segments = []
    for a, b in links:
        for start, end in ((a, b), (b, a)):
            segments.append({"id": len(segments) + 1, "startNodeId": start, "endNodeId": end, "speed": 1.0})
    data = {
        "schema": "xmsmap-v4.0",
        "nodes": [{"id": node, "x": x, "y": y} for node, (x, y) in nodes.items()],
        "segments": segments,
        "ports": [{"id": f"P{i}", "nodeId": port} for i, port in enumerate(ports)],
        "vehicleModels": [{
            "dimension": {"length": 0.5, "width": 0.5},
            "safetyMargin": {"front": 0.0, "rear": 0.0, "left": 0.0, "right": 0.0},
        }],
    }
    if info is not None:
        data["info"] = info
    (tmp_path / f"{name}_map.json").write_text(json.dumps(data), encoding="utf-8")
    config = {
        "name": name,
        "map": f"{name}_map.json",  # relative to the config's folder
        "units_per_meter": 1.0,
        "accel_mps2": 0.5,
        "decel_mps2": 0.5,
        "rotation_s_per_90deg": 2.5,
        "dwell_s": 10.0,
        "collision": "grid",
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def tiny_world(tmp_path, vehicle_m: float = 0.5, collision: str = "grid") -> World:
    """World of the tiny map (see tiny_config)."""
    return _world_from_config(tiny_config(tmp_path, vehicle_m, collision))


def world_of(name: str) -> World:
    """World of one of the four real maps, from configs/maps/<name>.json."""
    return _world_from_config(CONFIGS / f"{name}.json")


def daim_world(name: str) -> World:
    """World of one DAIM songdo map, from configs/maps/daim/<name>.json."""
    return world_of_config(DAIM_CONFIGS / f"{name}.json")


def daim_map_info(name: str) -> dict:
    """The "info" block of the map file that configs/maps/daim/<name>.json points at."""
    return load_xmsmap(load_map_config(DAIM_CONFIGS / f"{name}.json").map_path).info


def world_of_config(config_path: Path) -> World:
    """World of one config. No conflicts are built, so a map the cell rule would refuse still loads."""
    cfg = load_map_config(config_path)
    return build_world(cfg, load_xmsmap(cfg.map_path))


def _world_from_config(config_path: Path) -> World:
    """M1's name for world_of_config, kept for the callers that use it."""
    return world_of_config(config_path)


def fake_room_with_spurs(
    move_s: float = 2.0, dwell_s: float = 10.0
) -> tuple[StateGraph, Conflicts, tuple[str, str]]:
    """room_with_spurs in our types: one heading (0) per node, no R; DAIM's affects made symmetric, with L."""
    edges = [(f"c{x}{y}", f"c{x + 1}{y}") for x in range(5) for y in range(3)]
    edges += [(f"c{x}{y}", f"c{x}{y + 1}") for x in range(6) for y in range(2)]
    edges += [("c02", "c03"), ("c22", "c23"), ("c42", "c43")]
    ports = ("c00", "c50")
    nbr = {}
    for a, b in edges:
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    touching = {n: set() for n in nbr}
    for a in nbr:
        for b in nbr[a]:
            touching[a].add(f"M,{a},{b}")
            touching[b].add(f"M,{a},{b}")
    graph = {}
    for a in nbr:
        for b in sorted(nbr[a]):
            sid = f"M,{a},{b}"
            graph[sid] = State(sid, "M", (a, b), 0, None, None, move_s, (f"S,{b},0",))
    for n in nbr:
        nxt = [f"M,{n},{b}" for b in nbr[n]] + ([f"L,{n},0"] if n in ports else [])
        graph[f"S,{n},0"] = State(f"S,{n},0", "S", (n,), 0, None, None, None, tuple(sorted(nxt)))
    for p in ports:
        graph[f"L,{p},0"] = State(f"L,{p},0", "L", (p,), 0, None, None, dwell_s, (f"S,{p},0",))
    graph = {sid: graph[sid] for sid in sorted(graph)}
    pairs = set()
    for n in nbr:  # S,n,0 - every move that takes n (DAIM: S affect = touching moves)
        for m in touching[n]:
            pairs.add((f"S,{n},0", m))
    for a in nbr:  # M,a,b - the other moves that take a or b (DAIM: M affect also lists S,a,0 and S,b,0)
        for b in nbr[a]:
            for m in (touching[a] | touching[b]) - {f"M,{a},{b}"}:
                pairs.add((f"M,{a},{b}", m))
    for p in ports:  # L,p,0 is where S,p,0 is: it collides with S,p,0 and with all that S collides with
        pairs.add((f"L,{p},0", f"S,{p},0"))
        for x, y in list(pairs):
            if x == f"S,{p},0" and not y.startswith("L,"):
                pairs.add((f"L,{p},0", y))
            if y == f"S,{p},0" and not x.startswith("L,"):
                pairs.add((x, f"L,{p},0"))
    conflicts = {sid: set() for sid in graph}
    for x, y in pairs:
        conflicts[x].add(y)
        conflicts[y].add(x)
    return graph, {sid: frozenset(v) for sid, v in conflicts.items()}, ports


SPUR_ROOM = ([(f"c{x}{y}", f"c{x + 1}{y}") for x in range(3) for y in range(2)]
             + [(f"c{x}0", f"c{x}1") for x in range(4)] + [("c11", "c12"), ("c12", "c13"), ("c31", "c32")])


def spur_room_world(tmp_path) -> LoadedWorld:
    """A 4 x 2 room c00..c31 with ports c00 and c30, and two spurs: c11-c12-c13 (c13 a dead end) and c31-c32."""
    nodes = {n: (float(n[1]), float(n[2])) for link in SPUR_ROOM for n in link}
    return load_world(link_config(tmp_path, nodes, SPUR_ROOM, ["c00", "c30"], "spur_room"))


POCKET = ([(f"r0c{c}", f"r0c{c + 1}") for c in range(4)] + [("r0c1", "r1c1"), ("r1c1", "r2c1"), ("r0c3", "r1c3"),
          ("r1c3", "r2c3"), ("r2c1", "r2c2"), ("r2c2", "r2c3")])


def pocket_world(tmp_path) -> LoadedWorld:
    """R07's pocket map: corridor r0c0..r0c4 with ports at its ends, sidings r1c1 and r1c3, and the pocket r2c1..r2c3
    that only the two sidings join to the corridor (R07_files/review_checks/pocket_example.py)."""
    nodes = {n: (float(n[3]), float(n[1])) for link in POCKET for n in link}
    return load_world(link_config(tmp_path, nodes, POCKET, ["r0c0", "r0c4"], "pocket"))


def path_digest(path) -> str:
    """sha256 of the JSON of a path [(state, time), ...], as the evidence scripts print it."""
    return hashlib.sha256(json.dumps([[s, t] for s, t in path]).encode()).hexdigest()


def timelines_digest(timelines) -> str:
    """sha256 of the JSON of {vehicle: path}, vehicles in order and named as strings."""
    data = {str(v): [[s, t] for s, t in timelines[v]] for v in sorted(timelines)}
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def _undirected(edges):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return {v: tuple(sorted(us)) for v, us in sorted(adj.items())}


_LOLLIPOP = [("0", "1"), ("1", "2"), ("2", "3"), ("3", "0"), ("0", "4"), ("4", "5"), ("5", "6")]
SMALL_GRAPHS = {  # name -> (adjacency, ports); R26 verdicts: plans/2026-10-24-m6a-fallback-mode_files/ref_cases.out.txt
    "lollipop": (_undirected(_LOLLIPOP), ("2", "6")),
    "cutport": (_undirected(_LOLLIPOP), ("2", "5")),
    "spider": (_undirected([("0", "1"), ("1", "2"), ("2", "3"), ("0", "4"), ("4", "5"), ("5", "6"), ("0", "7")]),
               ("3", "6")),
    "comb": (_undirected([("0", "1"), ("1", "2"), ("2", "3"), ("3", "0"), ("0", "4"), ("4", "5"), ("5", "6"),
                          ("2", "7"), ("7", "8")]), ("6",)),
    "longlane": (_undirected([("0", "1"), ("1", "2"), ("2", "0"), ("0", "3"), ("3", "4"), ("4", "5"), ("5", "6"),
                              ("6", "7"), ("7", "8")]), ("1", "5")),
}
