"""One call from a map config to the world, its state graph and its conflict sets."""
from dataclasses import dataclass
from pathlib import Path

from agvtb.world.collision_grid import check_grid_rule, grid_conflicts
from agvtb.world.collision_poly import poly_conflicts
from agvtb.world.config import load_map_config
from agvtb.world.mapfile import load_xmsmap
from agvtb.world.model import World, build_world
from agvtb.world.profile import load_collision_profile
from agvtb.world.states import Conflicts, StateGraph, build_state_graph


@dataclass(frozen=True)
class LoadedWorld:
    """A map's world with its state graph and the conflict sets of its states."""

    world: World
    graph: StateGraph
    conflicts: Conflicts


def load_world(config_path: str | Path) -> LoadedWorld:
    """The World, StateGraph and Conflicts of a map config, built as its collision mode says.

    grid: rule-built states, the cell rule checked (GridRuleError if the vehicle is too big for it),
    then the cell conflicts. polygon: rule-built states and polygon conflicts, with no cell-rule
    check, which songdo_grid_bi fails by design. profile: states and conflicts as the shipped
    profile gives them (ProfileMismatchError if it was not made for this map file).
    """
    cfg = load_map_config(config_path)
    mf = load_xmsmap(cfg.map_path)
    world = build_world(cfg, mf)
    if cfg.collision == "grid":
        graph = build_state_graph(world)
        check_grid_rule(world)
        conflicts = grid_conflicts(world, graph)
    elif cfg.collision == "polygon":
        graph = build_state_graph(world)
        conflicts = poly_conflicts(world, graph)
    else:  # "profile": load_map_config allows no other mode, and a profile path comes with it
        graph, conflicts = load_collision_profile(world, mf, cfg.profile_path)
    return LoadedWorld(world, graph, conflicts)
