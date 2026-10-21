import re

import pytest

from agvtb.sidings.choice import SidingChoice, candidates_of, parse_siding_choice
from agvtb.sidings.lattice import LatticeError
from agvtb.world.config import load_map_config
from agvtb.world.loader import load_world
from helpers import CONFIGS, DAIM_CONFIGS, daim_map_info

KAIROS_12 = ["na.17", "na.196", "na.197", "na.198", "na.202", "na.203", "na.204", "na.40", "na.41", "na.42",
             "na.59", "na.60"]  # facts.out.txt (c)


@pytest.mark.parametrize("text, choice", [
    ("grid-bridge-s00", SidingChoice("grid", mode="bridge", shift=(0, 0))),
    ("grid-legacy-s11", SidingChoice("grid", mode="legacy", shift=(1, 1))),
    ("strict", SidingChoice("strict")),
    ("strict-drop:na.34,na.53", SidingChoice("strict", drop=("na.34", "na.53"))),
    ("recorded", SidingChoice("recorded")),
    ("list:c03,c23", SidingChoice("list", nodes=("c03", "c23"))),
])
def test_parse_and_name(text, choice):
    assert parse_siding_choice(text) == choice and choice.name == text


@pytest.mark.parametrize("text", ["", "grid", "grid-bridge-s02", "grid-fast-s00", "strict-drop:",
                                  "strict-drop:a,,b", "list:", "random"])
def test_rejects_unknown_choices(text):
    with pytest.raises(ValueError, match=re.escape(repr(text))):
        parse_siding_choice(text)


def test_variant_ga_candidates():  # facts.out.txt (c)
    mini = load_world(CONFIGS / "mini_deadend.json")
    assert len(candidates_of(mini, parse_siding_choice("grid-bridge-s00"))) == 37
    kairos = load_world(CONFIGS / "KAIROS_AMR.json")
    assert candidates_of(kairos, parse_siding_choice("strict-drop:na.34,na.53")) == KAIROS_12
    config_choice = load_map_config(CONFIGS / "KAIROS_AMR.json").sidings  # Claude's call 12: the config's fixed list
    assert candidates_of(kairos, parse_siding_choice(config_choice)) == KAIROS_12


def test_recorded_is_the_map_files_list_and_equals_ga_on_manual():  # R06 §4.3
    lw = load_world(DAIM_CONFIGS / "grid_songdo_manual.json")
    recorded = candidates_of(lw, parse_siding_choice("recorded"))
    assert recorded == sorted({str(s) for s in daim_map_info("grid_songdo_manual")["sidings"]})
    assert len(recorded) == 214 and recorded == candidates_of(lw, parse_siding_choice("grid-bridge-s00"))


def test_choice_errors_name_the_map_or_the_ids():
    kairos = load_world(CONFIGS / "KAIROS_AMR.json")
    with pytest.raises(LatticeError):
        candidates_of(kairos, parse_siding_choice("grid-bridge-s00"))
    mini = load_world(CONFIGS / "mini_deadend.json")
    with pytest.raises(ValueError, match=r"\bzz\b"):
        candidates_of(mini, parse_siding_choice("strict-drop:zz"))
    with pytest.raises(ValueError, match="mini_deadend"):
        candidates_of(mini, parse_siding_choice("recorded"))
    assert candidates_of(mini, parse_siding_choice("list:G2_2,zz,G2_2")) == ["G2_2", "zz"]
