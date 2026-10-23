"""The repo carries every map file it reads, under data/, byte for byte as data/sources.json records them."""
import hashlib
import json

from agvtb.world.config import load_map_config
from helpers import CONFIGS, DAIM, GRID_MAP, REPO

DATA = REPO / "data"


def test_every_map_config_reads_files_inside_the_repo():
    configs = sorted(CONFIGS.rglob("*.json"))
    assert configs
    for path in configs:
        config = load_map_config(path)
        for f in (config.map_path, config.profile_path):
            if f is not None:
                assert f.resolve().is_relative_to(DATA), (path.name, f)
                assert f.is_file(), (path.name, f)


def test_the_tests_read_the_map_data_inside_the_repo():
    assert GRID_MAP.resolve().is_relative_to(DATA)
    assert DAIM.resolve().is_relative_to(DATA)


def test_the_data_files_are_the_recorded_bytes():
    files = json.loads((DATA / "sources.json").read_text(encoding="utf-8"))["files"]
    on_disk = {p.relative_to(DATA).as_posix() for p in DATA.rglob("*") if p.is_file()}
    assert {f["path"] for f in files} == on_disk - {"sources.json", "README.md"}
    for f in files:
        assert hashlib.sha256((DATA / f["path"]).read_bytes()).hexdigest() == f["sha256"], f["path"]
