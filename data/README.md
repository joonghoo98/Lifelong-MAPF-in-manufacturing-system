# data

Map data the configs and tests read, copied byte for byte so that a clone runs on its own.

- `grid_map/` holds copies of files from the `grid_map` folder next to `agv_testbed`.
- `daim/` holds copies of files from the DAIM simulator repo (`DAIM_simulation/DAIM_simulation`), branch `Planning`, commit `9c04deb`. Each file is tracked and clean there.
- `sources.json` lists every file: where it came from, its size and sha256, and for DAIM files the git blob id. `tests/world/test_data_files.py` checks the bytes against it.
- `.gitattributes` stores `data/**` as binary (`-text`), so git never rewrites line endings and the sha256 holds on every system.
