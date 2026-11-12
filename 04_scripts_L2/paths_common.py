# -*- coding: utf-8 -*-
"""Shared path resolution for the paper-level (04_scripts_L2) scripts.

One import serves every script that needs data/figure/engine locations, in
either supported layout, so no script hard-codes a machine-specific path:

* release repository layout (GitHub / reproducibility pack)::

      <root>/01_paper   02_figures  03_data  04_scripts_L2  05_predecessor_assets
      <root>/06_engines/{z3_hexagonal_P0, z3_31212, z3_4612, z3_488, z4_3464,
                        z4_kagome, z4_square, z5_*, z6_triangular, penrose}

* development project layout::

      <root>/L2_metric_random_tiling/{scripts, data, figs}
      <root>/z3_hexagonal/<hc-leaf from dev_name_map.json>, z3_31212, ...

The layout is detected once by walking up from this file and probing for the
marker directories.  Every consumer resolves locations through this module
only; nothing below may assume the current working directory.
"""
import json
from pathlib import Path

_REPO_MARKER = "06_engines"
_PROJ_MARKER = "L2_metric_random_tiling"


def _detect():
    p = Path(__file__).resolve().parent
    for _ in range(6):
        if (p / _REPO_MARKER).is_dir():
            return p, "repo"
        if (p / _PROJ_MARKER).is_dir():
            return p, "proj"
        if p.parent == p:
            break
        p = p.parent
    raise RuntimeError(
        "paths_common: cannot locate the repository root (looking for "
        f"'{_REPO_MARKER}' or '{_PROJ_MARKER}') above {Path(__file__)}")


ROOT, LAYOUT = _detect()

if LAYOUT == "repo":
    DATA = ROOT / "03_data" / "uniform"       # ensemble / figure data (CSV, NPZ)
    DATA_ROOT = ROOT / "03_data"              # all delivered data
    FIGURES = ROOT / "02_figures"             # delivered PDF/PNG figures
    PAPER = ROOT / "01_paper"                 # paper PDFs (letter + SI + RS)
    SCRIPTS = ROOT / "04_scripts_L2"          # this folder
    PREDECESSOR = ROOT / "05_predecessor_assets"
    ENGINES = ROOT / "06_engines"             # engine root
    HC = ENGINES / "z3_hexagonal_P0"          # honeycomb engine (flattened)
else:
    L2 = ROOT / _PROJ_MARKER
    DATA = L2 / "data" / "uniform"
    DATA_ROOT = L2 / "data"
    FIGURES = L2 / "figs"
    PAPER = L2
    SCRIPTS = L2 / "scripts"
    PREDECESSOR = None                        # only exists in the release layout
    ENGINES = ROOT                            # engines sit at the project root
    # the development tree keeps its original (non-ASCII) honeycomb engine
    # directory name; the leaf name lives in dev_name_map.json (data file),
    # so this module itself stays pure ASCII
    _dev = json.loads((L2 / "scripts" / "dev_name_map.json").read_text(encoding="utf-8"))
    HC = ROOT / "z3_hexagonal" / _dev["_hc_leaf_name"]

#: engine short name -> directory, valid in both layouts
_ENGINES = {
    "hc": HC,
    "z3_hexagonal_P0": HC,
    "z3_31212": ENGINES / "z3_31212",
    "z3_4612": ENGINES / "z3_4612",
    "z3_488": ENGINES / "z3_488",
    "z4_3464": ENGINES / "z4_3464",
    "z4_kagome": ENGINES / "z4_kagome",
    "z4_square": ENGINES / "z4_square",
    "z6_triangular": ENGINES / "z6_triangular",
    "penrose": ENGINES / "penrose",
    "z5_snubhex": ENGINES / "z5_snubhex",
    "z5_snubsquare": ENGINES / "z5_snubsquare",
    "z5_33344": ENGINES / "z5_33344",
}


def engine(name):
    """Engine directory by short name; raises KeyError with the known names."""
    return _ENGINES[name]


_DEV_MAP = None


def fname(repo_name):
    """Map a repository (ASCII) filename to the on-disk name of this layout.

    The release repository renamed engine data/module files to ASCII names;
    the development tree kept the original
    names.  Scripts always write the ASCII name and call this helper for the
    actual file, so one source runs in both layouts.  The ASCII -> original
    mapping lives in the data file ``dev_name_map.json`` next to this module
    (JSON, not code, so script sources stay free of non-ASCII characters).
    Unknown names are returned unchanged.
    """
    if LAYOUT == "repo":
        return repo_name
    global _DEV_MAP
    if _DEV_MAP is None:
        import json
        _DEV_MAP = json.loads(
            (SCRIPTS / "dev_name_map.json").read_text(encoding="utf-8"))
    return _DEV_MAP.get(repo_name, repo_name)
