"""P0-M0 three-reference data loading (shared module).

Three-reference set (the organizing principle of all figures):
  REF-C  crystal: pure-222 honeycomb disk n = 6 (fixed extract_geometry layout
         convention);
  PC     pseudocrystal: binary {222,223} alloy n = 6 random solution (same as
         above);
  REF-A  true amorphous: deliverable v6 (beta-annealed by
         P0_REFA_amorphous_annealing.py, CRN with 294 vertices, coordinates +
         bond table; the bond table must be the deliverable's, since
         distance-threshold reconstruction is ambiguous at high density).

Common convention: coordinates divided by each system's mean bond length =>
r in units of the mean bond length (comparable across the three references).
Geometry-fix note: the extract_geometry double-add bug is fixed; the old version
produced floating layouts for n >= 3 (split vertices, non-tree edges not
coincident); counting-type conclusions are unaffected; D1/D1b/D3 and this module
all use the fixed coordinates.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pseudocrystal_lib as pl

N_RING = 6
PC_ALPHABET = ('222', '223')
PC_SEED = 7206            # same family as the D1 size series (rng 7000+100k+n; k=2, n=6)
REFA_COORD = ROOT / "P0_REFA_true_amorphous_coordinates.csv"
REFA_BONDS = ROOT / "P0_REFA_true_amorphous_bond_table.csv"


def load_refc():
    lib = pl.build_library(('222',))
    g = pl.crystal_disk_points(N_RING)
    pts = g['metal_nodes']
    bonds = np.array([[e[0], e[1]] for e in g['edges']], dtype=int)
    # edge endpoints are coordinate keys; map them to the metal_nodes row order
    key2idx = {k: i for i, k in enumerate(g['node_keys'])}
    bonds = np.array([[key2idx[e[0]], key2idx[e[1]]] for e in g['edges']], dtype=int)
    return pts, bonds


def load_pc():
    lib = pl.build_library(PC_ALPHABET)
    res = pl.solve_tessellation(N_RING, lib, periodic=False,
                                rng=__import__('random').Random(PC_SEED))
    sol, _ = res
    g = pl.extract_geometry(sol, lib)
    pts = g['metal_nodes']
    key2idx = {k: i for i, k in enumerate(g['node_keys'])}
    bonds = np.array([[key2idx[e[0]], key2idx[e[1]]] for e in g['edges']], dtype=int)
    return pts, bonds


def load_refa():
    df = pd.read_csv(REFA_COORD)
    pts = df[['X_Coordinate', 'Y_Coordinate']].to_numpy(dtype=float)
    eb = pd.read_csv(REFA_BONDS)
    bonds = eb[['i', 'j']].to_numpy(dtype=int)
    return pts, bonds


def load_all():
    """Returns {name: (pts, bonds, bond_len_mean)}. Coordinates normalized to
    mean bond length = 1."""
    out = {}
    for name, (pts, bonds) in {
        'REF-C': load_refc(), 'PC': load_pc(), 'REF-A': load_refa()
    }.items():
        d = np.linalg.norm(pts[bonds[:, 0]] - pts[bonds[:, 1]], axis=1)
        bl = float(d.mean())
        out[name] = (pts / bl, bonds, bl)
    return out


# ----------------------------------------------------------------------
# Ensemble loading (M0 optional: multi-configuration averaging to suppress
# single-configuration statistical fluctuations)
# ----------------------------------------------------------------------

PC_SEEDS_ENSEMBLE = [7006, 7106, 7206, 7306, 7406]   # same family as the D1 size series
SERIES_DIR = ROOT.parent / "_temp" / "M0_REFA_series" / "series"
REFA_EXTRA = ['s6_seed21', 's6_seed33']              # v6 body = s6_seed15


def load_pc_configs(seeds=None):
    """n = 6 binary {222,223}, multiple configurations."""
    seeds = seeds or PC_SEEDS_ENSEMBLE
    lib = pl.build_library(PC_ALPHABET)
    out = []
    for sd in seeds:
        sol, _ = pl.solve_tessellation(N_RING, lib, periodic=False,
                                       rng=__import__('random').Random(sd))
        g = pl.extract_geometry(sol, lib)
        pts = g['metal_nodes']
        key2idx = {k: i for i, k in enumerate(g['node_keys'])}
        bonds = np.array([[key2idx[e[0]], key2idx[e[1]]] for e in g['edges']],
                         dtype=int)
        out.append((pts, bonds))
    return out


def load_refa_configs():
    """REF-A ensemble: deliverable v6 (s6_seed15) + two same-size chains from
    the server size series (seed21/33).
    Degrades to a single configuration (v6 only) if the series directory is
    missing."""
    out = [load_refa()]
    for tag in REFA_EXTRA:
        d = SERIES_DIR / tag
        if not d.exists():
            continue
        df = pd.read_csv(d / "coords.csv")
        pts = df[['X_Coordinate', 'Y_Coordinate']].to_numpy(dtype=float)
        eb = pd.read_csv(d / "bonds_idx.csv")
        out.append((pts, eb[['i', 'j']].to_numpy(dtype=int)))
    return out


def load_all_ensembles():
    """Returns {name: [(pts, bonds, bl), ...]} (coordinates normalized to mean
    bond length = 1).

    REF-C single configuration (unique); PC 5 configurations; REF-A 3
    (1 if the series directory is missing).
    """
    out = {}
    raw = {'REF-C': [load_refc()], 'PC': load_pc_configs(),
           'REF-A': load_refa_configs()}
    for name, cfgs in raw.items():
        lst = []
        for pts, bonds in cfgs:
            d = np.linalg.norm(pts[bonds[:, 0]] - pts[bonds[:, 1]], axis=1)
            bl = float(d.mean())
            lst.append((pts / bl, bonds, bl))
        out[name] = lst
    return out


if __name__ == "__main__":
    for name, (pts, bonds, bl) in load_all().items():
        d = np.linalg.norm(pts[bonds[:, 0]] - pts[bonds[:, 1]], axis=1)
        deg = np.bincount(bonds.ravel(), minlength=len(pts))
        print(f"{name}: vertices {len(pts)} bonds {len(bonds)} "
              f"<bond length>={bl:.3f} normalized bond length {d.min():.3f}-{d.max():.3f} "
              f"degree distribution {dict(zip(*[a.tolist() for a in np.unique(deg, return_counts=True)]))}")
