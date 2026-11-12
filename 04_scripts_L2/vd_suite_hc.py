# -*- coding: utf-8 -*-
"""vd_suite_hc.py -- configuration-bootstrap intervals for the halo/core ratio.

The suite archives carried bootstrap intervals for f_coh and r only; halo/core
was computed on the ensemble-mean |F|^2 map and stored without an interval.
This script recomputes, per configuration, the same window sums (core:
|q-G| <= 0.22; halo: 0.22 <= |q-G| <= 1.10 around each of the six spots) and
bootstraps the ratio H/C over the 20 configurations (B = 200).

Writes:
  data/uniform/spots_<tag>.csv   (adds hc_lo / hc_hi columns)
  data/uniform/ladder_ext.csv    (adds hc_lo / hc_hi for the extension rungs)
Gate: the recomputed ensemble mean must reproduce the archived halo/core.

usage: python vd_suite_hc.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "external"))
import vd_ladder_ext as LX                      # noqa: E402  (reference builders)
import vd_suite3 as S                           # noqa: E402
from vd_patch import embed_config               # noqa: E402
import pseudocrystal_lib as pl                  # noqa: E402

DATA = PROJ / "L2_metric_random_tiling" / "data" / "uniform"
RAW = DATA / "raw"
Q_MAX, Q_PIX = S.Q_MAX, S.Q_PIX
B = 200


def per_config_hc(ref, cfgs):
    G, T, bonds = ref["G"], ref["T"], ref["bonds"]
    pix, cc, dqq = ref["pix"], ref["cc"], ref["dqq"]
    C, H = [], []
    for ic in range(len(cfgs)):
        pts, _cv = embed_config(G, T, cfgs[ic])
        bl = float(np.mean([np.hypot(*(pts[a] - pts[b])) for a, b in bonds]))
        I, _ = pl.structure_factor(pts / bl, q_max=Q_MAX, q_pixels=Q_PIX)
        c = h = 0.0
        for kk, gx, gy, ix, iy in pix:
            WIN = S.WIN
            y0, y1 = max(int(iy - WIN / dqq), 0), min(int(iy + WIN / dqq) + 1, Q_PIX)
            x0, x1 = max(int(ix - WIN / dqq), 0), min(int(ix + WIN / dqq) + 1, Q_PIX)
            sub = I[y0:y1, x0:x1]
            yy2, xx2 = np.mgrid[y0:y1, x0:x1]
            rrc = np.hypot((xx2 - cc) * dqq - gx, (yy2 - cc) * dqq - gy)
            c += float(sub[rrc <= 0.22].sum())
            h += float(sub[(rrc >= 0.22) & (rrc <= 1.10)].sum())
        C.append(c)
        H.append(h)
    return np.array(C), np.array(H)


def boot_ratio(C, H, rng):
    n = len(C)
    idx = rng.integers(0, n, (B, n))
    r = H[idx].sum(axis=1) / C[idx].sum(axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main():
    ref = LX.build_reference(13, 7)
    rng = np.random.default_rng(20261014)
    print("=== suite domains ===", flush=True)
    for tag in ("b12", "a123", "a234", "a345", "a456", "a567"):
        d = np.load(RAW / f"torus13_{tag}.npz")
        cfgs = d["cfgs"].astype(np.int64)
        C, H = per_config_hc(ref, cfgs)
        hc = H.sum() / C.sum()
        lo, hi = boot_ratio(C, H, rng)
        f = DATA / f"spots_{tag}.csv"
        df = pd.read_csv(f)
        arch = df.halo.sum() / df.core.sum()
        print(f"  {tag}: recomputed hc={hc:.3f}  archived={arch:.3f}  "
              f"diff={abs(hc-arch)/arch:.2%}  interval=[{lo:.2f},{hi:.2f}]", flush=True)
        df["hc_lo"] = lo
        df["hc_hi"] = hi
        df.to_csv(f, index=False, encoding="utf-8-sig")
    print("=== extension rungs ===", flush=True)
    d = np.load(RAW / "torus13_a123.npz")
    base = d["cfgs"].astype(np.int64)
    ext = pd.read_csv(DATA / "ladder_ext.csv")
    for tag, c in (("a678", 5), ("a789", 6)):
        C, H = per_config_hc(ref, base + c)
        lo, hi = boot_ratio(C, H, rng)
        ext.loc[ext.tag == tag, "hc_lo"] = lo
        ext.loc[ext.tag == tag, "hc_hi"] = hi
        print(f"  {tag}: hc={H.sum()/C.sum():.3f}  interval=[{lo:.2f},{hi:.2f}]", flush=True)
    ext.to_csv(DATA / "ladder_ext.csv", index=False, encoding="utf-8-sig")
    print("done", flush=True)


if __name__ == "__main__":
    main()
