# -*- coding: utf-8 -*-
"""Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

vd_tension_v2.py -- readouts for the re-sampled tension ensembles.

Inputs:  <DATA>/raw/vd2_h*_c*.npz  (4 h values x 4 chains x 5 configs)
Outputs: <DATA>/tension_v2.csv     (direct and mirror-readout rows)

The +/- h ensembles are exact mirror pairs of the target measure (l -> 4-l,
SI S10.4 identity 2).  For each |h| we therefore read the ensemble twice:
direct, and through the mirror of the opposite-h ensemble, which is an
exact-distribution sample of the same measure.

usage:  python vd_tension_v2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # L2 scripts dir
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(HERE / "external"))
import vd_ladder_ext as LX                       # noqa: E402

RAW = PC.DATA / "raw"
OUT = PC.DATA
H_LIST = (-0.30, -0.15, 0.15, 0.30)
NCHAIN = 4


def pool(h):
    parts = []
    for k in range(NCHAIN):
        f = RAW / f"vd2_h{h:+.2f}_c{k}.npz"
        d = np.load(f)
        parts.append(d["cfgs"].astype(np.int64))
    return np.concatenate(parts, axis=0), d


def main():
    ref = LX.build_reference(13, 7)
    rows = []
    for h in H_LIST:
        X, d = pool(h)
        r = LX.run_domain(ref, X, (1, 2, 3), tag=f"direct_h{h:+.2f}")
        r["h"] = h
        r["readout"] = "direct"
        rows.append(r)
        print(f"[direct h={h:+.2f}] n={len(X)} f_coh={r['f_coh']:.4g} "
              f"f_inc={r['f_inc']:.4g} halo/core={r['halo_core']:.3f} "
              f"r={r['r']:.3f}", flush=True)
    for h in H_LIST:
        X, d = pool(-h)
        Xm = 4 - X                                     # exact mirror
        r = LX.run_domain(ref, Xm, (1, 2, 3), tag=f"mirror_h{h:+.2f}")
        r["h"] = h
        r["readout"] = "mirror"
        rows.append(r)
        print(f"[mirror h={h:+.2f}] n={len(Xm)} f_coh={r['f_coh']:.4g} "
              f"f_inc={r['f_inc']:.4g} halo/core={r['halo_core']:.3f} "
              f"r={r['r']:.3f}", flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "ratios"} for r in rows])
    df.to_csv(OUT / "tension_v2.csv", index=False, encoding="utf-8-sig")
    print("wrote", OUT / "tension_v2.csv", flush=True)


if __name__ == "__main__":
    main()
