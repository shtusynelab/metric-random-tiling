# -*- coding: utf-8 -*-
"""vd_wrms.py -- w_rms(L) size ladder: metric-field roughness curves
(three value domains).
Definition (following the P0_T1_wrms.py calibre): w^2(L) = ensemble mean over
configurations of sum_i |u_i|^2 / N, where u is the displacement of the
configuration positions (mean-bond-length normalised) relative to the ideal
equilateral honeycomb reference of the same size, centroid-aligned with a
per-configuration optimal scale, template nodes = identity topology match.
Data source: tor{L}_{tag}.npz (vd_torus_run size ladder) + torus13_{tag}.npz
(L = 13), read from DATA/raw.
Output: wrms_curve.csv (tag, L, ncfg, wrms2_mean, wrms2_sem, h).

usage: python vd_wrms.py [out_dir]
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(HERE / "external"))
from vd_patch import (build_patch_graph, build_patch_template,  # noqa: E402
                      embed_config, patch_bonds)

def interior_mask_of(bonds, n_nodes):
    """Mask of interior nodes at graph distance >= 2 from the boundary
    (nodes of degree < 3); removes the O(1/L) de-slit contamination."""
    from collections import defaultdict, deque
    nb = defaultdict(set)
    for a, b, e in bonds:
        nb[a].add(b)
        nb[b].add(a)
    boundary = {n for n, s in nb.items() if len(s) < 3}
    dist = {n: 0 for n in boundary}
    q = deque(boundary)
    while q:
        u = q.popleft()
        for v in nb[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    mask = np.array([dist.get(n, 0) >= 2 for n in range(n_nodes)])
    return mask


RAW = PC.DATA / "raw"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else PC.DATA
TAGS = ["b12", "a123", "a456"]
RADII = [int(x) for x in os.environ.get(
    "VD_WRMS_RADII", "3,4,5,6,7,8,9,10").split(",")]
SRC_OVR = int(os.environ.get("VD_WRMS_SRC", "0"))   # probe: force the source ensemble to 13/21
# disc-radius ladder; R <= 6 uses the 13x13 ensembles, R >= 7 the 21x21
# ensembles (tor21_{tag}.npz)

rows = []
for R in RADII:
    Lsrc = SRC_OVR if SRC_OVR else (13 if R <= 6 else 21)  # graph built at the source-ensemble size
    GR = build_patch_graph(Lsrc, Lsrc, root_rings=R)
    TR = build_patch_template(GR)
    x_ideal = np.ones(GR["E"], dtype=np.int64)
    R0, cv_id = embed_config(GR, TR, x_ideal)
    bonds3 = patch_bonds(GR, TR, cv_id)
    bonds = [(a, b) for a, b, e in bonds3]
    bl_id = float(np.mean([np.hypot(*(R0[a] - R0[b])) for a, b in bonds]))
    R0 = R0 / bl_id
    R0c = R0 - R0.mean(axis=0)
    mask = interior_mask_of(bonds3, len(R0))
    for tag in TAGS:
        name = ("torus13_" if Lsrc == 13 else "tor21_") + f"{tag}.npz"
        npzf = RAW / name              # canonical data home: DATA/raw/
        if not npzf.exists():
            print(f"skip {tag} R={R} (no npz {npzf.name})", flush=True)
            continue
        d = np.load(npzf)
        cfgs = d["cfgs"]
        ws, ws_int = [], []
        for ic in range(len(cfgs)):
            pts, _cv = embed_config(GR, TR, cfgs[ic])
            bl = float(np.mean([np.hypot(*(pts[a] - pts[b]))
                                for a, b in bonds]))
            u = pts / bl
            u -= u.mean(axis=0)
            s_ = float((u * R0c).sum() / (R0c * R0c).sum())
            u = u - s_ * R0c
            ws.append(float((u ** 2).sum() / len(u)))
            ui = u[mask]
            ws_int.append(float((ui ** 2).sum() / len(ui)))
        ws = np.array(ws)
        wi = np.array(ws_int)
        rows.append(dict(tag=tag, R=R, L=2 * R + 1, ncfg=len(ws),
                         h=float(d.get("h", 0.0)),
                         n_nodes=len(R0), n_int=int(mask.sum()),
                         wrms2_mean=float(ws.mean()),
                         wrms2_sem=float(ws.std() / np.sqrt(len(ws))),
                         wrms2_int_mean=float(wi.mean()),
                         wrms2_int_sem=float(wi.std() / np.sqrt(len(wi)))))
        print(f"{tag} R={R}(L={2*R+1}): all {ws.mean():.4f}"
              f"+-{ws.std()/np.sqrt(len(ws)):.4f}  interior({int(mask.sum())}) "
              f"{wi.mean():.4f}+-{wi.std()/np.sqrt(len(wi)):.4f}", flush=True)
df = pd.DataFrame(rows)
out = OUT / "wrms_curve.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"saved {out}", flush=True)
