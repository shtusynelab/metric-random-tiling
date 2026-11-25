# -*- coding: utf-8 -*-
"""coh_bias_audit.py -- P0-2: raw vs cross-sample unbiased coherent fraction, all c, both chains.

f_raw(spot) = |mean_j F_j|^2 / H_ref          (the paper's estimator)
f_unb(spot) = (M|mean F|^2 - mean|F|^2)/(M-1) / H_ref   (unbiased for |E F|^2, iid case)
Also reports the relative length dispersion sigma/mu per set and the DW proxy ratio.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]   # repo root (script lives in 04_scripts_L2)
sys.path.insert(0, str(BASE / '04_scripts_L2'))
sys.path.insert(0, str(BASE / '04_scripts_L2/external'))
sys.path.insert(0, str(BASE / '06_engines/z3_hexagonal_P0'))
from vd_patch import build_patch_graph, build_patch_template, embed_config, patch_bonds
import pseudocrystal_lib as pl
from vd_suite3 import spot_pixels_ideal

Q_MAX, Q_PIX = 10.5, 601
DATA = BASE / '03_data/uniform'
RAW = DATA / 'raw'
t0 = time.time()

G = build_patch_graph(13, 13)
T = build_patch_template(G)
x_ideal = np.ones(G['E'], dtype=np.int64)
pts_id, cv_id = embed_config(G, T, x_ideal)
bonds = [(a, b) for a, b, e in patch_bonds(G, T, cv_id)]
bl_id = float(np.mean([np.hypot(*(pts_id[a] - pts_id[b])) for a, b in bonds]))
I_id, q1d = pl.structure_factor(pts_id / bl_id, q_max=Q_MAX, q_pixels=Q_PIX)
pix, cc, dqq = spot_pixels_ideal(I_id, q1d)
print("setup", time.time() - t0, "s", flush=True)

def F_six(p):
    """Complex amplitude at the six spot centres, all vertices (SI S6.1 convention)."""
    F = []
    for kk, gx, gy, ix, iy in pix:
        qx, qy = (ix - cc) * dqq, (iy - cc) * dqq
        F.append(np.exp(-1j * (qx * p[:, 0] + qy * p[:, 1])).sum())
    return np.array(F)

H_ref = np.abs(F_six(pts_id / bl_id)) ** 2
print("H_ref per spot:", np.round(H_ref, 1), flush=True)

rows = []
for chain, npzf in [('prod', RAW / 'torus13_a123.npz'), ('rep', RAW / 'torus13_a123r.npz')]:
    cfgs = np.load(npzf, allow_pickle=True)['cfgs']
    M = len(cfgs)
    for c in range(7):
        Fs, rel = [], []
        for i in range(M):
            cfg = cfgs[i] + c
            pts, _ = embed_config(G, T, cfg)
            bl = float(np.mean([np.hypot(*(pts[a] - pts[b])) for a, b in bonds]))
            Fs.append(F_six(pts / bl))
            rel.append(float(np.std(cfg) / np.mean(cfg)))
        rel_alphabet = float(np.sqrt(2.0 / 3.0) / (2.0 + c))   # paper's DW proxy uses the nominal alphabet dispersion
        Farr = np.array(Fs)                       # (M, 6)
        mF = Farr.mean(axis=0)
        raw = np.abs(mF) ** 2
        unb = (M * raw - (np.abs(Farr) ** 2).mean(axis=0)) / (M - 1)
        f_raw = float((raw / H_ref).mean())
        f_unb = float((unb / H_ref).mean())
        f_unb_spot = unb / H_ref
        rows.append(dict(chain=chain, c=c, f_coh_raw=f_raw, f_coh_unb=f_unb,
                         unb_min=float(f_unb_spot.min()),
                         rel_disp_actual=float(np.mean(rel)),
                         rel_disp_alphabet=rel_alphabet))
        print(f"{chain} c={c}: f_raw={f_raw:.6g} f_unb={f_unb:.6g} "
              f"(min spot {f_unb_spot.min():.3g}) rel={np.mean(rel):.3f}", flush=True)

df = pd.DataFrame(rows)
out = DATA / 'coh_bias_audit.csv'
df.to_csv(out, index=False, encoding='utf-8-sig')

for chain in ('prod', 'rep'):
    g = df[df.chain == chain]
    r_raw = g.f_coh_raw.iloc[6] / g.f_coh_raw.iloc[0]
    r_unb = g.f_coh_unb.iloc[6] / g.f_coh_unb.iloc[0]
    print(f"{chain}: endpoint ratio raw={r_raw:.1f} unbiased={r_unb:.1f}")
    # DW proxy: measured-to-predicted at each c; predicted = exp(-|G|^2 (sigma/mu)^2), |G|^2=52.638
    pred = np.exp(-52.638 * g.rel_disp_alphabet.values ** 2)   # paper convention: nominal alphabet dispersion
    print(f"  DW measured/predicted: raw={np.round(g.f_coh_raw.values/pred,2)}")
    print(f"  DW measured/predicted: unb={np.round(g.f_coh_unb.values/pred,2)}")
print(f"total {time.time()-t0:.0f}s -> {out}", flush=True)
