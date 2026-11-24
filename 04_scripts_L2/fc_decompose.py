# -*- coding: utf-8 -*-
"""fc_decompose.py -- Fig 4c coherent/diffuse decomposition (Codex memo 9.2).

For the matched translation series (c=0..6, configs from torus13_a123.npz
shifted by c, and the independent chain torus13_a123r.npz), compute per spot:
  I_tot  = < |F|^2 >            (per-config vertex mask = convex-hull incircle
                                 x0.99, replicating structure_factor)
  I_coh  = | <F> |^2            (same masked vertex convention)
  I_diff = I_tot - I_coh        (additivity asserted)
  I_d1   = < |F^(1)|^2 > - | <F^(1)> |^2,  F^(1) = -i*alpha*sum_j (q.u_j) e^-iqRj
and run the PRODUCTION halo_anisotropy operator verbatim on each map
(r_tot, r_coh, r_diff, r_d1), plus a fixed-center / no-background additive
variant (r_*_raw) whose mixture identity r_tot_raw = weighted mix is checked.
Also: per-config r_tot for paired deltas; replication check vs spots_*.csv.

Memory: << 0.1 GB. Runtime: ~5-10 min (local). Output: fc_decompose_results.csv
"""
import sys, math, time, importlib
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(r'C:/Users/Tsing/Desktop/Group projects/MOFtronics/MOF pseudocrystals/metric-random-tiling')
sys.path.insert(0, str(BASE / '04_scripts_L2'))
sys.path.insert(0, str(BASE / '04_scripts_L2/external'))
sys.path.insert(0, str(BASE / '06_engines/z3_hexagonal_P0'))
from vd_patch import build_patch_graph, build_patch_template, embed_config, patch_bonds
import pseudocrystal_lib as pl
from vd_suite3 import spot_pixels_ideal
from scipy.spatial import ConvexHull

import paths_common as PC
DATA = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else PC.DATA
RAW = DATA / 'raw'

_halo = importlib.import_module('P0_L1_halo_anisotropy')
halo_anisotropy = _halo.halo_anisotropy
K_LO, K_HI = _halo.K_LO, _halo.K_HI
Q_MAX, Q_PIX = 10.5, 601
MARGIN = 50          # px half-width filled around ideal spot pixel (covers argmax +/-10 + window +/-37)
NSHIFT = 7           # c = 0..6
BATCH = 2000         # pixel batch for amplitude computation

t0 = time.time()

# ---------- setup: identical to vd_suite3.hscan_mode ----------
G = build_patch_graph(13, 13)
T = build_patch_template(G)
x_ideal = np.ones(G['E'], dtype=np.int64)
pts_id, cv_id = embed_config(G, T, x_ideal)
bonds = [(a, b) for a, b, e in patch_bonds(G, T, cv_id)]
bl_id = float(np.mean([np.hypot(*(pts_id[a] - pts_id[b])) for a, b in bonds]))
I_id, q1d = pl.structure_factor(pts_id / bl_id, q_max=Q_MAX, q_pixels=Q_PIX)
pix, cc, dqq = spot_pixels_ideal(I_id, q1d)
dq = float(q1d[1] - q1d[0])
print(f"setup done {time.time()-t0:.1f}s | dq={dq:.5f} spots={[(gx,gy) for _,gx,gy,_,_ in pix]}", flush=True)

R = pts_id / bl_id                      # ideal normalized positions (template order, Nv)

def incircle_keep(p):
    """Replicate structure_factor's convex-hull incircle x0.99 vertex mask."""
    pts = p - (p.max(axis=0) + p.min(axis=0)) / 2
    hull = ConvexHull(pts)
    r_in = np.inf
    for s1, s2 in hull.simplices:
        p1, p2 = pts[s1], pts[s2]
        d = math.hypot(*(p2 - p1))
        if d > 0:
            r_in = min(r_in, abs(p1[0] * p2[1] - p2[0] * p1[1]) / d)
    r_in *= 0.99
    return np.linalg.norm(pts, axis=1) <= r_in

def amps(p, gx, gy, keep):
    """Complex F at all pixels of the local window around the ideal spot pixel."""
    ix0, iy0 = int(round(cc + gx / dq)), int(round(cc + gy / dq))
    y0, y1 = iy0 - MARGIN, iy0 + MARGIN + 1
    x0, x1 = ix0 - MARGIN, ix0 + MARGIN + 1
    xs = (np.arange(x0, x1) - cc) * dq
    ys = (np.arange(y0, y1) - cc) * dq
    QX, QY = np.meshgrid(xs, ys)                 # (H, W)
    pv = p[keep]
    out = np.empty(QX.shape, dtype=complex)
    px_, py_ = pv[:, 0], pv[:, 1]
    for i in range(0, QX.size, BATCH):
        qx = QX.ravel()[i:i + BATCH][:, None]
        qy = QY.ravel()[i:i + BATCH][:, None]
        ph = qx * px_[None, :] + qy * py_[None, :]
        out.ravel()[i:i + BATCH] = np.exp(-1j * ph).sum(axis=1)
    return out.ravel(), (x0, y0), (xs, ys)

def quv(gx, gy, xs, ys):
    """(q.u1),(q.u2) grids relative to unit vectors radial/tangential of (gx,gy)."""
    Gm = np.hypot(gx, gy)
    u = np.array([gx, gy]) / Gm
    return xs * u[0] + ys * u[1], -xs * u[1] + ys * u[0]

def fill_map(loc, x0, y0):
    """Embed a local window into a 601x602 frame (index-safe) for the engine."""
    full = np.zeros((Q_PIX, Q_PIX + 1), dtype=float)
    h, w = loc.shape
    full[y0:y0 + h, x0:x0 + w] = loc
    return full

def engine_r(map601, gx, gy):
    return halo_anisotropy(map601[:, :Q_PIX], q1d, gx, gy)

def raw_moments(loc_w, kpa, kpe, krr):
    """Fixed-center (ideal spot), no-background second moments in the
    halo-local frame k = q - G; additive-valid across components."""
    m = (krr >= K_LO) & (krr <= K_HI)
    W = loc_w[m]
    kp, ke = kpa[m], kpe[m]
    Mpar = float((W * kp ** 2).sum())
    Mperp = float((W * ke ** 2).sum())
    return Mpar, Mperp, float(W.sum())

# ---------- sanity: stored shifted ensembles equal cfg + c ----------
d0 = np.load(RAW / 'torus13_a123.npz', allow_pickle=True)
cfg0 = d0['cfgs']
for c, f in [(1, 'torus13_a234.npz'), (2, 'torus13_a345.npz'),
             (3, 'torus13_a456.npz'), (4, 'torus13_a567.npz'),
             (5, 'torus13_a678.npz'), (6, 'torus13_a789.npz')]:
    dc = np.load(RAW / f, allow_pickle=True)
    assert np.array_equal(dc['cfgs'], cfg0 + c), f'{f} != a123 + {c}'
print('matched-config assert OK (a234..a789 = a123 + c)', flush=True)

chains = [('prod', RAW / 'torus13_a123.npz'), ('rep', RAW / 'torus13_a123r.npz')]
rows = []
pc_rows = []
for chain, npzf in chains:
    d = np.load(npzf, allow_pickle=True)
    cfgs = d['cfgs']
    ncfg = len(cfgs)
    # per-config c=0 embedding -> normalized displacement u_i (template order)
    Us, BL0 = [], []
    for i in range(ncfg):
        pts0, _ = embed_config(G, T, cfgs[i])
        bl0 = float(np.mean([np.hypot(*(pts0[a] - pts0[b])) for a, b in bonds]))
        Us.append(pts0 / bl0 - R)
        BL0.append(bl0)
    # spot-local geometry precompute
    geo = []
    for kk, gx, gy, ix, iy in pix:
        ix0, iy0 = int(round(cc + gx / dq)), int(round(cc + gy / dq))
        xs = (np.arange(ix0 - MARGIN, ix0 + MARGIN + 1) - cc) * dq
        ys = (np.arange(iy0 - MARGIN, iy0 + MARGIN + 1) - cc) * dq
        KX, KY = np.meshgrid(xs - gx, ys - gy)   # k = q - G: halo-local frame
        Gm = np.hypot(gx, gy)
        ex, ey = gx / Gm, gy / Gm
        kpa = KX * ex + KY * ey
        kpe = -KX * ey + KY * ex
        krr = np.hypot(KX, KY)
        geo.append((ix0, iy0, xs, ys, kpa, kpe, krr))
    for c in range(NSHIFT):
        acc = [np.zeros((geo[s][2].size * geo[s][3].size,), dtype=complex) for s in range(6)]      # sum F
        accF2 = [np.zeros((geo[s][2].size * geo[s][3].size,), dtype=float) for s in range(6)]      # sum |F|^2
        acc1 = [np.zeros((geo[s][2].size * geo[s][3].size,), dtype=complex) for s in range(6)]     # sum F1
        accF12 = [np.zeros((geo[s][2].size * geo[s][3].size,), dtype=float) for s in range(6)]     # sum |F1|^2
        pc_r = [[] for _ in range(6)]
        for i in range(ncfg):
            cfg = cfgs[i] + c
            pts, _cv = embed_config(G, T, cfg)
            bl = float(np.mean([np.hypot(*(pts[a] - pts[b])) for a, b in bonds]))
            keep = incircle_keep(pts / bl)
            alpha = BL0[i] / bl
            t = time.time()
            for s in range(6):
                kk, gx, gy, ix, iy = pix[s]
                ix0, iy0, xs, ys, kpa, kpe, rr = geo[s]
                p = pts / bl
                F, (x0, y0), _ = amps(p, gx, gy, keep)
                QX2, QY2 = np.meshgrid(xs, ys)            # (ny, nx)
                qx2 = QX2.ravel()[:, None]
                qy2 = QY2.ravel()[:, None]
                Rk = R[keep]
                Uk = Us[i][keep]
                qR = qx2 * Rk[None, :, 0] + qy2 * Rk[None, :, 1]      # (npx, Nv)
                qu = qx2 * Uk[None, :, 0] + qy2 * Uk[None, :, 1]
                F1 = -1j * alpha * (qu * np.exp(-1j * qR)).sum(axis=1)
                f2 = np.abs(F) ** 2
                f12 = np.abs(F1) ** 2
                acc[s] += F; accF2[s] += f2; acc1[s] += F1; accF12[s] += f12
                # per-config r_tot (production operator on this config's map)
                loc = f2.reshape(ys.size, xs.size)
                mfull = fill_map(loc, x0, y0)
                pc_r[s].append(engine_r(mfull, gx, gy)['aniso'])
            if i == 0 and c == 0:
                print(f'  per-config pass {time.time()-t:.2f}s/config', flush=True)
        for s in range(6):
            kk, gx, gy, ix, iy = pix[s]
            ix0, iy0, xs, ys, kpa, kpe, rr = geo[s]
            Itot = (accF2[s] / ncfg).reshape(ys.size, xs.size)
            Fm = acc[s] / ncfg
            Icoh = (np.abs(Fm) ** 2).reshape(ys.size, xs.size)
            Idiff = Itot - Icoh
            assert Idiff.min() > -1e-8 * max(Itot.max(), 1.0), 'additivity violated'
            F1m = acc1[s] / ncfg
            Id1 = (accF12[s] / ncfg - np.abs(F1m) ** 2).reshape(ys.size, xs.size)
            x0, y0 = ix0 - MARGIN, iy0 - MARGIN
            res = {}
            for name, mp in [('tot', Itot), ('coh', Icoh), ('diff', Idiff), ('d1', Id1)]:
                res[name] = engine_r(fill_map(mp, x0, y0), gx, gy)
            # additive variant: fixed ideal-spot center, k = q - G frame, no background
            M = {}
            for name, mp in [('tot', Itot), ('coh', Icoh), ('diff', Idiff), ('d1', Id1)]:
                M[name] = raw_moments(mp, kpa, kpe, krr)
            # finite-sampling-corrected coherent map (E[I_coh] = I_coh + I_diff/M)
            Ic_c = np.clip(Icoh - Idiff / (ncfg - 1), 0, None)
            res['cohM'] = engine_r(fill_map(Ic_c, x0, y0), gx, gy)
            M['cohM'] = raw_moments(Ic_c, kpa, kpe, krr)
            mix = (M['coh'][1] + M['diff'][1]) / (M['coh'][0] + M['diff'][0])
            rows.append(dict(
                chain=chain, c=c, spot=s,
                r_tot=res['tot']['aniso'], r_coh=res['coh']['aniso'],
                r_diff=res['diff']['aniso'], r_d1=res['d1']['aniso'],
                r_cohM=res['cohM']['aniso'],
                r_tot_raw=M['tot'][1] / M['tot'][0], r_coh_raw=M['coh'][1] / M['coh'][0],
                r_cohM_raw=M['cohM'][1] / M['cohM'][0],
                r_diff_raw=M['diff'][1] / M['diff'][0], r_d1_raw=M['d1'][1] / M['d1'][0],
                w_coh=M['coh'][0] / (M['coh'][0] + M['diff'][0]),
                mix_resid=M['tot'][1] / M['tot'][0] - mix,
                Wsum_coh=M['coh'][2], Wsum_diff=M['diff'][2],
                Itot_sum=float(Itot.sum()), Idiff_sum=float(np.clip(Idiff, 0, None).sum())))
            pc_rows.append(dict(chain=chain, c=c, spot=s, mean_r_tot=float(np.mean(pc_r[s]))))
        mr = np.mean([r['r_tot'] for r in rows if r['chain'] == chain and r['c'] == c])
        md = np.mean([r['r_diff'] for r in rows if r['chain'] == chain and r['c'] == c])
        m1 = np.mean([r['r_d1'] for r in rows if r['chain'] == chain and r['c'] == c])
        mc = np.mean([r['r_coh'] for r in rows if r['chain'] == chain and r['c'] == c])
        print(f'{chain} c={c}: r_tot={mr:.4f} r_coh={mc:.4f} r_diff={md:.4f} r_d1={m1:.4f}', flush=True)

out = DATA / 'fc_decompose_results.csv'
pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
pd.DataFrame(pc_rows).to_csv(out.with_name('fc_decompose_percfg.csv'), index=False, encoding='utf-8-sig')

# ---------- replication check vs production spots_*.csv ----------
print('\n--- replication: prod chain r_tot (c=0..4) vs spots_*.csv aniso means ---', flush=True)
prod = pd.DataFrame(rows)
worst = 0.0
for j, tag in enumerate(['a123', 'a234', 'a345', 'a456', 'a567']):
    ref = pd.read_csv(DATA / f'spots_{tag}.csv')['aniso'].mean()
    mine = prod[(prod.chain == 'prod') & (prod.c == j)]['r_tot'].mean()
    dv = abs(mine - ref)
    worst = max(worst, dv)
    print(f'  c={j} ({tag}): mine={mine:.6f} prod={ref:.6f} |d|={dv:.2e}')
print(f'worst |delta| = {worst:.2e}  (expect <~1e-3: prod maps were float32-stacked)')
print(f'total {time.time()-t0:.0f}s', flush=True)
