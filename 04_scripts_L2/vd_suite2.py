# -*- coding: utf-8 -*-
"""Historical v2 helper (superseded by vd_suite3.py); kept for provenance.

Value-domain ensemble suite v2 (theory-paper section 2.3 / Fig 5 data).
Seven ensembles: base value domains {1,2}/{1,2,3}/{1,2,4} + the arithmetic
translation ladder {2,3,4}/{3,4,5}/{4,5,6}/{5,6,7}.
Physical question: the translation knob -- counting (shape classes,
translation invariant) vs diffraction (relative dispersion sigma/mu,
translation sensitive).
Per ensemble 20 configurations, n_rings = 8 (N = 169 pores), seeds
20261100+k (rejection-extended).
Metrics:
  f_coh  = |<F>|^2 / H_ideal (coherent-core share of the ensemble-mean
           complex amplitude; H_ideal = 97344 unified reference)
  H_ens  = local spot peak of mean |F|^2 (legacy estimator, kept for linkage)
  halo/core, r (bootstrap CI), S_perp/S_par (equal excitation, halo window
  0.22-1.10, mean-bond-length normalisation)
Outputs: <DATA>/(per-ensemble meanF2 npy + per-configuration F2 npz + metrics
CSV + summary).
Calibre: same as v1 (sampler = single-seed single-restart DFS, non-uniform;
uniformity audited separately)."""
import sys, random
import importlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(PC.HC))
import pseudocrystal_lib as pl  # noqa: E402
_halo = importlib.import_module(PC.fname("P0_L1_halo_anisotropy.py")[:-3])  # noqa: E402
halo_anisotropy = _halo.halo_anisotropy
G_SQRT3, WIN, K_LO, K_HI = _halo.G_SQRT3, _halo.WIN, _halo.K_LO, _halo.K_HI

N_RING = 8
N_CFG = 20
SEED0 = 20261100
ENSEMBLES = [
    ("b12", (1, 2)), ("a123", (1, 2, 3)), ("n124", (1, 2, 4)),
    ("a234", (2, 3, 4)), ("a345", (3, 4, 5)), ("a456", (4, 5, 6)),
    ("a567", (5, 6, 7)),
]
Q_MAX, Q_PIX = 10.5, 601
H_IDEAL = 97344.0   # uniform-tile disc, n=8, first-shell spot, mean-bond-length
                    # normalised (identical for all three domains; measured by
                    # vd_ideal_ref)
OUT = PC.DATA       # v2 legacy generator: that output generation is retired
                    # (see git history); the canonical products come from
                    # vd_suite3 into DATA
OUT.mkdir(exist_ok=True)


def metric_library(alphabet):
    from itertools import product
    ws = [w for w in product(sorted(alphabet), repeat=6) if pl.closure_ok(w)]
    W = np.array(ws, dtype=np.uint8)
    return pl.TileLibrary(
        alphabet=tuple(alphabet), allowed_c3=frozenset({0, 1, 2, 3}),
        colors=(W - min(alphabet)).astype(np.uint8), weights=W,
        all_yellow=np.zeros(len(W), bool),
        perimeter=W.sum(axis=1).astype(int), n_base=len(ws))


def geom_of(sol, lib):
    g = pl.extract_geometry(sol, lib)
    pts = g["metal_nodes"]
    key2idx = {k: i for i, k in enumerate(g["node_keys"])}
    bonds = np.array([[key2idx[e[0]], key2idx[e[1]]] for e in g["edges"]], dtype=int)
    bl = float(np.mean([np.linalg.norm(pts[key2idx[e[0]]] - pts[key2idx[e[1]]])
                        for e in g["edges"]]))
    return pts, bonds, bl, g


def topo_vertex_map(geom):
    coord2idx = {tuple(p): i for i, p in enumerate(geom["metal_nodes"])}
    vm = {}
    for e in geom["edges"]:
        vm[e[3]] = coord2idx[tuple(e[0])]
        vm[e[4]] = coord2idx[tuple(e[1])]
    return vm


def spot_pixels(q1d):
    """Six spot-centre pixel coordinates (expected positions) and the
    neighbourhood-window (+/-0.35) indices."""
    c = Q_PIX // 2
    dq = q1d[1] - q1d[0]
    out = []
    for kk in range(6):
        phi = np.radians(30 + 60 * kk)
        gx, gy = G_SQRT3 * np.cos(phi), G_SQRT3 * np.sin(phi)
        px, py = c + gx / dq, c + gy / dq
        ix, iy = int(round(px)), int(round(py))
        out.append((kk, gx, gy, ix, iy))
    return out, c, dq


def F_at(pts, bl, pixels, c, dq):
    """Complex amplitude of a configuration at the spot centres +/- 1 px
    (normalised coordinates; w = 1)."""
    p = pts / bl
    F = []
    for kk, gx, gy, ix, iy in pixels:
        qx = (ix - c) * dq
        qy = (iy - c) * dq
        ph = qx * p[:, 0] + qy * p[:, 1]
        F.append(np.exp(-1j * ph).sum())
    return np.array(F)


def main(smoke=False):
    import pandas as pd
    n_cfg = 2 if smoke else N_CFG
    summary, spot_rows = [], []
    for name, A in ENSEMBLES:
        lib = metric_library(A)
        mu = float(np.mean(A))
        sig = float(np.std(A))
        disp = sig / mu
        print(f"[{name}] A={A} tiles={lib.size} sigma/mu={disp:.3f}", flush=True)
        if A == (1, 2, 3):
            assert lib.size == 45
        if A == (1, 2, 4):
            assert lib.size == 33

        cfgs, seed = [], SEED0
        while len(cfgs) < n_cfg:
            try:
                sol, _ = pl.solve_tessellation(N_RING, lib, periodic=False,
                                               rng=random.Random(seed))
                pts, bonds, bl, g = geom_of(sol, lib)
                cfgs.append((seed, pts, bonds, bl, g))
            except Exception:
                pass
            seed += 1
            if seed - SEED0 > 400:
                raise RuntimeError("seed exhaustion")
        print(f"  sampled {len(cfgs)} cfgs (seeds {cfgs[0][0]}..{cfgs[-1][0]})", flush=True)

        # per-configuration |F|^2 + spot-centre complex amplitudes
        I_sum, q1d = None, None
        pix, c, dq = None, None, None
        Fs = []
        F2s = []
        for seed, pts, bonds, bl, g in cfgs:
            I, q1d = pl.structure_factor(pts / bl, q_max=Q_MAX, q_pixels=Q_PIX)
            if pix is None:
                pix, c, dq = spot_pixels(q1d)
            F2s.append(I.astype(np.float32))
            Fs.append(F_at(pts, bl, pix, c, dq))
            I_sum = I if I_sum is None else I_sum + I
        I_mean = (I_sum / len(cfgs)).astype(np.float64)
        np.save(OUT / f"meanF2_{name}.npy", I_mean)
        np.savez_compressed(OUT / f"F2_{name}.npz",
                            F2=np.stack(F2s), seeds=np.array([x[0] for x in cfgs]))
        Farr = np.array(Fs)                     # (M,6) complex
        Fm = Farr.mean(axis=0)
        f_coh = np.abs(Fm) ** 2 / H_IDEAL
        f_inc = (np.abs(Farr) ** 2).mean(axis=0) / H_IDEAL - f_coh
        print(f"  f_coh/spot: {['%.2g' % v for v in f_coh]}  "
              f"mean={f_coh.mean():.3g}  f_inc_mean={f_inc.mean():.3g}", flush=True)

        # halo metrics + bootstrap CI of r
        rows = []
        r_vals = []
        for kk, gx, gy, ix, iy in pix:
            r0 = halo_anisotropy(I_mean, q1d, gx, gy)
            r_vals.append(r0["aniso"])
            # core/halo shares
            y0, y1 = max(int(iy - WIN / dq), 0), min(int(iy + WIN / dq) + 1, Q_PIX)
            x0, x1 = max(int(ix - WIN / dq), 0), min(int(ix + WIN / dq) + 1, Q_PIX)
            sub = I_mean[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]
            rr = np.hypot((xx - c) * dq - gx, (yy - c) * dq - gy)
            core = float(sub[rr <= 0.22].sum())
            halo = float(sub[(rr >= 0.22) & (rr <= 1.10)].sum())
            rows.append(dict(spot=kk, H_ens=float(sub.max()), core=core,
                             halo=halo, aniso=r0["aniso"], f_coh=float(f_coh[kk]),
                             f_inc=float(f_inc[kk])))
        # bootstrap r / f_coh (B = 200, configuration resampling)
        rngb = np.random.default_rng(20261011)
        B = 200
        rb = np.empty((B, 6))
        fb = np.empty((B, 6))
        F2arr = np.stack(F2s)
        for b in range(B):
            idx = rngb.integers(0, len(cfgs), len(cfgs))
            Im = F2arr[idx].mean(axis=0).astype(np.float64)
            Fb = Farr[idx].mean(axis=0)
            fb[b] = np.abs(Fb) ** 2 / H_IDEAL
            for kk, gx, gy, ix, iy in pix:
                rb[b, kk] = halo_anisotropy(Im, q1d, gx, gy)["aniso"]
        for kk in range(6):
            rows[kk]["r_lo"] = float(np.percentile(rb[:, kk], 2.5))
            rows[kk]["r_hi"] = float(np.percentile(rb[:, kk], 97.5))
            rows[kk]["fcoh_lo"] = float(np.percentile(fb[:, kk], 2.5))
            rows[kk]["fcoh_hi"] = float(np.percentile(fb[:, kk], 97.5))
        pd.DataFrame(rows).assign(system=name).to_csv(
            OUT / f"spots_{name}.csv", index=False, encoding="utf-8-sig")
        cores = sum(r["core"] for r in rows)
        halos = sum(r["halo"] for r in rows)
        print(f"  halo/core={halos/cores:.2f}  r={np.mean(r_vals):.2f}"
              f"+{np.percentile(rb, 97.5)-np.mean(rb):.2f}/"
              f"-{np.mean(rb)-np.percentile(rb, 2.5):.2f}", flush=True)

        # displacement-field spectra (mean-bond-length normalised)
        idx_u = int(np.argmax(np.all(lib.weights == min(A), axis=1)))
        ref_sol = {cc: idx_u for cc in pl.hex_coords(N_RING - 1)}
        ref = pl.extract_geometry(ref_sol, lib)
        R = np.array(ref["metal_nodes"], float)
        ref_map = topo_vertex_map(ref)
        pts_u, bonds_u, bl_u, _ = geom_of(ref_sol, lib)
        R0 = R / bl_u
        us = []
        for seed, pts, bonds, bl, g in cfgs:
            vmap = topo_vertex_map(g)
            assert set(vmap.keys()) == set(ref_map.keys())
            r = np.zeros_like(R0)
            for tk, i in vmap.items():
                r[ref_map[tk]] = pts[i] / bl
            r -= r.mean(axis=0)
            Rc = R0 - R0.mean(axis=0)
            s = float((r * Rc).sum() / (Rc * Rc).sum())
            us.append(r - s * Rc)
        kmax = K_HI * 1.02
        kx = np.arange(-kmax, kmax + 1e-9, 0.03)
        KX, KY = np.meshgrid(kx, kx)
        K = np.hypot(KX, KY)
        m_ring = (K >= K_LO) & (K <= K_HI)
        kxs, kys = KX[m_ring], KY[m_ring]
        kmag = np.hypot(kxs, kys)
        khat_x, khat_y = kxs / kmag, kys / kmag
        phase = np.exp(-1j * (kxs[:, None] * R0[None, :, 0]
                              + kys[:, None] * R0[None, :, 1]))
        S_par = np.zeros(len(kxs))
        S_perp = np.zeros(len(kxs))
        for u in us:
            ux_k = phase @ u[:, 0]
            uy_k = phase @ u[:, 1]
            upar = ux_k * khat_x + uy_k * khat_y
            upx = ux_k - upar * khat_x
            upy = uy_k - upar * khat_y
            S_par += np.abs(upar) ** 2
            S_perp += np.abs(upx) ** 2 + np.abs(upy) ** 2
        S_par /= len(us)
        S_perp /= len(us)
        K_BIN = 0.08
        for b in range(int((K_HI - K_LO) / K_BIN)):
            lo, hi = K_LO + b * K_BIN, K_LO + (b + 1) * K_BIN
            m = (kmag >= lo) & (kmag < hi)
            if m.sum() < 3:
                continue
            summary.append(dict(system=name, A=str(A), disp=disp,
                                k_mid=0.5 * (lo + hi),
                                S_par=float(S_par[m].mean()),
                                S_perp=float(S_perp[m].mean()),
                                ratio=float(S_perp[m].mean() / S_par[m].mean()),
                                n_k=int(m.sum())))
        mid = [s for s in summary if s["system"] == name]
        rr_ = [s["ratio"] for s in mid]
        print(f"  S_perp/S_par = {np.mean(rr_):.2f}+-{np.std(rr_):.2f}", flush=True)
        np.savez_compressed(OUT / f"ens_{name}.npz",
                            seeds=np.array([x[0] for x in cfgs]),
                            bl=np.array([x[3] for x in cfgs]),
                            disp=np.array([disp]))
    pd.DataFrame(summary).to_csv(OUT / "displacement_spectra.csv",
                                 index=False, encoding="utf-8-sig")
    print("done", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
