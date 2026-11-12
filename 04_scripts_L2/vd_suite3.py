# -*- coding: utf-8 -*-
"""vd_suite3.py -- value-domain ensemble suite v3.1 (exact-uniform sampling;
data source = torus13_*.npz ensembles under DATA/raw).
Pipeline: vd_patch template disc -> per-configuration |F|^2 (convex-hull
incircle mask, mean-bond-length normalisation) + spot-centre complex
amplitudes (f_coh) + halo metrics + displacement-field spectra (template
nodes = identity topology matching).
Reference H_id = first-shell peak of the ideal patch (x = 1) on the same grid.
v3.1 changelog: spot positions are now data-driven via spot_pixels_ideal
(fixes a 30-degree orientation mismatch: the fixed 30deg+60k convention only
matched the DFS library geometry, while torus-disc geometry has spots at
0deg+60k -> spurious f_coh collapse / inverted halo ordering); fixed the
embed unpacking of the displacement segments; added the f_coh_al (centroid +
scale-aligned gauge) and f_pk (per-configuration fixed-spot power) columns.
v3.2 changelog: the f_coh reference is now H_ref (the ideal peak under the
same full-vertex raw-sum estimator, per spot) -- H_id (incircle truncation)
differs by ~1.5x, and mixing the two pushes f_coh past its <= 1 bound.
usage: python vd_suite3.py <out_dir> [--smoke]; h-sweep: <out_dir> --hscan"""
import sys, time
import importlib
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(HERE / "external"))
sys.path.insert(0, str(PC.HC))
from vd_patch import (build_patch_graph, build_patch_template,  # noqa: E402
                      embed_config, patch_bonds, validate_ideal)
from vd_suite2 import F_at  # noqa: E402
import pseudocrystal_lib as pl  # noqa: E402
_halo = importlib.import_module(PC.fname("P0_L1_halo_anisotropy.py")[:-3])  # noqa: E402
halo_anisotropy = _halo.halo_anisotropy
G_SQRT3, WIN, K_LO, K_HI = _halo.G_SQRT3, _halo.WIN, _halo.K_LO, _halo.K_HI

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else PC.DATA
RAW = PC.DATA / "raw"
Q_MAX, Q_PIX = 10.5, 601
ENSEMBLES = [("b12", (1, 2)), ("a123", (1, 2, 3)), ("n124", (1, 2, 4)),
             ("a234", (2, 3, 4)), ("a345", (3, 4, 5)), ("a456", (4, 5, 6)),
             ("a567", (5, 6, 7))]


def F_pts(p, pix, cc, dqq):
    """Complex amplitudes at the six spot centres, normalised coordinates p."""
    F = []
    for kk, gx, gy, ix, iy in pix:
        qx, qy = (ix - cc) * dqq, (iy - cc) * dqq
        F.append(np.exp(-1j * (qx * p[:, 0] + qy * p[:, 1])).sum())
    return np.array(F)


def spot_pixels_ideal(I_id, q1d, g=G_SQRT3):
    """Pixel coordinates of the six spot centres, data-driven from the ideal
    patch |F|^2 map (orientation convention consistent with the patch
    geometry).  argmax of the shell ring 0.85-1.15g -> polar angle phi0 ->
    six spots at phi0 + 60deg*k -> snapped to the local argmax within a
    +/-0.35 window.
    v3.1 revision: the fixed 30deg+60deg*k convention only matched the
    DFS-library geometry; torus-disc geometry is rotated by 30 deg (spots at
    0deg+60deg*k), so the fixed convention missed every spot and f_coh
    collapsed spuriously."""
    c = I_id.shape[0] // 2
    dq = q1d[1] - q1d[0]
    yy, xx = np.mgrid[0:I_id.shape[0], 0:I_id.shape[1]]
    rr = np.hypot((xx - c) * dq, (yy - c) * dq)
    ann = (rr >= 0.85 * g) & (rr <= 1.15 * g)
    iy0, ix0 = np.unravel_index(np.argmax(np.where(ann, I_id, 0.0)), I_id.shape)
    r0 = float(np.hypot((ix0 - c) * dq, (iy0 - c) * dq))
    phi0 = float(np.arctan2((iy0 - c) * dq, (ix0 - c) * dq))
    rad = int(0.35 / dq)
    out, vals = [], []
    for kk in range(6):
        phi = phi0 + np.pi / 3 * kk
        ix = int(round(c + r0 * np.cos(phi) / dq))
        iy = int(round(c + r0 * np.sin(phi) / dq))
        sub = I_id[max(iy - rad, 0):iy + rad + 1, max(ix - rad, 0):ix + rad + 1]
        dy, dx = np.unravel_index(np.argmax(sub), sub.shape)
        ix2, iy2 = max(ix - rad, 0) + dx, max(iy - rad, 0) + dy
        out.append((kk, (ix2 - c) * dq, (iy2 - c) * dq, ix2, iy2))
        vals.append(float(I_id[iy2, ix2]))
    vals = np.array(vals)
    assert vals.min() > 0.9 * vals.max(), f"spot orientation gate: {vals}"
    print("  spots(pix):", [(ix, iy) for _, _, _, ix, iy in out],
          " I_id fracs:", [round(float(v / vals.max()), 3) for v in vals], flush=True)
    return out, c, dq


def hscan_mode(DATA):
    """h-sweep mode: process torus13_*_h*.npz (metric-bias ensembles) with the
    same metric set plus the m(h) response."""
    G = build_patch_graph(13, 13)
    T = build_patch_template(G)
    x_ideal = np.ones(G["E"], dtype=np.int64)
    pts_id, cv_id = embed_config(G, T, x_ideal)
    bonds = [(a, b) for a, b, e in patch_bonds(G, T, cv_id)]
    bl_id = float(np.mean([np.hypot(*(pts_id[a] - pts_id[b])) for a, b in bonds]))
    I_id, q1d = pl.structure_factor(pts_id / bl_id, q_max=Q_MAX, q_pixels=Q_PIX)
    c = Q_PIX // 2
    dq = q1d[1] - q1d[0]
    yy, xx = np.mgrid[0:Q_PIX, 0:Q_PIX]
    rr = np.hypot((xx - c) * dq, (yy - c) * dq)
    H_ID = float(I_id[(rr >= 0.85 * G_SQRT3) & (rr <= 1.15 * G_SQRT3)].max())
    pix, cc, dqq = spot_pixels_ideal(I_id, q1d)
    H_ref = np.abs(F_pts(pts_id / bl_id, pix, cc, dqq)) ** 2
    files = sorted(RAW.glob("torus13_*_h*.npz"))
    # h=0 anchor = the uniform ensemble (integration lower bound)
    zero = RAW / "torus13_a123.npz"
    print(f"hscan: {len(files)} biased + h=0 anchor, H_id={H_ID:.6g}", flush=True)
    rows = []
    jobs = [(zero, 0.0)] + [(f, float(np.load(f)["h"])) for f in files]
    for npz, h in jobs:
        d = np.load(npz)
        A = tuple(int(v) for v in d["alphabet"])
        tag = npz.stem.replace("torus13_", "")
        cfgs = d["cfgs"]
        F = G["F"]
        m_h = float(np.mean([cfgs[i].sum() for i in range(len(cfgs))])) / F
        I_sum, F2s, Fs = None, [], []
        for ic in range(len(cfgs)):
            pts, _cv = embed_config(G, T, cfgs[ic])
            bl = float(np.mean([np.hypot(*(pts[a] - pts[b])) for a, b in bonds]))
            I, _ = pl.structure_factor(pts / bl, q_max=Q_MAX, q_pixels=Q_PIX)
            F2s.append(I.astype(np.float32))
            Fs.append(F_at(pts, bl, pix, cc, dqq))
            I_sum = I if I_sum is None else I_sum + I
        I_mean = (I_sum / len(cfgs)).astype(np.float64)
        np.save(DATA / f"meanF2_{tag}.npy", I_mean)
        np.savez_compressed(DATA / f"F2_{tag}.npz", F2=np.stack(F2s))
        Farr = np.array(Fs)
        f_coh = float((np.abs(Farr.mean(axis=0)) ** 2 / H_ref).mean())
        rows_h = []
        for kk, gx, gy, ix, iy in pix:
            r0 = halo_anisotropy(I_mean, q1d, gx, gy)
            y0, y1 = max(int(iy - WIN / dqq), 0), min(int(iy + WIN / dqq) + 1, Q_PIX)
            x0, x1 = max(int(ix - WIN / dqq), 0), min(int(ix + WIN / dqq) + 1, Q_PIX)
            sub = I_mean[y0:y1, x0:x1]
            yy2, xx2 = np.mgrid[y0:y1, x0:x1]
            rrc = np.hypot((xx2 - cc) * dqq - gx, (yy2 - cc) * dqq - gy)
            rows_h.append(dict(core=float(sub[rrc <= 0.22].sum()),
                               halo=float(sub[(rrc >= 0.22) & (rrc <= 1.10)].sum()),
                               aniso=r0["aniso"]))
        hc = sum(r["halo"] for r in rows_h) / sum(r["core"] for r in rows_h)
        rows.append(dict(tag=tag, h=h, A=str(A), ncfg=len(cfgs),
                         Sigma_l_per_F=m_h, f_coh=f_coh, halo_core=hc,
                         r=float(np.mean([r["aniso"] for r in rows_h]))))
        print(f"  {tag}: m(h)={m_h:.4f}  f_coh={f_coh:.3g}  "
              f"halo/core={hc:.2f}", flush=True)
    # entropy cost: s(h) = ln Z(h)/F + h*m(h) => ds/dh = h*m'(h) =>
    # Delta_s(h) = h*m(h) - int_0^h m(h') dh' (integration by parts; trapezoid
    # rule along the measured m(h) curve, no absolute ln Z needed; the h=0
    # row is the uniform-ensemble anchor)
    rs = sorted(rows, key=lambda r: r["h"])
    hs = np.array([r["h"] for r in rs])
    ms = np.array([r["Sigma_l_per_F"] for r in rs])
    cum = np.concatenate([[0.0], np.cumsum((ms[:-1] + ms[1:]) / 2
                                           * np.diff(hs))])
    i0 = int(np.argmin(np.abs(hs)))          # v3.2 fix: integrate from the h=0 anchor
    cum = cum - cum[i0]
    for r, hcur, icum in zip(rs, hs, cum):
        r["Delta_s"] = float(hcur * r["Sigma_l_per_F"] - icum)
    pd.DataFrame(rows).to_csv(DATA / "hscan_summary.csv", index=False,
                              encoding="utf-8-sig")
    print("hscan done (m(h), Delta_s(h) integrated)", flush=True)


def main(smoke=False):
    DATA.mkdir(parents=True, exist_ok=True)
    G = build_patch_graph(13, 13)
    T = build_patch_template(G)
    x_ideal = np.ones(G["E"], dtype=np.int64)
    pts_id, cv_id = embed_config(G, T, x_ideal)
    validate_ideal(G, T)
    bonds = [(a, b) for a, b, e in patch_bonds(G, T, cv_id)]
    bl_id = float(np.mean([np.hypot(*(pts_id[a] - pts_id[b])) for a, b in bonds]))
    I_id, q1d = pl.structure_factor(pts_id / bl_id, q_max=Q_MAX, q_pixels=Q_PIX)
    c = Q_PIX // 2
    dq = q1d[1] - q1d[0]
    yy, xx = np.mgrid[0:Q_PIX, 0:Q_PIX]
    rr = np.hypot((xx - c) * dq, (yy - c) * dq)
    m_shell = (rr >= 0.85 * G_SQRT3) & (rr <= 1.15 * G_SQRT3)
    H_ID = float(I_id[m_shell].max())
    print(f"H_id (ideal patch, shell) = {H_ID:.6g}", flush=True)
    pix, cc, dqq = spot_pixels_ideal(I_id, q1d)
    # reference convention (v3.2 fix): the f_coh complex amplitude is the
    # per-configuration full-vertex raw sum, so the reference must be the
    # ideal value under the same estimator, H_ref (per spot); H_id (incircle
    # mask truncation, drops the boundary-ring vertices) differs by ~1.5x --
    # mixing the two pushes the uniform ensemble's f_coh up to 1.5.
    H_ref = np.abs(F_pts(pts_id / bl_id, pix, cc, dqq)) ** 2
    print(f"  H_ref/H_id per spot: {[round(float(v), 3) for v in H_ref / H_ID]}",
          flush=True)

    # displacement-spectrum grid (template nodes = identity matching)
    R0 = pts_id / bl_id
    R0c = R0 - R0.mean(axis=0)
    kmax = K_HI * 1.02
    kx = np.arange(-kmax, kmax + 1e-9, 0.03)
    KX, KY = np.meshgrid(kx, kx)
    K = np.hypot(KX, KY)
    m_ring = (K >= K_LO) & (K <= K_HI)
    kxs, kys = KX[m_ring], KY[m_ring]
    kmag = np.hypot(kxs, kys)
    khat_x, khat_y = kxs / kmag, kys / kmag
    phase = np.exp(-1j * (kxs[:, None] * R0c[None, :, 0]
                          + kys[:, None] * R0c[None, :, 1]))

    summary, t0 = [], time.time()
    for tag, A in ENSEMBLES:
        npz = RAW / f"torus13_{tag}.npz"
        if not npz.exists():
            print(f"[{tag}] npz missing -- skip", flush=True)
            continue
        d = np.load(npz)
        cfgs = d["cfgs"]
        ncfg = 2 if smoke else len(cfgs)
        mu, sig = float(np.mean(A)), float(np.std(A))
        print(f"[{tag}] A={A} ncfg={ncfg} sigma/mu={sig/mu:.3f} "
              f"sampler-seed={int(d['seed'])}", flush=True)
        I_sum, F2s, Fs, Fs_al = None, [], [], []
        c0 = R0.mean(axis=0)
        for ic in range(ncfg):
            pts, _cv = embed_config(G, T, cfgs[ic])
            bl = float(np.mean([np.hypot(*(pts[a] - pts[b])) for a, b in bonds]))
            pn = pts / bl
            I, _ = pl.structure_factor(pn, q_max=Q_MAX, q_pixels=Q_PIX)
            F2s.append(I.astype(np.float32))
            Fs.append(F_pts(pn, pix, cc, dqq))
            pc = pn - pn.mean(axis=0)
            s = float((pc * R0c).sum() / (R0c * R0c).sum())
            Fs_al.append(F_pts(pc / s + c0, pix, cc, dqq))
            I_sum = I if I_sum is None else I_sum + I
        I_mean = (I_sum / ncfg).astype(np.float64)
        np.save(DATA / f"meanF2_{tag}.npy", I_mean)
        np.savez_compressed(DATA / f"F2_{tag}.npz", F2=np.stack(F2s))
        Farr = np.array(Fs)
        Farr_al = np.array(Fs_al)
        Fm = Farr.mean(axis=0)
        f_coh = np.abs(Fm) ** 2 / H_ref
        f_coh_al = np.abs(Farr_al.mean(axis=0)) ** 2 / H_ref
        f_pk = (np.abs(Farr) ** 2).mean(axis=0) / H_ref
        f_inc = f_pk - f_coh
        rows = []
        r_vals = []
        for kk, gx, gy, ix, iy in pix:
            r0 = halo_anisotropy(I_mean, q1d, gx, gy)
            r_vals.append(r0["aniso"])
            y0, y1 = max(int(iy - WIN / dqq), 0), min(int(iy + WIN / dqq) + 1, Q_PIX)
            x0, x1 = max(int(ix - WIN / dqq), 0), min(int(ix + WIN / dqq) + 1, Q_PIX)
            sub = I_mean[y0:y1, x0:x1]
            yy2, xx2 = np.mgrid[y0:y1, x0:x1]
            rrc = np.hypot((xx2 - cc) * dqq - gx, (yy2 - cc) * dqq - gy)
            core = float(sub[rrc <= 0.22].sum())
            halo = float(sub[(rrc >= 0.22) & (rrc <= 1.10)].sum())
            rows.append(dict(spot=kk, H_ens=float(sub.max()), core=core,
                             halo=halo, aniso=r0["aniso"],
                             f_coh=float(f_coh[kk]), f_inc=float(f_inc[kk]),
                             f_coh_al=float(f_coh_al[kk]), f_pk=float(f_pk[kk])))
        rngb = np.random.default_rng(20261011)
        B = 200
        F2arr = np.stack(F2s)
        rb = np.empty((B, 6))
        fb = np.empty((B, 6))
        for b in range(B):
            idx = rngb.integers(0, ncfg, ncfg)
            Im = F2arr[idx].mean(axis=0).astype(np.float64)
            Fb = Farr[idx].mean(axis=0)
            fb[b] = np.abs(Fb) ** 2 / H_ref
            for kk, gx, gy, ix, iy in pix:
                rb[b, kk] = halo_anisotropy(Im, q1d, gx, gy)["aniso"]
        for kk in range(6):
            rows[kk]["r_lo"] = float(np.percentile(rb[:, kk], 2.5))
            rows[kk]["r_hi"] = float(np.percentile(rb[:, kk], 97.5))
            rows[kk]["fcoh_lo"] = float(np.percentile(fb[:, kk], 2.5))
            rows[kk]["fcoh_hi"] = float(np.percentile(fb[:, kk], 97.5))
        pd.DataFrame(rows).assign(system=tag).to_csv(
            DATA / f"spots_{tag}.csv", index=False, encoding="utf-8-sig")
        cores = sum(r["core"] for r in rows)
        halos = sum(r["halo"] for r in rows)
        print(f"  f_coh mean={f_coh.mean():.3g} (aligned {f_coh_al.mean():.3g})  "
              f"f_pk={f_pk.mean():.3g}  halo/core={halos/cores:.2f}  "
              f"r={np.mean(r_vals):.2f}", flush=True)
        np.savez_compressed(DATA / f"ens_{tag}.npz", alphabet=np.array(A),
                            bl_id=bl_id, H_id=H_ID,
                            sampler_meta=np.array([int(d["seed"]), int(d["eq"]),
                                                   int(d["thin"]), float(d["swap"])]))
    # displacement spectra (run for every ensemble; positions re-embedded once)
    if not smoke:
        for tag, A in ENSEMBLES:
            npz = RAW / f"torus13_{tag}.npz"
            if not npz.exists():
                continue
            d = np.load(npz)
            us = []
            for ic in range(len(d["cfgs"])):
                pts, _cv = embed_config(G, T, d["cfgs"][ic])
                bl = float(np.mean([np.hypot(*(pts[a] - pts[b])) for a, b in bonds]))
                u = pts / bl
                u -= u.mean(axis=0)
                s = float((u * R0c).sum() / (R0c * R0c).sum())
                us.append(u - s * R0c)
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
                summary.append(dict(system=tag, A=str(A),
                                    disp=float(np.std(A) / np.mean(A)),
                                    k_mid=0.5 * (lo + hi),
                                    S_par=float(S_par[m].mean()),
                                    S_perp=float(S_perp[m].mean()),
                                    ratio=float(S_perp[m].mean() / S_par[m].mean()),
                                    n_k=int(m.sum())))
            mid = [s for s in summary if s["system"] == tag]
            rr_ = [s["ratio"] for s in mid]
            print(f"  [{tag}] S_perp/S_par = {np.mean(rr_):.2f}"
                  f"+-{np.std(rr_):.2f}", flush=True)
        pd.DataFrame(summary).to_csv(DATA / "displacement_spectra.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    if "--hscan" in sys.argv:
        data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") \
            else PC.DATA
        data_dir.mkdir(parents=True, exist_ok=True)
        hscan_mode(data_dir)
    else:
        main(smoke="--smoke" in sys.argv)
