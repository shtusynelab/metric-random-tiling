# -*- coding: utf-8 -*-
"""vd_ladder_ext.py -- L2 revision extension runs.

B3: ladder rungs {6,7,8} and {7,8,9}.  The arithmetic ladder is an exact
    translation relabelling of the {1,2,3} sample (verified to 3e-11 in the
    SI), so the configurations are x + c with c = 5, 6.  No new sampling.
B4: size scaling.  Hexagonal discs R = 6..10 (N = 3R(R+1)+1 = 127..331 faces)
    cut from the archived 21x21 ensembles (tor21_a123 / tor21_a456 /
    tor21_b12), same sample, same estimator.

Self-check gate: the pipeline below must reproduce the canonical a123
readings (f_coh = 9.0e-4, halo/core = 16.7) from the archived 13x13 npz.
If it does, the extension numbers are on the same calibre.

usage:
  python vd_ladder_ext.py ladder      # B3 -> <DATA>/ canonical files
  python vd_ladder_ext.py size        # B4 -> <DATA>/size_scaling.csv
ASCII source on purpose.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(HERE / "external"))
import vd_suite3 as S                      # noqa: E402  (module import, no run)
from vd_patch import (build_patch_graph, build_patch_template,  # noqa: E402
                      embed_config, patch_bonds, validate_ideal)
import pseudocrystal_lib as pl             # noqa: E402

DATA = PC.DATA
RAW = DATA / "raw"
OUT = PC.DATA
OUT.mkdir(parents=True, exist_ok=True)

Q_MAX, Q_PIX = S.Q_MAX, S.Q_PIX


def build_reference(L, R):
    """Ideal reference for a hexagonal disc of radius R cut from an LxL torus."""
    G = build_patch_graph(L, L, root_rings=R)
    T = build_patch_template(G)
    x_id = np.ones(G["E"], dtype=np.int64)
    pts_id, cv_id = embed_config(G, T, x_id)
    validate_ideal(G, T)
    bonds = [(a, b) for a, b, e in patch_bonds(G, T, cv_id)]
    bl_id = float(np.mean([np.hypot(*(pts_id[a] - pts_id[b])) for a, b in bonds]))
    I_id, q1d = pl.structure_factor(pts_id / bl_id, q_max=Q_MAX, q_pixels=Q_PIX)
    c = Q_PIX // 2
    dq = q1d[1] - q1d[0]
    yy, xx = np.mgrid[0:Q_PIX, 0:Q_PIX]
    rr = np.hypot((xx - c) * dq, (yy - c) * dq)
    m_shell = (rr >= 0.85 * S.G_SQRT3) & (rr <= 1.15 * S.G_SQRT3)
    H_ID = float(I_id[m_shell].max())
    pix, cc, dqq = S.spot_pixels_ideal(I_id, q1d)
    H_ref = np.abs(S.F_pts(pts_id / bl_id, pix, cc, dqq)) ** 2
    # displacement-spectrum grid (template nodes, identity matching)
    R0 = pts_id / bl_id
    R0c = R0 - R0.mean(axis=0)
    kmax = S.K_HI * 1.02
    kx = np.arange(-kmax, kmax + 1e-9, 0.03)
    KX, KY = np.meshgrid(kx, kx)
    K = np.hypot(KX, KY)
    m_ring = (K >= S.K_LO) & (K <= S.K_HI)
    kxs, kys = KX[m_ring], KY[m_ring]
    kmag = np.hypot(kxs, kys)
    khat_x, khat_y = kxs / kmag, kys / kmag
    phase = np.exp(-1j * (kxs[:, None] * R0c[None, :, 0]
                          + kys[:, None] * R0c[None, :, 1]))
    return dict(G=G, T=T, pts_id=pts_id, cv_id=cv_id, bonds=bonds, bl_id=bl_id,
                H_ID=H_ID, pix=pix, cc=cc, dqq=dqq, H_ref=H_ref, R0c=R0c,
                kmag=kmag, khat_x=khat_x, khat_y=khat_y, phase=phase, q1d=q1d,
                L=L, R=R)


def run_domain(ref, cfgs, A, boot=True, nboot=200, tag=""):
    G, T, bonds = ref["G"], ref["T"], ref["bonds"]
    pix, cc, dqq, H_ref, R0c = ref["pix"], ref["cc"], ref["dqq"], ref["H_ref"], ref["R0c"]
    ncfg = len(cfgs)
    I_sum = None
    Fs = []
    us = []
    for ic in range(ncfg):
        pts, _cv = embed_config(G, T, cfgs[ic])
        bl = float(np.mean([np.hypot(*(pts[a] - pts[b])) for a, b in bonds]))
        pn = pts / bl
        I, _ = pl.structure_factor(pn, q_max=Q_MAX, q_pixels=Q_PIX)
        I_sum = I if I_sum is None else I_sum + I
        Fs.append(S.F_pts(pn, pix, cc, dqq))
        pc = pn - pn.mean(axis=0)
        s = float((pc * R0c).sum() / (R0c * R0c).sum())
        us.append(pc - s * R0c)
    I_mean = (I_sum / ncfg).astype(np.float64)
    Farr = np.array(Fs)
    f_coh = np.abs(Farr.mean(axis=0)) ** 2 / H_ref
    f_pk = (np.abs(Farr) ** 2).mean(axis=0) / H_ref
    f_inc = f_pk - f_coh
    rows = []
    for kk, gx, gy, ix, iy in pix:
        r0 = S.halo_anisotropy(I_mean, ref["q1d"], gx, gy)
        WIN = S.WIN
        y0, y1 = max(int(iy - WIN / dqq), 0), min(int(iy + WIN / dqq) + 1, Q_PIX)
        x0, x1 = max(int(ix - WIN / dqq), 0), min(int(ix + WIN / dqq) + 1, Q_PIX)
        sub = I_mean[y0:y1, x0:x1]
        yy2, xx2 = np.mgrid[y0:y1, x0:x1]
        rrc = np.hypot((xx2 - cc) * dqq - gx, (yy2 - cc) * dqq - gy)
        rows.append(dict(spot=kk, core=float(sub[rrc <= 0.22].sum()),
                         halo=float(sub[(rrc >= 0.22) & (rrc <= 1.10)].sum()),
                         aniso=r0["aniso"], f_coh=float(f_coh[kk]),
                         f_inc=float(f_inc[kk])))
    hc = sum(r["halo"] for r in rows) / sum(r["core"] for r in rows)
    # displacement spectra in the ten resolved windows
    S_par = np.zeros(len(ref["khat_x"]))
    S_perp = np.zeros(len(ref["khat_x"]))
    for u in us:
        ux_k = ref["phase"] @ u[:, 0]
        uy_k = ref["phase"] @ u[:, 1]
        upar = ux_k * ref["khat_x"] + uy_k * ref["khat_y"]
        upx = ux_k - upar * ref["khat_x"]
        upy = uy_k - upar * ref["khat_y"]
        S_par += np.abs(upar) ** 2
        S_perp += np.abs(upx) ** 2 + np.abs(upy) ** 2
    S_par /= len(us)
    S_perp /= len(us)
    kmid = ref["kmag"]
    KBIN = 0.08
    ratios = []
    for lo in np.arange(0.26, 1.06 + 1e-9, KBIN):
        m = (kmid >= lo) & (kmid < lo + KBIN)
        if m.sum():
            ratios.append((round(lo, 2), float(S_perp[m].mean() / S_par[m].mean())))
    r10 = [v for k, v in ratios if k >= 0.34 - 1e-9]
    rr = np.mean(r10)
    rsd = np.std(r10, ddof=1)
    out = dict(tag=tag, ncfg=ncfg, F=ref["G"]["F"], f_coh=float(f_coh.mean()),
               f_inc=float(f_inc.mean()), halo_core=float(hc),
               r=float(np.mean([x["aniso"] for x in rows])),
               Sperp_Spar=float(rr), Sperp_Spar_sd=float(rsd), ratios=ratios)
    if boot:
        rngb = np.random.default_rng(20261011)
        fb = np.empty((nboot, 6))
        for b in range(nboot):
            idx = rngb.integers(0, ncfg, ncfg)
            fb[b] = np.abs(Farr[idx].mean(axis=0)) ** 2 / H_ref
        out["fcoh_lo"] = float(np.percentile(fb.mean(axis=1), 2.5))
        out["fcoh_hi"] = float(np.percentile(fb.mean(axis=1), 97.5))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "ladder"
    t0 = time.time()

    if mode == "ladder":
        # --- self check on the archived 13x13 a123 (canonical calibre) ---
        ref13 = build_reference(13, 7)
        d = np.load(RAW / "torus13_a123.npz")
        c123 = d["cfgs"].astype(np.int64)
        chk = run_domain(ref13, c123, (1, 2, 3), tag="a123_recheck")
        print(f"[check] a123 recheck: f_coh={chk['f_coh']:.3g} "
              f"(canonical 9.0e-4)  halo/core={chk['halo_core']:.2f} "
              f"(canonical 16.7)  r={chk['r']:.2f}  ({time.time()-t0:.0f}s)",
              flush=True)
        res = [chk]
        # --- B3: new rungs by exact translation ---
        for c, tag, A in ((5, "a678", (6, 7, 8)), (6, "a789", (7, 8, 9))):
            cfgs = c123 + c
            r = run_domain(ref13, cfgs, A, tag=tag)
            res.append(r)
            # archive the relabelled raw ensemble for provenance
            np.savez_compressed(RAW / f"torus13_{tag}.npz", cfgs=cfgs,
                                alphabet=np.array(A), eq=d["eq"], thin=d["thin"],
                                seed=d["seed"], h=0.0, L=13,
                                derived_from=np.array(["torus13_a123", str(c)]),
                                swap=d["swap"], acc=d["acc"], wall=d["wall"])
            print(f"[{tag}] f_coh={r['f_coh']:.4g} f_inc={r['f_inc']:.3g} "
                  f"halo/core={r['halo_core']:.2f} r={r['r']:.3f} "
                  f"Sperp/Spar={r['Sperp_Spar']:.4f}+-{r['Sperp_Spar_sd']:.4f}",
                  flush=True)
        rows = []
        for r in res:
            rr = dict(tag=r["tag"], npairs=r["ncfg"], F=r["F"],
                      sigma_mu=None, f_coh=r["f_coh"], f_inc=r["f_inc"],
                      fcoh_lo=r.get("fcoh_lo"), fcoh_hi=r.get("fcoh_hi"),
                      halo_core=r["halo_core"], r=r["r"],
                      Sperp_Spar=r["Sperp_Spar"], Sperp_Spar_sd=r["Sperp_Spar_sd"])
            rows.append(rr)
        pd.DataFrame(rows).to_csv(OUT / "ladder_ext.csv", index=False,
                                  encoding="utf-8-sig")
        print(f"ladder extension done ({time.time()-t0:.0f}s)", flush=True)

    elif mode == "size":
        rows = []
        for tag, A in (("a123", (1, 2, 3)), ("a456", (4, 5, 6)),
                       ("b12", (1, 2))):
            d = np.load(RAW / f"tor21_{tag}.npz")
            cfgs = d["cfgs"].astype(np.int64)
            for R in (6, 7, 8, 9, 10):
                ref = build_reference(21, R)
                r = run_domain(ref, cfgs, A, boot=False, tag=f"{tag}_R{R}")
                r["sigma_mu"] = float(np.std(A) / np.mean(A))
                r["domain"] = tag
                r["R"] = R
                r["N"] = ref["G"]["F"]
                rows.append(r)
                print(f"[{tag} R={R} N={r['N']}] f_coh={r['f_coh']:.4g} "
                      f"f_inc={r['f_inc']:.4g} halo/core={r['halo_core']:.2f} "
                      f"r={r['r']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(OUT / "size_scaling.csv", index=False,
                                  encoding="utf-8-sig")
        print(f"size scaling done ({time.time()-t0:.0f}s)", flush=True)
    else:
        raise SystemExit("mode must be ladder|size")


if __name__ == "__main__":
    import sys as _s
    main()
