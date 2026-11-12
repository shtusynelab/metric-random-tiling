# -*- coding: utf-8 -*-
# Recovered 2026-11-11 from the development build directory; original
# development location not preserved.
"""Historical v0.1 ensemble suite (superseded by vd_suite2.py/vd_suite3.py);
recovered copy, kept for provenance.

Value-domain ensemble suite v0.1 (theory-paper section 2.3 / Fig 5 data).
Three pure-metric value domains {1,2}/{1,2,3}/{1,2,4}, n_rings = 8
(N = 169 pores), 20 configurations per domain.
Sampling: pl.solve_tessellation (growth DFS; the pure-metric library has no
vertex constraints); seeds 20261100+k.
Library: TileLibrary built directly from closure_ok (1-based edge values;
asserted {1,2,3} -> 45, {1,2,4} -> 33, matching S6.2).
Metrics: ensemble-mean |F(q)|^2 (q_max = 10.5, 601 px, mean-bond-length
      normalised) -> first-shell six-spot peak heights H, spot core/halo
      shares, halo anisotropy r (halo_anisotropy calibre); displacement-field
      spectra S_perp/S_par (topology-identity mapping, mean-bond-length
      normalised, halo window 0.22-1.10).
Outputs: <DATA>/(ensemble npz, mean|F|^2 npy, metrics CSV, summary CSV).
Sampler note: single-seed single-restart DFS (non-uniform, same class as the
      legacy growth solver; v1 states this openly); the small-size uniformity
      audit was listed as a TODO."""
import sys, json, random
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
CLASSES = [("binary", (1, 2)), ("arith", (1, 2, 3)), ("nonarith", (1, 2, 4))]
Q_MAX, Q_PIX = 10.5, 601
OUT = PC.DATA
OUT.mkdir(exist_ok=True)


def metric_library(alphabet):
    """Pure-metric value-domain library: built directly on 1-based edge
    values (no vertex constraints / no all-5 adjacency ban / no perimeter
    bound)."""
    from itertools import product
    ws = [w for w in product(sorted(alphabet), repeat=6) if pl.closure_ok(w)]
    assert ws, f"empty library {alphabet}"
    W = np.array(ws, dtype=np.uint8)
    lib = pl.TileLibrary(
        alphabet=tuple(alphabet), allowed_c3=frozenset({0, 1, 2, 3}),
        colors=(W - min(alphabet)).astype(np.uint8), weights=W,
        all_yellow=np.zeros(len(W), bool),
        perimeter=W.sum(axis=1).astype(int), n_base=len(ws))
    return lib


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


def displacement_R0(n, lib):
    ref_sol = {cc: IDX_UNIFORM for cc in pl.hex_coords(n - 1)}
    ref = pl.extract_geometry(ref_sol, lib)
    R = np.array(ref["metal_nodes"], float)
    vm = topo_vertex_map(ref)
    return R, vm


def first_shell_metrics(I_mean, q1d):
    """First-shell six spots: peak height H (local argmax), core share, halo
    share (0.22-1.10 ring)."""
    c = I_mean.shape[0] // 2
    dq = q1d[1] - q1d[0]
    rows = []
    for kk in range(6):
        phi = np.radians(30 + 60 * kk)
        gx, gy = G_SQRT3 * np.cos(phi), G_SQRT3 * np.sin(phi)
        r = halo_anisotropy(I_mean, q1d, gx, gy)
        # core/halo shares: polar rings centred on the local spot centre
        px = c + gx / dq
        py = c + gy / dq
        y0, y1 = int(py - WIN / dq), int(py + WIN / dq) + 1
        x0, x1 = int(px - WIN / dq), int(px + WIN / dq) + 1
        y0, y1 = max(y0, 0), min(y1, I_mean.shape[0])
        x0, x1 = max(x0, 0), min(x1, I_mean.shape[1])
        sub = I_mean[y0:y1, x0:x1].astype(float)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot((xx - c) * dq - gx, (yy - c) * dq - gy)
        # spot-centre local correction (same scheme as halo_anisotropy)
        ix0, iy0 = int(round(px)), int(round(py))
        r0 = int(0.35 / dq)
        subc = I_mean[iy0 - r0:iy0 + r0 + 1, ix0 - r0:ix0 + r0 + 1]
        dy, dx = np.unravel_index(np.argmax(subc), subc.shape)
        pcx = (ix0 - r0 + dx - c) * dq
        pcy = (iy0 - r0 + dy - c) * dq
        rr = np.hypot((xx - c) * dq - pcx, (yy - c) * dq - pcy)
        H = float(subc.max())
        core = float(sub[rr <= 0.22].sum())
        halo = float(sub[(rr >= 0.22) & (rr <= 1.10)].sum())
        rows.append(dict(spot=kk, H=H, core=core, halo=halo,
                         aniso=r.get("aniso"), sx=r.get("sx"), sy=r.get("sy")))
    return rows


def main(smoke=False):
    n_cfg = 2 if smoke else N_CFG
    summary = []
    for name, A in CLASSES:
        lib = metric_library(A)
        print(f"[{name}] alphabet={A} tiles={lib.size}", flush=True)
        if A == (1, 2, 3):
            assert lib.size == 45, f"{name}: tiles={lib.size} != 45 (S6.2)"
        if A == (1, 2, 4):
            assert lib.size == 33, f"{name}: tiles={lib.size} != 33 (S6.2)"
        global IDX_UNIFORM
        IDX_UNIFORM = int(np.argmax(np.all(lib.weights == lib.weights[0], axis=1)))
        assert np.all(lib.weights[IDX_UNIFORM] == lib.weights[0])

        # sampling
        cfgs = []
        k = 0
        seed = SEED0
        while len(cfgs) < n_cfg:
            try:
                res = pl.solve_tessellation(N_RING, lib, periodic=False,
                                            rng=random.Random(seed))
                sol, _ = res
                pts, bonds, bl, g = geom_of(sol, lib)
                cfgs.append((seed, pts, bonds, bl, g))
                print(f"  cfg {len(cfgs)-1}: seed={seed} ok (V={len(pts)})", flush=True)
            except Exception as e:
                print(f"  seed={seed} reject ({type(e).__name__})", flush=True)
            seed += 1
            k += 1
            if k > 400:
                raise RuntimeError("seed exhaustion")

        # structure-factor ensemble + metrics
        I_mean, q1d = None, None
        for seed, pts, bonds, bl, g in cfgs:
            I, q1d = pl.structure_factor(pts / bl, q_max=Q_MAX, q_pixels=Q_PIX)
            I_mean = I if I_mean is None else I_mean + I
        I_mean /= len(cfgs)
        np.save(OUT / f"meanF2_{name}.npy", I_mean)
        rows = first_shell_metrics(I_mean, q1d)
        for r in rows:
            r["H_bg_sub"] = r["H"]
        Hs = [r["H"] for r in rows]
        cores = sum(r["core"] for r in rows)
        halos = sum(r["halo"] for r in rows)
        anisos = [r["aniso"] for r in rows]
        print(f"  [{name}] H per spot: {['%.3g' % h for h in Hs]}  "
              f"mean H={np.mean(Hs):.4g}", flush=True)
        print(f"  [{name}] 6-spot core total={cores:.4g} "
              f"halo total (0.22-1.10)={halos:.4g} "
              f"halo/core={halos/cores:.3f}", flush=True)
        print(f"  [{name}] halo anisotropy r = {['%.2f' % a for a in anisos]} "
              f"mean {np.mean(anisos):.2f}+-{np.std(anisos):.2f}", flush=True)

        # displacement-field spectra (mean-bond-length normalised)
        R0, ref_map = displacement_R0(N_RING, lib)
        bl_ref = np.mean([np.linalg.norm(
            R0[ref_map[e[3]]] - R0[ref_map[e[4]]])
            for cc in [0] for e in []] ) if False else None
        # R0 normalisation: uniform-tile bond length = alphabet minimum? Use
        # the reference geometry directly instead
        seg = []
        for e in pl.extract_geometry({cc: IDX_UNIFORM for cc in pl.hex_coords(N_RING - 1)},
                                     lib)["edges"]:
            pass
        # directly: R0 / bl_u
        gref = pl.extract_geometry({cc: IDX_UNIFORM for cc in pl.hex_coords(N_RING - 1)}, lib)
        pts_u, bonds_u, bl_u, _ = geom_of({cc: IDX_UNIFORM for cc in pl.hex_coords(N_RING - 1)}, lib)
        R0 = R0 / bl_u
        us = []
        for seed, pts, bonds, bl, g in cfgs:
            vmap = topo_vertex_map(g)
            assert set(vmap.keys()) == set(ref_map.keys()), "topology graph mismatch"
            r = np.zeros_like(R0)
            for tk, i in vmap.items():
                r[ref_map[tk]] = pts[i] / bl
            r = r - r.mean(axis=0)
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
        nb = int((K_HI - K_LO) / K_BIN)
        for b in range(nb):
            lo, hi = K_LO + b * K_BIN, K_LO + (b + 1) * K_BIN
            m = (kmag >= lo) & (kmag < hi)
            if m.sum() < 3:
                continue
            sp, st = S_par[m].mean(), S_perp[m].mean()
            summary.append(dict(system=name, k_mid=0.5 * (lo + hi),
                                S_par=sp, S_perp=st, ratio=st / sp,
                                n_k=int(m.sum())))
        mid = [s for s in summary if s["system"] == name]
        ratios = [s["ratio"] for s in mid]
        print(f"  [{name}] halo-window S_perp/S_par = {np.mean(ratios):.2f}"
              f"+-{np.std(ratios):.2f} "
              f"(K1/K2 equal-excitation readout, halo window {K_LO}-{K_HI})",
              flush=True)

        # ensemble configuration archive (per-configuration pts/bonds/bl/seed)
        np.savez_compressed(OUT / f"ens_{name}.npz",
                            seeds=np.array([c[0] for c in cfgs]),
                            bl=np.array([c[3] for c in cfgs]),
                            meta=np.array([N_RING, len(cfgs), SEED0]))
        import pandas as pd
        pd.DataFrame(rows).assign(system=name).to_csv(
            OUT / f"spots_{name}.csv", index=False, encoding="utf-8-sig")

    import pandas as pd
    pd.DataFrame(summary).to_csv(OUT / "displacement_spectra.csv",
                                 index=False, encoding="utf-8-sig")
    print("done", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
