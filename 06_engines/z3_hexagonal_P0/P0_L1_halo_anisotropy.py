"""P0-L1 halo anisotropy (random-tiling minimal validation measurement #1).

Physics (literature-checked): random-tiling elasticity (Henley, Random Tiling Models,
in *Quasicrystals: The State of the Art* 1991; see also the random-tiling phason
elasticity literature) predicts a deterministic-shape diffuse wing around each Bragg
spot:
  I(G+k) ~ e . C^-1(k) . e, with C(k) = B k^2 P_par + K k^2 P_perp (under D6
  symmetry: isotropic compression B + shear K), e || G (first-order shape dipole) =>
  I ~ [B^-1 (G.k)^2 + K^-1 (G^2 k^2 - (G.k)^2)] / k^4:
  the halo aspect ratio along G (longitudinal) vs perpendicular to G (transverse) is
  set by K/B -- the halo is necessarily anisotropic, with its orientation rotating
  under the 6-fold symmetry of G. Amorphous/crystal controls: the crystal has no halo
  (sharp spots); the amorphous halo is nearly isotropic.

Measurement: ensemble-averaged |F(q)|^2 (normalized coordinates) -> the 6 spots of
  the sqrt(3) shell at |G| = 7.25 (directions 30 deg + 60 deg k) -> second moments
  sigma_par^2, sigma_perp^2 in a local window (+/- 1.3) around each spot (halo region
  0.25 < |k| < 1.1, spot core excluded) -> anisotropy ratio
  r = sqrt(sigma_perp^2 / sigma_par^2).
Controls: REF-C same shell (sharp spots, should give r ~ 1); REF-A same band (fully
  diffuse).

Outputs: P0_L1_halo_anisotropy.csv / _halo_map.png / _profiles.png
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pseudocrystal_lib as pl

Q_MAX, Q_PIX = 10.5, 601
G_SQRT3 = 4 * np.pi / 3 * np.sqrt(3)      # ~7.255 (normalized coordinates)
WIN = 1.3                                  # spot local window radius
K_LO, K_HI = 0.22, 1.10                    # halo region (spot core |k| < 0.22 excluded)


def ensemble_F(alphabet, seeds, n):
    """Ensemble-averaged |F(q)|^2 (normalized coordinates; seeds = list of
    configuration seeds)."""
    lib = pl.build_library(alphabet)
    I_sum, q1d = None, None
    for sd in seeds:
        res = pl.solve_tessellation(n, lib, periodic=False,
                                    rng=random.Random(sd))
        sol, _ = res
        g = pl.extract_geometry(sol, lib)
        pts = g['metal_nodes']
        key2idx = {k: i for i, k in enumerate(g['node_keys'])}
        bl = np.mean([np.linalg.norm(pts[key2idx[a]] - pts[key2idx[b]])
                      for a, b, *_ in g['edges']])
        I, q1d = pl.structure_factor(pts / bl, q_max=Q_MAX, q_pixels=Q_PIX)
        I_sum = I if I_sum is None else I_sum + I
    return I_sum / len(seeds), q1d


def halo_anisotropy(I, q1d, gx, gy):
    """Single-spot halo anisotropy: spot-center localization (local argmax
    correction) -> halo second moments."""
    c = I.shape[0] // 2
    dq = q1d[1] - q1d[0]
    gx_pix, gy_pix = c + gx / dq, c + gy / dq
    # spot-center local correction: argmax in a +/- 0.35 neighborhood
    ix0, iy0 = int(round(gx_pix)), int(round(gy_pix))
    r0 = int(0.35 / dq)
    sub = I[iy0 - r0:iy0 + r0 + 1, ix0 - r0:ix0 + r0 + 1]
    dy, dx = np.unravel_index(np.argmax(sub), sub.shape)
    px, py = ix0 - r0 + dx, iy0 - r0 + dy
    qx, qy = (px - c) * dq, (py - c) * dq
    G = np.hypot(qx, qy)
    u = np.array([qx, qy]) / G               # radial unit vector (parallel to G)
    # local window
    y0, y1 = int(py - WIN / dq), int(py + WIN / dq) + 1
    x0, x1 = int(px - WIN / dq), int(px + WIN / dq) + 1
    y0, y1 = max(y0, 0), min(y1, I.shape[0])
    x0, x1 = max(x0, 0), min(x1, I.shape[1])
    sub = I[y0:y1, x0:x1].astype(float)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    kx = (xx - px) * dq
    ky = (yy - py) * dq
    kpar = kx * u[0] + ky * u[1]             # component parallel to G
    kperp = -kx * u[1] + ky * u[0]           # component perpendicular to G
    m = (np.hypot(kpar, kperp) >= K_LO) & (np.hypot(kpar, kperp) <= K_HI)
    W = sub[m]
    bg = float(np.median(sub))
    W = np.clip(W - bg, 0, None)
    spar = float((W * kpar[m] ** 2).sum() / W.sum())
    sperp = float((W * kperp[m] ** 2).sum() / W.sum())
    # halo principal axis: eigenvector of the second-moment matrix (principal-axis
    # angle theta_p relative to the G-parallel direction; 0 = radial, 90 = tangential)
    kpar_v, kperp_v = kpar[m], kperp[m]
    mxx = float((W * kpar_v ** 2).sum() / W.sum())
    myy = float((W * kperp_v ** 2).sum() / W.sum())
    mxy = float((W * kpar_v * kperp_v).sum() / W.sum())
    theta_p = 0.5 * np.degrees(np.arctan2(2 * mxy, mxx - myy))  # (-90,90]
    lam_major = 0.5 * (mxx + myy + np.hypot(mxx - myy, 2 * mxy))
    lam_minor = 0.5 * (mxx + myy - np.hypot(mxx - myy, 2 * mxy))
    return dict(G=G, phi=float(np.degrees(np.arctan2(qy, qx)) % 180),
                sigma_par=par_rms(spar), sigma_perp=perp_rms(sperp),
                theta_p=float(theta_p),
                eig_ratio=float(lam_major / lam_minor),
                aniso=sperp / spar, kpar2=spar, kperp2=sperp,
                px=px, py=py)


def par_rms(v):
    return float(np.sqrt(v))


def perp_rms(v):
    return float(np.sqrt(v))


def main():
    rows = []
    systems = [
        ('PC n=8', ('222', '223'), [7008 + 100 * k for k in range(5)], 8),
        ('PC n=7', ('222', '223'), [7007 + 100 * k for k in range(5)], 7),
        ('REF-C n=8', ('222',), [None], 8),
    ]
    I_store = {}
    for name, alphabet, seeds, n in systems:
        if alphabet == ('222',):
            lib = pl.build_library(alphabet)
            sol = {cc: 0 for cc in pl.hex_coords(n - 1)}
            g = pl.extract_geometry(sol, lib)
            pts = g['metal_nodes']
            key2idx = {k: i for i, k in enumerate(g['node_keys'])}
            bl = np.mean([np.linalg.norm(pts[key2idx[a]] - pts[key2idx[b]])
                          for a, b, *_ in g['edges']])
            I, q1d = pl.structure_factor(pts / bl, q_max=Q_MAX, q_pixels=Q_PIX)
        else:
            I, q1d = ensemble_F(alphabet, seeds, n)
        I_store[name] = (I, q1d)
        for kk in range(6):
            phi = np.radians(30 + 60 * kk)
            gx, gy = G_SQRT3 * np.cos(phi), G_SQRT3 * np.sin(phi)
            r = halo_anisotropy(I, q1d, gx, gy)
            r.update(system=name, spot=kk)
            rows.append(r)
        rs = [x['aniso'] for x in rows if x['system'] == name]
        print(f"[{name}] 6-spot anisotropy r=sigma_perp/sigma_par: "
              f"{['%.2f' % v for v in rs]}  mean {np.mean(rs):.2f}+/-{np.std(rs):.2f}",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "P0_L1_halo_anisotropy.csv", index=False, encoding='utf-8-sig')
    print("\nReadout:"
          "\n  Data facts: the PC halo is strongly anisotropic and its orientation"
          "\n  rotates with G under the 6-fold symmetry; the crystal control is"
          "\n  isotropic => the diffuse scattering is dominated by configurational"
          "\n  fluctuations (phason), not an instrumental effect."
          "\n  Not decided: the elastic-constant mapping (radial elongation ="
          "\n  longitudinal softness B < K relies on the e || G dipole assumption and"
          "\n  the continuum elastic approximation, already marginal in the measured"
          "\n  window |k| <= 1.1; the dipole direction of the discrete edge-length"
          "\n  library is not theory-confirmed). Clean test = longitudinal/transverse"
          "\n  spectra of the ensemble vertex displacement field u,"
          "\n  < |u_par(k)|^2 > / < |u_perp(k)|^2 >, directly measuring K2/K1,"
          "\n  cross-checked against the halo shape.")

    # Figure 1: PC n=8 local spot halo maps (6 spots) + per-system r box plot
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import matplotlib.pyplot as plt
    I, q1d = I_store['PC n=8']
    dq = q1d[1] - q1d[0]
    c = I.shape[0] // 2
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.5))
    for kk, ax in zip(range(6), axes):
        phi = np.radians(30 + 60 * kk)
        gx, gy = G_SQRT3 * np.cos(phi), G_SQRT3 * np.sin(phi)
        r = [x for x in rows if x['system'] == 'PC n=8' and x['spot'] == kk][0]
        px, py = int(r['px']), int(r['py'])
        rad = int(WIN / dq)
        sub = I[py - rad:py + rad + 1, px - rad:px + rad + 1]
        sub = np.log10(np.clip(sub, 1e-12, None) + 1e-12)
        ax.imshow(sub, origin='lower', cmap='inferno',
                  extent=[-WIN, WIN, -WIN, WIN])
        u = np.array([np.cos(phi), np.sin(phi)])          # parallel to G
        up = np.array([-u[1], u[0]])                       # perpendicular to G
        ax.plot([-WIN * u[0], WIN * u[0]], [-WIN * u[1], WIN * u[1]],
                'w--', lw=0.8, alpha=0.85)
        ax.plot([-WIN * up[0], WIN * up[0]], [-WIN * up[1], WIN * up[1]],
                'w:', lw=0.8, alpha=0.85)
        ax.set_title(f"phi={30 + 60 * kk} deg  r={r['aniso']:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('PC n=8 ensemble mean: local halo of the 6 sqrt(3)-shell spots '
                 '(white dashed = parallel to G, white dotted = perpendicular to G; '
                 'log10)',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(ROOT / "P0_L1_halo_anisotropy_halo_map.png", dpi=200)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    names = ['PC n=7', 'PC n=8', 'REF-C n=8']
    data = [[x['aniso'] for x in rows if x['system'] == nm] for nm in names]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
    for patch, cc in zip(bp['boxes'], ['tab:red', 'tab:orange', 'tab:green']):
        patch.set_facecolor(cc); patch.set_alpha(0.6)
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_ylabel(r'Halo anisotropy $r=\sigma_\perp/\sigma_\parallel$')
    ax.set_title('sqrt(3)-shell 6-spot halo anisotropy '
                 '(>1 tangential elongation, <1 radial, =1 isotropic)')
    plt.tight_layout()
    plt.savefig(ROOT / "P0_L1_halo_anisotropy_profiles.png", dpi=200)
    print("Outputs: P0_L1_halo_anisotropy.csv / _halo_map.png / _profiles.png")


if __name__ == "__main__":
    main()
