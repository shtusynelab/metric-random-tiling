# -*- coding: utf-8 -*-
"""gen_fig1_carrier_assets.py - Fig 1b four-carrier pattern asset generation.
Output (-> <FIGURES>/assets_carriers/):
  31212_triangles.png  3.12.12 patch: only triangles colored (dodecagons
                       white with outline); geometry = real coordinates from
                       z3_31212/step3_dp_strips.build_31212_rect.
  4612_types.png       4.6.12 patch: face-type coloring (three colours:
                       dodecagon / hexagon / square); analytic construction -
                       dodecagon centres on a triangular lattice (lattice
                       spacing 1/tan(15 deg) + 1), squares fit the two-body
                       gaps, hexagons fit the triangular gaps, all closures
                       equiangular and exact.
Both figures show **ideal geometry** (equal edge length 1) patterns; the
metric disorder is read out by the counting built on them (Fig 1c)."""
import math
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

OUT = PC.FIGURES / "assets_carriers"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PC.engine("z3_31212")))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402


def gen_31212():
    import step3_dp_strips as s3
    t = s3.build_31212_rect(4, 4)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    for f in t.faces:
        pts = np.asarray(f[1], dtype=float)
        if f[2] == 3:
            ax.add_patch(Polygon(pts, closed=True, facecolor='#e6811e',
                                 edgecolor='#8a4b00', lw=0.9, zorder=3))
        else:
            ax.add_patch(Polygon(pts, closed=True, facecolor='#f4f7fa',
                                 edgecolor='#2f5d8a', lw=0.8, zorder=2))
    allp = np.vstack([np.asarray(f[1], dtype=float) for f in t.faces])
    ax.set_xlim(allp[:, 0].min() - 0.2, allp[:, 0].max() + 0.2)
    ax.set_ylim(allp[:, 1].min() - 0.2, allp[:, 1].max() + 0.2)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0.05)
    p = OUT / "31212_triangles.png"
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("saved", p)


def gen_4612():
    """Analytic 4.6.12: dodecagons (edge 1, vertex angles 15 deg + 30k deg)
    centred on a triangular lattice d0 = 1/tan(15 deg) + 1;
    squares fill the adjacent-pair gaps (edge 1); hexagons fill the
    triangular gaps (edge 1, touching three dodecagons exactly along edges)."""
    s = 1.0
    R12 = s / (2 * math.sin(math.pi / 12))
    ap12 = s / (2 * math.tan(math.pi / 12))
    d0 = 2 * ap12 + s                       # centre-to-centre distance of adjacent dodecagons
    a1 = np.array([d0, 0.0])
    a2 = np.array([d0 / 2, d0 * math.sqrt(3) / 2])
    cells = [(m, n) for m in range(-2, 5) for n in range(-2, 5)]
    centers = {c: c[0] * a1 + c[1] * a2 for c in cells}

    def regular(cx, cy, R, k, phi0):
        return np.array([[cx + R * math.cos(phi0 + 2 * math.pi * t / k),
                          cy + R * math.sin(phi0 + 2 * math.pi * t / k)]
                         for t in range(k)])

    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    phi12 = math.radians(15)                # vertex angle 15 deg + 30k deg
    for c in cells:
        cx, cy = centers[c]
        ax.add_patch(Polygon(regular(cx, cy, R12, 12, phi12), closed=True,
                             facecolor='#dce9f5', edgecolor='#2f5d8a',
                             lw=0.7, zorder=2))
    # squares: adjacent-pair gaps
    sq_done = set()
    R2, phi_sq = s / math.sqrt(2), math.radians(0)
    for m in range(-2, 4):
        for n in range(-2, 4):
            for d in [(1, 0), (0, 1), (-1, 1)]:
                p = (m, n)
                q = (m + d[0], n + d[1])
                if p not in centers or q not in centers:
                    continue
                key = tuple(sorted([p, q]))
                if key in sq_done:
                    continue
                sq_done.add(key)
                mid = (centers[p] + centers[q]) / 2
                ang = math.atan2(centers[q][1] - centers[p][1],
                                 centers[q][0] - centers[p][0])
                ax.add_patch(Polygon(regular(mid[0], mid[1], R2, 4,
                                             ang - math.radians(45)),
                                     closed=True, facecolor='#fdeaea',
                                     edgecolor='#a33', lw=0.7, zorder=3))
    # hexagons: one per triangular gap, two orientations (2 per unit cell)
    R6 = s
    for m in range(-2, 4):
        for n in range(-2, 4):
            for trio in ([(m, n), (m + 1, n), (m, n + 1)],
                         [(m + 1, n), (m, n + 1), (m + 1, n + 1)]):
                if not all(c in centers for c in trio):
                    continue
                cen = np.mean([centers[c] for c in trio], axis=0)
                # edge normals must point at the three dodecagon centres:
                # take the first corner direction, phi0 = (theta - 30) % 60
                th = math.degrees(math.atan2(centers[trio[0]][1] - cen[1],
                                             centers[trio[0]][0] - cen[0]))
                phi0 = math.radians((th - 30.0) % 60.0)
                ax.add_patch(Polygon(regular(cen[0], cen[1], R6, 6, phi0),
                                     closed=True, facecolor='#e2f0e2',
                                     edgecolor='#2d6a2d', lw=0.7, zorder=2))
    allp = np.array([centers[c] for c in cells])
    ax.set_xlim(allp[:, 0].min() - 2.2, allp[:, 0].max() + 2.2)
    ax.set_ylim(allp[:, 1].min() - 2.2, allp[:, 1].max() + 2.2)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=0.05)
    p = OUT / "4612_types.png"
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    gen_31212()
    gen_4612()
