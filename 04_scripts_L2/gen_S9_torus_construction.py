# Torus-construction drawing module (flat parallelogram -> cylinder -> torus).
# The standalone SI figure S9 was retired: the three panel-drawing functions
# are reused directly by the main Fig 2 (panel a imports
# gen_S9_torus_construction).  main() is for debugging only; rerunning it
# regenerates the standalone S9 files in the figures directory (not cited
# by the main text).
#
# The torus structure of the transfer-matrix counting (opposite-edge
# identification of a parallelogram honeycomb domain) carries the s_inf
# argument of the whole paper, so the deformation is drawn in full.  Three
# panels:
#   a  tiled counting domain: Lx x Ly pore cells (60 deg axes), opposite
#      edges marked A-A' (E2-direction pair) and B-B' (E1-direction pair);
#      dashed ghosts outside the domain = periodic copies; matched markers =
#      points identified as the same point after gluing.
#   b  roll-up (glue A-A'): the A seam becomes an axial line of the tube,
#      the B edges become end rings.  Honeycomb cells are mapped onto an
#      isometric cylinder (R = perimeter/2pi, a 45 deg arc per cell, shape
#      readable).
#   c  closure (glue B-B'): smooth torus (an 8x3 honeycomb wrap on the torus
#      must stretch; the honeycomb information is carried by panels a/b, so
#      this panel draws only the surface + two-colour loops) + subtle shading.
# Convention note: the drawing uses the standard shear-free embedding, while
# the counting uses the affine (sheared) fundamental domain; both give the
# same s_inf for the two families.
# Data: none (pure schematic); memory < 100 MB.
# Discipline: English-only labels, no panel titles, two-colour A/B through
# all three panels.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

S = 1.0
E1 = np.array([np.sqrt(3) * S, 0.0])          # axis 1 (horizontal)
E2 = np.array([np.sqrt(3) * S / 2, 1.5 * S])  # axis 2 (60 deg)
ANG = np.radians(np.arange(30, 390, 60))      # pointy-top hexagon vertices
HEX = np.stack([S * np.cos(ANG), S * np.sin(ANG)], axis=1)
LX, LY = 8, 3
CA, CB = '#e6811e', '#2166ac'                 # A=orange, B=blue, through all panels
GREY = '#4a4a4a'


# ------------------------------------------------------------------ panel a
def panel_a(ax):
    """Tiled counting domain: Lx x Ly solid hexagons + one ring of ghosts
    + parallelogram domain lines (two-colour A/B)
    + matched markers (identified pairs) + Lx/Ly arrows."""
    t = 0.5                                    # domain lines midway between pore-centre rows (e-param)
    d = -t * (E1 + E2)                         # domain corner
    for i in range(-1, LX + 1):
        for j in range(-1, LY + 1):
            if 0 <= i < LX and 0 <= j < LY:
                continue
            c = i * E1 + j * E2
            ax.add_patch(plt.Polygon(c + HEX, closed=True, facecolor='none',
                                     edgecolor='#6a6a6a', lw=0.55,
                                     ls=(0, (2.2, 1.2)), zorder=1.5))
    for i in range(LX):
        for j in range(LY):
            c = i * E1 + j * E2
            ax.add_patch(plt.Polygon(c + HEX, closed=True, facecolor='0.955',
                                     edgecolor=GREY, lw=0.5, zorder=2))
    # domain lines: A = left/right sides (E2 direction), B = top/bottom (E1 direction)
    A1 = [d, d + LY * E2]
    A2 = [d + LX * E1, d + LX * E1 + LY * E2]
    B1 = [d, d + LX * E1]
    B2 = [d + LY * E2, d + LY * E2 + LX * E1]
    for seg in (A1, A2):
        ax.plot(*zip(*seg), color=CA, lw=2.2, zorder=4, solid_capstyle='round')
    for seg in (B1, B2):
        ax.plot(*zip(*seg), color=CB, lw=2.2, zorder=4, solid_capstyle='round')
    # matched markers (identified pairs: filled = one side, hollow = the other)
    pmL = d + 0.5 * LY * E2
    pmR = d + LX * E1 + 0.5 * LY * E2
    pbB = d + 0.5 * LX * E1
    pbT = d + LY * E2 + 0.5 * LX * E1
    for ppt, mk, fc, ec in ((pmL, '^', CA, CA), (pmR, '^', 'white', CA),
                            (pbB, 'v', CB, CB), (pbT, 'v', 'white', CB)):
        ax.plot(*zip(ppt), marker=mk, ms=6.6, mfc='white', mec='none',
                zorder=5.6)                       # white underlay (so the thick line does not swallow the marker)
        ax.plot(*zip(ppt), marker=mk, ms=4.0, mfc=fc, mec=ec,
                mew=0.9 if fc == 'white' else 0.0, zorder=5.7)
    # side labels (offset outward, not on the line)
    ax.text(*(pmL + [-1.05, -0.55]), '$A$', fontsize=7.0, ha='center',
            va='center', color=CA)
    ax.text(*(pmR + [0.95, -0.10]), "$A'$", fontsize=7.0, ha='center',
            va='center', color=CA)
    ax.text(*(pbB + [0.0, -0.80]), '$B$', fontsize=7.0, ha='center',
            va='center', color=CB)
    ax.text(*(pbT + [0.0, 0.88]), "$B'$", fontsize=7.0, ha='center',
            va='center', color=CB)
    # Lx/Ly arrows (inside the domain)
    o = d + 1.15 * E1 + 0.55 * E2
    a1 = o + (LX - 1.9) * E1
    a2 = o + (LY - 1.55) * E2
    kw = dict(arrowstyle='-|>', lw=0.7, color='#1a1a1a', mutation_scale=3,
              shrinkA=0, shrinkB=0)
    ax.annotate('', xy=a1, xytext=o, arrowprops=kw)
    ax.annotate('', xy=a2, xytext=o, arrowprops=kw)
    ax.text(a1[0] - 0.10, a1[1] + 0.30, r'$L_x$', fontsize=6.6,
            ha='center', va='bottom', color='#1a1a1a')
    ax.text(a2[0] - 0.12, a2[1] + 0.12, r'$L_y$', fontsize=6.6, ha='right',
            va='bottom', color='#1a1a1a')
    ax.set_xlim(-1.35, 18.6)                   # tighten white margin: content fills the frame
    ax.set_ylim(-2.05, 4.70)                   # narrow band below: ghost bottom edge not cropped
    ax.set_aspect('equal')
    ax.axis('off')


# ---------------------------------------------------------- 3D helpers
def draw_mini_domain(ax, LXm=4, LYm=3, fs=6.0):
    """Miniature pore-cell domain + Lx/Ly arrows (reused as the inset of
    Fig 2 panel c; annotates the sheet structure, no A/B gluing semantics).
    N_F = Lx x Ly is self-evident."""
    for i in range(LXm):
        for j in range(LYm):
            c = i * E1 + j * E2
            ax.add_patch(plt.Polygon(c + HEX, closed=True, facecolor='0.955',
                                     edgecolor=GREY, lw=0.5, zorder=2))
    o = 1.15 * E1 + 0.35 * E2
    a1 = o + (LXm - 1.6) * E1
    a2 = o + (LYm - 1.15) * E2
    kw = dict(arrowstyle='-|>', lw=0.7, color='#1a1a1a', mutation_scale=3,
              shrinkA=0, shrinkB=0)
    ax.annotate('', xy=a1, xytext=o, arrowprops=kw)
    ax.annotate('', xy=a2, xytext=o, arrowprops=kw)
    ax.text(a1[0] - 0.10, a1[1] + 0.30, r'$L_x$', fontsize=fs,
            ha='center', va='bottom', color='#1a1a1a')
    ax.text(a2[0] - 0.12, a2[1] + 0.12, r'$L_y$', fontsize=fs,
            ha='right', va='bottom', color='#1a1a1a')
    ax.set_xlim(-1.1, LXm * np.sqrt(3) + 1.1)
    ax.set_ylim(-1.2, (LYm - 1) * 1.5 + 1.3)
    ax.set_aspect('equal')
    ax.axis('off')


def cam_dir(elev, azim):
    el, az = np.radians(elev), np.radians(azim)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                     np.sin(el)])


def add_sorted_polys(ax, polys3, tocam):
    """Painter's algorithm (with computed_zorder=False, draw order = add
    order): add cells one by one in ascending centroid depth."""
    order = sorted(polys3, key=lambda p: np.dot(np.mean(p, axis=0), tocam))
    for p in order:
        ax.add_collection3d(Poly3DCollection(
            [p], facecolors='0.955', edgecolors=GREY, linewidths=0.5))


def cell_polys(map_fn, normal_fn):
    """Per-cell polygons + per-vertex outward normals (for the
    all-vertices-visible criterion)."""
    polys3, vnrms = [], []
    for j in range(LY):
        for i in range(LX):
            c = i * E1 + j * E2
            v3 = np.array([map_fn(*(c + v)) for v in HEX])
            polys3.append(v3)
            vnrms.append(np.array([normal_fn(*(c + v)) for v in HEX]))
    return polys3, vnrms


def add_cells_front(ax, polys3, vnrms, tocam, keep=-0.05):
    """All-vertices-visible criterion (removes silhouette spikes)
    + painter-ordered addition of the honeycomb cells."""
    keep_idx = [k for k in range(len(polys3))
                if (vnrms[k] @ tocam).min() > keep]
    add_sorted_polys(ax, [polys3[k] for k in keep_idx], tocam)


def split_visible(pts3, normals, tocam):
    vis = (normals @ tocam) > 0
    segs_v, segs_h = [], []
    cur = [pts3[0]]
    for k in range(1, len(pts3)):
        cur.append(pts3[k])
        if vis[k] != vis[k - 1]:
            (segs_v if vis[k - 1] else segs_h).append(np.array(cur))
            cur = [pts3[k]]
    (segs_v if vis[-1] else segs_h).append(np.array(cur))
    return segs_v, segs_h


def loop_segs(pts3, normals, tocam):
    return split_visible(pts3, normals, tocam)


def draw_segs(ax, segs, color, lw, visible=True):
    for s_ in segs:
        if visible:
            ax.plot(s_[:, 0], s_[:, 1], s_[:, 2], color=color, lw=lw,
                    zorder=3, solid_capstyle='round')
        else:
            ax.plot(s_[:, 0], s_[:, 1], s_[:, 2], color=color, lw=lw,
                    ls=(0, (2.5, 1.6)), zorder=1)


# ------------------------------------------------------------------ panel b
def panel_b(ax, elev=22, azim=-55, th0=0.83):
    """Roll-up (glue A-A'): front view of the isometric cylinder; the A seam
    = axial line (rotated to face front),
    B = end rings (ellipses seen end-on)."""
    P = LX * np.sqrt(3)
    R = P / (2 * np.pi)                        # isometric radius (no tangential stretching of cells)
    H = LY * 1.5

    def cyl(x, y):
        th = th0 + 2 * np.pi * x / P
        return np.array([R * np.sin(th), y - H / 2, R * np.cos(th)])

    tocam = cam_dir(elev, azim)

    def cyl_nrm(x, y):
        th = th0 + 2 * np.pi * x / P
        return np.array([np.sin(th), 0, np.cos(th)])

    ths = np.linspace(0, 2 * np.pi, 160)
    nrm = np.stack([np.sin(ths), 0 * ths, np.cos(ths)], axis=1)
    ring_segs = [loop_segs(np.array([cyl(R * th_, y) for th_ in ths]),
                           nrm, tocam) for y in (0.0, H)]
    # opaque tube surface (subtle shading, same style as panel c): boundary
    # cells hug the surface, no floating patches
    # (hidden segments of the far ring not drawn: no information under the
    # opaque surface, and a dashed line would spill past the surface boundary)
    nth, ny = 72, 18
    thv = np.linspace(0, 2 * np.pi, nth + 1)
    yv = np.linspace(0, H, ny + 1)
    quads, shades = [], []
    for it in range(nth):
        for iy in range(ny):
            p00 = cyl(R * thv[it], yv[iy])
            p10 = cyl(R * thv[it + 1], yv[iy])
            p11 = cyl(R * thv[it + 1], yv[iy + 1])
            p01 = cyl(R * thv[it], yv[iy + 1])
            quads.append(np.array([p00, p10, p11, p01]))
            thm = 0.5 * (thv[it] + thv[it + 1])
            n = np.array([np.sin(thm), 0, np.cos(thm)])
            shades.append(0.5 + 0.5 * float(np.dot(n, tocam)))
    order = sorted(range(len(quads)),
                   key=lambda k: np.dot(np.mean(quads[k], axis=0), tocam))
    for k in order:
        g_ = 0.86 + 0.11 * shades[k]
        ax.add_collection3d(Poly3DCollection([quads[k]],
                                             facecolors=(g_, g_, g_),
                                             edgecolors='none'))
    polys3, vnrms = cell_polys(cyl, cyl_nrm)
    add_cells_front(ax, polys3, vnrms, tocam, keep=0.12)
    seam = np.array([cyl(0, y) for y in np.linspace(0, H, 60)])
    # note: th0 is only a visual rotation (the glued seam stays at the
    # domain x=0 line); cells rotate with it, gluing relations unchanged
    ax.plot(seam[:, 0], seam[:, 1], seam[:, 2], color=CA, lw=2.2, zorder=3,
            solid_capstyle='round')
    near_ring = np.array([cyl(R * th_, 0.0) for th_ in ths])   # near ring fully visible
    ax.plot(near_ring[:, 0], near_ring[:, 1], near_ring[:, 2], color=CB,
            lw=2.0, zorder=3, solid_capstyle='round')
    draw_segs(ax, ring_segs[1][0], CB, 2.0, visible=True)      # far ring visible arcs
    draw_segs(ax, ring_segs[1][1], CB, 1.3, visible=False)     # far ring hidden segments dashed
    # (drawn on top: engineering-drawing convention, edges hidden from the view are dashed)
    seam_mid = cyl(0, H * 0.50)
    ax.text(seam_mid[0] - 0.58, seam_mid[1] + 0.02, seam_mid[2] + 0.52,
            '$A$', fontsize=7.0, color=CA, zorder=6,
            path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
    ring_lab = cyl(R * (np.pi + 0.62), -H / 2)
    ax.text(ring_lab[0] - 0.30, -H / 2 - 0.52, ring_lab[2] - 0.10,
            '$B$', fontsize=7.0, color=CB, ha='center')
    Lb = max(R + 0.35, H / 2 + 0.55)       # equal-scale limit (keeps sort/culling criteria exact)
    Lb = Lb * 0.74                          # tighten white margin (do not squeeze further: at 0.62
    # the data window sits below the cylinder display range and both sides get
    # vertically cropped - regression-tested)
    ax.set_xlim(-Lb, Lb)
    ax.set_ylim(-Lb, Lb)
    ax.set_zlim(-Lb, Lb)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')
    ax.axis('off')


# ------------------------------------------------------------------ panel c
def panel_c(ax, elev=40, azim=-74):
    """Closure (glue B-B'): smooth subtly-shaded torus + A meridian (outer
    equator) + B tube circle."""
    r, R2 = 1.15, 2.35
    P = LX * np.sqrt(3)
    Q = LY * 1.5

    def tor(th, ph):
        return np.array([(R2 + r * np.cos(th)) * np.cos(ph),
                         r * np.sin(th),
                         (R2 + r * np.cos(th)) * np.sin(ph)])

    tocam = cam_dir(elev, azim)
    # record the loops' hidden segments first (surface drawn as cells, visible segments last)
    ths = np.linspace(0, 2 * np.pi, 160)
    loopB = np.array([tor(th_, 0.0) for th_ in ths])
    nrmB = np.stack([np.cos(ths), np.sin(ths), 0 * ths], axis=1)
    phs = np.linspace(0, 2 * np.pi, 160)
    loopA = np.array([tor(0.0, ph_) for ph_ in phs])
    nrmA = np.stack([np.cos(phs), 0 * phs, np.sin(phs)], axis=1)
    segsB = loop_segs(loopB, nrmB, tocam)
    segsA = loop_segs(loopA, nrmA, tocam)
    # fine parametric-surface mesh (subtle shading, no edges)
    nth, nph = 72, 120
    thv = np.linspace(0, 2 * np.pi, nth + 1)
    phv = np.linspace(0, 2 * np.pi, nph + 1)
    quads, shades = [], []
    for it in range(nth):
        for ip in range(nph):
            c00 = tor(thv[it], phv[ip])
            c10 = tor(thv[it + 1], phv[ip])
            c11 = tor(thv[it + 1], phv[ip + 1])
            c01 = tor(thv[it], phv[ip + 1])
            quads.append(np.array([c00, c10, c11, c01]))
            n = np.cross(c10 - c00, c01 - c00)
            n = n / np.linalg.norm(n)
            shades.append(0.5 + 0.5 * float(np.dot(n, tocam)))
    order = sorted(range(len(quads)),
                   key=lambda k: np.dot(np.mean(quads[k], axis=0), tocam))
    for k in order:
        g = 0.88 + 0.08 * shades[k]
        ax.add_collection3d(Poly3DCollection([quads[k]], facecolors=(g, g, g),
                                             edgecolors='none'))
    # B tube circle and A meridian: hidden segments dashed + visible segments
    # solid, both drawn above the surface
    # (engineering-drawing convention: edges hidden from the view are dashed)
    draw_segs(ax, segsB[1], CB, 1.3, visible=False)
    draw_segs(ax, segsA[1], CA, 1.5, visible=False)
    draw_segs(ax, segsB[0], CB, 2.0, visible=True)
    draw_segs(ax, segsA[0], CA, 2.2, visible=True)
    halo = [pe.withStroke(linewidth=2.4, foreground='white')]
    ax.text(*(tor(0.0, 0.55) + [0.35, 0.30, 0.0]), '$A$', fontsize=7.0,
            color=CA, zorder=6, path_effects=halo)
    ax.text(*(tor(-0.60, 0.0) + [0.88, -0.05, 0.14]), '$B$', fontsize=7.0,
            color=CB, zorder=6, path_effects=halo)
    ext = R2 + r + 0.4
    xr = ext + 0.8                             # right margin for the B label
    ext_s = ext * 0.51                         # tighten white margin + enlarge: aligns the right edge of the B label
    ax.set_xlim(-ext_s, xr * 0.51)             # matches the Fig 2 right edge R1 (0.56 measured too small)
    ax.set_ylim(-ext_s, ext_s)
    ax.set_zlim(-ext_s, ext_s)
    ax.set_box_aspect((ext + xr, 2 * ext, 2 * ext))  # proportional to the ranges = unit equal scale
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type('ortho')
    ax.axis('off')


def main():
    # path resolution only here, so importing this module for its drawing
    # functions has no side effects
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paths_common as PC
    out_dir = PC.FIGURES

    fig = plt.figure(figsize=(7.2, 2.55))
    ax_a = fig.add_axes([0.005, 0.02, 0.275, 0.96])
    panel_a(ax_a)
    ax_b = fig.add_axes([0.335, 0.02, 0.29, 0.96], projection='3d',
                        computed_zorder=False)
    panel_b(ax_b)
    ax_c = fig.add_axes([0.685, 0.02, 0.31, 0.96], projection='3d',
                        computed_zorder=False)
    panel_c(ax_c)
    for x0, x1 in ((0.288, 0.335), (0.638, 0.685)):
        fig.add_artist(mpatches.FancyArrowPatch(
            (x0, 0.50), (x1, 0.50), transform=fig.transFigure,
            arrowstyle='-|>', mutation_scale=9, lw=1.1, color='#555555'))
    fig.text(0.3115, 0.60, "roll up: glue $A$-$A'$", fontsize=6.8, ha='center',
             color='#333333')
    fig.text(0.6615, 0.60, "close: glue $B$-$B'$", fontsize=6.8, ha='center',
             color='#333333')
    fig.savefig(out_dir / "S9_torus_construction.png", dpi=300)
    fig.savefig(out_dir / "S9_torus_construction.pdf")
    print('[QA] domain Lx=%d Ly=%d (schematic; symbolic L_x/L_y labels)'
          % (LX, LY))
    print('[QA] panel b: isometric cylinder R=P/2pi, cells %d around tube x '
          '%d along axis, back-face culled' % (LX, LY))
    print('[QA] panel c: smooth shaded torus r=%.2f R2=%.2f; loops A(meridian)'
          ' B(tube circle), hidden parts dashed' % (1.15, 2.35))
    print('[QA] colors A=%s B=%s through all panels' % (CA, CB))
    print('saved:', out_dir / "S9_torus_construction.png / .pdf")


if __name__ == "__main__":
    main()
