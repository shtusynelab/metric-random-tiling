# -*- coding: utf-8 -*-
"""Historical reference calculator (v1/v2 era; superseded by the H_ref
estimator in vd_suite3.py); kept for provenance.

First-shell peak height of the ideal reference (uniform-tile disc): the
per-domain H_ideal, used for suppression ratios H / H_ideal.
Uniform tiling = the all-minimum tile (hexagon all-{min}); same n = 8, same
q grid, mean-bond-length normalised."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(PC.HC))
import pseudocrystal_lib as pl  # noqa: E402
from vd_suite import metric_library, geom_of, CLASSES, N_RING, Q_MAX, Q_PIX  # noqa: E402

for name, A in CLASSES:
    lib = metric_library(A)
    idx_u = int(np.argmax(np.all(lib.weights == min(A), axis=1)))
    assert np.all(lib.weights[idx_u] == min(A))
    sol = {cc: idx_u for cc in pl.hex_coords(N_RING - 1)}
    pts, bonds, bl, _ = geom_of(sol, lib)
    I, q1d = pl.structure_factor(pts / bl, q_max=Q_MAX, q_pixels=Q_PIX)
    H = float(I.max())
    # maximum inside the first-shell ring (excludes any low-q artefact peak)
    c = I.shape[0] // 2
    dq = q1d[1] - q1d[0]
    yy, xx = np.mgrid[0:I.shape[0], 0:I.shape[1]]
    rr = np.hypot((xx - c) * dq, (yy - c) * dq)
    m = (rr >= 0.85 * 4 * np.pi / 3 * np.sqrt(3)) & (rr <= 1.15 * 4 * np.pi / 3 * np.sqrt(3))
    H_shell = float(I[m].max())
    print(f"[{name}] alphabet={A} H_ideal(global)={H:.6g}  H_ideal(shell)={H_shell:.6g}",
          flush=True)
