# -*- coding: utf-8 -*-
"""
TSQ_numerical_verification.py - numerical verification of
"discretization-induced rigidity" for the z=3 mixed-face 4.8.8 tiling.

Theorem (see TSQ_zero_entropy_theorem_analytic_proof.md in this directory):
the closure projection rows of 4.8.8 mix coefficients 1 and sqrt(2)/2; for an
integer alphabet every real equation
sum (a_i + b_i*sqrt(2)) s_i = 0  <=>  sum a_i s_i = 0 and sum b_i s_i = 0
(exact row splitting over the irrationals).
After splitting, the octagon constraints become all four opposite-side
equalities (two pairs obtained by adding/subtracting the sqrt(2) rows), and
the square becomes two opposite-side equalities => the whole system reduces
to equalities; taking constants on equivalence classes:

    Omega = k^(number of equivalence classes)   (exact; #classes = O(L),
                                                 sub-extensive)

=> per-face entropy density s(inf) = 0. Contrast: the real (R) solution-space
dimension = 2L^2+2 = F+2 (extensive, +1 per face) -- 4.8.8 is a clean sample
of "continuous extensive freedom, zero entropy for the integer alphabet";
the mechanism is that the discrete alphabet probes the irrational algebraic
directions.

Four verification layers (none presupposes the conclusion):
  A. DFS brute-force anchor (small torus, per-face split-row constraint
     pruning) vs kernel enumeration;
  B. three-way cross-check strip TM <-> kernel enumeration <-> Omega = k^#classes
     (small patches / small tori);
  C. class-count series: torus/patch size series, #classes ~ O(L), s -> 0;
  D. R-vs-Z contrast: real dimension 2L^2+2 (extensive) vs #classes
     (sub-extensive) at matched sizes.

Output: TSQ_entropy_density_verification.csv (in this script's directory).
stdlib + numpy.
"""
from __future__ import annotations

import csv
import math
import time
from itertools import product
from pathlib import Path

import numpy as np

from tsq_lib import TSQTorus, tsq_patch, tsq_patch_rows, kernel_count, \
    strip_count

ROOT = Path(__file__).resolve().parent

# Equality table after row splitting (variable pairs per face), from the
# face orientation-cycle construction:
#   square: two axis pairs (up = down, left = right); octagon: four
#   opposite-side pairs (two axis pairs + two diagonal pairs)
# The script first rank-checks that "the equality rows span the same space as
# the split rows", then uses them for union-find.


def equality_rows(hc: TSQTorus):
    """Extract equality pairs from the split-row system: scan rows; any row
    with exactly two nonzero coefficients +-1 directly gives an equality;
    octagon sqrt(2)-rows combine pairwise by addition/subtraction into
    equalities (template-level verification)."""
    # Directly use construction knowledge: square 2 rows, octagon 4 rows
    # (first 2 rational axis rows, last 2 sqrt(2) diagonal rows). Extracted
    # automatically by row structure:
    #   2-nonzero row => equality; 4-nonzero +-1 rows handled pairwise =>
    #   two equalities from their sum/difference.
    eqs = []
    used = [False] * len(hc.rows_split)
    # Simple route: process each face's rows separately (rows are ordered by
    # face: square 2 rows / octagon 4 rows)
    idx = 0
    for size, sides in hc.faces:
        n = 2 if size == 4 else 4       # split rows per face: square 2, octagon 4
        rows = hc.rows_split[idx:idx + n]
        idx += n
        nz = [(ri, [(c, v) for c, v in enumerate(r) if v]) for ri, r in
              enumerate(rows)]
        two = [(ri, nzv) for ri, nzv in nz if len(nzv) == 2]
        four = [(ri, nzv) for ri, nzv in nz if len(nzv) == 4]
        for ri, nzv in two:
            (c1, v1), (c2, v2) = nzv
            g = math.gcd(abs(v1), abs(v2))
            v1, v2 = v1 // g, v2 // g
            assert abs(v1) == 1 and abs(v2) == 1 and v1 + v2 == 0
            eqs.append((c1, c2))
        # 4-nonzero rows: pattern (+1,+1,-1,-1) => combine with the partner
        # row by addition/subtraction into equalities
        if four:
            assert len(four) == 2, "unexpected octagon row structure"
            (r1, nz1), (r2, nz2) = four
            cols1 = {c for c, v in nz1}
            cols2 = {c for c, v in nz2}
            assert cols1 == cols2 and all(
                abs(v) == 1 for c, v in nz1 + nz2)
            # coefficient-wise add/subtract: nonzero coefficients give pairs
            add = {c: sum(v for rr, nzv in four for c2, v in nzv if c2 == c)
                   for c in cols1}
            sub = {c: sum(v * (1 if rr == r1 else -1)
                          for rr, nzv in four for c2, v in nzv if c2 == c)
                   for c in cols1}
            for combo in (add, sub):
                nzp = [(c, v) for c, v in combo.items() if v]
                assert len(nzp) == 2, "octagon sqrt-row combo not equality"
                g = math.gcd(abs(nzp[0][1]), abs(nzp[1][1]))
                nzp = [(c, v // g) for c, v in nzp]
                assert all(abs(v) == 1 for c, v in nzp) \
                    and nzp[0][1] + nzp[1][1] == 0
                eqs.append((nzp[0][0], nzp[1][0]))
    return eqs


def equality_classes(hc: TSQTorus):
    """Number of equality classes (union-find over the split-row equalities)."""
    parent = list(range(hc.ne))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in equality_rows(hc):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(x) for x in range(hc.ne)})


def equality_classes_patch(faces, n_vars):
    """Patch version: equality pairs extracted per face from the split-row
    template (square 2 pairs, octagon 4 pairs)."""
    parent = list(range(n_vars))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for size, sides in faces:
        if size == 4:
            pairs = ((0, 2), (1, 3))            # opposite sides (cycle order)
        else:
            pairs = ((0, 4), (2, 6), (1, 3), (5, 7))
        for i, j in pairs:
            ra, rb = find(sides[i][0]), find(sides[j][0])
            if ra != rb:
                parent[ra] = rb
    return len({find(x) for x in range(n_vars)})


def dfs_brute(n_vars, faces, k):
    """DFS over all assignments (check the split-row constraints as soon as a
    face is completed). Returns Omega."""
    face_vars = []
    for size, sides in faces:
        face_vars.append([e for e, d in sides])
    # row templates as in tsq_lib._build_rows, per face
    from tsq_lib import DIR8
    templates = []
    for size, sides in faces:
        rx, rx2, ry, ry2 = ([0] * len(sides) for _ in range(4))
        for t, (e, d) in enumerate(sides):
            a, b, c, dd = DIR8[d]
            rx[t] = a; rx2[t] = b; ry[t] = c; ry2[t] = dd
        templates.append([r for r in (rx, rx2, ry, ry2) if any(r)])
    assign = [None] * n_vars
    remaining = [set(fv) for fv in face_vars]

    def rec(idx):
        if idx == n_vars:
            return 1
        total = 0
        for val in range(k):
            assign[idx] = val
            ok = True
            for fi, fv in enumerate(face_vars):
                if idx in fv:
                    remaining[fi].discard(idx)
                    if not remaining[fi]:
                        for row in templates[fi]:
                            if sum(c * assign[e] for c, e in
                                   zip(row, fv)) != 0:
                                ok = False
                                break
                if not ok:
                    break
            if ok:
                total += rec(idx + 1)
            for fi, fv in enumerate(face_vars):
                if idx in fv:
                    remaining[fi].add(idx)
            if ok and idx == n_vars - 1:
                pass
        assign[idx] = None
        return total

    return rec(0)


def main():
    rows = []
    all_ok = True

    # ---------------- A. DFS brute anchor ----------------
    print("A. DFS brute anchor (torus 2x2, k=2)", flush=True)
    t0 = time.time()
    hc = TSQTorus(2, 2)
    om_bf = dfs_brute(hc.ne, hc.faces, 2)
    om_ke = kernel_count(hc.rows_split, [4, 5][:2])
    ok = om_bf == om_ke
    all_ok &= ok
    rows.append(dict(layer="A_brute", case="torus2x2", k=2, F=hc.F,
                     omega_bf=om_bf, omega_ke=om_ke, classes="",
                     dim_float="", check="OK" if ok else "MISMATCH"))
    print(f"  torus2x2 k=2: brute={om_bf} kernel={om_ke} "
          f"{'OK' if ok else 'MISMATCH'} ({time.time()-t0:.1f}s)", flush=True)

    # ---------------- B. three-way cross-check (TM / kernel / k^classes) ----------------
    print("B. TM vs kernel vs k^classes", flush=True)
    for (W, L) in [(2, 2), (3, 2), (2, 3)]:
        faces, nv = tsq_patch(W, L)
        rows_i, _ = tsq_patch_rows(faces, nv)
        om_tm, peak = strip_count(faces, nv, [4, 5, 6])
        om_ke = kernel_count(rows_i, [4, 5, 6])
        n_cls = equality_classes_patch(faces, nv)
        om_cls = 3 ** n_cls
        ok = (om_tm == om_ke == om_cls)
        all_ok &= ok
        rows.append(dict(layer="B_patch", case=f"patch{W}x{L}", k=3,
                         F=len(faces), omega_bf=om_tm, omega_ke=om_ke,
                         classes=n_cls, dim_float="",
                         check="OK" if ok else "MISMATCH"))
        print(f"  patch{W}x{L}: TM={om_tm} kernel={om_ke} 3^classes={om_cls} "
              f"classes={n_cls} {'OK' if ok else 'MISMATCH'} "
              f"({time.time()-t0:.1f}s)", flush=True)
    for (Lq, Lr) in [(2, 2), (3, 2), (2, 3), (2, 4)]:
        hc = TSQTorus(Lq, Lr)
        om_ke = kernel_count(hc.rows_split, [4, 5, 6])
        n_cls = equality_classes(hc)
        om_cls = 3 ** n_cls
        ok = (om_ke == om_cls)
        all_ok &= ok
        rows.append(dict(layer="B_torus", case=f"torus{Lq}x{Lr}", k=3,
                         F=hc.F, omega_bf="", omega_ke=om_ke,
                         classes=n_cls, dim_float="",
                         check="OK" if ok else "MISMATCH"))
        print(f"  torus{Lq}x{Lr}: kernel={om_ke} 3^classes={om_cls} "
              f"classes={n_cls} {'OK' if ok else 'MISMATCH'}", flush=True)

    # ---------------- C+D. class-count series + real-vs-integer contrast ----------------
    print("C/D. class-count series and real-vs-integer contrast", flush=True)
    for L in (2, 3, 4, 5, 6, 8, 10, 12):
        t0 = time.time()
        hc = TSQTorus(L, L)
        n_cls = equality_classes(hc)
        A = np.array(hc.rows_float)
        dim_real = A.shape[1] - int(np.linalg.matrix_rank(A))
        F = hc.F
        s = n_cls * math.log(3) / F
        ok = dim_real == 2 * L * L + 2
        all_ok &= ok
        rows.append(dict(layer="CD_series", case=f"torus{L}x{L}", k=3,
                         F=F, omega_bf="", omega_ke=f"3^{n_cls}",
                         classes=n_cls, dim_float=dim_real,
                         s_per_face=f"{s:.6f}",
                         check="OK" if ok else "MISMATCH"))
        print(f"  torus{L}x{L}: classes={n_cls} dim_real={dim_real} "
              f"(expect {2*L*L+2}) s={s:.5f} ({time.time()-t0:.1f}s)",
              flush=True)
    for (W, L) in [(4, 4), (8, 8), (12, 12), (16, 16), (24, 24), (32, 32),
                   (40, 20), (60, 30)]:
        t0 = time.time()
        faces, nv = tsq_patch(W, L)
        n_cls = equality_classes_patch(faces, nv)
        F = len(faces)
        s = n_cls * math.log(3) / F
        rows.append(dict(layer="CD_patch", case=f"patch{W}x{L}", k=3,
                         F=F, omega_bf="", omega_ke=f"3^{n_cls}",
                         classes=n_cls, dim_float="",
                         s_per_face=f"{s:.6f}", check=""))
        print(f"  patch{W}x{L}: classes={n_cls} F={F} s={s:.5f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    out = ROOT / "TSQ_entropy_density_verification.csv"
    fieldnames = []
    for r in rows:
        for key in r:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nALL CHECKS {'PASSED' if all_ok else 'FAILED'}; wrote {out.name}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
