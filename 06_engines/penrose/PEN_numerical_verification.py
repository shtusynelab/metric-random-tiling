# -*- coding: utf-8 -*-
"""
PEN_numerical_verification.py -- numerical verification of the strand
theorem for Penrose rhombus tilings.

Theorem (see PEN_strip_theorem_analytic_proof.md in this directory): in
any isogonal tiling by rhombi (parallelogram faces), the per-face
constraint is exactly "opposite sides equal"; the pairing propagates into
strands and the solution is constant on each strand => Omega =
k^(number of strands) exactly. In the Penrose pentagrid dual, strands <->
grid lines => the number of strands is O(window radius), s(infinity) = 0.

Verification layers:
  A. Structure: exactly 10 face types {i,j}, i != j; each face has 4 edges,
     opposite edges in the same family;
  B. Scaling: strands/faces -> 0 (strands O(R), faces O(R^2)), strand
     macroscopic length grows;
  C. Exact Omega: kernel enumeration of the equality rows = 3^(number of
     strands); DFS brute-force anchor on a micro patch.

Output: PEN_strip_verification.csv. stdlib + numpy.
"""
from __future__ import annotations

import csv
import math
import time
from itertools import product
from pathlib import Path

import numpy as np

from penrose_lib import make_penrose_patch, strand_analysis

ROOT = Path(__file__).resolve().parent


def equality_rows(faces, edges):
    """Full equality rows: each face has two pairs of opposite edges
    (e0,e2), (e1,e3) => row = e_a - e_b."""
    eid = {e: i for i, e in enumerate(edges)}
    rows = []
    for ck, sides in faces.items():
        for (ea, _), (eb, _) in ((sides[0], sides[2]), (sides[1], sides[3])):
            r = [0] * len(edges)
            r[eid[ea]] -= 1
            r[eid[eb]] += 1
            rows.append(r)
    return rows


def dfs_brute(faces, edges, k):
    """Micro-patch DFS: check both pairs of opposite edges for equality
    as soon as a face is completed."""
    eid = {e: i for i, e in enumerate(edges)}
    face_vars = [[eid[e] for e, f in sides] for sides in faces.values()]
    n_vars = len(edges)
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
                        a, b, c, d = fv
                        if not (assign[a] == assign[c]
                                and assign[b] == assign[d]):
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                total += rec(idx + 1)
            for fi, fv in enumerate(face_vars):
                if idx in fv:
                    remaining[fi].add(idx)
        assign[idx] = None
        return total

    return rec(0)


def kernel_count(rows, alphabet):
    from fractions import Fraction
    ncol = len(rows[0])
    mat = [[Fraction(x) for x in r] for r in rows]
    m = len(mat)
    pivots = []
    r = 0
    for c in range(ncol):
        p = None
        for rr in range(r, m):
            if mat[rr][c] != 0:
                p = rr
                break
        if p is None:
            continue
        mat[r], mat[p] = mat[p], mat[r]
        pv = mat[r][c]
        mat[r] = [x / pv for x in mat[r]]
        for rr in range(m):
            if rr != r and mat[rr][c]:
                f = mat[rr][c]
                mat[rr] = [x - f * y for x, y in zip(mat[rr], mat[r])]
        pivots.append(c)
        r += 1
        if r == m:
            break
    free = [c for c in range(ncol) if c not in set(pivots)]
    if not free:
        return 1
    basis = []
    for fc in free:
        vec = [Fraction(0)] * ncol
        vec[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            vec[pc] = -mat[i][fc]
        basis.append(vec)
    D = 1
    for vec in basis:
        for x in vec:
            if x:
                D = D * x.denominator // math.gcd(D, x.denominator)
    B = np.array([[int(x * D) for x in vec] for vec in basis], dtype=np.int64)
    A = np.array(sorted(set(alphabet)), dtype=np.int64)
    # hard budget gate: reject outright when (enumeration size) x (row width)
    # exceeds the limit
    n_enum = len(A) ** len(free)
    if n_enum * ncol > 2e8:
        raise RuntimeError(f"kernel enumeration {n_enum:.3g} x {ncol} "
                           f"exceeds budget 2e8; use smaller case")
    combos = np.array(list(product(A, repeat=len(free))), dtype=np.int64)
    X = combos @ B
    ok = (X % D == 0)
    return int(np.count_nonzero((ok & np.isin(X // D, A)).all(axis=1)))


def main():
    rows_out = []
    all_ok = True

    # ---------------- A. structure checks ----------------
    print("A. structure checks", flush=True)
    faces, edges = make_penrose_patch(3.0)
    n_str, n_faces, maxlen, hist = strand_analysis(faces, edges)
    types = {tuple(sorted((s[0][1], s[1][1]))) for s in faces.values()}
    ok = (len(types) == 10 and all(a < b for a, b in types)
          and all(len(s) == 4 for s in faces.values()))
    all_ok &= ok
    rows_out.append(dict(layer="A_struct", case="R=3", n_faces=n_faces,
                         n_edges=len(edges), strands=n_str, ratio="",
                         omega="", check="OK" if ok else "MISMATCH"))
    print(f"  R=3: faces={n_faces} types={len(types)} "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)

    # ---------------- B. scaling series ----------------
    print("B. scaling series", flush=True)
    for R in (2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 13.0):
        t0 = time.time()
        faces, edges = make_penrose_patch(R)
        n_str, n_faces, maxlen, hist = strand_analysis(faces, edges)
        s = n_str * math.log(3) / n_faces
        rows_out.append(dict(layer="B_scale", case=f"R={R}", n_faces=n_faces,
                             n_edges=len(edges), strands=n_str,
                             ratio=f"{n_str / n_faces:.5f}",
                             s_per_face=f"{s:.6f}", omega="", check=""))
        print(f"  R={R}: faces={n_faces} strands={n_str} "
              f"ratio={n_str/n_faces:.4f} s={s:.5f} max_len={maxlen} "
              f"({time.time()-t0:.1f}s)", flush=True)

    # ---------------- C. exact Omega ----------------
    print("C. exact Omega (small patches)", flush=True)
    for R in (1.0, 1.2):
        t0 = time.time()
        faces, edges = make_penrose_patch(R)
        n_str, n_faces, maxlen, hist = strand_analysis(faces, edges)
        eqr = equality_rows(faces, edges)
        om_ke = kernel_count(eqr, [4, 5, 6])
        om_cls = 3 ** n_str
        ok = om_ke == om_cls
        all_ok &= ok
        rows_out.append(dict(layer="C_omega", case=f"R={R}", n_faces=n_faces,
                             n_edges=len(edges), strands=n_str, ratio="",
                             omega=f"{om_ke}=3^{n_str}",
                             check="OK" if ok else "MISMATCH"))
        print(f"  R={R}: kernel={om_ke} 3^strands={om_cls} strands={n_str} "
              f"{'OK' if ok else 'MISMATCH'} ({time.time()-t0:.1f}s)",
              flush=True)
    # DFS brute-force anchor (micro window: pure-Python backtracking,
    # 2^classes as upper bound)
    faces, edges = make_penrose_patch(1.2)
    n_str, n_faces, maxlen, hist = strand_analysis(faces, edges)
    t0 = time.time()
    om_bf = dfs_brute(faces, edges, 2)
    ok = om_bf == 2 ** n_str
    all_ok &= ok
    rows_out.append(dict(layer="C_brute", case="R=1.2 k=2",
                         n_faces=n_faces, n_edges=len(edges),
                         strands=n_str, ratio="",
                         omega=f"{om_bf}=2^{n_str}",
                         check="OK" if ok else "MISMATCH"))
    print(f"  R=1.2 k=2: brute={om_bf} 2^strands={2**n_str} "
          f"{'OK' if ok else 'MISMATCH'} ({time.time()-t0:.1f}s)",
          flush=True)

    out = ROOT / "PEN_strip_verification.csv"
    fieldnames = []
    for r in rows_out:
        for key in r:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nALL CHECKS {'PASSED' if all_ok else 'FAILED'}; wrote {out.name}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
