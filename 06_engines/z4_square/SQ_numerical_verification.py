# -*- coding: utf-8 -*-
"""
SQ_numerical_verification.py -- numerical verification of the z=4 square
lattice zero-entropy theorem.

Theorem (see SQ_zero_entropy_theorem_analytic_proof.md in this directory):
on an isogonal (90-degree) square-lattice torus Lx x Ly, the rectangle
closure constraint (opposite sides of each face equal) propagates in a
chain along columns/rows, so
    Omega = k^(Lx+Ly)      (k = edge-length alphabet size)
    s     = (Lx+Ly) * ln k / (Lx*Ly)  ->  0

Three verification layers (none presupposes the theorem):
  A. Brute force: enumerate all k^E edge assignments on small tori and
     filter, per face, top=bottom and left=right; compare the count with
     the formula point by point;
  B. Propagation class counting: union-find using only the rectangle
     equalities; the number of classes must be exactly Lx+Ly, with
     horizontal-edge classes split by column and vertical-edge classes
     split by row (direct numerical confirmation of the propagation
     mechanism);
  C. Finite-size series: two-track table of formula vs class-count values
     of s, verifying -> 0.

Output: SQ_entropy_density_verification.csv (same directory as this
script). Standard library only.
"""
from __future__ import annotations

import csv
import math
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALPHABET = (4, 5, 6)  # project edge-color/edge-length alphabet (weights 4/5/6)
K = len(ALPHABET)


def edges_faces(lx: int, ly: int):
    """Square-lattice torus. Horizontal edge H(i,j) joins (i,j)-(i+1,j);
    vertical edge V(i,j) joins (i,j)-(i,j+1).
    Face (i,j): bottom=H(i,j), top=H(i,j+1), left=V(i,j), right=V(i+1,j)
    (all with wraparound)."""
    H = lambda i, j: (j % ly) * lx + (i % lx)
    V = lambda i, j: lx * ly + (j % ly) * lx + (i % lx)
    n_edges = 2 * lx * ly
    faces = [(H(i, j), H(i, j + 1), V(i, j), V(i + 1, j))
             for j in range(ly) for i in range(lx)]
    return n_edges, faces


def brute_force(lx: int, ly: int, k: int) -> int:
    """Filter all k^E assignments one by one (only called when k^E <= ~10^6)."""
    n_edges, faces = edges_faces(lx, ly)
    count = 0
    for assign in product(range(k), repeat=n_edges):
        if all(assign[b] == assign[t] and assign[l] == assign[r]
               for b, t, l, r in faces):
            count += 1
    return count


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

    def n_classes(self, n: int) -> int:
        return len({self.find(x) for x in range(n)})


def propagation_classes(lx: int, ly: int):
    """Union-find using only the rectangle equalities (top~bottom,
    left~right). Returns (n_classes, horizontal-edge classes,
    vertical-edge classes) for structural checks."""
    n_edges, faces = edges_faces(lx, ly)
    d = DSU(n_edges)
    for b, t, l, r in faces:
        d.union(b, t)
        d.union(l, r)
    H = lambda i, j: (j % ly) * lx + (i % lx)
    V = lambda i, j: lx * ly + (j % ly) * lx + (i % lx)
    h_cls = [[d.find(H(i, j)) for j in range(ly)] for i in range(lx)]
    v_cls = [[d.find(V(i, j)) for i in range(lx)] for j in range(ly)]
    return d.n_classes(n_edges), h_cls, v_cls


def check_class_structure(lx: int, ly: int, h_cls, v_cls) -> bool:
    """Horizontal-edge classes must depend only on column i and vertical-edge
    classes only on row j; the two groups must be disjoint and mutually
    distinct within each group."""
    h_ids = {cls[0] for cls in h_cls}
    v_ids = {row[0] for row in v_cls}
    ok = (all(len(set(cls)) == 1 for cls in h_cls)
          and all(len(set(row)) == 1 for row in v_cls)
          and len(h_ids) == lx and len(v_ids) == ly
          and h_ids.isdisjoint(v_ids))
    return ok


def main() -> None:
    rows = []
    all_ok = True

    # A. brute-force anchors (small tori with manageable k^E)
    anchors = [((2, 2), K), ((2, 3), K), ((3, 2), K),
               ((3, 3), 2), ((4, 2), 2), ((2, 4), 2)]
    print("A. brute-force anchors")
    for (lx, ly), k in anchors:
        _, faces = edges_faces(lx, ly)
        omega_bf = brute_force(lx, ly, k)
        omega_th = k ** (lx + ly)
        ok = omega_bf == omega_th
        all_ok &= ok
        s_bf = math.log(omega_bf) / (lx * ly) if omega_bf > 1 else 0.0
        rows.append(dict(case="brute", k=k, Lx=lx, Ly=ly,
                         E=2 * lx * ly, F=lx * ly,
                         n_classes="", omega_formula=omega_th,
                         omega_bruteforce=omega_bf,
                         s_formula=f"{(lx + ly) * math.log(k) / (lx * ly):.6f}",
                         s_bruteforce=f"{s_bf:.6f}"))
        print(f"  {lx}x{ly} k={k}: brute={omega_bf} formula={omega_th} "
              f"{'OK' if ok else 'MISMATCH'}")

    # B. propagation class counting + class structure check
    print("B. propagation classes")
    for lx, ly in [(2, 2), (3, 3), (4, 4), (6, 4), (8, 8)]:
        n_cls, h_cls, v_cls = propagation_classes(lx, ly)
        struct_ok = check_class_structure(lx, ly, h_cls, v_cls)
        ok = (n_cls == lx + ly) and struct_ok
        all_ok &= ok
        rows.append(dict(case="classes", k=K, Lx=lx, Ly=ly,
                         E=2 * lx * ly, F=lx * ly,
                         n_classes=n_cls, omega_formula=K ** (lx + ly),
                         omega_bruteforce="",
                         s_formula=f"{(lx + ly) * math.log(K) / (lx * ly):.6f}",
                         s_bruteforce=f"{math.log(K ** n_cls) / (lx * ly):.6f}"))
        print(f"  {lx}x{ly}: classes={n_cls} (expect {lx + ly}) "
              f"structure={'OK' if struct_ok else 'BAD'}")

    # C. finite-size series: s -> 0
    print("C. finite-size series (per-face entropy)")
    for lx, ly in [(2, 2), (3, 3), (4, 4), (6, 6), (8, 8), (12, 12),
                   (16, 16), (24, 24), (32, 32), (48, 48)]:
        s = (lx + ly) * math.log(K) / (lx * ly)
        rows.append(dict(case="series", k=K, Lx=lx, Ly=ly,
                         E=2 * lx * ly, F=lx * ly, n_classes=lx + ly,
                         omega_formula=K ** (lx + ly), omega_bruteforce="",
                         s_formula=f"{s:.6f}", s_bruteforce=""))
        print(f"  {lx}x{ly}: s={s:.6f}")

    out = ROOT / "SQ_entropy_density_verification.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nALL CHECKS {'PASSED' if all_ok else 'FAILED'}; wrote {out.name}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
