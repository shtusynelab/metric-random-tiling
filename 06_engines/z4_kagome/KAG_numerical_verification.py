# -*- coding: utf-8 -*-
"""
KAG_numerical_verification.py -- numerical verification of the z=4 kagome
zero-entropy theorem.

Theorem (see KAG_zero_entropy_theorem_analytic_proof.md in this
directory): the isogonal metric kagome model = triangle variables t
(honeycomb vertices) + hexagon pair-sum closure constraints (2 per
hexagon). The constraint matrix M is a 2F x 2F square matrix; the Bloch
decomposition gives the symbol determinant

    det Mhat(k) = 3 (a-1)(b-1)(a-b) / a^2 b,   a = e^{ik1}, b = e^{ik2},

whose zero set is the winding triple of lines on the k torus
{k1=0} U {k2=0} U {k1=k2} =>

    dim ker M(Lq,Lr) = Lq + Lr + gcd(Lq,Lr) - 1   (exact, any size)

The nullity is sub-extensive (O(Lq+Lr)) => Omega <= k^dim => s(inf) = 0.

Four verification layers (none presupposes the conclusion):
  A. Brute-force anchors: filter all k^(2F) torus t assignments vs kernel
     enumeration vs the bound; on open-boundary patches, strip TM vs
     kernel enumeration cross-checks.
  B. Nullity table: dim vs formula Lq+Lr+gcd-1 (Fraction-exact for small
     matrices, large-prime field for large ones); continuous-k residuals
     of the symbol determinant against its factorization; zero-set grid
     counts and per-point 2x2 nullities (2 at the origin, 1 elsewhere).
  C. Exact Omega series (kernel enumeration): Omega <= k^dim holds point
     by point, s = lnOmega/F -> 0.
  D. Strip series (open-boundary TM): s(W,L) ~ c(W+L)/(WL) decay (the
    signature of open-boundary nullity O(perimeter)), consistent with the
    theorem; small patches already cross-checked bit-for-bit against
    kernel enumeration in A.

Output: KAG_entropy_density_verification.csv (same directory as this
script). stdlib + numpy.
"""
from __future__ import annotations

import csv
import math
import time
from itertools import product
from pathlib import Path

import numpy as np

from kagome_lib import (KagomeTorus, build_patch, count_assignments_reduced,
                        kernel_count_int, nullity_exact, nullity_modp,
                        patch_matrix, strip_count)

ROOT = Path(__file__).resolve().parent

# Local offsets of the symbol determinant: at corner k of face (q,r) the
# gauge cell of the vertex = (q,r) + delta_k
DELTA = {0: (0, 0), 1: (0, 0), 2: (-1, 1), 3: (-1, 0), 4: (-1, 0), 5: (0, -1)}


def symbol_det(hc: KagomeTorus, k1: float, k2: float) -> complex:
    """det Mhat(k1,k2): build the 2x2 symbol directly from the local
    offsets of the 6 corners of face 0."""
    Mh = np.zeros((2, 2), dtype=complex)
    corners = hc.hex_corners[0]
    for ri in range(2):
        for kk in range(6):
            v = corners[kk]
            d = None
            for ke, dd in hc.hex_oriented[0]:
                if hc.kedges[ke][0] == v:
                    d = dd[ri]
                    break
            dx, dy = DELTA[kk]
            Mh[ri, kk % 2] += d * np.exp(1j * (k1 * dx + k2 * dy))
    return complex(np.linalg.det(Mh))


def symbol_det_formula(k1: float, k2: float) -> complex:
    """3(a-1)(b-1)(a-b)/a^2 b with a = e^{ik1}, b = e^{ik2}."""
    a = np.exp(1j * k1)
    b = np.exp(1j * k2)
    return 3 * (a - 1) * (b - 1) * (a - b) / (a ** 2 * b)


def main() -> None:
    rows = []
    all_ok = True

    # ---------------- A. brute-force anchors ----------------
    print("A. brute-force anchors (torus) + strip/kernel cross-checks",
          flush=True)
    anchors = [((2, 2), 3), ((3, 2), 3), ((2, 3), 3), ((4, 2), 2),
               ((3, 3), 2), ((2, 4), 2)]
    for (Lq, Lr), k in anchors:
        t0 = time.time()
        hc = KagomeTorus(Lq, Lr)
        alph = list(range(k))
        om_bf = count_assignments_reduced(hc, alph)
        om_ke = kernel_count_int(hc.M_reduced, alph)
        dim = nullity_exact(hc.M_reduced)
        bound = k ** dim
        ok = (om_bf == om_ke) and (om_bf <= bound)
        all_ok &= ok
        rows.append(dict(layer="A_brute", case=f"torus{Lq}x{Lr}", k=k,
                         F=hc.F, n_vars=hc.nv, dim_formula="", dim_num=dim,
                         omega=om_bf, bound=bound,
                         s_per_hole=f"{math.log(om_bf) / hc.F:.6f}",
                         check="OK" if ok else "MISMATCH"))
        print(f"  torus{Lq}x{Lr} k={k}: brute={om_bf} kernel={om_ke} "
              f"bound={bound} {'OK' if ok else 'MISMATCH'} "
              f"({time.time() - t0:.1f}s)", flush=True)

    patches = [(2, 2), (3, 2), (2, 3), (3, 3), (3, 4), (4, 3)]
    for (W, L) in patches:
        t0 = time.time()
        om_tm, _ = strip_count(W, L, [4, 5, 6])
        om_ke = kernel_count_int(patch_matrix(W, L), [4, 5, 6])
        ok = om_tm == om_ke
        all_ok &= ok
        rows.append(dict(layer="A_patch", case=f"patch{W}x{L}", k=3,
                         F=W * L, n_vars="", dim_formula="", dim_num="",
                         omega=om_tm, bound="",
                         s_per_hole=f"{math.log(om_tm) / (W * L):.6f}",
                         check="OK" if ok else "MISMATCH"))
        print(f"  patch{W}x{L} k=3: TM={om_tm} kernel={om_ke} "
              f"{'OK' if ok else 'MISMATCH'} ({time.time() - t0:.1f}s)",
              flush=True)

    # ---------------- B. nullity table + symbol factorization ----------------
    print("B. nullity table vs formula Lq+Lr+gcd-1; symbol factorization",
          flush=True)
    sizes = [(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (8, 2), (3, 3), (4, 3),
             (5, 3), (6, 3), (4, 4), (5, 4), (6, 4), (8, 4), (5, 5), (6, 6),
             (9, 6), (10, 4), (12, 6), (7, 5), (12, 8)]
    for (Lq, Lr) in sizes:
        t0 = time.time()
        hc = KagomeTorus(Lq, Lr)
        g = math.gcd(Lq, Lr)
        formula = Lq + Lr + g - 1
        if hc.F <= 12:
            dim = nullity_exact(hc.M_reduced)
            method = "exact"
        else:
            dim = nullity_modp(hc.M_reduced)
            method = "modp"
        ok = dim == formula
        all_ok &= ok
        rows.append(dict(layer="B_nullity", case=f"torus{Lq}x{Lr}", k="",
                         F=hc.F, n_vars=hc.nv, dim_formula=formula,
                         dim_num=f"{dim}({method})", omega="", bound="",
                         s_per_hole="", check="OK" if ok else "MISMATCH"))
        print(f"  torus{Lq}x{Lr}: dim={dim} ({method}) formula={formula} "
              f"{'OK' if ok else 'MISMATCH'} ({time.time() - t0:.1f}s)",
              flush=True)

    # symbol determinant = 3(a-1)(b-1)(a-b)/a^2 b: residual over random
    # continuous-k samples
    rng = np.random.default_rng(11)
    hc = KagomeTorus(6, 5)
    max_res = 0.0
    for _ in range(200):
        k1, k2 = rng.uniform(0, 2 * np.pi, 2)
        res = abs(symbol_det(hc, k1, k2) - symbol_det_formula(k1, k2))
        max_res = max(max_res, res)
    ok = max_res < 1e-8
    all_ok &= ok
    rows.append(dict(layer="B_symbol", case="det vs factorization", k="",
                     F="", n_vars="", dim_formula="", dim_num="", omega="",
                     bound=f"maxres={max_res:.2e}",
                     s_per_hole="", check="OK" if ok else "MISMATCH"))
    print(f"  symbol factorization max residual = {max_res:.2e} "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)

    # zero grid: count = Lq+Lr+gcd-2; per-point 2x2 nullity 2 at the
    # origin, 1 elsewhere
    for (Lq, Lr) in [(6, 5), (8, 4), (6, 6)]:
        hc = KagomeTorus(Lq, Lr)
        g = math.gcd(Lq, Lr)
        zeros = ({(n1, 0) for n1 in range(Lq)}
                 | {(0, n2) for n2 in range(Lr)}
                 | {(t * Lq // g % Lq, t * Lr // g % Lr) for t in range(g)})
        count_ok, null_ok = True, True
        for (n1, n2) in sorted(zeros):
            # build the 2x2 symbol matrix directly (no determinant)
            M2 = np.zeros((2, 2), dtype=complex)
            corners = hc.hex_corners[0]
            for ri in range(2):
                for kk in range(6):
                    v = corners[kk]
                    d = None
                    for ke, dd in hc.hex_oriented[0]:
                        if hc.kedges[ke][0] == v:
                            d = dd[ri]
                            break
                    dx, dy = DELTA[kk]
                    M2[ri, kk % 2] += d * np.exp(
                        1j * (2 * np.pi * n1 / Lq * dx
                              + 2 * np.pi * n2 / Lr * dy))
            nz = 2 - np.linalg.matrix_rank(M2, tol=1e-9)
            null_ok &= (nz == (2 if (n1, n2) == (0, 0) else 1))
        count_ok = len(zeros) == Lq + Lr + g - 2
        ok = count_ok and null_ok
        all_ok &= ok
        rows.append(dict(layer="B_zeros", case=f"torus{Lq}x{Lr}", k="",
                         F=hc.F, n_vars="", dim_formula=Lq + Lr + g - 2,
                         dim_num=len(zeros), omega="", bound="",
                         s_per_hole="", check="OK" if ok else "MISMATCH"))
        print(f"  torus{Lq}x{Lr}: grid zeros={len(zeros)} "
              f"(expect {Lq + Lr + g - 2}), per-zero nullities "
              f"{'OK' if ok else 'MISMATCH'}", flush=True)

    # ---------------- C. exact Omega series ----------------
    print("C. exact Omega via kernel enumeration (torus)", flush=True)
    series3 = [(2, 2), (3, 2), (2, 3), (4, 2), (2, 4), (4, 3), (3, 3),
               (5, 2), (3, 4), (5, 3), (6, 2), (2, 6), (4, 4), (6, 3),
               (5, 4), (6, 4)]
    for (Lq, Lr) in series3:
        t0 = time.time()
        hc = KagomeTorus(Lq, Lr)
        dim = Lq + Lr + math.gcd(Lq, Lr) - 1
        om = kernel_count_int(hc.M_reduced, [4, 5, 6])
        bound = 3 ** dim
        ok = om <= bound
        all_ok &= ok
        rows.append(dict(layer="C_omega_k3", case=f"torus{Lq}x{Lr}", k=3,
                         F=hc.F, n_vars=hc.nv, dim_formula=dim, dim_num="",
                         omega=om, bound=bound,
                         s_per_hole=f"{math.log(om) / hc.F:.6f}",
                         s_bound=f"{math.log(bound) / hc.F:.6f}",
                         check="OK" if ok else "MISMATCH"))
        print(f"  torus{Lq}x{Lr} k=3: Omega={om} bound={bound} "
              f"s={math.log(om) / hc.F:.4f} (bound s="
              f"{math.log(bound) / hc.F:.4f}) ({time.time() - t0:.1f}s)",
              flush=True)
    series2 = [(6, 6), (8, 4), (8, 6), (10, 5), (12, 4)]
    for (Lq, Lr) in series2:
        t0 = time.time()
        hc = KagomeTorus(Lq, Lr)
        dim = Lq + Lr + math.gcd(Lq, Lr) - 1
        om = kernel_count_int(hc.M_reduced, [4, 5])
        bound = 2 ** dim
        ok = om <= bound
        all_ok &= ok
        rows.append(dict(layer="C_omega_k2", case=f"torus{Lq}x{Lr}", k=2,
                         F=hc.F, n_vars=hc.nv, dim_formula=dim, dim_num="",
                         omega=om, bound=bound,
                         s_per_hole=f"{math.log(om) / hc.F:.6f}",
                         s_bound=f"{math.log(bound) / hc.F:.6f}",
                         check="OK" if ok else "MISMATCH"))
        print(f"  torus{Lq}x{Lr} k=2: Omega={om} bound={bound} "
              f"s={math.log(om) / hc.F:.4f} ({time.time() - t0:.1f}s)",
              flush=True)

    # ---------------- D. strip series (open-boundary TM) ----------------
    print("D. open-boundary strip series", flush=True)
    for W, Ls in [(2, range(4, 25, 4)), (3, range(4, 25, 4)),
                  (4, range(4, 25, 4))]:
        for L in Ls:
            t0 = time.time()
            om, peak = strip_count(W, L, [4, 5, 6])
            rows.append(dict(layer="D_strip_k3", case=f"patch{W}x{L}", k=3,
                             F=W * L, n_vars="", dim_formula="", dim_num="",
                             omega=om, bound=f"peak_states={peak}",
                             s_per_hole=f"{math.log(om) / (W * L):.6f}",
                             s_bound="", check=""))
            print(f"  patch{W}x{L} k=3: s={math.log(om) / (W * L):.5f} "
                  f"peak={peak} ({time.time() - t0:.1f}s)", flush=True)
    for W, Ls in [(2, range(4, 25, 4)), (3, range(4, 25, 4)),
                  (4, range(4, 25, 4)), (5, (4, 8, 12, 16, 20, 24))]:
        for L in Ls:
            t0 = time.time()
            om, peak = strip_count(W, L, [4, 5])
            rows.append(dict(layer="D_strip_k2", case=f"patch{W}x{L}", k=2,
                             F=W * L, n_vars="", dim_formula="", dim_num="",
                             omega=om, bound=f"peak_states={peak}",
                             s_per_hole=f"{math.log(om) / (W * L):.6f}",
                             s_bound="", check=""))
            print(f"  patch{W}x{L} k=2: s={math.log(om) / (W * L):.5f} "
                  f"peak={peak} ({time.time() - t0:.1f}s)", flush=True)

    out = ROOT / "KAG_entropy_density_verification.csv"
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
