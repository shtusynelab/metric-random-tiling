# -*- coding: utf-8 -*-
"""
TSQ_Lambdalattice_structure_SNF_analysis.py - Route 3: Z-structure analysis
of the solution lattice of the 4.8.8 Lambda library via the Smith normal form.

Goal: growth order in W of the Smith normal form, covolume and basis coupling
structure of the solution lattice ker_Z(M) subset Z^{2E}, to test whether the
rank extensivity (dim_Lambda = 4W^2+4) materializes as counting extensivity.

Analytic preparation (hand derivation, numerically double-checked):
  square face (edge directions 3,5,7,1; b components = (-1,-1,1,1),
  d components = (1,-1,-1,1)):
    X1: sum(B n) = 0  =>  n(NE)+n(SW) = n(NW)+n(SE)    (b: 3,7 negative; 5,1 positive)
    X2: sum(2B m) = 0  =>  same pattern in m
    Y1: sum(D n) = 0;  Y2: same pattern in m
    X1-Y1 => n3 = n7 and n1 = n5; X2-Y2 => m3 = m7 and m1 = m5
    => square: m and n each satisfy two opposite-side equalities
       (dir3 = dir7, dir1 = dir5).
  octagon face: the four rows give
    m0-m4 = m2-m6   (equal m-differences on the axis-opposite pairs)
    n0-n4 = n2-n6   (equal n-differences on the axis-opposite pairs)
    n1-n3-n5+n7 = -2(m0-m4)   (parity coupling of diagonal pairs to the
                               axis m-difference)
    m1-m3-m5+m7 = -(n0-n4)
    => not purely opposite-side equality: difference-type constraints plus
       parity coupling (the candidate source of the library's new entropy).

Output: TSQ_Lambda_SNF_series.csv (invariant-factor summary / covolume /
support) beside this script.
Counting / linear adjudication uses int/Fraction throughout; the covolume
logarithm uses high-precision mpmath.
"""
import sys, math, time, csv
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mpmath as mp
from tsq_lambda_lib import TSQTorus, lambda_system, rank_rows

OUT = HERE / "TSQ_Lambda_SNF_series.csv"


# ---------------------------------------------------------------- SNF (with transforms)

def snf_transforms(A):
    """Smith normal form of an integer matrix; returns (D, V) with D the SNF
    of A@V (V unimodular). Only column transforms are needed to obtain the
    kernel basis (kernel columns of V); the nonzero diagonal of D gives the
    invariant factors. Algorithm: textbook Euclidean reduction + smallest
    element pivoting."""
    m, n = len(A), len(A[0])
    B = [row[:] for row in A]
    V = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    t = 0
    for t in range(min(m, n)):
        while True:
            # smallest nonzero element -> (t,t)
            piv = None
            for i in range(t, m):
                for j in range(t, n):
                    if B[i][j] and (piv is None or abs(B[i][j]) < abs(piv[0])):
                        piv = (B[i][j], i, j)
            if piv is None:
                return B, V   # rank exhausted: remaining rows all zero, zero diagonal tail
            _, pi, pj = piv
            if pi != t:
                B[t], B[pi] = B[pi], B[t]
            if pj != t:
                for r in range(m):
                    B[r][t], B[r][pj] = B[r][pj], B[r][t]
                for r in range(n):
                    V[r][t], V[r][pj] = V[r][pj], V[r][t]
            p = B[t][t]
            # column elimination: columns t+1..n-1 of row t
            done = False
            for j in range(t + 1, n):
                if B[t][j]:
                    q = B[t][j] // p
                    for r in range(m):
                        B[r][j] -= q * B[r][t]
                    for r in range(n):
                        V[r][j] -= q * V[r][t]
                    if abs(B[t][j]) < abs(p) and B[t][j] != 0:
                        done = True   # new smallest element appeared, restart
            # row elimination: rows t+1..m-1 of column t
            for i in range(t + 1, m):
                if B[i][t]:
                    q = B[i][t] // p
                    B[i] = [x - q * y for x, y in zip(B[i], B[t])]
                    if abs(B[i][t]) < abs(p) and B[i][t] != 0:
                        done = True
            if done:
                continue
            # divisibility check: on violation, add the violating row to row t
            # and re-eliminate
            viol = False
            for i in range(t + 1, m):
                for j in range(t + 1, n):
                    if B[i][j] % p:
                        B[t] = [x + y for x, y in zip(B[t], B[i])]
                        viol = True
                        break
                if viol:
                    break
            if not viol:
                break
    return B, V


def analyze(W):
    t = TSQTorus(W, W)
    E = t.ne
    rows = lambda_system(t)
    nvar = 2 * E
    M = [[row.get(c, 0) for c in range(nvar)] for row in rows]
    t0 = time.time()
    D, V = snf_transforms(M)
    dt = time.time() - t0
    diag = [abs(D[i][i]) for i in range(min(len(D), nvar))]
    diag = [d for d in diag if d]
    r = len(diag)
    dimL = nvar - r
    assert r == 8 * W * W - 4, f"unexpected SNF rank {r}"
    # kernel basis = columns r..nvar-1 of V
    kerB = [[V[i][j] for j in range(r, nvar)] for i in range(nvar)]  # nvar x dimL
    # assert M@kerB == 0
    for row in M:
        for c in range(dimL):
            assert sum(row[i] * kerB[i][c] for i in range(nvar)) == 0, \
                "kernel basis violates constraints"
    # support statistics
    supp = [sum(1 for i in range(nvar) if kerB[i][c]) for c in range(dimL)]
    linf = [max(abs(kerB[i][c]) for i in range(nvar)) for c in range(dimL)]
    # covolume (Euclidean): ln sqrt(det Gram), high-precision hand-written
    # Cholesky in mpmath
    mp.mp.dps = 50
    dim = dimL
    Gm = mp.matrix(dim, dim)
    for a in range(dim):
        for b in range(dim):
            Gm[a, b] = mp.mpf(sum(kerB[i][a] * kerB[i][b] for i in range(nvar)))
    L = mp.matrix(dim, dim)
    for i in range(dim):
        for j in range(i):
            s = Gm[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            L[i, j] = s / L[j, j]
        s = Gm[i, i]
        for k in range(i):
            s -= L[i, k] ** 2
        L[i, i] = mp.sqrt(s)
    logdet = sum(2 * mp.log(L[i, i]) for i in range(dim))
    logcovol = mp.mpf("0.5") * logdet
    max_d = max(diag)
    n_d1 = sum(1 for d in diag if d == 1)
    print(f"W={W}: E={E} nvar={nvar} rows={len(M)} rank={r} dim_L={dimL} "
          f"t={dt:.1f}s", flush=True)
    print(f"  invariant factors: max={max_d} #d>1={r - n_d1} "
          f"nontrivial={sorted(set(d for d in diag if d > 1))[:20]}", flush=True)
    print(f"  covolume: ln covol = {mp.nstr(logcovol, 10)}, "
          f"ln covol / W^2 = {mp.nstr(logcovol / (W * W), 10)}", flush=True)
    print(f"  basis support: mean={sum(supp) / len(supp):.1f} max={max(supp)} "
          f"linf_max={max(linf)}", flush=True)
    return dict(W=W, E=E, nvar=nvar, rank=r, dimL=dimL, t=round(dt, 1),
                max_inv=max_d, n_nontorsion=r - n_d1,
                logcovol=float(logcovol), logcovol_W2=float(logcovol / (W * W)),
                supp_mean=round(sum(supp) / len(supp), 2), supp_max=max(supp),
                linf_max=max(linf))


if __name__ == "__main__":
    res = []
    for W in (2, 3, 4, 5):
        try:
            res.append(analyze(W))
        except Exception as ex:
            print(f"W={W} failed: {ex}", flush=True)
            break
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print(f"CSV -> {OUT}")
