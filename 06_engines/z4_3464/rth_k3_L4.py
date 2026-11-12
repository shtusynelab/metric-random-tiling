# -*- coding: utf-8 -*-
"""
rth_k3_L4.py - exact counting supplement for the 3.4.6.4 k=3 case
(arithmetic-progression alphabet {0,1,2}).

Purpose: test that alpha ~ 3.3 (bits per value) is independent of k
(a signature of the perimeter law / propagation type).
Known anchors: k=2 Omega = 128/1036/8800/86956 (L=2..5); k=3 AP Omega(2)=2187,
Omega(3)=68301 (log_3(Omega) = 10.13 ~ log_2(Omega, k=2) = 10.02).
This script adds Omega(L=4, k=3): 3^19 ~ 1.16e9 integer-core enumeration.

Method: same mathematics as verify_dsh_audit.count_exact
(x_pivot = -T*f; D=1 integer basis; enumerate f in {alpha0..alpha0+k-1}^dim;
check that pivot coordinates lie inside the box), with two harmless
optimizations only:
  1. skip the mod-D test when D=1 (always true); fall back to the original
     path when D != 1;
  2. use int32 for N/F (range |N| <= sum of |rows| <= 60, Mvals <= 60*2,
     far inside int32).
Correctness is guaranteed by the anchor chain: L=2 k=2 -> 128, L=2 k=3 -> 2187,
L=3 k=2 -> 1036, L=3 k=3 -> 68301, L=4 k=2 -> 8800 must be reproduced exactly
before L=4 k=3 is counted.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verify_dsh_audit as V


def count_exact_fast(T, free_cols, E, k, alpha0=0, chunk=1 << 20):
    """count_exact fast path for D=1 (same mathematics; anchor-chain self-check)."""
    P = len(T)
    if P == 0:
        return k ** len(free_cols)
    D = 1
    for row in T:
        for v in row:
            D = D * v.denominator // math.gcd(D, v.denominator)
    assert D == 1, f"D={D} != 1; fall back to the original count_exact path"
    N = np.array([[int(v) for v in row] for row in T], dtype=np.int32)
    amax = int(np.abs(N).sum(axis=1).max())
    assert amax * (alpha0 + k) < 2 ** 30, "int32 overflow risk"
    dim = len(free_cols)
    total = k ** dim
    cnt = 0
    t0 = time.time()
    for start in range(0, total, chunk):
        n = min(chunk, total - start)
        idx = np.arange(start, start + n, dtype=np.int64)
        cols = []
        tmp = idx.copy()
        for _ in range(dim):
            cols.append((tmp % k + alpha0).astype(np.int32))
            tmp //= k
        F = np.stack(cols, axis=1)              # n x dim, int32
        Mv = F @ N.T                            # n x P, int32 (x_pivot = -Mv)
        ok = (Mv <= -alpha0).all(axis=1) & (Mv >= -(alpha0 + k - 1)).all(axis=1)
        cnt += int(ok.sum())
        done = min(start + chunk, total)
        if done % max(1, (1 << 24)) < chunk or done == total:
            el = time.time() - t0
            print(f"    {done}/{total} ({100*done/total:.1f}%) cnt={cnt} "
                  f"({el:.0f}s)", flush=True)
    return cnt


def main():
    ap = argparse.ArgumentParser(
        description="Exact counting for the 3.4.6.4 k=3 case at L=4.")
    ap.add_argument("--log-dir", default=None,
                    help="Directory for rth_k3_L4.log "
                         "(default: this script's directory)")
    args = ap.parse_args()
    log_dir = Path(args.log_dir) if args.log_dir else HERE

    print(f"start {time.strftime('%F %T')}", flush=True)
    anchors = [  # (L, k, expect)
        (2, 2, 128), (2, 3, 2187),
        (3, 2, 1036), (3, 3, 68301),
        (4, 2, 8800),
    ]
    param = {}
    for L in (2, 3, 4):
        t = V.build_3464(L)
        T, piv, free, dim = V.rref_and_param(t.xrows, t.yrows, t.E)
        param[L] = (T, free, dim)
        print(f"L={L}: E={t.E} dim={dim}", flush=True)

    ok_all = True
    for L, k, expect in anchors:
        T, free, dim = param[L]
        E = {2: 48, 3: 108, 4: 192}[L]
        t0 = time.time()
        got = count_exact_fast(T, free, E, k)
        ok = (got == expect)
        ok_all &= ok
        print(f"ANCHOR L={L} k={k}: got={got} expect={expect} "
              f"{'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)
    if not ok_all:
        print("Anchor check failed; aborting; L=4 k=3 not counted", flush=True)
        return 1

    T, free, dim = param[4]
    t0 = time.time()
    omega = count_exact_fast(T, free, 192, 3)
    log3 = math.log(omega, 3)
    log2_ref = math.log(8800, 2)
    print(f"RESULT Omega(L=4,k=3,alpha0=0) = {omega}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  log3 = {log3:.3f}   vs log2(Omega k=2, L=4) = {log2_ref:.3f}", flush=True)
    print(f"  s(per face) = {math.log(omega)/(6*16):.4f}", flush=True)
    print(f"  Perimeter-law sequence log_k Omega: L=2 {math.log(2187,3):.2f}/{math.log(128,2):.2f}, "
          f"L=3 {math.log(68301,3):.2f}/{math.log(1036,2):.2f}, "
          f"L=4 {log3:.2f}/{log2_ref:.2f}", flush=True)

    t0 = time.time()
    omega1 = count_exact_fast(T, free, 192, 3, alpha0=1)
    print(f"RESULT Omega(L=4,k=3,alpha0=1) = {omega1} "
          f"({'translation-invariance OK' if omega1 == omega else 'translation-invariance FAIL!'}) "
          f"({time.time()-t0:.0f}s)", flush=True)
    with open(log_dir / "rth_k3_L4.log", "a") as f:
        f.write(f"{time.strftime('%F %T')} L4k3={omega} L4k3_a1={omega1} "
                f"log3={log3:.4f}\n")
    print(f"done {time.strftime('%F %T')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
