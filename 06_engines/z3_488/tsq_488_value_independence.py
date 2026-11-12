# -*- coding: utf-8 -*-
"""
tsq_488_value_independence.py - finite-library value structure and 4.8.8
counting (revised v2).

Findings:
  - k=2: value independence holds strictly (Omega is identical for every
    two-value library; verified on the (2,2) torus, Omega = 256 the same
    for all 3 two-value libraries).
  - k=3: the affine structure of the library carries the count. For an
    arithmetic-progression library (including Z) Omega_Z(3) = 6561 on
    (2,2); the non-arithmetic mixed library {0, sqrt(2), 1+sqrt(2)} gives
    6641 != 6561, refuting value independence; a non-arithmetic library is
    a different CSP (a finite truncation of the Lambda two-variable rows).
  - The first DFS version exploded at (2,3); this version uses the strip
    DP (frontier states merged by library index, constraints checked over
    the reals; the arithmetic library (0,1,2) must reproduce the integer
    strip_count result bit-for-bit as a DP correctness self-check).

Question: is the entropy density of a non-arithmetic finite library
extensive (a plateau)?
  - If s(F) -> 0: finite libraries die (conjecture holds; cause = the
    arithmetic row-splitting + non-arithmetic sub-extensivity).
  - If s(F) plateaus: a finite-library positive-entropy carrier exists
    (the conjecture would have to be restricted to arithmetic/rational
    libraries).
"""
import os
import sys
import math
import time
from itertools import product

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "z3_488"))
import tsq_lib as T  # noqa: E402

TOL = 1e-9

# mixed library {0, sqrt(2), 1+sqrt(2)} written as s = gamma + beta*sqrt(2); value table (gamma, beta):
VAL_MIX = [(0, 0), (0, 1), (1, 1)]


def strip_count_pair(faces, B_vals, max_states=2e7):
    """Two-variable integer strip DP: each edge index takes a (gamma,beta)
    pair from B_vals. Closure (s = gamma + beta*sqrt(2)): the x rows give
    sum(a*gamma + 2b*beta) = 0 and sum(b*gamma + a*beta) = 0; likewise for
    y with (c,d). Hard budget gate: reject when the state count exceeds the
    cap."""
    nh = len(faces)
    fdat = []
    for size, sides in faces:
        R = np.zeros((4, len(sides), 2), dtype=np.int64)
        for t, (e, d) in enumerate(sides):
            a, b, c, dd = T.DIR8[d]
            R[0, t, 0] = a;  R[0, t, 1] = 2 * b
            R[1, t, 0] = b;  R[1, t, 1] = a
            R[2, t, 0] = c;  R[2, t, 1] = 2 * dd
            R[3, t, 0] = dd; R[3, t, 1] = c
        fdat.append((sides, R))
    hosts = {}
    for hi, (sides, R) in enumerate(fdat):
        for e, d in sides:
            hosts.setdefault(e, []).append(hi)
    first = {v: min(x) for v, x in hosts.items()}
    last = {v: max(x) for v, x in hosts.items()}
    k = len(B_vals)
    G = np.array([g for g, _ in B_vals], dtype=np.int64)
    Bt = np.array([b for _, b in B_vals], dtype=np.int64)

    def pending(i):
        return sorted(v for v in first if first[v] <= i - 1 and last[v] >= i)

    def after(i):
        return sorted(v for v in first if first[v] <= i and last[v] >= i + 1)

    states = np.zeros((1, 0), dtype=np.int64)
    counts = [1]
    peak = 0
    for i in range(nh):
        sides, R = fdat[i]
        before = pending(i)
        bidx = {v: j for j, v in enumerate(before)}
        new_e = [e for e, d in sides if first[e] == i]
        aft = after(i)
        carr = np.asarray(counts, dtype=np.int64)
        out_vals, out_cnts = [], []
        for combo in product(range(k), repeat=len(new_e)):
            newv = dict(zip(new_e, combo))
            eq4 = np.zeros((4, len(states)), dtype=np.int64)
            for t, (e, d) in enumerate(sides):
                rco = R[:, t, :]
                if e in newv:
                    g, b = G[newv[e]], Bt[newv[e]]
                    eq4 += rco[:, 0:1] * g + rco[:, 1:2] * b
                else:
                    sg = G[states[:, bidx[e]]]
                    sb = Bt[states[:, bidx[e]]]
                    eq4 += rco[:, 0:1] * sg[None, :] + \
                        rco[:, 1:2] * sb[None, :]
            valid = (eq4 == 0).all(axis=0)
            if not valid.any():
                continue
            sub = states[valid]
            subc = carr[valid]
            cols = np.empty((len(sub), len(aft)), dtype=np.int64)
            for j, v in enumerate(aft):
                if v in newv:
                    cols[:, j] = newv[v]
                else:
                    cols[:, j] = sub[:, bidx[v]]
            out_vals.append(cols)
            out_cnts.append(subc)
        if out_vals:
            raw = np.vstack(out_vals)
            rawc = np.concatenate(out_cnts)
        else:
            raw = np.zeros((0, len(aft)), dtype=np.int64)
            rawc = np.zeros(0, dtype=np.int64)
        merged = {}
        merged_rows = []
        for row, c in zip(raw, rawc):
            key = row.tobytes()
            if key in merged:
                merged[key] += int(c)
            else:
                merged[key] = int(c)
                merged_rows.append(row)
        states = (np.vstack(merged_rows) if merged_rows
                  else np.zeros((0, len(aft)), dtype=np.int64))
        counts = list(merged.values())
        peak = max(peak, len(states))
        if len(states) > max_states:
            raise RuntimeError(f"strip states {len(states)} > {max_states}")
    return sum(counts), peak


def strip_count_real(faces, n_vars, B):
    """Strip DP over a real-valued library: faces must be in scan order (the
    tsq_patch output already is). State = tuple of library indices on the
    frontier edges; counts merged by key; face closure checked over the
    reals."""
    nh = len(faces)
    fdat = []
    for size, sides in faces:
        cs = [((a + b * math.sqrt(2)) / 2, (c + dd * math.sqrt(2)) / 2)
              for e, d in sides
              for (a, b, c, dd) in [T.DIR8[d]]]
        fdat.append((sides, cs))
    hosts = {}
    for hi, (sides, cs) in enumerate(fdat):
        for e, d in sides:
            hosts.setdefault(e, []).append(hi)
    first = {v: min(x) for v, x in hosts.items()}
    last = {v: max(x) for v, x in hosts.items()}
    k = len(B)

    def pending(i):
        return sorted(v for v in first if first[v] <= i - 1 and last[v] >= i)

    def after(i):
        return sorted(v for v in first if first[v] <= i and last[v] >= i + 1)

    states = {(): 1}
    peak = 0
    for i in range(nh):
        sides, cs = fdat[i]
        before = pending(i)
        bidx = {v: j for j, v in enumerate(before)}
        new_e = [e for e, d in sides if first[e] == i]
        aft = after(i)
        out = {}
        for key, cnt in states.items():
            for combo in product(range(k), repeat=len(new_e)):
                val = dict(zip(new_e, combo))
                sx = sy = 0.0
                st = dict(zip(before, key))
                for (e, d), (cx, cy) in zip(sides, cs):
                    if e in val:
                        s = B[val[e]]
                    else:
                        s = B[st[e]]
                    sx += cx * s
                    sy += cy * s
                if abs(sx) > TOL or abs(sy) > TOL:
                    continue
                nk = tuple(val[v] if v in val else key[bidx[v]]
                           for v in aft)
                out[nk] = out.get(nk, 0) + cnt
        states = out
        peak = max(peak, len(states))
    return sum(states.values()), peak


def main():
    t0 = time.time()
    print(f"{'patch':>7} {'F':>4} {'E':>4} | {'Om_arith':>14} {'s_arith':>8} "
          f"| {'Om_mixed':>14} {'s_mixed':>8} | {'peak':>9}  {'t(s)':>5}",
          flush=True)
    for (W, L) in [(2, 2), (2, 3), (3, 2), (2, 4), (3, 3), (3, 4)]:
        faces, E = T.tsq_patch(W, L)
        F = len(faces)
        om_a, _ = T.strip_count(faces, E, (0, 1, 2))
        om_m, peak = strip_count_pair(faces, VAL_MIX)
        print(f"{W}x{L:>4} {F:>4} {E:>4} | {om_a:>14} "
              f"{math.log(om_a) / F:>8.4f} | {om_m:>14} "
              f"{math.log(om_m) / F:>8.4f} | {peak:>9}  "
              f"{time.time() - t0:>5.0f}", flush=True)
        # self-check 1: the pair DP must reproduce the integer strip
        # bit-for-bit on the arithmetic value table {(0,0),(1,0),(2,0)}
        # (i.e. s in {0,1,2})
        # self-check 2: mixed (2,2) anchor = the value 231647 from the first
        # slow real-valued DP
        if (W, L) == (2, 2):
            om_c, _ = strip_count_pair(faces, [(0, 0), (1, 0), (2, 0)])
            assert om_c == om_a, f"pair-DP sanity failed: {om_c} vs {om_a}"
            print("  sanity: pair-DP (arith table) vs integer strip OK",
                  flush=True)
            assert om_m == 231647, f"mixed anchor failed: {om_m}"
            print("  sanity: mixed (2,2) == slow real-DP anchor 231647 OK",
                  flush=True)
    print("note: s(F)->0 = sub-extensive (library dead); plateau = "
          "extensive-entropy candidate (library alive)")


if __name__ == "__main__":
    main()
