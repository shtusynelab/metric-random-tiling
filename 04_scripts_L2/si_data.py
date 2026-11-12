# -*- coding: utf-8 -*-
"""SI figure data: H_comp / H_cond size series (main agent, 2026-10-14).

For every (alphabet, Lx, Ly): exact enumeration -> ln Omega, H_comp, H_cond
(the splitting identity is asserted at every point).

Edge-count convention (erratum fix, v3.0c): the honeycomb torus carries
E = 3F edges for F hexagonal faces (three edges per fundamental cell, SI
Sect. S1.2); E is taken from the construction's actual edge set (len of the
edge-key set), never estimated from F.

Output: <DATA>/si_entropy_split.csv (SI Fig. S11 data source).
"""
import csv
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(PC.HC))
from edge_alphabet_affine_check import build_honeycomb_torus  # noqa: E402

OUT = PC.DATA / "si_entropy_split.csv"
ANCHORS = {((3, 3), (1, 2)): 518, ((3, 3), (1, 2, 3)): 22767,
           ((3, 3), (1, 2, 4)): 19707, ((4, 3), (1, 2, 3)): 30321}


def enumerate_valid(Lx, Ly, alpha, budget_s=1800.0):
    edge_set, faces = build_honeycomb_torus(Lx, Ly)
    E = len(edge_set)
    t0 = time.time()
    nodes = [0]
    order, seen = [], set()
    for f in faces:
        for e in f:
            if e not in seen:
                seen.add(e)
                order.append(e)
    pos = {e: k for k, e in enumerate(order)}
    faces_at = {}
    for last, f in sorted((max(pos[e] for e in f), f) for f in faces):
        faces_at.setdefault(last, []).append(f)
    x = [None] * E
    valid = []

    class Budget(Exception):
        pass

    def check_face(f):
        L = [x[pos[e]] for e in f]
        return (L[1] + L[2] == L[4] + L[5] and
                2 * L[0] + L[1] + L[5] == 2 * L[3] + L[2] + L[4])

    def rec(k):
        if k == E:
            valid.append(tuple(x))
            return
        nodes[0] += 1
        if nodes[0] % 5_000_000 == 0 and time.time() - t0 > budget_s:
            raise Budget()
        for v in alpha:
            x[k] = v
            if all(check_face(f) for f in faces_at.get(k, ())):
                rec(k + 1)
        x[k] = None

    try:
        rec(0)
    except Budget:
        return None
    return E, faces, valid


def metrics(Lx, Ly, alpha):
    r = enumerate_valid(Lx, Ly, alpha)
    if r is None:
        return None
    E, faces, valid = r
    F = len(faces)
    assert E == 3 * F, f"honeycomb torus must carry E = 3F, got E = {E}, F = {F}"
    omega = len(valid)
    key = ((Lx, Ly), tuple(alpha))
    if key in ANCHORS:
        assert omega == ANCHORS[key], f"{key}: {omega}"
    a = sorted(alpha)
    comp_of = {v: i for i, v in enumerate(a)}
    k = len(a)
    cc = Counter()
    for cfg in valid:
        cnt = [0] * k
        for v in cfg:
            cnt[comp_of[v]] += 1
        cc[tuple(cnt)] += 1
    Q = {c: w / omega for c, w in cc.items()}
    H_comp = -sum(q * math.log(q) for q in Q.values())
    H_cond = sum(Q[c] * math.log(cc[c]) for c in cc)
    assert abs((H_comp + H_cond) - math.log(omega)) < 1e-6
    return {"Lx": Lx, "Ly": Ly, "F": F, "E": E, "omega": omega,
            "ln_Omega": math.log(omega),
            "H_comp": H_comp, "H_cond": H_cond,
            "H_comp_per_F": H_comp / F, "H_cond_per_F": H_cond / F,
            "n_comp": len(cc)}


def main():
    # row set matches the delivered table (binary block incl. the (3,4)
    # transposed companion row and the 6x3 rows, then ternary blocks)
    runs = [
        (2, 2, (1, 2)), (2, 3, (1, 2)),
        (3, 3, (1, 2)), (3, 4, (1, 2)), (4, 3, (1, 2)), (5, 3, (1, 2)),
        (4, 4, (1, 2)), (6, 3, (1, 2)), (5, 4, (1, 2)),
        (2, 2, (1, 2, 3)), (2, 3, (1, 2, 3)),
        (3, 3, (1, 2, 3)), (4, 3, (1, 2, 3)), (5, 3, (1, 2, 3)),
        (2, 2, (1, 2, 4)), (2, 3, (1, 2, 4)),
        (3, 3, (1, 2, 4)), (4, 3, (1, 2, 4)), (5, 3, (1, 2, 4)), (6, 3, (1, 2, 4)),
    ]
    rows = []
    for Lx, Ly, alpha in runs:
        t0 = time.time()
        r = metrics(Lx, Ly, alpha)
        tag = "".join(str(v) for v in alpha)
        if r is None:
            print(f"[{tag} {Lx}x{Ly}] TIMEOUT ({time.time()-t0:.0f}s)",
                  flush=True)
            continue
        r["alpha"] = tag
        r["secs"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"[{tag} {Lx}x{Ly}] F={r['F']:2d} E={r['E']:2d} "
              f"lnOmega={r['ln_Omega']:8.4f} "
              f"Hcomp/F={r['H_comp_per_F']:.4f} Hcond/F={r['H_cond_per_F']:.4f}"
              f"  ({r['secs']}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "alpha", "Lx", "Ly", "F", "E", "omega", "ln_Omega",
            "H_comp", "H_cond", "H_comp_per_F", "H_cond_per_F",
            "n_comp", "secs"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
