# -*- coding: utf-8 -*-
"""
TSQ_Lambda_Knuth_estimator.py - Route 2: Knuth (Rosenbluth growth) estimator
for the 4.8.8 Lambda library (corroboration only; not part of the main
adjudication).

Port of the knuth_strip_one routine from the z3_hexagonal P0 Knuth
reliability-verification script:
  same enumeration tree (face order = torus monotone order of
  tsq_lambda_lib.mono_order; branching = library-value combinations of the
  face's newly added edges; survival = the face-closure 4 rows pass; the
  second-level lookahead la2 = children still have a feasible continuation),
  single-path weight = product of survivor counts, E[weight] = Omega unbiased.

Purpose: W=2..4 cross-check against exact DP (+/- s.d.); W=6..8 trend scan
as corroboration.
The dead-branch rate / branch product as functions of W form an independent
diagnostic for the sign of s_Lambda.
The linear-extension adjudication (Fraction) is not enabled: under this
growth order "immediate closure" is equivalent to DP pruning; the lookahead
is la2. The Fraction-based check is kept only as a debugging switch.

Usage:
  python TSQ_Lambda_Knuth_estimator.py --W 4 --n 200 --seeds 1 2 3 [--lib exp|int]
Output CSV rows are appended to TSQ_Lambda_Knuth.csv beside this script.
"""
import sys, math, random, argparse, csv, time, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tsq_lambda_lib import TSQTorus, face_rows, mono_order, M_EXP, M_INT

OUT = HERE / "TSQ_Lambda_Knuth.csv"


class Tree:
    """Precompiled enumeration tree: face order, per-face rows (sparsified
    by sides position), first/last appearance."""

    def __init__(self, W):
        t = TSQTorus(W, W)
        E = t.ne
        self.E = E
        self.faces = [t.faces[i] for i in mono_order(W)]
        self.rows = [face_rows(sides, E) for size, sides in self.faces]
        # New edge = edge appearing for the first time on this face
        # (first[e] == fi); frontier edges already valued are read from val
        first, _last = {}, {}
        for fi, (size, sides) in enumerate(self.faces):
            for e, d in sides:
                if e not in first:
                    first[e] = fi
                _last[e] = fi
        self.new_of = []
        for fi, (size, sides) in enumerate(self.faces):
            self.new_of.append([e for e, d in sides if first[e] == fi])
        # Sparse form of each face row: (sides position, m coefficient, n
        # coefficient)
        self.spec = []
        for fi, (size, sides) in enumerate(self.faces):
            pos = {e: i for i, (e, d) in enumerate(sides)}
            sp = []
            for row in self.rows[fi]:
                s = []
                for e, i in pos.items():
                    c1, c2 = row.get(e, 0), row.get(E + e, 0)
                    if c1 or c2:
                        s.append((i, c1, c2))
                sp.append(s)
            self.spec.append(sp)


def knuth_one(tree, M, rng, la2=True):
    """Single Rosenbluth path; returns the weight (dead branches 0).
    M = list of library-value tuples."""
    from itertools import product as _product
    E = tree.E
    k = len(M)
    val = {}
    mult = 1
    nF = len(tree.faces)
    for fi in range(nF):
        sides = tree.faces[fi][1]
        new_e = tree.new_of[fi]
        pos = {e: i for i, (e, d) in enumerate(sides)}
        side_val = [None] * len(sides)
        for e, i in pos.items():
            if e in val:
                side_val[i] = val[e]
        new_pos = [pos[e] for e in new_e]

        def feasible(fi, side_val):
            for sp in tree.spec[fi]:
                s = 0
                for i, c1, c2 in sp:
                    m, n = side_val[i]
                    s += c1 * m + c2 * n
                if s != 0:
                    return False
            return True

        survivors = []
        for combo in _product(range(k), repeat=len(new_e)):
            ok = True
            for ci, p in enumerate(new_pos):
                side_val[p] = M[combo[ci]]
            if feasible(fi, side_val):
                survivors.append(combo)
        if not survivors:
            return 0
        if la2 and fi + 1 < nF:
            sides2 = tree.faces[fi + 1][1]
            pos2 = {e: i for i, (e, d) in enumerate(sides2)}
            new_e2 = tree.new_of[fi + 1]
            new_pos2 = [pos2[e] for e in new_e2]
            viable = []
            for combo in survivors:
                for ci, p in enumerate(new_pos):
                    side_val[p] = M[combo[ci]]
                # Next-face values: trial values from this face (side_val)
                # first, else already-committed val; remaining Nones must be
                # new edges of the next face, assigned by combo2
                sv2 = []
                for e, d in sides2:
                    if e in pos:
                        sv2.append(side_val[pos[e]] if side_val[pos[e]]
                                   is not None else val[e])
                    else:
                        sv2.append(val.get(e))
                alive = False
                for combo2 in _product(range(k), repeat=len(new_e2)):
                    for ci, p in enumerate(new_pos2):
                        sv2[p] = M[combo2[ci]]
                    ok2 = True
                    for sp in tree.spec[fi + 1]:
                        s = 0
                        for i, c1, c2 in sp:
                            m, n = sv2[i]
                            s += c1 * m + c2 * n
                        if s != 0:
                            ok2 = False
                            break
                    if ok2:
                        alive = True
                        break
                if alive:
                    viable.append(combo)
            if not viable:
                return 0
            survivors = viable
        mult *= len(survivors)
        pick = survivors[rng.randrange(len(survivors))]
        for ci, p in enumerate(new_pos):
            side_val[p] = M[pick[ci]]
        for e, i in pos.items():
            if side_val[i] is not None and e in new_e:
                val[e] = side_val[i]
    return mult


def run_W(W, M, n_samples, seeds, la2=True):
    tree = Tree(W)
    rows_out = []
    for sd in seeds:
        rng = random.Random(sd)
        ws = []
        t0 = time.time()
        for _ in range(n_samples):
            ws.append(knuth_one(tree, M, rng, la2=la2))
        dt = time.time() - t0
        alive = sum(1 for x in ws if x > 0)
        mean = math.fsum(ws) / len(ws)
        var = math.fsum((x - mean) ** 2 for x in ws) / (len(ws) - 1) if len(ws) > 1 else float("nan")
        sem = math.sqrt(var / len(ws)) if len(ws) > 1 else float("nan")
        rows_out.append((W, sd, n_samples, mean, sem, alive / len(ws), round(dt, 1)))
        print(f"W={W} seed={sd}: omega_hat={mean:.6g} +/- {sem:.3g}  "
              f"alive={alive}/{n_samples}  t={dt:.1f}s", flush=True)
    return rows_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--lib", choices=["exp", "int"], default="exp")
    ap.add_argument("--no-la2", action="store_true")
    args = ap.parse_args()
    M = M_EXP if args.lib == "exp" else M_INT
    res = run_W(args.W, M, args.n, args.seeds, la2=not args.no_la2)
    newfile = not os.path.exists(OUT)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(["W", "lib", "seed", "n", "omega_hat", "sem", "alive_rate", "t_s"])
        for r in res:
            w.writerow([r[0], args.lib, r[1], r[2], r[3], r[4], r[5], r[6]])
    print(f"appended -> {OUT}")
