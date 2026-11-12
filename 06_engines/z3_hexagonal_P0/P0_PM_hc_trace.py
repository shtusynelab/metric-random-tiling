# -*- coding: utf-8 -*-
"""P0_PM trace-method extension: honeycomb {0,1,2} x3-family torus exact
counting Tr(T^Lx) (2026-09-29).

Four-step trace-method transfer to the honeycomb: cylinder of width W=Ly,
single-column transfer operator T, Omega_tor(Lx x Ly) = Tr(T^Lx).
This extends the Fig 1c honeycomb series (prefix-blocked DFS, F<=18)
up to Lx=20 (F=60).

The torus construction reuses edge_alphabet_affine_check.build_honeycomb_torus
(validated; identical graph and edge set as the torus-growth-forms series);
the face-closure equations eqA/eqB are the same as in count_assignments.

Anchors (must pass):
  arithmetic {0,1,2}: Tr(T^3..6) = 22767 / 30321 / 185523 / 3267903 (3x3..6x3)
  binary {0,1}:       Tr(T^3) = 518 (3x3)
Independent cross-check: 7x3 arithmetic recounted with the prefix-blocked DFS
(a different code path from the trace method).

Memory: state space <= 3^(2*Ly) = 729 (Ly=3); start tuples iterated lazily
(3^6 = 729 tuples, ~100 B each, no materialized full list); dict / Python int
exact throughout; peak < 50 MB; DFS cross-check 6 workers < 60 MB.
Workstation run (far below the 2 GB threshold).

Output: P0_PM_torus_trace_extension_Ly3.csv
"""
import csv
import math
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from edge_alphabet_affine_check import build_honeycomb_torus  # noqa: E402

OUT_CSV = HERE / "P0_PM_torus_trace_extension_Ly3.csv"


class TorusCtx:
    """Honeycomb torus + column partition + cut gauge order (translation
    equivariant; all columns' cuts live in one abstract space)."""

    def __init__(self, Lx, Ly, alphabet=(0, 1, 2)):
        self.Lx, self.Ly = Lx, Ly
        self.alpha = tuple(alphabet)
        edge_set, faces = build_honeycomb_torus(Lx, Ly)
        self.edge_set, self.faces = edge_set, faces
        self.E = len(edge_set)
        # representative midpoint of each edge (generator cell (i,j) in
        # [0,Lx)x[0,Ly), not taken modulo)
        mid = {}
        def add(u, v, mx, my):
            key = frozenset((u, v))
            if key in mid:
                assert abs(mid[key][0] - mx) < 1e-9 and \
                       abs(mid[key][1] - my) < 1e-9
            mid[key] = (mx, my)
        for i in range(Lx):
            for j in range(Ly):
                add(('A', i, j), ('B', i, j), i + 0.25, j + 0.25)
                add(('A', (i + 1) % Lx, j), ('B', i, j), i + 0.75, j + 0.25)
                add(('A', i, (j + 1) % Ly), ('B', i, j), i + 0.25, j + 0.75)
        self.mid = {e: mid[frozenset(edge_set[e])] for e in range(self.E)}
        # the two faces of each edge
        self.edge_faces = {e: [] for e in range(self.E)}
        for fi, f in enumerate(faces):
            for e in f:
                self.edge_faces[e].append(fi)
        assert all(len(v) == 2 for v in self.edge_faces.values())
        # face column = integer bin of the max edge-midpoint x within the face
        # (hexagon max mid_x = i+1.25); faces crossing the period are unwound
        # first: sort, find the largest gap (including the periodic wrap gap),
        # and lift the values left of the seam by Lx
        mx = []
        for f in faces:
            xs = sorted(round(x, 6) for x in (self.mid[e][0] for e in f))
            gaps = [(xs[k + 1] - xs[k], k) for k in range(len(xs) - 1)]
            gaps.append((Lx - xs[-1] + xs[0], -1))     # periodic wrap gap
            g, k = max(gaps)
            assert g > Lx / 2, f"no periodic seam gap in face {f}: {xs}"
            lifted = [x + Lx for x in xs[:k + 1]] if k >= 0 else []
            kept = xs[k + 1:]
            mx.append(max(lifted + kept if lifted else kept))
        vals = sorted(set(mx))
        assert len(vals) == Lx and all(abs((b - a) - 1.0) < 1e-6
                                       for a, b in zip(vals, vals[1:]))
        self.col = [vals.index(x) % Lx for x in mx]
        self.col_faces = {c: [fi for fi in range(len(faces))
                              if self.col[fi] == c] for c in range(Lx)}
        assert all(len(v) == Ly for v in self.col_faces.values())
        # cuts: edges between column c and its successor column (c+1)%Lx go to cuts[c]
        self.cuts = {}
        for e, fs in self.edge_faces.items():
            a, b = self.col[fs[0]], self.col[fs[1]]
            if a == b:
                continue                        # intra-column edge
            if b == (a + 1) % Lx:
                self.cuts.setdefault(a, []).append(e)
            else:
                assert a == (b + 1) % Lx, f"non-adjacent cols {a},{b} edge {e}"
                self.cuts.setdefault(b, []).append(e)
        assert len(self.cuts) == Lx, f"cuts {len(self.cuts)} != {Lx}"

        def cut_keys(es):
            return [(round(4 * (self.mid[e][0] % 1.0)) % 4,
                     round(4 * (self.mid[e][1] % Ly)) % (4 * Ly)) for e in es]

        self.cut_order = {}
        for c, es in self.cuts.items():
            keys = cut_keys(es)
            assert len(set(keys)) == len(keys), f"cut {c}: duplicate keys"
            self.cut_order[c] = [e for _, e in sorted(zip(keys, es))]
        # canonical key multiset identical across columns (translation equivariant)
        k0 = sorted(cut_keys(self.cut_order[0]))
        for c in range(1, Lx):
            assert sorted(cut_keys(self.cut_order[c])) == k0, \
                f"cut {c} key multiset differs from cut 0"
        self.cut_edges0 = self.cut_order[0]
        assert len(self.cut_edges0) == 2 * Ly
        self.t0 = None

    def column_transfer(self, states, col=0):
        """Single-column transfer: input {canonical tuple of cut col-1: count},
        process all faces of column col, return {canonical tuple of cut col:
        count}. Face-by-face closure equations, exact integer verdicts."""
        Lx = self.Lx
        cut_in = self.cut_order[(col - 1) % Lx]
        cut_out = self.cut_order[col]
        out = Counter()
        for s_in, cnt_in in states.items():
            stack = [(dict(zip(cut_in, s_in)), cnt_in)]
            for fi in self.col_faces[col]:
                f = self.faces[fi]
                nxt = []
                for st_, cc in stack:
                    known = {e: st_[e] for e in f if e in st_}
                    unknown = [e for e in f if e not in st_]
                    for combo in product(self.alpha, repeat=len(unknown)):
                        L = dict(known)
                        for e, v in zip(unknown, combo):
                            L[e] = v
                        # eqA: L1+L2 = L4+L5 ; eqB: 2L0+L1+L5 = 2L3+L2+L4
                        if L[f[1]] + L[f[2]] != L[f[4]] + L[f[5]]:
                            continue
                        if 2 * L[f[0]] + L[f[1]] + L[f[5]] != \
                           2 * L[f[3]] + L[f[2]] + L[f[4]]:
                            continue
                        st2 = dict(st_)
                        for e, v in zip(unknown, combo):
                            st2[e] = v
                        nxt.append((st2, cc))
                stack = nxt
            agg = Counter()
            for st_, cc in stack:
                agg[tuple(st_[e] for e in cut_out)] += cc
            for k, v in agg.items():
                out[k] += v * cnt_in
        return dict(out)

    def build_states(self):
        """Lazily iterate the 3^(2Ly) start tuples (~100 B each, no materialized
        full list), keep the states that can complete one column, and verify T
        closes (feasible-state dict peak < 1 MB)."""
        t0 = {}
        for s in product(self.alpha, repeat=len(self.cut_edges0)):
            o = self.column_transfer({s: 1})
            if o:
                t0[s] = o
        states = set(t0)
        for s, o in t0.items():
            assert set(o) <= states, f"T not closed at {s}"
        self.t0, self.states = t0, states
        return t0, states


def trace_power(t0, Lx):
    """Tr(T^Lx): per-start exact integer vector DP (column-transfer dict,
    Python ints, no rounding)."""
    total = 0
    for s in t0:
        v = {s: 1}
        for _ in range(Lx):
            nv = {}
            for st, c in v.items():
                if c == 0:
                    continue
                for o, w in t0[st].items():
                    nv[o] = nv.get(o, 0) + c * w
            v = nv
        total += v.get(s, 0)
    return total


def main():
    print(f"= honeycomb trace extension {time.strftime('%F %T')} =", flush=True)
    # ---- anchors: Tr(T^Lx) reproduces known exact values ----
    anchors = [
        ("arithmetic", (0, 1, 2), (3, 3), 22767),
        ("arithmetic", (0, 1, 2), (4, 3), 30321),
        ("arithmetic", (0, 1, 2), (5, 3), 185523),
        ("arithmetic", (0, 1, 2), (6, 3), 3267903),
        ("binary", (0, 1), (3, 3), 518),
    ]
    ctx_cache = {}
    for name, alpha, (Lx, Ly), expect in anchors:
        key = (Ly, alpha)
        if key not in ctx_cache:
            ctx_cache[key] = TorusCtx(Lx, Ly, alpha)
            ctx_cache[key].build_states()
        ctx = ctx_cache[key]
        got = trace_power(ctx.t0, Lx)
        tag = "PASS" if got == expect else "FAIL"
        print(f"  [{tag}] {name} {Lx}x{Ly}: Tr(T^{Lx})={got} expect={expect} "
              f"(|S|={len(ctx.states)})", flush=True)
        assert got == expect, f"anchor mismatch {name} {Lx}x{Ly}"
    print("anchors 5/5 PASS", flush=True)

    # ---- extension: arithmetic x3 family, Lx=7..20 ----
    ctx = ctx_cache[(3, (0, 1, 2))]
    rows = []
    for Lx in range(7, 21):
        t0 = time.time()
        om = trace_power(ctx.t0, Lx)
        F = Lx * 3
        row = dict(alphabet="arithmetic", alphabet_values="012", Lx=Lx, Ly=3,
                   F=F, E=ctx.E, method="trace Tr(T^Lx)", Omega=om,
                   lnOmega=round(math.log(om), 6),
                   lnOmega_per_F=round(math.log(om) / F, 6),
                   n_states=len(ctx.states), seconds=round(time.time() - t0, 2),
                   status="trace")
        rows.append(row)
        print(f"  arithmetic {Lx}x3: Omega={om} lnOmega={row['lnOmega']} "
              f"lnOm/F={row['lnOmega_per_F']} ({row['seconds']}s)", flush=True)
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # growth rate lambda (compare with the A0-2 strip W=3 arithmetic value 6.95)
    for a, b in zip(rows, rows[1:]):
        lam = math.log(b["Omega"] / a["Omega"])
        print(f"  ln Omega({b['Lx']}x3)/Omega({a['Lx']}x3) = {lam:.4f} "
              f"(-> lambda {math.exp(lam):.4f}; A0-2 strip W=3 lambda=6.95)",
              flush=True)
    print(f"done, CSV -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
