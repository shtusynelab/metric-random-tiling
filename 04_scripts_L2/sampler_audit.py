# -*- coding: utf-8 -*-
"""Historical audit: the single-seed DFS sampler it measures is retired; the
production ensembles use the exact-uniform PT sampler (vd_torus_run*.py).

Sampler uniformity audit (small-size exact enumeration vs the empirical
distribution of the DFS sampler).
Object: the solve_tessellation single-seed sampling of vd_suite2
(metric_library libraries).
System: the {1,2} value domain, n = 4 disc (N = 37 pores).  Steps:
  1) exact enumeration of all closed assignments (independent of the counter:
     per-cell DFS over the full tree, counting leaves) -> Omega and the
     configuration set;
  2) sample M = 600 independent seeds (rejecting seeds that miss the value
     domain), count per-configuration hit frequencies;
  3) criteria against the uniform distribution -- TV distance, chi2,
     max-hit multiple, number of zero-hit configurations.
Expected: the DFS is non-uniform (early cells strongly constrained before
later ones); the deviation is quantified for the S9 calibre note."""
import sys, random
from pathlib import Path
from itertools import product
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(PC.HC))
import pseudocrystal_lib as pl  # noqa: E402
from vd_suite2 import metric_library  # noqa: E402

N_RING = 4
M = 600
A = (1, 2)
lib = metric_library(A)
valid = [tuple(w) for w in lib.weights]
nv = len(valid)

cells = list(pl.hex_coords(N_RING - 1))
cellset = set(cells)
order = [cells[0]]
seen = {cells[0]}
import queue as _q
bq = _q.Queue()
bq.put(cells[0])
while not bq.empty():
    c = bq.get()
    for d in pl.DIRECTIONS:
        nb = (c[0] + d[0], c[1] + d[1])
        if nb in cellset and nb not in seen:
            seen.add(nb)
            order.append(nb)
            bq.put(nb)
slot_edge = {}
for c in cells:
    for k, d in enumerate(pl.DIRECTIONS):
        nb = (c[0] + d[0], c[1] + d[1])
        key = tuple(sorted([(c, k), (nb, (k + 3) % 6)])) if nb in cellset else ((c, k),)
        slot_edge[(c, k)] = key

# 1) exact enumeration (full tree, no randomness)
sols = {}
def rec(idx, assign):
    if idx == len(order):
        key = tuple(assign[slot_edge[(c, k)]] for c in order for k in range(6))
        sols[key] = sols.get(key, 0) + 1
        return
    c = order[idx]
    known = {k: assign[e] for k in range(6) if (e := slot_edge[(c, k)]) in assign}
    for t in valid:
        if all(t[k] == v for k, v in known.items()):
            new = {slot_edge[(c, k)]: t[k] for k in range(6)
                   if slot_edge[(c, k)] not in assign}
            assign.update(new)
            rec(idx + 1, assign)
            for e in new:
                del assign[e]
rec(0, {})
Omega = sum(sols.values())
print(f"exact enumeration: Omega({A}, n={N_RING} disc) = {Omega}, "
      f"distinct edge-assignments = {len(sols)}", flush=True)

# 2) sampling (via solve_tessellation, the same path as the retired
#    production pipeline; tile_idx -> edge-weight key)
def sol_key(sol):
    return tuple(lib.weights[sol[c]][k] for c in order for k in range(6))

hits = {k: 0 for k in sols}
got = 0
seed = 990001
while got < M:
    try:
        sol, _ = pl.solve_tessellation(N_RING, lib, periodic=False,
                                       rng=random.Random(seed))
        k = sol_key(sol)
        hits[k] = hits.get(k, 0) + 1
        got += 1
    except Exception:
        pass
    seed += 1

# 3) criteria
cnt = np.array([hits.get(k, 0) for k in sols], dtype=float)
p_uni = 1.0 / len(sols)
exp = M * p_uni
tv = 0.5 * np.abs(cnt / M - p_uni).sum()
chi2 = ((cnt - exp) ** 2 / exp).sum()
dof = len(sols) - 1
zero = int((cnt == 0).sum())
top = cnt.max()
print(f"sampler: M={M} draws over {len(sols)} configs")
print(f"  TV(distance to uniform) = {tv:.4f}")
print(f"  chi2 = {chi2:.1f} (dof={dof}), reduced = {chi2/dof:.2f}")
print(f"  zero-hit configs = {zero}/{len(sols)} (uniform expect "
      f"{len(sols)*np.exp(-exp):.1f})")
print(f"  max hits = {top:.0f} vs uniform mean {exp:.2f} "
      f"(x{top/exp:.1f})")
print(f"  reject rate: {seed-990001-M} rejects / {seed-990001} seeds")
