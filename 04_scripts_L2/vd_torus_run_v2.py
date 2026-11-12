# -*- coding: utf-8 -*-
"""Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

vd_torus_run_v2.py -- tension-ensemble rerun (multi-chain, densified PT
ladder, mirror gate support).

Compared with v1 (vd_torus_run.py):
  * denser temperature ladder (13 rungs) to raise the replica-exchange flow
    toward the hot replica (v1 hot-delta acceptance at |h|=0.30 was ~1e-5);
  * one process = one independent chain; several chains per h give an honest
    ensemble scatter (v1 used 20 snapshots of a single chain);
  * per-configuration sum(l) logged so the exact mirror identity
    m(h) + m(-h) = 4 per edge (12 per face) can be checked across the
    +h / -h chain pools post hoc.

usage:
  python3 vd_torus_run_v2.py <h> <chain> <outdir> [eq] [thin] [ncfg] [L] [seed]

output: <outdir>/vd2_h{+/-x.xx}_c{k}.npz with cfgs (ncfg, E) int64 and
        metadata (eq, thin, seed, h, L, swap, acc, wall, suml per config).
ASCII-only on purpose (heredoc/encoding safety).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "external"))
from metric_hc import build_torus, closure_rows          # noqa: E402
from sampler2 import ParallelTemperingSampler            # noqa: E402

ALPHABET = (1, 2, 3)
TS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
      0.95, 1.0)


def main():
    h = float(sys.argv[1])
    chain = int(sys.argv[2])
    outdir = Path(sys.argv[3]).expanduser()
    eq = int(sys.argv[4]) if len(sys.argv) > 4 else 1_500_000
    thin = int(sys.argv[5]) if len(sys.argv) > 5 else 100_000
    ncfg = int(sys.argv[6]) if len(sys.argv) > 6 else 5
    L = int(sys.argv[7]) if len(sys.argv) > 7 else 13
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 20262100 + 97 * chain

    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"h{h:+.2f}_c{chain}"
    t0 = time.time()
    print(f"[{tag}] A={ALPHABET} L={L} eq={eq} thin={thin} ncfg={ncfg} "
          f"seed={seed} Ts={len(TS)}rungs", flush=True)

    pt = ParallelTemperingSampler(L, L, ALPHABET, seed=seed, Ts=TS, h=h)
    E = pt.E
    rows = closure_rows(pt.reps[0].faces, E)
    print(f"[{tag}] E={E} K={pt.reps[0].K}", flush=True)

    for it in range(eq):
        pt.step()
        if (it + 1) % 200_000 == 0:
            sa = pt.swaps_accepted / max(pt.swaps_proposed, 1)
            ha = pt.reps[-1].steps_accepted / max(pt.reps[-1].steps_proposed, 1)
            print(f"[{tag}] eq {it+1}/{eq} swap={sa:.2f} hot-acc={ha:.2e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            sys.stdout.flush()

    cfgs = pt.sample(ncfg, thin=thin)
    X = np.array(cfgs, dtype=np.int64)

    bad = 0
    for x in cfgs:
        if any(sum(c * int(x[k]) for k, c in r.items()) != 0 for r in rows):
            bad += 1
    swap = pt.swaps_accepted / max(pt.swaps_proposed, 1)
    acc = pt.reps[-1].steps_accepted / max(pt.reps[-1].steps_proposed, 1)
    suml = X.sum(axis=1)
    print(f"[{tag}] closure-bad={bad}/{ncfg}  <l>={X.mean():.4f}  "
          f"sum_l={suml.tolist()}  swap={swap:.2f} hot-acc={acc:.2e}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    out = outdir / f"vd2_{tag}.npz"
    np.savez_compressed(out, cfgs=X, alphabet=np.array(ALPHABET), eq=eq,
                        thin=thin, seed=seed, h=h, L=L, chain=chain,
                        swap=swap, acc=acc, wall=time.time() - t0, suml=suml)
    print(f"[{tag}] saved {out}  closure-bad={bad}", flush=True)
    if bad:
        sys.exit(3)


if __name__ == "__main__":
    main()
