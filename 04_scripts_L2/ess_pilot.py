# -*- coding: utf-8 -*-
"""ess_pilot.py -- mixing diagnostic for the production PT sampler (v6.2.1).

Replicates the vd_torus_run protocol (same PT ladder, same eq) and logs the
total length sum(l) of the output (hot) replica every dt steps over a
production window. The archived series gives the integrated autocorrelation
time tau_int(sum l) and the correlation at the production thinning lag
(4e4 steps), i.e. how many effectively independent snapshots the 20-snapshot
production protocol carries.

usage: python3 ess_pilot.py <tag> <a1,a2[,a3]> <seed> [eq] [prod] [dt]
output: /home/hzw/mof_psc_vd/ess_<tag>.npz
ASCII only on purpose.
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/home/hzw/mof_psc_vd")
from sampler2 import ParallelTemperingSampler  # noqa: E402


def main():
    tag = sys.argv[1]
    alphabet = tuple(int(v) for v in sys.argv[2].split(","))
    seed = int(sys.argv[3])
    eq = int(sys.argv[4]) if len(sys.argv) > 4 else 1_500_000
    prod = int(sys.argv[5]) if len(sys.argv) > 5 else 1_500_000
    dt = int(sys.argv[6]) if len(sys.argv) > 6 else 100
    t0 = time.time()
    print(f"[{tag}] A={alphabet} seed={seed} eq={eq} prod={prod} dt={dt}",
          flush=True)
    pt = ParallelTemperingSampler(13, 13, alphabet, seed=seed)
    for it in range(eq):
        pt.step()
        if (it + 1) % 500_000 == 0:
            print(f"[{tag}] eq {it+1}/{eq} ({time.time()-t0:.0f}s)", flush=True)
    hot = pt.reps[-1]
    n = prod // dt
    series = np.empty(n, dtype=np.float64)
    for it in range(prod):
        pt.step()
        if (it + 1) % dt == 0:
            series[(it + 1) // dt - 1] = float(np.sum(hot.sample_constants()))
        if (it + 1) % 500_000 == 0:
            print(f"[{tag}] prod {it+1}/{prod} ({time.time()-t0:.0f}s)",
                  flush=True)
    swap = pt.swaps_accepted / max(pt.swaps_proposed, 1)
    acc = pt.reps[-1].steps_accepted / max(pt.reps[-1].steps_proposed, 1)
    out = f"/home/hzw/mof_psc_vd/ess_{tag}.npz"
    np.savez_compressed(out, series=series, dt=dt, eq=eq, prod=prod,
                        alphabet=np.array(alphabet), seed=seed, swap=swap,
                        acc=acc, wall=time.time() - t0)
    print(f"[{tag}] saved {out} swap={swap:.3f} acc={acc:.3f} "
          f"wall={time.time()-t0:.0f}s mean={series.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
