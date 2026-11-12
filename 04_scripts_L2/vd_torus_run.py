# -*- coding: utf-8 -*-
"""Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

vd_torus_run.py -- exact-uniform sampling of one value-domain ensemble.
Parallel tempering (sampler2 PT) on the 13x13 honeycomb torus (= 169 pores,
the same pore count as the L2 diffraction suite).
Output: npz (cfgs (n, E) int64 + metadata) + gate record (closure /
reflection criterion / swap rate), written to <DATA>/raw by default.
usage: python3 vd_torus_run.py <tag> <a1,a2[,a3]> [eq] [thin] [ncfg] [seed] [h] [L] [out_dir]
Memory ~0.5-1 GB per task; 7 parallel tasks ~4-7 GB (cluster execution)."""
import sys, os, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths_common as PC  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "external"))
from metric_hc import build_torus, closure_rows, count_solutions
from sampler2 import ParallelTemperingSampler

LX = LY = 13


def reflect_mean(alphabet):
    """Exact <l> of the reflection-symmetric ensemble (per-edge mean under
    the mirror map); used by the gate."""
    a = sorted(alphabet)
    return sum(a) / len(a)


def main():
    tag = sys.argv[1]
    alphabet = tuple(int(v) for v in sys.argv[2].split(","))
    eq = int(sys.argv[3]) if len(sys.argv) > 3 else 1_500_000
    thin = int(sys.argv[4]) if len(sys.argv) > 4 else 40_000
    ncfg = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 20261100
    h = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    L = int(sys.argv[8]) if len(sys.argv) > 8 else 13
    out_dir = Path(sys.argv[9]) if len(sys.argv) > 9 else PC.DATA / "raw"
    globals()["LX"] = globals()["LY"] = L
    t0 = time.time()
    print(f"[{tag}] A={alphabet} Lx=Ly={L} eq={eq} thin={thin} ncfg={ncfg} "
          f"seed={seed} h={h}", flush=True)

    # gate 0: independent exact counter reachable (small sizes only)
    if LX * LY <= 12:
        om = count_solutions(LX, LY, alphabet)
        print(f"[{tag}] gate0 exact Omega={om}", flush=True)

    pt = ParallelTemperingSampler(LX, LY, alphabet, seed=seed, h=h)
    E = pt.E
    rows = closure_rows(pt.reps[0].faces, E)
    print(f"[{tag}] E={E} K={pt.reps[0].K} ncyc={pt.reps[0].ncyc}", flush=True)

    for it in range(eq):
        pt.step()
        if (it + 1) % 200_000 == 0:
            sa = pt.swaps_accepted / max(pt.swaps_proposed, 1)
            print(f"[{tag}] eq {it+1}/{eq} swap={sa:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    cfgs = pt.sample(ncfg, thin=thin)
    X = np.array(cfgs, dtype=np.int64)
    # gate 1: closure (all rows, every configuration)
    bad = 0
    for x in cfgs:
        if any(sum(c * int(x[k]) for k, c in r.items()) != 0 for r in rows):
            bad += 1
    # gate 2: value domain + reflection criterion
    rm = reflect_mean(alphabet)
    ml = float(X.mean())
    # gate 3: diagnostics
    swap = pt.swaps_accepted / max(pt.swaps_proposed, 1)
    acc = pt.reps[-1].steps_accepted / max(pt.reps[-1].steps_proposed, 1)
    print(f"[{tag}] closure-bad={bad}/{ncfg}  <l>={ml:.4f} (reflect {rm:.4f})  "
          f"swap={swap:.2f} hot-acc={acc:.2f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    assert bad == 0, "closure violated"
    assert X.min() >= min(alphabet) and X.max() <= max(alphabet)

    # save before the final gates (a gate failure must not lose the sample)
    if L == 13 and h == 0.0:
        out = out_dir / f"torus13_{tag}.npz"
    else:
        out = out_dir / (
            f"tor{L}_{tag}.npz" if h == 0.0
            else f"tor{L}_{tag}_h{h:+.2f}.npz")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cfgs=X, alphabet=np.array(alphabet),
                        eq=eq, thin=thin, seed=seed, h=h, L=L,
                        swap=swap, acc=acc, wall=time.time() - t0)
    print(f"[{tag}] saved {out}", flush=True)

    # reflection criterion: {1,2}/{1,2,3}/{1,2,4} are anchored by exact
    # enumeration (at 3x3/4x3, <l> = mean(A) edge-wise exact); non-arithmetic
    # alphabets that miss it at large size only warn
    if h == 0.0 and not os.environ.get("VD_RELAX_GATE"):
        assert abs(ml - rm) < 0.02 * rm, "reflection criterion failed"


if __name__ == "__main__":
    main()
