# -*- coding: utf-8 -*-
"""P0-A2b transfer-matrix exact counting (all alphabets, within reach).

Strategy: exact strip counting under the arm-number-field convention.
Restricted alphabets (strong constraints, unreachable by Knuth sampling)
are the biggest incremental target of exact counting; loose alphabets have
large state spaces and are only run on short strips (order-of-magnitude
reference), their high-quality Knuth data being already in P0_A2.
Output: per-(alphabet, W) row-wise entropy density s = lnOmega/(W*L) and a
1/W extrapolation.
"""
import sys, time, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pseudocrystal_lib as pl

L_ROWS = 10
MAX_STATES = 600_000
W_RANGE = range(3, 6)


def run_one(args):
    alpha, W = args
    try:
        lib = pl.build_library(alpha)
        t0 = time.time()
        omega, oh = pl.tm_count_strip(W, lib, n_rows=L_ROWS, max_states=MAX_STATES)
    except Exception as e:  # memory guard: a single task failure must not kill the run
        return dict(alphabet='+'.join(alpha), W=W, status=f'error:{type(e).__name__}',
                    seconds=0.0)
    if omega is None:
        return dict(alphabet='+'.join(alpha), W=W, status='states_exceeded',
                    seconds=time.time() - t0)
    ss = [math.log(o) / (W * (i + 1)) for i, o in enumerate(oh)]
    return dict(alphabet='+'.join(alpha), W=W, status='ok', omega=omega,
                L=L_ROWS,
                s_rows=';'.join(f'{x:.4f}' for x in ss),
                s=ss[-1], s_prev=ss[-2] if len(ss) > 1 else None,
                seconds=time.time() - t0)


def main():
    t0 = time.time()
    tasks = [(a, W) for a in pl.ALL_SUBSETS for W in W_RANGE]
    print(f"A2b transfer matrix: {len(pl.ALL_SUBSETS)} subsets x "
          f"W={list(W_RANGE)} = {len(tasks)} tasks", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, res in enumerate(ex.map(run_one, tasks)):
            results.append(res)
            if res['status'] == 'ok':
                print(f"[{i+1}/{len(tasks)}] {res['alphabet']:22s} W={res['W']} "
                      f"s={res['s']:.4f} (prev {res['s_prev']:.4f}) "
                      f"Omega={res['omega']:.3e} {res['seconds']:.0f}s", flush=True)
            else:
                print(f"[{i+1}/{len(tasks)}] {res['alphabet']:22s} W={res['W']} "
                      f"{res['status']} {res['seconds']:.0f}s", flush=True)

    import pandas as pd
    pd.DataFrame(results).to_csv(ROOT / "P0_A2b_TM_entropy_density.csv",
                                 index=False, encoding='utf-8-sig')
    # 1/W extrapolation (only status=ok alphabets with >= 2 points at W>=3)
    print("\n=== s(W) extrapolation ===")
    df = pd.DataFrame(results)
    summ = []
    for alpha, g in df[df['status'] == 'ok'].groupby('alphabet'):
        g = g.sort_values('W')
        if len(g) >= 2:
            import numpy as np
            x = 1.0 / g['W'].values
            y = g['s'].values
            b, s0 = np.polyfit(x, y, 1)
            summ.append(dict(alphabet=alpha, s_inf=s0, slope=b, kpts=len(g),
                             Ws=';'.join(map(str, g['W'].values))))
            print(f"{alpha:22s} s(inf)~{s0:+.4f} (slope {b:+.3f}, W={list(g['W'].values)})")
    pd.DataFrame(summ).to_csv(ROOT / "P0_A2b_TM_extrapolation.csv",
                              index=False, encoding='utf-8-sig')
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
