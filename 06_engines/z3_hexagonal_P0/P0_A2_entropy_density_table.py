# -*- coding: utf-8 -*-
"""P0-A2 entropy density table, first round (proposal deliverable #1).

For every subset of {222,223,233,333} x n=2..6, estimate Omega_N with the
Knuth estimator (pattern-level convention: edge-color-field greedy +
arm-number realizability post-verification, open boundary), report
s_n = lnOmega_N / N_holes, and extrapolate linearly in 1/N -> entropy
density s(L).

Conventions and approximations (see the pseudocrystal_lib.py module
docstring):
- each counted solution = one assembled pattern with a realizable
  arm-number field (each pattern counted once);
- boundary effects, heavy-tailed variance and verify survival are reported
  as they are;
- the Knuth Omega distribution is heavy-tailed; sem is indicative only,
  and s is read mainly through order-of-magnitude separation.
"""
import sys, time, math, random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pseudocrystal_lib as pl

SAMPLE_LADDER = [4000, 20000, 100000, 400000]
ALIVE_TARGET = 0.02
MAX_N = 6


def run_one(args):
    alphabet, n, seed0 = args
    lib = pl.build_library(alphabet)
    out = []
    for li, samples in enumerate(SAMPLE_LADDER):
        r = pl.knuth_estimate(n, lib, periodic=False, n_samples=samples,
                              seed=seed0 * 1000 + li)
        out.append(r)
        if r['alive_rate'] >= ALIVE_TARGET:
            break
        if r['alive_rate'] == 0.0 and li >= 1:
            break  # two ladder levels both zero; even larger samples add little
    # report the level with the highest alive rate (survival first, then sample size)
    best = max(out, key=lambda r: (r['alive_rate'], r['n_samples']))
    return dict(alphabet='+'.join(alphabet), n=n, N_holes=best['N_holes'],
                samples=best['n_samples'], alive_rate=best['alive_rate'],
                omega_mean=best['omega_mean'], omega_sem=best['omega_sem'],
                ln_omega=best['ln_omega'], ln_omega_sem_lnsp=best['ln_omega_sem_lnsp'],
                ladder_tried=len(out))


def main():
    t0 = time.time()
    tasks = []
    for subset in pl.ALL_SUBSETS:
        for n in range(2, MAX_N + 1):
            tasks.append((subset, n, 2026))
    print(f"A2 entropy density table: {len(pl.ALL_SUBSETS)} subsets x "
          f"n=2..{MAX_N} = {len(tasks)} tasks")

    results = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, res in enumerate(ex.map(run_one, tasks)):
            results.append(res)
            print(f"[{i+1}/{len(tasks)}] {res['alphabet']} n={res['n']} "
                  f"alive={res['alive_rate']:.3f} lnOmega={res['ln_omega']:.2f} "
                  f"s_n={res['ln_omega']/res['N_holes']:.3f} "
                  f"(samples={res['samples']}, {time.time()-t0:.0f}s)", flush=True)

    # Scaling extrapolation: for an open disk the boundary correction is
    # O(sqrt(N)) (lnOmega_N = s*N - c*sqrt(N) + ...); main variable 1/sqrt(N)
    # with 1/N as a cross-check; fit with >= 2 points, report the intercept.
    print("\n=== entropy density extrapolation (s_n vs 1/sqrt(N) and 1/N "
          "linear-fit intercepts) ===")
    import pandas as pd
    df = pd.DataFrame(results)
    summary = []
    for alpha, g in df.groupby('alphabet'):
        gg = g[(g['alive_rate'] > 0) & (g['ln_omega'] > 0)].sort_values('n')
        row = dict(alphabet=alpha, kpts=len(gg))
        if len(gg) >= 2:
            inv_sqrtN = 1.0 / np.sqrt(gg['N_holes'].values)
            inv_N = 1.0 / gg['N_holes'].values
            y = (gg['ln_omega'] / gg['N_holes']).values
            b1, s1 = np.polyfit(inv_sqrtN, y, 1)
            b2, s2 = np.polyfit(inv_N, y, 1)
            row.update(s_extrap_invsqrtn=s1, slope_invsqrtn=b1,
                       s_extrap_invN=s2, slope_invN=b2)
            summary.append(row)
            print(f"{alpha:22s} s(1/sqrt(N))~{s1:+.3f}  s(1/N)~{s2:+.3f}  "
                  f"kpts={len(gg)}")
        elif len(gg) == 1:
            r0 = gg.iloc[0]
            row['s_extrap_invsqrtn'] = r0['ln_omega'] / r0['N_holes']
            summary.append(row)
            print(f"{alpha:22s} single point s_n(n={r0['n']}) = "
                  f"{r0['ln_omega']/r0['N_holes']:+.3f}")
        else:
            summary.append(row)
            print(f"{alpha:22s} unreachable by sampling (alive rate 0)")

    df.to_csv(ROOT / "P0_A2_entropy_density_table.csv", index=False,
              encoding='utf-8-sig')
    pd.DataFrame(summary).to_csv(ROOT / "P0_A2_entropy_density_extrapolation.csv",
                                 index=False, encoding='utf-8-sig')
    print(f"\ndone in {time.time()-t0:.0f}s; CSVs written to the script directory.")


if __name__ == "__main__":
    import numpy as np
    main()
