# -*- coding: utf-8 -*-
"""P0-A2b collection and 1/W extrapolation v2 (server side; triggered by the
inspection driver).

Merge strategy:
  1. ok rows of the main CSV (L=10 exact) -> source=exact;
  2. short-strip CSV: exact Omega at L=3..6 for states_exceeded cells ->
     s_L = lnOmega/(W*L) extrapolated linearly in 1/L -> s(W),
     source=exact_shortL (requires >= 2 feasible L values; cells with
     fewer go on the missing list);
  3. for each alphabet, the s(W) points with source in {exact,
     exact_shortL} are extrapolated linearly in 1/W -> s(inf).
Writes the extrapolation CSV and prints a summary.
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', default=str(HERE / 'P0_A2b_TM_entropy_density.csv'),
                    help='main CSV (L=10 exact rows)')
    ap.add_argument('--short',
                    default=str(HERE / 'P0_A2b_TM_short_strip_exact_counting.csv'),
                    help='short-strip CSV (exact Omega at L=3..6)')
    ap.add_argument('--out', default=str(HERE / 'P0_A2b_TM_extrapolation.csv'),
                    help='output extrapolation CSV (default: here)')
    args = ap.parse_args()
    MAIN, SHORT, OUT = args.main, args.short, args.out

    rows = {}
    df_m = pd.read_csv(MAIN)
    for _, r in df_m.iterrows():
        if r['status'] == 'ok':
            rows[(r['alphabet'], int(r['W']))] = dict(
                alphabet=r['alphabet'], W=int(r['W']), s=float(r['s']),
                source='exact', kL='')
    try:
        df_s = pd.read_csv(SHORT)
    except FileNotFoundError:
        df_s = pd.DataFrame()
    still_missing = []
    if len(df_s):
        for (alpha, W), g in df_s[df_s['status'] == 'ok'].groupby(
                ['alphabet', 'W']):
            if (alpha, W) in rows:
                continue
            g = g.sort_values('L')
            if len(g) < 2:
                still_missing.append((alpha, W, 'too few feasible L points'))
                continue
            x = 1.0 / g['L'].values
            y = np.array([math_log_omega(float(om), int(L), W)
                          for om, L in zip(g['omega'], g['L'])])
            k = min(len(g), 3)
            b, s0 = np.polyfit(x[-k:], y[-k:], 1)
            rows[(alpha, W)] = dict(alphabet=alpha, W=W, s=round(s0, 4),
                                    source='exact_shortL',
                                    kL=';'.join(map(str, g['L'].values)))
    df = pd.DataFrame(list(rows.values()))
    print(f"merged: {len(df)} (alphabet, W) points "
          f"(exact {int((df['source']=='exact').sum())}, "
          f"exact_shortL {int((df['source']=='exact_shortL').sum())})")
    summ = []
    for alpha, g in df.groupby('alphabet'):
        g = g.sort_values('W')
        if len(g) >= 2:
            x = 1.0 / g['W'].values
            y = g['s'].values
            b, s0 = np.polyfit(x, y, 1)
            summ.append(dict(alphabet=alpha, s_inf=round(s0, 4),
                             slope=round(b, 4), kpts=len(g),
                             Ws=';'.join(map(str, g['W'].values)),
                             s_values=';'.join(f'{v:.4f}' for v in y),
                             source='+'.join(sorted(set(g['source'])))))
            print(f"{alpha:22s} s(inf)~{s0:+.4f} (slope {b:+.3f}, "
                  f"W={list(g['W'].values)}, "
                  f"{'+'.join(sorted(set(g['source'])))})")
        else:
            print(f"{alpha:22s} only {len(g)} point(s), cannot extrapolate "
                  f"(W={list(g['W'].values)}, {list(g['source'])})")
    pd.DataFrame(summ).to_csv(OUT, index=False, encoding='utf-8-sig')
    if still_missing:
        print('cells with too few short-strip L points, not extrapolated:')
        for a, w, why in still_missing:
            print(f'  {a} W={w}: {why}')
    print(f'extrapolation table written to {OUT}; '
          f'{len(summ)} alphabets extrapolable')


def math_log_omega(omega, L, W):
    import math
    return math.log(omega) / (W * L)


if __name__ == '__main__':
    main()
