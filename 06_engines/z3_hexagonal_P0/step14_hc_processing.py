# -*- coding: utf-8 -*-
"""Step 14: honeycomb TM data processing (arithmetic ternary + binary
control) -- finalization of the s_inf reading.

Data: TSDHC_tm_rho.csv (binary Ly=2..8, seven points; ternary {0,1,2}
Ly=2..7, six points; all Tr(T^Lx) gates passed against archived exact
counts).
Three fit forms (per-width lambda(W) = ln rho(W)):
  A area law: lambda = s_inf*W + c (linear least squares);
  B 1/W correction: lambda/W = s_inf + c/W (reparameterization of A, equal
    weights);
  C second order: lambda = s_inf*W + c + d/W.
Reading:
  ternary: compare against the old trace-method value 0.45 (does the TM
           support / revise it upward?);
  binary : is s_inf nonzero (quantified death of the binary-honeycomb
           zero-entropy anchor; cross-checked against the 3.12.12 binary
           mid per-cell s x 3 -- same system).
Output: TSDHC_processing.csv + stdout reading.
"""
import csv
import math
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'TSDHC_tm_rho.csv')
OUT = os.path.join(HERE, 'TSDHC_processing.csv')


def load():
    d = {}
    with open(SRC, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            tag = r['alphabet']
            Ly = int(r['Ly'])
            ln = float(r['ln_rho'])
            rel = float(r['relerr'])
            d.setdefault(tag, []).append((Ly, ln, rel))
    for t in d:
        d[t].sort()
    return d


def linfit(xs, ys):
    A = np.polyfit(xs, ys, 1)
    pred = np.polyval(A, xs)
    resid = ys - pred
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return A[0], A[1], rms, resid


def fits_for(Ws, lns):
    x_inv = 1.0 / np.array(Ws)
    y = np.array(lns)
    # A: lambda = s*W + c
    sA, cA, rmsA, _ = linfit(np.array(Ws, float), y)
    # B: lambda/W = s + c/W (y' = lambda/W fit against 1/W; all points
    # equal weight)
    y2 = y / np.array(Ws, float)
    sB, cB, rmsB, _ = linfit(x_inv, y2)
    # C: lambda = s*W + c + d/W (3-parameter least squares, normal equations)
    M = np.column_stack([np.array(Ws, float), np.ones(len(Ws)), x_inv])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    sC, cC, dC = coef
    pred = M @ np.array([sC, cC, dC])
    rmsC = float(np.sqrt(np.mean((y - pred) ** 2)))
    return dict(A=(sA, cA, rmsA), B=(sB, cB, rmsB), C=(sC, cC, dC, rmsC))


def main():
    data = load()
    out = []
    print('=== honeycomb TM data processing (per-width lambda(W) = ln rho(W)) ===')
    verdict = {}
    for tag, lab, dom in (('ternary', 'arithmetic ternary', '{1,2,3}'),
                          ('binary', 'binary', '{1,2}')):
        pts = data[tag]
        Ws = [p[0] for p in pts]
        lns = [p[1] for p in pts]
        rels = [p[2] for p in pts]
        f = fits_for(Ws, lns)
        sA, cA, rmsA = f['A']
        sB, cB, rmsB = f['B']
        sC, cC, dC, rmsC = f['C']
        print(f'\n--- {lab} {dom} ({len(Ws)} points, W={min(Ws)}..{max(Ws)};'
              f' max rel={max(rels):.1e}) ---')
        print(f'  lambda(W) = {lns}')
        print(f'  lambda/W  = {[round(l/w, 4) for l, w in zip(lns, Ws)]}')
        print(f'  A area law:       lambda = {sA:.4f}*W + {cA:.4f}   '
              f'(rms {rmsA:.4f})  => s_inf = {sA:.4f}')
        print(f'  B 1/W correction: lambda/W = {sB:.4f} + {cB:.4f}/W '
              f'(rms {rmsB:.4f})  => s_inf = {sB:.4f}')
        print(f'  C 3-parameter:    lambda = {sC:.4f}*W + {cC:.4f} + '
              f'{dC:.4f}/W  (rms {rmsC:.4f})  => s_inf = {sC:.4f}')
        s_inf = 0.5 * (sA + sB)
        verdict[tag] = (s_inf, f)
        out.append([tag, len(Ws), f'{min(Ws)}-{max(Ws)}',
                    f'{sA:.4f}', f'{cA:.4f}', f'{rmsA:.4f}',
                    f'{sB:.4f}', f'{cB:.4f}', f'{rmsB:.4f}',
                    f'{sC:.4f}', f'{cC:.4f}', f'{dC:.4f}', f'{rmsC:.4f}',
                    f'{s_inf:.4f}'])
    with open(OUT, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['alphabet', 'n_points', 'W_range',
                    'sA', 'cA', 'rmsA', 'sB', 'cB', 'rmsB',
                    'sC', 'cC', 'dC', 'rmsC', 's_inf_avg'])
        w.writerows(out)
    print(f'\nwritten {OUT}')
    # cross-validation: compare against the 3.12.12 mid (same system)
    print('\n=== cross-validation (two TM pipelines, same system) ===')
    print('  binary : honeycomb s_inf fit vs 3.12.12 mid k=2 extrapolation '
          f'0.0584 x 3 = {0.0584 * 3:.4f} per hole')
    print('  ternary: honeycomb s_inf fit vs 3.12.12 mid k=3 extrapolation '
          f'0.1638 x 3 = {0.1638 * 3:.4f} per hole')
    print('  (convention: honeycomb per hole = 3.12.12 per cell x 3; '
          'intercept-form extrapolation, window W <= 7/8)')


if __name__ == '__main__':
    main()
