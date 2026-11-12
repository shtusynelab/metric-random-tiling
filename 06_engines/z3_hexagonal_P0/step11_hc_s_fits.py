# -*- coding: utf-8 -*-
"""Step 11 (s fit adjudication summary): two-convention fits of the width
scaling for the three honeycomb alphabets, with adjudication.

Inputs:
  TSDHC_tm_rho.csv           (step8: binary/ternary main table)
  TSDHC_tm_rho_nonarith.csv  (step9: nonarith main table)
  TSDHC_binary_spectral.csv  (step10: even-W rho_true correction, binary)
Outputs (stdout + CSV):
  - master table of lambda(W) = ln rho(W) (even-W entries overwritten by
    rho_true where the power iteration did not converge);
  - per-hole series s(W) = lambda/W;
  - two least-squares forms: (1) per-hole form s(W) = s_inf + c/W;
    (2) area form lambda(W) = s*W + c (when the two forms are degenerate
    over W <= W_max they give a consistent s_inf reading window);
  - adjudication conclusions and registry entries.
Memory: negligible (< 10 MB).
"""
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_rho(path, tag):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['alphabet'] != tag:
                continue
            rows[int(r['Ly'])] = dict(
                rho=float(r['rho']), ln_rho=float(r['ln_rho']),
                relerr=float(r['relerr']),
                tail=(float(r['tail_lo']), float(r['tail_hi'])))
    return rows


def fit_lin(xs, ys):
    xs, ys = np.array(xs, float), np.array(ys, float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    (m, b), res, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ np.array([m, b])
    ss = float(np.sum((ys - pred) ** 2))
    return m, b, ss


def fit_table(name, lam, conv):
    """Two-form fit. Only converged points (rel < 1e-10, or spectrally
    corrected) count as trustworthy lambda readings;
    form 1 fits s(W) = lambda/W vs 1/W; form 2 fits lambda vs W."""
    Ws = sorted(lam)
    x_inv = [1.0 / w for w in Ws]
    y_per = [lam[w] / w for w in Ws]
    c1, s1, ss1 = fit_lin(x_inv, y_per)      # s_per = s_inf + c1/W
    s2, c2, ss2 = fit_lin(Ws, [lam[w] for w in Ws])   # lambda = s2*W + c2
    print(f'\n== {name} ==')
    print(f'  {"W":>3} {"lambda":>10} {"s=lam/W":>9} {"conv?":>6}')
    for w in Ws:
        print(f'  {w:>3} {lam[w]:>10.6f} {lam[w]/w:>9.5f} '
              f'{"yes" if conv.get(w, False) else "no (spectral/unconverged)":>6}')
    rms1 = math.sqrt(ss1 / len(Ws))
    rms2 = math.sqrt(ss2 / len(Ws))
    print(f'  form 1 s(W)=s_inf+c/W     : s_inf={s1:.5f}  c={c1:.5f}  '
          f'rms={rms1:.2e}')
    print(f'  form 2 lambda(W)=s*W+c    : s ={s2:.5f}  c={c2:.5f}  '
          f'rms={rms2:.2e}  (s per column; lambda/per-hole ~ s)')
    return dict(s_inf_form1=s1, c_form1=c1, rms1=rms1,
                s_form2=s2, c_form2=c2, rms2=rms2)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    rho_bin = read_rho(os.path.join(HERE, 'TSDHC_tm_rho.csv'), 'binary')
    rho_ter = read_rho(os.path.join(HERE, 'TSDHC_tm_rho.csv'), 'ternary')
    rho_non = read_rho(os.path.join(HERE, 'TSDHC_tm_rho_nonarith.csv'),
                       'nonarith')
    spec = {}
    p_spec = os.path.join(HERE, 'TSDHC_binary_spectral.csv')
    if os.path.exists(p_spec):
        with open(p_spec, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                spec[int(r['Ly'])] = float(r['rho_true'])
    spec_non = {}
    p_specn = os.path.join(HERE, 'TSDHC_nonarith_spectral.csv')
    if os.path.exists(p_specn):
        with open(p_specn, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                spec_non[int(r['Ly'])] = float(r['rho_true'])

    # ---- binary: even W uses spectral rho_true; odd W uses converged readings ----
    lam_bin, conv_bin = {}, {}
    for w, d in rho_bin.items():
        if w in spec:
            lam_bin[w] = math.log(spec[w])
            conv_bin[w] = True          # spectral correction counts as trustworthy
        elif d['relerr'] < 1e-10:
            lam_bin[w] = d['ln_rho']
            conv_bin[w] = True
    fit_bin = fit_table('binary honeycomb {0,1} (after rho_true spectral correction)',
                        lam_bin, conv_bin)

    # ---- non-arithmetic ternary {0,1,3}: even W uses spectral rho_true;
    # odd W uses converged readings ----
    # convergence threshold 1e-6: at W=6 the power iteration has
    # relerr = 1.17e-8 with tail width 8e-8, rho stable to 7 digits, no
    # effect on the lambda/W fit (a 1e-10 threshold would misjudge it as
    # unconverged)
    lam_non, conv_non = {}, {}
    lam_non_raw = {}
    for w, d in rho_non.items():
        lam_non_raw[w] = d['ln_rho']
        if w in spec_non:
            lam_non[w] = math.log(spec_non[w])
            conv_non[w] = True
        elif d['relerr'] < 1e-6:
            lam_non[w] = d['ln_rho']
            conv_non[w] = True
    if lam_non:
        fit_non = fit_table(f'nonarith {{0,1,3}} (spectral-corrected + '
                            f'converged W={sorted(lam_non)})', lam_non,
                            conv_non)
    print('\n  nonarith even-W unconverged power readings without spectral '
          'correction (pending eigs):')
    for w, d in rho_non.items():
        if d['relerr'] >= 1e-10 and w not in spec_non:
            print(f'    W={w}: tail=[{d["tail"][0]:.4f},{d["tail"][1]:.4f}]'
                  f'  lambda/W in [{math.log(d["tail"][0])/w:.4f},'
                  f'{math.log(d["tail"][1])/w:.4f}]')

    # ---- arithmetic ternary {0,1,2} (control) ----
    lam_ter, conv_ter = {}, {}
    for w, d in rho_ter.items():
        if d['relerr'] < 1e-6:
            lam_ter[w] = d['ln_rho']
            conv_ter[w] = True
    if len(lam_ter) >= 2:
        fit_ter = fit_table('arithmetic ternary {0,1,2} (control)',
                            lam_ter, conv_ter)

    # ---- summary CSV ----
    out = os.path.join(HERE, 'TSDHC_s_fits_summary.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        wtr = csv.writer(f)
        wtr.writerow(['alphabet', 'W', 'lambda', 's_per_hole', 'converged',
                      'rho_source'])
        for tag, lam, conv, rho_tab, spc in (
                ('binary', lam_bin, conv_bin, rho_bin, spec),
                ('nonarith', {**lam_non, **{w: v for w, v in
                                            lam_non_raw.items()
                                            if w not in lam_non}},
                 conv_non, rho_non, spec_non),
                ('ternary', lam_ter, conv_ter, rho_ter, {})):
            for w in sorted(lam):
                src = ('spectral' if w in spc else
                       ('power_converged' if conv.get(w) else 'power_uncorrected'))
                wtr.writerow([tag, w, f'{lam[w]:.8f}', f'{lam[w]/w:.8f}',
                              'yes' if conv.get(w) else 'no', src])
    print(f'\nsummary CSV -> {out}', flush=True)


if __name__ == '__main__':
    main()
