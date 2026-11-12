# -*- coding: utf-8 -*-
"""collect_w5.py - aggregate the W=5 adjudication: merge the 8-seed probes,
final table, and the ln g growth-shape fit."""
import glob
import math
import numpy as np

Z20 = 3 ** 20
LN3 = math.log(3)


def load_probes(pattern, required=True):
    vals = []
    files = sorted(glob.glob(pattern))
    if not files:
        if required:
            raise SystemExit(f"no files: {pattern}")
        print(f"  (no files, skipping: {pattern})")
        return np.array([0.0])
    for f in files:
        v = np.loadtxt(f, skiprows=1)
        vals.append(v)
        print(f"  {f}: n={len(v)} mean={v.mean():.6g} "
              f"se={v.std(ddof=1)/math.sqrt(len(v)):.4g}")
    return np.concatenate(vals)


print("== W=5 int rehearsal (anchor 3^20) ==")
pi = load_probes("server_logs_extract/hutch_W5_int_s2*.csv")
mi, si = pi.mean(), pi.std(ddof=1) / math.sqrt(len(pi))
print(f"int: n={len(pi)} mean={mi:.10g} se={si:.4g}  anchor={Z20} "
      f"deviation={abs(mi-Z20):.4g} ({abs(mi-Z20)/Z20*100:.5f}%)")

print("== W=5 exp ==")
pe = load_probes("server_logs_extract/hutch_W5_exp_s1*.csv", required=False)
me, se = pe.mean(), pe.std(ddof=1) / math.sqrt(len(pe))
ri = round(me)
print(f"exp: n={len(pe)} mean={me:.10g} se={se:.4g} ({se/me*100:.4f}%)")
print(f"    integer snap {ri} (deviation {abs(me-ri):.4g}, {abs(me-ri)/me*100:.4f}%)")
lnO = math.log(me)
s5 = lnO / 50
g5 = me / Z20
ds5 = s5 - 2 * LN3 / 5
print(f"    lnOmega(5) = {lnO:.6f}; s(5) = {s5:.6f}; g(5) = {g5:.6f}; "
      f"ds(5) = {ds5:.6f}")

print("== adjudication table (chain) ==")
W = [2, 3, 4, 5]
Om = [6641, 563553, 52864817, me]
for w, om in zip(W, Om):
    s = math.log(om) / (2 * w * w)
    g = om / 3 ** (4 * w)
    ds = s - 2 * LN3 / w
    print(f"  W={w}: Omega={om:.10g} s={s:.6f} 2ln3/W={2*LN3/w:.6f} "
          f"ds={ds:.6f} g={g:.6f} ln g={math.log(g):.6f} "
          f"ln g/W={math.log(g)/w:.6f}")

lg = [math.log(om / 3 ** (4 * w)) for w, om in zip(W, Om)]
print("== ln g growth shape ==")
print("  ln g:", [f"{x:.4f}" for x in lg])
print("  successive ratios:", [f"{lg[i+1]/lg[i]:.2f}" for i in range(len(lg)-1)])
print("  ln g/W:", [f"{x/w:.4f}" for x, w in zip(lg, W)])
# ln g = a W + b fit (W=3..5)
A = np.vstack([W[1:], np.ones(3)]).T
coef, res, *_ = np.linalg.lstsq(A, np.array(lg[1:]), rcond=None)
pred6 = coef[0] * 6 + coef[1]
print(f"  linear fit (W=3..5): a={coef[0]:.4f} b={coef[1]:.4f} "
      f"=> ln g(6)~={pred6:.3f} (linear => s_Lambda->0)")
# comparison against a W^p law: pointwise exponent
for i in range(1, len(lg)):
    p = math.log(lg[i] / lg[i-1]) / math.log(W[i] / W[i-1])
    print(f"  effective exponent p(W={W[i-1]}->{W[i]}) = {p:.2f}", end="")
    print()
s_inf_upper = coef[0] / 2
print(f"  if ln g = aW + O(1) asymptotically: s_Lambda(W) = 2ln3/W + a/(2W) "
      f"+ O(1/W^2) => W=6 predicted s ~= {2*LN3/6 + coef[0]/12:.4f}")
