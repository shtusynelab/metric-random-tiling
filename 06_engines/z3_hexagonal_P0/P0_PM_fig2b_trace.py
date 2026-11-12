# -*- coding: utf-8 -*-
"""P0_PM Fig 2b trace-method data: shape-class map, dense exact torus counting
for three alphabets x Ly=2/3 (2026-09-29; Fig 2b data points obtained with the
trace method, same instrument as Fig 1c).

Four-step trace-method transfer (honeycomb as the primary target; reuses
P0_PM_hc_trace.TorusCtx / trace_power):
  alphabet in {binary (0,1), arithmetic (0,1,2), nonarith (0,1,3)}
  x Ly in {2, 3}  x  Lx = 2..20  ->  Omega_tor(Lx x Ly) = Tr(T^Lx), exact integer.
Display labels use the translation-normalized {1,2}/{1,2,3}/{1,2,4} (the
closure equations are invariant under L->L+1 and Omega is identical digit-wise;
alphabet affine invariance).

Anchor: all 30 rows of the three-alphabet torus-growth-forms series
(P0_PM_torus_growth_forms.csv; 2x2..7x2 and 3x3..6x3) checked bit-for-bit;
on failure nothing is written.

Memory estimate: state space <= 3^(2*Ly) = 729 (Ly=3) / 81 (Ly=2); trace_power
< 1 s per size (hc_trace measured 0.2 s/size); exact Python ints throughout;
peak < 100 MB; total ~1-2 min -- far below the 2 GB threshold on a workstation.

Output: P0_PM_torus_trace_shape_classes_Ly2Ly3.csv
"""
import csv
import math
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from P0_PM_hc_trace import TorusCtx, trace_power  # noqa: E402

# Release layout: this engine directory IS the data/output root.
ROOT = HERE
GEO = ROOT
OUT_CSV = GEO / "P0_PM_torus_trace_shape_classes_Ly2Ly3.csv"

ALPHABETS = {"binary": (0, 1), "arithmetic": (0, 1, 2),
             "nonarith": (0, 1, 3)}
VALUES = {"binary": "01", "arithmetic": "012", "nonarith": "013"}
LX_RANGE = range(2, 21)          # Ly=3 starts at Lx=2 (no 2x3 anchor; produced anyway)
BUILD_LX = {2: 3, 3: 3}          # T is Lx-independent; build once with Lx=3


def main():
    print(f"= Fig2b trace-method shape classes {time.strftime('%F %T')} =", flush=True)
    ref = pd.read_csv(GEO / "P0_PM_torus_growth_forms.csv")
    ctxs = {}
    rows = []
    n_anchor = n_pass = 0
    for Ly in (2, 3):
        for name, alpha in ALPHABETS.items():
            key = (Ly, alpha)
            if key not in ctxs:
                ctxs[key] = TorusCtx(BUILD_LX[Ly], Ly, alpha)
                ctxs[key].build_states()
                print(f"  built {name} Ly={Ly}: |S|={len(ctxs[key].states)} "
                      f"E={ctxs[key].E}", flush=True)
            ctx = ctxs[key]
            for Lx in LX_RANGE:
                om = trace_power(ctx.t0, Lx)
                is_anchor = False
                m = ref[(ref.alphabet == name) & (ref.Lx == Lx) &
                        (ref.Ly == Ly)]
                if len(m) == 1:
                    n_anchor += 1
                    is_anchor = True
                    ok = int(m.Omega.iloc[0]) == om
                    n_pass += ok
                    tag = "PASS" if ok else "FAIL"
                    print(f"  [anchor {tag}] {name} {Lx}x{Ly}: trace={om} "
                          f"forms-series={int(m.Omega.iloc[0])}", flush=True)
                    assert ok, f"anchor mismatch {name} {Lx}x{Ly}"
                rows.append(dict(
                    alphabet=name, alphabet_values=VALUES[name], Lx=Lx, Ly=Ly,
                    F=Lx * Ly, E=ctx.E, method="trace Tr(T^Lx)", Omega=om,
                    lnOmega=round(math.log(om), 6),
                    lnOmega_per_F=round(math.log(om) / (Lx * Ly), 6),
                    n_states=len(ctx.states), is_anchor=is_anchor,
                    seconds=None, status="trace"))
    print(f"anchors {n_pass}/{n_anchor} PASS (all forms-series rows bit-for-bit)", flush=True)

    # supplementary timing (a full recomputation is too costly; record one
    # representative measured magnitude: the first production case)
    t0 = time.time()
    trace_power(ctxs[(3, (0, 1, 2))].t0, 20)
    per20 = time.time() - t0
    for r in rows:
        r["seconds"] = round(per20, 2) if (r["Ly"] == 3 and r["Lx"] == 20
                                           and r["alphabet"] == "arithmetic") \
            else None

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"done, {len(rows)} rows -> {OUT_CSV}", flush=True)

    # reading: growth rate (adjacent ratio) of the three series and decay of
    # the parity oscillation
    df = pd.DataFrame(rows)
    for Ly in (2, 3):
        for name in ALPHABETS:
            d = df[(df.alphabet == name) & (df.Ly == Ly)].sort_values("Lx")
            ln = d.lnOmega.values
            lam_even = ln[-2] - ln[-4]      # end-of-series growth over two steps (same parity)
            wig = [abs(ln[k + 1] - ln[k] - (ln[k] - ln[k - 1])) / 2
                   for k in range(1, len(ln) - 1)]
            print(f"  Ly={Ly} {name:10s} lnOm(Lx={d.Lx.iloc[0]}.."
                  f"{d.Lx.iloc[-1]}): {ln[0]:.2f}->{ln[-1]:.2f} "
                  f"end slope/step={lam_even / 2:.3f} "
                  f"oscillation amp first={wig[0]:.3f} mid={wig[len(wig) // 2]:.3f} "
                  f"last={wig[-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
