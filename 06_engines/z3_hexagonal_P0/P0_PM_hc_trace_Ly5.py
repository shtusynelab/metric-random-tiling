# -*- coding: utf-8 -*-
"""P0_PM trace-method Ly=5 family: honeycomb {0,1}/{0,1,2}/{0,1,3} x5 torus
exact counting Tr(T^Lx), Lx=2..20 (2026-09-30; second width-scaling family for
fig2 panel c; Ly=3 + Ly=5 chosen over Ly=2 + Ly=3 to avoid the narrow-width
non-generic effects of Ly=2: parity splitting, doubled-row adjacency, winding
enhancement).

Method: same TorusCtx / column transfer as P0_PM_hc_trace.py (T is
Lx-independent; built once at Lx=3 and reused for all Lx; at Ly=3 this reuse
was validated bit-for-bit against the 7x3 DFS).
Optimization: trace_all runs one forward DP pass while recording all diagonal
elements for k=2..20 (Tr(T^k) = sum_s (T^k)[s,s]); one pass per alphabet,
saving a 19x recomputation.

Anchors (independent code path, DFS cross-check; on failure stop):
  2x5 three alphabets: count_serial
  3x5 three alphabets: count_chunked (6 workers)
  (DFS = the enumerator of P0_PM_torus_growth, a different implementation
  from the trace method)

Memory estimate: state space <= 3^10 = 59049; t0 dict (with out-edge Counters)
estimated 25-400 MB; trace DP per-start vector <= |S| entries < 6 MB; total
peak < 1 GB < 2 GB threshold, workstation run.
Output: P0_PM_torus_trace_shape_classes_Ly5.csv
"""
import csv
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT_CSV = HERE / "P0_PM_torus_trace_shape_classes_Ly5.csv"

ALPHAS = [("binary", (0, 1)), ("arithmetic", (0, 1, 2)),
          ("nonarith", (0, 1, 3))]
LX_MIN, LX_MAX = 2, 20


def trace_all(ctx, n_max):
    """One forward DP pass recording all diagonal elements for k=2..n_max.
    Returns {k: Tr(T^k)}."""
    diag = {k: 0 for k in range(2, n_max + 1)}
    t0, states = ctx.t0, ctx.states
    for idx, s in enumerate(states):
        v = {s: 1}
        for k in range(1, n_max + 1):
            nv = {}
            for st, c in v.items():
                for o, w in t0[st].items():
                    nv[o] = nv.get(o, 0) + c * w
            v = nv
            if k >= 2:
                diag[k] += v.get(s, 0)
        if (idx + 1) % 5000 == 0:
            print(f"    [trace] {idx + 1}/{len(states)} starts, "
                  f"k={n_max}", flush=True)
    return diag


def main():
    import P0_PM_hc_trace as hc
    import P0_PM_torus_growth as tg

    print(f"= honeycomb trace method Ly=5 {time.strftime('%F %T')} =", flush=True)
    rows = []
    for name, alpha in ALPHAS:
        # T is Lx-independent: build once at Lx=3 (the column-seam criterion is
        # inapplicable at Lx=2), then trace any Lx (at Ly=3 this reuse was
        # validated bit-for-bit against the 7x3 DFS)
        ctx = hc.TorusCtx(3, 5, alpha)
        ctx.build_states()
        print(f"  [{name}] T built (Lx=3): |S|={len(ctx.states)}", flush=True)
        # ---- DFS anchors (independent code path) ----
        for Lx in (2, 3):
            t0 = time.time()
            ctx_dfs = tg.Ctx(Lx, 5, alpha)
            if Lx == 2:
                om_dfs = tg.count_serial(ctx_dfs)
            else:
                om_dfs, _nch, timed_out = tg.count_chunked(
                    ctx_dfs, n_workers=6, deadline=3600)
                assert not timed_out and om_dfs is not None,                     f"DFS chunked timeout at {Lx}x5 {name}"
            print(f"  [DFS] {name} {Lx}x5: Omega={om_dfs} "
                  f"({time.time() - t0:.1f}s)", flush=True)
            om_tr = hc.trace_power(ctx.t0, Lx)
            tag = "PASS" if om_tr == om_dfs else "FAIL"
            print(f"  [{tag}] trace {name} {Lx}x5: Tr(T^{Lx})={om_tr} "
                  f"vs DFS {om_dfs}", flush=True)
            assert om_tr == om_dfs, f"anchor mismatch {name} {Lx}x5"
            rows.append(dict(alphabet=name,
                             alphabet_values="".join(str(v) for v in alpha),
                             Lx=Lx, Ly=5, F=Lx * 5, E=ctx.E,
                             method="trace Tr(T^Lx)", Omega=om_tr,
                             lnOmega=round(math.log(om_tr), 6),
                             lnOmega_per_F=round(math.log(om_tr) /
                                                 (Lx * 5), 6),
                             n_states=len(ctx.states),
                             seconds=round(time.time() - t0, 2),
                             status="trace"))
        # ---- extension Lx=4..20 ----
        t0 = time.time()
        diag = trace_all(ctx, LX_MAX)
        print(f"  [{name}] trace_all done in {time.time() - t0:.0f}s",
              flush=True)
        for Lx in range(4, LX_MAX + 1):
            om = diag[Lx]
            row = dict(alphabet=name,
                       alphabet_values="".join(str(v) for v in alpha),
                       Lx=Lx, Ly=5, F=Lx * 5, E=ctx.E,
                       method="trace Tr(T^Lx)", Omega=om,
                       lnOmega=round(math.log(om), 6),
                       lnOmega_per_F=round(math.log(om) / (Lx * 5), 6),
                       n_states=len(ctx.states), seconds=0.0,
                       status="trace")
            rows.append(row)
            print(f"  {name} {Lx}x5: Omega={om} lnOmega={row['lnOmega']} "
                  f"lnOm/F={row['lnOmega_per_F']}", flush=True)
        # growth rate lambda (compare with the A0-2 strip values)
        lx_rows = [r for r in rows if r["alphabet"] == name]
        for a, b in zip(lx_rows, lx_rows[1:]):
            lam = math.log(b["Omega"] / a["Omega"])
            print(f"    ln Omega({b['Lx']}x5)/Omega({a['Lx']}x5) = {lam:.4f}",
                  flush=True)
        with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"done, CSV -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
