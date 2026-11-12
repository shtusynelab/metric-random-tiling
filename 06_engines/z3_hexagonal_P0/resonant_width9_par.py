# -*- coding: utf-8 -*-
"""Parallel version of resonant-family counting: chunk start states -> Pool (fork shares t0).

Verification gate (do not proceed on failure):
  3x3 = 518, 8x2 = 5696, 7x7 = 11662694 (bit-for-bit)
Then 9x9 (expected 22285903280, cross-checked against this machine's single-thread run),
only after passing does production run 18x9 / 27x9 / 36x9.

Memory: parent builds ctx (~0.6 GB), worker forks share via COW, single-worker frontier <0.3 GB,
40 workers < 15 GB total -- far below the 2 GB local-machine threshold, no objection to running on server.
Output: resonant_width9_par_result.csv + DONE marker.
"""
import csv
import math
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from P0_PM_hc_trace import TorusCtx  # noqa: E402

G_T0 = None  # shared by fork after parent builds (COW)


def _trace_chunk(args):
    starts, Lx = args
    t0 = G_T0
    total = 0
    for s in starts:
        v = {s: 1}
        for _ in range(Lx):
            nv = {}
            for st, c in v.items():
                if c == 0:
                    continue
                for o, w in t0[st].items():
                    nv[o] = nv.get(o, 0) + c * w
            v = nv
        total += v.get(s, 0)
    return total


def trace_parallel(ctx, Lx, workers, chunks):
    global G_T0
    G_T0 = ctx.t0
    states = list(ctx.t0.keys())
    n = len(states)
    step = (n + chunks - 1) // chunks
    tasks = [(states[i:i + step], Lx) for i in range(0, n, step)]
    t0 = time.time()
    done = [0]
    with Pool(workers) as pool:
        total = 0
        for k, part in enumerate(pool.imap_unordered(_trace_chunk, tasks)):
            total += part
            done[0] += 1
            if (k + 1) % 40 == 0:
                print(f"    chunks {k+1}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"    Omega = {total}  ({time.time()-t0:.0f}s, {workers}w/{chunks}chunks)", flush=True)
    return total


def run_size(Lx, Ly, workers=40, chunks=320):
    t = time.time()
    ctx = TorusCtx(Lx, Ly, (0, 1))
    ctx.build_states()
    print(f"  {Lx}x{Ly}: |S| = {len(ctx.states)} (build {time.time()-t:.0f}s)", flush=True)
    return trace_parallel(ctx, Lx, workers, chunks)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    log = open(os.path.join(HERE, "par_progress.txt"), "a", buffering=1)
    def say(msg):
        print(msg, flush=True)
        log.write(f"{time.strftime('%m-%d %H:%M:%S')} {msg}\n")

    say(f"=== resonant_width9_par launched, workers={workers} ===")
    # 1) anchor verification gate
    anchors = [((3, 3), 518), ((8, 2), 5696), ((7, 7), 11662694)]
    for (Lx, Ly), expect in anchors:
        om = run_size(Lx, Ly, workers=workers, chunks=64)
        ok = (om == expect)
        say(f"  ANCHOR {Lx}x{Ly}: Omega={om} expect={expect} {'PASS' if ok else 'FAIL'}")
        if not ok:
            say("ABORT: anchor mismatch, production not launched")
            log.write("ABORTED\n")
            return
    # 2) 9x9 cross-check against single-thread run
    om9 = run_size(9, 9, workers=workers, chunks=320)
    ok9 = (om9 == 22285903280)
    say(f"  9x9 = {om9} expect 22285903280 {'PASS' if ok9 else 'FAIL'}")
    if not ok9:
        say("ABORT: 9x9 mismatch")
        log.write("ABORTED\n")
        return
    # 3) production: 18/27/36 x9
    results = []
    for Lx in (18, 27, 36):
        om = run_size(Lx, 9, workers=workers, chunks=320)
        cf = 2 ** (Lx + 10)
        F = Lx * 9
        exc = math.log(om) - (Lx + 10) * math.log(2)
        row = (Lx, om, math.log(om), math.log(om) / F, cf, exc / F)
        results.append(row)
        say(f"  {Lx}x9 [RESONANT] Omega={om} lnOmega={math.log(om):.4f} "
            f"per-tile={math.log(om)/F:.4f} closed={cf} excess/tile={exc/F:.4f}")
    with open(os.path.join(HERE, "resonant_width9_par_result.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Lx", "Ly", "alphabet", "Omega", "lnOmega", "lnOmega_per_F",
                    "closed_form", "excess_per_tile", "method"])
        for Lx, om, ln, per, cf, exc in results:
            w.writerow([Lx, 9, "binary01", om, f"{ln:.4f}", f"{per:.4f}", cf,
                        f"{exc:.4f}", "parallel DP, 40 workers, anchors passed"])
    say("DONE")
    log.write("DONE\n")


if __name__ == "__main__":
    main()
