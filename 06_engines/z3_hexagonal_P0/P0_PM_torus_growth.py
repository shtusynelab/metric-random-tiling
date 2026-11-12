# -*- coding: utf-8 -*-
"""P0_PM A0-1: pure-metric-layer torus growth-form series (main evidence for
Fig 2b).

Three alphabets: two-value {0,1} (perimeter-law form) / arithmetic {0,1,2}
(positive-entropy direction) / non-AP {0,1,3} (strictly fewer); sizes
2x2-7x2 and 3x3-6x3. Reuses edge_alphabet_affine_check.build_honeycomb_torus
(validated torus construction); counter = the same DFS (edge-by-edge
assignment + per-face closure-equation pruning) plus prefix-blocked parallelism
(serial arithmetic large sizes were projected beyond 30 min; blocked, each
case < 30 min).

Anchors (must pass; bit-for-bit against the serial original counter):
  binary  (2x2, 3x2, 3x3) = 64 / 64 / 518
  arith   (2x2, 3x2, 3x3, 4x3) = 729 / 1017 / 22767 / 30321   [alphabet (4,5,6)]
  nonarith (3x3) = 19707
  extra: {0,1,2} and {4,5,6} equal bit-for-bit at (2x2, 3x2, 3x3) (arithmetic-
         class affine invariance; the basis for labeling {0,1,2} on the graph).

Output: P0_PM_torus_growth_forms.csv (all columns per size)
Memory: per-worker DFS <10 MB (no materialized state table / solution set),
8 workers <100 MB, workstation run.
"""
import csv
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from edge_alphabet_affine_check import build_honeycomb_torus  # noqa: E402

OUT_CSV = HERE / "P0_PM_torus_growth_forms.csv"

WORKERS = 6
CASE_DEADLINE = 1800.0          # stop after 30 min per case
ALPHABETS = [("binary", (0, 1)), ("arithmetic", (0, 1, 2)), ("nonarith", (0, 1, 3))]
SIZES = [(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (3, 3), (4, 3), (5, 3), (6, 3)]


class Ctx:
    """Torus counting context (order / triggers construction word-for-word the
    same as affine_check.count_assignments, only extracted into a reusable
    object)."""
    __slots__ = ("Lx", "Ly", "E", "alpha", "order", "triggers")

    def __init__(self, Lx, Ly, alphabet):
        edge_set, faces = build_honeycomb_torus(Lx, Ly)
        E = len(edge_set)
        order, seen = [], set()
        for f in faces:
            for e in f:
                if e not in seen:
                    seen.add(e)
                    order.append(e)
        assert len(order) == E
        pos_in_order = {e: k for k, e in enumerate(order)}
        eqA = [(f[1], f[2], f[4], f[5]) for f in faces]
        eqB = [(f[0], f[1], f[5], f[3], f[2], f[4]) for f in faces]
        triggers = {}
        for a, b, c, d in eqA:
            terms = tuple(pos_in_order[e] for e in (a, b, c, d))
            triggers.setdefault(max(terms), []).append(("A", terms))
        for e0, e1, e5, e3, e2, e4 in eqB:
            terms = tuple(pos_in_order[e] for e in (e0, e1, e5, e3, e2, e4))
            triggers.setdefault(max(terms), []).append(("B", terms))
        self.Lx, self.Ly, self.E = Lx, Ly, E
        self.alpha = list(alphabet)
        self.order, self.triggers = order, triggers


def _trigger_ok(typ, terms, vals):
    if typ == "A":
        a, b, c, d = terms
        return vals[a] + vals[b] == vals[c] + vals[d]
    e0, e1, e5, e3, e2, e4 = terms
    return 2 * vals[e0] + vals[e1] + vals[e5] == 2 * vals[e3] + vals[e2] + vals[e4]


def count_serial(ctx):
    """Serial DFS identical to the original version (anchor cross-check)."""
    vals = [0] * ctx.E
    cnt = 0

    def dfs(k):
        nonlocal cnt
        if k == ctx.E:
            cnt += 1
            return
        for w in ctx.alpha:
            vals[k] = w
            if all(_trigger_ok(t, m, vals) for t, m in ctx.triggers.get(k, ())):
                dfs(k + 1)

    dfs(0)
    return cnt


def enum_prefixes(ctx, k):
    """All feasible prefixes of the first k edges (check logic word-for-word
    the same as the full DFS)."""
    out = []
    vals = [0] * ctx.E

    def rec(i):
        if i == k:
            out.append(tuple(vals[:k]))
            return
        for w in ctx.alpha:
            vals[i] = w
            if all(_trigger_ok(t, m, vals) for t, m in ctx.triggers.get(i, ())):
                rec(i + 1)

    rec(0)
    return out


def count_chunk(args):
    """Worker: given a prefix, complete the remaining DFS counting."""
    ctx, prefix = args
    vals = list(prefix) + [0] * (ctx.E - len(prefix))
    cnt = 0

    def dfs(k):
        nonlocal cnt
        if k == ctx.E:
            cnt += 1
            return
        for w in ctx.alpha:
            vals[k] = w
            if all(_trigger_ok(t, m, vals) for t, m in ctx.triggers.get(k, ())):
                dfs(k + 1)

    dfs(len(prefix))
    return cnt


def count_chunked(ctx, n_workers, deadline, k_min=8, chunk_timeout=30.0):
    """Prefix-blocked parallel counting; stops if a case exceeds deadline.
    Returns (Omega|None, n_chunks, timed_out).
    chunk_timeout = upper bound for waiting on a single prefix block (a large
    tree can exceed 30 s on one block; increase if needed)."""
    k = k_min
    while k < ctx.E - 2:
        prefs = enum_prefixes(ctx, k)
        if len(prefs) >= 3 * n_workers:
            break
        k += 1
    total = 0
    t0 = time.time()
    pool = mp.Pool(n_workers)
    try:
        it = pool.imap_unordered(count_chunk, [(ctx, p) for p in prefs],
                                 chunksize=1)
        while True:
            try:
                c = it.next(timeout=chunk_timeout)
            except StopIteration:
                break
            total += c
            if time.time() - t0 > deadline:
                raise TimeoutError
    except TimeoutError:
        pool.terminate()
        pool.join()
        return None, len(prefs), True
    pool.close()
    pool.join()
    return total, len(prefs), False


def run_case(alpha_name, alphabet, Lx, Ly, known=None):
    """Single case: use the known anchor value if available, otherwise timed
    prefix-blocked parallel counting. Returns a CSV row dict."""
    ctx = Ctx(Lx, Ly, alphabet)
    F = Lx * Ly
    row = dict(alphabet=alpha_name, alphabet_values="".join(map(str, alphabet)),
               Lx=Lx, Ly=Ly, F=F, E=ctx.E)
    t0 = time.time()
    if known is not None:
        omega, n_chunks, status = known, 1, "anchor"
    else:
        omega, n_chunks, _ = count_chunked(ctx, WORKERS, CASE_DEADLINE)
        status = "chunked"
        if omega is None:
            row.update(Omega="", lnOmega="", lnOmega_per_F="",
                       seconds=round(time.time() - t0, 1),
                       n_chunks=n_chunks, status="dnf_timeout")
            print(f"  {alpha_name} {alphabet} {Lx}x{Ly}: DNF "
                  f"(>{CASE_DEADLINE:.0f}s, stopped per protocol)", flush=True)
            return row
    row.update(Omega=omega, lnOmega=round(math.log(omega), 6),
               lnOmega_per_F=round(math.log(omega) / F, 6),
               seconds=round(time.time() - t0, 1),
               n_chunks=n_chunks, status=status)
    print(f"  {alpha_name} {alphabet} {Lx}x{Ly}: Omega={omega} "
          f"({row['seconds']}s, {status}, {n_chunks} chunks)", flush=True)
    return row


ANCHORS = {
    ("binary", (2, 2)): 64,
    ("binary", (3, 2)): 64,
    ("binary", (3, 3)): 518,
    ("arithmetic", (2, 2)): 729,
    ("arithmetic", (3, 2)): 1017,
    ("arithmetic", (3, 3)): 22767,
    ("arithmetic", (4, 3)): 30321,
    ("nonarith", (3, 3)): 19707,
}


def anchor_check():
    """Anchor cross-check: serial original counter vs known values; affine
    class (0,1,2) vs (4,5,6) bit-for-bit."""
    print("= anchor cross-check (serial original counter) = ", flush=True)
    ctx_cache = {}
    for (name, (Lx, Ly)), expect in ANCHORS.items():
        alpha = dict(ALPHABETS)[name]
        ctx = ctx_cache.setdefault((name, Lx, Ly), Ctx(Lx, Ly, alpha))
        got = count_serial(ctx)
        tag = "PASS" if got == expect else "FAIL"
        print(f"  [{tag}] {name} {Lx}x{Ly}: got={got} expect={expect}", flush=True)
        assert got == expect, f"anchor mismatch {name} {Lx}x{Ly}"
    # arithmetic-class affine: (0,1,2) vs (4,5,6) at three sizes, bit-for-bit
    for (Lx, Ly) in [(2, 2), (3, 2), (3, 3)]:
        a = count_serial(Ctx(Lx, Ly, (0, 1, 2)))
        b = count_serial(Ctx(Lx, Ly, (4, 5, 6)))
        tag = "PASS" if a == b else "FAIL"
        print(f"  [{tag}] affine (0,1,2)={a} vs (4,5,6)={b} at {Lx}x{Ly}", flush=True)
        assert a == b
    print("all anchors PASS", flush=True)


def main():
    print(f"= P0_PM A0-1 torus growth forms {time.strftime('%m-%d %H:%M:%S')} "
          f"(workers={WORKERS})=", flush=True)
    anchor_check()
    rows = []
    for name, alpha in ALPHABETS:
        for (Lx, Ly) in SIZES:
            known = ANCHORS.get((name, (Lx, Ly)))
            rows.append(run_case(name, alpha, Lx, Ly, known=known))
            with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print(f"done, CSV -> {OUT_CSV}", flush=True)
    # summary table (stdout record)
    print(f"\n{'alphabet':>10} | " + " | ".join(f"{s[0]}x{s[1]}" for s in SIZES))
    for name, _ in ALPHABETS:
        cells = [f"{r['Omega']}" for r in rows if r["alphabet"] == name]
        print(f"{name:>10} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
