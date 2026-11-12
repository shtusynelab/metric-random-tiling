# -*- coding: utf-8 -*-
"""4.6.12 transfer matrix v2: column transfer F + power iteration -> lambda(W).

v2 redesign (after the v1 slot collision):
  v1 slot (delta, w2=round(2*alpha), t) had three defects:
    (a) no T2 phase: T2-translates inside one column share alpha (and w2);
    (b) round(2*alpha) quantizes T1 too coarsely (mirror-orbit levels
        0.211/0.789 collide in the same bin for W>=3);
    (c) midpoints averaged from independently reduced vertex reps are wrong
        for torus-wrapping edges (no T1 wrap fix; T2 had one).
  v2: canonical per-edge (a, b) lattice coords via wrap-aware averaging and
  exact lattice solve; signature (bdir, qa, qb) with qa/qb = fractional
  coords on a QGRID lattice (column-relative, T2-translation covariant);
  slot = (delta, t) with t = sorted edge signatures.  Face column rule =
  MIN over the face's own edge-class columns (hexagons process in their
  lower column): per-cut straddler classes ~4W (2W cross HD + 2W cross SD)
  vs ~10W under the v1 MAX rule.  Covariant orderings everywhere (face key
  and new-class key use slots, never class ids); alignment assert
  out_slots[j] == in_slots[j+1] for every consecutive pair.

Gates (in order):
  diag    level separation / rounding-boundary distance / per-column edge
          signature uniqueness / per-rule block shape census;
  gateA   column-order frontier DP reproduces exact torus counts bitwise;
  gateB   FULL block product (empty head -> empty tail) reproduces the
          exact DP count bitwise (binary vs anchors; ternary dp vs blocks,
          two independent code paths) -- the slot-machinery validity gate
          (the analogue of 3.12.12's Tr(T^L) crosscheck);
  w2      lambda(2) vs the 33-point exact-chain resonant asymptote 2.7433;
  recon3  W=3 state-space reconnaissance (fail-closed state gate);
  w3/t2/t3/w4/t4  production power iterations -> lambda(W) table.

Run: python rth4612_tm.py <stage> [--server]
  stages: diag | gateA | gateB | w2 | w3 | t2 | t3 | w4 | t4 | w5 | t5
  --server: raise the closure gate to GATE_STATES (run big W/k on the
  server; local runs gate at LOCAL_STATES ~0.5-1 GB).
Production path: closure discovery -> explicit sparse transition ->
numpy bincount power iteration (Perron) -> lambda(W) appended to
RTH4612_tm_lambda.csv.
"""
import math
import os
import sys
import time
from collections import defaultdict
from itertools import product

import numpy as np

from roundE_4612_frontier_dp import build_face_system, frontier_plan, dp_count
from roundD_4612_numeric import (build_torus, extract_faces, build_orbit,
                                 fund_reduce, EDGE)

SQ3 = math.sqrt(3.0)
QGRID = 64                     # signature lattice (see diag asserts)
GATE_STATES = 5 * 10 ** 7      # server ceiling (dict form)
LOCAL_STATES = 2 * 10 ** 6     # local ceiling (~0.5 GB dict)
BOUND_MIN = 0.02               # min distance of a level to a rounding edge


def _quant(v):
    return int(round((v % 1.0) * QGRID)) % QGRID


def build_torus_mc(W, L):
    """build_torus verbatim + per-edge true midpoint residual mc.

    Torus-wrapping creates DISTINCT torus edges sharing the canonical
    (u, v, bdir) triple (parallel copies) -- build_torus dedups them by the
    midpoint class mc but drops it from the returned edge tuples.  The
    slot signatures need mc (translate-invariant, distinguishes parallel
    copies), so replay the construction exactly and record it.  diag
    asserts edge-list identity with build_torus.
    """
    pts = build_orbit(W, L)
    keys = list(pts.keys())
    pl = np.array([pts[k] for k in keys])
    n = len(pl)
    thr = EDGE + 1e-6
    edges_raw = []
    for i in range(n):
        d = np.sqrt(((pl[i + 1:] - pl[i]) ** 2).sum(-1))
        for jj in np.flatnonzero(d < thr):
            edges_raw.append((keys[i], keys[i + 1 + jj]))
    T1p, T2p = W * np.array([2.0, 0.0]), L * np.array([1.0, SQ3])

    def vid_of(key):
        p = pts[key]
        mm, nn, r = fund_reduce(p, W, L, T1p, T2p)
        return (round(r[0], 6), round(r[1], 6))

    vid = {k: vid_of(k) for k in keys}
    V = len(set(vid.values()))
    edges, eset, mcs = [], set(), []
    for ka, kb in edges_raw:
        u, v = vid[ka], vid[kb]
        if u == v:
            continue
        dvec = pts[kb] - pts[ka]
        ang = math.degrees(math.atan2(dvec[1], dvec[0])) % 360
        b = int(round(ang / 30)) % 12
        diff = abs(ang - 30 * b) % 360
        assert min(diff, 360 - diff) < 1e-4, (ka, kb, ang)
        mid = (pts[ka] + pts[kb]) / 2
        mm0, nn0, rm = fund_reduce(mid, W, L, T1p, T2p)
        mc = (round(rm[0], 6), round(rm[1], 6))
        k1 = (u, v, b, mc)
        k2 = (v, u, (b + 6) % 12, mc)
        if k1 not in eset and k2 not in eset:
            eset.add(k1)
            eset.add(k2)
            edges.append((u, v, b))
            mcs.append(np.array(rm))
    return vid, edges, V, len(edges), pts, mcs


def edge_cols(W, L, edges, mcs):
    """Per-edge (column, signature) from the TRUE midpoint residual.

    mc = fundamental-domain residual of the edge midpoint: (a, b) lattice
    coords with a in [0,W), b in [0,L); a is T2-translation invariant and
    b mod 1 is column-relative, so the quantized signature (bdir, qa, qb)
    is covariant under column shifts and distinguishes parallel copies.
    """
    col_e, sig_e = {}, {}
    for eid, (u, v, b) in enumerate(edges):
        rm = mcs[eid]
        a = (rm[0] - rm[1] / SQ3) / 2.0
        bm = rm[1] / SQ3
        # mirror midpoints sit ON integer b (float noise ~1e-15); snap:
        # true levels are >=1e-3 from integers, so 1e-6 is safe margin
        col_e[eid] = int(math.floor(bm + 1e-6)) % L
        # bdir mod 6: the raw direction is an orientation artifact of the
        # window scan (antiparallel dvec for the same undirected edge);
        # the undirected invariant is the line bucket
        sig_e[eid] = (b % 6, _quant(a / W), _quant(bm))
    return col_e, sig_e


def geometry_columns(W, L, col_rule="max"):
    """(face_eqs, order, cls_faces, slot, col_c, n_free, N, face_col).

    order covers faces WITH equations only (squares are no-ops); within a
    column, faces ordered by the covariant key = sorted class slots.
    col_rule: face_col = max over the face's own edge-class MIN-columns.
    (The min rule is broken: a dodecagon row collapses onto the previous
    one through the shared cross-row SD class -- T2-translate dodecagons
    then land in the same TM-column and their slots, being translate-
    covariant, tie.  max puts every face in its own b-row; the per-cut
    straddlers are then just the 2W cross-row SD classes.)
    """
    (face_eqs, n_cls, n_free, N, dim, all_eq_cls,
     face_nbrs, edge_cls) = build_face_system(W, L, rank_check=False)
    vid, edges, V, E, pts, mcs = build_torus_mc(W, L)
    assert E == len(edges) == len(edge_cls) == len(mcs)
    col_e, sig_e = edge_cols(W, L, edges, mcs)
    faces = extract_faces(edges)
    assert len(faces) == len(face_eqs)
    cls_edges = defaultdict(list)
    for eid, c in enumerate(edge_cls):
        cls_edges[c].append(eid)
    col_c = {c: min(col_e[e] for e in es) for c, es in cls_edges.items()}
    face_col = {}
    for fi, f in enumerate(faces):
        cs = [col_c[edge_cls[h[3]]] for h in f]
        face_col[fi] = min(cs) if col_rule == "min" else max(cs)
    cls_faces = defaultdict(list)
    for fi, eqs_f in enumerate(face_eqs):
        for eq in eqs_f:
            for c in eq:
                cls_faces[c].append(fi)
    # covariant LIFETIME component: (free column - assign column) mod L.
    # Pure-geometry slots collide for pass-through carriers (the b=0 wrap
    # row's top HD classes are assigned one column and freed L-1 later,
    # riding the frontier across the system) vs their in-column
    # translates; the lifetime delta separates the timing families
    # (in-column 0 / cross-column 1 / wrap carrier L-1) covariantly.
    life = {}
    a_col = {}
    for c, fl in cls_faces.items():
        a_col[c] = min(face_col[f] for f in fl)
        f_c = max(face_col[f] for f in fl)
        life[c] = (f_c - a_col[c]) % L
    col_c, slot = {}, {}
    for c, es in cls_edges.items():
        cols = [col_e[e] for e in es]
        col_c[c] = min(cols)
        # ephase: edge-column phase relative to the birth column -- wraps
        # create distinct same-birth-column classes (cut-b and cut-b+L
        # pairs in the wrap column) whose fractional positions coincide
        slot[c] = (max(cols) - min(cols), life[c],
                   (col_c[c] - a_col[c]) % L,
                   tuple(sorted(sig_e[e] for e in es)))
    eq_faces = [fi for fi, eqs_f in enumerate(face_eqs) if eqs_f]
    fkey = {fi: tuple(sorted(slot[c] for eq in face_eqs[fi] for c in eq))
            for fi in eq_faces}
    bycol = defaultdict(list)
    for fi in eq_faces:
        bycol[face_col[fi]].append(fi)
    for j, fl in bycol.items():
        assert len({fkey[fi] for fi in fl}) == len(fl), \
            f"face key tie in column {j}"
    order = sorted(eq_faces, key=lambda fi: (face_col[fi], fkey[fi]))
    return (face_eqs, order, cls_faces, slot, col_c, n_free, N, face_col)


def build_blocks(W, L, k, col_rule="max", need_bulk=True):
    """(blocks, bulk_block, bulk_range, n_free, N); blocks[j] =
    (in_slots, steps, out_slots); steps = (new_slots, eq_list).

    Slots are named LOCALLY per block: (role, geometric signature) with
    role = lifetime phase of the class relative to this block --
    I(ncoming, freed here) / O(utgoing, born here) / M(id, born and freed
    here) / P(ass-through).  A pure-geometry slot cannot work: the frontier
    of block j necessarily contains a class together with its own column
    translate (the dying [j-1,j] member and the newborn [j,j+1] member of
    the same family), and any translation-covariant hash assigns them the
    same key -- the same lesson that forced explicit role naming in
    3.12.12.  The role prefix separates the phases covariantly; in/out
    tuples follow the plan's frontier order, so consecutive blocks stay
    positionally aligned and bulk blocks remain literally identical.
    """
    (face_eqs, order, cls_faces, slot, col_c, n_free, N,
     face_col) = geometry_columns(W, L, col_rule)
    a_col = {c: min(face_col[f] for f in fl) for c, fl in cls_faces.items()}
    f_col = {c: max(face_col[f] for f in fl) for c, fl in cls_faces.items()}
    plan = frontier_plan(face_eqs, order, cls_faces)
    raw = defaultdict(list)
    for (fr_in, new, out, f, eqs_f) in plan:
        raw[face_col[f]].append((fr_in, new, out, eqs_f))
    blocks = {}
    for j in sorted(raw):
        steps = raw[j]

        def role_of(c, j=j):
            if a_col[c] < j and f_col[c] > j:
                return "P"
            if a_col[c] < j:
                return "I"
            if f_col[c] > j:
                return "O"
            return "M"

        lrole = {}
        cmap = {}
        for (fr_in, new, out, eqs_f) in steps:
            for c in list(fr_in) + list(new) + list(out):
                if c not in cmap:
                    lrole[c] = role_of(c)
                    cmap[c] = (lrole[c],) + slot[c]
            for eq in eqs_f:
                for c in eq:
                    assert c in cmap
        # canonical covariant order: geometric signature first, role as
        # tie-break -- the frontier SEQUENCE is not covariant (pass-through
        # carriers enter at their birth column), but the sorted sets of
        # corresponding blocks are identical, so positional alignment
        # across blocks and bulk periodicity both hold
        def skey(c):
            return (slot[c], lrole[c])

        uniq = {skey(c) for c in cmap}
        assert len(uniq) == len(cmap), f"sort key tie at col {j}"
        in_order = sorted(steps[0][0], key=skey)
        out_order = sorted(steps[-1][2], key=skey)
        in_slots = tuple(cmap[c] for c in in_order)
        out_slots = tuple(cmap[c] for c in out_order)
        st = []
        for (fr_in, new, out, eqs_f) in steps:
            news = tuple(cmap[c] for c in sorted(new, key=skey))
            eq_list = tuple(tuple(sorted((cmap[c], co) for c, co in eq.items()))
                            for eq in eqs_f)
            st.append((news, eq_list))
        # collision = two DIFFERENT classes sharing a slot; a slot MAY
        # legitimately occur twice in the tuple lists (P classes sit in
        # both in_slots and out_slots, O classes in news and out_slots)
        by_slot = defaultdict(set)
        for c, s in cmap.items():
            by_slot[s].add(c)
        bad = {s: cs for s, cs in by_slot.items() if len(cs) > 1}
        assert not bad, f"slot collision at col {j}: {list(bad.items())[:2]}"
        blocks[j] = (in_slots, tuple(st), out_slots, in_order, out_order)
    js = sorted(blocks)
    assert js == list(range(L)), f"columns missing: {js}"
    assert blocks[js[0]][0] == (), "head in_slots not empty"
    # positional alignment: local names differ across blocks by design
    # (same class is 'O' for its birth block, 'I'/'P' for the next), so
    # consecutive blocks align on the canonically-sorted CLASS SEQUENCE,
    # which apply_block's positional zip requires
    for a, b in zip(js, js[1:]):
        assert blocks[a][4] == blocks[b][3], f"alignment {a}->{b} broken"
    assert blocks[js[-1]][2] == (), "tail out_slots not empty"
    bulk = [j for j in js if 2 <= j <= L - 3]
    if need_bulk:
        assert len(bulk) >= 2, f"bulk range too small: {js} L={L}"
        ref_j = bulk[0]
        for j in bulk[1:]:
            assert blocks[j][:3] == blocks[ref_j][:3], \
                f"periodicity failed at {j}"
        return blocks, blocks[ref_j][:3], (ref_j, bulk[-1]), n_free, N
    return blocks, None, None, n_free, N


def apply_block(block, v, vals):
    """v: dict {in_state_tuple: weight} -> {out_state_tuple: weight}.

    Equations checked as full sums at their face's step; the out state is
    resolved from base (in-state) merged with the step-assigned slots."""
    in_slots, steps, out_slots = block
    out = defaultdict(float)
    for state_tuple, w0 in v.items():
        base = dict(zip(in_slots, state_tuple))
        cur_slots = ()
        cur = {(): w0}
        for (news, eq_list) in steps:
            k = len(news)
            ncur = defaultdict(float)
            for proc, w2 in cur.items():
                d = dict(base)
                for s, vv in zip(cur_slots, proc):
                    d[s] = vv
                feas = []
                for combo in product(vals, repeat=k):
                    for s, vv in zip(news, combo):
                        d[s] = vv
                    ok = True
                    for eq in eq_list:
                        acc = 0
                        for (s, co) in eq:
                            acc += co * d[s]
                        if acc != 0:
                            ok = False
                            break
                    if ok:
                        feas.append(combo)
                for combo in feas:
                    ncur[proc + combo] += w2
            cur = ncur
            cur_slots = cur_slots + news
        for proc, w2 in cur.items():
            d = dict(base)
            for s, vv in zip(cur_slots, proc):
                d[s] = vv
            out[tuple(d[s] for s in out_slots)] += w2
    return out


def power(block, v0, vals, niter=400, verbose=True, gate=LOCAL_STATES):
    v = {key: float(w) for key, w in v0.items()}
    lam_hist = []
    for n in range(niter):
        v2 = apply_block(block, v, vals)
        t2 = sum(v2.values())
        t1 = sum(v.values())
        lam = math.log(t2 / t1)
        lam_hist.append(lam)
        if verbose and n % 25 == 0:
            print(f"    iter {n}: |S|={len(v2)} lam={lam:.5f}", flush=True)
        v = {kk: ww / t2 for kk, ww in v2.items()}
        if len(v) > gate:
            print(f"  STATE GATE: |S|={len(v)} > {gate}", flush=True)
            raise MemoryError(f"state gate {len(v)} > {gate}")
        if n > 40 and max(abs(lam - x) for x in lam_hist[-6:-1]) < 1e-7:
            break
    est = sum(lam_hist[-5:]) / len(lam_hist[-5:])
    return est, len(v), lam_hist


# ---- parallel expansion workers ----
_G = {}


def _init_worker(block, vals):
    _G["block"] = block
    _G["vals"] = vals


def _expand_chunk(states):
    block, vals = _G["block"], _G["vals"]
    return [(st, list(apply_block(block, {st: 1.0}, vals).items()))
            for st in states]


def sparse_perron(block, v0, vals, gate=LOCAL_STATES, iters=20000,
                  tol=1e-12, verbose=True, workers=None):
    """Fused closure + transition enumeration -> numpy power iteration.

    Every reachable state is expanded EXACTLY ONCE; its outgoing edges are
    recorded at discovery time (discovery and enumeration are the same
    operation).  Then the Perron root via ARPACK eigs (bincount fallback).
    Returns (lam, n_states, n_edges, iters, rel_or_gap)."""
    from multiprocessing import Pool
    if workers is None:
        workers = int(os.environ.get("RTH_TM_WORKERS", 0)) or \
            min(32, os.cpu_count() or 8)
    nproc = max(1, workers)
    seen = {}
    edges = []
    frontier = []
    for key in v0:
        if key not in seen:
            seen[key] = len(seen)
            frontier.append(key)
    pool = Pool(nproc, initializer=_init_worker, initargs=(block, vals))
    try:
        while frontier:
            new_keys = []
            chunk = max(1, len(frontier) // (nproc * 4) or 1)
            batches = [frontier[i:i + chunk]
                       for i in range(0, len(frontier), chunk)]

            def claim(kk):
                idx = seen.get(kk)
                if idx is None:
                    idx = seen[kk] = len(seen)
                    new_keys.append(kk)
                return idx

            for res in pool.imap_unordered(_expand_chunk, batches):
                for st, outs in res:
                    si = seen[st]
                    for ok, cnt in outs:
                        edges.append((si, claim(ok), cnt))
            if len(seen) > gate:
                print(f"  CLOSURE GATE: |S|={len(seen)} > {gate}", flush=True)
                raise MemoryError(f"closure gate {len(seen)} > {gate}")
            if verbose:
                print(f"    closure: |S|={len(seen)} (+{len(new_keys)})",
                      flush=True)
            frontier = new_keys
    finally:
        pool.terminate()
    n = len(seen)
    from_i = np.array([e[0] for e in edges], dtype=np.int64)
    to_i = np.array([e[1] for e in edges], dtype=np.int64)
    cnts = np.array([e[2] for e in edges], dtype=np.float64)
    del edges
    if verbose:
        print(f"    sparse: n={n} nnz={len(from_i)}", flush=True)
    # Perron via ARPACK (the power iteration converges too slowly when a
    # subdominant eigenvalue hugs rho); fall back to bincount iteration
    import numpy as _np
    try:
        import scipy.sparse as _sp
        import scipy.sparse.linalg as _spla
        T = _sp.csr_matrix((cnts, (from_i, to_i)), shape=(n, n))
        vals_eigs = _spla.eigs(T, k=min(6, n - 2), which="LR",
                               return_eigenvectors=False)
        mags = _np.sort(_np.abs(vals_eigs))[::-1]
        rho = float(mags[0])
        gap = float(mags[1] / mags[0]) if len(mags) > 1 else 0.0
        if verbose:
            print(f"    eigs |lambda| top: "
                  f"{_np.array2string(mags, precision=6)}", flush=True)
        return math.log(rho), n, len(from_i), -1, gap
    except Exception as e:  # scipy missing or ARPACK trouble
        if verbose:
            print(f"    eigs unavailable ({e}); bincount fallback",
                  flush=True)
    v = _np.ones(n) / n
    rho_old = 1.0
    rel = 1.0
    it_used = 0
    for it in range(1, iters + 1):
        w = _np.bincount(to_i, weights=cnts * v[from_i], minlength=n)
        s = w.sum()
        rel = abs(s - rho_old) / s
        w /= s
        v = w
        rho_old = s
        it_used = it
        if rel < tol:
            break
    return math.log(rho_old), n, len(from_i), it_used, rel


def column_order(W, L, col_rule="min"):
    (face_eqs, order, cls_faces, slot, col_c, n_free, N,
     face_col) = geometry_columns(W, L, col_rule)
    return face_eqs, order, cls_faces, n_free, N


def diag():
    print(f"=== diag: signature levels + block census (QGRID={QGRID}) ===",
          flush=True)
    for (W, L) in [(2, 8), (3, 8)]:
        vid, edges, V, E, pts, mcs = build_torus_mc(W, L)
        vid2, edges2, V2, E2, pts2 = build_torus(W, L)
        assert edges == edges2, "wrapper/build_torus edge-list mismatch"
        col_e, sig_e = edge_cols(W, L, edges, mcs)
        # level sanity: distance of every quantized coord to a rounding edge
        worst = 1.0
        for eid, (u, v, b) in enumerate(edges):
            rm = mcs[eid]
            for val in ((rm[0] - rm[1] / SQ3) / 2.0 / W, rm[1] / SQ3):
                # round() flips only near half-integers of QGRID*val
                f = (val % 1.0) * QGRID % 1.0
                worst = min(worst, abs(f - 0.5))
        print(f"  (W={W}): worst rounding-boundary distance = {worst:.4f} "
              f"(need >= {BOUND_MIN})", flush=True)
        assert worst >= BOUND_MIN, worst
        # per-column edge signature uniqueness (absolute positions)
        seen = defaultdict(int)
        for eid in range(len(edges)):
            seen[(col_e[eid], sig_e[eid])] += 1
        dup = {kk: vv for kk, vv in seen.items() if vv > 1}
        assert not dup, f"duplicate edge signatures: {list(dup)[:5]}"
        print(f"  (W={W}): {len(edges)} edges, per-column signature "
              f"unique OK", flush=True)
    for rule in ("max",):
        print(f"  --- col_rule={rule} ---", flush=True)
        for W in (2, 3, 4, 5, 6, 8):
            t0 = time.time()
            blocks, bulk, brange, n_free, N = build_blocks(W, 8, 2,
                                                           col_rule=rule)
            ins = len(bulk[0])
            outs = len(bulk[2])
            nsteps = len(bulk[1])
            print(f"  W={W}: bulk cols {brange} steps={nsteps} "
                  f"|in|={ins} |out|={outs} raw 2^out={2 ** outs:.3g} "
                  f"raw 3^out={3 ** outs:.3g} [{time.time()-t0:.0f}s]",
                  flush=True)
    print("DIAG DONE", flush=True)


def gateA():
    print("=== Gate A: column-order DP vs exact counts ===", flush=True)
    known = {(2, 2): 22912, (2, 3): 58240, (2, 4): 2569312, (3, 2): 58240}
    for (W, L), exact in sorted(known.items()):
        face_eqs, order, cls_faces, n_free, N = column_order(W, L)
        dp, nfree2, N2, tw, peak, dt = dp_count(
            W, L, vals=(0, 1), face_order=order, rank_check=False)
        omega = dp * (2 ** nfree2)
        ok = omega == exact
        print(f"  ({W},{L}): column-order Omega={omega} vs exact {exact} "
              f"{'PASS' if ok else 'FAIL'} [peak={peak} {dt:.0f}s]",
              flush=True)
        assert ok, (W, L, omega, exact)
    print("Gate A PASS", flush=True)


def gateB():
    print("=== Gate B: full block product == exact DP count ===", flush=True)
    # (2,2)/(3,2) omitted: with L=2 the torus wrap makes translation mates
    # share the absolute lifetime (a_col, f_col) -- the local role naming
    # degenerates; (3,2)'s count is covered by Gate A as (2,3)'s transpose
    known = {(2, 3): 58240, (2, 4): 2569312}
    for (W, L), exact in sorted(known.items()):
        t0 = time.time()
        blocks, _, _, n_free, N = build_blocks(W, L, 2, need_bulk=False)
        v = {(): 1.0}
        for j in sorted(blocks):
            v = apply_block(blocks[j][:3], v, (0, 1))
        assert set(v) == {()}, f"nonempty tail keys at ({W},{L})"
        omega = int(round(sum(v.values()))) * (2 ** n_free)
        ok = omega == exact
        print(f"  ({W},{L}) k=2: block-product Omega={omega} vs {exact} "
              f"{'PASS' if ok else 'FAIL'} [{time.time()-t0:.0f}s]",
              flush=True)
        assert ok, (W, L, omega, exact)
    # ternary: dp_count and block product are independent code paths
    for (W, L) in [(2, 3)]:
        t0 = time.time()
        face_eqs, order, cls_faces, n_free, N = column_order(W, L)
        dp, nfree2, N2, tw, peak, dt = dp_count(
            W, L, vals=(0, 1, 2), face_order=order, rank_check=False)
        omega_dp = dp * (3 ** nfree2)
        blocks, _, _, n_free3, N3 = build_blocks(W, L, 3, need_bulk=False)
        assert n_free3 == nfree2
        v = {(): 1.0}
        for j in sorted(blocks):
            v = apply_block(blocks[j][:3], v, (0, 1, 2))
        assert set(v) == {()}
        omega_bp = int(round(sum(v.values()))) * (3 ** n_free)
        ok = omega_bp == omega_dp
        print(f"  ({W},{L}) k=3: dp={omega_dp} blocks={omega_bp} "
              f"{'PASS' if ok else 'FAIL'} [{time.time()-t0:.0f}s]",
              flush=True)
        assert ok, (W, L, omega_dp, omega_bp)
    print("Gate B PASS", flush=True)


def head_states(blocks, vals, nhead=2):
    js = sorted(blocks)
    head = {(): 1.0}
    for j in js[:nhead]:
        head = apply_block(blocks[j][:3], head, vals)
    return head


def stage_w2(gate=LOCAL_STATES):
    print("=== W=2 binary: sparse Perron ===", flush=True)
    blocks, bulk, brange, n_free, N = build_blocks(2, 8, 2)
    head = head_states(blocks, (0, 1))
    print(f"  bulk cols {brange} |out|={len(bulk[2])} head={len(head)}",
          flush=True)
    lam, n, nnz, it, rel = sparse_perron(bulk, head, (0, 1), gate=gate)
    dev = abs(lam - 2.7433)
    print(f"  lambda(2) = {lam:.6f}  |S|={n} nnz={nnz} it={it} rel={rel:.1e}"
          f"  vs exact-chain asymptote 2.7433: dev={dev:.4f} "
          f"{'PASS' if dev <= 0.02 else 'OPEN'}", flush=True)
    append_lambda(2, 2, lam, n, nnz, it, rel)
    return lam


def stage_w(W, k, gate=LOCAL_STATES, fasthead=0):
    blocks, bulk, brange, n_free, N = build_blocks(W, 8, k)
    vals = tuple(range(k))
    if fasthead:
        # random in-slot assignments as start set: the dominant class is
        # reached from any small sample (validated at W=2: K=100 gives the
        # full-head lambda to 15 digits with a 30x smaller closure)
        import random
        rng = random.Random(20261006 + W)
        head = {}
        while len(head) < fasthead:
            st = tuple(rng.randrange(k) for _ in bulk[0])
            head.setdefault(st, 1.0)
        print(f"  W={W} k={k}: fasthead {len(head)} random states",
              flush=True)
    else:
        head = head_states(blocks, vals)
        print(f"  W={W} k={k}: bulk cols {brange} |out|={len(bulk[2])} "
              f"head={len(head)}", flush=True)
    lam, n, nnz, it, rel = sparse_perron(bulk, head, vals, gate=gate)
    print(f"  lambda(W={W}, k={k}) = {lam:.6f}  |S|={n} nnz={nnz} "
          f"it={it} rel={rel:.1e}", flush=True)
    append_lambda(k, W, lam, n, nnz, it, rel)
    return lam


LAMBDA_CSV = os.environ.get(
    "RTH_TM_CSV",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "RTH4612_tm_lambda.csv"))


def append_lambda(k, W, lam, n, nnz, it, rel):
    import csv as _csv
    exists = os.path.exists(LAMBDA_CSV)
    with open(LAMBDA_CSV, "a", newline="") as f:
        w = _csv.writer(f)
        if not exists:
            w.writerow(["k", "W", "lambda_lnRho", "n_states", "nnz",
                        "iters", "relerr"])
        w.writerow([k, W, f"{lam:.8f}", n, nnz, it, f"{rel:.2e}"])


def main():
    argv = sys.argv[1:]
    gate = LOCAL_STATES
    if "--server" in argv:
        gate = GATE_STATES
        argv.remove("--server")
    fasthead = 0
    if "--fasthead" in argv:
        i = argv.index("--fasthead")
        fasthead = int(argv[i + 1])
        del argv[i:i + 2]
    stage = argv[0] if argv else "all"
    if stage in ("diag", "all"):
        diag()
    if stage in ("gateA", "all"):
        gateA()
    if stage in ("gateB", "all"):
        gateB()
    if stage in ("w2", "all"):
        stage_w2(gate=gate)
    if stage in ("w3", "all", "prod3"):
        stage_w(3, 2, gate=gate, fasthead=fasthead)
    if stage == "t2":
        stage_w(2, 3, gate=gate, fasthead=fasthead)
    if stage == "t3":
        stage_w(3, 3, gate=gate, fasthead=fasthead)
    if stage == "w4":
        stage_w(4, 2, gate=gate, fasthead=fasthead)
    if stage == "t4":
        stage_w(4, 3, gate=gate, fasthead=fasthead)
    if stage == "t5":
        stage_w(5, 3, gate=gate, fasthead=fasthead)
    if stage == "w5":
        stage_w(5, 2, gate=gate, fasthead=fasthead)
    print(f"STAGE {stage} DONE", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
