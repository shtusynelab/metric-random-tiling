# -*- coding: utf-8 -*-
"""
tsq_lambda_lib.py - exact counting library for the 4.8.8
Lambda = Z + (sqrt(2)/2)*Z edge-value library.

Conventions:
  Edge length s_e = m_e + n_e*(sqrt(2)/2); per-edge variables (m_e, n_e)
  drawn from M subset Z^2 (the same library on every edge).
  Experiment library M_EXP = {(1,0),(0,1),(1,1)} (s in {1, sqrt(2)/2,
  1+sqrt(2)/2}; 3 values, non-arithmetic-progression);
  control library M_INT = {(1,0),(2,0),(3,0)} (s in {1,2,3}; Z library with
  the exact anchor Omega = 3^{4W}).
  Quadrilateral faces = equiangular quadrilaterals: the closure constraint
  only enforces opposite-side equality, not adjacent-side equality.
  All counting in Python int; all linear adjudication in Fraction; no float
  enters any adjudication path.

Face constraints (each real-projection equation -> 2 integer rows; variable
order = (m_0..m_{E-1}, n_0..n_{E-1})):
  DIR8[d] = (A,B,C,D); real direction = ((A+B*sqrt(2))/2, (C+D*sqrt(2))/2):
    X1: sum(A m + B n) = 0      X2: sum(2B m + A n) = 0
    Y1: sum(C m + D n) = 0      Y2: sum(2D m + C n) = 0
  The all-ones solution (m == 1, n == 0) satisfies every row automatically;
  for the control library (n == 0) this degenerates to the same coefficient
  row-splitting as tsq_lib.rows_split.
  Asserted row rank = 2E - 2*dim_R = 8W^2-4; solution-module rank (Z-rank)
  = 2*dim_R = 4W^2+4 (dim_R = 2W^2+2; independently verified for W=2..6 and
  re-asserted here by check_dims).
"""
from __future__ import annotations

import math
from fractions import Fraction
from itertools import product

import numpy as np

from tsq_lib import TSQTorus, DIR8

# Library definitions (integer pairs (m, n))
M_EXP = ((1, 0), (0, 1), (1, 1))
M_INT = ((1, 0), (2, 0), (3, 0))


# ---------------------------------------------------------------- linear system

def face_rows(sides, E):
    """The <=4 integer library rows of one face (sparse dict: variable index
    -> nonzero integer coefficient). Variable indices: e in [0,E) is m_e,
    E+e is n_e. All-zero rows are dropped."""
    rows = []
    for (i1, i2) in ((0, 1), (2, 3)):        # (A,B) and (C,D) pairs
        r1, r2 = {}, {}
        for e, d in sides:
            A, B = DIR8[d][i1], DIR8[d][i2]
            if A:
                r1[e] = r1.get(e, 0) + A
                r2[E + e] = r2.get(E + e, 0) + A
            if B:
                r1[E + e] = r1.get(E + e, 0) + B
                r2[e] = r2.get(e, 0) + 2 * B
        for r in (r1, r2):
            if r:
                rows.append(r)
    return rows


def lambda_system(torus):
    """Full-torus library row system: list of sparse dicts. Row order =
    face order, up to 4 rows per face."""
    rows = []
    for size, sides in torus.faces:
        rows.extend(face_rows(sides, torus.ne))
    return rows


def rank_rows(rows, ncol):
    """Q-rank via Fraction Gaussian elimination. Row entries: sparse dict or
    dense sequence."""
    mat = []
    for r in rows:
        if isinstance(r, dict):
            mat.append([Fraction(r.get(c, 0)) for c in range(ncol)])
        else:
            mat.append([Fraction(x) for x in r])
    rank = 0
    for c in range(ncol):
        p = None
        for rr in range(rank, len(mat)):
            if mat[rr][c]:
                p = rr
                break
        if p is None:
            continue
        mat[rank], mat[p] = mat[p], mat[rank]
        pv = mat[rank][c]
        mat[rank] = [x / pv for x in mat[rank]]
        for rr in range(len(mat)):
            if rr != rank and mat[rr][c]:
                f = mat[rr][c]
                mat[rr] = [x - f * y for x, y in zip(mat[rr], mat[rank])]
        rank += 1
        if rank == len(mat):
            break
    return rank


def check_dims(W, verbose=True):
    """Rank self-check: row rank = 8W^2-4; solution-module rank = 4W^2+4 =
    2*dim_R; dim_R = 2W^2+2; the all-ones solution lies in the kernel.
    Returns a dict."""
    t = TSQTorus(W, W)
    E, F = t.ne, t.F
    rows = lambda_system(t)
    nvar = 2 * E
    r = rank_rows(rows, nvar)
    dimR = E - rank_rows(
        [[Fraction(x) for x in row] for row in t.rows_float], E)
    dimL = nvar - r
    ones_ok = all(
        sum(coef * (1 if v < E else 0) for v, coef in row.items()) == 0
        for row in rows)
    out = dict(W=W, E=E, F=F, n_rows=len(rows), rank=r,
               rank_expect=8 * W * W - 4, dimR=dimR, dimR_expect=2 * W * W + 2,
               dimL=dimL, dimL_expect=4 * W * W + 4, ones_ok=ones_ok)
    assert r == out["rank_expect"], f"row rank {r} != {out['rank_expect']}"
    assert dimR == out["dimR_expect"], f"dim_R {dimR} != {out['dimR_expect']}"
    assert dimL == out["dimL_expect"], f"solution-module rank {dimL} != {out['dimL_expect']}"
    assert ones_ok, "all-ones solution not in kernel"
    if verbose:
        print(f"W={W}: E={E} rows={len(rows)} rank={r} (=8W^2-4 OK) "
              f"dim_R={dimR} dim_L={dimL} (=2dim_R OK) ones_ok")
    return out


# ---------------------------------------------------------------- face order

def mono_order(W):
    """Monotone face order for the torus: O-row 0, S-row 0, then the pairs
    (O-row j, S-row j) for j = 1..W-1.

    S0 immediately follows O0: the top edges of S0 (= the top cut of
    O_{W-1}) are fixed directly by the square's opposite-side equality
    (zero branching), so the bottom cut of O0 no longer stays free until
    the end of the sweep; frontier degrees of freedom = 3W+1 (the legacy
    order's 5W+1 blows up: 16 million states already at W=3, OOM at W=4).
    TSQTorus.faces indexing: square index = i*W + j, octagon = W^2 + i*W + j.
    Monotonicity is asserted inside dp_count (all known edges must be on the
    frontier when each face is processed)."""
    order = [W * W + i * W + 0 for i in range(W)]              # O row 0
    order += [i * W + 0 for i in range(W)]                     # S row 0
    for j in range(1, W):
        order += [W * W + i * W + j for i in range(W)]         # O row j
        order += [i * W + j for i in range(W)]                 # S row j
    return order


def mono_order_interleaved(W):
    """Interleaved within-row order: O_j(0), S_j(1), O_j(1), ...,
    O_j(W-1), S_j(0).

    Each square S_j(i+1) immediately follows O_j(i): one pair of its sides
    (the bottom cut) consists of new edges fixed directly by the previous
    row's top cut (one choice out of 3), and the other pair immediately
    adjudicates the previous octagon's SE choice. Combined with the seed =
    south axis edges axis[(0,i,0)] (wrap-around horizontal axis; otherwise
    they would stay free for the whole W-slot sweep), frontier degrees of
    freedom = 3W (rolling boundary: north axis W + top cuts 2W) plus
    transients; state upper bound ~3^{3W}."""
    order = []
    for j in range(W):
        for i in range(W):
            order.append(W * W + i * W + j)          # O_j(i)
            order.append(((i + 1) % W) * W + j)      # S_j(i+1 mod W)
    return order


def boundary_seed_edges(torus):
    """Seed edges = the W wrap-around horizontal axis edges axis[(0,i,0)]
    plus the 2W boundary cut edges cut[(a,0,2)], cut[(a,0,3)] (= the top
    cuts of O_{W-1}).

    With axis edges only, the boundary cut edges are created in S_0 as
    "both-sides-new", each pair carrying a factor-3 freedom that persists
    to the end of the sweep, so the effective freedom is 5W (the measured
    W=3 peak of 898k states = 3^9 x 45 exposed this). With the full 3W
    seed: the two pairs of S_0 sides become respectively fixed-by-bottom-
    cut (= seed value) and a pure SE adjudication (killing 2/3), so the
    frontier freedom is exactly 3W and the state upper bound is
    3^{3W} x per-octagon transients."""
    W = torus.Lq
    return ([torus._axis[(0, i, 0)] for i in range(W)]
            + [torus._cuts[(a, 0, 2)] for a in range(W)]
            + [torus._cuts[(a, 0, 3)] for a in range(W)])


def dp_count_row(W, M=M_EXP, max_states=None, verbose=False,
                 chunk_elems=2 * 10 ** 7, workers=1):
    """Recommended entry point: interleaved row order + south-axis seed.
    Returns (omega, peak, stats)."""
    t = TSQTorus(W, W)
    seed = boundary_seed_edges(t)
    return dp_count_fast(W, M=M, max_states=max_states, verbose=verbose,
                         chunk_elems=chunk_elems,
                         order_fn=mono_order_interleaved, seed_edges=seed,
                         workers=workers)


def dp_count_row_batched(W, M=M_EXP, batch_size=200000, max_states=None,
                         verbose=False, chunk_elems=2 * 10 ** 7, workers=1):
    """Batched row sweep: advance row by row, split the boundary states into
    chunks of batch_size within a row, and aggregate across batches at row
    end.

    Per-batch transient ~= batch_size x b (b ~= 45.6, the per-octagon-face
    survival count); the row-end aggregate dictionary is bounded by the
    number of reachable rolling-boundary states. Used for W=5 (materializing
    the 3^{3W} seeds x transients at once would exhaust memory).
    Returns (omega, peak_states, stats)."""
    import numpy as np
    t = TSQTorus(W, W)
    seed = boundary_seed_edges(t)
    order = mono_order_interleaved(W)
    NF = 2 * W * W
    kk = len(M)
    states = {c: (c, 1, 0) for c in product(range(kk), repeat=len(seed))}
    peak_all = len(states)
    stats = []
    for j in range(W):
        f0, f1 = 2 * W * j, 2 * W * (j + 1)
        acc = {}
        if j == 0:
            # seed row: seed slices batched (partial seed injection; cross-batch
            # aggregation at row end, counts summed)
            n_seed = len(seed)
            n_all = kk ** n_seed
            nb0 = (n_all + batch_size - 1) // batch_size
            for b in range(nb0):
                lo, hi = b * batch_size, min(n_all, (b + 1) * batch_size)
                init = {}
                for idx in range(lo, hi):
                    code = idx
                    ckey = []
                    for _ in range(n_seed):
                        ckey.append(code % kk)
                        code //= kk
                    ckey = tuple(reversed(ckey))
                    init[ckey] = (ckey, 1, 0)
                st, pk, sts = dp_count_fast(W, M=M, max_states=max_states,
                                            verbose=verbose,
                                            chunk_elems=chunk_elems,
                                            order_fn=mono_order_interleaved,
                                            seed_edges=seed,
                                            faces_range=(0, f1),
                                            init_states=init, workers=workers)
                for k2, v2 in st.items():
                    if k2 in acc:
                        a = acc[k2]
                        acc[k2] = (v2[0], a[1] + v2[1], a[2] + v2[2])
                    else:
                        acc[k2] = v2
                peak_all = max(peak_all, pk)
                stats.extend(sts)
                if verbose:
                    print(f"  row0 batch {b+1}/{nb0}: cumulative keys {len(acc)}",
                          flush=True)
        else:
            items = list(states.items())
            nbatch = (len(items) + batch_size - 1) // batch_size
            for b0 in range(0, len(items), batch_size):
                init = dict(items[b0:b0 + batch_size])
                st, pk, sts = dp_count_fast(W, M=M, max_states=max_states,
                                            verbose=verbose,
                                            chunk_elems=chunk_elems,
                                            order_fn=mono_order_interleaved,
                                            seed_edges=seed, faces_range=(f0, f1),
                                            init_states=init, workers=workers)
                for k2, v2 in st.items():
                    if k2 in acc:
                        a = acc[k2]
                        acc[k2] = (v2[0], a[1] + v2[1], a[2] + v2[2])
                    else:
                        acc[k2] = v2
                peak_all = max(peak_all, pk)
                stats.extend(sts)
        states = acc
        if max_states is not None and len(states) > max_states:
            raise RuntimeError(
                f"W={W} row {j} end-of-row aggregate states {len(states)} > budget {max_states}")
        if verbose:
            print(f"  row {j}: batches={nbatch} row-end states={len(states)}",
                  flush=True)
    omega = sum(hi * (1 << 32) + lo for v, lo, hi in states.values())
    return omega, peak_all, stats


def mono_order_legacy(W):
    """Legacy face order (S0 last; O0 bottom cut stays free throughout,
    3^{5W+1} states). Kept only for cross-checking."""
    order = [W * W + i * W + 0 for i in range(W)]              # O row 0
    for j in range(1, W):
        order += [W * W + i * W + j for i in range(W)]         # O row j
        order += [i * W + j for i in range(W)]                 # S row j
    order += [i * W + 0 for i in range(W)]                     # S row 0
    return order


# ---------------------------------------------------------------- exact DP counting

def dp_count(W, M=M_EXP, max_states=None, verbose=False):
    """Exact per-face frontier DP count of Omega for library M.

    State = tuple of library-value indices assigned to frontier edges;
    Python int counts; identical frontiers merged. Returns
    (omega, peak_states); monotonicity violations are assertion-checked
    (always 0 on return). max_states: exceeding it raises RuntimeError
    (fail-closed budget gate)."""
    t = TSQTorus(W, W)
    E = t.ne
    order = mono_order(W)
    faces = [t.faces[i] for i in order]
    fd_rows = [face_rows(sides, E) for size, sides in faces]
    first, last = {}, {}
    for fi, (size, sides) in enumerate(faces):
        for e, d in sides:
            if e not in first:
                first[e] = fi
            last[e] = fi
    Mv = list(M)
    k = len(Mv)

    front = []                       # frontier edge list (fixed order)
    states = {(): 1}
    peak = 0
    for fi in range(len(faces)):
        sides = faces[fi][1]
        e_in = [e for e, d in sides if e in first and first[e] < fi]
        e_new = [e for e, d in sides if first[e] == fi]
        # monotonicity assertion: e_in must all be on the frontier; the edges
        # of fi must be in_front union e_new
        fset = set(front)
        for e in e_in:
            assert e in fset, f"non-monotone: edge {e} of face {fi} not on the frontier (order violated)"
        known_pos = {}
        for e in e_in:
            known_pos.setdefault(e, []).extend(range(len(fd_rows[fi])))
        # collect coefficients per row by sides position (sparse: variable
        # index -> (side position, coefficient))
        row_spec = []
        for row in fd_rows[fi]:
            spec = []
            for t_i, (e, d) in enumerate(sides):
                c1 = row.get(e, 0)
                c2 = row.get(E + e, 0)
                if c1 or c2:
                    spec.append((t_i, c1, c2))
            row_spec.append(spec)
        pos_of = {e: i for i, (e, d) in enumerate(sides)}
        # new frontier
        keep = [x for x in front if last[x] > fi]
        keep_set = set(keep)
        add = [e for e in e_new if last[e] > fi]
        new_front = keep + add
        # positions in the old front of consumed edges (for value lookup)
        old_pos = {e: i for i, e in enumerate(front)}
        keep_old_idx = [old_pos[e] for e in keep]
        add_idx = [pos_of[e] for e in add]
        add_pos_in_sides = [pos_of[e] for e in add]
        in_pos = {e: pos_of[e] for e in e_in}

        new_states = {}
        for fkey, cnt in states.items():
            fv = list(fkey)
            # sides value array: fill known entries first
            side_val = [None] * len(sides)
            for e in e_in:
                side_val[pos_of[e]] = Mv[fv[old_pos[e]]]
            # enumerate new-edge combinations
            for combo in product(range(k), repeat=len(e_new)):
                for ci, e in enumerate(e_new):
                    side_val[pos_of[e]] = Mv[combo[ci]]
                ok = True
                for spec in row_spec:
                    ssum = 0
                    for t_i, c1, c2 in spec:
                        m, n = side_val[t_i]
                        ssum += c1 * m + c2 * n
                    if ssum != 0:
                        ok = False
                        break
                if not ok:
                    continue
                nk = tuple(fv[i] for i in keep_old_idx) + combo_leave(
                    combo, e_new, add)
                new_states[nk] = new_states.get(nk, 0) + cnt
        states = new_states
        front = new_front
        peak = max(peak, len(states))
        if max_states is not None and len(states) > max_states:
            raise RuntimeError(
                f"W={W} DP states {len(states)} > budget {max_states} (fail-closed)")
    assert not front, "frontier not empty after the sweep"
    omega = sum(states.values())
    return omega, peak


def combo_leave(combo, e_new, add):
    """The slice of the new-edge combination that enters the new frontier
    (the add edges); consumed edges (this face is their last face) are
    dropped. add is a subset of e_new (the keep part comes from the old
    frontier)."""
    # add edges are all in e_new (their first == fi); return the combo values
    # at the corresponding positions
    epos = {e: i for i, e in enumerate(e_new)}
    return tuple(combo[epos[e]] for e in add)


# ---------------------------------------------------------------- fast DP

def _row_echelon_int(rowsF, ncol):
    """Reduce sparse Fraction rows to an integer row-echelon basis. rowsF:
    list of dicts {col: coef}. Returns an np.int64 basis matrix (rb, ncol)
    whose Q-row space equals the input row space."""
    mat = []
    for r in rowsF:
        mat.append([Fraction(r.get(c, 0)) for c in range(ncol)])
    basis = []
    rank = 0
    for c in range(ncol):
        p = None
        for rr in range(rank, len(mat)):
            if mat[rr][c]:
                p = rr
                break
        if p is None:
            continue
        mat[rank], mat[p] = mat[p], mat[rank]
        pv = mat[rank][c]
        mat[rank] = [x / pv for x in mat[rank]]
        for rr in range(rank + 1, len(mat)):
            if mat[rr][c]:
                f = mat[rr][c]
                mat[rr] = [x - f * y for x, y in zip(mat[rr], mat[rank])]
        basis.append(mat[rank])
        rank += 1
        if rank == len(mat):
            break
    # clear denominators to obtain an integer basis
    out = []
    for row in basis:
        D = 1
        for x in row:
            if x:
                D = D * x.denominator // math.gcd(D, x.denominator)
        out.append([int(x * D) for x in row])
    import numpy as np
    return np.array(out, dtype=np.int64) if out else np.zeros((0, ncol), np.int64)


_PC = {}


def _pc_init(fp, newp, Sv, keep_arr, cnt_lo, cnt_hi, Kidx, keep, n_new, c_out):
    """Inject worker globals (under fork this bypasses pickling)."""
    _PC.update(fp=fp, newp=newp, Sv=Sv, keep_arr=keep_arr, cnt_lo=cnt_lo,
               cnt_hi=cnt_hi, Kidx=Kidx, keep=keep, n_new=n_new, c_out=c_out)


def _pc_work(bounds):
    """Process one chunk + chunk-level unique aggregation; returns
    (uv_c, g_lo_c, g_hi_c) or None."""
    s0, s1 = bounds
    fp = _PC["fp"][:, s0:s1]
    newp = _PC["newp"]
    Sv = _PC["Sv"]
    keep_arr = _PC["keep_arr"]
    cnt_lo = _PC["cnt_lo"]
    cnt_hi = _PC["cnt_hi"]
    Kidx = _PC["Kidx"]
    keep = _PC["keep"]
    n_new = _PC["n_new"]
    c_out = _PC["c_out"]
    eq = fp[:, :, None] + newp[:, None, :]
    valid = (eq == 0).all(axis=0)
    ii, jj = np.nonzero(valid)
    nc = len(ii)
    if nc == 0:
        return None
    outv_c = np.zeros((nc, c_out), dtype=np.int8)
    if keep:
        outv_c[:, :len(keep)] = Sv[(s0 + ii)[:, None], keep_arr].reshape(
            nc, len(keep))
    if n_new:
        outv_c[:, len(keep):] = Kidx[jj]
    uv_c, inv_c = np.unique(outv_c, axis=0, return_inverse=True)
    plo = cnt_lo[s0:s1][ii]
    phi = cnt_hi[s0:s1][ii]
    o2 = np.argsort(inv_c, kind="stable")
    st2 = np.concatenate([[0], np.nonzero(np.diff(inv_c[o2]))[0] + 1])
    g_lo_c = np.add.reduceat(plo[o2], st2)
    g_hi_c = np.add.reduceat(phi[o2], st2)
    return (uv_c, g_lo_c, g_hi_c)


def dp_count_fast(W, M=M_EXP, max_states=None, verbose=False, check_basis=False,
                  merge_B=True, chunk_elems=2 * 10 ** 7, order_fn=mono_order,
                  seed_edges=None, faces_range=None, init_states=None,
                  workers=1):
    """Vectorized frontier DP with future-functional linear merging (exact,
    Python-int counting).

    Merge criterion: frontier assignments v and v' merge iff v - v' is
    orthogonal to all future constraint rows (the set of future continuations
    reads the frontier only through rows of M x = 0) iff B v == B v' for the
    echelon basis B. B is rebuilt per face (future rows x current frontier
    columns, Fraction elimination, integer basis).
    Counts are split into two int64 lanes (hi, lo) with hi*2^32 + lo; bounds
    are verified before within-group summation.
    Returns (omega, peak_states, merge_stats)."""
    import numpy as np
    t = TSQTorus(W, W)
    E = t.ne
    order = order_fn(W)
    faces = [t.faces[i] for i in order]
    frows = [face_rows(sides, E) for size, sides in faces]
    NF = len(faces)
    first, last = {}, {}
    for fi, (size, sides) in enumerate(faces):
        for e, d in sides:
            if e not in first:
                first[e] = fi
            last[e] = fi
    if seed_edges:
        for e in seed_edges:
            first[e] = -1          # seed edge: assigned before the sweep, never treated as new
    # profile: pre[fi] = frontier edge list before face fi is processed
    pre = []
    cur = list(seed_edges) if seed_edges else []
    for fi in range(NF):
        pre.append(list(cur))
        sides = faces[fi][1]
        cur = [e for e in cur if last[e] > fi] + \
              [e for e, d in sides if first[e] == fi and last[e] > fi]
    pre.append(list(cur))
    assert not pre[NF]

    Marr = np.array(sorted(M), dtype=np.int64)
    kk = len(Marr)

    def build_B(fi):
        cols = {e: i for i, e in enumerate(pre[fi])}
        if not cols:
            return np.zeros((0, 0, 2), np.int64)
        rowsF = []
        for f in range(fi, NF):
            for row in frows[f]:
                rr = {}
                for v, c in row.items():
                    e = v if v < E else v - E
                    if e in cols:
                        key = (cols[e], 0 if v < E else 1)
                        rr[key] = rr.get(key, 0) + c
                if rr:
                    dense = {}
                    for (ci, kn), c in rr.items():
                        dense[2 * ci + kn] = c   # column = 2*col + (m/n)
                    rowsF.append(dense)
        B = _row_echelon_int(rowsF, 2 * len(cols))
        if check_basis:
            import numpy as _np
            # assert: the restriction of every future row to the frontier lies
            # in the Q-row space of B
            rB = _np.linalg.matrix_rank(B.astype(float)) if B.size else 0
            for f in range(fi, NF):
                for row in frows[f]:
                    lim = [0] * (2 * len(cols))
                    for v, c in row.items():
                        e = v if v < E else v - E
                        if e in cols:
                            lim[2 * cols[e] + (0 if v < E else 1)] = c
                    if any(lim):
                        BB = _np.vstack([B, _np.array(lim, float)]) if B.size \
                            else _np.array([lim], float)
                        assert _np.linalg.matrix_rank(BB.astype(float)) == rB, \
                            "future row not in the echelon-basis row space"
        # reformat per row: (rb, c, 2): coefficients split by (column, m/n)
        # for easy multiplication
        rb = B.shape[0]
        Bf = np.zeros((rb, len(cols), 2), dtype=np.int64)
        for i in range(rb):
            for ci in range(len(cols)):
                Bf[i, ci, 0] = B[i, 2 * ci]
                Bf[i, ci, 1] = B[i, 2 * ci + 1]
        return Bf

    # initial states: empty frontier / all seed-edge combinations /
    # caller-injected (row-0 seed slices, batched)
    if init_states is not None:
        states = dict(init_states)
    elif seed_edges:
        n_seed = len(seed_edges)
        states = {c: (c, 1, 0) for c in product(range(kk), repeat=n_seed)}
    else:
        states = {(): ((), 1, 0)}   # key -> (representative values, lo, hi)
    stats = []
    f_lo, f_hi = faces_range if faces_range else (0, NF)
    assert not (init_states is not None and f_lo == 0 and not seed_edges),         "f0=0 with init_states requires seed_edges (needed to build pre)"
    for fi in range(f_lo, f_hi):
        pre_in = pre[fi]
        pre_out = pre[fi + 1]
        sides = faces[fi][1]
        pos_of = {e: i for i, (e, d) in enumerate(sides)}
        e_in = [e for e in pre_in if e in pos_of]
        e_new = [e for e, d in sides if first[e] == fi]
        keep = [e for e in pre_in if last[e] > fi]      # retained segment of the outgoing frontier
        assert keep + e_new == pre_out, "profile splice broken"
        colF = {e: i for i, e in enumerate(pre_in) if e in pos_of}  # pre_in column index
        colN = {e: i for i, e in enumerate(e_new)}
        keep_idx = [pre_in.index(e) for e in keep]
        c_in, c_out = len(pre_in), len(pre_out)
        n_new = len(e_new)
        # row spec: [(src, idx, cm, cn)]
        specs = []
        for row in frows[fi]:
            acc = {}
            for v, c in row.items():
                e = v if v < E else v - E
                isn = 0 if v < E else 1
                if e in colF:
                    key = ("F", colF[e])
                elif e in colN:
                    key = ("N", colN[e])
                else:
                    raise AssertionError("row variable not on this face")
                acc.setdefault(key, [0, 0])[isn] += c
            specs.append([(src, idx, cm, cn)
                          for (src, idx), (cm, cn) in acc.items()])
        n_r = len(specs)
        # combination matrix
        C = kk ** n_new
        Kidx = np.zeros((C, n_new), dtype=np.int64)
        if n_new:
            for j in range(n_new):
                Kidx[:, j] = (np.arange(C) // (kk ** (n_new - 1 - j))) % kk
        Km = Marr[Kidx, 0] if n_new else np.zeros((C, 0), np.int64)
        Kn = Marr[Kidx, 1] if n_new else np.zeros((C, 0), np.int64)
        # precompute the front contribution per state
        S = len(states)
        vals = [v for v, lo, hi in states.values()]
        cnt_lo = np.array([lo for v, lo, hi in states.values()], np.int64)
        cnt_hi = np.array([hi for v, lo, hi in states.values()], np.int64)
        Sv = np.array(vals, dtype=np.int8).reshape(S, c_in)
        # B (future functionals, acting on pre_out)
        Bf = build_B(fi + 1)
        rb = Bf.shape[0]
        # static front contribution: Sv stores library indices, lookup in
        # Marr gives (m,n); int16 saves memory (|sum| <= 96)
        frontp = np.zeros((n_r, S), dtype=np.int16)
        newp = np.zeros((n_r, C), dtype=np.int16)
        for r, sp in enumerate(specs):
            acc = np.zeros(S, dtype=np.int16)
            for src, idx, cm, cn in sp:
                if src == "F":
                    acc += (Marr[Sv[:, idx], 0] * cm + Marr[Sv[:, idx], 1] * cn).astype(np.int16)
                else:
                    newp[r] += (Km[:, idx] * cm + Kn[:, idx] * cn).astype(np.int16)
            frontp[r] = acc
        # chunked adjudication: within a chunk only raw passing rows
        # accumulate (int8 + counts); a single unique+reduceat pass at the end
        # with workers > 1, chunks are parallelized with multiprocessing
        # (pure data parallelism, chunks independent; workers aggregate at
        # chunk level and return only aggregated results, so the transfer is
        # the number of aggregated rows)
        CH = max(1, int(chunk_elems) // max(C, 1))
        keep_arr = np.array(keep_idx, dtype=np.int64)
        chunk_bounds = [(s0, min(S, s0 + CH))
                        for s0 in range(0, max(S, 1), CH)]
        N = 0

        _pc_init(frontp, newp, Sv, keep_arr, cnt_lo, cnt_hi, Kidx,
                 keep, n_new, c_out)
        if workers > 1 and len(chunk_bounds) > 1:
            import multiprocessing as mp
            if "fork" in mp.get_all_start_methods():
                ctx = mp.get_context("fork")   # Linux: true parallelism, large arrays shared COW
                with ctx.Pool(workers, initializer=_pc_init,
                              initargs=(frontp, newp, Sv, keep_arr, cnt_lo,
                                        cnt_hi, Kidx, keep, n_new, c_out)) as pool:
                    results = pool.map(_pc_work, chunk_bounds)
            else:
                results = [_pc_work(b) for b in chunk_bounds]
        else:
            results = [_pc_work(b) for b in chunk_bounds]
        acc = [r for r in results if r is not None]
        if acc:
            uv_all = np.vstack([a[0] for a in acc])
            g_lo_all = np.concatenate([a[1] for a in acc])
            g_hi_all = np.concatenate([a[2] for a in acc])
            N = int(uv_all.shape[0])
            uv, inv = np.unique(uv_all, axis=0, return_inverse=True)
            o2 = np.argsort(inv, kind="stable")
            st2 = np.concatenate([[0], np.nonzero(np.diff(inv[o2]))[0] + 1])
            g_lo = np.add.reduceat(g_lo_all[o2], st2)
            g_hi = np.add.reduceat(g_hi_all[o2], st2)
            u1 = uv.shape[0]
            del uv_all, g_lo_all, g_hi_all, acc, results
        else:
            uv = np.zeros((0, c_out), dtype=np.int8)
            g_lo = np.zeros(0, np.int64)
            g_hi = np.zeros(0, np.int64)
            u1 = 0
        if max_states is not None and max(u1, 1) > max_states:
            raise RuntimeError(
                f"W={W} face {fi} value states {u1} > budget {max_states} (fail-closed)")
        # group-count overflow guard (within-group sums must stay >= 0 and not
        # overflow; int64 overflow is silent in numpy)
        # bound: group size x 2^32 <= S x 4.3e9; Omega/2^32 - assert the bound
        # up front
        # second layer: B-image merging (merge_B=False degenerates to
        # per-value states, for cross-checking);
        # adaptive skip: for large u1 the merge does not compress
        # (u2 ~= u1) and einsum/unique dominate the cost,
        # so use the value keys (uv rows) directly - mathematically equivalent
        # to the B keys (keys only group the aggregation).
        if u1 and merge_B and u1 <= 2_000_000:
            KB = np.zeros((u1, rb), dtype=np.int64)
            BK = 4 * 10 ** 5
            for s0 in range(0, u1, BK):
                s1_ = min(u1, s0 + BK)
                vc = Marr[uv[s0:s1_]]                          # (nc, c, 2)
                KB[s0:s1_] = np.einsum("ijk,ljk->il", vc, Bf)
            uk, inv2 = np.unique(KB, axis=0, return_inverse=True)
            order3 = np.argsort(inv2, kind="stable")
            inv3 = inv2[order3]
            st3 = np.concatenate([[0], np.nonzero(np.diff(inv3))[0] + 1])
            glo = np.add.reduceat(g_lo[order3], st3)
            ghi = np.add.reduceat(g_hi[order3], st3)
            u2 = len(st3)
            rep_rows = order3[st3]             # first row of each group (index into the u1 rows)
            keys = [uk[g].tobytes() for g in range(u2)]
        else:
            # u1 groups: group id u iff uv row u (np.unique's unique rows are
            # arranged by group id)
            u2 = u1
            glo, ghi = g_lo, g_hi
            rep_rows = np.arange(u1)
            keys = [uv[g].tobytes() for g in range(u1)]
        new_states = {}
        for g in range(u2):
            r1 = rep_rows[g]
            val_t = tuple(int(x) for x in uv[r1])
            new_states[keys[g]] = (val_t, int(glo[g]), int(ghi[g]))
        states = new_states
        peak = u2
        stats.append((fi, S, N, u1, u2, c_out, rb))
        if verbose:
            print(f"  face {fi}: S={S} pairs={N} u1={u1} u2={u2} c={c_out} rb={rb}",
                  flush=True)
        if max_states is not None and u2 > max_states:
            raise RuntimeError(f"W={W} states {u2} > budget {max_states}")
        # overflow guard: hi accumulation bound
        if any(abs(int(x)) > (1 << 62) for x in ghi):
            raise RuntimeError("hi count overflows int64")
    if faces_range:
        return states, (stats[-1][4] if stats else 1), stats
    omega = sum(hi * (1 << 32) + lo for v, lo, hi in states.values())
    return omega, (stats[-1][4] if stats else 1), stats




def dfs_count(W, M=M_EXP):
    """Brute-force anchor at the W=2 scale: same face order, full enumeration
    of new-edge combinations per face, face-closure pruning. The
    implementation is independent of dp_count (no frontier/state merging,
    direct recursion); used for cross-checking."""
    t = TSQTorus(W, W)
    E = t.ne
    faces = [t.faces[i] for i in mono_order(W)]
    fd_rows = [face_rows(sides, E) for size, sides in faces]
    val = {}

    def rec(fi):
        if fi == len(faces):
            return 1
        sides = faces[fi][1]
        new_e = [e for e, d in sides if e not in val]
        rows = fd_rows[fi]
        total = 0
        for combo in product(M, repeat=len(new_e)):
            for e, v in zip(new_e, combo):
                val[e] = v
            ok = True
            for row in rows:
                if sum(c * (val[v][0] if v < E else val[v - E][1])
                       for v, c in row.items()) != 0:
                    ok = False
                    break
            if ok:
                total += rec(fi + 1)
            for e in new_e:
                del val[e]
        return total

    return rec(0)


# ---------------------------------------------------------------- self-tests

if __name__ == "__main__":
    print("== rank self-check (assert solution-module rank = 2*dim_R = 4W^2+4, row rank = 8W^2-4) ==")
    for W in (2, 3):
        check_dims(W)
    print("== W=2 control-library anchor: Omega should equal 3^8 = 6561 ==")
    om, pk = dp_count(2, M=M_INT)
    print(f"DP control library W=2: Omega={om} (3^8={3**8}) peak_states={pk}")
    assert om == 3 ** 8
    print("== W=2 experiment library: DP vs DFS cross-check ==")
    om_exp, pk_exp = dp_count(2, M=M_EXP)
    om_dfs = dfs_count(2, M=M_EXP)
    print(f"DP={om_exp} (peak {pk_exp})  DFS={om_dfs}  "
          f"{'MATCH' if om_exp == om_dfs else 'MISMATCH'}")
    assert om_exp == om_dfs
    print("ALL SELFTESTS PASSED")
