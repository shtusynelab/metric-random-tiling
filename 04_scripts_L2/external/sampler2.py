# -*- coding: utf-8 -*-
"""sampler2.py -- two-stage exact uniform sampler (ensemble-fragmentation
fix v3).

Structure theorem:
  solutions <-> (delta, constant groups) bijection. delta = the type-1 line
  difference matrix D[i,j] = u_j - u_{j-1}, subject to:
    (a) row sums = 0;  (b) column sums = 0;  (c) anti-diagonal classes
    (i+j) mod g sum to 0, g = gcd(Lx, Ly).
  Constant groups = a_i (Lx of them) + b_j (Ly) + c-cycle constants
  (g of them; the c-graph = the pairing graph of type-0 edges has exactly
  g cycles).
  Dimension identity: [(Lx-1)(Ly-1) - (g-1)] + [Lx+Ly+g] = F+2 = dim K.
  Given delta, the feasible constant set = independent intervals per
  line/cycle => N(delta) = product of the feasible shift counts (closed
  form).
  Strong check: sum over delta of N(delta) = Omega holds exactly on
  4x3 / 5x4 / 6x4 / 6x5 / 3x3 (ternary) / 6x6.

Sampling:
  Stage 1: delta chain with stationary distribution prop. N(delta).
    Proposal = sparse basis combination v = sum (+-)delta_k; s drawn from
    mixed weights (locally s=1/2 dominant + occasional s=3/4 crossing the
    "box corridor"), symmetric => Metropolis min(1, N(delta')/N(delta)).
  Stage 2: for each output sample, draw the constants uniformly given
    delta => exactly uniform on the full solution set.
"""
from __future__ import annotations

import math
import random
from collections import deque

import numpy as np
import scipy.sparse as sp

from metric_hc import build_torus, kernel_zbasis, reduce_basis


# ---------------------------------------------------------------- metadata

def edge_meta(Lx, Ly):
    """(edge_set, faces, types, key, posi). type1: line = key,
    position = posi(j); type2: line = key(j), position = posi(i);
    type0: (key = i, posi = j)."""
    edge_set, faces = build_torus(Lx, Ly)
    E = len(edge_set)
    types = np.zeros(E, dtype=np.int64)
    key = np.full(E, -1, dtype=np.int64)
    posi = np.full(E, -1, dtype=np.int64)
    for ei, e in enumerate(edge_set):
        (u, v) = e
        a_ = u if u[0] == "A" else v
        b_ = v if u[0] == "A" else u
        di = (a_[1] - b_[1]) % Lx
        dj = (a_[2] - b_[2]) % Ly
        if di == 0 and dj == 0:
            types[ei] = 0; key[ei] = a_[1]; posi[ei] = a_[2]
        elif di == 1 and dj == 0:
            types[ei] = 1; key[ei] = a_[1]; posi[ei] = a_[2]
        else:
            types[ei] = 2; key[ei] = a_[2]; posi[ei] = a_[1]
    return edge_set, faces, types, key, posi


def saturated_delta_basis(Lx, Ly):
    """Saturated Z-basis of the legal delta lattice (row / column /
    anti-diagonal class sums = 0; kernel_zbasis unimodular elimination)."""
    g = math.gcd(Lx, Ly)
    n = Lx * Ly
    rows = []
    for i in range(Lx):
        rows.append({i * Ly + j: 1 for j in range(Ly)})
    for j in range(Ly):
        rows.append({i * Ly + j: 1 for i in range(Lx)})
    for k in range(g):
        rows.append({i * Ly + j: 1 for i in range(Lx) for j in range(Ly)
                     if (i + j) % g == k})
    zb, rank = kernel_zbasis(rows, n)
    basis = reduce_basis(zb)
    return [np.array([int(x) for x in v], dtype=np.int64).reshape(Lx, Ly)
            for v in basis]


def _var_of(types, key, posi, a, b, Lx, Ly):
    ta = types[a]
    assert ta == types[b], "mixed-type pair"
    if ta == 1:
        i = int(key[a]); pa, pb = int(posi[a]), int(posi[b])
        j = pa if (pa - pb) % Ly == 1 else pb
        return ("u", i, int(j)), (1 if pa == j else -1)
    if ta == 2:
        j = int(key[a]); pa, pb = int(posi[a]), int(posi[b])
        i = pa if (pa - pb) % Lx == 1 else pb
        return ("w", j, int(i)), (1 if pa == i else -1)
    return ("c", min(a, b), max(a, b)), (1 if a == min(a, b) else -1)


class TwoStageSampler:
    """Two-stage exact uniform sampler (state = the delta difference
    matrix)."""

    # proposal mix: s -> weight (symmetry: each component is symmetric in
    # itself); can be overridden per temperature
    S_WEIGHTS = {1: 0.50, 2: 0.28, 3: 0.14, 4: 0.08}

    def __init__(self, Lx, Ly, alphabet, seed=0, dbasis=None,
                 cache_cap=2_000_000, s_weights=None, h=0.0):
        self.Lx, self.Ly = Lx, Ly
        self.h = float(h)          # metric pressure field: weight prop
        # exp(-h * sum l); h=0 uniform ensemble
        self.alph = np.asarray(sorted(set(int(a) for a in alphabet)),
                               dtype=np.int64)
        self.contiguous = bool(self.alph.max() - self.alph.min() + 1
                               == len(self.alph))
        self.banned = (np.setdiff1d(np.arange(int(self.alph.min()),
                                              int(self.alph.max()) + 1),
                                    self.alph) if not self.contiguous
                       else np.zeros(0, dtype=np.int64))
        self.rng = random.Random(seed)
        self.edge_set, self.faces, self.types, self.key, self.posi = \
            edge_meta(Lx, Ly)
        self.E = len(self.edge_set)
        self.dbasis = (dbasis if dbasis is not None
                       else saturated_delta_basis(Lx, Ly))
        self.K = len(self.dbasis)
        g = math.gcd(Lx, Ly)
        exp_dim = (Lx - 1) * (Ly - 1) - (g - 1)
        assert self.K == exp_dim, f"delta basis dim {self.K} != {exp_dim}"
        self._B3 = np.stack(self.dbasis)               # (K, Lx, Ly)
        key_bytes = Lx * Ly * 8
        # global budget ~60 MB per replica (7 replicas ~0.5 GB): guards
        # against an N-cache memory leak
        self.cache_cap = max(2_000, min(cache_cap, 60_000_000 // key_bytes))
        self._ncache = {}
        self._precompute()
        n0 = self._N(np.zeros((Lx, Ly), dtype=np.int64))
        assert n0 > 0, "delta=0 infeasible?"
        self.delta = np.zeros((Lx, Ly), dtype=np.int64)
        self.n_cur = n0
        self._ncache[self.delta.tobytes()] = n0
        self._lwcache = {}
        self.lw_cur = self.logw(self.delta)
        assert self.lw_cur > -math.inf, "delta=0 weight not finite?"
        self.steps_proposed = 0
        self.steps_accepted = 0
        self.s_hist = {s: 0 for s in self.S_WEIGHTS}
        self.s_acc = {s: 0 for s in self.S_WEIGHTS}

    # ---------------- precompute ----------------
    def _precompute(self):
        Lx, Ly = self.Lx, self.Ly
        types, key, posi = self.types, self.key, self.posi
        adj = {}
        for f in self.faces:
            for (p, q), (r, s) in ((((1, 4), (5, 2))), (((0, 3), (2, 5)))):
                v1, s1 = _var_of(types, key, posi, f[p], f[q], Lx, Ly)
                v2, s2 = _var_of(types, key, posi, f[r], f[s], Lx, Ly)
                adj.setdefault(v1, []).append((v2, s1 * s2))
                adj.setdefault(v2, []).append((v1, s1 * s2))
        # eps[i,j] (structure constant): delta w^(j)_i = eps * delta u^(i)_j
        eps = np.zeros((Lx, Ly), dtype=np.int64)
        for i in range(Lx):
            for j in range(Ly):
                val = {("u", i, j): 1}
                q = deque([("u", i, j)])
                while q:
                    u = q.popleft()
                    for (v, r) in adj.get(u, []):
                        if v not in val:
                            val[v] = r * val[u]
                            q.append(v)
                eps[i, j] = val.get(("w", j, i), 0)
        assert set(eps.flatten().tolist()) <= {-1, 0, 1}
        self._eps = eps
        # per-face c-difference vector as a linear expression:
        # dvec[fi] = dsign[fi] * delta u[fsrc]
        nf = len(self.faces)
        fsrc_i = np.zeros(nf, dtype=np.int64)
        fsrc_j = np.zeros(nf, dtype=np.int64)
        dsign = np.zeros(nf, dtype=np.int64)
        for fi, f in enumerate(self.faces):
            t1 = [e for e in f if types[e] == 1][0]
            t2 = [e for e in f if types[e] == 2][0]
            i, j = int(key[t1]), int(key[t2])
            cvar = ("c", *sorted([e for e in f if types[e] == 0]))
            got = None
            for (v, r) in adj.get(cvar, []):
                if v[0] == "u" and v[1] == i and v[2] == j:
                    got = r
                    break
            if got is None:
                for (v, r) in adj.get(cvar, []):
                    if v[0] == "w" and v[1] == j and v[2] == i:
                        got = r * int(eps[i, j])
                        break
            assert got is not None, f"face{fi} missing its c term"
            fsrc_i[fi], fsrc_j[fi], dsign[fi] = i, j, got
        self._fsrc_i, self._fsrc_j, self._dsign = fsrc_i, fsrc_j, dsign
        # type-0 graph: propagation map Mmap (nt0 x nf) + cycle sums Mcyc
        # + cycle members
        t0_edges = sorted([e for e in range(self.E) if types[e] == 0])
        node = {e: k for k, e in enumerate(t0_edges)}
        nt0 = len(t0_edges)
        gadj = {k: [] for k in range(nt0)}
        for fi, f in enumerate(self.faces):
            cs = sorted([e for e in f if types[e] == 0])
            a, b = node[cs[0]], node[cs[1]]
            gadj[a].append((b, fi, -1))
            gadj[b].append((a, fi, +1))
        Mmap = np.zeros((nt0, nf), dtype=np.int64)
        seen = set()
        comp_of = -np.ones(nt0, dtype=np.int64)
        ncomp = 0
        for s0 in range(nt0):
            if s0 in seen:
                continue
            seen.add(s0)
            comp_of[s0] = ncomp
            q = deque([s0])
            while q:
                uu = q.popleft()
                for (vv, fi, sg) in gadj[uu]:
                    if vv not in seen:
                        seen.add(vv)
                        comp_of[vv] = ncomp
                        Mmap[vv] = Mmap[uu]
                        Mmap[vv, fi] += sg
                        q.append(vv)
            ncomp += 1
        rows = []
        for fi, f in enumerate(self.faces):
            cs = sorted([e for e in f if types[e] == 0])
            a, b = node[cs[0]], node[cs[1]]
            if not np.array_equal(Mmap[b] - Mmap[a],
                                  np.where(np.arange(nf) == fi, -1, 0)):
                r = np.zeros(nf, dtype=np.int64)
                r[fi] += 1
                r -= (Mmap[b] - Mmap[a])
                rows.append(r)
        self._Mcyc = (sp.csr_matrix(np.array(rows, dtype=np.int64))
                      if rows else sp.csr_matrix((0, nf), dtype=np.int64))
        self._Mmap = sp.csr_matrix(Mmap)
        self.t0_edges = t0_edges
        self.node = node
        self._cyc_members = [np.nonzero(comp_of == c)[0]
                             for c in range(ncomp)]
        self.ncyc = ncomp

    # ---------------- N(delta) and profiles ----------------
    def _skeleton(self, delta):
        """(U0 (Lx,Ly), W0 (Ly,Lx), C0 (nt0,)) or None."""
        Lx, Ly = self.Lx, self.Ly
        dw = self._eps * delta
        dvec = self._dsign * delta[self._fsrc_i, self._fsrc_j]
        if self._Mcyc.shape[0] and np.any(self._Mcyc @ dvec != 0):
            return None
        C0 = np.asarray(self._Mmap @ dvec).reshape(-1)
        U0 = np.zeros((Lx, Ly), dtype=np.int64)
        U0[:, 1:] = np.cumsum(delta[:, 1:], axis=1)
        W0 = np.zeros((Ly, Lx), dtype=np.int64)
        W0[:, 1:] = np.cumsum(dw.T[:, 1:], axis=1)
        return U0, W0, C0

    def _shift_counts(self, P):
        """P: (n_lines, .) profile array -> per row #{shift: p+shift inside
        the alphabet}.
        Contiguous alphabet: O(1), fully vectorised; non-contiguous:
        subtract the shifts hitting banned values."""
        lo, hi = int(self.alph.min()), int(self.alph.max())
        pmin = P.min(axis=1).astype(np.int64)
        pmax = P.max(axis=1).astype(np.int64)
        base = (hi - lo + 1) - (pmax - pmin)
        if self.contiguous:
            return np.maximum(base, 0)
        out = np.maximum(base, 0)
        for idx in np.nonzero(out)[0]:
            p = P[idx]
            lo_, hi_ = lo - int(pmin[idx]), hi - int(pmax[idx])
            bad = set()
            for v in self.banned:
                for x in np.unique(p):
                    sv = int(v) - int(x)
                    if lo_ <= sv <= hi_:
                        bad.add(sv)
            out[idx] = max(int(base[idx]) - len(bad), 0)
        return out


    def logN(self, delta):
        if self.h != 0.0:
            return self._logw(delta)
        n = self._N(delta)
        return math.log(n) if n > 0 else -math.inf

    def logw(self, delta):
        """log weight: h=0 -> log N(delta) (count);
        h != 0 -> log N_h(delta) (Boltzmann count)."""
        return self._logw(delta)

    def _valid_shifts(self, p):
        """Legal shift set of a row profile: p+a in the alphabet.
        Contiguous alphabet = contiguous interval."""
        lo, hi = int(self.alph.min()), int(self.alph.max())
        pmin, pmax = int(p.min()), int(p.max())
        rng = range(lo - pmin, hi - pmax + 1)
        if self.contiguous:
            return list(rng)
        return [a for a in rng if np.all(np.isin(p + a, self.alph))]

    def _logw(self, delta):
        """log N_h(delta). At h=0 this reduces bitwise to logN(delta)
        (the count).
        Factorisation: given delta, sum l = sum over lines of
        (mult*shift + profile sum); each line enters the Boltzmann weight
        independently => N_h(delta) = prod over lines of
        sum over legal shifts of exp(-h(mult*a + sum p))."""
        k = None
        if len(self._lwcache) < self.cache_cap:
            k = delta.tobytes()
            v = self._lwcache.get(k)
            if v is not None:
                return v
        sk = self._skeleton(delta)
        if sk is None:
            if k is not None:
                self._lwcache[k] = -math.inf
            return -math.inf
        U0, W0, C0 = sk
        tot = 0.0
        for P, mult in ((U0, self.Ly), (W0, self.Lx)):
            for i in range(P.shape[0]):
                vs = self._valid_shifts(P[i])
                if not vs:
                    return -math.inf
                terms = [-self.h * mult * a for a in vs]
                m = max(terms)
                tot += m + math.log(sum(math.exp(t - m) for t in terms))
                tot -= self.h * float(P[i].sum())
        for memb in self._cyc_members:
            p = C0[memb]
            vs = self._valid_shifts(p)
            if not vs:
                return -math.inf
            mult = len(memb)
            terms = [-self.h * mult * a for a in vs]
            m = max(terms)
            tot += m + math.log(sum(math.exp(t - m) for t in terms))
            tot -= self.h * float(p.sum())
        if k is not None and len(self._lwcache) < self.cache_cap:
            self._lwcache[k] = tot
        return tot

    def _pick_shift(self, p, mult):
        """Draw a shift with the h weights: h=0 uniform (original path),
        h != 0 Boltzmann exp(-h*mult*a)."""
        vs = self._valid_shifts(p)
        assert vs, "no legal shift"
        if self.h == 0.0:
            return vs[self.rng.randrange(len(vs))]
        w = [math.exp(-self.h * mult * a) for a in vs]
        r = self.rng.random() * sum(w)
        acc = 0.0
        for a, wv in zip(vs, w):
            acc += wv
            if r < acc:
                return a
        return vs[-1]

    def _N(self, delta):
        k = None
        if len(self._ncache) < self.cache_cap:
            k = delta.tobytes()
            c = self._ncache.get(k)
            if c is not None:
                return c
        sk = self._skeleton(delta)
        if sk is None:
            if k is not None:
                self._ncache[k] = 0
            return 0
        U0, W0, C0 = sk
        c_all = np.concatenate([
            self._shift_counts(U0), self._shift_counts(W0),
            self._shift_counts(np.stack([C0[m] for m in self._cyc_members])
                               if self._cyc_members else
                               np.zeros((0, 1), dtype=np.int64))])
        tot = math.prod(int(x) for x in c_all) if c_all.size else 1
        if k is not None and len(self._ncache) < self.cache_cap:
            self._ncache[k] = tot
        return tot

    # ---------------- sampling ----------------
    def _pick_s(self):
        r = self.rng.random()
        acc = 0.0
        for s, w in self.S_WEIGHTS.items():
            acc += w
            if r < acc:
                return s
        return 1

    def _propose(self):
        s = min(self._pick_s(), self.K)
        ks = self.rng.sample(range(self.K), s)
        signs = np.array([1 if self.rng.getrandbits(1) else -1
                          for _ in range(s)], dtype=np.int64)
        self._cur_s = s
        return np.tensordot(signs, self._B3[ks], axes=1)

    def step(self):
        v = self._propose()
        d2 = self.delta + v
        self.steps_proposed += 1
        self.s_hist[self._cur_s] += 1
        if self.h != 0.0:
            lw2 = self._logw(d2)
            if lw2 == -math.inf:
                return False
            T = getattr(self, "T", 1.0)
            if T <= 0.0:
                accept = True        # target prop 1 (uniform on the feasible set)
            else:
                dlog = T * (lw2 - self.lw_cur)
                accept = dlog >= 0 or self.rng.random() < math.exp(dlog)
            if accept:
                self.delta = d2
                self.lw_cur = lw2
                self.steps_accepted += 1
                self.s_acc[self._cur_s] += 1
                return True
            return False
        n2 = self._N(d2)
        if n2 > 0:
            T = getattr(self, "T", 1.0)
            if T >= 1.0:
                accept = n2 >= self.n_cur or                     self.rng.random() * self.n_cur < n2
            elif T <= 0.0:
                accept = True          # target prop 1 (uniform on the feasible set)
            else:
                dlog = T * (math.log(n2) - math.log(self.n_cur))
                accept = dlog >= 0 or self.rng.random() < math.exp(dlog)
            if accept:
                self.delta = d2
                self.n_cur = n2
                self.lw_cur = self.logN(self.delta)
                self.steps_accepted += 1
                self.s_acc[self._cur_s] += 1
                return True
        return False

    def sample_constants(self, delta=None):
        """Draw the constants uniformly given delta and assemble the full
        configuration, shape (E,) int64."""
        if delta is None:
            delta = self.delta
        sk = self._skeleton(delta)
        assert sk is not None, "current delta infeasible"
        U0, W0, C0 = sk

        a = np.array([self._pick_shift(U0[i], self.Ly)
                      for i in range(self.Lx)], dtype=np.int64)
        b = np.array([self._pick_shift(W0[j], self.Lx)
                      for j in range(self.Ly)], dtype=np.int64)
        cc = np.array([self._pick_shift(C0[m], len(m))
                       for m in self._cyc_members], dtype=np.int64)
        Cvals = C0.copy()
        for ci, m in enumerate(self._cyc_members):
            Cvals[m] += cc[ci]
        x = np.zeros(self.E, dtype=np.int64)
        t1m = self.types == 1
        t2m = self.types == 2
        t0m = self.types == 0
        x[t1m] = a[self.key[t1m]] + U0[self.key[t1m], self.posi[t1m]]
        x[t2m] = b[self.key[t2m]] + W0[self.key[t2m], self.posi[t2m]]
        x[t0m] = Cvals[[self.node[e] for e in np.nonzero(t0m)[0]]]
        return x

    def sample(self, n_samples, thin=1):
        out = []
        for it in range(n_samples * thin):
            self.step()
            if (it + 1) % thin == 0:
                out.append(self.sample_constants())
        return out


class ParallelTemperingSampler:
    """Parallel-tempering enhanced two-stage sampler: replica target prop
    N(delta)^T; the T=0 replica diffuses freely over the feasible delta
    set, and swaps drive the tail equilibration of the T=1 replica. Output
    from the T=1 replica."""

    def __init__(self, Lx, Ly, alphabet, seed=0,
                 Ts=(0.0, 0.25, 0.45, 0.65, 0.82, 0.93, 1.0), h=0.0):
        self.Ts = tuple(Ts)
        self.nT = len(self.Ts)
        self.reps = []
        for i, T in enumerate(self.Ts):
            if T <= 0.25:
                # cold replicas: flat s weights up to 6 (far-corner "box
                # corridor" connections need s=3-5 combinations)
                w = {1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20, 5: 0.10, 6: 0.10}
            else:
                smax = 2 + int(round(3 * (1.0 - T)))
                w = {s: 1.0 / smax for s in range(1, smax + 1)}
            self.reps.append(TwoStageSampler(
                Lx, Ly, alphabet, seed=seed + 100 * i, s_weights=w, h=h))
        for rep, T in zip(self.reps, self.Ts):
            rep.T = T
        self.rng = random.Random(seed + 999)
        self.Lx, self.Ly, self.E = Lx, Ly, self.reps[0].E
        self.types = self.reps[0].types
        self.alph = self.reps[0].alph
        self.swaps_proposed = 0
        self.swaps_accepted = 0

    def step(self):
        for rep in self.reps:
            rep.step()
        # adjacent swaps (all pairs each step, improves swap throughput)
        for i in range(self.nT - 1):
            a, b = self.reps[i], self.reps[i + 1]
            la, lb = a.logN(a.delta), b.logN(b.delta)
            Ta, Tb = a.T, b.T
            dlog = (Ta - Tb) * (lb - la)
            self.swaps_proposed += 1
            if dlog >= 0 or self.rng.random() < math.exp(dlog):
                a.delta, b.delta = b.delta, a.delta
                a.n_cur, b.n_cur = b.n_cur, a.n_cur
                a.lw_cur, b.lw_cur = b.lw_cur, a.lw_cur
                self.swaps_accepted += 1

    def sample(self, n_samples, thin=1):
        out = []
        hot = self.reps[-1]
        for it in range(n_samples * thin):
            self.step()
            if (it + 1) % thin == 0:
                out.append(hot.sample_constants())
        return out
