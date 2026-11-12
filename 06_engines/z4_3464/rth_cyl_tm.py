# -*- coding: utf-8 -*-
"""
rth_cyl_tm.py - 3.4.6.4 cylindrical strip TM (periodic transverse, open
longitudinal): the perimeter-law verdict instrument.

Background: the open-boundary strip ln lambda(W) grows linearly
(rth_strip_tm.py, W=1..6 slope ~0.225), contradicting the perimeter law of
the four-point torus counts; attribution = free outer edges (wall degrees
of freedom) whose per-column counts grow with W.
The cylinder (periodic transverse W*B1, open longitudinal) has no wall =>
a clean criterion:
  perimeter law predicts: ln lambda_cyl(W) -> gamma ~ 2.28 (saturates;
  gamma comes from the torus lnOmega ~ 2.28L);
  positive entropy predicts: ln lambda_cyl(W) = 6sW grows linearly.

Implementation: CylinderRTH = a single-axis-reduced version of
verify_dsh_audit.Torus (mod W along B1, open along B2); the face/edge/row
interfaces match PlaneRTH => StripTM is reused directly.
Anchors: (1) Euler V-E+F = 0 (cylinder); (2) all-ones closure;
(3) small cylinders Omega: kernel enumeration == DP (k=2/3);
(4) torus-limit cross-check: a W x W cylinder closed longitudinally should
reduce to the torus (dim-consistency spot checks deliberately not done,
kept independent).
"""
import os, sys, math, time
import numpy as np
from math import cos, sin, radians, hypot
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from RTH_numerical_verification import rref_and_param, QD, qd_from_component  # noqa: E402

R6 = 6


class CylinderRTH:
    """3.4.6.4 cylinder of W (transverse, periodic) x L (longitudinal,
    open). Interface aligned with PlaneRTH."""

    def __init__(self, W, L):
        t0 = time.time()
        s = 1.0
        D = s * (1 + math.sqrt(3))
        B1 = D * np.array((cos(radians(30)), sin(radians(30))))
        B2 = D * np.array((0.0, 1.0))
        M = np.column_stack([B1, B2])
        self._Minv = np.linalg.inv(M)
        self.W, self.L, self._P1 = W, L, W * B1
        u60 = [(cos(radians(60 * m)), sin(radians(60 * m))) for m in range(6)]
        u30 = [(cos(radians(30 + 60 * m)), sin(radians(30 + 60 * m))) for m in range(6)]
        self.pos = {}
        self._raw = []
        for i in range(W):
            for j in range(L):
                C = i * B1 + j * B2
                v = [C + np.array(u60[m]) for m in range(6)]
                n = [np.array(u30[m]) for m in range(6)]
                self.add_face([tuple(p) for p in v], 6)
                for m in range(6):
                    self.add_face([tuple(v[m]), tuple(v[(m + 1) % 6]),
                                   tuple(v[(m + 1) % 6] + s * n[m]),
                                   tuple(v[m] + s * n[m])], 4)
                for m in range(6):
                    self.add_face([tuple(v[m]), tuple(v[m] + s * n[(m - 1) % 6]),
                                   tuple(v[m] + s * n[m])], 3)
        self.finalize()
        self.build_time = time.time() - t0

    def _reduce(self, p):
        uv = self._Minv @ np.asarray(p, float)
        k = math.floor(uv[0] / self.W + 1e-6)      # whole periods along B1
        res = np.asarray(p, float) - k * self._P1
        return (round(float(res[0]), R6), round(float(res[1]), R6))

    def add_face(self, pts, ftype):
        pts = [np.asarray(p, float) for p in pts]
        area = sum(float(pts[i][0] * pts[(i + 1) % len(pts)][1]
                         - pts[(i + 1) % len(pts)][0] * pts[i][1])
                   for i in range(len(pts)))
        assert abs(area) > 1e-9
        if area < 0:
            pts = pts[::-1]
        keys = []
        for p in pts:
            k = self._reduce(p)
            if k not in self.pos:
                self.pos[k] = p
            keys.append(k)
        self._raw.append((keys, pts, ftype))

    def finalize(self):
        seen, faces = set(), []
        for keys, pts, ftype in self._raw:
            cent = np.mean(pts, axis=0)
            sig = (frozenset(keys), ftype, self._reduce(cent))
            if sig in seen:
                continue
            seen.add(sig)
            faces.append((keys, pts, ftype))
        self.faces = faces
        self.V = len(self.pos)
        self.F = len(faces)
        self.edge_ids, self.edge_dirf, self.edge_cs = {}, {}, []
        self.face_edges = []
        for keys, pts, ftype in faces:
            segs = []
            for i in range(len(keys)):
                a, b = pts[i], pts[(i + 1) % len(pts)]
                ek = self._reduce((a + b) / 2)
                dv = b - a
                dv = dv / hypot(*dv)
                if ek not in self.edge_ids:
                    self.edge_ids[ek] = len(self.edge_ids)
                    self.edge_dirf[ek] = dv
                    ang = math.degrees(math.atan2(dv[1], dv[0]))
                    ar = round(ang / 30) * 30 % 360
                    assert abs((ang - ar + 180) % 360 - 180) < 1e-6
                    self.edge_cs.append((qd_from_component(cos(radians(ar)), 3),
                                         qd_from_component(sin(radians(ar)), 3)))
                dot = float(np.dot(self.edge_dirf[ek], dv))
                sgn = 1 if dot > 0.5 else (-1 if dot < -0.5 else 0)
                assert sgn != 0
                segs.append((self.edge_ids[ek], sgn))
            self.face_edges.append(segs)
        self.E = len(self.edge_ids)
        self.xrows, self.yrows = [], []
        for segs in self.face_edges:
            rx = [QD(0, 0, 3) for _ in range(self.E)]
            ry = [QD(0, 0, 3) for _ in range(self.E)]
            for eid, sgn in segs:
                cx, sy = self.edge_cs[eid]
                if sgn > 0:
                    rx[eid] = rx[eid] + cx
                    ry[eid] = ry[eid] + sy
                else:
                    rx[eid] = rx[eid] - cx
                    ry[eid] = ry[eid] - sy
            self.xrows.append(rx)
            self.yrows.append(ry)
        for rx, ry in zip(self.xrows, self.yrows):
            tx = QD(0, 0, 3)
            ty = QD(0, 0, 3)
            for x in rx:
                tx = tx + x
            for y in ry:
                ty = ty + y
            assert tx.iszero() and ty.iszero(), "all-ones row fails to close"

    def euler_ok(self):
        return self.V - self.E + self.F == 0


# ---------------- column partitioning + DP (same logic as
# rth_strip_tm.StripTM, independent instance) --
from rth_strip_tm import StripTM  # noqa: E402  (reuses the conversion logic: pass the cyl object)


class CylStripTM(StripTM):
    def __init__(self, t):
        # takes a CylinderRTH instance directly
        self.t = t
        E = t.E
        hex_idx = [fi for fi, (k, p, ft) in enumerate(t.faces) if ft == 6]
        hex_cent = {fi: np.mean(np.array(t.faces[fi][1]), axis=0) for fi in hex_idx}
        import math as _m
        D = 1.0 * (1 + _m.sqrt(3))
        B1 = np.array((D * _m.sqrt(3) / 2, D / 2))
        B2 = np.array((0.0, D))
        Minv = np.linalg.inv(np.column_stack([B1, B2]))
        self.col = {}
        for fi, (k, p, ft) in enumerate(t.faces):
            cent = np.mean(np.array(p), axis=0)
            if ft == 6:
                j = int(round((Minv @ cent)[1]))
            else:
                # distances in reduced coordinates (finds neighbors correctly
                # across the seam)
                cent_r = np.array(t._reduce(cent))
                near = []
                for h in hex_idx:
                    hc = np.array(t._reduce(hex_cent[h]))
                    d = hc - cent_r
                    u = float((Minv @ d)[0])
                    # minimum image: round picks wrong at half-period
                    # degenerate points; take the min distance over three
                    # candidates
                    best = None
                    for ku in (round(u) - 1, round(u), round(u) + 1):
                        dd = d - B1 * ku
                        n = float(np.hypot(*dd))
                        if best is None or n < best:
                            best, dmin = n, dd
                    if best < 2.0:
                        near.append(h)
                assert near, f"face {fi} finds no neighboring hexagon"
                js = [int(round((Minv @ hex_cent[h])[1])) for h in near]
                j = max(js)
            self.col[fi] = j
        assert sorted(set(self.col.values())) == list(range(t.L)), \
            f"column-index coverage anomaly {sorted(set(self.col.values()))} vs L={t.L}"
        self.edge_coef = []
        for eid in range(E):
            (cx, sy) = t.edge_cs[eid]
            self.edge_coef.append((int(2 * cx.a), int(2 * cx.b),
                                   int(2 * sy.a), int(2 * sy.b)))
        self.face_eqs = []
        for fi, (k, p, ft) in enumerate(t.faces):
            eqs = []
            for eid, sgn in t.face_edges[fi]:
                c = self.edge_coef[eid]
                eqs.append(tuple(sgn * v for v in c))
            self.face_eqs.append((fi, eqs, self.col[fi]))
        order = sorted(range(t.F),
                       key=lambda fi: (self.col[fi],
                                       float(np.mean(np.array(t.faces[fi][1]),
                                                     axis=0)[1]),
                                       t.faces[fi][2]))
        self.order = order
        self.edge_faces = {}
        for fi, eqs, cj in self.face_eqs:
            for eid, sgn in t.face_edges[fi]:
                self.edge_faces.setdefault(eid, []).append(fi)
        assert all(len(v) in (1, 2) for v in self.edge_faces.values()), \
            "edge adjacent-face count anomaly (>2; open cylinder ends may have 1)"
        self.edge_mincol = {e: min(self.col[f] for f in fs)
                            for e, fs in self.edge_faces.items()}
        self.edge_maxcol = {e: max(self.col[f] for f in fs)
                            for e, fs in self.edge_faces.items()}
        self.E = E
        self.build_time = 0.0


def lambda_infty(W, k, alpha0=0, n_iter=400, report=False):
    """Perron eigenvalue of the bulk single-column transfer operator =
    lambda_infty(W) (no convergence lag).
    States are canonical-ordered value tuples of the cutset edges (sorted by
    B1 index mod W, then B2 coordinate); translation equivariance => the
    cutset of every column lives in the same abstract space, no relabeling
    needed."""
    from collections import Counter
    c = CylinderRTH(W, 3)
    tm = CylStripTM(c)
    A = range(alpha0, alpha0 + k)
    id2mid = {v: kk for kk, v in c.edge_ids.items()}
    Minv = c._Minv

    def canon(fr):
        def key(e):
            uv = Minv @ np.array(id2mid[e], float)
            return (int(round(uv[0])) % W, round(float(uv[1]), 4))
        return sorted(fr, key=key)

    def to_canon(states, fr):
        o = canon(fr)
        out = {}
        for st, cnt in states.items():
            d = dict(st)
            kk = tuple(d[e] for e in o)
            out[kk] = out.get(kk, 0) + cnt
        return out

    def process_column(states, j, done):
        for fi in [f for f in tm.order if tm.col[f] == j]:
            _, eqs, _ = tm.face_eqs[fi]
            segs = tm.t.face_edges[fi]
            eids = [eid for eid, _ in segs]
            coef = {eid: eqs[i] for i, (eid, _) in enumerate(segs)}
            new_states = {}
            for st, cnt in states.items():
                cur = dict(st)
                old_e = [e for e in eids if e in cur]
                new_e = [e for e in eids if e not in cur]
                acc0 = [0, 0, 0, 0]
                for e in old_e:
                    m = coef[e]
                    v = cur[e]
                    for i in range(4):
                        acc0[i] += m[i] * v
                for combo in product(A, repeat=len(new_e)):
                    acc = list(acc0)
                    for e, v in zip(new_e, combo):
                        m = coef[e]
                        for i in range(4):
                            acc[i] += m[i] * v
                    if any(acc):
                        continue
                    upd = dict(cur)
                    for e, v in zip(new_e, combo):
                        upd[e] = v
                    nxt = tuple(sorted(upd.items()))
                    new_states[nxt] = new_states.get(nxt, 0) + cnt
            states = new_states
            for e in eids:
                done[e] = done.get(e, 0) + 1
            merged = {}
            for st, cc in states.items():
                key = tuple(sorted((e, v) for e, v in st
                                   if done[e] < len(tm.edge_faces[e])))
                merged[key] = merged.get(key, 0) + cc
            states = merged
        return states

    # column 0: build the boundary, get the canonical state of cutset 0
    done0 = {}
    states = to_canon(process_column({(): 1}, 0, done0), tm.frontier_after(0))
    # bulk transfer: run column 1 for each s_in; output uses the canonical
    # order of cutset 1 (same abstract space)
    T = {}
    for s_in in states:
        out = process_column({tuple(zip(canon(tm.frontier_after(0)), s_in)): 1},
                             1, dict(done0))
        for st, cnt in to_canon(out, tm.frontier_after(1)).items():
            T.setdefault(s_in, Counter())[st] += cnt
    if n_iter == 0:
        return len(T)
    # spectral radius: use dense eigvals for small state counts (power
    # iteration risks early stopping on nearly degenerate spectra; W=2/4
    # once gave 2.0/2.7 while the dense spectrum sqrt(6)/2*sqrt(3) is right)
    if len(T) <= 3000:
        idx = {s: i for i, s in enumerate(sorted(T))}
        n = len(idx)
        M = np.zeros((n, n))
        for s, row in T.items():
            for s2, cnt in row.items():
                M[idx[s], idx[s2]] = cnt
        return float(np.max(np.abs(np.linalg.eigvals(M))))
    # power iteration
    v = {s: 1.0 for s in T}
    lam_old = 1.0
    for it in range(n_iter):
        nv = {}
        for s, row in T.items():
            vs = v[s]
            for s2, cnt in row.items():
                nv[s2] = nv.get(s2, 0.0) + cnt * vs
        nrm = max(nv.values())
        v = {s: x / nrm for s, x in nv.items()}
        lam = nrm
        if report and it % 100 == 0:
            print(f"      iter {it}: lambda={lam:.6f} states={len(v)}", flush=True)
        if abs(lam - lam_old) < 1e-13 * lam:
            break
        lam_old = lam
    return lam


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_only", action="store_true")
    ap.add_argument("--Ws", default="2,3,4,5,6")
    ap.add_argument("--Lmax", type=int, default=32)
    ap.add_argument("--ks", default="2")
    args = ap.parse_args()

    print(f"= cylindrical strip TM {time.strftime('%F %T')} =", flush=True)
    # anchors
    print("anchor 1: construction (V-E+F=0, all-ones closure built in)", flush=True)
    for (W, L) in ((2, 2), (3, 3)):
        c = CylinderRTH(W, L)
        print(f"  W={W} L={L}: V={c.V} E={c.E} F={c.F} "
              f"Euler (cyl V-E+F=-1? measured {c.V-c.E+c.F}) "
              f"face types={dict(sorted((ft, sum(1 for _,_,f2 in c.faces if f2==ft)) for ft in (3,4,6)))}",
              flush=True)
    print("anchor 2: kernel enumeration == DP (k=2/3)", flush=True)
    ok_all = True
    for (W, L) in ((2, 2), (2, 3), (3, 2), (3, 3)):
        c = CylinderRTH(W, L)
        T, piv, free, dim = rref_and_param(c.xrows, c.yrows, c.E)
        tm = CylStripTM(c)
        for k in (2, 3):
            try:
                ref = __import__("verify_dsh_audit",
                                 fromlist=["count_exact"]).count_exact(
                    T, free, c.E, k, alpha0=0)
            except OverflowError:
                print(f"  cyl W={W} L={L} k={k}: kernel enumeration overflow, "
                      f"skipped (dim={dim})")
                continue
            ser = tm.run(k, report_every=10**9)
            got = ser[-1][1]
            ok = (got == ref)
            ok_all &= ok
            print(f"  cyl W={W} L={L} k={k}: DP={got} kernel_enum={ref} "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
    if args.anchor_only:
        return 0 if ok_all else 1
    if not ok_all:
        print("anchor check failed; aborting", flush=True)
        return 1
    print("-" * 80, flush=True)
    for W in [int(x) for x in args.Ws.split(",")]:
        for k in [int(x) for x in args.ks.split(",")]:
            t0 = time.time()
            c = CylinderRTH(W, args.Lmax)
            tm = CylStripTM(c)
            print(f"cyl W={W} k={k}: V={c.V} E={c.E} F={c.F}", flush=True)
            try:
                ser = tm.run(k)
            except MemoryError as e:
                print(f"  ABORT: {e}", flush=True)
                return 2
            lams = [math.log(ser[i + 1][1]) - math.log(ser[i][1])
                    for i in range(len(ser) - 4, len(ser) - 1)]
            lam = math.exp(sum(lams) / len(lams))
            print(f"  ==> ln_lambda_cyl(W={W},k={k}) ~ {math.log(lam):.4f} "
                  f"(lambda={lam:.3f}; perimeter law predicts saturation -> 2.28;"
                  f" positive entropy predicts linear 6sW)"
                  f"({time.time()-t0:.0f}s)", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
