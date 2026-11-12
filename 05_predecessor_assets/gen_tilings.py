# -*- coding: utf-8 -*-
"""Generate and save the four tilings illustrating the random-tiling theory
introduction document.

(a) periodic honeycomb (regular hexagons); (b) Penrose quasiperiodic tiling
(substitution method); (c) random lozenge tiling (honeycomb-graph perfect
matching + flip Monte Carlo); (d) Voronoi irregular network (rule-free
random tiling, amorphous control).

Output: tilings.pkl next to this script (per tiling: edge set Nx4 in a
circular window).
"""
import math
import pickle
import numpy as np

PHI = (1 + 5 ** 0.5) / 2


# ----------------------------------------------------------------------
# (a) periodic honeycomb: regular hexagonal tiling, circular window
# ----------------------------------------------------------------------
def periodic_honeycomb(radius=12.0, edge=1.0):
    s = edge
    dx, dy = 1.5 * s, math.sqrt(3) / 2 * s
    edges = {}
    n = int(radius / dx) + 3
    for q in range(-n, n + 1):
        for r in range(-n, n + 1):
            cx, cy = q * dx, (2 * r + (q % 2)) * dy
            if cx * cx + cy * cy > (radius + s) ** 2:
                continue
            verts = []
            for i in range(6):
                a = math.pi / 180 * (60 * i + 30)
                verts.append((cx + s * math.cos(a), cy + s * math.sin(a)))
            for i in range(6):
                a, b = verts[i], verts[(i + 1) % 6]
                key = tuple(sorted([(round(a[0], 6), round(a[1], 6)),
                                    (round(b[0], 6), round(b[1], 6))]))
                edges[key] = (a[0], a[1], b[0], b[1])
    out = []
    R2 = radius * radius
    for (k1, k2), v in edges.items():
        if k1[0] ** 2 + k1[1] ** 2 <= R2 and k2[0] ** 2 + k2[1] ** 2 <= R2:
            out.append(v)
    return np.array(out)


# ----------------------------------------------------------------------
# (b) Penrose P3 substitution (Robinson triangles)
# ----------------------------------------------------------------------
def penrose_half_triangles(n_gen=7):
    tris = []
    for i in range(10):
        B = complex(math.cos((2 * i - 1) * math.pi / 10),
                    math.sin((2 * i - 1) * math.pi / 10))
        C = complex(math.cos((2 * i + 1) * math.pi / 10),
                    math.sin((2 * i + 1) * math.pi / 10))
        if i % 2 == 0:                     # mirror every other wedge (substitution is chiral)
            B, C = C, B
        tris.append((0, complex(0, 0), B, C))
    for _ in range(n_gen):
        out = []
        for color, A, B, C in tris:
            if color == 0:                 # fat half
                P = A + (B - A) / PHI
                out.append((0, C, P, B))
                out.append((1, P, C, A))
            else:                          # thin half
                Q = B + (A - B) / PHI
                R = B + (C - B) / PHI
                out.append((1, R, C, A))
                out.append((1, Q, R, B))
                out.append((0, R, Q, A))
        tris = out
    return tris


def penrose_edges(radius=12.0, n_gen=7):
    """Visible edges = non-diagonal edges; the diagonal is the fat
    half-triangle's longest side / the thin half-triangle's shortest side."""
    tris = penrose_half_triangles(n_gen)
    diag = set()
    cnt = {}
    for color, A, B, C in tris:
        pts = [(A.real, A.imag), (B.real, B.imag), (C.real, C.imag)]
        keys = []
        lens = []
        for i in range(3):
            k = tuple(sorted([(round(pts[i][0], 9), round(pts[i][1], 9)),
                              (round(pts[(i + 1) % 3][0], 9),
                               round(pts[(i + 1) % 3][1], 9))]))
            keys.append(k)
            lens.append(math.dist(k[0], k[1]))
        idx = int(np.argmin(lens)) if color == 0 else int(np.argmax(lens))
        diag.add(keys[idx])
        for i, k in enumerate(keys):
            if i != idx:
                cnt[k] = cnt.get(k, 0) + 1
    lens = [math.dist(k1, k2) for (k1, k2) in cnt]
    med = float(np.median(lens))
    segs = []
    R2 = radius * radius
    for (k1, k2), c in cnt.items():
        x1, y1 = k1[0] / med, k1[1] / med
        x2, y2 = k2[0] / med, k2[1] / med
        if x1 * x1 + y1 * y1 <= R2 and x2 * x2 + y2 * y2 <= R2:
            segs.append((x1, y1, x2, y2))
    return np.array(segs)


# ----------------------------------------------------------------------
# (c) random lozenge tiling: honeycomb-graph perfect matching + flip MC (torus)
# ----------------------------------------------------------------------
class LozengeTiling:
    """Triangular-lattice torus indexed by (i,j,k); k=0: {T(i,j),T(i+1,j),
    T(i,j+1)}; k=1: {T(i+1,j),T(i+1,j+1),T(i,j+1)}; e1=(1,0), e2=(1/2,sqrt3/2).
    Honeycomb adjacency: (i,j,0) ~ (i,j-1,1) [e1 edge], (i-1,j,1) [e2 edge],
    (i,j,1) [e1-e2 edge].
    flip = alternating 6-cycle of the honeycomb graph (the deterministic
    search from a fixed vertex guarantees the same cycle from either
    direction)."""

    def __init__(self, Nu, Nv, seed=0):
        self.Nu, self.Nv = Nu, Nv
        self.rng = np.random.default_rng(seed)
        self.adj = {}
        for i in range(Nu):
            for j in range(Nv):
                a = (i, j, 0)
                self.adj[a] = [((i, j - 1, 1)), ((i, j, 1)), ((i - 1, j, 1))]
                b = (i, j, 1)
                self.adj[b] = [((i, j + 1, 0)), ((i, j, 0)), ((i + 1, j, 0))]
        self.match = {}
        self.match_edges = {}
        for i in range(Nu):
            for j in range(Nv):
                a, b = (i, j, 0), (i, j, 1)
                self.match[a] = b
                self.match[b] = a
                self.match_edges[tuple(sorted([a, b]))] = True

    def random_perfect_matching(self):
        """Build a perfect matching by Kuhn's augmenting algorithm in random
        order (random initial state; ensemble uniformity is provided by the
        flip MC)."""
        import sys
        sys.setrecursionlimit(50000)
        rng = self.rng
        k0 = sorted([t for t in self.adj if t[2] == 0])
        adj_rand = {}
        for x in self.adj:
            lst = [self._wrap(n) for n in self.adj[x]]
            perm = rng.permutation(len(lst))
            adj_rand[x] = [lst[i] for i in perm]
        match = {}                            # k1 vertex -> k0 vertex

        def try_kuhn(x, visited):
            for y in adj_rand[x]:
                if y in visited:
                    continue
                visited.add(y)
                if y not in match or try_kuhn(match[y], visited):
                    match[y] = x
                    return True
            return False

        for t0 in [k0[i] for i in rng.permutation(len(k0))]:
            try_kuhn(t0, set())
        assert len(match) == len(k0), 'greedy matching failed'
        self.match = {}
        self.match_edges = {}
        for y, x in match.items():
            self.match[x] = y
            self.match[y] = x
            self.match_edges[tuple(sorted([x, y]))] = True

    def _wrap(self, t):
        return (t[0] % self.Nu, t[1] % self.Nv, t[2])

    def find_flip(self, t0):
        """Search from t0 for the alternating 6-cycle
        t0-E-n1-M-n2-E-n3-M-n4-E-n5-M-t0.
        Returns (m0, n1, n3, n5) or None; the deterministic search selects
        the same cycle from either direction."""
        m0 = self.match[t0]
        for n1 in self.adj[t0]:
            n1 = self._wrap(n1)
            if n1 == m0:
                continue
            n2 = self.match[n1]
            for n3 in self.adj[n2]:
                n3 = self._wrap(n3)
                if n3 == n1:
                    continue
                n4 = self.match[n3]
                for n5 in self.adj[n4]:
                    n5 = self._wrap(n5)
                    if n5 == n3:
                        continue
                    if self.match[n5] == t0:
                        return (m0, n1, n3, n5)
        return None

    def apply_flip(self, flip):
        """Cycle t0-E-n1-M-n2-E-n3-M-n4-E-n5-M-t0: remove the 3 M edges,
        add the 3 E edges."""
        m0, n1, n3, n5 = flip
        t0 = self.match[m0]
        n2, n4 = self.match[n1], self.match[n3]
        remove = [tuple(sorted([n1, n2])), tuple(sorted([n3, n4])),
                  tuple(sorted([n5, t0]))]
        add = [(t0, n1), (n2, n3), (n4, n5)]
        for e in remove:
            del self.match_edges[e]
        for u, v in remove:
            del self.match[u]
            del self.match[v]
        for u, v in add:
            self.match[u] = v
            self.match[v] = u
            self.match_edges[tuple(sorted([u, v]))] = True

    def run_mc(self, n_sweeps=60):
        keys = sorted(self.adj.keys())
        n = len(keys)
        for _ in range(n_sweeps):
            idx = self.rng.integers(0, n, size=n)
            for t in idx:
                t0 = keys[t]
                f = self.find_flip(t0)
                if f is not None:
                    self.apply_flip((f[0], f[1], f[2], f[3]))

    def orientation_stats(self):
        """Counts of the three lozenge orientations (MC convergence check:
        should approach equal shares)."""
        e1 = (1.0, 0.0)
        e2 = (0.5, math.sqrt(3) / 2)
        dmap = {(0, 0): 0, (self.Nu - 1, 0): 1, (0, self.Nv - 1): 2}
        counts = [0, 0, 0]
        for (i, j, k), m in self.match.items():
            if k != 0:
                continue
            d = ((m[0] - i) % self.Nu, (m[1] - j) % self.Nv)
            counts[dmap[d]] += 1
        return counts

    def visible_edges(self, center, radius):
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.5, math.sqrt(3) / 2])
        d3 = e1 - e2
        R2 = radius * radius
        segs = []
        Nu, Nv = self.Nu, self.Nv
        for i in range(Nu):
            for j in range(Nv):
                T = i * e1 + j * e2
                for d, pair in [(e1, ((i, j, 0), (i, j - 1, 1))),
                                (e2, ((i, j, 0), (i - 1, j, 1))),
                                (d3, ((i, j, 0), (i, j, 1)))]:
                    e0 = T + d
                    if (np.dot(T - center, T - center) > R2
                            or np.dot(e0 - center, e0 - center) > R2):
                        continue
                    if tuple(sorted(pair)) in self.match_edges:
                        continue
                    segs.append((T[0] - center[0], T[1] - center[1],
                                 T[0] + d[0] - center[0], T[1] + d[1] - center[1]))
        return np.array(segs)


def random_lozenge(radius=12.0, Nu=48, Nv=48, seed=11, sweeps=150):
    t = LozengeTiling(Nu, Nv, seed=seed)
    t.random_perfect_matching()
    print('  after random init:', t.orientation_stats())
    t.run_mc(sweeps)
    print('  after MC:', t.orientation_stats())
    center = np.array([Nu / 2 + 0.25, Nv / 2 * math.sqrt(3) / 2 + 0.25])
    return t.visible_edges(center, radius)


# ----------------------------------------------------------------------
# (d) Voronoi irregular network
# ----------------------------------------------------------------------
def voronoi_network(radius=12.0, seed=23):
    from scipy.spatial import Voronoi
    rng = np.random.default_rng(seed)
    n_pts = int(4.5 * (3.6 * radius) ** 2)
    pts = rng.random((n_pts, 2)) * 3.6 * radius - 1.8 * radius
    vor = Voronoi(pts)
    segs = []
    lens = []
    R2 = radius * radius
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        a, b = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
        if a[0] ** 2 + a[1] ** 2 > R2 or b[0] ** 2 + b[1] ** 2 > R2:
            continue
        segs.append((a[0], a[1], b[0], b[1]))
        lens.append(math.dist(a, b))
    med = float(np.median(lens))
    out = []
    for s in segs:
        x1, y1, x2, y2 = [v / med for v in s]
        if x1 * x1 + y1 * y1 <= R2 and x2 * x2 + y2 * y2 <= R2:
            out.append((x1, y1, x2, y2))
    return np.array(out)


# ----------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    R = 12.0
    data = {}
    data['honeycomb'] = periodic_honeycomb(radius=R)
    print('penrose...')
    data['penrose'] = penrose_edges(radius=R)
    print('lozenge MC...')
    data['lozenge'] = random_lozenge(radius=R)
    data['voronoi'] = voronoi_network(radius=R)

    with open(__file__.replace('gen_tilings.py', 'tilings.pkl'), 'wb') as f:
        pickle.dump(data, f)

    names = [('honeycomb', 'a periodic honeycomb'),
             ('penrose', 'b Penrose'),
             ('lozenge', 'c random lozenge'),
             ('voronoi', 'd Voronoi')]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6))
    for ax, (key, label) in zip(axes, names):
        segs = data[key]
        for x1, y1, x2, y2 in segs:
            ax.plot([x1, x2], [y1, y2], 'k-', lw=0.8)
        ax.set_aspect('equal')
        ax.set_title(f'{label}  ({len(segs)} edges)', fontsize=9)
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
    fig.tight_layout()
    fig.savefig(__file__.replace('gen_tilings.py', 'check_tilings.png'), dpi=110)
    print('saved. edge counts:', {k: len(v) for k, v in data.items()})
