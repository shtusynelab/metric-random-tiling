// disk_count.c -- honeycomb metric tilings: open-boundary disk counting
// (column-recurrence DP, binary {0,1})
//
// Convention bitwise-identical to the Python three-way cross-check set
// (disk_lib.count_dp_column / DFS / generic frontier DP):
//   face (a,b): l0=A(a,b), l1=B(a,b), l5=C(a,b-1), l4=B(a,b-1),
//               l3=A(a+1,b-1), l2=C(a+1,b-1)
//   u = B(a,b-1)-B(a,b); l3 = l0-u; l2 = l5+u; all values in {0,1}
// Column DP state = the pairs (A(a+1,b), C(a+1,b-1)) of the next column,
// 4 values per pair -> 4^n packed bits.
// Transfer (scatter, forward): DFS over the B chain; old cell i produces
//   A'_i = A_i-u_i (-> new position i'=i+shift, high bit) and
//   C'_i = C_i+u_i (-> new position (i+1)', low bit);
//   shift = b0-b0n-1 in {0,-1}.
// Transfer (gather, reverse, race-free): for each target state walk the B
//   chain and recover the old positions:
//   A_i = high(d'_{i+shift}) + u_i; C_i = low(d'_{i+shift+1}) - u_i;
//   out-of-range components (dropped boundary values) branch by
//   enumeration over A (each value = one independent configuration).
// Initial: t = B(a,b0-1), 2 branches; prevC = C'_{-1}: when shift==0 and
//   nn>=1 it equals low(d'_0) (carried by the target state), otherwise a
//   dummy value (no branch).
// Counts use __int128 (from R=6 on, Omega ~ 10^24 overflows int64).
//
// Build: gcc -O3 -fopenmp -o disk_count disk_count.c
// Usage: ./disk_count R mode [nthreads] [verbose]   mode: s=scatter g=gather

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef int64_t i64;
typedef __int128 i128;

static int R;
static int ncol;
static int col_a[512], col_n[512], col_b0[512];

static inline int in_disk(int a, int b) {
    int x = a + b; if (x < 0) x = -x;
    int y = a;     if (y < 0) y = -y;
    int z = b;     if (z < 0) z = -z;
    int m = x > y ? x : y;
    if (z > m) m = z;
    return m <= R;
}

static void build_geom(void) {
    int amax = 2 * R + 2;
    ncol = 0;
    for (int a = -amax; a <= amax; a++) {
        int b0 = 0, b1 = 0, has = 0;
        for (int b = -3 * R - 3; b <= 3 * R + 3; b++) {
            if (in_disk(a, b)) { if (!has) { b0 = b; has = 1; } b1 = b; }
        }
        if (has) {
            col_a[ncol] = a; col_n[ncol] = b1 - b0 + 1; col_b0[ncol] = b0;
            ncol++;
        }
    }
}

static void print128(i128 x) {
    if (x == 0) { printf("0\n"); return; }
    char buf[64]; int k = 0;
    while (x > 0) { buf[k++] = '0' + (int)(x % 10); x /= 10; }
    for (int i = k - 1; i >= 0; i--) putchar(buf[i]);
    putchar('\n');
}

// ---------------- scatter (forward) ----------------
typedef struct {
    int n, nn, shift, j0;
    i128 v;
} CtxS;

static i128 *nd_s;

static void rec_s(const CtxS *cx, i64 s, int i, int Bprev, int prevC, i64 nidx) {
    const int n = cx->n, nn = cx->nn, shift = cx->shift;
    if (i == n) {
        if (cx->j0 >= nn) {
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            nd_s[nidx] += cx->v;
            return;
        }
        if (cx->j0 + 1 >= nn) {
            for (int A2 = 0; A2 < 2; A2++) {
                i64 k = nidx | ((i64)((A2 << 1) | prevC)) << (2 * cx->j0);
                #ifdef _OPENMP
                #pragma omp atomic
                #endif
                nd_s[k] += cx->v;
            }
            return;
        }
        if (cx->j0 + 2 >= nn) {
            for (int A2 = 0; A2 < 2; A2++)
            for (int A3 = 0; A3 < 2; A3++)
            for (int C3 = 0; C3 < 2; C3++) {
                i64 k = nidx | ((i64)((A2 << 1) | prevC)) << (2 * cx->j0);
                k |= ((i64)((A3 << 1) | C3)) << (2 * (cx->j0 + 1));
                #ifdef _OPENMP
                #pragma omp atomic
                #endif
                nd_s[k] += cx->v;
            }
            return;
        }
        fprintf(stderr, "ERROR: top width nn-j0=%d\n", nn - cx->j0);
        exit(2);
    }
    int A = (int)((s >> (2 * i + 1)) & 1);
    int C = (int)((s >> (2 * i)) & 1);
    for (int Bn = 0; Bn < 2; Bn++) {
        int u = Bprev - Bn;
        int Ap = A - u, Cp = C + u;
        if ((unsigned)Ap > 1u || (unsigned)Cp > 1u) continue;
        int ip = i + shift;
        i64 nn2 = nidx;
        if (0 <= ip && ip < nn)
            nn2 |= ((i64)((Ap << 1) | prevC)) << (2 * ip);
        rec_s(cx, s, i + 1, Bn, Cp, nn2);
    }
}

// ---------------- gather (reverse, race-free) ----------------
typedef struct {
    int n, nn, shift;
    const i128 *dp;
    i128 *nd;
} CtxG;

static void rec_g(const CtxG *cx, i64 sp, int i, int Bprev, i64 oidx) {
    const int n = cx->n, nn = cx->nn, shift = cx->shift;
    if (i == n) { cx->nd[sp] += cx->dp[oidx]; return; }
    int ipA = i + shift, ipC = ipA + 1;
    for (int Bi = 0; Bi < 2; Bi++) {
        int u = Bprev - Bi;
        int Aopts[2], na = 0;
        if (0 <= ipA && ipA < nn) {
            int A = (int)((sp >> (2 * ipA + 1)) & 1) + u;
            if ((unsigned)A < 2u) Aopts[na++] = A;
        } else {
            for (int A = 0; A < 2; A++) {
                int Ap = A - u;
                if ((unsigned)Ap < 2u) Aopts[na++] = A;
            }
        }
        if (!na) continue;
        int Copts[2], nc = 0;
        if (0 <= ipC && ipC < nn) {
            int C = (int)((sp >> (2 * ipC)) & 1) - u;
            if ((unsigned)C < 2u) Copts[nc++] = C;
        } else {
            for (int C = 0; C < 2; C++) {
                int Cp = C + u;
                if ((unsigned)Cp < 2u) Copts[nc++] = C;
            }
        }
        if (!nc) continue;
        for (int x = 0; x < na; x++)
            for (int y = 0; y < nc; y++) {
                i64 no = oidx | ((i64)Aopts[x] << (2 * i + 1)) | ((i64)Copts[y] << (2 * i));
                rec_g(cx, sp, i + 1, Bi, no);
            }
    }
}

static i128 count(const char *mode, int verbose) {
    int n0 = col_n[0];
    i64 sz = (i64)1 << (2 * n0);
    i128 *dp = (i128 *)malloc((size_t)sz * sizeof(i128));
    if (!dp) { fprintf(stderr, "malloc fail\n"); exit(1); }
    for (i64 i = 0; i < sz; i++) dp[i] = 1;

    for (int ci = 0; ci < ncol; ci++) {
        int n = col_n[ci];
        int nn = (ci + 1 < ncol) ? col_n[ci + 1] : 0;
        int b0 = col_b0[ci], b0n = (ci + 1 < ncol) ? col_b0[ci + 1] : 0;
        int shift = (ci + 1 < ncol) ? (b0 - b0n - 1) : 0;
        i64 szn = (i64)1 << (2 * nn);
        i128 *nd = (i128 *)calloc((size_t)szn, sizeof(i128));
        if (!nd) { fprintf(stderr, "calloc fail szn=%lld\n", (long long)szn); exit(1); }
        i64 old_sz = sz;
        if (mode[0] == 'g') {
            CtxG cx; cx.n = n; cx.nn = nn; cx.shift = shift;
            cx.dp = dp; cx.nd = nd;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 64)
            #endif
            for (i64 sp = 0; sp < szn; sp++) {
                for (int t = 0; t < 2; t++)
                    rec_g(&cx, sp, 0, t, 0);
            }
        } else {
            nd_s = nd;
            CtxS cx; cx.n = n; cx.nn = nn; cx.shift = shift;
            cx.j0 = n + shift;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic, 256)
            #endif
            for (i64 s = 0; s < old_sz; s++) {
                i128 v = dp[s];
                if (v == 0) continue;
                CtxS cx2 = cx; cx2.v = v;
                int npc = (shift == 0 && nn >= 1) ? 2 : 1;
                for (int t = 0; t < 2; t++)
                    for (int q = 0; q < npc; q++)
                        rec_s(&cx2, s, 0, t, q, 0);
            }
        }
        free(dp);
        dp = nd; sz = szn;
        if (verbose) {
            i64 nz = 0;
            for (i64 i = 0; i < sz; i++) if (dp[i]) nz++;
            fprintf(stderr, "  col %d (a=%d n=%d -> nn=%d shift=%d): nz=%lld/%lld\n",
                    ci, col_a[ci], n, nn, shift, (long long)nz, (long long)sz);
        }
    }
    i128 total = 0;
    for (i64 i = 0; i < sz; i++) total += dp[i];
    free(dp);
    return total;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s R mode(s|g) [nthreads] [verbose]\n", argv[0]); return 1; }
    R = atoi(argv[1]);
    const char *mode = argv[2];
#ifdef _OPENMP
    if (argc >= 4) omp_set_num_threads(atoi(argv[3]));
#endif
    int verbose = (argc >= 5) ? atoi(argv[4]) : 0;
    build_geom();
    if (verbose) {
        fprintf(stderr, "R=%d ncol=%d\n", R, ncol);
        for (int i = 0; i < ncol; i++)
            fprintf(stderr, "  [a=%d n=%d b0=%d]\n", col_a[i], col_n[i], col_b0[i]);
    }
    print128(count(mode, verbose));
    return 0;
}
