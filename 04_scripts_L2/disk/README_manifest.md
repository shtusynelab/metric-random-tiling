# scripts/disk -- exact-counting engine for the planar open-boundary disk

| File | Role |
|---|---|
| `disk_lib.py` | Python reference implementation: torus/disk graph construction, full DFS enumeration, generic key-merged frontier DP, column-recurrence DP, torus column trace |
| `blocked_multi.py` | blocked DFS (pure enumeration inside each block + interface-state matching), a cross-check path independent of the DP |
| `disk_count.c` | C `__int128` column DP (required for R=6/7): `s` = scatter version (OpenMP atomics), `g` = gather version (blocked by target state, race-free); build with `gcc -O3 -std=gnu11 -fopenmp -o disk_count disk_count.c` |

Conventions: face (a,b) slot order [F0(a,b), F1(a,b), F2(a+1,b-1), F0(a+1,b-1), F1(a,b-1), F2(a,b-1)];
closure eqA: l1+l2 = l4+l5, eqB': l0+l5 = l2+l3; disk = cube coords (a+b,-a,-b) with max-norm <= R (N = 3R(R+1)+1).
Exact counts delivered at `03_data/uniform/disk_counts_exact.csv` (development layout: `data/uniform/`);
audit record: the development-tree verification note "disk-face counting verdict for the planar
open-boundary entropy density" dated 2026-10-15 (anchors and audit tables).
