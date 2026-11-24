# Reproducibility Supplement

**Metric random tilings: counted geometric disorder and its diffraction**

Manuscript companion. Section, figure and table numbers refer to the accompanying Letter and Supplemental Material.

## RS1. The computational register

*Exact arithmetic and guards.* Honeycomb/3.4.6.4 geometries are built in ℚ(√2), ℚ(√3); every matrix reduction is verified by three independent rank computations (Fraction elimination, two modular primes, float SVD); count consumers assert Euler-type identities (V = 2F, E = 3F on tori) before counting.

*Counting ladder.* Engines are organized in three tiers, ordered by cost and reach; every tier is anchored in the tier below it, and Table R1 maps each engine to its reach, usage, and verification.

*Tier 1: exact enumeration.* DFS over partial edge assignments with integer kernel parametrization, prefix-blocked parallelism, and brute-force 3^E on the smallest systems. Reach F ≲ 20; role: verification anchors and small exact counts (honeycomb 2×2–5×2).

*Tier 2: transfer matrices, traces, elimination (the exact workhorse).* (Newman & Henley 1995) A keyed dynamic program over partial assignments (states merged by key before vectorization, since path counts are not state counts) builds the single-column strip operator T. Strips are counted by iterating T; the torus count is the exact trace Ω(L_x×L_y) = Tr(T^{L_x}), computed from one operator for all lengths (honeycomb: |S| ≤ 3^6 = 729, sub-second per size, series to L_x = 20). At scale T is held in CSR sparse form and the trace is evaluated in chunked unit-vector blocks with per-batch/segment checkpoints and an integrality assert on every partial sum (3.4.6.4: |S| = 2^{2L+1}; the L = 10 point = 4 002 790 772 868 took 36 320 s, ~215 GB peak). Systems whose constraints are naturally face-based rather than cut-based (4.6.12) are counted in the same exact tier by constraint-space elimination: bucket elimination over elimination orderings, and a parameterized DFS with exact rational back-substitution where the bucket width explodes. Closed forms replace counting where theorems provide them (square Ω = k^{L_x+L_y}; Penrose strips Ω = k^{#strips}; honeycomb binary/non-arithmetic classes Ω = k^{L_x+L_y+1} on coprime tori) and are verified bitwise at every measured size.

*Cylinder spectral radii.* Wide-strip and face-constraint systems are adjudicated by the cylinder transfer-matrix spectral radius ρ(W) (the exact asymptotic per-column growth rate, with no finite-length transient), computed on explicit sparse operators with ARPACK (leading spectra are typically degenerate: ± pairs, W-th roots-of-unity families, accidental multiplicities; power iteration is inadmissible). The verdict discipline has four gates, each once omitted at cost: (i) machine-bitwise anchors: frontier DP reproduces exact torus counts, and the full block product from empty head to empty tail equals the exact count; (ii) head-side purity: K-sample true-head closures, excluding unreachable-class inflation; (iii) tail connectivity: reverse-BFS trunk restriction with a restricted Perron computation (a Perron value over states that cannot reach the empty tail overestimates); (iv) known-answer calibration: the whole procedure reproduces the exactly known W = 2 rate with zero deviation. The honeycomb trace criterion and its tested domain are given in SI S5.1.

*Tier 3: calibrated stochastic estimators.* Deployed only where exact counts cap out, and only under an anchor ladder. (a) *Knuth-type unbiased tree sampling* (Knuth 1975) (two-step lookahead): raw estimates miss by 40–62% at Ω level (heavy-tailed weights) but only ≈ 0.01–0.02 at entropy-density level, and values are calibrated against exact anchors (24-task calibration, deviations +0.0001…−0.0049, statistical error ±0.012; RS1). It is deployed on the 4.6.12 ternary ℤ series (Table R4). It becomes ineffective on the 4.8.8 Λ lattice at W ≥ 3: the same heavy tails. (b) *Hutchinson stochastic trace* (Hutchinson 1990) estimates Tr(T^L) = ⟨vᵀ T^L v⟩ over Rademacher probes: unbiased by the trace identity, and reliable precisely where constraint locking compresses the transfer matrix so probe statistics stay near-Gaussian (kurtosis ≤ 3.3 measured across 3.4.6.4 L = 2…8). Deployment requires a calibration pass at known-answer sizes; it fails otherwise (honeycomb binary L_y = 5: probe variance diverges). Deployed: 4.8.8 Λ W = 5 (20 000 probes × 8 seeds, ± 0.0022%).

*Verification protocol (cross-cutting).* (i) An anchor ladder ties every stochastic estimator to exact values at overlapping sizes, and the exact tiers to brute force (three-way DFS/DP/brute agreement on all narrow-strip cases). (ii) Independent code paths (DFS enumeration, keyed dynamic program, sparse trace) agree bitwise at every overlapping size (729/1017/22 767 on the ternary class). (iii) Exact symmetries are asserted at run time: transposition Ω(L_x×L_y) = Ω(L_y×L_x), mirror-pair equality under 2↔3 (edge-slot labels), state-count laws (|S| = 2^{2L+1} for the 3.4.6.4 cylinder), Euler-type identities (V = 2F, E = 3F) before counting. (iv) Expensive extrapolations are guarded by registered out-of-sample predictions (ln Ω(9), ln Ω(10) hit). (v) Fail-closed budgets: any enumeration whose materialized size would exceed the registered cap (k^dim × row width ≤ 2×10⁸; per-worker intermediate-state gates) is refused before allocation; long runs are checkpointed and resumable.

Table R2 identifies the generators and input data for the four main figures and five SI figures. It maps the ternary certificates to their exact checkers and the disk extrapolations to the independent enumeration implementations. The uniform and tension simulations record seeds and use the sampling protocols in SI S6.

---

## RS2. Counting engines

**Table R1 | Counting engines: reach, usage, and verification.**

| Engine (tier) | Core technical point | Reach / applicable systems | Used for | Anchored by |
|---|---|---|---|---|
| DFS + brute force (1) | partial-assignment search; integer kernel parametrization; prefix-blocked parallelism | F ≲ 20, any network | anchors everywhere; small exact counts | three-way DFS/DP/brute agreement (34/34 strip cases) |
| Keyed DP → strip TM (2) | states merged by key before vectorization (path counts ≠ state counts); one operator, all lengths via Tr(T^{L_x}) | strips and tori with compressible cut states | honeycomb series (L_x ≤ 20 at L_y = 2, 3, 5; L_y = 7, 9 by transposition), narrow strips W = 2–4, 4.8.8 ℤ and Λ W ≤ 4 | DFS anchors digit-for-digit (729/1017/22 767) |
| Exact sparse trace (2) | CSR operator; chunked unit-vector blocks; checkpoints + integrality asserts | triangle-locked cut states, \|S\| = 2^{2L+1} | 3.4.6.4 L = 6–10 | L = 2–5 ≡ Fraction/DP values |
| Cylinder TM spectral radius (2) | explicit sparse cylinder operator; ARPACK (degenerate leading spectra (± pairs, roots-of-unity families, accidental degeneracies) defeat power iteration); trunk/tail-connectivity + known-answer calibration gates | wide-strip and face-constraint systems: 3.12.12 mid (W = 2…8), honeycomb three classes (W = 2…9), 4.6.12 binary (W = 2…4) | entropy signs and asymptotes (S3.2, S5.2, S3.3) | Gate A/B bitwise anchors; Tr-vs-DP identity; W = 2 trunk calibration (zero deviation) |
| Constraint-space elimination (2) | bucket elimination over orderings; parameterized DFS with exact rational back-substitution | face-based constraint systems (4.6.12: 12-edge faces, 4 equations/face) | 4.6.12 L = 1–2 exact; L = 3, 4 by calibrated Knuth | L = 1 brute force; four affine anchors 64/64/518/22 767 |
| Frontier DP, blocked DFS, strip Knuth (2) | face-closure-ordering keyed DP (states merged by key); prefix-parameterized blocked DFS; strip Knuth with X = Πf weights | 4.6.12 binary {0,1}: square/rectangular tori and W×L strips | binary exact to L = 3 tori and 4×3; W = 2 strips to L = 14 (DP); calibrated strip Knuth over W = 2…5 (reach L ≈ 10/5/4 at W = 2/3/4) | anchors 112 / 22 912 / 58 240 / 11 261 320 / 15 300 096 dual-engine; Knuth pinned at four strip points (0.03–2%) |
| Knuth tree sampler (3) | unbiased self-avoiding tree sampling, two-step lookahead; entropy-density calibration | 4.6.12 ternary ℤ L = 3, 4 | 4.6.12 ternary series (Table R4) | 24-task calibration vs exact anchors (+0.0001…−0.0049, ±0.012); ineffective on Λ at W ≥ 3 |
| Hutchinson trace (3) | Tr(T^L) = ⟨vᵀ T^L v⟩, Rademacher; exactly unbiased; kurtosis-monitored | locked/compressed operators with near-Gaussian probe tails | 4.8.8 Λ W = 5 (± 0.0022%) | mandatory known-answer calibration pass; fail precedent: honeycomb binary L_y = 5 |
| Exact and finite-grid counts | Strand propagation and transfer-matrix traces | Square counts are exact at all sizes. The honeycomb coprime-period criterion is verified on the grid reported in SI S5.1. | S4.1 and S5.1 | Direct counts at overlapping sizes |
| Constructive-certificate checkers | Two compatible completions on a uniform background | Four ternary carriers | SI S2, Fig. S1 | Face closure checked in exact arithmetic, boundary and alphabet checks, support of translated copies |

---

## RS3. File map

**Table R2 | File map: paper items to data and code.** Scripts without a path prefix are canonical copies in `04_scripts_L2/`; counting-engine scripts are the exception and live beside their data in the per-network directories under `06_engines/` (the Table R1 row gives their paths in full); items marked (dev) are development-stage scripts, archived with this repository snapshot. Paths are repository-relative (`02_figures/`, `03_data/`, `04_scripts_L2/`, `05_predecessor_assets/`, `06_engines/`). SI Tables S1 and S2 and sections S1–S6 follow the manuscript numbering. Tables R3–R6 give numerical values behind the figures.

| Item | Data / code |
|---|---|
| Fig. 1 (model & classification) | `02_figures/fig1_model_classification.*`; script `fig1_model_classification.py`; assets `02_figures/assets_carriers/`, a/b/c from the shared asset family (lozenge `G1_lozenge_color.png`, honeycomb `G5b_honeycomb_faces.png`, 3.4.6.4 `G4_rth_colorful.png`; generators `gen_G_pieces.py`, `gen_K3_kagome_portrait.py`, `gen_R4_rth_colorful.py`, `gen_R4_rth_binary.py`, `gen_G_batch2.py`, `gen_G_batch3.py` in `05_predecessor_assets/`) + d/e rendered by `gen_metric_patches.py` (black-edge style) |
| Fig. 1f counts | `06_engines/z3_hexagonal_P0/P0_PM_torus_growth_forms.csv`; `06_engines/z3_hexagonal_P0/P0_PM_torus_trace_extension_Ly3.csv`; `06_engines/z4_3464/RTH_torus_area_law_8pts.csv`; `06_engines/z3_31212/TSD31212_dp_ternary.csv`; the 4.6.12 strict bound s₂ ≥ 0.186 comes from the roundE counting engine (Table R1; the 4.8.8 file `TSQ_Lambda_DP_exp.csv` belongs to Fig. 2c) |
| Fig. 2 (propagation, spectral restriction, row splitting and rigidity) | `02_figures/fig2_constraints.pdf`; script `fig2_decay_mechanisms.py`; data `06_engines/z4_square/SQ_entropy_density_verification.csv`, `06_engines/z4_kagome/KAG_entropy_density_verification.csv`, `06_engines/z3_488/TSQ_Lambda_DP_exp.csv`; pattern assets `02_figures/assets_decay/` (G2/G3/G4 from the shared asset family + the drawn triangular patch) |
| Fig. 3 (honeycomb geometric limit) | `02_figures/fig3_honeycomb_metric_entropy.*` (file named to the main-text number); script `fig3_honeycomb_metric_entropy.py` (panels c/d = canonical `fig2cd_symmetric_forms.py` functions, 1:1) + `gen_S9_torus_construction.py`; data `06_engines/z3_hexagonal_P0/P0_PM_torus_trace_shape_classes_Ly2Ly3.csv`, `…_Ly5.csv` (short exact counts, sub-law branch) + `TSDHC_long_trace{,_arith5}.csv` (long chains) |
| Fig. 3e planar-disk markers (S5.2) | `03_data/uniform/disk_counts_exact.csv` (exact disk counts, binary R = 0–7 and arithmetic R = 0–4, with anchors and fit summary); engines in `04_scripts_L2/disk/`: `disk_count.c` (C, `__int128`, column-recursion DP, scatter/gather twin modes; `gcc -O3 -std=gnu11 -fopenmp`), `disk_lib.py` (reference implementation: torus/disk construction, DFS, frontier DP, column DP, torus column trace), `blocked_multi.py` (blocked-DFS cross-check); torus-side anchors against the vendored `metric_hc.py`; alignment document available from the authors (2026-10-15) |
| Fig. 4 (paired diffraction) and Fig. S5 (metric tension) | Figure files: `fig4_value_domain_halos.py`, `fig4_fixed_entropy.pdf` and `S5_metric_tension.pdf`. The extended endpoints in Fig. 4b use `ladder_ext.csv` and the matched `raw/torus13_a678.npz` and `raw/torus13_a789.npz`. The intermediate tension rows use `hscan_summary.csv`. Generating script `fig4_value_domain_halos.py` (seven-panel v2, 2026-10-12; + modules `vd_suite3.py`, `vd_patch.py`, `vd_suite2.py` (helper), external `sampler2.py`/`metric_hc.py`, `P0_L1_halo_anisotropy.py`); data `03_data/uniform/` (`meanF2_*.npy`, `spots_*.csv` (per-spot f_coh/f_inc/f_coh_al/f_pk + bootstraps), `hscan_summary.csv`, `displacement_spectra.csv`, `wrms_curve_tor21.csv` (single-ensemble w_rms, panel d), `ens_*.npz`; raw per-configuration ensembles + sampler metadata in `03_data/uniform/raw/` (incl. `tor21_*.npz` and the h-scan `torus13_a123_h∓0.15/±0.30.npz` for the panel-a schematic)); server drivers `vd_torus_run.py` (ensemble generation), `vd_wrms.py` (patch-radius ladder); Fig. 4c halo-shape mechanism decomposition (SI S6.2): `fc_decompose.py` with per-spot results `fc_decompose_results.csv` and per-configuration, per-shift six-spot-mean r in `fc_decompose_percfg.csv` |
| Fig. S1 (lower-bound certificates, SI S2) | `02_figures/S1_lower_bound_certificates.*`; composer `figS1_lower_bound_certs.py`; source renderings `02_figures/assets_certificates/` (`flip_certificate.png`, `cert_3464.png`, `cert_31212.png`, `cert_4612.png`), drawn by `flip_certificate_fig.py` and `a4_cert_figs_all.py` (real-metric, edge length = edge value) |
| Fig. S2 (3.4.6.4 area law) | `02_figures/S2_area_law_3464.*`; script `figSI_area_law.py`; data `06_engines/z4_3464/RTH_torus_area_law_8pts.csv` |
| Fig. S3 (4.6.12: ternary series + spectral determination) | `02_figures/S3_4612_spectral.*`; generator `figS3_4612_spectral.py`; data `06_engines/z3_4612/RTH4612_knuth_estimate_series.csv` (panel a), rate values from `06_engines/z3_4612/RTH4612_metric_entropy_TM_readjudication_2026-10-06.md` §1 rate table (incl. the 2026-10-09 W = 4 row) + logs `RTH4612_t4k1_trunk_2026-10-09.log` / `RTH4612_t4k5_rssabort_2026-10-09.log` (panel b; rates asserted as constants in the generator) |
| Fig. S4 (resonant sectors) | `02_figures/S4_resonance_sectors.*`; data `06_engines/z3_hexagonal_P0/P0_PM_torus_trace_shape_classes_Ly2Ly3.csv`, `…_Ly5.csv` (resonant rows) + `TSDHC_long_trace{,_arith5}.csv` (long chains) |
| Table S1 (model comparison) | The SI table cites the underlying random-tiling, Wang-matching and framework-counting references directly. |
| Table S2 (classification) | theorem documents + verification CSVs in `06_engines/z3_hexagonal_P0/`, `06_engines/z4_square/`, `06_engines/z6_triangular/`, `06_engines/z4_kagome/`, `06_engines/z3_488/`, `06_engines/z3_31212/`, `06_engines/z4_3464/`, `06_engines/z5_snubsquare/`, `06_engines/z5_snubhex/`, `06_engines/z5_33344/`, `06_engines/z3_4612/` |
| Table R5 (λ grid + mixed model) | `06_engines/z3_hexagonal_P0/TSDHC_tm_rho.csv`, `TSDHC_tm_rho_nonarith.csv`, `TSDHC_binary_spectral.csv`, `TSDHC_nonarith_spectral.csv`; counters `step8_hc_tm.py`, `step9_hc_tm_nonarith.py`, `step10_hc_binary_spec.py`, `step11_hc_s_fits.py`, `step12_long_trace.py` (+`step12b_parallel.py`), `step14_hc_processing.py`; legacy exact-count CSVs `P0_PM_torus_trace_shape_classes_Ly2Ly3.csv`, `…_Ly5.csv` (sub-law branch data, S5.1) |
| S5.1 commensurate-sector counts | `06_engines/z3_hexagonal_P0/P0_PM_resonant_families_Ly9.csv` (+ anchor-gate log `P0_PM_resonant_families_Ly9_anchor_gate.log`); counters `P0_PM_hc_trace.py` + parallel driver `resonant_width9_par.py` |
| S5.1 exact-size rule grid | `03_data/uniform/exact_size_rule_grid.csv` (2,232 rows, 2,088 distinct alphabet–size points, 30 alphabets k = 2–8, L_y = 2–6, L_x ≤ 150; per-row alphabet flags and counts, the criterion prediction re-derivable from the gcd and 3-AP columns); criterion working documents available from the authors (analytical proof in preparation) |
| Table R3 (3.4.6.4 counts) | `06_engines/z4_3464/RTH_torus_area_law_8pts.csv`; counter `rth_torus_trace_sparse.py` |
| Table R4 (4.6.12 ℤ counts) | `06_engines/z3_4612/RTH4612_three_rank_dimR_dimQs3_dimSplit.csv` (ranks), `06_engines/z3_4612/RTH4612_constraint_matrix_layer_results_2026-10-01.md` (pipeline), `06_engines/z3_4612/RTH4612_knuth_estimate_series.csv` (s_ℤ series; head records column semantics, estimator version, and the L = 1, 2 anchor-validation runs, with pre-fix rows marked superseded); counters `roundD_4612_numeric.py`, `roundE_bucket_count.py`, `roundE_4612_paramdfs.py`, `roundE_4612_knuth.py`; binary width-scaling determination (S3.3): `06_engines/z3_4612/RTH4612_binary_anchors.csv` (exact anchors, dual-engine), `06_engines/z3_4612/RTH4612_binary_strips.csv` (W = 2 frontier-DP series), `06_engines/z3_4612/RTH4612_binary_L4x3_summary.csv` (+ `RTH4612_binary_L4x3_blocks.csv`, depth-split 4×3), `06_engines/z3_4612/RTH4612_knuth_strips2.csv` (strip-Knuth series, extracted from `run_knuth_strips2.log`); counters `roundE_4612_binary_anchors.py`, `roundF_server_blocks_binary.py`, `roundE_4612_frontier_dp.py`, `roundE_4612_strip_series.py`, `roundE_4612_knuth_strips2.py`; transfer-matrix spectral determination (S3.3): see the "Transfer-matrix determinations" row |
| Table R6 (value-domain suite, tension and patch blocks) | `03_data/uniform/`: `meanF2_*.npy`, `spots_*.csv` (per-spot f_coh/f_inc/aniso + bootstraps; in these CSVs the `f_coh` column is normalized by H_id while `f_coh_ref` is the H_ref-normalized paper calibre quoted in Table R6 — compare against `f_coh_ref`), `ens_*.npz`, `displacement_spectra.csv`, `hscan_summary.csv`, `wrms_curve_tor21.csv`, `pdf_*`, `slq_*`, `halo_orient_*`; raw configurations in `03_data/uniform/raw/` (incl. `tor21_*.npz`, `torus13_a123_h∓0.15/±0.30.npz`, the ladder extensions `torus13_a678.npz`, `torus13_a789.npz`, and the v6.2 independent-chain replicas `torus13_a123r.npz`, `torus13_b12r.npz`, `tor21_b12r21.npz`); generators `vd_suite3.py`, `vd_ladder_ext.py`, `vd_wrms.py`, `vd_repchains.py` (v6.2 replication), server drivers `vd_torus_run.py`, `vd_torus_run_v2.py` (dev), vendored `sampler2.py`, `metric_hc.py`; revision-round tables `03_data/uniform/ladder_ext.csv` (extended rungs), `03_data/uniform/size_scaling_tor21.csv` (patch-size series), `03_data/uniform/repchain_summary.csv` + `repchain_gates.json` (v6.2 replication), `03_data/uniform/fc_decompose_results.csv` + `fc_decompose_percfg.csv` (halo coherent/diffuse decomposition, S6.2); generator `fc_decompose.py` |
| Table R1 (counting engines) | counters `06_engines/z3_hexagonal_P0/pseudocrystal_lib.py`, `06_engines/z3_hexagonal_P0/P0_PM_hc_trace.py`/`P0_PM_hc_trace_Ly5.py`, `06_engines/z4_3464/rth_torus_trace_sparse.py`, `06_engines/z4_3464/hutch_rth.py`, `06_engines/z3_488/tsq_lambda/T_builder_v2.py`, `06_engines/z3_4612/roundE_bucket_count.py` + `06_engines/z3_4612/roundE_4612_paramdfs.py`; calibrations `06_engines/z3_hexagonal_P0/P0_A2b_Knuth_calibration_v2.csv`, `06_engines/z3_488/TSQ_Lambda_DP_exp.csv`, `06_engines/z3_4612/dev_calibrations/hutch_locW*_result.json` (dev) |
| Transfer-matrix determinations (S3.2, S5.2, S3.3) | 3.12.12: `06_engines/z3_31212/step7_tm.py`, `step7_cross.py`, `step7_deep.py`, `step12_rho_solution.py`, `step13_rho_method.py` + `TSD31212_tm_*.csv`, `TSD31212_symmetric_fit.csv`; honeycomb: `06_engines/z3_hexagonal_P0/step8_hc_tm.py`, `step9_hc_tm_nonarith.py`, `step10_hc_binary_spec.py`, `step11_hc_s_fits.py`, `step12_long_trace.py` + `step12b_parallel.py`, `step14_hc_processing.py` + `TSDHC_tm_rho*.csv`, `TSDHC_*_spectral.csv`, `TSDHC_long_trace{,_arith5}.csv`, `TSDHC_processing.csv`; 4.6.12: `06_engines/z3_4612/rth4612_tm.py` (+ `roundE_4612_knuth_ternary_strips.py`, run logs) + `06_engines/z3_4612/RTH4612_metric_entropy_TM_readjudication_2026-10-06.md`, `RTH4612_binary_positive_entropy_adjudication_audit_2026-10-05.md`, the procedures in SI S3.3 |
| SI S2 ternary certificates | Scripts in `04_scripts_L2/`: `flip_hc_ternary.py` (honeycomb flower), `flip_3464_31212.py` (3.4.6.4 and 3.12.12), `flip_488lam_4612.py` (4.6.12). The per-network results and modes are in `03_data/lower_bounds/`. Additional binary open-strip constructions and superseded width-scaling interpretations are archived with the v6.1.3 supplement and the original research records. |
| Shared library | `06_engines/z3_hexagonal_P0/pseudocrystal_lib.py` |

---

## RS4. Data tables of record

*The four tables of this section carry the precise data-point values behind the main-text figures (Figs. 1f, 3e and 4b–d and Fig. S5a–c); they are data of record, not new analysis. Raw counting outputs and per-configuration archives live in `06_engines/` and `03_data/uniform/` (Table R2).*

**Table R3 | 3.4.6.4 exact torus counts (binary integer alphabet {1,2}, translation-equivalent to {4,5}; see S3.1; L×L tori, F = 6L²).**

| L | Ω | ln Ω | s·L = ln Ω/(6L) |
|---|---|---|---|
| 2 | 128 | 4.8520 | 0.4043 |
| 3 | 1 036 | 6.9431 | 0.3857 |
| 4 | 8 800 | 9.0825 | 0.3784 |
| 5 | 86 956 | 11.3732 | 0.3791 |
| 6 | 1 114 700 | 13.9241 | 0.3868 |
| 7 | 23 325 388 | 16.9651 | 0.4039 |
| 8 | 796 865 888 | 20.4962 | 0.4270 |
| 9 | 44 571 806 560 | 24.5204 | 0.4541 |
| 10 | 4 002 790 772 868 | 29.0178 | 0.4836 |

**Table R4 | 4.6.12 ℤ counts and split-lattice ranks** ({1,2,3} alphabet; L×L tori, N_F = 6L²). Ranks exact (Fraction RREF + float-SVD cross-check; pipeline validated against the 3.12.12 dimensions 11/19). Ω_ℤ at L = 1, 2 exact (brute force / parametrized DFS); at L = 3, 4 Knuth tree-sampling estimates (final estimator version, anchored at L = 1, 2 against the exact totals, estimates of 2 346 and 2.971 × 10⁷ against exact counts of 2 349 and 29 723 517, respectively).

| L | N_F | dim_ℝ | dim_split | Ω_ℤ | s_ℤ = ln Ω/N_F | method |
|---|---|---|---|---|---|---|
| 1 | 6 | — (degenerate)* | — | 2 349 | 1.294 | exact (brute force) |
| 2 | 24 | 26 = F+2 | 20 = 4L²+4 | 29 723 517 | 0.717 | exact (parametrized DFS) |
| 3 | 54 | 56 = F+2 | 40 = 4L²+4 | ≈ 6.57 × 10¹² | 0.5465 ± 0.0005 | Knuth 160 k |
| 4 | 96 | 98 = F+2 | 68 = 4L²+4 | ≈ 8.35 × 10¹⁹ | 0.4778 ± 0.0001 | Knuth 2 M |

(* = the L = 1 torus is degenerate (equations merge at L = 1) and is used only as the brute-force anchor; the 4L²+4 split dimension is exact at L = 2…4.)

**Table R5 | Transfer-matrix spectral radii and the mixed global model.** λ(L_y) = ln ρ(L_y), per column (Perron roots; even-width values are spectral moduli: ±-pair or root-of-unity families, S5.1). Asymptotes from the mixed global fit λ/L_y = s + c/L_y + A′q^{L_y} (binary: even/odd branch lines sharing s, c, q); fit ranges in S5.2.

| Class | L_y = 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | s∞ (mixed) |
|---|---|---|---|---|---|---|---|---|---|
| binary {1,2} (k=2) | 0.89588 | 0.96657 | 1.24245 | 1.32452 | 1.59197 | 1.73615 | 1.99572 | 2.17424 | **0.209** (A′ ratio pinned; 0.248 free) |
| non-arithmetic {1,2,4} (k=3) | 1.24245 | 1.38629 | 1.50444 | 1.58305 | 1.78945 | 1.89771 | — | — | **0.224** (free fit; same band as 0.209) |
| arithmetic {1,2,3} (k=3) | 1.60944 | 2.01215 | 2.50211 | 3.01488 | 3.54934 | 4.09316 | 4.643 | — | **0.544** |

**Table R6 | The value-domain ensemble suite.** N = 169 holes (13×13 torus, cut open); 20 configurations per domain; two-stage sampler targeting the uniform measure (PT, seed family 20261100 + domain offset, per-configuration seeds archived); f_coh = |⟨F⟩|²/H_ref (H_ref: the ideal-patch reference in the same raw-sum estimator, S6.1), the halo-to-core ratio and r — the halo second-moment ratio σ⊥²/σ∥² (S6.1) — are six-spot means with configuration-bootstrap 95% intervals (B = 200; per-spot tables archived, halo/core intervals in the `hc_lo`/`hc_hi` columns; within a domain the six spots split into inequivalent classes, per-spot f_coh spreading by up to a factor of five at the disordered end). f_inc = ⟨|F|²⟩/H_ref − f_coh is the incoherent share at the spot positions. S⊥/S∥ is the halo-window displacement-spectrum ratio, mean ± s.d. over the ten resolved κ-bins κ = 0.34–1.06 (the innermost bin of the halo window lies below the 13×13 supercell's fundamental Bloch shell and is excluded; S6.2). The non-arithmetic {1,2,4} row is deferred (S6.1); the {1,2} row carries the ensemble-level systematic of S6.1. The last two arithmetic rows ({6,7,8}, {7,8,9}) extend the ladder to the relative dispersion of alloyed frameworks; they are exact translation relabelings of the {1,2,3} sample (no new sampling). The halo-to-core column is a raw-sum ratio of the two windows whose flat-field value is 24.8 (S6.1); the coherent fraction is size-dependent at every domain, most strongly at the disordered end (S6.3). Tension block: the biased ensembles of the arithmetic domain {1,2,3} (metric equation of state m(h) and entropy cost Δs(h), S6.4; Fig. S5a–c); every tension is sampled twice (directly and through the mirror of the opposite-tension ensemble) and the quoted f_coh, halo/core and r are the mean of the two samplings with half their absolute difference as the plotted spread (S6.4).

| Domain | Class | Tiles | σ/μ | f_coh | f_inc | halo / core | r | S⊥/S∥ |
|---|---|---|---|---|---|---|---|---|
| {1,2} | binary | 10 | 0.333 | 1.8×10⁻⁴ | 1.8×10⁻³ | 18.9 | 0.36 | 0.17 ± 0.06 |
| {1,2,3} | arithmetic | 45 | 0.408 | 9.0×10⁻⁴ | 5.6×10⁻³ | 16.7 | 0.29 | 0.14 ± 0.05 |
| {2,3,4} | arithmetic | 45 | 0.272 | 2.5×10⁻² | 3.7×10⁻² | 4.4 | 0.30 | 0.14 ± 0.05 |
| {3,4,5} | arithmetic | 45 | 0.204 | 0.118 | 0.081 | 2.0 | 0.34 | 0.14 ± 0.05 |
| {4,5,6} | arithmetic | 45 | 0.163 | 0.251 | 0.101 | 1.2 | 0.37 | 0.14 ± 0.05 |
| {5,6,7} | arithmetic | 45 | 0.136 | 0.380 | 0.101 | 0.9 | 0.40 | 0.14 ± 0.05 |
| {6,7,8} | arithmetic | 45 | 0.117 | 0.489 | 0.093 | 0.71 | 0.43 | 0.13 ± 0.04 |
| {7,8,9} | arithmetic | 45 | 0.102 | 0.577 | 0.083 | 0.60 | 0.47 | 0.13 ± 0.04 |

*Tension block (arithmetic domain {1,2,3}):*

| h | m(h) [per face] | Δs(h) [k_B/face] | f_coh (pooled ± spread) | halo/core | r |
|---|---|---|---|---|---|
| −0.30 | 8.912 | −0.488 | 0.545 ± 0.00939 | 0.62 | 0.59 |
| −0.15 | 7.112 | −0.083 | 4.32×10⁻³ ± 2.85×10⁻³ | 9.02 | 0.30 |
| 0 | 6.008 | 0 | 9.0×10⁻⁴ | 16.7 | 0.29 |
| +0.15 | 4.915 | −0.082 | 1.19×10⁻⁴ ± 0.66×10⁻⁴ | 21.33 | 0.95 |
| +0.30 | 3.100 | −0.490 | 0.192 ± 0.00701 | 1.42 | 0.32 |

*Each tension is read twice, directly and through the mirror (l → 4 − l) of the opposite-tension ensemble; both sampling procedures target the same measure, so the pooled value is their mean and the plotted spread is half their absolute difference (S6.4). The mirror table:*

| measure | direct | mirror-read |
|---|---|---|
| f_coh(h = +0.30) | 0.185 | 0.199 |
| f_coh(h = +0.15) | 5.34×10⁻⁵ | 1.85×10⁻⁴ |
| f_coh(h = −0.15) | 7.17×10⁻³ | 1.47×10⁻³ |
| f_coh(h = −0.30) | 0.555 | 0.536 |

*Patch-size scaling (21×21 ensembles; SI S6.3, N = 127–331):*

| domain (σ/μ) | f_coh, N = 127 → 331 | halo/core, N = 127 → 331 | r, N = 127 → 331 |
|---|---|---|---|
| {1,2,3} (0.408) | 1.18×10⁻³ → 3.2×10⁻⁴ (×0.27; flat beyond N = 271) | 14.7 → 15.1 (≤ 5%) | 0.29 → 0.34 |
| {4,5,6} (0.163) | 0.281 → 0.195 (×0.69) | 1.31 → 0.92 | 0.37 → 0.30 |
| {1,2} (0.333) | 1.1×10⁻² → 5.9×10⁻³ (×0.55) | 3.7 → 2.4 | 0.33 → 0.28 |

**Independent-chain replication (SI S6.1–S6.2; v6.2).** Three runs of the production driver `vd_torus_run.py` with fresh seeds and otherwise identical protocol (eq 1.5×10⁶ — 3×10⁶ at 21×21, thinning 4×10⁴, 20 configurations, PT): `raw/torus13_a123r.npz` ({1,2,3}, 13×13, seed 20264100), `raw/torus13_b12r.npz` ({1,2}, 13×13, seed 20264200), `raw/tor21_b12r21.npz` ({1,2}, 21×21, seed 20264300). All sampler gates pass (closure 0/20; reflection criterion met; swap rates 0.796 / 0.757 / 0.939 against production 0.797 / 0.741 / 0.934; wall times 3 530 / 3 440 / 7 213 s against production 3 541 / 3 416 / 7 126 s). The anchored analysis (`vd_repchains.py`) first reproduces the production readings on the archived ensembles — f_coh = 9.013×10⁻⁴ with halo/core 16.69 ({1,2,3}), 1.774×10⁻⁴ (binary 13×13), 1.005×10⁻² (binary 21×21, R = 7 patch) — before reading the new chains.

| observable | independent chain | production chain |
|---|---|---|
| f_coh at {1,2,3} (13×13) | 9.24×10⁻⁴ | 9.0×10⁻⁴ |
| f_coh at {7,8,9} (paired +6 shift) | 0.533 | 0.577 |
| endpoint ratio (paired 95% quantiles) | 575 (227–1291) | 640 (257–1260) |
| f_coh, binary 13×13 | 2.68×10⁻⁴ | 1.8×10⁻⁴ |
| f_coh, binary 21×21 (R = 7) | 0.88×10⁻² | 1.0×10⁻² |

The paired-translation series rises monotonically in both chains (largest intermediate-shift deviation ×1.8 at {2,3,4}, within the single-chain spread of this observable); each chain's bootstrap interval characterizes snapshot-level (within-chain) uncertainty, and the binary parent-size contrast (×30–60) dominates the within-parent seed-to-seed spread (×1.1–1.5), addressing the confound disclosed in S6.1 (with two chains per parent size, a full decomposition of winding-sector sampling weights remains open). Mixing diagnostic (`ess_pilot.py`, fresh seeds 20265100/20265200, same PT ladder and equilibration): τ_int(Σℓ) = 147 ± 7 steps ({1,2,3}) and 213 ± 12 steps ({1,2}) by Sokal windowing on Σℓ series logged every 100 steps (`raw/ess_a123.npz`, `raw/ess_b12.npz`), both well below the production thinning interval of 4×10⁴ steps. This diagnostic bounds length-sector mixing only: the total edge length is invariant under the entropy-carrying flower flips (each flip exchanges six short and six long edges), so it does not by itself certify full-configuration independence of the 20-snapshot chains; the within-chain snapshot bootstrap intervals and the two-chain replication (SI S6.2) carry that part of the evidence. Full series, per-shift values and gates: `03_data/uniform/repchain_summary.csv`, `03_data/uniform/repchain_gates.json`.

---

## References

1. Newman, M. E. J. & Henley, C. L. Phason elasticity of a three-dimensional quasicrystal: a transfer-matrix method. *Phys. Rev. B* **52**, 6386–6399 (1995).
2. Knuth, D. E. Estimating the efficiency of backtrack programs. *Math. Comp.* **29**, 121–136 (1975).
3. Hutchinson, M. F. A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Commun. Stat. Simul. Comput.* **19**, 433–450 (1990).
