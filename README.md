# Metric random tilings: exact entropy and the diffraction of counted disorder

Code, data, and figure release for the paper *Metric random tilings: exact
entropy and the diffraction of counted disorder* (Xinyan Wu, Qingqing Ji;
arXiv link to be inserted at submission). The Letter, its Supplementary
Information, and the Reproducibility Supplement (RS v1.2) are included as
PDFs under `01_paper/`; RS Table R2 is the authoritative file map for
everything in this repository.

Archived: Zenodo DOI [10.5281/zenodo.24156133](https://doi.org/10.5281/zenodo.24156133) (concept DOI; resolves to the release accompanying the paper).

Random metric disorder on the eleven Archimedean/Kagome-type edge networks:
every tiling keeps one vertex type and one face type, the graph is fixed, and
the *edge lengths* fluctuate under exact per-face closure constraints. The
scripts count configurations exactly, certify constructive entropy lower
bounds, sample value-domain ensembles, and compute exact structure factors.

## Layout

| Folder | Contents |
|---|---|
| `01_paper/` | the Letter, SI, and Reproducibility Supplement as PDFs |
| `02_figures/` | all delivered figures (PDF + PNG, S1–S11 and Fig. 1–4) with rendering assets |
| `03_data/` | delivered data: `uniform/` (ensembles, structure factors, exact counts, NPZ raw configurations), `lower_bounds/` (certificate records) |
| `04_scripts_L2/` | paper-level scripts: one script per figure, the value-domain suite (`vd_*`), SI data generators, constructive-certificate checkers (`flip_*`, `h2l_*`), shared style, and `paths_common.py` (see below) |
| `05_predecessor_assets/` | shared pattern-asset generators (Fig. 1/2 input art) |
| `06_engines/` | per-network counting engines (one folder per tiling network), kept beside their data |

## Quick start

```bash
pip install -r requirements.txt
# SI Fig. S11 data + figure (exact enumeration, seconds per point):
python 04_scripts_L2/si_data.py
python 04_scripts_L2/fig_si_entropy_split.py
# official small-torus anchors (affine invariance, brute-force cross-checks):
python 06_engines/z3_hexagonal_P0/edge_alphabet_affine_check.py
python 06_engines/z3_hexagonal_P0/edge_alphabet_binary_check.py
```

Everything resolves paths through `04_scripts_L2/paths_common.py`, from any
working directory, so no script carries machine-specific paths. The same
sources also run in the authors' development tree (the resolver detects the
layout); `dev_name_map.json` next to it maps the repository's ASCII filenames
to the development originals (byte-identical, only names changed).

## Counting engines (`06_engines/`)

| Folder | Network | Character |
|---|---|---|
| `z3_hexagonal_P0` | honeycomb (3.6.3.6) | the positive-entropy carrier; exact traces, TM spectra, shape classes |
| `z3_31212` | 3.12.12 | mid-system positive entropy, TM ladder |
| `z3_4612` | 4.6.12 | perimeter-law falsification, strict width rates |
| `z3_488` | 4.8.8 | ℤ zero entropy / ℝ extensivity, Λ-library counts |
| `z4_3464` | 3.4.6.4 | triangle locking, sparse-trace area law |
| `z4_kagome` / `z4_square` / `z6_triangular` | kagome / square / triangular | zero-entropy mechanisms, closed forms |
| `penrose` | Penrose rhombi | strip theorem Ω = k^#strips |
| `z5_snubhex` / `z5_snubsquare` / `z5_33344` | snub hexagon / snub square / 3.3.3.4.4 | rigidity and sub-extensive cases |

## Reproducing the paper figures

Each main/SI figure has one generating script in `04_scripts_L2/` (map in RS
Table R2). Figures are rendered from delivered data by default; regenerating
raw ensembles needs the sampler drivers (`vd_torus_run*.py`, server-scale)
and the exact-uniform suite (`vd_suite3.py`, `vd_patch.py`). The constructive
certificates of SI S8.3 re-verify in seconds-to-minutes via `flip_*.py` /
`h2l_*.py` (records land in `03_data/lower_bounds/`, matching the delivered
JSONs bit-for-bit).

## License

Code and datasets: MIT — see `LICENSE`. The paper PDFs in `01_paper/` are
distributed under their arXiv license. If you use this code or data,
please cite the paper.
