# Calibration collapse under time-split validation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23583593.svg)](https://doi.org/10.5281/zenodo.23583593)
[![checks](https://github.com/noahnoah0713/calibration-collapse-time-split/actions/workflows/checks.yml/badge.svg)](https://github.com/noahnoah0713/calibration-collapse-time-split/actions/workflows/checks.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](environment.yml)

![Summary: splitting two FDA toxicity lists at approval year 2010 collapses calibration while discrimination barely moves, and refitting with recalibration recovers most of it](figures/graphical_abstract_hires.png)

Hepatotoxicity (DILIrank) and cardiotoxicity (DICTrank) models lose about
0.08 AUROC when the held-out set is defined by approval year rather than at
random. Their expected calibration error more than doubles, and with
prevalence held equal on both sides of the split it is still 1.37x and 1.21x
the matched control. This repository holds the analysis code and every
intermediate result behind that claim.

| | DILIrank | DICTrank |
|---|---|---|
| Compounds (training / held out) | 796 (640 / 156) | 919 (652 / 267) |
| AUROC, time split vs random | −0.088 | −0.077 |
| Calibration error, ratio to a size-matched random split | **×3.35** | **×2.08** |
| …after matching the positive rate on both sides | **×1.37** | **×1.21** |
| Share of the Brier increase that is calibration, not discrimination | 86.1% | 82.5% |
| Conformal coverage at a nominal 90% | 73.7% | 82.4% |
| Calibration error recovered by refitting with recalibration | −0.165 | −0.176 |

That table is the one place in this repository where figures are quoted by
hand, so here is where each comes from: the AUROC drops from
`results/permutation_test.csv`, both calibration ratios from
`results/prevalence_control.csv`, the Brier shares from
`results/brier_decomposition.csv`, the coverage from
`results/conformal_prediction.csv`, and the recalibration effect from
`results/prequential_retraining_delta.csv`. Everywhere else — the report, the
tables and the figures — the numbers are read from those files by script.

Everything in the paper — each number, table and figure — is regenerated
from `data/processed/` by the scripts here. Nothing is typed in by hand.

---

## Running it

```bash
pip install -r requirements.txt
bash rerun_all.sh
```

| Output | Where |
|---|---|
| Results report, with all numbers and tables | `RESULTS_REPORT.md` |
| LaTeX tables for the manuscript | `manuscript/tables_main.tex`, `tables_supp.tex` (the run creates the folder) |
| Figures, vector and raster | `figures/` |
| Every intermediate result | `results/` |

`RESULTS_REPORT.md` is written in Korean. Every number in it is read from
the CSV files in `results/`, whose column names are English, so the
underlying values are readable without it.

`rerun_all.sh` runs 77 scripts in 81 calls. The report and the two
document-side verifiers run twice, because the second pass picks up a cell
count the first pass produces. At the end it checks that every result file
the report, tables and figures read actually exists, and exits non-zero with
a list if any is missing. If a script fails, the run stops there rather than
carrying on with stale inputs.

Two things to know before you start.

The run **overwrites** `results/`. Copy it first if you want to keep the
values a particular version produced (`cp -r results results_before`).

It takes several hours. Most permutation tests and control arms repeat 200
to 2,000 times. To look at one analysis quickly, run its script directly and
lower the repetition count — 23 of them take `--reps`, for example
`python src/prevalence_control.py --reps 10`.

---

## Data

| Source | Origin | In this repository |
|---|---|---|
| DILIrank | FDA Liver Toxicity Knowledge Base (LTKB) | processed set only |
| DICTrank | FDA Drug-Induced Cardiotoxicity Rank | processed set only |
| Approval years | Drugs@FDA bulk files (Products / Submissions / Applications) | not included, 42 MB |
| ATC classes | KEGG DRUG–ATC linkage | mapping result only |

All three sources are public FDA releases. What ships here is the processed
analysis set, not the raw downloads; the path from raw files back to these
is in the next section. The KEGG mapping is a cached lookup and is subject
to KEGG's own terms.

`data/processed/` is all you need. Of the 89 scripts in `src/`, five read
raw files: `map_approval_year.py` and `dictrank_pipeline.py` build the
processed data, and `ev_years.py`, `check_years.py` and `check_labels.py`
audit it against the raw source. The first two are not called by
`rerun_all.sh`; the other three skip themselves when `data/raw/` is absent.
The remaining 84 touch only `data/processed/` and `results/`.

One consequence: `results/year_evidence.csv`, `results/year_corrections.csv`
and `results/year_match_types.csv` — the evidence behind the approval-year
corrections, and the tally of which name-matching rule assigned each year —
cannot be rebuilt without the raw files. All three are committed and the
report reads those copies.

| File | Contents |
|---|---|
| `dilirank2_with_year.csv` | DILIrank analysis set, with approval year, SMILES and InChIKey14 |
| `dictrank_processed.csv` | DICTrank analysis set |
| `*_prefilter.csv` | state before the structure filter, used by the sensitivity analyses |
| `*_preconflict.csv` | state before the label-conflict filter |
| `kegg_atc_map.csv` | ATC class assignments |
| `smiles_failed.csv`, `year_unmatched.csv` | records of what did not match |

### Starting from the raw files

Download the Drugs@FDA bulk files and the two dataset spreadsheets into
`data/raw/`, then:

```bash
python src/fetch_smiles.py          # SMILES from PubChem (needs network)
python src/map_approval_year.py     # approval years from Drugs@FDA
python src/dictrank_pipeline.py     # DICTrank: spreadsheet, years and ATC
python src/apply_structure_filter.py
python src/apply_label_conflict_filter.py
bash rerun_all.sh
```

Both filters back up the originals as `*_prefilter.csv` and
`*_preconflict.csv` on first use and never overwrite an existing backup, so
repeated runs keep the pre-filter data the sensitivity analyses depend on.

---

## How the numbers stay honest

No figure in the report is transcribed. `src/make_report.py` and
`src/make_tables.py` read every value out of `results/` and format it, so a
copying error is not possible by construction. Five scripts then check the
result from both directions:

- `verify9.py` traces every number in the report back to the file it came
  from, and re-runs the generator to confirm the document on disk is its
  output.
- `verify10.py` goes the other way: it builds the expected strings from
  `results/` and looks for them in the document, cell by cell.
- `verify11.py` checks that the 32-cell model × representation grid agrees
  between the two scripts that compute it.
- `verify12.py` does the same for the manuscript prose, and skips itself
  when the manuscript is absent, as it is here.
- `verify_state.py` records hashes of the data, the code and the results so
  drift between them is visible.

Randomness is seeded through `numpy.random.default_rng`. LightGBM's thread
dependence is disabled and RandomForest accumulates predictions on a single
thread, so repeated runs reproduce bit for bit (`src/_shared.py`).

### Known answers

The checks above compare results with each other. `tests/` compares them with
arithmetic done by hand: 54 unit tests in four files, driven by `unittest`, so
there is nothing extra to install.

```bash
python -m unittest discover -s tests -v
```

- `test_metrics.py` pins the shared metric functions to worked examples — the
  ECE of a two-bin case, the Murphy decomposition of an eight-point forecast
  against its own Brier score, the bin a forecast of exactly 1.0 belongs to,
  and the four constants the whole paper rests on.
- `test_duplicate_definitions.py` applies those same answers to every copy.
  `ece` is defined seven times across the analysis scripts, `calib_slope`,
  `bin_table` and `murphy_decomposition` three times each. `check_determinism.py`
  checks the copies agree; agreeing is not the same as being right, so each one
  is measured against the hand-computed value separately. The file also pins
  the number of definitions, so an unregistered eighth copy fails the run.
- `test_invariants.py` covers what must hold for any input: ECE is unchanged by
  row order and stays within its bounds, the decomposition terms stay in range
  and track the Brier score to within the binning error, and scaling the logit
  by c divides the calibration slope by exactly c.
- `test_modelling.py` pins the contract of the modelling code rather than its
  values, which move with the library versions: the feature matrix is 2048 bits
  plus six descriptors in the documented order, the fingerprint block is binary,
  the same molecules give the same matrix and the same seed gives the same
  predictions. It also pins the model grid, which is where the fifth defect
  below came from.

Every test was written by breaking the function it watches and confirming the
suite turns red. Eight deliberate faults were tried and all eight were
caught: the `float32` cast removed at each of the three sites that carry it,
the last ECE bin left open, reliability inflated by one percent, the logit
clip loosened from `1e-6` to `1e-3`, the uncertainty formula altered in one
of the three decomposition copies, and rounding replaced by truncation in
the prevalence matching.

`.github/workflows/checks.yml` runs three jobs on every push: the linters, the
unit tests installed from `requirements.txt`, and a re-run of the document
verifiers that confirms the committed report is still exactly what the
generator produces from the committed results.

### What the checks actually caught

These were written to find mistakes, not to pass. Five turned up, and until
each was found every check was green. The source comments carry the dates
and the reasoning.

| Found | How it surfaced | What it affected |
|---|---|---|
| Eight logit paths that never widened `float32` to `float64` | the same calibration slope printed differently in two sections | two sections reported one quantity as two values |
| Two hand-run steps missing from the pipeline | comparing a full re-run against the previous results cell by cell | §13.5 and §33 would not reproduce for anyone else |
| Four result files a resume cache had skipped entirely | a result file older than the code that writes it | two full re-runs had reported them as complete |
| Duplicate rows accumulated by append-mode writes | counting exactly repeated rows across every result file | the multiple-comparison correction counted 73 tests instead of 57 |
| A model grid that silently drops from 32 cells to 24 when LightGBM is missing | writing a unit test for the model factory | `verify11.py` compared the two grid files with each other, so both shrinking together passed |

The first one took two wrong guesses to find. The difference was blamed on
lbfgs reaching a different optimum each run; a determinism check ruled that
out, and `src/diff_ladder.py` then ran both paths side by side in one
process and compared fifteen stages before the dtype of the logit turned up.

So each new check is tested by deliberately breaking the code it watches and
confirming it notices. That threw out two useless checks. One used
prevalence values that divided evenly, so swapping rounding for truncation
changed nothing; the other compared file timestamps and raised forty false
alarms the moment a formatter touched every file. The first got harder
inputs, the second now compares syntax trees.

### What is still unchecked

- **The method itself.** The checks confirm that the written rules were
  executed and that the numbers agree with each other. Whether those are the
  right rules — the choice of control arm, estimator and test — is not
  something any script here can settle.
- **The analysis-specific maths.** Of the 245 functions in `src/`, eleven are
  pinned to hand-computed answers. The bootstrap intervals, the recalibrators,
  the conformal coverage and the multiple-comparison correction are checked
  only by agreeing with each other and by the end-to-end re-run. The `float32`
  defect showed what that misses: seven copies of one metric agreed with each
  other for weeks while all seven were wrong.
- **Figures.** `make_figures.py` keeps no record of the values it plotted,
  so a figure that disagrees with its table would go unnoticed. The headline
  values were matched against the figures by hand once, by recording every
  number the plotting calls received: sixteen table values appear in a figure
  and all sixteen agreed, and the rest are not plotted at all. Nothing
  re-checks this when the figures change.
- **The modelling code.** `test_modelling.py` pins shape, order and
  determinism, not the values themselves — nothing says the fingerprints or
  the fitted probabilities are the right numbers, only that they are stable
  and well formed.
- **A clean-room run.** The checks workflow regenerates the report and
  re-runs the verifiers from a fresh checkout on every push, and seven of the
  analysis scripts have been run from a clean clone on a deliberately
  different stack (Python 3.10, NumPy 2.2, pandas 2.3, scikit-learn 1.7,
  RDKit 2026.09) to see what moves: they all ran, `check_determinism.py`
  still reported no failures, and 41 cells shifted in the fourth decimal —
  temporal AUROC 0.6476 to 0.6489, temporal ECE 0.4384 to 0.4445. The
  remaining seventy scripts, and `rerun_all.sh` end to end, have not been run
  that way.

---

## Layout

```
src/              96 analysis and verification scripts (_shared.py is the common module)
tests/            unit tests for the shared metric functions
data/processed/   processed data the analyses read
results/          every intermediate and final result: 102 CSV, 70 TXT, 2 JSON
figures/          figures for the manuscript, PDF and 600 dpi PNG, plus the
                  graphical abstract in three forms: .svg and a 920x300 .png
                  at the size the journal specifies, and _hires.png at 4x for
                  reading on screen
_backup/          results from before the precision fix; record_precision_fix.py
                  reads them to document what it changed
```

Of the 174 files in `results/`, 97 are read by name by the report, tables and
figures. The CSV files are the machine-readable record and their column names
are English; the TXT files are per-analysis summaries written in Korean. The rest are per-analysis summaries a reader can skim without
running anything, permutation null distributions kept so the tests need not
be repeated, and audit records for SMILES, approval years, labels and ATC
assignments. The resume caches the permutation scripts write are not
included, so a fresh clone recomputes those null distributions instead of
reading a finished run back.

### Scripts `rerun_all.sh` does not call

| Script | Why |
|---|---|
| `_shared.py` | common module: loading, featurisation, models, metrics |
| `fetch_smiles.py`, `fetch_smiles_fix.py` | need the PubChem API |
| `map_approval_year.py`, `dictrank_pipeline.py` | need `data/raw/` |
| `record_env.py` | re-running it would overwrite the recorded environment |
| `diagnose.py`, `diagnose_atc.py` | API diagnostics; they write no results |
| `atc_analysis.py`, `baseline.py` | superseded by `atc_kegg.py` and `temporal_split.py` + `baseline_cv.py` |
| `diff_ladder.py`, `diff_results.py` | diagnostics used while tracking down the defects above |

Everything the report depends on is inside `rerun_all.sh`.

---

## Environment

Python 3.11.14 on macOS 14.3.1 (arm64), conda environment `pharmml`. The
full package list is in `results/environment.json` and
`results/environment.txt`; `requirements.txt` notes which packages need
conda rather than pip. Results are reproducible bit for bit on the recorded
versions — see the determinism note above.

## Citing

This repository is archived on Zenodo. To cite the exact version an
analysis was run with:

> Lee, N. (2026). *Beyond AUROC: Quantifying Calibration Collapse in
> Time-Split Validation of Hepatotoxicity and Cardiotoxicity Models*
> (v0.1) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23583594

Zenodo mints two DOIs. The one above belongs to v0.1 and never moves, which
is what a paper should cite. `10.5281/zenodo.23583593` is the concept DOI:
it always resolves to the newest release, and that is the one in the badge
at the top.

`CITATION.cff` in the repository root holds the same details in the
machine-readable form that GitHub's "Cite this repository" button reads.

Bibliographic details for the paper itself will be added once it is
published.

## Licence

Code in `src/` and `rerun_all.sh` is MIT; see `LICENSE`.

The data follow their original terms. DILIrank and DICTrank are public
releases from the FDA/NCTR, and the ATC classification comes from the WHO
Collaborating Centre for Drug Statistics Methodology. Check each source
before redistributing.
