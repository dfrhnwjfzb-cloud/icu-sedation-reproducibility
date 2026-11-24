# Clinical sedation study: reproducibility code

This project reproduces the primary MIMIC-IV analysis, the supportive eICU analyses, and both rounds of baseline severity robustness analyses.

## Layout

| Path | Purpose |
|---|---|
| `run_pipeline.py` | Command-line entry point and analysis order |
| `src/design.py`, `src/effects.py` | Baseline feature matrix, propensity model, overlap weights and augmented estimates |
| `src/external.py`, `src/transport_*.py` | eICU analyses and transport comparisons |
| `src/primary/` | Primary outcomes, sensitivity analyses and manuscript comparisons |
| `src/round1/` | Pre-treatment SOFA, original-weight balance, severity strata and ICU restrictions |
| `src/round2/` | Measurement audits, fixed severity specifications and patient bootstrap |
| `src/runtime.py`, `src/settings.py` | Paths, verified inputs and fixed settings |
| `config/analysis_settings.json` | Analysis specifications, source hashes and file paths |
| `sql/` | Original SQL texts; official reference SQL retains the upstream license |
| `tests/` | Score, temporality and second-round unit tests |

## Environment

Tested with Python 3.12.14, numpy 2.3.5, pandas 3.0.1, scipy 1.18.0 and lxml 6.1.1. Optional figure exports use matplotlib 3.11.0. BLAS threads are set to one before numerical imports.

```powershell
python -m pip install -r requirements.txt
python -B run_pipeline.py --check
python -B run_pipeline.py --help
```

## Inputs

Data are supplied outside the source directory under the applicable database access agreements. The program verifies the primary input manifest and frozen reference values.

| Argument | Required materials |
|---|---|
| `--primary-inputs` | `SOURCE_MANIFEST.json` and every listed file: `inputs/`, `expected/mimic/`, `expected/eicu/`, `reference_archive/`, the frozen manuscript DOCX, and the listed provenance files |
| `--frozen-inputs` | `mimic_design_candidates.csv`, `mimic_outcomes.csv`, `original_weights.csv`, retaining the original row order |
| `--round1-cache` | `data/sofa_observations.csv`, `data/windows.csv`, `data/rebuilt_candidates.csv`, `outputs/historical_lab_tie_audit.csv` |
| `--round2-cache` | `data/raw_labs.csv`, `raw_charts.csv`, `raw_vaso.csv`, `raw_urine.csv`, `derived_urine.csv`, `urine_rate_history.csv` |
| `--reference-dir` | The ten frozen aggregate reference files listed below |

References: `round1_model_comparison.csv`, `round1_subgroups.csv`, `round1_icu_restriction.csv`, `round1_urine_sensitivity.csv`, `round1_original_balance.csv`, `round2_model_comparison.csv`, `round2_bootstrap.csv`, `round1_expected_effects.csv`, `round1_expected_balance.csv`, `round1_expected_diagnostics.json`. A `benchmarks/` subdirectory is also accepted.

The primary inputs include MIMIC candidates/outcomes, eICU external candidates, and the archived MIMIC/eICU transport cohorts. `original_weights.csv` contains the propensity scores and overlap weights. The primary analysis writes these values to `derived/removed_weights.csv`. References are used for comparison, not as outcome-model inputs.

## Run

Use a separate output workspace. Input directories and source code must not overlap that workspace.

```powershell
python -B run_pipeline.py --task primary --workspace "D:\study_output" --primary-inputs "D:\primary_inputs"
python -B run_pipeline.py --task severity --workspace "D:\study_output" --frozen-inputs "D:\frozen_inputs" --round1-cache "D:\round1_cache" --round2-cache "D:\round2_cache" --reference-dir "D:\references" --workers 4
```

`--task all` runs both commands' analyses with all input arguments supplied. `round1` and `round2` permit separate execution; round 2 needs completed round-1 outputs in the same workspace. `validate` compares existing results; `bootstrap-pilot` recomputes the ten pilot draws. `--figures` exports the round-1 plots.

An external JSON file may be supplied with `--config`. Its keys are `workspace`, `primary_inputs`, `frozen_inputs`, `round1_cache`, `round2_cache`, `reference_dir`, `bootstrap_workers` and `postgres`. Paths are relative to the configuration file; command-line arguments override corresponding values. The effective configuration is written to `workspace/run_config.json`.

The workspace contains `primary/`, `round1/`, `round2/`, `logs/`, `validation/` and `supporting_inputs/`. Analysis modules write CSV tables, JSON diagnostics and execution logs to these directories.

## Database extraction

`--extract` uses a fresh workspace and the local MIMIC-IV 3.1 database with `mimiciv_derived`, replacing the severity extraction caches. Frozen inputs and references are still required. `--task db-check` checks 21 required tables and 140 columns. PostgreSQL options are `--psql`, `--db-host`, `--db-port`, `--db-user` and `--db-name`; use PostgreSQL credential facilities for authentication.

```powershell
python -B run_pipeline.py --list-sql
python -B run_pipeline.py --show-sql 61_review_round1_pre_t0_sofa.sql
python -B run_pipeline.py --export-sql --workspace "D:\sql_export"
```

The 28 SQL files include five primary-study queries, fourteen official MIT-LCP references, the frozen cohort source and eight parameterized cohort queries. Some official references use BigQuery syntax; the executed PostgreSQL extraction queries are constructed by the corresponding analysis modules. The parameterized queries require an authorized cohort binding. Extraction uses read-only SELECT/CTE statements. The primary runner starts from frozen extracted cohorts; database ingestion, derived-table construction and upstream primary-study extraction require separate setup.

## Statistical implementation

The primary cohort contains 3,669 stays (DEX 2,778; MIDAZ 891). The propensity ridge penalty is 1 and the outcome ridge penalty is 2. The primary models use L-BFGS-B with `maxiter=800` and `ftol=1e-10`. Only the outcome regression is cross-fitted, using row position modulo five. Feature construction, imputation, model specifications and severity-analysis settings are defined in the corresponding modules and `config/analysis_settings.json`.

The patient bootstrap uses seed 2026112202. Ten pilot draws are excluded from formal intervals; six specifications each have 500 formal draws. Every draw refits propensity scores, weights and twenty outcome fold models. Copies of a source stay use the same fold label. Per-draw JSON records support resumption; a fresh workspace computes all draws. Bootstrap logs record convergence status and estimated risks outside the unit interval.

These post hoc analyses provide adjusted comparative estimates subject to substantial residual confounding, including sedation intent and weaning readiness. Incomplete GCS/urine observations and late recording limit the severity measurements. The available data do not support a standard APACHE II score. eICU provides supportive external triangulation in a distinct population. MIMIC and eICU results are reported separately.

Tied historical laboratory timestamps can produce discrepancies in three stays when re-extracting upstream cohorts. Input checksums and frozen references identify such differences. They are reported rather than resolved by modifying model parameters.

## Tests and validation

The runner executes eight round-1 tests and eight round-2 tests. For standard unittest discovery after a completed run:

```powershell
$env:RESEARCH_PIPELINE_CONFIG = "D:\study_output\run_config.json"
python -B -m unittest discover -s tests -v
```

Primary numeric comparisons use a 1e-6 tolerance; severity reference comparisons use 1e-9. Checks cover 2,608 primary numeric values, 126 manuscript statistics and 232 severity fields. Comparison results and test logs are written to the workspace. Figure definitions and exports are implemented in `src/round1/figures.py`.

## License

Official reference SQL is from MIT-LCP `mimic-code`, commit `303d26c623dcc9c49cc0f204468d4acc2f063797`, under the MIT license included in `LICENSE`. The upstream license does not grant access to restricted research data.
