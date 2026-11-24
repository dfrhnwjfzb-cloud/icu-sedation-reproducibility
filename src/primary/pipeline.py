import json
import logging
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

import src.design as design_module
import src.effects as effects_module
import src.external as external_module
import src.primary.additional as additional_module
import src.primary.io as c
import src.primary.manuscript as manuscript_module

logger = logging.getLogger(__name__)


sys.dont_write_bytecode = True


R = c.ROOT
CHECKS = []


def compare(actual, expected, label, keys):
    a = pd.read_csv(actual)
    b = pd.read_csv(expected)
    if keys:
        a = a.set_index(keys).sort_index()
        b = b.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise AssertionError(label + " row identities differ")
    for col in b.columns:
        if col not in a:
            assert not pd.api.types.is_numeric_dtype(b[col]), (
                label + " missing numeric column " + col
            )
            continue
        if not pd.api.types.is_numeric_dtype(b[col]) or pd.api.types.is_bool_dtype(
            b[col]
        ):
            continue
        x = pd.to_numeric(a[col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(b[col], errors="coerce").to_numpy(float)
        ok = np.isclose(x, y, atol=1e-06, rtol=0, equal_nan=True)
        for idx, xx, yy, passed in zip(b.index, x, y, ok):
            CHECKS.append(
                {
                    "analysis": label,
                    "row": str(idx),
                    "field": col,
                    "reproduced": xx,
                    "frozen": yy,
                    "absolute_difference": abs(xx - yy),
                    "pass": bool(passed),
                }
            )
    pd.DataFrame(CHECKS).to_csv(
        R / "validation/numeric_checks.csv", index=False, encoding="utf-8-sig"
    )
    assert all(q["pass"] for q in CHECKS), (
        "Numerical reproduction failed; see validation/numeric_checks.csv"
    )


def main():
    needed = [
        R / "SOURCE_MANIFEST.json",
        R / "inputs",
        R / "expected",
        R / "reference_archive",
    ]
    missing = [str(p.relative_to(R)) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Code-only distribution: supply authorized inputs and reference files first. Missing: "
            + ", ".join(missing)
            + ". See README.md."
        )
    started = time.time()
    before = c.snapshot()
    for name in ["results", "derived", "logs", "validation"]:
        (R / name).mkdir(exist_ok=True)
    manifests = json.loads((R / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    for m in manifests:
        assert c.sha(R / m["package_path"]) == m["sha256"], m["package_path"]
    design = design_module
    est = effects_module
    est.design = design
    d = pd.read_csv(R / "inputs/mimic_design_candidates.csv", dtype={"stay_id": str})
    d = design.add_derived(
        d[d.treatment.isin(["DEX", "MIDAZ"])].drop(columns=c.REMOVED[:6]).copy()
    )
    o = pd.read_csv(R / "inputs/mimic_outcomes.csv", dtype={"stay_id": str})
    joined = d.merge(o, on="stay_id", how="inner", suffixes=("", "_out"))
    m = joined[(joined.prop_active_t0 == 1) & (joined.invasive_mv_t0 == 1)].copy()
    assert (
        len(m) == 3669 and m.stay_id.is_unique and (not set(c.REMOVED).intersection(m))
    )
    m.to_csv(
        R / "derived/mimic_primary_no_indication_proxies.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _, diag = c.diagnostics(design, m, "removed")
    rows = []
    for outcome in c.BINARY + c.CONTINUOUS:
        q = (
            est.estimate(m, outcome=outcome)
            if outcome in c.BINARY
            else est.continuous_ow(m, outcome)
        )
        q["outcome"] = outcome
        rows.append(q)
        logger.info("MIMIC outcome reproduced: %s", outcome)
    pd.DataFrame(rows).to_csv(
        R / "results/removed_proxy_effects.csv", index=False, encoding="utf-8-sig"
    )
    compare(
        R / "results/removed_proxy_effects.csv",
        R / "expected/mimic/removed_proxy_effects.csv",
        "MIMIC all 11 outcomes",
        ["outcome"],
    )
    compare(
        R / "results/removed_balance.csv",
        R / "expected/mimic/removed_balance.csv",
        "MIMIC balance",
        ["variable"],
    )
    ext = external_module
    ext.ROOT = R / "eicu_output"
    ext.T = ext.ROOT / "tables"
    ext.D = ext.ROOT / "derived"
    for p in [ext.T, ext.D, ext.ROOT / "logs"]:
        p.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(
        R / "inputs/eicu_external_candidates.csv",
        ext.D / "eicu_external_candidates.csv",
    )
    ext.main()
    logger.info(
        "eICU models, site effects, leave-one-out and positive controls refitted."
    )
    for file, keys in [
        ("eicu_primary_effect.csv", ["analysis"]),
        ("eicu_baseline_balance.csv", ["variable"]),
        ("eicu_cohort_flow.csv", ["step"]),
        ("hospital_support.csv", ["hospital_id"]),
        ("site_specific_effects.csv", ["hospital_id"]),
        ("leave_one_hospital_out.csv", ["omitted_hospital_id"]),
        ("bradycardia_validation.csv", ["window_hours", "hr_threshold"]),
        ("hypotension_validation.csv", ["window_hours", "map_threshold"]),
    ]:
        compare(ext.T / file, R / "expected/eicu" / file, "eICU " + file, keys)
    reproduce_additional = additional_module
    reproduce_additional.main()
    for file, keys in [
        ("sensitivity_results.csv", ["sensitivity_id"]),
        ("heterogeneity_results.csv", ["subgroup_variable", "subgroup_level"]),
        ("transported_effects.csv", ["analysis"]),
        ("cohort_flow.csv", ["step_order"]),
        ("unmeasured_confounding.csv", []),
    ]:
        compare(
            R / "results/updated" / file,
            R / "expected/mimic" / file,
            "MIMIC additional " + file,
            keys,
        )
    descriptive = []
    missing = []
    for var in m:
        for tx, g in m.groupby("treatment"):
            x = g[var]
            missing.append(
                {
                    "variable": var,
                    "treatment": tx,
                    "n": len(g),
                    "missing_n": int(x.isna().sum()),
                    "missing_pct": x.isna().mean() * 100,
                }
            )
            if pd.api.types.is_numeric_dtype(x):
                qs = x.quantile([0.25, 0.5, 0.75])
                descriptive.append(
                    {
                        "variable": var,
                        "treatment": tx,
                        "observed_n": int(x.notna().sum()),
                        "q1": qs.iloc[0],
                        "median": qs.iloc[1],
                        "q3": qs.iloc[2],
                        "mean": x.mean(),
                    }
                )
            elif var in [
                "sex",
                "icu_group",
                "anchor_year_group",
                "admission_type",
                "sedation_stratum",
            ]:
                for level, n in x.value_counts(dropna=False).items():
                    descriptive.append(
                        {
                            "variable": var,
                            "treatment": tx,
                            "level": str(level),
                            "count": n,
                            "percent": 100 * n / len(g),
                        }
                    )
    pd.DataFrame(descriptive).to_csv(
        R / "results/raw_baseline_descriptives.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(missing).to_csv(
        R / "results/complete_missingness.csv", index=False, encoding="utf-8-sig"
    )
    audit = manuscript_module.audit
    claim_summary = audit(R, m, diag, rows, ext)
    assert c.snapshot() == before, "Immutable input/reference files changed"
    result = {
        "numeric_check_count": len(CHECKS),
        "all_numeric_checks_pass": all(q["pass"] for q in CHECKS),
        "maximum_absolute_difference": max(
            q["absolute_difference"]
            for q in CHECKS
            if np.isfinite(q["absolute_difference"])
        ),
        "manuscript": claim_summary,
        "inputs_and_reference_files_unchanged": True,
        "execution_level": "frozen extracted cohorts to estimates; raw SQL extraction not rerun",
        "seconds": time.time() - started,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "blas_threads": 1,
        "folds": "MIMIC row position modulo five; eICU no cross-fitting in frozen implementation",
    }
    c.dump("validation/RUN_VALIDATION.json", result)
    (R / "requirements-tested.txt").write_text(
        f"numpy=={np.__version__}\npandas=={pd.__version__}\nscipy=={scipy.__version__}\nlxml\n",
        encoding="utf-8",
    )
    logger.info(
        "Primary validation: %d numeric checks; maximum difference %.3g",
        result["numeric_check_count"],
        result["maximum_absolute_difference"],
    )
