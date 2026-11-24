"""SOFA augmentation, severity strata and ICU restriction analyses."""

import datetime
import json
import logging
import types

import numpy as np
import pandas as pd

import src.design as base
import src.effects as est
import src.round1.setup as p

logger = logging.getLogger(__name__)


R = p.R


ORGS = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]
OUTCOMES = ["alive_sustainably_liberated_day28", "hospital_mortality"]


def freeze():
    path = R / "analysis_config.json"
    cfg = {
        "created_before_new_effects": True,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "strata": {"Low": [0, 5], "Intermediate": [6, 9], "High": [10, 24]},
        "rationale": "Fixed clinically interpretable integer score bands, pooled distribution reported separately; no threshold selected using treatment-outcome effects. Short windows and missing-zero scores limit clinical calibration.",
        "score": "MIT-LCP component thresholds; observed pre-t0 history; absent organs explicitly zero plus missingness indicators",
        "model_S": "Original baseline terms plus centered/scaled SOFA and SOFA squared and six organ missing indicators; constants removed",
        "model_G": "Same augmentation using raw complete GCS components recorded and stored before t0; add raw GCS missing indicator",
        "ridge_ps": 1.0,
        "ridge_outcome": 2.0,
        "folds": "row position modulo 5, exactly original implementation",
        "sparse_warning_arm_n": 60,
        "no_estimate_if_arm_n_below": 20,
        "ess_warning_per_arm": 50,
        "balance_warning_max_abs_smd": 0.1,
        "post_hoc": True,
        "APACHE_II": "INFEASIBLE unless all critical domains verifiable; no score/no Model A",
        "seed": "Deterministic; no random split",
        "source_cohort_sha256": p.sha(R / "data/model0_cohort.csv"),
    }
    if path.exists():
        existing = json.loads(path.read_text())
        assert existing["source_cohort_sha256"] == cfg["source_cohort_sha256"]
        return existing
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


def matrix(x, score_col=None, alternative=False):
    X = base.design_matrix(x, alternative=alternative)
    if score_col:
        for name, z in [
            (score_col, x[score_col].astype(float)),
            (score_col + "_squared", x[score_col].astype(float) ** 2),
        ]:
            X[name] = (z - z.mean()) / max(z.std(ddof=0), 1e-09)
        for c in [o + "_missing" for o in ORGS]:
            X[c] = x[c].astype(float)
        if score_col == "pre_t0_sofa_raw_gcs":
            X["cns_raw_missing"] = x.cns_raw_observed.isna().astype(float)
    return X.loc[:, X.nunique(dropna=False) > 1]


def design_for(score_col=None):
    d = types.SimpleNamespace(
        **{k: getattr(base, k) for k in dir(base) if not k.startswith("__")}
    )
    d.design_matrix = lambda x, alternative=False: matrix(x, score_col, alternative)
    return d


def balance(x, weights, variables):
    a = (x.treatment == "DEX").astype(int).to_numpy()
    rows = []
    for c in variables:
        v = pd.to_numeric(x[c], errors="coerce").to_numpy(float)
        rr = {
            "variable": c,
            "raw_smd": base.smd(v, a),
            "weighted_smd": base.smd(v, a, weights),
        }
        for arm, label in [(1, "DEX"), (0, "MIDAZ")]:
            k = a == arm
            rr[label + "_observed_n"] = int(np.isfinite(v[k]).sum())
            rr[label + "_missing_pct"] = 100 * np.mean(~np.isfinite(v[k]))
            rr[label + "_raw_mean"] = base.wmean(v[k], np.ones(k.sum()))
            rr[label + "_weighted_mean"] = base.wmean(v[k], weights[k])
        rows.append(rr)
    return pd.DataFrame(rows)


def fit(x, model, group, score_col):
    x = x.copy()
    a = (x.treatment == "DEX").astype(int).to_numpy()
    X = matrix(x, score_col)
    assert np.isfinite(X.to_numpy()).all()
    beta, e = base.ridge_logit(X.to_numpy(), a, 1.0)
    ow = np.where(a == 1, 1 - e, e)
    assert np.isfinite(e).all() and (e > 0).all() and (e < 1).all()
    Xbal = balance(pd.concat([x[["treatment"]], X], axis=1), ow, X.columns)
    Xbal.insert(0, "group", group)
    Xbal.insert(0, "model", model)
    Xbal.to_csv(R / "outputs" / f"balance_{group}_{model}.csv", index=False)
    sb = balance(x, ow, ["pre_t0_sofa"] + ORGS + [o + "_missing" for o in ORGS])
    sb.to_csv(R / "outputs" / f"severity_balance_{group}_{model}.csv", index=False)
    diag = {
        "model": model,
        "group": group,
        "n": len(x),
        "DEX_n": int(a.sum()),
        "MIDAZ_n": int((1 - a).sum()),
        "model_terms": len(X.columns),
        "ESS_DEX": base.ess(ow[a == 1]),
        "ESS_MIDAZ": base.ess(ow[a == 0]),
        "max_abs_smd": Xbal.weighted_smd.abs().max(),
        "max_abs_score_smd": sb.weighted_smd.abs().max(),
        "n_smd_gt_01": int((Xbal.weighted_smd.abs() > 0.1).sum()),
        "ps_min": e.min(),
        "ps_max": e.max(),
        "ps_extreme_pct": 100 * np.mean((e < 0.05) | (e > 0.95)),
        "ps_01_overlap_low": max(
            np.quantile(e[a == 1], 0.01), np.quantile(e[a == 0], 0.01)
        ),
        "ps_99_overlap_high": min(
            np.quantile(e[a == 1], 0.99), np.quantile(e[a == 0], 0.99)
        ),
        "matrix_rank": int(np.linalg.matrix_rank(X.to_numpy())),
        "condition_ridge": float(
            np.linalg.cond(X.to_numpy().T @ X.to_numpy() + np.eye(X.shape[1]))
        ),
    }
    diag["caution"] = (
        "; ".join(
            (
                q
                for q, test in [
                    ("Small arm", min(diag["DEX_n"], diag["MIDAZ_n"]) < 60),
                    ("ESS below 50", min(diag["ESS_DEX"], diag["ESS_MIDAZ"]) < 50),
                    ("Residual imbalance", diag["max_abs_smd"] > 0.1),
                    ("Severity components imbalance", diag["max_abs_score_smd"] > 0.1),
                    (
                        "Limited central PS overlap",
                        diag["ps_01_overlap_low"] >= diag["ps_99_overlap_high"],
                    ),
                ]
                if test
            )
        )
        or "No listed support warning; residual confounding remains high"
    )
    ps = x[["stay_id", "treatment", "pre_t0_sofa"]].copy()
    ps["propensity"] = e
    ps["overlap_weight"] = ow
    ps.to_csv(R / "data" / f"weights_{group}_{model}.csv", index=False)
    est.design = design_for(score_col)
    rows = []
    for out in OUTCOMES:
        rr = est.estimate(x, out)
        rr.update(diag)
        rr["outcome"] = out
        for arm, label in [(1, "DEX"), (0, "MIDAZ")]:
            rr[label + "_events"] = int(x.loc[a == arm, out].sum())
            rr[label + "_raw_risk"] = x.loc[a == arm, out].mean()
        rows.append(rr)
    logger.info(
        f"{group} / {model}: n={len(x)}, terms={X.shape[1]}, max SMD={diag['max_abs_smd']:.3f}"
    )
    return rows


def main():
    cfg = freeze()
    x = p.load_cohort()
    scores = pd.read_csv(R / "data/pre_t0_sofa.csv", dtype={"stay_id": str})
    x = x.merge(scores, on="stay_id", validate="one_to_one")
    assert len(x) == 3669 and x.stay_id.is_unique
    w = pd.read_csv(R / "inputs/original_weights.csv")
    assert x.stay_id.astype(str).tolist() == w.stay_id.astype(str).tolist()
    a = (x.treatment == "DEX").astype(int).to_numpy()
    _, e = base.ridge_logit(matrix(x).to_numpy(), a, 1.0)
    assert np.max(abs(e - w.propensity)) < 1e-10
    scorevars = (
        [
            "pre_t0_sofa",
            "pre_t0_sofa_strict",
            "missing_organs",
            "window_hours",
            "short_window",
        ]
        + ORGS
        + [o + "_missing" for o in ORGS]
        + ["cns_raw_observed", "gcs_default15_suspect_n"]
    )
    b = balance(x, w.overlap_weight.to_numpy(), scorevars)
    b.to_csv(R / "outputs/severity_balance_original_weights.csv", index=False)
    b.to_csv(R / "outputs/score_coverage_balance.csv", index=False)
    pd.crosstab(x.pre_t0_sofa, x.treatment).to_csv(
        R / "outputs/pooled_sofa_distribution.csv"
    )
    rows = []
    for model, col in [
        ("Model0", None),
        ("ModelS", "pre_t0_sofa"),
        ("ModelG_rawGCS", "pre_t0_sofa_raw_gcs"),
    ]:
        rows.extend(fit(x, model, "Full", col))
    pd.DataFrame(rows).to_csv(R / "outputs/model_comparison.csv", index=False)
    expected = pd.read_csv(R / "outputs/model0_reproduction.csv").set_index("outcome")
    for row in rows[:2]:
        assert (
            abs(
                row["risk_difference"] - expected.loc[row["outcome"], "risk_difference"]
            )
            < 1e-09
        )
    sub = []
    for group, (lo, hi) in cfg["strata"].items():
        z = x[x.pre_t0_sofa.between(lo, hi)]
        if (
            min((z.treatment == "DEX").sum(), (z.treatment == "MIDAZ").sum())
            < cfg["no_estimate_if_arm_n_below"]
        ):
            sub.append({"group": group, "n": len(z), "status": "NOT_ESTIMABLE_SPARSE"})
            continue
        sub.extend(fit(z, "ModelS", group, "pre_t0_sofa"))
    pd.DataFrame(sub).to_csv(R / "outputs/sofa_subgroups.csv", index=False)
    icu = []
    for group, z in [
        ("NonCardiacCV", x[x.icu_group != "Cardiac/CV"]),
        ("CardiacCV", x[x.icu_group == "Cardiac/CV"]),
    ]:
        for model, col in [("Model0", None), ("ModelS", "pre_t0_sofa")]:
            icu.extend(fit(z, model, group, col))
    pd.DataFrame(icu).to_csv(R / "outputs/icu_restriction.csv", index=False)
    pd.DataFrame(rows + sub + icu).to_csv(R / "outputs/all_models.csv", index=False)
    logger.info("Model fitting complete; APACHE II inputs are incomplete.")
