"""Baseline feature construction, ridge propensity models and balance diagnostics."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import src.runtime as runtime_module

ROOT = runtime_module.WORK / "primary"
TABLES = ROOT / "tables"
DERIVED = ROOT / "derived"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def ridge_logit(X, y, lam=1.0):
    X1 = np.column_stack([np.ones(len(X)), X])

    def fn(b):
        p = sigmoid(X1 @ b)
        loss = -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        return loss + 0.5 * lam * np.sum(b[1:] ** 2)

    def gr(b):
        p = sigmoid(X1 @ b)
        g = X1.T @ (p - y)
        g[1:] += lam * b[1:]
        return g

    # Unpenalized intercept; L2 penalty applies to the remaining coefficients.
    out = minimize(
        fn,
        np.zeros(X1.shape[1]),
        jac=gr,
        method="L-BFGS-B",
        options={"maxiter": 800, "ftol": 1e-10},
    )
    if not out.success:
        raise RuntimeError(out.message)
    return out.x, sigmoid(X1 @ out.x)


def design_matrix(df, alternative=False):
    x = df.copy()
    numeric = [
        "age",
        "hours_from_icu_admit",
        "hours_since_first_prop_start",
        "prop_overlap_hours_24h",
        "prop_rate_t0",
        "prop_rate_mean_0_2h",
        "prop_rate_mean_2_6h",
        "hours_since_sedation",
        "gcs_6h",
        "hours_since_mv_start",
        "peep",
        "fio2",
        "rr_spont",
        "rr_set",
        "heart_rate",
        "map",
        "sbp",
        "heart_rate_min_6h",
        "map_min_6h",
        "creatinine",
        "bun",
        "bicarbonate",
        "bilirubin",
        "lactate",
        "pf_ratio",
        "opioid_types_6h",
        "opioid_starts_6h",
        "fentanyl_rate_mcg_h_t0",
        "vasopressor_count_t0",
    ]
    binary = [
        "first_icu_stay",
        "invasive_mv_t0",
        "opioid_active_t0",
        "fentanyl_active_t0",
        "hydromorphone_active_t0",
        "morphine_active_t0",
        "nmba_active_t0",
        "nmba_recent_2h",
        "nmba_recent_4h",
        "nmba_recent_6h",
        "vasopressor_active_t0",
        "controlled_mode",
        "spontaneous_mode",
        "deep_sedation",
        "agitated",
        "bradycardia_proxy",
        "hypotension_proxy",
    ]
    categorical = [
        "sex",
        "anchor_year_group",
        "icu_group",
        "sedation_stratum",
        "admission_type",
    ]
    if alternative:
        numeric = [
            z
            for z in numeric
            if z not in {"bilirubin", "pf_ratio", "fentanyl_rate_mcg_h_t0"}
        ]
        categorical = ["sex", "anchor_year_group", "icu_group", "sedation_stratum"]
    out = pd.DataFrame(index=x.index)
    for c in numeric:
        z = pd.to_numeric(x[c], errors="coerce")
        miss = z.isna().astype(float)
        med = z.median()
        if not np.isfinite(med):
            med = 0.0
        z = z.fillna(med)
        sd = z.std(ddof=0)
        out[c] = (z - z.mean()) / (sd if sd > 1e-09 else 1.0)
        if miss.any():
            out[c + "__missing"] = miss
    for c in binary:
        out[c] = pd.to_numeric(x[c], errors="coerce").fillna(0).astype(float)
    cats = pd.get_dummies(
        x[categorical].fillna("Unknown").astype(str),
        prefix=categorical,
        drop_first=True,
        dtype=float,
    )
    out = pd.concat([out, cats], axis=1)
    out = out.loc[:, out.nunique(dropna=False) > 1]
    return out


def wmean(x, w):
    ok = np.isfinite(x) & np.isfinite(w)
    return np.sum(x[ok] * w[ok]) / np.sum(w[ok]) if np.sum(w[ok]) else np.nan


def smd(x, a, w=None):
    x = np.asarray(x, float)
    a = np.asarray(a, int)
    if w is None:
        w = np.ones(len(x))
    w = np.asarray(w, float)
    m1 = wmean(x[a == 1], w[a == 1])
    m0 = wmean(x[a == 0], w[a == 0])
    v1 = wmean((x[a == 1] - m1) ** 2, w[a == 1])
    v0 = wmean((x[a == 0] - m0) ** 2, w[a == 0])
    den = np.sqrt((v1 + v0) / 2)
    return (m1 - m0) / den if den > 1e-12 else 0.0


def ess(w):
    w = np.asarray(w, float)
    return float(np.sum(w) ** 2 / np.sum(w * w)) if np.sum(w * w) > 0 else 0.0


def add_derived(d):
    mode = (
        d.ventilator_mode.fillna("") + " " + d.ventilator_mode_hamilton.fillna("")
    ).str.upper()
    d["controlled_mode"] = mode.str.contains(
        "CMV|ASSIST|PRVC|VOL/AC|PCV|APV|CONTROL", regex=True
    ).astype(int)
    d["spontaneous_mode"] = mode.str.contains("CPAP|PSV|SPONT|SBT", regex=True).astype(
        int
    )
    score = pd.to_numeric(d.sedation_score, errors="coerce")
    rass = d.sedation_scale == "RASS"
    sas = d.sedation_scale == "SAS"
    d["deep_sedation"] = (rass & (score <= -3) | sas & (score <= 2)).astype(int)
    d["agitated"] = (rass & (score >= 1) | sas & (score >= 5)).astype(int)
    d["sedation_stratum"] = np.select(
        [
            d.deep_sedation == 1,
            rass & score.between(-2, -1) | sas & score.between(3, 3),
            rass & score.between(0, 0) | sas & score.between(4, 4),
            d.agitated == 1,
        ],
        ["deep", "moderate", "light/awake", "agitated"],
        default="unknown",
    )
    care = d.first_careunit.fillna("Unknown")
    d["icu_group"] = np.select(
        [
            care.str.contains("Neuro", case=False),
            care.str.contains("Cardiac|Coronary|CVICU|CCU", case=False),
            care.str.contains("Trauma", case=False),
            care.str.contains("Surgical|SICU", case=False),
            care.str.contains("Medical|MICU", case=False),
        ],
        ["Neuro", "Cardiac/CV", "Trauma", "Surgical", "Medical"],
        default="Other",
    )
    d["bradycardia_proxy"] = (
        pd.to_numeric(d.heart_rate_min_6h, errors="coerce") < 50
    ).astype(int)
    d["hypotension_proxy"] = (pd.to_numeric(d.map_min_6h, errors="coerce") < 60).astype(
        int
    )
    d["controlled_deep_context"] = (
        (d.deep_sedation == 1)
        | (d.nmba_recent_4h == 1)
        | (d.controlled_mode == 1) & (d.spontaneous_mode == 0)
    ).astype(int)
    return d


def summarize_risk(name, x):
    a = (x.treatment == "DEX").astype(int).to_numpy()
    X = design_matrix(x)
    beta, e = ridge_logit(X.to_numpy(), a, lam=1.0)
    ow = np.where(a == 1, 1 - e, e)
    lower = max(np.quantile(e[a == 1], 0.01), np.quantile(e[a == 0], 0.01))
    upper = min(np.quantile(e[a == 1], 0.99), np.quantile(e[a == 0], 0.99))
    cs = (e >= lower) & (e <= upper)
    rows = []
    for j, c in enumerate(X.columns):
        raw = smd(X.iloc[:, j], a)
        adj = smd(X.iloc[:, j], a, ow)
        rows.append(
            {
                "risk_set": name,
                "variable": c,
                "raw_smd": raw,
                "overlap_weighted_smd": adj,
            }
        )
    bal = pd.DataFrame(rows)
    result = {
        "risk_set": name,
        "n": len(x),
        "DEX_n": int(a.sum()),
        "MIDAZ_n": int((1 - a).sum()),
        "DEX_pct": 100 * a.mean(),
        "mv_pct": 100 * x.invasive_mv_t0.mean(),
        "rass_sas_observed_pct": 100 * (x.sedation_stratum != "unknown").mean(),
        "deep_pct": 100 * x.deep_sedation.mean(),
        "opioid_active_pct": 100 * x.opioid_active_t0.mean(),
        "nmba_recent4_pct": 100 * x.nmba_recent_4h.mean(),
        "e_lt_005_pct": 100 * np.mean(e < 0.05),
        "e_gt_095_pct": 100 * np.mean(e > 0.95),
        "common_support_pct": 100 * np.mean(cs),
        "overlap_ess_total": ess(ow),
        "overlap_ess_DEX": ess(ow[a == 1]),
        "overlap_ess_MIDAZ": ess(ow[a == 0]),
        "max_abs_raw_smd": bal.raw_smd.abs().max(),
        "max_abs_overlap_smd": bal.overlap_weighted_smd.abs().max(),
        "n_abs_overlap_smd_gt_0_1": int((bal.overlap_weighted_smd.abs() > 0.1).sum()),
        "ps_min": e.min(),
        "ps_p01": np.quantile(e, 0.01),
        "ps_p50": np.quantile(e, 0.5),
        "ps_p99": np.quantile(e, 0.99),
        "ps_max": e.max(),
        "cs_lower": lower,
        "cs_upper": upper,
    }
    return (
        result,
        bal,
        pd.DataFrame(
            {
                "stay_id": x.stay_id.astype(str),
                "risk_set": name,
                "treatment": x.treatment,
                "propensity": e,
                "overlap_weight": ow,
                "common_support": cs.astype(int),
            }
        ),
    )


def grouped_context(d, cols, name):
    rows = []
    for rs, mask in [
        ("A_all_active_propofol", d.prop_active_t0 == 1),
        ("B_invasive_mv", (d.prop_active_t0 == 1) & (d.invasive_mv_t0 == 1)),
        (
            "C_controlled_deep",
            (d.prop_active_t0 == 1)
            & (d.invasive_mv_t0 == 1)
            & (d.controlled_deep_context == 1),
        ),
    ]:
        x = d[mask]
        for tx, g in x.groupby("treatment"):
            row = {"risk_set": rs, "treatment": tx, "n": len(g)}
            for c in cols:
                z = pd.to_numeric(g[c], errors="coerce")
                row[c + "_mean_or_prop"] = z.mean()
                row[c + "_observed_n"] = int(z.notna().sum())
            rows.append(row)
    pd.DataFrame(rows).to_csv(TABLES / name, index=False, encoding="utf-8-sig")
