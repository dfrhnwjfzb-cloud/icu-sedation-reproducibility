"""Site-adjusted analyses of the archived eICU cohort."""

import json
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import src.runtime as runtime_module

logger = logging.getLogger(__name__)


ROOT = runtime_module.WORK / "primary"
T = ROOT / "tables"
D = ROOT / "derived"


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


def ridge(X, y, lam=1.0):
    Z = np.column_stack([np.ones(len(X)), X])

    def f(b):
        p = sigmoid(Z @ b)
        return -np.sum(
            y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)
        ) + 0.5 * lam * np.sum(b[1:] ** 2)

    def g(b):
        p = sigmoid(Z @ b)
        q = Z.T @ (p - y)
        q[1:] += lam * b[1:]
        return q

    r = minimize(
        f, np.zeros(Z.shape[1]), jac=g, method="L-BFGS-B", options={"maxiter": 600}
    )
    if not r.success:
        raise RuntimeError(r.message)
    return r.x, sigmoid(Z @ r.x)


def ess(w):
    return float(w.sum() ** 2 / (w @ w)) if w @ w else 0


def wmean(x, w):
    ok = np.isfinite(x) & np.isfinite(w)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if np.sum(w[ok]) else np.nan


def smd(x, a, w=None):
    x = np.asarray(x, float)
    a = np.asarray(a, int)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    m1 = wmean(x[a == 1], w[a == 1])
    m0 = wmean(x[a == 0], w[a == 0])
    v1 = wmean((x[a == 1] - m1) ** 2, w[a == 1])
    v0 = wmean((x[a == 0] - m0) ** 2, w[a == 0])
    den = np.sqrt((v1 + v0) / 2)
    return (m1 - m0) / den if den > 1e-12 else 0


def derive(d):
    z = d.copy()
    score = pd.to_numeric(z.sedation_score_6h, errors="coerce")
    scale = z.sedation_scale_6h.fillna("")
    z["deep_sedation"] = (
        (scale == "RASS") & (score <= -3) | (scale == "SAS") & (score <= 2)
    ).astype(int)
    z["agitated"] = (
        (scale == "RASS") & (score >= 1) | (scale == "SAS") & (score >= 5)
    ).astype(int)
    z["sedation_stratum"] = np.select(
        [
            z.deep_sedation == 1,
            (scale == "RASS") & score.between(-2, -1) | (scale == "SAS") & score.eq(3),
            (scale == "RASS") & score.eq(0) | (scale == "SAS") & score.eq(4),
            z.agitated == 1,
        ],
        ["deep", "moderate", "light/awake", "agitated"],
        default="unknown",
    )
    z["indication_group"] = np.select(
        [
            z.seizure == 1,
            z.withdrawal == 1,
            z.neurocritical == 1,
            z.postoperative == 1,
            z.delirium_proxy == 1,
            z.respiratory == 1,
        ],
        [
            "seizure/status",
            "withdrawal",
            "neurocritical",
            "postoperative",
            "delirium/agitation proxy",
            "respiratory",
        ],
        default="other/uncertain",
    )
    z["baseline_brady50"] = (
        pd.to_numeric(z.hr_min_pre6h, errors="coerce") < 50
    ).astype(int)
    z["baseline_hypotension60"] = (
        pd.to_numeric(z.map_min_pre6h, errors="coerce") < 60
    ).astype(int)
    z["icu_group"] = z.admission_type.fillna("Unknown").astype(str)
    return z


def matrix(d, include_hospital=True, alternative=False):
    num = [
        "age",
        "hours_from_icu_admit",
        "hours_since_last_prop_start",
        "gcs_6h",
        "hr_last",
        "map_last",
        "creatinine",
        "lactate",
        "minutes_since_last_prop",
    ]
    binary = [
        "first_icu_stay",
        "opioid_6h",
        "nmba_6h",
        "vasopressor_pre6h",
        "probable_mv",
        "seizure",
        "withdrawal",
        "neurocritical",
        "respiratory",
        "delirium_proxy",
        "postoperative",
        "deep_sedation",
        "agitated",
        "baseline_brady50",
        "baseline_hypotension60",
    ]
    cat = [
        "sex",
        "icu_group",
        "hospitaldischargeyear",
        "sedation_stratum",
        "indication_group",
    ]
    if include_hospital:
        cat.append("hospital_id")
    if alternative:
        num = ["age", "hours_from_icu_admit", "gcs_6h", "hr_last", "creatinine"]
        binary = [
            "first_icu_stay",
            "opioid_6h",
            "nmba_6h",
            "vasopressor_pre6h",
            "probable_mv",
            "seizure",
            "withdrawal",
            "neurocritical",
            "respiratory",
            "deep_sedation",
        ]
    out = pd.DataFrame(index=d.index)
    for c in num:
        x = pd.to_numeric(d[c], errors="coerce")
        mis = x.isna().astype(float)
        med = x.median()
        if not np.isfinite(med):
            med = 0
        x = x.fillna(med)
        sd = x.std(ddof=0)
        out[c] = (x - x.mean()) / (sd if sd > 1e-09 else 1)
        if mis.any():
            out[c + "__missing"] = mis
    for c in binary:
        out[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(float)
    out = pd.concat(
        [
            out,
            pd.get_dummies(
                d[cat].fillna("Unknown").astype(str),
                prefix=cat,
                drop_first=True,
                dtype=float,
            ),
        ],
        axis=1,
    )
    return out.loc[:, out.nunique(dropna=False) > 1]


def estimate(d, outcome="hospital_mortality", include_hospital=True, alternative=False):
    x = d.dropna(subset=[outcome]).copy()
    a = (x.treatment == "DEX").astype(int).to_numpy()
    y = x[outcome].astype(int).to_numpy()
    Xdf = matrix(x, include_hospital, alternative)
    X = Xdf.to_numpy()
    _, e = ridge(X, a, 2.0)
    m = []
    for arm in (0, 1):
        b, _ = ridge(X[a == arm], y[a == arm], 2.0)
        m.append(sigmoid(np.column_stack([np.ones(len(X)), X]) @ b))
    m0, m1 = m
    h = e * (1 - e)
    den = h.mean()
    z1 = h * (m1 + a / e * (y - m1))
    z0 = h * (m0 + (1 - a) / (1 - e) * (y - m0))
    mu1 = z1.mean() / den
    mu0 = z0.mean() / den
    psi1 = (z1 - mu1 * h) / den
    psi0 = (z0 - mu0 * h) / den
    rd = mu1 - mu0
    ifrd = psi1 - psi0
    se = ifrd.std(ddof=1) / np.sqrt(len(y))
    rr = mu1 / mu0
    iflog = psi1 / mu1 - psi0 / mu0
    selog = iflog.std(ddof=1) / np.sqrt(len(y))
    ow = np.where(a == 1, 1 - e, e)
    return {
        "n": len(x),
        "DEX_n": int(a.sum()),
        "MIDAZ_n": int((1 - a).sum()),
        "DEX_risk": mu1,
        "MIDAZ_risk": mu0,
        "risk_difference": rd,
        "rd_se": se,
        "rd_ci_low": rd - 1.96 * se,
        "rd_ci_high": rd + 1.96 * se,
        "risk_ratio": rr,
        "rr_ci_low": np.exp(np.log(rr) - 1.96 * selog),
        "rr_ci_high": np.exp(np.log(rr) + 1.96 * selog),
        "overlap_ess": ess(ow),
        "e_lt_005_pct": 100 * np.mean(e < 0.05),
        "e_gt_095_pct": 100 * np.mean(e > 0.95),
        "max_abs_weighted_smd": max(
            abs(smd(X[:, j], a, ow)) for j in range(X.shape[1])
        ),
        "_e": e,
        "_ow": ow,
        "_columns": list(Xdf.columns),
    }


def clean(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def continuous(d, outcome):
    x = d.dropna(subset=[outcome]).copy()
    a = (x.treatment == "DEX").astype(int).to_numpy()
    y = x[outcome].to_numpy(float)
    X = matrix(x).to_numpy()
    _, e = ridge(X, a, 2.0)
    w = np.where(a == 1, 1 - e, e)
    m1 = wmean(y[a == 1], w[a == 1])
    m0 = wmean(y[a == 0], w[a == 0])
    v1 = wmean((y[a == 1] - m1) ** 2, w[a == 1]) / ess(w[a == 1])
    v0 = wmean((y[a == 0] - m0) ** 2, w[a == 0]) / ess(w[a == 0])
    se = np.sqrt(v1 + v0)
    return {
        "n": len(x),
        "DEX_n": int(a.sum()),
        "MIDAZ_n": int((1 - a).sum()),
        "DEX_mean": m1,
        "MIDAZ_mean": m0,
        "mean_difference": m1 - m0,
        "mean_ci_low": m1 - m0 - 1.96 * se,
        "mean_ci_high": m1 - m0 + 1.96 * se,
        "overlap_ess": ess(w),
    }


def dl_meta(site):
    y = site.risk_difference.to_numpy()
    v = site.rd_se.to_numpy() ** 2
    w = 1 / v
    fixed = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - fixed) ** 2)
    df = len(y) - 1
    C = np.sum(w) - np.sum(w * w) / np.sum(w)
    tau = max(0, (Q - df) / C) if C > 0 else 0
    wr = 1 / (v + tau)
    mu = np.sum(wr * y) / np.sum(wr)
    se = np.sqrt(1 / np.sum(wr))
    I2 = max(0, (Q - df) / Q) * 100 if Q > 0 else 0
    return {
        "sites": len(y),
        "pooled_rd": mu,
        "ci_low": mu - 1.96 * se,
        "ci_high": mu + 1.96 * se,
        "tau2": tau,
        "Q": Q,
        "I2_pct": I2,
    }


def main():
    d = pd.read_csv(
        D / "eicu_external_candidates.csv", dtype={"stay_id": str, "hospital_id": str}
    )
    for c in d.columns:
        if c not in {
            "stay_id",
            "hospital_id",
            "treatment",
            "time_zero",
            "patient_id",
            "admission_id",
            "sex",
            "admission_type",
            "anchor_year_group",
            "sedation_scale_6h",
            "prop_exposure_level",
            "apacheadmissiondx",
            "last_airway_24h",
            "sedation_scale_4h",
            "raw_name",
        }:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = derive(d)
    d["probable_anchor"] = d.prop_exposure_level.isin(["STRICT", "PROBABLE"]).astype(
        int
    )
    support = (
        d[d.probable_anchor == 1]
        .groupby(["hospital_id", "treatment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for tx in ["DEX", "MIDAZ"]:
        if tx not in support:
            support[tx] = 0
    support["total"] = support.DEX + support.MIDAZ
    support["DEX_preference"] = support.DEX / support.total
    support["both_arm"] = ((support.DEX > 0) & (support.MIDAZ > 0)).astype(int)
    support["primary_support"] = ((support.DEX >= 5) & (support.MIDAZ >= 5)).astype(int)
    support["site_effect_support"] = (
        (support.DEX >= 10) & (support.MIDAZ >= 10)
    ).astype(int)
    support.to_csv(T / "hospital_support.csv", index=False, encoding="utf-8-sig")
    keep = set(support.loc[support.primary_support == 1, "hospital_id"].astype(str))
    primary = d[(d.probable_anchor == 1) & d.hospital_id.astype(str).isin(keep)].copy()
    e2 = primary[primary.probable_mv == 1].copy()
    strict = primary[primary.prop_exposure_level == "STRICT"].copy()
    primary.to_csv(D / "eicu_primary_cohort.csv", index=False, encoding="utf-8-sig")
    flow = [
        {
            "step": "BROAD propofol within 6h + next DEX/MIDAZ",
            "DEX_n": int((d.treatment == "DEX").sum()),
            "MIDAZ_n": int((d.treatment == "MIDAZ").sum()),
            "n": len(d),
        },
        {
            "step": "PROBABLE propofol within 60min (E1 candidate)",
            "DEX_n": int(((d.treatment == "DEX") & (d.probable_anchor == 1)).sum()),
            "MIDAZ_n": int(((d.treatment == "MIDAZ") & (d.probable_anchor == 1)).sum()),
            "n": int(d.probable_anchor.sum()),
        },
        {
            "step": "both-arm hospitals with >=5 per arm (primary E1)",
            "DEX_n": int((primary.treatment == "DEX").sum()),
            "MIDAZ_n": int((primary.treatment == "MIDAZ").sum()),
            "n": len(primary),
        },
        {
            "step": "E2 probable invasive-MV evidence",
            "DEX_n": int((e2.treatment == "DEX").sum()),
            "MIDAZ_n": int((e2.treatment == "MIDAZ").sum()),
            "n": len(e2),
        },
        {
            "step": "STRICT exposure within primary hospitals",
            "DEX_n": int((strict.treatment == "DEX").sum()),
            "MIDAZ_n": int((strict.treatment == "MIDAZ").sum()),
            "n": len(strict),
        },
    ]
    pd.DataFrame(flow).to_csv(
        T / "eicu_cohort_flow.csv", index=False, encoding="utf-8-sig"
    )
    r = estimate(primary)
    X = matrix(primary)
    a = (primary.treatment == "DEX").astype(int).to_numpy()
    _, e = ridge(X.to_numpy(), a, 2.0)
    ow = np.where(a == 1, 1 - e, e)
    bal = [
        {
            "variable": c,
            "raw_smd": smd(X.iloc[:, j], a),
            "overlap_weighted_smd": smd(X.iloc[:, j], a, ow),
        }
        for j, c in enumerate(X.columns)
    ]
    pd.DataFrame(bal).to_csv(
        T / "eicu_baseline_balance.csv", index=False, encoding="utf-8-sig"
    )
    rows = []
    for label, z in [
        ("PRIMARY_E1_PROBABLE_BOTH_ARM_HOSPITALS", primary),
        ("STRICT_EXPOSURE", strict),
        ("BROAD_EXPOSURE", d[d.hospital_id.astype(str).isin(keep)]),
        ("E2_PROBABLE_MV", e2),
        (
            "EXCLUDE_SEIZURE_WITHDRAWAL",
            primary[(primary.seizure == 0) & (primary.withdrawal == 0)],
        ),
        ("RASS_SAS_OBSERVED", primary[primary.sedation_stratum != "unknown"]),
    ]:
        if min((z.treatment == "DEX").sum(), (z.treatment == "MIDAZ").sum()) >= 30:
            q = clean(estimate(z))
            q.update(
                {
                    "analysis": label,
                    "outcome": "hospital_mortality",
                    "status": "ESTIMATED",
                }
            )
            rows.append(q)
        else:
            rows.append(
                {
                    "analysis": label,
                    "outcome": "hospital_mortality",
                    "status": "INSUFFICIENT_SUPPORT",
                    "n": len(z),
                }
            )
    q = clean(estimate(primary, "icu_mortality"))
    q.update(
        {
            "analysis": "SECONDARY_ICU_MORTALITY",
            "outcome": "icu_mortality",
            "status": "ESTIMATED",
        }
    )
    rows.append(q)
    for out in ["los_icu", "los_hospital"]:
        q = continuous(primary, out)
        q.update(
            {
                "analysis": "SECONDARY_" + out.upper(),
                "outcome": out,
                "status": "ESTIMATED",
            }
        )
        rows.append(q)
    pd.DataFrame(rows).to_csv(
        T / "eicu_primary_effect.csv", index=False, encoding="utf-8-sig"
    )
    site = []
    for h in support.loc[support.site_effect_support == 1, "hospital_id"].astype(str):
        z = primary[primary.hospital_id.astype(str) == h]
        q = clean(estimate(z, include_hospital=False, alternative=True))
        q.update({"hospital_id": h, "DEX_preference": (z.treatment == "DEX").mean()})
        site.append(q)
    site = pd.DataFrame(site)
    meta = dl_meta(site) if len(site) >= 2 else {}
    for k, v in meta.items():
        site["meta_" + k] = v
    site.to_csv(T / "site_specific_effects.csv", index=False, encoding="utf-8-sig")
    loo = []
    for h in sorted(primary.hospital_id.astype(str).unique()):
        z = primary[primary.hospital_id.astype(str) != h]
        q = clean(estimate(z))
        q.update({"omitted_hospital_id": h})
        loo.append(q)
    pd.DataFrame(loo).to_csv(
        T / "leave_one_hospital_out.csv", index=False, encoding="utf-8-sig"
    )
    br = []
    hy = []
    for hours in [6, 12, 24]:
        for threshold in [50, 55, 60]:
            col = f"hr_min_{hours}h"
            z = primary[pd.to_numeric(primary[col], errors="coerce").notna()].copy()
            z["event"] = (pd.to_numeric(z[col], errors="coerce") < threshold).astype(
                int
            )
            q = clean(estimate(z, "event"))
            q.update(
                {
                    "window_hours": hours,
                    "hr_threshold": threshold,
                    "outcome": "bradycardia",
                    "status": "ESTIMATED",
                }
            )
            br.append(q)
        for threshold in [60, 65]:
            col = f"map_min_{hours}h"
            z = primary[pd.to_numeric(primary[col], errors="coerce").notna()].copy()
            z["event"] = (pd.to_numeric(z[col], errors="coerce") < threshold).astype(
                int
            )
            q = clean(estimate(z, "event"))
            q.update(
                {
                    "window_hours": hours,
                    "map_threshold": threshold,
                    "outcome": "hypotension",
                    "status": "ESTIMATED",
                }
            )
            hy.append(q)
    z = primary.copy()
    z["event"] = ((z.vasopressor_pre6h == 0) & (z.vasopressor_post6h == 1)).astype(int)
    q = clean(estimate(z, "event"))
    q.update(
        {
            "window_hours": 6,
            "map_threshold": "new vasopressor",
            "outcome": "vasopressor_initiation",
            "status": "ESTIMATED",
        }
    )
    hy.append(q)
    pd.DataFrame(br).to_csv(
        T / "bradycardia_validation.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(hy).to_csv(
        T / "hypotension_validation.csv", index=False, encoding="utf-8-sig"
    )
    gate = {
        "frozen_on": "2026-10-10",
        "primary_population": "E1 PROBABLE propofol; both-arm hospitals >=5 per arm",
        "identification_state": "PROCEED_CAUSAL_WITH_CAVEATS",
        "hard_stops": [],
        "soft_warnings": [
            "point-record propofol continuity is probabilistic",
            "RASS/SAS missingness is high",
            "hospital mortality follow-up varies with discharge",
            "patient-level sedation/weaning intent remains proxy-measured",
        ],
        "positivity": "RESTRICTABLE",
        "measurement_validity": "PARTIAL",
        "primary_metrics": clean(r),
        "site_meta": meta,
    }
    (ROOT / "logs" / "external_design_gate.json").write_text(
        json.dumps(
            gate,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.item() if hasattr(x, "item") else str(x),
        ),
        encoding="utf-8",
    )
    logger.info(
        "eICU analysis complete: %d stays across %d hospitals", len(primary), len(site)
    )
