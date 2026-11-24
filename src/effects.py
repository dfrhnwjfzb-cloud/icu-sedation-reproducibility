"""Augmented overlap-weighted risk contrasts with fivefold outcome regression."""

import math

import numpy as np

import src.design as design
import src.runtime as runtime_module

ROOT = runtime_module.WORK / "primary"
TABLES = ROOT / "tables"
DERIVED = ROOT / "derived"


def outcome_models(X, a, y):
    m = np.zeros((len(y), 2))
    # Retain source row order: each stay keeps its original outcome-regression fold.
    folds = np.arange(len(y)) % 5
    for k in range(5):
        tr = folds != k
        te = ~tr
        for arm in (0, 1):
            sel = tr & (a == arm)
            b, _ = design.ridge_logit(X[sel], y[sel], lam=2.0)
            m[te, arm] = design.sigmoid(np.column_stack([np.ones(te.sum()), X[te]]) @ b)
    return m[:, 0], m[:, 1]


def estimate(
    df,
    outcome="alive_sustainably_liberated_day28",
    alternative=False,
    trim=None,
    unaugmented=False,
):
    x = df.dropna(subset=[outcome]).copy()
    y = x[outcome].astype(int).to_numpy()
    a = (x.treatment == "DEX").astype(int).to_numpy()
    Xdf = design.design_matrix(x, alternative=alternative)
    X = Xdf.to_numpy()
    _, e = design.ridge_logit(X, a, lam=1.0)
    if trim is not None:
        keep = (e >= trim) & (e <= 1 - trim)
        x = x.iloc[np.flatnonzero(keep)].copy()
        y = y[keep]
        a = a[keep]
        X = X[keep]
        e = e[keep]
    h = e * (1 - e)
    den = h.mean()
    if unaugmented:
        w1 = (1 - e) * a
        w0 = e * (1 - a)
        mu1 = np.sum(w1 * y) / np.sum(w1)
        mu0 = np.sum(w0 * y) / np.sum(w0)
        psi1 = w1 * (y - mu1) / (w1.mean() + 1e-12)
        psi0 = w0 * (y - mu0) / (w0.mean() + 1e-12)
    else:
        m0, m1 = outcome_models(X, a, y)
        z1 = h * (m1 + a / e * (y - m1))
        z0 = h * (m0 + (1 - a) / (1 - e) * (y - m0))
        mu1 = z1.mean() / den
        mu0 = z0.mean() / den
        psi1 = (z1 - mu1 * h) / den
        psi0 = (z0 - mu0 * h) / den
    rd = mu1 - mu0
    ifrd = psi1 - psi0
    se = ifrd.std(ddof=1) / math.sqrt(len(y))
    lo = rd - 1.96 * se
    hi = rd + 1.96 * se
    rr = mu1 / mu0 if mu0 > 0 else np.nan
    if mu1 > 0 and mu0 > 0:
        iflog = psi1 / mu1 - psi0 / mu0
        selog = iflog.std(ddof=1) / math.sqrt(len(y))
        rrlo = np.exp(np.log(rr) - 1.96 * selog)
        rrhi = np.exp(np.log(rr) + 1.96 * selog)
    else:
        rrlo = rrhi = np.nan
    return {
        "n": len(y),
        "DEX_n": int(a.sum()),
        "MIDAZ_n": int((1 - a).sum()),
        "DEX_standardized_risk": mu1,
        "MIDAZ_standardized_risk": mu0,
        "risk_difference": rd,
        "rd_ci_low": lo,
        "rd_ci_high": hi,
        "risk_ratio": rr,
        "rr_ci_low": rrlo,
        "rr_ci_high": rrhi,
        "overlap_ess": design.ess(np.where(a == 1, 1 - e, e)),
        "e_lt_005_pct": 100 * np.mean(e < 0.05),
        "e_gt_095_pct": 100 * np.mean(e > 0.95),
        "estimator": "overlap-weighted AIPW"
        if not unaugmented
        else "overlap weighting",
    }


def continuous_ow(df, outcome):
    x = df.dropna(subset=[outcome]).copy()
    a = (x.treatment == "DEX").astype(int).to_numpy()
    y = x[outcome].to_numpy(float)
    X = design.design_matrix(x).to_numpy()
    _, e = design.ridge_logit(X, a, 1.0)
    w = np.where(a == 1, 1 - e, e)
    m1 = design.wmean(y[a == 1], w[a == 1])
    m0 = design.wmean(y[a == 0], w[a == 0])
    rd = m1 - m0
    v1 = design.wmean((y[a == 1] - m1) ** 2, w[a == 1]) / design.ess(w[a == 1])
    v0 = design.wmean((y[a == 0] - m0) ** 2, w[a == 0]) / design.ess(w[a == 0])
    se = np.sqrt(v1 + v0)
    return {
        "outcome": outcome,
        "n": len(x),
        "DEX_mean": m1,
        "MIDAZ_mean": m0,
        "mean_difference": rd,
        "ci_low": rd - 1.96 * se,
        "ci_high": rd + 1.96 * se,
        "estimator": "overlap weighting",
    }


def evalue_rr(rr):
    r = rr if rr >= 1 else 1 / rr
    return r + math.sqrt(r * (r - 1)) if r >= 1 else 1.0
