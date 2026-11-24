"""Fixed severity model specifications and fitting diagnostics."""

import numpy as np
from scipy.optimize import minimize

import src.design as b
import src.round2.common as p

COUNTS = []


def ridge(X, y, lam=1.0):
    X1 = np.column_stack([np.ones(len(X)), X])

    def fn(beta):
        pr = b.sigmoid(X1 @ beta)
        return -np.sum(
            y * np.log(pr + 1e-12) + (1 - y) * np.log(1 - pr + 1e-12)
        ) + 0.5 * lam * np.sum(beta[1:] ** 2)

    def gr(beta):
        v = X1.T @ (b.sigmoid(X1 @ beta) - y)
        v[1:] += lam * beta[1:]
        return v

    r = minimize(
        fn,
        np.zeros(X1.shape[1]),
        jac=gr,
        method="L-BFGS-B",
        options={"maxiter": 800, "ftol": 1e-10},
    )
    COUNTS.append(
        {
            "success": bool(r.success),
            "nit": int(r.nit),
            "message": str(r.message),
            "grad_max": float(np.max(abs(r.jac))),
        }
    )
    # Continue the same weak-ridge objective from the current optimizer iterate.
    if not r.success and lam < 1 and (r.nit >= 800):
        r = minimize(
            fn, r.x, jac=gr, method="L-BFGS-B", options={"maxiter": 2400, "ftol": 1e-10}
        )
        COUNTS.append(
            {
                "success": bool(r.success),
                "nit": int(r.nit),
                "message": "Same-objective numerical continuation: " + str(r.message),
                "grad_max": float(np.max(abs(r.jac))),
            }
        )
    if not r.success:
        raise RuntimeError(r.message)
    return r.x, b.sigmoid(X1 @ r.x)


def scaled(z):
    return (z - z.mean()) / max(z.std(ddof=0), 1e-09)


def matrix(x, model):
    X = b.design_matrix(x)
    if model == "Model0":
        return X
    if model in ["ModelS", "G0"]:
        for col, z in [
            ("pre_t0_sofa", x.pre_t0_sofa.astype(float)),
            ("pre_t0_sofa_squared", x.pre_t0_sofa.astype(float) ** 2),
        ]:
            X[col] = scaled(z)
        for o in p.ORGS:
            X[o + "_missing"] = x[o + "_missing"].astype(float)
    elif model in ["G1", "G2"]:
        c = x["cns_" + model].astype(float)
        med = c.median() if c.notna().any() else 0.0
        idx = x.pre_t0_sofa - x.cns.fillna(0) + c.fillna(med)
        X["regression_SOFA_index_" + model] = scaled(idx)
        X["regression_SOFA_index_" + model + "_squared"] = scaled(idx**2)
        for o in p.ORGS:
            X[o + "_missing"] = (
                c.isna().astype(float)
                if o == "cns"
                else x[o + "_missing"].astype(float)
            )
    elif model in ["ModelM", "ModelE", "ModelW"]:
        for o in p.ORGS:
            z = x[o].astype(float)
            med = z.median() if z.notna().any() else 0.0
            X[o + "_regression_imputed"] = scaled(z.fillna(med))
            X[o + "_missing"] = z.isna().astype(float)
        X["severity_window_hours"] = scaled(x.window_hours)
        if model != "ModelM":
            X["pre_t0_sofa"] = scaled(x.pre_t0_sofa)
            X["pre_t0_sofa_squared"] = scaled(x.pre_t0_sofa**2)
            X["severity_window_squared"] = scaled(x.window_hours**2)
            X["age_squared"] = scaled(x.age**2)
            X["brady_x_awake"] = x.bradycardia_proxy.astype(float) * (
                x.sedation_stratum == "light/awake"
            ).astype(float)
    else:
        raise ValueError(model)
    return X.loc[:, X.nunique(dropna=False) > 1]


def subset(x, group):
    if group == "Full":
        return x.copy()
    if group == "Low":
        return x[x.pre_t0_sofa <= 5].copy()
    if group == "Intermediate":
        return x[x.pre_t0_sofa.between(6, 9)].copy()
    if group == "High":
        return x[x.pre_t0_sofa >= 10].copy()
    if group == "NonCardiacCV":
        return x[x.icu_group != "Cardiac/CV"].copy()
    if group == "CardiacCV":
        return x[x.icu_group == "Cardiac/CV"].copy()
    raise ValueError(group)


def fit(x, model, folds=None, diagnostics=False):
    global COUNTS
    COUNTS = []
    Xdf = matrix(x, model)
    X = Xdf.to_numpy()
    a = (x.treatment == "DEX").to_numpy(int)
    n = len(x)
    if min(a.sum(), n - a.sum()) < 2:
        raise ValueError("Missing treatment arm")
    if folds is None:
        folds = np.arange(n) % 5
    ps_beta, e = ridge(X, a, 0.01 if model == "ModelW" else 1.0)
    h = e * (1 - e)
    den = h.mean()
    ow = np.where(a == 1, 1 - e, e)
    rows = []
    influences = {}
    predictions = {}
    for outcome in p.OUTCOMES:
        y = x[outcome].to_numpy(int)
        pred = np.zeros((n, 2))
        for k in range(5):
            tr = folds != k
            te = ~tr
            if not te.any():
                raise ValueError("Empty held-out fold")
            for arm in [0, 1]:
                sel = tr & (a == arm)
                if sel.sum() < 2 or len(np.unique(y[sel])) < 2:
                    raise ValueError("Degenerate training arm/outcome")
                beta, _ = ridge(X[sel], y[sel], 2.0)
                pred[te, arm] = b.sigmoid(
                    np.column_stack([np.ones(te.sum()), X[te]]) @ beta
                )
        z1 = h * (pred[:, 1] + a / e * (y - pred[:, 1]))
        z0 = h * (pred[:, 0] + (1 - a) / (1 - e) * (y - pred[:, 0]))
        mu1 = z1.mean() / den
        mu0 = z0.mean() / den
        psi1 = (z1 - mu1 * h) / den
        psi0 = (z0 - mu0 * h) / den
        rd = mu1 - mu0
        rr = mu1 / mu0 if mu0 != 0 else np.nan
        ird = psi1 - psi0
        ilrr = psi1 / mu1 - psi0 / mu0 if mu1 != 0 and mu0 != 0 else np.full(n, np.nan)
        se = ird.std(ddof=1) / np.sqrt(n)
        selog = ilrr.std(ddof=1) / np.sqrt(n)
        row = {
            "outcome": outcome,
            "n": n,
            "DEX_n": int(a.sum()),
            "MIDAZ_n": int((1 - a).sum()),
            "DEX_risk": mu1,
            "MIDAZ_risk": mu0,
            "RD": rd,
            "RD_IF_low": rd - 1.96 * se,
            "RD_IF_high": rd + 1.96 * se,
            "RR": rr,
            "RR_IF_low": np.exp(np.log(rr) - 1.96 * selog) if rr > 0 else np.nan,
            "RR_IF_high": np.exp(np.log(rr) + 1.96 * selog) if rr > 0 else np.nan,
            "RD_IF_se": se,
            "logRR_IF_se": selog,
            "ESS_DEX": b.ess(ow[a == 1]),
            "ESS_MIDAZ": b.ess(ow[a == 0]),
            "risk_outside_unit": bool(not 0 <= mu1 <= 1 or not 0 <= mu0 <= 1),
            "model_terms": X.shape[1],
        }
        rows.append(row)
        influences[outcome] = np.column_stack([ird, ilrr])
        predictions[outcome] = pred
    info = {
        "ps": e,
        "ps_beta": ps_beta,
        "ow": ow,
        "X": Xdf,
        "influences": influences,
        "predictions": predictions,
        "convergence": COUNTS.copy(),
    }
    return rows, info
