import numpy as np
import pandas as pd

import src.external as ext
import src.runtime as runtime_module

ROOT = runtime_module.WORK / "primary"
MROOT = ROOT
T = ROOT / "tables"
D = ROOT / "derived"


def common(d, db):
    z = pd.DataFrame(index=d.index)
    z["age"] = pd.to_numeric(d.age, errors="coerce")
    z["hours_from_icu_admit"] = pd.to_numeric(d.hours_from_icu_admit, errors="coerce")
    z["gcs"] = pd.to_numeric(d.gcs_6h, errors="coerce")
    if db == "MIMIC":
        z["opioid"] = pd.to_numeric(d.opioid_active_t0, errors="coerce")
        z["nmba"] = pd.to_numeric(d.nmba_recent_6h, errors="coerce")
        z["vasopressor"] = pd.to_numeric(d.vasopressor_active_t0, errors="coerce")
    else:
        z["opioid"] = pd.to_numeric(d.opioid_6h, errors="coerce")
        z["nmba"] = pd.to_numeric(d.nmba_6h, errors="coerce")
        z["vasopressor"] = pd.to_numeric(d.vasopressor_pre6h, errors="coerce")
    for c in [
        "seizure",
        "withdrawal",
        "neurocritical",
        "respiratory",
        "delirium_proxy",
        "postoperative",
        "deep_sedation",
    ]:
        z[c] = pd.to_numeric(d[c], errors="coerce")
    z["sex"] = d.sex.fillna("Unknown").astype(str)
    z["sedation_stratum"] = d.sedation_stratum.fillna("unknown").astype(str)
    return z


def encode(m, e):
    c = pd.concat([m.assign(_db=0), e.assign(_db=1)], ignore_index=True)
    out = pd.DataFrame(index=c.index)
    for col in ["age", "hours_from_icu_admit", "gcs"]:
        x = c[col]
        miss = x.isna().astype(float)
        med = x.median()
        x = x.fillna(med)
        sd = x.std(ddof=0)
        out[col] = (x - x.mean()) / (sd if sd > 1e-09 else 1)
        out[col + "__missing"] = miss
    for col in [
        "opioid",
        "nmba",
        "vasopressor",
        "seizure",
        "withdrawal",
        "neurocritical",
        "respiratory",
        "delirium_proxy",
        "postoperative",
        "deep_sedation",
    ]:
        out[col] = c[col].fillna(0).astype(float)
    out = pd.concat(
        [
            out,
            pd.get_dummies(
                c[["sex", "sedation_stratum"]], drop_first=True, dtype=float
            ),
        ],
        axis=1,
    )
    return out.loc[:, out.nunique() > 1], c._db.to_numpy()


def weighted_effect(y, a, w):
    m1 = ext.wmean(y[a == 1], w[a == 1])
    m0 = ext.wmean(y[a == 0], w[a == 0])
    rd = m1 - m0
    v1 = ext.wmean((y[a == 1] - m1) ** 2, w[a == 1]) / ext.ess(w[a == 1])
    v0 = ext.wmean((y[a == 0] - m0) ** 2, w[a == 0]) / ext.ess(w[a == 0])
    se = np.sqrt(v1 + v0)
    rr = m1 / m0
    selog = np.sqrt(v1 / (m1 * m1) + v0 / (m0 * m0))
    return {
        "DEX_risk": m1,
        "MIDAZ_risk": m0,
        "risk_difference": rd,
        "rd_ci_low": rd - 1.96 * se,
        "rd_ci_high": rd + 1.96 * se,
        "risk_ratio": rr,
        "rr_ci_low": np.exp(np.log(rr) - 1.96 * selog),
        "rr_ci_high": np.exp(np.log(rr) + 1.96 * selog),
        "combined_ess": ext.ess(w),
        "DEX_ess": ext.ess(w[a == 1]),
        "MIDAZ_ess": ext.ess(w[a == 0]),
    }
