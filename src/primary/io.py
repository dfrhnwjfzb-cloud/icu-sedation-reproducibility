"""Study input hashes, descriptive statistics and baseline diagnostics."""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import src.runtime as runtime_module

logger = logging.getLogger(__name__)


ROOT = runtime_module.WORK / "primary"
sys.dont_write_bytecode = True
for env_name in ["TMP", "TEMP", "TMPDIR", "XDG_CACHE_HOME", "MPLCONFIGDIR"]:
    os.environ[env_name] = str(ROOT / "tmp")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for env_name in [
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[env_name] = "1"
(ROOT / "tmp").mkdir(exist_ok=True)


ARCHIVE = ROOT.parent
REMOVED = [
    "seizure",
    "withdrawal",
    "neurocritical",
    "delirium_proxy",
    "respiratory",
    "postoperative",
    "indication_group",
]
PRIMARY = "alive_sustainably_liberated_day28"
BINARY = [
    PRIMARY,
    "hospital_mortality",
    "mortality_day28",
    "icu_mortality",
    "bradycardia_24h",
    "hypotension_24h",
    "vasopressor_start_6h",
    "reintubation_7d",
]
CONTINUOUS = [
    "days_to_sustainable_liberation",
    "icu_los_after_t0",
    "hospital_los_after_t0",
]


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    (ROOT / path).write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else str(x),
        ),
        encoding="utf-8",
    )


def snapshot():
    return {
        str(p.relative_to(ROOT)): sha(p)
        for folder in ["inputs", "expected", "manuscript", "reference_archive"]
        for p in (ROOT / folder).rglob("*")
        if p.is_file()
    }


def progress(text):
    logger.info("%s", text)
    with (ROOT / "logs" / "progress.log").open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def diagnostics(design, data, name):
    a = (data.treatment == "DEX").astype(int).to_numpy()
    X = design.design_matrix(data)
    _, e = design.ridge_logit(X.to_numpy(), a, lam=1.0)
    w = np.where(a == 1, 1 - e, e)
    balance = []
    for col in X:
        balance.append(
            {
                "variable": col,
                "raw_smd": design.smd(X[col].to_numpy(), a),
                "weighted_smd": design.smd(X[col].to_numpy(), a, w),
            }
        )
    b = pd.DataFrame(balance)
    b.to_csv(
        ROOT / "results" / f"{name}_balance.csv", index=False, encoding="utf-8-sig"
    )
    lower = max(np.quantile(e[a == 1], 0.01), np.quantile(e[a == 0], 0.01))
    upper = min(np.quantile(e[a == 1], 0.99), np.quantile(e[a == 0], 0.99))
    out = {
        "version": name,
        "n": len(data),
        "DEX_n": int(a.sum()),
        "MIDAZ_n": int((1 - a).sum()),
        "design_columns": len(X.columns),
        "raw_max_abs_smd": float(b.raw_smd.abs().max()),
        "weighted_max_abs_smd": float(b.weighted_smd.abs().max()),
        "weighted_smd_gt_010_n": int((b.weighted_smd.abs() > 0.1).sum()),
        "overlap_ess": float(design.ess(w)),
        "DEX_ess": float(design.ess(w[a == 1])),
        "MIDAZ_ess": float(design.ess(w[a == 0])),
        "ps_below_005_pct": float(100 * (e < 0.05).mean()),
        "ps_above_095_pct": float(100 * (e > 0.95).mean()),
        "ps_common_1_99_lower": float(lower),
        "ps_common_1_99_upper": float(upper),
        "ps_common_1_99_pct": float(100 * ((e >= lower) & (e <= upper)).mean()),
        "ps_quantiles": {
            arm: dict(
                zip(
                    ["min", "p01", "p05", "p50", "p95", "p99", "max"],
                    map(
                        float,
                        np.quantile(e[a == armid], [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]),
                    ),
                )
            )
            for arm, armid in [("DEX", 1), ("MIDAZ", 0)]
        },
        "weight_quantiles": {
            arm: dict(
                zip(
                    ["min", "p01", "p05", "p50", "p95", "p99", "max"],
                    map(
                        float,
                        np.quantile(w[a == armid], [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1]),
                    ),
                )
            )
            for arm, armid in [("DEX", 1), ("MIDAZ", 0)]
        },
    }
    pd.DataFrame(
        {
            "stay_id": data.stay_id,
            "treatment": data.treatment,
            "propensity": e,
            "overlap_weight": w,
            "overlap_tilt": e * (1 - e),
        }
    ).to_csv(
        ROOT / "derived" / f"{name}_weights.csv", index=False, encoding="utf-8-sig"
    )
    dump(Path("results") / f"{name}_diagnostics.json", out)
    return X, out
