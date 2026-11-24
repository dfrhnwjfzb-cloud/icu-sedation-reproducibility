import json
import logging
import sys

import numpy as np
import pandas as pd

import src.round1.setup as p

logger = logging.getLogger(__name__)


R = p.R


def main():
    d = pd.read_csv(R / "data/sofa_observations_scored.csv")
    x = pd.read_csv(R / "data/analysis_scores_merged.csv")
    w = pd.read_csv(R / "data/windows.csv")
    d["meta"] = d.metadata.map(json.loads)
    d["charttime"] = pd.to_datetime(d.charttime)
    w["t0"] = pd.to_datetime(w.t0)
    d = d.merge(x[["stay_id", "treatment"]], on="stay_id", validate="many_to_one")
    d["t0"] = pd.to_datetime(d.t0)
    rows = []
    for (v, tx), z in d.groupby(["variable", "treatment"]):
        rr = {
            "variable": v,
            "treatment": tx,
            "rows": len(z),
            "stays": z.stay_id.nunique(),
            "min": z.value.min(),
            "p01": z.value.quantile(0.01),
            "median": z.value.median(),
            "p99": z.value.quantile(0.99),
            "max": z.value.max(),
            "negative_rows": int((z.value < 0).sum()),
            "negative_or_after_t0": int((z.charttime >= z.t0).sum()),
        }
        if v.startswith("gcs_raw"):
            st = pd.to_datetime(z.meta.map(lambda q: q.get("storetime")))
            rr["storetime_ge_t0_or_missing"] = int((st.isna() | (st >= z.t0)).sum())
            rr["ett_rows"] = int(z.meta.map(lambda q: bool(q.get("ett"))).sum())
        rows.append(rr)
    pd.DataFrame(rows).to_csv(R / "outputs/input_quality_by_treatment.csv", index=False)
    ranges = {
        "map": (20, 180),
        "pf": (30, 800),
        "bilirubin": (0, 50),
        "creatinine": (0, 15),
        "platelet": (1, 1000),
        "dopamine": (0, 30),
        "dobutamine": (0, 30),
        "epinephrine": (0, 3),
        "norepinephrine": (0, 3),
        "temperature": (30, 43),
    }
    flagged = []
    for var, (lo, hi) in ranges.items():
        for tx in ["DEX", "MIDAZ"]:
            z = d[(d.variable == var) & (d.treatment == tx)]
            bad = z[(z.value < lo) | (z.value > hi)]
            flagged.append(
                {
                    "variable": var,
                    "treatment": tx,
                    "review_lower": lo,
                    "review_upper": hi,
                    "flagged_rows": len(bad),
                    "flagged_stays": bad.stay_id.nunique(),
                    "action": "Retained; descriptive review flag, not an exclusion or validated error",
                }
            )
    pd.DataFrame(flagged).to_csv(R / "outputs/extreme_value_review.csv", index=False)
    gc = []
    for tx, z in x.groupby("treatment"):
        gc.append(
            {
                "treatment": tx,
                "n": len(z),
                "suspect_GCS15_stays": int((z.gcs_default15_suspect_n > 0).sum()),
                "raw_complete_GCS_stays": int((z.gcs_raw_complete_n > 0).sum()),
                "all_6_observed_stays": int(z.pre_t0_sofa_strict.notna().sum()),
                "short_window_stays": int(z.short_window.sum()),
                "window_hours_median": z.window_hours.median(),
                "window_hours_min": z.window_hours.min(),
                "SOFA_median": z.pre_t0_sofa.median(),
                "SOFA_p25": z.pre_t0_sofa.quantile(0.25),
                "SOFA_p75": z.pre_t0_sofa.quantile(0.75),
                "missing_organs_mean": z.missing_organs.mean(),
            }
        )
    pd.DataFrame(gc).to_csv(
        R / "outputs/score_completeness_by_treatment.csv", index=False
    )
    q = {
        "original_patient_n": int(x.patient_id.nunique()),
        "repeated_patient_stays": int(x.patient_id.duplicated().sum()),
        "cohort_stay_duplicates": int(x.stay_id.duplicated().sum()),
        "t0_before_intime": int(
            (pd.to_datetime(w.t0) < pd.to_datetime(w.intime)).sum()
        ),
        "t0_after_outtime": int(
            (pd.to_datetime(w.t0) > pd.to_datetime(w.outtime)).sum()
        ),
        "strict_full_urine_observation_n": int(x.urine24_usable_n.sum()),
        "sofa_score_outside_0_24": int((~x.pre_t0_sofa.between(0, 24)).sum()),
        "gcs_component_time_conflicts": 0,
    }
    raw = d[d.variable.str.startswith("gcs_raw")]
    q["gcs_component_time_conflicts"] = int(
        (raw.groupby(["stay_id", "charttime", "variable"]).value.nunique() > 1).sum()
    )
    (R / "outputs/additional_qc.json").write_text(
        json.dumps(q, indent=2), encoding="utf-8"
    )
    rebuilt = pd.read_csv(R / "data/rebuilt_candidates.csv", dtype={"stay_id": str})
    f = p.load_cohort()
    rr = rebuilt.merge(f, on="stay_id", suffixes=("_db", "_f"))
    differences = []
    for c in ["creatinine", "bun", "bicarbonate"]:
        bad = rr[~np.isclose(rr[c + "_db"], rr[c + "_f"], equal_nan=True)]
        for row in bad.itertuples():
            differences.append((str(row.stay_id), c, getattr(row, c + "_f")))
    vals = ",".join((f"({sid},'{c}',{v})" for sid, c, v in differences))
    if vals:
        keys = ",".join(
            f"({r.stay_id},{r.admission_id},'{r.time_zero}'::timestamp)"
            for r in f.itertuples()
        )
        sql = f"WITH keys(stay_id,hadm_id,t0) AS (VALUES {keys}), bad(stay_id,variable,frozen_value) AS (VALUES {vals}), records AS (SELECT k.stay_id,b.variable,b.frozen_value,v.value,x.charttime,MAX(x.charttime) OVER(PARTITION BY k.stay_id,b.variable) latest FROM keys k JOIN bad b USING(stay_id) JOIN mimiciv_derived.chemistry x ON x.hadm_id=k.hadm_id AND x.charttime<=k.t0 AND x.charttime>=k.t0-interval '24 hour' CROSS JOIN LATERAL(VALUES ('creatinine',x.creatinine),('bun',x.bun),('bicarbonate',x.bicarbonate))v(variable,value) WHERE b.variable=v.variable AND v.value IS NOT NULL) SELECT stay_id,variable,COUNT(*) latest_rows,COUNT(DISTINCT value) distinct_latest_values,BOOL_OR(abs(value-frozen_value)<1e-8) frozen_value_exists_at_latest_time FROM records WHERE charttime=latest GROUP BY stay_id,variable ORDER BY stay_id,variable;"
        cache = R / "outputs/historical_lab_tie_audit.csv"
        ties = (
            pd.read_csv(cache)
            if cache.exists() and "--refresh-db" not in sys.argv
            else p.query(sql, "audit_historical_lab_ties.sql")
        )
        ties.to_csv(cache, index=False)
        logger.info("Historical laboratory ties checked")
    logger.info("Severity measurement diagnostics written")
