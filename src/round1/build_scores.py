import json
import logging

import numpy as np
import pandas as pd

import src.round1.scores as scores_module
import src.round1.setup as p
import src.round1.temporality as temporality_module

logger = logging.getLogger(__name__)


score = scores_module.score
usable_urine = temporality_module.usable_urine
R = p.R


def main():
    d = pd.read_csv(R / "data/sofa_observations.csv")
    w = pd.read_csv(R / "data/windows.csv")
    for c in ["charttime", "endtime"]:
        d[c] = pd.to_datetime(d[c])
    for c in ["t0", "ws", "intime", "outtime"]:
        w[c] = pd.to_datetime(w[c])
    d = d.merge(w[["stay_id", "t0", "ws"]], on="stay_id", validate="many_to_one")
    assert (d.charttime < d.t0).all() and (d.charttime >= d.ws).all()
    assert (d.endtime.dropna() <= d.loc[d.endtime.notna(), "t0"]).all()
    duplicates = d.duplicated(
        ["stay_id", "variable", "value", "charttime", "endtime", "metadata"]
    ).sum()
    d = d.drop_duplicates(
        ["stay_id", "variable", "value", "charttime", "endtime", "metadata"]
    )
    d["meta"] = d.metadata.map(json.loads)
    d["subscore"] = np.nan
    for i, r in d.iterrows():
        if r.variable.startswith("gcs_raw") or r.variable in [
            "temperature",
            "heart_rate",
            "resp_rate",
            "sodium",
            "potassium",
            "bicarbonate",
            "hematocrit",
            "wbc",
        ]:
            continue
        if r.variable == "urine24":
            h = r.meta.get("hours")
            start = r.meta.get("period_start")
            if start is None or not usable_urine(h, pd.Timestamp(start), r.ws):
                continue
        if (
            r.variable == "gcs"
            and (not 3 <= r.value <= 15)
            or (r.variable != "gcs" and r.value < 0)
        ):
            continue
        d.loc[i, "subscore"] = score(r.variable, r.value, r.meta.get("vent", False))
    records = []
    orgs = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]
    mapping = {
        "respiration": ["pf"],
        "coagulation": ["platelet"],
        "liver": ["bilirubin"],
        "cardiovascular": [
            "map",
            "dopamine",
            "dobutamine",
            "epinephrine",
            "norepinephrine",
        ],
        "cns": ["gcs"],
        "renal": ["creatinine", "urine24"],
    }
    for row in w.itertuples():
        z = d[d.stay_id == row.stay_id]
        r = {
            "stay_id": row.stay_id,
            "window_hours": row.window_hours,
            "short_window": int(row.window_hours < 24),
        }
        for organ, variables in mapping.items():
            sub = z[z.variable.isin(variables) & z.subscore.notna()]
            r[organ] = sub.subscore.max() if len(sub) else np.nan
            r[organ + "_missing"] = int(not len(sub))
            worst = (
                sub.sort_values(
                    ["subscore", "charttime"], ascending=[False, False]
                ).iloc[0]
                if len(sub)
                else None
            )
            for attr in ["value", "variable", "charttime", "source"]:
                r[organ + "_worst_" + attr] = worst[attr] if worst is not None else None
            r[organ + "_observations"] = len(sub)
        g = z[z.variable == "gcs"]
        r["gcs_default15_suspect_n"] = sum(
            rr.value == 15
            and (
                rr.meta.get("unable") == 1
                or any(rr.meta.get(c) is None for c in ["motor", "eyes", "verbal"])
                or sum(rr.meta.get(c) or 0 for c in ["motor", "eyes", "verbal"])
                != rr.value
            )
            for rr in g.itertuples()
        )
        raw = z[z.variable.str.startswith("gcs_raw")].copy()
        rawok = []
        for t, q in raw.groupby("charttime"):
            vals = {
                rr.variable: rr.value
                for rr in q.itertuples()
                if rr.meta.get("storetime") is not None
                and pd.Timestamp(rr.meta["storetime"]) < row.t0
                and (not rr.meta.get("ett"))
            }
            if (
                len(vals) == 3
                and 1 <= vals.get("gcs_raw_eyes", 0) <= 4
                and (1 <= vals.get("gcs_raw_motor", 0) <= 6)
                and (1 <= vals.get("gcs_raw_verbal", 0) <= 5)
            ):
                rawok.append(sum(vals.values()))
        r["gcs_raw_complete_n"] = len(rawok)
        r["cns_raw_observed"] = score("gcs", min(rawok)) if rawok else np.nan
        for v in ["gcs_raw_eyes", "gcs_raw_motor", "gcs_raw_verbal"]:
            r[v + "_n"] = int((raw.variable == v).sum())
        r["urine24_usable_n"] = int(
            ((z.variable == "urine24") & z.subscore.notna()).sum()
        )
        scores = [r[x] for x in orgs]
        r["missing_organs"] = sum(pd.isna(x) for x in scores)
        r["pre_t0_sofa_strict"] = sum(scores) if not r["missing_organs"] else np.nan
        r["pre_t0_sofa"] = sum(0 if pd.isna(x) else x for x in scores)
        r["pre_t0_sofa_raw_gcs"] = (
            r["pre_t0_sofa"]
            - (0 if pd.isna(r["cns"]) else r["cns"])
            + (0 if pd.isna(r["cns_raw_observed"]) else r["cns_raw_observed"])
        )
        records.append(r)
    out = pd.DataFrame(records)
    out.to_csv(R / "data/pre_t0_sofa.csv", index=False)
    d.drop(columns="meta").to_csv(R / "data/sofa_observations_scored.csv", index=False)
    m = pd.read_csv(R / "data/model0_cohort.csv")
    merged = m.merge(out, on="stay_id", validate="one_to_one")
    assert len(merged) == 3669
    merged.to_csv(R / "data/analysis_scores_merged.csv", index=False)
    summary = {
        "rows": len(out),
        "duplicate_observations_removed": int(duplicates),
        "time_leakage_rows": 0,
        "short_window_n": int(out.short_window.sum()),
        "all_six_observed_n": int(out.pre_t0_sofa_strict.notna().sum()),
        "urine24_usable_stays": int((out.urine24_usable_n > 0).sum()),
        "gcs_suspect_stays": int((out.gcs_default15_suspect_n > 0).sum()),
        "raw_gcs_complete_stays": int((out.gcs_raw_complete_n > 0).sum()),
        "sofa_distribution": out.pre_t0_sofa.describe().to_dict(),
    }
    (R / "outputs/score_qc.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("SOFA scoring complete: %d stays", summary["rows"])
