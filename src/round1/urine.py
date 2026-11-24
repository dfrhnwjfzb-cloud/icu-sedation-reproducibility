import json
import logging

import numpy as np
import pandas as pd

import src.round1.models as models_module
import src.round1.scores as scores_module
import src.round1.setup as p

logger = logging.getLogger(__name__)


score = scores_module.score
fit = models_module.fit
R = p.R


def main():
    config = {
        "definition": "Official urine normalization for documented 22-30 h coverage, only if entire coverage interval lies within strict pre-t0 window; NOT a fully measured 24-h volume",
        "hours_min": 22,
        "hours_max": 30,
        "normalization": "urineoutput_24hr / uo_tm_24hr * 24",
        "use": "measurement sensitivity only; primary conservative renal score unchanged",
    }
    (R / "urine_sensitivity_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    d = pd.read_csv(R / "data/sofa_observations.csv")
    d = d[d.variable == "urine24"].copy()
    w = pd.read_csv(R / "data/windows.csv")
    d = d.merge(w[["stay_id", "ws", "t0"]], on="stay_id", validate="many_to_one")
    d["meta"] = d.metadata.map(json.loads)
    scored = []
    for r in d.itertuples():
        h = r.meta.get("hours")
        start = r.meta.get("period_start")
        if (
            h is not None
            and 22 <= h <= 30
            and (start is not None)
            and (pd.Timestamp(start) >= pd.Timestamp(r.ws))
            and (pd.Timestamp(r.charttime) < pd.Timestamp(r.t0))
        ):
            v = r.value / h * 24
            scored.append(
                {
                    "stay_id": r.stay_id,
                    "normalized_volume": v,
                    "hours": h,
                    "charttime": r.charttime,
                    "period_start": start,
                    "urine_score": score("urine24", v),
                }
            )
    if not scored:
        (R / "outputs/urine_sensitivity_qc.json").write_text(
            json.dumps(
                {"eligible_stays": 0, "status": "Not estimable; no qualified coverage"}
            ),
            encoding="utf-8",
        )
        return
    z = pd.DataFrame(scored)
    z.to_csv(R / "data/urine22_qualifying_records.csv", index=False)
    x = p.load_cohort()
    sofa = pd.read_csv(R / "data/pre_t0_sofa.csv", dtype={"stay_id": str})
    x = x.merge(sofa, on="stay_id", validate="one_to_one")
    worst = z.groupby("stay_id").urine_score.max()
    x["uo22_score"] = x.stay_id.astype(int).map(worst)
    renal = x.renal.fillna(0)
    updated = np.maximum(renal, x.uo22_score.fillna(0))
    x["pre_t0_sofa_urine22"] = x.pre_t0_sofa - renal + updated
    x.loc[x.uo22_score.notna(), "renal_missing"] = 0
    qc = {
        "qualifying_rows": len(z),
        "eligible_stays": int(x.uo22_score.notna().sum()),
        "changed_total_stays": int((x.pre_t0_sofa_urine22 != x.pre_t0_sofa).sum()),
        "status": "Executed as explicitly adapted urine sensitivity, not primary SOFA",
    }
    (R / "outputs/urine_sensitivity_qc.json").write_text(
        json.dumps(qc, indent=2), encoding="utf-8"
    )
    rows = fit(x, "ModelU_urine22", "Full", "pre_t0_sofa_urine22")
    pd.DataFrame(rows).to_csv(R / "outputs/urine_sensitivity.csv", index=False)
    logger.info("Urine sensitivity complete: %d eligible stays", qc["eligible_stays"])
