import json
import logging

import pandas as pd

import src.round2.common as p

logger = logging.getLogger(__name__)


R = p.R


def main():
    policy = {
        "fixed_window": "[t0-24h,t0), ICU history at least24h",
        "first_within_hours": 1,
        "last_within_hours": 1,
        "minimum_span_hours": 22,
        "maximum_internal_gap_hours": 2,
        "volume": "Sum actual official urine_output net mL; no normalization/extrapolation",
        "record_availability": "Every constituent raw event storetime<t0",
        "interpretation": "Near-complete documented fixed-window sensitivity, not proof of continuous 24h measurement or standard validated urine SOFA",
    }
    path = R / "urine_policy.json"
    if not path.exists():
        path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    head = p.keys_cte()
    queries = {
        "raw_labs.csv": "SELECT c.stay_id,l.labevent_id,l.specimen_id,l.itemid,l.charttime,l.storetime,l.valuenum,l.valueuom FROM c JOIN mimiciv_hosp.labevents l ON l.hadm_id=c.hadm_id AND l.subject_id=c.subject_id AND l.charttime>=c.ws AND l.charttime<c.t0 AND l.itemid IN(50912,50885,51265,50821,50816) WHERE l.valuenum IS NOT NULL;",
        "raw_charts.csv": "SELECT c.stay_id,x.itemid,x.charttime,x.storetime,x.valuenum,x.value FROM c JOIN mimiciv_icu.chartevents x ON x.stay_id=c.stay_id AND x.subject_id=c.subject_id AND x.charttime>=c.ws AND x.charttime<c.t0 AND x.itemid IN(220739,223900,223901,220052,220181,225312,223835) WHERE x.valuenum IS NOT NULL;",
        "raw_vaso.csv": "SELECT c.stay_id,x.itemid,x.starttime,x.endtime,x.storetime,x.rate,x.rateuom,x.patientweight,x.linkorderid FROM c JOIN mimiciv_icu.inputevents x ON x.stay_id=c.stay_id AND x.subject_id=c.subject_id AND x.starttime<c.t0 AND x.endtime>c.ws AND x.itemid IN(221662,221653,221289,221906) WHERE x.rate>0;",
        "raw_urine.csv": "SELECT c.stay_id,x.itemid,x.charttime,x.storetime,x.value,x.valueuom FROM c JOIN mimiciv_icu.outputevents x ON x.stay_id=c.stay_id AND x.subject_id=c.subject_id AND x.charttime>=c.ws AND x.charttime<c.t0 AND x.itemid IN(226559,226560,226561,226584,226563,226564,226565,226567,226557,226558,227488,227489) WHERE x.value IS NOT NULL;",
        "derived_urine.csv": "SELECT c.stay_id,x.charttime,x.urineoutput FROM c JOIN mimiciv_derived.urine_output x ON x.stay_id=c.stay_id AND x.charttime>=c.ws AND x.charttime<c.t0;",
        "urine_rate_history.csv": "SELECT c.stay_id,x.charttime,x.urineoutput_6hr,x.urineoutput_12hr,x.urineoutput_24hr,x.uo_tm_6hr,x.uo_tm_12hr,x.uo_tm_24hr FROM c JOIN mimiciv_derived.urine_output_rate x ON x.stay_id=c.stay_id AND x.charttime>=c.ws AND x.charttime<c.t0;",
    }
    for name, sql in queries.items():
        target = R / "data" / name
        if not target.exists():
            d = p.query(head + sql, name.replace(".csv", ".sql"))
            d.to_csv(target, index=False)
        else:
            d = pd.read_csv(target)
        logger.info(f"{name}: {len(d)} rows")
