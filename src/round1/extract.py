import json

import pandas as pd

import src.round1.setup as p

R = p.R


def main():
    m = pd.read_csv(R / "data/model0_cohort.csv")
    vals = ",".join(
        f"({r.stay_id},{r.admission_id},{r.patient_id},'{r.time_zero}'::timestamp)"
        for r in m.itertuples()
    )
    head = f"WITH keys(stay_id,hadm_id,subject_id,t0) AS (VALUES {vals}), c AS MATERIALIZED (SELECT k.*,i.intime,i.outtime,GREATEST(i.intime,k.t0-interval '24 hour') ws FROM keys k JOIN mimiciv_icu.icustays i USING(stay_id)) "
    meta = p.query(
        head
        + "SELECT stay_id,intime,outtime,t0,ws,extract(epoch FROM(t0-ws))/3600 window_hours FROM c ORDER BY stay_id;",
        "cohort_window.sql",
    )
    meta.to_csv(R / "data/windows.csv", index=False)
    unions = []

    def obs(
        table, key, varvalue, where="true", extras="'{}'::jsonb", time="x.charttime"
    ):
        unions.append(
            f"SELECT c.stay_id,v.variable,v.value::float8,{time} charttime,NULL::timestamp endtime,'{table}' source,{extras} metadata FROM c JOIN mimiciv_derived.{table} x ON c.{key}=x.{key} AND {time}>=c.ws AND {time}<c.t0 CROSS JOIN LATERAL (VALUES {varvalue}) v(variable,value) WHERE ({where}) AND v.value IS NOT NULL"
        )

    obs(
        "vitalsign",
        "stay_id",
        "('map',mbp),('temperature',temperature),('heart_rate',heart_rate),('resp_rate',resp_rate)",
    )
    obs(
        "chemistry",
        "hadm_id",
        "('creatinine',creatinine),('sodium',sodium),('potassium',potassium),('bicarbonate',bicarbonate)",
        "x.subject_id=c.subject_id",
    )
    obs(
        "enzyme",
        "hadm_id",
        "('bilirubin',bilirubin_total)",
        "x.subject_id=c.subject_id",
    )
    obs(
        "complete_blood_count",
        "hadm_id",
        "('platelet',platelet),('hematocrit',hematocrit),('wbc',wbc)",
        "x.subject_id=c.subject_id",
    )
    obs(
        "gcs",
        "stay_id",
        "('gcs',gcs)",
        extras="jsonb_build_object('motor',gcs_motor,'eyes',gcs_eyes,'verbal',gcs_verbal,'unable',gcs_unable)",
    )
    vent = "EXISTS(SELECT 1 FROM mimiciv_derived.ventilation vd WHERE vd.stay_id=c.stay_id AND vd.starttime<=x.charttime AND vd.endtime>=x.charttime AND vd.ventilation_status='InvasiveVent')"
    obs(
        "bg",
        "hadm_id",
        "('pf',pao2fio2ratio)",
        "x.specimen='ART.' AND x.subject_id=c.subject_id",
        f"jsonb_build_object('vent',{vent},'po2',po2,'pco2',pco2,'fio2',fio2,'fio2_chart',fio2_chartevents,'aado2',aado2_calc,'ph',ph)",
    )
    obs(
        "urine_output_rate",
        "stay_id",
        "('urine24',urineoutput_24hr)",
        extras="jsonb_build_object('hours',uo_tm_24hr,'period_start',x.charttime-uo_tm_24hr*interval '1 hour')",
    )
    for table in ["dopamine", "dobutamine", "epinephrine", "norepinephrine"]:
        unions.append(
            f"SELECT c.stay_id,'{table}',x.vaso_rate::float8,GREATEST(x.starttime,c.ws),LEAST(x.endtime,c.t0),'{table}',jsonb_build_object('rate_unit','mcg/kg/min','original_start',x.starttime,'original_end',x.endtime) FROM c JOIN mimiciv_derived.{table} x ON c.stay_id=x.stay_id AND x.starttime<c.t0 AND x.endtime>c.ws WHERE x.vaso_rate>0"
        )
    unions.append(
        "SELECT c.stay_id,CASE x.itemid WHEN 220739 THEN 'gcs_raw_eyes' WHEN 223900 THEN 'gcs_raw_verbal' ELSE 'gcs_raw_motor' END,x.valuenum::float8,x.charttime,NULL::timestamp,'chartevents',jsonb_build_object('storetime',x.storetime,'ett',x.value='No Response-ETT') FROM c JOIN mimiciv_icu.chartevents x ON c.stay_id=x.stay_id AND x.subject_id=c.subject_id AND x.charttime>=c.ws AND x.charttime<c.t0 AND x.itemid IN (220739,223900,223901) WHERE x.valuenum IS NOT NULL"
    )
    sql = (
        head
        + "SELECT * FROM ("
        + " UNION ALL ".join(unions)
        + ") obs ORDER BY stay_id,charttime,variable;"
    )
    d = p.query(sql, "pre_t0_sofa.sql")
    d.to_csv(R / "data/sofa_observations.csv", index=False)
    eq = p.query(
        head
        + "SELECT 'gcs' source,count(*) at_t0 FROM c JOIN mimiciv_derived.gcs x ON c.stay_id=x.stay_id AND x.charttime=c.t0 UNION ALL SELECT 'vitalsign',count(*) FROM c JOIN mimiciv_derived.vitalsign x ON c.stay_id=x.stay_id AND x.charttime=c.t0 UNION ALL SELECT 'lab',count(*) FROM c JOIN mimiciv_hosp.labevents x ON c.hadm_id=x.hadm_id AND x.charttime=c.t0;",
        "t0_equality_audit.sql",
    )
    eq.to_csv(R / "outputs/t0_equality_counts.csv", index=False)
    print(
        json.dumps(
            {
                "observation_rows": len(d),
                "stays": d.stay_id.nunique(),
                "sources": d.source.nunique(),
            }
        ),
        flush=True,
    )
