"""Shared cohorts, source windows and database queries for severity analyses."""

import pandas as pd

import src.runtime as rt

R = rt.R2
OLD = rt.R1
for name in [
    "analysis",
    "data",
    "outputs",
    "sql",
    "logs",
    "tmp",
    "bootstrap",
    "tests",
    "sources",
]:
    (R / name).mkdir(exist_ok=True)
ORGS = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]
OUTCOMES = ["alive_sustainably_liberated_day28", "hospital_mortality"]
sha = rt.sha


def query(sql, name):
    return rt.query(sql, name, R)


def baseline():
    x = rt.cohort(OLD)
    s = pd.read_csv(OLD / "data/pre_t0_sofa.csv", dtype={"stay_id": str})
    return x.merge(s, on="stay_id", validate="one_to_one")


def keys_cte():
    x = baseline()
    v = ",".join(
        f"({r.stay_id},{r.admission_id},{r.patient_id},'{r.time_zero}'::timestamp)"
        for r in x.itertuples()
    )
    return f"WITH keys(stay_id,hadm_id,subject_id,t0) AS (VALUES {v}), c AS MATERIALIZED (SELECT k.*,i.intime,i.outtime,greatest(i.intime,k.t0-interval '24 hour') ws FROM keys k JOIN mimiciv_icu.icustays i USING(stay_id)) "
