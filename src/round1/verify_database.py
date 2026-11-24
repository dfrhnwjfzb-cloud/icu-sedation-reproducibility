import json
import logging
import re

import numpy as np
import pandas as pd

import src.round1.setup as p

logger = logging.getLogger(__name__)


R = p.R


def cohort_cte(source):
    s = re.sub("--[^\\n]*", "", source)
    parts = s.split(";")
    ctes = []
    final = None
    for stmt in parts:
        stmt = stmt.strip()
        if not stmt:
            continue
        if stmt.startswith("CREATE INDEX") or stmt.startswith(
            "CREATE TEMP TABLE _drug_map("
        ):
            continue
        if stmt.startswith("INSERT INTO _drug_map VALUES"):
            ctes.append(
                "_drug_map(itemid,drug) AS (VALUES " + stmt.split("VALUES", 1)[1] + ")"
            )
            continue
        m = re.match("CREATE TEMP TABLE (\\w+) AS\\s+(.*)", stmt, re.S)
        if m:
            ctes.append(m[1] + " AS MATERIALIZED (" + m[2] + ")")
            continue
        if stmt.startswith("SELECT"):
            final = stmt
        else:
            raise ValueError("Unexpected SQL statement category")
    assert final and len(ctes) > 5
    return (
        "WITH "
        + ",\n".join(ctes)
        + ", final_cohort AS ("
        + final
        + ") SELECT * FROM final_cohort ORDER BY stay_id;"
    )


def main():
    src = R / "sources/frozen_design_cohort.sql"
    assert src.exists(), "Bundled frozen cohort SQL missing"
    sql = cohort_cte(src.read_text(encoding="utf-8-sig"))
    raw = p.query(sql, "rebuild_original_cohort_readonly.sql")
    raw.to_csv(R / "data/rebuilt_candidates.csv", index=False)
    x = raw[
        raw.treatment.isin(["DEX", "MIDAZ"])
        & (raw.prop_active_t0 == 1)
        & (raw.invasive_mv_t0 == 1)
    ].copy()
    frozen = pd.read_csv(R / "data/model0_cohort.csv")
    x = x.sort_values("stay_id")
    frozen = frozen.sort_values("stay_id")
    assert len(x) == 3669 and x.stay_id.tolist() == frozen.stay_id.tolist()
    checks = []
    for col in x:
        if col not in frozen:
            continue
        aa = x[col].reset_index(drop=True)
        bb = frozen[col].reset_index(drop=True)
        try:
            equal = np.isclose(
                pd.to_numeric(aa),
                pd.to_numeric(bb),
                equal_nan=True,
                atol=1e-06,
                rtol=1e-08,
            )
        except (ValueError, TypeError):
            equal = aa.fillna("<NA>").astype(str) == bb.fillna("<NA>").astype(str)
        checks.append({"variable": col, "mismatches": int((~equal).sum())})
    pd.DataFrame(checks).to_csv(
        R / "outputs/database_cohort_comparison.csv", index=False
    )
    disagreements = {q["variable"]: q["mismatches"] for q in checks if q["mismatches"]}
    assert not set(disagreements) - {"creatinine", "bun", "bicarbonate"}, (
        "Unexpected cohort differences"
    )
    vals = ",".join(
        f"({r.stay_id},{r.admission_id},{r.patient_id},'{r.time_zero}'::timestamp)"
        for r in frozen.itertuples()
    )
    sql = f"WITH c(stay_id,hadm_id,subject_id,t0) AS (VALUES {vals}), l AS (\n SELECT c.stay_id, MIN(v.endtime) FILTER(WHERE v.ventilation_status='InvasiveVent' AND v.endtime>c.t0 AND v.endtime<=c.t0+interval '28 day' AND NOT EXISTS(SELECT 1 FROM mimiciv_derived.ventilation v2 WHERE v2.stay_id=c.stay_id AND v2.ventilation_status='InvasiveVent' AND v2.starttime>=v.endtime AND v2.starttime<v.endtime+interval '48 hour')) liberation FROM c LEFT JOIN mimiciv_derived.ventilation v ON v.stay_id=c.stay_id GROUP BY c.stay_id)\n SELECT c.stay_id,CASE WHEN l.liberation IS NOT NULL AND (p.dod IS NULL OR p.dod>(c.t0+interval '28 day')::date) THEN 1 ELSE 0 END alive_sustainably_liberated_day28,a.hospital_expire_flag hospital_mortality FROM c JOIN l USING(stay_id) JOIN mimiciv_hosp.patients p ON c.subject_id=p.subject_id JOIN mimiciv_hosp.admissions a ON a.hadm_id=c.hadm_id ORDER BY c.stay_id;"
    outs = p.query(sql, "rebuild_original_endpoints.sql")
    out = frozen.merge(outs, on="stay_id", suffixes=("", "_db"), validate="one_to_one")
    qc = {
        "rebuilt_candidate_n": len(raw),
        "primary_n": len(x),
        "DEX_n": int((x.treatment == "DEX").sum()),
        "MIDAZ_n": int((x.treatment == "MIDAZ").sum()),
        "cohort_field_mismatches": sum(q["mismatches"] for q in checks),
        "differing_fields": disagreements,
        "baseline_used": "Frozen input CSV; raw SQL lab tie issue audited separately; not overwritten",
    }
    for name in ["alive_sustainably_liberated_day28", "hospital_mortality"]:
        qc[name + "_mismatches"] = int((out[name] != out[name + "_db"]).sum())
        assert qc[name + "_mismatches"] == 0
    (R / "outputs/database_reproduction_qc.json").write_text(
        json.dumps(qc, indent=2), encoding="utf-8"
    )
    logger.info("Cohort and endpoint verification complete")
