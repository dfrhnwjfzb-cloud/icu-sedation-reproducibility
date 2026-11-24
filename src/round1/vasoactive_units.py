import json
import re
import urllib.request

import pandas as pd

import src.round1.setup as p

R = p.R


def main():
    src = json.loads((R / "sources/manifest.json").read_text())
    commit = src[0]["commit"]
    drugs = ["dopamine", "dobutamine", "epinephrine", "norepinephrine"]
    ids = {}
    for drug in drugs:
        path = R / "sources" / f"medication__{drug}.sql"
        url = f"https://raw.githubusercontent.com/MIT-LCP/mimic-code/{commit}/mimic-iv/concepts/medication/{drug}.sql"
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(url, timeout=30).read())
        ids[drug] = int(re.search("WHERE itemid\\s*=\\s*(\\d+)", path.read_text())[1])
        if not any(s["file"] == path.name for s in src):
            src.append(
                {"url": url, "commit": commit, "file": path.name, "sha256": p.sha(path)}
            )
    (R / "sources/manifest.json").write_text(
        json.dumps(src, indent=2), encoding="utf-8"
    )
    x = pd.read_csv(R / "data/model0_cohort.csv")
    keys = ",".join(f"({r.stay_id},'{r.time_zero}'::timestamp)" for r in x.itertuples())
    parts = []
    for drug, item in ids.items():
        rate = (
            "CASE WHEN i.rateuom='mg/kg/min' AND i.patientweight=1 THEN i.rate WHEN i.rateuom='mg/kg/min' THEN i.rate*1000.0 ELSE i.rate END"
            if drug == "norepinephrine"
            else "i.rate"
        )
        parts.append(
            f"SELECT '{drug}' drug,i.rateuom,count(*) matched_records,count(*) FILTER(WHERE abs(v.vaso_rate-({rate}))>1e-6) rate_differences FROM keys k JOIN mimiciv_icu.icustays s USING(stay_id) JOIN mimiciv_derived.{drug} v ON v.stay_id=k.stay_id AND v.starttime<k.t0 AND v.endtime>greatest(s.intime,k.t0-interval '24 hour') JOIN mimiciv_icu.inputevents i ON i.stay_id=v.stay_id AND i.itemid={item} AND i.linkorderid=v.linkorderid AND i.starttime=v.starttime AND i.endtime=v.endtime GROUP BY i.rateuom"
        )
    result = p.query(
        "WITH keys(stay_id,t0) AS (VALUES "
        + keys
        + ") "
        + " UNION ALL ".join(parts)
        + ";",
        "audit_vaso_units.sql",
    )
    result.to_csv(R / "outputs/vaso_unit_audit.csv", index=False)
    print(result.to_string(index=False), flush=True)
