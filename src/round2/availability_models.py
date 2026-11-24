import datetime
import json

import pandas as pd

import src.round2.common as p
import src.round2.diagnostics as diagnostics_module
import src.round2.engine as en

R = p.R


def main():
    policy = {
        "created_before_this_effect_estimation": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "trigger": "A1 demonstrated charttime<t0 but storetime>=t0 in selected inputs",
        "rule": "One fixed conservative rule: new component evidence only when every matched contributing source storetime<t0; partial/unmatched chains excluded; CNS from G1; renal from verified creatinine only",
        "model": "Same component-median + missingness + window model as ModelM; original 84 covariates unchanged",
        "scope": "Sensitivity of newly added severity evidence; cannot remove availability uncertainty in frozen original covariates",
        "limitations": "Informative availability/missingness; not a true physiologic gold standard; IF CI only",
    }
    (R / "availability_policy.json").write_text(
        json.dumps(policy, indent=2), encoding="utf-8"
    )
    x = p.baseline()
    g = pd.read_csv(R / "data/round2_scoring_audit.csv", dtype={"stay_id": str})[
        ["stay_id", "cns_G1", "cns_G2"]
    ]
    x = x.merge(g, on="stay_id", validate="one_to_one")
    s = pd.read_csv(p.OLD / "data/sofa_observations_scored.csv")
    av = pd.read_csv(R / "data/component_availability_records.csv")
    keys = ["stay_id", "variable", "charttime", "value"]
    for q in [s, av]:
        q["charttime"] = pd.to_datetime(q.charttime)
    av["known"] = av.availability == "All_matched_sources_pre_t0"
    known = av.groupby(keys, dropna=False).known.all().reset_index()
    s = s.merge(known, on=keys, how="left", validate="many_to_one")
    s["included"] = s.known.fillna(False) & s.subscore.notna()
    mp = {
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
        "renal": ["creatinine"],
    }
    for o, vars in mp.items():
        vals = s[s.included & s.variable.isin(vars)].groupby("stay_id").subscore.max()
        x[o] = x.stay_id.astype(int).map(vals)
        x[o + "_missing"] = x[o].isna().astype(float)
    x["cns"] = x.cns_G1
    x["cns_missing"] = x.cns.isna().astype(float)
    x[["stay_id", "treatment"] + p.ORGS + [o + "_missing" for o in p.ORGS]].to_csv(
        R / "data/availability_components.csv", index=False
    )
    rows, info = en.fit(x, "ModelM")
    diag = diagnostics_module
    bal, dg, tails = diag.diagnostics(x, "Availability_components", "Full", info)
    for rr in rows:
        rr.update(dg)
    pd.DataFrame(rows).to_csv(
        R / "outputs/availability_effect_sensitivity.csv", index=False
    )
    coverage = []
    for tx, z in x.groupby("treatment"):
        for o in p.ORGS:
            coverage.append(
                {
                    "treatment": tx,
                    "organ": o,
                    "n": len(z),
                    "known_component_n": int(z[o].notna().sum()),
                    "missing_pct": 100 * z[o].isna().mean(),
                }
            )
    pd.DataFrame(coverage).to_csv(R / "outputs/availability_coverage.csv", index=False)
    print(
        pd.DataFrame(rows)[
            ["outcome", "RD", "RD_IF_low", "RD_IF_high", "RR"]
        ].to_string(index=False),
        flush=True,
    )
