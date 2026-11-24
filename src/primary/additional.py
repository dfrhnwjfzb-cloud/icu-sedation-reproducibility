import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import src.design as design_module
import src.effects as effects_module
import src.external as external_module
import src.primary.io as closed
import src.transport_control as transport_control_module
import src.transport_reduced as transport_reduced_module

logger = logging.getLogger(__name__)


sys.dont_write_bytecode = True


R = closed.ROOT
A = R / "reference_archive"
U = R / "results/updated"
E = R / "results/retained_external"


def save(name, df):
    df.to_csv(U / name, index=False, encoding="utf-8-sig")


def log(t):
    logger.info("%s", t)


def main():
    for p in [U, E, R / "validation"]:
        p.mkdir(parents=True, exist_ok=True)
    before = closed.snapshot()
    source_meta = []
    reduced = design_module
    est = effects_module
    est.design = reduced
    d = pd.read_csv(
        R / "inputs" / "mimic_design_candidates.csv", dtype={"stay_id": str}
    )
    d = d[d.treatment.isin(["DEX", "MIDAZ"])].copy()
    d = reduced.add_derived(d.drop(columns=closed.REMOVED[:6]))
    o = pd.read_csv(R / "inputs" / "mimic_outcomes.csv", dtype={"stay_id": str})
    allx = d.merge(o, on="stay_id", how="inner", suffixes=("", "_out"))
    primary = allx[(allx.prop_active_t0 == 1) & (allx.invasive_mv_t0 == 1)].copy()
    assert len(primary) == 3669 and (
        not set(closed.REMOVED).intersection(primary.columns)
    )
    first = pd.read_csv(R / "results" / "removed_proxy_effects.csv")
    pr = first.iloc[0].dropna().to_dict()
    risk = {
        "A_all_active_propofol": d[d.prop_active_t0 == 1].copy(),
        "B_invasive_mv": d[(d.prop_active_t0 == 1) & (d.invasive_mv_t0 == 1)].copy(),
        "C_controlled_deep": d[
            (d.prop_active_t0 == 1)
            & (d.invasive_mv_t0 == 1)
            & (d.controlled_deep_context == 1)
        ].copy(),
    }
    risk_rows = []
    balance = []
    ps = []
    for name, x in risk.items():
        a, b, p = reduced.summarize_risk(name, x)
        risk_rows.append(a)
        balance.append(b)
        ps.append(p)
    save("risk_set_comparison.csv", pd.DataFrame(risk_rows))
    save("all_risk_set_balance.csv", pd.concat(balance))
    save("all_risk_set_propensities.csv", pd.concat(ps))
    flow = []
    for i, (name, z) in enumerate(
        [
            ("adult first actual initiation", d),
            ("propofol active/recent within 6h", d[d.prop_n_6h > 0]),
            ("propofol active at t0", risk["A_all_active_propofol"]),
            ("invasive MV at t0; primary", risk["B_invasive_mv"]),
            ("controlled/deep context; separate risk set", risk["C_controlled_deep"]),
        ],
        1,
    ):
        flow.append(
            {
                "step_order": i,
                "step": name,
                "DEX_n": int((z.treatment == "DEX").sum()),
                "MIDAZ_n": int((z.treatment == "MIDAZ").sum()),
                "n": len(z),
            }
        )
    save("cohort_flow.csv", pd.DataFrame(flow))
    for var, name in [
        ("sedation_stratum", "sedation_distribution.csv"),
        ("anchor_year_group", "calendar_era.csv"),
        ("icu_group", "icu_group.csv"),
    ]:
        rows = []
        for rn, z in risk.items():
            q = (
                z.groupby(["treatment", var], dropna=False)
                .size()
                .rename("n")
                .reset_index()
            )
            q["risk_set"] = rn
            q["percent"] = 100 * q.n / q.groupby("treatment").n.transform("sum")
            rows.append(q)
        save(name, pd.concat(rows))
    for filename, cols in [
        (
            "nmba_context.csv",
            ["nmba_active_t0", "nmba_recent_2h", "nmba_recent_4h", "nmba_recent_6h"],
        ),
        (
            "opioid_context.csv",
            [
                "opioid_active_t0",
                "fentanyl_active_t0",
                "hydromorphone_active_t0",
                "morphine_active_t0",
                "opioid_types_6h",
                "opioid_starts_6h",
                "fentanyl_rate_mcg_h_t0",
            ],
        ),
        (
            "propofol_context.csv",
            [
                "prop_rate_t0",
                "prop_rate_mean_0_2h",
                "prop_rate_mean_2_6h",
                "prop_overlap_hours_24h",
                "hours_since_first_prop_start",
            ],
        ),
    ]:
        rows = []
        for rn, z in risk.items():
            for tx, g in z.groupby("treatment"):
                row = {"risk_set": rn, "treatment": tx, "n": len(g)}
                for c in cols:
                    row[c + "_mean_or_proportion"] = g[c].mean()
                    row[c + "_observed_n"] = int(g[c].notna().sum())
                rows.append(row)
        save(filename, pd.DataFrame(rows))
    vp = []
    for rn, z in risk.items():
        phase = np.select(
            [
                z.invasive_mv_t0 == 0,
                z.spontaneous_mode == 1,
                (z.controlled_mode == 1) & (z.spontaneous_mode == 0),
            ],
            [
                "not invasively ventilated",
                "spontaneous/weaning proxy",
                "controlled ventilation proxy",
            ],
            default="invasive/phase uncertain",
        )
        q = (
            z.assign(ventilation_phase=phase)
            .groupby(["treatment", "ventilation_phase"])
            .size()
            .rename("n")
            .reset_index()
        )
        q["risk_set"] = rn
        q["percent"] = 100 * q.n / q.groupby("treatment").n.transform("sum")
        vp.append(q)
    save("ventilation_phase.csv", pd.concat(vp))
    log("Design diagnostics and context outputs updated.")
    common_icus = [
        g
        for g, z in primary.groupby("icu_group")
        if len(z) >= 50 and 0.1 <= (z.treatment == "DEX").mean() <= 0.9
    ]
    specs = [
        ("S1", "Active propofol at t0", primary, {}),
        (
            "S2",
            "Recent propofol within 4 h",
            allx[(allx.prop_n_4h > 0) & (allx.invasive_mv_t0 == 1)],
            {},
        ),
        ("S3", "Invasive MV at t0", primary, {}),
        ("S4", "Exclude recent NMBA", primary[primary.nmba_recent_6h == 0], {}),
        ("S5", "RASS/SAS observed", primary[primary.sedation_stratum != "unknown"], {}),
        ("S6", "Exclude deep sedation", primary[primary.deep_sedation == 0], {}),
        ("S7", "Original duplicate model setting", primary, {}),
        (
            "S11",
            "Calendar era 2011–2022",
            primary[primary.anchor_year_group.astype(str) != "2008 - 2010"],
            {},
        ),
        (
            "S12",
            "Common ICU/service support",
            primary[primary.icu_group.isin(common_icus)],
            {},
        ),
        ("S13", "Propensity trimming 0.05–0.95", primary, {"trim": 0.05}),
        (
            "S14",
            "Overlap weighting without augmentation",
            primary,
            {"unaugmented": True},
        ),
        (
            "S15",
            "Complete-information subset",
            primary.dropna(
                subset=["sedation_score", "prop_rate_t0", "map", "creatinine"]
            ),
            {},
        ),
        ("S16", "Alternative treatment model", primary, {"alternative": True}),
    ]
    sens = []
    for sid, label, z, kw in specs:
        duplicate = not kw and z.stay_id.tolist() == primary.stay_id.tolist()
        if duplicate:
            r = {k: v for k, v in pr.items() if k != "outcome"}
            status = "DUPLICATE_PRIMARY"
        elif (
            len(z) < 200
            or min((z.treatment == "DEX").sum(), (z.treatment == "MIDAZ").sum()) < 50
        ):
            r = {"n": len(z)}
            status = "NOT_ESTIMABLE"
        else:
            r = est.estimate(z, **kw)
            status = "ESTIMATED"
        r.update(
            {
                "sensitivity_id": sid,
                "analysis": label,
                "status": status,
                "independent_sensitivity": not duplicate,
            }
        )
        sens.append(r)
        log("Updated sensitivity " + sid + " " + status)
    for sid, label in [
        ("S8", "Exclude seizure/status"),
        ("S9", "Exclude withdrawal"),
        ("S10", "Exclude neurocritical indication"),
    ]:
        sens.append(
            {
                "sensitivity_id": sid,
                "analysis": label,
                "status": "RETIRED_TEMPORAL_VALIDITY",
                "independent_sensitivity": False,
                "reason": "Requires removed untimed MIMIC diagnosis proxy",
            }
        )
    sens = pd.DataFrame(sens)
    sens["_sort"] = sens.sensitivity_id.str[1:].astype(int)
    sens = sens.sort_values("_sort").drop(columns="_sort")
    save("sensitivity_results.csv", sens)
    het = []
    for var, levels in [("deep_sedation", [0, 1]), ("nmba_recent_6h", [0, 1])]:
        for value in levels:
            z = primary[primary[var] == value]
            if (
                len(z) >= 300
                and min((z.treatment == "DEX").sum(), (z.treatment == "MIDAZ").sum())
                >= 60
            ):
                r = est.estimate(z)
                status = "ESTIMATED"
            else:
                r = {"n": len(z)}
                status = "INSUFFICIENT_SUPPORT"
            r.update(
                {
                    "subgroup_variable": var,
                    "subgroup_level": str(value),
                    "status": status,
                }
            )
            het.append(r)
    obs = primary.dropna(subset=["prop_rate_t0"])
    median = obs.prop_rate_t0.median()
    for level, z in [
        ("low", obs[obs.prop_rate_t0 <= median]),
        ("high", obs[obs.prop_rate_t0 > median]),
    ]:
        if min((z.treatment == "DEX").sum(), (z.treatment == "MIDAZ").sum()) >= 60:
            r = est.estimate(z)
            status = "ESTIMATED"
        else:
            r = {"n": len(z)}
            status = "INSUFFICIENT_SUPPORT"
        r.update(
            {
                "subgroup_variable": "propofol_intensity",
                "subgroup_level": level,
                "status": status,
                "propofol_observed_median": median,
            }
        )
        het.append(r)
    for val in ["respiratory", "neurocritical"]:
        het.append(
            {
                "subgroup_variable": "indication_group",
                "subgroup_level": val,
                "status": "RETIRED_TEMPORAL_VALIDITY",
                "reason": "No longer define subgroup using untimed diagnosis",
            }
        )
    save("heterogeneity_results.csv", pd.DataFrame(het))
    rr = float(pr["risk_ratio"])
    lim = float(pr["rr_ci_low"])
    uc = [
        {
            "analysis": "E-value",
            "observed_rr": rr,
            "ci_limit_closest_to_null": lim,
            "strength_parameter": est.evalue_rr(rr),
        },
        {
            "analysis": "E-value CI",
            "observed_rr": rr,
            "ci_limit_closest_to_null": lim,
            "strength_parameter": est.evalue_rr(lim),
        },
    ]
    for strength in [1.25, 1.5, 2, 2.5, 3, 4]:
        bf = strength**2 / (2 * strength - 1)
        uc.append(
            {
                "analysis": "bias-factor grid",
                "observed_rr": rr,
                "strength_parameter": strength,
                "bias_factor": bf,
                "adjusted_rr": rr / bf,
            }
        )
    save("unmeasured_confounding.csv", pd.DataFrame(uc))
    log("Remaining heterogeneity and quantitative confounding outputs updated.")
    ext = external_module
    tc = transport_control_module
    tn = transport_reduced_module
    m0 = pd.read_csv(
        R / "inputs" / "mimic_primary_design_cohort.csv", dtype={"stay_id": str}
    )
    m0 = m0.merge(
        o[["stay_id", "hospital_mortality"]],
        on="stay_id",
        how="left",
        suffixes=("", "_out"),
    )
    m0["hospital_mortality"] = m0.hospital_mortality_out
    e = ext.derive(
        pd.read_csv(
            R / "inputs" / "eicu_primary_cohort.csv",
            dtype={"stay_id": str, "hospital_id": str},
        )
    )
    e = e[e.probable_mv == 1].copy()

    def transport(module, m):
        cm, ce = (module.common(m, "MIMIC"), module.common(e, "eICU"))
        X, s = module.encode(cm, ce)
        _, p = ext.ridge(X.to_numpy(), s, 2.0)
        pm = p[: len(m)]
        raw = pm / (1 - pm)
        lo, hi = np.quantile(raw, [0.01, 0.99])
        tw = np.clip(raw, lo, hi)
        tw = tw / tw.mean()
        ow = pd.to_numeric(m.overlap_weight).to_numpy()
        combined = tw * ow
        y = pd.to_numeric(m.hospital_mortality).to_numpy()
        a = (m.treatment == "DEX").astype(int).to_numpy()
        ok = np.isfinite(y) & np.isfinite(combined)
        eff = module.weighted_effect(y[ok], a[ok], combined[ok])
        eff.update(
            {
                "n": len(m),
                "eicu_target_n": len(e),
                "transport_ess": ext.ess(tw),
                "membership_columns": len(X.columns),
            }
        )
        weights = pd.DataFrame(
            {
                "stay_id": m.stay_id,
                "sample_membership_probability": pm,
                "transport_odds_raw": raw,
                "transport_weight_truncated": tw,
                "mimic_overlap_weight": ow,
                "combined_weight": combined,
            }
        )
        return eff, weights, list(X.columns)

    old_tr, _, _ = transport(tc, m0)
    frozen_cross = pd.read_csv(A / "tables" / "Table_5.csv")
    old_transport = frozen_cross.iloc[-1]
    tr_checks = {
        k: abs(old_tr[k] - float(old_transport[k]))
        for k in [
            "risk_difference",
            "rd_ci_low",
            "rd_ci_high",
            "risk_ratio",
            "rr_ci_low",
            "rr_ci_high",
        ]
    }
    assert max(tr_checks.values()) <= 1e-06, tr_checks
    m = primary.copy()
    ow = pd.read_csv(R / "derived" / "removed_weights.csv", dtype={"stay_id": str})[
        ["stay_id", "overlap_weight"]
    ]
    m = m.merge(ow, on="stay_id", how="left")
    m["hospital_mortality"] = m.hospital_mortality_out
    new_tr, weights, tr_cols = transport(tn, m)
    assert not set(closed.REMOVED).intersection(tr_cols)
    save("transportability_weights.csv", weights)
    hm = first[first.outcome == "hospital_mortality"].iloc[0]
    transported = [
        {
            "analysis": "MIMIC revised overlap association",
            "status": "UPDATED_11_12",
            **{
                k: hm[k]
                for k in [
                    "risk_difference",
                    "rd_ci_low",
                    "rd_ci_high",
                    "risk_ratio",
                    "rr_ci_low",
                    "rr_ci_high",
                ]
            },
        },
        {
            "analysis": "MIMIC revised association transported to eICU E2",
            "status": "UPDATED_11_12",
            **new_tr,
        },
    ]
    save("transported_effects.csv", pd.DataFrame(transported))
    cross = frozen_cross.copy()
    cross.loc[0, "database_or_analysis"] = "MIMIC revised 11.12"
    for k in [
        "risk_difference",
        "rd_ci_low",
        "rd_ci_high",
        "risk_ratio",
        "rr_ci_low",
        "rr_ci_high",
    ]:
        cross.loc[0, k] = hm[k]
        cross.loc[len(cross) - 1, k] = new_tr[k]
    cross.loc[len(cross) - 1, "database_or_analysis"] = (
        "MIMIC revised transported to eICU E2"
    )
    save("cross_database_effects.csv", cross)
    closed.dump(
        Path("results") / "updated" / "transport_validation.json",
        {
            "old_frozen_reproduction_absolute_differences": tr_checks,
            "new_transport": new_tr,
            "membership_columns": tr_cols,
            "eICU_effects_reestimated": False,
            "MIMIC_eICU_pooling": False,
        },
    )
    closed.dump(
        "COMPLETION_ANALYSIS_SUMMARY.json",
        {
            "sensitivity_distinct_estimated": int((sens.status == "ESTIMATED").sum()),
            "sensitivity_duplicate": int((sens.status == "DUPLICATE_PRIMARY").sum()),
            "sensitivity_retired": int(
                (sens.status == "RETIRED_TEMPORAL_VALIDITY").sum()
            ),
            "heterogeneity": het,
            "unmeasured_confounding": uc,
            "transport": new_tr,
            "source_copies": source_meta,
            "eicu_diagnosis_timing": "diagnosisoffset BETWEEN -1440 AND t0; frozen eICU effects retained",
        },
    )
    after = closed.snapshot()
    changed = [k for k, v in before.items() if after.get(k) != v]
    added = sorted(set(after) - set(before))
    closed.dump(
        Path("validation") / "analysis_integrity.json",
        {
            "outside_files_checked": len(before),
            "outside_modified": changed,
            "outside_added": added,
            "first_primary_results_hash": closed.sha(
                R / "results" / "removed_proxy_effects.csv"
            ),
            "no_removed_proxy_in_MIMIC_models": True,
            "no_cross_database_pooling": True,
            "eICU_frozen_inherited": True,
        },
    )
    assert not changed and not added
    log("COMPLETE analysis extension; original files unchanged.")
