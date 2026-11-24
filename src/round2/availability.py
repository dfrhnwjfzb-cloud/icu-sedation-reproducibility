import json
import logging

import numpy as np
import pandas as pd

import src.round1.scores as sc
import src.round2.common as p
import src.round2.diagnostics as diagnostics_module
import src.round2.engine as en

logger = logging.getLogger(__name__)


R = p.R


def timeparse(d, cols):
    for c in cols:
        d[c] = pd.to_datetime(d[c])
    return d


def main():
    x = p.baseline()
    windows = timeparse(
        pd.read_csv(p.OLD / "data/windows.csv"), ["t0", "ws", "intime", "outtime"]
    ).set_index("stay_id")
    obs = timeparse(
        pd.read_csv(p.OLD / "data/sofa_observations.csv"), ["charttime", "endtime"]
    )
    obs["meta"] = obs.metadata.map(json.loads)
    lab = timeparse(pd.read_csv(R / "data/raw_labs.csv"), ["charttime", "storetime"])
    charts = timeparse(
        pd.read_csv(R / "data/raw_charts.csv"), ["charttime", "storetime"]
    )
    vaso = timeparse(
        pd.read_csv(R / "data/raw_vaso.csv"), ["starttime", "endtime", "storetime"]
    )
    ur = timeparse(pd.read_csv(R / "data/raw_urine.csv"), ["charttime", "storetime"])
    du = timeparse(pd.read_csv(R / "data/derived_urine.csv"), ["charttime"])
    rate = timeparse(pd.read_csv(R / "data/urine_rate_history.csv"), ["charttime"])
    groups = []
    for name, d in [("lab", lab), ("chart", charts), ("vaso", vaso), ("urine", ur)]:
        q = d.merge(windows[["t0", "ws"]], on="stay_id", validate="many_to_one")
        q = q.merge(
            x[["stay_id", "treatment"]].assign(stay_id=lambda a: a.stay_id.astype(int)),
            on="stay_id",
            validate="many_to_one",
        )
        tm = "starttime" if name == "vaso" else "charttime"
        assert (q[tm] < q.t0).all()
        for (tx, item), z in q.groupby(["treatment", "itemid"]):
            groups.append(
                {
                    "table": name,
                    "treatment": tx,
                    "itemid": item,
                    "rows": len(z),
                    "stays": z.stay_id.nunique(),
                    "storetime_missing": int(z.storetime.isna().sum()),
                    "chart_pre_t0_store_ge_t0": int((z.storetime >= z.t0).sum()),
                    "store_minus_event_hours_median": (
                        (z.storetime - z[tm]).dt.total_seconds() / 3600
                    ).median(),
                }
            )
    pd.DataFrame(groups).to_csv(R / "outputs/raw_storetime_audit.csv", index=False)
    lab["variable"] = lab.itemid.map(
        {
            50912: "creatinine",
            50885: "bilirubin",
            51265: "platelet",
            50821: "po2",
            50816: "fio2_lab",
        }
    )
    lab["matchvalue"] = lab.valuenum
    fio = lab.variable == "fio2_lab"
    lab.loc[fio & (lab.matchvalue >= 0.2) & (lab.matchvalue <= 1), "matchvalue"] *= 100
    lm = (
        lab.groupby(["stay_id", "variable", "charttime", "matchvalue"])
        .agg(
            store_min=("storetime", "min"),
            store_max=("storetime", "max"),
            source_rows=("storetime", "size"),
            missing_store=("storetime", lambda s: s.isna().sum()),
        )
        .reset_index()
    )
    common = obs[obs.variable.isin(["creatinine", "bilirubin", "platelet"])].merge(
        lm,
        left_on=["stay_id", "variable", "charttime", "value"],
        right_on=["stay_id", "variable", "charttime", "matchvalue"],
        how="left",
        validate="many_to_one",
    )
    cv = (
        charts[
            charts.itemid.isin([220052, 220181, 225312])
            & charts.valuenum.between(1e-12, 299.999999)
        ]
        .groupby(["stay_id", "charttime"])
        .agg(
            raw_map=("valuenum", "mean"),
            store_min=("storetime", "min"),
            store_max=("storetime", "max"),
            source_rows=("storetime", "size"),
            missing_store=("storetime", lambda s: s.isna().sum()),
        )
        .reset_index()
    )
    maps = obs[obs.variable == "map"].merge(
        cv, on=["stay_id", "charttime"], how="left", validate="many_to_one"
    )
    assert np.nanmax(abs(maps.value - maps.raw_map)) < 1e-07
    records = []
    for q in [common, maps]:
        for r in q.itertuples():
            records.append(
                {
                    "stay_id": r.stay_id,
                    "variable": r.variable,
                    "charttime": r.charttime,
                    "value": r.value,
                    "store_min": r.store_min,
                    "store_max": r.store_max,
                    "missing_store": r.missing_store,
                    "match": "matched" if pd.notna(r.source_rows) else "unmatched",
                }
            )
    vm = {
        221662: "dopamine",
        221653: "dobutamine",
        221289: "epinephrine",
        221906: "norepinephrine",
    }
    vaso["variable"] = vaso.itemid.map(vm)
    vi = (
        vaso.groupby(["stay_id", "variable", "starttime", "endtime"])
        .agg(
            store_min=("storetime", "min"),
            store_max=("storetime", "max"),
            missing_store=("storetime", lambda s: s.isna().sum()),
        )
        .reset_index()
    )
    lookup = {
        (r.stay_id, r.variable, r.starttime, r.endtime): r for r in vi.itertuples()
    }
    for r in obs[obs.variable.isin(vm.values())].itertuples():
        k = (
            r.stay_id,
            r.variable,
            pd.Timestamp(r.meta["original_start"]),
            pd.Timestamp(r.meta["original_end"]),
        )
        z = lookup.get(k)
        records.append(
            {
                "stay_id": r.stay_id,
                "variable": r.variable,
                "charttime": r.charttime,
                "value": r.value,
                "store_min": z.store_min if z else pd.NaT,
                "store_max": z.store_max if z else pd.NaT,
                "missing_store": z.missing_store if z else np.nan,
                "match": "matched" if z else "unmatched",
            }
        )
    for r in obs[obs.variable == "gcs"].itertuples():
        records.append(
            {
                "stay_id": r.stay_id,
                "variable": "gcs",
                "charttime": r.charttime,
                "value": r.value,
                "store_min": pd.NaT,
                "store_max": pd.NaT,
                "missing_store": np.nan,
                "match": "derived_carry_default_unconfirmed",
            }
        )
    li = {
        (r.stay_id, r.variable, r.charttime, round(r.matchvalue, 7)): r
        for r in lm.itertuples()
    }
    cf = charts[charts.itemid == 223835].copy()
    cf["fio2"] = np.where(cf.valuenum.between(0.2, 1), cf.valuenum * 100, cf.valuenum)
    cf = cf[cf.fio2.between(21, 100)]
    cgroups = {sid: q for sid, q in cf.groupby("stay_id")}
    for r in obs[obs.variable == "pf"].itertuples():
        po = li.get((r.stay_id, "po2", r.charttime, round(r.meta.get("po2") or -1, 7)))
        fi = None
        if r.meta.get("fio2") is not None:
            fi = li.get((r.stay_id, "fio2_lab", r.charttime, round(r.meta["fio2"], 7)))
        elif r.stay_id in cgroups:
            cand = cgroups[r.stay_id]
            cand = cand[
                (cand.charttime <= r.charttime)
                & (cand.charttime >= r.charttime - pd.Timedelta(hours=4))
            ]
            if len(cand):
                last = cand[cand.charttime == cand.charttime.max()]
                target = r.meta.get("fio2_chart")
                if target is not None and np.isclose(last.fio2.mean(), target):
                    fi = type(
                        "Fi",
                        (),
                        {
                            "store_min": last.storetime.min(),
                            "store_max": last.storetime.max(),
                            "missing_store": int(last.storetime.isna().sum()),
                        },
                    )()
        stores = [
            z.store_max for z in [po, fi] if z is not None and pd.notna(z.store_max)
        ]
        mins = [
            z.store_min for z in [po, fi] if z is not None and pd.notna(z.store_min)
        ]
        good = po is not None and fi is not None and (len(stores) == 2)
        records.append(
            {
                "stay_id": r.stay_id,
                "variable": "pf",
                "charttime": r.charttime,
                "value": r.value,
                "store_min": min(mins) if mins else pd.NaT,
                "store_max": max(stores) if stores else pd.NaT,
                "missing_store": sum(
                    z.missing_store for z in [po, fi] if z is not None
                ),
                "match": "matched" if good else "partial_input_chain",
            }
        )
    av = pd.DataFrame(records).merge(
        windows[["t0", "ws"]], on="stay_id", validate="many_to_one"
    )
    av["availability"] = np.select(
        [
            av["match"] != "matched",
            av.store_min.isna() | (av.missing_store > 0),
            av.store_max < av.t0,
            av.store_min >= av.t0,
        ],
        [
            "Unconfirmed",
            "Missing_storetime",
            "All_matched_sources_pre_t0",
            "All_matched_sources_late",
        ],
        default="Mixed_pre_and_late",
    )
    av.to_csv(R / "data/component_availability_records.csv", index=False)
    av = av.merge(
        x[["stay_id", "treatment"]].assign(stay_id=lambda a: a.stay_id.astype(int)),
        on="stay_id",
        validate="many_to_one",
    )
    av.groupby(["treatment", "variable", "availability"]).agg(
        records=("stay_id", "size"), stays=("stay_id", "nunique")
    ).to_csv(R / "outputs/component_availability_summary.csv")
    winning = []
    winner_lookup = (
        av.drop_duplicates(["stay_id", "variable", "charttime"])
        .set_index(["stay_id", "variable", "charttime"])
        .availability.to_dict()
    )
    for row in x.itertuples():
        for organ in p.ORGS:
            var = getattr(row, organ + "_worst_variable")
            tm = getattr(row, organ + "_worst_charttime")
            status = (
                winner_lookup.get(
                    (int(row.stay_id), var, pd.Timestamp(tm)), "Missing_or_unmatched"
                )
                if pd.notna(tm)
                else "Missing_or_unmatched"
            )
            winning.append(
                {
                    "stay_id": row.stay_id,
                    "treatment": row.treatment,
                    "organ": organ,
                    "source_variable": var,
                    "clinical_time": tm,
                    "availability": status,
                }
            )
    pd.DataFrame(winning).to_csv(
        R / "data/worst_component_availability.csv", index=False
    )
    pd.DataFrame(winning).groupby(["treatment", "organ", "availability"]).size().rename(
        "n"
    ).to_csv(R / "outputs/worst_component_availability_summary.csv")
    ur["net"] = np.where((ur.itemid == 227488) & (ur.value > 0), -ur.value, ur.value)
    net = (
        ur.groupby(["stay_id", "charttime"])
        .agg(
            urineoutput=("net", "sum"),
            store_max=("storetime", "max"),
            missing_store=("storetime", lambda s: s.isna().sum()),
        )
        .reset_index()
    )
    cmp = du.merge(
        net,
        on=["stay_id", "charttime"],
        how="outer",
        suffixes=("_derived", "_raw"),
        indicator=True,
    )
    mismatch = (cmp._merge != "both") | ~np.isclose(
        cmp.urineoutput_derived, cmp.urineoutput_raw, equal_nan=True
    )
    urine_qc = {
        "raw_rows": len(ur),
        "raw_stays": ur.stay_id.nunique(),
        "derived_rows": len(du),
        "exact_comparison_mismatches": int(mismatch.sum()),
        "negative_net_timepoints": int((net.urineoutput < 0).sum()),
        "normalization": "None; actual volumes only",
    }
    cmp.to_csv(R / "data/raw_vs_derived_urine.csv", index=False)
    records = []
    for row in x.itertuples():
        sid = int(row.stay_id)
        q = net[net.stay_id == sid].sort_values("charttime")
        t0 = windows.loc[sid, "t0"]
        ws = windows.loc[sid, "ws"]
        gaps = q.charttime.diff().dt.total_seconds() / 3600
        first = q.charttime.min()
        last = q.charttime.max()
        span = (last - first).total_seconds() / 3600 if len(q) > 1 else 0.0
        maxgap = gaps.max() if len(q) > 1 else np.nan
        cover = bool(
            row.window_hours >= 24
            and len(q) > 1
            and (first <= ws + pd.Timedelta(hours=1))
            and (last >= t0 - pd.Timedelta(hours=1))
            and (span >= 22)
            and (maxgap <= 2)
            and (q.urineoutput >= 0).all()
        )
        known = bool(
            cover
            and q.store_max.notna().all()
            and (q.store_max < t0).all()
            and (q.missing_store.sum() == 0)
        )
        v = q.urineoutput.sum() if len(q) else np.nan
        score = sc.score("urine24", v) if known else np.nan
        records.append(
            {
                "stay_id": row.stay_id,
                "treatment": row.treatment,
                "icu_group": row.icu_group,
                "window_hours": row.window_hours,
                "raw_urine_timepoints": len(q),
                "observed_span_h": span,
                "max_gap_h": maxgap,
                "first_time": first,
                "last_time": last,
                "documented_mL": v,
                "near_complete_chart_coverage": cover,
                "known_available_near_complete": known,
                "urine_sensitivity_score": score,
            }
        )
    uu = pd.DataFrame(records)
    uu.to_csv(R / "data/urine_fixed_window_audit.csv", index=False)
    urine_qc.update(
        {
            "near_complete_chart_coverage_stays": int(
                uu.near_complete_chart_coverage.sum()
            ),
            "known_available_near_complete_stays": int(
                uu.known_available_near_complete.sum()
            ),
        }
    )
    for h in [6, 12, 24]:
        q = rate[rate["uo_tm_" + str(h) + "hr"] >= h]
        urine_qc["derived_" + str(h) + "h_duration_stays"] = int(q.stay_id.nunique())
    (R / "outputs/urine_deep_audit.json").write_text(
        json.dumps(
            urine_qc,
            default=lambda v: int(v) if isinstance(v, np.integer) else str(v),
            indent=2,
        ),
        encoding="utf-8",
    )
    associations = []
    for strat in ["treatment", "icu_group"]:
        for lev, q in uu.groupby(strat):
            associations.append(
                {
                    "stratifier": strat,
                    "level": lev,
                    "n": len(q),
                    "any_raw_urine_pct": 100 * (q.raw_urine_timepoints > 0).mean(),
                    "near_complete_chart_pct": 100
                    * q.near_complete_chart_coverage.mean(),
                    "available_near_complete_pct": 100
                    * q.known_available_near_complete.mean(),
                    "median_span_h": q.observed_span_h.median(),
                }
            )
    pd.DataFrame(associations).to_csv(
        R / "outputs/urine_coverage_associations.csv", index=False
    )
    if uu.known_available_near_complete.any():
        xx = x.merge(
            uu[["stay_id", "urine_sensitivity_score"]],
            on="stay_id",
            validate="one_to_one",
        )
        xx["renal_original"] = xx.renal
        xx["renal"] = xx[["renal", "urine_sensitivity_score"]].max(axis=1, skipna=True)
        xx["renal_missing"] = xx.renal.isna().astype(float)
        xx["pre_t0_sofa"] = (
            xx.pre_t0_sofa - xx.renal_original.fillna(0) + xx.renal.fillna(0)
        )
        result, info = en.fit(xx, "ModelS")
        rules = pd.read_csv(
            R / "data/round2_scoring_audit.csv", dtype={"stay_id": str}
        )[["stay_id", "cns_G1", "cns_G2"]]
        xx = xx.merge(rules, on="stay_id", validate="one_to_one")
        helper = diagnostics_module
        _, dg, _ = helper.diagnostics(
            xx, "Urine_raw_fixed_window_sensitivity", "Full", info
        )
        for r in result:
            r.update(dg)
        pd.DataFrame(result).assign(
            changed_score_stays=int(
                (xx.renal.fillna(0) != xx.renal_original.fillna(0)).sum()
            )
        ).to_csv(R / "outputs/urine_effect_sensitivity.csv", index=False)
    logger.info("Recording availability and urine diagnostics complete")
