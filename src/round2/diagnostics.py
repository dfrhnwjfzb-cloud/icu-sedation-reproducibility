import json
import logging

import numpy as np
import pandas as pd

import src.round1.scores as sc
import src.round2.common as p
import src.round2.engine as en

logger = logging.getLogger(__name__)


R = p.R


def gcs_versions(x, obs):
    raw = obs[obs.variable.str.startswith("gcs_raw")].copy()
    raw["meta"] = raw.metadata.map(json.loads)
    raw["charttime"] = pd.to_datetime(raw.charttime)
    windows = pd.read_csv(p.OLD / "data/windows.csv").set_index("stay_id")
    records = []
    scores = []
    for row in x.itertuples():
        z = raw[raw.stay_id == int(row.stay_id)]
        t0 = pd.Timestamp(windows.loc[int(row.stay_id), "t0"])
        state = {}
        seen = {1: [], 2: []}
        for t, q in z.groupby("charttime", sort=True):
            current = {}
            for rr in q.itertuples():
                st = rr.meta.get("storetime")
                part = rr.variable.replace("gcs_raw_", "")
                v = rr.value
                if st is None or pd.Timestamp(st) >= t0:
                    continue
                if rr.meta.get("ett") or (part == "verbal" and v == 0):
                    state.pop("verbal", None)
                    continue
                maxval = {"eyes": 4, "motor": 6, "verbal": 5}[part]
                if not 1 <= v <= maxval:
                    state.pop(part, None)
                    continue
                item = {"value": v, "charttime": t, "storetime": pd.Timestamp(st)}
                current[part] = item
                state[part] = item
            valid = {
                k: v
                for k, v in state.items()
                if t - v["charttime"] <= pd.Timedelta(hours=6)
            }
            for rule, parts in [(1, current), (2, valid)]:
                if len(parts) != 3:
                    continue
                total = sum(v["value"] for v in parts.values())
                cns = sc.score("gcs", total)
                seen[rule].append(cns)
                out = {
                    "stay_id": row.stay_id,
                    "rule": "G" + str(rule),
                    "evaluation_time": t,
                    "GCS": total,
                    "CNS": cns,
                }
                for k, v in parts.items():
                    for a, vl in v.items():
                        out[k + "_" + a] = vl
                records.append(out)
        scores.append(
            {
                "stay_id": row.stay_id,
                "cns_G1": max(seen[1]) if seen[1] else np.nan,
                "cns_G2": max(seen[2]) if seen[2] else np.nan,
                "G1_records": len(seen[1]),
                "G2_records": len(seen[2]),
            }
        )
    out = pd.DataFrame(scores)
    pd.DataFrame(records).to_csv(R / "data/gcs_auditable_triplets.csv", index=False)
    assert np.allclose(out.cns_G1.fillna(-1), x.cns_raw_observed.fillna(-1)), (
        "G1 failed to reproduce raw-complete CNS from round1"
    )
    x = x.merge(out, on="stay_id", validate="one_to_one")
    for rule in ["G1", "G2"]:
        x["SOFA_" + rule + "_CNS_observed"] = np.where(
            x["cns_" + rule].notna(),
            x.pre_t0_sofa - x.cns.fillna(0) + x["cns_" + rule],
            np.nan,
        )
        x["SOFA_" + rule + "_strict"] = np.where(
            (x[[o + "_missing" for o in p.ORGS if o != "cns"]].sum(axis=1) == 0)
            & x["cns_" + rule].notna(),
            x["SOFA_" + rule + "_CNS_observed"],
            np.nan,
        )
    return x


def summarize_gcs(x, obs):
    rows = []
    for tx, z in x.groupby("treatment"):
        for rule, col in [("G0", "cns"), ("G1", "cns_G1"), ("G2", "cns_G2")]:
            v = z[col]
            r = {
                "treatment": tx,
                "rule": rule,
                "n": len(z),
                "CNS_observed_n": int(v.notna().sum()),
                "CNS_missing_pct": 100 * v.isna().mean(),
                "CNS_mean_observed": v.mean(),
            }
            for s in range(5):
                r["CNS_" + str(s) + "_n"] = int((v == s).sum())
            if rule != "G0":
                change = z["SOFA_" + rule + "_CNS_observed"] - z.pre_t0_sofa
                r["SOFA_comparable_n"] = int(change.notna().sum())
                r["SOFA_change_mean_observed_CNS"] = change.mean()
                r["SOFA_change_median_observed_CNS"] = change.median()
                r["SOFA_strict_n"] = int(z["SOFA_" + rule + "_strict"].notna().sum())
            rows.append(r)
    pd.DataFrame(rows).to_csv(R / "outputs/gcs_rule_coverage.csv", index=False)
    g = obs[obs.variable == "gcs"].copy()
    g["meta"] = g.metadata.map(json.loads)
    g["suspect15"] = [
        r.value == 15
        and (
            r.meta.get("unable") == 1
            or any(r.meta.get(c) is None for c in ["motor", "eyes", "verbal"])
            or sum(r.meta.get(c) or 0 for c in ["motor", "eyes", "verbal"]) != r.value
        )
        for r in g.itertuples()
    ]
    g["GCS15_case"] = [
        "Not15"
        if r.value != 15
        else "ETT_default_flag"
        if r.meta.get("unable") == 1 or r.meta.get("verbal") == 0
        else "Missing_or_nonadditive"
        if r.suspect15
        else "Additive_components"
        for r in g.itertuples()
    ]
    g = g.merge(
        x[["stay_id", "treatment", "cns_worst_charttime"]].assign(
            stay_id=lambda z: z.stay_id.astype(int)
        ),
        on="stay_id",
        validate="many_to_one",
    )
    g["winner_time"] = pd.to_datetime(g.charttime) == pd.to_datetime(
        g.cns_worst_charttime
    )
    g.groupby(["treatment", "GCS15_case"]).agg(
        records=("stay_id", "size"),
        stays=("stay_id", "nunique"),
        selected_worst_records=("winner_time", "sum"),
    ).to_csv(R / "outputs/gcs15_classification.csv")
    g[
        ["stay_id", "charttime", "value", "GCS15_case", "suspect15", "winner_time"]
    ].to_csv(R / "data/gcs15_record_audit.csv", index=False)


def window_audit(x, obs):
    w = pd.read_csv(p.OLD / "data/windows.csv", dtype={"stay_id": str})
    t = x.merge(
        w[["stay_id", "intime", "outtime", "t0", "ws"]],
        on="stay_id",
        validate="one_to_one",
    )
    t["window_bin"] = pd.cut(
        t.window_hours,
        [-1, 6, 12, 24 - 1e-08, 25],
        labels=["<6h", "6-12h", "12-<24h", "24h"],
    )
    obs = obs.copy()
    obs.charttime = pd.to_datetime(obs.charttime)
    obs.endtime = pd.to_datetime(obs.endtime)
    ot = obs.merge(
        w[["stay_id", "t0", "ws"]].assign(stay_id=lambda z: z.stay_id.astype(int)),
        on="stay_id",
        validate="many_to_one",
    )
    assert (ot.charttime < pd.to_datetime(ot.t0)).all() and (
        ot.charttime >= pd.to_datetime(ot.ws)
    ).all()
    t.to_csv(R / "data/window_stay_audit.csv", index=False)
    rows = []
    for (tx, bin), q in t.groupby(["treatment", "window_bin"], observed=True):
        r = {
            "treatment": tx,
            "window_bin": str(bin),
            "n": len(q),
            "window_hours_median": q.window_hours.median(),
            "SOFA_mean": q.pre_t0_sofa.mean(),
            "SOFA_median": q.pre_t0_sofa.median(),
            "missing_organs_mean": q.missing_organs.mean(),
            "all_organs_observed_pct": 100 * q.pre_t0_sofa_strict.notna().mean(),
        }
        for o in p.ORGS:
            r[o + "_observed_pct"] = 100 * (1 - q[o + "_missing"].mean())
        rows.append(r)
    pd.DataFrame(rows).to_csv(R / "outputs/window_by_treatment.csv", index=False)
    scored = pd.read_csv(p.OLD / "data/sofa_observations_scored.csv")
    scored.charttime = pd.to_datetime(scored.charttime)
    scored.endtime = pd.to_datetime(scored.endtime)
    nested = []
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
        "cns": ["gcs"],
        "renal": ["creatinine", "urine24"],
    }
    for r in t[t.window_hours >= 24].itertuples():
        sub = scored[scored.stay_id == int(r.stay_id)]
        rr = {"stay_id": r.stay_id, "treatment": r.treatment}
        for hours in [6, 12, 24]:
            cut = pd.Timestamp(r.t0) - pd.Timedelta(hours=hours)
            q = sub[(sub.charttime >= cut) | (sub.endtime > cut)]
            ss = [q.loc[q.variable.isin(v), "subscore"].max() for v in mp.values()]
            rr[f"SOFA_{hours}h"] = sum(0 if pd.isna(v) else v for v in ss)
            rr[f"missing_organs_{hours}h"] = sum(pd.isna(v) for v in ss)
        assert rr["SOFA_6h"] <= rr["SOFA_12h"] <= rr["SOFA_24h"]
        nested.append(rr)
    pd.DataFrame(nested).to_csv(
        R / "outputs/paired_nested_window_scores.csv", index=False
    )


def diagnostics(x, model, group, info):
    a = (x.treatment == "DEX").to_numpy(int)
    w = info["ow"]
    X = info["X"]
    records = []
    tails = []
    vars = {c: X[c].to_numpy(float) for c in X}
    for o in p.ORGS:
        vars["diagnostic_" + o] = x[o].to_numpy(float)
        vars["diagnostic_" + o + "_missing"] = x[o].isna().to_numpy(float)
    for c in [
        "pre_t0_sofa",
        "pre_t0_sofa_strict",
        "window_hours",
        "short_window",
        "cns_G1",
        "cns_G2",
    ]:
        vars["diagnostic_" + c] = x[c].to_numpy(float)
    for var, v in vars.items():
        rr = {
            "group": group,
            "model": model,
            "variable": var,
            "variable_set": "Diagnostic" if var.startswith("diagnostic_") else "Model",
            "raw_SMD": en.b.smd(v, a),
            "weighted_SMD": en.b.smd(v, a, w),
        }
        for arm, tx in [(1, "DEX"), (0, "MIDAZ")]:
            k = a == arm
            rr[tx + "_missing_pct"] = 100 * np.mean(~np.isfinite(v[k]))
            rr[tx + "_raw_mean"] = en.b.wmean(v[k], np.ones(k.sum()))
            rr[tx + "_weighted_mean"] = en.b.wmean(v[k], w[k])
            if var in [
                "bradycardia_proxy",
                "diagnostic_pre_t0_sofa",
                "diagnostic_cns",
                "diagnostic_window_hours",
                "diagnostic_renal",
                "diagnostic_liver",
            ]:
                vv = v[k]
                ww = w[k]
                ok = np.isfinite(vv)
                vv = vv[ok]
                ww = ww[ok]
                order = np.argsort(vv)
                vv = vv[order]
                ww = ww[order]
                if len(vv):
                    cdf = np.cumsum(ww) / ww.sum()
                    quant = np.interp([0.01, 0.05, 0.5, 0.95, 0.99], cdf, vv)
                    tails.append(
                        dict(
                            group=group,
                            model=model,
                            variable=var,
                            treatment=tx,
                            **dict(
                                zip(
                                    [
                                        "weighted_p01",
                                        "weighted_p05",
                                        "weighted_median",
                                        "weighted_p95",
                                        "weighted_p99",
                                    ],
                                    quant,
                                )
                            ),
                        )
                    )
        records.append(rr)
    bal = pd.DataFrame(records)
    path = R / "outputs" / f"{group}_{model}_balance.csv"
    bal.to_csv(path, index=False)
    bal.assign(absSMD=bal.weighted_SMD.abs()).sort_values(
        "absSMD", ascending=False
    ).head(15).to_csv(R / "outputs" / f"{group}_{model}_top15.csv", index=False)
    e = info["ps"]
    ps = x[["stay_id", "patient_id", "treatment"]].copy()
    ps["PS"] = e
    ps["OW"] = w
    ps.to_csv(R / "data" / f"{group}_{model}_weights.csv", index=False)
    smd_model = bal[bal.variable_set == "Model"].weighted_SMD.abs().max()
    eligible = bal[
        bal.variable.isin(
            ["diagnostic_" + o for o in p.ORGS]
            + ["diagnostic_" + o + "_missing" for o in p.ORGS]
        )
    ]
    smd_org = eligible.weighted_SMD.abs().max()
    dg = {
        "group": group,
        "model": model,
        "max_model_SMD": smd_model,
        "max_organ_missing_SMD": smd_org,
        "PS_extreme_pct": 100 * np.mean((e < 0.05) | (e > 0.95)),
        "PS_min": e.min(),
        "PS_max": e.max(),
        "weight_p01": np.quantile(w, 0.01),
        "weight_p50": np.quantile(w, 0.5),
        "weight_p99": np.quantile(w, 0.99),
        "ESS_DEX": en.b.ess(w[a == 1]),
        "ESS_MIDAZ": en.b.ess(w[a == 0]),
        "matrix_rank": int(np.linalg.matrix_rank(X.to_numpy())),
        "columns": X.shape[1],
        "ridge_condition": float(
            np.linalg.cond(
                X.to_numpy().T @ X.to_numpy()
                + np.eye(X.shape[1]) * (0.01 if model == "ModelW" else 1)
            )
        ),
        "PS_central_overlap_low": max(
            np.quantile(e[a == 1], 0.01), np.quantile(e[a == 0], 0.01)
        ),
        "PS_central_overlap_high": min(
            np.quantile(e[a == 1], 0.99), np.quantile(e[a == 0], 0.99)
        ),
    }
    return bal, dg, tails


def estimator_audit(x, info, rows):
    X = np.column_stack([np.ones(len(x)), info["X"].to_numpy()])
    beta = info["ps_beta"]
    a = (x.treatment == "DEX").to_numpy(int)
    e = info["ps"]
    h = e * (1 - e)
    den = h.mean()
    direction = np.linspace(-1, 1, len(beta))
    direction /= np.linalg.norm(direction)
    records = []
    cluster = []

    def value(bb, y, m):
        ee = en.b.sigmoid(X @ bb)
        hh = ee * (1 - ee)
        m1 = (hh * m[:, 1] + a * (1 - ee) * (y - m[:, 1])).mean() / hh.mean()
        m0 = (hh * m[:, 0] + (1 - a) * ee * (y - m[:, 0])).mean() / hh.mean()
        return m1 - m0

    for rr in rows:
        y = x[rr["outcome"]].to_numpy(int)
        m = info["predictions"][rr["outcome"]]
        de = e * (1 - e) * (X @ direction)
        analytic = (
            np.mean(
                (
                    (1 - 2 * e) * (m[:, 1] - rr["DEX_risk"])
                    - a * (y - m[:, 1])
                    - (1 - 2 * e) * (m[:, 0] - rr["MIDAZ_risk"])
                    - (1 - a) * (y - m[:, 0])
                )
                * de
            )
            / den
        )
        fd = (
            value(beta + 1e-05 * direction, y, m)
            - value(beta - 1e-05 * direction, y, m)
        ) / 2e-05
        assert abs(analytic - fd) < 1e-06
        records.append(
            {
                "outcome": rr["outcome"],
                "partial_RD_gradient_analytic": analytic,
                "partial_RD_gradient_finite_difference": fd,
                "absolute_error": abs(analytic - fd),
                "meaning": "Holding outcome predictions fixed; nonzero PS derivative shows omitted nuisance term need not vanish; not a complete sandwich derivation",
            }
        )
        phi = info["influences"][rr["outcome"]]
        g = (
            pd.DataFrame(phi, columns=["RD", "logRR"])
            .groupby(x.patient_id.to_numpy())
            .sum()
        )
        G = len(g)
        n = len(x)
        se = np.sqrt((g**2).sum().to_numpy() * G / (G - 1) / n**2)
        cluster.append(
            {
                "outcome": rr["outcome"],
                "n": n,
                "subjects": G,
                "extra_stays": n - G,
                "RD_IF_iid_se": rr["RD_IF_se"],
                "RD_IF_cluster_se": se[0],
                "RD_cluster_IF_low": rr["RD"] - 1.96 * se[0],
                "RD_cluster_IF_high": rr["RD"] + 1.96 * se[0],
                "RR_cluster_IF_low": np.exp(np.log(rr["RR"]) - 1.96 * se[1]),
                "RR_cluster_IF_high": np.exp(np.log(rr["RR"]) + 1.96 * se[1]),
            }
        )
    return records, cluster


def main():
    x = p.baseline()
    obs = pd.read_csv(p.OLD / "data/sofa_observations.csv")
    x = gcs_versions(x, obs)
    summarize_gcs(x, obs)
    window_audit(x, obs)
    x.to_csv(R / "data/round2_scoring_audit.csv", index=False)
    versions = [
        ("Full", m)
        for m in ["Model0", "ModelS", "G1", "G2", "ModelM", "ModelE", "ModelW"]
    ] + [
        (g, m)
        for g in ["Low", "Intermediate", "High", "NonCardiacCV", "CardiacCV"]
        for m in ["ModelS", "ModelE", "ModelW"]
    ]
    rows = []
    bals = []
    dgs = []
    tails = []
    deriv = []
    clus = []
    for group, model in versions:
        z = en.subset(x, group)
        res, info = en.fit(z, model)
        bal, dg, tail = diagnostics(z, model, group, info)
        bals.append(bal)
        dgs.append(dg)
        tails.extend(tail)
        (R / "logs" / f"convergence_{group}_{model}.json").write_text(
            json.dumps(info["convergence"], indent=2), encoding="utf-8"
        )
        for rr in res:
            rr.update(dg)
            rows.append(rr)
        if (group, model) in [
            tuple(v)
            for v in json.loads((R / "config.json").read_text())["bootstrap"]["models"]
        ]:
            dd, cc = estimator_audit(z, info, res)
            for r in dd:
                r.update(group=group, model=model)
                deriv.append(r)
            for r in cc:
                r.update(group=group, model=model)
                clus.append(r)
        logger.info(
            f"{group}/{model}: model SMD {dg['max_model_SMD']:.3f}; organs {dg['max_organ_missing_SMD']:.3f}; ESS {dg['ESS_DEX']:.1f}/{dg['ESS_MIDAZ']:.1f}"
        )
    pd.DataFrame(rows).to_csv(R / "05_model_comparison.csv", index=False)
    pd.concat(bals).to_csv(R / "04_subgroup_balance_diagnostics.csv", index=False)
    pd.DataFrame(dgs).to_csv(R / "outputs/model_support_diagnostics.csv", index=False)
    pd.DataFrame(tails).to_csv(R / "outputs/weighted_covariate_tails.csv", index=False)
    pd.DataFrame(deriv).to_csv(
        R / "outputs/PS_derivative_numerical_audit.csv", index=False
    )
    pd.DataFrame(clus).to_csv(R / "outputs/cluster_IF_comparison.csv", index=False)
    missing = pd.concat(bals)
    missing = missing[
        missing.variable.str.startswith("diagnostic_")
        & missing.variable.str.endswith("_missing")
    ]
    missing.to_csv(R / "03_missingness_analysis.csv", index=False)
    assoc = []
    for strat in ["icu_group", "short_window", "sedation_stratum"]:
        for (tx, level), z in x.groupby(["treatment", strat], dropna=False):
            rr = {
                "stratifier": strat,
                "level": str(level),
                "treatment": tx,
                "n": len(z),
                "window_hours_median": z.window_hours.median(),
                "other_than_liver_score_mean": (
                    z.pre_t0_sofa - z.liver.fillna(0)
                ).mean(),
                "creatinine_median": z.creatinine.median(),
            }
            for organ in p.ORGS:
                rr[organ + "_missing_pct"] = 100 * z[organ].isna().mean()
            assoc.append(rr)
    pd.DataFrame(assoc).to_csv(R / "outputs/missingness_associations.csv", index=False)
    logger.info(
        "GCS, nested windows, diagnostics and bounded point-model comparisons complete."
    )
