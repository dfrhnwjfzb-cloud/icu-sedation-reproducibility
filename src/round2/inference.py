import numpy as np
import pandas as pd

import src.round2.common as p

R = p.R


def main():
    d = pd.read_csv(R / "bootstrap/formal.csv")
    logs = pd.read_csv(R / "bootstrap/formal_convergence.csv")
    point = pd.read_csv(R / "outputs/bootstrap_point_estimates.csv")
    cluster = (
        pd.read_csv(R / "outputs/cluster_IF_comparison.csv")
        if (R / "outputs/cluster_IF_comparison.csv").exists()
        else pd.DataFrame()
    )
    rows = []
    for q in point.itertuples():
        z = d[
            (d.group == q.group) & (d.model == q.model) & (d.outcome == q.outcome)
        ].sort_values("replicate")
        ll = logs[(logs.group == q.group) & (logs.model == q.model)]
        assert len(z) >= 500
        rr = q._asdict()
        rr.pop("Index", None)
        rr["bootstrap_effective_n"] = len(z)
        rr["bootstrap_attempts"] = len(ll)
        rr["bootstrap_failures"] = int((~ll.success).sum())
        rr["bootstrap_risk_outside_unit_n"] = int(z.risk_outside_unit.sum())
        rr["bootstrap_nonpositive_RR_n"] = int((z.RR <= 0).sum())
        rr["bootstrap_valid_probability_RR_n"] = int(
            (~z.risk_outside_unit & (z.RR > 0)).sum()
        )
        rr["CI_interpretation_warning"] = (
            "All numerical replicates retained, including out-of-unit AOW risk; RR probability interpretation limited"
            if z.risk_outside_unit.any()
            else "Percentile interval; measurement error and unmeasured confounding not included"
        )
        for v in ["RD", "RR", "DEX_risk", "MIDAZ_risk"]:
            vals = z[v].to_numpy()
            assert np.isfinite(vals).all()
            lo, hi = np.quantile(vals, [0.025, 0.975])
            rr[v + "_bootstrap_low"] = lo
            rr[v + "_bootstrap_high"] = hi
            rr[v + "_bootstrap_SE"] = np.std(vals, ddof=1)
            rr[v + "_bootstrap_mean"] = np.mean(vals)
            half = np.quantile(vals[:250], [0.025, 0.975])
            rr[v + "_first250_low"] = half[0]
            rr[v + "_first250_high"] = half[1]
            rr[v + "_MC_endpoint_max_shift_250_to_500"] = max(
                abs(half - np.array([lo, hi]))
            )
        rr["RD_CI_width_ratio_bootstrap_to_IF"] = (
            rr["RD_bootstrap_high"] - rr["RD_bootstrap_low"]
        ) / (q.RD_IF_high - q.RD_IF_low)
        rr["bootstrap_ESS_DEX_p025"] = z.ESS_DEX.quantile(0.025)
        rr["bootstrap_ESS_MIDAZ_p025"] = z.ESS_MIDAZ.quantile(0.025)
        if len(cluster):
            v = cluster[
                (cluster.group == q.group)
                & (cluster.model == q.model)
                & (cluster.outcome == q.outcome)
            ]
            if len(v):
                rr.update(v.iloc[0].to_dict())
        rows.append(rr)
    out = pd.DataFrame(rows)
    support = pd.read_csv(R / "outputs/model_support_diagnostics.csv")
    out = out.merge(
        support[
            [
                "group",
                "model",
                "max_model_SMD",
                "max_organ_missing_SMD",
                "PS_extreme_pct",
            ]
        ],
        on=["group", "model"],
        how="left",
        validate="many_to_one",
    )
    out.to_csv(R / "06_bootstrap_inference.csv", index=False)
    print(
        out[
            [
                "group",
                "model",
                "outcome",
                "RD",
                "RD_IF_low",
                "RD_IF_high",
                "RD_bootstrap_low",
                "RD_bootstrap_high",
                "bootstrap_effective_n",
                "bootstrap_risk_outside_unit_n",
                "RD_CI_width_ratio_bootstrap_to_IF",
            ]
        ].to_string(index=False),
        flush=True,
    )
