"""Compare regenerated aggregate outputs against the frozen public references."""

import argparse
import json
import logging

import numpy as np
import pandas as pd

import src.runtime as rt

logger = logging.getLogger(__name__)


CHECKS = []


def compare(actual, reference, keys, columns=None):
    a = pd.read_csv(actual).set_index(keys).sort_index()
    b = pd.read_csv(reference).set_index(keys).sort_index()
    assert a.index.equals(b.index), "Reference row identities differ: " + reference.name
    if columns is None:
        columns = [
            c
            for c in b
            if c in a
            and pd.api.types.is_numeric_dtype(b[c])
            and (not pd.api.types.is_bool_dtype(b[c]))
        ]
    for c in columns:
        aa = pd.to_numeric(a[c], errors="coerce").to_numpy(float)
        bb = pd.to_numeric(b[c], errors="coerce").to_numpy(float)
        ok = np.isclose(aa, bb, rtol=0, atol=1e-09, equal_nan=True)
        CHECKS.append(
            {
                "table": reference.name,
                "field": c,
                "rows": len(aa),
                "pass": bool(ok.all()),
                "max_abs_difference": float(np.nanmax(abs(aa - bb)))
                if np.isfinite(aa - bb).any()
                else 0.0,
            }
        )
    assert all(v["pass"] for v in CHECKS), (
        "Frozen numeric differences detected; see validation outputs"
    )


def finalize_round2():
    root = rt.R2
    table = pd.read_csv(root / "05_model_comparison.csv")
    table = table[
        ~table.model.isin(
            ["Availability_components", "Urine_raw_fixed_window_sensitivity"]
        )
    ]
    extra = pd.concat(
        [
            pd.read_csv(root / "outputs/availability_effect_sensitivity.csv"),
            pd.read_csv(root / "outputs/urine_effect_sensitivity.csv"),
        ],
        ignore_index=True,
    )
    table = pd.concat([table, extra], ignore_index=True)
    bal = pd.read_csv(root / "04_subgroup_balance_diagnostics.csv")
    bal = bal[~bal.model.isin(extra.model.unique())]
    for model in extra.model.unique():
        bal = pd.concat(
            [bal, pd.read_csv(root / "outputs" / f"Full_{model}_balance.csv")],
            ignore_index=True,
        )
    bal.to_csv(root / "04_subgroup_balance_diagnostics.csv", index=False)
    for variables, name in [
        (None, "max_all_diagnostic_SMD"),
        (["diagnostic_cns_G1", "diagnostic_cns_G2"], "max_raw_G1G2_CNS_SMD"),
    ]:
        z = (
            bal[bal.variable_set == "Diagnostic"]
            if variables is None
            else bal[bal.variable.isin(variables)]
        )
        summary = (
            z.assign(absSMD=z.weighted_SMD.abs())
            .groupby(["group", "model"])
            .absSMD.max()
            .rename(name)
            .reset_index()
        )
        table = table.drop(columns=name, errors="ignore").merge(
            summary, on=["group", "model"], how="left"
        )
        boot = (
            pd.read_csv(root / "06_bootstrap_inference.csv")
            .drop(columns=name, errors="ignore")
            .merge(summary, on=["group", "model"], how="left")
        )
        boot.to_csv(root / "06_bootstrap_inference.csv", index=False)
    table["source_phase"] = np.where(
        table.model.isin(["Model0", "ModelS"]),
        "Round1 reproduced reference",
        "Round2 post hoc",
    )
    table["CI_method"] = "Original plug-in influence-function (not cluster bootstrap)"
    # Keep the published column order when validation is run repeatedly.
    trailing = [
        "max_all_diagnostic_SMD",
        "max_raw_G1G2_CNS_SMD",
        "source_phase",
        "CI_method",
    ]
    table = table[[c for c in table.columns if c not in trailing] + trailing]
    table.to_csv(root / "05_model_comparison.csv", index=False)
    miss = bal[
        bal.variable.str.startswith("diagnostic_")
        & bal.variable.str.endswith("_missing")
        | bal.variable.isin(
            [
                o + "_missing"
                for o in [
                    "respiration",
                    "coagulation",
                    "liver",
                    "cardiovascular",
                    "cns",
                    "renal",
                ]
            ]
        )
    ].copy()
    miss["missingness_role"] = np.where(
        miss.variable_set == "Model",
        "Model-specific missingness indicator",
        "Diagnostic organ missingness",
    )
    miss["DEX_observed_pct"] = 100 * (1 - miss.DEX_raw_mean)
    miss["MIDAZ_observed_pct"] = 100 * (1 - miss.MIDAZ_raw_mean)
    miss["raw_missing_difference_pp"] = 100 * (miss.DEX_raw_mean - miss.MIDAZ_raw_mean)
    miss.to_csv(root / "03_missingness_analysis.csv", index=False)
    for m in ["ModelS", "ModelE", "ModelW"]:
        z = bal[(bal.group == "High") & (bal.model == m)]
        z.assign(absSMD=z.weighted_SMD.abs()).sort_values(
            "absSMD", ascending=False
        ).head(15).to_csv(root / "outputs" / f"High_{m}_top15.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "round1", "round2"], default="all")
    args = parser.parse_args()
    bench = rt.ASSETS / "benchmarks"
    path = rt.WORK / "validation"
    path.mkdir(exist_ok=True)
    try:
        if args.stage in ["all", "round1"]:
            for actual, ref, keys in [
                (
                    "model_comparison.csv",
                    "round1_model_comparison.csv",
                    ["model", "group", "outcome"],
                ),
                (
                    "sofa_subgroups.csv",
                    "round1_subgroups.csv",
                    ["model", "group", "outcome"],
                ),
                (
                    "icu_restriction.csv",
                    "round1_icu_restriction.csv",
                    ["model", "group", "outcome"],
                ),
                (
                    "urine_sensitivity.csv",
                    "round1_urine_sensitivity.csv",
                    ["model", "group", "outcome"],
                ),
                (
                    "severity_balance_original_weights.csv",
                    "round1_original_balance.csv",
                    ["variable"],
                ),
            ]:
                compare(rt.R1 / "outputs" / actual, bench / ref, keys)
        if args.stage in ["all", "round2"]:
            finalize_round2()
            compare(
                rt.R2 / "05_model_comparison.csv",
                bench / "round2_model_comparison.csv",
                ["group", "model", "outcome"],
            )
            compare(
                rt.R2 / "06_bootstrap_inference.csv",
                bench / "round2_bootstrap.csv",
                ["group", "model", "outcome"],
            )
            log = pd.read_csv(rt.R2 / "bootstrap/formal_convergence.csv")
            assert len(log) == 3000 and log.success.all()
            assert (log.groupby(["group", "model"]).size() == 500).all()
            sample = pd.read_csv(rt.R2 / "bootstrap/formal.csv")
            assert len(sample) == 6000 and int(sample.risk_outside_unit.sum()) == 1
        result = {
            "status": "PASS",
            "stage": args.stage,
            "aggregate_numeric_fields_checked": len(CHECKS),
            "max_absolute_difference": max(q["max_abs_difference"] for q in CHECKS),
            "all_references_passed": True,
            "precision_tolerance": 1e-09,
            "scientific_interpretation": "Post hoc / Level C / high residual-confounding and measurement risk; numerical reproduction completed",
        }
    except Exception as e:
        result = {"status": "FAIL", "stage": args.stage, "error": str(e)}
        (path / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        pd.DataFrame(CHECKS).to_csv(path / "numeric_checks.csv", index=False)
        raise
    pd.DataFrame(CHECKS).to_csv(path / "numeric_checks.csv", index=False)
    (path / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(
        "Reference validation %s: %d numeric fields",
        result["status"],
        result.get("aggregate_numeric_fields_checked", 0),
    )
