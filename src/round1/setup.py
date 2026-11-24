"""Prepare frozen inputs and reproduce the baseline cohort and weights."""

import json
import logging
import shutil

import numpy as np
import pandas as pd

import src.design as design_module
import src.effects as effects_module
import src.runtime as rt

logger = logging.getLogger(__name__)


R = rt.R1
for name in [
    "data",
    "inputs",
    "analysis",
    "outputs",
    "figures",
    "logs",
    "sources",
    "sql",
    "tests",
    "tmp",
]:
    (R / name).mkdir(exist_ok=True)
sha = rt.sha


def query(sql, name):
    return rt.query(sql, name, R)


def load_cohort():
    return rt.cohort(R)


def prepare():
    rt.stage_models(R)
    for name in [
        "mimic_design_candidates.csv",
        "mimic_outcomes.csv",
        "original_weights.csv",
    ]:
        rt.copy_immutable(rt.configured("frozen_inputs") / name, R / "inputs" / name)
    for name, reference in [
        ("expected_removed_proxy_effects.csv", "round1_expected_effects.csv"),
        ("expected_removed_balance.csv", "round1_expected_balance.csv"),
        ("expected_removed_diagnostics.json", "round1_expected_diagnostics.json"),
    ]:
        shutil.copy2(rt.ASSETS / "benchmarks" / reference, R / "inputs" / name)
    for f in (rt.ASSETS / "sources/official").iterdir():
        if f.is_file():
            shutil.copy2(f, R / "sources" / f.name)
    shutil.copy2(
        rt.ASSETS / "sources/frozen_design_cohort.sql",
        R / "sources/frozen_design_cohort.sql",
    )
    shutil.copy2(rt.ASSETS / "config/round1.json", R / "analysis_config.json")
    cache = rt.configured("round1_cache", False)
    if cache:
        for folder, names in [
            (
                "data",
                ["sofa_observations.csv", "windows.csv", "rebuilt_candidates.csv"],
            ),
            ("outputs", ["historical_lab_tie_audit.csv"]),
        ]:
            for name in names:
                source = cache / folder / name
                if source.is_file() and (not (R / folder / name).exists()):
                    shutil.copy2(source, R / folder / name)
    m = load_cohort()
    design = design_module
    est = effects_module
    est.design = design
    rows = []
    expected = pd.read_csv(R / "inputs/expected_removed_proxy_effects.csv").set_index(
        "outcome"
    )
    for outcome in ["alive_sustainably_liberated_day28", "hospital_mortality"]:
        q = est.estimate(m, outcome)
        q["outcome"] = outcome
        rows.append(q)
        for c in [
            "n",
            "DEX_n",
            "MIDAZ_n",
            "risk_difference",
            "rd_ci_low",
            "rd_ci_high",
            "risk_ratio",
            "rr_ci_low",
            "rr_ci_high",
        ]:
            assert abs(q[c] - expected.loc[outcome, c]) < 1e-06
    X = design.design_matrix(m)
    a = (m.treatment == "DEX").to_numpy(int)
    _, e = design.ridge_logit(X.to_numpy(), a, 1.0)
    w = pd.read_csv(R / "inputs/original_weights.csv", dtype={"stay_id": str})
    assert m.stay_id.tolist() == w.stay_id.tolist()
    assert np.max(abs(e - w.propensity)) < 1e-10
    pd.DataFrame(rows).to_csv(R / "outputs/model0_reproduction.csv", index=False)
    m.to_csv(R / "data/model0_cohort.csv", index=False)
    assert (
        sha(R / "data/model0_cohort.csv")
        == json.loads((R / "analysis_config.json").read_text())["source_cohort_sha256"]
    ), (
        "Frozen model0 CSV checksum differs; inspect preprocessing/versions rather than changing results."
    )
    logger.info("Round 1 inputs and baseline estimates verified.")
