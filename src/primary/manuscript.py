import math
import re
from zipfile import ZipFile

import pandas as pd
from lxml import etree as E

import src.settings as settings_module


def audit(R, m, diag, rows, ext):
    target = R / "manuscript/MainDocument.docx"
    if not target.exists():
        target = R / settings_module.LEGACY_FILENAMES["manuscript_fallback"]
    with ZipFile(target) as z:
        tree = E.fromstring(z.read("word/document.xml"))
        ns = {
            "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
            "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
        }
        texts = [
            "".join(p.xpath(".//w:t/text()|.//m:t/text()", namespaces=ns))
            for p in tree.xpath("./w:body/w:p", namespaces=ns)
        ]
    (R / settings_module.LEGACY_FILENAMES["paragraph_extract"]).write_text(
        "\n\n".join((f"[{i}] {t}" for i, t in enumerate(texts) if t)), encoding="utf-8"
    )
    labels = [
        "Study Population",
        "Propensity Scores and Covariate Balance",
        "Primary Outcome",
        "Secondary Outcomes",
        "Sensitivity Analyses",
        "Exploratory Heterogeneity and Residual Confounding Analyses",
        "Multicenter External Consistency Validation",
    ]
    for i, t in enumerate(list(texts)):
        if t.strip() in labels and i + 1 < len(texts):
            texts[i] = t.strip() + " " + texts[i + 1].lstrip()
    checks = []

    def put(start, label, value, reported, digits=1, scale=1, source=""):
        hits = [(i, t) for i, t in enumerate(texts) if t.startswith(start)]
        assert len(hits) == 1, start
        i, t = hits[0]
        tokens = re.findall(
            "(?<![\\w])[-+−]?(?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?", t
        )
        tokens = [x.replace("−", "-").replace(",", "").lstrip("+") for x in tokens]
        s = str(reported).replace("−", "-").replace(",", "").lstrip("+")
        assert any(float(s) == float(x) for x in tokens), (start, reported)
        rounded = f"{value * scale:.{digits}f}"
        expected = f"{float(s):.{digits}f}"
        checks.append(
            {
                "paragraph": i,
                "claim": label,
                "reported": reported,
                "reproduced": value * scale,
                "display_reproduced": rounded,
                "source": source,
                "pass": rounded == expected,
            }
        )

    effect = {r["outcome"]: r for r in rows}
    primary = rows[0]
    flow = pd.read_csv(R / "results/updated/cohort_flow.csv")
    for j, n in enumerate([13416, 6862, 4533, 3669]):
        put(
            "Study Population",
            f"cohort step {j + 1}",
            flow.iloc[j]["n"],
            n,
            0,
            source="results/updated/cohort_flow.csv",
        )
    for tx, n in [("DEX", 2778), ("MIDAZ", 891)]:
        put(
            "Study Population",
            tx + " primary n",
            (m.treatment == tx).sum(),
            n,
            0,
            source="derived/mimic_primary_no_indication_proxies.csv",
        )
    for key, val, dig in [
        ("raw_max_abs_smd", 0.903, 3),
        ("overlap_ess", 1238.5, 1),
        ("DEX_ess", 882.0, 1),
        ("MIDAZ_ess", 477.1, 1),
        ("weighted_max_abs_smd", 0.078, 3),
    ]:
        put(
            "Propensity Scores and Covariate Balance",
            key,
            diag[key],
            val,
            dig,
            source="results/removed_diagnostics.json",
        )

    def result(start, outcome, values):
        for key, reported, digits, scale in values:
            put(
                start,
                outcome + ": " + key,
                effect[outcome][key],
                reported,
                digits,
                scale,
                "results/removed_proxy_effects.csv",
            )

    result(
        "Primary Outcome",
        "alive_sustainably_liberated_day28",
        [
            (k, v, d, s)
            for k, v, d, s in [
                ("DEX_standardized_risk", 85.9, 1, 100),
                ("MIDAZ_standardized_risk", 74.8, 1, 100),
                ("risk_difference", 11.1, 1, 100),
                ("rd_ci_low", 7.0, 1, 100),
                ("rd_ci_high", 15.2, 1, 100),
                ("risk_ratio", 1.15, 2, 1),
                ("rr_ci_low", 1.09, 2, 1),
                ("rr_ci_high", 1.21, 2, 1),
            ]
        ],
    )
    vals = [
        ("hospital_mortality", "Secondary Outcomes", [12.9, 24.3, -11.4, -15.4, -7.4]),
        ("mortality_day28", "Secondary Outcomes", [13.5, 23.4, -9.9, -13.8, -5.9]),
        ("icu_mortality", "Secondary Outcomes", [10.3, 20.3, -10.0, -13.8, -6.3]),
        ("bradycardia_24h", "For safety endpoints", [7.3, 8.4, -1.1, -4.1, 1.9]),
        ("hypotension_24h", "For safety endpoints", [51.7, 48.6, 3.1, -2.2, 8.3]),
        (
            "vasopressor_start_6h",
            "For safety endpoints",
            [52.1, 52.2, -0.02, -4.2, 4.1],
        ),
        ("reintubation_7d", "For safety endpoints", [19.2, 24.9, -5.7, -10.3, -1.1]),
    ]
    for out, start, values in vals:
        keys = [
            "DEX_standardized_risk",
            "MIDAZ_standardized_risk",
            "risk_difference",
            "rd_ci_low",
            "rd_ci_high",
        ]
        result(
            start,
            out,
            [
                (
                    k,
                    v,
                    2
                    if out == "vasopressor_start_6h" and k == "risk_difference"
                    else 1,
                    100,
                )
                for k, v in zip(keys, values)
            ],
        )
    result(
        "Secondary Outcomes",
        "hospital_mortality",
        [
            ("risk_ratio", 0.53, 2, 1),
            ("rr_ci_low", 0.43, 2, 1),
            ("rr_ci_high", 0.66, 2, 1),
        ],
    )
    for out, vs in [
        ("days_to_sustainable_liberation", [2.48, 5.34, -2.86, -3.4, -2.32]),
        ("icu_los_after_t0", [6.41, 10.37, -3.95, -5.13, -2.78]),
        ("hospital_los_after_t0", [13.08, 17.25, -4.16, -6.0, -2.33]),
    ]:
        result(
            "Among the 3,642 ICU stays",
            out,
            [
                (k, v, 2, 1)
                for k, v in zip(
                    ["DEX_mean", "MIDAZ_mean", "mean_difference", "ci_low", "ci_high"],
                    vs,
                )
            ],
        )
    put(
        "Among the 3,642 ICU stays",
        "liberation observed n",
        effect["days_to_sustainable_liberation"]["n"],
        3642,
        0,
    )
    ss = pd.read_csv(R / "results/updated/sensitivity_results.csv").query(
        "status=='ESTIMATED'"
    )
    put(
        "Sensitivity Analyses Across all nine",
        "sensitivity minimum RD",
        ss.risk_difference.min(),
        8.1,
        1,
        100,
    )
    put(
        "Sensitivity Analyses Across all nine",
        "sensitivity maximum RD",
        ss.risk_difference.max(),
        12.1,
        1,
        100,
    )
    checks.append(
        {
            "claim": "Nine sensitivity analyses, all positive CIs",
            "reproduced": len(ss),
            "reported": "nine; all CIs > 0",
            "pass": len(ss) == 9 and bool((ss.rd_ci_low > 0).all()),
            "source": "results/updated/sensitivity_results.csv",
        }
    )
    het = pd.read_csv(
        R / "results/updated/heterogeneity_results.csv", dtype={"subgroup_level": str}
    )
    for var, level, vs in [
        ("deep_sedation", "0", [8.1, 2.9, 13.3]),
        ("deep_sedation", "1", [12.6, 6.2, 19.0]),
        ("propofol_intensity", "low", [12.7, 5.4, 19.9]),
        ("propofol_intensity", "high", [7.5, 2.4, 12.5]),
    ]:
        row = het[(het.subgroup_variable == var) & (het.subgroup_level == level)].iloc[
            0
        ]
        for k, v in zip(["risk_difference", "rd_ci_low", "rd_ci_high"], vs):
            put(
                "Exploratory Heterogeneity and Residual Confounding Analyses",
                var + "/" + level + "/" + k,
                row[k],
                v,
                1,
                100,
                "results/updated/heterogeneity_results.csv",
            )
    uc = pd.read_csv(R / "results/updated/unmeasured_confounding.csv")
    for i, v in [(0, 1.56), (1, 1.4)]:
        put(
            "Exploratory Heterogeneity and Residual Confounding Analyses",
            "E-value " + str(i),
            uc.iloc[i].strength_parameter,
            v,
            2,
        )
    ef = pd.read_csv(ext.T / "eicu_primary_effect.csv")
    er = ef.iloc[0]
    ec = pd.read_csv(ext.D / "eicu_primary_cohort.csv")
    prefix = "Multicenter External Consistency Validation The primary"
    for label, value, reported in [
        ("E1 n", len(ec), 1284),
        ("hospitals", ec.hospital_id.nunique(), 20),
        ("mortality n", er["n"], 1270),
        ("DEX n", er.DEX_n, 658),
        ("MIDAZ n", er.MIDAZ_n, 612),
    ]:
        put(prefix, label, value, reported, 0)
    for k, v, dig, scale in [
        ("DEX_risk", 12.3, 1, 100),
        ("MIDAZ_risk", 26.2, 1, 100),
        ("risk_difference", -14.0, 1, 100),
        ("rd_ci_low", -18.7, 1, 100),
        ("rd_ci_high", -9.3, 1, 100),
        ("risk_ratio", 0.47, 2, 1),
        ("rr_ci_low", 0.36, 2, 1),
        ("rr_ci_high", 0.61, 2, 1),
    ]:
        put(
            prefix,
            k,
            er[k],
            v,
            dig,
            scale,
            "eicu_output/tables/eicu_primary_effect.csv",
        )
    bal = pd.read_csv(ext.T / "eicu_baseline_balance.csv")
    put(
        prefix,
        "E1 full-cohort max weighted SMD",
        bal.overlap_weighted_smd.abs().max(),
        0.077,
        3,
        source="eicu_output/tables/eicu_baseline_balance.csv",
    )
    for analysis, v in [
        ("STRICT_EXPOSURE", -14.0),
        ("BROAD_EXPOSURE", -13.6),
        ("E2_PROBABLE_MV", -13.8),
    ]:
        put(
            "Applying strict and broad",
            analysis,
            ef[ef.analysis == analysis].iloc[0].risk_difference,
            v,
            1,
            100,
        )
    e2 = ef[ef.analysis == "E2_PROBABLE_MV"].iloc[0]
    put("Applying strict and broad", "E2 n", e2["n"], 120, 0)
    for k, v in [("rd_ci_low", -21.9), ("rd_ci_high", -5.7)]:
        put("Applying strict and broad", k, e2[k], v, 1, 100)
    site = pd.read_csv(ext.T / "site_specific_effects.csv")
    loo = pd.read_csv(ext.T / "leave_one_hospital_out.csv")
    for label, value, v, dig, scale in [
        ("site n", len(site), 16, 0, 1),
        ("negative sites", (site.risk_difference < 0).sum(), 15, 0, 1),
        ("meta RD", site.iloc[0].meta_pooled_rd, -18.3, 1, 100),
        ("meta lower", site.iloc[0].meta_ci_low, -23.5, 1, 100),
        ("meta upper", site.iloc[0].meta_ci_high, -13.1, 1, 100),
        ("LOO min", loo.risk_difference.min(), -15.2, 1, 100),
        ("LOO max", loo.risk_difference.max(), -13.0, 1, 100),
    ]:
        put("Among the 16 hospitals", label, value, v, dig, scale)
    br = (
        pd.read_csv(ext.T / "bradycardia_validation.csv")
        .query("window_hours==24 and hr_threshold==60")
        .iloc[0]
    )
    tr = pd.read_csv(R / "results/updated/transported_effects.csv").iloc[-1]
    for label, row, vals in [
        ("bradycardia", br, [12.8, 7.9, 17.8]),
        ("transport", tr, [-5.9, -13.9, 2.2]),
    ]:
        for k, v in zip(["risk_difference", "rd_ci_low", "rd_ci_high"], vals):
            put("Among the 16 hospitals", label + " " + k, row[k], v, 1, 100)
    ceiling = 1 / primary["MIDAZ_standardized_risk"]
    ev = ceiling + math.sqrt(ceiling * (ceiling - 1))
    for label, val, reported, dig in [
        ("RR ceiling", ceiling, 1.34, 2),
        ("ceiling E-value", ev, 2.01, 2),
        ("NNT arithmetic", 1 / primary["risk_difference"], 9, 0),
    ]:
        put("Nevertheless, several important limitations", label, val, reported, dig)
    df = pd.DataFrame(checks)
    df.to_csv(
        R / "validation/MANUSCRIPT_RESULT_CHECKS.csv", index=False, encoding="utf-8-sig"
    )
    assert df["pass"].all(), df[~df["pass"]].to_dict("records")
    return {
        "statistics_checked": len(df),
        "all_displayed_statistics_pass": True,
        "note": "NNT summarizes the adjusted risk difference arithmetically.",
    }
