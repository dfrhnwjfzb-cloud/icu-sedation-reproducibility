"""Severity distributions, model comparisons and overlap diagnostics."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.plot_layout as plot_layout_module
import src.round1.setup as p

mpl.use("Agg")

require_matplotlib_panel_alignment = (
    plot_layout_module.require_matplotlib_panel_alignment
)
R = p.R
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 600,
    }
)
C = {"DEX": "#267B96", "MIDAZ": "#B76A40"}


def export(fig, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(R / "figures" / f"{name}.alignment.json"),
        overlay_svg=str(R / "figures" / f"{name}.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(R / "figures" / f"{name}.svg")
    fig.savefig(R / "figures" / f"{name}.pdf")
    fig.savefig(R / "figures" / f"{name}.png", dpi=600)
    fig.savefig(R / "figures" / f"{name}.tiff", dpi=600)
    plt.close(fig)


def letters(axes):
    for label, ax in zip("abcdefgh", np.ravel(axes)):
        ax.annotate(
            label,
            (0, 1),
            xytext=(-14, 10),
            textcoords="offset points",
            xycoords="axes fraction",
            weight="bold",
            fontsize=8,
        )


def histograms():
    x = pd.read_csv(R / "data/analysis_scores_merged.csv")
    w = pd.read_csv(R / "inputs/original_weights.csv")
    assert x.stay_id.tolist() == w.stay_id.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.19, top=0.82, wspace=0.24)
    letters(axs)
    bins = np.arange(-0.5, 24.6, 1)
    for ax, weighted, title in zip(
        axs, [False, True], ["Before weighting", "Original overlap weights"]
    ):
        for tx in ["DEX", "MIDAZ"]:
            k = x.treatment == tx
            weights = (
                w.loc[k, "overlap_weight"].to_numpy() if weighted else np.ones(k.sum())
            )
            weights = weights / weights.sum()
            ax.hist(
                x.loc[k, "pre_t0_sofa"],
                bins=bins,
                weights=weights,
                histtype="step",
                linewidth=1.2,
                label=tx,
                color=C[tx],
            )
        ax.set(
            title=title,
            xlabel="Pre-t0 SOFA (missing organs = 0)",
            xlim=(-0.5, 24.5),
            xticks=np.arange(0, 25, 4),
        )
        ax.legend(loc="upper right")
    axs[0].set_ylabel("Proportion within treatment")
    export(fig, "severity_balance_figure")
    b = pd.read_csv(R / "outputs/severity_balance_original_weights.csv")
    b = b[
        b.variable.isin(
            [
                "pre_t0_sofa",
                "pre_t0_sofa_strict",
                "short_window",
                "respiration",
                "coagulation",
                "liver",
                "cardiovascular",
                "cns",
                "renal",
            ]
        )
    ].copy()
    labels = {
        "pre_t0_sofa": "SOFA: missing organs = 0",
        "pre_t0_sofa_strict": "SOFA: all six organs observed",
        "short_window": "History shorter than 24 h",
        "respiration": "Respiratory component",
        "coagulation": "Coagulation component",
        "liver": "Liver component",
        "cardiovascular": "Cardiovascular component",
        "cns": "CNS component",
        "renal": "Renal component",
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.19, top=0.84)
    y = np.arange(len(b))
    ax.scatter(
        b.raw_smd.abs(), y, color="#999999", marker="o", s=16, label="Unweighted"
    )
    ax.scatter(
        b.weighted_smd.abs(),
        y,
        color=C["DEX"],
        marker="s",
        s=16,
        label="Original weights",
    )
    ax.axvline(0.1, color="#999999", ls="--", lw=0.7)
    ax.set(
        yticks=y,
        yticklabels=[labels[v] for v in b.variable],
        xlabel="Absolute standardized mean difference",
        xlim=(0, 0.5),
    )
    ax.invert_yaxis()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncol=2)
    export(fig, "severity_component_balance")


def forest(table, name, rowspec):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=True)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.17, top=0.84, wspace=0.24)
    letters(axs)
    for ax, out, title in zip(
        axs,
        ["alive_sustainably_liberated_day28", "hospital_mortality"],
        ["Alive and sustainably liberated\nby day 28", "Hospital mortality"],
    ):
        for j, (group, model, label) in enumerate(rowspec):
            q = table[
                (table.group == group) & (table.model == model) & (table.outcome == out)
            ].iloc[0]
            v = q.risk_difference * 100
            lo = q.rd_ci_low * 100
            hi = q.rd_ci_high * 100
            color = C["DEX"] if q.max_abs_smd <= 0.1 else C["MIDAZ"]
            ax.errorbar(
                v,
                j,
                xerr=np.array([[v - lo], [hi - v]]),
                fmt="o",
                ms=4,
                color=color,
                lw=1,
                capsize=2,
            )
        ax.axvline(0, color="#999999", ls="--", lw=0.7)
        ax.set(
            title=title,
            xlabel="Risk difference (percentage points)\nDEX minus MIDAZ",
            xlim=(-32, 32),
            xticks=[-30, -15, 0, 15, 30],
            ylim=(len(rowspec) - 0.5, -0.7),
        )
    axs[0].set_yticks(np.arange(len(rowspec)), [r[2] for r in rowspec])
    export(fig, name)


def ps_plots():
    spec = [
        ("Full", "Model0", "Full cohort: original"),
        ("Full", "ModelS", "Full cohort: + SOFA"),
        ("Low", "ModelS", "SOFA 0-5"),
        ("Intermediate", "ModelS", "SOFA 6-9"),
        ("High", "ModelS", "SOFA 10-24"),
        ("NonCardiacCV", "ModelS", "Non-Cardiac/CV"),
        ("CardiacCV", "ModelS", "Cardiac/CV"),
        ("Full", "ModelG_rawGCS", "Raw-GCS sensitivity"),
    ]
    fig, axs = plt.subplots(4, 2, figsize=(7.2, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.09, top=0.95, hspace=0.68, wspace=0.25
    )
    letters(axs)
    for ax, (g, m, title) in zip(axs.flat, spec):
        z = pd.read_csv(R / "data" / f"weights_{g}_{m}.csv")
        for tx in ["DEX", "MIDAZ"]:
            k = z.treatment == tx
            ww = z.loc[k, "overlap_weight"].to_numpy()
            ww = ww / ww.sum()
            ax.hist(
                z.loc[k, "propensity"],
                bins=np.linspace(0, 1, 21),
                weights=ww,
                histtype="step",
                color=C[tx],
                label=tx,
                lw=1.0,
            )
        ax.set(title=title, xlim=(0, 1))
        ax.legend(loc="upper left", ncol=2)
    for ax in axs[-1]:
        ax.set_xlabel("Propensity score after within-set refitting")
    for ax in axs[:, 0]:
        ax.set_ylabel("Weighted proportion")
    export(fig, "propensity_overlap_diagnostics")


def main():
    histograms()
    t = pd.read_csv(R / "outputs/all_models.csv")
    forest(
        t,
        "model_comparison_forest",
        [
            ("Full", "Model0", "Original 84-term model"),
            ("Full", "ModelS", "SOFA-augmented model"),
            ("Full", "ModelG_rawGCS", "Raw-GCS sensitivity"),
        ],
    )
    forest(
        t,
        "subgroup_forest",
        [
            ("Full", "ModelS", "Full cohort (n = 3,669)"),
            ("Low", "ModelS", "SOFA 0-5 (n = 2,091)"),
            ("Intermediate", "ModelS", "SOFA 6-9 (n = 1,163)"),
            ("High", "ModelS", "SOFA 10-24 (n = 415)"),
            ("NonCardiacCV", "ModelS", "Non-Cardiac/CV (n = 1,830)"),
            ("CardiacCV", "ModelS", "Cardiac/CV (n = 1,839)"),
        ],
    )
    ps_plots()
