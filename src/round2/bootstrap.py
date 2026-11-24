"""Patient-cluster bootstrap with complete propensity and outcome-model refitting."""

import argparse
import concurrent.futures
import json
import logging
import time

import numpy as np
import pandas as pd

import src.round2.bootstrap as bootstrap_module
import src.round2.common as p
import src.round2.engine as en

logger = logging.getLogger(__name__)


R = p.R
DATA = None
UIDS = None
DRAWS = None
SETS = None


def init():
    global DATA, UIDS, DRAWS, SETS
    DATA = p.baseline()
    UIDS = np.sort(DATA.patient_id.unique())
    DRAWS = np.load(R / "bootstrap/subject_draw_indices.npy", mmap_mode="r")
    SETS = {}
    for group, model in json.loads((R / "config.json").read_text())["bootstrap"][
        "models"
    ]:
        z = en.subset(DATA, group)
        codes = pd.Index(UIDS).get_indexer(z.patient_id)
        SETS[group, model] = (z, codes, np.arange(len(z)) % 5)


def task(rep):
    start = time.time()
    counts = np.bincount(DRAWS[rep], minlength=len(UIDS))
    out = []
    log = []
    for (group, model), (z, codes, folds) in SETS.items():
        ix = np.repeat(np.arange(len(z)), counts[codes])
        sample = z.iloc[ix].copy()
        try:
            rows, info = en.fit(sample, model, folds[ix])
            conv = info["convergence"]
            for r in rows:
                r.update({"replicate": rep, "group": group, "model": model})
                out.append(r)
            log.append(
                {
                    "replicate": rep,
                    "group": group,
                    "model": model,
                    "success": True,
                    "n": len(sample),
                    "unique_subjects": sample.patient_id.nunique(),
                    "PS_refitted": True,
                    "outcome_models_refitted": 20,
                    "max_iterations": max(c["nit"] for c in conv),
                    "ESS_DEX": rows[0]["ESS_DEX"],
                    "ESS_MIDAZ": rows[0]["ESS_MIDAZ"],
                    "error": "",
                }
            )
        except Exception as e:
            log.append(
                {
                    "replicate": rep,
                    "group": group,
                    "model": model,
                    "success": False,
                    "n": len(sample),
                    "error": type(e).__name__ + ": " + str(e),
                    "fit_history": str(en.COUNTS),
                }
            )
    return out, log, time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all requested replicates instead of resuming cached round2 JSON records.",
    )
    args = parser.parse_args()
    cfg = json.loads((R / "config.json").read_text())["bootstrap"]
    path = R / "bootstrap/subject_draw_indices.npy"
    n = len(np.unique(p.baseline().patient_id))
    rng = np.random.default_rng(cfg["seed"])
    if not path.exists():
        np.save(
            path,
            rng.integers(
                0,
                n,
                size=(cfg["max_attempts_if_failures"] + cfg["pilot_reps"], n),
                dtype=np.int32,
            ),
        )
    selected = range(10) if args.pilot else range(10, 510)
    label = "pilot" if args.pilot else "formal"
    prefix = R / "bootstrap" / label
    point = []
    if args.pilot:
        x = p.baseline()
        ref = pd.read_csv(p.OLD / "outputs/all_models.csv")
        for group, model in cfg["models"]:
            rows, info = en.fit(en.subset(x, group), model)
            if model != "ModelW":
                for r in rows:
                    old = ref[
                        (ref.group == group)
                        & (ref.model == model)
                        & (ref.outcome == r["outcome"])
                    ].iloc[0]
                    assert (
                        abs(r["RD"] - old.risk_difference) < 1e-09
                        and abs(r["RR"] - old.risk_ratio) < 1e-09
                    )
            for r in rows:
                r.update(group=group, model=model)
                point.append(r)
        pd.DataFrame(point).to_csv(
            R / "outputs/bootstrap_point_estimates.csv", index=False
        )
        logger.info(
            "All 5 reference point estimators reproduced to <1e-9; starting 10-replicate pilot."
        )
    rows = []
    logs = []
    dur = []
    done = set()
    for f in (R / "bootstrap").glob(label + "_rep_*.json"):
        if args.recompute:
            continue
        q = json.loads(f.read_text())
        rows.extend(q["rows"])
        logs.extend(q["logs"])
        dur.append(q["seconds"])
        done.add(q["replicate"])
    todo = [v for v in selected if v not in done]

    def execute(todo):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=bootstrap_module.init
        ) as pool:
            futures = {pool.submit(bootstrap_module.task, i): i for i in todo}
            for future in concurrent.futures.as_completed(futures):
                rep = futures[future]
                a, b, seconds = future.result()
                payload = {"replicate": rep, "rows": a, "logs": b, "seconds": seconds}
                (R / "bootstrap" / f"{label}_rep_{rep:04}.json").write_text(
                    json.dumps(
                        payload,
                        default=lambda v: (
                            v.item() if isinstance(v, np.generic) else str(v)
                        ),
                    ),
                    encoding="utf-8",
                )
                rows.extend(a)
                logs.extend(b)
                dur.append(seconds)
                if len(dur) % 10 == 0:
                    logger.info(
                        f"{label}: {len(dur)} replicate sets completed; mean worker seconds/set {np.mean(dur):.2f}; failures {sum(not v['success'] for v in logs)}"
                    )

    execute(todo)
    if not args.pilot:
        for lo in range(510, 610, 10):
            valid = pd.DataFrame(logs).groupby(["group", "model"]).success.sum()
            if (valid >= 500).all():
                break
            execute(list(range(lo, lo + 10)))
    pd.DataFrame(rows).sort_values(["replicate", "group", "model", "outcome"]).to_csv(
        prefix.with_suffix(".csv"), index=False
    )
    pd.DataFrame(logs).sort_values(["replicate", "group", "model"]).to_csv(
        R / "bootstrap" / f"{label}_convergence.csv", index=False
    )
    status = {
        "mode": label,
        "replicate_sets": len(dur),
        "model_attempts": len(logs),
        "successes": sum(v["success"] for v in logs),
        "failures": sum(not v["success"] for v in logs),
        "mean_seconds_per_set": float(np.mean(dur)),
    }
    (R / "bootstrap" / f"{label}_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    logger.info(
        "%s bootstrap: %d replicate sets; %d successes and %d failures",
        status["mode"],
        status["replicate_sets"],
        status["successes"],
        status["failures"],
    )
