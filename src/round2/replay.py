import json
import logging

import src.round2.bootstrap as bootstrap_module
import src.round2.common as p

logger = logging.getLogger(__name__)


R = p.R


def main():
    mod = bootstrap_module
    mod.init()
    rows, logs, seconds = mod.task(10)
    old = json.loads((R / "bootstrap/formal_rep_0010.json").read_text())
    lookup = {(q["group"], q["model"], q["outcome"]): q for q in old["rows"]}
    diff = []
    for q in rows:
        prior = lookup[q["group"], q["model"], q["outcome"]]
        for col in ["RD", "RR", "DEX_risk", "MIDAZ_risk", "ESS_DEX", "ESS_MIDAZ"]:
            diff.append(abs(q[col] - prior[col]))
    result = {
        "replayed_replicate": 10,
        "pipeline_models": len(logs),
        "model_outcome_rows": len(rows),
        "max_absolute_numeric_difference": max(diff),
        "all_fits_successful": all(q["success"] for q in logs),
    }
    assert (
        result["all_fits_successful"]
        and result["max_absolute_numeric_difference"] < 1e-09
    )
    (R / "outputs/bootstrap_replay_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    logger.info(
        "Bootstrap replicate %d reproduced across %d models",
        result["replayed_replicate"],
        result["pipeline_models"],
    )
