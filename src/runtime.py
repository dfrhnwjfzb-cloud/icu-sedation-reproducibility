"""Project paths, input verification and read-only database access."""

import hashlib
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

from .settings import (
    CONFIGS,
    OFFICIAL_MANIFEST,
    OFFICIAL_SQL_FILES,
    ORIGINAL_COHORT_SQL,
    REQUIRED_DATABASE_COLUMNS,
    SQL_SOURCES,
)

CONFIG = {}
WORK = None
PACKAGE = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE
ASSETS = R1 = R2 = None


def configure(config, persist=True):
    global CONFIG, WORK, ASSETS, R1, R2
    CONFIG = config
    WORK = Path(config["workspace"]).resolve()
    if WORK == PACKAGE or WORK.is_relative_to(PACKAGE) or PACKAGE.is_relative_to(WORK):
        raise ValueError("Workspace and source directory must be separate")
    for key in [
        "frozen_inputs",
        "reference_dir",
        "primary_inputs",
        "round1_cache",
        "round2_cache",
    ]:
        source = configured(key, False)
        if source:
            protected = (
                [source]
                if key in ["frozen_inputs", "reference_dir"]
                else [source / "data", source / "inputs", source / "outputs"]
            )
            if source == WORK or any(
                p.is_relative_to(WORK) or WORK.is_relative_to(p) for p in protected
            ):
                raise ValueError("Input and output paths overlap: " + key)
    ASSETS, R1, R2 = WORK / "supporting_inputs", WORK / "round1", WORK / "round2"
    for key in ["TMP", "TEMP", "TMPDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"]:
        os.environ[key] = str(WORK / "tmp")
    for path in [
        WORK,
        WORK / "tmp",
        WORK / "primary",
        R1,
        R2,
        WORK / "logs",
        WORK / "validation",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    os.environ["RESEARCH_PIPELINE_CONFIG"] = str(WORK / "run_config.json")
    if persist:
        (WORK / "run_config.json").write_text(
            json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def stage_models(root):
    """Create the legacy analysis output directory."""
    (root / "analysis").mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configured(key, required=True):
    value = CONFIG.get(key)
    if not value:
        if required:
            raise ValueError("Missing external input setting: " + key)
        return None
    return Path(value).expanduser().resolve()


def copy_immutable(source, target):
    source = Path(source)
    target = Path(target)
    if not source.is_file():
        raise FileNotFoundError("Missing external input: " + source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target):
            raise RuntimeError(
                "Different existing input; choose a fresh workspace: " + target.name
            )
    else:
        shutil.copy2(source, target)


def cohort(root):
    import pandas as pd

    import src.design as module_design

    design = module_design
    d = pd.read_csv(root / "inputs/mimic_design_candidates.csv", dtype={"stay_id": str})
    d = design.add_derived(
        d[d.treatment.isin(["DEX", "MIDAZ"])]
        .drop(
            columns=[
                "seizure",
                "withdrawal",
                "neurocritical",
                "delirium_proxy",
                "respiratory",
                "postoperative",
            ],
            errors="ignore",
        )
        .copy()
    )
    o = pd.read_csv(root / "inputs/mimic_outcomes.csv", dtype={"stay_id": str})
    d = d.merge(o, on="stay_id", suffixes=("", "_out"))
    x = d[(d.prop_active_t0 == 1) & (d.invasive_mv_t0 == 1)].copy()
    assert (
        len(x) == 3669
        and x.stay_id.is_unique
        and ((x.treatment == "DEX").sum() == 2778)
    )
    return x


def query(sql, name, root):
    import pandas as pd

    if not sql.lstrip().upper().startswith(("SELECT", "WITH")):
        raise ValueError("Read-only SELECT/CTE required")
    (root / "sql").mkdir(exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    path = root / "sql" / name
    path.write_text(sql, encoding="utf-8")
    db = CONFIG.get("postgres", {})
    if "password" in db:
        raise ValueError(
            "Use PostgreSQL credential facilities, not a password in configuration"
        )
    cmd = [
        db.get("psql", "psql"),
        "-h",
        str(db.get("host", "localhost")),
        "-p",
        str(db.get("port", 5432)),
        "-U",
        str(db.get("user", "postgres")),
        "-d",
        str(db.get("database", "mimiciv")),
        "-w",
        "-X",
        "-q",
        "--csv",
        "-P",
        "pager=off",
        "-v",
        "ON_ERROR_STOP=1",
        "-f",
        str(path),
    ]
    env = os.environ.copy()
    env["PGOPTIONS"] = "-c default_transaction_read_only=on"
    q = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=2400,
        env=env,
    )
    (root / "logs" / (name + ".log")).write_text(q.stderr, encoding="utf-8")
    if q.returncode:
        raise RuntimeError("Database error; inspect external log " + name + ".log")
    return pd.read_csv(io.StringIO(q.stdout))


def support():
    assets = ASSETS
    for name in ["benchmarks", "config", "sources/official", "docs"]:
        (assets / name).mkdir(parents=True, exist_ok=True)
    reference = configured("reference_dir")
    if (reference / "benchmarks").is_dir():
        reference = reference / "benchmarks"
    for name in [
        "round1_model_comparison.csv",
        "round1_subgroups.csv",
        "round1_icu_restriction.csv",
        "round1_urine_sensitivity.csv",
        "round1_original_balance.csv",
        "round2_model_comparison.csv",
        "round2_bootstrap.csv",
        "round1_expected_effects.csv",
        "round1_expected_balance.csv",
        "round1_expected_diagnostics.json",
    ]:
        copy_immutable(reference / name, assets / "benchmarks" / name)
    for name, value in CONFIGS.items():
        (assets / "config" / name).write_text(
            json.dumps(value, indent=2), encoding="utf-8"
        )
    for name, key in OFFICIAL_SQL_FILES.items():
        (assets / "sources/official" / name).write_text(
            SQL_SOURCES[key], encoding="utf-8"
        )
    (assets / "sources/official/manifest.json").write_text(
        json.dumps(OFFICIAL_MANIFEST, indent=2), encoding="utf-8"
    )
    (assets / "sources/frozen_design_cohort.sql").write_text(
        SQL_SOURCES[ORIGINAL_COHORT_SQL], encoding="utf-8"
    )
    (assets / "docs/REQUIRED_DATABASE_COLUMNS.json").write_text(
        json.dumps(REQUIRED_DATABASE_COLUMNS), encoding="utf-8"
    )


def prepare_round2():
    root = R2
    for name in [
        "analysis",
        "data",
        "outputs",
        "sql",
        "logs",
        "bootstrap",
        "tests",
        "sources",
    ]:
        (root / name).mkdir(exist_ok=True)
    for source, target in [
        ("round2.json", "config.json"),
        ("urine_round2.json", "urine_policy.json"),
        ("availability_round2.json", "availability_policy.json"),
    ]:
        shutil.copy2(ASSETS / "config" / source, root / target)
    cache = configured("round2_cache", False)
    if cache:
        for name in [
            "raw_labs.csv",
            "raw_charts.csv",
            "raw_vaso.csv",
            "raw_urine.csv",
            "derived_urine.csv",
            "urine_rate_history.csv",
        ]:
            if not (root / "data" / name).exists():
                copy_immutable(cache / "data" / name, root / "data" / name)


def prepare_primary():
    source = configured("primary_inputs")
    target = WORK / "primary"
    manifest = source / "SOURCE_MANIFEST.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            "Primary reproduction requires SOURCE_MANIFEST.json and its authorized input/reference files"
        )
    entries = json.loads(manifest.read_text(encoding="utf-8-sig"))
    for item in entries:
        relative = Path(item["package_path"])
        origin = (source / relative).resolve()
        destination = (target / relative).resolve()
        if not origin.is_relative_to(source) or not destination.is_relative_to(target):
            raise ValueError("Manifest path escapes input/output root")
        if sha(origin) != item["sha256"]:
            raise ValueError("Input hash mismatch: " + relative.as_posix())
        copy_immutable(origin, destination)
    copy_immutable(manifest, target / "SOURCE_MANIFEST.json")


if os.environ.get("RESEARCH_PIPELINE_CONFIG"):
    configure(
        json.loads(
            Path(os.environ["RESEARCH_PIPELINE_CONFIG"]).read_text(encoding="utf-8")
        ),
        persist=False,
    )
