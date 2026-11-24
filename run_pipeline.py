"""Reproduce the primary study and severity robustness analyses."""

import argparse
import ast
import io
import json
import logging
import os
import sys
import time
import unittest
from pathlib import Path

from src import runtime as rt
from src.settings import REQUIRED_DATABASE_COLUMNS, SQL_SOURCES

sys.dont_write_bytecode = True
for variable in [
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[variable] = "1"

CODE = Path(__file__).resolve()


def check():
    paths = [
        CODE,
        *CODE.parent.joinpath("src").rglob("*.py"),
        *CODE.parent.joinpath("tests").rglob("*.py"),
    ]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"))
    logging.info("Syntax checked: %d Python files", len(paths))


def run_step(section, args=()):
    previous = sys.argv
    sys.argv = [str(CODE), *args]
    try:
        logging.info("Running %s", section)
        if section == "legacy_pipeline":
            from src.primary.pipeline import main as execute

            return execute()
        if section == "db_check":
            from src.database import main as execute

            return execute()
        if section == "reference_validation":
            from src.validation import main as execute

            return execute()
        if section == "setup1":
            from src.round1.setup import prepare as execute

            return execute()
        if section == "verify_db":
            from src.round1.verify_database import main as execute

            return execute()
        if section == "extract1":
            from src.round1.extract import main as execute

            return execute()
        if section == "scores1":
            from src.round1.build_scores import main as execute

            return execute()
        if section == "details1":
            from src.round1.diagnostics import main as execute

            return execute()
        if section == "dose_audit":
            from src.round1.vasoactive_units import main as execute

            return execute()
        if section == "models1":
            from src.round1.models import main as execute

            return execute()
        if section == "urine1":
            from src.round1.urine import main as execute

            return execute()
        if section == "plots1":
            from src.round1.figures import main as execute

            return execute()
        if section == "extract2":
            from src.round2.extract import main as execute

            return execute()
        if section == "audit2":
            from src.round2.diagnostics import main as execute

            return execute()
        if section == "bootstrap":
            from src.round2.bootstrap import main as execute

            return execute()
        if section == "availability":
            from src.round2.availability import main as execute

            return execute()
        if section == "availability_effect":
            from src.round2.availability_models import main as execute

            return execute()
        if section == "bootstrap_summary":
            from src.round2.inference import main as execute

            return execute()
        if section == "replay":
            from src.round2.replay import main as execute

            return execute()
        raise ValueError("Unknown pipeline step: " + section)
    finally:
        sys.argv = previous


def run_tests(round_number):
    if round_number == 1:
        from tests import test_scores, test_temporality

        modules = [test_scores, test_temporality]
    else:
        from tests import test_round2

        modules = [test_round2]
    suite = unittest.TestSuite()
    for module in modules:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (rt.WORK / "logs" / f"round{round_number}_unit_tests.txt").write_text(
        stream.getvalue(), encoding="utf-8"
    )
    if not result.wasSuccessful():
        raise RuntimeError(stream.getvalue())
    return result.testsRun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list-sql", action="store_true")
    parser.add_argument("--show-sql")
    parser.add_argument("--export-sql", action="store_true")
    parser.add_argument(
        "--task",
        choices=[
            "all",
            "primary",
            "severity",
            "round1",
            "round2",
            "validate",
            "db-check",
            "bootstrap-pilot",
        ],
        default="all",
    )
    parser.add_argument("--config", type=Path)
    for key in [
        "workspace",
        "primary-inputs",
        "frozen-inputs",
        "round1-cache",
        "round2-cache",
        "reference-dir",
    ]:
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--psql")
    parser.add_argument("--db-host")
    parser.add_argument("--db-port", type=int)
    parser.add_argument("--db-user")
    parser.add_argument("--db-name")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if args.list_sql:
        print("\n".join(SQL_SOURCES))
        return
    if args.show_sql:
        if args.show_sql not in SQL_SOURCES:
            parser.error("Unknown SQL section; use --list-sql")
        print(SQL_SOURCES[args.show_sql])
        return
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
    base = args.config.resolve().parent if args.config else Path.cwd()
    for name in [
        "workspace",
        "primary_inputs",
        "frozen_inputs",
        "round1_cache",
        "round2_cache",
        "reference_dir",
    ]:
        value = getattr(args, name)
        if value is not None:
            cfg[name] = str(value.resolve())
        elif cfg.get(name):
            path = Path(cfg[name]).expanduser()
            cfg[name] = str((path if path.is_absolute() else base / path).resolve())
    if not cfg.get("workspace"):
        parser.error("--workspace outside the code directory is required")
    db = cfg.setdefault("postgres", {})
    for arg, key in [
        ("psql", "psql"),
        ("db_host", "host"),
        ("db_port", "port"),
        ("db_user", "user"),
        ("db_name", "database"),
    ]:
        if getattr(args, arg) is not None:
            db[key] = getattr(args, arg)
    if args.workers is not None:
        cfg["bootstrap_workers"] = args.workers
    if (
        args.task in ["all", "primary"]
        and not args.export_sql
        and not cfg.get("primary_inputs")
    ):
        parser.error("--primary-inputs is required for primary/all")
    if (
        args.task not in ["primary", "db-check"]
        and not args.export_sql
        and any(not cfg.get(k) for k in ["frozen_inputs", "reference_dir"])
    ):
        parser.error(
            "--frozen-inputs and --reference-dir are required for severity tasks"
        )
    rt.configure(cfg)
    if args.export_sql:
        target = rt.WORK / "sql_sources"
        target.mkdir(exist_ok=True)
        for name, sql in SQL_SOURCES.items():
            (target / name).write_text(sql.lstrip("\n"), encoding="utf-8")
        print("Exported readable SQL to external workspace")
        return
    if args.task == "db-check":
        assets = rt.ASSETS
        (assets / "docs").mkdir(parents=True, exist_ok=True)
        (assets / "docs/REQUIRED_DATABASE_COLUMNS.json").write_text(
            json.dumps(REQUIRED_DATABASE_COLUMNS), encoding="utf-8"
        )
        run_step("db_check")
        return
    started = time.time()
    if args.task in ["all", "primary"]:
        rt.prepare_primary()
        run_step("legacy_pipeline")
    if args.task == "primary":
        return
    rt.support()
    if args.task == "validate":
        run_step("reference_validation", ["--stage", "all"])
        return
    if args.task == "bootstrap-pilot":
        run_step(
            "bootstrap",
            [
                "--pilot",
                "--recompute",
                "--workers",
                str(cfg.get("bootstrap_workers", 4)),
            ],
        )
        return
    os.environ["SEVERITY_EXTRACT_FRESH"] = "1" if args.extract else "0"
    if args.extract and any((rt.R2 / "data").glob("raw_*.csv")):
        raise RuntimeError("Use a fresh workspace for --extract")
    if args.task in ["all", "severity", "round1"]:
        run_step("setup1")
        if args.extract:
            run_step("verify_db")
            run_step("extract1")
            run_step("dose_audit")
        else:
            required = [
                rt.R1 / "data" / name
                for name in [
                    "sofa_observations.csv",
                    "windows.csv",
                    "rebuilt_candidates.csv",
                ]
            ] + [rt.R1 / "outputs/historical_lab_tie_audit.csv"]
            if any(not path.is_file() for path in required):
                raise FileNotFoundError(
                    "Supply authorized round1 cache or use --extract"
                )
        run_step("scores1")
        run_step("details1", ["--refresh-db"] if args.extract else [])
        run_step("models1")
        run_step("urine1")
        run_tests(1)
        if args.figures:
            run_step("plots1")
    if args.task in ["all", "severity", "round2"]:
        if not (rt.R1 / "data/pre_t0_sofa.csv").is_file():
            raise FileNotFoundError("Run round1 in the same workspace first")
        rt.prepare_round2()
        if args.extract:
            run_step("extract2")
        elif any(
            not (rt.R2 / "data" / name).is_file()
            for name in [
                "raw_labs.csv",
                "raw_charts.csv",
                "raw_vaso.csv",
                "raw_urine.csv",
                "derived_urine.csv",
                "urine_rate_history.csv",
            ]
        ):
            raise FileNotFoundError(
                "Supply authorized round2 raw cache or use --extract"
            )
        run_step("audit2")
        workers = str(int(cfg.get("bootstrap_workers", 4)))
        run_step("bootstrap", ["--pilot", "--workers", workers])
        run_step("bootstrap", ["--workers", workers])
        run_step("availability")
        run_step("availability_effect")
        run_step("bootstrap_summary")
        run_step("replay")
        run_tests(2)
    stage = args.task if args.task in ["round1", "round2"] else "all"
    run_step("reference_validation", ["--stage", stage])
    logging.info("Completed %s in %.2f s", args.task, time.time() - started)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
