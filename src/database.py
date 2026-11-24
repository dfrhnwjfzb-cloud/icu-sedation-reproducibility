"""Inspect required source columns; never creates/updates database objects."""

import json
import logging

import src.runtime as rt

logger = logging.getLogger(__name__)


def main():
    required = json.loads(
        (rt.ASSETS / "docs/REQUIRED_DATABASE_COLUMNS.json").read_text()
    )
    schema = rt.query(
        "SELECT table_schema,table_name,column_name FROM information_schema.columns WHERE table_schema IN ('mimiciv_derived','mimiciv_icu','mimiciv_hosp');",
        "schema_preflight.sql",
        rt.R1,
    )
    found = set(zip(schema.table_schema + "." + schema.table_name, schema.column_name))
    missing = []
    for table, columns in required.items():
        for column in columns:
            if (table, column) not in found:
                missing.append({"table": table, "column": column})
    status = {
        "status": "PASS" if not missing else "FAIL",
        "required_tables": len(required),
        "required_columns": sum(len(v) for v in required.values()),
        "missing": missing,
        "scope": "Column/schema availability only; derived-table build commit and clinical validity are not certified",
    }
    (rt.WORK / "validation").mkdir(exist_ok=True)
    (rt.WORK / "validation/database_preflight.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    logger.info(
        "Database preflight: %s; %d tables and %d columns",
        status["status"],
        status["required_tables"],
        status["required_columns"],
    )
    if missing:
        raise RuntimeError(
            "Required MIMIC source columns unavailable; see private workspace validation/database_preflight.json"
        )
