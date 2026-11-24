"""Fixed analysis settings and SQL sources."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads(
    (PROJECT_ROOT / "config/analysis_settings.json").read_text(encoding="utf-8")
)
CONFIGS = SETTINGS["CONFIGS"]
OFFICIAL_SQL_FILES = SETTINGS["OFFICIAL_SQL_FILES"]
OFFICIAL_MANIFEST = SETTINGS["OFFICIAL_MANIFEST"]
REQUIRED_DATABASE_COLUMNS = SETTINGS["REQUIRED_DATABASE_COLUMNS"]
ORIGINAL_COHORT_SQL = SETTINGS["ORIGINAL_COHORT_SQL"]
LEGACY_FILENAMES = SETTINGS["LEGACY_FILENAMES"]
SQL_SOURCES = {
    p.name: p.read_text(encoding="utf-8")
    for p in sorted((PROJECT_ROOT / "sql").glob("*.sql"))
}
