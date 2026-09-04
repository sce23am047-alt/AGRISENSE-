"""
src/history.py
Lightweight SQLite storage for past soil analyses, so the dashboard's
History page has real records to show instead of mock data.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    location TEXT,
    n REAL, p REAL, k REAL,
    temperature REAL, moisture REAL, ph REAL,
    soil_health_score INTEGER,
    soil_health_label TEXT,
    recommended_crop TEXT,
    confidence REAL,
    fertilizer_status TEXT,
    recommended_fertilizer TEXT
);
"""


def _connect():
    config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.execute(SCHEMA)
    return conn


def save_analysis(record: dict):
    conn = _connect()
    conn.execute(
        """INSERT INTO analyses
           (created_at, location, n, p, k, temperature, moisture, ph,
            soil_health_score, soil_health_label, recommended_crop, confidence,
            fertilizer_status, recommended_fertilizer)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            record.get("location", ""),
            record.get("N"), record.get("P"), record.get("K"),
            record.get("temperature"), record.get("moisture"), record.get("ph"),
            record.get("soil_health_score"), record.get("soil_health_label"),
            record.get("recommended_crop"), record.get("confidence"),
            record.get("fertilizer_status"), record.get("recommended_fertilizer"),
        ),
    )
    conn.commit()
    conn.close()


def get_history(limit: int = 100):
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def clear_history():
    conn = _connect()
    conn.execute("DELETE FROM analyses")
    conn.commit()
    conn.close()
