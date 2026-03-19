import sqlite3
import cv2
import os
from datetime import datetime

DB_PATH        = 'violations.db'
SNAPSHOT_DIR   = 'violations'


class ViolationLogger:
    def __init__(self):
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                type      TEXT    NOT NULL,
                plate     TEXT,
                timestamp TEXT    NOT NULL,
                snapshot  TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log(self, violation_type, plate_text, frame=None):
        """Log a violation to the database and save snapshot."""
        ts       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot = None

        # Save snapshot
        if frame is not None:
            fname    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            snapshot = os.path.join(SNAPSHOT_DIR, f"{violation_type.replace(' ','_')}_{fname}.jpg")
            cv2.imwrite(snapshot, frame)

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT INTO violations (type, plate, timestamp, snapshot) VALUES (?, ?, ?, ?)',
            (violation_type, plate_text or 'Unknown', ts, snapshot)
        )
        conn.commit()
        conn.close()
        print(f"[ViolationLogger] {violation_type} | Plate: {plate_text} | {ts}")

    def get_recent(self, limit=20):
        """Fetch most recent violations."""
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            'SELECT id, type, plate, timestamp FROM violations ORDER BY id DESC LIMIT ?',
            (limit,)
        ).fetchall()
        conn.close()
        return [
            {"id": r[0], "type": r[1], "plate": r[2], "timestamp": r[3]}
            for r in rows
        ]

    def get_counts(self):
        """Get violation counts by type."""
        conn   = sqlite3.connect(DB_PATH)
        rows   = conn.execute(
            'SELECT type, COUNT(*) FROM violations GROUP BY type'
        ).fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}
