import json
import sqlite3
import threading
from datetime import datetime, timezone

from .config import DATA

LOCK = threading.RLock()


def connect():
    conn = sqlite3.connect(DATA / 'library.sqlite', timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init():
    with connect() as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')


def get(vid):
    with connect() as conn:
        row = conn.execute('SELECT payload FROM videos WHERE id=?', (vid,)).fetchone()
    return json.loads(row['payload']) if row else None


def all_videos():
    with connect() as conn:
        rows = conn.execute('SELECT payload FROM videos ORDER BY rowid DESC').fetchall()
    return [json.loads(row['payload']) for row in rows]


def update(vid, **changes):
    with LOCK:
        payload = get(vid) or {'id': vid, 'created_at': datetime.now(timezone.utc).isoformat()}
        payload.update(changes)
        payload['updated_at'] = datetime.now(timezone.utc).isoformat()
        with connect() as conn:
            conn.execute('INSERT OR REPLACE INTO videos VALUES (?,?)', (vid, json.dumps(payload, ensure_ascii=False)))
    return payload


def delete(vid):
    with LOCK:
        with connect() as conn:
            conn.execute('DELETE FROM videos WHERE id=?', (vid,))

