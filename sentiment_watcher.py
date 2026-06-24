#!/usr/bin/env python3
"""
sentiment_watcher.py — Watches trendyol.duckdb for changes and
                      auto-refreshes the Flask app's in-memory cache
                      so the dashboard always shows live data.

Usage:
    python3 sentiment_watcher.py              # foreground (Ctrl+C)
    python3 sentiment_watcher.py --daemon     # background process
    python3 sentiment_watcher.py --once       # just clear cache once
"""
from __future__ import annotations

import os
import sys
import time
import json
import signal
import logging
import urllib.request
import urllib.error
from pathlib import Path

DB_PATH = Path("/Users/macbook/Documents/codex/trendyol_research_mvp/data/trendyol.duckdb")
FLASK_REFRESH_URL = "http://127.0.0.1:5999/api/refresh"
FLASK_HEALTH_URL = "http://127.0.0.1:5999/api/health"
DEBOUNCE_SEC = 8
POLL_INTERVAL = 2.0

LOG_FILE = Path("/Users/macbook/sentiment_watcher.log")
PID_FILE = Path("/tmp/sentiment_watcher.pid")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("watcher")


def get_db_state() -> str:
    if not DB_PATH.exists():
        return ""
    s = DB_PATH.stat()
    return f"{s.st_size}_{s.st_mtime_ns}"


def ping_flask(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def wait_for_flask(timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(FLASK_HEALTH_URL, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ── Watcher loop ───────────────────────────────────────────────────────────

def watch():
    if not wait_for_flask():
        log.warning("⚠️  Flask server not reachable at %s — will retry on change", FLASK_HEALTH_URL)
    else:
        log.info("✅ Flask server reachable")

    last_state = get_db_state()

    # Initial refresh
    if last_state:
        log.info("📦 DB found (%d KB), clearing Flask cache", DB_PATH.stat().st_size // 1024)
        ping_flask(FLASK_REFRESH_URL)
        last_state = get_db_state()

    log.info("👀 Watching %s every %.1fs (debounce=%ds)", DB_PATH.name, POLL_INTERVAL, DEBOUNCE_SEC)

    dirty_time = 0.0
    notified = False

    while True:
        cur = get_db_state()

        if cur != last_state:
            if not notified:
                log.info("🔔 Change detected — waiting %ds for writes to settle", DEBOUNCE_SEC)
                dirty_time = time.time()
                notified = True
        elif notified and dirty_time > 0:
            if time.time() - dirty_time >= DEBOUNCE_SEC:
                log.info("🔔 File stable — clearing Flask cache")
                if ping_flask(FLASK_REFRESH_URL):
                    log.info("✅ Flask cache cleared, dashboard will re-query live data")
                else:
                    log.warning("⚠️  Flask not reachable, will retry on next change")
                last_state = cur
                dirty_time = 0
                notified = False

        time.sleep(POLL_INTERVAL)


# ── Modes ───────────────────────────────────────────────────────────────────

def daemonize():
    pid = os.fork()
    if pid > 0:
        PID_FILE.write_text(str(pid))
        print(f"✅ Watcher started (PID: {pid})")
        print(f"   Log: {LOG_FILE}")
        sys.exit(0)
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        sys.exit(0)
    sys.stdin.close()
    sys.stdout = open(LOG_FILE, "a")
    sys.stderr = open(LOG_FILE, "a")
    watch()


def run_once():
    log.info("🏃 One-shot: clearing Flask cache")
    if ping_flask(FLASK_REFRESH_URL):
        log.info("✅ Cache cleared")
    else:
        log.info("⚠️  Flask not running — start the server first with:")
        log.info("   cd /Users/macbook && python3 app.py")


if __name__ == "__main__":
    if "--once" in sys.argv:
        run_once()
        sys.exit(0)
    if "--daemon" in sys.argv:
        daemonize()
    try:
        watch()
    except KeyboardInterrupt:
        if PID_FILE.exists():
            PID_FILE.unlink()
        log.info("Stopped")
        print("\nWatcher stopped.")
