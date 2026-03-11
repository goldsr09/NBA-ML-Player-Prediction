#!/usr/bin/env python3
"""Fetch GameRotation and BoxScoreMatchupsV3 caches from stats.nba.com.

This hydrates the local cache consumed by `predict_player_props.py` for:
  - GameRotation: stint structure / rotation patterns
  - BoxScoreMatchupsV3: offensive matchup load by player
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import requests

from analyze_nba_2025_26_advanced import BOXSCORE_CACHE

PROJECT_ROOT = Path(os.environ.get("NBA_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
PROP_CACHE_DIR = PROJECT_ROOT / "analysis" / "output" / "prop_cache"
ROTATION_CACHE_DIR = PROP_CACHE_DIR / "game_rotation_raw"
MATCHUPS_CACHE_DIR = PROP_CACHE_DIR / "boxscore_matchups_raw"
STATS_NBA_BASE = "https://stats.nba.com/stats"
REQUEST_SLEEP = 1.0
MAX_RETRIES = 6
TIMEOUT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
}

_original_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: (
    _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
)


def _game_ids() -> list[str]:
    if not BOXSCORE_CACHE.is_dir():
        return []
    ids = sorted({f.stem for f in BOXSCORE_CACHE.glob("*.json") if f.stem})
    return ids


def _fetch(session: requests.Session, endpoint: str, game_id: str) -> dict | None:
    url = f"{STATS_NBA_BASE}/{endpoint}"
    params = {"GameID": game_id}
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, params=params, timeout=TIMEOUT)
            if resp.status_code == 200 and len(resp.content) > 50:
                return resp.json()
            if resp.status_code == 500:
                return None
        except Exception as exc:
            last_err = exc
        time.sleep(min(8.0, 1.2 * attempt))
    print(f"  Warning: {endpoint} failed for {game_id}: {last_err}", flush=True)
    return None


def _hydrate_endpoint(
    session: requests.Session,
    endpoint: str,
    cache_dir: Path,
    game_ids: list[str],
) -> tuple[int, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    for i, gid in enumerate(game_ids, start=1):
        path = cache_dir / f"{gid}.json"
        if path.exists():
            skipped += 1
            continue
        payload = _fetch(session, endpoint, gid)
        if payload is None:
            continue
        path.write_text(json.dumps(payload))
        saved += 1
        if i < len(game_ids):
            time.sleep(REQUEST_SLEEP)
        if i % 100 == 0:
            print(f"  {endpoint}: {i}/{len(game_ids)} processed | saved={saved} skipped={skipped}", flush=True)
    return saved, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GameRotation + BoxScoreMatchupsV3 caches")
    parser.add_argument("--max-games", type=int, default=0, help="Limit number of missing games fetched")
    args = parser.parse_args()

    game_ids = _game_ids()
    if not game_ids:
        print("No boxscore game ids found.", flush=True)
        return
    if args.max_games > 0:
        game_ids = game_ids[:args.max_games]

    session = requests.Session()

    print(f"Hydrating GameRotation cache for {len(game_ids)} games...", flush=True)
    rot_saved, rot_skipped = _hydrate_endpoint(session, "gamerotation", ROTATION_CACHE_DIR, game_ids)
    print(f"  GameRotation saved={rot_saved} skipped_existing={rot_skipped}", flush=True)

    print(f"Hydrating BoxScoreMatchupsV3 cache for {len(game_ids)} games...", flush=True)
    mt_saved, mt_skipped = _hydrate_endpoint(session, "boxscorematchupsv3", MATCHUPS_CACHE_DIR, game_ids)
    print(f"  BoxScoreMatchupsV3 saved={mt_saved} skipped_existing={mt_skipped}", flush=True)


if __name__ == "__main__":
    main()
