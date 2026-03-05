#!/usr/bin/env python3
"""
Fetch boxscoredefensivev2 and boxscorescoringv3 from stats.nba.com.

These endpoints provide data NOT available from BDL:
  - Defensive: matchup FGA/FGM/3PA/3PM allowed, switchesOn, partialPossessions
  - Scoring: % assisted/unassisted 2pt/3pt, % midrange, shot creation context

stats.nba.com is extremely flaky (~50% timeout rate), so this uses:
  - Forced IPv4 (IPv6 always times out)
  - Aggressive retries with exponential backoff
  - Incremental per-game caching (resume-safe)
  - Connection keep-alive via session

Usage:
    python scripts/fetch_nba_defensive_scoring.py [--max-games N] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(os.environ.get("NBA_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
OUT_DIR = _PROJECT_ROOT / "analysis" / "output"
PROP_CACHE_DIR = OUT_DIR / "prop_cache"
TRACKING_CACHE_DIR = PROP_CACHE_DIR / "player_tracking_raw"
DEFENSIVE_CACHE_DIR = PROP_CACHE_DIR / "boxscore_defensive_raw"
SCORING_CACHE_DIR = PROP_CACHE_DIR / "boxscore_scoring_raw"

# ---------------------------------------------------------------------------
# stats.nba.com config
# ---------------------------------------------------------------------------
STATS_NBA_BASE = "https://stats.nba.com/stats"
MAX_RETRIES = 8
BASE_TIMEOUT = 30  # seconds
RETRY_BACKOFF = 3  # seconds between retries (multiplied by attempt)
REQUEST_SLEEP = 1.0  # seconds between successful requests

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


# ---------------------------------------------------------------------------
# Force IPv4 (stats.nba.com IPv6 always times out)
# ---------------------------------------------------------------------------
_original_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: (
    _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------
def _fetch_endpoint(
    session: requests.Session,
    endpoint: str,
    game_id: str,
) -> dict | None:
    """Fetch a single endpoint for a game with retry logic."""
    url = f"{STATS_NBA_BASE}/{endpoint}"
    params = {"GameID": game_id}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, headers=HEADERS, params=params, timeout=BASE_TIMEOUT)
            if r.status_code == 200 and len(r.content) > 50:
                return r.json()
            elif r.status_code == 500:
                return None
            elif r.status_code == 429:
                wait = RETRY_BACKOFF * attempt * 2
                print(f"    Rate limited, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
        except requests.exceptions.Timeout:
            if attempt >= 3:
                print(f"    {game_id} timeout (attempt {attempt}/{MAX_RETRIES})", flush=True)
        except requests.exceptions.ConnectionError:
            if attempt >= 3:
                print(f"    {game_id} connection error (attempt {attempt})", flush=True)
        except Exception as exc:
            print(f"    {game_id} error: {exc}", flush=True)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF * min(attempt, 4))

    return None


def _parse_defensive_v2(data: dict, game_id: str) -> list[dict[str, Any]]:
    """Parse boxscoredefensivev2 response into flat rows."""
    rows = []
    root = None
    for key in ("boxScoreDefensive", "boxscoredefensivev2"):
        if key in data:
            root = data[key]
            break
    if root is None:
        root = data

    for side_key in ("homeTeam", "awayTeam"):
        side = root.get(side_key)
        if not isinstance(side, dict):
            continue
        team = str(
            side.get("teamTricode") or side.get("teamCode") or side.get("teamAbbreviation") or ""
        ).upper().strip()

        for p in side.get("players", []) or []:
            if not isinstance(p, dict):
                continue
            pid = p.get("personId") or p.get("playerId")
            if pid is None:
                continue
            stats = p.get("statistics") or {}
            rows.append({
                "game_id": str(game_id),
                "team": team,
                "player_id": int(pid),
                "def_matchup_fga": _to_float(stats.get("matchupFieldGoalsAttempted")),
                "def_matchup_fgm": _to_float(stats.get("matchupFieldGoalsMade")),
                "def_matchup_fg_pct": _to_float(stats.get("matchupFieldGoalPercentage")),
                "def_matchup_3pa": _to_float(stats.get("matchupThreePointersAttempted")),
                "def_matchup_3pm": _to_float(stats.get("matchupThreePointersMade")),
                "def_matchup_3pt_pct": _to_float(stats.get("matchupThreePointerPercentage")),
                "def_matchup_minutes": _parse_minutes(stats.get("matchupMinutes")),
                "def_matchup_assists": _to_float(stats.get("matchupAssists")),
                "def_matchup_tov": _to_float(stats.get("matchupTurnovers")),
                "def_matchup_player_pts": _to_float(stats.get("playerPoints")),
                "def_steals": _to_float(stats.get("steals")),
                "def_blocks": _to_float(stats.get("blocks")),
                "def_switches_on": _to_float(stats.get("switchesOn")),
                "def_partial_poss": _to_float(stats.get("partialPossessions")),
                "def_reb_defensive": _to_float(stats.get("defensiveRebounds")),
            })
    return rows


def _parse_scoring_v3(data: dict, game_id: str) -> list[dict[str, Any]]:
    """Parse boxscorescoringv3 response into flat rows."""
    rows = []
    root = None
    for key in ("boxScoreScoring", "boxscorescoringv3"):
        if key in data:
            root = data[key]
            break
    if root is None:
        root = data

    for side_key in ("homeTeam", "awayTeam"):
        side = root.get(side_key)
        if not isinstance(side, dict):
            continue
        team = str(
            side.get("teamTricode") or side.get("teamCode") or side.get("teamAbbreviation") or ""
        ).upper().strip()

        for p in side.get("players", []) or []:
            if not isinstance(p, dict):
                continue
            pid = p.get("personId") or p.get("playerId")
            if pid is None:
                continue
            stats = p.get("statistics") or {}
            rows.append({
                "game_id": str(game_id),
                "team": team,
                "player_id": int(pid),
                "scr_pct_assisted_2pt": _to_float(stats.get("percentageAssisted2pt")),
                "scr_pct_assisted_3pt": _to_float(stats.get("percentageAssisted3pt")),
                "scr_pct_assisted_fgm": _to_float(stats.get("percentageAssistedFGM")),
                "scr_pct_unassisted_2pt": _to_float(stats.get("percentageUnassisted2pt")),
                "scr_pct_unassisted_3pt": _to_float(stats.get("percentageUnassisted3pt")),
                "scr_pct_unassisted_fgm": _to_float(stats.get("percentageUnassistedFGM")),
                "scr_pct_fga_2pt": _to_float(stats.get("percentageFieldGoalsAttempted2pt")),
                "scr_pct_fga_3pt": _to_float(stats.get("percentageFieldGoalsAttempted3pt")),
                "scr_pct_pts_2pt": _to_float(stats.get("percentagePoints2pt")),
                "scr_pct_pts_3pt": _to_float(stats.get("percentagePoints3pt")),
                "scr_pct_pts_ft": _to_float(stats.get("percentagePointsFreeThrow")),
                "scr_pct_pts_paint": _to_float(stats.get("percentagePointsPaint")),
                "scr_pct_pts_midrange": _to_float(stats.get("percentagePointsMidrange2pt")),
                "scr_pct_pts_fastbreak": _to_float(stats.get("percentagePointsFastBreak")),
                "scr_pct_pts_off_to": _to_float(stats.get("percentagePointsOffTurnovers")),
            })
    return rows


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _parse_minutes(v: Any) -> float | None:
    """Parse 'MM:SS' or 'PTXXMYY.ZZS' minutes string to float minutes."""
    if v is None:
        return None
    s = str(v).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return None
    if s.startswith("PT") and s.endswith("S"):
        # ISO 8601 duration: PT31M36.00S
        s = s[2:-1]  # strip PT and S
        mins = 0.0
        if "M" in s:
            m_part, s = s.split("M")
            mins = float(m_part)
        if s:
            mins += float(s) / 60.0
        return mins
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def get_all_game_ids() -> list[str]:
    """Get all game IDs from the tracking cache (games we have BDL data for)."""
    files = sorted(TRACKING_CACHE_DIR.glob("*.json"))
    return [f.stem for f in files]


def fetch_all(max_games: int = 0, endpoint_filter: str = "all"):
    """Fetch defensive and scoring data for all games."""
    session = requests.Session()

    game_ids = get_all_game_ids()
    if not game_ids:
        print("No game IDs found in tracking cache. Run fetch_bdl_tracking.py first.")
        return

    DEFENSIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SCORING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which games need fetching
    endpoints = []
    if endpoint_filter in ("all", "defensive"):
        def_missing = [gid for gid in game_ids if not (DEFENSIVE_CACHE_DIR / f"{gid}.json").exists()]
        endpoints.append(("boxscoredefensivev2", DEFENSIVE_CACHE_DIR, _parse_defensive_v2, def_missing))
        print(f"Defensive: {len(def_missing)} games to fetch ({len(game_ids) - len(def_missing)} cached)")
    if endpoint_filter in ("all", "scoring"):
        scr_missing = [gid for gid in game_ids if not (SCORING_CACHE_DIR / f"{gid}.json").exists()]
        endpoints.append(("boxscorescoringv3", SCORING_CACHE_DIR, _parse_scoring_v3, scr_missing))
        print(f"Scoring: {len(scr_missing)} games to fetch ({len(game_ids) - len(scr_missing)} cached)")

    for ep_name, cache_dir, parser, missing_ids in endpoints:
        if max_games > 0:
            missing_ids = missing_ids[:max_games]

        print(f"\n--- Fetching {ep_name} for {len(missing_ids)} games ---")
        success = 0
        fail = 0
        skip = 0

        for i, gid in enumerate(missing_ids, 1):
            cache_path = cache_dir / f"{gid}.json"
            if cache_path.exists():
                skip += 1
                continue

            data = _fetch_endpoint(session, ep_name, gid)
            if data is not None:
                # Validate parse works before saving
                parsed = parser(data, gid)
                if parsed:
                    cache_path.write_text(json.dumps(data, separators=(",", ":")))
                    success += 1
                else:
                    # Save empty marker so we don't retry
                    cache_path.write_text(json.dumps({"empty": True}))
                    skip += 1
            else:
                fail += 1

            if i % 50 == 0 or i == len(missing_ids):
                print(
                    f"  [{i}/{len(missing_ids)}] "
                    f"ok={success} fail={fail} skip={skip}",
                    flush=True,
                )

            # Rate limit
            if data is not None:
                time.sleep(REQUEST_SLEEP)

        print(f"  Done: {success} fetched, {fail} failed, {skip} skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch defensive + scoring boxscores from stats.nba.com")
    parser.add_argument("--max-games", type=int, default=0, help="Max games to fetch per endpoint (0=all)")
    parser.add_argument("--endpoint", choices=["all", "defensive", "scoring"], default="all",
                        help="Which endpoint to fetch")
    args = parser.parse_args()
    fetch_all(max_games=args.max_games, endpoint_filter=args.endpoint)
