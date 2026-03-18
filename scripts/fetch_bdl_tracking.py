#!/usr/bin/env python3
"""
Fetch player tracking + hustle stats from BallDontLie V2 Advanced Stats API.

Populates the same tracking cache used by predict_player_props.py.
Maps BDL data to NBA game IDs and player IDs via boxscore cross-reference.

Uses date-based fetching (BDL's game_id filter is unreliable, but dates[] works).

Usage:
    python scripts/fetch_bdl_tracking.py [--seasons 2021 2022 2023 2024 2025] [--max-dates N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths — mirror predict_player_props.py conventions
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(os.environ.get("NBA_PROJECT_ROOT", Path(__file__).resolve().parent.parent))
OUT_DIR = _PROJECT_ROOT / "analysis" / "output"
PROP_CACHE_DIR = OUT_DIR / "prop_cache"
TRACKING_CACHE_DIR = PROP_CACHE_DIR / "player_tracking_raw"
HISTORICAL_CACHE = OUT_DIR / "historical_cache"
CURRENT_SEASON_CACHE = OUT_DIR / "nba_2025_26_advanced_cache" / "boxscores"

BDL_API_KEY = os.environ.get("BDL_API_KEY", "")
BDL_BASE = "https://api.balldontlie.io"
BDL_SLEEP = 0.12  # ~500 req/min to stay safe under 600/min limit

# BDL season number → our season string
BDL_SEASON_MAP = {
    2021: "2021-22",
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
    2025: "2025-26",
}


def _normalize(name: str) -> str:
    """Normalize player name for fuzzy matching."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return s.lower().strip().replace(".", "").replace("'", "").replace("-", " ").replace("  ", " ")


def _bdl_get(endpoint: str, params: dict | None = None) -> dict:
    """Make authenticated BDL API request with retry."""
    url = f"{BDL_BASE}{endpoint}"
    headers = {"Authorization": BDL_API_KEY}
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"  Rate limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt == 2:
                raise
            print(f"  Retry {attempt + 1}/3: {e}", flush=True)
            time.sleep(2 ** attempt)
    return {}


# Team abbreviation normalization (BDL may use slightly different codes)
_TEAM_ALIASES = {
    "PHX": "PHX", "PHO": "PHX",
    "BKN": "BKN", "BRK": "BKN",
    "CHA": "CHA", "CHO": "CHA",
    "GS": "GSW", "SA": "SAS",
    "NY": "NYK", "NO": "NOP",
    "NOP": "NOP", "NOH": "NOP",
    "WSH": "WAS",
    "UTH": "UTA",
}


def _norm_team(abbr: str) -> str:
    abbr = abbr.upper().strip()
    return _TEAM_ALIASES.get(abbr, abbr)


# ---------------------------------------------------------------------------
# Step 1: Build NBA game ID mapping from boxscore files
# ---------------------------------------------------------------------------
def _extract_game_date_et(game: dict) -> str:
    """Extract YYYY-MM-DD date in ET from a boxscore game dict."""
    et = game.get("gameEt") or game.get("gameTimeLocal") or game.get("gameTimeUTC") or ""
    return et[:10]


def build_nba_game_map() -> dict[str, dict]:
    """
    Scan all boxscore files → dict keyed by "YYYY-MM-DD|HOME|AWAY".
    Returns: { key: {"nba_game_id": str, "players": {(norm_name, team): nba_pid}} }
    """
    game_map: dict[str, dict] = {}
    boxscore_dirs = []

    # Historical seasons
    if HISTORICAL_CACHE.exists():
        for season_dir in sorted(HISTORICAL_CACHE.iterdir()):
            bs_dir = season_dir / "boxscores"
            if bs_dir.is_dir():
                boxscore_dirs.append(bs_dir)

    # Current season
    if CURRENT_SEASON_CACHE.is_dir():
        boxscore_dirs.append(CURRENT_SEASON_CACHE)

    total_files = 0
    for bs_dir in boxscore_dirs:
        for f in bs_dir.glob("*.json"):
            total_files += 1
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            game = data.get("game", data)
            nba_gid = str(game.get("gameId") or f.stem)
            date_str = _extract_game_date_et(game)
            if not date_str or len(date_str) != 10:
                continue

            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            home_tri = str(home.get("teamTricode", "")).upper()
            away_tri = str(away.get("teamTricode", "")).upper()
            if not home_tri or not away_tri:
                continue

            key = f"{date_str}|{home_tri}|{away_tri}"
            players: dict[tuple[str, str], int] = {}

            for side in (home, away):
                team_tri = str(side.get("teamTricode", "")).upper()
                for p in side.get("players", []):
                    pid = p.get("personId")
                    fname = p.get("firstName", "")
                    lname = p.get("familyName", "")
                    full_name = f"{fname} {lname}".strip()
                    if pid and full_name:
                        players[(_normalize(full_name), team_tri)] = int(pid)

            game_map[key] = {
                "nba_game_id": nba_gid,
                "players": players,
                "home": home_tri,
                "away": away_tri,
            }

    print(f"Built NBA game map: {len(game_map)} games from {total_files} boxscore files", flush=True)
    return game_map


# ---------------------------------------------------------------------------
# Step 2: Get unique game dates from BDL games endpoint
# ---------------------------------------------------------------------------
def fetch_bdl_game_dates(seasons: list[int]) -> dict[str, list[dict]]:
    """Fetch all BDL games for given seasons, grouped by date.
    Returns: {"YYYY-MM-DD": [game_dicts]}
    """
    dates: dict[str, list[dict]] = defaultdict(list)
    for season in seasons:
        print(f"Fetching BDL games for season {season} ({BDL_SEASON_MAP.get(season, '?')})...", flush=True)
        cursor = None
        page = 0
        total = 0
        while True:
            params: dict[str, Any] = {"seasons[]": season, "per_page": 100}
            if cursor:
                params["cursor"] = cursor
            data = _bdl_get("/v1/games", params)
            games = data.get("data", [])
            if not games:
                break
            for g in games:
                dates[g["date"]].append(g)
                total += 1
            page += 1
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            time.sleep(BDL_SLEEP)
        print(f"  Season {season}: {total} games across {page} pages", flush=True)
    print(f"Total: {sum(len(v) for v in dates.values())} games across {len(dates)} dates", flush=True)
    return dict(dates)


# ---------------------------------------------------------------------------
# BDL field → resultSets header mapping
# ---------------------------------------------------------------------------
_BDL_TRACKING_MAP = {
    "touches": "TOUCHES",
    "passes": "PASSES",
    "distance": "DIST_MILES",
    "speed": "AVG_SPEED",
    "contested_shots": "CONTESTED_SHOTS",
    "contested_shots_2pt": "CONTESTED_SHOTS_2PT",
    "contested_shots_3pt": "CONTESTED_SHOTS_3PT",
    "uncontested_fga": "UNCONTESTED_FGA",
    "uncontested_fgm": "UNCONTESTED_FGM",
    # Hustle stats
    "deflections": "DEFLECTIONS",
    "box_outs": "BOX_OUTS",
    "offensive_box_outs": "OFFENSIVE_BOX_OUTS",
    "defensive_box_outs": "DEFENSIVE_BOX_OUTS",
    "loose_balls_recovered_total": "LOOSE_BALLS",
    "screen_assists": "SCREEN_ASSISTS",
    "screen_assist_points": "SCREEN_ASSIST_PTS",
    "secondary_assists": "SECONDARY_ASSISTS",
    "charges_drawn": "CHARGES_DRAWN",
    "switches_on": "SWITCHES_ON",
    # Matchup defense
    "matchup_fg_pct": "MATCHUP_FG_PCT",
    "matchup_fga": "MATCHUP_FGA",
    "matchup_fgm": "MATCHUP_FGM",
    "matchup_3pt_pct": "MATCHUP_3PT_PCT",
    "matchup_3pa": "MATCHUP_3PA",
    "matchup_3pm": "MATCHUP_3PM",
    "matchup_player_points": "MATCHUP_PLAYER_PTS",
    # Rebound chances
    "rebound_chances_total": "REB_CHANCES_TOTAL",
    "rebound_chances_off": "REB_CHANCES_OFF",
    "rebound_chances_def": "REB_CHANCES_DEF",
    # Scoring breakdown
    "points_paint": "PTS_PAINT",
    "points_fast_break": "PTS_FAST_BREAK",
    "points_off_turnovers": "PTS_OFF_TO",
    "points_second_chance": "PTS_SECOND_CHANCE",
    # Advanced per-game
    "usage_percentage": "USG_PCT",
    "pace": "PACE",
    "pie": "PIE",
    "offensive_rating": "OFF_RATING",
    "defensive_rating": "DEF_RATING",
    "true_shooting_percentage": "TS_PCT",
}

_ALL_HEADERS = ["PLAYER_ID", "TEAM_ABBREVIATION"] + list(_BDL_TRACKING_MAP.values())


def _map_player_id(
    bdl_player: dict,
    bdl_team_abbr: str,
    nba_players: dict[tuple[str, str], int],
) -> int | None:
    """Map a BDL player to NBA player ID using name + team."""
    fname = bdl_player.get("first_name", "")
    lname = bdl_player.get("last_name", "")
    full = _normalize(f"{fname} {lname}")
    team = _norm_team(bdl_team_abbr)

    # Direct match
    pid = nba_players.get((full, team))
    if pid:
        return pid

    # Try last-name-only match within team (handles nickname variations)
    lname_norm = _normalize(lname)
    candidates = [(k, v) for k, v in nba_players.items() if k[1] == team and lname_norm in k[0]]
    if len(candidates) == 1:
        return candidates[0][1]

    return None


# ---------------------------------------------------------------------------
# Step 3: Date-based advanced stats fetching
# ---------------------------------------------------------------------------
def fetch_date_advanced_stats(date_str: str) -> list[dict]:
    """Fetch ALL advanced stat records for a given date via pagination."""
    all_records: list[dict] = []
    cursor = None
    while True:
        params: dict[str, Any] = {"dates[]": date_str, "per_page": 100}
        if cursor:
            params["cursor"] = cursor
        data = _bdl_get("/v2/stats/advanced", params)
        records = data.get("data", [])
        if not records:
            break
        all_records.extend(records)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(BDL_SLEEP)
    return all_records


def process_date(
    date_str: str,
    bdl_games_on_date: list[dict],
    nba_map: dict[str, dict],
    cache_dir: Path,
) -> tuple[int, int, int]:
    """
    Fetch and process all advanced stats for a date.
    Returns (games_saved, players_saved, players_unmapped).
    """
    # Build BDL game_id → NBA game mapping for this date
    bdl_to_nba: dict[int, dict] = {}
    for g in bdl_games_on_date:
        bdl_gid = g["id"]
        home = _norm_team(g["home_team"]["abbreviation"])
        away = _norm_team(g["visitor_team"]["abbreviation"])

        key1 = f"{date_str}|{home}|{away}"
        key2 = f"{date_str}|{away}|{home}"
        match = nba_map.get(key1) or nba_map.get(key2)
        if match:
            bdl_to_nba[bdl_gid] = match

    if not bdl_to_nba:
        return 0, 0, 0

    # Check which games are already cached
    uncached = {gid: info for gid, info in bdl_to_nba.items()
                if not (cache_dir / f"{info['nba_game_id']}.json").exists()}
    if not uncached:
        return 0, 0, 0

    # Fetch all advanced stats for this date
    records = fetch_date_advanced_stats(date_str)
    if not records:
        return 0, 0, 0

    # Group records by BDL game_id
    by_game: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        gid = rec.get("game", {}).get("id")
        if gid and gid in uncached:
            by_game[gid].append(rec)

    games_saved = 0
    total_players = 0
    total_unmapped = 0

    for bdl_gid, game_records in by_game.items():
        info = uncached[bdl_gid]
        nba_gid = info["nba_game_id"]
        nba_players = info["players"]
        cache_path = cache_dir / f"{nba_gid}.json"

        rowset: list[list] = []
        unmapped = 0

        for rec in game_records:
            player = rec.get("player", {})
            team = rec.get("team", {})
            team_abbr = _norm_team(team.get("abbreviation", ""))

            nba_pid = _map_player_id(player, team_abbr, nba_players)
            if nba_pid is None:
                unmapped += 1
                continue

            row = [nba_pid, team_abbr]
            for bdl_field in _BDL_TRACKING_MAP:
                row.append(rec.get(bdl_field))
            rowset.append(row)

        payload = {
            "resultSets": [{
                "headers": _ALL_HEADERS,
                "rowSet": rowset,
            }],
            "_source": "balldontlie_v2",
            "_bdl_game_id": bdl_gid,
        }
        cache_path.write_text(json.dumps(payload))
        games_saved += 1
        total_players += len(rowset)
        total_unmapped += unmapped

    return games_saved, total_players, total_unmapped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch BDL tracking + hustle stats")
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025],
        help="BDL season numbers (2021=2021-22, etc.)",
    )
    parser.add_argument("--max-dates", type=int, default=0, help="Max dates to process (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Map games but don't fetch stats")
    args = parser.parse_args()

    TRACKING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build NBA game mapping from boxscores
    print("=== Step 1: Building NBA game map from boxscores ===", flush=True)
    nba_map = build_nba_game_map()
    if not nba_map:
        print("ERROR: No boxscore data found. Run fetch_historical_seasons.py first.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Fetch BDL game dates
    print(f"\n=== Step 2: Fetching BDL game dates for seasons {args.seasons} ===", flush=True)
    date_games = fetch_bdl_game_dates(args.seasons)

    # Count how many games map to NBA
    total_mapped = 0
    for date_str, games in date_games.items():
        for g in games:
            home = _norm_team(g["home_team"]["abbreviation"])
            away = _norm_team(g["visitor_team"]["abbreviation"])
            key1 = f"{date_str}|{home}|{away}"
            key2 = f"{date_str}|{away}|{home}"
            if nba_map.get(key1) or nba_map.get(key2):
                total_mapped += 1
    print(f"\n{total_mapped} BDL games map to NBA boxscores", flush=True)

    if args.dry_run:
        already_cached = sum(1 for p in TRACKING_CACHE_DIR.glob("*.json"))
        print(f"Already cached: {already_cached} games")
        print(f"Dry run complete. {len(date_games)} dates, {total_mapped} mapped games.", flush=True)
        return

    # Step 3: Fetch advanced stats date by date
    dates_sorted = sorted(date_games.keys())
    if args.max_dates > 0:
        dates_sorted = dates_sorted[:args.max_dates]

    print(f"\n=== Step 3: Fetching advanced stats for {len(dates_sorted)} dates ===", flush=True)

    total_games_saved = 0
    total_players_saved = 0
    total_unmapped = 0
    errors = 0

    for i, date_str in enumerate(dates_sorted, 1):
        try:
            gs, ps, um = process_date(date_str, date_games[date_str], nba_map, TRACKING_CACHE_DIR)
            total_games_saved += gs
            total_players_saved += ps
            total_unmapped += um
            if gs > 0:
                time.sleep(BDL_SLEEP)
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on {date_str}: {e}", flush=True)
            elif errors == 6:
                print("  (suppressing further error messages)", flush=True)

        if i % 25 == 0 or i == len(dates_sorted):
            cached = sum(1 for _ in TRACKING_CACHE_DIR.glob("*.json"))
            print(
                f"  Progress: {i}/{len(dates_sorted)} dates | "
                f"{total_games_saved} games saved | "
                f"{total_players_saved} players | "
                f"{total_unmapped} unmapped | "
                f"{cached} total cached | "
                f"{errors} errors",
                flush=True,
            )

    cached_final = sum(1 for _ in TRACKING_CACHE_DIR.glob("*.json"))
    print(
        f"\nDone! {total_games_saved} games saved, "
        f"{total_players_saved} players, "
        f"{total_unmapped} unmapped, "
        f"{errors} errors, "
        f"{cached_final} total in cache.",
        flush=True,
    )


if __name__ == "__main__":
    main()
