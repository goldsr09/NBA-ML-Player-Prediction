"""Fetch historical NBA season boxscores (2021-22 through 2024-25).

Downloads boxscores from the NBA CDN and ESPN odds for ~5000 historical games.
Caches to analysis/output/historical_cache/{season}/.

Usage:
    python scripts/fetch_historical_seasons.py [--seasons 2021-22 2022-23 ...]
    python scripts/fetch_historical_seasons.py --skip-odds
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis" / "output"
HIST_CACHE_DIR = OUT_DIR / "historical_cache"

BOXSCORE_URL_TMPL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={yyyymmdd}"
)
ESPN_ODDS_LIST_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event_id}"
    "/competitions/{event_id}/odds?lang=en&region=us"
)

ESPN_ABBR_MAP = {
    "GS": "GSW", "SA": "SAS", "NO": "NOP", "NY": "NYK",
    "PHO": "PHX", "UTAH": "UTA", "WSH": "WAS", "BK": "BKN",
}

# Season code -> (2-digit code for game IDs, regular-season game count)
SEASON_INFO: dict[str, tuple[str, int]] = {
    "2021-22": ("21", 1230),
    "2022-23": ("22", 1230),
    "2023-24": ("23", 1230),
    "2024-25": ("24", 1230),
}

DEFAULT_SEASONS = list(SEASON_INFO.keys())


def _normalize_abbr(abbr: str) -> str:
    return ESPN_ABBR_MAP.get(abbr.upper(), abbr.upper()) if abbr else abbr


def fetch_json_cached(
    url: str, cache_path: Path, timeout: int = 30, retries: int = 3
) -> dict[str, Any] | None:
    """Fetch JSON with disk caching. Returns None on failure/404."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            cache_path.unlink()
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            payload = r.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload))
            return payload
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    return None


def generate_game_ids(season: str) -> list[str]:
    """Generate regular-season game IDs for a given season."""
    code, n_games = SEASON_INFO[season]
    return [f"002{code}{i:05d}" for i in range(1, n_games + 1)]


def fetch_season_boxscores(season: str, max_workers: int = 10) -> tuple[int, int]:
    """Fetch all boxscores for a season. Returns (success, errors)."""
    game_ids = generate_game_ids(season)
    cache_dir = HIST_CACHE_DIR / season / "boxscores"
    cache_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    errors = 0

    def fetch_one(gid: str) -> bool:
        path = cache_dir / f"{gid}.json"
        payload = fetch_json_cached(
            BOXSCORE_URL_TMPL.format(game_id=gid), path, timeout=30, retries=3
        )
        return payload is not None

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_one, gid): gid for gid in game_ids}
        for fut in cf.as_completed(futs):
            if fut.result():
                success += 1
            else:
                errors += 1
            total = success + errors
            if total % 200 == 0 or total == len(game_ids):
                print(
                    f"  [{season}] Boxscores: {total}/{len(game_ids)} "
                    f"({success} ok, {errors} failed)",
                    flush=True,
                )
    return success, errors


def extract_dates_from_boxscores(season: str) -> list[str]:
    """Extract unique game dates (YYYYMMDD) from cached boxscores."""
    cache_dir = HIST_CACHE_DIR / season / "boxscores"
    dates: set[str] = set()
    for f in cache_dir.glob("*.json"):
        try:
            game = json.loads(f.read_text())["game"]
            et = game.get("gameEt") or game.get("gameTimeUTC")
            if et:
                dates.add(str(et)[:10].replace("-", ""))
        except Exception:
            continue
    return sorted(dates)


def fetch_season_espn_odds(season: str, max_workers: int = 12) -> int:
    """Fetch ESPN odds for all games in a season. Returns count of events with odds."""
    dates = extract_dates_from_boxscores(season)
    if not dates:
        return 0

    sb_cache = HIST_CACHE_DIR / season / "espn_scoreboards"
    sb_cache.mkdir(parents=True, exist_ok=True)

    # Fetch ESPN scoreboards and extract event IDs
    event_map: dict[tuple[str, str, str], str] = {}
    for d in dates:
        payload = fetch_json_cached(
            ESPN_SCOREBOARD_URL.format(yyyymmdd=d), sb_cache / f"{d}.json", timeout=20, retries=3
        )
        if not payload:
            continue
        for e in payload.get("events", []):
            comp = (e.get("competitions") or [{}])[0]
            comps = comp.get("competitors", [])
            home = next((c for c in comps if c.get("homeAway") == "home"), None)
            away = next((c for c in comps if c.get("homeAway") == "away"), None)
            if home and away:
                h = _normalize_abbr(home.get("team", {}).get("abbreviation", ""))
                a = _normalize_abbr(away.get("team", {}).get("abbreviation", ""))
                event_map[(d, h, a)] = str(e.get("id"))

    # Fetch odds for matched events
    odds_cache = HIST_CACHE_DIR / season / "espn_odds"
    odds_cache.mkdir(parents=True, exist_ok=True)

    event_ids = list(set(event_map.values()))
    matched = 0

    def fetch_odds(eid: str) -> bool:
        path = odds_cache / f"{eid}.json"
        payload = fetch_json_cached(
            ESPN_ODDS_LIST_URL.format(event_id=eid), path, timeout=20, retries=3
        )
        return payload is not None

    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_odds, eid): eid for eid in event_ids}
        done_count = 0
        for fut in cf.as_completed(futs):
            if fut.result():
                matched += 1
            done_count += 1
            if done_count % 200 == 0 or done_count == len(event_ids):
                print(
                    f"  [{season}] ESPN odds: {done_count}/{len(event_ids)} ({matched} ok)",
                    flush=True,
                )

    # Save event mapping for later use
    mapping_path = HIST_CACHE_DIR / season / "espn_event_map.json"
    mapping_data = [
        {"date": d, "home": h, "away": a, "event_id": eid}
        for (d, h, a), eid in event_map.items()
    ]
    mapping_path.write_text(json.dumps(mapping_data, indent=2))
    return matched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical NBA season data.")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_SEASONS,
        help=f"Seasons to fetch (default: {DEFAULT_SEASONS})",
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Max concurrent requests (default: 10)"
    )
    parser.add_argument("--skip-odds", action="store_true", help="Skip ESPN odds fetching")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    HIST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    total_games = 0
    total_errors = 0
    total_odds = 0

    for season in args.seasons:
        if season not in SEASON_INFO:
            print(f"Unknown season: {season}, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Season: {season}")
        print(f"{'=' * 60}")

        success, errors = fetch_season_boxscores(season, max_workers=args.workers)
        total_games += success
        total_errors += errors
        print(f"  Boxscores: {success} fetched, {errors} errors")

        if not args.skip_odds:
            print(f"  Fetching ESPN odds for {season}...")
            odds_count = fetch_season_espn_odds(season, max_workers=args.workers)
            total_odds += odds_count
            print(f"  ESPN odds: {odds_count} events with odds data")

        time.sleep(1.0)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total boxscores fetched: {total_games}")
    print(f"Total errors: {total_errors}")
    if not args.skip_odds:
        print(f"Total ESPN odds events: {total_odds}")
    print(f"Cache location: {HIST_CACHE_DIR}")


if __name__ == "__main__":
    main()
