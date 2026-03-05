#!/usr/bin/env python3
"""Capture opening-line and line-movement snapshots from ESPN.

Designed to run as a cron job (e.g., twice daily) to build a time-series of
odds for each game date.  Each invocation fetches the ESPN scoreboard for the
target date(s) and saves a timestamped JSON snapshot.

Usage:
    # Snapshot today + tomorrow (default)
    python scripts/fetch_opening_lines.py

    # Snapshot a specific date
    python scripts/fetch_opening_lines.py --date 20260301

    # Snapshot multiple days ahead
    python scripts/fetch_opening_lines.py --days-ahead 3
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis" / "output"
SNAPSHOT_DIR = OUT_DIR / "odds_snapshots"

# ESPN endpoints (same as the monolith uses)
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    "?dates={yyyymmdd}"
)

ESPN_ABBR_MAP = {
    "GS": "GSW",
    "SA": "SAS",
    "NO": "NOP",
    "NY": "NYK",
    "PHO": "PHX",
    "UTAH": "UTA",
    "WSH": "WAS",
    "BK": "BKN",
}


def normalize_espn_abbr(abbr: str) -> str:
    if not abbr:
        return abbr
    return ESPN_ABBR_MAP.get(abbr, abbr)


def _to_float(v: Any) -> float | None:
    """Convert a value to float, returning None on failure."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def american_to_prob(odds: float | int | None) -> float | None:
    """Convert American odds to implied probability."""
    if odds is None:
        return None
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return None


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str, timeout: int = 20, retries: int = 3) -> dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def fetch_scoreboard_snapshot(date_str: str) -> dict[str, Any]:
    """Fetch the ESPN scoreboard for a given YYYYMMDD date and extract odds."""
    payload = fetch_json(ESPN_SCOREBOARD_URL.format(yyyymmdd=date_str))
    now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    games: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_abbr = normalize_espn_abbr(home.get("team", {}).get("abbreviation", ""))
        away_abbr = normalize_espn_abbr(away.get("team", {}).get("abbreviation", ""))
        event_id = str(event.get("id", ""))

        # Extract odds from the competition
        odds_entries = comp.get("odds", [])
        odds_data: list[dict[str, Any]] = []
        for od in odds_entries:
            provider = od.get("provider", {})
            home_ml = _to_float(od.get("homeTeamOdds", {}).get("moneyLine"))
            away_ml = _to_float(od.get("awayTeamOdds", {}).get("moneyLine"))
            spread = _to_float(od.get("spread"))
            over_under = _to_float(od.get("overUnder"))
            # Opening values (if available in this endpoint)
            open_spread = _to_float(od.get("spreadOdds", {}).get("open") if isinstance(od.get("spreadOdds"), dict) else None)
            details = od.get("details", "")

            odds_data.append({
                "provider_id": provider.get("id"),
                "provider_name": provider.get("name"),
                "spread": spread,
                "over_under": over_under,
                "details": details,
                "home_moneyline": home_ml,
                "away_moneyline": away_ml,
                "home_implied_prob": american_to_prob(home_ml),
                "away_implied_prob": american_to_prob(away_ml),
            })

        games.append({
            "espn_event_id": event_id,
            "home_team": home_abbr,
            "away_team": away_abbr,
            "game_name": event.get("name"),
            "game_start_utc": event.get("date"),
            "status": event.get("status", {}).get("type", {}).get("name"),
            "odds": odds_data,
        })

    return {
        "snapshot_time_utc": now_iso,
        "game_date": date_str,
        "games_count": len(games),
        "games": games,
    }


def save_snapshot(snapshot: dict[str, Any], date_str: str) -> Path:
    """Save a snapshot to the dated directory with a timestamp filename."""
    date_dir = SNAPSHOT_DIR / date_str
    date_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    filename = f"snapshot_{ts}.json"
    filepath = date_dir / filename

    filepath.write_text(json.dumps(snapshot, indent=2, default=str))
    return filepath


# ---------------------------------------------------------------------------
# Snapshot loading utilities (importable by other scripts)
# ---------------------------------------------------------------------------

def load_snapshots_for_date(date_str: str) -> list[dict[str, Any]]:
    """Load all snapshots for a given game date, sorted chronologically."""
    date_dir = SNAPSHOT_DIR / date_str
    if not date_dir.exists():
        return []

    snapshots = []
    for f in sorted(date_dir.glob("snapshot_*.json")):
        try:
            data = json.loads(f.read_text())
            snapshots.append(data)
        except Exception:
            continue
    return snapshots


def compute_line_movement(date_str: str) -> dict[str, dict[str, Any]]:
    """Compute opening/closing line movement from snapshots for a date.

    Returns a dict keyed by espn_event_id with opening and closing odds
    and computed movement.
    """
    snapshots = load_snapshots_for_date(date_str)
    if not snapshots:
        return {}

    # earliest = opening, latest = closing
    opening = snapshots[0]
    closing = snapshots[-1]

    movements: dict[str, dict[str, Any]] = {}

    # Build lookup from opening snapshot
    open_games = {g["espn_event_id"]: g for g in opening.get("games", [])}
    close_games = {g["espn_event_id"]: g for g in closing.get("games", [])}

    all_event_ids = set(open_games.keys()) | set(close_games.keys())

    for eid in all_event_ids:
        og = open_games.get(eid, {})
        cg = close_games.get(eid, {})

        # Use first odds provider in each snapshot (typically the primary book)
        open_odds = og.get("odds", [{}])[0] if og.get("odds") else {}
        close_odds = cg.get("odds", [{}])[0] if cg.get("odds") else {}

        open_spread = open_odds.get("spread")
        close_spread = close_odds.get("spread")
        open_total = open_odds.get("over_under")
        close_total = close_odds.get("over_under")
        open_home_prob = open_odds.get("home_implied_prob")
        close_home_prob = close_odds.get("home_implied_prob")

        spread_move = None
        if open_spread is not None and close_spread is not None:
            spread_move = close_spread - open_spread

        total_move = None
        if open_total is not None and close_total is not None:
            total_move = close_total - open_total

        ml_move = None
        if open_home_prob is not None and close_home_prob is not None:
            ml_move = close_home_prob - open_home_prob

        movements[eid] = {
            "home_team": og.get("home_team") or cg.get("home_team"),
            "away_team": og.get("away_team") or cg.get("away_team"),
            "snapshot_spread_open": open_spread,
            "snapshot_spread_close": close_spread,
            "snapshot_total_open": open_total,
            "snapshot_total_close": close_total,
            "snapshot_home_implied_prob_open": open_home_prob,
            "snapshot_home_implied_prob_close": close_home_prob,
            "snapshot_spread_move": spread_move,
            "snapshot_total_move": total_move,
            "snapshot_ml_move": ml_move,
            "snapshot_spread_move_abs": abs(spread_move) if spread_move is not None else None,
            "snapshot_total_move_abs": abs(total_move) if total_move is not None else None,
            "num_snapshots": len(snapshots),
            "opening_time_utc": opening.get("snapshot_time_utc"),
            "closing_time_utc": closing.get("snapshot_time_utc"),
        }

    return movements


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture ESPN odds snapshots for NBA games")
    p.add_argument("--date", type=str, default=None,
                   help="Target date YYYYMMDD (default: today)")
    p.add_argument("--days-ahead", type=int, default=1,
                   help="Number of additional days ahead to snapshot (default: 1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.date:
        base_date = datetime.strptime(args.date, "%Y%m%d")
    else:
        base_date = datetime.now()

    dates = []
    for offset in range(args.days_ahead + 1):
        d = base_date + timedelta(days=offset)
        dates.append(d.strftime("%Y%m%d"))

    print(f"Fetching odds snapshots for dates: {', '.join(dates)}", flush=True)

    for date_str in dates:
        print(f"\n--- {date_str} ---", flush=True)
        try:
            snapshot = fetch_scoreboard_snapshot(date_str)
            filepath = save_snapshot(snapshot, date_str)
            n_games = snapshot["games_count"]
            n_with_odds = sum(1 for g in snapshot["games"] if g.get("odds"))
            print(f"  Saved: {filepath.name} ({n_games} games, {n_with_odds} with odds)", flush=True)

            # Show summary of current odds
            for g in snapshot["games"]:
                odds_str = "no odds"
                if g.get("odds"):
                    od = g["odds"][0]
                    spread = od.get("spread")
                    total = od.get("over_under")
                    odds_str = f"spread={spread}, total={total}"
                print(f"    {g['away_team']} @ {g['home_team']}: {odds_str}", flush=True)

        except Exception as exc:
            print(f"  Error fetching {date_str}: {exc}", flush=True)

    # Show existing snapshots summary
    print("\n--- Existing snapshot coverage ---", flush=True)
    if SNAPSHOT_DIR.exists():
        for date_dir in sorted(SNAPSHOT_DIR.iterdir()):
            if date_dir.is_dir():
                count = len(list(date_dir.glob("snapshot_*.json")))
                print(f"  {date_dir.name}: {count} snapshot(s)", flush=True)
    else:
        print("  No snapshots yet.", flush=True)

    # Print cron setup instructions
    print("\n" + "=" * 60, flush=True)
    print("CRON SETUP INSTRUCTIONS", flush=True)
    print("=" * 60, flush=True)
    print("""
# Add to crontab for automated line capture:
# Edit crontab: crontab -e

# Morning snapshot (9 AM ET / 14:00 UTC):
0 14 * * * cd /Users/ryangoldstein/NBA && python3 scripts/fetch_opening_lines.py 2>&1 >> /tmp/nba_odds_snapshot.log

# Evening snapshot (6 PM ET / 23:00 UTC):
0 23 * * * cd /Users/ryangoldstein/NBA && python3 scripts/fetch_opening_lines.py 2>&1 >> /tmp/nba_odds_snapshot.log

# Optional: midday snapshot for extra granularity (1 PM ET / 18:00 UTC):
# 0 18 * * * cd /Users/ryangoldstein/NBA && python3 scripts/fetch_opening_lines.py 2>&1 >> /tmp/nba_odds_snapshot.log
""", flush=True)


if __name__ == "__main__":
    main()
