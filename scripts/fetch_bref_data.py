"""Fetch historical NBA data from Basketball Reference.

Scrapes box scores, player game logs, and opponent stats that fill gaps
not covered by NBA CDN, ESPN, or BDL sources. Key data:
  - Starting lineups (fixes train/serve asymmetry on confirmed_starter)
  - Per-game advanced stats (usage%, ORtg, DRtg per player)
  - Steals/blocks/turnovers per player per game (opens new prop markets)
  - Opponent defensive stats by position
  - Shot type breakdowns (at-rim, mid-range, 3PT)

Caches all responses to analysis/output/bref_cache/. Respects BRef rate
limits (3s between requests). Fully idempotent — re-running skips cached data.

Usage:
    # Fetch box scores for specific seasons
    python scripts/fetch_bref_data.py --seasons 2024-25 2023-24

    # Fetch all supported seasons (2021-22 through 2024-25)
    python scripts/fetch_bref_data.py --all-seasons

    # Fetch only specific data types
    python scripts/fetch_bref_data.py --seasons 2024-25 --only boxscores
    python scripts/fetch_bref_data.py --seasons 2024-25 --only opponent-stats
    python scripts/fetch_bref_data.py --seasons 2024-25 --only shooting

    # Export parsed data to CSV for downstream pipelines
    python scripts/fetch_bref_data.py --seasons 2024-25 --export-csv

    # Adjust rate limit (default 3.0s between requests)
    python scripts/fetch_bref_data.py --seasons 2024-25 --delay 4.0

    # Resume after interruption (skips already-cached pages)
    python scripts/fetch_bref_data.py --seasons 2024-25
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, Comment  # uses html.parser (no lxml needed)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis" / "output"
BREF_CACHE_DIR = OUT_DIR / "bref_cache"

BASE_URL = "https://www.basketball-reference.com"

# Basketball Reference season codes: "2024-25" -> 2025 (ending year)
SUPPORTED_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]

# BRef uses 3-letter abbreviations that mostly match ours
BREF_ABBR_MAP = {
    "BRK": "BKN",
    "CHO": "CHA",
    "PHO": "PHX",
    "NJN": "BKN",
    "NOH": "NOP",
    "NOK": "NOP",
    "SEA": "OKC",
    "VAN": "MEM",
    "WSB": "WAS",
}

# Reverse map for constructing BRef URLs
OUR_TO_BREF = {v: k for k, v in BREF_ABBR_MAP.items()}
OUR_TO_BREF.update({
    "BKN": "BRK",
    "CHA": "CHO",
    "PHX": "PHO",
})

TEAMS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

REQUEST_DELAY = 3.0  # seconds between requests (BRef rate limit)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _season_end_year(season: str) -> int:
    """'2024-25' -> 2025."""
    return int("20" + season.split("-")[1])


def _bref_team(our_abbr: str) -> str:
    """Convert our team abbreviation to BRef's."""
    return OUR_TO_BREF.get(our_abbr, our_abbr)


def _our_team(bref_abbr: str) -> str:
    """Convert BRef abbreviation to ours."""
    return BREF_ABBR_MAP.get(bref_abbr, bref_abbr)


def _fetch_page(url: str, cache_path: Path, delay: float = REQUEST_DELAY) -> str | None:
    """Fetch an HTML page with disk caching and rate limiting."""
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8")
        if text.strip():
            return text

    time.sleep(delay)
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 404:
            # Cache empty marker so we don't re-fetch
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("")
            return None
        if r.status_code == 429:
            print(f"  Rate limited — waiting 60s before retry...")
            time.sleep(60)
            r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        html = r.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")
        return html
    except Exception as exc:
        print(f"  ERROR fetching {url}: {exc}")
        return None


def _safe_float(val: str | None) -> float | None:
    """Parse a string to float, returning None on failure."""
    if val is None or val.strip() == "" or val.strip() == "-":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: str | None) -> int | None:
    if val is None or val.strip() == "" or val.strip() == "-":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_minutes(mp_str: str | None) -> float | None:
    """Parse BRef minutes format 'MM:SS' or plain number to decimal minutes."""
    if mp_str is None or mp_str.strip() == "" or mp_str.strip() == "-":
        return None
    mp_str = mp_str.strip()
    if ":" in mp_str:
        parts = mp_str.split(":")
        try:
            return int(parts[0]) + int(parts[1]) / 60.0
        except (ValueError, IndexError):
            return None
    return _safe_float(mp_str)


def _uncomment_tables(soup: BeautifulSoup) -> None:
    """BRef hides some tables inside HTML comments. Uncomment them."""
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            new_soup = BeautifulSoup(comment, "html.parser")
            comment.replace_with(new_soup)


# ---------------------------------------------------------------------------
# 1. Box score parsing — starters, per-game stats, advanced
# ---------------------------------------------------------------------------
def fetch_schedule(season: str, delay: float = REQUEST_DELAY) -> list[dict]:
    """Fetch the season schedule (all months) from BRef. Returns game metadata."""
    end_year = _season_end_year(season)
    cache_dir = BREF_CACHE_DIR / season / "schedule"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # BRef splits schedule by month
    months = ["october", "november", "december", "january", "february",
              "march", "april", "may", "june"]
    games = []

    for month in months:
        url = f"{BASE_URL}/leagues/NBA_{end_year}_games-{month}.html"
        cache_path = cache_dir / f"{month}.html"
        html = _fetch_page(url, cache_path, delay=delay)
        if html is None:
            continue

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id="schedule")
        if table is None:
            continue

        tbody = table.find("tbody")
        if tbody is None:
            continue

        for row in tbody.find_all("tr"):
            # Skip header rows within tbody
            if row.find("th", {"scope": "col"}):
                continue

            cells = row.find_all(["th", "td"])
            if len(cells) < 6:
                continue

            date_cell = cells[0]
            date_link = date_cell.find("a")
            if date_link is None:
                continue

            # Parse date from the link href: /boxscores/?month=X&day=Y&year=Z
            date_text = date_cell.get_text(strip=True)
            try:
                game_date = datetime.strptime(date_text, "%a, %b %d, %Y")
            except ValueError:
                try:
                    game_date = datetime.strptime(date_text, "%B %d, %Y")
                except ValueError:
                    continue

            # Visitor and Home teams
            visitor_cell = cells[2] if len(cells) > 2 else None
            home_cell = cells[4] if len(cells) > 4 else None
            if visitor_cell is None or home_cell is None:
                continue

            visitor_link = visitor_cell.find("a")
            home_link = home_cell.find("a")
            if visitor_link is None or home_link is None:
                continue

            # Extract team abbr from href: /teams/XXX/YYYY.html
            visitor_href = visitor_link.get("href", "")
            home_href = home_link.get("href", "")
            visitor_match = re.search(r"/teams/(\w+)/", visitor_href)
            home_match = re.search(r"/teams/(\w+)/", home_href)
            if visitor_match is None or home_match is None:
                continue

            visitor_abbr = _our_team(visitor_match.group(1))
            home_abbr = _our_team(home_match.group(1))

            # Box score link
            box_cell = cells[6] if len(cells) > 6 else None
            box_href = None
            if box_cell:
                box_link = box_cell.find("a")
                if box_link:
                    box_href = box_link.get("href", "")

            date_str = game_date.strftime("%Y-%m-%d")
            games.append({
                "date": date_str,
                "visitor": visitor_abbr,
                "home": home_abbr,
                "box_href": box_href,
            })

    # Save parsed schedule
    sched_path = BREF_CACHE_DIR / season / "schedule.json"
    sched_path.parent.mkdir(parents=True, exist_ok=True)
    sched_path.write_text(json.dumps(games, indent=2))
    print(f"  {season}: {len(games)} games in schedule")
    return games


def parse_boxscore(html: str, game_date: str) -> dict | None:
    """Parse a BRef box score page into structured data.

    Returns dict with:
      - game_date, home, visitor, home_pts, visitor_pts
      - players: list of player dicts with stats + starter flag
      - home_starters / visitor_starters: lists of player names
    """
    soup = BeautifulSoup(html, "html.parser")
    _uncomment_tables(soup)

    # Find the scorebox for team names
    scorebox = soup.find("div", class_="scorebox")
    if scorebox is None:
        return None

    team_divs = scorebox.find_all("div", recursive=False)
    if len(team_divs) < 2:
        return None

    team_names = []
    for td in team_divs[:2]:
        link = td.find("a", href=re.compile(r"/teams/\w+/"))
        if link:
            href = link.get("href", "")
            m = re.search(r"/teams/(\w+)/", href)
            if m:
                team_names.append(_our_team(m.group(1)))

    if len(team_names) < 2:
        return None

    visitor_abbr, home_abbr = team_names[0], team_names[1]

    # Parse player stats from basic box score tables
    players = []

    # Find all basic box score tables (pattern: box-XXX-game-basic)
    basic_tables = soup.find_all("table", id=re.compile(r"box-\w+-game-basic"))
    advanced_tables = soup.find_all("table", id=re.compile(r"box-\w+-game-advanced"))

    # Build advanced stats lookup by player
    advanced_lookup: dict[str, dict] = {}
    for adv_table in advanced_tables:
        table_id = adv_table.get("id", "")
        # Extract team from table id: box-XXX-game-advanced
        team_match = re.search(r"box-(\w+)-game-advanced", table_id)
        if team_match is None:
            continue
        bref_team = team_match.group(1)
        team_abbr = _our_team(bref_team)

        tbody = adv_table.find("tbody")
        if tbody is None:
            continue

        for row in tbody.find_all("tr"):
            if "thead" in row.get("class", []) or row.find("th", {"scope": "col"}):
                continue

            name_cell = row.find("th", {"data-stat": "player"})
            if name_cell is None:
                continue
            player_name = name_cell.get_text(strip=True)
            if player_name.lower() in ("reserves", "team totals", ""):
                continue

            cells = {td.get("data-stat"): td.get_text(strip=True) for td in row.find_all("td")}
            advanced_lookup[f"{team_abbr}_{player_name}"] = {
                "ts_pct": _safe_float(cells.get("ts_pct")),
                "efg_pct": _safe_float(cells.get("efg_pct")),
                "orb_pct": _safe_float(cells.get("orb_pct")),
                "drb_pct": _safe_float(cells.get("drb_pct")),
                "trb_pct": _safe_float(cells.get("trb_pct")),
                "ast_pct": _safe_float(cells.get("ast_pct")),
                "stl_pct": _safe_float(cells.get("stl_pct")),
                "blk_pct": _safe_float(cells.get("blk_pct")),
                "tov_pct": _safe_float(cells.get("tov_pct")),
                "usg_pct": _safe_float(cells.get("usg_pct")),
                "off_rtg": _safe_float(cells.get("off_rtg")),
                "def_rtg": _safe_float(cells.get("def_rtg")),
                "bpm": _safe_float(cells.get("bpm")),
            }

    for basic_table in basic_tables:
        table_id = basic_table.get("id", "")
        team_match = re.search(r"box-(\w+)-game-basic", table_id)
        if team_match is None:
            continue
        bref_team = team_match.group(1)
        team_abbr = _our_team(bref_team)
        is_home = team_abbr == home_abbr

        tbody = basic_table.find("tbody")
        if tbody is None:
            continue

        player_idx = 0
        for row in tbody.find_all("tr"):
            if "thead" in row.get("class", []) or row.find("th", {"scope": "col"}):
                continue

            name_cell = row.find("th", {"data-stat": "player"})
            if name_cell is None:
                continue
            player_name = name_cell.get_text(strip=True)
            if player_name.lower() in ("reserves", "team totals", ""):
                continue

            # Did Not Play / Did Not Dress / Inactive
            reason_cell = row.find("td", {"data-stat": "reason"})
            if reason_cell and reason_cell.get_text(strip=True):
                player_idx += 1
                continue

            cells = {td.get("data-stat"): td.get_text(strip=True) for td in row.find_all("td")}

            mp = _parse_minutes(cells.get("mp"))
            if mp is None or mp == 0:
                player_idx += 1
                continue

            # First 5 players are starters in BRef box scores
            is_starter = player_idx < 5

            # Merge advanced stats
            adv = advanced_lookup.get(f"{team_abbr}_{player_name}", {})

            player_data = {
                "player_name": player_name,
                "team": team_abbr,
                "is_home": is_home,
                "is_starter": is_starter,
                "mp": round(mp, 1),
                # Basic counting stats
                "fg": _safe_int(cells.get("fg")),
                "fga": _safe_int(cells.get("fga")),
                "fg_pct": _safe_float(cells.get("fg_pct")),
                "fg3": _safe_int(cells.get("fg3")),
                "fg3a": _safe_int(cells.get("fg3a")),
                "fg3_pct": _safe_float(cells.get("fg3_pct")),
                "ft": _safe_int(cells.get("ft")),
                "fta": _safe_int(cells.get("fta")),
                "ft_pct": _safe_float(cells.get("ft_pct")),
                "orb": _safe_int(cells.get("orb")),
                "drb": _safe_int(cells.get("drb")),
                "trb": _safe_int(cells.get("trb")),
                "ast": _safe_int(cells.get("ast")),
                "stl": _safe_int(cells.get("stl")),
                "blk": _safe_int(cells.get("blk")),
                "tov": _safe_int(cells.get("tov")),
                "pf": _safe_int(cells.get("pf")),
                "pts": _safe_int(cells.get("pts")),
                "plus_minus": _safe_float(cells.get("plus_minus")),
                # Advanced stats (from advanced table)
                **{f"adv_{k}": v for k, v in adv.items()},
            }
            players.append(player_data)
            player_idx += 1

    if not players:
        return None

    # Determine starters
    home_starters = [p["player_name"] for p in players if p["is_home"] and p["is_starter"]]
    visitor_starters = [p["player_name"] for p in players if not p["is_home"] and p["is_starter"]]

    # Get scores from scorebox
    scores = scorebox.find_all("div", class_="score")
    visitor_pts = _safe_int(scores[0].get_text(strip=True)) if len(scores) > 0 else None
    home_pts = _safe_int(scores[1].get_text(strip=True)) if len(scores) > 1 else None

    return {
        "game_date": game_date,
        "home": home_abbr,
        "visitor": visitor_abbr,
        "home_pts": home_pts,
        "visitor_pts": visitor_pts,
        "home_starters": home_starters,
        "visitor_starters": visitor_starters,
        "players": players,
    }


def fetch_boxscores(season: str, games: list[dict], delay: float = REQUEST_DELAY) -> list[dict]:
    """Fetch and parse all box scores for a season."""
    cache_dir = BREF_CACHE_DIR / season / "boxscores"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = BREF_CACHE_DIR / season / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(games)

    for i, game in enumerate(games):
        box_href = game.get("box_href")
        if box_href is None:
            continue

        # Cache key from href: /boxscores/202410220BOS.html -> 202410220BOS
        bref_id = box_href.split("/")[-1].replace(".html", "")
        cache_path = cache_dir / f"{bref_id}.html"
        parsed_path = parsed_dir / f"{bref_id}.json"

        # Skip if already parsed
        if parsed_path.exists():
            try:
                data = json.loads(parsed_path.read_text())
                if data:
                    results.append(data)
                    continue
            except json.JSONDecodeError:
                pass

        url = f"{BASE_URL}{box_href}"
        html = _fetch_page(url, cache_path, delay=delay)

        if html is None or html.strip() == "":
            continue

        data = parse_boxscore(html, game["date"])
        if data:
            data["bref_id"] = bref_id
            parsed_path.write_text(json.dumps(data, indent=2))
            results.append(data)

        if (i + 1) % 50 == 0:
            print(f"  {season}: {i + 1}/{total} box scores processed ({len(results)} parsed)")

    print(f"  {season}: {len(results)}/{total} box scores successfully parsed")
    return results


# ---------------------------------------------------------------------------
# 2. Opponent stats — team defensive metrics
# ---------------------------------------------------------------------------
def fetch_opponent_stats(season: str, delay: float = REQUEST_DELAY) -> dict | None:
    """Fetch team opponent (defensive) stats from the league page."""
    end_year = _season_end_year(season)
    cache_dir = BREF_CACHE_DIR / season / "team_stats"
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/leagues/NBA_{end_year}.html"
    cache_path = cache_dir / "league_page.html"
    html = _fetch_page(url, cache_path, delay=delay)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    _uncomment_tables(soup)

    # Find opponent stats per game table
    opp_table = soup.find("table", id="per_game-opponent")
    if opp_table is None:
        # Try alternate id
        opp_table = soup.find("table", id="opponent-stats-per_game")
    if opp_table is None:
        print(f"  {season}: opponent stats table not found")
        return None

    teams_data = {}
    tbody = opp_table.find("tbody")
    if tbody is None:
        return None

    for row in tbody.find_all("tr"):
        if "thead" in row.get("class", []):
            continue

        team_cell = row.find("td", {"data-stat": "team"})
        if team_cell is None:
            continue
        team_link = team_cell.find("a")
        if team_link is None:
            # Try text-based extraction (some rows lack links)
            continue
        href = team_link.get("href", "")
        m = re.search(r"/teams/(\w+)/", href)
        if m is None:
            continue
        team_abbr = _our_team(m.group(1))

        cells = {td.get("data-stat"): td.get_text(strip=True) for td in row.find_all("td")}

        teams_data[team_abbr] = {
            "opp_fg_per_g": _safe_float(cells.get("opp_fg")),
            "opp_fga_per_g": _safe_float(cells.get("opp_fga")),
            "opp_fg_pct": _safe_float(cells.get("opp_fg_pct")),
            "opp_fg3_per_g": _safe_float(cells.get("opp_fg3")),
            "opp_fg3a_per_g": _safe_float(cells.get("opp_fg3a")),
            "opp_fg3_pct": _safe_float(cells.get("opp_fg3_pct")),
            "opp_ft_per_g": _safe_float(cells.get("opp_ft")),
            "opp_fta_per_g": _safe_float(cells.get("opp_fta")),
            "opp_orb_per_g": _safe_float(cells.get("opp_orb")),
            "opp_drb_per_g": _safe_float(cells.get("opp_drb")),
            "opp_trb_per_g": _safe_float(cells.get("opp_trb")),
            "opp_ast_per_g": _safe_float(cells.get("opp_ast")),
            "opp_stl_per_g": _safe_float(cells.get("opp_stl")),
            "opp_blk_per_g": _safe_float(cells.get("opp_blk")),
            "opp_tov_per_g": _safe_float(cells.get("opp_tov")),
            "opp_pf_per_g": _safe_float(cells.get("opp_pf")),
            "opp_pts_per_g": _safe_float(cells.get("opp_pts")),
        }

    out_path = BREF_CACHE_DIR / season / "opponent_stats.json"
    out_path.write_text(json.dumps(teams_data, indent=2))
    print(f"  {season}: opponent stats for {len(teams_data)} teams")
    return teams_data


# ---------------------------------------------------------------------------
# 3. Team shooting splits — shot type breakdowns
# ---------------------------------------------------------------------------
def fetch_team_shooting(season: str, delay: float = REQUEST_DELAY) -> dict | None:
    """Fetch team shooting splits (by distance, shot type) from BRef."""
    end_year = _season_end_year(season)
    cache_dir = BREF_CACHE_DIR / season / "team_stats"
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/leagues/NBA_{end_year}.html"
    cache_path = cache_dir / "league_page.html"

    # Reuse cached league page from opponent_stats
    if cache_path.exists():
        html = cache_path.read_text(encoding="utf-8")
    else:
        html = _fetch_page(url, cache_path, delay=delay)

    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    _uncomment_tables(soup)

    # Look for the shooting table
    shooting_table = soup.find("table", id="shooting-team")
    if shooting_table is None:
        print(f"  {season}: team shooting table not found on league page")
        return None

    teams_data = {}
    tbody = shooting_table.find("tbody")
    if tbody is None:
        return None

    for row in tbody.find_all("tr"):
        if "thead" in row.get("class", []):
            continue

        team_cell = row.find("td", {"data-stat": "team"})
        if team_cell is None:
            continue
        team_link = team_cell.find("a")
        if team_link is None:
            continue
        href = team_link.get("href", "")
        m = re.search(r"/teams/(\w+)/", href)
        if m is None:
            continue
        team_abbr = _our_team(m.group(1))

        cells = {td.get("data-stat"): td.get_text(strip=True) for td in row.find_all("td")}

        teams_data[team_abbr] = {
            "avg_dist": _safe_float(cells.get("avg_dist")),
            "pct_fga_2p": _safe_float(cells.get("pct_fga_fg2a")),
            "pct_fga_0_3": _safe_float(cells.get("pct_fga_00_03")),
            "pct_fga_3_10": _safe_float(cells.get("pct_fga_03_10")),
            "pct_fga_10_16": _safe_float(cells.get("pct_fga_10_16")),
            "pct_fga_16_3pt": _safe_float(cells.get("pct_fga_16_xx")),
            "pct_fga_3p": _safe_float(cells.get("pct_fga_fg3a")),
            "fg_pct_0_3": _safe_float(cells.get("fg_pct_00_03")),
            "fg_pct_3_10": _safe_float(cells.get("fg_pct_03_10")),
            "fg_pct_10_16": _safe_float(cells.get("fg_pct_10_16")),
            "fg_pct_16_3pt": _safe_float(cells.get("fg_pct_16_xx")),
            "pct_ast_fg2": _safe_float(cells.get("pct_ast_fg2")),
            "pct_ast_fg3": _safe_float(cells.get("pct_ast_fg3")),
            # Dunk / layup / corner 3 stats if present
            "pct_fga_dunk": _safe_float(cells.get("pct_fga_dunk")),
            "fg_dunk": _safe_int(cells.get("fg_dunk")),
            "pct_fga_layup": _safe_float(cells.get("pct_fga_layup")),
            "fg_layup": _safe_int(cells.get("fg_layup")),
            "pct_fg3a_corner": _safe_float(cells.get("pct_fg3a_corner")),
            "fg3_pct_corner": _safe_float(cells.get("fg3_pct_corner")),
        }

    out_path = BREF_CACHE_DIR / season / "team_shooting.json"
    out_path.write_text(json.dumps(teams_data, indent=2))
    print(f"  {season}: shooting splits for {len(teams_data)} teams")
    return teams_data


# ---------------------------------------------------------------------------
# 4. Opponent shooting splits (defensive shot profile)
# ---------------------------------------------------------------------------
def fetch_opponent_shooting(season: str, delay: float = REQUEST_DELAY) -> dict | None:
    """Fetch opponent shooting splits (defensive shot profile) from BRef."""
    end_year = _season_end_year(season)
    cache_dir = BREF_CACHE_DIR / season / "team_stats"
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/leagues/NBA_{end_year}.html"
    cache_path = cache_dir / "league_page.html"

    if cache_path.exists():
        html = cache_path.read_text(encoding="utf-8")
    else:
        html = _fetch_page(url, cache_path, delay=delay)

    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    _uncomment_tables(soup)

    opp_shooting_table = soup.find("table", id="shooting-opponent")
    if opp_shooting_table is None:
        print(f"  {season}: opponent shooting table not found")
        return None

    teams_data = {}
    tbody = opp_shooting_table.find("tbody")
    if tbody is None:
        return None

    for row in tbody.find_all("tr"):
        if "thead" in row.get("class", []):
            continue

        team_cell = row.find("td", {"data-stat": "team"})
        if team_cell is None:
            continue
        team_link = team_cell.find("a")
        if team_link is None:
            continue
        href = team_link.get("href", "")
        m = re.search(r"/teams/(\w+)/", href)
        if m is None:
            continue
        team_abbr = _our_team(m.group(1))

        cells = {td.get("data-stat"): td.get_text(strip=True) for td in row.find_all("td")}

        teams_data[team_abbr] = {
            "opp_avg_dist": _safe_float(cells.get("opp_avg_dist")),
            "opp_pct_fga_2p": _safe_float(cells.get("opp_pct_fga_fg2a")),
            "opp_pct_fga_0_3": _safe_float(cells.get("opp_pct_fga_00_03")),
            "opp_pct_fga_3_10": _safe_float(cells.get("opp_pct_fga_03_10")),
            "opp_pct_fga_10_16": _safe_float(cells.get("opp_pct_fga_10_16")),
            "opp_pct_fga_16_3pt": _safe_float(cells.get("opp_pct_fga_16_xx")),
            "opp_pct_fga_3p": _safe_float(cells.get("opp_pct_fga_fg3a")),
            "opp_fg_pct_0_3": _safe_float(cells.get("opp_fg_pct_00_03")),
            "opp_fg_pct_3_10": _safe_float(cells.get("opp_fg_pct_03_10")),
            "opp_fg_pct_10_16": _safe_float(cells.get("opp_fg_pct_10_16")),
            "opp_fg_pct_16_3pt": _safe_float(cells.get("opp_fg_pct_16_xx")),
            "opp_pct_ast_fg2": _safe_float(cells.get("opp_pct_ast_fg2")),
            "opp_pct_ast_fg3": _safe_float(cells.get("opp_pct_ast_fg3")),
            "opp_pct_fga_dunk": _safe_float(cells.get("opp_pct_fga_dunk")),
            "opp_pct_fga_layup": _safe_float(cells.get("opp_pct_fga_layup")),
            "opp_pct_fg3a_corner": _safe_float(cells.get("opp_pct_fg3a_corner")),
            "opp_fg3_pct_corner": _safe_float(cells.get("opp_fg3_pct_corner")),
        }

    out_path = BREF_CACHE_DIR / season / "opponent_shooting.json"
    out_path.write_text(json.dumps(teams_data, indent=2))
    print(f"  {season}: opponent shooting splits for {len(teams_data)} teams")
    return teams_data


# ---------------------------------------------------------------------------
# 5. Incremental daily fetch — fetch box scores for a single date
# ---------------------------------------------------------------------------
def _date_to_season(date_str: str) -> str:
    """Map a date (YYYY-MM-DD or YYYYMMDD) to NBA season code (e.g. '2024-25').

    NBA season starts in October. Oct-Dec = first year, Jan-Jun = second year.
    """
    d = date_str.replace("-", "")
    year = int(d[:4])
    month = int(d[4:6])
    if month >= 10:
        # Oct-Dec: season starts this year
        start_year = year
    else:
        # Jan-Sep: season started previous year
        start_year = year - 1
    end_yr = (start_year + 1) % 100
    return f"{start_year}-{end_yr:02d}"


def fetch_date_boxscores(
    date_str: str, delay: float = REQUEST_DELAY
) -> list[dict]:
    """Fetch box scores for a single date. Used for daily incremental updates.

    Args:
        date_str: Date in YYYYMMDD or YYYY-MM-DD format.
        delay: Seconds between requests.

    Returns list of parsed box score dicts.
    """
    # Normalize date
    d = date_str.replace("-", "")
    yyyy_mm_dd = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    season = _date_to_season(d)

    cache_dir = BREF_CACHE_DIR / season / "boxscores"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = BREF_CACHE_DIR / season / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    # Fetch the BRef daily scoreboard page to find game links
    url = f"{BASE_URL}/boxscores/?month={d[4:6]}&day={d[6:8]}&year={d[:4]}"
    date_cache = BREF_CACHE_DIR / season / "schedule" / f"day_{d}.html"
    date_cache.parent.mkdir(parents=True, exist_ok=True)

    html = _fetch_page(url, date_cache, delay=delay)
    if html is None:
        print(f"  No BRef scoreboard for {yyyy_mm_dd}")
        return []

    soup = BeautifulSoup(html, "html.parser")
    # Find box score links on the daily page
    box_links = []
    for a in soup.find_all("a", href=re.compile(r"/boxscores/\d{9}\w+\.html")):
        href = a.get("href", "")
        if href and href not in box_links:
            box_links.append(href)

    if not box_links:
        print(f"  No games found on BRef for {yyyy_mm_dd}")
        return []

    results = []
    for href in box_links:
        bref_id = href.split("/")[-1].replace(".html", "")
        cache_path = cache_dir / f"{bref_id}.html"
        parsed_path = parsed_dir / f"{bref_id}.json"

        # Skip if already parsed
        if parsed_path.exists():
            try:
                data = json.loads(parsed_path.read_text())
                if data:
                    results.append(data)
                    continue
            except json.JSONDecodeError:
                pass

        page_url = f"{BASE_URL}{href}"
        page_html = _fetch_page(page_url, cache_path, delay=delay)
        if page_html is None or page_html.strip() == "":
            continue

        data = parse_boxscore(page_html, yyyy_mm_dd)
        if data:
            data["bref_id"] = bref_id
            parsed_path.write_text(json.dumps(data, indent=2))
            results.append(data)

    print(f"  {yyyy_mm_dd}: {len(results)}/{len(box_links)} box scores parsed")

    # Re-export CSV for this season to include the new games
    if results:
        export_csv(season)

    return results


# ---------------------------------------------------------------------------
# 6. CSV export — flatten parsed boxscores for downstream pipelines
# ---------------------------------------------------------------------------
def export_csv(season: str) -> None:
    """Export parsed box scores to CSV files for easy pipeline consumption."""
    parsed_dir = BREF_CACHE_DIR / season / "parsed"
    if not parsed_dir.exists():
        print(f"  {season}: no parsed data to export")
        return

    export_dir = BREF_CACHE_DIR / season / "csv"
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Player game logs CSV
    player_rows = []
    starter_rows = []

    for json_file in sorted(parsed_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except json.JSONDecodeError:
            continue
        if not data:
            continue

        game_date = data.get("game_date", "")
        home = data.get("home", "")
        visitor = data.get("visitor", "")
        bref_id = data.get("bref_id", json_file.stem)

        for p in data.get("players", []):
            row = {
                "game_date": game_date,
                "bref_id": bref_id,
                "home_team": home,
                "away_team": visitor,
                **p,
            }
            player_rows.append(row)

        # Starters
        for name in data.get("home_starters", []):
            starter_rows.append({
                "game_date": game_date,
                "bref_id": bref_id,
                "team": home,
                "opponent": visitor,
                "player_name": name,
                "is_home": True,
            })
        for name in data.get("visitor_starters", []):
            starter_rows.append({
                "game_date": game_date,
                "bref_id": bref_id,
                "team": visitor,
                "opponent": home,
                "player_name": name,
                "is_home": False,
            })

    if player_rows:
        fieldnames = list(player_rows[0].keys())
        csv_path = export_dir / "player_game_logs.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(player_rows)
        print(f"  {season}: exported {len(player_rows)} player game log rows to {csv_path.name}")

    if starter_rows:
        csv_path = export_dir / "starters.csv"
        fieldnames = list(starter_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(starter_rows)
        print(f"  {season}: exported {len(starter_rows)} starter rows to {csv_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fetch NBA data from Basketball Reference"
    )
    parser.add_argument(
        "--seasons", nargs="+", default=[],
        help="Seasons to fetch (e.g., 2024-25 2023-24)"
    )
    parser.add_argument(
        "--all-seasons", action="store_true",
        help="Fetch all supported seasons (2021-22 through 2024-25)"
    )
    parser.add_argument(
        "--only", choices=["boxscores", "opponent-stats", "shooting"],
        help="Fetch only a specific data type"
    )
    parser.add_argument(
        "--export-csv", action="store_true",
        help="Export parsed data to CSV after fetching"
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY,
        help=f"Delay between requests in seconds (default: {REQUEST_DELAY})"
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Fetch box scores for a single date (YYYYMMDD). For daily cron use."
    )
    args = parser.parse_args()

    # --date mode: incremental daily fetch
    if args.date:
        BREF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Fetching BRef box scores for {args.date}...")
        results = fetch_date_boxscores(args.date, delay=args.delay)
        print(f"Done! {len(results)} games fetched.")
        sys.exit(0)

    if args.all_seasons:
        seasons = SUPPORTED_SEASONS
    elif args.seasons:
        seasons = args.seasons
    else:
        print("Error: specify --seasons, --all-seasons, or --date YYYYMMDD")
        parser.print_help()
        sys.exit(1)

    # Validate seasons
    for s in seasons:
        if not re.match(r"^\d{4}-\d{2}$", s):
            print(f"Error: invalid season format '{s}' (expected YYYY-YY)")
            sys.exit(1)

    BREF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        print(f"\n{'='*60}")
        print(f"Season: {season}")
        print(f"{'='*60}")

        if args.only is None or args.only == "boxscores":
            # Step 1: Fetch schedule
            sched_path = BREF_CACHE_DIR / season / "schedule.json"
            if sched_path.exists():
                games = json.loads(sched_path.read_text())
                print(f"  Using cached schedule ({len(games)} games)")
            else:
                print("  Fetching schedule...")
                games = fetch_schedule(season, delay=args.delay)

            # Step 2: Fetch box scores
            if games:
                print(f"  Fetching box scores...")
                results = fetch_boxscores(season, games, delay=args.delay)

        if args.only is None or args.only == "opponent-stats":
            # Step 3: Opponent stats
            print("  Fetching opponent stats...")
            fetch_opponent_stats(season, delay=args.delay)

        if args.only is None or args.only == "shooting":
            # Step 4: Shooting splits
            print("  Fetching team shooting splits...")
            fetch_team_shooting(season, delay=args.delay)
            print("  Fetching opponent shooting splits...")
            fetch_opponent_shooting(season, delay=args.delay)

        if args.export_csv:
            print("  Exporting to CSV...")
            export_csv(season)

    print(f"\nDone! Cache stored in {BREF_CACHE_DIR}")


if __name__ == "__main__":
    main()
