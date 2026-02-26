from __future__ import annotations

import concurrent.futures as cf
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor


SEASON = "2025-26"
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
BOXSCORE_URL_TMPL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={yyyymmdd}"
ESPN_ODDS_LIST_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event_id}"
    "/competitions/{event_id}/odds?lang=en&region=us"
)

OUT_DIR = Path("analysis/output")
CACHE_DIR = OUT_DIR / "nba_2025_26_advanced_cache"
BOXSCORE_CACHE = CACHE_DIR / "boxscores"
ESPN_SB_CACHE = CACHE_DIR / "espn_scoreboards"
ESPN_ODDS_CACHE = CACHE_DIR / "espn_odds"

ADV_TEAM_GAMES_CSV = OUT_DIR / "nba_2025_26_advanced_team_games.csv"
ADV_GAMES_CSV = OUT_DIR / "nba_2025_26_advanced_games.csv"
ADV_SUMMARY_JSON = OUT_DIR / "nba_2025_26_advanced_summary.json"
ADV_PLAYER_GAMES_CSV = OUT_DIR / "nba_2025_26_player_games.csv"
ADV_ODDS_CSV = OUT_DIR / "nba_2025_26_espn_odds.csv"


TEAM_COORDS = {
    "ATL": (33.7573, -84.3963),
    "BOS": (42.3662, -71.0621),
    "BKN": (40.6827, -73.9751),
    "CHA": (35.2251, -80.8392),
    "CHI": (41.8807, -87.6742),
    "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8103),
    "DEN": (39.7487, -105.0077),
    "DET": (42.3410, -83.0551),
    "GSW": (37.7680, -122.3877),
    "HOU": (29.7508, -95.3621),
    "IND": (39.7639, -86.1555),
    "LAC": (34.0430, -118.2673),
    "LAL": (34.0430, -118.2673),
    "MEM": (35.1382, -90.0506),
    "MIA": (25.7814, -80.1870),
    "MIL": (43.0451, -87.9172),
    "MIN": (44.9795, -93.2760),
    "NOP": (29.9490, -90.0821),
    "NYK": (40.7505, -73.9934),
    "OKC": (35.4634, -97.5151),
    "ORL": (28.5392, -81.3839),
    "PHI": (39.9012, -75.1720),
    "PHX": (33.4458, -112.0712),
    "POR": (45.5316, -122.6668),
    "SAC": (38.5802, -121.4997),
    "SAS": (29.4270, -98.4375),
    "TOR": (43.6435, -79.3791),
    "UTA": (40.7683, -111.9011),
    "WAS": (38.8981, -77.0209),
}

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


def _to_float(v: Any) -> float:
    if v is None or v == "":
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace("%", ""))
        except ValueError:
            return np.nan
    return np.nan


def _safe_div(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def _minutes_to_float(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except Exception:
        if isinstance(v, str) and v.startswith("PT"):
            # ISO-ish format like PT32M14.00S
            mins = 0.0
            sec = 0.0
            rest = v[2:]
            if "M" in rest:
                m, rest = rest.split("M", 1)
                mins = float(m or 0)
            if "S" in rest:
                s = rest.replace("S", "")
                sec = float(s or 0)
            return mins + sec / 60.0
        return 0.0


def american_to_prob(odds: float | int | None) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def normalize_prob_pair(p1: float, p2: float) -> tuple[float, float]:
    if pd.isna(p1) or pd.isna(p2) or p1 + p2 == 0:
        return np.nan, np.nan
    s = p1 + p2
    return p1 / s, p2 / s


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fetch_json(url: str, cache_path: Path | None = None, timeout: int = 30, retries: int = 3) -> dict[str, Any]:
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text())
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            payload = r.json()
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload))
            return payload
        except Exception as exc:  # pragma: no cover
            last_err = exc
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def normalize_espn_abbr(abbr: str) -> str:
    if not abbr:
        return abbr
    a = abbr.upper()
    return ESPN_ABBR_MAP.get(a, a)


def fetch_schedule_df() -> pd.DataFrame:
    payload = fetch_json(SCHEDULE_URL, cache_path=CACHE_DIR / "nba_schedule_2025_26.json")
    rows: list[dict[str, Any]] = []
    for day in payload["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            gid = str(g.get("gameId", ""))
            if not gid.startswith("002"):
                continue
            if int(g.get("gameStatus", 0)) != 3:
                continue
            rows.append(
                {
                    "game_id": gid,
                    "game_date": day["gameDate"],
                    "game_time_utc": g.get("gameDateTimeUTC") or g.get("gameDateUTC"),
                    "game_date_est": g.get("gameDateEst"),
                    "game_code": g.get("gameCode"),
                    "arena_name": g.get("arenaName"),
                    "arena_city": g.get("arenaCity"),
                    "arena_state": g.get("arenaState"),
                    "home_team": g["homeTeam"]["teamTricode"],
                    "away_team": g["awayTeam"]["teamTricode"],
                    "home_score_sched": g["homeTeam"].get("score"),
                    "away_score_sched": g["awayTeam"].get("score"),
                }
            )
    df = pd.DataFrame(rows)
    df["game_time_utc"] = pd.to_datetime(df["game_time_utc"], utc=True)
    df["game_date_est"] = pd.to_datetime(df["game_date_est"]).dt.strftime("%Y%m%d")
    return df.sort_values(["game_time_utc", "game_id"]).reset_index(drop=True)


def parse_team_box_rows(game: dict[str, Any]) -> list[dict[str, Any]]:
    game_id = str(game["gameId"])
    game_time_utc = game.get("gameTimeUTC")
    home = game["homeTeam"]
    away = game["awayTeam"]

    def parse_side(team: dict[str, Any], opp: dict[str, Any], is_home: bool) -> dict[str, Any]:
        stats = team.get("statistics", {}) or {}
        opp_stats = opp.get("statistics", {}) or {}
        fga = _to_float(stats.get("fieldGoalsAttempted"))
        fgm = _to_float(stats.get("fieldGoalsMade"))
        fg3a = _to_float(stats.get("threePointersAttempted"))
        fg3m = _to_float(stats.get("threePointersMade"))
        fta = _to_float(stats.get("freeThrowsAttempted"))
        ftm = _to_float(stats.get("freeThrowsMade"))
        orb = _to_float(stats.get("reboundsOffensive"))
        drb = _to_float(stats.get("reboundsDefensive"))
        trb = _to_float(stats.get("reboundsTotal"))
        tov = _to_float(stats.get("turnoversTotal"))
        ast = _to_float(stats.get("assists"))
        stl = _to_float(stats.get("steals"))
        blk = _to_float(stats.get("blocks"))
        pf = _to_float(stats.get("foulsPersonal"))
        pts = _to_float(stats.get("points"))

        opp_fga = _to_float(opp_stats.get("fieldGoalsAttempted"))
        opp_fta = _to_float(opp_stats.get("freeThrowsAttempted"))
        opp_orb = _to_float(opp_stats.get("reboundsOffensive"))
        opp_drb = _to_float(opp_stats.get("reboundsDefensive"))
        opp_tov = _to_float(opp_stats.get("turnoversTotal"))
        opp_pts = _to_float(opp_stats.get("points"))
        poss = 0.5 * ((fga + 0.44 * fta - orb + tov) + (opp_fga + 0.44 * opp_fta - opp_orb + opp_tov))

        efg = _safe_div(fgm + 0.5 * fg3m, fga)
        ft_rate = _safe_div(fta, fga)
        tov_rate = _safe_div(tov, poss)
        orb_rate = _safe_div(orb, orb + opp_drb)
        drb_rate = _safe_div(drb, drb + opp_orb)
        ts_pct = _safe_div(pts, 2 * (fga + 0.44 * fta))
        off_rating = _safe_div(100 * pts, poss)
        def_rating = _safe_div(100 * opp_pts, poss)
        net_rating = off_rating - def_rating if not (pd.isna(off_rating) or pd.isna(def_rating)) else np.nan

        return {
            "game_id": game_id,
            "game_time_utc": game_time_utc,
            "team": team["teamTricode"],
            "opp": opp["teamTricode"],
            "is_home": int(is_home),
            "team_score": pts,
            "opp_score": opp_pts,
            "win": int(pts > opp_pts),
            "margin": pts - opp_pts,
            "fga": fga,
            "fgm": fgm,
            "fg3a": fg3a,
            "fg3m": fg3m,
            "fta": fta,
            "ftm": ftm,
            "orb": orb,
            "drb": drb,
            "trb": trb,
            "ast": ast,
            "stl": stl,
            "blk": blk,
            "pf": pf,
            "tov": tov,
            "possessions": poss,
            "efg": efg,
            "ts_pct": ts_pct,
            "ft_rate": ft_rate,
            "tov_rate": tov_rate,
            "orb_rate": orb_rate,
            "drb_rate": drb_rate,
            "three_pa_rate": _safe_div(fg3a, fga),
            "off_rating": off_rating,
            "def_rating": def_rating,
            "net_rating": net_rating,
        }

    return [parse_side(home, away, True), parse_side(away, home, False)]


def parse_player_box_rows(game: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    game_id = str(game["gameId"])
    game_time_utc = game.get("gameTimeUTC")
    for side_key in ("homeTeam", "awayTeam"):
        team = game[side_key]
        team_code = team["teamTricode"]
        opp_code = game["awayTeam"]["teamTricode"] if side_key == "homeTeam" else game["homeTeam"]["teamTricode"]
        is_home = int(side_key == "homeTeam")
        for p in team.get("players", []):
            stats = p.get("statistics") or {}
            played_flag = str(p.get("played", "0")) == "1"
            minutes = _minutes_to_float(stats.get("minutesCalculated") or stats.get("minutes"))
            rows.append(
                {
                    "game_id": game_id,
                    "game_time_utc": game_time_utc,
                    "team": team_code,
                    "opp": opp_code,
                    "is_home": is_home,
                    "player_id": int(p.get("personId")) if p.get("personId") else np.nan,
                    "player_name": f"{p.get('firstName','').strip()} {p.get('familyName','').strip()}".strip() or p.get("nameI"),
                    "position": p.get("position"),
                    "starter": int(str(p.get("starter", "0")) == "1"),
                    "played": int(played_flag),
                    "status": p.get("status"),
                    "minutes": minutes,
                    "points": _to_float(stats.get("points")) if played_flag else 0.0,
                    "rebounds": _to_float(stats.get("reboundsTotal")) if played_flag else 0.0,
                    "assists": _to_float(stats.get("assists")) if played_flag else 0.0,
                    "plus_minus": _to_float(stats.get("plusMinusPoints")) if played_flag else 0.0,
                }
            )
    return rows


def fetch_boxscore_payload(game_id: str) -> dict[str, Any]:
    return fetch_json(
        BOXSCORE_URL_TMPL.format(game_id=game_id),
        cache_path=BOXSCORE_CACHE / f"{game_id}.json",
        timeout=30,
        retries=4,
    )


def fetch_all_boxscores(schedule_df: pd.DataFrame, max_workers: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_ids = schedule_df["game_id"].tolist()
    team_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_boxscore_payload, gid): gid for gid in game_ids}
        done = 0
        for fut in cf.as_completed(futs):
            payload = fut.result()
            game = payload["game"]
            team_rows.extend(parse_team_box_rows(game))
            player_rows.extend(parse_player_box_rows(game))
            done += 1
            if done % 100 == 0 or done == len(game_ids):
                print(f"Boxscores {done}/{len(game_ids)}", flush=True)
    team_df = pd.DataFrame(team_rows).drop_duplicates(subset=["game_id", "team"])
    player_df = pd.DataFrame(player_rows).drop_duplicates(subset=["game_id", "team", "player_id"])
    team_df["game_time_utc"] = pd.to_datetime(team_df["game_time_utc"], utc=True)
    player_df["game_time_utc"] = pd.to_datetime(player_df["game_time_utc"], utc=True)
    return team_df, player_df


def add_rest_and_rolling_team_features(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.copy().sort_values(["team", "game_time_utc", "game_id"]).reset_index(drop=True)
    prev_time = df.groupby("team")["game_time_utc"].shift(1)
    df["days_since_prev"] = (df["game_time_utc"] - prev_time).dt.total_seconds() / 86400.0
    df["b2b"] = ((df["days_since_prev"] > 0) & (df["days_since_prev"] < 1.6)).astype(int)
    df["three_in_four"] = (
        ((df["game_time_utc"] - df.groupby("team")["game_time_utc"].shift(2)).dt.total_seconds() / 86400.0) < 4.1
    ).fillna(False).astype(int)
    df["four_in_six"] = (
        ((df["game_time_utc"] - df.groupby("team")["game_time_utc"].shift(3)).dt.total_seconds() / 86400.0) < 6.1
    ).fillna(False).astype(int)

    roll_cols = [
        "team_score",
        "opp_score",
        "margin",
        "possessions",
        "off_rating",
        "def_rating",
        "net_rating",
        "efg",
        "ts_pct",
        "ft_rate",
        "tov_rate",
        "orb_rate",
        "three_pa_rate",
        "ast",
        "tov",
    ]
    for col in roll_cols:
        grp = df.groupby("team")[col]
        df[f"pre_{col}_avg5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        df[f"pre_{col}_avg10"] = grp.transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
        df[f"pre_{col}_season"] = grp.transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
    return df


def add_travel_features(team_games: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    sched = schedule_df[["game_id", "home_team", "away_team"]].copy()
    sched["venue_team"] = sched["home_team"]
    df = team_games.merge(sched[["game_id", "venue_team"]], on="game_id", how="left")
    coords = pd.DataFrame(
        [{"venue_team": t, "venue_lat": lat, "venue_lon": lon} for t, (lat, lon) in TEAM_COORDS.items()]
    )
    df = df.merge(coords, on="venue_team", how="left")
    df = df.sort_values(["team", "game_time_utc", "game_id"]).reset_index(drop=True)
    df["prev_venue_lat"] = df.groupby("team")["venue_lat"].shift(1)
    df["prev_venue_lon"] = df.groupby("team")["venue_lon"].shift(1)
    dists = []
    for r in df[["prev_venue_lat", "prev_venue_lon", "venue_lat", "venue_lon"]].itertuples(index=False):
        if any(pd.isna(x) for x in r):
            dists.append(np.nan)
        else:
            dists.append(haversine_miles(r[0], r[1], r[2], r[3]))
    df["travel_miles_since_prev"] = dists
    df["travel_500_plus"] = (df["travel_miles_since_prev"] >= 500).fillna(False).astype(int)
    df["travel_1000_plus"] = (df["travel_miles_since_prev"] >= 1000).fillna(False).astype(int)
    df["b2b_travel_500_plus"] = ((df["b2b"] == 1) & (df["travel_500_plus"] == 1)).astype(int)

    # Home/road streak counters
    home_streak = []
    road_streak = []
    for _, grp in df.groupby("team", sort=False):
        hs = 0
        rs = 0
        for v in grp["is_home"].astype(int):
            if v == 1:
                hs += 1
                rs = 0
            else:
                rs += 1
                hs = 0
            home_streak.append(hs)
            road_streak.append(rs)
    df["home_stand_game_num"] = home_streak
    df["road_trip_game_num"] = road_streak
    df["road_trip_3_plus"] = (df["road_trip_game_num"] >= 3).astype(int)
    return df


def add_player_availability_proxy(player_games: pd.DataFrame) -> pd.DataFrame:
    pg = player_games.copy().sort_values(["team", "player_id", "game_time_utc", "game_id"]).reset_index(drop=True)
    for col in ["minutes", "points", "rebounds", "assists", "plus_minus", "played", "starter"]:
        grp = pg.groupby(["team", "player_id"])[col]
        if col in {"played", "starter"}:
            pg[f"pre_{col}_avg5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            pg[f"pre_{col}_sum5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
        else:
            pg[f"pre_{col}_avg5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            pg[f"pre_{col}_avg10"] = grp.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    # Team-game level availability / injury proxy from absent players who recently mattered.
    pg["rotation_candidate"] = (
        (pg["pre_minutes_avg5"].fillna(0) >= 8.0) & (pg["pre_played_sum5"].fillna(0) >= 2)
    ).astype(int)
    pg["recent_starter_candidate"] = (pg["pre_starter_avg5"].fillna(0) >= 0.4).astype(int)
    pg["absent_rotation"] = ((pg["rotation_candidate"] == 1) & (pg["played"] == 0)).astype(int)
    pg["absent_recent_starter"] = ((pg["recent_starter_candidate"] == 1) & (pg["played"] == 0)).astype(int)

    # Weight absences by pregame production/minutes.
    pg["absent_weighted_min"] = np.where(pg["played"] == 0, pg["pre_minutes_avg5"].fillna(0), 0.0)
    pg["absent_weighted_pts"] = np.where(pg["played"] == 0, pg["pre_points_avg5"].fillna(0), 0.0)
    pg["absent_weighted_ast"] = np.where(pg["played"] == 0, pg["pre_assists_avg5"].fillna(0), 0.0)
    pg["absent_weighted_reb"] = np.where(pg["played"] == 0, pg["pre_rebounds_avg5"].fillna(0), 0.0)

    team_inj = (
        pg.groupby(["game_id", "team", "game_time_utc", "is_home", "opp"])
        .agg(
            roster_count=("player_id", "nunique"),
            active_count=("played", "sum"),
            starters_active=("starter", "sum"),
            injury_proxy_absent_rotation_count=("absent_rotation", "sum"),
            injury_proxy_absent_recent_starter_count=("absent_recent_starter", "sum"),
            injury_proxy_missing_minutes5=("absent_weighted_min", "sum"),
            injury_proxy_missing_points5=("absent_weighted_pts", "sum"),
            injury_proxy_missing_assists5=("absent_weighted_ast", "sum"),
            injury_proxy_missing_rebounds5=("absent_weighted_reb", "sum"),
            injury_proxy_max_missing_minutes5=("absent_weighted_min", "max"),
        )
        .reset_index()
    )
    team_inj["injury_proxy_max_missing_minutes5"] = team_inj["injury_proxy_max_missing_minutes5"].fillna(0)
    return pg, team_inj


def fetch_espn_scoreboard_for_date(date_str: str) -> dict[str, Any]:
    return fetch_json(
        ESPN_SCOREBOARD_URL.format(yyyymmdd=date_str),
        cache_path=ESPN_SB_CACHE / f"{date_str}.json",
        timeout=20,
        retries=3,
    )


def fetch_espn_odds_for_event(event_id: str) -> dict[str, Any] | None:
    try:
        return fetch_json(
            ESPN_ODDS_LIST_URL.format(event_id=event_id),
            cache_path=ESPN_ODDS_CACHE / f"{event_id}.json",
            timeout=20,
            retries=3,
        )
    except Exception:
        return None


def extract_espn_events(schedule_df: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(schedule_df["game_date_est"].dropna().unique().tolist())
    event_rows: list[dict[str, Any]] = []
    for d in dates:
        sb = fetch_espn_scoreboard_for_date(str(d))
        for e in sb.get("events", []):
            comp = (e.get("competitions") or [{}])[0]
            comps = comp.get("competitors", [])
            home = next((c for c in comps if c.get("homeAway") == "home"), None)
            away = next((c for c in comps if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            event_rows.append(
                {
                    "espn_event_id": str(e.get("id")),
                    "game_date_est": str(d),
                    "home_team": normalize_espn_abbr(home.get("team", {}).get("abbreviation", "")),
                    "away_team": normalize_espn_abbr(away.get("team", {}).get("abbreviation", "")),
                    "espn_game_name": e.get("name"),
                    "espn_start": e.get("date"),
                }
            )
    return pd.DataFrame(event_rows).drop_duplicates(subset=["game_date_est", "home_team", "away_team"])


def _deep_get(d: dict[str, Any] | None, path: list[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _parse_spread_american(node: dict[str, Any] | None) -> float:
    v = _deep_get(node, ["pointSpread", "american"])
    return _to_float(v)


def _parse_total_american(node: dict[str, Any] | None) -> float:
    v = _deep_get(node, ["total", "american"])
    return _to_float(v)


def fetch_all_espn_odds(events_df: pd.DataFrame, max_workers: int = 16) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    ids = events_df["espn_event_id"].astype(str).unique().tolist()
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_espn_odds_for_event, eid): eid for eid in ids}
        done = 0
        for fut in cf.as_completed(futs):
            eid = futs[fut]
            payload = fut.result()
            done += 1
            if done % 100 == 0 or done == len(ids):
                print(f"ESPN odds {done}/{len(ids)}", flush=True)
            if not payload or not payload.get("items"):
                continue
            item = payload["items"][0]  # provider priority 1 typically DraftKings
            provider = item.get("provider", {})
            close = item.get("close", {}) or {}
            open_ = item.get("open", {}) or {}
            current = item.get("current", {}) or {}
            home_odds = item.get("homeTeamOdds", {}) or {}
            away_odds = item.get("awayTeamOdds", {}) or {}
            hml = _to_float(home_odds.get("moneyLine"))
            aml = _to_float(away_odds.get("moneyLine"))
            hip_raw = american_to_prob(hml)
            aip_raw = american_to_prob(aml)
            hip, aip = normalize_prob_pair(hip_raw, aip_raw)
            rows.append(
                {
                    "espn_event_id": eid,
                    "odds_provider_id": provider.get("id"),
                    "odds_provider_name": provider.get("name"),
                    "market_details": item.get("details"),
                    "market_over_under_current": _to_float(item.get("overUnder")),
                    "market_spread_current": _to_float(item.get("spread")),
                    "market_home_ml_current": hml,
                    "market_away_ml_current": aml,
                    "market_home_spread_close": _parse_spread_american(home_odds.get("close")) or _parse_spread_american(close),
                    "market_away_spread_close": _parse_spread_american(away_odds.get("close")),
                    "market_home_spread_open": _parse_spread_american(home_odds.get("open")) or _parse_spread_american(open_),
                    "market_away_spread_open": _parse_spread_american(away_odds.get("open")),
                    "market_total_close": _parse_total_american(close),
                    "market_total_open": _parse_total_american(open_),
                    "market_total_current": _parse_total_american(current),
                    "market_home_ml_close": _to_float(_deep_get(home_odds, ["close", "moneyLine", "american"])) or hml,
                    "market_away_ml_close": _to_float(_deep_get(away_odds, ["close", "moneyLine", "american"])) or aml,
                    "market_home_implied_prob_close_raw": hip_raw,
                    "market_away_implied_prob_close_raw": aip_raw,
                    "market_home_implied_prob_close": hip,
                    "market_away_implied_prob_close": aip,
                }
            )
    return pd.DataFrame(rows)


def join_espn_odds(schedule_df: pd.DataFrame) -> pd.DataFrame:
    events = extract_espn_events(schedule_df)
    merged = schedule_df.merge(events, on=["game_date_est", "home_team", "away_team"], how="left")
    odds = fetch_all_espn_odds(merged[["espn_event_id"]].dropna().drop_duplicates())
    merged = merged.merge(odds, on="espn_event_id", how="left")
    return merged


def build_team_games_and_players() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schedule_df = fetch_schedule_df()
    print(f"Completed regular-season games: {len(schedule_df)}", flush=True)
    team_games, player_games = fetch_all_boxscores(schedule_df)
    team_games = team_games.merge(
        schedule_df[
            ["game_id", "game_time_utc", "game_date_est", "home_team", "away_team", "arena_city", "arena_state", "arena_name"]
        ],
        on=["game_id", "game_time_utc"],
        how="left",
    )
    team_games = add_rest_and_rolling_team_features(team_games)
    team_games = add_travel_features(team_games, schedule_df)
    player_games, team_inj = add_player_availability_proxy(player_games)
    team_games = team_games.merge(
        team_inj.drop(columns=["game_time_utc", "is_home", "opp"]),
        on=["game_id", "team"],
        how="left",
    )
    return schedule_df, team_games, player_games


def build_game_level(team_games: pd.DataFrame, schedule_with_odds: pd.DataFrame) -> pd.DataFrame:
    tg = team_games.copy().sort_values(["game_time_utc", "game_id", "team"])
    home = tg[tg["is_home"] == 1].copy()
    away = tg[tg["is_home"] == 0].copy()

    keep_cols = [
        "game_id",
        "game_time_utc",
        "team",
        "opp",
        "team_score",
        "opp_score",
        "win",
        "margin",
        "days_since_prev",
        "b2b",
        "three_in_four",
        "four_in_six",
        "travel_miles_since_prev",
        "travel_500_plus",
        "travel_1000_plus",
        "b2b_travel_500_plus",
        "home_stand_game_num",
        "road_trip_game_num",
        "road_trip_3_plus",
        "injury_proxy_absent_rotation_count",
        "injury_proxy_absent_recent_starter_count",
        "injury_proxy_missing_minutes5",
        "injury_proxy_missing_points5",
        "injury_proxy_missing_assists5",
        "injury_proxy_missing_rebounds5",
        "injury_proxy_max_missing_minutes5",
        "active_count",
        "roster_count",
    ] + [c for c in tg.columns if c.startswith("pre_")]
    home = home[keep_cols].rename(columns={c: f"home_{c}" for c in keep_cols if c not in ["game_id", "game_time_utc"]})
    away = away[keep_cols].rename(columns={c: f"away_{c}" for c in keep_cols if c not in ["game_id", "game_time_utc"]})
    games = home.merge(away, on=["game_id", "game_time_utc"], how="inner")
    games["home_win"] = games["home_win"].astype(int)
    games["total_points"] = games["home_team_score"] + games["away_team_score"]
    games["home_margin"] = games["home_margin"]

    # Differential engineered features.
    diff_metrics = [
        "net_rating",
        "off_rating",
        "def_rating",
        "efg",
        "ts_pct",
        "ft_rate",
        "tov_rate",
        "orb_rate",
        "three_pa_rate",
        "margin",
        "team_score",
        "opp_score",
        "possessions",
        "ast",
        "tov",
        "injury_proxy_missing_minutes5",
        "injury_proxy_missing_points5",
        "travel_miles_since_prev",
        "days_since_prev",
    ]
    for m in diff_metrics:
        for w in ("avg5", "avg10", "season"):
            hc = f"home_pre_{m}_{w}"
            ac = f"away_pre_{m}_{w}"
            if hc in games.columns and ac in games.columns:
                games[f"diff_pre_{m}_{w}"] = games[hc] - games[ac]
    # Direct diffs for non-rolling features.
    for c in [
        "days_since_prev",
        "travel_miles_since_prev",
        "injury_proxy_missing_minutes5",
        "injury_proxy_missing_points5",
        "active_count",
        "injury_proxy_absent_rotation_count",
        "injury_proxy_absent_recent_starter_count",
    ]:
        hc = f"home_{c}"
        ac = f"away_{c}"
        if hc in games.columns and ac in games.columns:
            games[f"diff_{c}"] = games[hc] - games[ac]

    games["rest_diff"] = games["home_days_since_prev"] - games["away_days_since_prev"]
    games["home_b2b_adv"] = games["away_b2b"] - games["home_b2b"]
    games["travel_diff"] = games["home_travel_miles_since_prev"] - games["away_travel_miles_since_prev"]

    odds_cols = [
        "game_id",
        "espn_event_id",
        "market_home_spread_close",
        "market_away_spread_close",
        "market_home_spread_open",
        "market_total_close",
        "market_total_open",
        "market_home_ml_close",
        "market_away_ml_close",
        "market_home_implied_prob_close",
        "market_away_implied_prob_close",
        "odds_provider_name",
    ]
    games = games.merge(schedule_with_odds[odds_cols].drop_duplicates("game_id"), on="game_id", how="left")
    games["market_spread_move_home"] = games["market_home_spread_close"] - games["market_home_spread_open"]
    games["market_total_move"] = games["market_total_close"] - games["market_total_open"]
    return games.sort_values("game_time_utc").reset_index(drop=True)


def chron_split(df: pd.DataFrame, frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("game_time_utc").reset_index(drop=True)
    cut = int(math.floor(len(df) * frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def chron_split_team(df: pd.DataFrame, frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("game_time_utc").reset_index(drop=True)
    cut = int(math.floor(len(df) * frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def eval_win_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str, use_xgb: bool = True) -> dict[str, Any]:
    X_train = train[features]
    X_test = test[features]
    y_train = train["home_win"]
    y_test = test["home_win"]
    if use_xgb:
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_train)
        Xte = imp.transform(X_test)
        model = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        importances = model.feature_importances_
    else:
        pipe = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=500))]
        )
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        importances = np.abs(pipe.named_steps["model"].coef_[0])
    fi = sorted(
        [{"feature": f, "importance": float(v)} for f, v in zip(features, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )[:12]
    return {
        "label": label,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "baseline_home_win_rate_test": float(y_test.mean()),
        "top_features": fi,
    }


def eval_total_model(train: pd.DataFrame, test: pd.DataFrame, features: list[str], label: str, use_xgb: bool = True) -> dict[str, Any]:
    X_train = train[features]
    X_test = test[features]
    y_train = train["total_points"]
    y_test = test["total_points"]
    if use_xgb:
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(X_train)
        Xte = imp.transform(X_test)
        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
        )
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        importances = model.feature_importances_
    else:
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        importances = np.zeros(len(features))
    return {
        "label": label,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "test_avg_total": float(y_test.mean()),
        "top_features": sorted(
            [{"feature": f, "importance": float(v)} for f, v in zip(features, importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )[:12],
    }


def eval_market_baselines(test_games: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    win_df = test_games.dropna(subset=["market_home_implied_prob_close"]).copy()
    if len(win_df):
        p = win_df["market_home_implied_prob_close"].clip(1e-6, 1 - 1e-6)
        pred = (p >= 0.5).astype(int)
        y = win_df["home_win"]
        out["market_win_baseline"] = {
            "n_test": int(len(win_df)),
            "accuracy": float(accuracy_score(y, pred)),
            "auc": float(roc_auc_score(y, p)),
            "log_loss": float(log_loss(y, p)),
        }
    total_df = test_games.dropna(subset=["market_total_close"]).copy()
    if len(total_df):
        y = total_df["total_points"]
        pred = total_df["market_total_close"]
        out["market_total_baseline"] = {
            "n_test": int(len(total_df)),
            "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)),
        }
    return out


def run_advanced_models(games: pd.DataFrame) -> dict[str, Any]:
    models: dict[str, Any] = {}

    base_win_features = [
        "diff_pre_net_rating_season",
        "diff_pre_net_rating_avg5",
        "diff_pre_off_rating_avg5",
        "diff_pre_def_rating_avg5",
        "diff_pre_efg_avg5",
        "diff_pre_tov_rate_avg5",
        "diff_pre_orb_rate_avg5",
        "diff_pre_ft_rate_avg5",
        "diff_pre_possessions_avg5",
        "diff_pre_margin_avg10",
        "rest_diff",
        "home_b2b",
        "away_b2b",
        "home_b2b_adv",
    ]
    enh_win_features = base_win_features + [
        "home_four_in_six",
        "away_four_in_six",
        "home_travel_miles_since_prev",
        "away_travel_miles_since_prev",
        "home_b2b_travel_500_plus",
        "away_b2b_travel_500_plus",
        "diff_injury_proxy_missing_minutes5",
        "diff_injury_proxy_missing_points5",
        "diff_injury_proxy_absent_rotation_count",
        "diff_injury_proxy_absent_recent_starter_count",
        "home_road_trip_game_num",
        "away_road_trip_game_num",
    ]
    market_win_features = enh_win_features + [
        "market_home_spread_close",
        "market_total_close",
        "market_home_implied_prob_close",
        "market_spread_move_home",
        "market_total_move",
    ]

    common_win = sorted(set(base_win_features + enh_win_features + market_win_features + ["home_win"]))
    win_df = games.dropna(subset=["home_win"]).copy()
    win_train, win_test = chron_split(win_df, 0.8)
    models["win_model_baseline"] = eval_win_model(win_train, win_test, base_win_features, "baseline_team_rolling")
    models["win_model_enhanced"] = eval_win_model(win_train, win_test, enh_win_features, "plus_travel_injury_proxy")
    market_win_df = games.dropna(subset=["market_home_implied_prob_close"]).copy()
    mtrain, mtest = chron_split(market_win_df, 0.8)
    models["win_model_with_market"] = eval_win_model(mtrain, mtest, market_win_features, "plus_market_lines")

    base_total_features = [
        "home_pre_possessions_avg5",
        "away_pre_possessions_avg5",
        "home_pre_off_rating_avg5",
        "away_pre_off_rating_avg5",
        "home_pre_def_rating_avg5",
        "away_pre_def_rating_avg5",
        "home_pre_efg_avg5",
        "away_pre_efg_avg5",
        "home_pre_tov_rate_avg5",
        "away_pre_tov_rate_avg5",
        "home_pre_ft_rate_avg5",
        "away_pre_ft_rate_avg5",
        "home_b2b",
        "away_b2b",
        "rest_diff",
    ]
    enh_total_features = base_total_features + [
        "home_four_in_six",
        "away_four_in_six",
        "home_travel_miles_since_prev",
        "away_travel_miles_since_prev",
        "home_b2b_travel_500_plus",
        "away_b2b_travel_500_plus",
        "home_injury_proxy_missing_minutes5",
        "away_injury_proxy_missing_minutes5",
        "home_injury_proxy_missing_points5",
        "away_injury_proxy_missing_points5",
        "home_active_count",
        "away_active_count",
    ]
    market_total_features = enh_total_features + [
        "market_total_close",
        "market_home_spread_close",
        "market_total_move",
    ]

    total_df = games.copy()
    ttrain, ttest = chron_split(total_df, 0.8)
    models["total_model_baseline"] = eval_total_model(ttrain, ttest, base_total_features, "baseline_team_rolling")
    models["total_model_enhanced"] = eval_total_model(ttrain, ttest, enh_total_features, "plus_travel_injury_proxy")
    market_total_df = games.dropna(subset=["market_total_close"]).copy()
    mttrain, mttest = chron_split(market_total_df, 0.8)
    models["total_model_with_market"] = eval_total_model(mttrain, mttest, market_total_features, "plus_market_lines")

    # Explicit market baselines on aligned test sets:
    models["market_baselines_win_test"] = eval_market_baselines(mtest)
    models["market_baselines_total_test"] = eval_market_baselines(mttest)
    return models


def summarize_data(schedule_odds: pd.DataFrame, team_games: pd.DataFrame, player_games: pd.DataFrame, games: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    summary["coverage"] = {
        "season": SEASON,
        "games": int(len(games)),
        "team_games": int(len(team_games)),
        "player_game_rows": int(len(player_games)),
        "utc_start": str(games["game_time_utc"].min()),
        "utc_end": str(games["game_time_utc"].max()),
    }
    summary["odds_coverage"] = {
        "games_with_espn_event_match": int(schedule_odds["espn_event_id"].notna().sum()),
        "games_with_market_total_close": int(schedule_odds["market_total_close"].notna().sum()),
        "games_with_market_home_ml_close": int(schedule_odds["market_home_ml_close"].notna().sum()),
        "provider_mode": schedule_odds["odds_provider_name"].dropna().mode().tolist()[:3],
    }
    summary["league_baseline"] = {
        "home_win_rate": float(games["home_win"].mean()),
        "avg_total_points": float(games["total_points"].mean()),
        "avg_home_points": float(games["home_team_score"].mean()),
        "avg_away_points": float(games["away_team_score"].mean()),
        "avg_team_possessions": float(team_games["possessions"].mean()),
        "avg_team_off_rating": float(team_games["off_rating"].mean()),
        "avg_team_efg": float(team_games["efg"].mean()),
        "avg_3pa_rate": float(team_games["three_pa_rate"].mean()),
    }
    travel = team_games["travel_miles_since_prev"].dropna()
    summary["travel"] = {
        "avg_travel_miles_since_prev": float(travel.mean()) if len(travel) else None,
        "median_travel_miles_since_prev": float(travel.median()) if len(travel) else None,
        "b2b_travel_500_plus_rate": float(team_games["b2b_travel_500_plus"].mean()),
    }
    inj_cols = [
        "injury_proxy_absent_rotation_count",
        "injury_proxy_absent_recent_starter_count",
        "injury_proxy_missing_minutes5",
        "injury_proxy_missing_points5",
    ]
    summary["injury_proxy"] = {
        col: float(team_games[col].fillna(0).mean()) for col in inj_cols if col in team_games.columns
    }

    # Practical bucket heuristics.
    buckets = {}
    valid = games.dropna(subset=["diff_pre_net_rating_season"]).copy()
    valid["nr_bin"] = pd.cut(valid["diff_pre_net_rating_season"], bins=[-999, -8, -4, -2, 0, 2, 4, 8, 999])
    nr_bucket_df = valid.groupby("nr_bin", observed=True)["home_win"].agg(["count", "mean"]).reset_index()
    nr_bucket_df["nr_bin"] = nr_bucket_df["nr_bin"].astype(str)
    buckets["home_win_by_pregame_net_rating_diff"] = nr_bucket_df.to_dict(orient="records")
    pace_df = games.dropna(subset=["home_pre_possessions_avg5", "away_pre_possessions_avg5"]).copy()
    pace_df["pre_pace_avg"] = (pace_df["home_pre_possessions_avg5"] + pace_df["away_pre_possessions_avg5"]) / 2
    pace_df["pace_quartile"] = pd.qcut(pace_df["pre_pace_avg"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    buckets["totals_by_pregame_pace_quartile"] = (
        pace_df.groupby("pace_quartile", observed=True)["total_points"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .to_dict(orient="records")
    )
    market_df = games.dropna(subset=["market_home_spread_close"]).copy()
    market_df["home_beat_spread"] = (market_df["home_margin"] > -market_df["market_home_spread_close"]).astype(int)
    summary["market_shape"] = {
        "avg_close_total": float(market_df["market_total_close"].dropna().mean()),
        "avg_close_home_spread": float(market_df["market_home_spread_close"].dropna().mean()),
        "home_cover_rate_vs_close_spread": float(market_df["home_beat_spread"].mean()) if len(market_df) else None,
    }
    summary["heuristic_buckets"] = buckets
    return summary


def print_results(summary: dict[str, Any], models: dict[str, Any]) -> None:
    print("\n=== ADVANCED COVERAGE ===")
    c = summary["coverage"]
    print(
        f"{c['season']} through {c['utc_end']}: {c['games']} games, {c['team_games']} team-games, "
        f"{c['player_game_rows']} player rows"
    )
    o = summary["odds_coverage"]
    print(
        f"Odds coverage: event matches {o['games_with_espn_event_match']}/{c['games']}, "
        f"totals {o['games_with_market_total_close']}, moneylines {o['games_with_market_home_ml_close']}"
    )
    print(f"Primary odds provider(s): {o['provider_mode']}")

    print("\n=== FEATURE CONTEXT ===")
    print(
        f"Travel avg miles between games: {summary['travel']['avg_travel_miles_since_prev']:.1f} "
        f"(median {summary['travel']['median_travel_miles_since_prev']:.1f}); "
        f"B2B+500mi rate: {summary['travel']['b2b_travel_500_plus_rate']:.3f}"
    )
    print("Injury/availability proxy averages (team-game):")
    for k, v in summary["injury_proxy"].items():
        print(f"  {k}: {v:.3f}")

    print("\n=== WIN MODELS (OUT-OF-SAMPLE) ===")
    for k in ["win_model_baseline", "win_model_enhanced", "win_model_with_market"]:
        m = models[k]
        print(
            f"{m['label']}: acc={m['accuracy']:.3f}, AUC={m['auc']:.3f}, logloss={m['log_loss']:.3f} "
            f"(n_test={m['n_test']})"
        )
    if "market_baselines_win_test" in models and "market_win_baseline" in models["market_baselines_win_test"]:
        m = models["market_baselines_win_test"]["market_win_baseline"]
        print(
            f"Market baseline (moneyline implied prob): acc={m['accuracy']:.3f}, AUC={m['auc']:.3f}, "
            f"logloss={m['log_loss']:.3f} (n_test={m['n_test']})"
        )

    print("\n=== TOTALS MODELS (OUT-OF-SAMPLE) ===")
    for k in ["total_model_baseline", "total_model_enhanced", "total_model_with_market"]:
        m = models[k]
        print(f"{m['label']}: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R2={m['r2']:.3f} (n_test={m['n_test']})")
    if "market_baselines_total_test" in models and "market_total_baseline" in models["market_baselines_total_test"]:
        m = models["market_baselines_total_test"]["market_total_baseline"]
        print(f"Market baseline (closing total): MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R2={m['r2']:.3f}")

    print("\n=== TOP WIN FEATURES (WITH MARKET) ===")
    for row in models["win_model_with_market"]["top_features"][:10]:
        print(f"  {row['feature']}: {row['importance']:.4f}")
    print("\n=== TOP TOTAL FEATURES (WITH MARKET) ===")
    for row in models["total_model_with_market"]["top_features"][:10]:
        print(f"  {row['feature']}: {row['importance']:.4f}")

    print("\n=== PRACTICAL BUCKETS ===")
    print("Pregame net rating diff buckets -> home win%:")
    for row in summary["heuristic_buckets"]["home_win_by_pregame_net_rating_diff"]:
        print(f"  {row['nr_bin']}: n={int(row['count'])}, win%={row['mean']:.3f}")
    print("Pregame pace quartile -> total points:")
    for row in summary["heuristic_buckets"]["totals_by_pregame_pace_quartile"]:
        print(
            f"  {row['pace_quartile']}: n={int(row['count'])}, avg={row['mean']:.1f}, median={row['median']:.1f}"
        )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BOXSCORE_CACHE.mkdir(parents=True, exist_ok=True)
    ESPN_SB_CACHE.mkdir(parents=True, exist_ok=True)
    ESPN_ODDS_CACHE.mkdir(parents=True, exist_ok=True)

    print("Building team/player game datasets with travel + availability proxy...", flush=True)
    schedule_df, team_games, player_games = build_team_games_and_players()
    print("Matching ESPN events and pulling odds...", flush=True)
    schedule_with_odds = join_espn_odds(schedule_df)

    games = build_game_level(team_games, schedule_with_odds)

    team_games.to_csv(ADV_TEAM_GAMES_CSV, index=False)
    player_games.to_csv(ADV_PLAYER_GAMES_CSV, index=False)
    games.to_csv(ADV_GAMES_CSV, index=False)
    schedule_with_odds.to_csv(ADV_ODDS_CSV, index=False)

    print("Training advanced models...", flush=True)
    models = run_advanced_models(games)
    summary = summarize_data(schedule_with_odds, team_games, player_games, games)

    payload = {"summary": summary, "models": models}
    ADV_SUMMARY_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print_results(summary, models)

    print(f"\nSaved: {ADV_TEAM_GAMES_CSV}")
    print(f"Saved: {ADV_PLAYER_GAMES_CSV}")
    print(f"Saved: {ADV_GAMES_CSV}")
    print(f"Saved: {ADV_ODDS_CSV}")
    print(f"Saved: {ADV_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
