from __future__ import annotations

import concurrent.futures as cf
import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import joblib
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

from nba_evaluate import (
    evaluate_win_model_comprehensive,
    evaluate_total_model_comprehensive,
    evaluate_cv_folds,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")

SEASON = "2025-26"
SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
BOXSCORE_URL_TMPL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={yyyymmdd}"
ESPN_ODDS_LIST_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{event_id}"
    "/competitions/{event_id}/odds?lang=en&region=us"
)
ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis" / "output"
CACHE_DIR = OUT_DIR / "nba_2025_26_advanced_cache"
HIST_CACHE_DIR = OUT_DIR / "historical_cache"
BOXSCORE_CACHE = CACHE_DIR / "boxscores"
ESPN_SB_CACHE = CACHE_DIR / "espn_scoreboards"
ESPN_ODDS_CACHE = CACHE_DIR / "espn_odds"
INJURY_REPORT_CACHE = CACHE_DIR / "injury_reports"
MODEL_DIR = OUT_DIR / "models"
MIN_ROI_BETS_PRINT = 20

ADV_TEAM_GAMES_CSV = OUT_DIR / "nba_2025_26_advanced_team_games.csv"
ADV_GAMES_CSV = OUT_DIR / "nba_2025_26_advanced_games.csv"
ADV_SUMMARY_JSON = OUT_DIR / "nba_2025_26_advanced_summary.json"
ADV_PLAYER_GAMES_CSV = OUT_DIR / "nba_2025_26_player_games.csv"
ADV_ODDS_CSV = OUT_DIR / "nba_2025_26_espn_odds.csv"

# Module-level session for HTTP connection pooling
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (nba-analysis)"})


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
    "LAC": (33.9192, -118.3381),  # Intuit Dome
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


def _first_valid(a: float, b: float) -> float:
    """Return *a* if it is a real number, otherwise *b*.  Handles NaN correctly."""
    return a if pd.notna(a) else b


def _nan_or(value: float, default: float) -> float:
    """Return *value* if it is a real number, otherwise *default*."""
    return value if pd.notna(value) else default


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
            r = _SESSION.get(url, timeout=timeout)
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


def load_historical_season(season: str) -> tuple[pd.DataFrame, list[dict], list[dict]]:
    """Load schedule, team-box rows, and player-box rows from historical cache.

    Returns (schedule_df, team_rows, player_rows) for a single historical season.
    """
    box_dir = HIST_CACHE_DIR / season / "boxscores"
    if not box_dir.exists():
        return pd.DataFrame(), [], []

    sched_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []

    for f in sorted(box_dir.glob("*.json")):
        try:
            payload = json.loads(f.read_text())
            game = payload["game"]
            gid = str(game["gameId"])
            game_status = int(game.get("gameStatus", 0))
            if game_status != 3:
                continue

            sched_rows.append({
                "game_id": gid,
                "season": season,
                "game_date": "",
                "game_time_utc": game.get("gameTimeUTC"),
                "game_date_est": (game.get("gameEt") or "")[:10].replace("-", "") or None,
                "game_code": game.get("gameCode"),
                "arena_name": game.get("arena", {}).get("arenaName"),
                "arena_city": game.get("arena", {}).get("arenaCity"),
                "arena_state": game.get("arena", {}).get("arenaState"),
                "home_team": game["homeTeam"]["teamTricode"],
                "away_team": game["awayTeam"]["teamTricode"],
                "home_score_sched": game["homeTeam"].get("score"),
                "away_score_sched": game["awayTeam"].get("score"),
            })
            team_rows.extend(parse_team_box_rows(game))
            player_rows.extend(parse_player_box_rows(game))
        except Exception:
            continue

    if not sched_rows:
        return pd.DataFrame(), [], []

    sched_df = pd.DataFrame(sched_rows)
    sched_df["game_time_utc"] = pd.to_datetime(sched_df["game_time_utc"], utc=True)
    sched_df = sched_df.sort_values(["game_time_utc", "game_id"]).reset_index(drop=True)
    return sched_df, team_rows, player_rows


def load_historical_espn_odds(season: str) -> pd.DataFrame:
    """Load ESPN event mapping and odds for a historical season."""
    mapping_path = HIST_CACHE_DIR / season / "espn_event_map.json"
    odds_dir = HIST_CACHE_DIR / season / "espn_odds"
    if not mapping_path.exists() or not odds_dir.exists():
        return pd.DataFrame()

    event_map = json.loads(mapping_path.read_text())
    odds_rows: list[dict[str, Any]] = []
    for entry in event_map:
        eid = entry["event_id"]
        odds_path = odds_dir / f"{eid}.json"
        if not odds_path.exists():
            continue
        try:
            payload = json.loads(odds_path.read_text())
            if not payload or not payload.get("items"):
                continue
            item = payload["items"][0]
            provider = item.get("provider", {})
            close = item.get("close", {}) or {}
            open_ = item.get("open", {}) or {}
            home_odds = item.get("homeTeamOdds", {}) or {}
            away_odds = item.get("awayTeamOdds", {}) or {}
            hml = _to_float(home_odds.get("moneyLine"))
            aml = _to_float(away_odds.get("moneyLine"))
            hip_raw = american_to_prob(hml)
            aip_raw = american_to_prob(aml)
            hip, aip = normalize_prob_pair(hip_raw, aip_raw)
            odds_rows.append({
                "espn_event_id": eid,
                "game_date_est": entry["date"],
                "home_team": entry["home"],
                "away_team": entry["away"],
                "odds_provider_name": provider.get("name"),
                "market_home_spread_close": _first_valid(_parse_spread_american(home_odds.get("close")), _parse_spread_american(close)),
                "market_home_spread_open": _first_valid(_parse_spread_american(home_odds.get("open")), _parse_spread_american(open_)),
                "market_total_close": _parse_total_american(close),
                "market_total_open": _parse_total_american(open_),
                "market_home_ml_close": _first_valid(_to_float(_deep_get(home_odds, ["close", "moneyLine", "american"])), hml),
                "market_away_ml_close": _first_valid(_to_float(_deep_get(away_odds, ["close", "moneyLine", "american"])), aml),
                "market_home_implied_prob_close": hip,
                "market_away_implied_prob_close": aip,
            })
        except Exception:
            continue
    return pd.DataFrame(odds_rows)


SCHEDULE_CACHE_TTL_HOURS = 6


def fetch_schedule_df() -> pd.DataFrame:
    sched_cache = CACHE_DIR / "nba_schedule_2025_26.json"
    # Invalidate schedule cache if older than TTL so new games are picked up
    if sched_cache.exists():
        age_hours = (time.time() - sched_cache.stat().st_mtime) / 3600.0
        if age_hours > SCHEDULE_CACHE_TTL_HOURS:
            sched_cache.unlink()
    payload = fetch_json(SCHEDULE_URL, cache_path=sched_cache)
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


def parse_officials(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract official (referee) info from a boxscore game dict.

    Returns list of dicts with personId, name, assignment for each official.
    """
    officials = game.get("officials", [])
    game_id = str(game["gameId"])
    game_time_utc = game.get("gameTimeUTC")
    rows: list[dict[str, Any]] = []
    for o in officials:
        rows.append({
            "game_id": game_id,
            "game_time_utc": game_time_utc,
            "ref_id": int(o.get("personId", 0)),
            "ref_name": o.get("name", ""),
            "ref_assignment": o.get("assignment", ""),
        })
    return rows


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
        failed = 0
        for fut in cf.as_completed(futs):
            gid = futs[fut]
            try:
                payload = fut.result()
                game = payload["game"]
                team_rows.extend(parse_team_box_rows(game))
                player_rows.extend(parse_player_box_rows(game))
            except Exception as exc:
                failed += 1
                print(f"  WARNING: boxscore fetch failed for {gid}: {exc}", flush=True)
            done += 1
            if done % 100 == 0 or done == len(game_ids):
                print(f"Boxscores {done}/{len(game_ids)}" + (f" ({failed} failed)" if failed else ""), flush=True)
    team_df = pd.DataFrame(team_rows).drop_duplicates(subset=["game_id", "team"])
    player_df = pd.DataFrame(player_rows).drop_duplicates(subset=["game_id", "team", "player_id"])
    team_df["game_time_utc"] = pd.to_datetime(team_df["game_time_utc"], utc=True)
    player_df["game_time_utc"] = pd.to_datetime(player_df["game_time_utc"], utc=True)
    return team_df, player_df


def add_rest_and_rolling_team_features(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.copy().sort_values(["team", "game_time_utc", "game_id"]).reset_index(drop=True)

    # Use (team, season) grouping so rolling features reset at season boundaries
    group_key = ["team", "season"] if "season" in df.columns else ["team"]

    prev_time = df.groupby(group_key)["game_time_utc"].shift(1)
    df["days_since_prev"] = (df["game_time_utc"] - prev_time).dt.total_seconds() / 86400.0
    df["b2b"] = ((df["days_since_prev"] > 0) & (df["days_since_prev"] < 1.6)).astype(int)
    df["three_in_four"] = (
        ((df["game_time_utc"] - df.groupby(group_key)["game_time_utc"].shift(2)).dt.total_seconds() / 86400.0) < 4.1
    ).fillna(False).astype(int)
    df["four_in_six"] = (
        ((df["game_time_utc"] - df.groupby(group_key)["game_time_utc"].shift(3)).dt.total_seconds() / 86400.0) < 6.1
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
        grp = df.groupby(group_key)[col]
        df[f"pre_{col}_avg5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        df[f"pre_{col}_avg10"] = grp.transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
        df[f"pre_{col}_season"] = grp.transform(lambda s: s.shift(1).expanding(min_periods=5).mean())

    # Exponentially weighted moving averages -- recent games weighted more heavily
    ewm_cols = ["net_rating", "off_rating", "def_rating", "margin", "efg", "ts_pct", "tov_rate", "possessions"]
    for col in ewm_cols:
        grp = df.groupby(group_key)[col]
        df[f"pre_{col}_ewm10"] = grp.transform(
            lambda s: s.shift(1).ewm(span=10, min_periods=5).mean()
        )

    # Win percentage entering the game (powerful baseline feature)
    win_grp_wp = df.groupby(group_key)["win"]
    df["pre_win_pct"] = win_grp_wp.transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    df["pre_win_pct_last10"] = win_grp_wp.transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())

    # Defensive rebounding rate (drb_rate) rolling averages
    drb_grp = df.groupby(group_key)["drb_rate"]
    df["pre_drb_rate_avg5"] = drb_grp.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    df["pre_drb_rate_avg10"] = drb_grp.transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())

    # Rolling standard deviation of margin (consistency measure)
    margin_grp = df.groupby(group_key)["margin"]
    df["pre_margin_std10"] = margin_grp.transform(
        lambda s: s.shift(1).rolling(10, min_periods=5).std()
    )

    # Phase 3A: Streak & Trend Features
    # Win/loss streak entering game
    win_grp = df.groupby(group_key)["win"]
    df["pre_win_streak"] = win_grp.transform(lambda s: _compute_streak(s.shift(1)))
    # Trend: improving vs declining (avg5 - avg10 for net rating)
    df["pre_net_rating_trend"] = df["pre_net_rating_avg5"] - df["pre_net_rating_avg10"]
    df["pre_margin_trend"] = df["pre_margin_avg5"] - df["pre_margin_avg10"]
    # Home/away-specific rolling averages
    home_mask = df["is_home"] == 1
    away_mask = df["is_home"] == 0
    for col in ["net_rating", "margin"]:
        # Home-only rolling average
        df[f"pre_{col}_home_avg5"] = np.nan
        df[f"pre_{col}_away_avg5"] = np.nan
        for keys, grp in df.groupby(group_key):
            home_vals = grp.loc[grp["is_home"] == 1, col].shift(1).rolling(5, min_periods=2).mean()
            away_vals = grp.loc[grp["is_home"] == 0, col].shift(1).rolling(5, min_periods=2).mean()
            df.loc[home_vals.index, f"pre_{col}_home_avg5"] = home_vals
            df.loc[away_vals.index, f"pre_{col}_away_avg5"] = away_vals

    # Phase 3B: Opponent-Adjusted Metrics
    # Simple Rating System: adj_off = raw_off - (opp_def - league_avg_def)
    league_avg_off = df["off_rating"].mean()
    league_avg_def = df["def_rating"].mean()
    # Compute opponent's season average defense for adjustment
    df["opp_def_rating_season"] = df.groupby(group_key + (["opp"] if "opp" in df.columns else []))["def_rating"].transform(
        lambda s: s.shift(1).expanding(min_periods=5).mean()
    ) if "opp" in df.columns else np.nan
    df["pre_adj_off_rating"] = df["pre_off_rating_avg5"] - (df["opp_def_rating_season"] - league_avg_def)
    df["pre_adj_off_rating"] = df["pre_adj_off_rating"].fillna(df["pre_off_rating_avg5"])

    # Opponent-adjusted defensive rating
    df["opp_off_rating_season"] = df.groupby(group_key + (["opp"] if "opp" in df.columns else []))["off_rating"].transform(
        lambda s: s.shift(1).expanding(min_periods=5).mean()
    ) if "opp" in df.columns else np.nan
    df["pre_adj_def_rating"] = df["pre_def_rating_avg5"] - (df["opp_off_rating_season"] - league_avg_off)
    df["pre_adj_def_rating"] = df["pre_adj_def_rating"].fillna(df["pre_def_rating_avg5"])

    # EWM-based trend (recent form signal)
    df["pre_net_rating_ewm_trend"] = df["pre_net_rating_ewm10"] - df["pre_net_rating_season"]

    # --- Pythagorean Win Expectation (Item 2) ---
    # pyth_win_pct = pts_scored^exp / (pts_scored^exp + pts_allowed^exp), Morey exp=13.91
    PYTH_EXP = 13.91
    pts_scored_cum = df.groupby(group_key)["team_score"].transform(
        lambda s: s.shift(1).expanding(min_periods=5).sum()
    )
    pts_allowed_cum = df.groupby(group_key)["opp_score"].transform(
        lambda s: s.shift(1).expanding(min_periods=5).sum()
    )
    # Avoid division by zero; use a small epsilon
    scored_exp = np.power(pts_scored_cum.clip(lower=1), PYTH_EXP)
    allowed_exp = np.power(pts_allowed_cum.clip(lower=1), PYTH_EXP)
    df["pre_pyth_win_pct"] = scored_exp / (scored_exp + allowed_exp)
    df.loc[pts_scored_cum.isna(), "pre_pyth_win_pct"] = np.nan

    # --- Five-in-seven (Item 3) ---
    df["five_in_seven"] = 0
    if len(df) > 0:
        df["five_in_seven"] = (
            ((df["game_time_utc"] - df.groupby(group_key)["game_time_utc"].shift(4)).dt.total_seconds() / 86400.0) < 7.1
        ).fillna(False).astype(int)

    # --- Games played this season and late-season flag (Item 3) ---
    df["games_played_season"] = df.groupby(group_key).cumcount() + 1
    df["late_season"] = (df["games_played_season"] > 65).astype(int)

    return df


def _compute_streak(series: pd.Series) -> pd.Series:
    """Compute current win/loss streak. Positive = winning, negative = losing."""
    result = pd.Series(0.0, index=series.index)
    streak = 0.0
    for i, val in series.items():
        if pd.isna(val):
            result.at[i] = 0.0
        else:
            if val == 1:
                streak = streak + 1 if streak > 0 else 1
            else:
                streak = streak - 1 if streak < 0 else -1
            result.at[i] = streak
    return result


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

    # B2B + heavy travel (>1000 miles) interaction (Item 3)
    df["b2b_travel_heavy"] = ((df["b2b"] == 1) & (df["travel_1000_plus"] == 1)).astype(int)

    # --- Cross-timezone travel direction (Item 3) ---
    # Approximate timezone from longitude: each 15 degrees ~ 1 timezone
    # Eastern US ~ -75, Central ~ -90, Mountain ~ -105, Pacific ~ -120
    tz_changes = []
    eastbound_flags = []
    for r in df[["prev_venue_lon", "venue_lon"]].itertuples(index=False):
        if pd.isna(r[0]) or pd.isna(r[1]):
            tz_changes.append(0)
            eastbound_flags.append(0)
        else:
            lon_diff = r[1] - r[0]  # positive = eastward
            tz_crossed = abs(lon_diff) / 15.0  # approximate timezone zones
            tz_changes.append(round(tz_crossed, 1))
            eastbound_flags.append(int(lon_diff > 7.5))  # >0.5 timezone eastward
    df["timezone_change"] = tz_changes
    df["eastbound_travel"] = eastbound_flags

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


def build_referee_game_features(team_games: pd.DataFrame) -> pd.DataFrame:
    """Build referee tendency profiles from cached boxscores and produce per-game crew features.

    For each completed game, extracts the officials from the cached boxscore JSON,
    builds rolling referee tendencies (shifted to avoid leakage), and averages
    across the crew for game-level features.

    Returns a DataFrame keyed by game_id with referee crew feature columns.
    """
    # Collect (game_id, game_time_utc, total_points, combined_fta, combined_pf, possessions)
    # along with officials from boxscores
    game_stats: dict[str, dict[str, Any]] = {}
    game_officials: dict[str, list[dict[str, Any]]] = {}

    # Build game-level stats from team_games (which has both sides)
    for gid, grp in team_games.groupby("game_id"):
        if len(grp) < 2:
            continue
        total_pts = float(grp["team_score"].sum())
        total_fta = float(grp["fta"].sum())
        total_pf = float(grp["pf"].sum())
        total_poss = float(grp["possessions"].mean())  # average of both sides
        gt = grp["game_time_utc"].iloc[0]
        game_stats[str(gid)] = {
            "game_id": str(gid),
            "game_time_utc": gt,
            "total_pts": total_pts,
            "total_fta": total_fta,
            "total_pf": total_pf,
            "avg_poss": total_poss,
        }

    # Extract officials from cached boxscores (current + historical)
    for cache_dir in [BOXSCORE_CACHE, *(HIST_CACHE_DIR / s / "boxscores" for s in SEASONS[:-1])]:
        if not cache_dir.exists():
            continue
        for f in cache_dir.glob("*.json"):
            try:
                payload = json.loads(f.read_text())
                game = payload["game"]
                gid = str(game["gameId"])
                if gid not in game_stats:
                    continue
                officials = game.get("officials", [])
                if officials:
                    game_officials[gid] = [
                        {"ref_id": int(o.get("personId", 0)), "ref_name": o.get("name", "")}
                        for o in officials
                    ]
            except Exception:
                continue

    if not game_officials:
        return pd.DataFrame()

    # Build per-ref-per-game rows sorted chronologically
    ref_game_rows: list[dict[str, Any]] = []
    for gid, refs in game_officials.items():
        gs = game_stats.get(gid)
        if gs is None:
            continue
        for ref in refs:
            ref_game_rows.append({
                "game_id": gid,
                "game_time_utc": gs["game_time_utc"],
                "ref_id": ref["ref_id"],
                "ref_name": ref["ref_name"],
                "total_pts": gs["total_pts"],
                "total_fta": gs["total_fta"],
                "total_pf": gs["total_pf"],
                "avg_poss": gs["avg_poss"],
            })

    if not ref_game_rows:
        return pd.DataFrame()

    ref_df = pd.DataFrame(ref_game_rows)
    ref_df["game_time_utc"] = pd.to_datetime(ref_df["game_time_utc"], utc=True)
    ref_df = ref_df.sort_values(["ref_id", "game_time_utc", "game_id"]).reset_index(drop=True)

    # Compute rolling ref tendencies (shifted for no leakage)
    for col, out_col in [
        ("total_pts", "ref_avg_total"),
        ("avg_poss", "ref_avg_pace"),
        ("total_fta", "ref_avg_fta"),
        ("total_pf", "ref_avg_fouls"),
    ]:
        ref_df[out_col] = ref_df.groupby("ref_id")[col].transform(
            lambda s: s.shift(1).expanding(min_periods=5).mean()
        )

    # Compute league averages for comparison
    league_avg_total = ref_df.drop_duplicates("game_id")["total_pts"].mean()
    league_avg_pace = ref_df.drop_duplicates("game_id")["avg_poss"].mean()

    # Aggregate per game: average referee tendencies across the crew
    crew_features = (
        ref_df.groupby("game_id")
        .agg(
            ref_crew_avg_total=("ref_avg_total", "mean"),
            ref_crew_avg_fta=("ref_avg_fta", "mean"),
            ref_crew_avg_fouls=("ref_avg_fouls", "mean"),
            ref_crew_avg_pace=("ref_avg_pace", "mean"),
            ref_crew_size=("ref_id", "count"),
        )
        .reset_index()
    )

    # Deviations from league average
    crew_features["ref_crew_total_over_league_avg"] = crew_features["ref_crew_avg_total"] - league_avg_total
    crew_features["ref_crew_pace_over_league_avg"] = crew_features["ref_crew_avg_pace"] - league_avg_pace

    print(f"  Referee features computed for {len(crew_features)} games ({ref_df['ref_id'].nunique()} unique refs)", flush=True)
    return crew_features


def add_player_availability_proxy(player_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pg = player_games.copy().sort_values(["team", "player_id", "game_time_utc", "game_id"]).reset_index(drop=True)

    # Use (team, player_id, season) grouping if season column exists
    player_group = ["team", "player_id", "season"] if "season" in pg.columns else ["team", "player_id"]

    for col in ["minutes", "points", "rebounds", "assists", "plus_minus", "played", "starter"]:
        grp = pg.groupby(player_group)[col]
        if col in {"played", "starter"}:
            pg[f"pre_{col}_avg5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            pg[f"pre_{col}_sum5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
        else:
            pg[f"pre_{col}_avg5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
            pg[f"pre_{col}_avg10"] = grp.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    # Game score approximation for star identification: pts + 0.4*ast + 0.7*reb + 0.5*plus_minus
    pg["pre_game_score5"] = (
        pg["pre_points_avg5"].fillna(0)
        + 0.4 * pg["pre_assists_avg5"].fillna(0)
        + 0.7 * pg["pre_rebounds_avg5"].fillna(0)
    )
    pg["pre_impact5"] = pg["pre_minutes_avg5"].fillna(0) * pg["pre_game_score5"]

    # Rotation and starter candidates
    pg["rotation_candidate"] = (
        (pg["pre_minutes_avg5"].fillna(0) >= 8.0) & (pg["pre_played_sum5"].fillna(0) >= 2)
    ).astype(int)
    pg["recent_starter_candidate"] = (pg["pre_starter_avg5"].fillna(0) >= 0.4).astype(int)
    pg["absent_rotation"] = ((pg["rotation_candidate"] == 1) & (pg["played"] == 0)).astype(int)
    pg["absent_recent_starter"] = ((pg["recent_starter_candidate"] == 1) & (pg["played"] == 0)).astype(int)

    # Weight absences by pregame production/minutes
    pg["absent_weighted_min"] = np.where(pg["played"] == 0, pg["pre_minutes_avg5"].fillna(0), 0.0)
    pg["absent_weighted_pts"] = np.where(pg["played"] == 0, pg["pre_points_avg5"].fillna(0), 0.0)
    pg["absent_weighted_ast"] = np.where(pg["played"] == 0, pg["pre_assists_avg5"].fillna(0), 0.0)
    pg["absent_weighted_reb"] = np.where(pg["played"] == 0, pg["pre_rebounds_avg5"].fillna(0), 0.0)

    # Weighted availability: absence weighted by minutes share + production
    pg["minutes_share"] = pg.groupby(["game_id", "team"])["pre_minutes_avg5"].transform(
        lambda s: s / s.sum() if s.sum() > 0 else 0
    )
    pg["absent_weighted_impact"] = np.where(
        pg["played"] == 0,
        pg["minutes_share"].fillna(0) * pg["pre_game_score5"],
        0.0,
    )

    # Active roster plus/minus: aggregate recent plus_minus of active players
    pg["active_plus_minus_contribution"] = np.where(
        pg["played"] == 1,
        pg["pre_plus_minus_avg5"].fillna(0) * pg["pre_minutes_avg5"].fillna(0) / 48.0,
        0.0,
    )

    # Star identification: top-2 players per team-game by pre_impact5
    pg["_rank_in_game"] = pg.groupby(["game_id", "team"])["pre_impact5"].rank(
        ascending=False, method="first"
    )
    pg["is_star_top2"] = (pg["_rank_in_game"] <= 2).astype(int)
    pg["star_absent"] = ((pg["is_star_top2"] == 1) & (pg["played"] == 0)).astype(int)

    # Lineup continuity: fraction of recent 5-game starters who are present
    pg["recent_starter_present"] = (
        (pg["recent_starter_candidate"] == 1) & (pg["played"] == 1)
    ).astype(int)

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
            # Phase 2A: New player impact features
            injury_proxy_weighted_availability=("absent_weighted_impact", "sum"),
            star_players_absent=("star_absent", "sum"),
            recent_starter_present_count=("recent_starter_present", "sum"),
            recent_starter_candidate_count=("recent_starter_candidate", "sum"),
            active_roster_plus_minus=("active_plus_minus_contribution", "sum"),
        )
        .reset_index()
    )
    team_inj["injury_proxy_max_missing_minutes5"] = team_inj["injury_proxy_max_missing_minutes5"].fillna(0)
    # Lineup continuity: fraction of recent starters who are present
    team_inj["lineup_continuity"] = np.where(
        team_inj["recent_starter_candidate_count"] > 0,
        team_inj["recent_starter_present_count"] / team_inj["recent_starter_candidate_count"],
        1.0,
    )
    # Star flag: any star player absent
    team_inj["star_player_absent_flag"] = (team_inj["star_players_absent"] > 0).astype(int)
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
        failed = 0
        for fut in cf.as_completed(futs):
            eid = futs[fut]
            try:
                payload = fut.result()
            except Exception as exc:
                failed += 1
                print(f"  WARNING: ESPN odds fetch failed for {eid}: {exc}", flush=True)
                done += 1
                continue
            done += 1
            if done % 100 == 0 or done == len(ids):
                print(f"ESPN odds {done}/{len(ids)}" + (f" ({failed} failed)" if failed else ""), flush=True)
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
                    "market_home_spread_close": _first_valid(_parse_spread_american(home_odds.get("close")), _parse_spread_american(close)),
                    "market_away_spread_close": _parse_spread_american(away_odds.get("close")),
                    "market_home_spread_open": _first_valid(_parse_spread_american(home_odds.get("open")), _parse_spread_american(open_)),
                    "market_away_spread_open": _parse_spread_american(away_odds.get("open")),
                    "market_total_close": _parse_total_american(close),
                    "market_total_open": _parse_total_american(open_),
                    "market_total_current": _parse_total_american(current),
                    "market_home_ml_close": _first_valid(_to_float(_deep_get(home_odds, ["close", "moneyLine", "american"])), hml),
                    "market_away_ml_close": _first_valid(_to_float(_deep_get(away_odds, ["close", "moneyLine", "american"])), aml),
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


# ---------------------------------------------------------------------------
# Injury Report: fetch, parse, and feature engineering
# ---------------------------------------------------------------------------

INJURY_STATUS_PROB = {
    "out": 0.0,
    "doubtful": 0.10,
    "questionable": 0.35,
    "probable": 0.85,
    "day-to-day": 0.50,
    "suspension": 0.0,
}


def fetch_espn_injury_report(cache_key: str | None = None) -> list[dict[str, Any]]:
    """Fetch the current NBA injury report from ESPN and return parsed rows.

    Each row contains: team (NBA tricode), player_name, espn_player_id,
    status, status_prob, injury_type, injury_detail, report_date.

    Results are cached by cache_key (typically YYYYMMDD date string).
    """
    cache_path = None
    if cache_key:
        INJURY_REPORT_CACHE.mkdir(parents=True, exist_ok=True)
        cache_path = INJURY_REPORT_CACHE / f"espn_injuries_{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    try:
        payload = fetch_json(ESPN_INJURIES_URL, timeout=20, retries=3)
    except Exception as exc:
        print(f"  Warning: Could not fetch ESPN injury report: {exc}", flush=True)
        return []

    rows: list[dict[str, Any]] = []
    for team_entry in payload.get("injuries", []):
        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            team_info = athlete.get("team", {})
            espn_abbr = team_info.get("abbreviation", "")
            team_tricode = normalize_espn_abbr(espn_abbr)

            # Extract ESPN player ID from links
            espn_player_id = None
            for link in athlete.get("links", []):
                href = link.get("href", "")
                if "/player/_/id/" in href:
                    parts = href.split("/id/")
                    if len(parts) > 1:
                        try:
                            espn_player_id = int(parts[1].split("/")[0])
                        except (ValueError, IndexError):
                            pass
                    break

            raw_status = (inj.get("status") or "").strip().lower()
            status_prob = INJURY_STATUS_PROB.get(raw_status, 0.5)
            details = inj.get("details", {}) or {}

            rows.append({
                "team": team_tricode,
                "player_name": athlete.get("displayName", ""),
                "espn_player_id": espn_player_id,
                "status": inj.get("status", ""),
                "status_lower": raw_status,
                "status_prob": status_prob,
                "injury_type": details.get("type", ""),
                "injury_detail": details.get("detail", ""),
                "injury_side": details.get("side", ""),
                "return_date": details.get("returnDate", ""),
                "report_date": inj.get("date", ""),
            })

    if cache_path and rows:
        cache_path.write_text(json.dumps(rows, default=str))
    return rows


def match_injury_report_to_players(
    injury_rows: list[dict[str, Any]],
    player_games: pd.DataFrame,
) -> pd.DataFrame:
    """Match ESPN injury report entries to player_games rows by name and team.

    Returns a DataFrame with columns: team, player_name, espn_player_id,
    player_id (NBA personId), status, status_prob, plus player recent stats.
    """
    if not injury_rows:
        return pd.DataFrame()

    inj_df = pd.DataFrame(injury_rows)

    # Build a lookup of latest player stats per team from player_games
    # We need: player_id, player_name, team, and recent averages
    pg = player_games.copy()
    pg = pg.sort_values(["team", "player_id", "game_time_utc", "game_id"])

    # Get latest 5-game averages per player
    latest_stats: list[dict[str, Any]] = []
    for (team, pid), grp in pg.groupby(["team", "player_id"], sort=False):
        if pd.isna(pid):
            continue
        recent = grp.tail(5)
        if recent.empty:
            continue
        latest_stats.append({
            "team": team,
            "player_id": int(pid),
            "player_name_roster": recent.iloc[-1].get("player_name", ""),
            "recent_minutes_avg": float(recent["minutes"].mean()),
            "recent_points_avg": float(recent["points"].mean()),
            "recent_assists_avg": float(recent["assists"].mean()),
            "recent_rebounds_avg": float(recent["rebounds"].mean()),
            "recent_played_rate": float(recent["played"].mean()),
            "recent_starter_rate": float(recent["starter"].mean()),
        })

    if not latest_stats:
        return pd.DataFrame()

    roster_df = pd.DataFrame(latest_stats)

    # Match injury report to roster by team + player name (fuzzy matching)
    # Normalize names for matching: lowercase, strip whitespace
    inj_df["_name_norm"] = inj_df["player_name"].str.lower().str.strip()
    roster_df["_name_norm"] = roster_df["player_name_roster"].str.lower().str.strip()

    # Try exact merge first
    matched = inj_df.merge(
        roster_df, on=["team", "_name_norm"], how="left"
    )

    # For unmatched entries, try partial name matching (last name)
    unmatched_mask = matched["player_id"].isna()
    if unmatched_mask.any():
        inj_df["_last_name"] = inj_df["player_name"].str.split().str[-1].str.lower().str.strip()
        roster_df["_last_name"] = roster_df["player_name_roster"].str.split().str[-1].str.lower().str.strip()

        for idx in matched[unmatched_mask].index:
            team = matched.at[idx, "team"]
            last_name = inj_df.loc[inj_df["_name_norm"] == matched.at[idx, "_name_norm"], "_last_name"]
            if last_name.empty:
                continue
            last_name = last_name.iloc[0]
            candidates = roster_df[
                (roster_df["team"] == team)
                & (roster_df["_last_name"] == last_name)
            ]
            if len(candidates) == 1:
                for col in ["player_id", "recent_minutes_avg", "recent_points_avg",
                            "recent_assists_avg", "recent_rebounds_avg",
                            "recent_played_rate", "recent_starter_rate"]:
                    matched.at[idx, col] = candidates.iloc[0][col]

    # Clean up temp columns
    for col in ["_name_norm", "_last_name"]:
        if col in matched.columns:
            matched.drop(columns=[col], inplace=True, errors="ignore")

    return matched


def compute_injury_report_features(
    injury_matched: pd.DataFrame,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """Compute team-level injury report features from matched injury data.

    Returns one row per team with injury report feature columns.
    """
    if injury_matched.empty:
        return pd.DataFrame()

    im = injury_matched.copy()
    # Fill missing stats with 0 (unmatched players contribute nothing)
    for col in ["recent_minutes_avg", "recent_points_avg", "recent_assists_avg",
                "recent_rebounds_avg", "recent_played_rate", "recent_starter_rate"]:
        if col in im.columns:
            im[col] = im[col].fillna(0.0)
        else:
            im[col] = 0.0

    # Compute per-player impact: (1 - availability_prob) * stat
    im["unavail_prob"] = 1.0 - im["status_prob"]
    im["missing_minutes_weighted"] = im["unavail_prob"] * im["recent_minutes_avg"]
    im["missing_impact_weighted"] = im["unavail_prob"] * (
        im["recent_points_avg"]
        + 0.7 * im["recent_assists_avg"]
        + 0.5 * im["recent_rebounds_avg"]
    )
    im["is_out"] = (im["status_lower"] == "out").astype(int) | (im["status_lower"] == "suspension").astype(int)
    im["is_questionable"] = im["status_lower"].isin(["questionable", "day-to-day"]).astype(int)

    # Filter to rotation players only (recent avg >= 8 min)
    im["is_rotation"] = (im["recent_minutes_avg"] >= 8.0).astype(int)

    # Identify top-2 impact players per team
    im["_impact_score"] = im["recent_minutes_avg"] * (
        im["recent_points_avg"]
        + 0.4 * im["recent_assists_avg"]
        + 0.7 * im["recent_rebounds_avg"]
    )

    team_features: list[dict[str, Any]] = []
    all_teams = teams or im["team"].unique().tolist()

    for team in all_teams:
        team_inj = im[im["team"] == team]
        if team_inj.empty:
            team_features.append({
                "team": team,
                "injury_report_missing_minutes": 0.0,
                "injury_report_missing_impact": 0.0,
                "injury_report_star_status_top1": 1.0,
                "injury_report_star_status_top2": 1.0,
                "injury_report_count_out": 0,
                "injury_report_count_questionable": 0,
                "injury_report_total_risk": 0.0,
                "injury_report_rotation_out": 0,
            })
            continue

        # Top-2 impact players on the injury report
        top2 = team_inj.nlargest(2, "_impact_score")
        star1_prob = float(top2.iloc[0]["status_prob"]) if len(top2) >= 1 else 1.0
        star2_prob = float(top2.iloc[1]["status_prob"]) if len(top2) >= 2 else 1.0

        rotation_inj = team_inj[team_inj["is_rotation"] == 1]

        team_features.append({
            "team": team,
            "injury_report_missing_minutes": float(team_inj["missing_minutes_weighted"].sum()),
            "injury_report_missing_impact": float(team_inj["missing_impact_weighted"].sum()),
            "injury_report_star_status_top1": star1_prob,
            "injury_report_star_status_top2": star2_prob,
            "injury_report_count_out": int(team_inj["is_out"].sum()),
            "injury_report_count_questionable": int(team_inj["is_questionable"].sum()),
            "injury_report_total_risk": float(team_inj["unavail_prob"].sum()),
            "injury_report_rotation_out": int(rotation_inj["is_out"].sum()),
        })

    return pd.DataFrame(team_features)


def add_injury_report_features(
    team_games: pd.DataFrame,
    player_games: pd.DataFrame,
    game_date_est: str | None = None,
) -> pd.DataFrame:
    """Fetch the ESPN injury report and merge injury features onto team_games.

    For current-season use: fetches the live injury report (cached per date).
    The injury features are only populated for games where a report is available;
    historical games will have NaN (the existing proxy system covers those).

    Parameters
    ----------
    team_games : DataFrame with team-game rows (must have 'team' column).
    player_games : DataFrame with player-game rows for building stat lookups.
    game_date_est : Date string (YYYYMMDD) for caching, or None to auto-detect.

    Returns
    -------
    team_games with additional injury_report_* columns (NaN where unavailable).
    """
    from datetime import datetime as _dt

    cache_key = game_date_est or _dt.now().strftime("%Y%m%d")
    injury_rows = fetch_espn_injury_report(cache_key=cache_key)

    if not injury_rows:
        # No injury data available -- add NaN columns
        for col in [
            "injury_report_missing_minutes",
            "injury_report_missing_impact",
            "injury_report_star_status_top1",
            "injury_report_star_status_top2",
            "injury_report_count_out",
            "injury_report_count_questionable",
            "injury_report_total_risk",
            "injury_report_rotation_out",
        ]:
            team_games[col] = np.nan
        return team_games

    # Match injuries to player roster
    injury_matched = match_injury_report_to_players(injury_rows, player_games)

    # Compute team-level features
    all_teams = team_games["team"].unique().tolist()
    team_features = compute_injury_report_features(injury_matched, teams=all_teams)

    if team_features.empty:
        for col in [
            "injury_report_missing_minutes",
            "injury_report_missing_impact",
            "injury_report_star_status_top1",
            "injury_report_star_status_top2",
            "injury_report_count_out",
            "injury_report_count_questionable",
            "injury_report_total_risk",
            "injury_report_rotation_out",
        ]:
            team_games[col] = np.nan
        return team_games

    # The injury report applies to upcoming/current games only.
    # We merge it onto ALL team_games rows for the matching teams, but since
    # the report is a point-in-time snapshot, only current-date games benefit.
    # Historical games will already have NaN for these columns.
    team_games = team_games.merge(team_features, on="team", how="left")
    return team_games


def compute_elo_ratings(team_games: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo ratings for all teams across all seasons.

    For each game, we assign a pre-game Elo (``pre_elo``) that reflects the
    team's strength entering that contest.  Ratings are updated after each game
    using a margin-of-victory-adjusted K-factor to reduce autocorrelation.

    Between seasons the ratings are regressed toward 1500 by 30%.
    """
    ELO_START = 1500.0
    K_BASE = 20.0
    HOME_ADV_ELO = 100.0
    REGRESS_FRAC = 0.30  # regress 30% toward mean between seasons

    df = team_games.copy().sort_values(["game_time_utc", "game_id", "team"]).reset_index(drop=True)

    # Current Elo for every team
    elo: dict[str, float] = {}
    # Track which season we last saw, for regression at season boundaries
    last_season_seen: str | None = None

    # Pre-allocate the columns
    df["pre_elo"] = np.nan
    df["post_elo"] = np.nan

    # Build a game-level view: for each game_id, we need home/away teams and
    # the margin.  Process games in chronological order.
    game_ids_ordered = (
        df.drop_duplicates(subset=["game_id"])
        .sort_values("game_time_utc")["game_id"]
        .tolist()
    )

    # Index into df for fast lookup by game_id
    game_id_groups = df.groupby("game_id")

    for gid in game_ids_ordered:
        rows = game_id_groups.get_group(gid)
        if len(rows) < 2:
            continue

        home_row = rows[rows["is_home"] == 1]
        away_row = rows[rows["is_home"] == 0]
        if home_row.empty or away_row.empty:
            continue

        home_team = str(home_row["team"].iloc[0])
        away_team = str(away_row["team"].iloc[0])
        home_margin = float(home_row["margin"].iloc[0])
        season = str(home_row["season"].iloc[0]) if "season" in home_row.columns else None

        # Between-season regression
        if season is not None and last_season_seen is not None and season != last_season_seen:
            for t in list(elo.keys()):
                elo[t] = ELO_START + (1.0 - REGRESS_FRAC) * (elo[t] - ELO_START)
        last_season_seen = season

        # Initialize teams if first appearance
        if home_team not in elo:
            elo[home_team] = ELO_START
        if away_team not in elo:
            elo[away_team] = ELO_START

        # Pre-game Elo
        home_elo = elo[home_team]
        away_elo = elo[away_team]

        # Store pre-game Elo on the DataFrame
        df.loc[home_row.index, "pre_elo"] = home_elo
        df.loc[away_row.index, "pre_elo"] = away_elo

        # --- Update Elo ---
        # Expected score with home advantage
        expected_home = 1.0 / (1.0 + 10.0 ** (-(home_elo - away_elo + HOME_ADV_ELO) / 400.0))
        actual_home = 1.0 if home_margin > 0 else (0.0 if home_margin < 0 else 0.5)

        # Margin-of-victory multiplier (reduces autocorrelation)
        abs_margin = abs(home_margin)
        if home_margin > 0:
            winner_elo_diff = home_elo - away_elo
        else:
            winner_elo_diff = away_elo - home_elo
        mov_mult = math.log(abs_margin + 1) * 2.2 / (winner_elo_diff * 0.001 + 2.2)

        k = K_BASE * max(mov_mult, 0.5)  # floor at half K to avoid near-zero updates

        new_home_elo = home_elo + k * (actual_home - expected_home)
        new_away_elo = away_elo + k * (expected_home - actual_home)
        elo[home_team] = new_home_elo
        elo[away_team] = new_away_elo

        # Store post-game Elo (used by prediction script for current ratings)
        df.loc[home_row.index, "post_elo"] = new_home_elo
        df.loc[away_row.index, "post_elo"] = new_away_elo

    print(f"  Elo ratings computed for {len(elo)} teams across {len(game_ids_ordered)} games", flush=True)
    return df


def build_team_games_and_players(include_historical: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load current season
    schedule_df = fetch_schedule_df()
    print(f"Current season completed games: {len(schedule_df)}", flush=True)
    team_games, player_games = fetch_all_boxscores(schedule_df)
    schedule_df["season"] = SEASON
    team_games["season"] = SEASON
    player_games["season"] = SEASON
    team_games = team_games.merge(
        schedule_df[
            ["game_id", "game_time_utc", "game_date_est", "home_team", "away_team", "arena_city", "arena_state", "arena_name"]
        ],
        on=["game_id", "game_time_utc"],
        how="left",
    )

    # Load historical seasons
    if include_historical:
        for hist_season in SEASONS[:-1]:  # All except current
            hist_sched, hist_team_rows, hist_player_rows = load_historical_season(hist_season)
            if hist_sched.empty:
                print(f"  No cached data for {hist_season}, skipping", flush=True)
                continue
            hist_sched["season"] = hist_season
            hist_tg = pd.DataFrame(hist_team_rows).drop_duplicates(subset=["game_id", "team"])
            hist_pg = pd.DataFrame(hist_player_rows).drop_duplicates(subset=["game_id", "team", "player_id"])
            hist_tg["game_time_utc"] = pd.to_datetime(hist_tg["game_time_utc"], utc=True)
            hist_pg["game_time_utc"] = pd.to_datetime(hist_pg["game_time_utc"], utc=True)
            hist_tg["season"] = hist_season
            hist_pg["season"] = hist_season
            hist_tg = hist_tg.merge(
                hist_sched[["game_id", "game_time_utc", "game_date_est", "home_team", "away_team",
                            "arena_city", "arena_state", "arena_name"]],
                on=["game_id", "game_time_utc"],
                how="left",
            )
            schedule_df = pd.concat([hist_sched, schedule_df], ignore_index=True)
            team_games = pd.concat([hist_tg, team_games], ignore_index=True)
            player_games = pd.concat([hist_pg, player_games], ignore_index=True)
            print(f"  Loaded {hist_season}: {len(hist_sched)} games", flush=True)

    schedule_df = schedule_df.sort_values(["game_time_utc", "game_id"]).reset_index(drop=True)
    team_games = team_games.sort_values(["game_time_utc", "game_id", "team"]).reset_index(drop=True)
    player_games = player_games.sort_values(["game_time_utc", "game_id", "team"]).reset_index(drop=True)
    print(f"Total games across all seasons: {schedule_df['game_id'].nunique()}", flush=True)

    team_games = add_rest_and_rolling_team_features(team_games)
    team_games = add_travel_features(team_games, schedule_df)
    team_games = compute_elo_ratings(team_games)
    player_games, team_inj = add_player_availability_proxy(player_games)
    team_games = team_games.merge(
        team_inj.drop(columns=[c for c in ["game_time_utc", "is_home", "opp"] if c in team_inj.columns]),
        on=["game_id", "team"],
        how="left",
    )
    return schedule_df, team_games, player_games


# ---------------------------------------------------------------------------
# Odds snapshot overlay: enrich games with snapshot-based line movement
# ---------------------------------------------------------------------------
ODDS_SNAPSHOT_DIR = OUT_DIR / "odds_snapshots"


def load_odds_snapshots(game_date: str) -> dict[str, dict[str, Any]]:
    """Load all odds snapshots for a game date and compute opening/closing movement.

    Returns a dict keyed by espn_event_id with snapshot-derived odds data.
    """
    date_dir = ODDS_SNAPSHOT_DIR / game_date
    if not date_dir.exists():
        return {}

    snapshots: list[dict[str, Any]] = []
    for f in sorted(date_dir.glob("snapshot_*.json")):
        try:
            data = json.loads(f.read_text())
            snapshots.append(data)
        except Exception:
            continue

    if not snapshots:
        return {}

    opening = snapshots[0]
    closing = snapshots[-1]

    open_games = {g["espn_event_id"]: g for g in opening.get("games", [])}
    close_games = {g["espn_event_id"]: g for g in closing.get("games", [])}

    result: dict[str, dict[str, Any]] = {}
    for eid in set(open_games.keys()) | set(close_games.keys()):
        og = open_games.get(eid, {})
        cg = close_games.get(eid, {})

        open_odds = og.get("odds", [{}])[0] if og.get("odds") else {}
        close_odds = cg.get("odds", [{}])[0] if cg.get("odds") else {}

        o_spread = open_odds.get("spread")
        c_spread = close_odds.get("spread")
        o_total = open_odds.get("over_under")
        c_total = close_odds.get("over_under")
        o_hprob = open_odds.get("home_implied_prob")
        c_hprob = close_odds.get("home_implied_prob")

        result[eid] = {
            "snapshot_spread_open": o_spread,
            "snapshot_spread_close": c_spread,
            "snapshot_total_open": o_total,
            "snapshot_total_close": c_total,
            "snapshot_home_prob_open": o_hprob,
            "snapshot_home_prob_close": c_hprob,
            "snapshot_spread_move": (c_spread - o_spread) if c_spread is not None and o_spread is not None else None,
            "snapshot_total_move": (c_total - o_total) if c_total is not None and o_total is not None else None,
            "snapshot_ml_move": (c_hprob - o_hprob) if c_hprob is not None and o_hprob is not None else None,
            "snapshot_count": len(snapshots),
        }

    return result


def _overlay_odds_snapshots(games: pd.DataFrame) -> pd.DataFrame:
    """Overlay snapshot-based line movement onto the games DataFrame.

    Where snapshot data is available and the ESPN-based open/close data has NaN,
    we fill from snapshots.  We also add snapshot-specific columns for models
    that have richer snapshot coverage.
    """
    if "game_date_est" not in games.columns or "espn_event_id" not in games.columns:
        return games

    # Collect all unique game dates
    dates = games["game_date_est"].dropna().unique().tolist()

    # Load all snapshots across dates
    all_snapshots: dict[str, dict[str, Any]] = {}
    for d in dates:
        date_movements = load_odds_snapshots(str(d))
        all_snapshots.update(date_movements)

    if not all_snapshots:
        return games

    # Build a snapshot DataFrame keyed by espn_event_id
    snap_rows = []
    for eid, data in all_snapshots.items():
        row = {"espn_event_id": eid}
        row.update(data)
        snap_rows.append(row)
    snap_df = pd.DataFrame(snap_rows)

    if snap_df.empty:
        return games

    # Merge snapshot data
    games = games.merge(snap_df, on="espn_event_id", how="left", suffixes=("", "_snap"))

    # Use snapshot spread/total as fallback for open values when ESPN data is NaN
    if "snapshot_spread_open" in games.columns:
        mask = games["market_home_spread_open"].isna() & games["snapshot_spread_open"].notna()
        games.loc[mask, "market_home_spread_open"] = games.loc[mask, "snapshot_spread_open"]

    if "snapshot_total_open" in games.columns:
        mask = games["market_total_open"].isna() & games["snapshot_total_open"].notna()
        games.loc[mask, "market_total_open"] = games.loc[mask, "snapshot_total_open"]

    # Recompute movement features if we filled in open values
    games["market_spread_move_home"] = games["market_home_spread_close"] - games["market_home_spread_open"]
    games["market_total_move"] = games["market_total_close"] - games["market_total_open"]
    games["market_spread_move_abs"] = games["market_spread_move_home"].abs()
    games["market_total_move_abs"] = games["market_total_move"].abs()

    _hs_open = games["market_home_spread_open"]
    games["market_home_implied_prob_open"] = 1.0 / (1.0 + np.exp(_hs_open / 6.5))
    games["market_ml_move"] = games["market_home_implied_prob_close"] - games["market_home_implied_prob_open"]

    return games


def build_game_level(team_games: pd.DataFrame, schedule_with_odds: pd.DataFrame, ref_features: pd.DataFrame | None = None) -> pd.DataFrame:
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
        "b2b_travel_heavy",
        "five_in_seven",
        "games_played_season",
        "late_season",
        "timezone_change",
        "eastbound_travel",
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
        "injury_proxy_weighted_availability",
        "star_players_absent",
        "star_player_absent_flag",
        "lineup_continuity",
        "active_roster_plus_minus",
        "active_count",
        "roster_count",
        # Injury report features (NaN for historical games)
        "injury_report_missing_minutes",
        "injury_report_missing_impact",
        "injury_report_star_status_top1",
        "injury_report_star_status_top2",
        "injury_report_count_out",
        "injury_report_count_questionable",
        "injury_report_total_risk",
        "injury_report_rotation_out",
    ] + [c for c in tg.columns if c.startswith("pre_")]
    # Filter keep_cols to only those present in the DataFrame (injury report cols may be absent for historical data)
    keep_cols = [c for c in keep_cols if c in tg.columns]
    home = home[keep_cols].rename(columns={c: f"home_{c}" for c in keep_cols if c not in ["game_id", "game_time_utc"]})
    away = away[keep_cols].rename(columns={c: f"away_{c}" for c in keep_cols if c not in ["game_id", "game_time_utc"]})
    games = home.merge(away, on=["game_id", "game_time_utc"], how="inner")
    games["home_win"] = games["home_win"].astype(int)
    games["total_points"] = games["home_team_score"] + games["away_team_score"]

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
    # EWM differentials
    ewm_diff_metrics = ["net_rating", "off_rating", "def_rating", "margin", "efg", "ts_pct", "tov_rate", "possessions"]
    for m in ewm_diff_metrics:
        hc = f"home_pre_{m}_ewm10"
        ac = f"away_pre_{m}_ewm10"
        if hc in games.columns and ac in games.columns:
            games[f"diff_pre_{m}_ewm10"] = games[hc] - games[ac]
    # Win pct differentials
    for w in ["win_pct", "win_pct_last10"]:
        hc = f"home_pre_{w}"
        ac = f"away_pre_{w}"
        if hc in games.columns and ac in games.columns:
            games[f"diff_pre_{w}"] = games[hc] - games[ac]
    # Adjusted rating differentials
    for r in ["adj_off_rating", "adj_def_rating"]:
        hc = f"home_pre_{r}"
        ac = f"away_pre_{r}"
        if hc in games.columns and ac in games.columns:
            games[f"diff_pre_{r}"] = games[hc] - games[ac]
    # Defensive rebounding diff
    for w in ["avg5", "avg10"]:
        hc = f"home_pre_drb_rate_{w}"
        ac = f"away_pre_drb_rate_{w}"
        if hc in games.columns and ac in games.columns:
            games[f"diff_pre_drb_rate_{w}"] = games[hc] - games[ac]
    # Margin consistency diff
    if "home_pre_margin_std10" in games.columns and "away_pre_margin_std10" in games.columns:
        games["diff_pre_margin_std10"] = games["home_pre_margin_std10"] - games["away_pre_margin_std10"]
    # EWM trend diff
    if "home_pre_net_rating_ewm_trend" in games.columns and "away_pre_net_rating_ewm_trend" in games.columns:
        games["diff_pre_net_rating_ewm_trend"] = games["home_pre_net_rating_ewm_trend"] - games["away_pre_net_rating_ewm_trend"]
    # Elo differential and expected margin
    if "home_pre_elo" in games.columns and "away_pre_elo" in games.columns:
        games["diff_pre_elo"] = games["home_pre_elo"] - games["away_pre_elo"]
        # Convert Elo diff to expected point spread (~28 Elo points per 1 point of margin)
        games["pre_elo_diff_expected_margin"] = games["diff_pre_elo"] / 28.0
    # Direct diffs for non-rolling features.
    for c in [
        "days_since_prev",
        "travel_miles_since_prev",
        "injury_proxy_missing_minutes5",
        "injury_proxy_missing_points5",
        "active_count",
        "injury_proxy_absent_rotation_count",
        "injury_proxy_absent_recent_starter_count",
        "injury_proxy_weighted_availability",
        "lineup_continuity",
        "active_roster_plus_minus",
        "star_player_absent_flag",
        # Injury report features (NaN for historical games without reports)
        "injury_report_missing_minutes",
        "injury_report_missing_impact",
        "injury_report_count_out",
        "injury_report_count_questionable",
        "injury_report_total_risk",
        "injury_report_rotation_out",
    ]:
        hc = f"home_{c}"
        ac = f"away_{c}"
        if hc in games.columns and ac in games.columns:
            games[f"diff_{c}"] = games[hc] - games[ac]

    games["rest_diff"] = games["home_days_since_prev"] - games["away_days_since_prev"]
    games["home_b2b_adv"] = games["away_b2b"] - games["home_b2b"]
    games["travel_diff"] = games["home_travel_miles_since_prev"] - games["away_travel_miles_since_prev"]

    # --- Referee crew features (Item 1) ---
    if ref_features is not None and not ref_features.empty:
        ref_cols_to_merge = [c for c in ref_features.columns if c != "game_id" or c == "game_id"]
        games = games.merge(ref_features, on="game_id", how="left")

    # --- Home/away venue splits (Item 2) ---
    # Use each team's venue-specific rolling stats at the correct venue
    if "home_pre_net_rating_home_avg5" in games.columns:
        games["home_pre_net_rating_venue_split"] = games["home_pre_net_rating_home_avg5"]
    if "away_pre_net_rating_away_avg5" in games.columns:
        games["away_pre_net_rating_venue_split"] = games["away_pre_net_rating_away_avg5"]
    if "home_pre_margin_home_avg5" in games.columns:
        games["home_pre_margin_venue_split"] = games["home_pre_margin_home_avg5"]
    if "away_pre_margin_away_avg5" in games.columns:
        games["away_pre_margin_venue_split"] = games["away_pre_margin_away_avg5"]
    # Venue split differentials
    if "home_pre_net_rating_venue_split" in games.columns and "away_pre_net_rating_venue_split" in games.columns:
        games["diff_pre_net_rating_venue_split"] = games["home_pre_net_rating_venue_split"] - games["away_pre_net_rating_venue_split"]
    if "home_pre_margin_venue_split" in games.columns and "away_pre_margin_venue_split" in games.columns:
        games["diff_pre_margin_venue_split"] = games["home_pre_margin_venue_split"] - games["away_pre_margin_venue_split"]

    # --- Pythagorean win expectation differential (Item 2) ---
    if "home_pre_pyth_win_pct" in games.columns and "away_pre_pyth_win_pct" in games.columns:
        games["diff_pre_pyth_win_pct"] = games["home_pre_pyth_win_pct"] - games["away_pre_pyth_win_pct"]

    # --- Situational features (Item 3) ---
    # five_in_seven, b2b_travel_heavy diffs
    for c in ["five_in_seven", "b2b_travel_heavy", "games_played_season", "late_season"]:
        hc = f"home_{c}"
        ac = f"away_{c}"
        if hc in games.columns and ac in games.columns:
            games[f"diff_{c}"] = games[hc] - games[ac]

    # Altitude effect: game is in Denver (home_team == DEN)
    games["altitude_game"] = (games["home_team"] == "DEN").astype(int)
    games["altitude_short_rest"] = (
        (games["altitude_game"] == 1) & (games["away_days_since_prev"] <= 1.6)
    ).fillna(False).astype(int)

    # Cross-timezone travel direction features
    for side in ("home", "away"):
        tz_col = f"{side}_timezone_change"
        eb_col = f"{side}_eastbound_travel"
        if tz_col not in games.columns:
            games[tz_col] = 0
        if eb_col not in games.columns:
            games[eb_col] = 0

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

    # Derived line-movement features
    games["market_spread_move_abs"] = games["market_spread_move_home"].abs()
    games["market_total_move_abs"] = games["market_total_move"].abs()

    # Opening moneyline implied probability: derive from open spread using logistic approx
    # spread-to-prob: P(home win) ~ logistic(-spread / 6.5)  (empirical NBA conversion)
    _hs_open = games["market_home_spread_open"]
    games["market_home_implied_prob_open"] = 1.0 / (1.0 + np.exp(_hs_open / 6.5))
    games["market_ml_move"] = games["market_home_implied_prob_close"] - games["market_home_implied_prob_open"]

    # Overlay snapshot-based line movement if available
    games = _overlay_odds_snapshots(games)

    return games.sort_values("game_time_utc").reset_index(drop=True)


def chron_split(df: pd.DataFrame, frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("game_time_utc").reset_index(drop=True)
    cut = int(math.floor(len(df) * frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def time_series_cv_folds(df: pd.DataFrame, n_splits: int = 5, min_train: int = 200) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate expanding-window time-series CV folds."""
    df = df.sort_values("game_time_utc").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, test_idx in tscv.split(df):
        if len(train_idx) >= min_train:
            folds.append((df.iloc[train_idx].copy(), df.iloc[test_idx].copy()))
    return folds


def optuna_tune_xgb_classifier(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    target: str = "home_win",
    n_trials: int = 100,
) -> dict[str, Any]:
    """Tune XGBoost classifier hyperparameters using Optuna over CV folds."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
        scores = []
        for train_df, val_df in folds:
            fold_features = usable_feature_list(train_df, features, val_df)
            if not fold_features:
                continue
            imp = SimpleImputer(strategy="median")
            Xtr = imp.fit_transform(train_df[fold_features])
            Xval = imp.transform(val_df[fold_features])
            model = XGBClassifier(**params, eval_metric="logloss", random_state=42, verbosity=0)
            model.fit(Xtr, train_df[target])
            proba = model.predict_proba(Xval)[:, 1]
            scores.append(log_loss(val_df[target], proba))
        return float(np.mean(scores)) if scores else 1e6

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optuna_tune_xgb_regressor(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    target: str = "total_points",
    n_trials: int = 100,
) -> dict[str, Any]:
    """Tune XGBoost regressor hyperparameters using Optuna over CV folds."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
        scores = []
        for train_df, val_df in folds:
            fold_features = usable_feature_list(train_df, features, val_df)
            if not fold_features:
                continue
            imp = SimpleImputer(strategy="median")
            Xtr = imp.fit_transform(train_df[fold_features])
            Xval = imp.transform(val_df[fold_features])
            model = XGBRegressor(**params, random_state=42, verbosity=0)
            model.fit(Xtr, train_df[target])
            pred = model.predict(Xval)
            scores.append(mean_absolute_error(val_df[target], pred))
        return float(np.mean(scores)) if scores else 1e6

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def shap_feature_importance(
    model: Any, X_train: np.ndarray, features: list[str], max_samples: int = 500
) -> list[dict[str, Any]]:
    """Compute SHAP feature importances. Returns sorted list of (feature, importance)."""
    if shap is None:
        return [{"feature": f, "importance": 0.0} for f in features]
    # Subsample for speed
    if X_train.shape[0] > max_samples:
        idx = np.random.default_rng(42).choice(X_train.shape[0], max_samples, replace=False)
        X_sample = X_train[idx]
    else:
        X_sample = X_train
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classifiers, use class 1
    mean_abs = np.abs(shap_values).mean(axis=0)
    fi = sorted(
        [{"feature": f, "importance": float(v)} for f, v in zip(features, mean_abs)],
        key=lambda x: x["importance"],
        reverse=True,
    )
    return fi


def select_features_by_shap(
    shap_importances: list[dict[str, Any]], threshold: float = 0.005
) -> list[str]:
    """Select features with SHAP importance above threshold."""
    total = sum(f["importance"] for f in shap_importances)
    if total == 0:
        return [f["feature"] for f in shap_importances]
    return [
        f["feature"]
        for f in shap_importances
        if f["importance"] / total >= threshold
    ]


def add_market_comparison_deltas(
    model_metrics: dict[str, Any],
    market_metrics: dict[str, Any] | None,
    metric_names: list[str],
    prefix: str = "vs_market_",
) -> dict[str, Any]:
    """Attach model-minus-market deltas and which side is better."""
    if not market_metrics:
        return model_metrics
    higher_better = {"accuracy", "auc", "r2"}
    for metric in metric_names:
        if metric not in model_metrics or metric not in market_metrics:
            continue
        mv = model_metrics[metric]
        bv = market_metrics[metric]
        if mv is None or bv is None:
            continue
        delta = float(mv - bv)
        model_metrics[f"{prefix}{metric}_delta"] = delta
        if metric in higher_better:
            model_metrics[f"{prefix}{metric}_better"] = "model" if delta > 0 else "market"
        else:
            model_metrics[f"{prefix}{metric}_better"] = "model" if delta < 0 else "market"
    return model_metrics


def usable_feature_list(
    train_df: pd.DataFrame,
    features: list[str],
    test_df: pd.DataFrame | None = None,
) -> list[str]:
    """Keep only features present and with at least one observed train value.

    This prevents sklearn's imputer from dropping all-NaN columns mid-fold and changing shape.
    """
    out: list[str] = []
    test_cols = set(test_df.columns) if test_df is not None else None
    for f in features:
        if f not in train_df.columns:
            continue
        if test_cols is not None and f not in test_cols:
            continue
        s = train_df[f]
        if s.notna().any():
            out.append(f)
    return out


def eval_win_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_name: str,
    xgb_params: dict[str, Any] | None = None,
    calibrate: bool = False,
    calibration_method: str = "sigmoid",
    compute_shap: bool = True,
) -> dict[str, Any]:
    """Train and evaluate a win probability model."""
    feats = usable_feature_list(train, features, test)
    if not feats:
        raise ValueError(f"No usable features for {model_name}")
    X_train = train[feats]
    X_test = test[feats]
    y_train = train["home_win"]
    y_test = test["home_win"]

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xte = imp.transform(X_test)

    params = xgb_params or {
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    }
    base_model = XGBClassifier(**params, eval_metric="logloss", random_state=42, verbosity=0)
    base_model.fit(Xtr, y_train)
    raw_proba = base_model.predict_proba(Xte)[:, 1]
    proba = raw_proba
    if calibrate and len(train) >= 100 and y_train.nunique() > 1:
        cal_est = clone(base_model)
        cal_cv = 3 if len(train) >= 150 else 2
        calibrator = CalibratedClassifierCV(estimator=cal_est, method=calibration_method, cv=cal_cv)
        calibrator.fit(Xtr, y_train)
        proba = calibrator.predict_proba(Xte)[:, 1]
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    pred = (proba >= 0.5).astype(int)
    importances = base_model.feature_importances_

    # SHAP importance on base model only (skip for CV speed)
    shap_fi = shap_feature_importance(base_model, Xtr, feats) if compute_shap else []

    fi = sorted(
        [{"feature": f, "importance": float(v)} for f, v in zip(feats, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )[:12]
    return {
        "label": model_name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "calibrated": bool(calibrate),
        "n_features_used": int(len(feats)),
        "features_used": feats,
        "baseline_home_win_rate_test": float(y_test.mean()),
        "top_features": fi,
        "shap_features": shap_fi[:12],
        "xgb_params": params,
    }


def eval_total_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_name: str,
    xgb_params: dict[str, Any] | None = None,
    compute_shap: bool = True,
) -> dict[str, Any]:
    """Train and evaluate a total points model."""
    feats = usable_feature_list(train, features, test)
    if not feats:
        raise ValueError(f"No usable features for {model_name}")
    X_train = train[feats]
    X_test = test[feats]
    y_train = train["total_points"]
    y_test = test["total_points"]

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xte = imp.transform(X_test)

    params = xgb_params or {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    }
    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    importances = model.feature_importances_

    shap_fi = shap_feature_importance(model, Xtr, feats) if compute_shap else []

    return {
        "label": model_name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "test_avg_total": float(y_test.mean()),
        "n_features_used": int(len(feats)),
        "features_used": feats,
        "top_features": sorted(
            [{"feature": f, "importance": float(v)} for f, v in zip(feats, importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )[:12],
        "shap_features": shap_fi[:12],
        "xgb_params": params,
    }


def eval_win_model_cv(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    model_name: str,
    xgb_params: dict[str, Any] | None = None,
    calibrate: bool = False,
) -> dict[str, Any]:
    """Evaluate win model across CV folds. Returns mean + std metrics."""
    fold_results = []
    for train_df, val_df in folds:
        result = eval_win_model(
            train_df,
            val_df,
            features,
            model_name,
            xgb_params,
            calibrate=calibrate,
            compute_shap=False,
        )
        fold_results.append(result)
    metrics = ["accuracy", "auc", "log_loss", "brier_score"]
    summary = {"label": model_name, "n_folds": len(fold_results)}
    for m in metrics:
        values = [f[m] for f in fold_results if m in f]
        if values:
            summary[f"{m}_mean"] = float(np.mean(values))
            summary[f"{m}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    summary["fold_results"] = fold_results
    return summary


def eval_total_model_cv(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    model_name: str,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate total model across CV folds."""
    fold_results = []
    for train_df, val_df in folds:
        result = eval_total_model(train_df, val_df, features, model_name, xgb_params, compute_shap=False)
        fold_results.append(result)
    metrics = ["mae", "rmse", "r2"]
    summary = {"label": model_name, "n_folds": len(fold_results)}
    for m in metrics:
        values = [f[m] for f in fold_results if m in f]
        if values:
            summary[f"{m}_mean"] = float(np.mean(values))
            summary[f"{m}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    summary["fold_results"] = fold_results
    return summary


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


def eval_market_baselines_cv(folds: list[tuple[pd.DataFrame, pd.DataFrame]]) -> dict[str, Any]:
    """Aggregate market baselines over CV validation folds."""
    out: dict[str, Any] = {}
    win_folds: list[dict[str, Any]] = []
    total_folds: list[dict[str, Any]] = []
    for _, val_df in folds:
        fold_res = eval_market_baselines(val_df)
        if "market_win_baseline" in fold_res:
            win_folds.append(fold_res["market_win_baseline"])
        if "market_total_baseline" in fold_res:
            total_folds.append(fold_res["market_total_baseline"])

    def summarize(rows: list[dict[str, Any]], metrics: list[str], label: str) -> dict[str, Any]:
        summary: dict[str, Any] = {"label": label, "n_folds": len(rows)}
        if rows:
            summary["fold_results"] = rows
            for metric in metrics:
                vals = [float(r[metric]) for r in rows if metric in r]
                if vals:
                    summary[f"{metric}_mean"] = float(np.mean(vals))
                    summary[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return summary

    if win_folds:
        out["market_win_baseline_cv"] = summarize(win_folds, ["accuracy", "auc", "log_loss"], "market_win_baseline_cv")
    if total_folds:
        out["market_total_baseline_cv"] = summarize(total_folds, ["mae", "rmse", "r2"], "market_total_baseline_cv")
    return out


def eval_market_residual_regression(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    residual_target_col: str,
    actual_target_col: str,
    market_baseline_pred_col: str,
    model_name: str,
    xgb_params: dict[str, Any] | None = None,
    compute_shap: bool = True,
) -> dict[str, Any]:
    """Train a regressor on residuals vs market and report both residual and reconstructed metrics."""
    feats = usable_feature_list(train, features, test)
    if not feats:
        raise ValueError(f"No usable features for {model_name}")
    X_train = train[feats]
    X_test = test[feats]
    y_train_resid = train[residual_target_col]
    y_test_resid = test[residual_target_col]
    y_test_actual = test[actual_target_col]
    market_pred = test[market_baseline_pred_col]

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xte = imp.transform(X_test)

    params = xgb_params or {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    }
    model = XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(Xtr, y_train_resid)
    pred_resid = model.predict(Xte)
    pred_actual = market_pred.to_numpy(dtype=float) + pred_resid
    importances = model.feature_importances_
    shap_fi = shap_feature_importance(model, Xtr, feats) if compute_shap else []

    out = {
        "label": model_name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "residual_target_col": residual_target_col,
        "actual_target_col": actual_target_col,
        "market_baseline_pred_col": market_baseline_pred_col,
        "residual_mae": float(mean_absolute_error(y_test_resid, pred_resid)),
        "residual_rmse": float(np.sqrt(mean_squared_error(y_test_resid, pred_resid))),
        "residual_r2": float(r2_score(y_test_resid, pred_resid)),
        "mae": float(mean_absolute_error(y_test_actual, pred_actual)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_actual, pred_actual))),
        "r2": float(r2_score(y_test_actual, pred_actual)),
        "market_baseline_mae": float(mean_absolute_error(y_test_actual, market_pred)),
        "market_baseline_rmse": float(np.sqrt(mean_squared_error(y_test_actual, market_pred))),
        "market_baseline_r2": float(r2_score(y_test_actual, market_pred)),
        "n_features_used": int(len(feats)),
        "features_used": feats,
        "top_features": sorted(
            [{"feature": f, "importance": float(v)} for f, v in zip(feats, importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )[:12],
        "shap_features": shap_fi[:12],
        "xgb_params": params,
    }
    add_market_comparison_deltas(out, {"mae": out["market_baseline_mae"], "rmse": out["market_baseline_rmse"], "r2": out["market_baseline_r2"]}, ["mae", "rmse", "r2"])
    return out


def eval_market_residual_regression_cv(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    residual_target_col: str,
    actual_target_col: str,
    market_baseline_pred_col: str,
    model_name: str,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate market residual regressor over CV folds."""
    fold_results: list[dict[str, Any]] = []
    for train_df, val_df in folds:
        fold_results.append(
            eval_market_residual_regression(
                train_df,
                val_df,
                features,
                residual_target_col,
                actual_target_col,
                market_baseline_pred_col,
                model_name,
                xgb_params,
                compute_shap=False,
            )
        )
    summary: dict[str, Any] = {"label": model_name, "n_folds": len(fold_results), "fold_results": fold_results}
    for metric in ["mae", "rmse", "r2", "residual_mae", "residual_rmse", "residual_r2"]:
        vals = [float(f[metric]) for f in fold_results if metric in f]
        if vals:
            summary[f"{metric}_mean"] = float(np.mean(vals))
            summary[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if fold_results:
        market_summary = {
            "mae": float(np.mean([f["market_baseline_mae"] for f in fold_results])),
            "rmse": float(np.mean([f["market_baseline_rmse"] for f in fold_results])),
            "r2": float(np.mean([f["market_baseline_r2"] for f in fold_results])),
        }
        summary["market_baseline_means"] = market_summary
        add_market_comparison_deltas(summary, market_summary, ["mae", "rmse", "r2"])
    return summary


def _build_lgbm_classifier(params: dict[str, Any] | None = None) -> Any:
    """Build a LightGBM classifier with given params (or defaults)."""
    if not _HAS_LGBM:
        return None
    p = params or {}
    return LGBMClassifier(
        n_estimators=p.get("n_estimators", 300),
        max_depth=p.get("max_depth", 5),
        learning_rate=p.get("learning_rate", 0.05),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
        reg_lambda=p.get("reg_lambda", 1.0),
        reg_alpha=p.get("reg_alpha", 0.1),
        min_child_samples=p.get("min_child_samples", 20),
        num_leaves=p.get("num_leaves", 31),
        random_state=42,
        verbosity=-1,
    )


def _build_lgbm_regressor(params: dict[str, Any] | None = None) -> Any:
    """Build a LightGBM regressor with given params (or defaults)."""
    if not _HAS_LGBM:
        return None
    p = params or {}
    return LGBMRegressor(
        n_estimators=p.get("n_estimators", 300),
        max_depth=p.get("max_depth", 5),
        learning_rate=p.get("learning_rate", 0.05),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
        reg_lambda=p.get("reg_lambda", 1.0),
        reg_alpha=p.get("reg_alpha", 0.1),
        min_child_samples=p.get("min_child_samples", 20),
        num_leaves=p.get("num_leaves", 31),
        random_state=42,
        verbosity=-1,
    )


def optuna_tune_lgbm_classifier(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    target: str = "home_win",
    n_trials: int = 60,
) -> dict[str, Any]:
    """Tune LightGBM classifier hyperparameters using Optuna over CV folds."""
    if not _HAS_LGBM:
        return {}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        }
        scores = []
        for train_df, val_df in folds:
            fold_features = usable_feature_list(train_df, features, val_df)
            if not fold_features:
                continue
            imp = SimpleImputer(strategy="median")
            Xtr = imp.fit_transform(train_df[fold_features])
            Xval = imp.transform(val_df[fold_features])
            model = LGBMClassifier(**params, random_state=42, verbosity=-1)
            model.fit(Xtr, train_df[target])
            proba = model.predict_proba(Xval)[:, 1]
            scores.append(log_loss(val_df[target], proba))
        return float(np.mean(scores)) if scores else 1e6

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def eval_ensemble_win_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_name: str,
    xgb_params: dict[str, Any] | None = None,
    lgbm_params: dict[str, Any] | None = None,
    calibrate: bool = True,
    compute_shap: bool = True,
) -> dict[str, Any]:
    """Train XGBoost + LightGBM ensemble with logistic regression stacking for win prediction.

    Uses out-of-fold predictions from both base models to train a logistic meta-learner,
    then evaluates on the test set. This combines the strengths of both tree models.
    """
    feats = usable_feature_list(train, features, test)
    if not feats:
        raise ValueError(f"No usable features for {model_name}")
    X_train = train[feats].copy()
    X_test = test[feats].copy()
    y_train = train["home_win"].to_numpy()
    y_test = test["home_win"].to_numpy()

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xte = imp.transform(X_test)

    # Base model 1: XGBoost
    xgb_p = xgb_params or {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.04}
    xgb_model = XGBClassifier(**xgb_p, eval_metric="logloss", random_state=42, verbosity=0)

    # Base model 2: LightGBM
    lgbm_model = _build_lgbm_classifier(lgbm_params) if _HAS_LGBM else None

    # Generate out-of-fold predictions for stacking
    n_stack_folds = 5
    tscv = TimeSeriesSplit(n_splits=n_stack_folds)
    oof_xgb = np.full(len(Xtr), np.nan)
    oof_lgbm = np.full(len(Xtr), np.nan) if lgbm_model else None

    for fold_train_idx, fold_val_idx in tscv.split(Xtr):
        if len(fold_train_idx) < 100:
            continue
        Xf_tr, Xf_val = Xtr[fold_train_idx], Xtr[fold_val_idx]
        yf_tr = y_train[fold_train_idx]

        xgb_fold = XGBClassifier(**xgb_p, eval_metric="logloss", random_state=42, verbosity=0)
        xgb_fold.fit(Xf_tr, yf_tr)
        oof_xgb[fold_val_idx] = xgb_fold.predict_proba(Xf_val)[:, 1]

        if lgbm_model is not None:
            lgbm_fold = _build_lgbm_classifier(lgbm_params)
            lgbm_fold.fit(Xf_tr, yf_tr)
            oof_lgbm[fold_val_idx] = lgbm_fold.predict_proba(Xf_val)[:, 1]

    # Train full base models on all training data
    xgb_model.fit(Xtr, y_train)
    test_xgb = xgb_model.predict_proba(Xte)[:, 1]

    if lgbm_model is not None:
        lgbm_model.fit(Xtr, y_train)
        test_lgbm = lgbm_model.predict_proba(Xte)[:, 1]

        # Build stacking meta-features
        valid_oof = ~np.isnan(oof_xgb) & ~np.isnan(oof_lgbm)
        if valid_oof.sum() >= 50:
            meta_train = np.column_stack([oof_xgb[valid_oof], oof_lgbm[valid_oof]])
            meta_test = np.column_stack([test_xgb, test_lgbm])
            meta_y = y_train[valid_oof]

            meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            meta_model.fit(meta_train, meta_y)
            proba = meta_model.predict_proba(meta_test)[:, 1]
        else:
            # Fallback: simple average
            proba = 0.5 * test_xgb + 0.5 * test_lgbm
    else:
        proba = test_xgb

    if calibrate and len(train) >= 100 and train["home_win"].nunique() > 1:
        # Calibrate the ensemble output
        cal_model = CalibratedClassifierCV(
            estimator=clone(xgb_model), method="isotonic", cv=3 if len(train) >= 200 else 2
        )
        cal_model.fit(Xtr, y_train)
        cal_proba = cal_model.predict_proba(Xte)[:, 1]
        # Blend calibrated single model with stacked ensemble
        if lgbm_model is not None:
            proba = 0.6 * proba + 0.4 * cal_proba
        else:
            proba = cal_proba

    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    pred = (proba >= 0.5).astype(int)
    importances = xgb_model.feature_importances_

    shap_fi = shap_feature_importance(xgb_model, Xtr, feats) if compute_shap else []

    fi = sorted(
        [{"feature": f, "importance": float(v)} for f, v in zip(feats, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )[:12]
    return {
        "label": model_name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "calibrated": bool(calibrate),
        "ensemble": True,
        "has_lgbm": _HAS_LGBM,
        "n_features_used": int(len(feats)),
        "features_used": feats,
        "baseline_home_win_rate_test": float(y_test.mean()),
        "top_features": fi,
        "shap_features": shap_fi[:12],
        "xgb_params": xgb_p,
    }


def eval_ensemble_win_model_cv(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    model_name: str,
    xgb_params: dict[str, Any] | None = None,
    lgbm_params: dict[str, Any] | None = None,
    calibrate: bool = True,
) -> dict[str, Any]:
    """Evaluate ensemble win model across CV folds."""
    fold_results = []
    for train_df, val_df in folds:
        result = eval_ensemble_win_model(
            train_df, val_df, features, model_name,
            xgb_params, lgbm_params, calibrate=calibrate, compute_shap=False,
        )
        fold_results.append(result)
    metrics = ["accuracy", "auc", "log_loss", "brier_score"]
    summary = {"label": model_name, "n_folds": len(fold_results)}
    for m in metrics:
        values = [f[m] for f in fold_results if m in f]
        if values:
            summary[f"{m}_mean"] = float(np.mean(values))
            summary[f"{m}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    summary["fold_results"] = fold_results
    return summary


def eval_ensemble_total_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    model_name: str,
    xgb_params: dict[str, Any] | None = None,
    lgbm_params: dict[str, Any] | None = None,
    compute_shap: bool = True,
) -> dict[str, Any]:
    """Train XGBoost + LightGBM ensemble for total points prediction."""
    feats = usable_feature_list(train, features, test)
    if not feats:
        raise ValueError(f"No usable features for {model_name}")
    X_train = train[feats].copy()
    X_test = test[feats].copy()
    y_train = train["total_points"].to_numpy(dtype=float)
    y_test = test["total_points"].to_numpy(dtype=float)

    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X_train)
    Xte = imp.transform(X_test)

    xgb_p = xgb_params or {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.04}
    xgb_model = XGBRegressor(**xgb_p, random_state=42, verbosity=0)
    xgb_model.fit(Xtr, y_train)
    pred_xgb = xgb_model.predict(Xte)

    if _HAS_LGBM:
        lgbm_model = _build_lgbm_regressor(lgbm_params)
        lgbm_model.fit(Xtr, y_train)
        pred_lgbm = lgbm_model.predict(Xte)
        # Use out-of-fold to find optimal blend weight
        n_stack_folds = 5
        tscv = TimeSeriesSplit(n_splits=n_stack_folds)
        oof_xgb = np.full(len(Xtr), np.nan)
        oof_lgbm = np.full(len(Xtr), np.nan)
        for fold_train_idx, fold_val_idx in tscv.split(Xtr):
            if len(fold_train_idx) < 100:
                continue
            xgb_f = XGBRegressor(**xgb_p, random_state=42, verbosity=0)
            xgb_f.fit(Xtr[fold_train_idx], y_train[fold_train_idx])
            oof_xgb[fold_val_idx] = xgb_f.predict(Xtr[fold_val_idx])
            lgbm_f = _build_lgbm_regressor(lgbm_params)
            lgbm_f.fit(Xtr[fold_train_idx], y_train[fold_train_idx])
            oof_lgbm[fold_val_idx] = lgbm_f.predict(Xtr[fold_val_idx])
        valid = ~np.isnan(oof_xgb) & ~np.isnan(oof_lgbm)
        if valid.sum() >= 50:
            meta_X = np.column_stack([oof_xgb[valid], oof_lgbm[valid]])
            meta_y = y_train[valid]
            meta = LinearRegression()
            meta.fit(meta_X, meta_y)
            pred = meta.predict(np.column_stack([pred_xgb, pred_lgbm]))
        else:
            pred = 0.5 * pred_xgb + 0.5 * pred_lgbm
    else:
        pred = pred_xgb

    importances = xgb_model.feature_importances_
    shap_fi = shap_feature_importance(xgb_model, Xtr, feats) if compute_shap else []

    return {
        "label": model_name,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "test_avg_total": float(y_test.mean()),
        "ensemble": True,
        "has_lgbm": _HAS_LGBM,
        "n_features_used": int(len(feats)),
        "features_used": feats,
        "top_features": sorted(
            [{"feature": f, "importance": float(v)} for f, v in zip(feats, importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )[:12],
        "shap_features": shap_fi[:12],
        "xgb_params": xgb_p,
    }


def eval_ensemble_total_model_cv(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    model_name: str,
    xgb_params: dict[str, Any] | None = None,
    lgbm_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate ensemble total model across CV folds."""
    fold_results = []
    for train_df, val_df in folds:
        result = eval_ensemble_total_model(
            train_df, val_df, features, model_name, xgb_params, lgbm_params, compute_shap=False,
        )
        fold_results.append(result)
    metrics = ["mae", "rmse", "r2"]
    summary = {"label": model_name, "n_folds": len(fold_results)}
    for m in metrics:
        values = [f[m] for f in fold_results if m in f]
        if values:
            summary[f"{m}_mean"] = float(np.mean(values))
            summary[f"{m}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    summary["fold_results"] = fold_results
    return summary


def oof_calibrate_probabilities(
    folds: list[tuple[pd.DataFrame, pd.DataFrame]],
    features: list[str],
    target: str,
    xgb_params: dict[str, Any] | None = None,
    lgbm_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Generate out-of-fold predictions and fit an isotonic regression calibrator.

    Returns ``(calibrator, diagnostics_dict)`` where *calibrator* is a fitted
    ``sklearn.isotonic.IsotonicRegression`` that maps raw predicted probabilities
    to calibrated ones.

    When *lgbm_params* is provided the function calibrates an ensemble
    (XGBoost + LightGBM + margin-consistent) rather than raw XGBoost alone.
    """
    from sklearn.isotonic import IsotonicRegression

    oof_probs: list[float] = []
    oof_labels: list[int] = []

    for train_df, val_df in folds:
        feats = [f for f in features if f in train_df.columns and f in val_df.columns]
        if not feats:
            continue
        imp = SimpleImputer(strategy="median")
        X_train = imp.fit_transform(train_df[feats])
        X_val = imp.transform(val_df[feats])
        y_train = train_df[target].values
        y_val = val_df[target].values

        # XGBoost prediction
        p = xgb_params or {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.04}
        xgb_model = XGBClassifier(
            **{k: v for k, v in p.items() if k in {"n_estimators", "max_depth", "learning_rate",
                                                     "subsample", "colsample_bytree", "reg_lambda",
                                                     "reg_alpha", "min_child_weight"}},
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        p_xgb = xgb_model.predict_proba(X_val)[:, 1]

        if lgbm_params is not None and _HAS_LGBM:
            from lightgbm import LGBMClassifier
            lgbm_model = LGBMClassifier(
                n_estimators=lgbm_params.get("n_estimators", 200),
                max_depth=lgbm_params.get("max_depth", 4),
                learning_rate=lgbm_params.get("learning_rate", 0.05),
                subsample=lgbm_params.get("subsample", 0.8),
                colsample_bytree=lgbm_params.get("colsample_bytree", 0.8),
                reg_lambda=lgbm_params.get("reg_lambda", 1.0),
                random_state=42, verbose=-1,
            )
            lgbm_model.fit(X_train, y_train)
            p_lgb = lgbm_model.predict_proba(X_val)[:, 1]
            # Ensemble: simple average of xgb and lgbm
            fold_probs = 0.5 * p_xgb + 0.5 * p_lgb
        else:
            fold_probs = p_xgb

        oof_probs.extend(fold_probs.tolist())
        oof_labels.extend(y_val.tolist())

    oof_probs_arr = np.array(oof_probs, dtype=float)
    oof_labels_arr = np.array(oof_labels, dtype=int)

    # Fit isotonic regression calibrator
    calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    calibrator.fit(oof_probs_arr, oof_labels_arr)

    # Compute diagnostics: raw vs calibrated ECE
    cal_probs = calibrator.predict(oof_probs_arr)
    from nba_evaluate import calibration_error, calibration_by_decile, brier_score

    raw_ece = calibration_error(oof_labels_arr, oof_probs_arr)
    cal_ece = calibration_error(oof_labels_arr, cal_probs)
    raw_brier = brier_score(oof_labels_arr, oof_probs_arr)
    cal_brier = brier_score(oof_labels_arr, cal_probs)

    diagnostics = {
        "raw_ece": round(float(raw_ece), 4),
        "calibrated_ece": round(float(cal_ece), 4),
        "raw_brier": round(float(raw_brier), 4),
        "calibrated_brier": round(float(cal_brier), 4),
        "n_oof_samples": len(oof_probs),
        "calibration_buckets_raw": calibration_by_decile(oof_labels_arr, oof_probs_arr),
        "calibration_buckets_calibrated": calibration_by_decile(oof_labels_arr, cal_probs),
    }
    return calibrator, diagnostics


def print_calibration_diagnostics(diagnostics: dict[str, Any]) -> None:
    """Print calibration bucket table and ECE comparison."""
    print("\n  === CALIBRATION DIAGNOSTICS ===", flush=True)
    print(f"  OOF samples: {diagnostics['n_oof_samples']}", flush=True)
    print(f"  Raw ECE:        {diagnostics['raw_ece']:.4f}   Brier: {diagnostics['raw_brier']:.4f}", flush=True)
    print(f"  Calibrated ECE: {diagnostics['calibrated_ece']:.4f}   Brier: {diagnostics['calibrated_brier']:.4f}", flush=True)
    improvement_pct = 100 * (diagnostics["raw_ece"] - diagnostics["calibrated_ece"]) / max(diagnostics["raw_ece"], 1e-6)
    print(f"  ECE improvement: {improvement_pct:+.1f}%", flush=True)

    # Calibration bucket table
    print("\n  Calibration buckets (out-of-fold, AFTER isotonic calibration):", flush=True)
    print(f"  {'Bin':>10}  {'Count':>6}  {'Pred':>7}  {'Actual':>7}  {'|Error|':>7}", flush=True)
    print(f"  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}", flush=True)
    for b in diagnostics.get("calibration_buckets_calibrated", []):
        bin_label = f"{b['bin_lo']:.1f}-{b['bin_hi']:.1f}"
        print(
            f"  {bin_label:>10}  {b['n_games']:>6}  {b['predicted_mean']:>7.3f}  {b['actual_mean']:>7.3f}  {b['abs_error']:>7.3f}",
            flush=True,
        )


def run_advanced_models(games: pd.DataFrame, tune: bool = True, n_tune_trials: int = 100) -> dict[str, Any]:
    models: dict[str, Any] = {}

    # --- Feature lists ---
    base_win_features = [
        "diff_pre_net_rating_season",
        "diff_pre_net_rating_avg5",
        "diff_pre_net_rating_avg10",
        "diff_pre_off_rating_avg5",
        "diff_pre_def_rating_avg5",
        "diff_pre_efg_avg5",
        "diff_pre_tov_rate_avg5",
        "diff_pre_orb_rate_avg5",
        "diff_pre_ft_rate_avg5",
        "diff_pre_possessions_avg5",
        "diff_pre_margin_avg10",
        "diff_pre_margin_season",
        "rest_diff",
        "home_b2b",
        "away_b2b",
        "home_b2b_adv",
    ]
    enh_win_features = base_win_features + [
        # EWM features (recency-weighted)
        "diff_pre_net_rating_ewm10",
        "diff_pre_off_rating_ewm10",
        "diff_pre_def_rating_ewm10",
        "diff_pre_margin_ewm10",
        # Win percentage
        "diff_pre_win_pct",
        "diff_pre_win_pct_last10",
        # Adjusted ratings
        "diff_pre_adj_off_rating",
        "diff_pre_adj_def_rating",
        # Defensive rebounding
        "diff_pre_drb_rate_avg5",
        # Margin consistency
        "diff_pre_margin_std10",
        # EWM trend
        "diff_pre_net_rating_ewm_trend",
        # Elo rating differential (opponent-adjusted strength)
        "diff_pre_elo",
        "pre_elo_diff_expected_margin",
        # Travel and fatigue
        "home_travel_miles_since_prev",
        "away_travel_miles_since_prev",
        "home_b2b_travel_500_plus",
        "away_b2b_travel_500_plus",
        # Player availability
        "diff_injury_proxy_missing_minutes5",
        "diff_injury_proxy_absent_rotation_count",
        "diff_injury_proxy_weighted_availability",
        "diff_lineup_continuity",
        "diff_active_roster_plus_minus",
        # Streaks and trends
        "home_pre_win_streak",
        "away_pre_win_streak",
        "home_pre_net_rating_trend",
        "away_pre_net_rating_trend",
        "home_road_trip_game_num",
        "away_road_trip_game_num",
        # Official injury report features (NaN for historical games)
        "diff_injury_report_missing_minutes",
        "diff_injury_report_missing_impact",
        "diff_injury_report_count_out",
        "diff_injury_report_total_risk",
        "home_injury_report_star_status_top1",
        "away_injury_report_star_status_top1",
        "home_injury_report_star_status_top2",
        "away_injury_report_star_status_top2",
        "diff_injury_report_rotation_out",
        # Home/away venue splits (Item 2)
        "diff_pre_net_rating_venue_split",
        "diff_pre_margin_venue_split",
        # Pythagorean win expectation (Item 2)
        "diff_pre_pyth_win_pct",
        # Situational features (Item 3)
        "home_three_in_four",
        "away_three_in_four",
        "home_four_in_six",
        "away_four_in_six",
        "home_five_in_seven",
        "away_five_in_seven",
        "home_b2b_travel_heavy",
        "away_b2b_travel_heavy",
        "altitude_game",
        "altitude_short_rest",
        "home_late_season",
        "away_late_season",
        "home_timezone_change",
        "away_timezone_change",
        "home_eastbound_travel",
        "away_eastbound_travel",
    ]
    market_win_features = enh_win_features + [
        "market_home_spread_close",
        "market_total_close",
        "market_home_implied_prob_close",
        "market_spread_move_home",
        "market_total_move",
        "market_home_implied_prob_open",
        "market_ml_move",
        "market_spread_move_abs",
    ]
    market_margin_residual_features = enh_win_features + [
        "home_pre_possessions_avg5",
        "away_pre_possessions_avg5",
        "home_pre_off_rating_avg5",
        "away_pre_off_rating_avg5",
        "home_pre_def_rating_avg5",
        "away_pre_def_rating_avg5",
        "home_pre_possessions_ewm10",
        "away_pre_possessions_ewm10",
        "market_home_spread_close",
        "market_total_close",
        "market_spread_move_home",
        "market_total_move",
        "market_spread_move_abs",
        "market_ml_move",
    ]

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
        # EWM pace features
        "home_pre_possessions_ewm10",
        "away_pre_possessions_ewm10",
        "home_pre_off_rating_ewm10",
        "away_pre_off_rating_ewm10",
        "home_pre_def_rating_ewm10",
        "away_pre_def_rating_ewm10",
        # Elo rating differential (opponent-adjusted strength)
        "diff_pre_elo",
        "pre_elo_diff_expected_margin",
        # Travel and fatigue
        "home_travel_miles_since_prev",
        "away_travel_miles_since_prev",
        "home_b2b_travel_500_plus",
        "away_b2b_travel_500_plus",
        # Player availability
        "home_injury_proxy_missing_minutes5",
        "away_injury_proxy_missing_minutes5",
        "home_lineup_continuity",
        "away_lineup_continuity",
        "home_active_roster_plus_minus",
        "away_active_roster_plus_minus",
        "home_active_count",
        "away_active_count",
        # Official injury report features (NaN for historical games)
        "home_injury_report_missing_minutes",
        "away_injury_report_missing_minutes",
        "home_injury_report_missing_impact",
        "away_injury_report_missing_impact",
        "home_injury_report_count_out",
        "away_injury_report_count_out",
        "home_injury_report_total_risk",
        "away_injury_report_total_risk",
        # Referee crew features (Item 1) -- primarily useful for totals
        "ref_crew_avg_total",
        "ref_crew_avg_fta",
        "ref_crew_avg_fouls",
        "ref_crew_total_over_league_avg",
        "ref_crew_pace_over_league_avg",
        # Situational features (Item 3)
        "home_three_in_four",
        "away_three_in_four",
        "home_four_in_six",
        "away_four_in_six",
        "home_five_in_seven",
        "away_five_in_seven",
        "altitude_game",
        "altitude_short_rest",
    ]
    market_total_features = enh_total_features + [
        "market_total_close",
        "market_home_spread_close",
        "market_total_move",
        "market_total_move_abs",
    ]
    # Purpose-built totals residual feature set (not just a copy of the market total model).
    # Keep this narrow to avoid overfitting; add a few totals-relevant signals on top of market_total_features.
    market_total_residual_features = enh_total_features + [
        "home_pre_possessions_avg10",
        "away_pre_possessions_avg10",
        "home_pre_off_rating_avg10",
        "away_pre_off_rating_avg10",
        "home_pre_def_rating_avg10",
        "away_pre_def_rating_avg10",
        "home_injury_proxy_missing_points5",
        "away_injury_proxy_missing_points5",
        "home_four_in_six",
        "away_four_in_six",
        "market_total_close",
        "market_home_spread_close",
        "market_total_move",
        "market_total_move_abs",
    ]

    # Derived market-baseline targets for residual modeling.
    df_sorted = games.sort_values("game_time_utc").reset_index(drop=True).copy()
    if "market_home_spread_close" in df_sorted.columns:
        df_sorted["market_home_margin_implied"] = -df_sorted["market_home_spread_close"]
        df_sorted["market_home_margin_residual"] = df_sorted["home_margin"] - df_sorted["market_home_margin_implied"]
    if "market_total_close" in df_sorted.columns:
        df_sorted["market_total_residual"] = df_sorted["total_points"] - df_sorted["market_total_close"]

    # --- Hold out final test set (last 20% of current season) ---
    # Find where the current season starts for final test holdout
    if "season" in df_sorted.columns:
        current_mask = df_sorted["season"] == SEASON
        if current_mask.any():
            current_start = current_mask.idxmax()
            current_len = current_mask.sum()
            final_test_start = current_start + int(current_len * 0.8)
            cv_df = df_sorted.iloc[:final_test_start].copy()
            final_test = df_sorted.iloc[final_test_start:].copy()
        else:
            cut = int(len(df_sorted) * 0.8)
            cv_df = df_sorted.iloc[:cut].copy()
            final_test = df_sorted.iloc[cut:].copy()
    else:
        cut = int(len(df_sorted) * 0.8)
        cv_df = df_sorted.iloc[:cut].copy()
        final_test = df_sorted.iloc[cut:].copy()

    print(f"  CV set: {len(cv_df)} games, Final test: {len(final_test)} games", flush=True)

    # --- Generate CV folds ---
    folds = time_series_cv_folds(cv_df, n_splits=5, min_train=200)
    print(f"  CV folds: {len(folds)}", flush=True)

    # Filter features to only those present in the data
    def filter_features(feats: list[str], df: pd.DataFrame) -> list[str]:
        return [f for f in feats if f in df.columns]

    base_win_features = filter_features(base_win_features, games)
    enh_win_features = filter_features(enh_win_features, games)
    market_win_features = filter_features(market_win_features, games)
    market_margin_residual_features = filter_features(market_margin_residual_features, games)
    base_total_features = filter_features(base_total_features, games)
    enh_total_features = filter_features(enh_total_features, games)
    market_total_features = filter_features(market_total_features, games)
    market_total_residual_features = filter_features(market_total_residual_features, games)

    # Market-only CV folds (filter to games with market data); built before tuning for subset-specific tuning.
    market_folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_df, val_df in folds:
        mt = train_df.dropna(subset=["market_home_implied_prob_close"])
        mv = val_df.dropna(subset=["market_home_implied_prob_close"])
        if len(mt) >= 100 and len(mv) >= 20:
            market_folds.append((mt, mv))

    market_total_folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_df, val_df in folds:
        mt = train_df.dropna(subset=["market_total_close"])
        mv = val_df.dropna(subset=["market_total_close"])
        if len(mt) >= 100 and len(mv) >= 20:
            market_total_folds.append((mt, mv))

    market_margin_resid_folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_df, val_df in folds:
        mt = train_df.dropna(subset=["market_home_margin_implied", "market_home_margin_residual"])
        mv = val_df.dropna(subset=["market_home_margin_implied", "market_home_margin_residual"])
        if len(mt) >= 100 and len(mv) >= 20:
            market_margin_resid_folds.append((mt, mv))

    market_total_resid_folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_df, val_df in folds:
        mt = train_df.dropna(subset=["market_total_close", "market_total_residual"])
        mv = val_df.dropna(subset=["market_total_close", "market_total_residual"])
        if len(mt) >= 100 and len(mv) >= 20:
            market_total_resid_folds.append((mt, mv))

    # --- Optuna tuning on CV folds (if enabled) ---
    win_params_base = None
    win_params_enh = None
    win_params_market = None
    total_params_base = None
    total_params_enh = None
    total_params_market = None
    margin_resid_params = None
    total_resid_params = None
    if tune and len(folds) >= 2:
        models["tuned_params"] = {}

        print("  Tuning enhanced win model hyperparameters...", flush=True)
        win_params_enh = optuna_tune_xgb_classifier(folds, enh_win_features, "home_win", n_trials=n_tune_trials)
        models["tuned_params"]["win_enhanced"] = win_params_enh

        print("  Tuning enhanced total model hyperparameters...", flush=True)
        total_params_enh = optuna_tune_xgb_regressor(folds, enh_total_features, "total_points", n_trials=n_tune_trials)
        models["tuned_params"]["total_enhanced"] = total_params_enh

        if market_folds:
            print("  Tuning market win model hyperparameters...", flush=True)
            win_params_market = optuna_tune_xgb_classifier(
                market_folds, market_win_features, "home_win", n_trials=max(10, n_tune_trials // 2)
            )
            models["tuned_params"]["win_market"] = win_params_market
        if market_total_folds:
            print("  Tuning market total model hyperparameters...", flush=True)
            total_params_market = optuna_tune_xgb_regressor(
                market_total_folds, market_total_features, "total_points", n_trials=max(10, n_tune_trials // 2)
            )
            models["tuned_params"]["total_market"] = total_params_market
        if market_margin_resid_folds:
            print("  Tuning market margin residual model hyperparameters...", flush=True)
            margin_resid_params = optuna_tune_xgb_regressor(
                market_margin_resid_folds,
                market_margin_residual_features,
                "market_home_margin_residual",
                n_trials=max(10, n_tune_trials // 2),
            )
            models["tuned_params"]["margin_residual_market"] = margin_resid_params
        if market_total_resid_folds:
            print("  Tuning market total residual model hyperparameters...", flush=True)
            total_resid_params = optuna_tune_xgb_regressor(
                market_total_resid_folds,
                market_total_features,
                "market_total_residual",
                n_trials=max(10, n_tune_trials // 2),
            )
            models["tuned_params"]["total_residual_market"] = total_resid_params

    # --- LightGBM tuning ---
    lgbm_win_params_enh = None
    lgbm_win_params_market = None
    if tune and _HAS_LGBM and len(folds) >= 2:
        print("  Tuning LightGBM enhanced win model...", flush=True)
        lgbm_win_params_enh = optuna_tune_lgbm_classifier(
            folds, enh_win_features, "home_win", n_trials=max(10, n_tune_trials // 2)
        )
        models["tuned_params"]["lgbm_win_enhanced"] = lgbm_win_params_enh
        if market_folds:
            print("  Tuning LightGBM market win model...", flush=True)
            lgbm_win_params_market = optuna_tune_lgbm_classifier(
                market_folds, market_win_features, "home_win", n_trials=max(10, n_tune_trials // 3)
            )
            models["tuned_params"]["lgbm_win_market"] = lgbm_win_params_market

    # --- CV evaluation ---
    print("  Running CV evaluation...", flush=True)
    models["win_cv_baseline"] = eval_win_model_cv(folds, base_win_features, "baseline_cv", win_params_base)
    models["win_cv_enhanced"] = eval_win_model_cv(folds, enh_win_features, "enhanced_cv", win_params_enh)
    models["win_cv_enhanced_calibrated"] = eval_win_model_cv(
        folds, enh_win_features, "enhanced_cv_calibrated", win_params_enh, calibrate=True
    )
    # Ensemble CV evaluation
    models["win_cv_ensemble"] = eval_ensemble_win_model_cv(
        folds, enh_win_features, "ensemble_cv", win_params_enh, lgbm_win_params_enh, calibrate=True
    )
    models["total_cv_baseline"] = eval_total_model_cv(folds, base_total_features, "baseline_cv", total_params_base)
    models["total_cv_enhanced"] = eval_total_model_cv(folds, enh_total_features, "enhanced_cv", total_params_enh)
    models["total_cv_ensemble"] = eval_ensemble_total_model_cv(
        folds, enh_total_features, "ensemble_cv", total_params_enh
    )

    def cv_mean_metric_map(summary: dict[str, Any], metrics: list[str]) -> dict[str, Any]:
        return {m: summary.get(f"{m}_mean") for m in metrics if f"{m}_mean" in summary}

    if market_folds:
        market_cv_baselines = eval_market_baselines_cv(market_folds)
        if "market_win_baseline_cv" in market_cv_baselines:
            models["win_cv_market_baseline"] = market_cv_baselines["market_win_baseline_cv"]
        models["win_cv_baseline_on_market_subset"] = eval_win_model_cv(
            market_folds, base_win_features, "baseline_on_market_subset_cv", win_params_base
        )
        models["win_cv_enhanced_on_market_subset"] = eval_win_model_cv(
            market_folds, enh_win_features, "enhanced_on_market_subset_cv", win_params_enh
        )
        models["win_cv_market"] = eval_win_model_cv(market_folds, market_win_features, "market_cv", win_params_market)
        models["win_cv_market_calibrated"] = eval_win_model_cv(
            market_folds, market_win_features, "market_cv_calibrated", win_params_market, calibrate=True
        )
        models["win_cv_market_ensemble"] = eval_ensemble_win_model_cv(
            market_folds, market_win_features, "market_ensemble_cv",
            win_params_market, lgbm_win_params_market, calibrate=True,
        )
        if "win_cv_market_baseline" in models:
            market_mean = cv_mean_metric_map(models["win_cv_market_baseline"], ["accuracy", "auc", "log_loss"])
            for key in [
                "win_cv_baseline_on_market_subset",
                "win_cv_enhanced_on_market_subset",
                "win_cv_market",
                "win_cv_market_calibrated",
                "win_cv_market_ensemble",
            ]:
                if key in models:
                    add_market_comparison_deltas(models[key], market_mean, ["accuracy", "auc", "log_loss"], prefix="vs_market_cv_")

    if market_total_folds:
        market_total_cv_baselines = eval_market_baselines_cv(market_total_folds)
        if "market_total_baseline_cv" in market_total_cv_baselines:
            models["total_cv_market_baseline"] = market_total_cv_baselines["market_total_baseline_cv"]
        models["total_cv_baseline_on_market_subset"] = eval_total_model_cv(
            market_total_folds, base_total_features, "baseline_on_market_subset_cv", total_params_base
        )
        models["total_cv_enhanced_on_market_subset"] = eval_total_model_cv(
            market_total_folds, enh_total_features, "enhanced_on_market_subset_cv", total_params_enh
        )
        models["total_cv_market"] = eval_total_model_cv(
            market_total_folds, market_total_features, "market_cv", total_params_market
        )
        if "total_cv_market_baseline" in models:
            market_total_mean = cv_mean_metric_map(models["total_cv_market_baseline"], ["mae", "rmse", "r2"])
            for key in [
                "total_cv_baseline_on_market_subset",
                "total_cv_enhanced_on_market_subset",
                "total_cv_market",
            ]:
                add_market_comparison_deltas(models[key], market_total_mean, ["mae", "rmse", "r2"], prefix="vs_market_cv_")

    if market_margin_resid_folds:
        models["margin_cv_market_residual"] = eval_market_residual_regression_cv(
            market_margin_resid_folds,
            market_margin_residual_features,
            "market_home_margin_residual",
            "home_margin",
            "market_home_margin_implied",
            "margin_residual_market_cv",
            margin_resid_params,
        )
    if market_total_resid_folds:
        models["total_cv_market_residual"] = eval_market_residual_regression_cv(
            market_total_resid_folds,
            market_total_residual_features,
            "market_total_residual",
            "total_points",
            "market_total_close",
            "total_residual_market_cv",
            total_resid_params,
        )

    # --- Out-of-fold recalibration ---
    print("  Fitting OOF isotonic recalibration...", flush=True)
    oof_calibrator_xgb, oof_diag_xgb = oof_calibrate_probabilities(
        folds, enh_win_features, "home_win", xgb_params=win_params_enh,
    )
    models["oof_calibration_xgb"] = oof_diag_xgb
    print_calibration_diagnostics(oof_diag_xgb)

    # Ensemble OOF calibration (XGB + LightGBM)
    oof_calibrator_ensemble, oof_diag_ensemble = oof_calibrate_probabilities(
        folds, enh_win_features, "home_win",
        xgb_params=win_params_enh, lgbm_params=lgbm_win_params_enh,
    )
    models["oof_calibration_ensemble"] = oof_diag_ensemble
    models["oof_calibrator"] = oof_calibrator_ensemble  # for downstream use
    print(f"  Ensemble OOF ECE: raw={oof_diag_ensemble['raw_ece']:.4f} -> calibrated={oof_diag_ensemble['calibrated_ece']:.4f}", flush=True)

    # --- SHAP-based feature selection ---
    print("  Running SHAP feature selection...", flush=True)
    # Train on full CV set, get SHAP importances
    imp = SimpleImputer(strategy="median")
    Xtr_full = imp.fit_transform(cv_df[enh_win_features])
    win_model_full = XGBClassifier(
        **(win_params_enh or {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.04}),
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    win_model_full.fit(Xtr_full, cv_df["home_win"])
    shap_win = shap_feature_importance(win_model_full, Xtr_full, enh_win_features)
    selected_win = select_features_by_shap(shap_win)
    models["shap_win_features"] = shap_win
    models["selected_win_features"] = selected_win
    print(f"  SHAP selected {len(selected_win)}/{len(enh_win_features)} win features")

    # --- Final test set evaluation ---
    print("  Evaluating on final held-out test set...", flush=True)
    train_all = cv_df.copy()

    models["win_model_baseline"] = eval_win_model(
        train_all, final_test, base_win_features, "baseline_team_rolling", win_params_base
    )
    models["win_model_enhanced"] = eval_win_model(
        train_all, final_test, enh_win_features, "plus_travel_injury_player", win_params_enh
    )
    models["win_model_enhanced_calibrated"] = eval_win_model(
        train_all,
        final_test,
        enh_win_features,
        "plus_travel_injury_player_calibrated",
        win_params_enh,
        calibrate=True,
    )
    if selected_win:
        models["win_model_shap_selected"] = eval_win_model(
            train_all, final_test, selected_win, "shap_selected", win_params_enh
        )

    # Ensemble models (XGBoost + LightGBM stacking)
    models["win_model_ensemble"] = eval_ensemble_win_model(
        train_all, final_test, enh_win_features, "ensemble_xgb_lgbm",
        win_params_enh, lgbm_win_params_enh, calibrate=True,
    )
    models["total_model_ensemble"] = eval_ensemble_total_model(
        train_all, final_test, enh_total_features, "ensemble_xgb_lgbm_total",
        total_params_enh,
    )

    # Market models on games with market data
    market_train = train_all.dropna(subset=["market_home_implied_prob_close"])
    market_test = final_test.dropna(subset=["market_home_implied_prob_close"])
    if len(market_train) >= 100 and len(market_test) >= 20:
        models["win_model_baseline_on_market_subset"] = eval_win_model(
            market_train, market_test, base_win_features, "baseline_on_market_subset", win_params_base
        )
        models["win_model_enhanced_on_market_subset"] = eval_win_model(
            market_train, market_test, enh_win_features, "enhanced_on_market_subset", win_params_enh
        )
        models["win_model_with_market"] = eval_win_model(
            market_train, market_test, market_win_features, "plus_market_lines", win_params_market
        )
        models["win_model_with_market_calibrated"] = eval_win_model(
            market_train,
            market_test,
            market_win_features,
            "plus_market_lines_calibrated",
            win_params_market,
            calibrate=True,
        )
        models["win_model_with_market_ensemble"] = eval_ensemble_win_model(
            market_train, market_test, market_win_features, "market_ensemble_xgb_lgbm",
            win_params_market, lgbm_win_params_market, calibrate=True,
        )

    models["total_model_baseline"] = eval_total_model(
        train_all, final_test, base_total_features, "baseline_team_rolling", total_params_base
    )
    models["total_model_enhanced"] = eval_total_model(
        train_all, final_test, enh_total_features, "plus_travel_injury_player", total_params_enh
    )
    market_total_train = train_all.dropna(subset=["market_total_close"])
    market_total_test = final_test.dropna(subset=["market_total_close"])
    if len(market_total_train) >= 100 and len(market_total_test) >= 20:
        models["total_model_baseline_on_market_subset"] = eval_total_model(
            market_total_train, market_total_test, base_total_features, "baseline_on_market_subset", total_params_base
        )
        models["total_model_enhanced_on_market_subset"] = eval_total_model(
            market_total_train, market_total_test, enh_total_features, "enhanced_on_market_subset", total_params_enh
        )
        models["total_model_with_market"] = eval_total_model(
            market_total_train, market_total_test, market_total_features, "plus_market_lines", total_params_market
        )

    market_margin_train = train_all.dropna(subset=["market_home_margin_implied", "market_home_margin_residual"])
    market_margin_test = final_test.dropna(subset=["market_home_margin_implied", "market_home_margin_residual"])
    if len(market_margin_train) >= 100 and len(market_margin_test) >= 20:
        models["margin_model_market_residual"] = eval_market_residual_regression(
            market_margin_train,
            market_margin_test,
            market_margin_residual_features,
            "market_home_margin_residual",
            "home_margin",
            "market_home_margin_implied",
            "margin_residual_market",
            margin_resid_params,
        )
    if len(market_total_train) >= 100 and len(market_total_test) >= 20 and "market_total_residual" in market_total_train.columns:
        models["total_model_market_residual"] = eval_market_residual_regression(
            market_total_train,
            market_total_test,
            market_total_residual_features,
            "market_total_residual",
            "total_points",
            "market_total_close",
            "total_residual_market",
            total_resid_params,
        )

    # Market baselines
    if len(market_test) >= 20:
        models["market_baselines_win_test"] = eval_market_baselines(market_test)
        market_win_baseline = models["market_baselines_win_test"].get("market_win_baseline")
        if market_win_baseline:
            for key in [
                "win_model_baseline_on_market_subset",
                "win_model_enhanced_on_market_subset",
                "win_model_with_market",
                "win_model_with_market_calibrated",
            ]:
                if key in models:
                    add_market_comparison_deltas(models[key], market_win_baseline, ["accuracy", "auc", "log_loss"])
    if len(market_total_test) >= 20:
        models["market_baselines_total_test"] = eval_market_baselines(market_total_test)
        market_total_baseline = models["market_baselines_total_test"].get("market_total_baseline")
        if market_total_baseline:
            for key in [
                "total_model_baseline_on_market_subset",
                "total_model_enhanced_on_market_subset",
                "total_model_with_market",
                "total_model_market_residual",
                ]:
                    if key in models:
                        add_market_comparison_deltas(models[key], market_total_baseline, ["mae", "rmse", "r2"])

    # Comprehensive evaluation summaries (nba_evaluate module): held-out + CV confidence intervals
    def _predict_win_proba(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list[str],
        params: dict[str, Any] | None,
        calibrate: bool = False,
    ) -> np.ndarray:
        feats = usable_feature_list(train_df, features, test_df)
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(train_df[feats])
        Xte = imp.transform(test_df[feats])
        p = params or {}
        base_model = XGBClassifier(
            n_estimators=p.get("n_estimators", 250),
            max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.04),
            subsample=p.get("subsample", 0.9),
            colsample_bytree=p.get("colsample_bytree", 0.9),
            reg_lambda=p.get("reg_lambda", 1.0),
            reg_alpha=p.get("reg_alpha", 0.0),
            min_child_weight=p.get("min_child_weight", 1),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        base_model.fit(Xtr, train_df["home_win"])
        if calibrate and len(train_df) >= 100 and train_df["home_win"].nunique() > 1:
            cal_est = clone(base_model)
            cal = CalibratedClassifierCV(estimator=cal_est, method="sigmoid", cv=3 if len(train_df) >= 150 else 2)
            cal.fit(Xtr, train_df["home_win"])
            proba = cal.predict_proba(Xte)[:, 1]
        else:
            proba = base_model.predict_proba(Xte)[:, 1]
        return np.clip(proba, 1e-6, 1 - 1e-6)

    def _predict_regression(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list[str],
        target_col: str,
        params: dict[str, Any] | None,
    ) -> np.ndarray:
        feats = usable_feature_list(train_df, features, test_df)
        imp = SimpleImputer(strategy="median")
        Xtr = imp.fit_transform(train_df[feats])
        Xte = imp.transform(test_df[feats])
        p = params or {}
        model = XGBRegressor(
            n_estimators=p.get("n_estimators", 300),
            max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.04),
            subsample=p.get("subsample", 0.9),
            colsample_bytree=p.get("colsample_bytree", 0.9),
            reg_lambda=p.get("reg_lambda", 1.0),
            reg_alpha=p.get("reg_alpha", 0.0),
            min_child_weight=p.get("min_child_weight", 1),
            random_state=42,
            verbosity=0,
        )
        model.fit(Xtr, train_df[target_col])
        return model.predict(Xte)

    def _predict_market_resid_actual(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list[str],
        residual_target_col: str,
        market_pred_col: str,
        params: dict[str, Any] | None,
    ) -> np.ndarray:
        resid_pred = _predict_regression(train_df, test_df, features, residual_target_col, params)
        return test_df[market_pred_col].to_numpy(dtype=float) + resid_pred

    models["comprehensive_eval"] = {}
    # CV confidence summaries from fold-level results
    for src_key, dst_key in [
        ("win_cv_enhanced_calibrated", "win_cv_enhanced_calibrated_ci"),
        ("win_cv_ensemble", "win_cv_ensemble_ci"),
        ("win_cv_market_calibrated", "win_cv_market_calibrated_ci"),
        ("win_cv_market_ensemble", "win_cv_market_ensemble_ci"),
        ("total_cv_enhanced", "total_cv_enhanced_ci"),
        ("total_cv_ensemble", "total_cv_ensemble_ci"),
        ("total_cv_market", "total_cv_market_ci"),
        ("total_cv_market_residual", "total_cv_market_residual_ci"),
    ]:
        if src_key in models and isinstance(models[src_key], dict) and models[src_key].get("fold_results"):
            models["comprehensive_eval"][dst_key] = evaluate_cv_folds(models[src_key]["fold_results"])

    # Held-out comprehensive win eval on market subset (includes market comparison/profit + ATS if margin residual model available)
    if len(market_train) >= 100 and len(market_test) >= 20:
        win_eval_df = market_test.dropna(subset=["market_home_spread_close"]).copy()
        if len(win_eval_df) >= 20:
            win_proba_mkt_cal = _predict_win_proba(
                market_train, win_eval_df, market_win_features, win_params_market, calibrate=True
            )
            pred_margin = None
            if len(market_margin_train) >= 100:
                margin_eval_df = win_eval_df.dropna(subset=["market_home_margin_implied"]).copy()
                if len(margin_eval_df) == len(win_eval_df):
                    pred_margin = _predict_market_resid_actual(
                        market_margin_train,
                        margin_eval_df,
                        market_margin_residual_features,
                        "market_home_margin_residual",
                        "market_home_margin_implied",
                        margin_resid_params,
                    )
            models["comprehensive_eval"]["win_market_calibrated_holdout"] = evaluate_win_model_comprehensive(
                win_eval_df["home_win"].to_numpy(dtype=int),
                win_proba_mkt_cal,
                market_prob=win_eval_df["market_home_implied_prob_close"].to_numpy(dtype=float),
                spread=(-win_eval_df["market_home_spread_close"]).to_numpy(dtype=float),
                pred_margin=pred_margin,
                y_margin=win_eval_df["home_margin"].to_numpy(dtype=float) if pred_margin is not None else None,
            )

    # Held-out comprehensive totals evals on market subset
    if len(market_total_train) >= 100 and len(market_total_test) >= 20:
        if "total_model_with_market" in models:
            pred_total_market = _predict_regression(
                market_total_train, market_total_test, market_total_features, "total_points", total_params_market
            )
            models["comprehensive_eval"]["total_market_holdout"] = evaluate_total_model_comprehensive(
                market_total_test["total_points"].to_numpy(dtype=float),
                pred_total_market,
                market_total=market_total_test["market_total_close"].to_numpy(dtype=float),
            )
        if "total_model_market_residual" in models:
            pred_total_resid = _predict_market_resid_actual(
                market_total_train,
                market_total_test,
                market_total_residual_features,
                "market_total_residual",
                "market_total_close",
                total_resid_params,
            )
            models["comprehensive_eval"]["total_market_residual_holdout"] = evaluate_total_model_comprehensive(
                market_total_test["total_points"].to_numpy(dtype=float),
                pred_total_resid,
                market_total=market_total_test["market_total_close"].to_numpy(dtype=float),
            )

    # Store feature lists for downstream use
    models["feature_lists"] = {
        "base_win": base_win_features,
        "enhanced_win": enh_win_features,
        "market_win": market_win_features,
        "market_margin_residual": market_margin_residual_features,
        "shap_selected_win": selected_win,
        "base_total": base_total_features,
        "enhanced_total": enh_total_features,
        "market_total": market_total_features,
        "market_total_residual": market_total_residual_features,
    }

    return models


def summarize_data(schedule_odds: pd.DataFrame, team_games: pd.DataFrame, player_games: pd.DataFrame, games: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    seasons_present = sorted(games["season"].unique().tolist()) if "season" in games.columns else [SEASON]
    summary["coverage"] = {
        "season": SEASON,
        "seasons": seasons_present,
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
    # Injury report coverage
    inj_report_cols = [
        "injury_report_missing_minutes",
        "injury_report_count_out",
        "injury_report_total_risk",
    ]
    inj_report_avail = {
        col: int(team_games[col].notna().sum()) for col in inj_report_cols if col in team_games.columns
    }
    if inj_report_avail:
        summary["injury_report"] = {
            "games_with_injury_report": inj_report_avail,
            "note": "Injury report features are only available for current-date games (not historical).",
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
        f"Through {c['utc_end']}: {c['games']} games, {c['team_games']} team-games, "
        f"{c['player_game_rows']} player rows"
    )
    if "seasons" in c:
        print(f"Seasons: {c['seasons']}")
    o = summary.get("odds_coverage", {})
    if o:
        print(
            f"Odds coverage: event matches {o.get('games_with_espn_event_match', 'N/A')}/{c['games']}, "
            f"totals {o.get('games_with_market_total_close', 'N/A')}, "
            f"moneylines {o.get('games_with_market_home_ml_close', 'N/A')}"
        )
        print(f"Primary odds provider(s): {o.get('provider_mode', [])}")

    print("\n=== FEATURE CONTEXT ===")
    if summary.get("travel"):
        t = summary["travel"]
        print(
            f"Travel avg miles between games: {t['avg_travel_miles_since_prev']:.1f} "
            f"(median {t['median_travel_miles_since_prev']:.1f}); "
            f"B2B+500mi rate: {t['b2b_travel_500_plus_rate']:.3f}"
        )
    if summary.get("injury_proxy"):
        print("Injury/availability proxy averages (team-game):")
        for k, v in summary["injury_proxy"].items():
            print(f"  {k}: {v:.3f}")

    # --- CV Results ---
    print("\n=== CROSS-VALIDATION RESULTS ===")
    for k in [
        "win_cv_baseline",
        "win_cv_enhanced",
        "win_cv_enhanced_calibrated",
        "win_cv_ensemble",
        "win_cv_baseline_on_market_subset",
        "win_cv_enhanced_on_market_subset",
        "win_cv_market",
        "win_cv_market_calibrated",
        "win_cv_market_ensemble",
    ]:
        if k in models:
            m = models[k]
            extra = ""
            if "vs_market_cv_log_loss_delta" in m:
                extra = f", dLL_vs_mkt={m['vs_market_cv_log_loss_delta']:+.3f}"
            print(
                f"Win {m['label']}: acc={m.get('accuracy_mean', 0):.3f}+/-{m.get('accuracy_std', 0):.3f}, "
                f"AUC={m.get('auc_mean', 0):.3f}+/-{m.get('auc_std', 0):.3f}{extra} ({m['n_folds']} folds)"
            )
    if "win_cv_market_baseline" in models:
        m = models["win_cv_market_baseline"]
        print(
            f"Win market baseline CV: acc={m.get('accuracy_mean', 0):.3f}, "
            f"AUC={m.get('auc_mean', 0):.3f}, logloss={m.get('log_loss_mean', 0):.3f}"
        )
    for k in [
        "total_cv_baseline",
        "total_cv_enhanced",
        "total_cv_ensemble",
        "total_cv_baseline_on_market_subset",
        "total_cv_enhanced_on_market_subset",
        "total_cv_market",
        "margin_cv_market_residual",
        "total_cv_market_residual",
    ]:
        if k in models:
            m = models[k]
            extra = ""
            if "vs_market_cv_mae_delta" in m:
                extra = f", dMAE_vs_mkt={m['vs_market_cv_mae_delta']:+.2f}"
            print(
                f"Total {m['label']}: MAE={m.get('mae_mean', 0):.2f}+/-{m.get('mae_std', 0):.2f}, "
                f"R2={m.get('r2_mean', 0):.3f}+/-{m.get('r2_std', 0):.3f}{extra} ({m['n_folds']} folds)"
            )
    if "total_cv_market_baseline" in models:
        m = models["total_cv_market_baseline"]
        print(
            f"Total market baseline CV: MAE={m.get('mae_mean', 0):.2f}, "
            f"RMSE={m.get('rmse_mean', 0):.2f}, R2={m.get('r2_mean', 0):.3f}"
        )

    # --- Tuned params ---
    if "tuned_params" in models:
        print("\nTuned params:")
        for name, p in models["tuned_params"].items():
            print(
                f"  {name}: depth={p.get('max_depth')}, lr={p.get('learning_rate', 0):.4f}, "
                f"n_est={p.get('n_estimators')}"
            )

    # --- Final test results ---
    print("\n=== WIN MODELS (HELD-OUT TEST) ===")
    for k in [
        "win_model_baseline",
        "win_model_enhanced",
        "win_model_enhanced_calibrated",
        "win_model_ensemble",
        "win_model_shap_selected",
        "win_model_baseline_on_market_subset",
        "win_model_enhanced_on_market_subset",
        "win_model_with_market",
        "win_model_with_market_calibrated",
        "win_model_with_market_ensemble",
    ]:
        if k in models:
            m = models[k]
            extra = f", brier={m['brier_score']:.3f}" if "brier_score" in m else ""
            if "vs_market_log_loss_delta" in m:
                extra += f", dLL_vs_mkt={m['vs_market_log_loss_delta']:+.3f}"
            print(
                f"{m['label']}: acc={m['accuracy']:.3f}, AUC={m['auc']:.3f}, "
                f"logloss={m['log_loss']:.3f}{extra} (n_test={m['n_test']})"
            )
    if "market_baselines_win_test" in models and "market_win_baseline" in models.get("market_baselines_win_test", {}):
        m = models["market_baselines_win_test"]["market_win_baseline"]
        print(
            f"Market baseline (moneyline implied prob): acc={m['accuracy']:.3f}, AUC={m['auc']:.3f}, "
            f"logloss={m['log_loss']:.3f} (n_test={m['n_test']})"
        )

    print("\n=== TOTALS MODELS (HELD-OUT TEST) ===")
    for k in [
        "total_model_baseline",
        "total_model_enhanced",
        "total_model_ensemble",
        "total_model_baseline_on_market_subset",
        "total_model_enhanced_on_market_subset",
        "total_model_with_market",
        "margin_model_market_residual",
        "total_model_market_residual",
    ]:
        if k in models:
            m = models[k]
            extra = ""
            if "vs_market_mae_delta" in m:
                extra = f", dMAE_vs_mkt={m['vs_market_mae_delta']:+.2f}"
            print(
                f"{m['label']}: MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R2={m['r2']:.3f}{extra} "
                f"(n_test={m['n_test']})"
            )
    if "market_baselines_total_test" in models and "market_total_baseline" in models.get("market_baselines_total_test", {}):
        m = models["market_baselines_total_test"]["market_total_baseline"]
        print(f"Market baseline (closing total): MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, R2={m['r2']:.3f}")

    # --- SHAP features ---
    if "shap_win_features" in models:
        print("\n=== SHAP WIN FEATURE IMPORTANCE ===")
        for row in models["shap_win_features"][:10]:
            print(f"  {row['feature']}: {row['importance']:.4f}")
    if "selected_win_features" in models:
        print(f"\nSHAP-selected features ({len(models['selected_win_features'])}): "
              f"{', '.join(models['selected_win_features'][:8])}...")

    # --- Top features for market models ---
    if "win_model_with_market" in models:
        print("\n=== TOP WIN FEATURES (WITH MARKET) ===")
        for row in models["win_model_with_market"]["top_features"][:10]:
            print(f"  {row['feature']}: {row['importance']:.4f}")
    if "total_model_with_market" in models:
        print("\n=== TOP TOTAL FEATURES (WITH MARKET) ===")
        for row in models["total_model_with_market"]["top_features"][:10]:
            print(f"  {row['feature']}: {row['importance']:.4f}")

    if "comprehensive_eval" in models and models["comprehensive_eval"]:
        ce = models["comprehensive_eval"]
        print("\n=== COMPREHENSIVE EVAL (NBA_EVALUATE) ===")
        if "win_market_calibrated_holdout" in ce:
            w = ce["win_market_calibrated_holdout"]
            print(
                f"Win market-calibrated holdout: Brier={w.get('brier_score', float('nan')):.4f}, "
                f"ECE={w.get('calibration_error', float('nan')):.4f}, n={w.get('n_games', 0)}"
            )
            if "ats" in w and w["ats"].get("n", 0):
                print(f"  ATS={w['ats']['accuracy']:.3f} (n={w['ats']['n']})")
            if "profit_loss" in w:
                pl_rows = [r for r in w["profit_loss"] if r.get("n_bets", 0) >= MIN_ROI_BETS_PRINT]
                if pl_rows:
                    best = max(pl_rows, key=lambda r: r.get("roi_pct", -1e9))
                    sample_note = ""
                    if best["n_bets"] < 50:
                        sample_note = "  (caution: small sample)"
                    print(
                        f"  Best simulated ROI threshold={best['edge_threshold']:.0%}: "
                        f"ROI={best['roi_pct']:+.1f}% on {best['n_bets']} bets{sample_note}"
                    )
                else:
                    print(
                        f"  Best simulated ROI: insufficient sample size "
                        f"(need >= {MIN_ROI_BETS_PRINT} bets per threshold)"
                    )
        for key in ["total_market_holdout", "total_market_residual_holdout"]:
            if key in ce:
                t = ce[key]
                ou = t.get("over_under", {})
                ou_txt = f", O/U={ou.get('accuracy', float('nan')):.3f} (n={ou.get('n', 0)})" if ou.get("n", 0) else ""
                print(f"{key}: MAE={t.get('mae', 0):.2f}, RMSE={t.get('rmse', 0):.2f}, R2={t.get('r2', 0):.3f}{ou_txt}")
        for key in [
            "win_cv_enhanced_calibrated_ci",
            "win_cv_ensemble_ci",
            "win_cv_market_calibrated_ci",
            "win_cv_market_ensemble_ci",
            "total_cv_enhanced_ci",
            "total_cv_ensemble_ci",
            "total_cv_market_ci",
            "total_cv_market_residual_ci",
        ]:
            if key in ce:
                ci = ce[key]
                parts = []
                for metric in ["auc", "log_loss", "brier_score", "mae", "r2"]:
                    if metric in ci and isinstance(ci[metric], dict):
                        parts.append(f"{metric}={ci[metric]['mean']:.3f}+/-{ci[metric]['std']:.3f}")
                if parts:
                    print(f"{key}: " + ", ".join(parts))

    # --- Practical buckets ---
    if summary.get("heuristic_buckets"):
        print("\n=== PRACTICAL BUCKETS ===")
        if "home_win_by_pregame_net_rating_diff" in summary["heuristic_buckets"]:
            print("Pregame net rating diff buckets -> home win%:")
            for row in summary["heuristic_buckets"]["home_win_by_pregame_net_rating_diff"]:
                print(f"  {row['nr_bin']}: n={int(row['count'])}, win%={row['mean']:.3f}")
        if "totals_by_pregame_pace_quartile" in summary["heuristic_buckets"]:
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
    INJURY_REPORT_CACHE.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Building team/player game datasets with travel + availability proxy...", flush=True)
    schedule_df, team_games, player_games = build_team_games_and_players(include_historical=True)

    # Merge ESPN odds for current season (live fetch)
    print("Matching ESPN events and pulling odds for current season...", flush=True)
    current_sched = schedule_df[schedule_df["season"] == SEASON].copy()
    current_with_odds = join_espn_odds(current_sched)

    # Load historical ESPN odds from cache
    all_odds_dfs = [current_with_odds]
    for hist_season in SEASONS[:-1]:
        hist_odds = load_historical_espn_odds(hist_season)
        if not hist_odds.empty:
            hist_sched = schedule_df[schedule_df["season"] == hist_season].copy()
            merged = hist_sched.merge(
                hist_odds, on=["game_date_est", "home_team", "away_team"], how="left"
            )
            all_odds_dfs.append(merged)
            print(f"  Loaded {hist_season} odds: {hist_odds['espn_event_id'].notna().sum()} events", flush=True)

    schedule_with_odds = pd.concat(all_odds_dfs, ignore_index=True)
    schedule_with_odds = schedule_with_odds.sort_values(["game_time_utc", "game_id"]).reset_index(drop=True)

    print("Building referee tendency profiles...", flush=True)
    ref_features = build_referee_game_features(team_games)

    games = build_game_level(team_games, schedule_with_odds, ref_features=ref_features)
    # Propagate season column to game level
    if "season" in team_games.columns:
        season_map = team_games[team_games["is_home"] == 1][["game_id", "season"]].drop_duplicates("game_id")
        if "season" not in games.columns:
            games = games.merge(season_map, on="game_id", how="left")

    team_games.to_csv(ADV_TEAM_GAMES_CSV, index=False)
    player_games.to_csv(ADV_PLAYER_GAMES_CSV, index=False)
    games.to_csv(ADV_GAMES_CSV, index=False)
    schedule_with_odds.to_csv(ADV_ODDS_CSV, index=False)

    print("Training advanced models with CV + tuning...", flush=True)
    models = run_advanced_models(games, tune=True, n_tune_trials=100)
    summary = summarize_data(schedule_with_odds, team_games, player_games, games)

    payload = {"summary": summary, "models": models}
    ADV_SUMMARY_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print_results(summary, models)

    # Save trained models for prediction script
    print("Persisting models to disk...", flush=True)
    joblib.dump(models, MODEL_DIR / "advanced_models.joblib")
    joblib.dump({"games": games, "team_games": team_games}, MODEL_DIR / "training_data.joblib")

    print(f"\nSaved: {ADV_TEAM_GAMES_CSV}")
    print(f"Saved: {ADV_PLAYER_GAMES_CSV}")
    print(f"Saved: {ADV_GAMES_CSV}")
    print(f"Saved: {ADV_ODDS_CSV}")
    print(f"Saved: {ADV_SUMMARY_JSON}")
    print(f"Saved: {MODEL_DIR / 'advanced_models.joblib'}")


if __name__ == "__main__":
    main()
