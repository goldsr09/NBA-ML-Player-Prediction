#!/usr/bin/env python3
"""Player props prediction pipeline.

Predicts player-level outcomes (points, rebounds, assists, minutes, 3-pointers
made) for upcoming NBA games. Uses the existing player-game data from the
monolith and builds player-level features with team/opponent context.

Features:
  - Two-stage modeling: predict minutes first, then per-minute rates
  - Enhanced features: EWM, venue splits, matchup context, usage dynamics
  - ESPN / The Odds API / manual CSV prop line fetching
  - Walk-forward backtest with season-based folds
  - CLV tracking for prop predictions

Usage:
    cd /Users/ryangoldstein/NBA

    # Predict player props for today's games
    python3 scripts/predict_player_props.py --date 20260227

    # Backtest on historical data
    python3 scripts/predict_player_props.py --backtest

    # Walk-forward backtest (season-based folds)
    python3 scripts/predict_player_props.py --walk-forward

    # Backtest prop edge signals
    python3 scripts/predict_player_props.py --backtest-props

    # Track CLV after games complete
    python3 scripts/predict_player_props.py --track-clv --date 20260226
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import unicodedata
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from pandas.errors import PerformanceWarning
from scipy import stats as sp_stats
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from analyze_nba_2025_26_advanced import (
    BOXSCORE_CACHE,
    CACHE_DIR,
    ESPN_SCOREBOARD_URL,
    HIST_CACHE_DIR,
    SCHEDULE_URL,
    SEASON,
    SEASONS,
    TEAM_COORDS,
    _minutes_to_float,
    _nan_or,
    _to_float,
    add_player_availability_proxy,
    build_referee_game_features,
    build_team_games_and_players,
    fetch_espn_injury_report,
    fetch_json,
    haversine_miles,
    join_espn_odds,
    load_historical_espn_odds,
    normalize_espn_abbr,
)
from nba_evaluate import prop_brier_score, prop_calibration_by_bucket

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=PerformanceWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis" / "output"
PREDICTIONS_DIR = OUT_DIR / "predictions"
MODEL_DIR = OUT_DIR / "models"
PROP_LINES_DIR = OUT_DIR / "prop_lines"
PROP_CACHE_DIR = OUT_DIR / "prop_cache"
PROP_LOG_DIR = OUT_DIR / "prop_logs"
BOX_ADV_CACHE_DIR = PROP_CACHE_DIR / "boxscore_advanced_v3_raw"
MARKET_MODEL_MIN_ROWS = 500
MARKET_CALIB_MIN_ROWS = 250
_EXTENDED_STATS_CACHE: pd.DataFrame | None = None  # Invalidated when new fields are added
_BOX_ADV_CACHE: pd.DataFrame | None = None
PLAYER_FEATURE_CACHE_FILE = MODEL_DIR / "player_features_cache.pkl"
PLAYER_FEATURE_CACHE_META = MODEL_DIR / "player_features_cache_meta.json"
PLAYER_FEATURE_CACHE_VERSION = "v4"  # v4: market line features + residual model
NO_LINES_RETRY_SECS_SAME_DAY = 45 * 60
BOX_ADV_REQUEST_SLEEP_SECS = 1.0
BOX_ADV_DEFAULT_RETRIES = 5
BOX_ADV_DEFAULT_TIMEOUT = 20

# Props we predict
PROP_TARGETS = ["points", "rebounds", "assists", "minutes"]

# Minimum games for a player to be modeled
DEFAULT_MIN_GAMES = 20

# Betting parameters for props
VIG_FACTOR = 0.9524        # net payout per $1 at ~-110 juice
BREAKEVEN_PROB = 1.0 / (1.0 + VIG_FACTOR)  # ~0.5122

# Signal thresholds
MIN_EDGE_PCT = 15.0        # minimum edge% to signal (e.g., pred 23 vs line 20 = 15%)
MIN_EV = 0.20              # minimum EV to signal (20 cents per dollar)
BEST_BET_EV = 0.40         # EV threshold for "best bet" flag
MAX_SIGNALS_PER_DAY = 10   # cap total signals to avoid overexposure
USE_BOXSCORE_ADV_FEATURES = True

# Signal policy controls
SIGNAL_POINTS_ONLY = False
MIN_SIGNAL_PRED_MINUTES = 20.0
MIN_SIGNAL_PRE_MINUTES_AVG10 = 18.0

# Side-specific thresholds (stricter for OVER to counter observed over-bias)
MIN_EDGE_PCT_BY_SIDE = {
    "OVER": 18.0,
    "UNDER": MIN_EDGE_PCT,
}
MIN_EV_BY_SIDE = {
    "OVER": 0.25,
    "UNDER": MIN_EV,
}

# Paper-trading activation thresholds
PAPER_PHASE_TOTAL_MIN_ROWS = 500
PAPER_PHASE_PER_STAT_MIN_ROWS = 200
PAPER_PHASE_STATS = ("points", "rebounds", "assists", "fg3m")
PAPER_PHASE_MIN_READY_STATS = 3  # "ideally all 4"; require 3+ to start exploratory mode

# Calibration drift thresholds (from weekly actionable backtest)
MIN_SETTLED_FOR_DRIFT_CHECK = 30
CALIB_DRIFT_TIGHTEN_THRESHOLD = 0.08

# Calibration alert thresholds (Phase 2: calibration monitoring)
CALIB_ALERT_THRESHOLDS = {
    "hit_rate_vs_predicted_gap": 0.08,  # |hit_rate - mean_p_hit| > 8%
    "brier_score_max": 0.30,            # worse than random (0.25)
    "roi_floor_pct": -8.0,              # ROI below -8%
}
CALIB_MIN_SAMPLE = 50       # minimum graded rows per slice for calibration metrics
CALIB_DEFAULT_LOOKBACK_DAYS = 90
CALIB_RELIABILITY_LOOKBACK_DAYS = 30
CALIB_RELIABILITY_MIN_SAMPLE = 50

# Canonical results history (Phase 1)
PROP_RESULTS_HISTORY_FILE = PREDICTIONS_DIR / "prop_results_history.csv"
MODEL_VERSION = "v4"  # Bump when model architecture changes significantly

# Persistent monitoring logs
MARKET_PROGRESS_LOG = PROP_LOG_DIR / "market_data_progress.csv"
MARKET_WEEKLY_LOG = PROP_LOG_DIR / "market_weekly_actionable_backtest.csv"

# Signal policy presets (baseline -> exploratory -> tightened)
SIGNAL_POLICY_PRESETS: dict[str, dict[str, Any]] = {
    "baseline": {
        "signal_points_only": SIGNAL_POINTS_ONLY,
        "min_pred_minutes": MIN_SIGNAL_PRED_MINUTES,
        "min_pre_minutes_avg10": MIN_SIGNAL_PRE_MINUTES_AVG10,
        "min_edge_pct_by_side": dict(MIN_EDGE_PCT_BY_SIDE),
        "min_ev_by_side": dict(MIN_EV_BY_SIDE),
        "best_bet_ev": BEST_BET_EV,
        "max_signals_per_day": MAX_SIGNALS_PER_DAY,
    },
    "exploratory": {
        "signal_points_only": False,
        "min_pred_minutes": 18.0,
        "min_pre_minutes_avg10": 16.0,
        "min_edge_pct_by_side": {"OVER": 16.0, "UNDER": 13.0},
        "min_ev_by_side": {"OVER": 0.22, "UNDER": 0.17},
        "best_bet_ev": 0.35,
        "max_signals_per_day": 12,
    },
    "tightened": {
        "signal_points_only": False,
        "min_pred_minutes": 22.0,
        "min_pre_minutes_avg10": 20.0,
        "min_edge_pct_by_side": {"OVER": 20.0, "UNDER": 17.0},
        "min_ev_by_side": {"OVER": 0.30, "UNDER": 0.24},
        "best_bet_ev": 0.45,
        "max_signals_per_day": 8,
    },
}
ACTIVE_SIGNAL_POLICY_MODE = "baseline"

# Minimum absolute edge per stat type (prevents noise on low lines like assists 2.5)
MIN_ABS_EDGE = {
    "points": 3.0,
    "rebounds": 1.5,
    "assists": 1.5,
    "fg3m": 0.8,
    "minutes": 3.0,
    "steals": 0.5,
}

# Position mapping for matchup features (approximate)
POSITION_GROUPS = {
    "G": ["PG", "SG", "G"],
    "F": ["SF", "PF", "F"],
    "C": ["C"],
}

ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
INJURY_REMOVE_STATUSES = {"out", "suspension", "ofs"}
INJURY_HIGH_RISK_STATUSES = {"doubtful"}


def normalize_player_name(name: Any) -> str:
    """Normalize player name for robust matching across data sources."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def generate_prediction_id(date: str, name_norm: str, stat_type: str, team: str = "") -> str:
    """Deterministic dedup key: '{date}_{team}_{name_norm}_{stat_type}'.

    Team is included to avoid same-name collisions across teams.
    """
    team_norm = (team or "").upper().strip()
    return f"{date}_{team_norm}_{name_norm}_{stat_type}"


def _injury_key(team: str, player_name: str) -> str:
    return f"{(team or '').upper()}|{normalize_player_name(player_name)}"


def fetch_injury_status_map(target_date: str) -> dict[str, dict[str, Any]]:
    """Fetch injury report rows indexed by `TEAM|normalized_player_name`."""
    injuries = fetch_espn_injury_report(cache_key=target_date)
    out: dict[str, dict[str, Any]] = {}
    for inj in injuries:
        team = str(inj.get("team", "")).upper()
        name = str(inj.get("player_name", ""))
        if not team or not name:
            continue
        status = str(inj.get("status", "")).strip().lower()
        avail_prob = _to_float(inj.get("status_prob"))
        if pd.isna(avail_prob):
            avail_prob = 0.5
        out[_injury_key(team, name)] = {
            "team": team,
            "player_name": name,
            "status": status,
            "availability_prob": float(np.clip(avail_prob, 0.0, 1.0)),
        }
    return out


def fetch_confirmed_starters_for_event(espn_event_id: str) -> dict[str, set[str]]:
    """Return confirmed starters from ESPN summary boxscore for an event.

    Output shape: {TEAM_TRICODE: {"normalized name", ...}, ...}
    """
    if not espn_event_id:
        return {}
    try:
        payload = fetch_json(ESPN_SUMMARY_URL.format(event_id=espn_event_id), timeout=20, retries=2)
    except Exception:
        return {}

    starters: dict[str, set[str]] = defaultdict(set)
    teams = payload.get("boxscore", {}).get("players", [])
    for team_entry in teams:
        team_obj = (team_entry.get("team", {}) if isinstance(team_entry, dict) else {}) or {}
        team_abbr = team_obj.get("abbreviation")
        team = normalize_espn_abbr(str(team_abbr or ""))
        if not team:
            continue
        stats_blocks = team_entry.get("statistics", []) or []
        for block in stats_blocks:
            for ath in block.get("athletes", []) or []:
                if not ath.get("starter", False):
                    continue
                athlete_obj = ath.get("athlete", {}) or {}
                name = athlete_obj.get("displayName", "")
                norm_name = normalize_player_name(name)
                if norm_name:
                    starters[team].add(norm_name)
    return dict(starters)


def fetch_confirmed_starters(
    upcoming: pd.DataFrame,
    lock_minutes: int = 30,
    force: bool = False,
) -> dict[tuple[str, str, str], dict[str, set[str]]]:
    """Fetch confirmed starters for games near tip.

    Returns mapping key=(game_date_est, home_team, away_team) -> {team: starters}.
    """
    if upcoming.empty:
        return {}
    now_utc = pd.Timestamp.now(tz="UTC")
    out: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    for _, row in upcoming.iterrows():
        key = (
            str(row.get("game_date_est", "")),
            str(row.get("home_team", "")),
            str(row.get("away_team", "")),
        )
        start_utc = pd.to_datetime(row.get("game_start_utc"), utc=True, errors="coerce")
        in_window = False
        if pd.notna(start_utc):
            mins = (start_utc - now_utc).total_seconds() / 60.0
            in_window = (mins >= -15) and (mins <= float(lock_minutes))
        if (not force) and (not in_window):
            continue
        event_id = str(row.get("espn_event_id", ""))
        starters = fetch_confirmed_starters_for_event(event_id)
        if starters:
            out[key] = starters
    return out


def apply_injury_status_to_predictions(
    pred_df: pd.DataFrame,
    injury_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Attach per-player injury availability fields to predictions rows."""
    if pred_df.empty:
        return pred_df
    out = pred_df.copy()
    status: list[str] = []
    avail: list[float] = []
    for _, row in out.iterrows():
        key = _injury_key(str(row.get("team", "")), str(row.get("player_name", "")))
        inj = injury_map.get(key)
        if inj:
            status.append(str(inj.get("status", "")))
            avail.append(float(inj.get("availability_prob", 0.5)))
        else:
            status.append("")
            avail.append(np.nan)
    out["injury_status"] = status
    out["injury_availability_prob"] = avail
    out["injury_unavailability_prob"] = 1.0 - out["injury_availability_prob"].astype(float)
    out["injury_is_out"] = out["injury_status"].isin(sorted(INJURY_REMOVE_STATUSES)).astype(int)
    out["injury_is_doubtful"] = out["injury_status"].isin(sorted(INJURY_HIGH_RISK_STATUSES)).astype(int)
    out["injury_is_questionable"] = out["injury_status"].eq("questionable").astype(int)
    out["injury_is_probable"] = out["injury_status"].eq("probable").astype(int)
    return out


def filter_out_inactive(
    pred_df: pd.DataFrame,
    injury_map: dict[str, dict[str, Any]],
    remove_doubtful: bool = True,
) -> pd.DataFrame:
    """Remove players with status that should not be projected for betting."""
    if pred_df.empty or not injury_map:
        return pred_df
    blocked = set(INJURY_REMOVE_STATUSES)
    if remove_doubtful:
        blocked |= INJURY_HIGH_RISK_STATUSES
    keys_to_remove = {
        key for key, inj in injury_map.items()
        if str(inj.get("status", "")).strip().lower() in blocked
    }
    if not keys_to_remove:
        return pred_df
    key_series = pred_df.apply(
        lambda r: _injury_key(str(r.get("team", "")), str(r.get("player_name", ""))),
        axis=1,
    )
    mask = key_series.isin(keys_to_remove)
    removed = pred_df[mask]
    if not removed.empty:
        names = removed["player_name"].unique().tolist()
        print(f"  Filtered {len(names)} high-risk inactive players: {', '.join(names)}", flush=True)
    return pred_df[~mask].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Prop line fetching: ESPN API + The Odds API + manual CSV
# ---------------------------------------------------------------------------

def _american_odds_to_prob(odds: Any) -> float:
    """Convert American odds to implied probability."""
    if pd.isna(odds) or odds is None:
        return np.nan
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return np.nan
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    elif odds > 0:
        return 100.0 / (odds + 100.0)
    return np.nan


def _american_odds_to_decimal(odds: Any) -> float:
    """Convert American odds to decimal payout (profit per $1 wagered)."""
    if pd.isna(odds) or odds is None:
        return np.nan
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return np.nan
    if odds < 0:
        return 100.0 / (-odds)
    elif odds > 0:
        return odds / 100.0
    return np.nan


def _normalize_team_series(series: pd.Series) -> pd.Series:
    """Normalize team abbreviations while preserving missing values as empty strings."""
    out = series.where(series.notna(), "").astype(str).str.upper().str.strip()
    return out.replace({"NAN": "", "NONE": "", "NULL": ""})


STATS_NBA_ADV_URL = "https://stats.nba.com/stats/boxscoreadvancedv3"
_STATS_NBA_SESSION = requests.Session()
_STATS_NBA_SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
)


def _stats_nba_fetch_json(
    url: str,
    params: dict[str, Any],
    cache_path: Path,
    timeout: int = BOX_ADV_DEFAULT_TIMEOUT,
    retries: int = BOX_ADV_DEFAULT_RETRIES,
) -> dict[str, Any]:
    """Fetch stats.nba.com JSON with retries/backoff and raw response caching."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _STATS_NBA_SESSION.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload))
            return payload
        except Exception as exc:
            last_err = exc
            time.sleep(min(6.0, 0.8 * (attempt + 1)))
    raise RuntimeError(f"Failed stats.nba fetch for {params}: {last_err}")


def _parse_boxscore_advanced_payload(payload: dict[str, Any], game_id: str) -> list[dict[str, Any]]:
    """Parse player-level advanced metrics from a BoxScoreAdvancedV3 payload."""
    rows: list[dict[str, Any]] = []

    # Shape A: resultSets style
    result_sets = payload.get("resultSets")
    if isinstance(result_sets, list):
        for rs in result_sets:
            headers = rs.get("headers") or rs.get("Headers") or []
            rowset = rs.get("rowSet") or rs.get("row_set") or []
            if not headers or not rowset:
                continue
            h = [str(x) for x in headers]

            def idx(*names: str) -> int:
                for n in names:
                    if n in h:
                        return h.index(n)
                return -1

            pid_i = idx("PLAYER_ID", "PERSON_ID")
            team_i = idx("TEAM_ABBREVIATION", "TEAM_TRICODE")
            usage_i = idx("USG_PCT", "USG_PCT_EST")
            pace_i = idx("PACE")
            poss_i = idx("POSS")
            off_i = idx("OFF_RATING", "OFFRTG")
            ast_i = idx("AST_PCT")
            reb_i = idx("REB_PCT")
            ts_i = idx("TS_PCT")
            if pid_i < 0 or team_i < 0:
                continue

            for r in rowset:
                if not isinstance(r, (list, tuple)):
                    continue
                pid = r[pid_i] if pid_i < len(r) else None
                team = str(r[team_i]).upper().strip() if team_i < len(r) else ""
                if pid is None or not team:
                    continue
                rows.append(
                    {
                        "game_id": str(game_id),
                        "team": team,
                        "player_id": int(pid),
                        "adv_usage_pct": _to_float(r[usage_i]) if usage_i >= 0 and usage_i < len(r) else np.nan,
                        "adv_pace": _to_float(r[pace_i]) if pace_i >= 0 and pace_i < len(r) else np.nan,
                        "adv_possessions": _to_float(r[poss_i]) if poss_i >= 0 and poss_i < len(r) else np.nan,
                        "adv_off_rating": _to_float(r[off_i]) if off_i >= 0 and off_i < len(r) else np.nan,
                        "adv_ast_pct": _to_float(r[ast_i]) if ast_i >= 0 and ast_i < len(r) else np.nan,
                        "adv_reb_pct": _to_float(r[reb_i]) if reb_i >= 0 and reb_i < len(r) else np.nan,
                        "adv_ts_pct": _to_float(r[ts_i]) if ts_i >= 0 and ts_i < len(r) else np.nan,
                    }
                )
        if rows:
            return rows

    # Shape B: boxScoreAdvanced with homeTeam/awayTeam players
    adv_root = payload.get("boxScoreAdvanced") or payload.get("boxscoreadvancedv3") or payload
    for side_key in ("homeTeam", "awayTeam"):
        side = adv_root.get(side_key) if isinstance(adv_root, dict) else None
        if not isinstance(side, dict):
            continue
        team = str(
            side.get("teamTricode")
            or side.get("teamCode")
            or side.get("teamAbbreviation")
            or ""
        ).upper().strip()
        for p in side.get("players", []) or []:
            if not isinstance(p, dict):
                continue
            pid = p.get("personId") or p.get("playerId") or p.get("PLAYER_ID")
            if pid is None:
                continue
            stats = p.get("statistics") or {}
            if not team:
                team = str(p.get("teamTricode") or p.get("teamCode") or "").upper().strip()
            rows.append(
                {
                    "game_id": str(game_id),
                    "team": team,
                    "player_id": int(pid),
                    "adv_usage_pct": _to_float(stats.get("usagePercentage") or stats.get("usgPct") or stats.get("USG_PCT")),
                    "adv_pace": _to_float(stats.get("pace") or stats.get("PACE")),
                    "adv_possessions": _to_float(stats.get("possessions") or stats.get("POSS")),
                    "adv_off_rating": _to_float(stats.get("offensiveRating") or stats.get("offRating") or stats.get("OFF_RATING")),
                    "adv_ast_pct": _to_float(stats.get("assistPercentage") or stats.get("astPct") or stats.get("AST_PCT")),
                    "adv_reb_pct": _to_float(stats.get("reboundPercentage") or stats.get("rebPct") or stats.get("REB_PCT")),
                    "adv_ts_pct": _to_float(stats.get("trueShootingPercentage") or stats.get("tsPct") or stats.get("TS_PCT")),
                }
            )
    return rows


def load_boxscore_advanced_stats(
    game_ids: list[str] | pd.Series | np.ndarray | None = None,
    fetch_missing: bool = False,
    max_fetch: int = 0,
) -> pd.DataFrame:
    """Load BoxScoreAdvancedV3 player stats from local cache, optionally fetching misses."""
    global _BOX_ADV_CACHE
    BOX_ADV_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if game_ids is None:
        wanted_ids = None
    else:
        wanted_ids = sorted({str(g) for g in list(game_ids) if str(g).strip()})
        if not wanted_ids:
            return pd.DataFrame()

    if fetch_missing and wanted_ids:
        missing_ids = [gid for gid in wanted_ids if not (BOX_ADV_CACHE_DIR / f"{gid}.json").exists()]
        if max_fetch > 0:
            missing_ids = missing_ids[:max_fetch]
        for i, gid in enumerate(missing_ids, start=1):
            cache_path = BOX_ADV_CACHE_DIR / f"{gid}.json"
            try:
                _stats_nba_fetch_json(
                    STATS_NBA_ADV_URL,
                    params={"GameID": gid},
                    cache_path=cache_path,
                )
                if i < len(missing_ids):
                    time.sleep(BOX_ADV_REQUEST_SLEEP_SECS)
            except Exception as exc:
                print(f"  Warning: BoxScoreAdvancedV3 fetch failed for {gid}: {exc}", flush=True)

    # Build cache once per process from available raw files.
    if _BOX_ADV_CACHE is None:
        rows: list[dict[str, Any]] = []
        files = sorted(BOX_ADV_CACHE_DIR.glob("*.json"))
        for f in files:
            gid = f.stem
            try:
                payload = json.loads(f.read_text())
                rows.extend(_parse_boxscore_advanced_payload(payload, gid))
            except Exception:
                continue
        if not rows:
            _BOX_ADV_CACHE = pd.DataFrame()
        else:
            df = pd.DataFrame(rows).drop_duplicates(subset=["game_id", "team", "player_id"], keep="last")
            _BOX_ADV_CACHE = df.reset_index(drop=True)

    if _BOX_ADV_CACHE is None or _BOX_ADV_CACHE.empty:
        return pd.DataFrame()
    out = _BOX_ADV_CACHE
    if wanted_ids is not None:
        out = out[out["game_id"].astype(str).isin(wanted_ids)]
    return out.copy()


def apply_signal_policy(mode: str) -> None:
    """Apply one of the predefined signal policy presets globally."""
    global SIGNAL_POINTS_ONLY
    global MIN_SIGNAL_PRED_MINUTES
    global MIN_SIGNAL_PRE_MINUTES_AVG10
    global MIN_EDGE_PCT_BY_SIDE
    global MIN_EV_BY_SIDE
    global BEST_BET_EV
    global MAX_SIGNALS_PER_DAY
    global ACTIVE_SIGNAL_POLICY_MODE
    preset = SIGNAL_POLICY_PRESETS.get(mode) or SIGNAL_POLICY_PRESETS["baseline"]
    SIGNAL_POINTS_ONLY = bool(preset["signal_points_only"])
    MIN_SIGNAL_PRED_MINUTES = float(preset["min_pred_minutes"])
    MIN_SIGNAL_PRE_MINUTES_AVG10 = float(preset["min_pre_minutes_avg10"])
    MIN_EDGE_PCT_BY_SIDE = dict(preset["min_edge_pct_by_side"])
    MIN_EV_BY_SIDE = dict(preset["min_ev_by_side"])
    BEST_BET_EV = float(preset["best_bet_ev"])
    MAX_SIGNALS_PER_DAY = int(preset["max_signals_per_day"])
    ACTIVE_SIGNAL_POLICY_MODE = str(mode)


def summarize_market_coverage(diag: dict[str, Any]) -> dict[str, Any]:
    """Summarize matched market rows overall and per stat from diagnostics."""
    per_stat_diag = diag.get("per_stat", {}) if isinstance(diag, dict) else {}
    per_stat_rows = {
        stat: int((per_stat_diag.get(stat) or {}).get("n_rows", 0))
        for stat in PAPER_PHASE_STATS
    }
    total_rows = int(diag.get("rows", 0)) if isinstance(diag, dict) else 0
    ready_stat_count = int(sum(v >= PAPER_PHASE_PER_STAT_MIN_ROWS for v in per_stat_rows.values()))
    return {
        "total_rows": total_rows,
        "per_stat_rows": per_stat_rows,
        "ready_total": total_rows >= PAPER_PHASE_TOTAL_MIN_ROWS,
        "ready_stat_count": ready_stat_count,
        "ready_all_stats": ready_stat_count == len(PAPER_PHASE_STATS),
    }


def _estimate_rows_per_day(
    progress_df: pd.DataFrame,
    value_col: str,
    lookback_days: int = 14,
) -> tuple[float, float]:
    """Return (rows_per_day, eta_days) for a value column; eta_days uses global target from caller."""
    if progress_df.empty or value_col not in progress_df.columns:
        return np.nan, np.nan
    snap = progress_df[["as_of_date", value_col]].copy()
    snap["as_of_date"] = pd.to_datetime(snap["as_of_date"], errors="coerce")
    snap = snap.dropna(subset=["as_of_date", value_col]).sort_values("as_of_date")
    if snap.empty:
        return np.nan, np.nan
    end_dt = snap["as_of_date"].max()
    start_cut = end_dt - pd.Timedelta(days=lookback_days)
    recent = snap[snap["as_of_date"] >= start_cut]
    if len(recent) < 2:
        recent = snap.tail(2)
    if len(recent) < 2:
        return np.nan, np.nan
    start_row = recent.iloc[0]
    end_row = recent.iloc[-1]
    delta_days = (end_row["as_of_date"] - start_row["as_of_date"]).days
    delta_rows = float(end_row[value_col] - start_row[value_col])
    if delta_days <= 0 or delta_rows <= 0:
        return np.nan, np.nan
    return delta_rows / float(delta_days), np.nan


def record_market_progress(
    as_of_date: str,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    """Persist market-data progress and print current pace/ETA."""
    PROP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    per_stat = coverage.get("per_stat_rows", {})
    row = {
        "as_of_date": str(as_of_date),
        "matched_rows_total": int(coverage.get("total_rows", 0)),
        "rows_points": int(per_stat.get("points", 0)),
        "rows_rebounds": int(per_stat.get("rebounds", 0)),
        "rows_assists": int(per_stat.get("assists", 0)),
        "rows_fg3m": int(per_stat.get("fg3m", 0)),
        "ready_total": int(bool(coverage.get("ready_total", False))),
        "ready_stat_count": int(coverage.get("ready_stat_count", 0)),
        "ready_all_stats": int(bool(coverage.get("ready_all_stats", False))),
    }

    if MARKET_PROGRESS_LOG.exists():
        try:
            hist = pd.read_csv(MARKET_PROGRESS_LOG)
        except Exception:
            hist = pd.DataFrame()
    else:
        hist = pd.DataFrame()
    if not hist.empty and "as_of_date" in hist.columns:
        hist["as_of_date"] = hist["as_of_date"].astype(str)
        hist = hist[hist["as_of_date"] != str(row["as_of_date"])]
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    if "as_of_date" in hist.columns:
        hist["as_of_date"] = hist["as_of_date"].astype(str)
        hist = hist.sort_values("as_of_date").reset_index(drop=True)
    hist.to_csv(MARKET_PROGRESS_LOG, index=False)

    total = int(row["matched_rows_total"])
    remaining_total = max(PAPER_PHASE_TOTAL_MIN_ROWS - total, 0)
    pace_total, _ = _estimate_rows_per_day(hist, "matched_rows_total")
    eta_total = np.nan
    if pd.notna(pace_total) and pace_total > 0:
        eta_total = remaining_total / float(pace_total)
    pace_total_s = f"{pace_total:.1f}/day" if pd.notna(pace_total) else "n/a"
    eta_total_s = "n/a"
    if pd.notna(eta_total):
        eta_total_date = (pd.to_datetime(as_of_date, errors="coerce") + pd.Timedelta(days=int(math.ceil(eta_total))))
        if pd.notna(eta_total_date):
            eta_total_s = eta_total_date.strftime("%Y-%m-%d")

    stat_chunks: list[str] = []
    stat_eta_chunks: list[str] = []
    for st in PAPER_PHASE_STATS:
        col = f"rows_{st}"
        cur = int(row.get(col, 0))
        rem = max(PAPER_PHASE_PER_STAT_MIN_ROWS - cur, 0)
        pace_st, _ = _estimate_rows_per_day(hist, col)
        eta_st_s = "n/a"
        if pd.notna(pace_st) and pace_st > 0:
            eta_days = rem / float(pace_st)
            eta_dt = (pd.to_datetime(as_of_date, errors="coerce") + pd.Timedelta(days=int(math.ceil(eta_days))))
            if pd.notna(eta_dt):
                eta_st_s = eta_dt.strftime("%Y-%m-%d")
        stat_chunks.append(f"{st}={cur}/{PAPER_PHASE_PER_STAT_MIN_ROWS}")
        stat_eta_chunks.append(f"{st} ETA {eta_st_s}")

    print("\nMarket data progress:", flush=True)
    print(
        f"  Matched rows: {total}/{PAPER_PHASE_TOTAL_MIN_ROWS} "
        f"(remaining {remaining_total}, pace {pace_total_s}, ETA {eta_total_s})",
        flush=True,
    )
    print(f"  Per-stat rows: {', '.join(stat_chunks)}", flush=True)
    print(f"  Per-stat ETAs: {', '.join(stat_eta_chunks)}", flush=True)
    print(f"  Progress log: {MARKET_PROGRESS_LOG}", flush=True)

    row["pace_total_rows_per_day"] = float(pace_total) if pd.notna(pace_total) else np.nan
    row["eta_total_date"] = eta_total_s
    return row


def append_weekly_market_check(result: dict[str, Any], as_of_date: str, max_dates: int) -> None:
    """Append one actionable market-backtest snapshot for weekly tracking."""
    PROP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "as_of_date": str(as_of_date),
        "max_dates": int(max_dates),
        "status": str(result.get("status", "unknown")),
        "n_signals": int(result.get("n_signals", 0)) if str(result.get("n_signals", "")).strip() else 0,
        "n_settled": int(result.get("n_settled", 0)) if str(result.get("n_settled", "")).strip() else 0,
        "wins": int(result.get("wins", 0)) if str(result.get("wins", "")).strip() else 0,
        "hit_rate": result.get("hit_rate", np.nan),
        "roi_pct": result.get("roi_pct", np.nan),
        "pnl": result.get("pnl", np.nan),
        "avg_clv_line_pts": result.get("avg_clv_line_pts", np.nan),
        "avg_ev_at_signal": result.get("avg_ev_at_signal", np.nan),
        "calibration_drift_abs": result.get("calibration_drift_abs", np.nan),
        "brier_score": result.get("brier_score", np.nan),
    }
    per_stat = result.get("per_stat", {}) if isinstance(result, dict) else {}
    for st in PAPER_PHASE_STATS:
        st_row = per_stat.get(st, {})
        row[f"{st}_bets"] = int(st_row.get("n_bets", 0)) if st_row else 0
        row[f"{st}_roi_pct"] = st_row.get("roi_pct", np.nan) if st_row else np.nan

    if MARKET_WEEKLY_LOG.exists():
        try:
            hist = pd.read_csv(MARKET_WEEKLY_LOG)
        except Exception:
            hist = pd.DataFrame()
    else:
        hist = pd.DataFrame()
    if not hist.empty and "as_of_date" in hist.columns:
        hist["as_of_date"] = hist["as_of_date"].astype(str)
        hist = hist[hist["as_of_date"] != str(row["as_of_date"])]
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    if "as_of_date" in hist.columns:
        hist["as_of_date"] = hist["as_of_date"].astype(str)
        hist = hist.sort_values("as_of_date").reset_index(drop=True)
    hist.to_csv(MARKET_WEEKLY_LOG, index=False)
    print(f"  Weekly actionable backtest snapshot saved: {MARKET_WEEKLY_LOG}", flush=True)


def load_latest_weekly_market_check() -> dict[str, Any]:
    """Load latest successful actionable weekly market-backtest snapshot."""
    if not MARKET_WEEKLY_LOG.exists():
        return {}
    try:
        df = pd.read_csv(MARKET_WEEKLY_LOG)
    except Exception:
        return {}
    if df.empty or "status" not in df.columns:
        return {}
    ok = df[df["status"].astype(str).str.lower() == "ok"].copy()
    if ok.empty:
        return {}
    if "as_of_date" in ok.columns:
        ok = ok.sort_values("as_of_date")
    return ok.iloc[-1].to_dict()


def choose_signal_policy_mode(
    coverage: dict[str, Any],
    latest_weekly_check: dict[str, Any],
) -> tuple[str, str]:
    """Choose signal policy mode from market coverage + weekly calibration drift."""
    total_rows = int(coverage.get("total_rows", 0))
    per_stat_rows = coverage.get("per_stat_rows", {})
    ready_stats = int(sum(int(per_stat_rows.get(st, 0)) >= PAPER_PHASE_PER_STAT_MIN_ROWS for st in PAPER_PHASE_STATS))
    ready_for_paper = (
        total_rows >= PAPER_PHASE_TOTAL_MIN_ROWS
        and ready_stats >= PAPER_PHASE_MIN_READY_STATS
    )
    if not ready_for_paper:
        return (
            "baseline",
            (
                f"insufficient market coverage (total={total_rows}/{PAPER_PHASE_TOTAL_MIN_ROWS}, "
                f"ready_stats={ready_stats}/{PAPER_PHASE_MIN_READY_STATS})"
            ),
        )

    settled = _to_float(latest_weekly_check.get("n_settled")) if latest_weekly_check else np.nan
    drift = _to_float(latest_weekly_check.get("calibration_drift_abs")) if latest_weekly_check else np.nan

    # Phase 2: also check canonical calibration report for drift signals
    degraded = get_calibration_degraded_stats()
    if degraded:
        return (
            "tightened",
            f"calibration drift for {', '.join(sorted(degraded))} (canonical report)",
        )

    if (
        pd.notna(settled)
        and int(settled) >= MIN_SETTLED_FOR_DRIFT_CHECK
        and pd.notna(drift)
        and float(drift) >= CALIB_DRIFT_TIGHTEN_THRESHOLD
    ):
        return (
            "tightened",
            f"calibration drift elevated ({float(drift):.3f} >= {CALIB_DRIFT_TIGHTEN_THRESHOLD:.3f})",
        )
    if pd.notna(settled) and int(settled) >= MIN_SETTLED_FOR_DRIFT_CHECK and pd.notna(drift):
        return ("exploratory", f"paper phase active, drift acceptable ({float(drift):.3f})")
    return ("exploratory", "paper phase active (awaiting weekly drift sample)")


def _get_espn_event_ids(date_str: str) -> list[dict[str, Any]]:
    """Get ESPN event IDs and team info for games on a given date."""
    try:
        payload = fetch_json(ESPN_SCOREBOARD_URL.format(yyyymmdd=date_str), timeout=20, retries=3)
    except Exception:
        return []

    events = []
    for event in payload.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        events.append({
            "event_id": str(event.get("id", "")),
            "home_team": normalize_espn_abbr(home.get("team", {}).get("abbreviation", "")),
            "away_team": normalize_espn_abbr(away.get("team", {}).get("abbreviation", "")),
        })
    return events


def _resolve_espn_athlete(athlete_id: str, athlete_cache: dict[str, str]) -> str:
    """Resolve an ESPN athlete ID to a display name, with in-memory + disk caching."""
    if athlete_id in athlete_cache:
        return athlete_cache[athlete_id]

    # Check disk cache
    disk_cache = PROP_CACHE_DIR / "espn_athlete_cache.json"
    if not athlete_cache and disk_cache.exists():
        try:
            loaded = json.loads(disk_cache.read_text())
            athlete_cache.update(loaded)
            if athlete_id in athlete_cache:
                return athlete_cache[athlete_id]
        except Exception:
            pass

    # Fetch from ESPN
    url = (
        f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
        f"seasons/2026/athletes/{athlete_id}"
    )
    try:
        data = fetch_json(url, timeout=10, retries=2)
        name = data.get("displayName", data.get("fullName", ""))
        if name:
            athlete_cache[athlete_id] = name
            # Persist to disk periodically (every 20 new entries)
            if len(athlete_cache) % 20 == 0:
                _save_athlete_cache(athlete_cache)
            return name
    except Exception:
        pass

    return ""


def _save_athlete_cache(athlete_cache: dict[str, str]) -> None:
    """Persist athlete ID -> name cache to disk."""
    disk_cache = PROP_CACHE_DIR / "espn_athlete_cache.json"
    PROP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Merge with existing on-disk data
        existing: dict[str, str] = {}
        if disk_cache.exists():
            existing = json.loads(disk_cache.read_text())
        existing.update(athlete_cache)
        disk_cache.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass


# ESPN propBets type name -> our internal stat_type
_ESPN_PROP_TYPE_MAP = {
    "Total Points": "points",
    "Total Rebounds": "rebounds",
    "Total Assists": "assists",
    "Total 3-Point Field Goals": "fg3m",
    "Total Steals": "steals",
}


def fetch_espn_player_props(date_str: str) -> pd.DataFrame:
    """Fetch player props from ESPN propBets endpoint.

    Uses: /v2/sports/basketball/leagues/nba/events/{eid}/competitions/{eid}/odds/100/propBets
    Handles pagination (limit=100&page=N), resolves athlete IDs to names,
    groups over/under pairs, and returns a DataFrame with columns:
        player_name, team, stat_type, line, over_odds, under_odds,
        open_line, open_over_odds, open_under_odds, source
    """
    events = _get_espn_event_ids(date_str)
    if not events:
        return pd.DataFrame()

    # Check JSON cache first
    json_cache = PROP_CACHE_DIR / f"espn_props_{date_str}.json"
    if json_cache.exists():
        try:
            cached_rows = json.loads(json_cache.read_text())
            if cached_rows:
                print(f"  Loaded {len(cached_rows)} cached ESPN prop entries for {date_str}", flush=True)
                df = pd.DataFrame(cached_rows)
                df["player_name"] = df["player_name"].str.strip()
                df["stat_type"] = df["stat_type"].str.strip().str.lower()
                return df
        except Exception:
            pass

    # Load athlete name cache (ID -> name)
    athlete_cache: dict[str, str] = {}
    disk_cache_path = PROP_CACHE_DIR / "espn_athlete_cache.json"
    if disk_cache_path.exists():
        try:
            athlete_cache = json.loads(disk_cache_path.read_text())
        except Exception:
            pass

    all_rows: list[dict[str, Any]] = []

    for ev in events:
        eid = ev["event_id"]
        home_team = ev["home_team"]
        away_team = ev["away_team"]

        print(f"  Fetching props for {away_team} @ {home_team} (event {eid})...", flush=True)

        # Collect all raw prop items across pages
        raw_items: list[dict[str, Any]] = []
        page = 1
        max_pages = 10  # safety limit

        while page <= max_pages:
            url = (
                f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/"
                f"events/{eid}/competitions/{eid}/odds/100/propBets"
                f"?limit=100&page={page}"
            )
            try:
                data = fetch_json(url, timeout=15, retries=2)
            except Exception as exc:
                print(f"    Page {page} fetch failed: {exc}", flush=True)
                break

            items = data.get("items", [])
            if not items:
                break

            raw_items.extend(items)

            page_count = data.get("pageCount", 1)
            if page >= page_count:
                break
            page += 1
            time.sleep(0.15)  # light rate limiting

        if not raw_items:
            print(f"    No prop items found for event {eid}", flush=True)
            continue

        print(f"    Fetched {len(raw_items)} raw prop items across {page} page(s)", flush=True)

        # Group raw items into over/under pairs.
        # ESPN returns two consecutive entries per athlete+type:
        #   first entry = over, second entry = under.
        # We group by (athlete_id, type_id) and expect exactly 2 entries per group.
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

        for item in raw_items:
            # Extract athlete ID from $ref URL
            athlete_ref = item.get("athlete", {}).get("$ref", "")
            athlete_id = ""
            if "/athletes/" in athlete_ref:
                athlete_id = athlete_ref.split("/athletes/")[-1].split("?")[0].split("/")[0]

            type_info = item.get("type", {})
            type_name = type_info.get("name", "")

            if not athlete_id or not type_name:
                continue

            grouped[(athlete_id, type_name)].append(item)

        # Process each athlete+type group
        for (athlete_id, type_name), entries in grouped.items():
            stat_type = _ESPN_PROP_TYPE_MAP.get(type_name)
            if not stat_type:
                continue

            # Resolve athlete name
            player_name = _resolve_espn_athlete(athlete_id, athlete_cache)
            if not player_name:
                continue

            # Parse over/under from entries.
            # First entry = over, second = under (per ESPN convention).
            over_entry = entries[0] if len(entries) >= 1 else {}
            under_entry = entries[1] if len(entries) >= 2 else {}

            # Current line (should be same on both)
            current_over = over_entry.get("current", {}).get("target", {})
            line = current_over.get("value", np.nan)
            if pd.isna(line) or line is None:
                continue

            # Open line
            open_over = over_entry.get("open", {}).get("target", {})
            open_line = open_over.get("value", np.nan)

            # Over odds (from first entry)
            over_odds_data = over_entry.get("odds", {})
            over_american = over_odds_data.get("american", {}).get("value", np.nan)
            open_over_american = over_odds_data.get("american", {}).get("open", np.nan)

            # Under odds (from second entry)
            under_odds_data = under_entry.get("odds", {})
            under_american = under_odds_data.get("american", {}).get("value", np.nan)
            open_under_american = under_odds_data.get("american", {}).get("open", np.nan)

            # Convert odds strings to floats
            def _safe_float(v: Any) -> float:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return np.nan
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return np.nan

            all_rows.append({
                "player_name": player_name,
                "team": "",  # will be enriched below if possible
                "stat_type": stat_type,
                "line": float(line),
                "over_odds": _safe_float(over_american),
                "under_odds": _safe_float(under_american),
                "open_line": _safe_float(open_line),
                "open_over_odds": _safe_float(open_over_american),
                "open_under_odds": _safe_float(open_under_american),
                "source": "espn_propbets",
                "home_team": home_team,
                "away_team": away_team,
            })

    # Save athlete cache
    _save_athlete_cache(athlete_cache)

    if not all_rows:
        return pd.DataFrame()

    # Cache raw results to JSON
    PROP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    json_cache.write_text(json.dumps(all_rows, indent=2, default=str))
    print(f"  Cached {len(all_rows)} ESPN prop lines to {json_cache}", flush=True)

    df = pd.DataFrame(all_rows)
    df["player_name"] = df["player_name"].str.strip()
    df["stat_type"] = df["stat_type"].str.strip().str.lower()
    return df


def fetch_odds_api_player_props(date_str: str) -> pd.DataFrame:
    """Fetch player props from The Odds API (free tier).

    Requires ODDS_API_KEY environment variable or config file.
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        key_file = PROJECT_ROOT / ".odds_api_key"
        if key_file.exists():
            api_key = key_file.read_text().strip()
    if not api_key:
        return pd.DataFrame()

    import requests

    # Get events for this date
    base_url = "https://api.the-odds-api.com/v4/sports/basketball_nba"
    try:
        resp = requests.get(
            f"{base_url}/events",
            params={"apiKey": api_key, "dateFormat": "iso"},
            timeout=20,
        )
        resp.raise_for_status()
        events = resp.json()
    except Exception as exc:
        print(f"  Odds API events fetch failed: {exc}", flush=True)
        return pd.DataFrame()

    # Filter events for the target date
    target = datetime.strptime(date_str, "%Y%m%d").date()
    matching_events = []
    for ev in events:
        try:
            ev_date = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")).date()
            if ev_date == target:
                matching_events.append(ev)
        except Exception:
            continue

    if not matching_events:
        return pd.DataFrame()

    # Stat type mapping for Odds API market names
    market_map = {
        "player_points": "points",
        "player_rebounds": "rebounds",
        "player_assists": "assists",
        "player_threes": "fg3m",
    }
    markets_str = ",".join(market_map.keys())

    rows: list[dict[str, Any]] = []
    for ev in matching_events:
        eid = ev["id"]
        try:
            resp = requests.get(
                f"{base_url}/events/{eid}/odds",
                params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": markets_str,
                    "oddsFormat": "american",
                },
                timeout=20,
            )
            resp.raise_for_status()
            odds_data = resp.json()
        except Exception as exc:
            print(f"  Odds API props fetch failed for {eid}: {exc}", flush=True)
            continue

        # Parse bookmaker data - take first available bookmaker
        for bk in odds_data.get("bookmakers", []):
            for market in bk.get("markets", []):
                market_key = market.get("key", "")
                stat_type = market_map.get(market_key)
                if not stat_type:
                    continue

                outcomes = market.get("outcomes", [])
                # Group outcomes by player (Over/Under pairs)
                player_lines: dict[str, dict[str, Any]] = {}
                for oc in outcomes:
                    name = oc.get("description", "")
                    if not name:
                        continue
                    side = oc.get("name", "").lower()
                    price = oc.get("price", np.nan)
                    point = oc.get("point", np.nan)

                    if name not in player_lines:
                        player_lines[name] = {"line": point}
                    if side == "over":
                        player_lines[name]["over_odds"] = price
                        player_lines[name]["line"] = point
                    elif side == "under":
                        player_lines[name]["under_odds"] = price

                for pname, pdata in player_lines.items():
                    rows.append({
                        "player_name": pname,
                        "team": "",
                        "stat_type": stat_type,
                        "line": float(pdata.get("line", np.nan)),
                        "over_odds": float(pdata.get("over_odds", np.nan)),
                        "under_odds": float(pdata.get("under_odds", np.nan)),
                        "source": f"odds_api_{bk.get('key', '')}",
                    })
            break  # just take first bookmaker to avoid duplicates

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["player_name"] = df["player_name"].str.strip()
    return df


def load_manual_prop_lines(target_date: str, override_path: str | None = None) -> pd.DataFrame:
    """Load player prop lines from a manual CSV file.

    Expected CSV columns: date, player_name, team, stat_type, line, over_odds, under_odds
    """
    if override_path:
        csv_path = Path(override_path)
    else:
        csv_path = PROP_LINES_DIR / f"prop_lines_{target_date}.csv"

    if not csv_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  Warning: could not read prop lines from {csv_path}: {exc}", flush=True)
        return pd.DataFrame()

    required_cols = {"player_name", "stat_type", "line"}
    if not required_cols.issubset(df.columns):
        print(f"  Warning: prop lines CSV missing required columns. Need: {required_cols}", flush=True)
        return pd.DataFrame()

    df["stat_type"] = df["stat_type"].str.strip().str.lower()
    df["player_name"] = df["player_name"].str.strip()
    if "team" in df.columns:
        df["team"] = _normalize_team_series(df["team"])
    else:
        df["team"] = ""
    if "source" not in df.columns:
        df["source"] = "manual"

    return df


def fetch_player_prop_lines(date_str: str, override_path: str | None = None) -> pd.DataFrame:
    """Fetch player prop lines from multiple sources with fallback.

    Priority: 1) Manual CSV (if exists), 2) ESPN API, 3) The Odds API
    Results are cached to PROP_CACHE_DIR.
    """
    PROP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = PROP_CACHE_DIR / f"prop_lines_{date_str}.csv"
    no_lines_file = PROP_CACHE_DIR / f"prop_lines_{date_str}.none.json"

    # Skip repeated refetches when we recently observed no lines.
    if no_lines_file.exists():
        try:
            marker = json.loads(no_lines_file.read_text())
            now_ts = time.time()
            marker_ts = float(marker.get("ts", 0))
            today_str = datetime.now().strftime("%Y%m%d")
            is_past_date = date_str < today_str
            recent_same_day_miss = (date_str == today_str) and ((now_ts - marker_ts) < NO_LINES_RETRY_SECS_SAME_DAY)
            if is_past_date or recent_same_day_miss:
                reason = "historical no-lines marker" if is_past_date else "recent no-lines marker"
                print(f"  Skipping line fetch for {date_str} ({reason})", flush=True)
                return pd.DataFrame()
        except Exception:
            pass

    # 1) Check for manual override
    manual = load_manual_prop_lines(date_str, override_path)
    if not manual.empty:
        print(f"  Loaded {len(manual)} manual prop lines", flush=True)
        _add_implied_probs(manual)
        manual.to_csv(cache_file, index=False)
        no_lines_file.unlink(missing_ok=True)
        return manual

    # 2) Check cache
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file)
            if not cached.empty:
                print(f"  Loaded {len(cached)} cached prop lines for {date_str}", flush=True)
                _add_implied_probs(cached)
                no_lines_file.unlink(missing_ok=True)
                return cached
        except Exception:
            pass

    # 3) Try ESPN
    print("  Trying ESPN player props API...", flush=True)
    espn_lines = fetch_espn_player_props(date_str)
    if not espn_lines.empty:
        print(f"  Found {len(espn_lines)} ESPN prop lines", flush=True)
        _add_implied_probs(espn_lines)
        espn_lines.to_csv(cache_file, index=False)
        no_lines_file.unlink(missing_ok=True)
        return espn_lines

    # 4) Try The Odds API
    print("  ESPN props not available. Trying The Odds API...", flush=True)
    odds_api_lines = fetch_odds_api_player_props(date_str)
    if not odds_api_lines.empty:
        print(f"  Found {len(odds_api_lines)} Odds API prop lines", flush=True)
        _add_implied_probs(odds_api_lines)
        odds_api_lines.to_csv(cache_file, index=False)
        no_lines_file.unlink(missing_ok=True)
        return odds_api_lines

    # 5) No lines available
    print(f"  No prop lines found for {date_str}.", flush=True)
    print(f"  To add manually, create: {PROP_LINES_DIR / f'prop_lines_{date_str}.csv'}", flush=True)
    print(f"  Required columns: player_name, stat_type, line", flush=True)
    print(f"  Optional columns: team, over_odds, under_odds, date", flush=True)
    try:
        no_lines_file.write_text(json.dumps({"date": date_str, "ts": time.time()}, indent=2))
    except Exception:
        pass
    return pd.DataFrame()


def load_prop_lines_for_dates(
    dates: list[str],
    fetch_missing: bool = False,
) -> pd.DataFrame:
    """Load prop lines for a list of YYYYMMDD dates from cache (or fetch on demand)."""
    all_lines: list[pd.DataFrame] = []
    for date_str in sorted(set(dates)):
        cache_file = PROP_CACHE_DIR / f"prop_lines_{date_str}.csv"
        df = pd.DataFrame()
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file)
            except Exception:
                df = pd.DataFrame()
        elif fetch_missing:
            df = fetch_player_prop_lines(date_str)

        if df.empty:
            continue

        if "game_date_est" not in df.columns:
            if "date" in df.columns:
                df["game_date_est"] = (
                    df["date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
                )
            else:
                df["game_date_est"] = date_str
        else:
            df["game_date_est"] = df["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)

        if "stat_type" in df.columns:
            df["stat_type"] = df["stat_type"].astype(str).str.strip().str.lower()
        if "player_name" in df.columns:
            df["player_name"] = df["player_name"].astype(str).str.strip()
        if "team" in df.columns:
            df["team"] = _normalize_team_series(df["team"])
        else:
            df["team"] = ""

        _add_implied_probs(df)
        all_lines.append(df)

    if not all_lines:
        return pd.DataFrame()
    out = pd.concat(all_lines, ignore_index=True)
    return out


def _add_implied_probs(df: pd.DataFrame) -> None:
    """Add implied probability columns to prop lines DataFrame (in place)."""
    if "over_odds" in df.columns:
        df["over_implied_prob"] = df["over_odds"].apply(_american_odds_to_prob)
    else:
        df["over_implied_prob"] = np.nan
    if "under_odds" in df.columns:
        df["under_implied_prob"] = df["under_odds"].apply(_american_odds_to_prob)
    else:
        df["under_implied_prob"] = np.nan


# ---------------------------------------------------------------------------
# Extended player-game parsing: add fg3m, fta, tov, fga, fgm, steals, blocks
# ---------------------------------------------------------------------------

def _parse_extended_player_stats(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse extended player stats from a boxscore game dict."""
    rows: list[dict[str, Any]] = []
    game_id = str(game["gameId"])
    # OT detection: period > 4 means overtime
    game_period = int(game.get("period", 4))
    n_ot_periods = max(0, game_period - 4)
    for side_key in ("homeTeam", "awayTeam"):
        team = game[side_key]
        team_code = team["teamTricode"]
        for p in team.get("players", []):
            stats = p.get("statistics") or {}
            played_flag = str(p.get("played", "0")) == "1"
            if not played_flag:
                continue
            pid = p.get("personId")
            if pid is None:
                continue
            rows.append({
                "game_id": game_id,
                "team": team_code,
                "player_id": int(pid),
                "fg3m": _nan_or(_to_float(stats.get("threePointersMade")), 0.0),
                "fg3a": _nan_or(_to_float(stats.get("threePointersAttempted")), 0.0),
                "fga": _nan_or(_to_float(stats.get("fieldGoalsAttempted")), 0.0),
                "fgm": _nan_or(_to_float(stats.get("fieldGoalsMade")), 0.0),
                "fta": _nan_or(_to_float(stats.get("freeThrowsAttempted")), 0.0),
                "ftm": _nan_or(_to_float(stats.get("freeThrowsMade")), 0.0),
                "tov": _nan_or(_to_float(stats.get("turnovers") or stats.get("turnoversTotal")), 0.0),
                "steals": _nan_or(_to_float(stats.get("steals")), 0.0),
                "blocks": _nan_or(_to_float(stats.get("blocks")), 0.0),
                "orb": _nan_or(_to_float(stats.get("reboundsOffensive")), 0.0),
                "drb": _nan_or(_to_float(stats.get("reboundsDefensive")), 0.0),
                # New stats: fouls, scoring breakdown
                "fouls_personal": _nan_or(_to_float(stats.get("foulsPersonal")), 0.0),
                "fouls_drawn": _nan_or(_to_float(stats.get("foulsDrawn")), 0.0),
                "pts_in_paint": _nan_or(_to_float(stats.get("pointsInThePaint")), 0.0),
                "pts_fast_break": _nan_or(_to_float(stats.get("pointsFastBreak")), 0.0),
                # OT detection
                "n_ot_periods": n_ot_periods,
            })
    return rows


def load_extended_player_stats() -> pd.DataFrame:
    """Load extended player stats from all cached boxscores."""
    global _EXTENDED_STATS_CACHE
    if _EXTENDED_STATS_CACHE is not None:
        return _EXTENDED_STATS_CACHE.copy()

    all_rows: list[dict[str, Any]] = []

    # Current season cache
    if BOXSCORE_CACHE.exists():
        for f in BOXSCORE_CACHE.glob("*.json"):
            try:
                payload = json.loads(f.read_text())
                game = payload["game"]
                if int(game.get("gameStatus", 0)) != 3:
                    continue
                all_rows.extend(_parse_extended_player_stats(game))
            except Exception:
                continue

    # Historical season caches
    for season in SEASONS[:-1]:
        box_dir = HIST_CACHE_DIR / season / "boxscores"
        if not box_dir.exists():
            continue
        for f in box_dir.glob("*.json"):
            try:
                payload = json.loads(f.read_text())
                game = payload["game"]
                if int(game.get("gameStatus", 0)) != 3:
                    continue
                all_rows.extend(_parse_extended_player_stats(game))
            except Exception:
                continue

    if not all_rows:
        _EXTENDED_STATS_CACHE = pd.DataFrame()
        return _EXTENDED_STATS_CACHE.copy()

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["game_id", "team", "player_id"])
    _EXTENDED_STATS_CACHE = df
    return df.copy()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rolling_beta_shifted(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 20,
    min_periods: int = 6,
) -> np.ndarray:
    """Shifted rolling beta(y ~ x): each row uses prior observations only."""
    out = np.full(len(x), np.nan, dtype=float)
    if len(x) == 0:
        return out
    for i in range(len(x)):
        s = max(0, i - window)
        xs = x[s:i]
        ys = y[s:i]
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() < min_periods:
            continue
        xv = xs[mask]
        yv = ys[mask]
        var_x = float(np.var(xv))
        if var_x < 1e-8:
            continue
        cov_xy = float(np.mean((xv - xv.mean()) * (yv - yv.mean())))
        out[i] = cov_xy / var_x
    return out


def _safe_logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def load_game_odds_lookup(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Build a game_id → (market_total_close, market_home_spread_close) lookup.

    Combines historical cached odds + current-season ESPN odds so that
    Vegas features can be backfilled into training data.
    """
    all_odds: list[pd.DataFrame] = []

    # Historical seasons: load from cached JSON
    for season in SEASONS[:-1]:
        odds = load_historical_espn_odds(season)
        if odds.empty:
            continue
        sched = schedule_df[schedule_df["season"] == season]
        if sched.empty:
            continue
        merged = sched[["game_id", "game_date_est", "home_team", "away_team"]].merge(
            odds[["game_date_est", "home_team", "away_team",
                  "market_total_close", "market_home_spread_close"]],
            on=["game_date_est", "home_team", "away_team"],
            how="inner",
        )
        all_odds.append(merged[["game_id", "market_total_close", "market_home_spread_close"]])

    # Current season: fetch via ESPN (cached)
    current = schedule_df[schedule_df["season"] == SEASON].copy()
    if not current.empty:
        try:
            current_with_odds = join_espn_odds(current)
            cols_needed = ["game_id", "market_total_close", "market_home_spread_close"]
            if all(c in current_with_odds.columns for c in cols_needed):
                all_odds.append(current_with_odds[cols_needed].dropna())
        except Exception as exc:
            print(f"  Warning: Could not load current-season odds: {exc}", flush=True)

    if not all_odds:
        return pd.DataFrame(columns=["game_id", "market_total_close", "market_home_spread_close"])
    result = pd.concat(all_odds, ignore_index=True).drop_duplicates(subset="game_id")
    print(f"  Loaded Vegas odds for {len(result)} games", flush=True)
    return result


def _player_feature_cache_key(
    player_games: pd.DataFrame,
    team_games: pd.DataFrame,
    game_odds: pd.DataFrame,
    min_games: int,
    ref_features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build a cache key for the engineered player-feature table."""
    last_game_time = pd.to_datetime(player_games.get("game_time_utc"), utc=True, errors="coerce")
    last_ts = ""
    if isinstance(last_game_time, pd.Series) and not last_game_time.dropna().empty:
        last_ts = str(last_game_time.max())
    return {
        "version": PLAYER_FEATURE_CACHE_VERSION,
        "min_games": int(min_games),
        "n_player_rows": int(len(player_games)),
        "n_team_rows": int(len(team_games)),
        "n_odds_rows": int(len(game_odds)),
        "n_ref_rows": int(len(ref_features)) if ref_features is not None else 0,
        "last_game_time_utc": last_ts,
    }


def load_or_build_player_features(
    player_games: pd.DataFrame,
    team_games: pd.DataFrame,
    game_odds: pd.DataFrame,
    min_games: int = DEFAULT_MIN_GAMES,
    ref_features: pd.DataFrame | None = None,
    box_adv_fetch_missing: bool = False,
    box_adv_max_fetch: int = 0,
) -> pd.DataFrame:
    """Load player features from cache when possible; otherwise rebuild and cache."""
    cache_key = _player_feature_cache_key(
        player_games,
        team_games,
        game_odds,
        min_games,
        ref_features=ref_features,
    )
    if PLAYER_FEATURE_CACHE_FILE.exists() and PLAYER_FEATURE_CACHE_META.exists():
        try:
            meta = json.loads(PLAYER_FEATURE_CACHE_META.read_text())
            if meta == cache_key:
                cached = pd.read_pickle(PLAYER_FEATURE_CACHE_FILE)
                if isinstance(cached, pd.DataFrame) and not cached.empty:
                    print(f"  Loaded cached player features: {len(cached)} rows", flush=True)
                    return cached
        except Exception:
            pass

    built = build_player_features(
        player_games,
        team_games,
        min_games=min_games,
        game_odds=game_odds,
        ref_features=ref_features,
        box_adv_fetch_missing=box_adv_fetch_missing,
        box_adv_max_fetch=box_adv_max_fetch,
    )
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        built.to_pickle(PLAYER_FEATURE_CACHE_FILE)
        PLAYER_FEATURE_CACHE_META.write_text(json.dumps(cache_key, indent=2, default=str))
    except Exception as exc:
        print(f"  Warning: could not write player feature cache: {exc}", flush=True)
    return built


# ---------------------------------------------------------------------------
# Enhanced player-level feature engineering
# ---------------------------------------------------------------------------

def build_player_features(player_games: pd.DataFrame, team_games: pd.DataFrame,
                          min_games: int = DEFAULT_MIN_GAMES,
                          game_odds: pd.DataFrame | None = None,
                          ref_features: pd.DataFrame | None = None,
                          box_adv_fetch_missing: bool = False,
                          box_adv_max_fetch: int = 0) -> pd.DataFrame:
    """Build player-level features for props modeling.

    Enhanced with: EWM averages, venue splits, matchup features, usage dynamics,
    per-minute rates, blowout risk context, referee crew data, OT regulation
    adjustment, and BoxScoreAdvancedV3 usage/pace metrics.
    """
    pg = player_games.copy()
    pg = pg.sort_values(["team", "player_id", "game_time_utc", "game_id"]).reset_index(drop=True)

    # Only keep rows where the player actually played
    pg = pg[pg["played"] == 1].copy()

    # --- Load and merge extended stats ---
    print("  Loading extended player stats (fg3m, fta, tov, fouls, paint/fastbreak, OT)...", flush=True)
    ext = load_extended_player_stats()
    _ext_cols = ["fg3m", "fg3a", "fga", "fgm", "fta", "ftm", "tov", "steals", "blocks",
                 "orb", "drb", "fouls_personal", "fouls_drawn", "pts_in_paint",
                 "pts_fast_break", "n_ot_periods"]
    if not ext.empty:
        pg = pg.merge(ext, on=["game_id", "team", "player_id"], how="left")
        for c in _ext_cols:
            if c not in pg.columns:
                pg[c] = np.nan
    else:
        for c in _ext_cols:
            pg[c] = np.nan

    # --- Load and merge BoxScoreAdvancedV3 stats (usage / pace / advanced rates) ---
    adv_cols = [
        "adv_usage_pct",
        "adv_pace",
        "adv_possessions",
        "adv_off_rating",
        "adv_ast_pct",
        "adv_reb_pct",
        "adv_ts_pct",
    ]
    adv = load_boxscore_advanced_stats(
        game_ids=pg["game_id"].astype(str).unique().tolist(),
        fetch_missing=box_adv_fetch_missing,
        max_fetch=box_adv_max_fetch,
    )
    if not adv.empty:
        pg = pg.merge(
            adv[["game_id", "team", "player_id"] + [c for c in adv_cols if c in adv.columns]],
            on=["game_id", "team", "player_id"],
            how="left",
        )
    for c in adv_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    # --- OT regulation adjustment ---
    # Scale counting stats to regulation-equivalent for rolling averages.
    # OT inflates counting stats by ~10% per OT period; per-minute rates are unaffected.
    pg["n_ot_periods"] = pg["n_ot_periods"].fillna(0).astype(int)
    pg["is_ot"] = (pg["n_ot_periods"] > 0).astype(float)
    reg_factor = 48.0 / (48.0 + 5.0 * pg["n_ot_periods"])
    # Create regulation-adjusted counting stats for rolling averages
    _counting_stats = ["points", "rebounds", "assists", "fg3m", "fga", "fgm",
                       "fg3a", "fta", "ftm", "tov", "steals", "blocks", "orb", "drb",
                       "fouls_personal", "fouls_drawn", "pts_in_paint", "pts_fast_break"]
    for col in _counting_stats:
        if col in pg.columns:
            pg[f"{col}_reg"] = pg[col] * reg_factor

    # --- Per-minute rates (for two-stage modeling) ---
    safe_mins = pg["minutes"].clip(lower=1)
    pg["pts_per_min"] = pg["points"] / safe_mins
    pg["reb_per_min"] = pg["rebounds"] / safe_mins
    pg["ast_per_min"] = pg["assists"] / safe_mins
    pg["fg3m_per_min"] = pg["fg3m"] / safe_mins
    pg["fga_per_min"] = pg["fga"] / safe_mins
    pg["fg3a_per_min"] = pg["fg3a"] / safe_mins
    pg["fta_per_min"] = pg["fta"] / safe_mins
    pg["fouls_drawn_per_min"] = pg["fouls_drawn"] / safe_mins

    # --- Player rolling features ---
    player_group = ["team", "player_id", "season"] if "season" in pg.columns else ["team", "player_id"]

    # Drop any pre-existing rolling features from the monolith (we recompute with OT adjustment).
    _old_rolling = [c for c in pg.columns if c.startswith("pre_") and any(
        w in c for w in ("_avg5", "_avg10", "_avg3", "_ewm5", "_ewm10", "_std10", "_season")
    )]
    if _old_rolling:
        pg.drop(columns=_old_rolling, inplace=True, errors="ignore")

    # Use regulation-adjusted counting stats for rolling averages to avoid OT inflation.
    # Per-minute rates are already OT-independent and use raw values.
    rolling_cols = ["points_reg", "rebounds_reg", "assists_reg", "minutes",
                    "fg3m_reg", "fta_reg", "tov_reg",
                    "fga_reg", "fgm_reg", "fg3a_reg", "steals_reg", "blocks_reg",
                    "orb_reg", "drb_reg",
                    "fouls_personal_reg", "fouls_drawn_reg",
                    "pts_in_paint_reg", "pts_fast_break_reg",
                    "adv_usage_pct", "adv_pace", "adv_possessions", "adv_off_rating",
                    "adv_ast_pct", "adv_reb_pct", "adv_ts_pct",
                    "pts_per_min", "reb_per_min", "ast_per_min", "fg3m_per_min",
                    "fga_per_min", "fg3a_per_min", "fta_per_min", "fouls_drawn_per_min"]
    # Last-3-game hot streak features (faster than avg5 for capturing streaks)
    avg3_cols = ["points_reg", "rebounds_reg", "assists_reg", "minutes", "fg3m_reg",
                 "fg3a_reg", "fouls_drawn_reg", "adv_usage_pct",
                 "pts_per_min", "reb_per_min", "ast_per_min", "fg3m_per_min"]
    for col in rolling_cols:
        if col not in pg.columns:
            continue
        grp = pg.groupby(player_group)[col]
        if col in avg3_cols:
            pg[f"pre_{col}_avg3"] = grp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        pg[f"pre_{col}_avg5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        pg[f"pre_{col}_avg10"] = grp.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
        pg[f"pre_{col}_season"] = grp.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())

    # --- EWM averages (span=5 and span=10) ---
    # Use regulation-adjusted counting stats for EWM too
    ewm_cols = ["points_reg", "rebounds_reg", "assists_reg", "minutes",
                "fg3m_reg", "fta_reg", "tov_reg", "orb_reg", "drb_reg",
                "fouls_drawn_reg",
                "adv_usage_pct", "adv_pace", "adv_possessions", "adv_off_rating",
                "adv_ast_pct", "adv_reb_pct", "adv_ts_pct",
                "pts_per_min", "reb_per_min", "ast_per_min", "fg3m_per_min",
                "fga_per_min", "fg3a_per_min", "fta_per_min", "fouls_drawn_per_min"]
    for col in ewm_cols:
        if col not in pg.columns:
            continue
        grp = pg.groupby(player_group)[col]
        pg[f"pre_{col}_ewm5"] = grp.transform(lambda s: s.shift(1).ewm(span=5, min_periods=1).mean())
        pg[f"pre_{col}_ewm10"] = grp.transform(lambda s: s.shift(1).ewm(span=10, min_periods=1).mean())

    # --- Variance / consistency features ---
    for col in ["points_reg", "rebounds_reg", "assists_reg", "minutes"]:
        if col not in pg.columns:
            continue
        grp = pg.groupby(player_group)[col]
        pg[f"pre_{col}_std10"] = grp.transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())

    # Defragment before rename to avoid duplicate-column issues on fragmented frames.
    pg = pg.copy()

    # --- Rename regulation-adjusted rolling features to standard names ---
    # Convert pre_points_reg_avg5 → pre_points_avg5, etc. so downstream code stays compatible.
    rename_map = {}
    for c in pg.columns:
        if "_reg_avg" in c or "_reg_ewm" in c or "_reg_std" in c or "_reg_season" in c:
            new_name = c.replace("_reg_", "_")
            rename_map[c] = new_name
    if rename_map:
        pg = pg.rename(columns=rename_map)
    # Drop the intermediate _reg columns (raw counting stat × reg_factor)
    reg_drop = [c for c in pg.columns if c.endswith("_reg") and c.startswith(("points_", "rebounds_", "assists_", "fg3m_", "fga_", "fgm_", "fg3a_", "fta_", "ftm_", "tov_", "steals_", "blocks_", "orb_", "drb_", "fouls_personal_", "fouls_drawn_", "pts_in_paint_", "pts_fast_break_"))]
    if reg_drop:
        pg.drop(columns=reg_drop, inplace=True, errors="ignore")

    # Starter rate
    grp_starter = pg.groupby(player_group)["starter"]
    pg["pre_starter_rate"] = grp_starter.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    # Defragment after heavy rolling-window feature construction.
    pg = pg.copy()

    # Minutes trend: avg5 - avg10 (positive = increasing role)
    if "pre_minutes_avg5" in pg.columns and "pre_minutes_avg10" in pg.columns:
        pg["pre_minutes_trend"] = pg["pre_minutes_avg5"] - pg["pre_minutes_avg10"]

    # Points trend
    if "pre_points_avg5" in pg.columns and "pre_points_avg10" in pg.columns:
        pg["pre_points_trend"] = pg["pre_points_avg5"] - pg["pre_points_avg10"]

    # Usage proxy: (points + 0.44*fta + tov) / minutes
    pre_pts = pg["pre_points_avg5"].fillna(0)
    pre_fta = pg["pre_fta_avg5"].fillna(0) if "pre_fta_avg5" in pg.columns else 0
    pre_tov = pg["pre_tov_avg5"].fillna(0) if "pre_tov_avg5" in pg.columns else 0
    pre_min = pg["pre_minutes_avg5"].fillna(1).clip(lower=1)
    pg["pre_usage_proxy"] = (pre_pts + 0.44 * pre_fta + pre_tov) / pre_min

    # --- Venue splits (home vs away performance) ---
    # Build columns in batches to avoid DataFrame fragmentation from repeated inserts.
    venue_cols: dict[str, pd.Series] = {}
    home_mask = pg["is_home"] == 1
    away_mask = pg["is_home"] == 0
    group_keys = [pg[c] for c in player_group]
    for col in ["points", "rebounds", "assists", "minutes", "fg3m"]:
        if col not in pg.columns:
            continue
        home_series = pd.Series(np.nan, index=pg.index, dtype=float)
        away_series = pd.Series(np.nan, index=pg.index, dtype=float)

        if home_mask.any():
            home_avg = pg.loc[home_mask].groupby(player_group)[col].transform(
                lambda s: s.shift(1).expanding(min_periods=3).mean()
            )
            home_series.loc[home_mask] = home_avg.to_numpy(dtype=float)
        if away_mask.any():
            away_avg = pg.loc[away_mask].groupby(player_group)[col].transform(
                lambda s: s.shift(1).expanding(min_periods=3).mean()
            )
            away_series.loc[away_mask] = away_avg.to_numpy(dtype=float)

        home_series = home_series.groupby(group_keys).transform(lambda s: s.ffill())
        away_series = away_series.groupby(group_keys).transform(lambda s: s.ffill())
        venue_cols[f"pre_{col}_home_avg"] = home_series
        venue_cols[f"pre_{col}_away_avg"] = away_series
        venue_cols[f"pre_{col}_venue_diff"] = home_series.fillna(0.0) - away_series.fillna(0.0)

    if venue_cols:
        pg = pd.concat([pg, pd.DataFrame(venue_cols, index=pg.index)], axis=1)

    # Defragment before adding remaining scalar columns.
    pg = pg.copy()
    # Player game count (for filtering)
    pg["player_game_num"] = pg.groupby(player_group).cumcount() + 1

    # --- Player rest and home/away ---
    pg["player_is_home"] = pg["is_home"]

    # Player days rest
    prev_time = pg.groupby(player_group)["game_time_utc"].shift(1)
    pg["player_days_rest"] = (pg["game_time_utc"] - prev_time).dt.total_seconds() / 86400.0

    # --- B2B and schedule fatigue ---
    pg["is_b2b"] = (pg["player_days_rest"] <= 1.5).astype(float)
    # 3-in-4-nights: check if 3rd game in 4 calendar days
    prev2_time = pg.groupby(player_group)["game_time_utc"].shift(2)
    days_for_3 = (pg["game_time_utc"] - prev2_time).dt.total_seconds() / 86400.0
    pg["is_3_in_4"] = (days_for_3 <= 4.0).astype(float)
    pg["is_3_in_4"] = pg["is_3_in_4"].fillna(0.0)
    # Fatigue interactions
    pg["b2b_x_minutes"] = pg["is_b2b"] * pg.get("pre_minutes_avg5", pd.Series(0.0, index=pg.index)).fillna(0)
    pg["b2b_x_starter"] = pg["is_b2b"] * pg.get("pre_starter_rate", pd.Series(0.0, index=pg.index)).fillna(0)

    # --- Team context features ---
    tg = team_games.copy()
    team_cols_to_merge = ["game_id", "team"]
    team_feature_cols = []

    for col in ["possessions", "off_rating", "def_rating", "net_rating", "efg"]:
        for window in ["avg5", "avg10", "season"]:
            feat = f"pre_{col}_{window}"
            if feat in tg.columns:
                team_feature_cols.append(feat)

    if team_feature_cols:
        tg_subset = tg[team_cols_to_merge + team_feature_cols].copy()
        tg_subset = tg_subset.rename(columns={c: f"team_{c}" for c in team_feature_cols})
        pg = pg.merge(tg_subset, on=["game_id", "team"], how="left")

    # --- Opponent context features ---
    if "opp" in pg.columns:
        opp_tg = tg[team_cols_to_merge + team_feature_cols].copy()
        opp_tg = opp_tg.rename(columns={"team": "opp", **{c: f"opp_{c}" for c in team_feature_cols}})
        pg = pg.merge(opp_tg, on=["game_id", "opp"], how="left")

    # --- Matchup features: opponent's defensive strength context ---
    # Approximate: how many points does the opponent allow per game (already captured
    # in opp_pre_def_rating). Add pace interaction and differential.
    if "team_pre_possessions_avg5" in pg.columns and "opp_pre_possessions_avg5" in pg.columns:
        pg["matchup_pace_avg"] = (
            pg["team_pre_possessions_avg5"].fillna(0) + pg["opp_pre_possessions_avg5"].fillna(0)
        ) / 2.0
    else:
        pg["matchup_pace_avg"] = 0.0

    if "team_pre_off_rating_avg5" in pg.columns and "opp_pre_def_rating_avg5" in pg.columns:
        pg["matchup_off_vs_def"] = (
            pg["team_pre_off_rating_avg5"].fillna(0) - pg["opp_pre_def_rating_avg5"].fillna(0)
        )
    else:
        pg["matchup_off_vs_def"] = 0.0

    # --- Opponent pace differential vs player's recent games ---
    # Positive = this game expected to be faster than player's recent norm
    if "team_pre_possessions_avg5" in pg.columns:
        # Player's recent average team pace (proxy for game pace they've played in)
        player_recent_pace = pg.groupby(player_group)["team_pre_possessions_avg5"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=1).mean()
        )
        pg["pace_diff_vs_recent"] = pg["matchup_pace_avg"] - player_recent_pace.fillna(pg["matchup_pace_avg"])
    else:
        pg["pace_diff_vs_recent"] = 0.0

    # --- Teammate injury context ---
    inj_cols = ["game_id", "team"]
    for ic in ["injury_proxy_missing_minutes5", "injury_proxy_missing_points5",
               "star_player_absent_flag", "active_count"]:
        if ic in tg.columns:
            inj_cols.append(ic)
    if len(inj_cols) > 2:
        inj_sub = tg[inj_cols].copy()
        inj_sub = inj_sub.rename(columns={c: f"team_{c}" for c in inj_cols if c not in ["game_id", "team"]})
        pg = pg.merge(inj_sub, on=["game_id", "team"], how="left")

    # --- Usage boost from missing teammates ---
    # If star players are absent, remaining players get more usage
    if "team_injury_proxy_missing_points5" in pg.columns:
        pg["team_injury_pressure"] = pg["team_injury_proxy_missing_points5"].fillna(0)
        pg["usage_boost_proxy"] = pg["team_injury_pressure"] * pg["pre_usage_proxy"].fillna(0)
        pg["minutes_x_injury_pressure"] = pg["pre_minutes_avg5"].fillna(0) * pg["team_injury_pressure"]
    else:
        pg["team_injury_pressure"] = 0.0
        pg["minutes_x_injury_pressure"] = 0.0
        pg["usage_boost_proxy"] = pg["pre_usage_proxy"].fillna(0)

    # --- Blowout risk proxy ---
    # If team's net rating advantage is large, starters may play fewer minutes
    if "team_pre_net_rating_avg5" in pg.columns and "opp_pre_net_rating_avg5" in pg.columns:
        pg["net_rating_diff"] = (
            pg["team_pre_net_rating_avg5"].fillna(0) - pg["opp_pre_net_rating_avg5"].fillna(0)
        )
        pg["blowout_risk"] = pg["net_rating_diff"].abs()
    else:
        pg["net_rating_diff"] = 0.0
        pg["blowout_risk"] = 0.0

    pg["pace_x_injury_pressure"] = pg["matchup_pace_avg"].fillna(0) * pg["team_injury_pressure"].fillna(0)

    # --- Role shift / on-off style response to teammate absences ---
    # Beta > 0 means player output tends to rise when teammate injury pressure rises.
    role_specs = [
        ("points", "role_pts_injury_beta20"),
        ("rebounds", "role_reb_injury_beta20"),
        ("assists", "role_ast_injury_beta20"),
        ("minutes", "role_min_injury_beta20"),
    ]
    for _, out_col in role_specs:
        pg[out_col] = np.nan

    for _, idx in pg.groupby(player_group).groups.items():
        sub = pg.loc[idx]
        x = sub["team_injury_pressure"].to_numpy(dtype=float)
        for y_col, out_col in role_specs:
            if y_col not in sub.columns:
                continue
            y = sub[y_col].to_numpy(dtype=float)
            pg.loc[idx, out_col] = _rolling_beta_shifted(x, y, window=20, min_periods=6)

    # --- Rebound-opportunity and center-depth features ---
    if {"game_id", "game_time_utc", "team", "opp", "rebounds", "orb", "drb", "fga", "fgm"}.issubset(pg.columns):
        reb_profile = (
            pg.groupby(["game_id", "game_time_utc", "team", "opp"], as_index=False)
            .agg(
                team_rebounds=("rebounds", "sum"),
                team_orb=("orb", "sum"),
                team_drb=("drb", "sum"),
                team_fga=("fga", "sum"),
                team_fgm=("fgm", "sum"),
            )
        )
        reb_profile["team_missed_fg"] = (reb_profile["team_fga"] - reb_profile["team_fgm"]).clip(lower=0.0)
        opp_map = reb_profile[
            ["game_id", "team", "team_rebounds", "team_orb", "team_drb", "team_missed_fg"]
        ].rename(
            columns={
                "team": "opp",
                "team_rebounds": "opp_rebounds",
                "team_orb": "opp_orb",
                "team_drb": "opp_drb",
                "team_missed_fg": "opp_missed_fg",
            }
        )
        reb_profile = reb_profile.merge(opp_map, on=["game_id", "opp"], how="left")
        reb_profile["total_rebounds"] = reb_profile["team_rebounds"] + reb_profile["opp_rebounds"]
        reb_profile["total_missed_fg"] = reb_profile["team_missed_fg"] + reb_profile["opp_missed_fg"]
        reb_profile["team_reb_share"] = np.where(
            reb_profile["total_rebounds"] > 0,
            reb_profile["team_rebounds"] / reb_profile["total_rebounds"],
            np.nan,
        )
        reb_profile["team_orb_share"] = np.where(
            (reb_profile["team_orb"] + reb_profile["opp_drb"]) > 0,
            reb_profile["team_orb"] / (reb_profile["team_orb"] + reb_profile["opp_drb"]),
            np.nan,
        )
        reb_profile["team_drb_share"] = np.where(
            (reb_profile["team_drb"] + reb_profile["opp_orb"]) > 0,
            reb_profile["team_drb"] / (reb_profile["team_drb"] + reb_profile["opp_orb"]),
            np.nan,
        )

        reb_profile = reb_profile.sort_values("game_time_utc")
        reb_roll_cols = [
            "team_rebounds",
            "team_orb",
            "team_drb",
            "team_missed_fg",
            "opp_missed_fg",
            "total_missed_fg",
            "team_reb_share",
            "team_orb_share",
            "team_drb_share",
        ]
        for col in reb_roll_cols:
            reb_profile[f"pre_{col}_avg10"] = reb_profile.groupby("team")[col].transform(
                lambda s: s.shift(1).rolling(10, min_periods=3).mean()
            )

        reb_merge_cols = ["game_id", "team"] + [f"pre_{c}_avg10" for c in reb_roll_cols]
        pg = pg.merge(reb_profile[reb_merge_cols], on=["game_id", "team"], how="left")

        team_reb_avg10 = pg.get("pre_team_rebounds_avg10", pd.Series(np.nan, index=pg.index)).clip(lower=1.0)
        team_orb_avg10 = pg.get("pre_team_orb_avg10", pd.Series(np.nan, index=pg.index)).clip(lower=1.0)
        team_drb_avg10 = pg.get("pre_team_drb_avg10", pd.Series(np.nan, index=pg.index)).clip(lower=1.0)
        pg["pre_player_reb_share_avg10"] = pg.get("pre_rebounds_avg10", pd.Series(np.nan, index=pg.index)) / team_reb_avg10
        pg["pre_player_orb_share_avg10"] = pg.get("pre_orb_avg10", pd.Series(np.nan, index=pg.index)) / team_orb_avg10
        pg["pre_player_drb_share_avg10"] = pg.get("pre_drb_avg10", pd.Series(np.nan, index=pg.index)) / team_drb_avg10
        pg["pre_player_reb_opp_proxy_avg10"] = (
            pg["pre_player_reb_share_avg10"]
            * pg.get("pre_total_missed_fg_avg10", pd.Series(np.nan, index=pg.index))
        )
    else:
        pg["pre_team_rebounds_avg10"] = np.nan
        pg["pre_team_orb_avg10"] = np.nan
        pg["pre_team_drb_avg10"] = np.nan
        pg["pre_team_missed_fg_avg10"] = np.nan
        pg["pre_opp_missed_fg_avg10"] = np.nan
        pg["pre_total_missed_fg_avg10"] = np.nan
        pg["pre_team_reb_share_avg10"] = np.nan
        pg["pre_team_orb_share_avg10"] = np.nan
        pg["pre_team_drb_share_avg10"] = np.nan
        pg["pre_player_reb_share_avg10"] = np.nan
        pg["pre_player_orb_share_avg10"] = np.nan
        pg["pre_player_drb_share_avg10"] = np.nan
        pg["pre_player_reb_opp_proxy_avg10"] = np.nan

    if "position" in pg.columns and "minutes" in pg.columns:
        center_mask = pg["position"].astype(str).str.upper().eq("C")
        team_center_minutes = (
            pg.assign(_center_minutes=np.where(center_mask, pg["minutes"].fillna(0.0), 0.0))
            .groupby(["game_id", "team"])["_center_minutes"]
            .sum()
            .rename("team_center_minutes")
            .reset_index()
        )
        pg = pg.merge(team_center_minutes, on=["game_id", "team"], how="left")
        pg["other_center_minutes"] = pg["team_center_minutes"].fillna(0.0) - np.where(
            center_mask,
            pg["minutes"].fillna(0.0),
            0.0,
        )
        pg["other_center_minutes"] = pg["other_center_minutes"].clip(lower=0.0)
        pg["pre_other_center_minutes_avg10"] = pg.groupby(player_group)["other_center_minutes"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean()
        )
        pg["pre_center_depth_risk"] = (pg["pre_other_center_minutes_avg10"] < 18.0).astype(float)
        pg.drop(columns=["team_center_minutes", "other_center_minutes"], inplace=True, errors="ignore")
    else:
        pg["pre_other_center_minutes_avg10"] = np.nan
        pg["pre_center_depth_risk"] = np.nan

    # --- Opponent positional defense features ---
    # Compute how many points/rebounds/assists each team allows to each position group,
    # then merge rolling averages back onto player rows.
    if "position" in pg.columns:
        pos_map = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}
        pg["pos_group"] = pg["position"].map(pos_map).fillna("F")

        # Build per-game defense profile: stats scored BY each position group AGAINST each team
        defense_stats = pg.groupby(["game_id", "game_time_utc", "opp", "pos_group"]).agg(
            pts_allowed=("points", "sum"),
            reb_allowed=("rebounds", "sum"),
            ast_allowed=("assists", "sum"),
        ).reset_index()

        # Compute rolling averages per defending team and position group
        defense_stats = defense_stats.sort_values("game_time_utc")
        for stat in ["pts", "reb", "ast"]:
            col = f"{stat}_allowed"
            grp = defense_stats.groupby(["opp", "pos_group"])[col]
            defense_stats[f"opp_{stat}_allowed_to_pos_avg10"] = grp.transform(
                lambda s: s.shift(1).rolling(10, min_periods=3).mean()
            )

        # Merge back: player's opp + pos_group + game_id -> opponent defensive profile
        defense_merge = defense_stats[[
            "game_id", "opp", "pos_group",
            "opp_pts_allowed_to_pos_avg10",
            "opp_reb_allowed_to_pos_avg10",
            "opp_ast_allowed_to_pos_avg10",
        ]].copy()
        pg = pg.merge(defense_merge, on=["game_id", "opp", "pos_group"], how="left")
    else:
        pg["pos_group"] = "F"
        pg["opp_pts_allowed_to_pos_avg10"] = np.nan
        pg["opp_reb_allowed_to_pos_avg10"] = np.nan
        pg["opp_ast_allowed_to_pos_avg10"] = np.nan

    # --- Live-only fields (filled at prediction time) ---
    # Keep these columns in training so feature schemas stay stable.
    pg["injury_availability_prob"] = np.nan
    pg["injury_unavailability_prob"] = np.nan
    pg["injury_is_out"] = 0
    pg["injury_is_doubtful"] = 0
    pg["injury_is_questionable"] = 0
    pg["injury_is_probable"] = 0
    pg["lineup_confirmed"] = 0
    pg["confirmed_starter"] = np.nan
    pg["pred_starter_prob"] = np.nan

    # --- Vegas game total / spread (backfilled from historical + current odds) ---
    if game_odds is not None and not game_odds.empty:
        odds_sub = game_odds[["game_id", "market_total_close", "market_home_spread_close"]].drop_duplicates(subset="game_id")
        pg = pg.merge(odds_sub, on="game_id", how="left")
        pg["implied_total"] = pg["market_total_close"]
        # Convert home spread to team-perspective spread
        pg["implied_spread"] = np.where(
            pg["is_home"] == 1,
            pg["market_home_spread_close"],
            -pg["market_home_spread_close"],
        )
        total = pg["implied_total"]
        spread = pg["implied_spread"]
        pg["implied_team_total"] = np.where(
            pg["is_home"] == 1,
            total / 2 - spread / 2,
            total / 2 + spread / 2,
        )
        pg["abs_spread"] = pg["implied_spread"].abs()
        starter_rate = pg.get("pre_starter_rate", pd.Series(0.0, index=pg.index)).fillna(0)
        pg["spread_x_starter"] = pg["abs_spread"] * starter_rate
        pg["is_big_favorite"] = (pg["abs_spread"] > 8).astype(float)
        pg.drop(columns=["market_total_close", "market_home_spread_close"], inplace=True, errors="ignore")
    else:
        pg["implied_total"] = np.nan
        pg["implied_spread"] = np.nan
        pg["implied_team_total"] = np.nan
        pg["abs_spread"] = np.nan
        pg["spread_x_starter"] = np.nan
        pg["is_big_favorite"] = np.nan

    # --- Player career / age-rest interactions (computed from training data) ---
    # Cumulative game count across all seasons for this player
    pg["player_career_games"] = pg.groupby("player_id").cumcount() + 1
    pg["is_veteran"] = (pg["player_career_games"] > 200).astype(float)
    rest = pg["player_days_rest"].fillna(2.0)
    pg["veteran_rest_effect"] = pg["is_veteran"] * rest
    pg["career_x_rest"] = pg["player_career_games"] * rest

    # --- Referee crew features (merged from game-level monolith data) ---
    _ref_cols = ["ref_crew_avg_total", "ref_crew_avg_fta", "ref_crew_avg_fouls",
                 "ref_crew_avg_pace", "ref_crew_total_over_league_avg",
                 "ref_crew_pace_over_league_avg"]
    if ref_features is not None and not ref_features.empty:
        ref_merge = ref_features[["game_id"] + [c for c in _ref_cols if c in ref_features.columns]].drop_duplicates(subset="game_id")
        pg = pg.merge(ref_merge, on="game_id", how="left")
    for c in _ref_cols:
        if c not in pg.columns:
            pg[c] = np.nan
    # Legacy placeholders kept for schema stability (always NaN at training time)
    pg["ref_foul_rate"] = np.nan
    pg["ref_pace_factor"] = np.nan

    # --- Filter to players with enough history ---
    pg = pg[pg["player_game_num"] >= min_games].copy()

    return pg


# ---------------------------------------------------------------------------
# Market Line Features (Phase 3)
# ---------------------------------------------------------------------------


def add_market_line_features(
    player_df: pd.DataFrame,
    prop_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Add prop market line features to a player-game DataFrame.

    For each row (player/date/stat), looks up cached prop_lines_YYYYMMDD.csv
    files to find the opening line. Uses only open_line (safe timestamp).

    Features added:
      - prop_open_line: opening line for this player/stat
      - prop_line_vs_avg5: open_line - pre_{stat}_avg5
      - prop_line_vs_avg10: open_line - pre_{stat}_avg10
      - implied_over_prob: market-implied P(over) from opening odds
      - line_available: 1 if a line was found, 0 otherwise
    """
    df = player_df.copy()
    cache_dir = prop_cache_dir or PROP_CACHE_DIR

    # Initialize columns
    for col in ["prop_open_line", "prop_line_vs_avg5", "prop_line_vs_avg10",
                "implied_over_prob", "line_available"]:
        df[col] = np.nan
    df["line_available"] = 0.0

    if "game_date_est" not in df.columns:
        return df

    # Build lookup from cached prop line files
    line_lookup: dict[str, dict[str, float]] = {}  # "{date}_{name_norm}_{stat}" -> {open_line, over_implied, ...}

    # Check both prop_cache and predictions directories for prop line files
    search_dirs = [cache_dir, PREDICTIONS_DIR, PROP_LINES_DIR]
    seen_files: set[str] = set()

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for f in search_dir.glob("prop_lines_*.csv"):
            if f.name in seen_files:
                continue
            seen_files.add(f.name)
            # Extract date from filename
            date_match = re.search(r"prop_lines_(\d{8})", f.name)
            if not date_match:
                continue
            file_date = date_match.group(1)
            try:
                lines_df = pd.read_csv(f)
            except Exception:
                continue
            if lines_df.empty or "player_name" not in lines_df.columns:
                continue
            for _, lr in lines_df.iterrows():
                name_norm = normalize_player_name(lr.get("player_name", ""))
                stat = str(lr.get("stat_type", ""))
                if not name_norm or not stat:
                    continue
                key = f"{file_date}_{name_norm}_{stat}"
                open_line = lr.get("open_line", lr.get("line", np.nan))
                over_implied = lr.get("over_implied_prob", np.nan)
                if pd.notna(open_line):
                    line_lookup[key] = {
                        "open_line": float(open_line),
                        "over_implied": float(over_implied) if pd.notna(over_implied) else np.nan,
                    }

    if not line_lookup:
        return df

    # For each stat type that has market lines, add features
    stat_types = ["points", "rebounds", "assists", "fg3m", "minutes"]

    # Process row by row, matching player + date + stat_type
    # We need to know which stat this row's features are for — but in training data,
    # each row is a player-game, and we have all stats. So we look up each stat type.
    dates = df["game_date_est"].astype(str)
    player_norms = df["player_name"].map(normalize_player_name) if "player_name" in df.columns else pd.Series("", index=df.index)

    # For the primary stat types, find the best-matching line
    # In training, we pick the stat with the highest coverage for market context
    for stat in stat_types:
        avg5_col = f"pre_{stat}_avg5"
        avg10_col = f"pre_{stat}_avg10"
        if avg5_col not in df.columns:
            continue

        keys = dates + "_" + player_norms + f"_{stat}"
        matched = keys.map(line_lookup).dropna()
        if matched.empty:
            continue

        for idx, match_data in matched.items():
            if not isinstance(match_data, dict):
                continue
            open_line = match_data.get("open_line", np.nan)
            if pd.isna(open_line):
                continue
            df.loc[idx, "prop_open_line"] = open_line
            df.loc[idx, "line_available"] = 1.0
            if pd.notna(match_data.get("over_implied")):
                df.loc[idx, "implied_over_prob"] = match_data["over_implied"]
            # Compute line vs averages
            avg5_val = df.loc[idx, avg5_col] if avg5_col in df.columns else np.nan
            avg10_val = df.loc[idx, avg10_col] if avg10_col in df.columns else np.nan
            if pd.notna(avg5_val):
                df.loc[idx, "prop_line_vs_avg5"] = open_line - float(avg5_val)
            if pd.notna(avg10_val):
                df.loc[idx, "prop_line_vs_avg10"] = open_line - float(avg10_val)

    n_with_lines = int(df["line_available"].sum())
    print(f"  Market line features: {n_with_lines}/{len(df)} rows with prop lines", flush=True)
    return df


def _run_market_line_ablation(player_df: pd.DataFrame) -> None:
    """Walk-forward backtest comparing models with/without market line features.

    Trains two sets of models: baseline (no market features) and enhanced (with
    prop market line features). Compares MAE and Brier score.
    """
    # Add market features to data
    df_with = add_market_line_features(player_df)
    n_with = int(df_with["line_available"].sum())
    print(f"  Rows with market lines: {n_with}/{len(df_with)}", flush=True)

    if n_with < 200:
        print("  Insufficient market line coverage for ablation. Need >= 200 rows.", flush=True)
        return

    # Use time-series split
    from sklearn.model_selection import TimeSeriesSplit

    if "game_date_est" in df_with.columns:
        df_with = df_with.sort_values("game_date_est").reset_index(drop=True)

    n_splits = min(3, max(2, len(df_with) // 2000))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    stat_targets = ["points", "rebounds", "assists"]
    results: dict[str, dict[str, list[float]]] = {
        st: {"baseline_mae": [], "market_mae": []} for st in stat_targets
    }

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(df_with)):
        train = df_with.iloc[train_idx].copy()
        test = df_with.iloc[test_idx].copy()
        print(f"\n  Fold {fold_i + 1}/{n_splits}: train={len(train)}, test={len(test)}", flush=True)

        for target in stat_targets:
            if target not in train.columns or train[target].notna().sum() < 200:
                continue

            # Baseline (no market features)
            feats_base = get_feature_list(target, two_stage=False, use_market_features=False)
            try:
                imp_b, mod_b, used_b = train_prop_model(train, feats_base, target)
                pred_b = predict_prop(imp_b, mod_b, used_b, test)
                valid = test[target].notna()
                mae_b = float(mean_absolute_error(test.loc[valid, target], pred_b[valid]))
                results[target]["baseline_mae"].append(mae_b)
            except ValueError:
                continue

            # With market features
            feats_mkt = get_feature_list(target, two_stage=False, use_market_features=True)
            try:
                imp_m, mod_m, used_m = train_prop_model(train, feats_mkt, target)
                pred_m = predict_prop(imp_m, mod_m, used_m, test)
                mae_m = float(mean_absolute_error(test.loc[valid, target], pred_m[valid]))
                results[target]["market_mae"].append(mae_m)
            except ValueError:
                continue

    print(f"\n  === Market Line Feature Ablation Results ===", flush=True)
    for target, vals in results.items():
        if vals["baseline_mae"] and vals["market_mae"]:
            b_mean = np.mean(vals["baseline_mae"])
            m_mean = np.mean(vals["market_mae"])
            diff = m_mean - b_mean
            pct = diff / b_mean * 100 if b_mean > 0 else 0
            better = "IMPROVED" if diff < 0 else "DEGRADED"
            print(
                f"    {target:>10s}: baseline MAE={b_mean:.3f}  market MAE={m_mean:.3f}  "
                f"diff={diff:+.3f} ({pct:+.1f}%) [{better}]",
                flush=True,
            )
        else:
            print(f"    {target:>10s}: insufficient data for comparison", flush=True)


# ---------------------------------------------------------------------------
# Feature lists for each prop target
# ---------------------------------------------------------------------------

def get_feature_list(target: str, two_stage: bool = False, use_market_features: bool = False) -> list[str]:
    """Get the feature list for a given prop target.

    If two_stage=True, returns features for the per-minute-rate model
    (which includes predicted minutes as a feature).
    """
    # Common features for all prop models
    common = [
        # Player rolling averages (last 3/5/10 — no season avg to prevent stale anchoring)
        "pre_minutes_avg3", "pre_minutes_avg5", "pre_minutes_avg10",
        "pre_minutes_ewm5", "pre_minutes_ewm10",
        "pre_minutes_std10",
        "pre_starter_rate",
        "pre_minutes_trend",
        "pre_usage_proxy",
        "pred_starter_prob",
        "confirmed_starter",
        "lineup_confirmed",
        "player_days_rest",
        "player_is_home",
        # B2B and schedule fatigue
        "is_b2b",
        "is_3_in_4",
        "b2b_x_minutes",
        "b2b_x_starter",
        # Team context
        "team_pre_possessions_avg5", "team_pre_possessions_avg10",
        "team_pre_off_rating_avg5", "team_pre_off_rating_avg10",
        "team_pre_net_rating_avg5",
        # Opponent context
        "opp_pre_def_rating_avg5", "opp_pre_def_rating_avg10",
        "opp_pre_possessions_avg5", "opp_pre_possessions_avg10",
        "opp_pre_net_rating_avg5",
        # Matchup features
        "matchup_pace_avg",
        "matchup_off_vs_def",
        "pace_diff_vs_recent",
        # Teammate injury context
        "team_injury_proxy_missing_minutes5",
        "team_injury_proxy_missing_points5",
        "team_star_player_absent_flag",
        "team_active_count",
        "team_injury_pressure",
        # Live injury report context
        "injury_availability_prob",
        "injury_unavailability_prob",
        "injury_is_out",
        "injury_is_doubtful",
        "injury_is_questionable",
        "injury_is_probable",
        # Usage boost
        "usage_boost_proxy",
        "minutes_x_injury_pressure",
        "pace_x_injury_pressure",
        "role_pts_injury_beta20",
        "role_reb_injury_beta20",
        "role_ast_injury_beta20",
        "role_min_injury_beta20",
        # Blowout risk
        "net_rating_diff",
        "blowout_risk",
        # Vegas game total / spread context (backfilled from historical odds)
        "implied_total",
        "implied_spread",
        "implied_team_total",
        # Blowout-adjusted minutes (from Vegas spread)
        "abs_spread",
        "spread_x_starter",
        "is_big_favorite",
        # Opponent positional defense
        "opp_pts_allowed_to_pos_avg10",
        "opp_reb_allowed_to_pos_avg10",
        "opp_ast_allowed_to_pos_avg10",
        # Player career / age-rest interactions
        "player_career_games",
        "is_veteran",
        "veteran_rest_effect",
        "career_x_rest",
        # Referee crew features (from game-level monolith data)
        "ref_crew_avg_total",
        "ref_crew_avg_fta",
        "ref_crew_avg_fouls",
        "ref_crew_avg_pace",
        "ref_crew_total_over_league_avg",
        "ref_crew_pace_over_league_avg",
        # OT detection
        "is_ot",
        # Foul trouble / engagement
        "pre_fouls_personal_avg5",
        "pre_fouls_personal_avg10",
    ]
    if USE_BOXSCORE_ADV_FEATURES:
        common += [
            "pre_adv_usage_pct_avg5", "pre_adv_usage_pct_avg10", "pre_adv_usage_pct_ewm5",
            "pre_adv_pace_avg5", "pre_adv_possessions_avg5",
            "pre_adv_off_rating_avg5", "pre_adv_ts_pct_avg5",
        ]

    # Target-specific features
    # NOTE: season averages + venue splits (expanding averages) excluded to prevent
    # anchoring predictions to stale early-season production. avg3/5/10 + EWM are sufficient.
    if target == "minutes":
        # Minutes model: no per-minute features to avoid circularity
        specific = [
            "pre_points_avg3", "pre_points_avg5",
        ]
    elif target == "points":
        specific = [
            "pre_points_avg3", "pre_points_avg5", "pre_points_avg10",
            "pre_points_ewm5", "pre_points_ewm10",
            "pre_points_std10",
            "pre_points_trend",
            "pre_fga_avg5", "pre_fga_avg10",
            "pre_fta_avg5", "pre_fta_avg10",
            "pre_fg3m_avg3", "pre_fg3m_avg5", "pre_fg3m_avg10",
            # Fouls drawn → FTA predictor (stable scoring component)
            "pre_fouls_drawn_avg3", "pre_fouls_drawn_avg5", "pre_fouls_drawn_avg10",
            "pre_fouls_drawn_per_min_avg5",
            # Scoring breakdown (interior vs perimeter, transition)
            "pre_pts_in_paint_avg5", "pre_pts_in_paint_avg10",
            "pre_pts_fast_break_avg5",
            # Shot volume per-minute rates
            "pre_fga_per_min_avg5", "pre_fta_per_min_avg5",
            # Per-minute rate
            "pre_pts_per_min_avg3", "pre_pts_per_min_avg5", "pre_pts_per_min_ewm5",
        ]
        if USE_BOXSCORE_ADV_FEATURES:
            specific += [
                "pre_adv_usage_pct_avg3",
                "pre_adv_off_rating_avg10",
                "pre_adv_ts_pct_avg10",
            ]
        if two_stage:
            specific.append("pred_minutes")
    elif target == "rebounds":
        specific = [
            "pre_rebounds_avg3", "pre_rebounds_avg5", "pre_rebounds_avg10",
            "pre_rebounds_ewm5", "pre_rebounds_ewm10",
            "pre_rebounds_std10",
            "pre_orb_avg5", "pre_drb_avg5",
            "pre_orb_avg10", "pre_drb_avg10",
            # Rebound opportunity / share features
            "pre_team_rebounds_avg10",
            "pre_team_orb_avg10",
            "pre_team_drb_avg10",
            "pre_team_missed_fg_avg10",
            "pre_opp_missed_fg_avg10",
            "pre_total_missed_fg_avg10",
            "pre_team_reb_share_avg10",
            "pre_team_orb_share_avg10",
            "pre_team_drb_share_avg10",
            "pre_player_reb_share_avg10",
            "pre_player_orb_share_avg10",
            "pre_player_drb_share_avg10",
            "pre_player_reb_opp_proxy_avg10",
            "pre_other_center_minutes_avg10",
            "pre_center_depth_risk",
            # Per-minute rate
            "pre_reb_per_min_avg3", "pre_reb_per_min_avg5", "pre_reb_per_min_ewm5",
        ]
        if USE_BOXSCORE_ADV_FEATURES:
            specific += [
                "pre_adv_reb_pct_avg5", "pre_adv_reb_pct_avg10",
                "pre_adv_possessions_avg10",
            ]
        if two_stage:
            specific.append("pred_minutes")
    elif target == "assists":
        specific = [
            "pre_assists_avg3", "pre_assists_avg5", "pre_assists_avg10",
            "pre_assists_ewm5", "pre_assists_ewm10",
            "pre_assists_std10",
            "pre_tov_avg5", "pre_tov_avg10",
            # Per-minute rate
            "pre_ast_per_min_avg3", "pre_ast_per_min_avg5", "pre_ast_per_min_ewm5",
        ]
        if USE_BOXSCORE_ADV_FEATURES:
            specific += [
                "pre_adv_ast_pct_avg5", "pre_adv_ast_pct_avg10",
                "pre_adv_usage_pct_avg3",
            ]
        if two_stage:
            specific.append("pred_minutes")
    elif target == "fg3m":
        specific = [
            "pre_fg3m_avg3", "pre_fg3m_avg5", "pre_fg3m_avg10",
            "pre_fg3m_ewm5", "pre_fg3m_ewm10",
            # 3PA volume (stickier than make rate)
            "pre_fg3a_avg3", "pre_fg3a_avg5", "pre_fg3a_avg10",
            "pre_fg3a_per_min_avg5",
            "pre_points_avg5",
            # Per-minute rate
            "pre_fg3m_per_min_avg3", "pre_fg3m_per_min_avg5", "pre_fg3m_per_min_ewm5",
        ]
        if USE_BOXSCORE_ADV_FEATURES:
            specific += [
                "pre_adv_usage_pct_avg3",
                "pre_adv_pace_avg10",
            ]
        if two_stage:
            specific.append("pred_minutes")
    else:
        specific = []

    # Phase 3: Market line features (optional, gated behind use_market_features)
    market_feats: list[str] = []
    if use_market_features:
        market_feats = [
            "prop_open_line",
            "prop_line_vs_avg5",
            "prop_line_vs_avg10",
            "implied_over_prob",
            "line_available",
        ]

    return common + specific + market_feats


def filter_features(features: list[str], df: pd.DataFrame) -> list[str]:
    """Filter to features present in DataFrame and not entirely NaN."""
    out: list[str] = []
    for f in features:
        if f not in df.columns:
            continue
        s = df[f]
        if hasattr(s, "notna") and not bool(s.notna().any()):
            continue
        out.append(f)
    return out


def get_residual_feature_list(target: str) -> list[str]:
    """Feature list for Stage 3 residual model (Phase 4).

    Includes base features plus OOF predictions and interaction features.
    """
    base = get_feature_list(target, two_stage=True, use_market_features=True)
    residual_specific = [
        f"oof_pred_{target}",    # the base OOF prediction itself
        "oof_pred_minutes",       # OOF minutes prediction
        # Market context
        "prop_open_line",
        "oof_pred_vs_line",       # oof_pred - open_line
        # Interaction features
        "oof_pred_x_b2b",            # oof_pred * is_b2b
        "oof_pred_x_injury_pressure",  # oof_pred * team_injury_pressure
    ]
    return base + residual_specific


# ---------------------------------------------------------------------------
# Model training and prediction
# ---------------------------------------------------------------------------

def train_prop_model(
    train_df: pd.DataFrame,
    features: list[str],
    target: str,
    params: dict[str, Any] | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> tuple[SimpleImputer, XGBRegressor, list[str]]:
    """Train an XGBoost regressor for a player prop target."""
    train = train_df.dropna(subset=[target]).copy()
    if train.empty:
        raise ValueError(f"No training data for target {target}")
    feats = [f for f in filter_features(features, train) if train[f].notna().any()]
    if not feats:
        raise ValueError(f"No usable features for target {target}")

    X = train[feats]
    y = train[target]

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    sw = None
    if sample_weight is not None:
        if isinstance(sample_weight, pd.Series):
            sw = sample_weight.reindex(train.index).to_numpy(dtype=float)
        else:
            sw = np.asarray(sample_weight, dtype=float)
            if len(sw) != len(train):
                sw = None
        if sw is not None:
            sw = np.where(np.isfinite(sw), sw, 1.0)
            sw = np.clip(sw, 0.05, None)

    p = params or {}
    default_params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.025,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
    }
    if target == "minutes":
        default_params.update(
            {
                "n_estimators": 500,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
                "reg_alpha": 0.25,
                "min_child_weight": 5,
            }
        )
    model = XGBRegressor(
        n_estimators=p.get("n_estimators", default_params["n_estimators"]),
        max_depth=p.get("max_depth", default_params["max_depth"]),
        learning_rate=p.get("learning_rate", default_params["learning_rate"]),
        subsample=p.get("subsample", default_params["subsample"]),
        colsample_bytree=p.get("colsample_bytree", default_params["colsample_bytree"]),
        reg_lambda=p.get("reg_lambda", default_params["reg_lambda"]),
        reg_alpha=p.get("reg_alpha", default_params["reg_alpha"]),
        min_child_weight=p.get("min_child_weight", default_params["min_child_weight"]),
        eval_metric="mae",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_imp, y, sample_weight=sw)
    return imp, model, feats


def predict_prop(
    imp: SimpleImputer,
    model: XGBRegressor,
    features: list[str],
    pred_df: pd.DataFrame,
) -> np.ndarray:
    """Generate predictions for a prop model."""
    feats_present = [f for f in features if f in pred_df.columns]
    X = pred_df[feats_present].copy()

    for f in features:
        if f not in X.columns:
            X[f] = np.nan
    X = X[features]

    X_imp = imp.transform(X)
    return model.predict(X_imp)


def _time_series_oof_predictions(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    sample_weight: pd.Series | np.ndarray | None = None,
    min_rows: int = 500,
) -> pd.Series:
    """Generate time-series out-of-fold predictions for leakage-safe stacked features."""
    valid = df.dropna(subset=[target]).copy()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if len(valid) < min_rows:
        return out

    n_splits = min(5, max(2, len(valid) // 1200))
    if n_splits < 2:
        return out

    splitter = TimeSeriesSplit(n_splits=n_splits)
    oof = pd.Series(np.nan, index=valid.index, dtype=float)

    for train_idx, val_idx in splitter.split(valid):
        fold_train = valid.iloc[train_idx].copy()
        fold_val = valid.iloc[val_idx].copy()
        if fold_train.empty or fold_val.empty:
            continue

        fold_sw = None
        if sample_weight is not None:
            if isinstance(sample_weight, pd.Series):
                fold_sw = sample_weight.reindex(fold_train.index)
            else:
                sw_arr = np.asarray(sample_weight, dtype=float)
                if len(sw_arr) == len(df):
                    fold_sw = pd.Series(sw_arr, index=df.index).reindex(fold_train.index)

        try:
            imp, model, used_feats = train_prop_model(
                fold_train,
                features,
                target,
                sample_weight=fold_sw,
            )
            fold_pred = predict_prop(imp, model, used_feats, fold_val)
            oof.loc[fold_val.index] = fold_pred
        except ValueError:
            continue

    out.loc[oof.index] = oof
    return out


def train_oof_residual_models(
    train_df: pd.DataFrame,
    two_stage_models: dict[str, Any],
    min_oof_rows: int = 500,
) -> dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]]:
    """Train Stage 3 OOF residual models (Phase 4).

    For each stat target:
    1. Generate OOF base predictions using _time_series_oof_predictions
    2. Compute residuals: actual - oof_pred_base
    3. Train residual XGBoost to predict residuals
    4. Only train when >= min_oof_rows valid OOF rows exist

    Returns dict mapping target -> (imputer, model, features) for residual models.
    """
    residual_models: dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]] = {}
    stat_targets = ["points", "rebounds", "assists"]
    if "fg3m" in train_df.columns and train_df["fg3m"].notna().sum() > 200:
        stat_targets.append("fg3m")

    # Need OOF minutes predictions for residual features
    min_features = get_feature_list("minutes", two_stage=False)
    oof_minutes = _time_series_oof_predictions(
        train_df, min_features, "minutes", min_rows=min_oof_rows,
    )
    train_df = train_df.copy()
    train_df["oof_pred_minutes"] = oof_minutes

    for target in stat_targets:
        if target not in train_df.columns:
            continue

        # Generate OOF base predictions for this stat
        stat_features = get_feature_list(target, two_stage=True)
        oof_preds = _time_series_oof_predictions(
            train_df, stat_features, target, min_rows=min_oof_rows,
        )

        # Only use rows where OOF prediction and actual are both available
        oof_col = f"oof_pred_{target}"
        train_df[oof_col] = oof_preds
        valid_mask = train_df[oof_col].notna() & train_df[target].notna()
        valid_df = train_df[valid_mask].copy()

        if len(valid_df) < min_oof_rows:
            print(f"    Residual {target}: skipped (only {len(valid_df)}/{min_oof_rows} valid OOF rows)", flush=True)
            continue

        # Compute residuals
        residual_target = f"_resid_{target}"
        valid_df[residual_target] = valid_df[target] - valid_df[oof_col]

        # Build interaction features for residual model
        valid_df["oof_pred_vs_line"] = valid_df[oof_col] - valid_df.get("prop_open_line", np.nan)
        valid_df["oof_pred_x_b2b"] = valid_df[oof_col] * valid_df.get("is_b2b", 0).astype(float)
        valid_df["oof_pred_x_injury_pressure"] = valid_df[oof_col] * valid_df.get("team_injury_pressure", 0).astype(float)

        # Conservative residual model hyperparameters
        resid_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "reg_lambda": 5.0,
            "reg_alpha": 0.5,
            "min_child_weight": 5,
        }

        resid_features = get_residual_feature_list(target)

        try:
            imp, model, used_feats = train_prop_model(
                valid_df, resid_features, residual_target, params=resid_params,
            )
            residual_models[target] = (imp, model, used_feats)
            # Report residual stats
            resid_vals = valid_df[residual_target]
            print(
                f"    Residual {target}: trained on {len(valid_df)} rows  "
                f"mean_resid={resid_vals.mean():.2f}  std_resid={resid_vals.std():.2f}",
                flush=True,
            )
        except ValueError as e:
            print(f"    Residual {target}: training failed ({e})", flush=True)
            continue

    return residual_models


def train_two_stage_models(
    train_df: pd.DataFrame,
) -> dict[str, Any]:
    """Train two-stage models: minutes first, then stats with predicted minutes.

    Stage 1: Train minutes model
    Stage 2: For each stat target, train model that includes predicted minutes as feature
    """
    models: dict[str, Any] = {}

    # Stage 1: Train minutes model
    min_features = get_feature_list("minutes", two_stage=False)
    train_for_minutes = train_df.copy()
    train_for_minutes["pred_starter_prob"] = train_for_minutes.get("pre_starter_rate", np.nan)
    if "confirmed_starter" in train_for_minutes.columns:
        mask_known = train_for_minutes["confirmed_starter"].notna()
        if mask_known.any():
            train_for_minutes.loc[mask_known, "pred_starter_prob"] = train_for_minutes.loc[mask_known, "confirmed_starter"].astype(float)

    recency_weight = pd.Series(1.0, index=train_for_minutes.index, dtype=float)
    if "game_time_utc" in train_for_minutes.columns:
        order = train_for_minutes["game_time_utc"].rank(method="first")
        recency_weight = 0.4 + 0.6 * (order / max(order.max(), 1.0))
    if "starter" in train_for_minutes.columns:
        recency_weight = recency_weight * (1.0 + 0.2 * (train_for_minutes["starter"].fillna(0) > 0).astype(float))

    try:
        min_imp, min_model, min_feats = train_prop_model(
            train_for_minutes,
            min_features,
            "minutes",
            sample_weight=recency_weight,
        )
        models["minutes"] = (min_imp, min_model, min_feats)

        # Leakage-safe Stage 2 input: prefer OOF minutes predictions.
        stage2_train = train_for_minutes.copy()
        oof_minutes = _time_series_oof_predictions(
            stage2_train,
            min_features,
            "minutes",
            sample_weight=recency_weight,
            min_rows=700,
        )
        fallback_minutes = predict_prop(min_imp, min_model, min_feats, stage2_train)
        stage2_train["pred_minutes"] = oof_minutes.reindex(stage2_train.index)
        stage2_train["pred_minutes"] = stage2_train["pred_minutes"].fillna(
            pd.Series(fallback_minutes, index=stage2_train.index)
        )
        stage2_train["pred_minutes"] = stage2_train["pred_minutes"].clip(lower=0.0)
    except ValueError:
        # Fall back to non-two-stage
        return {}

    # Stage 2: Train stat models with predicted minutes
    stat_targets = ["points", "rebounds", "assists"]
    if "fg3m" in stage2_train.columns and stage2_train["fg3m"].notna().sum() > 100:
        stat_targets.append("fg3m")

    # Recency weights for stat models too (same as minutes model)
    stat_recency_weight = pd.Series(1.0, index=stage2_train.index, dtype=float)
    if "game_time_utc" in stage2_train.columns:
        stat_order = stage2_train["game_time_utc"].rank(method="first")
        stat_recency_weight = 0.4 + 0.6 * (stat_order / max(stat_order.max(), 1.0))

    for target in stat_targets:
        features = get_feature_list(target, two_stage=True)
        try:
            imp, model, feats = train_prop_model(
                stage2_train, features, target, sample_weight=stat_recency_weight,
            )
            models[target] = (imp, model, feats)
        except ValueError:
            # Fall back: train without pred_minutes
            features_no_stage = get_feature_list(target, two_stage=False)
            try:
                imp, model, feats = train_prop_model(
                    stage2_train, features_no_stage, target, sample_weight=stat_recency_weight,
                )
                models[target] = (imp, model, feats)
            except ValueError:
                continue

    # Stage 3 (Phase 4): OOF residual models
    print("  Training Stage 3 residual models (OOF)...", flush=True)
    residual_models = train_oof_residual_models(stage2_train, models, min_oof_rows=500)
    if residual_models:
        models["_residual"] = residual_models
        print(f"  Stage 3 residual models: {', '.join(sorted(residual_models.keys()))}", flush=True)
    else:
        print("  Stage 3: no residual models trained (insufficient OOF data)", flush=True)

    return models


def predict_two_stage(
    models: dict[str, Any],
    pred_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate two-stage predictions: minutes first, then use predicted minutes for stats."""
    pred_df = pred_df.copy()

    # Stage 0: starter probability from historical rate with confirmed starter override.
    pred_df["pred_starter_prob"] = pred_df.get("pre_starter_rate", np.nan)

    # Apply confirmed lineup override if available
    if "lineup_confirmed" in pred_df.columns and "confirmed_starter" in pred_df.columns:
        known_mask = pred_df["lineup_confirmed"].fillna(0).astype(int).eq(1) & pred_df["confirmed_starter"].notna()
        if known_mask.any():
            pred_df.loc[known_mask, "pred_starter_prob"] = pred_df.loc[known_mask, "confirmed_starter"].astype(float)

    # Stage 1: Predict minutes
    if "minutes" in models:
        min_imp, min_model, min_feats = models["minutes"]
        pred_df["pred_minutes"] = predict_prop(min_imp, min_model, min_feats, pred_df)
        pred_df["pred_minutes"] = pred_df["pred_minutes"].clip(lower=0.0)
    else:
        return pred_df

    # Dampen minutes by injury risk in pregame mode (keeps questionable tags from over-projecting).
    if "injury_unavailability_prob" in pred_df.columns:
        unavail = pred_df["injury_unavailability_prob"].astype(float)
        base_adj = 1.0 - 0.25 * unavail.fillna(0.0)
        if "injury_is_doubtful" in pred_df.columns:
            base_adj = base_adj * np.where(pred_df["injury_is_doubtful"].astype(float) > 0, 0.7, 1.0)
        if "injury_is_out" in pred_df.columns:
            base_adj = np.where(pred_df["injury_is_out"].astype(float) > 0, 0.0, base_adj)
        pred_df["pred_minutes"] = pred_df["pred_minutes"] * np.clip(base_adj, 0.0, 1.0)

    # Stage 2: Predict stats using predicted minutes
    for target in ["points", "rebounds", "assists", "fg3m"]:
        if target in models:
            imp, model, feats = models[target]
            pred_df[f"pred_{target}"] = predict_prop(imp, model, feats, pred_df)

    # Stage 3 (Phase 4): Apply residual correction with clipping
    residual_models = models.get("_residual", {})
    if residual_models:
        for target, (r_imp, r_model, r_feats) in residual_models.items():
            pred_col = f"pred_{target}"
            if pred_col not in pred_df.columns:
                continue

            # Build residual interaction features for prediction
            oof_col = f"oof_pred_{target}"
            pred_df[oof_col] = pred_df[pred_col]  # at prediction time, base pred serves as "oof" input
            pred_df["oof_pred_minutes"] = pred_df.get("pred_minutes", np.nan)
            pred_df["oof_pred_vs_line"] = pred_df[pred_col] - pred_df.get("prop_open_line", np.nan)
            pred_df["oof_pred_x_b2b"] = pred_df[pred_col] * pred_df.get("is_b2b", 0).astype(float)
            pred_df["oof_pred_x_injury_pressure"] = pred_df[pred_col] * pred_df.get("team_injury_pressure", 0).astype(float)

            correction = predict_prop(r_imp, r_model, r_feats, pred_df)

            # Clip correction to ±20% of base prediction
            base_val = pred_df[pred_col].to_numpy(dtype=float)
            max_correction = np.abs(base_val) * 0.20
            correction = np.clip(correction, -max_correction, max_correction)

            pred_df[pred_col] = base_val + correction

    return pred_df


def train_prediction_models(
    player_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]]]:
    """Train core prediction models once for reuse in the run."""
    two_stage_models = train_two_stage_models(player_df)
    single_models: dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]] = {}

    targets_to_model = list(PROP_TARGETS)
    if "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 100:
        targets_to_model.append("fg3m")

    for target in targets_to_model:
        if target in two_stage_models:
            continue
        features = get_feature_list(target)
        feats = filter_features(features, player_df)
        if not feats:
            continue
        try:
            imp, model, used_feats = train_prop_model(player_df, features, target)
            single_models[target] = (imp, model, used_feats)
        except ValueError:
            continue

    return two_stage_models, single_models


# ---------------------------------------------------------------------------
# Prop edge computation
# ---------------------------------------------------------------------------

def compute_prop_residual_stds(
    player_df: pd.DataFrame,
    test_frac: float = 0.2,
    two_stage_models: dict[str, Any] | None = None,
    single_models: dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]] | None = None,
    leakage_safe: bool = True,
) -> dict[str, float]:
    """Compute residual standard deviations for each prop target."""
    player_df = player_df.sort_values("game_time_utc").reset_index(drop=True)
    cut = int(len(player_df) * (1.0 - test_frac))
    train = player_df.iloc[:cut].copy()
    test = player_df.iloc[cut:].copy()

    # Use train-split-fitted models by default to avoid in-sample leakage in uncertainty estimates.
    if leakage_safe or two_stage_models is None or single_models is None:
        two_stage_models, single_models = train_prediction_models(train)

    residual_stds: dict[str, float] = {}
    all_targets = list(PROP_TARGETS) + ["fg3m"]

    for target in all_targets:
        if target not in player_df.columns or player_df[target].isna().all():
            continue

        test_valid = test.dropna(subset=[target]).copy()
        if test_valid.empty:
            continue

        preds: np.ndarray | None = None
        if target == "minutes":
            if "minutes" in two_stage_models:
                imp, model, feats = two_stage_models["minutes"]
                tv = test_valid.copy()
                tv["pred_starter_prob"] = tv.get("pre_starter_rate", np.nan)
                preds = predict_prop(imp, model, feats, tv)
        elif target in two_stage_models:
            tv = test_valid.copy()
            tv["pred_starter_prob"] = tv.get("pre_starter_rate", np.nan)
            if "minutes" in two_stage_models:
                min_imp, min_model, min_feats = two_stage_models["minutes"]
                tv["pred_minutes"] = predict_prop(min_imp, min_model, min_feats, tv)
            imp, model, feats = two_stage_models[target]
            preds = predict_prop(imp, model, feats, tv)
        elif target in single_models:
            imp, model, feats = single_models[target]
            preds = predict_prop(imp, model, feats, test_valid)

        if preds is None:
            continue
        actual = test_valid[target].to_numpy(dtype=float)
        residual_std = float(np.std(actual - preds))
        residual_stds[target] = residual_std

    return residual_stds


def _fit_probability_calibrator(
    p_raw: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any] | None:
    """Fit per-stat probability calibrator (Platt vs isotonic) with OOF selection."""
    x = np.clip(np.asarray(p_raw, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=int)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 80 or len(np.unique(y)) < 2:
        return None

    n_splits = min(5, max(2, len(x) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_iso = np.full(len(x), np.nan, dtype=float)
    oof_platt = np.full(len(x), np.nan, dtype=float)

    for tr_idx, va_idx in tscv.split(np.arange(len(x))):
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_va = x[va_idx]
        if len(np.unique(y_tr)) < 2:
            continue

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x_tr, y_tr)
        oof_iso[va_idx] = iso.predict(x_va)

        pl = LogisticRegression(solver="lbfgs", max_iter=1000)
        pl.fit(_safe_logit(x_tr).reshape(-1, 1), y_tr)
        oof_platt[va_idx] = pl.predict_proba(_safe_logit(x_va).reshape(-1, 1))[:, 1]

    valid_iso = np.isfinite(oof_iso)
    valid_platt = np.isfinite(oof_platt)
    brier_iso = np.mean((oof_iso[valid_iso] - y[valid_iso]) ** 2) if valid_iso.any() else np.inf
    brier_platt = np.mean((oof_platt[valid_platt] - y[valid_platt]) ** 2) if valid_platt.any() else np.inf

    if brier_iso <= brier_platt:
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(x, y)
        return {"method": "isotonic", "model": model, "brier_oof": float(brier_iso), "n_train": int(len(x))}

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(_safe_logit(x).reshape(-1, 1), y)
    return {"method": "platt", "model": model, "brier_oof": float(brier_platt), "n_train": int(len(x))}


def _apply_probability_calibrator(p_raw: float, calibrator: dict[str, Any] | None) -> float:
    if calibrator is None or pd.isna(p_raw):
        return float(p_raw) if pd.notna(p_raw) else np.nan
    p = float(np.clip(p_raw, 1e-6, 1 - 1e-6))
    method = str(calibrator.get("method", ""))
    model = calibrator.get("model")
    if method == "isotonic" and model is not None:
        return float(np.clip(model.predict([p])[0], 1e-6, 1 - 1e-6))
    if method == "platt" and model is not None:
        return float(np.clip(model.predict_proba(_safe_logit(np.array([p])).reshape(-1, 1))[0, 1], 1e-6, 1 - 1e-6))
    return p


def load_cached_prop_lines(max_dates: int = 180) -> pd.DataFrame:
    """Load cached prop lines from PROP_CACHE_DIR."""
    if not PROP_CACHE_DIR.exists():
        return pd.DataFrame()
    rows: list[tuple[str, Path]] = []
    for f in PROP_CACHE_DIR.glob("prop_lines_*.csv"):
        m = re.search(r"prop_lines_(\d{8})\.csv$", f.name)
        if not m:
            continue
        rows.append((m.group(1), f))
    if not rows:
        return pd.DataFrame()
    rows.sort(key=lambda t: t[0])
    if max_dates > 0 and len(rows) > max_dates:
        rows = rows[-max_dates:]
    frames: list[pd.DataFrame] = []
    for date_str, path in rows:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        if "game_date_est" not in df.columns:
            if "date" in df.columns:
                df["game_date_est"] = df["date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
            else:
                df["game_date_est"] = date_str
        df["game_date_est"] = df["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
        if "stat_type" in df.columns:
            df["stat_type"] = df["stat_type"].astype(str).str.lower().str.strip()
        if "team" in df.columns:
            df["team"] = _normalize_team_series(df["team"])
        else:
            df["team"] = ""
        if "player_name" in df.columns:
            df["player_name_norm"] = df["player_name"].map(normalize_player_name)
        else:
            continue
        _add_implied_probs(df)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_market_training_frame(
    player_df: pd.DataFrame,
    max_dates: int = 180,
    pretrained_models: tuple[dict[str, Any], dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]]] | None = None,
) -> pd.DataFrame:
    """Join cached market lines to historical rows with model predictions.

    If ``pretrained_models`` is provided as (two_stage_models, single_models),
    those are reused instead of training from scratch — avoids expensive
    duplicate training within a single run.
    """
    lines = load_cached_prop_lines(max_dates=max_dates)
    if lines.empty:
        return pd.DataFrame()

    hist = player_df.copy()
    if "game_date_est" not in hist.columns:
        hist["game_date_est"] = pd.to_datetime(hist["game_time_utc"], utc=True, errors="coerce").dt.strftime("%Y%m%d")
    hist["game_date_est"] = hist["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    hist["player_name_norm"] = hist["player_name"].map(normalize_player_name)
    hist = hist.sort_values("game_time_utc").reset_index(drop=True)
    cut = int(len(hist) * 0.8)
    fit_df = hist.iloc[:cut].copy()
    score_df = hist.iloc[cut:].copy()
    if fit_df.empty or score_df.empty:
        return pd.DataFrame()

    # Out-of-sample predictions for residual training (avoid leakage).
    if pretrained_models is not None:
        two_stage, single_models = pretrained_models
    else:
        two_stage, single_models = train_prediction_models(fit_df)
    pred_hist = score_df.copy()
    if two_stage:
        pred_hist = predict_two_stage(two_stage, pred_hist)
    for t in ["points", "rebounds", "assists", "fg3m"]:
        pred_col = f"pred_{t}"
        if pred_col not in pred_hist.columns and t in fit_df.columns:
            bundle = single_models.get(t)
            if bundle is None:
                continue
            imp, model, feats = bundle
            pred_hist[pred_col] = predict_prop(imp, model, feats, pred_hist)

    all_rows: list[pd.DataFrame] = []
    for stat in ["points", "rebounds", "assists", "fg3m"]:
        pred_col = f"pred_{stat}"
        if pred_col not in pred_hist.columns or stat not in pred_hist.columns:
            continue
        keep_cols = [
            "game_date_est",
            "team",
            "player_name_norm",
            "player_name",
            pred_col,
            stat,
            "pred_minutes",
            "pre_minutes_avg5",
            "pre_minutes_avg10",
            "pre_usage_proxy",
            "player_days_rest",
            "team_pre_net_rating_avg5",
            "opp_pre_def_rating_avg5",
            "matchup_pace_avg",
            "matchup_off_vs_def",
            "team_injury_pressure",
            "pred_starter_prob",
            "confirmed_starter",
            "injury_unavailability_prob",
            f"pre_{stat}_avg5",
            f"pre_{stat}_avg10",
        ]
        keep_cols = [c for c in keep_cols if c in pred_hist.columns]
        st = pred_hist[keep_cols].copy()
        st = st.rename(columns={pred_col: "pred_value", stat: "actual_value"})
        st["stat_type"] = stat
        all_rows.append(st)
    if not all_rows:
        return pd.DataFrame()
    base = pd.concat(all_rows, ignore_index=True)

    merge_keys = ["game_date_est", "player_name_norm", "stat_type"]
    has_team_keys = "team" in lines.columns and lines["team"].astype(str).str.strip().ne("").any()
    if has_team_keys:
        merged = lines.merge(
            base,
            on=["game_date_est", "team", "player_name_norm", "stat_type"],
            how="inner",
        )
        if merged.empty:
            merged = lines.merge(base, on=merge_keys, how="inner")
    else:
        merged = lines.merge(base, on=merge_keys, how="inner")
    if merged.empty:
        return merged
    merged["line"] = merged["line"].astype(float)
    merged["edge"] = merged["pred_value"] - merged["line"]
    merged["edge_pct"] = np.where(merged["line"] != 0, 100.0 * merged["edge"] / merged["line"], np.nan)
    merged = merged[np.isfinite(merged["pred_value"]) & np.isfinite(merged["actual_value"]) & np.isfinite(merged["line"])]
    return merged.reset_index(drop=True)


def train_market_residual_models(
    player_df: pd.DataFrame,
    max_dates: int = 180,
    pretrained_models: tuple[dict[str, Any], dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Train per-stat market residual and calibration models."""
    market_df = _build_market_training_frame(player_df, max_dates=max_dates, pretrained_models=pretrained_models)
    residual_models: dict[str, dict[str, Any]] = {}
    calibrators: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {"rows": int(len(market_df)), "per_stat": {}}
    if market_df.empty:
        return residual_models, calibrators, diagnostics

    for stat, st_df in market_df.groupby("stat_type"):
        st_df = st_df.sort_values("game_date_est").reset_index(drop=True)
        # Residual model: actual - line
        st_df["target_resid"] = st_df["actual_value"] - st_df["line"]
        feat_cols = [
            "pred_value",
            "line",
            "edge",
            "edge_pct",
            "over_implied_prob",
            "under_implied_prob",
            "over_odds",
            "under_odds",
            "pred_minutes",
            "pre_minutes_avg5",
            "pre_minutes_avg10",
            f"pre_{stat}_avg5",
            f"pre_{stat}_avg10",
            "pre_usage_proxy",
            "player_days_rest",
            "team_pre_net_rating_avg5",
            "opp_pre_def_rating_avg5",
            "matchup_pace_avg",
            "matchup_off_vs_def",
            "team_injury_pressure",
            "pred_starter_prob",
            "confirmed_starter",
            "injury_unavailability_prob",
        ]
        feat_cols = [c for c in feat_cols if c in st_df.columns]
        use = st_df.dropna(subset=["target_resid"]).copy()
        if len(use) >= 80 and feat_cols:
            imp = SimpleImputer(strategy="median")
            X = imp.fit_transform(use[feat_cols])
            y = use["target_resid"].to_numpy(dtype=float)
            model = XGBRegressor(
                n_estimators=260,
                max_depth=4,
                learning_rate=0.045,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=2.5,
                reg_alpha=0.1,
                min_child_weight=2,
                eval_metric="mae",
                random_state=42,
                verbosity=0,
            )
            model.fit(X, y)
            residual_models[stat] = {"imputer": imp, "model": model, "features": feat_cols, "n_train": int(len(use))}

        # Probability calibrators on historical market lines (by stat + side).
        resid_std = float(np.std(st_df["actual_value"].to_numpy(dtype=float) - st_df["pred_value"].to_numpy(dtype=float)))
        resid_std = max(resid_std, 0.01)
        z = (st_df["line"].to_numpy(dtype=float) - st_df["pred_value"].to_numpy(dtype=float)) / resid_std
        p_over_raw = 1.0 - sp_stats.norm.cdf(z)
        p_under_raw = 1.0 - p_over_raw
        non_push = np.abs(st_df["actual_value"].to_numpy(dtype=float) - st_df["line"].to_numpy(dtype=float)) > 1e-9
        labels_over = (st_df["actual_value"].to_numpy(dtype=float) > st_df["line"].to_numpy(dtype=float)).astype(int)
        labels_under = (st_df["actual_value"].to_numpy(dtype=float) < st_df["line"].to_numpy(dtype=float)).astype(int)
        calib_over = _fit_probability_calibrator(p_over_raw[non_push], labels_over[non_push])
        calib_under = _fit_probability_calibrator(p_under_raw[non_push], labels_under[non_push])
        stat_calibs: dict[str, Any] = {}
        if calib_over is not None:
            stat_calibs["over"] = calib_over
        if calib_under is not None:
            stat_calibs["under"] = calib_under
        if stat_calibs:
            calibrators[stat] = stat_calibs

        diagnostics["per_stat"][stat] = {
            "n_rows": int(len(st_df)),
            "residual_model": int(len(use)) if len(use) else 0,
            "calibrator_rows": int(non_push.sum()),
            "calib_over": bool(calib_over is not None),
            "calib_under": bool(calib_under is not None),
        }
    return residual_models, calibrators, diagnostics


def train_synthetic_probability_calibrators(
    player_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Fallback calibration from synthetic lines when market history is sparse."""
    calibrators: dict[str, dict[str, Any]] = {}
    player_df = player_df.sort_values("game_time_utc").reset_index(drop=True)
    cut = int(len(player_df) * 0.8)
    train = player_df.iloc[:cut].copy()
    test = player_df.iloc[cut:].copy()
    for stat in ["points", "rebounds", "assists", "fg3m"]:
        if stat not in player_df.columns:
            continue
        line_col = f"pre_{stat}_avg10" if f"pre_{stat}_avg10" in test.columns else f"pre_{stat}_season"
        if line_col not in test.columns:
            continue
        try:
            imp, model, feats = train_prop_model(train, get_feature_list(stat), stat)
        except ValueError:
            continue
        tv = test.dropna(subset=[stat, line_col]).copy()
        if len(tv) < 120:
            continue
        pred = predict_prop(imp, model, feats, tv)
        actual = tv[stat].to_numpy(dtype=float)
        line = tv[line_col].to_numpy(dtype=float)
        resid_std = max(float(np.std(actual - pred)), 0.01)
        p_over_raw = 1.0 - sp_stats.norm.cdf((line - pred) / resid_std)
        non_push = np.abs(actual - line) > 1e-9
        labels = (actual > line).astype(int)
        calib = _fit_probability_calibrator(p_over_raw[non_push], labels[non_push])
        if calib is not None:
            calibrators[stat] = calib
    return calibrators


def _predict_market_residual_adjustment(
    stat_type: str,
    pred_row: pd.Series,
    line_row: pd.Series,
    pred_val: float,
    residual_models: dict[str, dict[str, Any]] | None,
) -> float:
    if not residual_models or stat_type not in residual_models:
        return 0.0
    bundle = residual_models.get(stat_type, {})
    imp = bundle.get("imputer")
    model = bundle.get("model")
    feats = bundle.get("features", [])
    if imp is None or model is None or not feats:
        return 0.0

    line_val = float(line_row.get("line", np.nan))
    row_dict = {
        "pred_value": pred_val,
        "line": line_val,
        "edge": pred_val - line_val,
        "edge_pct": (100.0 * (pred_val - line_val) / line_val) if line_val else 0.0,
        "over_implied_prob": line_row.get("over_implied_prob", np.nan),
        "under_implied_prob": line_row.get("under_implied_prob", np.nan),
        "over_odds": line_row.get("over_odds", np.nan),
        "under_odds": line_row.get("under_odds", np.nan),
        "pred_minutes": pred_row.get("pred_minutes", np.nan),
        "pre_minutes_avg5": pred_row.get("pre_minutes_avg5", np.nan),
        "pre_minutes_avg10": pred_row.get("pre_minutes_avg10", np.nan),
        f"pre_{stat_type}_avg5": pred_row.get(f"pre_{stat_type}_avg5", np.nan),
        f"pre_{stat_type}_avg10": pred_row.get(f"pre_{stat_type}_avg10", np.nan),
        "pre_usage_proxy": pred_row.get("pre_usage_proxy", np.nan),
        "player_days_rest": pred_row.get("player_days_rest", np.nan),
        "team_pre_net_rating_avg5": pred_row.get("team_pre_net_rating_avg5", np.nan),
        "opp_pre_def_rating_avg5": pred_row.get("opp_pre_def_rating_avg5", np.nan),
        "matchup_pace_avg": pred_row.get("matchup_pace_avg", np.nan),
        "matchup_off_vs_def": pred_row.get("matchup_off_vs_def", np.nan),
        "team_injury_pressure": pred_row.get("team_injury_pressure", np.nan),
        "pred_starter_prob": pred_row.get("pred_starter_prob", np.nan),
        "confirmed_starter": pred_row.get("confirmed_starter", np.nan),
        "injury_unavailability_prob": pred_row.get("injury_unavailability_prob", np.nan),
    }
    X = pd.DataFrame([{f: row_dict.get(f, np.nan) for f in feats}])
    X_imp = imp.transform(X)
    resid = float(model.predict(X_imp)[0])
    if not np.isfinite(resid):
        return 0.0
    return resid


def compute_prop_edges(
    predictions: pd.DataFrame,
    prop_lines: pd.DataFrame,
    residual_stds: dict[str, float],
    market_residual_models: dict[str, dict[str, Any]] | None = None,
    prob_calibrators: dict[str, dict[str, Any]] | None = None,
    calibration_degraded_stats: set[str] | None = None,
) -> pd.DataFrame:
    """Compute edges between model predictions and prop lines."""
    if predictions.empty or prop_lines.empty:
        return pd.DataFrame()

    results = []

    for _, line_row in prop_lines.iterrows():
        player_name = line_row["player_name"]
        stat_type = line_row["stat_type"]
        line_val = float(line_row["line"])
        team = line_row.get("team", "")
        over_implied = line_row.get("over_implied_prob", np.nan)
        under_implied = line_row.get("under_implied_prob", np.nan)
        over_odds_raw = line_row.get("over_odds", np.nan)
        under_odds_raw = line_row.get("under_odds", np.nan)

        pred_col = f"pred_{stat_type}"
        if pred_col not in predictions.columns:
            continue

        # Match player name robustly across accent/punctuation variants
        pred_norm = predictions["player_name"].map(normalize_player_name)
        line_norm = normalize_player_name(player_name)
        mask = pred_norm.eq(line_norm)
        if not mask.any() and line_norm:
            mask = pred_norm.str.contains(re.escape(line_norm), na=False)
        if team and pd.notna(team):
            mask = mask & (predictions["team"] == team)

        matched = predictions[mask]
        if matched.empty:
            continue

        pred_row = matched.iloc[0]
        pred_base = float(pred_row[pred_col])
        if pd.isna(pred_base):
            continue

        resid_adj = _predict_market_residual_adjustment(
            stat_type,
            pred_row,
            line_row,
            pred_base,
            market_residual_models,
        )
        pred_val = pred_base + resid_adj

        edge = pred_val - line_val
        edge_pct = (edge / line_val * 100) if line_val != 0 else np.nan

        # Compute p_over and p_under using blended (player-specific + global) std
        resid_std = residual_stds.get(stat_type, np.nan)
        player_std = pred_row.get(f"pre_{stat_type}_std10", np.nan)
        if pd.notna(player_std) and player_std > 0 and pd.notna(resid_std):
            n_games = pred_row.get("player_game_num", 10)
            player_weight = min(n_games / 30, 0.7)  # max 70% player-specific
            blended_std = player_weight * player_std + (1 - player_weight) * resid_std
        else:
            blended_std = resid_std
        if pd.notna(blended_std) and blended_std > 0:
            z = (line_val - pred_val) / blended_std
            # Use t-distribution (df=7) instead of Gaussian for heavier tails.
            # Counting stats (rebounds, assists) have fat tails that Gaussian
            # underestimates, leading to overconfident probabilities near the tails.
            p_over_raw = float(1.0 - sp_stats.t.cdf(z, df=7))
            stat_calibs = (prob_calibrators or {}).get(stat_type, {})
            if isinstance(stat_calibs, dict) and ("over" in stat_calibs or "under" in stat_calibs):
                p_under_raw = 1.0 - p_over_raw
                p_over = _apply_probability_calibrator(p_over_raw, stat_calibs.get("over"))
                p_under = _apply_probability_calibrator(p_under_raw, stat_calibs.get("under"))
                p_sum = p_over + p_under
                if pd.notna(p_sum) and p_sum > 0:
                    p_over = float(np.clip(p_over / p_sum, 1e-6, 1 - 1e-6))
                    p_under = float(np.clip(p_under / p_sum, 1e-6, 1 - 1e-6))
            else:
                p_over = _apply_probability_calibrator(p_over_raw, stat_calibs if isinstance(stat_calibs, dict) else None)
                p_under = 1.0 - p_over
        else:
            p_over_raw = np.nan
            p_over = np.nan
            p_under = np.nan

        # Compute EV using actual market odds
        if pd.notna(over_implied) and over_implied > 0:
            total_implied = (over_implied + under_implied) if pd.notna(under_implied) else 1.0
            over_break_even = over_implied / total_implied if total_implied > 0 else 0.5
            under_break_even = under_implied / total_implied if pd.notna(under_implied) and total_implied > 0 else 0.5
            over_payout = _american_odds_to_decimal(over_odds_raw) if pd.notna(over_odds_raw) else VIG_FACTOR
            under_payout = _american_odds_to_decimal(under_odds_raw) if pd.notna(under_odds_raw) else VIG_FACTOR
        else:
            over_break_even = BREAKEVEN_PROB
            under_break_even = BREAKEVEN_PROB
            over_payout = VIG_FACTOR
            under_payout = VIG_FACTOR

        ev_over = np.nan
        ev_under = np.nan
        if pd.notna(p_over):
            ev_over = p_over * over_payout - (1.0 - p_over)
            ev_under = p_under * under_payout - (1.0 - p_under)

        # Generate signal with thresholds + policy gates
        signal = "NO BET"
        confidence = ""
        signal_blocked_reason = ""
        min_abs = MIN_ABS_EDGE.get(stat_type, 2.0)
        stat_gate_ok = (not SIGNAL_POINTS_ONLY) or (stat_type == "points")
        pred_minutes = pd.to_numeric(pred_row.get("pred_minutes"), errors="coerce")
        pre_minutes_avg10 = pd.to_numeric(pred_row.get("pre_minutes_avg10"), errors="coerce")
        minutes_gate_ok = (
            pd.notna(pred_minutes) and pred_minutes >= MIN_SIGNAL_PRED_MINUTES
            and pd.notna(pre_minutes_avg10) and pre_minutes_avg10 >= MIN_SIGNAL_PRE_MINUTES_AVG10
        )

        # Calibration reliability gate (Phase 2)
        calib_gate_ok = True
        if calibration_degraded_stats and stat_type in calibration_degraded_stats:
            calib_gate_ok = False

        if not stat_gate_ok:
            signal_blocked_reason = "non_points_stat_filtered"
        elif not minutes_gate_ok:
            signal_blocked_reason = "minutes_gate"
        elif not calib_gate_ok:
            signal_blocked_reason = "calibration_drift"

        if stat_gate_ok and minutes_gate_ok and calib_gate_ok and pd.notna(ev_over) and pd.notna(ev_under) and abs(edge) >= min_abs:
            if (
                ev_over > MIN_EV_BY_SIDE["OVER"]
                and p_over > over_break_even
                and abs(edge_pct) >= MIN_EDGE_PCT_BY_SIDE["OVER"]
            ):
                signal = "OVER"
                if ev_over >= BEST_BET_EV:
                    confidence = "BEST BET"
                else:
                    confidence = "LEAN"
            elif (
                ev_under > MIN_EV_BY_SIDE["UNDER"]
                and p_under > under_break_even
                and abs(edge_pct) >= MIN_EDGE_PCT_BY_SIDE["UNDER"]
            ):
                signal = "UNDER"
                if ev_under >= BEST_BET_EV:
                    confidence = "BEST BET"
                else:
                    confidence = "LEAN"

        # Line movement signal: compare current line to open line
        open_line_val = line_row.get("open_line", np.nan)
        if pd.notna(open_line_val) and float(open_line_val) > 0:
            open_line_val = float(open_line_val)
            line_move = line_val - open_line_val
            line_move_pct = (line_move / open_line_val) * 100
            model_says_over = pred_val > line_val
            line_moved_up = line_move > 0
            line_confirms_model = (model_says_over and line_moved_up) or (not model_says_over and not line_moved_up)
            # Upgrade LEAN to BEST BET if line movement confirms model direction
            if signal != "NO BET" and confidence == "LEAN" and line_confirms_model and abs(line_move_pct) >= 2.0:
                confidence = "BEST BET"
        else:
            line_move = np.nan
            line_move_pct = np.nan
            line_confirms_model = np.nan

        results.append({
            "game_date_est": str(line_row.get("game_date_est", pred_row.get("game_date_est", ""))),
            "player_name": player_name,
            "team": team if (team and pd.notna(team)) else pred_row.get("team", ""),
            "opp": pred_row.get("opp", ""),
            "home_team": pred_row.get("home_team", ""),
            "away_team": pred_row.get("away_team", ""),
            "stat_type": stat_type,
            "prop_line": line_val,
            "pred_value": round(pred_val, 1),
            "pred_value_base": round(pred_base, 1),
            "market_resid_adj": round(resid_adj, 2),
            "edge": round(edge, 1),
            "edge_pct": round(edge_pct, 1) if pd.notna(edge_pct) else np.nan,
            "p_over_raw": round(p_over_raw, 3) if pd.notna(p_over_raw) else np.nan,
            "p_over": round(p_over, 3) if pd.notna(p_over) else np.nan,
            "p_under": round(p_under, 3) if pd.notna(p_under) else np.nan,
            "ev_over": round(ev_over, 3) if pd.notna(ev_over) else np.nan,
            "ev_under": round(ev_under, 3) if pd.notna(ev_under) else np.nan,
            "signal": signal,
            "confidence": confidence,
            "source": line_row.get("source", ""),
            "over_odds": over_odds_raw,
            "under_odds": under_odds_raw,
            "open_line": line_row.get("open_line", np.nan),
            "open_over_odds": line_row.get("open_over_odds", np.nan),
            "open_under_odds": line_row.get("open_under_odds", np.nan),
            "line_move": round(line_move, 1) if pd.notna(line_move) else np.nan,
            "line_move_pct": round(line_move_pct, 1) if pd.notna(line_move_pct) else np.nan,
            "line_confirms_model": line_confirms_model if not isinstance(line_confirms_model, float) else np.nan,
            "residual_std": round(resid_std, 2) if pd.notna(resid_std) else np.nan,
            "blended_std": round(blended_std, 2) if pd.notna(blended_std) else np.nan,
            "player_std": round(float(player_std), 2) if pd.notna(player_std) else np.nan,
            "pre_avg5": pred_row.get(f"pre_{stat_type}_avg5", np.nan),
            "pre_avg10": pred_row.get(f"pre_{stat_type}_avg10", np.nan),
            "pre_season": pred_row.get(f"pre_{stat_type}_season", np.nan),
            "lineup_confirmed": pred_row.get("lineup_confirmed", np.nan),
            "confirmed_starter": pred_row.get("confirmed_starter", np.nan),
            "injury_status": pred_row.get("injury_status", ""),
            "injury_unavailability_prob": pred_row.get("injury_unavailability_prob", np.nan),
            "signal_blocked_reason": signal_blocked_reason,
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Flag correlated signals on the same player (multiple props same direction)
    result_df["correlated"] = False
    if not result_df.empty:
        signal_mask = result_df["signal"] != "NO BET"
        player_signals = result_df[signal_mask].groupby("player_name")
        for player, group in player_signals:
            if len(group) > 1:
                directions = group["signal"].unique()
                # All same direction = likely correlated (pts/reb/ast all OVER)
                is_corr = len(directions) == 1
                result_df.loc[group.index, "correlated"] = is_corr

    # Sort by absolute EV (strongest signals first)
    result_df["abs_ev"] = result_df.apply(
        lambda r: max(r["ev_over"], r["ev_under"]) if pd.notna(r["ev_over"]) and pd.notna(r["ev_under"]) else 0,
        axis=1,
    )
    result_df = result_df.sort_values("abs_ev", ascending=False).drop(columns=["abs_ev"])
    return result_df.reset_index(drop=True)


def apply_lineup_lock_gate(
    prop_edges: pd.DataFrame,
    upcoming: pd.DataFrame,
    lock_minutes: int = 30,
    enforce: bool = False,
) -> pd.DataFrame:
    """Attach lineup-lock timing context and optionally suppress pre-lock signals.

    A signal is lineup-lock-eligible when the game is scheduled and tip-off is within
    `lock_minutes` from now.
    """
    if prop_edges.empty:
        return prop_edges

    out = prop_edges.copy()
    up = upcoming.copy()
    if up.empty:
        out["game_start_utc"] = pd.NaT
        out["status"] = ""
        out["minutes_to_tip"] = np.nan
        out["lineup_lock_ok"] = False
        return out

    up["game_date_est"] = up["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    up["game_start_utc"] = pd.to_datetime(up["game_start_utc"], utc=True, errors="coerce")
    out["game_date_est"] = out["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)

    key_cols = ["game_date_est", "home_team", "away_team"]
    up_small = up[key_cols + ["game_start_utc", "status"]].drop_duplicates(key_cols)
    out = out.merge(up_small, on=key_cols, how="left")

    now_utc = pd.Timestamp.now(tz="UTC")
    out["minutes_to_tip"] = (
        (out["game_start_utc"] - now_utc).dt.total_seconds() / 60.0
    )
    status_upper = out["status"].fillna("").astype(str).str.upper()
    status_sched = status_upper.str.contains("SCHEDULED|PRE", regex=True)
    out["lineup_lock_ok"] = (
        status_sched
        & out["minutes_to_tip"].notna()
        & (out["minutes_to_tip"] >= 0)
        & (out["minutes_to_tip"] <= float(lock_minutes))
    )
    if "lineup_confirmed" in out.columns:
        out["lineup_confirmed"] = out["lineup_confirmed"].fillna(0).astype(int)
        out["lineup_lock_ok"] = out["lineup_lock_ok"] & out["lineup_confirmed"].eq(1)

    if enforce:
        mask = (out["signal"] != "NO BET") & (~out["lineup_lock_ok"])
        out.loc[mask, "signal"] = "NO BET"
        out.loc[mask, "confidence"] = "WAIT_CONFIRMED_STARTERS"

    return out


def run_market_line_backtest(
    player_df: pd.DataFrame,
    test_frac: float = 0.2,
    bet_size: float = 100.0,
    max_dates: int = 30,
    fetch_missing_lines: bool = False,
) -> dict[str, Any]:
    """Backtest prop signals against actual cached market lines (not synthetic lines)."""
    player_df = player_df.sort_values("game_time_utc").reset_index(drop=True)
    cut = int(len(player_df) * (1.0 - test_frac))
    train = player_df.iloc[:cut].copy()
    test = player_df.iloc[cut:].copy()
    print(f"\n  Market-line backtest split: {len(train)} train, {len(test)} test", flush=True)

    targets = [t for t in (list(PROP_TARGETS) + ["fg3m"]) if t in player_df.columns]

    pred_base_cols = ["game_date_est", "home_team", "away_team", "team", "opp", "player_name", "player_id"]
    pred_df = test[[c for c in pred_base_cols if c in test.columns]].copy()
    if "game_date_est" not in pred_df.columns:
        if "game_time_utc" in test.columns:
            pred_df["game_date_est"] = pd.to_datetime(
                test["game_time_utc"], utc=True, errors="coerce"
            ).dt.strftime("%Y%m%d")
        else:
            pred_df["game_date_est"] = ""
    pred_df["game_date_est"] = pred_df["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)

    residual_stds: dict[str, float] = {}
    model_perf: dict[str, Any] = {}
    for target in targets:
        train_valid = train.dropna(subset=[target]).copy()
        test_valid = test.dropna(subset=[target]).copy()
        if train_valid.empty or test_valid.empty:
            continue
        feats = filter_features(get_feature_list(target), train_valid)
        if not feats:
            continue
        try:
            imp, model, used_feats = train_prop_model(train_valid, get_feature_list(target), target)
        except ValueError:
            continue

        train_preds = predict_prop(imp, model, used_feats, train_valid)
        test_preds = predict_prop(imp, model, used_feats, test_valid)
        residual_stds[target] = float(np.std(train_valid[target].to_numpy(dtype=float) - train_preds))

        pred_col = f"pred_{target}"
        pred_df.loc[test_valid.index, pred_col] = test_preds

        mae = float(mean_absolute_error(test_valid[target].to_numpy(dtype=float), test_preds))
        rmse = float(math.sqrt(mean_squared_error(test_valid[target].to_numpy(dtype=float), test_preds)))
        r2 = float(r2_score(test_valid[target].to_numpy(dtype=float), test_preds))
        model_perf[target] = {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 3), "n_test": len(test_valid)}

    if not residual_stds:
        print("  No prop models could be trained for market-line backtest.", flush=True)
        return {"status": "no_models"}

    dates = sorted([d for d in pred_df["game_date_est"].dropna().astype(str).unique() if len(d) == 8])
    if max_dates > 0 and len(dates) > max_dates:
        dates = dates[-max_dates:]
    print(f"  Loading market prop lines for {len(dates)} test dates (max_dates={max_dates})...", flush=True)
    prop_lines = load_prop_lines_for_dates(dates, fetch_missing=fetch_missing_lines)
    if prop_lines.empty:
        print("  No market prop lines available for selected backtest dates.", flush=True)
        return {"status": "no_prop_lines", "n_dates": len(dates), "model_perf": model_perf}

    edges = compute_prop_edges(pred_df, prop_lines, residual_stds)
    if edges.empty:
        print("  No model/line matches found for market-line backtest.", flush=True)
        return {"status": "no_matches", "n_lines": len(prop_lines), "model_perf": model_perf}

    sig = edges[edges["signal"] != "NO BET"].copy()
    if sig.empty:
        print("  No actionable signals on matched market lines.", flush=True)
        return {"status": "no_signals", "n_lines": len(prop_lines), "n_matches": len(edges), "model_perf": model_perf}

    # Build lookup for actual outcomes
    base_actual_cols = ["team", "player_name"] + [t for t in targets if t in test.columns]
    actual = test[[c for c in base_actual_cols if c in test.columns]].copy()
    if "game_date_est" in test.columns:
        actual["game_date_est"] = (
            test["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
        )
    elif "game_time_utc" in test.columns:
        actual["game_date_est"] = pd.to_datetime(
            test["game_time_utc"], utc=True, errors="coerce"
        ).dt.strftime("%Y%m%d")
    else:
        actual["game_date_est"] = ""
    if "team" not in actual.columns or "player_name" not in actual.columns:
        return {
            "status": "missing_required_actual_columns",
            "required": ["team", "player_name"],
            "available": list(test.columns),
            "model_perf": model_perf,
        }
    actual["game_date_est"] = actual["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    actual["player_name_norm"] = actual["player_name"].map(normalize_player_name)

    eval_rows: list[dict[str, Any]] = []
    unmatched = 0
    for _, r in sig.iterrows():
        date = str(r.get("game_date_est", ""))
        team = str(r.get("team", ""))
        stat = str(r.get("stat_type", ""))
        side = str(r.get("signal", ""))
        if stat not in actual.columns:
            continue
        pname_norm = normalize_player_name(r.get("player_name", ""))
        m = (
            (actual["game_date_est"] == date)
            & (actual["team"] == team)
            & (actual["player_name_norm"] == pname_norm)
        )
        matches = actual[m]
        if matches.empty and pname_norm:
            m2 = (
                (actual["game_date_est"] == date)
                & (actual["team"] == team)
                & (actual["player_name_norm"].str.contains(re.escape(pname_norm), na=False))
            )
            matches = actual[m2]
        if matches.empty:
            unmatched += 1
            continue
        a = matches.iloc[0]
        actual_val = a.get(stat, np.nan)
        if pd.isna(actual_val):
            continue

        line = float(r["prop_line"])
        if abs(float(actual_val) - line) < 1e-9:
            result = "PUSH"
            hit = np.nan
            pnl = 0.0
        elif side == "OVER":
            result = "WIN" if float(actual_val) > line else "LOSS"
            hit = 1 if result == "WIN" else 0
            payout = _american_odds_to_decimal(r.get("over_odds", np.nan))
            payout = payout if pd.notna(payout) else VIG_FACTOR
            pnl = (payout * bet_size) if result == "WIN" else -bet_size
        else:
            result = "WIN" if float(actual_val) < line else "LOSS"
            hit = 1 if result == "WIN" else 0
            payout = _american_odds_to_decimal(r.get("under_odds", np.nan))
            payout = payout if pd.notna(payout) else VIG_FACTOR
            pnl = (payout * bet_size) if result == "WIN" else -bet_size

        open_line = r.get("open_line", np.nan)
        clv_line = np.nan
        if pd.notna(open_line):
            clv_line = (line - float(open_line)) if side == "OVER" else (float(open_line) - line)

        ev_chosen = r.get("ev_over", np.nan) if side == "OVER" else r.get("ev_under", np.nan)
        p_hit = r.get("p_over", np.nan) if side == "OVER" else r.get("p_under", np.nan)
        eval_rows.append(
            {
                "game_date_est": date,
                "player_name": r.get("player_name", ""),
                "team": team,
                "stat_type": stat,
                "signal": side,
                "prop_line": line,
                "open_line": open_line,
                "clv_line_pts": clv_line,
                "pred_value": r.get("pred_value", np.nan),
                "p_hit_model": p_hit,
                "ev_at_signal": ev_chosen,
                "actual_value": float(actual_val),
                "result": result,
                "hit": hit,
                "pnl": round(float(pnl), 2),
            }
        )

    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        print("  No signals could be matched to actual outcomes in market-line backtest.", flush=True)
        return {
            "status": "no_matched_signals",
            "n_signals": len(sig),
            "n_unmatched": unmatched,
            "model_perf": model_perf,
        }

    settled = eval_df[eval_df["result"] != "PUSH"].copy()
    n_bets = len(settled)
    n_wins = int(settled["hit"].fillna(0).sum())
    hit_rate = (n_wins / n_bets) if n_bets > 0 else np.nan
    total_pnl = float(settled["pnl"].sum())
    roi = (100.0 * total_pnl / (n_bets * bet_size)) if n_bets > 0 else np.nan
    avg_ev = float(settled["ev_at_signal"].dropna().mean()) if settled["ev_at_signal"].notna().any() else np.nan
    avg_clv = float(settled["clv_line_pts"].dropna().mean()) if settled["clv_line_pts"].notna().any() else np.nan
    calibration_drift_abs = np.nan
    brier_score = np.nan
    if "p_hit_model" in settled.columns and settled["p_hit_model"].notna().any():
        calib_df = settled.dropna(subset=["p_hit_model", "hit"]).copy()
        if not calib_df.empty:
            mean_pred = float(calib_df["p_hit_model"].mean())
            mean_real = float(calib_df["hit"].mean())
            calibration_drift_abs = abs(mean_real - mean_pred)
            brier_score = float(np.mean((calib_df["hit"] - calib_df["p_hit_model"]) ** 2))

    print("\n  --- Market-Line Backtest Summary ---", flush=True)
    print(
        f"  Signals={len(sig)}  Settled={n_bets}  Wins={n_wins}  "
        f"HitRate={hit_rate:.1%}  P/L=${total_pnl:+.0f}  ROI={roi:.1f}%",
        flush=True,
    )
    if pd.notna(avg_ev):
        print(f"  Avg EV at signal: {avg_ev:+.3f}", flush=True)
    if pd.notna(avg_clv):
        print(f"  Avg line CLV (pts): {avg_clv:+.3f}", flush=True)
    if pd.notna(calibration_drift_abs):
        print(f"  Calibration drift |hit - p(hit)|: {calibration_drift_abs:.3f}", flush=True)
    if pd.notna(brier_score):
        print(f"  Brier score: {brier_score:.4f}", flush=True)

    per_stat: dict[str, Any] = {}
    for st, g in settled.groupby("stat_type"):
        st_bets = len(g)
        st_wins = int(g["hit"].fillna(0).sum())
        st_pnl = float(g["pnl"].sum())
        st_hr = (st_wins / st_bets) if st_bets > 0 else np.nan
        st_roi = (100.0 * st_pnl / (st_bets * bet_size)) if st_bets > 0 else np.nan
        per_stat[st] = {
            "n_bets": st_bets,
            "wins": st_wins,
            "hit_rate": round(float(st_hr), 4) if pd.notna(st_hr) else np.nan,
            "pnl": round(st_pnl, 2),
            "roi_pct": round(float(st_roi), 2) if pd.notna(st_roi) else np.nan,
        }
        print(f"    {st:10s}: bets={st_bets:3d} hit={st_hr:.1%} pnl=${st_pnl:+.0f} roi={st_roi:.1f}%", flush=True)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    eval_csv = PREDICTIONS_DIR / "market_line_prop_backtest_signals.csv"
    eval_df.to_csv(eval_csv, index=False)
    out = {
        "status": "ok",
        "n_signals": int(len(sig)),
        "n_settled": int(n_bets),
        "wins": int(n_wins),
        "hit_rate": round(float(hit_rate), 4) if pd.notna(hit_rate) else np.nan,
        "pnl": round(total_pnl, 2),
        "roi_pct": round(float(roi), 2) if pd.notna(roi) else np.nan,
        "avg_ev_at_signal": round(float(avg_ev), 4) if pd.notna(avg_ev) else np.nan,
        "avg_clv_line_pts": round(float(avg_clv), 4) if pd.notna(avg_clv) else np.nan,
        "calibration_drift_abs": round(float(calibration_drift_abs), 4) if pd.notna(calibration_drift_abs) else np.nan,
        "brier_score": round(float(brier_score), 6) if pd.notna(brier_score) else np.nan,
        "n_unmatched_signals": int(unmatched),
        "per_stat": per_stat,
        "model_perf": model_perf,
        "eval_csv": str(eval_csv),
    }
    out_json = PREDICTIONS_DIR / "market_line_prop_backtest_summary.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"  Signal-level results saved to {eval_csv}", flush=True)
    print(f"  Summary saved to {out_json}", flush=True)
    return out


def run_box_advanced_ablation(
    player_df: pd.DataFrame,
    max_dates: int = 60,
    fetch_missing_lines: bool = False,
) -> dict[str, Any]:
    """Compare actionable market-line performance with and without BoxScoreAdvancedV3 features."""
    global USE_BOXSCORE_ADV_FEATURES
    original_flag = USE_BOXSCORE_ADV_FEATURES
    modes = [("baseline", False), ("box_advanced", True)]
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    print("\n--- BoxScoreAdvancedV3 Ablation ---", flush=True)
    print(f"  max_dates={max_dates}, actionable-only market-line metrics", flush=True)

    for name, enabled in modes:
        USE_BOXSCORE_ADV_FEATURES = enabled
        print(f"\n  Running mode={name} (USE_BOXSCORE_ADV_FEATURES={enabled})", flush=True)
        res = run_market_line_backtest(
            player_df,
            test_frac=0.2,
            bet_size=100.0,
            max_dates=max_dates,
            fetch_missing_lines=fetch_missing_lines,
        )
        if not isinstance(res, dict):
            res = {"status": "unknown"}
        details[name] = res
        row = {
            "mode": name,
            "status": str(res.get("status", "unknown")),
            "n_signals": int(res.get("n_signals", 0)) if str(res.get("n_signals", "")).strip() else 0,
            "n_settled": int(res.get("n_settled", 0)) if str(res.get("n_settled", "")).strip() else 0,
            "wins": int(res.get("wins", 0)) if str(res.get("wins", "")).strip() else 0,
            "hit_rate": res.get("hit_rate", np.nan),
            "roi_pct": res.get("roi_pct", np.nan),
            "pnl": res.get("pnl", np.nan),
            "avg_clv_line_pts": res.get("avg_clv_line_pts", np.nan),
            "avg_ev_at_signal": res.get("avg_ev_at_signal", np.nan),
            "calibration_drift_abs": res.get("calibration_drift_abs", np.nan),
            "brier_score": res.get("brier_score", np.nan),
        }
        rows.append(row)

    USE_BOXSCORE_ADV_FEATURES = original_flag
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return {"status": "no_results"}

    print("\n  Ablation summary:", flush=True)
    for _, r in out_df.iterrows():
        print(
            f"    {r['mode']:12s} status={r['status']:<12s} "
            f"signals={int(r['n_signals'])} settled={int(r['n_settled'])} "
            f"roi={r['roi_pct']} clv={r['avg_clv_line_pts']}",
            flush=True,
        )

    delta_row: dict[str, Any] = {"status": "ok", "details": details}
    try:
        base = out_df[out_df["mode"] == "baseline"].iloc[0]
        adv = out_df[out_df["mode"] == "box_advanced"].iloc[0]
        delta_row["delta_roi_pct"] = (
            float(adv["roi_pct"]) - float(base["roi_pct"])
            if pd.notna(adv["roi_pct"]) and pd.notna(base["roi_pct"])
            else np.nan
        )
        delta_row["delta_clv_pts"] = (
            float(adv["avg_clv_line_pts"]) - float(base["avg_clv_line_pts"])
            if pd.notna(adv["avg_clv_line_pts"]) and pd.notna(base["avg_clv_line_pts"])
            else np.nan
        )
        delta_row["delta_drift_abs"] = (
            float(adv["calibration_drift_abs"]) - float(base["calibration_drift_abs"])
            if pd.notna(adv["calibration_drift_abs"]) and pd.notna(base["calibration_drift_abs"])
            else np.nan
        )
    except Exception:
        pass

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = PREDICTIONS_DIR / f"box_adv_ablation_{ts}.csv"
    json_path = PREDICTIONS_DIR / f"box_adv_ablation_{ts}.json"
    out_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "summary": delta_row,
                "rows": out_df.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )
    print(f"  Ablation CSV saved to {csv_path}", flush=True)
    print(f"  Ablation JSON saved to {json_path}", flush=True)
    return {"status": "ok", "csv": str(csv_path), "json": str(json_path), **delta_row}


# ---------------------------------------------------------------------------
# Backtest: simple chronological split
# ---------------------------------------------------------------------------

def backtest_prop_edges(
    player_df: pd.DataFrame,
    test_frac: float = 0.2,
    bet_size: float = 100.0,
) -> dict[str, dict[str, Any]]:
    """Backtest prop edge signals against actual outcomes."""
    player_df = player_df.sort_values("game_time_utc").reset_index(drop=True)
    cut = int(len(player_df) * (1.0 - test_frac))
    train = player_df.iloc[:cut].copy()
    test = player_df.iloc[cut:].copy()

    print(f"\n  Prop edge backtest: {len(train)} train, {len(test)} test", flush=True)

    results: dict[str, dict[str, Any]] = {}
    all_targets = list(PROP_TARGETS) + (
        ["fg3m"] if "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 100 else []
    )

    for target in all_targets:
        features = get_feature_list(target)
        feats = filter_features(features, train)
        if not feats:
            continue
        try:
            imp, model, used_feats = train_prop_model(train, features, target)
        except ValueError:
            continue

        test_valid = test.dropna(subset=[target]).copy()
        if test_valid.empty:
            continue

        preds = predict_prop(imp, model, used_feats, test_valid)
        actual = test_valid[target].to_numpy(dtype=float)
        residual_std = float(np.std(actual - preds))

        # Use season averages as synthetic lines
        if f"pre_{target}_season" in test_valid.columns:
            synthetic_lines = test_valid[f"pre_{target}_season"].to_numpy(dtype=float)
        elif f"pre_{target}_avg10" in test_valid.columns:
            synthetic_lines = test_valid[f"pre_{target}_avg10"].to_numpy(dtype=float)
        else:
            continue

        valid_mask = ~np.isnan(synthetic_lines)
        if valid_mask.sum() < 50:
            continue

        preds_v = preds[valid_mask]
        actual_v = actual[valid_mask]
        lines_v = synthetic_lines[valid_mask]

        edges = preds_v - lines_v
        z_scores = (lines_v - preds_v) / max(residual_std, 0.01)
        p_overs = 1.0 - sp_stats.norm.cdf(z_scores)

        over_mask = p_overs > (BREAKEVEN_PROB + 0.03)
        under_mask = p_overs < (1.0 - BREAKEVEN_PROB - 0.03)

        n_over = int(over_mask.sum())
        n_under = int(under_mask.sum())
        over_hit = int((actual_v[over_mask] > lines_v[over_mask]).sum()) if n_over > 0 else 0
        under_hit = int((actual_v[under_mask] < lines_v[under_mask]).sum()) if n_under > 0 else 0

        over_wins = over_hit
        over_losses = n_over - over_hit
        under_wins = under_hit
        under_losses = n_under - under_hit
        total_profit = (
            (over_wins + under_wins) * bet_size * VIG_FACTOR
            - (over_losses + under_losses) * bet_size
        )
        n_bets = n_over + n_under

        results[target] = {
            "residual_std": round(residual_std, 2),
            "n_over_signals": n_over,
            "over_hit_rate": round(over_hit / n_over, 3) if n_over > 0 else np.nan,
            "n_under_signals": n_under,
            "under_hit_rate": round(under_hit / n_under, 3) if n_under > 0 else np.nan,
            "total_bets": n_bets,
            "total_wins": over_wins + under_wins,
            "total_win_rate": round((over_wins + under_wins) / n_bets, 3) if n_bets > 0 else np.nan,
            "profit_flat_100": round(total_profit, 2),
            "roi_pct": round(100 * total_profit / (n_bets * bet_size), 2) if n_bets > 0 else np.nan,
        }

        print(
            f"    {target:>10s}:  Overs={n_over} ({over_hit}/{n_over} hit="
            f"{results[target]['over_hit_rate']})  "
            f"Unders={n_under} ({under_hit}/{n_under} hit="
            f"{results[target]['under_hit_rate']})  "
            f"P/L=${total_profit:+.0f}  ROI={results[target]['roi_pct']}%",
            flush=True,
        )

    return results


def run_backtest(player_df: pd.DataFrame, test_frac: float = 0.2) -> dict[str, dict[str, float]]:
    """Run a chronological backtest on player props models."""
    player_df = player_df.sort_values("game_time_utc").reset_index(drop=True)
    cut = int(len(player_df) * (1.0 - test_frac))
    train = player_df.iloc[:cut].copy()
    test = player_df.iloc[cut:].copy()

    print(f"\n  Backtest split: {len(train)} train, {len(test)} test", flush=True)

    results: dict[str, dict[str, float]] = {}

    # --- Standard (single-stage) models ---
    print("\n  --- Single-Stage Models ---", flush=True)
    for target in PROP_TARGETS:
        features = get_feature_list(target)
        feats = filter_features(features, train)
        if not feats:
            print(f"    {target}: no features available, skipping", flush=True)
            continue
        try:
            imp, model, used_feats = train_prop_model(train, features, target)
        except ValueError as e:
            print(f"    {target}: {e}", flush=True)
            continue

        test_valid = test.dropna(subset=[target]).copy()
        if test_valid.empty:
            continue

        preds = predict_prop(imp, model, used_feats, test_valid)
        actual = test_valid[target].to_numpy(dtype=float)

        mae = mean_absolute_error(actual, preds)
        rmse = math.sqrt(mean_squared_error(actual, preds))
        r2 = r2_score(actual, preds)

        results[target] = {"mae": mae, "rmse": rmse, "r2": r2, "n_test": len(test_valid)}
        print(f"    {target:>10s}:  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}  (n={len(test_valid)})", flush=True)

        fi = pd.Series(model.feature_importances_, index=used_feats).sort_values(ascending=False)
        top_fi = fi.head(10)
        print(f"      Top features: {', '.join(f'{k}={v:.3f}' for k, v in top_fi.items())}", flush=True)

    # fg3m
    if "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 100:
        features = get_feature_list("fg3m")
        feats = filter_features(features, train)
        if feats:
            try:
                imp, model, used_feats = train_prop_model(train, features, "fg3m")
                test_valid = test.dropna(subset=["fg3m"]).copy()
                if not test_valid.empty:
                    preds = predict_prop(imp, model, used_feats, test_valid)
                    actual = test_valid["fg3m"].to_numpy(dtype=float)
                    mae = mean_absolute_error(actual, preds)
                    rmse = math.sqrt(mean_squared_error(actual, preds))
                    r2 = r2_score(actual, preds)
                    results["fg3m"] = {"mae": mae, "rmse": rmse, "r2": r2, "n_test": len(test_valid)}
                    print(f"    {'fg3m':>10s}:  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}  (n={len(test_valid)})", flush=True)
            except ValueError as e:
                print(f"    fg3m: {e}", flush=True)

    # --- Two-Stage Models ---
    print("\n  --- Two-Stage Models (Minutes -> Stats) ---", flush=True)
    two_stage_models = train_two_stage_models(train)
    two_stage_results: dict[str, dict[str, float]] = {}

    if two_stage_models:
        # Predict minutes on test set
        test_with_pred = test.copy()
        if "minutes" in two_stage_models:
            min_imp, min_model, min_feats = two_stage_models["minutes"]
            test_with_pred["pred_minutes"] = predict_prop(min_imp, min_model, min_feats, test_with_pred)

        for target in ["points", "rebounds", "assists", "fg3m"]:
            if target not in two_stage_models:
                continue
            imp, model, feats = two_stage_models[target]
            test_valid = test_with_pred.dropna(subset=[target]).copy()
            if test_valid.empty:
                continue

            preds = predict_prop(imp, model, feats, test_valid)
            actual = test_valid[target].to_numpy(dtype=float)

            mae = mean_absolute_error(actual, preds)
            rmse = math.sqrt(mean_squared_error(actual, preds))
            r2 = r2_score(actual, preds)

            two_stage_results[target] = {"mae": mae, "rmse": rmse, "r2": r2, "n_test": len(test_valid)}
            # Compare to single-stage
            ss_mae = results.get(target, {}).get("mae", np.nan)
            improve = f"  ({mae - ss_mae:+.2f} vs single)" if pd.notna(ss_mae) else ""
            print(f"    {target:>10s}:  MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}  (n={len(test_valid)}){improve}", flush=True)

    results["_two_stage"] = two_stage_results

    return results


# ---------------------------------------------------------------------------
# Walk-forward backtest with season-based folds
# ---------------------------------------------------------------------------

def run_walk_forward_backtest(
    player_df: pd.DataFrame,
    bet_size: float = 100.0,
) -> dict[str, Any]:
    """Walk-forward backtest with season-based folds.

    Uses seasons as fold boundaries. Train on all prior seasons, test on next.
    Evaluates: MAE, R2, hit rate vs synthetic lines, simulated P/L.
    """
    if "season" not in player_df.columns:
        print("  Error: season column required for walk-forward backtest.", flush=True)
        return {}

    seasons = sorted(player_df["season"].unique())
    if len(seasons) < 3:
        print(f"  Need at least 3 seasons for walk-forward. Have: {seasons}", flush=True)
        return {}

    print(f"\n  Walk-forward backtest across {len(seasons)} seasons: {seasons}", flush=True)

    all_targets = list(PROP_TARGETS) + (
        ["fg3m"] if "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 100 else []
    )

    fold_results: list[dict[str, Any]] = []

    # Use each season (except the first two) as test, all prior as train
    for i in range(2, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]
        train = player_df[player_df["season"].isin(train_seasons)].copy()
        test = player_df[player_df["season"] == test_season].copy()

        if len(train) < 500 or len(test) < 100:
            print(f"\n  Fold {i-1}: Train={train_seasons} -> Test={test_season} (skipped: insufficient data)", flush=True)
            continue

        print(f"\n  Fold {i-1}: Train={train_seasons} ({len(train)}) -> Test={test_season} ({len(test)})", flush=True)

        fold_metrics: dict[str, dict[str, Any]] = {}

        for target in all_targets:
            features = get_feature_list(target)
            feats = filter_features(features, train)
            if not feats:
                continue
            try:
                imp, model, used_feats = train_prop_model(train, features, target)
            except ValueError:
                continue

            test_valid = test.dropna(subset=[target]).copy()
            if test_valid.empty:
                continue

            preds = predict_prop(imp, model, used_feats, test_valid)
            actual = test_valid[target].to_numpy(dtype=float)
            residual_std = float(np.std(actual - preds))

            mae = mean_absolute_error(actual, preds)
            rmse = math.sqrt(mean_squared_error(actual, preds))
            r2 = r2_score(actual, preds)

            # Synthetic lines from season averages
            if f"pre_{target}_season" in test_valid.columns:
                synthetic_lines = test_valid[f"pre_{target}_season"].to_numpy(dtype=float)
            elif f"pre_{target}_avg10" in test_valid.columns:
                synthetic_lines = test_valid[f"pre_{target}_avg10"].to_numpy(dtype=float)
            else:
                fold_metrics[target] = {"mae": mae, "rmse": rmse, "r2": r2, "n_test": len(test_valid)}
                print(f"    {target:>10s}:  MAE={mae:.2f}  R2={r2:.3f}  (n={len(test_valid)})", flush=True)
                continue

            valid_mask = ~np.isnan(synthetic_lines)
            if valid_mask.sum() < 20:
                fold_metrics[target] = {"mae": mae, "rmse": rmse, "r2": r2, "n_test": len(test_valid)}
                print(f"    {target:>10s}:  MAE={mae:.2f}  R2={r2:.3f}  (n={len(test_valid)})", flush=True)
                continue

            preds_v = preds[valid_mask]
            actual_v = actual[valid_mask]
            lines_v = synthetic_lines[valid_mask]

            z_scores = (lines_v - preds_v) / max(residual_std, 0.01)
            p_overs = 1.0 - sp_stats.norm.cdf(z_scores)

            over_mask = p_overs > (BREAKEVEN_PROB + 0.03)
            under_mask = p_overs < (1.0 - BREAKEVEN_PROB - 0.03)

            n_over = int(over_mask.sum())
            n_under = int(under_mask.sum())
            over_hit = int((actual_v[over_mask] > lines_v[over_mask]).sum()) if n_over > 0 else 0
            under_hit = int((actual_v[under_mask] < lines_v[under_mask]).sum()) if n_under > 0 else 0

            n_bets = n_over + n_under
            total_profit = (
                (over_hit + under_hit) * bet_size * VIG_FACTOR
                - (n_bets - over_hit - under_hit) * bet_size
            )

            fold_metrics[target] = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "n_test": len(test_valid),
                "residual_std": residual_std,
                "n_over": n_over,
                "over_hit_rate": round(over_hit / n_over, 3) if n_over > 0 else np.nan,
                "n_under": n_under,
                "under_hit_rate": round(under_hit / n_under, 3) if n_under > 0 else np.nan,
                "n_bets": n_bets,
                "total_wins": over_hit + under_hit,
                "total_win_rate": round((over_hit + under_hit) / n_bets, 3) if n_bets > 0 else np.nan,
                "profit": round(total_profit, 2),
                "roi_pct": round(100 * total_profit / (n_bets * bet_size), 2) if n_bets > 0 else np.nan,
            }

            wr = fold_metrics[target].get("total_win_rate", np.nan)
            wr_s = f"{wr:.1%}" if pd.notna(wr) else "N/A"
            print(
                f"    {target:>10s}:  MAE={mae:.2f}  R2={r2:.3f}  "
                f"Bets={n_bets}  WR={wr_s}  "
                f"P/L=${total_profit:+.0f}  ROI={fold_metrics[target].get('roi_pct', 0):.1f}%",
                flush=True,
            )

        fold_results.append({
            "fold": i - 1,
            "train_seasons": train_seasons,
            "test_season": test_season,
            "n_train": len(train),
            "n_test": len(test),
            "metrics": fold_metrics,
        })

    # --- Aggregate results ---
    print(f"\n{'=' * 72}", flush=True)
    print("  WALK-FORWARD AGGREGATE RESULTS", flush=True)
    print(f"{'=' * 72}", flush=True)

    agg: dict[str, dict[str, list[float]]] = {}
    for fr in fold_results:
        for target, m in fr["metrics"].items():
            if target not in agg:
                agg[target] = {"mae": [], "r2": [], "profit": [], "n_bets": [], "win_rate": []}
            agg[target]["mae"].append(m.get("mae", np.nan))
            agg[target]["r2"].append(m.get("r2", np.nan))
            agg[target]["profit"].append(m.get("profit", 0))
            agg[target]["n_bets"].append(m.get("n_bets", 0))
            if pd.notna(m.get("total_win_rate")):
                agg[target]["win_rate"].append(m["total_win_rate"])

    total_profit_all = 0.0
    total_bets_all = 0
    for target, vals in agg.items():
        avg_mae = np.nanmean(vals["mae"])
        avg_r2 = np.nanmean(vals["r2"])
        sum_profit = sum(vals["profit"])
        sum_bets = sum(vals["n_bets"])
        avg_wr = np.nanmean(vals["win_rate"]) if vals["win_rate"] else np.nan
        roi = (100 * sum_profit / (sum_bets * bet_size)) if sum_bets > 0 else np.nan
        total_profit_all += sum_profit
        total_bets_all += int(sum_bets)

        wr_s = f"{avg_wr:.1%}" if pd.notna(avg_wr) else "N/A"
        roi_s = f"{roi:.1f}%" if pd.notna(roi) else "N/A"
        print(
            f"  {target:>10s}:  Avg MAE={avg_mae:.2f}  Avg R2={avg_r2:.3f}  "
            f"Total Bets={int(sum_bets)}  Avg WR={wr_s}  "
            f"P/L=${sum_profit:+.0f}  ROI={roi_s}",
            flush=True,
        )

    if total_bets_all > 0:
        overall_roi = 100 * total_profit_all / (total_bets_all * bet_size)
        print(
            f"\n  OVERALL: {total_bets_all} bets, ${total_profit_all:+.0f} P/L, "
            f"{overall_roi:.1f}% ROI",
            flush=True,
        )

        # GO / NO GO assessment
        print(f"\n  {'=' * 40}", flush=True)
        if overall_roi > 2.0 and total_bets_all >= 50:
            print("  ASSESSMENT: GO -- Positive ROI across walk-forward folds", flush=True)
        elif overall_roi > 0:
            print("  ASSESSMENT: CAUTIOUS GO -- Marginally positive, monitor closely", flush=True)
        else:
            print("  ASSESSMENT: NO GO -- Negative ROI in walk-forward validation", flush=True)
        print(f"  {'=' * 40}", flush=True)

    # Save results
    wf_out = PREDICTIONS_DIR / "walk_forward_props_results.json"
    wf_data = {
        "folds": fold_results,
        "aggregate": {
            target: {
                "avg_mae": float(np.nanmean(v["mae"])),
                "avg_r2": float(np.nanmean(v["r2"])),
                "total_profit": sum(v["profit"]),
                "total_bets": int(sum(v["n_bets"])),
            }
            for target, v in agg.items()
        },
        "total_profit": total_profit_all,
        "total_bets": total_bets_all,
    }
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    wf_out.write_text(json.dumps(wf_data, indent=2, default=str))
    print(f"\n  Walk-forward results saved to {wf_out}", flush=True)

    return wf_data


# ---------------------------------------------------------------------------
# Upcoming games prediction
# ---------------------------------------------------------------------------

def fetch_upcoming_schedule(target_date: str, days: int = 1) -> pd.DataFrame:
    """Fetch upcoming NBA games from the ESPN scoreboard."""
    rows: list[dict[str, Any]] = []
    base = datetime.strptime(target_date, "%Y%m%d")
    for offset in range(days):
        d = base + timedelta(days=offset)
        date_str = d.strftime("%Y%m%d")
        try:
            payload = fetch_json(ESPN_SCOREBOARD_URL.format(yyyymmdd=date_str), timeout=20, retries=3)
        except Exception as exc:
            print(f"  Warning: could not fetch scoreboard for {date_str}: {exc}", flush=True)
            continue

        for event in payload.get("events", []):
            comp = (event.get("competitions") or [{}])[0]
            competitors = comp.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            status_name = event.get("status", {}).get("type", {}).get("name", "")

            # Fetch Vegas odds (game total + spread) for this event
            implied_total = np.nan
            implied_spread = np.nan
            event_id = str(event.get("id", ""))
            if event_id:
                try:
                    odds_url = (
                        f"https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
                        f"/events/{event_id}/competitions/{event_id}/odds"
                    )
                    odds_payload = fetch_json(odds_url, timeout=15, retries=2)
                    for item in odds_payload.get("items", []):
                        ou = item.get("overUnder")
                        sp = item.get("spread")
                        if ou is not None and pd.notna(ou):
                            implied_total = float(ou)
                        if sp is not None and pd.notna(sp):
                            implied_spread = float(sp)
                        if pd.notna(implied_total):
                            break  # use first provider with data
                except Exception:
                    pass  # odds not available pre-game, leave NaN

            rows.append({
                "espn_event_id": event_id,
                "game_date_est": date_str,
                "home_team": normalize_espn_abbr(home.get("team", {}).get("abbreviation", "")),
                "away_team": normalize_espn_abbr(away.get("team", {}).get("abbreviation", "")),
                "game_name": event.get("name"),
                "game_start_utc": event.get("date"),
                "status": status_name,
                "implied_total": implied_total,
                "implied_spread": implied_spread,
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_player_predictions(
    player_df: pd.DataFrame,
    team_games: pd.DataFrame,
    player_games_raw: pd.DataFrame,
    upcoming: pd.DataFrame,
    min_games: int = DEFAULT_MIN_GAMES,
    injury_status_map: dict[str, dict[str, Any]] | None = None,
    confirmed_starters: dict[tuple[str, str, str], dict[str, set[str]]] | None = None,
    two_stage_models: dict[str, Any] | None = None,
    single_models: dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]] | None = None,
) -> pd.DataFrame:
    """Build predictions for players in upcoming games using two-stage modeling."""
    if two_stage_models is None or single_models is None:
        print("  Training two-stage player prop models...", flush=True)
        two_stage_models, single_models = train_prediction_models(player_df)

    all_models = {**single_models, **two_stage_models}
    if not all_models:
        print("  No models trained. Cannot generate predictions.", flush=True)
        return pd.DataFrame()

    for t, v in all_models.items():
        if t.startswith("_"):
            continue  # skip internal keys like _residual
        _, _, f = v
        print(f"    {t}: {len(f)} features", flush=True)

    # For each upcoming game, get players from each team
    print("  Building prediction rows for upcoming players...", flush=True)
    pred_rows: list[dict[str, Any]] = []
    injury_status_map = injury_status_map or {}
    confirmed_starters = confirmed_starters or {}

    pg = player_games_raw.copy()
    pg = pg[pg["played"] == 1].sort_values(["team", "player_id", "game_time_utc", "game_id"])

    # Merge extended stats (fouls, scoring breakdown, OT) for prediction-time rolling averages
    ext = load_extended_player_stats()
    _new_ext_cols = ["fouls_personal", "fouls_drawn", "pts_in_paint", "pts_fast_break",
                     "n_ot_periods", "fg3a", "orb", "drb"]
    if not ext.empty:
        # Only merge columns not already present
        ext_merge_cols = ["game_id", "team", "player_id"] + [c for c in _new_ext_cols if c in ext.columns and c not in pg.columns]
        if len(ext_merge_cols) > 3:
            pg = pg.merge(ext[ext_merge_cols].drop_duplicates(subset=["game_id", "team", "player_id"]),
                          on=["game_id", "team", "player_id"], how="left")
    for c in _new_ext_cols:
        if c not in pg.columns:
            pg[c] = 0.0 if c == "n_ot_periods" else np.nan
    pg["n_ot_periods"] = pg["n_ot_periods"].fillna(0).astype(int)

    # Merge BoxScoreAdvancedV3 stats from cache for prediction-time rolling features.
    adv_cols = [
        "adv_usage_pct",
        "adv_pace",
        "adv_possessions",
        "adv_off_rating",
        "adv_ast_pct",
        "adv_reb_pct",
        "adv_ts_pct",
    ]
    adv = load_boxscore_advanced_stats(game_ids=pg["game_id"].astype(str).unique().tolist(), fetch_missing=False)
    if not adv.empty:
        pg = pg.merge(
            adv[["game_id", "team", "player_id"] + [c for c in adv_cols if c in adv.columns]].drop_duplicates(
                subset=["game_id", "team", "player_id"]
            ),
            on=["game_id", "team", "player_id"],
            how="left",
        )
    for c in adv_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    # Merge team-level injury proxy context onto player history for role-shift features.
    inj_hist_cols = ["game_id", "team"]
    for c in ["injury_proxy_missing_points5", "injury_proxy_missing_minutes5"]:
        if c in team_games.columns:
            inj_hist_cols.append(c)
    if len(inj_hist_cols) > 2:
        inj_hist = team_games[inj_hist_cols].copy()
        inj_hist = inj_hist.rename(columns={c: f"team_{c}" for c in inj_hist_cols if c not in ["game_id", "team"]})
        pg = pg.merge(inj_hist, on=["game_id", "team"], how="left")
    if "team_injury_proxy_missing_points5" not in pg.columns:
        pg["team_injury_proxy_missing_points5"] = 0.0

    # Determine each player's current team (team of their most recent game by date, not game_id)
    player_current_team = pg.sort_values("game_time_utc").groupby("player_id")["team"].last()

    for _, game in upcoming.iterrows():
        home_team = game["home_team"]
        away_team = game["away_team"]
        game_date = game.get("game_date_est", "")
        event_id = game.get("espn_event_id", "")
        lineup_key = (str(game_date), str(home_team), str(away_team))
        game_starters = confirmed_starters.get(lineup_key, {})
        lineup_confirmed = int(bool(game_starters))

        for team, opp, is_home in [(home_team, away_team, 1), (away_team, home_team, 0)]:
            # Only include players whose CURRENT team matches (handles mid-season trades)
            current_pids = player_current_team[player_current_team == team].index
            if current_pids.empty:
                continue

            # Use ALL of each player's games (across teams) for features,
            # but only predict for players currently on this team
            eligible_pids = []
            for pid in current_pids:
                all_player_games = pg[pg["player_id"] == pid]
                if len(all_player_games) >= min_games:
                    eligible_pids.append(pid)

            for pid in eligible_pids:
                # Use all games across all teams for rolling averages
                p_games = pg[pg["player_id"] == pid].sort_values("game_time_utc")
                if p_games.empty:
                    continue

                latest = p_games.iloc[-1]
                recent5 = p_games.tail(5)
                recent10 = p_games.tail(10)
                all_games = p_games

                # Build feature row
                row: dict[str, Any] = {
                    "game_date_est": game_date,
                    "espn_event_id": event_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "team": team,
                    "opp": opp,
                    "player_id": int(pid),
                    "player_name": latest.get("player_name", ""),
                    "player_is_home": is_home,
                    "player_game_num": len(p_games),
                    "lineup_confirmed": lineup_confirmed,
                }

                # Player rolling averages + EWM
                stat_cols = ["points", "rebounds", "assists", "minutes", "fg3m", "fta", "tov",
                             "fga", "fgm", "fg3a", "steals", "blocks", "orb", "drb",
                             "fouls_personal", "fouls_drawn", "pts_in_paint", "pts_fast_break",
                             "adv_usage_pct", "adv_pace", "adv_possessions", "adv_off_rating",
                             "adv_ast_pct", "adv_reb_pct", "adv_ts_pct"]
                avg3_stats = {"points", "rebounds", "assists", "minutes", "fg3m",
                              "fg3a", "fouls_drawn", "adv_usage_pct"}
                recent3 = p_games.tail(3)

                # OT regulation adjustment for prediction-time rolling averages
                _counting_pred = {"points", "rebounds", "assists", "fg3m", "fta", "tov",
                                  "fga", "fgm", "fg3a", "steals", "blocks", "orb", "drb",
                                  "fouls_personal", "fouls_drawn", "pts_in_paint", "pts_fast_break"}
                ot_periods = p_games["n_ot_periods"].fillna(0).astype(int)
                _reg_factor = 48.0 / (48.0 + 5.0 * ot_periods)

                for col in stat_cols:
                    if col not in p_games.columns:
                        row[f"pre_{col}_avg5"] = np.nan
                        row[f"pre_{col}_avg10"] = np.nan
                        row[f"pre_{col}_season"] = np.nan
                        row[f"pre_{col}_ewm5"] = np.nan
                        row[f"pre_{col}_ewm10"] = np.nan
                        if col in avg3_stats:
                            row[f"pre_{col}_avg3"] = np.nan
                        continue
                    # Apply regulation adjustment for counting stats
                    if col in _counting_pred and col != "minutes":
                        vals_adj = (p_games[col] * _reg_factor).dropna()
                        r3_vals = recent3[col] * _reg_factor.iloc[-3:] if len(recent3) > 0 else recent3[col]
                        r5_vals = recent5[col] * _reg_factor.iloc[-5:] if len(recent5) > 0 else recent5[col]
                        r10_vals = recent10[col] * _reg_factor.iloc[-10:] if len(recent10) > 0 else recent10[col]
                        all_vals = p_games[col] * _reg_factor
                    else:
                        vals_adj = p_games[col].dropna()
                        r3_vals = recent3[col]
                        r5_vals = recent5[col]
                        r10_vals = recent10[col]
                        all_vals = p_games[col]

                    if col in avg3_stats:
                        row[f"pre_{col}_avg3"] = float(r3_vals.mean()) if not r3_vals.isna().all() else np.nan
                    row[f"pre_{col}_avg5"] = float(r5_vals.mean()) if not r5_vals.isna().all() else np.nan
                    row[f"pre_{col}_avg10"] = float(r10_vals.mean()) if not r10_vals.isna().all() else np.nan
                    row[f"pre_{col}_season"] = float(all_vals.mean()) if not all_vals.isna().all() else np.nan

                    # EWM (compute on all games, shifted)
                    if len(vals_adj) >= 2:
                        ewm5 = vals_adj.ewm(span=5, min_periods=1).mean().iloc[-1]
                        ewm10 = vals_adj.ewm(span=10, min_periods=1).mean().iloc[-1]
                        row[f"pre_{col}_ewm5"] = float(ewm5)
                        row[f"pre_{col}_ewm10"] = float(ewm10)
                    else:
                        row[f"pre_{col}_ewm5"] = row[f"pre_{col}_season"]
                        row[f"pre_{col}_ewm10"] = row[f"pre_{col}_season"]

                # Per-minute rates
                safe_min_avg5 = max(_nan_or(row.get("pre_minutes_avg5"), 1), 1)
                safe_min_avg3 = max(_nan_or(row.get("pre_minutes_avg3"), 1), 1)
                for stat, rate_prefix in [("points", "pts"), ("rebounds", "reb"),
                                          ("assists", "ast"), ("fg3m", "fg3m"),
                                          ("fga", "fga"), ("fg3a", "fg3a"),
                                          ("fta", "fta"), ("fouls_drawn", "fouls_drawn")]:
                    avg5 = row.get(f"pre_{stat}_avg5")
                    avg3 = row.get(f"pre_{stat}_avg3")
                    if pd.notna(avg3):
                        row[f"pre_{rate_prefix}_per_min_avg3"] = avg3 / safe_min_avg3
                    if pd.notna(avg5):
                        row[f"pre_{rate_prefix}_per_min_avg5"] = avg5 / safe_min_avg5
                        row[f"pre_{rate_prefix}_per_min_ewm5"] = row.get(f"pre_{stat}_ewm5", avg5) / safe_min_avg5
                        row[f"pre_{rate_prefix}_per_min_season"] = _nan_or(row.get(f"pre_{stat}_season"), avg5) / max(_nan_or(row.get("pre_minutes_season"), 1), 1)

                # Variance features
                for col in ["points", "rebounds", "assists", "minutes"]:
                    if col in p_games.columns and len(recent10) >= 3:
                        row[f"pre_{col}_std10"] = float(recent10[col].std()) if not recent10[col].isna().all() else np.nan
                    else:
                        row[f"pre_{col}_std10"] = np.nan

                # Starter rate
                row["pre_starter_rate"] = float(recent10["starter"].mean()) if "starter" in recent10.columns else np.nan

                # Minutes trend
                m5 = row.get("pre_minutes_avg5")
                m10 = row.get("pre_minutes_avg10")
                row["pre_minutes_trend"] = (m5 - m10) if m5 is not None and m10 is not None and not (pd.isna(m5) or pd.isna(m10)) else np.nan

                # Points trend
                p5 = row.get("pre_points_avg5")
                p10 = row.get("pre_points_avg10")
                row["pre_points_trend"] = (p5 - p10) if p5 is not None and p10 is not None and not (pd.isna(p5) or pd.isna(p10)) else np.nan

                # Usage proxy
                _pts = _nan_or(row.get("pre_points_avg5"), 0)
                _fta = _nan_or(row.get("pre_fta_avg5"), 0)
                _tov = _nan_or(row.get("pre_tov_avg5"), 0)
                _min = max(_nan_or(row.get("pre_minutes_avg5"), 1), 1)
                row["pre_usage_proxy"] = (_pts + 0.44 * _fta + _tov) / _min

                # Venue splits
                for col in ["points", "rebounds", "assists", "minutes", "fg3m"]:
                    if col not in p_games.columns:
                        continue
                    home_games = p_games[p_games["is_home"] == 1]
                    away_games = p_games[p_games["is_home"] == 0]
                    row[f"pre_{col}_home_avg"] = float(home_games[col].mean()) if len(home_games) >= 3 and not home_games[col].isna().all() else np.nan
                    row[f"pre_{col}_away_avg"] = float(away_games[col].mean()) if len(away_games) >= 3 and not away_games[col].isna().all() else np.nan
                    h = row.get(f"pre_{col}_home_avg")
                    a = row.get(f"pre_{col}_away_avg")
                    row[f"pre_{col}_venue_diff"] = (h - a) if pd.notna(h) and pd.notna(a) else np.nan

                # Player days rest
                last_game_time = latest.get("game_time_utc")
                if pd.notna(last_game_time):
                    now = pd.Timestamp.now(tz="UTC")
                    row["player_days_rest"] = (now - last_game_time).total_seconds() / 86400.0
                else:
                    row["player_days_rest"] = np.nan

                # B2B and schedule fatigue
                rest_val = row.get("player_days_rest", np.nan)
                row["is_b2b"] = float(pd.notna(rest_val) and rest_val <= 1.5)
                # 3-in-4: check 2nd-to-last game time
                if len(p_games) >= 2:
                    prev2_time = p_games.iloc[-2].get("game_time_utc")
                    if pd.notna(prev2_time):
                        days_for_3 = (pd.Timestamp.now(tz="UTC") - prev2_time).total_seconds() / 86400.0
                        row["is_3_in_4"] = float(days_for_3 <= 4.0)
                    else:
                        row["is_3_in_4"] = 0.0
                else:
                    row["is_3_in_4"] = 0.0
                row["b2b_x_minutes"] = row["is_b2b"] * _nan_or(row.get("pre_minutes_avg5"), 0)
                row["b2b_x_starter"] = row["is_b2b"] * _nan_or(row.get("pre_starter_rate"), 0)

                # Team context
                team_tg = team_games[team_games["team"] == team]
                if not team_tg.empty:
                    latest_team = team_tg.iloc[-1]
                    for col in ["possessions", "off_rating", "def_rating", "net_rating", "efg"]:
                        for window in ["avg5", "avg10", "season"]:
                            feat = f"pre_{col}_{window}"
                            if feat in latest_team.index:
                                row[f"team_{feat}"] = latest_team[feat]

                    for ic in ["injury_proxy_missing_minutes5", "injury_proxy_missing_points5",
                               "star_player_absent_flag", "active_count"]:
                        if ic in latest_team.index:
                            row[f"team_{ic}"] = latest_team[ic]

                # Opponent context
                opp_tg = team_games[team_games["team"] == opp]
                if not opp_tg.empty:
                    latest_opp = opp_tg.iloc[-1]
                    for col in ["possessions", "off_rating", "def_rating", "net_rating", "efg"]:
                        for window in ["avg5", "avg10", "season"]:
                            feat = f"pre_{col}_{window}"
                            if feat in latest_opp.index:
                                row[f"opp_{feat}"] = latest_opp[feat]

                # Matchup features
                t_pace = _nan_or(row.get("team_pre_possessions_avg5", 0), 0)
                o_pace = _nan_or(row.get("opp_pre_possessions_avg5", 0), 0)
                row["matchup_pace_avg"] = (t_pace + o_pace) / 2.0
                t_off = _nan_or(row.get("team_pre_off_rating_avg5", 0), 0)
                o_def = _nan_or(row.get("opp_pre_def_rating_avg5", 0), 0)
                row["matchup_off_vs_def"] = t_off - o_def

                # Pace differential vs player's recent games
                # Use team_pre_possessions_avg5 from recent games as proxy for game pace player has been in
                if "team_pre_possessions_avg5" in p_games.columns:
                    recent_pace = p_games["team_pre_possessions_avg5"].dropna()
                    if not recent_pace.empty:
                        player_recent_pace = float(recent_pace.tail(5).mean())
                        row["pace_diff_vs_recent"] = row["matchup_pace_avg"] - player_recent_pace
                    else:
                        row["pace_diff_vs_recent"] = 0.0
                else:
                    row["pace_diff_vs_recent"] = 0.0

                # Usage boost
                missing_pts = _nan_or(row.get("team_injury_proxy_missing_points5", 0), 0)
                usage = _nan_or(row.get("pre_usage_proxy", 0), 0)
                row["team_injury_pressure"] = missing_pts
                row["usage_boost_proxy"] = missing_pts * usage
                row["minutes_x_injury_pressure"] = _nan_or(row.get("pre_minutes_avg5"), 0) * missing_pts

                # Blowout risk
                t_net = _nan_or(row.get("team_pre_net_rating_avg5", 0), 0)
                o_net = _nan_or(row.get("opp_pre_net_rating_avg5", 0), 0)
                row["net_rating_diff"] = t_net - o_net
                row["blowout_risk"] = abs(t_net - o_net)
                row["pace_x_injury_pressure"] = row["matchup_pace_avg"] * missing_pts

                # Vegas game total / spread context (from upcoming schedule odds)
                row["implied_total"] = game.get("implied_total", np.nan)
                row["implied_spread"] = game.get("implied_spread", np.nan)
                total = row["implied_total"]
                spread = row["implied_spread"]
                if pd.notna(total) and pd.notna(spread):
                    if is_home:
                        row["implied_team_total"] = total / 2 - spread / 2
                    else:
                        row["implied_team_total"] = total / 2 + spread / 2
                else:
                    row["implied_team_total"] = np.nan

                # Blowout-adjusted minutes features (Task 4)
                if pd.notna(spread):
                    row["abs_spread"] = abs(spread)
                    starter_rate = _nan_or(row.get("pre_starter_rate", 0.5), 0.5)
                    row["spread_x_starter"] = abs(spread) * starter_rate
                    row["is_big_favorite"] = 1 if abs(spread) > 8 else 0
                else:
                    row["abs_spread"] = np.nan
                    row["spread_x_starter"] = np.nan
                    row["is_big_favorite"] = np.nan

                # Role-shift features: player's rolling response to teammate injury pressure.
                hist_pressure = p_games.get("team_injury_proxy_missing_points5", pd.Series(0.0, index=p_games.index)).fillna(0.0)
                for src_col, out_col in [
                    ("points", "role_pts_injury_beta20"),
                    ("rebounds", "role_reb_injury_beta20"),
                    ("assists", "role_ast_injury_beta20"),
                    ("minutes", "role_min_injury_beta20"),
                ]:
                    if src_col in p_games.columns and len(p_games) >= 7:
                        beta_series = _rolling_beta_shifted(
                            hist_pressure.to_numpy(dtype=float),
                            p_games[src_col].to_numpy(dtype=float),
                            window=20,
                            min_periods=6,
                        )
                        row[out_col] = float(beta_series[-1]) if len(beta_series) else np.nan
                    else:
                        row[out_col] = np.nan

                # orb/drb rolling (regulation-adjusted via stat_cols loop above)

                # Rebound opportunity / center-depth context (from latest historical feature row)
                reb_feature_cols = [
                    "pre_orb_avg10",
                    "pre_drb_avg10",
                    "pre_team_rebounds_avg10",
                    "pre_team_orb_avg10",
                    "pre_team_drb_avg10",
                    "pre_team_missed_fg_avg10",
                    "pre_opp_missed_fg_avg10",
                    "pre_total_missed_fg_avg10",
                    "pre_team_reb_share_avg10",
                    "pre_team_orb_share_avg10",
                    "pre_team_drb_share_avg10",
                    "pre_player_reb_share_avg10",
                    "pre_player_orb_share_avg10",
                    "pre_player_drb_share_avg10",
                    "pre_player_reb_opp_proxy_avg10",
                    "pre_other_center_minutes_avg10",
                    "pre_center_depth_risk",
                ]
                for feat_col in reb_feature_cols:
                    if feat_col in p_games.columns:
                        vals = p_games[feat_col].dropna()
                        row[feat_col] = float(vals.iloc[-1]) if not vals.empty else np.nan
                    else:
                        row[feat_col] = np.nan

                # Opponent positional defense features
                # Look up opponent's rolling avg of stats allowed to this player's position group
                pos_map = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}
                player_pos = latest.get("position", "F")
                player_pos_group = pos_map.get(str(player_pos), "F")
                row["pos_group"] = player_pos_group

                # Get the most recent opp defense profile for this pos group from player_games history
                opp_games_as_defender = pg[(pg["opp"] == opp) & (pg["pos_group"] == player_pos_group)] if "pos_group" in pg.columns else pd.DataFrame()
                if not opp_games_as_defender.empty and "opp_pts_allowed_to_pos_avg10" in pg.columns:
                    opp_latest_def = opp_games_as_defender.sort_values("game_time_utc")
                    for stat in ["pts", "reb", "ast"]:
                        feat_col = f"opp_{stat}_allowed_to_pos_avg10"
                        if feat_col in opp_latest_def.columns:
                            last_val = opp_latest_def[feat_col].dropna()
                            row[feat_col] = float(last_val.iloc[-1]) if not last_val.empty else np.nan
                        else:
                            row[feat_col] = np.nan
                else:
                    row["opp_pts_allowed_to_pos_avg10"] = np.nan
                    row["opp_reb_allowed_to_pos_avg10"] = np.nan
                    row["opp_ast_allowed_to_pos_avg10"] = np.nan

                # Player career / age-rest interactions
                total_career_games = len(pg[pg["player_id"] == pid])
                row["player_career_games"] = total_career_games
                row["is_veteran"] = 1 if total_career_games > 200 else 0
                rest = row.get("player_days_rest", np.nan)
                if pd.notna(rest):
                    row["veteran_rest_effect"] = row["is_veteran"] * rest
                    row["career_x_rest"] = total_career_games * rest
                else:
                    row["veteran_rest_effect"] = np.nan
                    row["career_x_rest"] = np.nan

                # Referee crew features (NaN for upcoming games — refs not known pre-game)
                row["ref_crew_avg_total"] = np.nan
                row["ref_crew_avg_fta"] = np.nan
                row["ref_crew_avg_fouls"] = np.nan
                row["ref_crew_avg_pace"] = np.nan
                row["ref_crew_total_over_league_avg"] = np.nan
                row["ref_crew_pace_over_league_avg"] = np.nan
                row["ref_foul_rate"] = np.nan
                row["ref_pace_factor"] = np.nan
                # OT flag (always 0 for predictions — we predict regulation stats)
                row["is_ot"] = 0.0

                # Injury report status (real-time) and confirmed starter flag (lineup refresh).
                inj = injury_status_map.get(_injury_key(team, row.get("player_name", "")), {})
                avail_prob = _to_float(inj.get("availability_prob"))
                status = str(inj.get("status", "")).strip().lower()
                row["injury_availability_prob"] = float(np.clip(avail_prob, 0.0, 1.0)) if pd.notna(avail_prob) else np.nan
                row["injury_unavailability_prob"] = 1.0 - row["injury_availability_prob"] if pd.notna(row["injury_availability_prob"]) else np.nan
                row["injury_status"] = status
                row["injury_is_out"] = int(status in INJURY_REMOVE_STATUSES)
                row["injury_is_doubtful"] = int(status in INJURY_HIGH_RISK_STATUSES)
                row["injury_is_questionable"] = int(status == "questionable")
                row["injury_is_probable"] = int(status == "probable")

                team_starters = game_starters.get(team, set())
                if lineup_confirmed:
                    row["confirmed_starter"] = float(normalize_player_name(row.get("player_name", "")) in team_starters)
                else:
                    row["confirmed_starter"] = np.nan

                pred_rows.append(row)

    if not pred_rows:
        print("  No eligible players found for upcoming games.", flush=True)
        return pd.DataFrame()

    pred_df = pd.DataFrame(pred_rows)

    # Generate two-stage predictions
    if two_stage_models:
        pred_df = predict_two_stage(two_stage_models, pred_df)
    else:
        # Single-stage fallback
        for target, (imp, model, used_feats) in single_models.items():
            pred_df[f"pred_{target}"] = predict_prop(imp, model, used_feats, pred_df)

    # Fill any missing predictions from single-stage models
    for target, (imp, model, used_feats) in single_models.items():
        pred_col = f"pred_{target}"
        if pred_col not in pred_df.columns:
            pred_df[pred_col] = predict_prop(imp, model, used_feats, pred_df)

    return pred_df


def format_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Format and sort prediction output."""
    if pred_df.empty:
        return pred_df

    out_cols = [
        "game_date_est", "home_team", "away_team", "team", "opp",
        "player_name", "player_id", "player_is_home",
    ]

    for target in PROP_TARGETS + ["fg3m"]:
        pred_col = f"pred_{target}"
        if pred_col in pred_df.columns:
            out_cols.append(pred_col)

    context_cols = [
        "pre_minutes_avg5", "pre_points_avg5", "pre_rebounds_avg5",
        "pre_assists_avg5", "pre_fg3m_avg5", "pre_starter_rate",
        "player_game_num",
    ]
    for c in context_cols:
        if c in pred_df.columns:
            out_cols.append(c)

    out_cols = [c for c in out_cols if c in pred_df.columns]
    result = pred_df[out_cols].copy()

    for c in result.columns:
        if c.startswith("pred_") or c.startswith("pre_"):
            result[c] = result[c].round(1)

    if "pred_minutes" in result.columns:
        result = result.sort_values(
            ["game_date_est", "home_team", "away_team", "team", "pred_minutes"],
            ascending=[True, True, True, True, False],
        ).reset_index(drop=True)
    else:
        result = result.sort_values(
            ["game_date_est", "home_team", "away_team", "team"],
        ).reset_index(drop=True)

    return result


def print_prop_edge_summary(prop_edges: pd.DataFrame) -> None:
    """Print a formatted summary of prop edge signals."""
    if prop_edges.empty:
        print("\n  No prop edge data available.", flush=True)
        return

    n_signals = int((prop_edges["signal"] != "NO BET").sum())
    n_best_bets = int((prop_edges["confidence"] == "BEST BET").sum())
    n_positive_ev = int(
        ((prop_edges["ev_over"] > 0) | (prop_edges["ev_under"] > 0)).sum()
    )

    print(f"\n{'=' * 80}", flush=True)
    print(f"  PLAYER PROP SIGNALS", flush=True)
    print(f"{'=' * 80}", flush=True)
    policy_desc = "points-only" if SIGNAL_POINTS_ONLY else "all-stats"
    print(
        f"  Signal policy: {policy_desc}, min_pred_minutes={MIN_SIGNAL_PRED_MINUTES:.1f}, "
        f"min_pre_minutes_avg10={MIN_SIGNAL_PRE_MINUTES_AVG10:.1f}, mode={ACTIVE_SIGNAL_POLICY_MODE}",
        flush=True,
    )
    print(f"  Total props analyzed: {len(prop_edges)}", flush=True)
    print(f"  Signals generated: {n_signals}", flush=True)
    print(f"  Best bets: {n_best_bets}", flush=True)
    print(f"  Positive EV props: {n_positive_ev}", flush=True)
    if "signal_blocked_reason" in prop_edges.columns:
        blocked_minutes = int((prop_edges["signal_blocked_reason"] == "minutes_gate").sum())
        blocked_non_points = int((prop_edges["signal_blocked_reason"] == "non_points_stat_filtered").sum())
        if blocked_minutes or blocked_non_points:
            print(
                f"  Blocked by gates: minutes={blocked_minutes}, non_points={blocked_non_points}",
                flush=True,
            )

    # Sort all signals by EV and cap at MAX_SIGNALS_PER_DAY
    all_signals = prop_edges[prop_edges["signal"] != "NO BET"].copy()
    all_signals["_best_ev"] = all_signals.apply(
        lambda r: r["ev_over"] if r["signal"] == "OVER" else r["ev_under"], axis=1
    )
    all_signals = all_signals.sort_values("_best_ev", ascending=False)

    # Cap at 2 signals per player to reduce correlation exposure
    signal_counts: dict[str, int] = {}
    keep_mask: list[bool] = []
    for idx, sig_row in all_signals.iterrows():
        player = sig_row["player_name"]
        count = signal_counts.get(player, 0)
        if count < 2:
            keep_mask.append(True)
            signal_counts[player] = count + 1
        else:
            keep_mask.append(False)
    all_signals = all_signals[keep_mask]
    n_correlated = int(all_signals.get("correlated", pd.Series(False)).sum()) if "correlated" in all_signals.columns else 0
    if n_correlated > 0:
        print(f"  Correlated signals (same player, same direction): {n_correlated}", flush=True)

    if n_best_bets > 0:
        best = all_signals[all_signals["confidence"] == "BEST BET"].head(MAX_SIGNALS_PER_DAY)
        print(f"\n  --- BEST BETS (top {len(best)}) ---", flush=True)
        _print_prop_table(best)

    remaining = MAX_SIGNALS_PER_DAY - min(n_best_bets, MAX_SIGNALS_PER_DAY)
    if remaining > 0 and n_signals > n_best_bets:
        lean = all_signals[all_signals["confidence"] != "BEST BET"].head(remaining)
        if not lean.empty:
            print(f"\n  --- LEAN ({len(lean)}) ---", flush=True)
            _print_prop_table(lean)

    print(f"{'=' * 80}\n", flush=True)


def _print_prop_table(df: pd.DataFrame) -> None:
    """Print a formatted table of prop signals."""
    print(f"  {'Player':25s} {'Team':5s} {'Stat':10s} {'Line':>6s} {'Pred':>6s} "
          f"{'Edge':>6s} {'Signal':8s} {'EV':>7s} {'p(hit)':>7s}", flush=True)
    print(f"  {'-' * 90}", flush=True)

    for _, row in df.iterrows():
        name = str(row["player_name"])[:25]
        team = str(row.get("team", ""))[:5]
        stat = str(row["stat_type"])[:10]
        line = row["prop_line"]
        pred = row["pred_value"]
        edge = row["edge"]
        signal = row["signal"]
        if signal == "OVER":
            ev = row.get("ev_over", np.nan)
            p_hit = row.get("p_over", np.nan)
        else:
            ev = row.get("ev_under", np.nan)
            p_hit = row.get("p_under", np.nan)

        ev_s = f"{ev:+.3f}" if pd.notna(ev) else "  N/A"
        p_s = f"{p_hit:.3f}" if pd.notna(p_hit) else "  N/A"
        print(f"  {name:25s} {team:5s} {stat:10s} {line:6.1f} {pred:6.1f} "
              f"{edge:+6.1f} {signal:8s} {ev_s:>7s} {p_s:>7s}", flush=True)


# ---------------------------------------------------------------------------
# CLV tracking for props
# ---------------------------------------------------------------------------

def track_prop_clv(target_date: str) -> None:
    """Track CLV for prop predictions by comparing to actual results.

    Loads predictions from the morning, compares to actual game results,
    and appends to a cumulative tracking CSV.
    """
    pred_file = PREDICTIONS_DIR / f"player_props_{target_date}.csv"
    edge_file = PREDICTIONS_DIR / f"player_prop_edges_{target_date}.csv"
    tracking_file = PREDICTIONS_DIR / "prop_tracking.csv"

    if not pred_file.exists():
        print(f"  No predictions file found for {target_date}: {pred_file}", flush=True)
        return

    if not edge_file.exists():
        print(f"  No edge predictions found for {target_date}: {edge_file}", flush=True)
        print("  CLV tracking requires prop edge signals.", flush=True)
        return

    print(f"  Loading predictions from {pred_file}", flush=True)
    preds = pd.read_csv(pred_file)
    edges = pd.read_csv(edge_file)

    # Load actual results from boxscores
    print("  Loading actual game results...", flush=True)
    schedule_df, team_games, player_games = build_team_games_and_players(include_historical=False)

    # Extended stats for fg3m
    ext = load_extended_player_stats()
    if not ext.empty:
        player_games = player_games.merge(ext, on=["game_id", "team", "player_id"], how="left")

    # Filter to the target date
    if "game_date_est" in player_games.columns:
        actual_games = player_games[player_games["game_date_est"] == target_date].copy()
    else:
        actual_games = pd.DataFrame()

    if actual_games.empty:
        print(f"  No completed games found for {target_date}. Games may not be finished yet.", flush=True)
        return

    # Match predictions to actuals
    tracking_rows: list[dict[str, Any]] = []

    for _, edge_row in edges.iterrows():
        player_name = edge_row["player_name"]
        stat_type = edge_row["stat_type"]
        signal = edge_row["signal"]
        prop_line = edge_row["prop_line"]
        pred_value = edge_row["pred_value"]

        if signal == "NO BET":
            continue

        # Find actual result
        mask = actual_games["player_name"].str.lower().str.contains(
            player_name.lower(), na=False
        )
        team = edge_row.get("team", "")
        if team:
            mask = mask & (actual_games["team"] == team)

        matched = actual_games[mask]
        if matched.empty:
            continue

        actual_row = matched.iloc[0]
        actual_val = actual_row.get(stat_type, np.nan)
        if pd.isna(actual_val):
            continue

        # Determine hit/miss
        if signal == "OVER":
            hit = 1 if actual_val > prop_line else 0
            ev = edge_row.get("ev_over", np.nan)
        else:
            hit = 1 if actual_val < prop_line else 0
            ev = edge_row.get("ev_under", np.nan)

        # P/L calculation
        pnl = VIG_FACTOR * 100 if hit else -100  # $100 flat bet

        tracking_rows.append({
            "date": target_date,
            "player_name": player_name,
            "team": team,
            "stat_type": stat_type,
            "prop_line": prop_line,
            "pred_value": pred_value,
            "actual_value": float(actual_val),
            "signal": signal,
            "confidence": edge_row.get("confidence", ""),
            "ev_at_signal": round(ev, 3) if pd.notna(ev) else np.nan,
            "hit": hit,
            "pnl": round(pnl, 2),
        })

    if not tracking_rows:
        print("  No prop signals could be matched to completed games.", flush=True)
        return

    new_tracking = pd.DataFrame(tracking_rows)

    # Append to cumulative tracking file
    if tracking_file.exists():
        existing = pd.read_csv(tracking_file)
        # Remove any existing entries for this date to avoid duplicates
        existing = existing[existing["date"] != target_date]
        combined = pd.concat([existing, new_tracking], ignore_index=True)
    else:
        combined = new_tracking

    combined.to_csv(tracking_file, index=False)

    # Print summary
    n_bets = len(new_tracking)
    n_hits = int(new_tracking["hit"].sum())
    total_pnl = new_tracking["pnl"].sum()
    hit_rate = n_hits / n_bets if n_bets > 0 else 0

    print(f"\n  --- CLV Tracking for {target_date} ---", flush=True)
    print(f"  Bets: {n_bets}  Hits: {n_hits}  Hit Rate: {hit_rate:.1%}  P/L: ${total_pnl:+.0f}", flush=True)

    for _, row in new_tracking.iterrows():
        result = "HIT" if row["hit"] else "MISS"
        print(
            f"    {row['player_name']:25s} {row['stat_type']:10s} "
            f"Line={row['prop_line']:5.1f}  Pred={row['pred_value']:5.1f}  "
            f"Actual={row['actual_value']:5.1f}  {row['signal']:6s}  {result}  "
            f"${row['pnl']:+.0f}",
            flush=True,
        )

    # Print cumulative stats
    if len(combined) > n_bets:
        cum_bets = len(combined)
        cum_hits = int(combined["hit"].sum())
        cum_pnl = combined["pnl"].sum()
        cum_hr = cum_hits / cum_bets if cum_bets > 0 else 0
        cum_roi = 100 * cum_pnl / (cum_bets * 100) if cum_bets > 0 else 0
        print(f"\n  --- Cumulative Stats ---", flush=True)
        print(
            f"  Total Bets: {cum_bets}  Hits: {cum_hits}  "
            f"Hit Rate: {cum_hr:.1%}  P/L: ${cum_pnl:+.0f}  ROI: {cum_roi:.1f}%",
            flush=True,
        )

    # Per-stat breakdown
    print(f"\n  --- Per-Stat Breakdown ---", flush=True)
    for st in combined["stat_type"].unique():
        st_df = combined[combined["stat_type"] == st]
        st_bets = len(st_df)
        st_hits = int(st_df["hit"].sum())
        st_pnl = st_df["pnl"].sum()
        st_hr = st_hits / st_bets if st_bets > 0 else 0
        print(f"    {st:10s}: {st_bets} bets, {st_hr:.1%} hit rate, ${st_pnl:+.0f} P/L", flush=True)

    print(f"\n  Tracking saved to {tracking_file}", flush=True)


# ---------------------------------------------------------------------------
# Calibration Monitoring (Phase 2)
# ---------------------------------------------------------------------------


def compute_calibration_report(
    min_sample: int = CALIB_MIN_SAMPLE,
    lookback_days: int = CALIB_DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    """Compute rolling calibration report from canonical results history.

    Slices by: stat_type, confidence, and edge_bucket.
    Only reports metrics when n >= min_sample.

    Returns a dict with:
      - lookback_days, as_of_date, total_graded
      - by_stat_type: list of per-stat-type metrics
      - by_confidence: list of per-confidence metrics
      - by_edge_bucket: list of per-edge-bucket metrics
      - alerts: list of any triggered alert conditions
    """
    if not PROP_RESULTS_HISTORY_FILE.exists():
        return {"error": "no history file"}

    history = pd.read_csv(PROP_RESULTS_HISTORY_FILE)
    if history.empty or "actual_value" not in history.columns:
        return {"error": "empty history"}

    # Filter to graded rows
    graded = history[history["actual_value"].notna()].copy()
    if graded.empty:
        return {"error": "no graded rows"}

    # Apply lookback window
    if "game_date_est" in graded.columns:
        graded["game_date_est"] = graded["game_date_est"].astype(str)
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        graded = graded[graded["game_date_est"] >= cutoff].copy()

    if graded.empty:
        return {"error": "no graded rows in lookback window"}

    # Compute p_hit: P(over) for OVER signals, P(under) for UNDER, P(over) for NO BET
    graded["p_hit"] = np.where(
        graded["signal"] == "UNDER",
        graded["p_under"].astype(float),
        graded["p_over"].astype(float),
    )

    # Edge buckets
    graded["edge_bucket"] = pd.cut(
        graded["edge_pct"].abs(),
        bins=[0, 5, 10, 15, 20, 100],
        labels=["0-5%", "5-10%", "10-15%", "15-20%", "20%+"],
        include_lowest=True,
    )

    # Signal-only for confidence/pnl metrics
    signal_graded = graded[graded["signal"] != "NO BET"].copy()

    report: dict[str, Any] = {
        "lookback_days": lookback_days,
        "as_of_date": datetime.now().strftime("%Y%m%d"),
        "total_graded": len(graded),
        "total_signal_graded": len(signal_graded),
    }

    alerts: list[dict[str, Any]] = []

    # --- By stat_type ---
    by_stat = prop_calibration_by_bucket(
        signal_graded if not signal_graded.empty else graded,
        "stat_type",
        min_sample=min_sample,
    )
    report["by_stat_type"] = by_stat

    # Check alert thresholds per stat_type
    for entry in by_stat:
        stat = entry["group"]
        if entry["gap"] > CALIB_ALERT_THRESHOLDS["hit_rate_vs_predicted_gap"]:
            alerts.append({
                "type": "miscalibration",
                "stat_type": stat,
                "gap": entry["gap"],
                "threshold": CALIB_ALERT_THRESHOLDS["hit_rate_vs_predicted_gap"],
            })
        if entry["brier"] > CALIB_ALERT_THRESHOLDS["brier_score_max"]:
            alerts.append({
                "type": "brier_degraded",
                "stat_type": stat,
                "brier": entry["brier"],
                "threshold": CALIB_ALERT_THRESHOLDS["brier_score_max"],
            })
        if entry["roi_pct"] < CALIB_ALERT_THRESHOLDS["roi_floor_pct"]:
            alerts.append({
                "type": "roi_floor",
                "stat_type": stat,
                "roi_pct": entry["roi_pct"],
                "threshold": CALIB_ALERT_THRESHOLDS["roi_floor_pct"],
            })

    # --- By confidence ---
    report["by_confidence"] = prop_calibration_by_bucket(
        signal_graded if not signal_graded.empty else graded,
        "confidence",
        min_sample=min_sample,
    )

    # --- By edge bucket ---
    report["by_edge_bucket"] = prop_calibration_by_bucket(
        signal_graded if not signal_graded.empty else graded,
        "edge_bucket",
        min_sample=min_sample,
    )

    # --- CLV summary (mean line movement in signal direction) ---
    if "line_move" in signal_graded.columns and not signal_graded.empty:
        with_move = signal_graded[signal_graded["line_move"].notna()].copy()
        if not with_move.empty:
            clv_dir = np.where(
                with_move["signal"] == "OVER",
                with_move["line_move"].astype(float),
                -with_move["line_move"].astype(float),
            )
            report["clv_mean"] = round(float(np.mean(clv_dir)), 3)
            report["clv_n"] = len(with_move)

    report["alerts"] = alerts
    return report


def print_calibration_report(report: dict[str, Any]) -> None:
    """Print calibration report to stdout."""
    if "error" in report:
        print(f"  Calibration report: {report['error']}", flush=True)
        return

    print(f"\n  === Calibration Report (last {report['lookback_days']} days) ===", flush=True)
    print(f"  Total graded: {report['total_graded']}  Signal bets: {report['total_signal_graded']}", flush=True)

    if report.get("by_stat_type"):
        print(f"\n  --- By Stat Type ---", flush=True)
        for e in report["by_stat_type"]:
            print(
                f"    {str(e['group']):10s}: n={e['n']:>4d}  "
                f"HitRate={e['hit_rate']:.1%}  Pred={e['mean_p_hit']:.1%}  "
                f"Gap={e['gap']:.3f}  Brier={e['brier']:.3f}  "
                f"ROI={e['roi_pct']:+.1f}%",
                flush=True,
            )

    if report.get("by_confidence"):
        print(f"\n  --- By Confidence ---", flush=True)
        for e in report["by_confidence"]:
            print(
                f"    {str(e['group']):10s}: n={e['n']:>4d}  "
                f"HitRate={e['hit_rate']:.1%}  Brier={e['brier']:.3f}  "
                f"ROI={e['roi_pct']:+.1f}%",
                flush=True,
            )

    if report.get("by_edge_bucket"):
        print(f"\n  --- By Edge Bucket ---", flush=True)
        for e in report["by_edge_bucket"]:
            print(
                f"    {str(e['group']):10s}: n={e['n']:>4d}  "
                f"HitRate={e['hit_rate']:.1%}  Brier={e['brier']:.3f}  "
                f"ROI={e['roi_pct']:+.1f}%",
                flush=True,
            )

    if "clv_mean" in report:
        print(f"\n  CLV (signal direction): mean={report['clv_mean']:+.3f} pts  (n={report['clv_n']})", flush=True)

    if report.get("alerts"):
        print(f"\n  !!! ALERTS ({len(report['alerts'])}) !!!", flush=True)
        for a in report["alerts"]:
            if a["type"] == "miscalibration":
                print(f"    MISCALIBRATION: {a['stat_type']} gap={a['gap']:.3f} > {a['threshold']:.3f}", flush=True)
            elif a["type"] == "brier_degraded":
                print(f"    BRIER DEGRADED: {a['stat_type']} brier={a['brier']:.3f} > {a['threshold']:.3f}", flush=True)
            elif a["type"] == "roi_floor":
                print(f"    ROI FLOOR: {a['stat_type']} roi={a['roi_pct']:.1f}% < {a['threshold']:.1f}%", flush=True)
    else:
        print(f"\n  No alerts triggered.", flush=True)


def get_calibration_degraded_stats(
    lookback_days: int = CALIB_RELIABILITY_LOOKBACK_DAYS,
    min_sample: int = CALIB_RELIABILITY_MIN_SAMPLE,
) -> set[str]:
    """Return set of stat_types where Brier > threshold in recent lookback window.

    Used as a reliability gate to suppress signals for degraded stat types.
    """
    if not PROP_RESULTS_HISTORY_FILE.exists():
        return set()

    try:
        history = pd.read_csv(PROP_RESULTS_HISTORY_FILE)
    except Exception:
        return set()

    graded = history[history["actual_value"].notna()].copy()
    if graded.empty:
        return set()

    if "game_date_est" in graded.columns:
        graded["game_date_est"] = graded["game_date_est"].astype(str)
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y%m%d")
        graded = graded[graded["game_date_est"] >= cutoff]

    # Only check signal rows
    signal = graded[graded["signal"] != "NO BET"].copy()
    if signal.empty:
        return set()

    signal["p_hit"] = np.where(
        signal["signal"] == "UNDER",
        signal["p_under"].astype(float),
        signal["p_over"].astype(float),
    )

    degraded: set[str] = set()
    for stat_type, grp in signal.groupby("stat_type"):
        valid = grp[grp["p_hit"].notna() & grp["hit"].notna()]
        if len(valid) < min_sample:
            continue
        brier = float(np.mean((valid["p_hit"].astype(float) - valid["hit"].astype(float)) ** 2))
        if brier > CALIB_ALERT_THRESHOLDS["brier_score_max"]:
            degraded.add(str(stat_type))
    return degraded


# ---------------------------------------------------------------------------
# Canonical Results (Phase 1)
# ---------------------------------------------------------------------------


def save_canonical_results(
    prop_edges: pd.DataFrame,
    target_date: str,
    policy_mode: str,
) -> Path:
    """Save ALL prop edge rows (signals + NO BET) to canonical results history.

    Predictions are stored with actual_value=NaN. Actuals are filled later by
    ``grade_canonical_results()``. Idempotent: replaces rows for the same date.

    Returns the path to the history file.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if prop_edges.empty:
        return PROP_RESULTS_HISTORY_FILE

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict[str, Any]] = []
    for _, r in prop_edges.iterrows():
        name_norm = normalize_player_name(r.get("player_name", ""))
        stat_type = str(r.get("stat_type", ""))
        team = str(r.get("team", ""))
        pid = generate_prediction_id(target_date, name_norm, stat_type, team=team)
        rows.append({
            "prediction_id": pid,
            "game_date_est": target_date,
            "player_name": r.get("player_name", ""),
            "player_name_norm": name_norm,
            "team": r.get("team", ""),
            "opp": r.get("opp", ""),
            "stat_type": stat_type,
            # Decision-time features
            "pred_value": r.get("pred_value", np.nan),
            "pred_value_base": r.get("pred_value_base", np.nan),
            "market_resid_adj": r.get("market_resid_adj", np.nan),
            "pre_avg5": r.get("pre_avg5", np.nan),
            "pre_avg10": r.get("pre_avg10", np.nan),
            "pre_season": r.get("pre_season", np.nan),
            "blended_std": r.get("blended_std", np.nan),
            "residual_std": r.get("residual_std", np.nan),
            "player_std": r.get("player_std", np.nan),
            "confirmed_starter": r.get("confirmed_starter", np.nan),
            "lineup_confirmed": r.get("lineup_confirmed", np.nan),
            "injury_status": r.get("injury_status", ""),
            "injury_unavailability_prob": r.get("injury_unavailability_prob", np.nan),
            # Line snapshot
            "prop_line": r.get("prop_line", np.nan),
            "open_line": r.get("open_line", np.nan),
            "over_odds": r.get("over_odds", np.nan),
            "under_odds": r.get("under_odds", np.nan),
            "open_over_odds": r.get("open_over_odds", np.nan),
            "open_under_odds": r.get("open_under_odds", np.nan),
            "line_move": r.get("line_move", np.nan),
            "line_move_pct": r.get("line_move_pct", np.nan),
            "source": r.get("source", ""),
            # Model outputs
            "p_over": r.get("p_over", np.nan),
            "p_under": r.get("p_under", np.nan),
            "ev_over": r.get("ev_over", np.nan),
            "ev_under": r.get("ev_under", np.nan),
            "edge": r.get("edge", np.nan),
            "edge_pct": r.get("edge_pct", np.nan),
            "signal": r.get("signal", "NO BET"),
            "confidence": r.get("confidence", ""),
            "signal_blocked_reason": r.get("signal_blocked_reason", ""),
            "signal_policy_mode": policy_mode,
            "lineup_lock_ok": r.get("lineup_lock_ok", np.nan),
            # Outcome (filled by grade_canonical_results)
            "actual_value": np.nan,
            "hit": np.nan,
            "pnl": np.nan,
            "push": np.nan,
            "graded_at": "",
            # Metadata
            "prediction_created_at": now_str,
            "model_version": MODEL_VERSION,
        })

    new_df = pd.DataFrame(rows)

    # Merge with existing history, keeping the FIRST-written row per prediction_id
    # to preserve decision-time snapshots (no hindsight bias from reruns).
    if PROP_RESULTS_HISTORY_FILE.exists():
        try:
            existing = pd.read_csv(PROP_RESULTS_HISTORY_FILE)
            existing_ids = set(existing["prediction_id"].astype(str))
            # Only insert rows whose prediction_id doesn't already exist
            truly_new = new_df[~new_df["prediction_id"].isin(existing_ids)]
            combined = pd.concat([existing, truly_new], ignore_index=True)
            n_new = len(truly_new)
            n_skipped = len(new_df) - n_new
        except Exception:
            combined = new_df
            n_new = len(new_df)
            n_skipped = 0
    else:
        combined = new_df
        n_new = len(new_df)
        n_skipped = 0

    combined.to_csv(PROP_RESULTS_HISTORY_FILE, index=False)
    skip_msg = f" ({n_skipped} already existed)" if n_skipped else ""
    print(f"  Canonical results saved: {n_new} new rows for {target_date}{skip_msg} -> {PROP_RESULTS_HISTORY_FILE}", flush=True)
    return PROP_RESULTS_HISTORY_FILE


def grade_canonical_results(target_date: str) -> pd.DataFrame:
    """Grade canonical results by filling actuals from boxscores.

    Loads the history file, fetches completed game data for target_date,
    matches predictions to actuals, computes hit/pnl, saves back.

    Returns the graded rows for the target date.
    """
    if not PROP_RESULTS_HISTORY_FILE.exists():
        print(f"  No canonical results history found at {PROP_RESULTS_HISTORY_FILE}", flush=True)
        return pd.DataFrame()

    history = pd.read_csv(PROP_RESULTS_HISTORY_FILE)
    date_mask = history["game_date_est"].astype(str) == str(target_date)
    target_rows = history[date_mask].copy()

    if target_rows.empty:
        print(f"  No predictions found for {target_date} in canonical history.", flush=True)
        return pd.DataFrame()

    already_graded = target_rows["actual_value"].notna().sum()
    if already_graded == len(target_rows):
        print(f"  All {len(target_rows)} rows already graded for {target_date}.", flush=True)
        return target_rows

    # Fetch actual results from boxscores
    print("  Loading actual game results...", flush=True)
    _schedule_df, _team_games, player_games = build_team_games_and_players(include_historical=False)

    ext = load_extended_player_stats()
    if not ext.empty:
        player_games = player_games.merge(ext, on=["game_id", "team", "player_id"], how="left")

    if "game_date_est" in player_games.columns:
        actual_games = player_games[player_games["game_date_est"] == target_date].copy()
    else:
        actual_games = pd.DataFrame()

    if actual_games.empty:
        print(f"  No completed games found for {target_date}. Games may not be finished yet.", flush=True)
        return pd.DataFrame()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    graded_count = 0

    for idx, row in target_rows.iterrows():
        if pd.notna(row.get("actual_value")):
            continue  # already graded

        player_name = str(row.get("player_name", ""))
        stat_type = str(row.get("stat_type", ""))
        team = str(row.get("team", ""))

        # Match by exact normalized name + team (no substring fallback to avoid
        # ambiguous matches like "Gary Harris" matching "Tobias Harris").
        actual_norm = actual_games["player_name"].map(normalize_player_name)
        pred_norm = normalize_player_name(player_name)
        mask = actual_norm.eq(pred_norm)
        if team:
            mask = mask & (actual_games["team"] == team)

        matched = actual_games[mask]
        if matched.empty:
            continue

        if len(matched) > 1:
            # Multiple matches (shouldn't happen with exact norm + team). Take first
            # but log a warning so it can be investigated.
            print(
                f"    Warning: multiple matches for {player_name} ({team}), using first",
                flush=True,
            )

        actual_row = matched.iloc[0]
        actual_val = actual_row.get(stat_type, np.nan)
        if pd.isna(actual_val):
            continue

        actual_val = float(actual_val)
        prop_line = float(row.get("prop_line", np.nan))
        signal = str(row.get("signal", "NO BET"))

        # Determine hit/push/pnl
        push = 0
        if pd.notna(prop_line) and actual_val == prop_line:
            push = 1
            hit = 0
            pnl = 0.0
        elif signal == "OVER":
            hit = 1 if actual_val > prop_line else 0
            pnl = VIG_FACTOR * 100 if hit else -100.0
        elif signal == "UNDER":
            hit = 1 if actual_val < prop_line else 0
            pnl = VIG_FACTOR * 100 if hit else -100.0
        else:
            # NO BET: still record actual_value but pnl=0
            hit = int(actual_val > prop_line) if pd.notna(prop_line) else np.nan
            pnl = 0.0

        history.loc[idx, "actual_value"] = actual_val
        history.loc[idx, "hit"] = hit
        history.loc[idx, "pnl"] = round(pnl, 2)
        history.loc[idx, "push"] = push
        history.loc[idx, "graded_at"] = now_str
        graded_count += 1

    history.to_csv(PROP_RESULTS_HISTORY_FILE, index=False)

    graded_rows = history[date_mask].copy()
    n_total = len(graded_rows)
    n_graded = graded_rows["actual_value"].notna().sum()
    n_signals = (graded_rows["signal"] != "NO BET").sum()
    n_hits = graded_rows.loc[graded_rows["signal"] != "NO BET", "hit"].sum()
    signal_total = graded_rows.loc[graded_rows["signal"] != "NO BET", "pnl"].sum()

    print(f"\n  --- Grading Results for {target_date} ---", flush=True)
    print(f"  Total predictions: {n_total}  Graded: {n_graded}  New: {graded_count}", flush=True)
    if n_signals > 0:
        sig_hr = n_hits / n_signals if n_signals > 0 else 0
        print(f"  Signal bets: {int(n_signals)}  Hits: {int(n_hits)}  Hit Rate: {sig_hr:.1%}  P/L: ${signal_total:+.0f}", flush=True)

    # Per-stat breakdown for signal bets
    sig_rows = graded_rows[(graded_rows["signal"] != "NO BET") & graded_rows["actual_value"].notna()]
    if not sig_rows.empty:
        print(f"\n  --- Per-Stat Breakdown ---", flush=True)
        for st in sorted(sig_rows["stat_type"].unique()):
            st_df = sig_rows[sig_rows["stat_type"] == st]
            st_bets = len(st_df)
            st_hits = int(st_df["hit"].sum())
            st_pnl = st_df["pnl"].sum()
            st_hr = st_hits / st_bets if st_bets > 0 else 0
            print(f"    {st:10s}: {st_bets} bets, {st_hr:.1%} hit rate, ${st_pnl:+.0f} P/L", flush=True)

    print(f"\n  Results saved to {PROP_RESULTS_HISTORY_FILE}", flush=True)
    return graded_rows


def migrate_tracking_to_canonical() -> None:
    """Migrate existing prop_tracking.csv data into canonical results history."""
    tracking_file = PREDICTIONS_DIR / "prop_tracking.csv"
    if not tracking_file.exists():
        print("  No prop_tracking.csv found to migrate.", flush=True)
        return

    tracking = pd.read_csv(tracking_file)
    if tracking.empty:
        print("  prop_tracking.csv is empty.", flush=True)
        return

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict[str, Any]] = []
    for _, r in tracking.iterrows():
        date = str(r.get("date", ""))
        name = str(r.get("player_name", ""))
        name_norm = normalize_player_name(name)
        stat_type = str(r.get("stat_type", ""))
        team = str(r.get("team", ""))
        pid = generate_prediction_id(date, name_norm, stat_type, team=team)
        rows.append({
            "prediction_id": pid,
            "game_date_est": date,
            "player_name": name,
            "player_name_norm": name_norm,
            "team": r.get("team", ""),
            "opp": "",
            "stat_type": stat_type,
            "pred_value": r.get("pred_value", np.nan),
            "pred_value_base": np.nan,
            "market_resid_adj": np.nan,
            "pre_avg5": np.nan,
            "pre_avg10": np.nan,
            "pre_season": np.nan,
            "blended_std": np.nan,
            "residual_std": np.nan,
            "player_std": np.nan,
            "confirmed_starter": np.nan,
            "lineup_confirmed": np.nan,
            "injury_status": "",
            "injury_unavailability_prob": np.nan,
            "prop_line": r.get("prop_line", np.nan),
            "open_line": np.nan,
            "over_odds": np.nan,
            "under_odds": np.nan,
            "open_over_odds": np.nan,
            "open_under_odds": np.nan,
            "line_move": np.nan,
            "line_move_pct": np.nan,
            "source": "",
            "p_over": np.nan,
            "p_under": np.nan,
            "ev_over": r.get("ev_at_signal", np.nan),
            "ev_under": np.nan,
            "edge": np.nan,
            "edge_pct": np.nan,
            "signal": r.get("signal", ""),
            "confidence": r.get("confidence", ""),
            "signal_blocked_reason": "",
            "signal_policy_mode": "baseline",
            "lineup_lock_ok": np.nan,
            "actual_value": r.get("actual_value", np.nan),
            "hit": r.get("hit", np.nan),
            "pnl": r.get("pnl", np.nan),
            "push": np.nan,
            "graded_at": now_str if pd.notna(r.get("actual_value")) else "",
            "prediction_created_at": "",
            "model_version": "migrated",
        })

    migrated = pd.DataFrame(rows)

    # Merge with existing history (deduplicate by prediction_id)
    if PROP_RESULTS_HISTORY_FILE.exists():
        existing = pd.read_csv(PROP_RESULTS_HISTORY_FILE)
        existing_ids = set(existing["prediction_id"].astype(str))
        new_rows = migrated[~migrated["prediction_id"].isin(existing_ids)]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = migrated

    combined.to_csv(PROP_RESULTS_HISTORY_FILE, index=False)
    print(f"  Migrated {len(migrated)} rows from prop_tracking.csv -> {PROP_RESULTS_HISTORY_FILE}", flush=True)


# ---------------------------------------------------------------------------
# Weekly Retrain
# ---------------------------------------------------------------------------


def run_weekly_retrain(
    player_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    team_games: pd.DataFrame,
    game_odds: pd.DataFrame,
    ref_features: pd.DataFrame | None,
    args: argparse.Namespace,
) -> None:
    """Run weekly retrain: fresh models, calibration report, market-line backtest.

    Steps:
    1. Invalidate feature cache (force rebuild)
    2. Train fresh models
    3. Run market-line backtest for performance snapshot
    4. Run calibration report
    5. Print summary
    """
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    print(f"  Weekly retrain as of {target_date}", flush=True)

    # Step 1: Invalidate feature cache
    if PLAYER_FEATURE_CACHE_FILE.exists():
        PLAYER_FEATURE_CACHE_FILE.unlink()
        print("  Invalidated player feature cache.", flush=True)
    if PLAYER_FEATURE_CACHE_META.exists():
        PLAYER_FEATURE_CACHE_META.unlink()

    # Step 2: Train fresh models
    print("\n  Training fresh core prop models...", flush=True)
    two_stage_models, single_models = train_prediction_models(player_df)
    n_two_stage = len([k for k in two_stage_models if not k.startswith("_")])
    n_residual = len(two_stage_models.get("_residual", {}))
    print(f"  Two-stage models: {n_two_stage}  Residual models: {n_residual}", flush=True)
    print(f"  Single models: {len(single_models)}", flush=True)

    # Step 3: Market-line backtest
    print("\n  Running market-line backtest...", flush=True)
    try:
        result = run_market_line_backtest(
            player_df,
            test_frac=0.2,
            bet_size=100.0,
            max_dates=args.market_backtest_max_dates,
            fetch_missing_lines=False,
        )
        if isinstance(result, dict):
            roi = result.get("roi_pct", "N/A")
            clv = result.get("avg_clv_line_pts", "N/A")
            print(f"  Market backtest ROI: {roi}  CLV: {clv}", flush=True)

            # Record snapshot
            append_weekly_market_check(result, as_of_date=target_date, max_dates=args.market_backtest_max_dates)
    except Exception as e:
        print(f"  Market-line backtest error: {e}", flush=True)

    # Step 4: Calibration report
    print("\n  Running calibration report...", flush=True)
    report = compute_calibration_report()
    print_calibration_report(report)

    # Save report
    report_path = PROP_LOG_DIR / f"calibration_report_{target_date}.json"
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Calibration report saved to {report_path}", flush=True)
    except Exception:
        pass

    # Step 5: Summary
    print(f"\n  === Weekly Retrain Complete ===", flush=True)
    print(f"  Date: {target_date}", flush=True)
    print(f"  Models: {n_two_stage} two-stage + {n_residual} residual + {len(single_models)} single", flush=True)
    if report.get("alerts"):
        print(f"  Calibration alerts: {len(report['alerts'])}", flush=True)
    else:
        print(f"  Calibration: OK (no alerts)", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict NBA player props")
    p.add_argument("--date", type=str, default=None,
                   help="Target date YYYYMMDD (default: today)")
    p.add_argument("--days", type=int, default=1,
                   help="Number of days to predict (default: 1)")
    p.add_argument("--backtest", action="store_true",
                   help="Run backtest evaluation")
    p.add_argument("--backtest-props", action="store_true",
                   help="Run backtest on prop edges (signal hit rates and P/L)")
    p.add_argument("--backtest-market-props", action="store_true",
                   help="Run backtest against actual market prop lines (cached/fetched)")
    p.add_argument("--walk-forward", action="store_true",
                   help="Run walk-forward backtest with season-based folds")
    p.add_argument("--track-clv", action="store_true",
                   help="Track CLV for a completed date's predictions")
    p.add_argument("--min-games", type=int, default=DEFAULT_MIN_GAMES,
                   help=f"Minimum games for a player (default: {DEFAULT_MIN_GAMES})")
    p.add_argument("--prop-lines", type=str, default=None,
                   help="Path to prop lines CSV (overrides default)")
    p.add_argument("--market-backtest-max-dates", type=int, default=30,
                   help="Max recent test dates to evaluate in market-line backtest (default: 30)")
    p.add_argument("--market-backtest-fetch-missing", action="store_true",
                   help="Fetch missing historical prop lines during market-line backtest")
    p.add_argument("--ablation-box-adv", action="store_true",
                   help="Run actionable market-line ablation: baseline vs BoxScoreAdvancedV3 features")
    p.add_argument("--ablation-max-dates", type=int, default=60,
                   help="Max recent dates for BoxScoreAdvancedV3 ablation market-line backtest (default: 60)")
    p.add_argument("--record-weekly-market-check", action="store_true",
                   help="Append actionable market-backtest ROI/CLV snapshot to weekly log")
    p.add_argument("--enforce-lineup-lock", action="store_true",
                   help="Suppress signals until game is within lineup lock window")
    p.add_argument("--lineup-lock-minutes", type=int, default=30,
                   help="Lineup lock window in minutes before tip (default: 30)")
    p.add_argument("--force-starter-refresh", action="store_true",
                   help="Force confirmed-starter refresh for all upcoming games (not just lock window)")
    p.add_argument("--keep-doubtful", action="store_true",
                   help="Keep doubtful players in output (default removes them)")
    p.add_argument("--market-model-max-dates", type=int, default=180,
                   help="How many cached prop-line dates to use for market residual/calibration training")
    p.add_argument("--enable-experimental-market-models", action="store_true",
                   help="Enable market residual + calibration (requires sufficient cached market history)")
    p.add_argument("--box-adv-fetch-missing", action="store_true",
                   help="Fetch missing BoxScoreAdvancedV3 raw payloads while building player features")
    p.add_argument("--box-adv-max-fetch", type=int, default=0,
                   help="Max missing BoxScoreAdvancedV3 games to fetch this run (0=all missing)")
    # Phase 1: Canonical results
    p.add_argument("--grade-results", action="store_true",
                   help="Grade canonical results for a completed date (fill actuals from boxscores)")
    p.add_argument("--migrate-tracking", action="store_true",
                   help="Migrate existing prop_tracking.csv into canonical results history")
    # Phase 2: Calibration monitoring
    p.add_argument("--calibration-report", action="store_true",
                   help="Generate standalone calibration report from graded results")
    # Phase 3: Market features ablation
    p.add_argument("--ablation-market-lines", action="store_true",
                   help="Walk-forward backtest with/without prop market line features")
    # Phase 4 + Weekly retrain
    p.add_argument("--weekly-retrain", action="store_true",
                   help="Run weekly retrain: fresh models, calibration report, market backtest")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROP_LINES_DIR.mkdir(parents=True, exist_ok=True)
    PROP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROP_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Grade results mode (Phase 1) ---
    if args.grade_results:
        target_date = args.date or (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        print(f"Grading canonical results for {target_date}...", flush=True)
        grade_canonical_results(target_date)
        return

    # --- Migrate tracking mode (Phase 1) ---
    if args.migrate_tracking:
        print("Migrating prop_tracking.csv to canonical results history...", flush=True)
        migrate_tracking_to_canonical()
        return

    # --- CLV tracking mode (legacy, delegates to grade_canonical_results when possible) ---
    if args.track_clv:
        target_date = args.date or (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        if PROP_RESULTS_HISTORY_FILE.exists():
            print(f"Grading canonical results for {target_date} (via --track-clv)...", flush=True)
            grade_canonical_results(target_date)
        else:
            print(f"Tracking CLV for {target_date}...", flush=True)
            track_prop_clv(target_date)
        return

    print("Loading game history and building team/player features...", flush=True)
    schedule_df, team_games, player_games = build_team_games_and_players(include_historical=True)

    # Load historical + current Vegas odds for training feature backfill
    print("Loading Vegas odds for training data...", flush=True)
    game_odds = load_game_odds_lookup(schedule_df)

    # Build referee crew features from cached boxscores
    print("Building referee crew features...", flush=True)
    ref_features = build_referee_game_features(team_games)
    if not ref_features.empty:
        print(f"  Referee features: {len(ref_features)} games, {ref_features['ref_crew_avg_total'].notna().sum()} with rolling data", flush=True)

    # Build enriched player features
    print("Building player-level features...", flush=True)
    player_df = load_or_build_player_features(
        player_games,
        team_games,
        game_odds,
        min_games=args.min_games,
        ref_features=ref_features,
        box_adv_fetch_missing=args.box_adv_fetch_missing,
        box_adv_max_fetch=args.box_adv_max_fetch,
    )
    print(f"  Player-game rows after filtering: {len(player_df)}", flush=True)
    print(f"  Unique players: {player_df['player_id'].nunique()}", flush=True)

    if player_df.empty:
        print("No player data available. Ensure historical boxscores are cached.", flush=True)
        return

    # --- Backtest mode ---
    if args.backtest:
        print("\nRunning player props backtest...", flush=True)
        results = run_backtest(player_df)

        if results:
            two_stage = results.pop("_two_stage", {})
            print("\n--- Backtest Summary (Single-Stage) ---", flush=True)
            for target, metrics in results.items():
                print(f"  {target:>10s}: MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.3f}", flush=True)
            if two_stage:
                print("\n--- Backtest Summary (Two-Stage) ---", flush=True)
                for target, metrics in two_stage.items():
                    print(f"  {target:>10s}: MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.3f}", flush=True)
        return

    # --- BoxScoreAdvancedV3 ablation mode ---
    if args.ablation_box_adv:
        print(
            "\nRunning BoxScoreAdvancedV3 ablation (baseline vs advanced; actionable market-line metrics)...",
            flush=True,
        )
        run_box_advanced_ablation(
            player_df,
            max_dates=args.ablation_max_dates,
            fetch_missing_lines=args.market_backtest_fetch_missing,
        )
        return

    # --- Prop edge backtest mode ---
    if args.backtest_props:
        print("\nRunning prop edge backtest (signal hit rates + P/L)...", flush=True)
        edge_results = backtest_prop_edges(player_df)

        if edge_results:
            print("\n--- Prop Edge Backtest Summary ---", flush=True)
            total_profit = 0.0
            total_bets = 0
            for target, metrics in edge_results.items():
                wr = metrics.get("total_win_rate", np.nan)
                wr_s = f"{wr:.1%}" if pd.notna(wr) else "N/A"
                print(
                    f"  {target:>10s}: Bets={metrics['total_bets']:>4d}  "
                    f"WinRate={wr_s}  "
                    f"P/L=${metrics['profit_flat_100']:+.0f}  "
                    f"ROI={metrics.get('roi_pct', 0):.1f}%",
                    flush=True,
                )
                total_profit += metrics.get("profit_flat_100", 0)
                total_bets += metrics.get("total_bets", 0)
            if total_bets > 0:
                print(
                    f"\n  TOTAL: {total_bets} bets, ${total_profit:+.0f} P/L, "
                    f"{100 * total_profit / (total_bets * 100):.1f}% ROI",
                    flush=True,
                )
        return

    # --- Market-line prop backtest mode ---
    if args.backtest_market_props:
        print("\nRunning market-line prop backtest (actual lines + outcomes)...", flush=True)
        result = run_market_line_backtest(
            player_df,
            test_frac=0.2,
            bet_size=100.0,
            max_dates=args.market_backtest_max_dates,
            fetch_missing_lines=args.market_backtest_fetch_missing,
        )
        if args.record_weekly_market_check:
            as_of_date = args.date or datetime.now().strftime("%Y%m%d")
            append_weekly_market_check(
                result if isinstance(result, dict) else {"status": "unknown"},
                as_of_date=as_of_date,
                max_dates=args.market_backtest_max_dates,
            )
        return

    # --- Walk-forward backtest mode ---
    if args.walk_forward:
        print("\nRunning walk-forward backtest for player props...", flush=True)
        run_walk_forward_backtest(player_df)
        return

    # --- Calibration report mode (Phase 2) ---
    if args.calibration_report:
        print("\nGenerating calibration report from canonical results...", flush=True)
        report = compute_calibration_report()
        print_calibration_report(report)
        # Save JSON
        report_path = PROP_LOG_DIR / f"calibration_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved to {report_path}", flush=True)
        return

    # --- Market lines ablation mode (Phase 3) ---
    if args.ablation_market_lines:
        print("\nRunning market-line feature ablation (walk-forward with/without market features)...", flush=True)
        _run_market_line_ablation(player_df)
        return

    # --- Weekly retrain mode ---
    if args.weekly_retrain:
        print("\nRunning weekly retrain...", flush=True)
        run_weekly_retrain(player_df, schedule_df, team_games, game_odds, ref_features, args)
        return

    # --- Prediction mode ---
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    print(f"\nFetching upcoming games for {target_date}...", flush=True)
    upcoming = fetch_upcoming_schedule(target_date, args.days)
    if upcoming.empty:
        print(f"No upcoming games found for {target_date}.", flush=True)
        return

    print(f"  Found {len(upcoming)} upcoming games", flush=True)
    for _, g in upcoming.iterrows():
        print(f"    {g['away_team']} @ {g['home_team']} ({g.get('status', '')})", flush=True)

    print("\nChecking injury report...", flush=True)
    injury_map = fetch_injury_status_map(target_date)
    if injury_map:
        by_status = pd.Series([v.get("status", "") for v in injury_map.values()]).value_counts()
        print(f"  Loaded injury statuses for {len(injury_map)} players", flush=True)
        print(f"  Status mix: {', '.join(f'{k}={int(v)}' for k, v in by_status.items())}", flush=True)
    else:
        print("  No injury data available", flush=True)

    print("\nRefreshing confirmed starters...", flush=True)
    confirmed_starters = fetch_confirmed_starters(
        upcoming,
        lock_minutes=args.lineup_lock_minutes,
        force=args.force_starter_refresh,
    )
    n_games_with_starters = len(confirmed_starters)
    print(f"  Confirmed starter games: {n_games_with_starters}/{len(upcoming)}", flush=True)

    # Merge extended stats into player_games for prediction lookup
    print("\nLoading extended player stats...", flush=True)
    ext = load_extended_player_stats()
    pg_extended = player_games.copy()
    if not ext.empty:
        pg_extended = pg_extended.merge(ext, on=["game_id", "team", "player_id"], how="left")

    print("\nTraining core prop models (single fit for this run)...", flush=True)
    two_stage_models, single_models = train_prediction_models(player_df)

    print("Generating player prop predictions...", flush=True)
    pred_df = build_player_predictions(
        player_df,
        team_games,
        pg_extended,
        upcoming,
        min_games=args.min_games,
        injury_status_map=injury_map,
        confirmed_starters=confirmed_starters,
        two_stage_models=two_stage_models,
        single_models=single_models,
    )

    if pred_df.empty:
        print("No predictions generated.", flush=True)
        return

    pred_df = apply_injury_status_to_predictions(pred_df, injury_map)
    pred_df = filter_out_inactive(pred_df, injury_map, remove_doubtful=(not args.keep_doubtful))

    market_residual_models: dict[str, dict[str, Any]] = {}
    prob_calibrators: dict[str, dict[str, Any]] = {}
    print("\nTraining market residual/calibration models from cached prop lines...", flush=True)
    residual_models, market_calibs, diag = train_market_residual_models(
        player_df,
        max_dates=args.market_model_max_dates,
        pretrained_models=(two_stage_models, single_models),
    )
    coverage = summarize_market_coverage(diag)
    market_rows = int(coverage.get("total_rows", 0))
    print(f"  Cached market rows available: {market_rows}", flush=True)
    progress_snapshot = record_market_progress(target_date, coverage)

    latest_weekly = load_latest_weekly_market_check()
    policy_mode, policy_reason = choose_signal_policy_mode(coverage, latest_weekly)
    apply_signal_policy(policy_mode)
    print(
        f"  Active signal policy: {ACTIVE_SIGNAL_POLICY_MODE} ({policy_reason})",
        flush=True,
    )
    if latest_weekly:
        lw_date = str(latest_weekly.get("as_of_date", ""))
        lw_roi = latest_weekly.get("roi_pct", np.nan)
        lw_clv = latest_weekly.get("avg_clv_line_pts", np.nan)
        lw_drift = latest_weekly.get("calibration_drift_abs", np.nan)
        print(
            f"  Latest weekly check [{lw_date}]: ROI={lw_roi}, CLV={lw_clv}, drift={lw_drift}",
            flush=True,
        )
    if progress_snapshot.get("ready_total", 0):
        print(
            "  Coverage threshold met for total rows; per-stat coverage determines paper-phase quality.",
            flush=True,
        )

    force_market_layers = bool(args.enable_experimental_market_models)
    if market_rows >= MARKET_MODEL_MIN_ROWS or force_market_layers:
        market_residual_models = residual_models
        if market_residual_models:
            print(f"  Trained residual models: {', '.join(sorted(market_residual_models.keys()))}", flush=True)
    else:
        print(
            f"  Skipping residual models: need >= {MARKET_MODEL_MIN_ROWS} rows.",
            flush=True,
        )

    if market_rows >= MARKET_CALIB_MIN_ROWS or force_market_layers:
        prob_calibrators = dict(market_calibs)
        if prob_calibrators:
            print(f"  Probability calibrators: {', '.join(sorted(prob_calibrators.keys()))}", flush=True)
    else:
        print(
            f"  Skipping probability calibration: need >= {MARKET_CALIB_MIN_ROWS} rows.",
            flush=True,
        )

    # --- Load prop lines and compute edges ---
    print(f"\nFetching prop lines for {target_date}...", flush=True)
    prop_lines = fetch_player_prop_lines(target_date, override_path=args.prop_lines)

    prop_edges = pd.DataFrame()
    if not prop_lines.empty:
        print(f"Loaded {len(prop_lines)} prop lines", flush=True)
        print("Computing residual standard deviations...", flush=True)
        residual_stds = compute_prop_residual_stds(
            player_df,
            leakage_safe=True,
        )
        for stat, std in residual_stds.items():
            print(f"  {stat}: residual_std = {std:.2f}", flush=True)

        # Phase 2: check calibration reliability gate
        degraded_stats = get_calibration_degraded_stats()
        if degraded_stats:
            print(f"  Calibration drift gate active for: {', '.join(sorted(degraded_stats))}", flush=True)

        print("Computing prop edges...", flush=True)
        prop_edges = compute_prop_edges(
            pred_df,
            prop_lines,
            residual_stds,
            market_residual_models=market_residual_models,
            prob_calibrators=prob_calibrators,
            calibration_degraded_stats=degraded_stats,
        )
        if not prop_edges.empty:
            pre_signals = int((prop_edges["signal"] != "NO BET").sum())
            prop_edges = apply_lineup_lock_gate(
                prop_edges,
                upcoming,
                lock_minutes=args.lineup_lock_minutes,
                enforce=args.enforce_lineup_lock,
            )
            post_signals = int((prop_edges["signal"] != "NO BET").sum())
            if args.enforce_lineup_lock:
                print(
                    f"  Lineup lock gate: {pre_signals} -> {post_signals} signals "
                    f"(window <= {args.lineup_lock_minutes} min)",
                    flush=True,
                )
            edge_out_path = PREDICTIONS_DIR / f"player_prop_edges_{target_date}.csv"
            prop_edges.to_csv(edge_out_path, index=False)
            print(f"  Prop edges saved to {edge_out_path}", flush=True)

            # Save canonical results (Phase 1): all predictions (signals + NO BET)
            save_canonical_results(prop_edges, target_date, ACTIVE_SIGNAL_POLICY_MODE)

    # Format and save predictions
    output = format_predictions(pred_df)
    out_path = PREDICTIONS_DIR / f"player_props_{target_date}.csv"
    output.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}", flush=True)
    print(f"  {len(output)} player predictions across {upcoming['home_team'].nunique()} games", flush=True)

    # Print predictions summary
    print(f"\n--- Player Props Predictions for {target_date} ---", flush=True)
    for _, game in upcoming.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        print(f"\n  {away} @ {home}:", flush=True)
        game_preds = output[
            (output["home_team"] == home) & (output["away_team"] == away)
        ]
        if game_preds.empty:
            print("    (no eligible players)", flush=True)
            continue

        for team in [home, away]:
            team_preds = game_preds[game_preds["team"] == team].head(8)
            if team_preds.empty:
                continue
            label = f"    {team}" + (" (Home)" if team == home else " (Away)")
            print(label, flush=True)
            for _, row in team_preds.iterrows():
                name = str(row.get("player_name", ""))[:25]
                pts = row.get("pred_points")
                reb = row.get("pred_rebounds")
                ast = row.get("pred_assists")
                mins = row.get("pred_minutes")
                fg3 = row.get("pred_fg3m")
                pts_s = f"{pts:.1f}" if pd.notna(pts) else "?"
                reb_s = f"{reb:.1f}" if pd.notna(reb) else "?"
                ast_s = f"{ast:.1f}" if pd.notna(ast) else "?"
                min_s = f"{mins:.1f}" if pd.notna(mins) else "?"
                fg3_str = f"  3PM={fg3:.1f}" if fg3 is not None and pd.notna(fg3) else ""
                print(f"      {name:25s}  PTS={pts_s:>5s}  REB={reb_s:>5s}  AST={ast_s:>5s}  MIN={min_s:>5s}{fg3_str}", flush=True)

    # Print prop edge signals
    if not prop_edges.empty:
        print_prop_edge_summary(prop_edges)

    # Phase 2: Brief calibration summary if enough graded data exists
    if PROP_RESULTS_HISTORY_FILE.exists():
        try:
            report = compute_calibration_report(min_sample=30, lookback_days=30)
            if report.get("total_signal_graded", 0) >= 30:
                print(f"\n  --- Quick Calibration Check (last 30 days) ---", flush=True)
                for entry in report.get("by_stat_type", []):
                    print(
                        f"    {str(entry['group']):10s}: n={entry['n']:>3d}  "
                        f"HR={entry['hit_rate']:.0%}  Brier={entry['brier']:.3f}  "
                        f"ROI={entry['roi_pct']:+.1f}%",
                        flush=True,
                    )
                if report.get("alerts"):
                    print(f"    ({len(report['alerts'])} alert(s) — run --calibration-report for details)", flush=True)
        except Exception:
            pass  # Don't let calibration summary crash the main pipeline


if __name__ == "__main__":
    main()
