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
import hashlib
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
from uuid import uuid4

import numpy as np
import pandas as pd
import requests
from pandas.errors import PerformanceWarning
from scipy import stats as sp_stats
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    import pdfplumber
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

from analyze_nba_2025_26_advanced import (
    BOXSCORE_CACHE,
    CACHE_DIR,
    ESPN_SCOREBOARD_URL,
    HIST_CACHE_DIR,
    SCHEDULE_URL,
    SEASON,
    SEASONS,
    TEAM_COORDS,
    INJURY_STATUS_PROB,
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
OFFICIAL_INJURY_DIR = PROP_CACHE_DIR / "official_injury_reports"
ODDS_API_SNAPSHOT_DIR = PROP_CACHE_DIR / "odds_api_snapshots"
GAME_ROTATION_CACHE_DIR = PROP_CACHE_DIR / "game_rotation_raw"
MATCHUPS_CACHE_DIR = PROP_CACHE_DIR / "boxscore_matchups_raw"
MARKET_MODEL_MIN_ROWS = 500
MARKET_CALIB_MIN_ROWS = 250
_EXTENDED_STATS_CACHE: pd.DataFrame | None = None  # Invalidated when new fields are added
_BOX_ADV_CACHE: pd.DataFrame | None = None
_GAME_ROTATION_CACHE: pd.DataFrame | None = None
_MATCHUPS_CACHE: pd.DataFrame | None = None
_ESPN_ATHLETE_DISK_CACHE_LOADED = False
PLAYER_FEATURE_CACHE_VERSION = "v16"  # v16: BRef advanced stats (BPM, def_rtg, efg, stl/blk%), BRef opponent defense


def _versioned_model_file(stem: str, suffix: str, version: str | None = None) -> Path:
    ver = version or PLAYER_FEATURE_CACHE_VERSION
    return MODEL_DIR / f"{stem}_{ver}{suffix}"


def _artifact_read_candidates(primary: Path, legacy: Path | None = None) -> list[Path]:
    candidates: list[Path] = [primary]
    if legacy is not None and legacy not in candidates:
        candidates.append(legacy)
    return [p for p in candidates if p.exists()]


PLAYER_FEATURE_CACHE_FILE = _versioned_model_file("player_features_cache", ".pkl")
PLAYER_FEATURE_CACHE_META = _versioned_model_file("player_features_cache_meta", ".json")
LEGACY_PLAYER_FEATURE_CACHE_FILE = MODEL_DIR / "player_features_cache.pkl"
LEGACY_PLAYER_FEATURE_CACHE_META = MODEL_DIR / "player_features_cache_meta.json"
NO_LINES_RETRY_SECS_SAME_DAY = 45 * 60
BOX_ADV_REQUEST_SLEEP_SECS = 1.0
BOX_ADV_DEFAULT_RETRIES = 5
BOX_ADV_DEFAULT_TIMEOUT = 20
OFFICIAL_INJURY_PAGE_URL = "https://official.nba.com/nba-injury-report-2020-21-season/"
OFFICIAL_INJURY_PAGE_TIMEOUT = 20
STATS_NBA_ROTATION_URL = "https://stats.nba.com/stats/gamerotation"
STATS_NBA_MATCHUPS_URL = "https://stats.nba.com/stats/boxscorematchupsv3"

# Props we predict
PROP_TARGETS = ["points", "rebounds", "assists", "minutes"]

# Minimum games for a player to be modeled
DEFAULT_MIN_GAMES = 20

# Betting parameters for props
VIG_FACTOR = 0.9524        # net payout per $1 at ~-105 juice
BREAKEVEN_PROB = 1.0 / (1.0 + VIG_FACTOR)  # ~0.5122
# Post-calibration clipping to avoid saturated 0/1 probabilities in outputs.
PROB_CLIP_FLOOR = 0.02
PROB_CLIP_CEILING = 0.98

# Signal thresholds
MIN_EDGE_PCT = 15.0        # minimum edge% to signal (e.g., pred 23 vs line 20 = 15%)
MIN_EV = 0.20              # minimum EV to signal (20 cents per dollar)
BEST_BET_EV = 0.40         # EV threshold for "best bet" flag
MAX_SIGNALS_PER_DAY = 10   # cap total signals to avoid overexposure
MAX_SIGNALS_PER_PLAYER = 2       # cap signals per player (already enforced in portfolio filter)
MAX_SIGNALS_PER_GAME = 4         # cap signals from a single game
MAX_SIGNALS_PER_TEAM = 3         # cap signals from one team's players
USE_BOXSCORE_ADV_FEATURES = True
USE_ROTATION_MATCHUP_FEATURES = True
USE_ENSEMBLE = _HAS_LGBM  # XGBoost + LightGBM stacking (requires lightgbm)
USE_QUANTILE_UNCERTAINTY = True  # Quantile regression for uncertainty (Step 2)
USE_PLAYER_TARGET_ENCODING = False  # Per-player target encoding (Step 4, experimental)
USE_MARKET_LINE_BLENDING = False  # Market line blending (Step 6, gated)
MARKET_LINE_BLEND_ALPHA = 0.5  # Weight on model prediction (vs 1-alpha on market line)

# OVER signal hardening (disabled — OVER now uses same thresholds as UNDER)
SUPPRESS_LEAN_OVER = False
OVER_REQUIRE_LINEUP_CONFIRMED = False
OVER_MAX_INJURY_PROB = 1.0  # effectively disabled

# Points bias correction (rolling residual shrinkage)
USE_POINTS_BIAS_CORRECTION = True
POINTS_BIAS_SHRINK_K = 120  # Shrinkage strength (higher = more conservative)
POINTS_BIAS_CAP = 1.5  # Max absolute correction in points
POINTS_BIAS_MIN_SAMPLE = 75  # Minimum graded rows to activate
POINTS_BIAS_LOOKBACK_DAYS = 30  # Rolling window for points bias estimation
POINTS_BIAS_FILE = PROP_LOG_DIR / "points_bias.json"

# Star-out heuristic floor boost
# When forward injury pressure is extreme (>p95 of training), apply a floor boost
# to the top remaining scorers to compensate for model under-extrapolation.
STAR_OUT_PRESSURE_THRESHOLD = 38.0  # ~p95 of training team_injury_pressure
STAR_OUT_BOOST_SCALE = 0.6  # Scale factor on role_beta * pressure for floor boost
STAR_OUT_MAX_BOOST = 5.0  # Max boost in raw stat units
STAR_OUT_TOP_N_PLAYERS = 3  # Apply boost to top N scorers per team
STAR_OUT_STAT_TARGETS = {"points", "rebounds", "assists"}  # Stats eligible for boost

# CLV gate robustness
DEPLOY_CLV_MIN_SAMPLE = 50  # Minimum rows with nonzero line movement for CLV gate

# Signal policy controls
SIGNAL_POINTS_ONLY = False
MIN_SIGNAL_PRED_MINUTES = 20.0
MIN_SIGNAL_PRE_MINUTES_AVG10 = 18.0

# Side-specific thresholds (symmetric — OVER and UNDER use same thresholds)
MIN_EDGE_PCT_BY_SIDE = {
    "OVER": MIN_EDGE_PCT,
    "UNDER": MIN_EDGE_PCT,
}
MIN_EV_BY_SIDE = {
    "OVER": MIN_EV,
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

# Phase 1: OOF validation fold minimum size
OOF_MIN_VAL_FOLD_SIZE = 100

# Phase 10: Deploy gate thresholds
DEPLOY_GATE_MIN_GRADED_PER_STAT = 100
DEPLOY_GATE_MIN_CLV = 0.0
DEPLOY_GATE_MAX_BRIER = 0.28
DEPLOY_GATE_MAX_MODEL_AGE_DAYS = 14
DEPLOY_GATES_ENFORCE = False  # advisory mode initially

# Canonical results history (Phase 1)
PROP_RESULTS_HISTORY_FILE = PREDICTIONS_DIR / "prop_results_history.csv"
MODEL_VERSION = "v6"  # v6: forward-injury-pressure inference override + coverage gates

# Persistent monitoring logs
MARKET_PROGRESS_LOG = PROP_LOG_DIR / "market_data_progress.csv"
MARKET_WEEKLY_LOG = PROP_LOG_DIR / "market_weekly_actionable_backtest.csv"

# Optuna tuning: cached best params per target/model type
TUNED_PARAMS_FILE = _versioned_model_file("prop_tuned_params", ".json")
LEGACY_TUNED_PARAMS_FILE = MODEL_DIR / "prop_tuned_params.json"

# Experiment tracking: append-only CSV log of all experiments
EXPERIMENT_LOG_FILE = PROP_LOG_DIR / "experiment_log.csv"

# Automated feature selection: persisted selected feature groups
FEATURE_SELECTION_FILE = _versioned_model_file("feature_selection", ".json")
LEGACY_FEATURE_SELECTION_FILE = MODEL_DIR / "feature_selection.json"

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
        "min_edge_pct_by_side": {"OVER": 18.0, "UNDER": 13.0},
        "min_ev_by_side": {"OVER": 0.26, "UNDER": 0.17},
        "best_bet_ev": 0.35,
        "max_signals_per_day": 12,
    },
    "tightened": {
        "signal_points_only": False,
        "min_pred_minutes": 22.0,
        "min_pre_minutes_avg10": 20.0,
        "min_edge_pct_by_side": {"OVER": 23.0, "UNDER": 17.0},
        "min_ev_by_side": {"OVER": 0.35, "UNDER": 0.24},
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
INJURY_FORWARD_STATUSES = {"out", "doubtful", "questionable", "suspension"}
INJURY_FORWARD_MIN_UNAVAIL = 0.20
INJURY_FEED_STALE_HOURS = 6.0
INJURY_FEED_MIN_ROWS = 8


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


def _match_actual_player_rows(
    actual_df: pd.DataFrame,
    player_name: str,
    team: str = "",
    player_id: Any | None = None,
) -> pd.DataFrame:
    """Strict player matcher for grading/backtests.

    Order:
    1) exact player_id (+team when provided)
    2) exact normalized name (+team when provided)
    Returns exactly-one-row match or empty frame.
    """
    if actual_df.empty:
        return actual_df.iloc[0:0]

    team = str(team or "")
    pid = _to_float(player_id)
    if pd.notna(pid) and "player_id" in actual_df.columns:
        ids = pd.to_numeric(actual_df["player_id"], errors="coerce")
        mask = ids.eq(pid)
        if team:
            mask = mask & actual_df["team"].astype(str).eq(team)
        matched = actual_df[mask]
        if len(matched) == 1:
            return matched

    name_norm = normalize_player_name(player_name)
    norm_series = actual_df["player_name"].map(normalize_player_name)
    mask = norm_series.eq(name_norm)
    if team:
        mask = mask & actual_df["team"].astype(str).eq(team)
    matched = actual_df[mask]
    if len(matched) == 1:
        return matched
    return actual_df.iloc[0:0]


def generate_prediction_id(
    date: str,
    name_norm: str,
    stat_type: str,
    team: str | None = None,
    player_id: Any | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
) -> str:
    """Deterministic dedup key for canonical results.

    Preferred key: '{date}_{home}_{away}_{team}_{player_id}_{stat_type}'
    Fallback key uses game/team context when available:
    '{date}_{home}_{away}_{team}_{name_norm}_{stat_type}'.
    Legacy fallback: '{date}_{name_norm}_{stat_type}'.
    """
    pid = _to_float(player_id)
    if pd.notna(pid):
        home = str(home_team or "").upper()
        away = str(away_team or "").upper()
        tm = str(team or "").upper()
        return f"{date}_{home}_{away}_{tm}_{int(pid)}_{stat_type}"
    home = str(home_team or "").upper()
    away = str(away_team or "").upper()
    tm = str(team or "").upper()
    if home or away or tm:
        return f"{date}_{home}_{away}_{tm}_{name_norm}_{stat_type}"
    return f"{date}_{name_norm}_{stat_type}"


def parse_asof_utc(asof_str: str | None) -> pd.Timestamp:
    """Parse optional run context timestamp; defaults to now UTC."""
    if not asof_str:
        return pd.Timestamp.now(tz="UTC")
    ts = pd.to_datetime(asof_str, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid --asof-utc: {asof_str}")
    return ts


def _canonical_latest_view(history: pd.DataFrame) -> pd.DataFrame:
    """Return latest snapshot rows per game date for canonical metrics/gates."""
    if history.empty:
        return history
    out = history.copy()
    if "is_latest" in out.columns:
        return out[out["is_latest"].fillna(0).astype(int) == 1].copy()
    if "prediction_created_at" in out.columns:
        out["_created"] = pd.to_datetime(out["prediction_created_at"], errors="coerce")
        if out["_created"].notna().sum() == 0:
            return out.drop(columns=["_created"]).copy()
        out = out.sort_values(["game_date_est", "_created"]).copy()
        idx = out.groupby("game_date_est")["_created"].transform("max") == out["_created"]
        return out[idx].drop(columns=["_created"]).copy()
    return out


def _injury_key(team: str, player_name: str) -> str:
    return f"{(team or '').upper()}|{normalize_player_name(player_name)}"


_OFFICIAL_TEAM_NAME_MAP = {
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LA CLIPPERS": "LAC",
    "LOS ANGELES CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",
}

_OFFICIAL_STATUS_MAP = {
    "out": "out",
    "out for season": "out",
    "doubtful": "doubtful",
    "questionable": "questionable",
    "probable": "probable",
    "game time decision": "questionable",
    "available": "probable",
    "suspension": "suspension",
}


def _official_status_to_prob(status: str) -> float:
    norm = _OFFICIAL_STATUS_MAP.get(str(status or "").strip().lower(), "")
    if norm:
        return float(INJURY_STATUS_PROB.get(norm, 0.5))
    return 0.5


def _fetch_official_injury_pdf_links(target_date: str) -> list[str]:
    """Scrape official NBA injury report PDF links for the target date."""
    month_token = f"{target_date[0:4]}/{target_date[4:6]}/"
    day_token = f"{target_date[4:6]}{target_date[6:8]}{target_date[0:4]}"
    links: list[str] = []
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(OFFICIAL_INJURY_PAGE_URL, timeout=OFFICIAL_INJURY_PAGE_TIMEOUT)
            resp.raise_for_status()
            found = re.findall(
                r"https://ak-static\.cms\.nba\.com/wp-content/uploads/sites/4/[^\"' >]+\.pdf",
                resp.text,
                flags=re.IGNORECASE,
            )
            for link in found:
                if month_token in link or day_token in link.replace("-", ""):
                    links.append(link)
            break
        except Exception as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    if not links and last_err is not None:
        print(f"  Warning: official NBA injury page fetch failed: {last_err}", flush=True)
    return sorted(set(links))


def _extract_official_injury_rows_from_pdf(pdf_path: Path, pdf_url: str, target_date: str) -> list[dict[str, Any]]:
    """Parse official NBA injury report PDF into normalized player rows."""
    if not _HAS_PDFPLUMBER:
        return []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception:
        return []
    if not text:
        return []

    rows: list[dict[str, Any]] = []
    current_team = ""
    report_ts = ""
    team_names = set(_OFFICIAL_TEAM_NAME_MAP)
    status_re = (
        r"(Out For Season|Game Time Decision|Questionable|Probable|Doubtful|Out|Available|Suspension)"
    )
    line_re = re.compile(
        rf"^(?P<name>[A-Za-zÀ-ÿ'.\- ]+?)\s+(?P<pos>G-F|F-G|F-C|C-F|G|F|C|F-G/C|G/F|F/G|C/F)?\s*"
        rf"(?P<status>{status_re})\s+(?P<detail>.+)$",
        flags=re.IGNORECASE,
    )
    ts_re = re.compile(
        r"(\d{1,2}:\d{2}\s*(?:a\.m\.|p\.m\.|AM|PM)\s+ET,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4})",
        flags=re.IGNORECASE,
    )

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", str(raw_line or "").strip())
        if not line:
            continue
        ts_match = ts_re.search(line)
        if ts_match and not report_ts:
            report_ts = ts_match.group(1)
        upper = line.upper()
        if upper in team_names:
            current_team = _OFFICIAL_TEAM_NAME_MAP[upper]
            continue
        if not current_team:
            continue
        if any(tok in upper for tok in ["PLAYER NAME", "POSITION", "STATUS", "REASON", "NOT YET SUBMITTED"]):
            continue
        match = line_re.match(line)
        if not match:
            continue
        raw_status = match.group("status").strip()
        status = _OFFICIAL_STATUS_MAP.get(raw_status.lower(), raw_status.lower())
        player_name = re.sub(r"\s+", " ", match.group("name")).strip(" -")
        if len(player_name) < 3:
            continue
        rows.append(
            {
                "team": current_team,
                "player_name": player_name,
                "status": status,
                "status_prob": _official_status_to_prob(raw_status),
                "report_date": report_ts or target_date,
                "source": "official_nba_pdf",
                "pdf_url": pdf_url,
                "cache_path": str(pdf_path),
            }
        )
    return rows


def fetch_official_nba_injury_report(target_date: str) -> list[dict[str, Any]]:
    """Fetch and parse official NBA injury report PDFs for the target date."""
    OFFICIAL_INJURY_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for url in _fetch_official_injury_pdf_links(target_date):
        filename = Path(url).name
        cache_path = OFFICIAL_INJURY_DIR / filename
        if not cache_path.exists():
            try:
                resp = requests.get(url, timeout=OFFICIAL_INJURY_PAGE_TIMEOUT)
                resp.raise_for_status()
                cache_path.write_bytes(resp.content)
            except Exception as exc:
                print(f"  Warning: official injury PDF fetch failed for {filename}: {exc}", flush=True)
                continue
        rows.extend(_extract_official_injury_rows_from_pdf(cache_path, url, target_date))
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    df["status_pri"] = df["status"].map(
        {"out": 5, "suspension": 5, "doubtful": 4, "questionable": 3, "probable": 2}
    ).fillna(1)
    df = df.sort_values(["team", "player_name", "status_pri"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["team", "player_name"], keep="first")
    return df.drop(columns=["status_pri"]).to_dict("records")


def fetch_injury_status_map(target_date: str, asof_utc: pd.Timestamp | None = None) -> dict[str, dict[str, Any]]:
    """Fetch injury report rows indexed by `TEAM|normalized_player_name`."""
    injuries = fetch_espn_injury_report(cache_key=target_date)
    official_rows = fetch_official_nba_injury_report(target_date)
    fetched_at_utc = (asof_utc if asof_utc is not None else pd.Timestamp.now(tz="UTC")).isoformat()
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
            "espn_player_id": inj.get("espn_player_id"),
            "report_date": inj.get("report_date", ""),
            "fetched_at_utc": fetched_at_utc,
            "injury_source": "espn",
        }
    for inj in official_rows:
        team = str(inj.get("team", "")).upper()
        name = str(inj.get("player_name", ""))
        if not team or not name:
            continue
        key = _injury_key(team, name)
        status = str(inj.get("status", "")).strip().lower()
        avail_prob = _to_float(inj.get("status_prob"))
        if pd.isna(avail_prob):
            avail_prob = 0.5
        base = out.get(key, {})
        out[key] = {
            "team": team,
            "player_name": name,
            "status": status or str(base.get("status", "")),
            "availability_prob": float(np.clip(avail_prob, 0.0, 1.0)),
            "espn_player_id": base.get("espn_player_id"),
            "report_date": inj.get("report_date", "") or base.get("report_date", ""),
            "fetched_at_utc": fetched_at_utc,
            "injury_source": "official_nba_pdf",
            "pdf_url": inj.get("pdf_url", ""),
            "cache_path": inj.get("cache_path", ""),
        }
    return out


def _forward_pressure_clip_bounds(team_games: pd.DataFrame) -> tuple[float, float]:
    """Get robust clip bounds for forward pressure from historical training range."""
    candidate_cols = ["injury_proxy_missing_points5", "team_injury_pressure"]
    for col in candidate_cols:
        if col in team_games.columns:
            s = pd.to_numeric(team_games[col], errors="coerce").dropna()
            if len(s) >= 100:
                lo = float(s.quantile(0.01))
                hi = float(s.quantile(0.99))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    return lo, hi
    return 0.0, 60.0


def evaluate_injury_feed_coverage(
    injury_status_map: dict[str, dict[str, Any]],
    teams: list[str],
    now_utc: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Assess injury-feed freshness/coverage, globally and per-team."""
    now_utc = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    by_team: dict[str, list[dict[str, Any]]] = {t: [] for t in teams}
    report_ts: list[pd.Timestamp] = []
    for inj in injury_status_map.values():
        team = str(inj.get("team", "")).upper()
        if team in by_team:
            by_team[team].append(inj)
        rts = pd.to_datetime(inj.get("report_date", ""), utc=True, errors="coerce")
        if pd.notna(rts):
            report_ts.append(rts)

    max_report_age_hours = np.nan
    if report_ts:
        latest_report = max(report_ts)
        max_report_age_hours = float((now_utc - latest_report).total_seconds() / 3600.0)

    global_reasons: list[str] = []
    total_rows = len(injury_status_map)
    if total_rows < INJURY_FEED_MIN_ROWS:
        global_reasons.append(f"low_rows:{total_rows}")
    if pd.notna(max_report_age_hours) and max_report_age_hours > INJURY_FEED_STALE_HOURS:
        global_reasons.append(f"stale_hours:{max_report_age_hours:.1f}")
    if total_rows == 0:
        global_reasons.append("empty_feed")
    global_stale = bool(global_reasons)

    team_meta: dict[str, dict[str, Any]] = {}
    for team in teams:
        rows = by_team.get(team, [])
        out_doubt = sum(
            str(r.get("status", "")).strip().lower() in {"out", "doubtful", "suspension"}
            for r in rows
        )
        team_reasons: list[str] = []
        if global_stale:
            team_reasons.append("global_stale")
        if len(rows) == 0:
            team_reasons.append("no_team_rows")
        if len(rows) > 0 and out_doubt == 0:
            team_reasons.append("zero_out_doubtful")
        team_meta[team] = {
            "rows": int(len(rows)),
            "out_doubtful_count": int(out_doubt),
            "zero_out_doubtful": int(out_doubt == 0),
            "stale": int(bool(team_reasons)),
            "stale_reason": "|".join(team_reasons),
        }

    return {
        "global_stale": int(global_stale),
        "global_reason": "|".join(global_reasons),
        "max_report_age_hours": max_report_age_hours,
        "total_rows": int(total_rows),
        "teams": team_meta,
    }


def compute_forward_injury_pressure(
    injury_status_map: dict[str, dict[str, Any]],
    player_games: pd.DataFrame,
    teams: list[str],
    clip_bounds: tuple[float, float],
) -> dict[str, dict[str, Any]]:
    """Compute forward-looking team injury pressure using current injury report + recent form."""
    teams_u = [str(t).upper() for t in teams]
    base = {
        t: {
            "fwd_missing_points": 0.0,
            "fwd_missing_minutes": 0.0,
            "fwd_missing_assists": 0.0,
            "fwd_missing_rebounds": 0.0,
            "fwd_star_absent_flag": 0,
            "fwd_top1_missing_points": 0.0,
            "fwd_top2_missing_points": 0.0,
            "top_players_out": [],
        }
        for t in teams_u
    }
    if not injury_status_map or player_games.empty:
        return base

    pg = player_games.copy()
    if "played" in pg.columns:
        pg = pg[pg["played"] == 1].copy()
    if pg.empty:
        return base
    if "game_time_utc" not in pg.columns or "player_id" not in pg.columns or "team" not in pg.columns:
        return base

    pg["team"] = pg["team"].astype(str).str.upper()
    pg = pg[pg["team"].isin(teams_u)].copy()
    if pg.empty:
        return base
    pg = pg.sort_values(["team", "player_id", "game_time_utc", "game_id"])
    pg["name_norm"] = pg["player_name"].map(normalize_player_name)
    if "espn_player_id" in pg.columns:
        pg["espn_player_id"] = pd.to_numeric(pg["espn_player_id"], errors="coerce")

    def _tail_mean(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return np.nan
        s = pd.to_numeric(df[col], errors="coerce")
        return float(s.mean()) if s.notna().any() else np.nan

    snapshots: list[dict[str, Any]] = []
    for (team, pid), grp in pg.groupby(["team", "player_id"], sort=False):
        tail = grp.tail(5)
        latest = grp.iloc[-1]
        snapshots.append({
            "team": team,
            "player_id": int(pid),
            "player_name": str(latest.get("player_name", "")),
            "name_norm": str(latest.get("name_norm", "")),
            "espn_player_id": _to_float(latest.get("espn_player_id", np.nan)),
            "pre_points_avg5": _tail_mean(tail, "points"),
            "pre_minutes_avg5": _tail_mean(tail, "minutes"),
            "pre_assists_avg5": _tail_mean(tail, "assists"),
            "pre_rebounds_avg5": _tail_mean(tail, "rebounds"),
        })

    snap_df = pd.DataFrame(snapshots)
    if snap_df.empty:
        return base

    # Team "star" reference = top 2 by recent points.
    top2_by_team: dict[str, set[int]] = {}
    for team, grp in snap_df.groupby("team"):
        top2 = grp.sort_values("pre_points_avg5", ascending=False).head(2)
        top2_by_team[team] = set(top2["player_id"].astype(int).tolist())

    lo_clip, hi_clip = clip_bounds
    for inj in injury_status_map.values():
        team = str(inj.get("team", "")).upper()
        if team not in base:
            continue
        status = str(inj.get("status", "")).strip().lower()
        unavail = 1.0 - float(np.clip(_nan_or(inj.get("availability_prob"), 0.5), 0.0, 1.0))
        if status not in INJURY_FORWARD_STATUSES and unavail < INJURY_FORWARD_MIN_UNAVAIL:
            continue

        team_rows = snap_df[snap_df["team"] == team]
        if team_rows.empty:
            continue

        matched = pd.DataFrame()
        espn_id = _to_float(inj.get("espn_player_id"))
        if pd.notna(espn_id) and "espn_player_id" in team_rows.columns:
            matched = team_rows[team_rows["espn_player_id"] == espn_id]

        if matched.empty:
            nn = normalize_player_name(inj.get("player_name", ""))
            if nn:
                matched = team_rows[team_rows["name_norm"] == nn]

        if matched.empty:
            continue

        m = matched.sort_values("pre_minutes_avg5", ascending=False).iloc[0]
        miss_pts = float(_nan_or(m.get("pre_points_avg5"), 0.0) * unavail)
        miss_min = float(_nan_or(m.get("pre_minutes_avg5"), 0.0) * unavail)
        miss_ast = float(_nan_or(m.get("pre_assists_avg5"), 0.0) * unavail)
        miss_reb = float(_nan_or(m.get("pre_rebounds_avg5"), 0.0) * unavail)
        is_star = int(int(m.get("player_id", -1)) in top2_by_team.get(team, set()))

        base[team]["fwd_missing_points"] += miss_pts
        base[team]["fwd_missing_minutes"] += miss_min
        base[team]["fwd_missing_assists"] += miss_ast
        base[team]["fwd_missing_rebounds"] += miss_reb
        if is_star:
            base[team]["fwd_star_absent_flag"] = 1
        base[team]["top_players_out"].append({
            "player_name": str(m.get("player_name", inj.get("player_name", ""))),
            "status": status,
            "availability_prob": float(1.0 - unavail),
            "unavailability_prob": float(unavail),
            "missing_points": round(miss_pts, 2),
            "missing_minutes": round(miss_min, 2),
            "is_team_top2_scorer": is_star,
        })

    for team in teams_u:
        base[team]["fwd_missing_points"] = float(np.clip(base[team]["fwd_missing_points"], lo_clip, hi_clip))
        # minutes tracks points scale to preserve proportionality after clipping
        base[team]["fwd_missing_minutes"] = float(max(0.0, base[team]["fwd_missing_minutes"]))
        players = sorted(base[team]["top_players_out"], key=lambda d: d.get("missing_points", 0.0), reverse=True)
        base[team]["top_players_out"] = players
        base[team]["fwd_top1_missing_points"] = float(players[0]["missing_points"]) if len(players) >= 1 else 0.0
        base[team]["fwd_top2_missing_points"] = float(sum(p["missing_points"] for p in players[:2])) if players else 0.0

    return base


def save_injury_snapshot(
    target_date: str,
    teams: list[str],
    injury_status_map: dict[str, dict[str, Any]],
    coverage: dict[str, Any],
    forward_pressure: dict[str, dict[str, Any]],
    asof_utc: pd.Timestamp | None = None,
) -> Path:
    """Persist a run-level injury snapshot for audit/debug."""
    PROP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = PROP_LOG_DIR / f"injury_snapshot_{target_date}.json"
    rows = sorted(
        [
            {
                "team": str(v.get("team", "")).upper(),
                "player_name": v.get("player_name", ""),
                "espn_player_id": v.get("espn_player_id"),
                "status": v.get("status", ""),
                "availability_prob": v.get("availability_prob"),
                "report_date": v.get("report_date", ""),
                "fetched_at_utc": v.get("fetched_at_utc", ""),
                "injury_source": v.get("injury_source", ""),
                "pdf_url": v.get("pdf_url", ""),
            }
            for v in injury_status_map.values()
        ],
        key=lambda r: (r["team"], normalize_player_name(r["player_name"])),
    )
    asof_utc = asof_utc if asof_utc is not None else pd.Timestamp.now(tz="UTC")
    payload = {
        "target_date": str(target_date),
        "generated_at_utc": asof_utc.isoformat(),
        "coverage": coverage,
        "teams": [str(t).upper() for t in teams],
        "injuries": rows,
        "forward_pressure": forward_pressure,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


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
    now_utc: pd.Timestamp | None = None,
) -> dict[tuple[str, str, str], dict[str, set[str]]]:
    """Fetch confirmed starters for games near tip.

    Returns mapping key=(game_date_est, home_team, away_team) -> {team: starters}.
    """
    if upcoming.empty:
        return {}
    now_utc = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
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


def _american_odds_to_implied_prob(odds: Any) -> float:
    """Convert American odds to implied probability."""
    if pd.isna(odds) or odds is None:
        return np.nan
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return np.nan
    if odds < 0:
        return (-odds) / (-odds + 100.0)
    elif odds > 0:
        return 100.0 / (odds + 100.0)
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
        fetched_any = False
        for i, gid in enumerate(missing_ids, start=1):
            cache_path = BOX_ADV_CACHE_DIR / f"{gid}.json"
            try:
                _stats_nba_fetch_json(
                    STATS_NBA_ADV_URL,
                    params={"GameID": gid},
                    cache_path=cache_path,
                )
                fetched_any = True
                if i < len(missing_ids):
                    time.sleep(BOX_ADV_REQUEST_SLEEP_SECS)
            except Exception as exc:
                print(f"  Warning: BoxScoreAdvancedV3 fetch failed for {gid}: {exc}", flush=True)
        # Invalidate in-memory cache so newly fetched games are visible in this run.
        if fetched_any:
            _BOX_ADV_CACHE = None

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


# ---------------------------------------------------------------------------
# Player Tracking Stats (touches, drives, catch-and-shoot, speed, distance)
# ---------------------------------------------------------------------------
STATS_NBA_TRACKING_URL = "https://stats.nba.com/stats/boxscoreplayertrackv3"
TRACKING_CACHE_DIR = PROP_CACHE_DIR / "player_tracking_raw"
_TRACKING_CACHE: pd.DataFrame | None = None
USE_TRACKING_FEATURES = True

# ---------------------------------------------------------------------------
# Basketball Reference Advanced Stats (BRef)
# ---------------------------------------------------------------------------
BREF_CACHE_DIR = OUT_DIR / "bref_cache"
USE_BREF_FEATURES = True
_BREF_GAME_LOGS_CACHE: pd.DataFrame | None = None
_BREF_OPP_STATS_CACHE: dict[str, dict[str, Any]] | None = None

BREF_ADV_COLS = [
    "bref_adv_usg_pct",
    "bref_adv_off_rtg",
    "bref_adv_def_rtg",
    "bref_adv_bpm",
    "bref_adv_ts_pct",
    "bref_adv_efg_pct",
    "bref_adv_ast_pct",
    "bref_adv_trb_pct",
    "bref_adv_stl_pct",
    "bref_adv_blk_pct",
    "bref_adv_orb_pct",
    "bref_adv_drb_pct",
    "bref_adv_tov_pct",
]

# BRef opponent defensive stats to use as per-game features
BREF_OPP_COLS = [
    "bref_opp_pts_per_g",
    "bref_opp_fg_pct",
    "bref_opp_fg3_pct",
    "bref_opp_ft_per_g",
    "bref_opp_trb_per_g",
    "bref_opp_ast_per_g",
    "bref_opp_tov_per_g",
    "bref_opp_stl_per_g",
    "bref_opp_blk_per_g",
    # Opponent shooting profile
    "bref_opp_avg_dist",
    "bref_opp_pct_fga_3p",
    "bref_opp_fg_pct_0_3",
    "bref_opp_fg_pct_16_3pt",
]


def load_bref_player_game_logs(seasons: list[str] | None = None) -> pd.DataFrame:
    """Load BRef player game logs with advanced stats. Returns DataFrame.

    Joins on game_date_est (YYYYMMDD) + team + normalized player name.
    Results are cached in memory for the session.
    """
    global _BREF_GAME_LOGS_CACHE
    if _BREF_GAME_LOGS_CACHE is not None:
        return _BREF_GAME_LOGS_CACHE

    if seasons is None:
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    frames: list[pd.DataFrame] = []
    for season in seasons:
        csv_path = BREF_CACHE_DIR / season / "csv" / "player_game_logs.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df["season"] = season
        frames.append(df)
    if not frames:
        _BREF_GAME_LOGS_CACHE = pd.DataFrame()
        return _BREF_GAME_LOGS_CACHE

    df = pd.concat(frames, ignore_index=True)

    # Normalize join keys
    df["game_date_est"] = df["game_date"].str.replace("-", "", regex=False)
    df["player_name_norm"] = df["player_name"].apply(
        lambda n: normalize_player_name(n) if pd.notna(n) else n
    )

    # Rename BRef advanced columns to our namespace to avoid collisions
    rename_map = {
        "adv_usg_pct": "bref_adv_usg_pct",
        "adv_off_rtg": "bref_adv_off_rtg",
        "adv_def_rtg": "bref_adv_def_rtg",
        "adv_bpm": "bref_adv_bpm",
        "adv_ts_pct": "bref_adv_ts_pct",
        "adv_efg_pct": "bref_adv_efg_pct",
        "adv_ast_pct": "bref_adv_ast_pct",
        "adv_trb_pct": "bref_adv_trb_pct",
        "adv_stl_pct": "bref_adv_stl_pct",
        "adv_blk_pct": "bref_adv_blk_pct",
        "adv_orb_pct": "bref_adv_orb_pct",
        "adv_drb_pct": "bref_adv_drb_pct",
        "adv_tov_pct": "bref_adv_tov_pct",
    }
    df = df.rename(columns=rename_map)

    _BREF_GAME_LOGS_CACHE = df
    return _BREF_GAME_LOGS_CACHE


def load_bref_opponent_stats(seasons: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Load BRef opponent (defensive) stats per team per season.

    Returns {season: {team: {stat: value}}}.
    Merges opponent_stats.json + opponent_shooting.json.
    """
    global _BREF_OPP_STATS_CACHE
    if _BREF_OPP_STATS_CACHE is not None:
        return _BREF_OPP_STATS_CACHE

    if seasons is None:
        seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    result: dict[str, dict[str, Any]] = {}
    for season in seasons:
        path = BREF_CACHE_DIR / season / "opponent_stats.json"
        if path.exists():
            result[season] = json.loads(path.read_text())
        opp_shoot_path = BREF_CACHE_DIR / season / "opponent_shooting.json"
        if opp_shoot_path.exists():
            opp_shoot = json.loads(opp_shoot_path.read_text())
            if season not in result:
                result[season] = {}
            for team, stats in opp_shoot.items():
                if team in result[season]:
                    result[season][team].update(stats)
                else:
                    result[season][team] = dict(stats)
    _BREF_OPP_STATS_CACHE = result
    return _BREF_OPP_STATS_CACHE


def _merge_bref_advanced_stats(pg: pd.DataFrame) -> pd.DataFrame:
    """Merge BRef advanced stats onto player-game rows.

    Joins on game_date_est + team + normalized player name.
    Fills NaN values in existing adv_ columns from BRef equivalents,
    and adds new BRef-only columns (def_rtg, bpm, efg, stl_pct, blk_pct, etc.).
    """
    if not USE_BREF_FEATURES:
        for c in BREF_ADV_COLS:
            if c not in pg.columns:
                pg[c] = np.nan
        return pg

    bref = load_bref_player_game_logs()
    if bref.empty:
        print("  BRef: no game logs found, skipping BRef merge.", flush=True)
        for c in BREF_ADV_COLS:
            if c not in pg.columns:
                pg[c] = np.nan
        return pg

    # Prepare join key on pg side
    if "game_date_est" not in pg.columns:
        print("  BRef: game_date_est not in pg, skipping BRef merge.", flush=True)
        for c in BREF_ADV_COLS:
            if c not in pg.columns:
                pg[c] = np.nan
        return pg

    pg["_bref_date"] = pg["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    pg["_bref_name"] = pg["player_name"].map(normalize_player_name)

    # Select BRef columns for merge
    bref_merge_cols = ["game_date_est", "team", "player_name_norm"] + [
        c for c in BREF_ADV_COLS if c in bref.columns
    ]
    bref_slim = bref[bref_merge_cols].drop_duplicates(
        subset=["game_date_est", "team", "player_name_norm"]
    ).copy()
    bref_slim = bref_slim.rename(columns={
        "game_date_est": "_bref_date",
        "player_name_norm": "_bref_name",
    })

    # Merge
    pg = pg.merge(bref_slim, on=["_bref_date", "team", "_bref_name"], how="left", suffixes=("", "_bref_dup"))

    # Fill NaN in existing adv_ columns from BRef equivalents
    _fill_map = {
        "adv_usage_pct": "bref_adv_usg_pct",
        "adv_off_rating": "bref_adv_off_rtg",
        "adv_ts_pct": "bref_adv_ts_pct",
        "adv_ast_pct": "bref_adv_ast_pct",
        "adv_reb_pct": "bref_adv_trb_pct",
    }
    for existing_col, bref_col in _fill_map.items():
        if existing_col in pg.columns and bref_col in pg.columns:
            pg[existing_col] = pg[existing_col].fillna(pg[bref_col])

    # Ensure all BRef columns exist
    for c in BREF_ADV_COLS:
        if c not in pg.columns:
            pg[c] = np.nan

    # Clean up temp columns
    pg.drop(columns=["_bref_date", "_bref_name"], inplace=True, errors="ignore")
    # Drop any _bref_dup columns from merge collisions
    bref_dup_cols = [c for c in pg.columns if c.endswith("_bref_dup")]
    if bref_dup_cols:
        pg.drop(columns=bref_dup_cols, inplace=True, errors="ignore")

    _bref_avail = pg[BREF_ADV_COLS].notna().any(axis=1).mean() * 100
    print(f"  Merged BRef advanced stats: {_bref_avail:.0f}% row coverage", flush=True)

    return pg


def _merge_bref_opponent_defense(pg: pd.DataFrame) -> pd.DataFrame:
    """Merge BRef opponent defensive stats as static per-game features.

    Joins on season + opponent team. These are season-level aggregates
    (e.g., opponent points per game, opponent FG%), so they are the same
    for every game in a season against that team.
    """
    if not USE_BREF_FEATURES:
        for c in BREF_OPP_COLS:
            if c not in pg.columns:
                pg[c] = np.nan
        return pg

    opp_stats = load_bref_opponent_stats()
    if not opp_stats:
        for c in BREF_OPP_COLS:
            if c not in pg.columns:
                pg[c] = np.nan
        return pg

    if "season" not in pg.columns or "opp" not in pg.columns:
        for c in BREF_OPP_COLS:
            if c not in pg.columns:
                pg[c] = np.nan
        return pg

    # Build a flat DataFrame from the nested dict: season x team -> stats
    opp_rows: list[dict[str, Any]] = []
    # Map from BRef stat names to our feature names
    _stat_rename = {
        "opp_pts_per_g": "bref_opp_pts_per_g",
        "opp_fg_pct": "bref_opp_fg_pct",
        "opp_fg3_pct": "bref_opp_fg3_pct",
        "opp_ft_per_g": "bref_opp_ft_per_g",
        "opp_trb_per_g": "bref_opp_trb_per_g",
        "opp_ast_per_g": "bref_opp_ast_per_g",
        "opp_tov_per_g": "bref_opp_tov_per_g",
        "opp_stl_per_g": "bref_opp_stl_per_g",
        "opp_blk_per_g": "bref_opp_blk_per_g",
        # From opponent_shooting.json
        "opp_avg_dist": "bref_opp_avg_dist",
        "opp_pct_fga_3p": "bref_opp_pct_fga_3p",
        "opp_fg_pct_0_3": "bref_opp_fg_pct_0_3",
        "opp_fg_pct_16_3pt": "bref_opp_fg_pct_16_3pt",
    }
    for season, teams in opp_stats.items():
        for team, stats in teams.items():
            row: dict[str, Any] = {"season": season, "opp": team}
            for bref_key, feat_name in _stat_rename.items():
                row[feat_name] = stats.get(bref_key, np.nan)
            opp_rows.append(row)

    if not opp_rows:
        for c in BREF_OPP_COLS:
            if c not in pg.columns:
                pg[c] = np.nan
        return pg

    opp_df = pd.DataFrame(opp_rows)
    pg = pg.merge(opp_df, on=["season", "opp"], how="left")

    for c in BREF_OPP_COLS:
        if c not in pg.columns:
            pg[c] = np.nan

    _opp_avail = pg[BREF_OPP_COLS].notna().any(axis=1).mean() * 100
    print(f"  Merged BRef opponent stats: {_opp_avail:.0f}% row coverage", flush=True)

    return pg


def _parse_tracking_payload(payload: dict[str, Any], game_id: str) -> list[dict[str, Any]]:
    """Parse player tracking stats from a BoxScorePlayerTrackV3 payload."""
    rows: list[dict[str, Any]] = []

    # Shape A: resultSets style (stats.nba.com standard)
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
            touches_i = idx("TOUCHES")
            drives_i = idx("DRIVES")
            passes_i = idx("PASSES")
            c_and_s_fga_i = idx("CATCH_SHOOT_FGA")
            c_and_s_fgm_i = idx("CATCH_SHOOT_FGM")
            c_and_s_fg3a_i = idx("CATCH_SHOOT_FG3A")
            c_and_s_fg3m_i = idx("CATCH_SHOOT_FG3M")
            pull_up_fga_i = idx("PULL_UP_FGA")
            pull_up_fgm_i = idx("PULL_UP_FGM")
            pull_up_fg3a_i = idx("PULL_UP_FG3A")
            dist_i = idx("DIST_MILES", "DIST")
            speed_i = idx("AVG_SPEED", "SPEED")
            cont_shots_i = idx("CONTESTED_SHOTS", "CONT_SHOTS")
            uncontested_fga_i = idx("UNCONTESTED_FGA")
            # Hustle stats (from BDL advanced)
            deflections_i = idx("DEFLECTIONS")
            box_outs_i = idx("BOX_OUTS")
            off_box_outs_i = idx("OFFENSIVE_BOX_OUTS")
            def_box_outs_i = idx("DEFENSIVE_BOX_OUTS")
            loose_balls_i = idx("LOOSE_BALLS")
            screen_ast_i = idx("SCREEN_ASSISTS")
            secondary_ast_i = idx("SECONDARY_ASSISTS")
            # Rebound chances
            reb_chances_i = idx("REB_CHANCES_TOTAL")
            reb_chances_off_i = idx("REB_CHANCES_OFF")
            reb_chances_def_i = idx("REB_CHANCES_DEF")
            # Scoring breakdown
            pts_paint_i = idx("PTS_PAINT")
            pts_fb_i = idx("PTS_FAST_BREAK")
            pts_off_to_i = idx("PTS_OFF_TO")
            if pid_i < 0 or team_i < 0:
                continue

            def _safe(r: list | tuple, i: int) -> Any:
                return _to_float(r[i]) if 0 <= i < len(r) else np.nan

            for r in rowset:
                if not isinstance(r, (list, tuple)):
                    continue
                pid = r[pid_i] if pid_i < len(r) else None
                team = str(r[team_i]).upper().strip() if team_i < len(r) else ""
                if pid is None or not team:
                    continue
                row_dict: dict[str, Any] = {
                    "game_id": str(game_id),
                    "team": team,
                    "player_id": int(pid),
                    "trk_touches": _safe(r, touches_i),
                    "trk_drives": _safe(r, drives_i),
                    "trk_passes": _safe(r, passes_i),
                    "trk_catch_shoot_fga": _safe(r, c_and_s_fga_i),
                    "trk_catch_shoot_fgm": _safe(r, c_and_s_fgm_i),
                    "trk_catch_shoot_fg3a": _safe(r, c_and_s_fg3a_i),
                    "trk_catch_shoot_fg3m": _safe(r, c_and_s_fg3m_i),
                    "trk_pull_up_fga": _safe(r, pull_up_fga_i),
                    "trk_pull_up_fgm": _safe(r, pull_up_fgm_i),
                    "trk_pull_up_fg3a": _safe(r, pull_up_fg3a_i),
                    "trk_dist_miles": _safe(r, dist_i),
                    "trk_avg_speed": _safe(r, speed_i),
                    "trk_contested_shots": _safe(r, cont_shots_i),
                    "trk_uncontested_fga": _safe(r, uncontested_fga_i),
                    # Hustle stats
                    "trk_deflections": _safe(r, deflections_i),
                    "trk_box_outs": _safe(r, box_outs_i),
                    "trk_off_box_outs": _safe(r, off_box_outs_i),
                    "trk_def_box_outs": _safe(r, def_box_outs_i),
                    "trk_loose_balls": _safe(r, loose_balls_i),
                    "trk_screen_assists": _safe(r, screen_ast_i),
                    "trk_secondary_assists": _safe(r, secondary_ast_i),
                    # Rebound chances
                    "trk_reb_chances": _safe(r, reb_chances_i),
                    "trk_reb_chances_off": _safe(r, reb_chances_off_i),
                    "trk_reb_chances_def": _safe(r, reb_chances_def_i),
                    # Scoring breakdown
                    "trk_pts_paint": _safe(r, pts_paint_i),
                    "trk_pts_fast_break": _safe(r, pts_fb_i),
                    "trk_pts_off_to": _safe(r, pts_off_to_i),
                }
                rows.append(row_dict)
        if rows:
            return rows

    # Shape B: nested homeTeam/awayTeam (newer NBA CDN format)
    trk_root = payload.get("boxScorePlayerTrack") or payload.get("boxscoreplayertrackv3") or payload
    for side_key in ("homeTeam", "awayTeam"):
        side = trk_root.get(side_key) if isinstance(trk_root, dict) else None
        if not isinstance(side, dict):
            continue
        team = str(
            side.get("teamTricode") or side.get("teamCode") or side.get("teamAbbreviation") or ""
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
            rows.append({
                "game_id": str(game_id),
                "team": team,
                "player_id": int(pid),
                "trk_touches": _to_float(stats.get("touches") or stats.get("TOUCHES")),
                "trk_drives": _to_float(stats.get("drives") or stats.get("DRIVES")),
                "trk_passes": _to_float(stats.get("passes") or stats.get("PASSES")),
                "trk_catch_shoot_fga": _to_float(stats.get("catchShootFga") or stats.get("CATCH_SHOOT_FGA")),
                "trk_catch_shoot_fgm": _to_float(stats.get("catchShootFgm") or stats.get("CATCH_SHOOT_FGM")),
                "trk_catch_shoot_fg3a": _to_float(stats.get("catchShootFg3a") or stats.get("CATCH_SHOOT_FG3A")),
                "trk_catch_shoot_fg3m": _to_float(stats.get("catchShootFg3m") or stats.get("CATCH_SHOOT_FG3M")),
                "trk_pull_up_fga": _to_float(stats.get("pullUpFga") or stats.get("PULL_UP_FGA")),
                "trk_pull_up_fgm": _to_float(stats.get("pullUpFgm") or stats.get("PULL_UP_FGM")),
                "trk_pull_up_fg3a": _to_float(stats.get("pullUpFg3a") or stats.get("PULL_UP_FG3A")),
                "trk_dist_miles": _to_float(stats.get("distanceMiles") or stats.get("DIST_MILES") or stats.get("dist")),
                "trk_avg_speed": _to_float(stats.get("averageSpeed") or stats.get("AVG_SPEED") or stats.get("speed")),
                "trk_contested_shots": _to_float(stats.get("contestedShots") or stats.get("CONTESTED_SHOTS")),
                "trk_uncontested_fga": _to_float(stats.get("uncontestedFga") or stats.get("UNCONTESTED_FGA")),
                # Hustle stats (populated by BDL via Shape A; CDN keys here as fallback)
                "trk_deflections": _to_float(stats.get("deflections") or stats.get("DEFLECTIONS")),
                "trk_box_outs": _to_float(stats.get("boxOuts") or stats.get("BOX_OUTS")),
                "trk_off_box_outs": _to_float(stats.get("offensiveBoxOuts") or stats.get("OFFENSIVE_BOX_OUTS")),
                "trk_def_box_outs": _to_float(stats.get("defensiveBoxOuts") or stats.get("DEFENSIVE_BOX_OUTS")),
                "trk_loose_balls": _to_float(stats.get("looseBallsRecoveredTotal") or stats.get("LOOSE_BALLS")),
                "trk_screen_assists": _to_float(stats.get("screenAssists") or stats.get("SCREEN_ASSISTS")),
                "trk_secondary_assists": _to_float(stats.get("secondaryAssists") or stats.get("SECONDARY_ASSISTS")),
                "trk_reb_chances": _to_float(stats.get("reboundChancesTotal") or stats.get("REB_CHANCES_TOTAL")),
                "trk_reb_chances_off": _to_float(stats.get("reboundChancesOff") or stats.get("REB_CHANCES_OFF")),
                "trk_reb_chances_def": _to_float(stats.get("reboundChancesDef") or stats.get("REB_CHANCES_DEF")),
                "trk_pts_paint": _to_float(stats.get("pointsPaint") or stats.get("PTS_PAINT")),
                "trk_pts_fast_break": _to_float(stats.get("pointsFastBreak") or stats.get("PTS_FAST_BREAK")),
                "trk_pts_off_to": _to_float(stats.get("pointsOffTurnovers") or stats.get("PTS_OFF_TO")),
            })
    return rows


def load_player_tracking_stats(
    game_ids: list[str] | pd.Series | np.ndarray | None = None,
    fetch_missing: bool = False,
    max_fetch: int = 0,
) -> pd.DataFrame:
    """Load player tracking stats from local cache, optionally fetching misses."""
    global _TRACKING_CACHE
    TRACKING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if game_ids is None:
        wanted_ids = None
    else:
        wanted_ids = sorted({str(g) for g in list(game_ids) if str(g).strip()})
        if not wanted_ids:
            return pd.DataFrame()

    if fetch_missing and wanted_ids:
        missing_ids = [gid for gid in wanted_ids if not (TRACKING_CACHE_DIR / f"{gid}.json").exists()]
        if max_fetch > 0:
            missing_ids = missing_ids[:max_fetch]
        fetched_any = False
        for i, gid in enumerate(missing_ids, start=1):
            cache_path = TRACKING_CACHE_DIR / f"{gid}.json"
            try:
                _stats_nba_fetch_json(
                    STATS_NBA_TRACKING_URL,
                    params={"GameID": gid},
                    cache_path=cache_path,
                )
                fetched_any = True
                if i < len(missing_ids):
                    time.sleep(BOX_ADV_REQUEST_SLEEP_SECS)
            except Exception as exc:
                print(f"  Warning: Tracking stats fetch failed for {gid}: {exc}", flush=True)
        if fetched_any:
            _TRACKING_CACHE = None

    # Build cache once per process from available raw files.
    if _TRACKING_CACHE is None:
        rows: list[dict[str, Any]] = []
        files = sorted(TRACKING_CACHE_DIR.glob("*.json"))
        for f in files:
            gid = f.stem
            try:
                payload = json.loads(f.read_text())
                rows.extend(_parse_tracking_payload(payload, gid))
            except Exception:
                continue
        if not rows:
            _TRACKING_CACHE = pd.DataFrame()
        else:
            df = pd.DataFrame(rows).drop_duplicates(subset=["game_id", "team", "player_id"], keep="first")
            _TRACKING_CACHE = df.reset_index(drop=True)

    if _TRACKING_CACHE is None or _TRACKING_CACHE.empty:
        return pd.DataFrame()
    out = _TRACKING_CACHE
    if wanted_ids is not None:
        out = out[out["game_id"].astype(str).isin(wanted_ids)]
    return out.copy()


# ---------------------------------------------------------------------------
# Defensive + Scoring boxscore stats (from stats.nba.com)
# ---------------------------------------------------------------------------
DEFENSIVE_CACHE_DIR = PROP_CACHE_DIR / "boxscore_defensive_raw"
SCORING_CACHE_DIR = PROP_CACHE_DIR / "boxscore_scoring_raw"
_DEFENSIVE_CACHE: pd.DataFrame | None = None
_SCORING_CACHE: pd.DataFrame | None = None


def _parse_minutes_str(v: Any) -> float:
    """Parse 'MM:SS' or 'PTXXMYY.ZZS' to float minutes."""
    if v is None:
        return np.nan
    s = str(v).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return np.nan
    if s.startswith("PT") and s.endswith("S"):
        s = s[2:-1]
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
        return np.nan


def load_defensive_stats(
    game_ids: list[str] | pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Load boxscoredefensivev2 data from local cache."""
    global _DEFENSIVE_CACHE
    if _DEFENSIVE_CACHE is None:
        rows: list[dict[str, Any]] = []
        if DEFENSIVE_CACHE_DIR.is_dir():
            for f in sorted(DEFENSIVE_CACHE_DIR.glob("*.json")):
                gid = f.stem
                try:
                    data = json.loads(f.read_text())
                except Exception:
                    continue
                if "empty" in data:
                    continue
                root = data.get("boxScoreDefensive") or data
                for side_key in ("homeTeam", "awayTeam"):
                    side = root.get(side_key)
                    if not isinstance(side, dict):
                        continue
                    team = str(
                        side.get("teamTricode") or side.get("teamCode") or ""
                    ).upper().strip()
                    for p in side.get("players") or []:
                        if not isinstance(p, dict):
                            continue
                        pid = p.get("personId") or p.get("playerId")
                        if pid is None:
                            continue
                        st = p.get("statistics") or {}
                        rows.append({
                            "game_id": str(gid),
                            "team": team,
                            "player_id": int(pid),
                            "def_matchup_fga": _to_float(st.get("matchupFieldGoalsAttempted")),
                            "def_matchup_fgm": _to_float(st.get("matchupFieldGoalsMade")),
                            "def_matchup_fg_pct": _to_float(st.get("matchupFieldGoalPercentage")),
                            "def_matchup_3pa": _to_float(st.get("matchupThreePointersAttempted")),
                            "def_matchup_3pm": _to_float(st.get("matchupThreePointersMade")),
                            "def_matchup_3pt_pct": _to_float(st.get("matchupThreePointerPercentage")),
                            "def_matchup_minutes": _parse_minutes_str(st.get("matchupMinutes")),
                            "def_matchup_assists": _to_float(st.get("matchupAssists")),
                            "def_matchup_tov": _to_float(st.get("matchupTurnovers")),
                            "def_matchup_player_pts": _to_float(st.get("playerPoints")),
                            "def_switches_on": _to_float(st.get("switchesOn")),
                            "def_partial_poss": _to_float(st.get("partialPossessions")),
                        })
        if not rows:
            _DEFENSIVE_CACHE = pd.DataFrame()
        else:
            df = pd.DataFrame(rows).drop_duplicates(
                subset=["game_id", "team", "player_id"], keep="first"
            )
            _DEFENSIVE_CACHE = df.reset_index(drop=True)

    if _DEFENSIVE_CACHE is None or _DEFENSIVE_CACHE.empty:
        return pd.DataFrame()
    out = _DEFENSIVE_CACHE
    if game_ids is not None:
        wanted = sorted({str(g) for g in list(game_ids) if str(g).strip()})
        out = out[out["game_id"].astype(str).isin(wanted)]
    return out.copy()


def load_scoring_stats(
    game_ids: list[str] | pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Load boxscorescoringv3 data from local cache."""
    global _SCORING_CACHE
    if _SCORING_CACHE is None:
        rows: list[dict[str, Any]] = []
        if SCORING_CACHE_DIR.is_dir():
            for f in sorted(SCORING_CACHE_DIR.glob("*.json")):
                gid = f.stem
                try:
                    data = json.loads(f.read_text())
                except Exception:
                    continue
                if "empty" in data:
                    continue
                root = data.get("boxScoreScoring") or data
                for side_key in ("homeTeam", "awayTeam"):
                    side = root.get(side_key)
                    if not isinstance(side, dict):
                        continue
                    team = str(
                        side.get("teamTricode") or side.get("teamCode") or ""
                    ).upper().strip()
                    for p in side.get("players") or []:
                        if not isinstance(p, dict):
                            continue
                        pid = p.get("personId") or p.get("playerId")
                        if pid is None:
                            continue
                        st = p.get("statistics") or {}
                        rows.append({
                            "game_id": str(gid),
                            "team": team,
                            "player_id": int(pid),
                            "scr_pct_assisted_2pt": _to_float(st.get("percentageAssisted2pt")),
                            "scr_pct_assisted_3pt": _to_float(st.get("percentageAssisted3pt")),
                            "scr_pct_assisted_fgm": _to_float(st.get("percentageAssistedFGM")),
                            "scr_pct_unassisted_2pt": _to_float(st.get("percentageUnassisted2pt")),
                            "scr_pct_unassisted_3pt": _to_float(st.get("percentageUnassisted3pt")),
                            "scr_pct_fga_2pt": _to_float(st.get("percentageFieldGoalsAttempted2pt")),
                            "scr_pct_fga_3pt": _to_float(st.get("percentageFieldGoalsAttempted3pt")),
                            "scr_pct_pts_2pt": _to_float(st.get("percentagePoints2pt")),
                            "scr_pct_pts_3pt": _to_float(st.get("percentagePoints3pt")),
                            "scr_pct_pts_ft": _to_float(st.get("percentagePointsFreeThrow")),
                            "scr_pct_pts_paint": _to_float(st.get("percentagePointsPaint")),
                            "scr_pct_pts_midrange": _to_float(st.get("percentagePointsMidrange2pt")),
                            "scr_pct_pts_fastbreak": _to_float(st.get("percentagePointsFastBreak")),
                            "scr_pct_pts_off_to": _to_float(st.get("percentagePointsOffTurnovers")),
                        })
        if not rows:
            _SCORING_CACHE = pd.DataFrame()
        else:
            df = pd.DataFrame(rows).drop_duplicates(
                subset=["game_id", "team", "player_id"], keep="first"
            )
            _SCORING_CACHE = df.reset_index(drop=True)

    if _SCORING_CACHE is None or _SCORING_CACHE.empty:
        return pd.DataFrame()
    out = _SCORING_CACHE
    if game_ids is not None:
        wanted = sorted({str(g) for g in list(game_ids) if str(g).strip()})
        out = out[out["game_id"].astype(str).isin(wanted)]
    return out.copy()


def _parse_rotation_time_to_minutes(v: Any) -> float:
    """Parse GameRotation in/out timestamps into elapsed minutes."""
    parsed = _parse_minutes_str(v)
    if pd.notna(parsed):
        return float(parsed)
    s = str(v or "").strip()
    if not s:
        return np.nan
    m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if not m:
        return np.nan
    parts = [int(x) for x in m.groups(default="0")]
    if len(parts) == 3:
        hh, mm, ss = parts
        return float(hh * 60 + mm + ss / 60.0)
    return float(parts[0] + parts[1] / 60.0)


def _parse_game_rotation_payload(payload: dict[str, Any], game_id: str) -> list[dict[str, Any]]:
    """Parse GameRotation payload into per-player stint aggregates."""
    agg: dict[tuple[str, int], dict[str, Any]] = {}
    result_sets = payload.get("resultSets")
    if not isinstance(result_sets, list):
        return []
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
        in_i = idx("IN_TIME_REAL", "IN_TIME")
        out_i = idx("OUT_TIME_REAL", "OUT_TIME")
        minutes_i = idx("PT_DIFF", "PLAY_TIME", "MINUTES")
        if pid_i < 0:
            continue
        for row in rowset:
            if not isinstance(row, (list, tuple)):
                continue
            pid = row[pid_i] if pid_i < len(row) else None
            if pid is None:
                continue
            team = ""
            if 0 <= team_i < len(row):
                team = str(row[team_i] or "").upper().strip()
            in_min = _parse_rotation_time_to_minutes(row[in_i]) if 0 <= in_i < len(row) else np.nan
            out_min = _parse_rotation_time_to_minutes(row[out_i]) if 0 <= out_i < len(row) else np.nan
            stint_min = np.nan
            if pd.notna(in_min) and pd.notna(out_min):
                stint_min = max(0.0, float(out_min - in_min))
            elif 0 <= minutes_i < len(row):
                stint_min = _parse_rotation_time_to_minutes(row[minutes_i])
            key = (team, int(pid))
            rec = agg.setdefault(
                key,
                {
                    "game_id": str(game_id),
                    "team": team,
                    "player_id": int(pid),
                    "rot_stints": 0.0,
                    "rot_total_stint_min": 0.0,
                    "rot_max_stint_min": 0.0,
                },
            )
            rec["rot_stints"] += 1.0
            if pd.notna(stint_min):
                rec["rot_total_stint_min"] += float(stint_min)
                rec["rot_max_stint_min"] = max(float(rec["rot_max_stint_min"]), float(stint_min))
    rows = list(agg.values())
    for rec in rows:
        stints = max(float(rec["rot_stints"]), 1.0)
        rec["rot_avg_stint_min"] = float(rec["rot_total_stint_min"]) / stints
    return rows


def load_game_rotation_stats(
    game_ids: list[str] | pd.Series | np.ndarray | None = None,
    fetch_missing: bool = False,
    max_fetch: int = 0,
) -> pd.DataFrame:
    """Load GameRotation data from local cache, optionally fetching misses."""
    global _GAME_ROTATION_CACHE
    GAME_ROTATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wanted_ids = None
    if game_ids is not None:
        wanted_ids = sorted({str(g) for g in list(game_ids) if str(g).strip()})
        if not wanted_ids:
            return pd.DataFrame()
    if fetch_missing and wanted_ids:
        missing_ids = [gid for gid in wanted_ids if not (GAME_ROTATION_CACHE_DIR / f"{gid}.json").exists()]
        if max_fetch > 0:
            missing_ids = missing_ids[:max_fetch]
        fetched_any = False
        for i, gid in enumerate(missing_ids, start=1):
            try:
                _stats_nba_fetch_json(
                    STATS_NBA_ROTATION_URL,
                    params={"GameID": gid},
                    cache_path=GAME_ROTATION_CACHE_DIR / f"{gid}.json",
                )
                fetched_any = True
                if i < len(missing_ids):
                    time.sleep(BOX_ADV_REQUEST_SLEEP_SECS)
            except Exception as exc:
                print(f"  Warning: GameRotation fetch failed for {gid}: {exc}", flush=True)
        if fetched_any:
            _GAME_ROTATION_CACHE = None
    if _GAME_ROTATION_CACHE is None:
        rows: list[dict[str, Any]] = []
        for f in sorted(GAME_ROTATION_CACHE_DIR.glob("*.json")):
            try:
                rows.extend(_parse_game_rotation_payload(json.loads(f.read_text()), f.stem))
            except Exception:
                continue
        _GAME_ROTATION_CACHE = (
            pd.DataFrame(rows).drop_duplicates(subset=["game_id", "player_id"], keep="first").reset_index(drop=True)
            if rows else pd.DataFrame()
        )
    if _GAME_ROTATION_CACHE is None or _GAME_ROTATION_CACHE.empty:
        return pd.DataFrame()
    out = _GAME_ROTATION_CACHE
    if wanted_ids is not None:
        out = out[out["game_id"].astype(str).isin(wanted_ids)]
    return out.copy()


def _parse_boxscore_matchups_payload(payload: dict[str, Any], game_id: str) -> list[dict[str, Any]]:
    """Parse BoxScoreMatchupsV3 payload into per-offensive-player aggregates."""
    result_sets = payload.get("resultSets")
    if not isinstance(result_sets, list):
        return []
    agg: dict[tuple[str, int], dict[str, Any]] = {}
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

        pid_i = idx("OFF_PLAYER_ID", "PLAYER_ID", "PERSON_ID")
        team_i = idx("OFF_TEAM_ABBREVIATION", "TEAM_ABBREVIATION", "TEAM_TRICODE")
        poss_i = idx("PARTIAL_POSSESSIONS", "PARTIAL_POSS", "POSS")
        fga_i = idx("MATCHUP_FGA", "FGA")
        fgm_i = idx("MATCHUP_FGM", "FGM")
        fg3a_i = idx("MATCHUP_3PA", "THREEPA", "FG3A")
        fg3m_i = idx("MATCHUP_3PM", "THREEPM", "FG3M")
        ast_i = idx("MATCHUP_AST", "AST")
        pts_i = idx("PLAYER_PTS", "MATCHUP_PTS", "PTS")
        if pid_i < 0:
            continue
        for row in rowset:
            if not isinstance(row, (list, tuple)):
                continue
            pid = row[pid_i] if pid_i < len(row) else None
            if pid is None:
                continue
            team = ""
            if 0 <= team_i < len(row):
                team = str(row[team_i] or "").upper().strip()
            key = (team, int(pid))
            rec = agg.setdefault(
                key,
                {
                    "game_id": str(game_id),
                    "team": team,
                    "player_id": int(pid),
                    "mtch_partial_poss": 0.0,
                    "mtch_fga": 0.0,
                    "mtch_fgm": 0.0,
                    "mtch_3pa": 0.0,
                    "mtch_3pm": 0.0,
                    "mtch_ast": 0.0,
                    "mtch_pts": 0.0,
                },
            )
            for col_name, col_idx in [
                ("mtch_partial_poss", poss_i),
                ("mtch_fga", fga_i),
                ("mtch_fgm", fgm_i),
                ("mtch_3pa", fg3a_i),
                ("mtch_3pm", fg3m_i),
                ("mtch_ast", ast_i),
                ("mtch_pts", pts_i),
            ]:
                if 0 <= col_idx < len(row):
                    rec[col_name] += float(_nan_or(_to_float(row[col_idx]), 0.0))
    rows = list(agg.values())
    for rec in rows:
        fga = float(rec["mtch_fga"])
        fg3a = float(rec["mtch_3pa"])
        rec["mtch_fg_pct"] = float(rec["mtch_fgm"]) / fga if fga > 0 else np.nan
        rec["mtch_3pt_pct"] = float(rec["mtch_3pm"]) / fg3a if fg3a > 0 else np.nan
    return rows


def load_boxscore_matchups_stats(
    game_ids: list[str] | pd.Series | np.ndarray | None = None,
    fetch_missing: bool = False,
    max_fetch: int = 0,
) -> pd.DataFrame:
    """Load BoxScoreMatchupsV3 data from local cache, optionally fetching misses."""
    global _MATCHUPS_CACHE
    MATCHUPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    wanted_ids = None
    if game_ids is not None:
        wanted_ids = sorted({str(g) for g in list(game_ids) if str(g).strip()})
        if not wanted_ids:
            return pd.DataFrame()
    if fetch_missing and wanted_ids:
        missing_ids = [gid for gid in wanted_ids if not (MATCHUPS_CACHE_DIR / f"{gid}.json").exists()]
        if max_fetch > 0:
            missing_ids = missing_ids[:max_fetch]
        fetched_any = False
        for i, gid in enumerate(missing_ids, start=1):
            try:
                _stats_nba_fetch_json(
                    STATS_NBA_MATCHUPS_URL,
                    params={"GameID": gid},
                    cache_path=MATCHUPS_CACHE_DIR / f"{gid}.json",
                )
                fetched_any = True
                if i < len(missing_ids):
                    time.sleep(BOX_ADV_REQUEST_SLEEP_SECS)
            except Exception as exc:
                print(f"  Warning: BoxScoreMatchupsV3 fetch failed for {gid}: {exc}", flush=True)
        if fetched_any:
            _MATCHUPS_CACHE = None
    if _MATCHUPS_CACHE is None:
        rows: list[dict[str, Any]] = []
        for f in sorted(MATCHUPS_CACHE_DIR.glob("*.json")):
            try:
                rows.extend(_parse_boxscore_matchups_payload(json.loads(f.read_text()), f.stem))
            except Exception:
                continue
        _MATCHUPS_CACHE = (
            pd.DataFrame(rows).drop_duplicates(subset=["game_id", "player_id"], keep="first").reset_index(drop=True)
            if rows else pd.DataFrame()
        )
    if _MATCHUPS_CACHE is None or _MATCHUPS_CACHE.empty:
        return pd.DataFrame()
    out = _MATCHUPS_CACHE
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
    global _ESPN_ATHLETE_DISK_CACHE_LOADED

    # Hydrate from disk once per process even when in-memory cache is partially preloaded.
    disk_cache = PROP_CACHE_DIR / "espn_athlete_cache.json"
    if not _ESPN_ATHLETE_DISK_CACHE_LOADED and disk_cache.exists():
        try:
            loaded = json.loads(disk_cache.read_text())
            if isinstance(loaded, dict):
                athlete_cache.update({str(k): str(v) for k, v in loaded.items()})
        except Exception:
            pass
        _ESPN_ATHLETE_DISK_CACHE_LOADED = True

    if athlete_id in athlete_cache:
        return athlete_cache[athlete_id]

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
    snapshot_rows: list[dict[str, Any]] = []
    snapshot_ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
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

        home_team = normalize_espn_abbr(str(ev.get("home_team", "")))
        away_team = normalize_espn_abbr(str(ev.get("away_team", "")))
        for bk in odds_data.get("bookmakers", []):
            bk_key = str(bk.get("key", "")).strip().lower()
            bk_title = str(bk.get("title", "")).strip()
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
                    row = {
                        "player_name": pname,
                        "team": "",
                        "stat_type": stat_type,
                        "line": float(pdata.get("line", np.nan)),
                        "over_odds": float(pdata.get("over_odds", np.nan)),
                        "under_odds": float(pdata.get("under_odds", np.nan)),
                        "source": f"odds_api_{bk_key}",
                        "bookmaker": bk_key,
                        "bookmaker_title": bk_title,
                        "event_id": eid,
                        "home_team": home_team,
                        "away_team": away_team,
                        "snapshot_at_utc": snapshot_ts,
                    }
                    rows.append(row)
                    snapshot_rows.append(row.copy())

    if snapshot_rows:
        snap_dir = ODDS_API_SNAPSHOT_DIR / date_str
        snap_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(snapshot_rows).to_csv(
            snap_dir / f"odds_api_player_props_{snapshot_ts}.csv",
            index=False,
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["player_name"] = df["player_name"].str.strip()
    return df


def load_odds_api_snapshot_history(date_str: str) -> pd.DataFrame:
    """Load all saved per-book Odds API snapshots for a date."""
    snap_dir = ODDS_API_SNAPSHOT_DIR / date_str
    if not snap_dir.is_dir():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for f in sorted(snap_dir.glob("odds_api_player_props_*.csv")):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return _normalize_and_dedupe_prop_lines(out, default_date=date_str)


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
        manual = _normalize_and_dedupe_prop_lines(manual, default_date=date_str)
        print(f"  Loaded {len(manual)} manual prop lines", flush=True)
        manual.to_csv(cache_file, index=False)
        no_lines_file.unlink(missing_ok=True)
        return manual

    # 2) Check cache
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file)
            if not cached.empty:
                cached = _normalize_and_dedupe_prop_lines(cached, default_date=date_str)
                print(f"  Loaded {len(cached)} cached prop lines for {date_str}", flush=True)
                today_str = datetime.now().strftime("%Y%m%d")
                if date_str == today_str:
                    odds_api_lines = fetch_odds_api_player_props(date_str)
                    if not odds_api_lines.empty:
                        print(f"  Refreshed {len(odds_api_lines)} Odds API prop lines (same-day snapshot)", flush=True)
                        cached = _normalize_and_dedupe_prop_lines(
                            pd.concat([cached, odds_api_lines], ignore_index=True),
                            default_date=date_str,
                        )
                        cached.to_csv(cache_file, index=False)
                no_lines_file.unlink(missing_ok=True)
                return cached
        except Exception:
            pass

    # 3) Try ESPN, then optionally merge Odds API for better executable pricing.
    print("  Trying ESPN player props API...", flush=True)
    espn_lines = fetch_espn_player_props(date_str)
    if not espn_lines.empty:
        print(f"  Found {len(espn_lines)} ESPN prop lines", flush=True)
        odds_api_lines = fetch_odds_api_player_props(date_str)
        combined = [espn_lines]
        if not odds_api_lines.empty:
            print(f"  Found {len(odds_api_lines)} Odds API prop lines (merge)", flush=True)
            combined.append(odds_api_lines)
        merged = _normalize_and_dedupe_prop_lines(pd.concat(combined, ignore_index=True), default_date=date_str)
        merged.to_csv(cache_file, index=False)
        no_lines_file.unlink(missing_ok=True)
        return merged

    # 4) Try The Odds API
    print("  ESPN props not available. Trying The Odds API...", flush=True)
    odds_api_lines = fetch_odds_api_player_props(date_str)
    if not odds_api_lines.empty:
        odds_api_lines = _normalize_and_dedupe_prop_lines(odds_api_lines, default_date=date_str)
        print(f"  Found {len(odds_api_lines)} Odds API prop lines", flush=True)
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

        norm_df = _normalize_and_dedupe_prop_lines(df, default_date=date_str)
        if not norm_df.empty:
            all_lines.append(norm_df)

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


def _prop_source_priority(source_val: Any) -> int:
    """Rank prop-line sources for dedupe tie-breaks (higher is preferred)."""
    src = str(source_val or "").lower()
    if src.startswith("manual"):
        return 5
    if "espn" in src:
        return 4
    if "odds_api" in src:
        return 3
    if src.startswith("cached"):
        return 2
    return 1


def _normalize_and_dedupe_prop_lines(
    lines_df: pd.DataFrame,
    default_date: str | None = None,
) -> pd.DataFrame:
    """Normalize schema and keep one best executable line per player/stat/date/team."""
    if lines_df is None or lines_df.empty:
        return pd.DataFrame()

    df = lines_df.copy()
    required = {"player_name", "stat_type", "line"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    if "game_date_est" not in df.columns:
        if "date" in df.columns:
            df["game_date_est"] = (
                df["date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
            )
        else:
            df["game_date_est"] = default_date or ""
    df["game_date_est"] = df["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["player_name_norm"] = df["player_name"].map(normalize_player_name)
    df["stat_type"] = df["stat_type"].astype(str).str.strip().str.lower()
    if "team" in df.columns:
        df["team"] = _normalize_team_series(df["team"])
    else:
        df["team"] = ""
    df["team"] = df["team"].fillna("").astype(str).str.strip().str.upper()
    if "source" not in df.columns:
        df["source"] = "unknown"

    for col in ["line", "open_line", "over_odds", "under_odds", "open_over_odds", "open_under_odds"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _add_implied_probs(df)
    over_imp = pd.to_numeric(df.get("over_implied_prob", np.nan), errors="coerce")
    under_imp = pd.to_numeric(df.get("under_implied_prob", np.nan), errors="coerce")
    df["vig"] = np.where(over_imp.notna() & under_imp.notna(), over_imp + under_imp - 1.0, np.inf)
    df["source_pri"] = df["source"].map(_prop_source_priority)
    df["has_team"] = df["team"].ne("")

    # Prefer team-scoped lines over blank-team rows when both exist.
    grp_base = ["game_date_est", "player_name_norm", "stat_type"]
    has_team_any = df.groupby(grp_base)["has_team"].transform("max").astype(bool)
    df = df[(~has_team_any) | df["has_team"]].copy()

    # One row per (date, player, stat, team), best vig then preferred source.
    sort_cols = ["game_date_est", "player_name_norm", "stat_type", "team", "vig", "source_pri"]
    df = df.sort_values(sort_cols, ascending=[True, True, True, True, True, False])
    df = df.drop_duplicates(subset=grp_base + ["team"], keep="first").copy()

    drop_cols = ["player_name_norm", "vig", "source_pri", "has_team"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns]).reset_index(drop=True)


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


def _role_bucket_series(starter_rate: pd.Series, minutes_avg: pd.Series) -> pd.Series:
    """Expected role bucket from pregame starter probability and minute load."""
    sr = pd.to_numeric(starter_rate, errors="coerce").fillna(0.0)
    mins = pd.to_numeric(minutes_avg, errors="coerce").fillna(0.0)
    minute_bucket = np.select(
        [mins >= 34.0, mins >= 28.0, mins >= 20.0],
        [3, 2, 1],
        default=0,
    )
    starter_bucket = (sr >= 0.5).astype(int) * 10
    return pd.Series((starter_bucket + minute_bucket).astype(int), index=starter_rate.index)


def _add_team_role_absence_context(raw_games: pd.DataFrame) -> pd.DataFrame:
    """Annotate raw player-game history with top-role teammate absences and WOWY deltas."""
    if raw_games.empty or "played" not in raw_games.columns:
        return raw_games

    df = raw_games.copy()
    df = df.sort_values(["team", "player_id", "game_time_utc", "game_id"]).reset_index(drop=True)
    player_group = ["team", "player_id", "season"] if "season" in df.columns else ["team", "player_id"]

    for col in ["points", "rebounds", "assists", "minutes"]:
        if col not in df.columns:
            df[col] = np.nan
        tmp = f"_{col}_played_only"
        df[tmp] = df[col].where(df["played"] == 1)
        df[f"_role_pre_{col}_avg10"] = df.groupby(player_group)[tmp].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean()
        )

    df["_creator_score"] = (
        df["_role_pre_assists_avg10"].fillna(0.0)
        + 0.15 * df["_role_pre_points_avg10"].fillna(0.0)
        + 0.03 * df["_role_pre_minutes_avg10"].fillna(0.0)
    )
    df["_rebounder_score"] = (
        df["_role_pre_rebounds_avg10"].fillna(0.0)
        + 0.03 * df["_role_pre_minutes_avg10"].fillna(0.0)
    )

    creator_top = (
        df.sort_values(["game_id", "team", "_creator_score"], ascending=[True, True, False])
        .dropna(subset=["_creator_score"])
        .drop_duplicates(subset=["game_id", "team"])
        [["game_id", "team", "player_id", "played"]]
        .rename(columns={"player_id": "team_top_creator_pid", "played": "_team_top_creator_played"})
    )
    rebound_top = (
        df.sort_values(["game_id", "team", "_rebounder_score"], ascending=[True, True, False])
        .dropna(subset=["_rebounder_score"])
        .drop_duplicates(subset=["game_id", "team"])
        [["game_id", "team", "player_id", "played"]]
        .rename(columns={"player_id": "team_top_rebounder_pid", "played": "_team_top_rebounder_played"})
    )
    df = df.merge(creator_top, on=["game_id", "team"], how="left")
    df = df.merge(rebound_top, on=["game_id", "team"], how="left")
    df["team_top_creator_out"] = (df["_team_top_creator_played"].fillna(0.0) == 0.0).astype(float)
    df["team_top_rebounder_out"] = (df["_team_top_rebounder_played"].fillna(0.0) == 0.0).astype(float)

    wowy_specs = [
        ("points", "team_top_creator_out", "wowy_points_top_creator_out_delta20"),
        ("assists", "team_top_creator_out", "wowy_assists_top_creator_out_delta20"),
        ("minutes", "team_top_creator_out", "wowy_minutes_top_creator_out_delta20"),
        ("rebounds", "team_top_rebounder_out", "wowy_rebounds_top_rebounder_out_delta20"),
        ("minutes", "team_top_rebounder_out", "wowy_minutes_top_rebounder_out_delta20"),
    ]
    for _, _, out_col in wowy_specs:
        df[out_col] = np.nan

    for stat_col, flag_col, out_col in wowy_specs:
        out_vals = df[stat_col].where((df["played"] == 1) & (df[flag_col] == 1))
        in_vals = df[stat_col].where((df["played"] == 1) & (df[flag_col] == 0))
        tmp_out = f"_{out_col}_raw_out"
        tmp_in = f"_{out_col}_raw_in"
        df[tmp_out] = out_vals
        df[tmp_in] = in_vals

        out_mean = df.groupby(player_group)[tmp_out].transform(
            lambda s: s.shift(1).rolling(20, min_periods=1).mean()
        )
        out_n = df.groupby(player_group)[tmp_out].transform(
            lambda s: s.shift(1).rolling(20, min_periods=1).count()
        )
        in_mean = df.groupby(player_group)[tmp_in].transform(
            lambda s: s.shift(1).rolling(20, min_periods=1).mean()
        )
        in_n = df.groupby(player_group)[tmp_in].transform(
            lambda s: s.shift(1).rolling(20, min_periods=1).count()
        )
        shrink = (out_n / (out_n + 5.0)).fillna(0.0) * (in_n / (in_n + 5.0)).fillna(0.0)
        df[out_col] = (out_mean - in_mean).fillna(0.0) * shrink

    drop_cols = [
        c
        for c in df.columns
        if c.startswith("_role_pre_")
        or c.endswith("_played_only")
        or c.startswith("_wowy_")
        or c in {
            "_creator_score",
            "_rebounder_score",
            "_team_top_creator_played",
            "_team_top_rebounder_played",
        }
    ]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    return df


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


def _migrate_cached_player_features_v13_to_v14(cached: pd.DataFrame) -> pd.DataFrame | None:
    """Fast-path migrate cached v13 features to v14 without full recomputation.

    v14 only changes training-side injury interaction scaling. All required source
    columns already exist in the cached feature frame.
    """
    required = [
        "team_injury_pressure",
        "pre_usage_proxy",
        "pre_minutes_avg5",
        "matchup_pace_avg",
    ]
    if cached.empty or any(col not in cached.columns for col in required):
        return None

    out = cached.copy()
    pressure_scaled = out["team_injury_pressure"].map(_compress_injury_pressure)
    out["usage_boost_proxy"] = pressure_scaled * out["pre_usage_proxy"].fillna(0.0)
    out["minutes_x_injury_pressure"] = out["pre_minutes_avg5"].fillna(0.0) * pressure_scaled
    out["pace_x_injury_pressure"] = out["matchup_pace_avg"].fillna(0.0) * pressure_scaled
    return out


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
    cache_pairs = [
        (PLAYER_FEATURE_CACHE_FILE, PLAYER_FEATURE_CACHE_META),
        (LEGACY_PLAYER_FEATURE_CACHE_FILE, LEGACY_PLAYER_FEATURE_CACHE_META),
    ]
    for cache_file, meta_file in cache_pairs:
        if not cache_file.exists() or not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
            if meta == cache_key:
                cached = pd.read_pickle(cache_file)
                if isinstance(cached, pd.DataFrame) and not cached.empty:
                    print(f"  Loaded cached player features: {len(cached)} rows from {cache_file.name}", flush=True)
                    return cached
            elif meta.get("version") == "v13" and cache_key.get("version") == "v14":
                cached = pd.read_pickle(cache_file)
                if isinstance(cached, pd.DataFrame) and not cached.empty:
                    migrated = _migrate_cached_player_features_v13_to_v14(cached)
                    if migrated is not None and not migrated.empty:
                        print(f"  Migrated cached player features v13 -> v14: {len(migrated)} rows", flush=True)
                        try:
                            MODEL_DIR.mkdir(parents=True, exist_ok=True)
                            migrated.to_pickle(PLAYER_FEATURE_CACHE_FILE)
                            PLAYER_FEATURE_CACHE_META.write_text(json.dumps(cache_key, indent=2, default=str))
                        except Exception as exc:
                            print(f"  Warning: could not write migrated player feature cache: {exc}", flush=True)
                        return migrated
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
    pg = _add_team_role_absence_context(player_games)
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

    # --- Load and merge Basketball Reference advanced stats (BRef fallback + new cols) ---
    print("  Loading BRef advanced stats (usage, efficiency, BPM, defensive rating)...", flush=True)
    pg = _merge_bref_advanced_stats(pg)

    # --- Load and merge Player Tracking stats (touches, drives, catch-and-shoot) ---
    trk_cols = [
        "trk_touches", "trk_drives", "trk_passes",
        "trk_catch_shoot_fga", "trk_catch_shoot_fgm",
        "trk_catch_shoot_fg3a", "trk_catch_shoot_fg3m",
        "trk_pull_up_fga", "trk_pull_up_fgm", "trk_pull_up_fg3a",
        "trk_dist_miles", "trk_avg_speed",
        "trk_contested_shots", "trk_uncontested_fga",
        # Hustle stats (from BDL advanced)
        "trk_deflections", "trk_box_outs", "trk_off_box_outs", "trk_def_box_outs",
        "trk_loose_balls", "trk_screen_assists", "trk_secondary_assists",
        # Rebound chances
        "trk_reb_chances", "trk_reb_chances_off", "trk_reb_chances_def",
        # Scoring breakdown
        "trk_pts_paint", "trk_pts_fast_break", "trk_pts_off_to",
    ]
    if USE_TRACKING_FEATURES:
        trk = load_player_tracking_stats(
            game_ids=pg["game_id"].astype(str).unique().tolist(),
            fetch_missing=box_adv_fetch_missing,
            max_fetch=box_adv_max_fetch,
        )
        if not trk.empty:
            pg = pg.merge(
                trk[["game_id", "team", "player_id"] + [c for c in trk_cols if c in trk.columns]],
                on=["game_id", "team", "player_id"],
                how="left",
            )
            _trk_avail = [c for c in trk_cols if c in trk.columns]
            _trk_pct = trk[_trk_avail].notna().any(axis=1).mean() * 100
            print(f"  Merged tracking stats: {len(trk)} rows, {_trk_pct:.0f}% coverage", flush=True)
    for c in trk_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    # --- Load and merge Defensive boxscore stats (matchup defense) ---
    def_cols = [
        "def_matchup_fga", "def_matchup_fgm", "def_matchup_fg_pct",
        "def_matchup_3pa", "def_matchup_3pm", "def_matchup_3pt_pct",
        "def_matchup_minutes", "def_matchup_assists", "def_matchup_tov",
        "def_matchup_player_pts", "def_switches_on", "def_partial_poss",
    ]
    if USE_TRACKING_FEATURES:
        def_df = load_defensive_stats(
            game_ids=pg["game_id"].astype(str).unique().tolist(),
        )
        if not def_df.empty:
            pg = pg.merge(
                def_df[["game_id", "team", "player_id"] + [c for c in def_cols if c in def_df.columns]],
                on=["game_id", "team", "player_id"],
                how="left",
            )
            _def_pct = def_df[[c for c in def_cols if c in def_df.columns]].notna().any(axis=1).mean() * 100
            print(f"  Merged defensive stats: {len(def_df)} rows, {_def_pct:.0f}% coverage", flush=True)
    for c in def_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    # --- Load and merge Scoring boxscore stats (shot creation context) ---
    scr_cols = [
        "scr_pct_assisted_2pt", "scr_pct_assisted_3pt", "scr_pct_assisted_fgm",
        "scr_pct_unassisted_2pt", "scr_pct_unassisted_3pt",
        "scr_pct_fga_2pt", "scr_pct_fga_3pt",
        "scr_pct_pts_2pt", "scr_pct_pts_3pt", "scr_pct_pts_ft",
        "scr_pct_pts_paint", "scr_pct_pts_midrange",
        "scr_pct_pts_fastbreak", "scr_pct_pts_off_to",
    ]
    if USE_TRACKING_FEATURES:
        scr_df = load_scoring_stats(
            game_ids=pg["game_id"].astype(str).unique().tolist(),
        )
        if not scr_df.empty:
            pg = pg.merge(
                scr_df[["game_id", "team", "player_id"] + [c for c in scr_cols if c in scr_df.columns]],
                on=["game_id", "team", "player_id"],
                how="left",
            )
            _scr_pct = scr_df[[c for c in scr_cols if c in scr_df.columns]].notna().any(axis=1).mean() * 100
            print(f"  Merged scoring stats: {len(scr_df)} rows, {_scr_pct:.0f}% coverage", flush=True)
    for c in scr_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    # --- Load and merge GameRotation stats (stint structure) ---
    rot_cols = ["rot_stints", "rot_total_stint_min", "rot_avg_stint_min", "rot_max_stint_min"]
    if USE_ROTATION_MATCHUP_FEATURES:
        rot_df = load_game_rotation_stats(
            game_ids=pg["game_id"].astype(str).unique().tolist(),
            fetch_missing=box_adv_fetch_missing,
            max_fetch=box_adv_max_fetch,
        )
        if not rot_df.empty:
            merge_cols = ["game_id", "player_id"] + [c for c in rot_cols if c in rot_df.columns]
            pg = pg.merge(rot_df[merge_cols], on=["game_id", "player_id"], how="left")
            _rot_pct = rot_df[[c for c in rot_cols if c in rot_df.columns]].notna().any(axis=1).mean() * 100
            print(f"  Merged GameRotation stats: {len(rot_df)} rows, {_rot_pct:.0f}% coverage", flush=True)
    for c in rot_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    # --- Load and merge BoxScoreMatchupsV3 stats (offensive matchup load) ---
    mtch_cols = [
        "mtch_partial_poss", "mtch_fga", "mtch_fgm", "mtch_fg_pct",
        "mtch_3pa", "mtch_3pm", "mtch_3pt_pct", "mtch_ast", "mtch_pts",
    ]
    if USE_ROTATION_MATCHUP_FEATURES:
        mtch_df = load_boxscore_matchups_stats(
            game_ids=pg["game_id"].astype(str).unique().tolist(),
            fetch_missing=box_adv_fetch_missing,
            max_fetch=box_adv_max_fetch,
        )
        if not mtch_df.empty:
            merge_cols = ["game_id", "player_id"] + [c for c in mtch_cols if c in mtch_df.columns]
            pg = pg.merge(mtch_df[merge_cols], on=["game_id", "player_id"], how="left")
            _mtch_pct = mtch_df[[c for c in mtch_cols if c in mtch_df.columns]].notna().any(axis=1).mean() * 100
            print(f"  Merged matchup stats: {len(mtch_df)} rows, {_mtch_pct:.0f}% coverage", flush=True)
    for c in mtch_cols:
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
                       "fouls_personal", "fouls_drawn", "pts_in_paint", "pts_fast_break",
                       "trk_touches", "trk_drives", "trk_passes",
                       "trk_catch_shoot_fga", "trk_catch_shoot_fgm",
                       "trk_catch_shoot_fg3a", "trk_catch_shoot_fg3m",
                       "trk_pull_up_fga", "trk_pull_up_fgm", "trk_pull_up_fg3a",
                       "trk_contested_shots", "trk_uncontested_fga",
                       # Hustle counting stats (OT-adjusted)
                       "trk_deflections", "trk_box_outs", "trk_off_box_outs",
                       "trk_def_box_outs", "trk_loose_balls", "trk_screen_assists",
                       "trk_secondary_assists", "trk_reb_chances", "trk_reb_chances_off",
                       "trk_reb_chances_def", "trk_pts_paint", "trk_pts_fast_break",
                       "trk_pts_off_to",
                       # Defensive matchup counting stats (OT-adjusted)
                       "def_matchup_fga", "def_matchup_fgm",
                       "def_matchup_3pa", "def_matchup_3pm",
                       "def_matchup_assists", "def_matchup_tov",
                       "def_matchup_player_pts", "def_switches_on",
                       # MatchupsV3 counting stats
                       "mtch_partial_poss", "mtch_fga", "mtch_fgm",
                       "mtch_3pa", "mtch_3pm", "mtch_ast", "mtch_pts"]
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
    # Tracking per-minute rates (more stable than raw counts across different minute loads)
    if "trk_touches" in pg.columns:
        pg["trk_touches_per_min"] = pg["trk_touches"].fillna(0) / safe_mins
    if "trk_drives" in pg.columns:
        pg["trk_drives_per_min"] = pg["trk_drives"].fillna(0) / safe_mins
    if "trk_passes" in pg.columns:
        pg["trk_passes_per_min"] = pg["trk_passes"].fillna(0) / safe_mins
    if "trk_catch_shoot_fg3a" in pg.columns:
        pg["trk_catch_shoot_fg3a_per_min"] = pg["trk_catch_shoot_fg3a"].fillna(0) / safe_mins
    # Hustle per-minute rates
    if "trk_deflections" in pg.columns:
        pg["trk_deflections_per_min"] = pg["trk_deflections"].fillna(0) / safe_mins
    if "trk_box_outs" in pg.columns:
        pg["trk_box_outs_per_min"] = pg["trk_box_outs"].fillna(0) / safe_mins
    if "trk_loose_balls" in pg.columns:
        pg["trk_loose_balls_per_min"] = pg["trk_loose_balls"].fillna(0) / safe_mins
    if "trk_reb_chances" in pg.columns:
        pg["trk_reb_chances_per_min"] = pg["trk_reb_chances"].fillna(0) / safe_mins
    if "trk_screen_assists" in pg.columns:
        pg["trk_screen_assists_per_min"] = pg["trk_screen_assists"].fillna(0) / safe_mins
    # Defensive per-minute rates
    if "def_matchup_fga" in pg.columns:
        pg["def_matchup_fga_per_min"] = pg["def_matchup_fga"].fillna(0) / safe_mins
    if "def_matchup_fgm" in pg.columns:
        pg["def_matchup_fgm_per_min"] = pg["def_matchup_fgm"].fillna(0) / safe_mins
    if "mtch_partial_poss" in pg.columns:
        pg["mtch_partial_poss_per_min"] = pg["mtch_partial_poss"].fillna(0) / safe_mins

    # --- Role-conditioned creation / rebounding context ---
    if "trk_passes" in pg.columns:
        team_passes = pg.groupby(["game_id", "team"])["trk_passes"].transform("sum").clip(lower=1.0)
        pg["player_pass_share"] = pg["trk_passes"].fillna(0.0) / team_passes
        pg["player_ast_per_pass"] = pg["assists"].fillna(0.0) / pg["trk_passes"].clip(lower=1.0)
    else:
        pg["player_pass_share"] = np.nan
        pg["player_ast_per_pass"] = np.nan

    if "trk_reb_chances" in pg.columns:
        team_reb_chances = pg.groupby(["game_id", "team"])["trk_reb_chances"].transform("sum").clip(lower=1.0)
        pg["player_reb_chance_share"] = pg["trk_reb_chances"].fillna(0.0) / team_reb_chances
        pg["player_reb_conversion"] = pg["rebounds"].fillna(0.0) / pg["trk_reb_chances"].clip(lower=1.0)
    else:
        pg["player_reb_chance_share"] = np.nan
        pg["player_reb_conversion"] = np.nan

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
                    "fga_per_min", "fg3a_per_min", "fta_per_min", "fouls_drawn_per_min",
                    "player_pass_share", "player_ast_per_pass",
                    "player_reb_chance_share", "player_reb_conversion",
                    # Player tracking stats (touches, drives, catch-and-shoot)
                    "trk_touches_reg", "trk_drives_reg", "trk_passes_reg",
                    "trk_catch_shoot_fga_reg", "trk_catch_shoot_fg3a_reg",
                    "trk_pull_up_fga_reg", "trk_pull_up_fg3a_reg",
                    "trk_contested_shots_reg", "trk_uncontested_fga_reg",
                    # Tracking per-minute rates
                    "trk_touches_per_min", "trk_drives_per_min",
                    "trk_passes_per_min",
                    "trk_catch_shoot_fg3a_per_min",
                    # Hustle stats (from BDL)
                    "trk_deflections_reg", "trk_box_outs_reg",
                    "trk_off_box_outs_reg", "trk_def_box_outs_reg",
                    "trk_loose_balls_reg", "trk_screen_assists_reg",
                    "trk_secondary_assists_reg",
                    "trk_reb_chances_reg", "trk_reb_chances_off_reg", "trk_reb_chances_def_reg",
                    "trk_pts_paint_reg", "trk_pts_fast_break_reg", "trk_pts_off_to_reg",
                    # Hustle per-minute rates
                    "trk_deflections_per_min", "trk_box_outs_per_min",
                    "trk_loose_balls_per_min", "trk_reb_chances_per_min",
                    "trk_screen_assists_per_min",
                    # Defensive matchup stats (from stats.nba.com)
                    "def_matchup_fga_reg", "def_matchup_fgm_reg",
                    "def_matchup_3pa_reg", "def_matchup_3pm_reg",
                    "def_matchup_fg_pct", "def_matchup_3pt_pct",
                    "def_matchup_player_pts_reg", "def_switches_on_reg",
                    "def_matchup_fga_per_min", "def_matchup_fgm_per_min",
                    # Rotation / matchup context
                    "rot_stints", "rot_total_stint_min", "rot_avg_stint_min", "rot_max_stint_min",
                    "mtch_partial_poss_reg", "mtch_fga_reg", "mtch_fgm_reg",
                    "mtch_3pa_reg", "mtch_3pm_reg", "mtch_ast_reg", "mtch_pts_reg",
                    "mtch_fg_pct", "mtch_3pt_pct", "mtch_partial_poss_per_min",
                    # Scoring context stats (from stats.nba.com)
                    "scr_pct_assisted_2pt", "scr_pct_assisted_3pt",
                    "scr_pct_unassisted_2pt", "scr_pct_unassisted_3pt",
                    "scr_pct_fga_2pt", "scr_pct_fga_3pt",
                    "scr_pct_pts_paint", "scr_pct_pts_midrange",
                    "scr_pct_pts_fastbreak",
                    # BRef advanced stats — only NEW columns not already covered
                    # by adv_usage_pct/adv_off_rating/adv_ts_pct/adv_ast_pct/adv_reb_pct
                    "bref_adv_def_rtg", "bref_adv_bpm", "bref_adv_efg_pct",
                    "bref_adv_stl_pct", "bref_adv_blk_pct",
                    "bref_adv_orb_pct", "bref_adv_drb_pct",
                    "bref_adv_tov_pct"]
    # Last-3-game hot streak features (faster than avg5 for capturing streaks)
    avg3_cols = ["points_reg", "rebounds_reg", "assists_reg", "minutes", "fg3m_reg",
                 "fg3a_reg", "fouls_drawn_reg", "adv_usage_pct",
                 "pts_per_min", "reb_per_min", "ast_per_min", "fg3m_per_min",
                 # Tracking: touches and drives are strong usage signals
                 "trk_touches_reg", "trk_drives_reg",
                 # Hustle: box_outs for rebounds, reb_chances
                 "trk_box_outs_reg", "trk_reb_chances_reg",
                 # Rotation + matchups
                 "rot_avg_stint_min", "rot_max_stint_min",
                 "mtch_partial_poss_reg", "mtch_fg_pct",
                 # Defensive matchup: recent matchup load
                 "def_matchup_fga_reg", "def_matchup_fg_pct",
                 # BRef: BPM and defensive rating for hot streak detection
                 "bref_adv_bpm", "bref_adv_def_rtg"]
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
                "fga_per_min", "fg3a_per_min", "fta_per_min", "fouls_drawn_per_min",
                "player_pass_share", "player_ast_per_pass",
                "player_reb_chance_share", "player_reb_conversion",
                # Tracking EWM (touches and drives most important)
                "trk_touches_reg", "trk_drives_reg", "trk_passes_per_min",
                "trk_catch_shoot_fg3a_reg", "trk_pull_up_fg3a_reg",
                # Hustle EWM
                "trk_deflections_reg", "trk_box_outs_reg",
                "trk_reb_chances_reg", "trk_screen_assists_reg",
                # Defensive matchup EWM
                "def_matchup_fga_reg", "def_matchup_fg_pct",
                "def_matchup_3pt_pct",
                # Rotation / matchups
                "rot_avg_stint_min", "rot_max_stint_min",
                "mtch_partial_poss_reg", "mtch_fg_pct", "mtch_3pt_pct",
                # Scoring context EWM
                "scr_pct_assisted_2pt", "scr_pct_unassisted_2pt",
                "scr_pct_pts_paint", "scr_pct_pts_midrange",
                # BRef advanced EWM (only new columns, not redundant with adv_*)
                "bref_adv_bpm", "bref_adv_def_rtg", "bref_adv_efg_pct"]
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
    reg_drop = [c for c in pg.columns if c.endswith("_reg") and c.startswith(("points_", "rebounds_", "assists_", "fg3m_", "fga_", "fgm_", "fg3a_", "fta_", "ftm_", "tov_", "steals_", "blocks_", "orb_", "drb_", "fouls_personal_", "fouls_drawn_", "pts_in_paint_", "pts_fast_break_", "trk_", "mtch_"))]
    if reg_drop:
        pg.drop(columns=reg_drop, inplace=True, errors="ignore")

    # Starter rate
    grp_starter = pg.groupby(player_group)["starter"]
    pg["pre_starter_rate"] = grp_starter.transform(lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    # --- Same-role distribution features ---
    # Condition recent stat ranges on expected role bucket instead of raw history.
    role_minutes_anchor = (
        pg.get("pre_minutes_avg5", pd.Series(np.nan, index=pg.index))
        .fillna(pg.get("pre_minutes_avg10", pd.Series(np.nan, index=pg.index)))
        .fillna(0.0)
    )
    pg["expected_role_bucket"] = _role_bucket_series(pg["pre_starter_rate"], role_minutes_anchor)
    same_role_group = player_group + ["expected_role_bucket"]
    for col in ["points", "rebounds", "assists", "fg3m", "minutes"]:
        if col not in pg.columns:
            continue
        grp_same = pg.groupby(same_role_group)[col]
        pg[f"pre_{col}_same_role_p75_20"] = grp_same.transform(
            lambda s: s.shift(1).rolling(20, min_periods=3).quantile(0.75)
        )
        pg[f"pre_{col}_same_role_p90_20"] = grp_same.transform(
            lambda s: s.shift(1).rolling(20, min_periods=3).quantile(0.90)
        )
        pg[f"pre_{col}_same_role_max20"] = grp_same.transform(
            lambda s: s.shift(1).rolling(20, min_periods=3).max()
        )

    # --- Phase 3: Enhanced minutes model features ---
    # DNP rate: fraction of recent games with < 5 minutes (shift(1) to avoid leakage)
    if "minutes" in pg.columns:
        pg["_dnp_flag"] = (pg["minutes"] < 5).astype(float)
        pg["dnp_rate_last10"] = pg.groupby(player_group)["_dnp_flag"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean()
        )
        pg.drop(columns=["_dnp_flag"], inplace=True, errors="ignore")

    # Foul trouble tendency: fouls per minute
    if "pre_fouls_personal_avg5" in pg.columns and "pre_minutes_avg5" in pg.columns:
        safe_min = pg["pre_minutes_avg5"].clip(lower=1.0)
        pg["pre_fouls_per_min_avg5"] = pg["pre_fouls_personal_avg5"].fillna(0) / safe_min

    # Starter benched rate: started but got < 15 min in last 3 games
    if "starter" in pg.columns and "minutes" in pg.columns:
        pg["_starter_benched"] = ((pg["starter"] > 0) & (pg["minutes"] < 15)).astype(float)
        pg["starter_benched_rate3"] = pg.groupby(player_group)["_starter_benched"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        pg.drop(columns=["_starter_benched"], inplace=True, errors="ignore")

    # --- Phase 7: Recency-weighted features ---
    recency_stat_cols = ["points", "rebounds", "assists", "fg3m", "minutes"]
    for col in recency_stat_cols:
        # Use the renamed rolling cols (no _reg suffix)
        avg_col = f"pre_{col}_avg5" if f"pre_{col}_avg5" in pg.columns else None
        season_col = f"pre_{col}_season" if f"pre_{col}_season" in pg.columns else None

        # EWM variance (captures inconsistency weighted toward recent games)
        raw_col = col if col in pg.columns else None
        if raw_col:
            pg[f"pre_{col}_ewm_var5"] = pg.groupby(player_group)[raw_col].transform(
                lambda s: s.shift(1).ewm(span=5, min_periods=3).var()
            )

        # Momentum / trend: slope of last 5 games via linear regression
        if raw_col:
            def _window_slope(arr: np.ndarray) -> float:
                valid = np.isfinite(arr)
                if valid.sum() < 3:
                    return np.nan
                x = np.arange(arr.shape[0], dtype=float)[valid]
                y = arr[valid].astype(float)
                x_centered = x - x.mean()
                denom = float(np.dot(x_centered, x_centered))
                if denom <= 0:
                    return np.nan
                y_centered = y - y.mean()
                return float(np.dot(x_centered, y_centered) / denom)

            pg[f"pre_{col}_trend5"] = pg.groupby(player_group)[raw_col].transform(
                lambda s: s.shift(1).rolling(5, min_periods=3).apply(_window_slope, raw=True)
            )

        # Recency ratio: avg3 / season_avg (>1 = hot streak)
        avg3_col = f"pre_{col}_avg3"
        if avg3_col in pg.columns and season_col and season_col in pg.columns:
            safe_season = pg[season_col].clip(lower=0.1)
            pg[f"pre_{col}_recent_vs_season"] = pg[avg3_col].fillna(0) / safe_season

    # --- Distribution-aware stat features ---
    # Capture whether recent production is out-of-distribution for the player's normal range.
    distribution_cols = ["points", "rebounds", "assists", "fg3m", "minutes"]
    for col in distribution_cols:
        if col not in pg.columns:
            continue
        grp = pg.groupby(player_group)[col]
        pg[f"pre_{col}_p75_20"] = grp.transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).quantile(0.75)
        )
        pg[f"pre_{col}_p90_20"] = grp.transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).quantile(0.90)
        )
        pg[f"pre_{col}_max20"] = grp.transform(
            lambda s: s.shift(1).rolling(20, min_periods=5).max()
        )
        season_col = f"pre_{col}_season"
        std_col = f"pre_{col}_std10"
        avg3_col = f"pre_{col}_avg3"
        if avg3_col in pg.columns and season_col in pg.columns:
            safe_std = pg.get(std_col, pd.Series(np.nan, index=pg.index)).fillna(
                grp.transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
            ).clip(lower=0.75)
            pg[f"pre_{col}_recent_zscore"] = (
                pg[avg3_col].fillna(0.0) - pg[season_col].fillna(0.0)
            ) / safe_std

    # --- Role persistence features ---
    # Distinguish stable role changes from one-off spikes in minutes/creation.
    if "pre_minutes_avg10" in pg.columns and "minutes" in pg.columns:
        safe_base_minutes = pg["pre_minutes_avg10"].clip(lower=8.0)
        minute_dev = (pg["minutes"].fillna(0.0) - safe_base_minutes) / safe_base_minutes
        pg["_role_min_consistent"] = (minute_dev.abs() <= 0.15).astype(float)
        pg["_role_min_expanded"] = (minute_dev >= 0.15).astype(float)
        pg["_role_min_reduced"] = (minute_dev <= -0.15).astype(float)
        grp_consistent = pg.groupby(player_group)["_role_min_consistent"]
        grp_expand = pg.groupby(player_group)["_role_min_expanded"]
        grp_reduce = pg.groupby(player_group)["_role_min_reduced"]
        pg["pre_role_minutes_consistency5"] = grp_consistent.transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).mean()
        )
        pg["pre_role_minutes_expansion3"] = grp_expand.transform(
            lambda s: s.shift(1).rolling(3, min_periods=2).mean()
        )
        pg["pre_role_minutes_expansion5"] = grp_expand.transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).mean()
        )
        pg["pre_role_minutes_reduction5"] = grp_reduce.transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).mean()
        )
        pg.drop(
            columns=["_role_min_consistent", "_role_min_expanded", "_role_min_reduced"],
            inplace=True,
            errors="ignore",
        )
    else:
        pg["pre_role_minutes_consistency5"] = np.nan
        pg["pre_role_minutes_expansion3"] = np.nan
        pg["pre_role_minutes_expansion5"] = np.nan
        pg["pre_role_minutes_reduction5"] = np.nan

    if "trk_passes_per_min" in pg.columns and "pre_trk_passes_avg10" in pg.columns and "pre_minutes_avg10" in pg.columns:
        base_passes_per_min = (
            pg["pre_trk_passes_avg10"].fillna(0.0) / pg["pre_minutes_avg10"].clip(lower=8.0)
        ).clip(lower=0.1)
        pass_dev = (pg["trk_passes_per_min"].fillna(0.0) - base_passes_per_min) / base_passes_per_min
        pg["_role_pass_expand"] = (pass_dev >= 0.15).astype(float)
        pg["pre_role_passes_expansion5"] = pg.groupby(player_group)["_role_pass_expand"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).mean()
        )
        pg.drop(columns=["_role_pass_expand"], inplace=True, errors="ignore")
    else:
        pg["pre_role_passes_expansion5"] = np.nan

    # --- Phase 12: Bayesian shrinkage features ---
    # Mild stabilizer: blend avg5 toward avg10/season to dampen streak-chasing.
    # Uses avg10 as primary anchor (more responsive than season) with season as fallback.
    # Weight on avg5 is high (0.6-0.85) so this acts as a gentle pull, not a heavy anchor.
    shrinkage_cols = ["points", "rebounds", "assists", "fg3m", "minutes"]
    _game_num = pg.groupby(player_group).cumcount() + 1
    for col in shrinkage_cols:
        avg5_col = f"pre_{col}_avg5"
        avg10_col = f"pre_{col}_avg10"
        season_col = f"pre_{col}_season"
        std10_col = f"pre_{col}_std10"
        if avg5_col not in pg.columns or season_col not in pg.columns:
            continue
        # Anchor: prefer avg10 (responsive), fall back to season for early-season
        anchor = pg[avg10_col].fillna(pg[season_col]) if avg10_col in pg.columns else pg[season_col]
        # w on avg5: ramp from 0.6 (few games, trust anchor more) to 0.85 (many games)
        games_w = (_game_num.clip(lower=1) / 40.0).clip(upper=0.85).clip(lower=0.6)
        # Volatility adjustment: high-variance players get slightly more shrinkage
        if std10_col in pg.columns:
            safe_mean = anchor.clip(lower=0.5)
            cv = (pg[std10_col].fillna(0) / safe_mean).clip(upper=2.0)
            # Reduce w by up to 0.15 for very volatile players
            stability_adj = (0.15 * cv).clip(upper=0.15)
        else:
            stability_adj = 0.0
        w = (games_w - stability_adj).clip(lower=0.5, upper=0.85)
        pg[f"pre_{col}_shrunk"] = w * pg[avg5_col].fillna(0) + (1.0 - w) * anchor.fillna(0)

    # --- Phase 12: Role-change detection flags ---
    # Detect sustained minutes/usage regime shifts so true role changes are not over-shrunk.
    if "pre_minutes_avg5" in pg.columns and "pre_minutes_avg10" in pg.columns:
        min_shift = pg["pre_minutes_avg5"] - pg["pre_minutes_avg10"]
        min_avg10 = pg["pre_minutes_avg10"].clip(lower=1.0)
        # Role change = avg5 minutes differ from avg10 by >15% (sustained, not noise)
        pg["role_change_flag"] = (min_shift.abs() / min_avg10 > 0.15).astype(float)
        # Direction: positive = expanded role, negative = reduced role
        pg["role_change_direction"] = np.sign(min_shift) * pg["role_change_flag"]
    # Usage regime shift: detect via points-per-minute ratio change (avg5 vs avg10)
    if "pre_pts_per_min_avg5" in pg.columns and "pre_pts_per_min_ewm5" in pg.columns:
        # Use per-minute scoring rate as usage proxy (available at this point)
        ppm5 = pg["pre_pts_per_min_avg5"].fillna(0)
        ppm_ewm = pg["pre_pts_per_min_ewm5"].fillna(0)
        # Compare short-term rate to its longer EWM; >15% divergence = usage regime shift
        safe_ppm = ppm_ewm.clip(lower=0.01)
        ppm_shift = (ppm5 - ppm_ewm).abs() / safe_ppm
        pg["usage_regime_shift"] = (ppm_shift > 0.15).astype(float)
    else:
        pg["usage_regime_shift"] = 0.0
    # Interaction: shrinkage should be weaker (trust recent more) during role changes
    for col in shrinkage_cols:
        shrunk_col = f"pre_{col}_shrunk"
        avg5_col = f"pre_{col}_avg5"
        if shrunk_col in pg.columns and avg5_col in pg.columns and "role_change_flag" in pg.columns:
            # During role changes, shift shrunk estimate 50% back toward avg5 (less shrinkage)
            rc = pg["role_change_flag"]
            pg[shrunk_col] = pg[shrunk_col] * (1.0 - 0.5 * rc) + pg[avg5_col].fillna(0) * (0.5 * rc)

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

    # --- Step 3: Cumulative season load features ---
    # Players degrade in the second half of season, especially high-minute players on B2Bs.
    # All features shifted by 1 to prevent leakage (value from before current game).
    pg["season_games_played"] = pg.groupby(player_group).cumcount()  # 0-indexed = shifted by 1
    if "minutes" in pg.columns:
        pg["season_total_minutes"] = pg.groupby(player_group)["minutes"].transform(
            lambda s: s.shift(1).cumsum()
        ).fillna(0.0)
    else:
        pg["season_total_minutes"] = 0.0
    pg["high_load_flag"] = (pg["season_total_minutes"] > 1800).astype(float)

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

    # Step 3: Load interaction features (load_x_age deferred until player_career_games exists)
    pg["load_x_b2b"] = pg["high_load_flag"] * pg["is_b2b"]

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
        pg["team_injury_pressure_bwd"] = pg["team_injury_pressure"]
        pg["injury_pressure_delta"] = 0.0
        pressure_scaled = pg["team_injury_pressure"].map(_compress_injury_pressure)
        pg["usage_boost_proxy"] = pressure_scaled * pg["pre_usage_proxy"].fillna(0)
        pg["minutes_x_injury_pressure"] = pg["pre_minutes_avg5"].fillna(0) * pressure_scaled
    else:
        pg["team_injury_pressure"] = 0.0
        pg["team_injury_pressure_bwd"] = 0.0
        pg["injury_pressure_delta"] = 0.0
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

    pressure_scaled = pg["team_injury_pressure"].map(_compress_injury_pressure)
    pg["pace_x_injury_pressure"] = pg["matchup_pace_avg"].fillna(0) * pressure_scaled

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
        _agg_dict: dict[str, tuple[str, str]] = {
            "pts_allowed": ("points", "sum"),
            "reb_allowed": ("rebounds", "sum"),
            "ast_allowed": ("assists", "sum"),
        }
        if "fg3m" in pg.columns:
            _agg_dict["fg3m_allowed"] = ("fg3m", "sum")
        if "fg3a" in pg.columns:
            _agg_dict["fg3a_allowed"] = ("fg3a", "sum")
        defense_stats = pg.groupby(["game_id", "game_time_utc", "opp", "pos_group"]).agg(
            **_agg_dict
        ).reset_index()

        # Compute rolling averages per defending team and position group
        defense_stats = defense_stats.sort_values("game_time_utc")
        for stat in ["pts", "reb", "ast", "fg3m", "fg3a"]:
            col = f"{stat}_allowed"
            if col not in defense_stats.columns:
                continue
            grp = defense_stats.groupby(["opp", "pos_group"])[col]
            defense_stats[f"opp_{stat}_allowed_to_pos_avg10"] = grp.transform(
                lambda s: s.shift(1).rolling(10, min_periods=3).mean()
            )

        # Derived: opponent 3-point percentage allowed to position group
        if "opp_fg3m_allowed_to_pos_avg10" in defense_stats.columns and "opp_fg3a_allowed_to_pos_avg10" in defense_stats.columns:
            safe_fg3a = defense_stats["opp_fg3a_allowed_to_pos_avg10"].clip(lower=0.1)
            defense_stats["opp_fg3_pct_allowed_to_pos_avg10"] = (
                defense_stats["opp_fg3m_allowed_to_pos_avg10"] / safe_fg3a
            )

        # League-relative suppression context by position group.
        for stat in ["pts", "reb", "ast", "fg3m"]:
            feat = f"opp_{stat}_allowed_to_pos_avg10"
            if feat not in defense_stats.columns:
                continue
            league_avg = defense_stats.groupby("pos_group")[feat].transform("mean")
            q25 = defense_stats.groupby("pos_group")[feat].transform(lambda s: s.quantile(0.25))
            defense_stats[f"{feat}_vs_league"] = defense_stats[feat] - league_avg
            defense_stats[f"{feat}_tough_flag"] = (
                defense_stats[feat] <= q25
            ).astype(float)

        # Merge back: player's opp + pos_group + game_id -> opponent defensive profile
        _def_merge_cols = [
            "game_id", "opp", "pos_group",
            "opp_pts_allowed_to_pos_avg10",
            "opp_reb_allowed_to_pos_avg10",
            "opp_ast_allowed_to_pos_avg10",
        ]
        for _dc in ["opp_fg3m_allowed_to_pos_avg10", "opp_fg3a_allowed_to_pos_avg10",
                     "opp_fg3_pct_allowed_to_pos_avg10",
                     "opp_pts_allowed_to_pos_avg10_vs_league", "opp_reb_allowed_to_pos_avg10_vs_league",
                     "opp_ast_allowed_to_pos_avg10_vs_league", "opp_fg3m_allowed_to_pos_avg10_vs_league",
                     "opp_pts_allowed_to_pos_avg10_tough_flag", "opp_reb_allowed_to_pos_avg10_tough_flag",
                     "opp_ast_allowed_to_pos_avg10_tough_flag", "opp_fg3m_allowed_to_pos_avg10_tough_flag"]:
            if _dc in defense_stats.columns:
                _def_merge_cols.append(_dc)
        defense_merge = defense_stats[_def_merge_cols].copy()
        pg = pg.merge(defense_merge, on=["game_id", "opp", "pos_group"], how="left")
    else:
        pg["pos_group"] = "F"
        pg["opp_pts_allowed_to_pos_avg10"] = np.nan
        pg["opp_reb_allowed_to_pos_avg10"] = np.nan
        pg["opp_ast_allowed_to_pos_avg10"] = np.nan
        pg["opp_fg3m_allowed_to_pos_avg10"] = np.nan
        pg["opp_fg3a_allowed_to_pos_avg10"] = np.nan
        pg["opp_fg3_pct_allowed_to_pos_avg10"] = np.nan
        pg["opp_pts_allowed_to_pos_avg10_vs_league"] = np.nan
        pg["opp_reb_allowed_to_pos_avg10_vs_league"] = np.nan
        pg["opp_ast_allowed_to_pos_avg10_vs_league"] = np.nan
        pg["opp_fg3m_allowed_to_pos_avg10_vs_league"] = np.nan
        pg["opp_pts_allowed_to_pos_avg10_tough_flag"] = np.nan
        pg["opp_reb_allowed_to_pos_avg10_tough_flag"] = np.nan
        pg["opp_ast_allowed_to_pos_avg10_tough_flag"] = np.nan
        pg["opp_fg3m_allowed_to_pos_avg10_tough_flag"] = np.nan

    # --- BRef opponent defensive stats (season-level, joined on season + opp) ---
    print("  Loading BRef opponent defensive stats...", flush=True)
    pg = _merge_bref_opponent_defense(pg)

    # --- Phase 8: Matchup delta features ---
    # Player's recent average minus what the opponent allows to that position
    if "pre_points_avg10" in pg.columns and "opp_pts_allowed_to_pos_avg10" in pg.columns:
        pg["player_vs_opp_pts_delta"] = (
            pg["pre_points_avg10"].fillna(0) - pg["opp_pts_allowed_to_pos_avg10"].fillna(0)
        )
    else:
        pg["player_vs_opp_pts_delta"] = 0.0
    if "pre_rebounds_avg10" in pg.columns and "opp_reb_allowed_to_pos_avg10" in pg.columns:
        pg["player_vs_opp_reb_delta"] = (
            pg["pre_rebounds_avg10"].fillna(0) - pg["opp_reb_allowed_to_pos_avg10"].fillna(0)
        )
    else:
        pg["player_vs_opp_reb_delta"] = 0.0
    if "pre_assists_avg10" in pg.columns and "opp_ast_allowed_to_pos_avg10" in pg.columns:
        pg["player_vs_opp_ast_delta"] = (
            pg["pre_assists_avg10"].fillna(0) - pg["opp_ast_allowed_to_pos_avg10"].fillna(0)
        )
    else:
        pg["player_vs_opp_ast_delta"] = 0.0
    if "pre_fg3m_avg10" in pg.columns and "opp_fg3m_allowed_to_pos_avg10" in pg.columns:
        pg["player_vs_opp_fg3m_delta"] = (
            pg["pre_fg3m_avg10"].fillna(0) - pg["opp_fg3m_allowed_to_pos_avg10"].fillna(0)
        )
    else:
        pg["player_vs_opp_fg3m_delta"] = 0.0

    # --- Live-only fields (filled at prediction time) ---
    # Keep these columns in training so feature schemas stay stable.
    pg["injury_availability_prob"] = np.nan
    pg["injury_unavailability_prob"] = np.nan
    pg["injury_is_out"] = 0
    pg["injury_is_doubtful"] = 0
    pg["injury_is_questionable"] = 0
    pg["injury_is_probable"] = 0
    pg["lineup_confirmed"] = 1
    pg["confirmed_starter"] = pg.get("starter", pd.Series(np.nan, index=pg.index)).astype(float)
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

    # --- Derived Vegas features: implied pace and minutes context ---
    # League average total (~228) and pace (~100) are rough anchors; the ratio
    # converts an implied game total into an implied pace.  A 240-total game
    # implies ~5% faster pace than average, meaning more possessions and more
    # counting-stat opportunities for everyone.
    _LEAGUE_AVG_TOTAL = 228.0
    _LEAGUE_AVG_PACE = 100.0
    impl_total = pg["implied_total"]
    impl_team_total = pg.get("implied_team_total", pd.Series(np.nan, index=pg.index))
    # Implied pace: proportional scaling of league-average pace by game total
    pg["implied_pace"] = impl_total / _LEAGUE_AVG_TOTAL * _LEAGUE_AVG_PACE
    # Implied team scoring rate relative to their recent average
    team_ppg = pg.get("team_pre_off_rating_avg5", pd.Series(np.nan, index=pg.index))
    team_poss = pg.get("team_pre_possessions_avg5", pd.Series(np.nan, index=pg.index))
    team_recent_ppg = (team_ppg.fillna(110) * team_poss.fillna(100) / 100.0)
    safe_recent_ppg = team_recent_ppg.clip(lower=80.0)
    # How many more/fewer points does Vegas expect vs the team's recent average?
    pg["implied_team_total_vs_recent"] = impl_team_total - safe_recent_ppg
    # Implied minutes context: if Vegas expects higher team scoring, starters
    # should play closer to full minutes (game stays competitive / high-paced)
    pre_min = pg.get("pre_minutes_avg5", pd.Series(np.nan, index=pg.index)).fillna(0)
    pg["implied_minutes_boost"] = pre_min * (impl_team_total / safe_recent_ppg - 1.0)
    # Implied pace delta vs player's recent pace environment
    player_recent_pace = pg.get("matchup_pace_avg", pd.Series(np.nan, index=pg.index))
    pg["implied_pace_delta"] = pg["implied_pace"] - player_recent_pace.fillna(pg["implied_pace"])

    # --- Player career / age-rest interactions (computed from training data) ---
    # Cumulative game count across all seasons for this player
    pg["player_career_games"] = pg.groupby("player_id").cumcount() + 1
    pg["is_veteran"] = (pg["player_career_games"] > 200).astype(float)
    rest = pg["player_days_rest"].fillna(2.0)
    pg["veteran_rest_effect"] = pg["is_veteran"] * rest
    pg["career_x_rest"] = pg["player_career_games"] * rest

    # Step 3: load_x_age deferred here so player_career_games exists
    pg["load_x_age"] = pg["season_total_minutes"] * pg["player_career_games"]

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
      - Generic fallback fields (`prop_open_line`, etc.) chosen once by fixed
        stat priority (no overwrite across stat types).
      - Stat-specific fields (`prop_open_line_points`, etc.) for coherent
        per-target modeling.
    """
    df = player_df.copy()
    cache_dir = prop_cache_dir or PROP_CACHE_DIR
    stat_types = ["points", "rebounds", "assists", "fg3m", "minutes"]

    # Initialize generic columns (backward compatibility)
    for col in ["prop_open_line", "prop_line_vs_avg5", "prop_line_vs_avg10",
                "implied_over_prob", "line_available"]:
        df[col] = np.nan
    df["prop_line_stat_type"] = ""
    df["line_available"] = 0.0

    # Initialize stat-specific columns
    for stat in stat_types:
        for col in [
            f"prop_open_line_{stat}",
            f"prop_line_vs_avg5_{stat}",
            f"prop_line_vs_avg10_{stat}",
            f"implied_over_prob_{stat}",
            f"line_available_{stat}",
        ]:
            df[col] = np.nan
        df[f"line_available_{stat}"] = 0.0

    if "game_date_est" not in df.columns:
        return df

    # Build lookup from cached prop line files
    line_lookup: dict[str, dict[str, float]] = {}  # "{date}_{name_norm}_{team}_{stat}" -> {open_line, over_implied, ...}

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
            lines_df = _normalize_and_dedupe_prop_lines(lines_df, default_date=file_date)
            if lines_df.empty:
                continue
            _add_implied_probs(lines_df)
            for _, lr in lines_df.iterrows():
                name_norm = normalize_player_name(lr.get("player_name", ""))
                stat = str(lr.get("stat_type", ""))
                team_val = lr.get("team", "")
                team_norm = normalize_espn_abbr(str(team_val).strip()) if pd.notna(team_val) and str(team_val).strip() else ""
                if not name_norm or not stat:
                    continue
                open_line = lr.get("open_line", lr.get("line", np.nan))
                over_implied = lr.get("over_implied_prob", np.nan)
                under_implied = lr.get("under_implied_prob", np.nan)
                if pd.isna(open_line):
                    continue
                vig = (
                    float(over_implied + under_implied - 1.0)
                    if pd.notna(over_implied) and pd.notna(under_implied)
                    else np.inf
                )
                key = f"{file_date}_{name_norm}_{team_norm}_{stat}"
                payload = {
                    "open_line": float(open_line),
                    "over_implied": float(over_implied) if pd.notna(over_implied) else np.nan,
                    "vig": vig,
                    "source_pri": _prop_source_priority(lr.get("source", "")),
                }
                prev = line_lookup.get(key)
                if (
                    prev is None
                    or payload["vig"] < float(prev.get("vig", np.inf))
                    or (
                        payload["vig"] == float(prev.get("vig", np.inf))
                        and payload["source_pri"] > int(prev.get("source_pri", 0))
                    )
                ):
                    line_lookup[key] = payload

    if not line_lookup:
        return df

    # Process row by row, matching player + date + stat_type
    dates = df["game_date_est"].astype(str)
    player_norms = df["player_name"].map(normalize_player_name) if "player_name" in df.columns else pd.Series("", index=df.index)
    if "team" in df.columns:
        team_norms = df["team"].map(
            lambda t: normalize_espn_abbr(str(t).strip()) if pd.notna(t) and str(t).strip() else ""
        )
    else:
        team_norms = pd.Series("", index=df.index)

    # Fill stat-specific fields for every available stat, and fill generic once
    # in a stable priority order to avoid last-write-wins overwrite behavior.
    for stat in stat_types:
        avg5_col = f"pre_{stat}_avg5"
        avg10_col = f"pre_{stat}_avg10"
        if avg5_col not in df.columns:
            continue

        keys_team = dates + "_" + player_norms + "_" + team_norms + f"_{stat}"
        keys_any_team = dates + "_" + player_norms + "__" + stat
        matched = keys_team.map(line_lookup).combine_first(keys_any_team.map(line_lookup)).dropna()
        if matched.empty:
            continue

        for idx, match_data in matched.items():
            if not isinstance(match_data, dict):
                continue
            open_line = match_data.get("open_line", np.nan)
            if pd.isna(open_line):
                continue
            # Stat-specific
            df.loc[idx, f"prop_open_line_{stat}"] = open_line
            df.loc[idx, f"line_available_{stat}"] = 1.0
            over_implied = match_data.get("over_implied")
            if pd.notna(over_implied):
                df.loc[idx, f"implied_over_prob_{stat}"] = over_implied
            # Compute line vs averages
            avg5_val = df.loc[idx, avg5_col] if avg5_col in df.columns else np.nan
            avg10_val = df.loc[idx, avg10_col] if avg10_col in df.columns else np.nan
            if pd.notna(avg5_val):
                df.loc[idx, f"prop_line_vs_avg5_{stat}"] = open_line - float(avg5_val)
            if pd.notna(avg10_val):
                df.loc[idx, f"prop_line_vs_avg10_{stat}"] = open_line - float(avg10_val)

            # Generic fallback: populate once using stat priority.
            if float(df.loc[idx, "line_available"]) == 0.0:
                df.loc[idx, "prop_open_line"] = open_line
                df.loc[idx, "line_available"] = 1.0
                df.loc[idx, "prop_line_stat_type"] = stat
                if pd.notna(over_implied):
                    df.loc[idx, "implied_over_prob"] = over_implied
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
# Phase 6: Generalized Feature Ablation Framework
# ---------------------------------------------------------------------------

FEATURE_GROUPS_FOR_ABLATION: dict[str, list[str]] = {
    "market_lines": [
        "prop_open_line", "prop_line_vs_avg5", "prop_line_vs_avg10",
        "implied_over_prob", "line_available",
        "prop_open_line_points", "prop_line_vs_avg5_points", "prop_line_vs_avg10_points",
        "implied_over_prob_points", "line_available_points",
        "prop_open_line_rebounds", "prop_line_vs_avg5_rebounds", "prop_line_vs_avg10_rebounds",
        "implied_over_prob_rebounds", "line_available_rebounds",
        "prop_open_line_assists", "prop_line_vs_avg5_assists", "prop_line_vs_avg10_assists",
        "implied_over_prob_assists", "line_available_assists",
        "prop_open_line_fg3m", "prop_line_vs_avg5_fg3m", "prop_line_vs_avg10_fg3m",
        "implied_over_prob_fg3m", "line_available_fg3m",
        "prop_open_line_minutes", "prop_line_vs_avg5_minutes", "prop_line_vs_avg10_minutes",
        "implied_over_prob_minutes", "line_available_minutes",
    ],
    "boxscore_advanced": [
        "pre_adv_usage_pct_avg5", "pre_adv_usage_pct_avg10", "pre_adv_usage_pct_ewm5",
        "pre_adv_usage_pct_avg3",
        "pre_adv_pace_avg5", "pre_adv_pace_avg10", "pre_adv_possessions_avg5",
        "pre_adv_possessions_avg10",
        "pre_adv_off_rating_avg5", "pre_adv_off_rating_avg10",
        "pre_adv_ts_pct_avg5", "pre_adv_ts_pct_avg10",
        "pre_adv_ast_pct_avg5", "pre_adv_ast_pct_avg10",
        "pre_adv_reb_pct_avg5", "pre_adv_reb_pct_avg10",
    ],
    "injury_context": [
        "team_injury_pressure", "team_injury_proxy_missing_minutes5",
        "team_injury_pressure_bwd",
        "team_injury_proxy_missing_points5", "team_star_player_absent_flag",
        "team_active_count",
        "usage_boost_proxy", "minutes_x_injury_pressure", "pace_x_injury_pressure",
        "role_pts_injury_beta20", "role_reb_injury_beta20",
        "role_ast_injury_beta20", "role_min_injury_beta20",
    ],
    "matchup": [
        "matchup_pace_avg", "matchup_off_vs_def", "pace_diff_vs_recent",
        "opp_pts_allowed_to_pos_avg10", "opp_reb_allowed_to_pos_avg10",
        "opp_ast_allowed_to_pos_avg10",
        "opp_fg3m_allowed_to_pos_avg10", "opp_fg3a_allowed_to_pos_avg10",
        "opp_fg3_pct_allowed_to_pos_avg10",
        "player_vs_opp_pts_delta", "player_vs_opp_reb_delta", "player_vs_opp_ast_delta",
        "player_vs_opp_fg3m_delta",
    ],
    "referee": [
        "ref_crew_avg_total", "ref_crew_avg_fta", "ref_crew_avg_fouls",
        "ref_crew_avg_pace", "ref_crew_total_over_league_avg", "ref_crew_pace_over_league_avg",
    ],
    "vegas_context": [
        "implied_total", "implied_spread", "implied_team_total",
        "abs_spread", "spread_x_starter", "is_big_favorite",
        "implied_pace", "implied_team_total_vs_recent",
        "implied_minutes_boost", "implied_pace_delta",
    ],
    "recency": [
        "pre_points_ewm_var5", "pre_points_trend5", "pre_points_recent_vs_season",
        "pre_rebounds_ewm_var5", "pre_rebounds_trend5", "pre_rebounds_recent_vs_season",
        "pre_assists_ewm_var5", "pre_assists_trend5", "pre_assists_recent_vs_season",
        "pre_fg3m_ewm_var5", "pre_fg3m_trend5", "pre_fg3m_recent_vs_season",
        "pre_minutes_ewm_var5", "pre_minutes_trend5", "pre_minutes_recent_vs_season",
    ],
    "distribution": [
        "pre_points_p75_20", "pre_points_p90_20", "pre_points_max20", "pre_points_recent_zscore",
        "pre_rebounds_p75_20", "pre_rebounds_p90_20", "pre_rebounds_max20", "pre_rebounds_recent_zscore",
        "pre_assists_p75_20", "pre_assists_p90_20", "pre_assists_max20", "pre_assists_recent_zscore",
        "pre_fg3m_p75_20", "pre_fg3m_p90_20", "pre_fg3m_max20", "pre_fg3m_recent_zscore",
        "pre_minutes_p75_20", "pre_minutes_p90_20", "pre_minutes_max20", "pre_minutes_recent_zscore",
    ],
    "shrinkage": [
        "pre_points_shrunk", "pre_rebounds_shrunk", "pre_assists_shrunk",
        "pre_fg3m_shrunk", "pre_minutes_shrunk",
        "role_change_flag", "role_change_direction", "usage_regime_shift",
    ],
    "tracking": [
        "pre_trk_touches_avg3", "pre_trk_touches_avg5", "pre_trk_touches_avg10",
        "pre_trk_touches_ewm5",
        "pre_trk_drives_avg3", "pre_trk_drives_avg5", "pre_trk_drives_avg10",
        "pre_trk_drives_ewm5",
        "pre_trk_passes_avg5", "pre_trk_passes_avg10",
        "pre_trk_catch_shoot_fga_avg5",
        "pre_trk_catch_shoot_fg3a_avg5", "pre_trk_catch_shoot_fg3a_avg10",
        "pre_trk_catch_shoot_fg3a_ewm5",
        "pre_trk_pull_up_fga_avg5", "pre_trk_pull_up_fg3a_avg5",
        "pre_trk_contested_shots_avg5",
        "pre_trk_uncontested_fga_avg5",
        "pre_trk_touches_per_min_avg5",
        "pre_trk_drives_per_min_avg5",
        "pre_trk_catch_shoot_fg3a_per_min_avg5",
    ],
    "implied_vegas": [
        "implied_pace", "implied_team_total_vs_recent",
        "implied_minutes_boost", "implied_pace_delta",
    ],
    "opp_3pt_defense": [
        "opp_fg3m_allowed_to_pos_avg10", "opp_fg3a_allowed_to_pos_avg10",
        "opp_fg3_pct_allowed_to_pos_avg10", "player_vs_opp_fg3m_delta",
    ],
    "hustle": [
        "pre_trk_deflections_avg5", "pre_trk_deflections_avg10",
        "pre_trk_deflections_per_min_avg5",
        "pre_trk_loose_balls_avg5",
        "pre_trk_box_outs_avg3", "pre_trk_box_outs_avg5", "pre_trk_box_outs_avg10",
        "pre_trk_box_outs_ewm5", "pre_trk_box_outs_per_min_avg5",
        "pre_trk_off_box_outs_avg5", "pre_trk_def_box_outs_avg5",
        "pre_trk_reb_chances_avg3", "pre_trk_reb_chances_avg5",
        "pre_trk_reb_chances_avg10", "pre_trk_reb_chances_ewm5",
        "pre_trk_reb_chances_per_min_avg5",
        "pre_trk_reb_chances_off_avg5", "pre_trk_reb_chances_def_avg5",
        "pre_trk_screen_assists_avg5", "pre_trk_screen_assists_avg10",
        "pre_trk_screen_assists_per_min_avg5",
        "pre_trk_secondary_assists_avg5", "pre_trk_secondary_assists_avg10",
        "pre_trk_pts_paint_avg5", "pre_trk_pts_paint_avg10",
        "pre_trk_pts_fast_break_avg5", "pre_trk_pts_off_to_avg5",
    ],
    "defensive_matchup": [
        "pre_def_matchup_fga_avg5", "pre_def_matchup_fga_avg10",
        "pre_def_matchup_fga_per_min_avg5",
        "pre_def_matchup_fg_pct_avg5", "pre_def_matchup_fg_pct_ewm5",
        "pre_def_matchup_3pt_pct_avg5", "pre_def_matchup_3pt_pct_ewm5",
        "pre_def_matchup_fgm_avg5",
        "pre_def_matchup_3pa_avg5", "pre_def_matchup_3pm_avg5",
        "pre_def_matchup_player_pts_avg5",
        "pre_def_switches_on_avg5",
    ],
    "scoring_context": [
        "pre_scr_pct_assisted_2pt_avg5", "pre_scr_pct_assisted_2pt_ewm5",
        "pre_scr_pct_assisted_3pt_avg5", "pre_scr_pct_assisted_3pt_ewm5",
        "pre_scr_pct_assisted_fgm_avg5",
        "pre_scr_pct_unassisted_2pt_avg5", "pre_scr_pct_unassisted_2pt_ewm5",
        "pre_scr_pct_unassisted_3pt_avg5",
        "pre_scr_pct_fga_2pt_avg5", "pre_scr_pct_fga_3pt_avg5",
        "pre_scr_pct_pts_paint_avg5", "pre_scr_pct_pts_paint_ewm5",
        "pre_scr_pct_pts_midrange_avg5", "pre_scr_pct_pts_midrange_ewm5",
        "pre_scr_pct_pts_fastbreak_avg5",
    ],
    "rotation": [
        "pre_rot_stints_avg5", "pre_rot_total_stint_min_avg5",
        "pre_rot_avg_stint_min_avg5", "pre_rot_max_stint_min_avg5",
        "pre_rot_avg_stint_min_ewm5", "pre_rot_max_stint_min_ewm5",
        "pre_role_minutes_consistency5", "pre_role_minutes_expansion3",
        "pre_role_minutes_expansion5", "pre_role_minutes_reduction5",
        "pre_role_passes_expansion5",
    ],
    "matchups_v3": [
        "pre_mtch_partial_poss_avg5", "pre_mtch_partial_poss_ewm5",
        "pre_mtch_partial_poss_per_min_avg5",
        "pre_mtch_fga_avg5", "pre_mtch_fg_pct_avg5",
        "pre_mtch_3pa_avg5", "pre_mtch_3pt_pct_avg5",
        "pre_mtch_ast_avg5", "pre_mtch_pts_avg5",
        "opp_pts_allowed_to_pos_avg10_vs_league", "opp_reb_allowed_to_pos_avg10_vs_league",
        "opp_ast_allowed_to_pos_avg10_vs_league", "opp_fg3m_allowed_to_pos_avg10_vs_league",
        "opp_pts_allowed_to_pos_avg10_tough_flag", "opp_reb_allowed_to_pos_avg10_tough_flag",
        "opp_ast_allowed_to_pos_avg10_tough_flag", "opp_fg3m_allowed_to_pos_avg10_tough_flag",
    ],
    "bref_advanced": [
        # BRef player advanced stats — only NEW columns not redundant with adv_*
        "pre_bref_adv_bpm_avg5", "pre_bref_adv_bpm_avg10", "pre_bref_adv_bpm_ewm5",
        "pre_bref_adv_bpm_avg3",
        "pre_bref_adv_def_rtg_avg5", "pre_bref_adv_def_rtg_avg10", "pre_bref_adv_def_rtg_ewm5",
        "pre_bref_adv_def_rtg_avg3",
        "pre_bref_adv_efg_pct_avg5", "pre_bref_adv_efg_pct_avg10", "pre_bref_adv_efg_pct_ewm5",
        "pre_bref_adv_stl_pct_avg5", "pre_bref_adv_stl_pct_avg10",
        "pre_bref_adv_blk_pct_avg5", "pre_bref_adv_blk_pct_avg10",
        "pre_bref_adv_orb_pct_avg5", "pre_bref_adv_orb_pct_avg10",
        "pre_bref_adv_drb_pct_avg5", "pre_bref_adv_drb_pct_avg10",
        "pre_bref_adv_tov_pct_avg5", "pre_bref_adv_tov_pct_avg10",
        # BRef opponent defense (season-level)
        "bref_opp_pts_per_g", "bref_opp_fg_pct", "bref_opp_fg3_pct",
        "bref_opp_trb_per_g", "bref_opp_ast_per_g", "bref_opp_tov_per_g",
        "bref_opp_avg_dist", "bref_opp_pct_fga_3p",
        "bref_opp_fg_pct_0_3", "bref_opp_fg_pct_16_3pt",
    ],
}


def run_feature_ablation(
    player_df: pd.DataFrame,
    feature_groups: dict[str, list[str]] | None = None,
    targets: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generalized walk-forward feature group ablation.

    For each feature group, runs walk-forward backtest twice:
    once with all features, once with the group removed.
    Reports per-group lift/drag with paired t-test significance.
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS_FOR_ABLATION
    if targets is None:
        targets = ["points", "rebounds", "assists"]
        if "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 200:
            targets.append("fg3m")

    if "season" not in player_df.columns:
        print("  Error: season column required for walk-forward ablation.", flush=True)
        return {}

    seasons = sorted(player_df["season"].unique())
    if len(seasons) < 3:
        print(f"  Need >= 3 seasons. Have: {seasons}", flush=True)
        return {}

    results: dict[str, dict[str, Any]] = {}

    for group_name, drop_features in feature_groups.items():
        print(f"\n  === Ablation: removing '{group_name}' ({len(drop_features)} features) ===", flush=True)
        group_results: dict[str, dict[str, list[float]]] = {
            t: {"full_mae": [], "ablated_mae": []} for t in targets
        }

        for i in range(2, len(seasons)):
            test_season = seasons[i]
            train_seasons = seasons[:i]
            train = player_df[player_df["season"].isin(train_seasons)].copy()
            test = player_df[player_df["season"] == test_season].copy()

            if len(train) < 500 or len(test) < 100:
                continue

            for target in targets:
                if target not in train.columns or train[target].notna().sum() < 200:
                    continue

                full_features = get_feature_list(target, two_stage=False)
                ablated_features = [f for f in full_features if f not in drop_features]

                try:
                    # Full model
                    imp_f, mod_f, used_f = train_prop_model(train, full_features, target)
                    test_valid = test.dropna(subset=[target]).copy()
                    if test_valid.empty:
                        continue
                    pred_f = predict_prop(imp_f, mod_f, used_f, test_valid)
                    mae_f = float(mean_absolute_error(test_valid[target], pred_f))
                    group_results[target]["full_mae"].append(mae_f)

                    # Ablated model (without group)
                    imp_a, mod_a, used_a = train_prop_model(train, ablated_features, target)
                    pred_a = predict_prop(imp_a, mod_a, used_a, test_valid)
                    mae_a = float(mean_absolute_error(test_valid[target], pred_a))
                    group_results[target]["ablated_mae"].append(mae_a)
                except ValueError:
                    continue

        # Report results for this group
        results[group_name] = {}
        for target in targets:
            full_maes = group_results[target]["full_mae"]
            ablated_maes = group_results[target]["ablated_mae"]
            if len(full_maes) >= 2 and len(ablated_maes) >= 2:
                f_mean = np.mean(full_maes)
                a_mean = np.mean(ablated_maes)
                diff = a_mean - f_mean  # positive = group helps (removing it hurts)
                pct = (diff / f_mean * 100) if f_mean > 0 else 0
                # Paired t-test on per-fold deltas
                deltas = [a - f for a, f in zip(ablated_maes, full_maes)]
                if len(deltas) >= 2 and np.std(deltas) > 0:
                    t_stat, p_val = sp_stats.ttest_1samp(deltas, 0)
                else:
                    t_stat, p_val = np.nan, np.nan
                sig = "*" if pd.notna(p_val) and p_val < 0.05 else ""
                verdict = "HELPS" if diff > 0 else "HURTS" if diff < 0 else "NEUTRAL"
                results[group_name][target] = {
                    "full_mae": round(f_mean, 3),
                    "ablated_mae": round(a_mean, 3),
                    "diff": round(diff, 4),
                    "diff_pct": round(pct, 2),
                    "p_value": round(float(p_val), 4) if pd.notna(p_val) else None,
                    "verdict": verdict,
                }
                print(
                    f"    {target:>10s}: full={f_mean:.3f}  w/o {group_name}={a_mean:.3f}  "
                    f"delta={diff:+.4f} ({pct:+.1f}%) [{verdict}]{sig}",
                    flush=True,
                )
            else:
                print(f"    {target:>10s}: insufficient folds for comparison", flush=True)

    # Log experiment results
    for group_name, group_data in results.items():
        metrics: dict[str, Any] = {}
        for target, target_data in group_data.items():
            metrics[f"mae_{target}"] = target_data.get("full_mae")
            metrics[f"ablated_mae_{target}"] = target_data.get("ablated_mae")
            metrics[f"diff_pct_{target}"] = target_data.get("diff_pct")
            metrics[f"verdict_{target}"] = target_data.get("verdict")
        append_experiment_result(
            experiment_type="feature_ablation",
            description=f"Ablation: removing '{group_name}'",
            metrics=metrics,
        )

    return results


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
        "team_injury_pressure_bwd",
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
        # Derived Vegas: implied pace and minutes context
        "implied_pace",
        "implied_team_total_vs_recent",
        "implied_minutes_boost",
        "implied_pace_delta",
        # Opponent positional defense
        "opp_pts_allowed_to_pos_avg10",
        "opp_reb_allowed_to_pos_avg10",
        "opp_ast_allowed_to_pos_avg10",
        "opp_fg3m_allowed_to_pos_avg10",
        "opp_fg3a_allowed_to_pos_avg10",
        "opp_fg3_pct_allowed_to_pos_avg10",
        # Player career / age-rest interactions
        "player_career_games",
        "is_veteran",
        "veteran_rest_effect",
        "career_x_rest",
        # Step 3: Cumulative season load
        "season_games_played",
        "season_total_minutes",
        "high_load_flag",
        "load_x_b2b",
        "load_x_age",
        # Referee crew features (from game-level monolith data)
        "ref_crew_avg_total",
        "ref_crew_avg_fta",
        "ref_crew_avg_fouls",
        "ref_crew_avg_pace",
        "ref_crew_total_over_league_avg",
        "ref_crew_pace_over_league_avg",
        # Foul trouble / engagement
        "pre_fouls_personal_avg5",
        "pre_fouls_personal_avg10",
        # Phase 8: Matchup deltas
        "player_vs_opp_pts_delta",
        "player_vs_opp_reb_delta",
        "player_vs_opp_ast_delta",
        "player_vs_opp_fg3m_delta",
        "pre_rot_avg_stint_min_avg5",
        "pre_rot_max_stint_min_avg5",
        "pre_mtch_partial_poss_avg5",
        "pre_role_minutes_consistency5",
        "pre_role_minutes_expansion5",
        "pre_role_passes_expansion5",
        "team_top_creator_out",
        "team_top_rebounder_out",
        "wowy_points_top_creator_out_delta20",
        "wowy_assists_top_creator_out_delta20",
        "wowy_minutes_top_creator_out_delta20",
        "wowy_rebounds_top_rebounder_out_delta20",
        "wowy_minutes_top_rebounder_out_delta20",
        "opp_pts_allowed_to_pos_avg10_vs_league",
        "opp_reb_allowed_to_pos_avg10_vs_league",
        "opp_ast_allowed_to_pos_avg10_vs_league",
        "opp_fg3m_allowed_to_pos_avg10_vs_league",
        "opp_pts_allowed_to_pos_avg10_tough_flag",
        "opp_reb_allowed_to_pos_avg10_tough_flag",
        "opp_ast_allowed_to_pos_avg10_tough_flag",
        "opp_fg3m_allowed_to_pos_avg10_tough_flag",
    ]
    # Player tracking features (touches, drives, catch-and-shoot)
    if USE_TRACKING_FEATURES:
        common += [
            "pre_trk_touches_avg5", "pre_trk_touches_avg10",
            "pre_trk_touches_avg3", "pre_trk_touches_ewm5",
            "pre_trk_drives_avg5", "pre_trk_drives_avg10",
            "pre_trk_drives_avg3", "pre_trk_drives_ewm5",
            "pre_trk_passes_avg5",
            "pre_trk_contested_shots_avg5",
            # Per-minute tracking rates
            "pre_trk_touches_per_min_avg5",
            "pre_trk_drives_per_min_avg5",
            "pre_trk_passes_per_min_avg5",
            # Hustle stats (from BDL advanced)
            "pre_trk_deflections_avg5", "pre_trk_deflections_avg10",
            "pre_trk_deflections_per_min_avg5",
            "pre_trk_loose_balls_avg5",
            "pre_player_pass_share_avg5", "pre_player_pass_share_avg10",
            "pre_player_ast_per_pass_avg5", "pre_player_ast_per_pass_avg10",
            "pre_player_reb_chance_share_avg5", "pre_player_reb_chance_share_avg10",
            "pre_player_reb_conversion_avg5", "pre_player_reb_conversion_avg10",
        ]
    if USE_BOXSCORE_ADV_FEATURES:
        common += [
            "pre_adv_usage_pct_avg5", "pre_adv_usage_pct_avg10", "pre_adv_usage_pct_ewm5",
            "pre_adv_pace_avg5", "pre_adv_possessions_avg5",
            "pre_adv_off_rating_avg5", "pre_adv_ts_pct_avg5",
        ]
    if USE_BREF_FEATURES:
        common += [
            # BRef advanced (fills gaps where stats.nba.com data is sparse)
            "pre_bref_adv_bpm_avg5", "pre_bref_adv_bpm_avg10", "pre_bref_adv_bpm_ewm5",
            "pre_bref_adv_def_rtg_avg5", "pre_bref_adv_def_rtg_avg10",
            "pre_bref_adv_efg_pct_avg5",
            # BRef opponent defense (season-level context)
            "bref_opp_pts_per_g", "bref_opp_fg_pct", "bref_opp_fg3_pct",
            "bref_opp_trb_per_g", "bref_opp_ast_per_g",
        ]

    # Target-specific features
    # NOTE: season averages + venue splits (expanding averages) excluded to prevent
    # anchoring predictions to stale early-season production. avg3/5/10 + EWM are sufficient.
    if target == "minutes":
        # Minutes model: no per-minute features to avoid circularity
        specific = [
            "pre_points_avg3", "pre_points_avg5",
            # Phase 3: Enhanced minutes model features
            "dnp_rate_last10",
            "pre_fouls_per_min_avg5",
            "starter_benched_rate3",
            "pre_rot_total_stint_min_avg5",
            "pre_rot_max_stint_min_avg5",
            # Phase 7: Recency features for minutes
            "pre_minutes_ewm_var5",
            "pre_minutes_trend5",
            "pre_minutes_recent_vs_season",
            "pre_minutes_p75_20", "pre_minutes_p90_20", "pre_minutes_max20",
            "pre_minutes_same_role_p75_20", "pre_minutes_same_role_p90_20", "pre_minutes_same_role_max20",
            "pre_minutes_recent_zscore",
            # Phase 12: Shrinkage + role-change
            "pre_minutes_shrunk",
            "role_change_flag", "role_change_direction", "usage_regime_shift",
            "pre_role_minutes_consistency5", "pre_role_minutes_expansion3",
            "pre_role_minutes_expansion5", "pre_role_minutes_reduction5",
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
            # Phase 7: Recency features
            "pre_points_ewm_var5", "pre_points_trend5", "pre_points_recent_vs_season",
            "pre_points_p75_20", "pre_points_p90_20", "pre_points_max20", "pre_points_recent_zscore",
            "pre_points_same_role_p75_20", "pre_points_same_role_p90_20", "pre_points_same_role_max20",
            # Phase 12: Shrinkage + role-change
            "pre_points_shrunk",
            "role_change_flag", "role_change_direction", "usage_regime_shift",
        ]
        if USE_BOXSCORE_ADV_FEATURES:
            specific += [
                "pre_adv_usage_pct_avg3",
                "pre_adv_off_rating_avg10",
                "pre_adv_ts_pct_avg10",
            ]
        if USE_BREF_FEATURES:
            specific += [
                # BRef: BPM (strong points predictor, not redundant with adv_off_rating)
                "pre_bref_adv_bpm_avg3",
                "pre_bref_adv_tov_pct_avg5",  # turnovers reduce scoring
                # Opponent paint/perimeter defense quality
                "bref_opp_fg_pct_0_3", "bref_opp_fg_pct_16_3pt",
            ]
        if USE_TRACKING_FEATURES:
            specific += [
                # Drives → paint scoring, pull-up volume → perimeter scoring
                "pre_trk_pull_up_fga_avg5",
                "pre_trk_uncontested_fga_avg5",
                # Scoring breakdown: paint + fast break points
                "pre_trk_pts_paint_avg5", "pre_trk_pts_paint_avg10",
                "pre_trk_pts_fast_break_avg5",
                "pre_trk_pts_off_to_avg5",
                # Scoring context: shot creation vs assisted (from stats.nba.com)
                "pre_scr_pct_unassisted_2pt_avg5", "pre_scr_pct_unassisted_2pt_ewm5",
                "pre_scr_pct_pts_paint_avg5", "pre_scr_pct_pts_paint_ewm5",
                "pre_scr_pct_pts_midrange_avg5", "pre_scr_pct_pts_midrange_ewm5",
                "pre_scr_pct_pts_fastbreak_avg5",
                "pre_scr_pct_fga_2pt_avg5", "pre_scr_pct_fga_3pt_avg5",
                # Defensive matchup: opponent FG% allowed (quality of defense)
                "pre_def_matchup_fg_pct_avg5", "pre_def_matchup_fg_pct_ewm5",
                "pre_mtch_fga_avg5", "pre_mtch_pts_avg5",
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
            # Phase 7: Recency features
            "pre_rebounds_ewm_var5", "pre_rebounds_trend5", "pre_rebounds_recent_vs_season",
            "pre_rebounds_p75_20", "pre_rebounds_p90_20", "pre_rebounds_max20", "pre_rebounds_recent_zscore",
            "pre_rebounds_same_role_p75_20", "pre_rebounds_same_role_p90_20", "pre_rebounds_same_role_max20",
            # Phase 12: Shrinkage + role-change
            "pre_rebounds_shrunk",
            "role_change_flag", "role_change_direction", "usage_regime_shift",
            "wowy_rebounds_top_rebounder_out_delta20",
            "wowy_minutes_top_rebounder_out_delta20",
        ]
        if USE_BOXSCORE_ADV_FEATURES:
            specific += [
                "pre_adv_reb_pct_avg5", "pre_adv_reb_pct_avg10",
                "pre_adv_possessions_avg10",
            ]
        if USE_BREF_FEATURES:
            specific += [
                # BRef: rebound % breakdown (off/def separate — not redundant with adv_reb_pct total)
                "pre_bref_adv_orb_pct_avg5", "pre_bref_adv_orb_pct_avg10",
                "pre_bref_adv_drb_pct_avg5", "pre_bref_adv_drb_pct_avg10",
                # Opponent rebound context
                "bref_opp_trb_per_g",
            ]
        if USE_TRACKING_FEATURES:
            specific += [
                # Box-outs directly predict rebound opportunity creation
                "pre_trk_box_outs_avg3", "pre_trk_box_outs_avg5", "pre_trk_box_outs_avg10",
                "pre_trk_box_outs_ewm5",
                "pre_trk_box_outs_per_min_avg5",
                "pre_trk_off_box_outs_avg5", "pre_trk_def_box_outs_avg5",
                # Rebound chances: how many loose boards the player contests
                "pre_trk_reb_chances_avg3", "pre_trk_reb_chances_avg5",
                "pre_trk_reb_chances_avg10", "pre_trk_reb_chances_ewm5",
                "pre_trk_reb_chances_per_min_avg5",
                "pre_trk_reb_chances_off_avg5", "pre_trk_reb_chances_def_avg5",
                "pre_player_reb_chance_share_avg5", "pre_player_reb_chance_share_avg10",
                "pre_player_reb_conversion_avg5", "pre_player_reb_conversion_avg10",
                # Defensive engagement: more matchup FGA = more involved on defense = more rebounding
                "pre_def_matchup_fga_avg5", "pre_def_matchup_fga_avg10",
                "pre_def_matchup_fga_per_min_avg5",
                "pre_mtch_partial_poss_avg5", "pre_mtch_partial_poss_per_min_avg5",
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
            # Phase 7: Recency features
            "pre_assists_ewm_var5", "pre_assists_trend5", "pre_assists_recent_vs_season",
            "pre_assists_p75_20", "pre_assists_p90_20", "pre_assists_max20", "pre_assists_recent_zscore",
            "pre_assists_same_role_p75_20", "pre_assists_same_role_p90_20", "pre_assists_same_role_max20",
            # Phase 12: Shrinkage + role-change
            "pre_assists_shrunk",
            "role_change_flag", "role_change_direction", "usage_regime_shift",
            "wowy_assists_top_creator_out_delta20",
            "wowy_minutes_top_creator_out_delta20",
        ]
        if USE_BOXSCORE_ADV_FEATURES:
            specific += [
                "pre_adv_ast_pct_avg5", "pre_adv_ast_pct_avg10",
                "pre_adv_usage_pct_avg3",
            ]
        if USE_BREF_FEATURES:
            specific += [
                # BRef: turnover % (playmaking efficiency — ast_pct redundant with adv_ast_pct)
                "pre_bref_adv_tov_pct_avg5",
                # Opponent turnover forcing (more turnovers = fewer assist opportunities for opponent)
                "bref_opp_tov_per_g",
            ]
        if USE_TRACKING_FEATURES:
            specific += [
                # Passes are the strongest leading indicator for assists
                "pre_trk_passes_avg5", "pre_trk_passes_avg10",
                "pre_trk_passes_per_min_avg5",
                # Screen assists + secondary assists capture playmaking style
                "pre_trk_screen_assists_avg5", "pre_trk_screen_assists_avg10",
                "pre_trk_screen_assists_per_min_avg5",
                "pre_trk_secondary_assists_avg5", "pre_trk_secondary_assists_avg10",
                "pre_player_pass_share_avg5", "pre_player_pass_share_avg10",
                "pre_player_ast_per_pass_avg5", "pre_player_ast_per_pass_avg10",
                # Scoring context: how much of team's offense is assisted (playmaker signal)
                "pre_scr_pct_assisted_2pt_avg5", "pre_scr_pct_assisted_2pt_ewm5",
                "pre_scr_pct_assisted_3pt_avg5",
                "pre_scr_pct_assisted_fgm_avg5",
                "pre_mtch_ast_avg5",
                "pre_role_passes_expansion5",
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
            # Phase 7: Recency features
            "pre_fg3m_ewm_var5", "pre_fg3m_trend5", "pre_fg3m_recent_vs_season",
            "pre_fg3m_p75_20", "pre_fg3m_p90_20", "pre_fg3m_max20", "pre_fg3m_recent_zscore",
            "pre_fg3m_same_role_p75_20", "pre_fg3m_same_role_p90_20", "pre_fg3m_same_role_max20",
            # Phase 12: Shrinkage + role-change
            "pre_fg3m_shrunk",
            "role_change_flag", "role_change_direction", "usage_regime_shift",
        ]
        if USE_BOXSCORE_ADV_FEATURES:
            specific += [
                "pre_adv_usage_pct_avg3",
                "pre_adv_pace_avg10",
            ]
        if USE_BREF_FEATURES:
            specific += [
                # BRef: eFG% captures 3pt efficiency better than TS% for this target
                "pre_bref_adv_efg_pct_avg5", "pre_bref_adv_efg_pct_avg10",
                # Opponent 3pt defense quality
                "bref_opp_fg3_pct", "bref_opp_pct_fga_3p",
            ]
        if USE_TRACKING_FEATURES:
            specific += [
                # Catch-and-shoot volume is the strongest predictor of 3PM
                "pre_trk_catch_shoot_fg3a_avg5", "pre_trk_catch_shoot_fg3a_avg10",
                "pre_trk_catch_shoot_fg3a_ewm5",
                "pre_trk_catch_shoot_fga_avg5",
                # Pull-up 3PA: secondary 3-point source
                "pre_trk_pull_up_fg3a_avg5",
                "pre_trk_catch_shoot_fg3a_per_min_avg5",
                # Scoring context: 3pt shot share + assisted vs self-created 3s
                "pre_scr_pct_fga_3pt_avg5",
                "pre_scr_pct_assisted_3pt_avg5", "pre_scr_pct_assisted_3pt_ewm5",
                "pre_scr_pct_unassisted_3pt_avg5",
                # Defensive matchup: 3pt% allowed (opponent perimeter D quality)
                "pre_def_matchup_3pt_pct_avg5", "pre_def_matchup_3pt_pct_ewm5",
                "pre_mtch_3pa_avg5", "pre_mtch_3pt_pct_avg5",
            ]
        if two_stage:
            specific.append("pred_minutes")
    else:
        specific = []

    # Phase 3: Market line features (optional, gated behind use_market_features)
    market_feats: list[str] = []
    if use_market_features:
        market_feats = [
            f"prop_open_line_{target}",
            f"prop_line_vs_avg5_{target}",
            f"prop_line_vs_avg10_{target}",
            f"implied_over_prob_{target}",
            f"line_available_{target}",
        ]

    return list(dict.fromkeys(common + specific + market_feats))


def _selected_group_exclusions(selected_groups: list[str] | None) -> set[str]:
    """Return optional-group features to exclude for the selected feature-group set."""
    if selected_groups is None:
        return set()
    selected = set(selected_groups)
    excluded: set[str] = set()
    for group_name, feats in FEATURE_GROUPS_FOR_ABLATION.items():
        if group_name not in selected:
            excluded.update(feats)
    return excluded


def get_effective_feature_list(
    target: str,
    two_stage: bool = False,
    use_market_features: bool = False,
    selected_groups: list[str] | None = None,
) -> list[str]:
    """Return the effective feature list after applying persisted group selection."""
    full_features = get_feature_list(
        target,
        two_stage=two_stage,
        use_market_features=use_market_features,
    )
    excluded = _selected_group_exclusions(selected_groups)
    if not excluded:
        return full_features
    return [f for f in full_features if f not in excluded]


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


def get_residual_feature_list(target: str, selected_groups: list[str] | None = None) -> list[str]:
    """Feature list for Stage 3 residual model (Phase 4).

    Includes base features plus OOF predictions and interaction features.
    """
    base = get_effective_feature_list(
        target,
        two_stage=True,
        use_market_features=True,
        selected_groups=selected_groups,
    )
    line_feature = f"prop_open_line_{target}"
    residual_specific = [
        f"oof_pred_{target}",    # the base OOF prediction itself
        "oof_pred_minutes",       # OOF minutes prediction
        # Market context
        line_feature,
        "oof_pred_vs_line",       # oof_pred - open_line
        # Interaction features
        "oof_pred_x_b2b",            # oof_pred * is_b2b
        "oof_pred_x_injury_pressure",  # oof_pred * team_injury_pressure
    ]
    # Step 4: Player target encoding (only when enabled)
    if USE_PLAYER_TARGET_ENCODING:
        residual_specific.append(f"player_resid_enc_{target}")
    return base + residual_specific


# ---------------------------------------------------------------------------
# Model training and prediction
# ---------------------------------------------------------------------------


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


def _compute_prop_recency_weights(
    df: pd.DataFrame,
    include_starter_bonus: bool = False,
) -> pd.Series:
    """Compute recency weights for prop model training."""
    weights = pd.Series(1.0, index=df.index, dtype=float)
    if "game_time_utc" in df.columns:
        order = df["game_time_utc"].rank(method="first")
        weights = 0.4 + 0.6 * (order / max(order.max(), 1.0))
    if include_starter_bonus and "starter" in df.columns:
        weights = weights * (1.0 + 0.2 * (df["starter"].fillna(0) > 0).astype(float))
    return weights


def _build_prop_feature_signature(
    player_df: pd.DataFrame,
    targets: list[str],
    selected_groups: list[str] | None = None,
) -> dict[str, Any]:
    """Build a stable signature for tuned-param and feature-selection reuse."""
    feature_map = {
        target: get_effective_feature_list(
            target,
            two_stage=(target != "minutes"),
            selected_groups=selected_groups,
        )
        for target in targets
    }
    feature_blob = json.dumps(feature_map, sort_keys=True)
    seasons = sorted(str(s) for s in player_df.get("season", pd.Series(dtype=str)).dropna().unique())
    return {
        "player_feature_cache_version": PLAYER_FEATURE_CACHE_VERSION,
        "selected_groups": sorted(selected_groups or []),
        "targets": sorted(targets),
        "seasons": seasons,
        "n_rows": int(len(player_df)),
        "feature_hash": hashlib.sha256(feature_blob.encode("utf-8")).hexdigest(),
    }


def _prepare_two_stage_training_context(
    train_df: pd.DataFrame,
    tuned_params: dict[str, dict[str, Any]] | None = None,
    selected_groups: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build the leakage-safe stage-2 training frame used by production models."""
    ensemble_active = USE_ENSEMBLE and _HAS_LGBM
    tp = tuned_params or {}

    min_features = get_effective_feature_list(
        "minutes",
        two_stage=False,
        selected_groups=selected_groups,
    )
    train_for_minutes = train_df.copy()
    train_for_minutes["pred_starter_prob"] = train_for_minutes.get("pre_starter_rate", np.nan)
    if "confirmed_starter" in train_for_minutes.columns:
        mask_known = train_for_minutes["confirmed_starter"].notna()
        if mask_known.any():
            train_for_minutes.loc[mask_known, "pred_starter_prob"] = (
                train_for_minutes.loc[mask_known, "confirmed_starter"].astype(float)
            )

    recency_weight = _compute_prop_recency_weights(train_for_minutes, include_starter_bonus=True)

    min_imp, min_model, min_feats = train_prop_model(
        train_for_minutes,
        min_features,
        "minutes",
        params=tp.get("xgb_minutes"),
        sample_weight=recency_weight,
    )
    minute_models: dict[str, Any] = {"minutes": (min_imp, min_model, min_feats)}

    if ensemble_active:
        try:
            lgbm_min_imp, lgbm_min_model, lgbm_min_feats = train_prop_model_lgbm(
                train_for_minutes,
                min_features,
                "minutes",
                params=tp.get("lgbm_minutes"),
                sample_weight=recency_weight,
            )
            minute_models["_lgbm_minutes"] = (lgbm_min_imp, lgbm_min_model, lgbm_min_feats)
        except ValueError:
            pass

    stage2_train = train_for_minutes.copy()
    if ensemble_active and "_lgbm_minutes" in minute_models:
        oof_min_xgb, oof_min_lgbm = _time_series_oof_ensemble(
            stage2_train,
            min_features,
            "minutes",
            sample_weight=recency_weight,
            min_rows=700,
        )
        valid_both = oof_min_xgb.notna() & oof_min_lgbm.notna()
        if valid_both.sum() >= 100:
            min_meta = train_stacked_model(
                oof_min_xgb,
                oof_min_lgbm,
                stage2_train["minutes"],
            )
            minute_models["_meta_minutes"] = min_meta
            oof_minutes = pd.Series(np.nan, index=stage2_train.index, dtype=float)
            if min_meta is not None:
                stacked = predict_stacked(
                    oof_min_xgb[valid_both].values,
                    oof_min_lgbm[valid_both].values,
                    min_meta,
                )
                oof_minutes.loc[valid_both[valid_both].index] = stacked
            else:
                oof_minutes.loc[valid_both[valid_both].index] = (
                    0.5 * oof_min_xgb[valid_both] + 0.5 * oof_min_lgbm[valid_both]
                )
            oof_minutes = oof_minutes.fillna(oof_min_xgb)
            if verbose:
                print(f"  Ensemble minutes: {valid_both.sum()} stacked OOF rows", flush=True)
        else:
            oof_minutes = _time_series_oof_predictions(
                stage2_train,
                min_features,
                "minutes",
                sample_weight=recency_weight,
                min_rows=700,
            )
    else:
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

    stat_targets = ["points", "rebounds", "assists"]
    if "fg3m" in stage2_train.columns and stage2_train["fg3m"].notna().sum() > 100:
        stat_targets.append("fg3m")

    stat_recency_weight = _compute_prop_recency_weights(stage2_train, include_starter_bonus=False)
    return {
        "minute_models": minute_models,
        "train_for_minutes": train_for_minutes,
        "minutes_features": min_features,
        "minutes_sample_weight": recency_weight,
        "stage2_train": stage2_train,
        "stat_targets": stat_targets,
        "stat_sample_weight": stat_recency_weight,
        "selected_groups": list(selected_groups or []),
    }


# ---------------------------------------------------------------------------
# Optuna Hyperparameter Tuning for Props Models
# ---------------------------------------------------------------------------

def optuna_tune_xgb_regressor_props(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    sample_weight: pd.Series | np.ndarray | None = None,
    n_trials: int = 50,
    n_splits: int = 3,
    min_train: int = 500,
) -> dict[str, Any]:
    """Tune XGBoost regressor hyperparameters for props using time-series CV."""
    if not _HAS_OPTUNA:
        print("  Optuna not installed — skipping tuning.", flush=True)
        return {}

    valid = df.dropna(subset=[target]).copy()
    feats = [f for f in filter_features(features, valid) if valid[f].notna().any()]
    if len(valid) < min_train or not feats:
        return {}

    valid = valid.sort_values("game_time_utc").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, val_idx in tscv.split(valid):
        if len(train_idx) >= min_train:
            folds.append((valid.iloc[train_idx], valid.iloc[val_idx]))
    if len(folds) < 2:
        return {}

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
        maes = []
        for fold_train, fold_val in folds:
            fold_feats = [f for f in feats if f in fold_train.columns and fold_train[f].notna().any()]
            if not fold_feats:
                continue
            fold_sw = None
            if sample_weight is not None:
                if isinstance(sample_weight, pd.Series):
                    fold_sw = sample_weight.reindex(fold_train.index).to_numpy(dtype=float)
                else:
                    sw_arr = np.asarray(sample_weight, dtype=float)
                    if len(sw_arr) == len(df):
                        fold_sw = pd.Series(sw_arr, index=df.index).reindex(fold_train.index).to_numpy(dtype=float)
                if fold_sw is not None:
                    fold_sw = np.where(np.isfinite(fold_sw), fold_sw, 1.0)
                    fold_sw = np.clip(fold_sw, 0.05, None)
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(fold_train[fold_feats])
            X_va = imp.transform(fold_val[fold_feats])
            model = XGBRegressor(
                **params, eval_metric="mae", random_state=42, verbosity=0,
            )
            model.fit(X_tr, fold_train[target], sample_weight=fold_sw)
            pred = model.predict(X_va)
            maes.append(mean_absolute_error(fold_val[target], pred))
        return float(np.mean(maes)) if maes else 1e6

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optuna_tune_lgbm_regressor_props(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    sample_weight: pd.Series | np.ndarray | None = None,
    n_trials: int = 50,
    n_splits: int = 3,
    min_train: int = 500,
) -> dict[str, Any]:
    """Tune LightGBM regressor hyperparameters for props using time-series CV."""
    if not _HAS_OPTUNA or not _HAS_LGBM:
        return {}

    valid = df.dropna(subset=[target]).copy()
    feats = [f for f in filter_features(features, valid) if valid[f].notna().any()]
    if len(valid) < min_train or not feats:
        return {}

    valid = valid.sort_values("game_time_utc").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_idx, val_idx in tscv.split(valid):
        if len(train_idx) >= min_train:
            folds.append((valid.iloc[train_idx], valid.iloc[val_idx]))
    if len(folds) < 2:
        return {}

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.001, 5.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        }
        maes = []
        for fold_train, fold_val in folds:
            fold_feats = [f for f in feats if f in fold_train.columns and fold_train[f].notna().any()]
            if not fold_feats:
                continue
            fold_sw = None
            if sample_weight is not None:
                if isinstance(sample_weight, pd.Series):
                    fold_sw = sample_weight.reindex(fold_train.index).to_numpy(dtype=float)
                else:
                    sw_arr = np.asarray(sample_weight, dtype=float)
                    if len(sw_arr) == len(df):
                        fold_sw = pd.Series(sw_arr, index=df.index).reindex(fold_train.index).to_numpy(dtype=float)
                if fold_sw is not None:
                    fold_sw = np.where(np.isfinite(fold_sw), fold_sw, 1.0)
                    fold_sw = np.clip(fold_sw, 0.05, None)
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(fold_train[fold_feats])
            X_va = imp.transform(fold_val[fold_feats])
            model = LGBMRegressor(
                **params, random_state=42, verbosity=-1,
            )
            model.fit(X_tr, fold_train[target], sample_weight=fold_sw)
            pred = model.predict(X_va)
            maes.append(mean_absolute_error(fold_val[target], pred))
        return float(np.mean(maes)) if maes else 1e6

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _run_props_tuning(
    train_df: pd.DataFrame,
    targets: list[str],
    n_trials: int = 50,
    selected_groups: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run Optuna tuning for all prop targets, returning best params per target/model.

    Returns dict like: {"xgb_points": {...}, "lgbm_points": {...}, ...}
    Persists results to TUNED_PARAMS_FILE for reuse.
    """
    tuned: dict[str, dict[str, Any]] = {}
    context = _prepare_two_stage_training_context(
        train_df,
        tuned_params=None,
        selected_groups=selected_groups,
        verbose=False,
    )

    for target in targets:
        if target == "minutes":
            tune_df = context["train_for_minutes"]
            features = context["minutes_features"]
            sample_weight = context["minutes_sample_weight"]
        else:
            tune_df = context["stage2_train"]
            features = get_effective_feature_list(
                target,
                two_stage=True,
                selected_groups=selected_groups,
            )
            sample_weight = context["stat_sample_weight"]

        print(f"  Tuning XGB for {target} ({n_trials} trials)...", flush=True)
        xgb_params = optuna_tune_xgb_regressor_props(
            tune_df,
            features,
            target,
            sample_weight=sample_weight,
            n_trials=n_trials,
        )
        if xgb_params:
            tuned[f"xgb_{target}"] = xgb_params
            print(f"    XGB {target}: lr={xgb_params.get('learning_rate', '?'):.4f} "
                  f"depth={xgb_params.get('max_depth', '?')} "
                  f"n_est={xgb_params.get('n_estimators', '?')}", flush=True)

        if _HAS_LGBM:
            print(f"  Tuning LGBM for {target} ({n_trials} trials)...", flush=True)
            lgbm_params = optuna_tune_lgbm_regressor_props(
                tune_df,
                features,
                target,
                sample_weight=sample_weight,
                n_trials=n_trials,
            )
            if lgbm_params:
                tuned[f"lgbm_{target}"] = lgbm_params
                print(f"    LGBM {target}: lr={lgbm_params.get('learning_rate', '?'):.4f} "
                      f"depth={lgbm_params.get('max_depth', '?')} "
                      f"n_est={lgbm_params.get('n_estimators', '?')}", flush=True)

    # Persist to disk
    if tuned:
        signature = _build_prop_feature_signature(
            train_df,
            targets,
            selected_groups=selected_groups,
        )
        payload = {
            "meta": {
                **signature,
                "n_trials": n_trials,
                "updated_at_utc": datetime.utcnow().isoformat(),
            },
            "params": tuned,
        }
        TUNED_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TUNED_PARAMS_FILE, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  Saved tuned params to {TUNED_PARAMS_FILE}", flush=True)

    return tuned


def _load_tuned_params(
    player_df: pd.DataFrame,
    targets: list[str],
    selected_groups: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load cached tuned params from disk, if available."""
    for tuned_file in _artifact_read_candidates(TUNED_PARAMS_FILE, LEGACY_TUNED_PARAMS_FILE):
        try:
            with open(tuned_file) as f:
                payload = json.load(f)
            if isinstance(payload, dict) and "params" in payload:
                expected = _build_prop_feature_signature(
                    player_df,
                    targets,
                    selected_groups=selected_groups,
                )
                meta = payload.get("meta", {})
                meta_subset = {
                    "player_feature_cache_version": meta.get("player_feature_cache_version"),
                    "selected_groups": meta.get("selected_groups", []),
                    "targets": meta.get("targets", []),
                    "seasons": meta.get("seasons", []),
                    "n_rows": meta.get("n_rows"),
                    "feature_hash": meta.get("feature_hash"),
                }
                if meta_subset == expected:
                    if tuned_file != TUNED_PARAMS_FILE:
                        print(f"  Loaded tuned params from legacy artifact {tuned_file.name}", flush=True)
                    return payload.get("params", {})
                print(f"  Ignoring stale tuned params cache {tuned_file.name} (feature/training signature mismatch).", flush=True)
                continue
            if isinstance(payload, dict):
                print(f"  Ignoring legacy tuned params cache without metadata: {tuned_file.name}", flush=True)
                continue
        except (json.JSONDecodeError, IOError):
            continue
    return {}


# ---------------------------------------------------------------------------
# Persistent Experiment Tracking
# ---------------------------------------------------------------------------

def append_experiment_result(
    experiment_type: str,
    description: str,
    metrics: dict[str, Any],
    feature_set_hash: str | None = None,
) -> None:
    """Append a single experiment result to the persistent experiment log CSV."""
    PROP_LOG_DIR.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment_type": experiment_type,
        "description": description,
        "feature_set_hash": feature_set_hash or "",
    }
    # Flatten per-target metrics: "mae_points", "r2_rebounds", etc.
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                row[f"{k}_{sub_k}"] = sub_v
        else:
            row[k] = v

    new_row = pd.DataFrame([row])
    write_header = not EXPERIMENT_LOG_FILE.exists()
    try:
        new_row.to_csv(
            EXPERIMENT_LOG_FILE, mode="a", header=write_header, index=False,
        )
    except Exception:
        # Fallback: full rewrite if append fails (e.g., schema mismatch)
        if EXPERIMENT_LOG_FILE.exists():
            try:
                existing = pd.read_csv(EXPERIMENT_LOG_FILE)
                combined = pd.concat([existing, new_row], ignore_index=True)
            except Exception:
                combined = new_row
        else:
            combined = new_row
        combined.to_csv(EXPERIMENT_LOG_FILE, index=False)


# ---------------------------------------------------------------------------
# Automated Feature Group Selection
# ---------------------------------------------------------------------------

def load_selected_feature_groups(
    player_df: pd.DataFrame | None = None,
    targets: list[str] | None = None,
) -> list[str] | None:
    """Load previously selected feature groups from disk. Returns None if not available."""
    for feature_file in _artifact_read_candidates(FEATURE_SELECTION_FILE, LEGACY_FEATURE_SELECTION_FILE):
        try:
            with open(feature_file) as f:
                data = json.load(f)
            if player_df is not None and targets is not None:
                meta = data.get("meta", {})
                expected = _build_prop_feature_signature(
                    player_df,
                    targets,
                    selected_groups=data.get("selected_groups"),
                )
                meta_subset = {
                    "player_feature_cache_version": meta.get("player_feature_cache_version"),
                    "selected_groups": meta.get("selected_groups", []),
                    "targets": meta.get("targets", []),
                    "seasons": meta.get("seasons", []),
                    "n_rows": meta.get("n_rows"),
                    "feature_hash": meta.get("feature_hash"),
                }
                if meta_subset != expected:
                    print(f"  Ignoring stale feature-selection cache {feature_file.name} (feature/training signature mismatch).", flush=True)
                    continue
            if feature_file != FEATURE_SELECTION_FILE:
                print(f"  Loaded feature-selection artifact from legacy file {feature_file.name}", flush=True)
            return data.get("selected_groups")
        except (json.JSONDecodeError, IOError):
            continue
    return None


def run_auto_feature_selection(
    player_df: pd.DataFrame,
    targets: list[str] | None = None,
    min_improvement_pct: float = 0.5,
    min_folds_pass: int = 2,
    total_folds: int = 3,
) -> list[str]:
    """Greedy forward selection of feature groups using walk-forward validation.

    Starts with base features (those not in any ablation group), iteratively adds
    the group that improves aggregate MAE the most, stopping when no group clears
    the improvement threshold across enough folds.

    Returns list of selected group names. Persists to FEATURE_SELECTION_FILE.
    """
    if targets is None:
        targets = ["points", "rebounds", "assists"]
        if "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 200:
            targets.append("fg3m")

    if "season" not in player_df.columns:
        print("  Error: season column required for feature selection.", flush=True)
        return []

    seasons = sorted(player_df["season"].unique())
    if len(seasons) < 3:
        print(f"  Need >= 3 seasons. Have: {seasons}", flush=True)
        return []

    feature_groups = FEATURE_GROUPS_FOR_ABLATION
    all_group_features = set()
    for feats in feature_groups.values():
        all_group_features.update(feats)

    # Build walk-forward folds
    folds = []
    for i in range(max(2, len(seasons) - total_folds), len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]
        train = player_df[player_df["season"].isin(train_seasons)].copy()
        test = player_df[player_df["season"] == test_season].copy()
        if len(train) >= 500 and len(test) >= 100:
            folds.append((train, test))

    if len(folds) < 2:
        print("  Insufficient walk-forward folds for feature selection.", flush=True)
        return []

    def _evaluate_groups(included_groups: list[str]) -> dict[str, list[float]]:
        """Evaluate MAE per target across folds with only included feature groups."""
        results: dict[str, list[float]] = {t: [] for t in targets}
        for train, test in folds:
            try:
                fold_models = train_two_stage_models(
                    train,
                    selected_groups=included_groups,
                    include_post_models=False,
                    verbose=False,
                )
            except ValueError:
                continue
            if not fold_models:
                continue
            pred_test = predict_two_stage(fold_models, test.copy())
            for target in targets:
                pred_col = f"pred_{target}"
                if pred_col not in pred_test.columns or target not in pred_test.columns:
                    continue
                test_valid = pred_test.dropna(subset=[target, pred_col])
                if len(test_valid) < 50:
                    continue
                mae = float(mean_absolute_error(test_valid[target], test_valid[pred_col]))
                results[target].append(mae)
        return results

    def _aggregate_mae(results: dict[str, list[float]]) -> float:
        """Mean MAE across all targets and folds."""
        all_maes = []
        for maes in results.values():
            all_maes.extend(maes)
        return float(np.mean(all_maes)) if all_maes else 1e6

    # Start: no optional groups (base features only)
    selected: list[str] = []
    remaining = list(feature_groups.keys())

    base_results = _evaluate_groups(selected)
    initial_base_mae = _aggregate_mae(base_results)
    best_mae = initial_base_mae
    print(f"  Base MAE (no optional groups): {best_mae:.4f}", flush=True)

    # Greedy forward selection
    while remaining:
        best_candidate = None
        best_candidate_mae = best_mae
        best_candidate_results: dict[str, list[float]] = {}

        baseline_results = base_results
        for candidate in remaining:
            trial_groups = selected + [candidate]
            trial_results = _evaluate_groups(trial_groups)
            trial_mae = _aggregate_mae(trial_results)

            if trial_mae < best_candidate_mae:
                best_candidate = candidate
                best_candidate_mae = trial_mae
                best_candidate_results = trial_results

        if best_candidate is None:
            print("  No group improves MAE. Stopping.", flush=True)
            break

        improvement_pct = (best_mae - best_candidate_mae) / best_mae * 100
        # Check per-target fold consistency: count targets where majority of folds improve
        folds_pass = 0
        for target in targets:
            base_maes = baseline_results.get(target, [])
            cand_maes = best_candidate_results.get(target, [])
            if len(base_maes) == len(cand_maes) and len(base_maes) >= 1:
                n_better = sum(1 for b, c in zip(base_maes, cand_maes) if c < b)
                if n_better >= len(base_maes) / 2:
                    folds_pass += 1

        if improvement_pct >= min_improvement_pct and folds_pass >= min_folds_pass:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_mae = best_candidate_mae
            base_results = best_candidate_results
            print(f"  + Added '{best_candidate}': MAE {best_mae:.4f} "
                  f"(improvement: {improvement_pct:.2f}%, {folds_pass}/{len(targets)} targets)", flush=True)
        else:
            print(f"  ~ Rejected '{best_candidate}': improvement {improvement_pct:.2f}% "
                  f"({folds_pass}/{len(targets)} targets pass) — below threshold", flush=True)
            remaining.remove(best_candidate)

    # Persist
    result_data = {
        "selected_groups": selected,
        "base_mae": float(initial_base_mae),
        "final_mae": float(best_mae),
        "timestamp": datetime.utcnow().isoformat(),
        "meta": _build_prop_feature_signature(player_df, targets, selected_groups=selected),
    }
    FEATURE_SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_SELECTION_FILE, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  Selected {len(selected)} feature groups: {selected}", flush=True)
    print(f"  Saved to {FEATURE_SELECTION_FILE}", flush=True)

    # Log experiment
    append_experiment_result(
        experiment_type="auto_feature_selection",
        description=f"Forward selection: {len(selected)} groups from {len(feature_groups)}",
        metrics={
            "base_mae": result_data["base_mae"],
            "final_mae": result_data["final_mae"],
            "n_selected": len(selected),
            "selected_groups": ",".join(selected),
        },
    )

    return selected


def train_prop_model_lgbm(
    train_df: pd.DataFrame,
    features: list[str],
    target: str,
    params: dict[str, Any] | None = None,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> tuple[SimpleImputer, Any, list[str]]:
    """Train a LightGBM regressor for a player prop target."""
    if not _HAS_LGBM:
        raise ValueError("LightGBM not available")
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
    lgbm_defaults = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.1,
        "min_child_samples": 20,
        "num_leaves": 31,
    }
    if target == "minutes":
        lgbm_defaults.update({
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.04,
            "reg_lambda": 2.0,
            "reg_alpha": 0.2,
            "min_child_samples": 25,
        })
    model = _build_lgbm_regressor({**lgbm_defaults, **p})
    model.fit(X_imp, y, sample_weight=sw)
    return imp, model, feats


def _time_series_oof_ensemble(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    sample_weight: pd.Series | np.ndarray | None = None,
    min_rows: int = 500,
) -> tuple[pd.Series, pd.Series]:
    """Generate OOF predictions from both XGBoost and LightGBM for stacking.

    Returns (oof_xgb, oof_lgbm) as separate Series aligned to df.index.
    """
    valid = df.dropna(subset=[target]).copy()
    oof_xgb = pd.Series(np.nan, index=df.index, dtype=float)
    oof_lgbm = pd.Series(np.nan, index=df.index, dtype=float)

    if len(valid) < min_rows or not _HAS_LGBM:
        return oof_xgb, oof_lgbm

    n_splits = min(5, max(2, len(valid) // 1200))
    if n_splits < 2:
        return oof_xgb, oof_lgbm

    splitter = TimeSeriesSplit(n_splits=n_splits)
    xgb_oof = pd.Series(np.nan, index=valid.index, dtype=float)
    lgbm_oof = pd.Series(np.nan, index=valid.index, dtype=float)

    for train_idx, val_idx in splitter.split(valid):
        fold_train = valid.iloc[train_idx].copy()
        fold_val = valid.iloc[val_idx].copy()
        if fold_train.empty or fold_val.empty:
            continue
        if len(val_idx) < OOF_MIN_VAL_FOLD_SIZE:
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
            imp_x, model_x, feats_x = train_prop_model(
                fold_train, features, target, sample_weight=fold_sw,
            )
            xgb_oof.loc[fold_val.index] = predict_prop(imp_x, model_x, feats_x, fold_val)
        except ValueError:
            pass

        try:
            imp_l, model_l, feats_l = train_prop_model_lgbm(
                fold_train, features, target, sample_weight=fold_sw,
            )
            lgbm_oof.loc[fold_val.index] = predict_prop(imp_l, model_l, feats_l, fold_val)
        except ValueError:
            pass

    oof_xgb.loc[xgb_oof.index] = xgb_oof
    oof_lgbm.loc[lgbm_oof.index] = lgbm_oof
    return oof_xgb, oof_lgbm


def train_stacked_model(
    oof_xgb: pd.Series,
    oof_lgbm: pd.Series,
    y: pd.Series,
) -> Ridge | None:
    """Train Ridge regression meta-learner on OOF predictions from XGB + LGBM.

    Returns fitted Ridge model or None if insufficient data.
    """
    valid = oof_xgb.notna() & oof_lgbm.notna() & y.notna()
    if valid.sum() < 100:
        return None
    meta_X = np.column_stack([oof_xgb[valid].values, oof_lgbm[valid].values])
    meta_y = y[valid].values
    meta = Ridge(alpha=1.0)
    meta.fit(meta_X, meta_y)
    return meta


def predict_stacked(
    xgb_pred: np.ndarray,
    lgbm_pred: np.ndarray,
    meta_model: Ridge | None,
) -> np.ndarray:
    """Generate stacked prediction from XGB + LGBM base predictions."""
    if meta_model is None:
        return 0.5 * xgb_pred + 0.5 * lgbm_pred
    meta_X = np.column_stack([xgb_pred, lgbm_pred])
    return meta_model.predict(meta_X)


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
    if hasattr(pred_df, "columns") and pred_df.columns.duplicated().any():
        pred_df = pred_df.loc[:, ~pred_df.columns.duplicated()].copy()
    # Primary schema comes from training tuple to avoid cross-model drift.
    expected_features = list(dict.fromkeys(features))

    # If imputer expects a different width, prefer its own fitted schema.
    n_expected = int(getattr(imp, "n_features_in_", len(expected_features)))
    if len(expected_features) != n_expected:
        imp_schema = list(getattr(imp, "feature_names_in_", []))
        if len(imp_schema) == n_expected:
            expected_features = imp_schema
        elif len(expected_features) > n_expected:
            expected_features = expected_features[:n_expected]
        else:
            expected_features = expected_features + [
                f"__pad_missing_feature_{i}" for i in range(n_expected - len(expected_features))
            ]

    # Build positional matrix directly so duplicate feature names (if present in model schema)
    # keep their width/order and do not collapse in a DataFrame.
    cols: list[np.ndarray] = []
    n_rows = len(pred_df)
    for f in expected_features:
        if f in pred_df.columns:
            v = pd.to_numeric(pred_df[f], errors="coerce")
            if isinstance(v, pd.DataFrame):
                v = pd.to_numeric(v.iloc[:, 0], errors="coerce")
            cols.append(v.to_numpy(dtype=float))
        else:
            cols.append(np.full(n_rows, np.nan, dtype=float))

    if cols:
        X_mat = np.column_stack(cols).astype(float, copy=False)
    else:
        X_mat = np.empty((n_rows, 0), dtype=float)

    # Defensive width normalization.
    if X_mat.shape[1] != n_expected:
        if X_mat.shape[1] > n_expected:
            X_mat = X_mat[:, :n_expected]
        else:
            pad = np.full((n_rows, n_expected - X_mat.shape[1]), np.nan, dtype=float)
            X_mat = np.hstack([X_mat, pad])

    # ndarray bypasses strict feature-name checks; width/order is explicitly controlled above.
    X_imp = imp.transform(X_mat)
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
        # Phase 1: skip tiny validation folds to avoid noisy OOF predictions
        if len(val_idx) < OOF_MIN_VAL_FOLD_SIZE:
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


def compute_player_target_encoding(
    df: pd.DataFrame,
    oof_preds: pd.Series,
    target: str,
    shrinkage_n: int = 30,
) -> pd.Series:
    """Compute per-player expanding-window mean residual from OOF predictions.

    For each player: expanding mean of (actual - oof_pred) with shift(1) to prevent
    leakage, shrunk toward zero: adj = (n * player_mean_resid) / (n + shrinkage_n).

    Returns a Series aligned to df.index with the shrunk residual encoding.
    """
    work = df[[target, "player_id"]].copy()
    work["_oof_pred"] = oof_preds
    work["_resid"] = work[target] - work["_oof_pred"]

    valid = work["_resid"].notna()
    result = pd.Series(0.0, index=df.index, dtype=float)

    if not valid.any():
        return result

    # Per-player expanding mean of residuals, shifted by 1
    grouped = work.loc[valid].groupby("player_id")["_resid"]
    expanding_mean = grouped.transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    expanding_count = grouped.transform(lambda s: s.shift(1).expanding(min_periods=3).count())

    # Bayesian shrinkage toward zero
    shrunk = (expanding_count * expanding_mean) / (expanding_count + shrinkage_n)
    result.loc[expanding_mean.index] = shrunk.fillna(0.0)

    return result


def train_oof_residual_models(
    train_df: pd.DataFrame,
    two_stage_models: dict[str, Any],
    min_oof_rows: int = 500,
    selected_groups: list[str] | None = None,
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
    min_features = get_effective_feature_list(
        "minutes",
        two_stage=False,
        selected_groups=selected_groups,
    )
    oof_minutes = _time_series_oof_predictions(
        train_df, min_features, "minutes", min_rows=min_oof_rows,
    )
    train_df = train_df.copy()
    train_df["oof_pred_minutes"] = oof_minutes

    for target in stat_targets:
        if target not in train_df.columns:
            continue

        # Generate OOF base predictions for this stat
        stat_features = get_effective_feature_list(
            target,
            two_stage=True,
            selected_groups=selected_groups,
        )
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
        line_col = f"prop_open_line_{target}"
        valid_df["oof_pred_vs_line"] = valid_df[oof_col] - valid_df.get(line_col, np.nan)
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

        resid_features = get_residual_feature_list(target, selected_groups=selected_groups)

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


def train_uncertainty_models(
    train_df: pd.DataFrame,
    two_stage_models: dict[str, Any],
    min_oof_rows: int = 500,
    selected_groups: list[str] | None = None,
) -> dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]]:
    """Train Phase 4 uncertainty models: predict abs(actual - oof_pred) per stat.

    Output is a predicted MAE; multiply by sqrt(pi/2) ≈ 1.253 to convert to
    approximate std for the t-distribution.

    Returns dict mapping target -> (imputer, model, features).
    """
    uncertainty_models: dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]] = {}
    stat_targets = ["points", "rebounds", "assists"]
    if "fg3m" in train_df.columns and train_df["fg3m"].notna().sum() > 200:
        stat_targets.append("fg3m")

    # Need OOF minutes predictions
    min_features = get_effective_feature_list(
        "minutes",
        two_stage=False,
        selected_groups=selected_groups,
    )
    oof_minutes = _time_series_oof_predictions(
        train_df, min_features, "minutes", min_rows=min_oof_rows,
    )
    train_df = train_df.copy()
    train_df["oof_pred_minutes"] = oof_minutes

    for target in stat_targets:
        if target not in train_df.columns:
            continue

        stat_features = get_effective_feature_list(
            target,
            two_stage=True,
            selected_groups=selected_groups,
        )
        oof_preds = _time_series_oof_predictions(
            train_df, stat_features, target, min_rows=min_oof_rows,
        )

        oof_col = f"oof_pred_{target}"
        train_df[oof_col] = oof_preds
        valid_mask = train_df[oof_col].notna() & train_df[target].notna()
        valid_df = train_df[valid_mask].copy()

        if len(valid_df) < min_oof_rows:
            continue

        # Target: absolute error
        abs_error_col = f"_abs_error_{target}"
        valid_df[abs_error_col] = np.abs(valid_df[target] - valid_df[oof_col])

        # Features: same as stat model + oof prediction + market features
        line_col = f"prop_open_line_{target}"
        unc_features = stat_features + [
            oof_col,
            "oof_pred_minutes",
            line_col,
        ]

        # Conservative hyperparameters for uncertainty model
        unc_params = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "reg_lambda": 3.0,
            "reg_alpha": 0.3,
            "min_child_weight": 5,
        }

        try:
            imp, model, used_feats = train_prop_model(
                valid_df, unc_features, abs_error_col, params=unc_params,
            )
            uncertainty_models[target] = (imp, model, used_feats)
            pred_mae = valid_df[abs_error_col]
            print(
                f"    Uncertainty {target}: trained on {len(valid_df)} rows  "
                f"mean_abs_error={pred_mae.mean():.2f}",
                flush=True,
            )
        except ValueError as e:
            print(f"    Uncertainty {target}: training failed ({e})", flush=True)
            continue

    return uncertainty_models


def train_quantile_uncertainty_models(
    train_df: pd.DataFrame,
    two_stage_models: dict[str, Any],
    min_oof_rows: int = 500,
    selected_groups: list[str] | None = None,
) -> dict[str, dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]]]:
    """Train quantile regression models for uncertainty estimation.

    For each stat target, trains XGBoost with reg:quantileerror objective at
    alpha=0.25 and alpha=0.75. Derives robust std from IQR / 1.35.
    Optionally adds 10th/90th percentile models for tail risk.

    Returns dict mapping target -> {"q25": (imp, model, feats), "q75": ...}.
    """
    quantile_models: dict[str, dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]]] = {}
    stat_targets = ["points", "rebounds", "assists"]
    if "fg3m" in train_df.columns and train_df["fg3m"].notna().sum() > 200:
        stat_targets.append("fg3m")

    # Need OOF minutes predictions
    min_features = get_effective_feature_list(
        "minutes",
        two_stage=False,
        selected_groups=selected_groups,
    )
    oof_minutes = _time_series_oof_predictions(
        train_df, min_features, "minutes", min_rows=min_oof_rows,
    )
    train_df = train_df.copy()
    train_df["oof_pred_minutes"] = oof_minutes

    quantile_params_base = {
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_lambda": 3.0,
        "reg_alpha": 0.3,
        "min_child_weight": 5,
    }

    for target in stat_targets:
        if target not in train_df.columns:
            continue

        stat_features = get_effective_feature_list(
            target,
            two_stage=True,
            selected_groups=selected_groups,
        )
        oof_preds = _time_series_oof_predictions(
            train_df, stat_features, target, min_rows=min_oof_rows,
        )
        oof_col = f"oof_pred_{target}"
        train_df[oof_col] = oof_preds
        valid_mask = train_df[oof_col].notna() & train_df[target].notna()
        valid_df = train_df[valid_mask].copy()

        if len(valid_df) < min_oof_rows:
            continue

        line_col = f"prop_open_line_{target}"
        unc_features = stat_features + [oof_col, "oof_pred_minutes", line_col]

        target_quantile_models: dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]] = {}
        for alpha, label in [(0.25, "q25"), (0.75, "q75"), (0.10, "q10"), (0.90, "q90")]:
            try:
                feats = [f for f in filter_features(unc_features, valid_df)
                         if valid_df[f].notna().any()]
                if not feats:
                    continue

                X = valid_df[feats]
                y = valid_df[target]

                imp = SimpleImputer(strategy="median")
                X_imp = imp.fit_transform(X)

                model = XGBRegressor(
                    n_estimators=quantile_params_base["n_estimators"],
                    max_depth=quantile_params_base["max_depth"],
                    learning_rate=quantile_params_base["learning_rate"],
                    subsample=quantile_params_base["subsample"],
                    colsample_bytree=quantile_params_base["colsample_bytree"],
                    reg_lambda=quantile_params_base["reg_lambda"],
                    reg_alpha=quantile_params_base["reg_alpha"],
                    min_child_weight=quantile_params_base["min_child_weight"],
                    objective="reg:quantileerror",
                    quantile_alpha=alpha,
                    eval_metric="mae",
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X_imp, y)
                target_quantile_models[label] = (imp, model, feats)
            except Exception as e:
                print(f"    Quantile {target} {label}: failed ({e})", flush=True)
                continue

        if "q25" in target_quantile_models and "q75" in target_quantile_models:
            quantile_models[target] = target_quantile_models
            print(
                f"    Quantile {target}: trained {', '.join(sorted(target_quantile_models.keys()))} "
                f"on {len(valid_df)} rows",
                flush=True,
            )

    return quantile_models


def predict_quantile_std(
    quantile_models: dict[str, dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]]],
    stat_type: str,
    pred_row: pd.Series,
) -> tuple[float, float, float]:
    """Compute std and P(over) from quantile models for a single prediction row.

    Returns (quantile_std, p_over_nonparam, iqr) where:
    - quantile_std = IQR / 1.35 (robust std estimate)
    - p_over_nonparam = non-parametric P(over) from quantile position (NaN if unavailable)
    - iqr = raw IQR for diagnostics
    """
    if stat_type not in quantile_models:
        return np.nan, np.nan, np.nan

    models = quantile_models[stat_type]
    row_df = pd.DataFrame([pred_row])

    predictions = {}
    for label, (imp, model, feats) in models.items():
        try:
            pred = predict_prop(imp, model, feats, row_df)
            predictions[label] = float(pred[0])
        except Exception:
            pass

    q25 = predictions.get("q25", np.nan)
    q75 = predictions.get("q75", np.nan)

    if not (np.isfinite(q25) and np.isfinite(q75)):
        return np.nan, np.nan, np.nan

    iqr = q75 - q25
    # Robust std: IQR / 1.35 (for normal distribution, IQR ≈ 1.35 * sigma)
    quantile_std = max(iqr / 1.35, 0.5)  # Floor at 0.5 to prevent degenerate std

    return quantile_std, np.nan, iqr


def train_two_stage_models(
    train_df: pd.DataFrame,
    tuned_params: dict[str, dict[str, Any]] | None = None,
    selected_groups: list[str] | None = None,
    include_post_models: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train two-stage models: minutes first, then stats with predicted minutes.

    Stage 1: Train minutes model (XGBoost, optionally + LightGBM ensemble)
    Stage 2: For each stat target, train model that includes predicted minutes as feature
    Stage 3: OOF residual correction models
    Phase 4: Uncertainty models (quantile or abs-error based)

    If tuned_params is provided, looks up per-target XGB/LGBM params
    (keys like "xgb_minutes", "lgbm_points", etc.).
    """
    models: dict[str, Any] = {}
    tp = tuned_params or {}
    try:
        context = _prepare_two_stage_training_context(
            train_df,
            tuned_params=tuned_params,
            selected_groups=selected_groups,
            verbose=verbose,
        )
    except ValueError:
        return {}

    models.update(context["minute_models"])
    stage2_train = context["stage2_train"]
    stat_targets = context["stat_targets"]
    stat_recency_weight = context["stat_sample_weight"]
    ensemble_active = USE_ENSEMBLE and _HAS_LGBM

    for target in stat_targets:
        features = get_effective_feature_list(
            target,
            two_stage=True,
            selected_groups=selected_groups,
        )
        try:
            imp, model, feats = train_prop_model(
                stage2_train, features, target,
                params=tp.get(f"xgb_{target}"),
                sample_weight=stat_recency_weight,
            )
            models[target] = (imp, model, feats)

            # Train LightGBM for ensemble
            if ensemble_active:
                try:
                    lgbm_imp, lgbm_model, lgbm_feats = train_prop_model_lgbm(
                        stage2_train, features, target,
                        params=tp.get(f"lgbm_{target}"),
                        sample_weight=stat_recency_weight,
                    )
                    models[f"_lgbm_{target}"] = (lgbm_imp, lgbm_model, lgbm_feats)

                    # Train stacking meta-learner from OOF predictions
                    oof_xgb, oof_lgbm = _time_series_oof_ensemble(
                        stage2_train, features, target,
                        sample_weight=stat_recency_weight, min_rows=500,
                    )
                    meta = train_stacked_model(oof_xgb, oof_lgbm, stage2_train[target])
                    if meta is not None:
                        models[f"_meta_{target}"] = meta
                        n_stacked = (oof_xgb.notna() & oof_lgbm.notna()).sum()
                        if verbose:
                            print(f"  Ensemble {target}: stacked meta-learner on {n_stacked} OOF rows", flush=True)
                except ValueError:
                    pass
        except ValueError:
            # Fall back: train without pred_minutes
            features_no_stage = get_effective_feature_list(
                target,
                two_stage=False,
                selected_groups=selected_groups,
            )
            try:
                imp, model, feats = train_prop_model(
                    stage2_train, features_no_stage, target,
                    params=tp.get(f"xgb_{target}"),
                    sample_weight=stat_recency_weight,
                )
                models[target] = (imp, model, feats)
            except ValueError:
                continue

    # Step 4: Per-player target encoding (experimental, off by default)
    if USE_PLAYER_TARGET_ENCODING and "player_id" in stage2_train.columns:
        print("  Computing per-player target encodings...", flush=True)
        for target in stat_targets:
            if target not in models:
                continue
            stat_features = get_effective_feature_list(
                target,
                two_stage=True,
                selected_groups=selected_groups,
            )
            oof_preds = _time_series_oof_predictions(
                stage2_train, stat_features, target, min_rows=500,
            )
            enc_col = f"player_resid_enc_{target}"
            stage2_train[enc_col] = compute_player_target_encoding(
                stage2_train, oof_preds, target,
            )
            n_nonzero = (stage2_train[enc_col].abs() > 0.01).sum()
            if verbose:
                print(f"    {target}: {n_nonzero} players with non-trivial encoding", flush=True)

    # Stage 3 (Phase 4): OOF residual models
    if include_post_models:
        if verbose:
            print("  Training Stage 3 residual models (OOF)...", flush=True)
        residual_models = train_oof_residual_models(
            stage2_train,
            models,
            min_oof_rows=500,
            selected_groups=selected_groups,
        )
        if residual_models:
            models["_residual"] = residual_models
            if verbose:
                print(f"  Stage 3 residual models: {', '.join(sorted(residual_models.keys()))}", flush=True)
        elif verbose:
            print("  Stage 3: no residual models trained (insufficient OOF data)", flush=True)

    # Phase 4: Uncertainty models
    if include_post_models and USE_QUANTILE_UNCERTAINTY:
        if verbose:
            print("  Training quantile uncertainty models...", flush=True)
        quantile_models = train_quantile_uncertainty_models(
            stage2_train,
            models,
            min_oof_rows=500,
            selected_groups=selected_groups,
        )
        if quantile_models:
            models["_uncertainty_quantile"] = quantile_models
            if verbose:
                print(f"  Quantile uncertainty: {', '.join(sorted(quantile_models.keys()))}", flush=True)
        else:
            if verbose:
                print("  Quantile uncertainty: no models trained", flush=True)
            # Fall back to abs-error uncertainty
            if verbose:
                print("  Falling back to abs-error uncertainty models...", flush=True)
            uncertainty_models = train_uncertainty_models(
                stage2_train,
                models,
                min_oof_rows=500,
                selected_groups=selected_groups,
            )
            if uncertainty_models:
                models["_uncertainty"] = uncertainty_models
    elif include_post_models:
        if verbose:
            print("  Training uncertainty models (Phase 4)...", flush=True)
        uncertainty_models = train_uncertainty_models(
            stage2_train,
            models,
            min_oof_rows=500,
            selected_groups=selected_groups,
        )
        if uncertainty_models:
            models["_uncertainty"] = uncertainty_models
            if verbose:
                print(f"  Uncertainty models: {', '.join(sorted(uncertainty_models.keys()))}", flush=True)
        elif verbose:
            print("  Uncertainty: no models trained (insufficient OOF data)", flush=True)

    return models


def predict_two_stage(
    models: dict[str, Any],
    pred_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate two-stage predictions: minutes first, then use predicted minutes for stats.

    When ensemble models (LightGBM + meta-learner) are available, generates stacked
    predictions from XGBoost + LightGBM via Ridge meta-learner.
    """
    pred_df = pred_df.copy()

    # Stage 0: starter probability from historical rate with confirmed starter override.
    pred_df["pred_starter_prob"] = pred_df.get("pre_starter_rate", np.nan)

    # Apply confirmed lineup override if available
    if "lineup_confirmed" in pred_df.columns and "confirmed_starter" in pred_df.columns:
        known_mask = pred_df["lineup_confirmed"].fillna(0).astype(int).eq(1) & pred_df["confirmed_starter"].notna()
        if known_mask.any():
            pred_df.loc[known_mask, "pred_starter_prob"] = pred_df.loc[known_mask, "confirmed_starter"].astype(float)

    # Stage 1: Predict minutes (with optional ensemble)
    if "minutes" in models:
        min_imp, min_model, min_feats = models["minutes"]
        xgb_minutes = predict_prop(min_imp, min_model, min_feats, pred_df)

        if "_lgbm_minutes" in models:
            lgbm_imp, lgbm_model, lgbm_feats = models["_lgbm_minutes"]
            lgbm_minutes = predict_prop(lgbm_imp, lgbm_model, lgbm_feats, pred_df)
            meta = models.get("_meta_minutes")
            pred_df["pred_minutes"] = predict_stacked(xgb_minutes, lgbm_minutes, meta)
        else:
            pred_df["pred_minutes"] = xgb_minutes

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

    # Stage 2: Predict stats using predicted minutes (with optional ensemble)
    for target in ["points", "rebounds", "assists", "fg3m"]:
        if target not in models:
            continue
        imp, model, feats = models[target]
        xgb_pred = predict_prop(imp, model, feats, pred_df)

        lgbm_key = f"_lgbm_{target}"
        meta_key = f"_meta_{target}"
        if lgbm_key in models:
            lgbm_imp, lgbm_model, lgbm_feats = models[lgbm_key]
            lgbm_pred = predict_prop(lgbm_imp, lgbm_model, lgbm_feats, pred_df)
            meta = models.get(meta_key)
            pred_df[f"pred_{target}"] = predict_stacked(xgb_pred, lgbm_pred, meta)
        else:
            pred_df[f"pred_{target}"] = xgb_pred

        pred_df = _apply_stat_prediction_bounds(pred_df, f"pred_{target}", target)

    # Stage 3 (Phase 4): Apply residual correction with clipping
    residual_models = models.get("_residual", {})
    if residual_models:
        for target, (r_imp, r_model, r_feats) in residual_models.items():
            pred_col = f"pred_{target}"
            if pred_col not in pred_df.columns:
                continue

            # Build residual interaction features for prediction
            oof_col = f"oof_pred_{target}"
            line_col = f"prop_open_line_{target}"
            pred_df[oof_col] = pred_df[pred_col]  # at prediction time, base pred serves as "oof" input
            pred_df["oof_pred_minutes"] = pred_df.get("pred_minutes", np.nan)
            pred_df["oof_pred_vs_line"] = pred_df[pred_col] - pred_df.get(line_col, np.nan)
            pred_df["oof_pred_x_b2b"] = pred_df[pred_col] * pred_df.get("is_b2b", 0).astype(float)
            pred_df["oof_pred_x_injury_pressure"] = pred_df[pred_col] * pred_df.get("team_injury_pressure", 0).astype(float)

            correction = predict_prop(r_imp, r_model, r_feats, pred_df)

            # Clip correction to ±20% of base prediction
            base_val = pred_df[pred_col].to_numpy(dtype=float)
            max_correction = np.abs(base_val) * 0.20
            correction = np.clip(correction, -max_correction, max_correction)

            pred_df[pred_col] = base_val + correction
            pred_df = _apply_stat_prediction_bounds(pred_df, pred_col, target)

    return pred_df


def train_prediction_models(
    player_df: pd.DataFrame,
    tune: bool = False,
    n_tune_trials: int = 50,
    selected_groups: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]]]:
    """Train core prediction models once for reuse in the run.

    When tune=True, runs Optuna hyperparameter tuning first (or loads cached params).
    """
    targets = list(PROP_TARGETS)
    if "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 100:
        targets.append("fg3m")
    selected_groups = (
        selected_groups
        if selected_groups is not None
        else load_selected_feature_groups(player_df, targets)
    )
    if selected_groups:
        print(
            f"  Applying selected feature groups ({len(selected_groups)}): {', '.join(selected_groups)}",
            flush=True,
        )
    tuned_params: dict[str, dict[str, Any]] | None = None
    if tune:
        print("  Loading/running Optuna tuning for props...", flush=True)
        # Try loading cached params first
        cached = _load_tuned_params(
            player_df,
            targets,
            selected_groups=selected_groups,
        )
        if cached:
            print(f"  Loaded cached tuned params for {len(cached)} model/target combos.", flush=True)
            tuned_params = cached
        else:
            tuned_params = _run_props_tuning(
                player_df,
                targets,
                n_trials=n_tune_trials,
                selected_groups=selected_groups,
            )

    two_stage_models = train_two_stage_models(
        player_df,
        tuned_params=tuned_params,
        selected_groups=selected_groups,
    )
    single_models: dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]] = {}

    targets_to_model = list(targets)

    tp = tuned_params or {}
    for target in targets_to_model:
        if target in two_stage_models:
            continue
        features = get_effective_feature_list(
            target,
            selected_groups=selected_groups,
        )
        feats = filter_features(features, player_df)
        if not feats:
            continue
        try:
            imp, model, used_feats = train_prop_model(
                player_df, features, target, params=tp.get(f"xgb_{target}"),
            )
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
    """Compute residual standard deviations for each prop target.

    When pre-trained models are supplied and ``leakage_safe`` is False, this
    reuses those models instead of retraining, which keeps a single-model
    version across downstream consumers in the same run.
    """
    player_df = player_df.sort_values("game_time_utc").reset_index(drop=True)
    cut = int(len(player_df) * (1.0 - test_frac))
    train = player_df.iloc[:cut].copy()
    test = player_df.iloc[cut:].copy()

    # Use train-split-fitted models by default to avoid in-sample leakage in
    # uncertainty estimates. Reuse pre-trained models when explicitly allowed.
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


def _sanitize_probability_pair(
    p_over: float,
    p_under: float,
    floor: float = PROB_CLIP_FLOOR,
    ceiling: float = PROB_CLIP_CEILING,
) -> tuple[float, float]:
    """Clip calibrated probabilities away from 0/1 and renormalize."""
    if not (pd.notna(p_over) and pd.notna(p_under)):
        return p_over, p_under
    po = float(np.clip(p_over, floor, ceiling))
    pu = float(np.clip(p_under, floor, ceiling))
    s = po + pu
    if not np.isfinite(s) or s <= 0:
        return np.nan, np.nan
    po = float(np.clip(po / s, floor, ceiling))
    pu = float(np.clip(1.0 - po, floor, ceiling))
    return po, pu


def _compress_injury_pressure(value: Any) -> float:
    """Compress large injury-pressure values before building interaction features."""
    val = _nan_or(value, 0.0)
    if not np.isfinite(val) or val <= 0:
        return 0.0
    return float(np.log1p(val))


_STAT_MINUTE_LIFT_CAP = {
    "points": 1.20,
    "rebounds": 1.12,
    "assists": 1.15,
    "fg3m": 1.18,
}

_STAT_ROLE_LIFT_SCALE = {
    "points": 0.10,
    "rebounds": 0.08,
    "assists": 0.12,
    "fg3m": 0.10,
}


def _stat_prediction_upper_bound(
    row: pd.Series,
    stat_type: str,
    pred_val: float,
) -> float:
    """Compute a player-specific upper bound for forecast sanity.

    The cap is anchored to the player's recent distribution and only relaxes when
    predicted minutes or role-expansion signals support it.
    """
    if not np.isfinite(pred_val):
        return pred_val

    avg5 = _nan_or(row.get(f"pre_{stat_type}_avg5"), np.nan)
    avg10 = _nan_or(row.get(f"pre_{stat_type}_avg10"), np.nan)
    p75 = _nan_or(row.get(f"pre_{stat_type}_p75_20"), np.nan)
    p90 = _nan_or(row.get(f"pre_{stat_type}_p90_20"), np.nan)
    max20 = _nan_or(row.get(f"pre_{stat_type}_max20"), np.nan)
    same_p75 = _nan_or(row.get(f"pre_{stat_type}_same_role_p75_20"), np.nan)
    same_p90 = _nan_or(row.get(f"pre_{stat_type}_same_role_p90_20"), np.nan)
    same_max20 = _nan_or(row.get(f"pre_{stat_type}_same_role_max20"), np.nan)
    std10 = _nan_or(row.get(f"pre_{stat_type}_std10"), np.nan)

    anchors = [p75, p90, max20, same_p75, same_p90, same_max20]
    if np.isfinite(avg10):
        spread = max(_nan_or(std10, 0.0), 0.0)
        anchors.append(avg10 + 1.25 * spread)
    finite_anchors = [float(a) for a in anchors if np.isfinite(a) and a > 0]
    if not finite_anchors:
        return pred_val

    base_anchor = max(finite_anchors)
    pre_minutes = max(_nan_or(row.get("pre_minutes_avg5"), row.get("pre_minutes_avg10", 1.0)), 1.0)
    pred_minutes = max(_nan_or(row.get("pred_minutes"), pre_minutes), 0.0)
    minute_lift = np.clip(pred_minutes / pre_minutes, 1.0, _STAT_MINUTE_LIFT_CAP.get(stat_type, 1.15))

    role_lift = max(
        _nan_or(row.get("pre_role_minutes_expansion5"), 0.0),
        _nan_or(row.get("pre_role_minutes_expansion3"), 0.0),
    )
    if stat_type == "assists":
        role_lift = max(role_lift, _nan_or(row.get("pre_role_passes_expansion5"), 0.0))
    role_mult = 1.0 + _STAT_ROLE_LIFT_SCALE.get(stat_type, 0.1) * np.clip(role_lift, 0.0, 1.0)

    upper = base_anchor * minute_lift * role_mult
    if np.isfinite(avg5) and avg5 > 0:
        upper = max(upper, avg5)
    return float(upper)


def _apply_stat_prediction_bounds(
    df: pd.DataFrame,
    pred_col: str,
    stat_type: str,
) -> pd.DataFrame:
    """Cap implausible upward predictions using player-specific recent distribution."""
    if pred_col not in df.columns or df.empty:
        return df

    capped = []
    for _, row in df.iterrows():
        pred_val = _nan_or(row.get(pred_col), np.nan)
        if not np.isfinite(pred_val):
            capped.append(pred_val)
            continue
        upper = _stat_prediction_upper_bound(row, stat_type, pred_val)
        if np.isfinite(upper):
            capped.append(float(min(pred_val, upper)))
        else:
            capped.append(pred_val)
    df[pred_col] = capped
    return df


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
        df = _normalize_and_dedupe_prop_lines(df, default_date=date_str)
        if df.empty:
            continue
        df["player_name_norm"] = df["player_name"].map(normalize_player_name)
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
    those are reused instead of retraining in this function.
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
    market_df = _build_market_training_frame(
        player_df,
        max_dates=max_dates,
        pretrained_models=pretrained_models,
    )
    residual_models: dict[str, dict[str, Any]] = {}
    calibrators: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {"rows": int(len(market_df)), "per_stat": {}}
    if market_df.empty:
        return residual_models, calibrators, diagnostics

    for stat, st_df in market_df.groupby("stat_type"):
        st_df = st_df.sort_values("game_date_est").reset_index(drop=True)
        # Residual model: actual - model prediction.
        # The learned residual is added back onto pred_value at inference time, so
        # the target must be prediction error, not actual-vs-line.
        st_df["target_resid"] = st_df["actual_value"] - st_df["pred_value"]
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
        ]
        feat_cols = [c for c in feat_cols if c in st_df.columns]
        use = st_df.dropna(subset=["target_resid"]).copy()
        holdout_mae = np.nan
        baseline_holdout_mae = np.nan
        if len(use) >= 100 and feat_cols:
            # Time-ordered inner holdout to reduce residual-layer overfit.
            split_idx = int(len(use) * 0.8)
            train_slice = use.iloc[:split_idx].copy()
            holdout_slice = use.iloc[split_idx:].copy()
            if len(train_slice) >= 80 and len(holdout_slice) >= 20:
                # Drop split-specific all-NaN features before imputation to avoid
                # unstable warnings and train/predict shape mismatch.
                fit_feat_cols = [c for c in feat_cols if train_slice[c].notna().any()]
                if fit_feat_cols:
                    imp = SimpleImputer(strategy="median")
                    X_train = imp.fit_transform(train_slice[fit_feat_cols])
                    y_train = train_slice["target_resid"].to_numpy(dtype=float)
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
                    model.fit(X_train, y_train)

                    X_holdout = imp.transform(holdout_slice[fit_feat_cols])
                    y_holdout = holdout_slice["target_resid"].to_numpy(dtype=float)
                    pred_holdout = model.predict(X_holdout)
                    holdout_mae = float(mean_absolute_error(y_holdout, pred_holdout))
                    baseline_holdout_mae = float(np.mean(np.abs(y_holdout)))  # zero-adjust baseline

                    if np.isfinite(holdout_mae) and np.isfinite(baseline_holdout_mae) and holdout_mae < baseline_holdout_mae:
                        residual_models[stat] = {
                            "imputer": imp,
                            "model": model,
                            "features": fit_feat_cols,
                            "n_train": int(len(train_slice)),
                            "n_holdout": int(len(holdout_slice)),
                            "holdout_mae": round(holdout_mae, 4),
                            "holdout_baseline_mae": round(baseline_holdout_mae, 4),
                        }

        # Probability calibrators on historical market lines (by stat + side).
        resid_std = float(np.std(st_df["actual_value"].to_numpy(dtype=float) - st_df["pred_value"].to_numpy(dtype=float)))
        resid_std = max(resid_std, 0.01)
        z = (st_df["line"].to_numpy(dtype=float) - st_df["pred_value"].to_numpy(dtype=float)) / resid_std
        # Use t(df=7) to match inference-time distribution (heavier tails for counting stats)
        p_over_raw = 1.0 - sp_stats.t.cdf(z, df=7)
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
            "residual_holdout_mae": round(float(holdout_mae), 4) if pd.notna(holdout_mae) else None,
            "residual_holdout_baseline_mae": round(float(baseline_holdout_mae), 4) if pd.notna(baseline_holdout_mae) else None,
            "calibrator_rows": int(non_push.sum()),
            "calib_over": bool(calib_over is not None),
            "calib_under": bool(calib_under is not None),
        }
    return residual_models, calibrators, diagnostics


def train_synthetic_probability_calibrators(
    player_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Fallback calibration from synthetic lines when market history is sparse.

    Phase 5: Fits separate OVER and UNDER calibrators per stat, stored as
    ``{stat: {"over": calib_over, "under": calib_under}}``.
    """
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
        # Use t(df=7) to match inference-time distribution (heavier tails for counting stats)
        p_over_raw = 1.0 - sp_stats.t.cdf((line - pred) / resid_std, df=7)
        p_under_raw = 1.0 - p_over_raw
        non_push = np.abs(actual - line) > 1e-9
        labels_over = (actual > line).astype(int)
        labels_under = (actual < line).astype(int)
        calib_over = _fit_probability_calibrator(p_over_raw[non_push], labels_over[non_push])
        calib_under = _fit_probability_calibrator(p_under_raw[non_push], labels_under[non_push])
        stat_calibs: dict[str, Any] = {}
        if calib_over is not None:
            stat_calibs["over"] = calib_over
        if calib_under is not None:
            stat_calibs["under"] = calib_under
        if stat_calibs:
            calibrators[stat] = stat_calibs
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
    stat_cap = {
        "points": 0.18,
        "rebounds": 0.10,
        "assists": 0.10,
        "fg3m": 0.12,
    }.get(stat_type, 0.12)
    max_adj = abs(pred_val) * stat_cap
    resid = float(np.clip(resid, -max_adj, max_adj))
    upper = _stat_prediction_upper_bound(pred_row, stat_type, pred_val)
    if np.isfinite(upper):
        resid = min(resid, max(upper - pred_val, 0.0))
    return resid


# ---------------------------------------------------------------------------
# Rolling Bias Correction (Points)
# ---------------------------------------------------------------------------

def compute_points_bias(history: pd.DataFrame) -> dict[str, Any]:
    """Compute rolling bias for points predictions from graded history.

    Uses Bayesian shrinkage: adj = (n * raw_bias) / (n + k), capped at ±POINTS_BIAS_CAP.
    Returns a dict with bias stats to be stored as JSON sidecar.
    Auto-disables (sets adj to 0) if sign of raw bias flips vs prior saved bias.
    """
    graded = history[
        (history["actual_value"].notna())
        & (history["stat_type"] == "points")
        & (history["pred_value"].notna())
    ].copy()

    # Rolling window to avoid anchoring on stale regime data.
    if not graded.empty and "game_date_est" in graded.columns:
        dates = graded["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
        cutoff = (datetime.now() - timedelta(days=POINTS_BIAS_LOOKBACK_DAYS)).strftime("%Y%m%d")
        graded = graded[dates >= cutoff].copy()

    result: dict[str, Any] = {
        "n": 0,
        "raw_bias": 0.0,
        "shrunk_bias": 0.0,
        "active": False,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lookback_days": POINTS_BIAS_LOOKBACK_DAYS,
        "reason": "",
    }

    if len(graded) < POINTS_BIAS_MIN_SAMPLE:
        result["reason"] = f"insufficient_sample ({len(graded)} < {POINTS_BIAS_MIN_SAMPLE})"
        return result

    residuals = graded["actual_value"].astype(float) - graded["pred_value"].astype(float)
    raw_bias = float(residuals.mean())
    n = len(graded)

    # Bayesian shrinkage
    shrunk = (n * raw_bias) / (n + POINTS_BIAS_SHRINK_K)
    shrunk = max(-POINTS_BIAS_CAP, min(POINTS_BIAS_CAP, shrunk))

    # Check for sign flip vs prior bias (auto-disable if sign changed)
    prior = load_points_bias()
    if prior.get("active") and prior.get("shrunk_bias", 0) != 0:
        prior_sign = np.sign(prior["shrunk_bias"])
        new_sign = np.sign(shrunk)
        if prior_sign != new_sign and new_sign != 0:
            result["n"] = n
            result["raw_bias"] = round(raw_bias, 4)
            result["shrunk_bias"] = 0.0
            result["active"] = False
            result["reason"] = f"sign_flip (prior={prior['shrunk_bias']:+.3f}, new={shrunk:+.3f})"
            return result

    result["n"] = n
    result["raw_bias"] = round(raw_bias, 4)
    result["shrunk_bias"] = round(shrunk, 4)
    result["active"] = True
    result["reason"] = "ok"
    return result


def save_points_bias(bias_info: dict[str, Any]) -> None:
    """Save bias correction to JSON sidecar."""
    if POINTS_BIAS_FILE is None:
        return
    POINTS_BIAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POINTS_BIAS_FILE, "w") as f:
        json.dump(bias_info, f, indent=2)


def load_points_bias() -> dict[str, Any]:
    """Load bias correction from JSON sidecar. Returns empty dict if not found."""
    if POINTS_BIAS_FILE is None or not POINTS_BIAS_FILE.exists():
        return {}
    try:
        with open(POINTS_BIAS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def get_active_points_bias() -> float:
    """Return the active bias correction amount for points, or 0.0 if inactive."""
    if not USE_POINTS_BIAS_CORRECTION:
        return 0.0
    bias = load_points_bias()
    if bias.get("active"):
        return float(bias.get("shrunk_bias", 0.0))
    return 0.0


def compute_prop_edges(
    predictions: pd.DataFrame,
    prop_lines: pd.DataFrame,
    residual_stds: dict[str, float],
    market_residual_models: dict[str, dict[str, Any]] | None = None,
    prob_calibrators: dict[str, dict[str, Any]] | None = None,
    calibration_degraded_stats: set[str] | None = None,
    uncertainty_models: dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]] | None = None,
    quantile_uncertainty_models: dict[str, dict[str, tuple[SimpleImputer, XGBRegressor, list[str]]]] | None = None,
) -> pd.DataFrame:
    """Compute edges between model predictions and prop lines."""
    if predictions.empty or prop_lines.empty:
        return pd.DataFrame()

    prop_lines = _normalize_and_dedupe_prop_lines(prop_lines)
    if prop_lines.empty:
        return pd.DataFrame()

    pred_work = predictions.copy()
    pred_work["_name_norm"] = pred_work["player_name"].map(normalize_player_name)
    if "game_date_est" in pred_work.columns:
        pred_work["_date_key"] = (
            pred_work["game_date_est"]
            .astype(str)
            .str.replace("-", "", regex=False)
            .str.slice(0, 8)
        )
    else:
        pred_work["_date_key"] = ""

    results = []

    for _, line_row in prop_lines.iterrows():
        player_name = line_row["player_name"]
        stat_type = line_row["stat_type"]
        line_date = (
            str(line_row.get("game_date_est", ""))
            .replace("-", "")
            .strip()[:8]
        )
        line_val = float(line_row["line"])
        team = line_row.get("team", "")
        over_implied = line_row.get("over_implied_prob", np.nan)
        under_implied = line_row.get("under_implied_prob", np.nan)
        over_odds_raw = line_row.get("over_odds", np.nan)
        under_odds_raw = line_row.get("under_odds", np.nan)
        closing_over_odds = line_row.get("closing_odds_over", line_row.get("close_over_odds", line_row.get("over_odds_close", np.nan)))
        closing_under_odds = line_row.get("closing_odds_under", line_row.get("close_under_odds", line_row.get("under_odds_close", np.nan)))

        pred_col = f"pred_{stat_type}"
        if pred_col not in predictions.columns:
            continue

        # Match by normalized name (+ team if available). Avoid fuzzy contains to
        # prevent ambiguous player attribution.
        line_norm = normalize_player_name(player_name)
        mask = pred_work["_name_norm"].eq(line_norm)
        if line_date:
            mask = mask & pred_work["_date_key"].eq(line_date)
        if team and pd.notna(team):
            mask = mask & pred_work["team"].astype(str).eq(str(team))

        matched = pred_work[mask]
        if matched.empty:
            continue
        if len(matched) > 1:
            lt_home = str(line_row.get("home_team", "") or "").strip()
            lt_away = str(line_row.get("away_team", "") or "").strip()
            if lt_home and lt_away and {"home_team", "away_team"}.issubset(matched.columns):
                by_matchup = matched[
                    matched["home_team"].astype(str).eq(lt_home)
                    & matched["away_team"].astype(str).eq(lt_away)
                ]
                if len(by_matchup) == 1:
                    matched = by_matchup
            if len(matched) > 1:
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

        # Step 6: Market line blending (gated, off by default)
        # Applied here where both model prediction and market line are available.
        if USE_MARKET_LINE_BLENDING and line_val > 0:
            alpha = MARKET_LINE_BLEND_ALPHA
            pred_val = alpha * pred_val + (1 - alpha) * line_val

        # Rolling bias correction for points: shift prediction by shrunk bias
        bias_adj = 0.0
        if stat_type == "points":
            bias_adj = get_active_points_bias()
            if bias_adj != 0.0:
                pred_val = pred_val + bias_adj

        pred_val = min(pred_val, _stat_prediction_upper_bound(pred_row, stat_type, pred_val))

        edge = pred_val - line_val
        edge_pct = (edge / line_val * 100) if line_val != 0 else np.nan

        # Compute p_over and p_under using blended (player-specific + global) std
        resid_std = residual_stds.get(stat_type, np.nan)

        # Quantile uncertainty (Step 2): use IQR-based std from quantile regression
        quantile_std = np.nan
        if quantile_uncertainty_models and stat_type in quantile_uncertainty_models:
            unc_row = pred_row.copy()
            unc_row[f"oof_pred_{stat_type}"] = pred_base
            unc_row["oof_pred_minutes"] = pred_row.get("pred_minutes", np.nan)
            try:
                quantile_std, _, _ = predict_quantile_std(
                    quantile_uncertainty_models, stat_type, unc_row,
                )
            except Exception:
                pass

        # Phase 4: Use abs-error uncertainty model if available (fallback)
        unc_std = np.nan
        if not (pd.notna(quantile_std) and quantile_std > 0):
            if uncertainty_models and stat_type in uncertainty_models:
                u_imp, u_model, u_feats = uncertainty_models[stat_type]
                unc_row = pred_row.copy()
                unc_row[f"oof_pred_{stat_type}"] = pred_base
                unc_row["oof_pred_minutes"] = pred_row.get("pred_minutes", np.nan)
                try:
                    unc_pred = predict_prop(u_imp, u_model, u_feats, pd.DataFrame([unc_row]))
                    predicted_mae = float(unc_pred[0])
                    if np.isfinite(predicted_mae) and predicted_mae > 0:
                        # Convert predicted MAE to std: std ≈ MAE * sqrt(pi/2)
                        unc_std = predicted_mae * 1.253
                except Exception:
                    pass

        player_std = pred_row.get(f"pre_{stat_type}_std10", np.nan)
        if pd.notna(quantile_std) and quantile_std > 0:
            # Priority 0: Quantile regression IQR-derived std
            blended_std = quantile_std
        elif pd.notna(unc_std) and unc_std > 0:
            # Priority 1: Abs-error uncertainty model
            blended_std = unc_std
        elif pd.notna(player_std) and player_std > 0 and pd.notna(resid_std):
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
            p_over, p_under = _sanitize_probability_pair(p_over, p_under)
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

            # --- OVER signal gates ---
            if signal == "OVER":
                # Gate: suppress LEAN OVER entirely
                if SUPPRESS_LEAN_OVER and confidence == "LEAN":
                    signal = "NO BET"
                    confidence = ""
                    signal_blocked_reason = "lean_over_suppressed"
                # Gate: require confirmed lineup for OVER
                elif OVER_REQUIRE_LINEUP_CONFIRMED:
                    lineup_conf = pd.to_numeric(pred_row.get("confirmed_starter"), errors="coerce")
                    if not (pd.notna(lineup_conf) and lineup_conf >= 1.0):
                        signal = "NO BET"
                        confidence = ""
                        signal_blocked_reason = "over_lineup_not_confirmed"
                # Gate: max injury probability for OVER
                if signal == "OVER" and OVER_MAX_INJURY_PROB < 1.0:
                    inj_prob = pd.to_numeric(pred_row.get("injury_unavailability_prob"), errors="coerce")
                    if pd.notna(inj_prob) and inj_prob > OVER_MAX_INJURY_PROB:
                        signal = "NO BET"
                        confidence = ""
                        signal_blocked_reason = "over_injury_gate"

            # Injury coverage stale gate (team-scoped/global): suppress OVER and downgrade confidence.
            stale_global = int(_nan_or(pred_row.get("injury_feed_global_stale"), 0)) == 1
            stale_team = int(_nan_or(pred_row.get("injury_feed_team_stale"), 0)) == 1
            zero_out_cov = int(_nan_or(pred_row.get("injury_feed_team_zero_out_doubtful"), 0)) == 1
            if stale_global or stale_team or zero_out_cov:
                if signal == "OVER":
                    signal = "NO BET"
                    confidence = ""
                    stale_tag = "GLOBAL" if stale_global else str(pred_row.get("team", "TEAM"))
                    signal_blocked_reason = f"injury_coverage_stale:{stale_tag}"
                elif signal != "NO BET" and confidence == "BEST BET":
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
            "player_id": pred_row.get("player_id", np.nan),
            "team": team if (team and pd.notna(team)) else pred_row.get("team", ""),
            "opp": pred_row.get("opp", ""),
            "home_team": pred_row.get("home_team", ""),
            "away_team": pred_row.get("away_team", ""),
            "run_id": pred_row.get("run_id", ""),
            "asof_utc": pred_row.get("asof_utc", ""),
            "stat_type": stat_type,
            "prop_line": line_val,
            "pred_value": round(pred_val, 1),
            "pred_value_base": round(pred_base, 1),
            "market_resid_adj": round(resid_adj, 2),
            "bias_adj": round(bias_adj, 3),
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
            "closing_odds_over": closing_over_odds,
            "closing_odds_under": closing_under_odds,
            "line_move": round(line_move, 1) if pd.notna(line_move) else np.nan,
            "line_move_pct": round(line_move_pct, 1) if pd.notna(line_move_pct) else np.nan,
            "line_confirms_model": line_confirms_model if not isinstance(line_confirms_model, float) else np.nan,
            "residual_std": round(resid_std, 2) if pd.notna(resid_std) else np.nan,
            "blended_std": round(blended_std, 2) if pd.notna(blended_std) else np.nan,
            "player_std": round(float(player_std), 2) if pd.notna(player_std) else np.nan,
            "pre_avg5": pred_row.get(f"pre_{stat_type}_avg5", np.nan),
            "pre_avg10": pred_row.get(f"pre_{stat_type}_avg10", np.nan),
            "pre_season": pred_row.get(f"pre_{stat_type}_season", np.nan),
            "player_vs_opp_pts_delta": pred_row.get("player_vs_opp_pts_delta", np.nan),
            "player_vs_opp_reb_delta": pred_row.get("player_vs_opp_reb_delta", np.nan),
            "player_vs_opp_ast_delta": pred_row.get("player_vs_opp_ast_delta", np.nan),
            "lineup_confirmed": pred_row.get("lineup_confirmed", np.nan),
            "confirmed_starter": pred_row.get("confirmed_starter", np.nan),
            "injury_status": pred_row.get("injury_status", ""),
            "injury_unavailability_prob": pred_row.get("injury_unavailability_prob", np.nan),
            "team_injury_pressure_fwd": pred_row.get("team_injury_pressure_fwd", np.nan),
            "team_injury_pressure_bwd": pred_row.get("team_injury_pressure_bwd", np.nan),
            "injury_pressure_delta": pred_row.get("injury_pressure_delta", np.nan),
            "injury_feed_global_stale": pred_row.get("injury_feed_global_stale", np.nan),
            "injury_feed_team_stale": pred_row.get("injury_feed_team_stale", np.nan),
            "injury_feed_team_zero_out_doubtful": pred_row.get("injury_feed_team_zero_out_doubtful", np.nan),
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
    targets = [t for t in (list(PROP_TARGETS) + ["fg3m"]) if t in player_df.columns]
    selected_groups = load_selected_feature_groups(player_df, targets)
    cut = int(len(player_df) * (1.0 - test_frac))
    train = player_df.iloc[:cut].copy()
    test = player_df.iloc[cut:].copy()
    print(f"\n  Market-line backtest split: {len(train)} train, {len(test)} test", flush=True)

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
        features = get_effective_feature_list(target, selected_groups=selected_groups)
        feats = filter_features(features, train_valid)
        if not feats:
            continue
        try:
            imp, model, used_feats = train_prop_model(train_valid, features, target)
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
        date_slice = actual[actual["game_date_est"] == date]
        matches = _match_actual_player_rows(
            date_slice,
            player_name=str(r.get("player_name", "")),
            team=team,
            player_id=r.get("player_id", np.nan),
        )
        if matches.empty or len(matches) != 1:
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
    all_targets = list(PROP_TARGETS) + (
        ["fg3m"] if "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 100 else []
    )
    selected_groups = load_selected_feature_groups(player_df, all_targets)
    cut = int(len(player_df) * (1.0 - test_frac))
    train = player_df.iloc[:cut].copy()
    test = player_df.iloc[cut:].copy()

    print(f"\n  Prop edge backtest: {len(train)} train, {len(test)} test", flush=True)

    results: dict[str, dict[str, Any]] = {}
    for target in all_targets:
        features = get_effective_feature_list(target, selected_groups=selected_groups)
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
        p_overs = 1.0 - sp_stats.t.cdf(z_scores, df=7)

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
    backtest_targets = list(PROP_TARGETS) + (
        ["fg3m"] if "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 100 else []
    )
    selected_groups = load_selected_feature_groups(player_df, backtest_targets)
    cut = int(len(player_df) * (1.0 - test_frac))
    train = player_df.iloc[:cut].copy()
    test = player_df.iloc[cut:].copy()

    print(f"\n  Backtest split: {len(train)} train, {len(test)} test", flush=True)

    results: dict[str, dict[str, float]] = {}

    # --- Standard (single-stage) models ---
    print("\n  --- Single-Stage Models ---", flush=True)
    for target in PROP_TARGETS:
        features = get_effective_feature_list(target, selected_groups=selected_groups)
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
        features = get_effective_feature_list("fg3m", selected_groups=selected_groups)
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
    two_stage_models = train_two_stage_models(train, selected_groups=selected_groups)
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

    def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
        if n <= 0:
            return np.nan, np.nan
        p = wins / n
        denom = 1.0 + (z * z) / n
        center = (p + (z * z) / (2.0 * n)) / denom
        spread = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) / denom
        return max(0.0, center - spread), min(1.0, center + spread)

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
            p_overs = 1.0 - sp_stats.t.cdf(z_scores, df=7)

            over_mask = p_overs > (BREAKEVEN_PROB + 0.03)
            under_mask = p_overs < (1.0 - BREAKEVEN_PROB - 0.03)

            n_over = int(over_mask.sum())
            n_under = int(under_mask.sum())
            over_hit = int((actual_v[over_mask] > lines_v[over_mask]).sum()) if n_over > 0 else 0
            under_hit = int((actual_v[under_mask] < lines_v[under_mask]).sum()) if n_under > 0 else 0

            n_bets = n_over + n_under
            total_wins = over_hit + under_hit
            wr_low, wr_high = _wilson_ci(total_wins, n_bets)
            total_profit = (
                total_wins * bet_size * VIG_FACTOR
                - (n_bets - total_wins) * bet_size
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
                "total_wins": total_wins,
                "total_win_rate": round(total_wins / n_bets, 3) if n_bets > 0 else np.nan,
                "total_win_rate_ci_low": round(wr_low, 3) if pd.notna(wr_low) else np.nan,
                "total_win_rate_ci_high": round(wr_high, 3) if pd.notna(wr_high) else np.nan,
                "profit": round(total_profit, 2),
                "roi_pct": round(100 * total_profit / (n_bets * bet_size), 2) if n_bets > 0 else np.nan,
            }

            wr = fold_metrics[target].get("total_win_rate", np.nan)
            wr_l = fold_metrics[target].get("total_win_rate_ci_low", np.nan)
            wr_h = fold_metrics[target].get("total_win_rate_ci_high", np.nan)
            wr_s = f"{wr:.1%}" if pd.notna(wr) else "N/A"
            ci_s = f"[{wr_l:.1%},{wr_h:.1%}]" if pd.notna(wr_l) and pd.notna(wr_h) else ""
            print(
                f"    {target:>10s}:  MAE={mae:.2f}  R2={r2:.3f}  "
                f"Bets={n_bets}  WR={wr_s} {ci_s}  "
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

    # --- Step 5: Gate Checks ---
    gate_results = _run_walk_forward_gate_checks(agg, fold_results, bet_size)

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
        "gate_checks": gate_results,
    }
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    wf_out.write_text(json.dumps(wf_data, indent=2, default=str))
    print(f"\n  Walk-forward results saved to {wf_out}", flush=True)

    # Log experiment
    exp_metrics: dict[str, Any] = {
        "total_profit": total_profit_all,
        "total_bets": total_bets_all,
    }
    for target, v in agg.items():
        exp_metrics[f"mae_{target}"] = float(np.nanmean(v["mae"]))
        exp_metrics[f"r2_{target}"] = float(np.nanmean(v["r2"]))
    append_experiment_result(
        experiment_type="walk_forward",
        description=f"Walk-forward: {len(fold_results)} folds",
        metrics=exp_metrics,
    )

    return wf_data


def _run_walk_forward_gate_checks(
    agg: dict[str, dict[str, list[float]]],
    fold_results: list[dict[str, Any]],
    bet_size: float,
) -> dict[str, Any]:
    """Run gate checks on walk-forward results.

    Gate checks (all must pass before proceeding to market blending):
    1. MAE per stat: test fold MAE below stat-specific threshold
    2. Win rate: avg win rate on synthetic lines >= 48% per stat
    3. Overall ROI: >= -5% across all folds (within vig tolerance)
    4. R² positive: model adds value over naive baseline for each stat
    """
    print(f"\n  {'=' * 72}", flush=True)
    print("  WALK-FORWARD GATE CHECKS", flush=True)
    print(f"  {'=' * 72}", flush=True)

    gates: dict[str, Any] = {}
    all_pass = True

    # Gate 1: MAE thresholds per stat (reasonable baselines)
    mae_thresholds = {
        "points": 7.0,
        "rebounds": 2.8,
        "assists": 2.2,
        "fg3m": 1.2,
        "minutes": 8.0,
    }
    for target, vals in agg.items():
        avg_mae = float(np.nanmean(vals["mae"]))
        threshold = mae_thresholds.get(target, 10.0)
        passed = avg_mae <= threshold
        gates[f"mae_{target}"] = {"value": round(avg_mae, 3), "threshold": threshold, "pass": passed}
        status = "PASS" if passed else "FAIL"
        print(f"    MAE gate {target}: {avg_mae:.3f} vs {threshold} -> {status}", flush=True)
        if not passed:
            all_pass = False

    # Gate 2: Win rate on test folds (proxy for calibration)
    for target, vals in agg.items():
        win_rates = vals.get("win_rate", [])
        if win_rates:
            avg_wr = float(np.nanmean(win_rates))
            # Minimum 48% win rate on synthetic lines (above random minus vig)
            passed = avg_wr >= 0.48
            gates[f"win_rate_{target}"] = {"value": round(avg_wr, 3), "threshold": 0.48, "pass": passed}
            status = "PASS" if passed else "FAIL"
            print(f"    WR gate {target}: {avg_wr:.1%} vs 48% -> {status}", flush=True)
            if not passed:
                all_pass = False

    # Gate 3: Overall ROI non-negative
    total_profit = sum(sum(v["profit"]) for v in agg.values())
    total_bets = sum(int(sum(v["n_bets"])) for v in agg.values())
    if total_bets > 0:
        overall_roi = 100 * total_profit / (total_bets * bet_size)
        passed = overall_roi >= -5.0  # Allow slight negative (within vig)
        gates["overall_roi"] = {"value": round(overall_roi, 2), "threshold": -5.0, "pass": passed}
        status = "PASS" if passed else "FAIL"
        print(f"    ROI gate: {overall_roi:.1f}% vs -5.0% -> {status}", flush=True)
        if not passed:
            all_pass = False

    # Gate 4: R2 positive for test folds (model adds value over baseline)
    for target, vals in agg.items():
        avg_r2 = float(np.nanmean(vals["r2"]))
        passed = avg_r2 > 0.0
        gates[f"r2_{target}"] = {"value": round(avg_r2, 4), "threshold": 0.0, "pass": passed}
        status = "PASS" if passed else "FAIL"
        print(f"    R2 gate {target}: {avg_r2:.4f} vs 0.0 -> {status}", flush=True)
        if not passed:
            all_pass = False

    # Overall assessment
    gates["all_pass"] = all_pass
    print(f"\n  {'=' * 40}", flush=True)
    if all_pass:
        print("  GATE CHECK: ALL GATES PASSED -- Safe to proceed to market blending", flush=True)
    else:
        failed = [k for k, v in gates.items() if isinstance(v, dict) and not v.get("pass", True)]
        print(f"  GATE CHECK: FAILED -- {len(failed)} gate(s) failed: {', '.join(failed)}", flush=True)
    print(f"  {'=' * 40}", flush=True)

    return gates


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


def apply_star_out_floor_boost(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Apply heuristic floor boost for extreme star-absence scenarios.

    When a team's forward injury pressure exceeds the training p95, the XGBoost
    model under-extrapolates because it rarely saw such extreme values in training.
    This function boosts the top remaining scorers' predictions toward the uplift
    observed in training data (role_beta * pressure), capped conservatively.

    Only applied at prediction time (not training), so it won't contaminate the
    model's learned relationships.
    """
    if pred_df.empty:
        return pred_df

    pressure_col = "team_injury_pressure_fwd"
    if pressure_col not in pred_df.columns:
        return pred_df

    pressure = pd.to_numeric(pred_df[pressure_col], errors="coerce").fillna(0)
    eligible_mask = pressure > STAR_OUT_PRESSURE_THRESHOLD
    if not eligible_mask.any():
        return pred_df

    pred_df = pred_df.copy()
    boost_log: list[str] = []

    # Identify top N remaining scorers per team (by pre_points_avg5)
    avg5_col = "pre_points_avg5"
    if avg5_col not in pred_df.columns:
        return pred_df

    eligible_teams = pred_df.loc[eligible_mask, "team"].unique()
    top_pids_per_team: dict[str, set[int]] = {}

    # Exclude players who are OUT/suspended — they won't play and get filtered later
    is_out_col = "injury_is_out"
    for team in eligible_teams:
        team_mask = (pred_df["team"] == team) & eligible_mask
        team_rows = pred_df[team_mask].drop_duplicates(subset=["player_id"])
        if is_out_col in team_rows.columns:
            team_rows = team_rows[team_rows[is_out_col] != 1]
        if team_rows.empty:
            continue
        top_players = team_rows.nlargest(STAR_OUT_TOP_N_PLAYERS, avg5_col)
        top_pids_per_team[team] = set(top_players["player_id"].astype(int).tolist())

    # Map stat targets to their role beta columns
    stat_beta_map = {
        "points": "role_pts_injury_beta20",
        "rebounds": "role_reb_injury_beta20",
        "assists": "role_ast_injury_beta20",
    }

    for stat in STAR_OUT_STAT_TARGETS:
        pred_col = f"pred_{stat}"
        avg_col = f"pre_{stat}_avg5"
        beta_col = stat_beta_map.get(stat, "")
        if pred_col not in pred_df.columns or avg_col not in pred_df.columns:
            continue

        for team in eligible_teams:
            top_pids = top_pids_per_team.get(team, set())
            if not top_pids:
                continue

            team_pressure = float(pred_df.loc[pred_df["team"] == team, pressure_col].iloc[0])
            mask = (
                (pred_df["team"] == team)
                & pred_df["player_id"].astype(int).isin(top_pids)
                & eligible_mask
            )
            if not mask.any():
                continue

            for idx in pred_df.index[mask]:
                row = pred_df.loc[idx]
                avg5 = float(_nan_or(row.get(avg_col), 0))
                model_pred = float(_nan_or(row.get(pred_col), avg5))
                beta = float(_nan_or(row.get(beta_col), 0.1) if beta_col and beta_col in pred_df.columns else 0.1)
                beta = max(beta, 0.05)  # Floor beta so boost is never zero

                # Floor = avg5 + scale * beta * pressure (capped)
                raw_boost = STAR_OUT_BOOST_SCALE * beta * team_pressure
                capped_boost = min(raw_boost, STAR_OUT_MAX_BOOST)
                floor_val = avg5 + capped_boost

                if floor_val > model_pred:
                    delta = floor_val - model_pred
                    pred_df.at[idx, pred_col] = floor_val
                    pred_df.at[idx, f"star_out_boost_{stat}"] = round(delta, 2)
                    boost_log.append(
                        f"    {row.get('player_name', '?')} ({team}) {stat}: "
                        f"{model_pred:.1f} -> {floor_val:.1f} (+{delta:.1f}, "
                        f"beta={beta:.3f}, pressure={team_pressure:.0f})"
                    )

    if boost_log:
        print(f"\n  Star-out floor boost applied ({len(boost_log)} adjustments):", flush=True)
        for line in boost_log:
            print(line, flush=True)

    return pred_df


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
    asof_utc: pd.Timestamp | None = None,
    run_id: str = "",
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
    asof_utc = asof_utc if asof_utc is not None else pd.Timestamp.now(tz="UTC")

    pg = _add_team_role_absence_context(player_games_raw)
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

    # Merge BRef advanced stats (same as training path)
    print("  Loading BRef advanced stats for prediction features...", flush=True)
    pg = _merge_bref_advanced_stats(pg)

    # Merge BRef opponent defense stats (same as training path)
    print("  Loading BRef opponent defense for prediction features...", flush=True)
    pg = _merge_bref_opponent_defense(pg)

    # Merge player tracking / scoring / rotation / matchup caches so serve-time features
    # match the training feature space for role and matchup context.
    trk_cols = [
        "trk_touches", "trk_drives", "trk_passes",
        "trk_catch_shoot_fga", "trk_catch_shoot_fg3a",
        "trk_pull_up_fga", "trk_pull_up_fg3a",
        "trk_contested_shots", "trk_uncontested_fga",
        "trk_deflections", "trk_box_outs", "trk_off_box_outs", "trk_def_box_outs",
        "trk_loose_balls", "trk_screen_assists", "trk_secondary_assists",
        "trk_reb_chances", "trk_reb_chances_off", "trk_reb_chances_def",
        "trk_pts_paint", "trk_pts_fast_break", "trk_pts_off_to",
    ]
    trk = load_player_tracking_stats(game_ids=pg["game_id"].astype(str).unique().tolist(), fetch_missing=False)
    if not trk.empty:
        pg = pg.merge(
            trk[["game_id", "team", "player_id"] + [c for c in trk_cols if c in trk.columns]].drop_duplicates(
                subset=["game_id", "team", "player_id"]
            ),
            on=["game_id", "team", "player_id"],
            how="left",
        )
    for c in trk_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    scr_cols = [
        "scr_pct_assisted_2pt", "scr_pct_assisted_3pt", "scr_pct_assisted_fgm",
        "scr_pct_unassisted_2pt", "scr_pct_unassisted_3pt",
        "scr_pct_fga_2pt", "scr_pct_fga_3pt",
        "scr_pct_pts_paint", "scr_pct_pts_midrange", "scr_pct_pts_fastbreak",
    ]
    scr = load_scoring_stats(game_ids=pg["game_id"].astype(str).unique().tolist())
    if not scr.empty:
        pg = pg.merge(
            scr[["game_id", "team", "player_id"] + [c for c in scr_cols if c in scr.columns]].drop_duplicates(
                subset=["game_id", "team", "player_id"]
            ),
            on=["game_id", "team", "player_id"],
            how="left",
        )
    for c in scr_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    rot_cols = ["rot_stints", "rot_total_stint_min", "rot_avg_stint_min", "rot_max_stint_min"]
    rot = load_game_rotation_stats(game_ids=pg["game_id"].astype(str).unique().tolist(), fetch_missing=False)
    if not rot.empty:
        pg = pg.merge(
            rot[["game_id", "player_id"] + [c for c in rot_cols if c in rot.columns]].drop_duplicates(
                subset=["game_id", "player_id"]
            ),
            on=["game_id", "player_id"],
            how="left",
        )
    for c in rot_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    mtch_cols = ["mtch_partial_poss", "mtch_fga", "mtch_fg_pct", "mtch_3pa", "mtch_3pt_pct", "mtch_ast", "mtch_pts"]
    mtch = load_boxscore_matchups_stats(game_ids=pg["game_id"].astype(str).unique().tolist(), fetch_missing=False)
    if not mtch.empty:
        pg = pg.merge(
            mtch[["game_id", "player_id"] + [c for c in mtch_cols if c in mtch.columns]].drop_duplicates(
                subset=["game_id", "player_id"]
            ),
            on=["game_id", "player_id"],
            how="left",
        )
    for c in mtch_cols:
        if c not in pg.columns:
            pg[c] = np.nan

    if "minutes" in pg.columns:
        _safe_pred_min = pg["minutes"].clip(lower=1.0)
        if "trk_passes" in pg.columns:
            pg["trk_passes_per_min"] = pg["trk_passes"].fillna(0.0) / _safe_pred_min
        if "trk_touches" in pg.columns:
            pg["trk_touches_per_min"] = pg["trk_touches"].fillna(0.0) / _safe_pred_min
        if "trk_drives" in pg.columns:
            pg["trk_drives_per_min"] = pg["trk_drives"].fillna(0.0) / _safe_pred_min
    if "trk_passes" in pg.columns:
        _team_passes = pg.groupby(["game_id", "team"])["trk_passes"].transform("sum").clip(lower=1.0)
        pg["player_pass_share"] = pg["trk_passes"].fillna(0.0) / _team_passes
        pg["player_ast_per_pass"] = pg["assists"].fillna(0.0) / pg["trk_passes"].clip(lower=1.0)
    else:
        pg["player_pass_share"] = np.nan
        pg["player_ast_per_pass"] = np.nan
    if "trk_reb_chances" in pg.columns:
        _team_reb_ch = pg.groupby(["game_id", "team"])["trk_reb_chances"].transform("sum").clip(lower=1.0)
        pg["player_reb_chance_share"] = pg["trk_reb_chances"].fillna(0.0) / _team_reb_ch
        pg["player_reb_conversion"] = pg["rebounds"].fillna(0.0) / pg["trk_reb_chances"].clip(lower=1.0)
    else:
        pg["player_reb_chance_share"] = np.nan
        pg["player_reb_conversion"] = np.nan

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

    teams_in_scope = sorted(set(upcoming["home_team"].astype(str)).union(set(upcoming["away_team"].astype(str))))
    clip_bounds = _forward_pressure_clip_bounds(team_games)
    forward_injury_pressure = compute_forward_injury_pressure(
        injury_status_map,
        pg,
        teams_in_scope,
        clip_bounds=clip_bounds,
    )
    injury_coverage = evaluate_injury_feed_coverage(
        injury_status_map,
        teams_in_scope,
        now_utc=asof_utc,
    )

    bwd_pressure_by_team: dict[str, float] = {}
    if "injury_proxy_missing_points5" in team_games.columns:
        latest_tg = team_games.sort_values("game_time_utc").groupby("team", sort=False).tail(1)
        bwd_pressure_by_team = {
            str(r["team"]).upper(): float(_nan_or(r.get("injury_proxy_missing_points5"), 0.0))
            for _, r in latest_tg.iterrows()
        }

    print("\n  Injury pressure audit (forward vs backward):", flush=True)
    for team in teams_in_scope:
        fwd = float(_nan_or(forward_injury_pressure.get(team, {}).get("fwd_missing_points"), 0.0))
        bwd = float(_nan_or(bwd_pressure_by_team.get(team, 0.0), 0.0))
        delta = fwd - bwd
        cov = injury_coverage.get("teams", {}).get(team, {})
        if int(cov.get("zero_out_doubtful", 0)) == 1:
            print(f"    [WARN] {team}: 0 Out/Doubtful/Suspension rows in injury feed", flush=True)
        print(f"    {team}: forward={fwd:.1f} backward={bwd:.1f} delta={delta:+.1f}", flush=True)
        top_out = forward_injury_pressure.get(team, {}).get("top_players_out", [])[:3]
        if top_out:
            top_s = ", ".join(
                f"{p.get('player_name','')}({p.get('status','')}, miss_pts={float(p.get('missing_points',0.0)):.1f})"
                for p in top_out
            )
            print(f"      top players out: {top_s}", flush=True)

    if int(injury_coverage.get("global_stale", 0)) == 1:
        print(f"  [WARN] Injury feed stale globally: {injury_coverage.get('global_reason', 'unknown')}", flush=True)

    snapshot_date = (
        str(upcoming["game_date_est"].iloc[0])
        if "game_date_est" in upcoming.columns and not upcoming.empty
        else datetime.now().strftime("%Y%m%d")
    )
    snapshot_path = save_injury_snapshot(
        snapshot_date,
        teams_in_scope,
        injury_status_map,
        injury_coverage,
        forward_injury_pressure,
        asof_utc=asof_utc,
    )
    print(f"  Injury snapshot saved: {snapshot_path}", flush=True)

    # Determine each player's current team (team of their most recent game by date, not game_id)
    player_current_team = pg.sort_values("game_time_utc").groupby("player_id")["team"].last()

    opp_def_profile = pd.DataFrame()
    if "position" in pg.columns:
        pos_map = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}
        pg["pos_group"] = pg["position"].map(pos_map).fillna("F")
        defense_agg: dict[str, tuple[str, str]] = {
            "pts_allowed": ("points", "sum"),
            "reb_allowed": ("rebounds", "sum"),
            "ast_allowed": ("assists", "sum"),
        }
        if "fg3m" in pg.columns:
            defense_agg["fg3m_allowed"] = ("fg3m", "sum")
        if "fg3a" in pg.columns:
            defense_agg["fg3a_allowed"] = ("fg3a", "sum")
        opp_def_profile = pg.groupby(["game_id", "game_time_utc", "opp", "pos_group"]).agg(
            **defense_agg
        ).reset_index().sort_values("game_time_utc")
        for stat in ["pts", "reb", "ast", "fg3m", "fg3a"]:
            col = f"{stat}_allowed"
            if col not in opp_def_profile.columns:
                continue
            grp = opp_def_profile.groupby(["opp", "pos_group"])[col]
            opp_def_profile[f"opp_{stat}_allowed_to_pos_avg10"] = grp.transform(
                lambda s: s.shift(1).rolling(10, min_periods=3).mean()
            )
        if {
            "opp_fg3m_allowed_to_pos_avg10",
            "opp_fg3a_allowed_to_pos_avg10",
        }.issubset(opp_def_profile.columns):
            safe_fg3a = opp_def_profile["opp_fg3a_allowed_to_pos_avg10"].clip(lower=0.1)
            opp_def_profile["opp_fg3_pct_allowed_to_pos_avg10"] = (
                opp_def_profile["opp_fg3m_allowed_to_pos_avg10"] / safe_fg3a
            )
        for stat in ["pts", "reb", "ast", "fg3m"]:
            feat = f"opp_{stat}_allowed_to_pos_avg10"
            if feat not in opp_def_profile.columns:
                continue
            league_avg = opp_def_profile.groupby("pos_group")[feat].transform("mean")
            q25 = opp_def_profile.groupby("pos_group")[feat].transform(lambda s: s.quantile(0.25))
            opp_def_profile[f"{feat}_vs_league"] = opp_def_profile[feat] - league_avg
            opp_def_profile[f"{feat}_tough_flag"] = (opp_def_profile[feat] <= q25).astype(float)

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

            team_role_context = {"team_top_creator_out": 0.0, "team_top_rebounder_out": 0.0}
            if eligible_pids:
                latest_team_rows = (
                    pg[(pg["team"] == team) & (pg["player_id"].isin(eligible_pids))]
                    .sort_values("game_time_utc")
                    .groupby("player_id", sort=False)
                    .tail(1)
                )
                if not latest_team_rows.empty:
                    creator_pool = latest_team_rows.assign(
                        _creator_score=latest_team_rows.get("assists", 0.0).fillna(0.0)
                        + 0.15 * latest_team_rows.get("points", 0.0).fillna(0.0)
                    )
                    rebound_pool = latest_team_rows.assign(
                        _rebounder_score=latest_team_rows.get("rebounds", 0.0).fillna(0.0)
                        + 0.03 * latest_team_rows.get("minutes", 0.0).fillna(0.0)
                    )

                    if not creator_pool.empty:
                        creator_name = str(creator_pool.sort_values("_creator_score", ascending=False).iloc[0].get("player_name", ""))
                        creator_inj = injury_status_map.get(_injury_key(team, creator_name), {})
                        creator_unavail = 1.0 - float(np.clip(_nan_or(creator_inj.get("availability_prob"), 1.0), 0.0, 1.0))
                        creator_status = str(creator_inj.get("status", "")).strip().lower()
                        team_role_context["team_top_creator_out"] = float(
                            creator_status in INJURY_FORWARD_STATUSES and creator_unavail >= INJURY_FORWARD_MIN_UNAVAIL
                        )

                    if not rebound_pool.empty:
                        rebound_name = str(rebound_pool.sort_values("_rebounder_score", ascending=False).iloc[0].get("player_name", ""))
                        rebound_inj = injury_status_map.get(_injury_key(team, rebound_name), {})
                        rebound_unavail = 1.0 - float(np.clip(_nan_or(rebound_inj.get("availability_prob"), 1.0), 0.0, 1.0))
                        rebound_status = str(rebound_inj.get("status", "")).strip().lower()
                        team_role_context["team_top_rebounder_out"] = float(
                            rebound_status in INJURY_FORWARD_STATUSES and rebound_unavail >= INJURY_FORWARD_MIN_UNAVAIL
                        )

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
                    "run_id": run_id,
                    "asof_utc": asof_utc.isoformat(),
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

                extra_roll_cols = [
                    "trk_touches", "trk_drives", "trk_passes",
                    "player_pass_share", "player_ast_per_pass",
                    "player_reb_chance_share", "player_reb_conversion",
                    "trk_touches_per_min", "trk_drives_per_min", "trk_passes_per_min",
                    "trk_box_outs", "trk_reb_chances", "trk_screen_assists",
                    "trk_secondary_assists",
                    "rot_stints", "rot_total_stint_min", "rot_avg_stint_min", "rot_max_stint_min",
                    "mtch_partial_poss", "mtch_fga", "mtch_fg_pct", "mtch_3pa", "mtch_3pt_pct", "mtch_ast", "mtch_pts",
                    # BRef advanced stats — only NEW (not redundant with adv_*)
                    "bref_adv_def_rtg", "bref_adv_bpm", "bref_adv_efg_pct",
                    "bref_adv_stl_pct", "bref_adv_blk_pct",
                    "bref_adv_orb_pct", "bref_adv_drb_pct",
                    "bref_adv_tov_pct",
                ]
                for col in extra_roll_cols:
                    if col not in p_games.columns:
                        continue
                    vals = p_games[col].dropna()
                    if vals.empty:
                        continue
                    r3_vals = recent3[col] if col in recent3.columns else pd.Series(dtype=float)
                    r5_vals = recent5[col] if col in recent5.columns else pd.Series(dtype=float)
                    r10_vals = recent10[col] if col in recent10.columns else pd.Series(dtype=float)
                    row[f"pre_{col}_avg3"] = float(r3_vals.mean()) if not r3_vals.dropna().empty else np.nan
                    row[f"pre_{col}_avg5"] = float(r5_vals.mean()) if not r5_vals.dropna().empty else np.nan
                    row[f"pre_{col}_avg10"] = float(r10_vals.mean()) if not r10_vals.dropna().empty else np.nan
                    row[f"pre_{col}_season"] = float(vals.mean()) if not vals.empty else np.nan
                    row[f"pre_{col}_ewm5"] = float(vals.ewm(span=5, min_periods=1).mean().iloc[-1]) if len(vals) >= 2 else row[f"pre_{col}_season"]
                    row[f"pre_{col}_ewm10"] = float(vals.ewm(span=10, min_periods=1).mean().iloc[-1]) if len(vals) >= 2 else row[f"pre_{col}_season"]

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
                if pd.notna(row.get("pre_trk_passes_avg5")):
                    row["pre_trk_passes_per_min_avg5"] = row["pre_trk_passes_avg5"] / safe_min_avg5

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

                # Distribution-aware features: quantify whether recent production is out of range.
                for col in ["points", "rebounds", "assists", "fg3m", "minutes"]:
                    if col not in p_games.columns:
                        continue
                    hist20 = p_games[col].dropna().tail(20)
                    if len(hist20) >= 5:
                        row[f"pre_{col}_p75_20"] = float(hist20.quantile(0.75))
                        row[f"pre_{col}_p90_20"] = float(hist20.quantile(0.90))
                        row[f"pre_{col}_max20"] = float(hist20.max())
                    else:
                        row[f"pre_{col}_p75_20"] = np.nan
                        row[f"pre_{col}_p90_20"] = np.nan
                        row[f"pre_{col}_max20"] = np.nan
                    avg3 = row.get(f"pre_{col}_avg3", np.nan)
                    season_val = row.get(f"pre_{col}_season", np.nan)
                    std_val = row.get(f"pre_{col}_std10", np.nan)
                    if pd.notna(avg3) and pd.notna(season_val):
                        if pd.isna(std_val) or float(std_val) < 0.75:
                            std_val = float(hist20.std()) if len(hist20) >= 5 else np.nan
                        row[f"pre_{col}_recent_zscore"] = (
                            (avg3 - season_val) / max(_nan_or(std_val, 1.0), 0.75)
                        )
                    else:
                        row[f"pre_{col}_recent_zscore"] = np.nan

                role_bucket = int(_role_bucket_series(
                    pd.Series([_nan_or(row.get("pre_starter_rate"), 0.0)]),
                    pd.Series([_nan_or(row.get("pre_minutes_avg5"), row.get("pre_minutes_avg10", 0.0))]),
                ).iloc[0])
                row["expected_role_bucket"] = role_bucket
                if "starter" in p_games.columns and "minutes" in p_games.columns:
                    hist_role_bucket = _role_bucket_series(
                        p_games["starter"].rolling(1, min_periods=1).mean(),
                        p_games["minutes"],
                    )
                    for col in ["points", "rebounds", "assists", "fg3m", "minutes"]:
                        if col not in p_games.columns:
                            continue
                        same_role_hist = p_games.loc[hist_role_bucket == role_bucket, col].dropna().tail(20)
                        if len(same_role_hist) >= 3:
                            row[f"pre_{col}_same_role_p75_20"] = float(same_role_hist.quantile(0.75))
                            row[f"pre_{col}_same_role_p90_20"] = float(same_role_hist.quantile(0.90))
                            row[f"pre_{col}_same_role_max20"] = float(same_role_hist.max())
                        else:
                            row[f"pre_{col}_same_role_p75_20"] = row.get(f"pre_{col}_p75_20", np.nan)
                            row[f"pre_{col}_same_role_p90_20"] = row.get(f"pre_{col}_p90_20", np.nan)
                            row[f"pre_{col}_same_role_max20"] = row.get(f"pre_{col}_max20", np.nan)

                # Role persistence: separate true role shifts from two-game spikes.
                baseline_minutes = max(_nan_or(row.get("pre_minutes_avg10"), row.get("pre_minutes_avg5", 0.0)), 8.0)
                recent_minutes_vals = recent5["minutes"].dropna()
                if not recent_minutes_vals.empty:
                    minute_dev = (recent_minutes_vals - baseline_minutes) / baseline_minutes
                    row["pre_role_minutes_consistency5"] = float((minute_dev.abs() <= 0.15).mean())
                    row["pre_role_minutes_expansion5"] = float((minute_dev >= 0.15).mean())
                    row["pre_role_minutes_reduction5"] = float((minute_dev <= -0.15).mean())
                    row["pre_role_minutes_expansion3"] = float((((recent3["minutes"].dropna() - baseline_minutes) / baseline_minutes) >= 0.15).mean()) if not recent3["minutes"].dropna().empty else np.nan
                else:
                    row["pre_role_minutes_consistency5"] = np.nan
                    row["pre_role_minutes_expansion5"] = np.nan
                    row["pre_role_minutes_reduction5"] = np.nan
                    row["pre_role_minutes_expansion3"] = np.nan
                if "trk_passes_per_min" in p_games.columns and not p_games["trk_passes_per_min"].dropna().empty:
                    recent_pass_ppm = p_games["trk_passes_per_min"].dropna().tail(5)
                    base_pass_ppm = max(
                        _nan_or(row.get("pre_trk_passes_avg10"), row.get("pre_trk_passes_avg5", 0.0)) / baseline_minutes,
                        0.1,
                    )
                    row["pre_role_passes_expansion5"] = float((((recent_pass_ppm - base_pass_ppm) / base_pass_ppm) >= 0.15).mean())
                else:
                    row["pre_role_passes_expansion5"] = np.nan

                for wow_col in [
                    "team_top_creator_out", "team_top_rebounder_out",
                    "wowy_points_top_creator_out_delta20", "wowy_assists_top_creator_out_delta20",
                    "wowy_minutes_top_creator_out_delta20", "wowy_rebounds_top_rebounder_out_delta20",
                    "wowy_minutes_top_rebounder_out_delta20",
                ]:
                    if wow_col in p_games.columns and not p_games[wow_col].dropna().empty:
                        row[wow_col] = float(p_games[wow_col].dropna().iloc[-1])
                    else:
                        row[wow_col] = np.nan

                # Venue splits
                for col in ["points", "rebounds", "assists", "minutes", "fg3m"]:
                    if col not in p_games.columns:
                        continue
                    home_series = p_games.loc[p_games["is_home"] == 1, col]
                    away_series = p_games.loc[p_games["is_home"] == 0, col]
                    # Match training-time definition: shifted expanding mean, then take latest value.
                    home_hist = home_series.shift(1).expanding(min_periods=3).mean().dropna()
                    away_hist = away_series.shift(1).expanding(min_periods=3).mean().dropna()
                    row[f"pre_{col}_home_avg"] = float(home_hist.iloc[-1]) if not home_hist.empty else np.nan
                    row[f"pre_{col}_away_avg"] = float(away_hist.iloc[-1]) if not away_hist.empty else np.nan
                    h = row.get(f"pre_{col}_home_avg")
                    a = row.get(f"pre_{col}_away_avg")
                    row[f"pre_{col}_venue_diff"] = (h - a) if pd.notna(h) and pd.notna(a) else np.nan

                # Player days rest
                last_game_time = latest.get("game_time_utc")
                if pd.notna(last_game_time):
                    row["player_days_rest"] = (asof_utc - last_game_time).total_seconds() / 86400.0
                else:
                    row["player_days_rest"] = np.nan

                # B2B and schedule fatigue
                rest_val = row.get("player_days_rest", np.nan)
                row["is_b2b"] = float(pd.notna(rest_val) and rest_val <= 1.5)
                # 3-in-4: check 2nd-to-last game time
                if len(p_games) >= 2:
                    prev2_time = p_games.iloc[-2].get("game_time_utc")
                    if pd.notna(prev2_time):
                        days_for_3 = (asof_utc - prev2_time).total_seconds() / 86400.0
                        row["is_3_in_4"] = float(days_for_3 <= 4.0)
                    else:
                        row["is_3_in_4"] = 0.0
                else:
                    row["is_3_in_4"] = 0.0
                row["b2b_x_minutes"] = row["is_b2b"] * _nan_or(row.get("pre_minutes_avg5"), 0)
                row["b2b_x_starter"] = row["is_b2b"] * _nan_or(row.get("pre_starter_rate"), 0)

                # Step 3: Season load features (scoped to team+season to match training groupby)
                if "season" in p_games.columns and "team" in p_games.columns:
                    current_season_games = p_games[
                        (p_games["season"] == p_games["season"].iloc[-1])
                        & (p_games["team"] == team)
                    ]
                elif "season" in p_games.columns:
                    current_season_games = p_games[p_games["season"] == p_games["season"].iloc[-1]]
                elif "team" in p_games.columns:
                    current_season_games = p_games[p_games["team"] == team]
                else:
                    current_season_games = p_games
                row["season_games_played"] = float(len(current_season_games))
                if "minutes" in current_season_games.columns and not current_season_games["minutes"].dropna().empty:
                    row["season_total_minutes"] = float(current_season_games["minutes"].sum())
                else:
                    row["season_total_minutes"] = 0.0
                row["high_load_flag"] = float(row["season_total_minutes"] > 1800)
                row["load_x_b2b"] = row["high_load_flag"] * row["is_b2b"]
                # load_x_age deferred until player_career_games is set below

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

                # Forward-looking injury pressure (prediction time) + backward proxy for audit.
                fwd_ctx = forward_injury_pressure.get(team, {})
                opp_fwd_ctx = forward_injury_pressure.get(opp, {})
                bwd_missing_pts = _nan_or(row.get("team_injury_proxy_missing_points5", 0), 0)
                fwd_missing_pts = _nan_or(fwd_ctx.get("fwd_missing_points"), bwd_missing_pts)
                row["team_injury_pressure_fwd"] = fwd_missing_pts
                row["team_injury_pressure_bwd"] = bwd_missing_pts
                row["injury_pressure_delta"] = fwd_missing_pts - bwd_missing_pts
                row["team_injury_pressure"] = fwd_missing_pts
                row["team_top_creator_out"] = float(team_role_context.get("team_top_creator_out", 0.0))
                row["team_top_rebounder_out"] = float(team_role_context.get("team_top_rebounder_out", 0.0))
                row["fwd_missing_minutes"] = _nan_or(fwd_ctx.get("fwd_missing_minutes"), 0.0)
                row["fwd_star_absent_flag"] = int(_nan_or(fwd_ctx.get("fwd_star_absent_flag"), 0))
                row["fwd_top1_missing_points"] = _nan_or(fwd_ctx.get("fwd_top1_missing_points"), 0.0)
                row["fwd_top2_missing_points"] = _nan_or(fwd_ctx.get("fwd_top2_missing_points"), 0.0)
                row["opp_fwd_injury_missing_points"] = _nan_or(opp_fwd_ctx.get("fwd_missing_points"), np.nan)
                row["opp_fwd_injury_missing_minutes"] = _nan_or(opp_fwd_ctx.get("fwd_missing_minutes"), np.nan)

                cov_team = injury_coverage.get("teams", {}).get(team, {})
                row["injury_feed_global_stale"] = int(injury_coverage.get("global_stale", 0))
                row["injury_feed_team_stale"] = int(cov_team.get("stale", 0))
                row["injury_feed_team_zero_out_doubtful"] = int(cov_team.get("zero_out_doubtful", 0))

                # Usage boost
                usage = _nan_or(row.get("pre_usage_proxy", 0), 0)
                fwd_pressure_scaled = _compress_injury_pressure(fwd_missing_pts)
                row["usage_boost_proxy"] = fwd_pressure_scaled * usage
                row["minutes_x_injury_pressure"] = _nan_or(row.get("pre_minutes_avg5"), 0) * fwd_pressure_scaled

                # Blowout risk
                t_net = _nan_or(row.get("team_pre_net_rating_avg5", 0), 0)
                o_net = _nan_or(row.get("opp_pre_net_rating_avg5", 0), 0)
                row["net_rating_diff"] = t_net - o_net
                row["blowout_risk"] = abs(t_net - o_net)
                row["pace_x_injury_pressure"] = row["matchup_pace_avg"] * fwd_pressure_scaled

                # Vegas game total / spread context (from upcoming schedule odds)
                row["implied_total"] = game.get("implied_total", np.nan)
                raw_spread = game.get("implied_spread", np.nan)
                # Keep sign convention aligned with training features:
                # team-perspective spread (positive=favored, negative=underdog).
                row["implied_spread"] = raw_spread if is_home else (-raw_spread if pd.notna(raw_spread) else np.nan)
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

                # Get the most recent opponent defensive profile for this position group.
                opp_latest_def = opp_def_profile[
                    (opp_def_profile["opp"] == opp) & (opp_def_profile["pos_group"] == player_pos_group)
                ] if not opp_def_profile.empty else pd.DataFrame()
                if not opp_latest_def.empty:
                    opp_latest_def = opp_latest_def.sort_values("game_time_utc")
                    latest_def = opp_latest_def.iloc[-1]
                    for feat_col in [
                        "opp_pts_allowed_to_pos_avg10", "opp_reb_allowed_to_pos_avg10",
                        "opp_ast_allowed_to_pos_avg10", "opp_fg3m_allowed_to_pos_avg10",
                        "opp_fg3a_allowed_to_pos_avg10", "opp_fg3_pct_allowed_to_pos_avg10",
                        "opp_pts_allowed_to_pos_avg10_vs_league", "opp_reb_allowed_to_pos_avg10_vs_league",
                        "opp_ast_allowed_to_pos_avg10_vs_league", "opp_fg3m_allowed_to_pos_avg10_vs_league",
                        "opp_pts_allowed_to_pos_avg10_tough_flag", "opp_reb_allowed_to_pos_avg10_tough_flag",
                        "opp_ast_allowed_to_pos_avg10_tough_flag", "opp_fg3m_allowed_to_pos_avg10_tough_flag",
                    ]:
                        row[feat_col] = latest_def.get(feat_col, np.nan)
                else:
                    row["opp_pts_allowed_to_pos_avg10"] = np.nan
                    row["opp_reb_allowed_to_pos_avg10"] = np.nan
                    row["opp_ast_allowed_to_pos_avg10"] = np.nan
                    row["opp_fg3m_allowed_to_pos_avg10"] = np.nan
                    row["opp_fg3a_allowed_to_pos_avg10"] = np.nan
                    row["opp_fg3_pct_allowed_to_pos_avg10"] = np.nan
                    row["opp_pts_allowed_to_pos_avg10_vs_league"] = np.nan
                    row["opp_reb_allowed_to_pos_avg10_vs_league"] = np.nan
                    row["opp_ast_allowed_to_pos_avg10_vs_league"] = np.nan
                    row["opp_fg3m_allowed_to_pos_avg10_vs_league"] = np.nan
                    row["opp_pts_allowed_to_pos_avg10_tough_flag"] = np.nan
                    row["opp_reb_allowed_to_pos_avg10_tough_flag"] = np.nan
                    row["opp_ast_allowed_to_pos_avg10_tough_flag"] = np.nan
                    row["opp_fg3m_allowed_to_pos_avg10_tough_flag"] = np.nan

                # Phase 8: player-vs-opponent matchup deltas
                row["player_vs_opp_pts_delta"] = (
                    _nan_or(row.get("pre_points_avg10"), 0.0)
                    - _nan_or(row.get("opp_pts_allowed_to_pos_avg10"), 0.0)
                )
                row["player_vs_opp_reb_delta"] = (
                    _nan_or(row.get("pre_rebounds_avg10"), 0.0)
                    - _nan_or(row.get("opp_reb_allowed_to_pos_avg10"), 0.0)
                )
                row["player_vs_opp_ast_delta"] = (
                    _nan_or(row.get("pre_assists_avg10"), 0.0)
                    - _nan_or(row.get("opp_ast_allowed_to_pos_avg10"), 0.0)
                )
                row["player_vs_opp_fg3m_delta"] = (
                    _nan_or(row.get("pre_fg3m_avg10"), 0.0)
                    - _nan_or(row.get("opp_fg3m_allowed_to_pos_avg10"), 0.0)
                )

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

                # Step 3: load_x_age now that player_career_games exists
                row["load_x_age"] = row["season_total_minutes"] * total_career_games

                # BRef opponent defensive stats for the upcoming opponent
                _bref_opp_data = load_bref_opponent_stats()
                # Use the latest available season for the opponent
                _bref_opp_seasons = sorted(_bref_opp_data.keys(), reverse=True) if _bref_opp_data else []
                _bref_opp_team_stats: dict[str, Any] = {}
                for _bs in _bref_opp_seasons:
                    if opp in _bref_opp_data.get(_bs, {}):
                        _bref_opp_team_stats = _bref_opp_data[_bs][opp]
                        break
                _bref_opp_rename = {
                    "opp_pts_per_g": "bref_opp_pts_per_g",
                    "opp_fg_pct": "bref_opp_fg_pct",
                    "opp_fg3_pct": "bref_opp_fg3_pct",
                    "opp_ft_per_g": "bref_opp_ft_per_g",
                    "opp_trb_per_g": "bref_opp_trb_per_g",
                    "opp_ast_per_g": "bref_opp_ast_per_g",
                    "opp_tov_per_g": "bref_opp_tov_per_g",
                    "opp_stl_per_g": "bref_opp_stl_per_g",
                    "opp_blk_per_g": "bref_opp_blk_per_g",
                    "opp_avg_dist": "bref_opp_avg_dist",
                    "opp_pct_fga_3p": "bref_opp_pct_fga_3p",
                    "opp_fg_pct_0_3": "bref_opp_fg_pct_0_3",
                    "opp_fg_pct_16_3pt": "bref_opp_fg_pct_16_3pt",
                }
                for _bref_key, _feat_name in _bref_opp_rename.items():
                    row[_feat_name] = _bref_opp_team_stats.get(_bref_key, np.nan)

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

    # Apply star-out heuristic floor boost for extreme injury-pressure scenarios
    pred_df = apply_star_out_floor_boost(pred_df)

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
        "player_vs_opp_pts_delta", "player_vs_opp_reb_delta", "player_vs_opp_ast_delta",
        "team_injury_pressure_fwd", "team_injury_pressure_bwd", "injury_pressure_delta",
        "opp_fwd_injury_missing_points", "opp_fwd_injury_missing_minutes",
        "injury_feed_global_stale", "injury_feed_team_stale", "injury_feed_team_zero_out_doubtful",
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


def apply_portfolio_caps(prop_edges: pd.DataFrame) -> pd.DataFrame:
    """Apply portfolio de-correlation caps directly to signal rows.

    Non-selected signals are converted to NO BET and annotated via
    `signal_blocked_reason` so persisted outputs match what is actionable.
    """
    if prop_edges.empty or "signal" not in prop_edges.columns:
        return prop_edges

    out = prop_edges.copy()
    signal_mask = out["signal"] != "NO BET"
    signals = out[signal_mask].copy()
    if signals.empty:
        return out

    signals["_best_ev"] = signals.apply(
        lambda r: r["ev_over"] if r["signal"] == "OVER" else r["ev_under"],
        axis=1,
    )
    signals = signals.sort_values("_best_ev", ascending=False)

    blocked_reasons: dict[int, str] = {}

    # Cap 1: per-player with same-direction suppression first.
    player_counts: dict[str, int] = {}
    player_directions: dict[str, set[str]] = defaultdict(set)
    keep_player: list[int] = []
    for idx, row in signals.iterrows():
        player = str(row.get("player_name", ""))
        direction = str(row.get("signal", ""))
        count = player_counts.get(player, 0)
        seen_dirs = player_directions[player]
        if direction in seen_dirs:
            blocked_reasons[idx] = "portfolio_player_same_direction"
            continue
        if count >= MAX_SIGNALS_PER_PLAYER:
            blocked_reasons[idx] = "portfolio_player_cap"
            continue
        keep_player.append(idx)
        player_counts[player] = count + 1
        player_directions[player].add(direction)
    signals = signals.loc[keep_player]

    # Cap 2: per-game
    game_counts: dict[tuple[str, str], int] = {}
    keep_game: list[int] = []
    for idx, row in signals.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        game_key = (home, away) if home else (away, "")
        count = game_counts.get(game_key, 0)
        if count >= MAX_SIGNALS_PER_GAME:
            blocked_reasons[idx] = "portfolio_game_cap"
            continue
        keep_game.append(idx)
        game_counts[game_key] = count + 1
    signals = signals.loc[keep_game]

    # Cap 3: per-team
    team_counts: dict[str, int] = {}
    keep_team: list[int] = []
    for idx, row in signals.iterrows():
        team = str(row.get("team", ""))
        count = team_counts.get(team, 0)
        if count >= MAX_SIGNALS_PER_TEAM:
            blocked_reasons[idx] = "portfolio_team_cap"
            continue
        keep_team.append(idx)
        team_counts[team] = count + 1
    signals = signals.loc[keep_team]

    keep_idx = set(signals.index.tolist())
    original_signal_idx = set(out[signal_mask].index.tolist())
    blocked_idx = sorted(original_signal_idx - keep_idx)
    if blocked_idx:
        out.loc[blocked_idx, "signal"] = "NO BET"
        if "confidence" in out.columns:
            out.loc[blocked_idx, "confidence"] = ""
        if "signal_blocked_reason" in out.columns:
            for idx in blocked_idx:
                reason = blocked_reasons.get(idx, "portfolio_cap")
                out.at[idx, "signal_blocked_reason"] = reason

    return out


def print_prop_edge_summary(prop_edges: pd.DataFrame) -> None:
    """Print a formatted summary of prop edge signals."""
    if prop_edges.empty:
        print("\n  No prop edge data available.", flush=True)
        return

    # Keep display output aligned with persisted actionable signals.
    prop_edges = apply_portfolio_caps(prop_edges)

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
# Phase 9: Production Monitoring — Daily Report
# ---------------------------------------------------------------------------

DAILY_METRICS_LOG = PROP_LOG_DIR / "daily_metrics.csv"


def generate_daily_report(
    prop_edges: pd.DataFrame,
    predictions: pd.DataFrame,
    target_date: str,
) -> dict[str, Any]:
    """Generate a daily health report for production monitoring.

    Checks:
    1. Feature missingness in today's predictions
    2. Prediction distribution vs historical norms
    3. Signal summary by stat/confidence/direction
    4. Model freshness (days since last retrain)

    Returns report dict and persists metrics to daily_metrics.csv.
    """
    report: dict[str, Any] = {"date": target_date, "per_stat": {}}

    # 1. Feature missingness
    # Use a fixed monitoring scope so --daily-report mode (formatted CSV) and full
    # pipeline mode (wide in-memory frame) report consistently.
    monitored_cols = [
        "pred_points",
        "pred_rebounds",
        "pred_assists",
        "pred_minutes",
        "pred_fg3m",
        "pre_points_avg5",
        "pre_rebounds_avg5",
        "pre_assists_avg5",
        "pre_minutes_avg5",
        "pre_fg3m_avg5",
        "pre_starter_rate",
        "player_days_rest",
        "injury_unavailability_prob",
        "lineup_confirmed",
        "confirmed_starter",
    ]
    missing_source = predictions if not predictions.empty else prop_edges
    present_monitored = [c for c in monitored_cols if c in missing_source.columns]

    feature_missing: dict[str, float] = {}
    if not missing_source.empty and present_monitored:
        for col in present_monitored:
            pct = float(missing_source[col].isna().mean() * 100)
            if pct > 0:
                feature_missing[col] = round(pct, 1)

    # If no lineups are confirmed today, starter flags being NaN is expected.
    if "confirmed_starter" in feature_missing and "lineup_confirmed" in missing_source.columns:
        lineup_confirmed_rate = float(pd.to_numeric(missing_source["lineup_confirmed"], errors="coerce").fillna(0).mean())
        if lineup_confirmed_rate <= 0.0:
            feature_missing.pop("confirmed_starter", None)

    max_missing_pct = max(feature_missing.values()) if feature_missing else 0.0
    high_missing = {k: v for k, v in feature_missing.items() if v > 50}
    report["feature_missing_pct_max"] = round(max_missing_pct, 1)
    report["high_missing_features"] = high_missing

    # 2. Prediction distribution per stat
    pred_cols = [c for c in predictions.columns if c.startswith("pred_") and c != "pred_starter_prob"]
    pred_stats: dict[str, dict[str, float]] = {}
    for col in pred_cols:
        vals = predictions[col].dropna()
        if vals.empty:
            continue
        pred_stats[col] = {
            "mean": round(float(vals.mean()), 2),
            "std": round(float(vals.std()), 2),
            "min": round(float(vals.min()), 2),
            "max": round(float(vals.max()), 2),
            "count": int(len(vals)),
        }
    report["prediction_distributions"] = pred_stats

    # 3. Signal summary
    signal_summary: dict[str, dict[str, int]] = {}
    if not prop_edges.empty:
        signals = prop_edges[prop_edges["signal"] != "NO BET"]
        for stat_type, grp in signals.groupby("stat_type"):
            signal_summary[str(stat_type)] = {
                "n_signals": int(len(grp)),
                "n_best_bets": int((grp["confidence"] == "BEST BET").sum()),
                "n_leans": int((grp["confidence"] == "LEAN").sum()),
                "n_over": int((grp["signal"] == "OVER").sum()),
                "n_under": int((grp["signal"] == "UNDER").sum()),
            }
    report["signal_summary"] = signal_summary

    # 4. Model freshness
    model_age_days = np.nan
    model_files = list(MODEL_DIR.glob("*.joblib"))
    if model_files:
        newest = max(model_files, key=lambda p: p.stat().st_mtime)
        age_secs = time.time() - newest.stat().st_mtime
        model_age_days = round(age_secs / 86400.0, 1)
    report["model_age_days"] = model_age_days

    # Print report
    print(f"\n{'=' * 60}", flush=True)
    print(f"  DAILY HEALTH REPORT — {target_date}", flush=True)
    print(f"{'=' * 60}", flush=True)

    if high_missing:
        print(f"  [ALERT] Monitored fields with >50% NaN:", flush=True)
        for feat, pct in sorted(high_missing.items(), key=lambda x: -x[1])[:10]:
            print(f"    {feat}: {pct:.1f}% missing", flush=True)
    else:
        print(f"  Feature missingness: OK (max {max_missing_pct:.1f}%)", flush=True)

    print(f"\n  Prediction distributions:", flush=True)
    for col, stats in pred_stats.items():
        print(f"    {col}: mean={stats['mean']:.1f} std={stats['std']:.1f} "
              f"[{stats['min']:.1f}, {stats['max']:.1f}] n={stats['count']}", flush=True)

    n_total_signals = sum(s["n_signals"] for s in signal_summary.values())
    n_total_best = sum(s["n_best_bets"] for s in signal_summary.values())
    print(f"\n  Signals: {n_total_signals} total ({n_total_best} best bets)", flush=True)
    for stat, counts in signal_summary.items():
        print(f"    {stat}: {counts['n_signals']} signals "
              f"({counts['n_over']}O/{counts['n_under']}U, "
              f"{counts['n_best_bets']} best)", flush=True)

    if pd.notna(model_age_days):
        freshness = "OK" if model_age_days < 14 else "STALE"
        print(f"\n  Model freshness: {model_age_days:.1f} days [{freshness}]", flush=True)
    else:
        print(f"\n  Model freshness: no model files found", flush=True)

    print(f"{'=' * 60}\n", flush=True)

    # Persist metrics to daily log
    PROP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict[str, Any]] = []
    stat_types = set(signal_summary.keys())
    if not prop_edges.empty and "stat_type" in prop_edges.columns:
        stat_types |= set(prop_edges["stat_type"].unique())
    for stat in sorted(stat_types):
        ss = signal_summary.get(stat, {})
        stat_signals = (
            prop_edges[
                (prop_edges["stat_type"] == stat) & (prop_edges["signal"] != "NO BET")
            ].copy()
            if not prop_edges.empty
            else pd.DataFrame()
        )
        mean_ev = np.nan
        if not stat_signals.empty:
            selected_ev = np.where(
                stat_signals["signal"] == "OVER",
                pd.to_numeric(stat_signals.get("ev_over"), errors="coerce"),
                pd.to_numeric(stat_signals.get("ev_under"), errors="coerce"),
            )
            if np.isfinite(selected_ev).any():
                mean_ev = float(np.nanmean(selected_ev))
        log_rows.append({
            "date": target_date,
            "stat_type": stat,
            "n_predictions": int(len(prop_edges[prop_edges["stat_type"] == stat])) if not prop_edges.empty else 0,
            "n_signals": ss.get("n_signals", 0),
            "n_best_bets": ss.get("n_best_bets", 0),
            "n_leans": ss.get("n_leans", 0),
            "mean_pred": pred_stats.get(f"pred_{stat}", {}).get("mean", np.nan),
            "mean_edge": float(prop_edges.loc[prop_edges["stat_type"] == stat, "edge"].mean())
                if not prop_edges.empty and "edge" in prop_edges.columns
                and len(prop_edges[prop_edges["stat_type"] == stat]) > 0
                else np.nan,
            "mean_ev": mean_ev,
            "feature_missing_pct_max": max_missing_pct,
            "model_age_days": model_age_days,
        })
    if log_rows:
        new_log = pd.DataFrame(log_rows)
        if DAILY_METRICS_LOG.exists():
            try:
                existing = pd.read_csv(DAILY_METRICS_LOG)
                existing = existing[existing["date"].astype(str) != str(target_date)]
                combined = pd.concat([existing, new_log], ignore_index=True)
            except Exception:
                combined = new_log
        else:
            combined = new_log
        combined.to_csv(DAILY_METRICS_LOG, index=False)

    return report


# ---------------------------------------------------------------------------
# Phase 10: Deploy Gates
# ---------------------------------------------------------------------------


def check_deploy_gates(target_date: str) -> dict[str, Any]:
    """Automated go/no-go checks before signals are published.

    Returns a dict with gate names mapped to pass/fail + details.
    """
    results: dict[str, Any] = {"date": target_date, "gates": {}}

    # Load graded history
    history = pd.DataFrame()
    if PROP_RESULTS_HISTORY_FILE.exists():
        try:
            history = pd.read_csv(PROP_RESULTS_HISTORY_FILE)
        except Exception:
            pass
    history = _canonical_latest_view(history)

    graded = history[history["actual_value"].notna()].copy() if not history.empty else pd.DataFrame()

    # Gate 1: Min graded per stat
    gate1: dict[str, Any] = {"passed": True, "details": {}, "affected_stats": set()}
    for stat in ["points", "rebounds", "assists", "fg3m"]:
        n = int(len(graded[graded["stat_type"] == stat])) if not graded.empty else 0
        gate1["details"][stat] = n
        if n < DEPLOY_GATE_MIN_GRADED_PER_STAT:
            gate1["passed"] = False
            gate1["affected_stats"].add(stat)
    results["gates"]["min_graded_per_stat"] = gate1

    # Gate 2: Positive trailing CLV (last 30 days) with minimum sample
    gate2: dict[str, Any] = {"passed": True, "details": {}, "status": "PASS"}
    if not graded.empty and "game_date_est" in graded.columns:
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        recent = graded[
            (graded["game_date_est"].astype(str) >= cutoff)
            & (graded["signal"] != "NO BET")
        ]
        if not recent.empty and "line_move" in recent.columns:
            with_line_snapshot = recent.copy()
            with_line_snapshot["line_move"] = pd.to_numeric(with_line_snapshot["line_move"], errors="coerce")
            with_line_snapshot = with_line_snapshot[with_line_snapshot["line_move"].notna()].copy()

            # CLV sample should only include rows with actual movement.
            with_move = with_line_snapshot[with_line_snapshot["line_move"].abs() > 1e-9].copy()
            n_with_move = len(with_move)
            gate2["details"]["n_signals_30d"] = int(len(recent))
            gate2["details"]["n_with_line_snapshot_30d"] = int(len(with_line_snapshot))
            gate2["details"]["n_with_line_move_30d"] = n_with_move
            gate2["details"]["min_sample_required"] = DEPLOY_CLV_MIN_SAMPLE

            if n_with_move >= DEPLOY_CLV_MIN_SAMPLE:
                clv_dir = np.where(
                    with_move["signal"] == "OVER",
                    with_move["line_move"].to_numpy(dtype=float),
                    -with_move["line_move"].to_numpy(dtype=float),
                )
                mean_clv = float(np.nanmean(clv_dir))
                gate2["details"]["avg_clv_line_pts_30d"] = round(mean_clv, 3)

                # Separate odds-based CLV if closing odds available
                if "closing_odds_over" in with_move.columns and "closing_odds_under" in with_move.columns:
                    odds_clv_vals = []
                    for _, r in with_move.iterrows():
                        open_over = pd.to_numeric(r.get("over_odds"), errors="coerce")
                        close_over = pd.to_numeric(r.get("closing_odds_over"), errors="coerce")
                        if pd.notna(open_over) and pd.notna(close_over):
                            # CLV = implied prob improvement in signal direction
                            open_ip = _american_odds_to_implied_prob(open_over)
                            close_ip = _american_odds_to_implied_prob(close_over)
                            if pd.notna(open_ip) and pd.notna(close_ip):
                                clv_odds = (close_ip - open_ip) if r.get("signal") == "OVER" else (open_ip - close_ip)
                                odds_clv_vals.append(clv_odds)
                    if odds_clv_vals:
                        gate2["details"]["avg_clv_odds_30d"] = round(float(np.mean(odds_clv_vals)), 4)
                        gate2["details"]["n_with_odds_clv_30d"] = len(odds_clv_vals)

                if mean_clv < DEPLOY_GATE_MIN_CLV:
                    gate2["passed"] = False
                    gate2["status"] = "FAIL"
            else:
                # Insufficient sample — treat as not deployment-ready.
                gate2["passed"] = False
                gate2["status"] = "INSUFFICIENT_DATA"
                gate2["details"]["reason"] = f"insufficient_sample ({n_with_move} < {DEPLOY_CLV_MIN_SAMPLE})"
        else:
            gate2["passed"] = False
            gate2["status"] = "INSUFFICIENT_DATA"
            gate2["details"]["n_signals_30d"] = 0 if recent.empty else int(len(recent))
            gate2["details"]["n_with_line_move_30d"] = 0
            gate2["details"]["reason"] = "no_line_move_data"
    else:
        gate2["passed"] = False
        gate2["status"] = "INSUFFICIENT_DATA"
        gate2["details"]["reason"] = "no graded history"
    results["gates"]["positive_trailing_clv"] = gate2

    # Gate 3: Calibration tolerance
    gate3: dict[str, Any] = {"passed": True, "details": {}, "affected_stats": set()}
    if not graded.empty and "game_date_est" in graded.columns:
        cutoff = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        recent_signal = graded[
            (graded["game_date_est"].astype(str) >= cutoff)
            & (graded["signal"] != "NO BET")
        ]
        if not recent_signal.empty:
            recent_signal = recent_signal.copy()
            recent_signal["p_hit"] = np.where(
                recent_signal["signal"] == "UNDER",
                recent_signal["p_under"].astype(float),
                recent_signal["p_over"].astype(float),
            )
            for stat, grp in recent_signal.groupby("stat_type"):
                valid = grp[grp["p_hit"].notna() & grp["hit"].notna()]
                if len(valid) < 30:
                    continue
                brier = float(np.mean(
                    (valid["p_hit"].astype(float) - valid["hit"].astype(float)) ** 2
                ))
                gate3["details"][str(stat)] = {"brier_30d": round(brier, 4), "n": int(len(valid))}
                if brier > DEPLOY_GATE_MAX_BRIER:
                    gate3["passed"] = False
                    gate3["affected_stats"].add(str(stat))
    results["gates"]["calibration_tolerance"] = gate3

    # Gate 4: Model freshness
    gate4: dict[str, Any] = {"passed": True, "details": {}}
    model_files = list(MODEL_DIR.glob("*.joblib"))
    if model_files:
        newest = max(model_files, key=lambda p: p.stat().st_mtime)
        age_days = (time.time() - newest.stat().st_mtime) / 86400.0
        gate4["details"]["model_age_days"] = round(age_days, 1)
        gate4["details"]["newest_model"] = newest.name
        if age_days > DEPLOY_GATE_MAX_MODEL_AGE_DAYS:
            gate4["passed"] = False
    else:
        gate4["passed"] = True  # No persisted models = training fresh each run
        gate4["details"]["reason"] = "no persisted model files (training fresh)"
    results["gates"]["model_freshness"] = gate4

    # Summary
    all_passed = all(g["passed"] for g in results["gates"].values())
    results["all_passed"] = all_passed
    return results


def _print_deploy_gate_status(gate_results: dict[str, Any]) -> None:
    """Pretty-print deploy gate status."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"  DEPLOY GATE STATUS — {gate_results.get('date', '')}", flush=True)
    print(f"{'=' * 60}", flush=True)

    for gate_name, gate_info in gate_results.get("gates", {}).items():
        status = str(gate_info.get("status", "PASS" if gate_info.get("passed", False) else "FAIL"))
        marker = "  " if gate_info.get("passed", False) else "  [!]"
        print(f"{marker} {gate_name}: {status}", flush=True)
        details = gate_info.get("details", {})
        if isinstance(details, dict):
            for k, v in details.items():
                print(f"      {k}: {v}", flush=True)
        affected = gate_info.get("affected_stats", set())
        if affected:
            print(f"      affected_stats: {', '.join(sorted(affected))}", flush=True)

    overall = "ALL GATES PASSED" if gate_results.get("all_passed", False) else "SOME GATES FAILED"
    enforce = "(ENFORCING)" if DEPLOY_GATES_ENFORCE else "(ADVISORY)"
    print(f"\n  {overall} {enforce}", flush=True)

    # Show experiment log trend if available
    if EXPERIMENT_LOG_FILE.exists():
        try:
            log = pd.read_csv(EXPERIMENT_LOG_FILE)
            recent = log[log["experiment_type"] == "weekly_retrain"].tail(5)
            if not recent.empty:
                print(f"\n  Recent weekly retrains ({len(recent)}):", flush=True)
                for _, row in recent.iterrows():
                    ts = str(row.get("timestamp", ""))[:10]
                    roi = row.get("roi", "N/A")
                    alerts = row.get("n_alerts", "N/A")
                    print(f"    {ts}: ROI={roi}  alerts={alerts}", flush=True)
        except Exception:
            pass
    print(f"{'=' * 60}\n", flush=True)


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
        team = str(edge_row.get("team", ""))
        matched = _match_actual_player_rows(
            actual_games,
            player_name=str(player_name),
            team=team,
            player_id=edge_row.get("player_id", np.nan),
        )
        if matched.empty or len(matched) != 1:
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
    history = _canonical_latest_view(history)
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

    # --- Star-out nights subset (forward pressure > p90 threshold) ---
    star_out_threshold = 30.0  # ~p90 of training team_injury_pressure
    fwd_col = "team_injury_pressure_fwd"
    base_df = signal_graded if not signal_graded.empty else graded
    if fwd_col in base_df.columns:
        fwd_pressure = pd.to_numeric(base_df[fwd_col], errors="coerce").fillna(0)
        star_out_mask = fwd_pressure > star_out_threshold
        normal_mask = ~star_out_mask
        star_out_rows = base_df[star_out_mask]
        normal_rows = base_df[normal_mask]

        star_out_metrics: list[dict[str, Any]] = []
        for label, subset in [("star_out", star_out_rows), ("normal", normal_rows)]:
            if len(subset) < 10:
                continue
            hit = subset["hit"].astype(float)
            p_hit = subset["p_hit"].astype(float)
            pnl = np.where(hit == 1, VIG_FACTOR, -1.0)
            star_out_metrics.append({
                "group": label,
                "n": len(subset),
                "hit_rate": round(float(hit.mean()), 4),
                "mean_p_hit": round(float(p_hit.mean()), 4),
                "gap": round(float(abs(hit.mean() - p_hit.mean())), 4),
                "roi_pct": round(float(pnl.mean()) * 100, 2),
            })
        if star_out_metrics:
            report["by_star_out"] = star_out_metrics

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

    if report.get("by_star_out"):
        print(f"\n  --- Star-Out Nights (fwd_pressure > 30) ---", flush=True)
        for e in report["by_star_out"]:
            print(
                f"    {str(e['group']):10s}: n={e['n']:>4d}  "
                f"HitRate={e['hit_rate']:.1%}  Pred={e['mean_p_hit']:.1%}  "
                f"Gap={e['gap']:.3f}  ROI={e['roi_pct']:+.1f}%",
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
    history = _canonical_latest_view(history)

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
    run_id: str = "",
    asof_utc: pd.Timestamp | None = None,
) -> Path:
    """Save ALL prop edge rows (signals + NO BET) to canonical results history.

    Predictions are stored with actual_value=NaN. Actuals are filled later by
    ``grade_canonical_results()``.

    Snapshot behavior:
    - Appends immutable run snapshots (no date overwrite).
    - Marks latest snapshot rows per date with ``is_latest=1``.
    - Prior runs are retained with ``is_latest=0``.

    Returns the path to the history file.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if prop_edges.empty:
        return PROP_RESULTS_HISTORY_FILE

    asof_utc = asof_utc if asof_utc is not None else pd.Timestamp.now(tz="UTC")
    now_str = asof_utc.strftime("%Y-%m-%d %H:%M:%S")
    run_id = str(run_id or f"run_{asof_utc.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}")
    rows: list[dict[str, Any]] = []
    for _, r in prop_edges.iterrows():
        name_norm = normalize_player_name(r.get("player_name", ""))
        stat_type = str(r.get("stat_type", ""))
        pid = generate_prediction_id(
            target_date,
            name_norm,
            stat_type,
            team=str(r.get("team", "")),
            player_id=r.get("player_id", np.nan),
            home_team=str(r.get("home_team", "")),
            away_team=str(r.get("away_team", "")),
        )
        rows.append({
            "prediction_id": pid,
            "game_date_est": target_date,
            "player_name": r.get("player_name", ""),
            "player_name_norm": name_norm,
            "player_id": r.get("player_id", np.nan),
            "team": r.get("team", ""),
            "opp": r.get("opp", ""),
            "home_team": r.get("home_team", ""),
            "away_team": r.get("away_team", ""),
            "stat_type": stat_type,
            # Decision-time features
            "pred_value": r.get("pred_value", np.nan),
            "pred_value_base": r.get("pred_value_base", np.nan),
            "market_resid_adj": r.get("market_resid_adj", np.nan),
            "bias_adj": r.get("bias_adj", np.nan),
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
            "closing_odds_over": r.get("closing_odds_over", np.nan),
            "closing_odds_under": r.get("closing_odds_under", np.nan),
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
            "run_id": run_id,
            "asof_utc": asof_utc.isoformat(),
            "is_latest": 1,
            "prediction_created_at": now_str,
            "model_version": MODEL_VERSION,
        })

    new_df = pd.DataFrame(rows)

    # Load existing history, mark prior date snapshots non-latest, append new.
    if PROP_RESULTS_HISTORY_FILE.exists():
        try:
            existing = pd.read_csv(PROP_RESULTS_HISTORY_FILE)
            if "is_latest" not in existing.columns:
                existing["is_latest"] = 1
            if "run_id" not in existing.columns:
                existing["run_id"] = ""
            if "asof_utc" not in existing.columns:
                existing["asof_utc"] = ""
            date_mask = existing["game_date_est"].astype(str) == str(target_date)
            existing.loc[date_mask, "is_latest"] = 0
            # Idempotent if same run_id/date is saved again.
            same_run = date_mask & (existing["run_id"].astype(str) == run_id)
            existing = existing[~same_run]
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df

    combined.to_csv(PROP_RESULTS_HISTORY_FILE, index=False)
    print(
        f"  Canonical results saved: {len(new_df)} rows for {target_date} "
        f"(run_id={run_id}) -> {PROP_RESULTS_HISTORY_FILE}",
        flush=True,
    )
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
    if "is_latest" in history.columns:
        date_mask = date_mask & history["is_latest"].fillna(0).astype(int).eq(1)
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

    # Derive game_date_est from game_time_utc if missing (Eastern time, YYYYMMDD format)
    if "game_date_est" not in player_games.columns and "game_time_utc" in player_games.columns:
        utc_ts = pd.to_datetime(player_games["game_time_utc"], utc=True, errors="coerce")
        eastern = utc_ts.dt.tz_convert("US/Eastern")
        player_games["game_date_est"] = eastern.dt.strftime("%Y%m%d")

    if "game_date_est" in player_games.columns:
        # Normalise both sides to string for comparison
        pg_dates = player_games["game_date_est"].astype(str)
        target_str = str(target_date).replace("-", "")
        actual_games = player_games[pg_dates == target_str].copy()
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

        # Match by player_id first, then strict normalized name + team.
        matched = _match_actual_player_rows(
            actual_games,
            player_name=player_name,
            team=team,
            player_id=row.get("player_id", np.nan),
        )
        if matched.empty or len(matched) != 1:
            continue

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

    # Recompute rolling points bias after grading
    if USE_POINTS_BIAS_CORRECTION:
        bias_info = compute_points_bias(history)
        save_points_bias(bias_info)
        status = "ACTIVE" if bias_info["active"] else "INACTIVE"
        print(f"  Points bias: {status} — raw={bias_info['raw_bias']:+.3f}, "
              f"shrunk={bias_info['shrunk_bias']:+.3f}, n={bias_info['n']} "
              f"({bias_info['reason']})", flush=True)

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
    migrate_run_id = f"migrate_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    rows: list[dict[str, Any]] = []
    for _, r in tracking.iterrows():
        date = str(r.get("date", ""))
        name = str(r.get("player_name", ""))
        name_norm = normalize_player_name(name)
        stat_type = str(r.get("stat_type", ""))
        pid = generate_prediction_id(date, name_norm, stat_type, team=str(r.get("team", "")))
        rows.append({
            "prediction_id": pid,
            "game_date_est": date,
            "player_name": name,
            "player_name_norm": name_norm,
            "player_id": np.nan,
            "team": r.get("team", ""),
            "opp": "",
            "home_team": "",
            "away_team": "",
            "stat_type": stat_type,
            "pred_value": r.get("pred_value", np.nan),
            "pred_value_base": np.nan,
            "market_resid_adj": np.nan,
            "bias_adj": np.nan,
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
            "closing_odds_over": np.nan,
            "closing_odds_under": np.nan,
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
            "run_id": migrate_run_id,
            "asof_utc": "",
            "is_latest": 1,
            "model_version": "migrated",
        })

    migrated = pd.DataFrame(rows)

    # Merge with existing history (deduplicate by prediction_id)
    if PROP_RESULTS_HISTORY_FILE.exists():
        existing = pd.read_csv(PROP_RESULTS_HISTORY_FILE)
        if "is_latest" not in existing.columns:
            existing["is_latest"] = 1
        if "run_id" not in existing.columns:
            existing["run_id"] = ""
        if "asof_utc" not in existing.columns:
            existing["asof_utc"] = ""
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
    1b. Optuna tuning (if --tune)
    1c. Auto feature selection (if --auto-feature-select)
    2. Train fresh models (with tuned params if available)
    3. Run market-line backtest for performance snapshot
    4. Run calibration report
    5. Print summary + log experiment
    """
    target_date = args.date or datetime.now().strftime("%Y%m%d")
    do_tune = getattr(args, "tune", False)
    n_tune_trials = getattr(args, "tune_trials", 50)
    do_auto_select = getattr(args, "auto_feature_select", False)
    print(f"  Weekly retrain as of {target_date}", flush=True)

    # Step 1: Invalidate feature cache
    invalidated_cache = False
    if PLAYER_FEATURE_CACHE_FILE.exists():
        PLAYER_FEATURE_CACHE_FILE.unlink()
        print("  Invalidated player feature cache.", flush=True)
        invalidated_cache = True
    if PLAYER_FEATURE_CACHE_META.exists():
        PLAYER_FEATURE_CACHE_META.unlink()
        invalidated_cache = True

    # Step 1a: Rebuild and persist the player feature cache that weekly retrain just invalidated.
    # Weekly retrain loads player_df before entering this branch, so without an explicit rebuild it
    # would train on the in-memory frame and leave no on-disk cache behind for subsequent runs.
    if invalidated_cache or not PLAYER_FEATURE_CACHE_FILE.exists() or not PLAYER_FEATURE_CACHE_META.exists():
        print("  Rebuilding player feature cache...", flush=True)
        player_df = load_or_build_player_features(
            player_games,
            team_games,
            game_odds,
            min_games=args.min_games,
            ref_features=ref_features,
            box_adv_fetch_missing=args.box_adv_fetch_missing,
            box_adv_max_fetch=args.box_adv_max_fetch,
        )
        if PLAYER_FEATURE_CACHE_FILE.exists() and PLAYER_FEATURE_CACHE_META.exists():
            print(f"  Rebuilt player feature cache: {len(player_df)} rows", flush=True)
        else:
            print("  Warning: player feature cache rebuild did not persist artifacts.", flush=True)

    targets = list(PROP_TARGETS)
    if "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 100:
        targets.append("fg3m")
    selected_groups = load_selected_feature_groups(player_df, targets)

    # Step 1b: Auto feature selection (BEFORE tuning, since tuning depends on feature set)
    if do_auto_select:
        print("\n  Running automated feature group selection...", flush=True)
        selected_groups = run_auto_feature_selection(player_df)
        print(f"  Selected {len(selected_groups)} feature groups.", flush=True)

    # Step 1c: Invalidate stale tuned params when re-tuning
    # (train_prediction_models handles the actual tuning/loading)
    if do_tune and TUNED_PARAMS_FILE.exists():
        TUNED_PARAMS_FILE.unlink()
        print("  Invalidated stale tuned params (will re-tune).", flush=True)

    # Step 2: Train fresh models (tune=True triggers Optuna inside train_prediction_models)
    print("\n  Training fresh core prop models...", flush=True)
    two_stage_models, single_models = train_prediction_models(
        player_df,
        tune=do_tune,
        n_tune_trials=n_tune_trials,
        selected_groups=selected_groups,
    )
    n_two_stage = len([k for k in two_stage_models if not k.startswith("_")])
    n_residual = len(two_stage_models.get("_residual", {}))
    print(f"  Two-stage models: {n_two_stage}  Residual models: {n_residual}", flush=True)
    print(f"  Single models: {len(single_models)}", flush=True)

    # Step 3: Market-line backtest
    roi_val = None
    clv_val = None
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
            roi_val = result.get("roi_pct", "N/A")
            clv_val = result.get("avg_clv_line_pts", "N/A")
            print(f"  Market backtest ROI: {roi_val}  CLV: {clv_val}", flush=True)

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

    # Step 5: Summary + experiment log
    print(f"\n  === Weekly Retrain Complete ===", flush=True)
    print(f"  Date: {target_date}", flush=True)
    print(f"  Models: {n_two_stage} two-stage + {n_residual} residual + {len(single_models)} single", flush=True)
    if report.get("alerts"):
        print(f"  Calibration alerts: {len(report['alerts'])}", flush=True)
    else:
        print(f"  Calibration: OK (no alerts)", flush=True)

    append_experiment_result(
        experiment_type="weekly_retrain",
        description=f"Weekly retrain {target_date}" + (" +tune" if do_tune else ""),
        metrics={
            "n_two_stage": n_two_stage,
            "n_residual": n_residual,
            "n_single": len(single_models),
            "n_alerts": len(report.get("alerts", [])),
            "roi": roi_val,
            "clv": clv_val,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict NBA player props")
    p.add_argument("--date", type=str, default=None,
                   help="Target date YYYYMMDD (default: today)")
    p.add_argument("--asof-utc", type=str, default=None,
                   help="Freeze run context time in UTC (ISO string, e.g. 2026-03-03T20:00:00Z)")
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
    # Step 4: Player target encoding (experimental)
    p.add_argument("--enable-player-encoding", action="store_true",
                   help="Enable experimental per-player target encoding features")
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
    # Phase 6: Feature ablation framework
    p.add_argument("--ablation-features", action="store_true",
                   help="Run generalized feature group ablation (walk-forward)")
    # Optuna hyperparameter tuning
    p.add_argument("--tune", action="store_true",
                   help="Run Optuna hyperparameter tuning for prop models (or use cached)")
    p.add_argument("--tune-trials", type=int, default=50,
                   help="Number of Optuna trials per model/target (default: 50)")
    # Automated feature group selection
    p.add_argument("--auto-feature-select", action="store_true",
                   help="Run greedy forward feature group selection (walk-forward)")
    # Phase 9: Production monitoring
    p.add_argument("--daily-report", action="store_true",
                   help="Generate daily health report from latest predictions")
    # Phase 10: Deploy gates
    p.add_argument("--deploy-status", action="store_true",
                   help="Print current deploy gate status without running predictions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_asof_utc = parse_asof_utc(getattr(args, "asof_utc", None))
    run_id = f"run_{run_asof_utc.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

    # Wire CLI flags to module-level globals
    global USE_PLAYER_TARGET_ENCODING
    if getattr(args, "enable_player_encoding", False):
        USE_PLAYER_TARGET_ENCODING = True

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROP_LINES_DIR.mkdir(parents=True, exist_ok=True)
    PROP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PROP_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Grade results mode (Phase 1) ---
    if args.grade_results:
        target_date = args.date or (run_asof_utc - timedelta(days=1)).strftime("%Y%m%d")
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
        target_date = args.date or (run_asof_utc - timedelta(days=1)).strftime("%Y%m%d")
        if PROP_RESULTS_HISTORY_FILE.exists():
            print(f"Grading canonical results for {target_date} (via --track-clv)...", flush=True)
            grade_canonical_results(target_date)
        else:
            print(f"Tracking CLV for {target_date}...", flush=True)
            track_prop_clv(target_date)
        return

    # --- Calibration report mode (history-only) ---
    if args.calibration_report:
        print("\nGenerating calibration report from canonical results...", flush=True)
        report = compute_calibration_report()
        print_calibration_report(report)
        report_path = PROP_LOG_DIR / f"calibration_report_{run_asof_utc.strftime('%Y%m%d')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved to {report_path}", flush=True)
        return

    # --- Daily report mode (history/prediction-file only) ---
    if args.daily_report:
        target_date = args.date or run_asof_utc.strftime("%Y%m%d")
        print(f"\nGenerating daily health report for {target_date}...", flush=True)
        pred_file = PREDICTIONS_DIR / f"player_props_{target_date}.csv"
        edge_file = PREDICTIONS_DIR / f"player_prop_edges_{target_date}.csv"
        preds = pd.read_csv(pred_file) if pred_file.exists() else pd.DataFrame()
        edges = pd.read_csv(edge_file) if edge_file.exists() else pd.DataFrame()
        generate_daily_report(edges, preds, target_date)
        return

    # --- Deploy status mode (history-only) ---
    if args.deploy_status:
        target_date = args.date or run_asof_utc.strftime("%Y%m%d")
        print(f"\nDeploy gate status for {target_date}...", flush=True)
        gate_results = check_deploy_gates(target_date)
        _print_deploy_gate_status(gate_results)
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

    # --- Market lines ablation mode (Phase 3) ---
    if args.ablation_market_lines or args.ablation_features:
        if args.ablation_features:
            print("\nRunning generalized feature group ablation...", flush=True)
            run_feature_ablation(player_df)
        else:
            print("\nRunning market-line feature ablation (walk-forward with/without market features)...", flush=True)
            _run_market_line_ablation(player_df)
        return

    # --- Auto feature selection mode ---
    if args.auto_feature_select and not args.weekly_retrain:
        print("\nRunning automated feature group selection...", flush=True)
        run_auto_feature_selection(player_df)
        return

    # --- Weekly retrain mode ---
    if args.weekly_retrain:
        print("\nRunning weekly retrain...", flush=True)
        run_weekly_retrain(player_df, schedule_df, team_games, game_odds, ref_features, args)
        return

    # --- Prediction mode ---
    target_date = args.date or run_asof_utc.strftime("%Y%m%d")
    print(f"\nFetching upcoming games for {target_date}...", flush=True)
    upcoming = fetch_upcoming_schedule(target_date, args.days)
    if upcoming.empty:
        print(f"No upcoming games found for {target_date}.", flush=True)
        return

    print(f"  Found {len(upcoming)} upcoming games", flush=True)
    for _, g in upcoming.iterrows():
        print(f"    {g['away_team']} @ {g['home_team']} ({g.get('status', '')})", flush=True)

    print("\nChecking injury report...", flush=True)
    injury_map = fetch_injury_status_map(target_date, asof_utc=run_asof_utc)
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
        now_utc=run_asof_utc,
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
    two_stage_models, single_models = train_prediction_models(
        player_df, tune=args.tune, n_tune_trials=args.tune_trials,
    )

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
        asof_utc=run_asof_utc,
        run_id=run_id,
    )

    if pred_df.empty:
        print("No predictions generated.", flush=True)
        return

    pred_df = apply_injury_status_to_predictions(pred_df, injury_map)
    pred_df = filter_out_inactive(pred_df, injury_map, remove_doubtful=(not args.keep_doubtful))

    market_residual_models: dict[str, dict[str, Any]] = {}
    prob_calibrators: dict[str, dict[str, Any]] = {}
    print("\nTraining market residual/calibration models from cached prop lines...", flush=True)
    try:
        residual_models, market_calibs, diag = train_market_residual_models(
            player_df,
            max_dates=args.market_model_max_dates,
            pretrained_models=(two_stage_models, single_models),
        )
    except Exception as e:
        print(
            f"  [WARN] Market residual/calibration training failed: {type(e).__name__}: {e}",
            flush=True,
        )
        residual_models, market_calibs, diag = {}, {}, {"rows": 0, "per_stat": {}, "error": str(e)}
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
            two_stage_models=two_stage_models,
            single_models=single_models,
            leakage_safe=False,
        )
        for stat, std in residual_stds.items():
            print(f"  {stat}: residual_std = {std:.2f}", flush=True)

        # Phase 2: check calibration reliability gate
        degraded_stats = get_calibration_degraded_stats()
        if degraded_stats:
            print(f"  Calibration drift gate active for: {', '.join(sorted(degraded_stats))}", flush=True)

        # Phase 4: Extract uncertainty models from two_stage_models
        unc_models = two_stage_models.get("_uncertainty", None) if two_stage_models else None
        q_unc_models = two_stage_models.get("_uncertainty_quantile", None) if two_stage_models else None

        # Log active bias correction
        pts_bias = get_active_points_bias()
        if pts_bias != 0.0:
            print(f"  Points bias correction active: {pts_bias:+.3f}", flush=True)

        print("Computing prop edges...", flush=True)
        prop_edges = compute_prop_edges(
            pred_df,
            prop_lines,
            residual_stds,
            market_residual_models=market_residual_models,
            prob_calibrators=prob_calibrators,
            calibration_degraded_stats=degraded_stats,
            uncertainty_models=unc_models,
            quantile_uncertainty_models=q_unc_models,
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

            # Phase 2: enforce portfolio caps in persisted outputs.
            cap_pre = int((prop_edges["signal"] != "NO BET").sum())
            prop_edges = apply_portfolio_caps(prop_edges)
            cap_post = int((prop_edges["signal"] != "NO BET").sum())
            if cap_post != cap_pre:
                print(f"  Portfolio caps: {cap_pre} -> {cap_post} signals", flush=True)

            # Phase 10: apply deploy gates before writing outputs.
            gate_results = check_deploy_gates(target_date)
            any_fail = any(not g.get("passed", True) for g in gate_results.get("gates", {}).values())
            if any_fail:
                failed_gates = [k for k, v in gate_results.get("gates", {}).items() if not v.get("passed", True)]
                print(f"\n  [DEPLOY GATE WARNING] Failed gates: {', '.join(failed_gates)}", flush=True)
                if DEPLOY_GATES_ENFORCE:
                    gate_blocked = 0
                    for gate_name in failed_gates:
                        gate_info = gate_results["gates"][gate_name]
                        affected = gate_info.get("affected_stats", set())
                        if affected:
                            for stat in affected:
                                mask = (prop_edges["stat_type"] == stat) & (prop_edges["signal"] != "NO BET")
                                gate_blocked += int(mask.sum())
                                prop_edges.loc[mask, "signal"] = "NO BET"
                                prop_edges.loc[mask, "confidence"] = ""
                                if "signal_blocked_reason" in prop_edges.columns:
                                    prop_edges.loc[mask, "signal_blocked_reason"] = f"deploy_gate:{gate_name}"
                        else:
                            mask = prop_edges["signal"] != "NO BET"
                            gate_blocked += int(mask.sum())
                            prop_edges.loc[mask, "signal"] = "NO BET"
                            prop_edges.loc[mask, "confidence"] = ""
                            if "signal_blocked_reason" in prop_edges.columns:
                                prop_edges.loc[mask, "signal_blocked_reason"] = f"deploy_gate:{gate_name}"
                    if gate_blocked:
                        print(f"  Deploy gate enforcement blocked {gate_blocked} signals", flush=True)

            edge_out_path = PREDICTIONS_DIR / f"player_prop_edges_{target_date}.csv"
            prop_edges.to_csv(edge_out_path, index=False)
            print(f"  Prop edges saved to {edge_out_path}", flush=True)

            # Save canonical results (Phase 1): all predictions (signals + NO BET)
            save_canonical_results(
                prop_edges,
                target_date,
                ACTIVE_SIGNAL_POLICY_MODE,
                run_id=run_id,
                asof_utc=run_asof_utc,
            )

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

    # Phase 9: Auto-generate daily health report
    try:
        generate_daily_report(prop_edges, pred_df, target_date)
    except Exception:
        pass  # Don't let report crash the main pipeline

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
