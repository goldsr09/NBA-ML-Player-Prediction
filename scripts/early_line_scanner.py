#!/usr/bin/env python3
"""Early-line value scanner and CLV (closing line value) tracker.

Compares the model's predictions against opening lines to flag value bets,
then tracks where lines close to measure closing line value -- the single
best indicator of whether the model has genuine edge.

Usage:
    # Morning: scan for early value
    python3 scripts/early_line_scanner.py --date 20260227

    # Evening: track where lines closed
    python3 scripts/early_line_scanner.py --date 20260227 --track

    # Weekly: check CLV performance
    python3 scripts/early_line_scanner.py --report

Cron setup (add to crontab -e):
    # Morning scan (9:30 AM ET / 14:30 UTC)
    30 14 * * * cd /Users/ryangoldstein/NBA && python3 scripts/early_line_scanner.py >> /tmp/nba_early_scan.log 2>&1
    # Evening track (7 PM ET / 00:00 UTC next day)
    0 0 * * * cd /Users/ryangoldstein/NBA && python3 scripts/early_line_scanner.py --track >> /tmp/nba_early_scan.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUT_DIR = PROJECT_ROOT / "analysis" / "output"
PREDICTIONS_DIR = OUT_DIR / "predictions"
MODEL_DIR = OUT_DIR / "models"
SIGNALS_DIR = PREDICTIONS_DIR  # early signals go here
CLV_TRACKING_PATH = PREDICTIONS_DIR / "clv_tracking.csv"

# Ensure scripts dir is on path for imports
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fetch_opening_lines import (
    fetch_scoreboard_snapshot,
    load_snapshots_for_date,
    save_snapshot,
    american_to_prob,
    SNAPSHOT_DIR,
)

# ---------------------------------------------------------------------------
# Configurable edge thresholds
# ---------------------------------------------------------------------------
DEFAULT_SPREAD_EDGE_THRESHOLD = 2.0    # points
DEFAULT_TOTAL_EDGE_THRESHOLD = 3.0     # points
DEFAULT_ML_EDGE_THRESHOLD = 0.05       # 5 percentage points


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Load model and generate predictions (reuse predict_upcoming_nba logic)
# ---------------------------------------------------------------------------

def load_models() -> dict[str, Any]:
    """Load persisted models from joblib."""
    import joblib
    models_path = MODEL_DIR / "advanced_models.joblib"
    if not models_path.exists():
        raise FileNotFoundError(
            f"No trained models found at {models_path}. "
            f"Run: cd scripts && python analyze_nba_2025_26_advanced.py"
        )
    return joblib.load(models_path)


def generate_predictions_for_date(target_date: str) -> pd.DataFrame:
    """Generate model predictions for a target date using predict_upcoming_nba machinery.

    Returns a DataFrame with columns: game_id, home_team, away_team, espn_event_id,
    home_win_prob, pred_home_margin, pred_total, model_spread (= -pred_home_margin expressed
    as a spread), model_ml_prob (= home_win_prob).
    """
    from predict_upcoming_nba import (
        build_training_history,
        regular_season_schedule_all,
        filter_upcoming,
        build_upcoming_team_states,
        build_upcoming_game_features,
        build_latest_player_form_lookup,
        attach_live_snapshots,
        fit_xgb_classifier,
        fit_xgb_regressor,
        fit_calibrated_classifier,
        fit_lgb_classifier,
        fit_correction_model,
        predict_with_model,
        predict_bayesian_blend,
        margin_consistent_win_prob,
        feature_intersection,
        load_tuned_params,
        _attach_injury_report_features,
        WIN_FEATURES_BASE,
        TOTAL_FEATURES_BASE,
        MARGIN_FEATURES_BASE,
        MARGIN_FEATURES_MARKET_RESIDUAL,
        TOTAL_FEATURES_MARKET_RESIDUAL,
    )
    from analyze_nba_2025_26_advanced import (
        join_espn_odds,
        SEASON,
    )

    tuned = load_tuned_params()
    win_params = tuned.get("win_params") if tuned else None
    total_params = tuned.get("total_params") if tuned else None
    margin_residual_params = tuned.get("margin_residual_params") if tuned else None
    total_residual_params = tuned.get("total_residual_params") if tuned else None
    feature_lists = (tuned or {}).get("feature_lists", {}) if tuned else {}

    win_base_features = feature_lists.get("enhanced_win", WIN_FEATURES_BASE)
    total_base_features = feature_lists.get("enhanced_total", TOTAL_FEATURES_BASE)
    market_margin_residual_features = feature_lists.get("market_margin_residual", MARGIN_FEATURES_MARKET_RESIDUAL)
    market_total_residual_features = feature_lists.get("market_total_residual", TOTAL_FEATURES_MARKET_RESIDUAL)

    print("Loading completed game history...", flush=True)
    team_games, player_games, games_hist = build_training_history()

    print("Loading schedule and upcoming games...", flush=True)
    schedule_all = regular_season_schedule_all()
    upcoming = filter_upcoming(schedule_all, target_date, 1, include_in_progress=False)
    if upcoming.empty:
        print(f"No upcoming games for {target_date}")
        return pd.DataFrame()

    print(f"Found {len(upcoming)} upcoming games for {target_date}", flush=True)
    upcoming_with_odds = join_espn_odds(upcoming)

    team_states = build_upcoming_team_states(upcoming, team_games)
    pred_df = build_upcoming_game_features(upcoming, team_states, upcoming_with_odds)
    player_form_lookup = build_latest_player_form_lookup(player_games)
    pred_df = attach_live_snapshots(pred_df, player_form_lookup)
    pred_df = _attach_injury_report_features(pred_df, player_games, target_date)

    # Sync features
    def _sync(name: str, feats: list[str]) -> list[str]:
        synced = feature_intersection(feats, games_hist, pred_df)
        if not synced:
            raise ValueError(f"No features for {name}")
        return synced

    win_base_features = _sync("win", win_base_features)
    total_base_features = _sync("total", total_base_features)
    margin_base_features = _sync("margin", MARGIN_FEATURES_BASE)
    market_margin_residual_features = _sync("margin_residual", market_margin_residual_features)
    market_total_residual_features = _sync("total_residual", market_total_residual_features)

    print("Training models on historical data...", flush=True)
    win_base_imp, win_base_model = fit_xgb_classifier(games_hist, win_base_features, "home_win", win_params)
    total_base_imp, total_base_model = fit_xgb_regressor(games_hist, total_base_features, "total_points", total_params)
    margin_base_imp, margin_base_model = fit_xgb_regressor(games_hist, margin_base_features, "home_margin", total_params)

    win_cal_imp, win_cal_model = fit_calibrated_classifier(games_hist, win_base_features, "home_win", win_params)
    win_lgb_imp, win_lgb_model = fit_lgb_classifier(games_hist, win_base_features, "home_win")

    correction = fit_correction_model(
        games_hist, win_base_imp, win_base_model, total_base_imp, total_base_model,
        win_base_features, total_base_features, shrinkage=1.0,
        margin_residual_features=market_margin_residual_features,
        total_residual_features=market_total_residual_features,
        margin_residual_params=margin_residual_params,
        total_residual_params=total_residual_params,
    )

    margin_preds_hist = predict_with_model(margin_base_imp, margin_base_model, games_hist, margin_base_features)
    residual_std = float(np.std(games_hist["home_margin"].values - margin_preds_hist))

    # Generate predictions
    print("Generating predictions...", flush=True)
    results = []
    for row in pred_df.itertuples(index=False):
        one = pd.DataFrame([row._asdict()])

        p_xgb = float(predict_with_model(win_base_imp, win_base_model, one, win_base_features, proba=True)[0])
        t_xgb = float(predict_with_model(total_base_imp, total_base_model, one, total_base_features)[0])
        p_cal = float(predict_with_model(win_cal_imp, win_cal_model, one, win_base_features, proba=True)[0])
        p_lgb = float(predict_with_model(win_lgb_imp, win_lgb_model, one, win_base_features, proba=True)[0])
        m_pred = float(predict_with_model(margin_base_imp, margin_base_model, one, margin_base_features)[0])
        p_margin = margin_consistent_win_prob(m_pred, residual_std)
        p_ensemble_base = float(np.mean([p_cal, p_lgb, p_margin]))

        has_market = (
            pd.notna(one["market_home_spread_close"].iloc[0])
            and pd.notna(one["market_total_close"].iloc[0])
            and pd.notna(one["market_home_implied_prob_close"].iloc[0])
        )

        if has_market and correction.get("fitted"):
            p_blend, t_blend, m_blend, method = predict_bayesian_blend(
                one, win_base_imp, win_base_model, total_base_imp, total_base_model,
                correction, win_base_features, total_base_features, margin_residual_std=residual_std,
            )
            p_final = float(np.clip(0.5 * p_ensemble_base + 0.5 * p_blend, 0.001, 0.999))
            t_final = float(t_blend)
            m_final = float(m_blend)
        elif has_market:
            p_market = float(one["market_home_implied_prob_close"].iloc[0])
            p_final = float(np.clip(0.6 * p_market + 0.4 * p_ensemble_base, 0.001, 0.999))
            t_market = float(one["market_total_close"].iloc[0])
            t_final = 0.6 * t_market + 0.4 * t_xgb
            m_market = float(-one["market_home_spread_close"].iloc[0])
            m_final = 0.6 * m_market + 0.4 * m_pred
        else:
            p_final = p_ensemble_base
            t_final = t_xgb
            m_final = m_pred

        results.append({
            "game_id": getattr(row, "game_id", ""),
            "espn_event_id": getattr(row, "espn_event_id", ""),
            "home_team": getattr(row, "home_team", ""),
            "away_team": getattr(row, "away_team", ""),
            "game_time_utc": getattr(row, "game_time_utc", ""),
            "home_win_prob": p_final,
            "pred_home_margin": m_final,
            "pred_total": t_final,
            "model_spread": -m_final,  # spread convention: negative = home favored
            "model_ml_prob": p_final,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Fetch opening lines from ESPN / snapshots
# ---------------------------------------------------------------------------

def fetch_current_lines(date_str: str) -> pd.DataFrame:
    """Fetch current ESPN lines for a date and return as DataFrame.

    Uses the fetch_opening_lines snapshot infrastructure.
    """
    snapshot = fetch_scoreboard_snapshot(date_str)

    rows = []
    for g in snapshot.get("games", []):
        odds_list = g.get("odds", [])
        if not odds_list:
            continue
        od = odds_list[0]  # primary book

        spread = od.get("spread")
        total = od.get("over_under")
        home_ml = od.get("home_moneyline")
        away_ml = od.get("away_moneyline")
        home_prob = od.get("home_implied_prob")
        away_prob = od.get("away_implied_prob")

        rows.append({
            "espn_event_id": g.get("espn_event_id", ""),
            "home_team": g.get("home_team", ""),
            "away_team": g.get("away_team", ""),
            "market_spread": spread,
            "market_total": total,
            "market_home_ml": home_ml,
            "market_away_ml": away_ml,
            "market_home_prob": home_prob,
            "market_away_prob": away_prob,
            "game_status": g.get("status", ""),
            "snapshot_time": snapshot.get("snapshot_time_utc", ""),
        })

    return pd.DataFrame(rows)


def get_opening_lines(date_str: str) -> pd.DataFrame:
    """Get the earliest available lines for a date.

    First checks stored snapshots, then falls back to live fetch.
    """
    snapshots = load_snapshots_for_date(date_str)
    if snapshots:
        # Use earliest snapshot as opening lines
        opening = snapshots[0]
        rows = []
        for g in opening.get("games", []):
            odds_list = g.get("odds", [])
            if not odds_list:
                continue
            od = odds_list[0]
            rows.append({
                "espn_event_id": g.get("espn_event_id", ""),
                "home_team": g.get("home_team", ""),
                "away_team": g.get("away_team", ""),
                "market_spread_open": od.get("spread"),
                "market_total_open": od.get("over_under"),
                "market_home_prob_open": od.get("home_implied_prob"),
                "opening_snapshot_time": opening.get("snapshot_time_utc", ""),
            })
        if rows:
            return pd.DataFrame(rows)

    # Fallback: fetch live (these are "current" lines, may be close to opening)
    current = fetch_current_lines(date_str)
    if current.empty:
        return pd.DataFrame()
    return current.rename(columns={
        "market_spread": "market_spread_open",
        "market_total": "market_total_open",
        "market_home_prob": "market_home_prob_open",
        "snapshot_time": "opening_snapshot_time",
    })


def get_closing_lines(date_str: str) -> pd.DataFrame:
    """Get the latest available lines for a date (closing or near-closing).

    Checks stored snapshots first, then fetches live.
    """
    snapshots = load_snapshots_for_date(date_str)
    if len(snapshots) >= 2:
        closing = snapshots[-1]
        rows = []
        for g in closing.get("games", []):
            odds_list = g.get("odds", [])
            if not odds_list:
                continue
            od = odds_list[0]
            rows.append({
                "espn_event_id": g.get("espn_event_id", ""),
                "home_team": g.get("home_team", ""),
                "away_team": g.get("away_team", ""),
                "market_spread_close": od.get("spread"),
                "market_total_close": od.get("over_under"),
                "market_home_prob_close": od.get("home_implied_prob"),
                "closing_snapshot_time": closing.get("snapshot_time_utc", ""),
            })
        if rows:
            return pd.DataFrame(rows)

    # Fallback: fetch live
    current = fetch_current_lines(date_str)
    if current.empty:
        return pd.DataFrame()
    return current.rename(columns={
        "market_spread": "market_spread_close",
        "market_total": "market_total_close",
        "market_home_prob": "market_home_prob_close",
        "snapshot_time": "closing_snapshot_time",
    })


# ---------------------------------------------------------------------------
# Core scanning logic
# ---------------------------------------------------------------------------

def compute_edges(
    predictions: pd.DataFrame,
    opening_lines: pd.DataFrame,
) -> pd.DataFrame:
    """Merge model predictions with opening lines and compute edges."""
    if predictions.empty or opening_lines.empty:
        return pd.DataFrame()

    # Merge on espn_event_id if available, else on home_team + away_team
    if "espn_event_id" in predictions.columns and "espn_event_id" in opening_lines.columns:
        merged = predictions.merge(opening_lines, on="espn_event_id", how="inner", suffixes=("", "_mkt"))
    else:
        merged = predictions.merge(
            opening_lines, on=["home_team", "away_team"], how="inner", suffixes=("", "_mkt")
        )

    if merged.empty:
        return pd.DataFrame()

    # Compute edges
    # Spread edge: model_spread vs market_spread_open
    # Convention: spread is from home perspective (negative = home favorite)
    merged["spread_edge"] = np.nan
    mask_spread = merged["market_spread_open"].notna() & merged["model_spread"].notna()
    merged.loc[mask_spread, "spread_edge"] = (
        merged.loc[mask_spread, "model_spread"] - merged.loc[mask_spread, "market_spread_open"]
    )
    # A negative spread_edge means model thinks home is MORE favored than market
    # For betting: we want |edge| > threshold

    # Total edge: model_total vs market_total_open
    merged["total_edge"] = np.nan
    mask_total = merged["market_total_open"].notna() & merged["pred_total"].notna()
    merged.loc[mask_total, "total_edge"] = (
        merged.loc[mask_total, "pred_total"] - merged.loc[mask_total, "market_total_open"]
    )

    # ML edge: model_ml_prob vs market_home_prob_open
    merged["ml_edge"] = np.nan
    mask_ml = merged["market_home_prob_open"].notna() & merged["model_ml_prob"].notna()
    merged.loc[mask_ml, "ml_edge"] = (
        merged.loc[mask_ml, "model_ml_prob"] - merged.loc[mask_ml, "market_home_prob_open"]
    )

    return merged


def flag_value_bets(
    edges_df: pd.DataFrame,
    spread_threshold: float = DEFAULT_SPREAD_EDGE_THRESHOLD,
    total_threshold: float = DEFAULT_TOTAL_EDGE_THRESHOLD,
    ml_threshold: float = DEFAULT_ML_EDGE_THRESHOLD,
) -> pd.DataFrame:
    """Flag games where edge exceeds configurable thresholds."""
    if edges_df.empty:
        return edges_df

    df = edges_df.copy()

    # Spread signal
    df["spread_signal"] = "NO BET"
    # If model spread is more negative than market (model sees home stronger),
    # bet HOME ATS. If model spread is more positive, bet AWAY ATS.
    spread_edge_abs = df["spread_edge"].abs()
    mask_home_ats = (df["spread_edge"] < -spread_threshold)
    mask_away_ats = (df["spread_edge"] > spread_threshold)
    df.loc[mask_home_ats, "spread_signal"] = "HOME ATS"
    df.loc[mask_away_ats, "spread_signal"] = "AWAY ATS"

    # Total signal
    df["total_signal"] = "NO BET"
    mask_over = (df["total_edge"] > total_threshold)
    mask_under = (df["total_edge"] < -total_threshold)
    df.loc[mask_over, "total_signal"] = "OVER"
    df.loc[mask_under, "total_signal"] = "UNDER"

    # ML signal
    df["ml_signal"] = "NO BET"
    mask_home_ml = (df["ml_edge"] > ml_threshold)
    mask_away_ml = (df["ml_edge"] < -ml_threshold)
    df.loc[mask_home_ml, "ml_signal"] = "HOME ML"
    df.loc[mask_away_ml, "ml_signal"] = "AWAY ML"

    # Overall: has at least one signal
    df["has_signal"] = (
        (df["spread_signal"] != "NO BET")
        | (df["total_signal"] != "NO BET")
        | (df["ml_signal"] != "NO BET")
    )

    return df


# ---------------------------------------------------------------------------
# CLV tracking
# ---------------------------------------------------------------------------

def track_clv(date_str: str) -> pd.DataFrame:
    """Track closing line value for a date's signals.

    Loads the morning signals file, fetches closing lines, and computes
    whether lines moved in the model's direction (CLV).
    """
    signals_path = SIGNALS_DIR / f"early_signals_{date_str}.csv"
    if not signals_path.exists():
        print(f"No morning signals file found at {signals_path}", flush=True)
        return pd.DataFrame()

    signals = pd.read_csv(signals_path)
    if signals.empty:
        print("Morning signals file is empty.", flush=True)
        return pd.DataFrame()

    # Take a new snapshot for closing lines
    print(f"Taking closing-line snapshot for {date_str}...", flush=True)
    try:
        snapshot = fetch_scoreboard_snapshot(date_str)
        save_snapshot(snapshot, date_str)
        print(f"  Snapshot saved ({snapshot['games_count']} games)", flush=True)
    except Exception as exc:
        print(f"  Warning: could not save snapshot: {exc}", flush=True)

    closing = get_closing_lines(date_str)
    if closing.empty:
        print("Could not fetch closing lines.", flush=True)
        return pd.DataFrame()

    # Merge signals with closing lines
    merge_key = "espn_event_id" if "espn_event_id" in signals.columns and "espn_event_id" in closing.columns else None
    if merge_key:
        # Ensure same type
        signals[merge_key] = signals[merge_key].astype(str)
        closing[merge_key] = closing[merge_key].astype(str)
        merged = signals.merge(closing, on=merge_key, how="left", suffixes=("", "_close"))
    else:
        merged = signals.merge(
            closing, on=["home_team", "away_team"], how="left", suffixes=("", "_close")
        )

    if merged.empty:
        return pd.DataFrame()

    results = []
    for _, row in merged.iterrows():
        r = {
            "date": date_str,
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "espn_event_id": str(row.get("espn_event_id", "")),
        }

        # Spread CLV: did the closing line move toward model's position?
        spread_open = row.get("market_spread_open")
        spread_close = row.get("market_spread_close")
        model_spread = row.get("model_spread")
        spread_signal = row.get("spread_signal", "NO BET")

        r["spread_signal"] = spread_signal
        r["model_spread"] = model_spread
        r["market_spread_open"] = spread_open
        r["market_spread_close"] = spread_close

        if pd.notna(spread_open) and pd.notna(spread_close) and pd.notna(model_spread):
            spread_move = spread_close - spread_open
            # If model had home more favored (model_spread < market_spread_open),
            # then line moving down (spread_close < spread_open) = moved our way
            model_direction = model_spread - spread_open
            r["spread_line_move"] = spread_move
            r["spread_clv_moved_our_way"] = int(
                (model_direction < 0 and spread_move < 0)
                or (model_direction > 0 and spread_move > 0)
            ) if abs(model_direction) > 0.5 else np.nan
            r["spread_clv_points"] = abs(spread_move) if r.get("spread_clv_moved_our_way") == 1 else -abs(spread_move)
        else:
            r["spread_line_move"] = np.nan
            r["spread_clv_moved_our_way"] = np.nan
            r["spread_clv_points"] = np.nan

        # Total CLV
        total_open = row.get("market_total_open")
        total_close = row.get("market_total_close")
        model_total = row.get("pred_total")
        total_signal = row.get("total_signal", "NO BET")

        r["total_signal"] = total_signal
        r["model_total"] = model_total
        r["market_total_open"] = total_open
        r["market_total_close"] = total_close

        if pd.notna(total_open) and pd.notna(total_close) and pd.notna(model_total):
            total_move = total_close - total_open
            model_direction_total = model_total - total_open
            r["total_line_move"] = total_move
            r["total_clv_moved_our_way"] = int(
                (model_direction_total > 0 and total_move > 0)
                or (model_direction_total < 0 and total_move < 0)
            ) if abs(model_direction_total) > 0.5 else np.nan
            r["total_clv_points"] = abs(total_move) if r.get("total_clv_moved_our_way") == 1 else -abs(total_move)
        else:
            r["total_line_move"] = np.nan
            r["total_clv_moved_our_way"] = np.nan
            r["total_clv_points"] = np.nan

        # ML CLV
        ml_open = row.get("market_home_prob_open")
        ml_close = row.get("market_home_prob_close")
        model_ml = row.get("model_ml_prob")
        ml_signal = row.get("ml_signal", "NO BET")

        r["ml_signal"] = ml_signal
        r["model_ml_prob"] = model_ml
        r["market_home_prob_open"] = ml_open
        r["market_home_prob_close"] = ml_close

        if pd.notna(ml_open) and pd.notna(ml_close) and pd.notna(model_ml):
            ml_move = ml_close - ml_open
            model_direction_ml = model_ml - ml_open
            r["ml_line_move"] = ml_move
            r["ml_clv_moved_our_way"] = int(
                (model_direction_ml > 0 and ml_move > 0)
                or (model_direction_ml < 0 and ml_move < 0)
            ) if abs(model_direction_ml) > 0.02 else np.nan
        else:
            r["ml_line_move"] = np.nan
            r["ml_clv_moved_our_way"] = np.nan

        results.append(r)

    tracking_df = pd.DataFrame(results)

    # Append to cumulative CLV tracking file
    if CLV_TRACKING_PATH.exists():
        existing = pd.read_csv(CLV_TRACKING_PATH)
        # Remove any existing rows for this date to avoid duplicates
        existing = existing[existing["date"] != date_str]
        tracking_df = pd.concat([existing, tracking_df], ignore_index=True)

    tracking_df.to_csv(CLV_TRACKING_PATH, index=False)
    print(f"CLV tracking data saved to {CLV_TRACKING_PATH}", flush=True)

    return tracking_df[tracking_df["date"] == date_str]


# ---------------------------------------------------------------------------
# CLV report
# ---------------------------------------------------------------------------

def generate_clv_report() -> None:
    """Generate a CLV summary report from all historical tracking data."""
    if not CLV_TRACKING_PATH.exists():
        print("No CLV tracking data found. Run --track first.", flush=True)
        return

    df = pd.read_csv(CLV_TRACKING_PATH)
    if df.empty:
        print("CLV tracking file is empty.", flush=True)
        return

    print("\n" + "=" * 72)
    print("  CLOSING LINE VALUE (CLV) REPORT")
    print("=" * 72)
    print(f"\n  Data covers: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total games tracked: {len(df)}")
    n_dates = df["date"].nunique()
    print(f"  Unique dates: {n_dates}")

    # --- Spread CLV ---
    spread_games = df[df["spread_signal"] != "NO BET"].copy()
    spread_with_clv = spread_games.dropna(subset=["spread_clv_moved_our_way"])
    print(f"\n  --- Spread CLV ---")
    print(f"  Games with spread signals: {len(spread_games)}")
    if not spread_with_clv.empty:
        n_moved = int(spread_with_clv["spread_clv_moved_our_way"].sum())
        n_total = len(spread_with_clv)
        clv_rate = n_moved / n_total * 100
        avg_clv_pts = spread_with_clv["spread_clv_points"].mean()
        print(f"  CLV rate (spread): {n_moved}/{n_total} = {clv_rate:.1f}%")
        print(f"  Average CLV (points): {avg_clv_pts:+.2f}")
        if clv_rate > 55:
            print(f"  Assessment: POSITIVE -- beating CLV on spreads")
        elif clv_rate > 50:
            print(f"  Assessment: MARGINAL -- near breakeven on CLV")
        else:
            print(f"  Assessment: NEGATIVE -- not beating CLV on spreads")
    else:
        print("  No spread CLV data available.")

    # --- Total CLV ---
    total_games = df[df["total_signal"] != "NO BET"].copy()
    total_with_clv = total_games.dropna(subset=["total_clv_moved_our_way"])
    print(f"\n  --- Total CLV ---")
    print(f"  Games with total signals: {len(total_games)}")
    if not total_with_clv.empty:
        n_moved = int(total_with_clv["total_clv_moved_our_way"].sum())
        n_total = len(total_with_clv)
        clv_rate = n_moved / n_total * 100
        avg_clv_pts = total_with_clv["total_clv_points"].mean()
        print(f"  CLV rate (total): {n_moved}/{n_total} = {clv_rate:.1f}%")
        print(f"  Average CLV (points): {avg_clv_pts:+.2f}")
        if clv_rate > 55:
            print(f"  Assessment: POSITIVE -- beating CLV on totals")
        elif clv_rate > 50:
            print(f"  Assessment: MARGINAL -- near breakeven on CLV")
        else:
            print(f"  Assessment: NEGATIVE -- not beating CLV on totals")
    else:
        print("  No total CLV data available.")

    # --- ML CLV ---
    ml_games = df[df["ml_signal"] != "NO BET"].copy()
    ml_with_clv = ml_games.dropna(subset=["ml_clv_moved_our_way"])
    print(f"\n  --- Moneyline CLV ---")
    print(f"  Games with ML signals: {len(ml_games)}")
    if not ml_with_clv.empty:
        n_moved = int(ml_with_clv["ml_clv_moved_our_way"].sum())
        n_total = len(ml_with_clv)
        clv_rate = n_moved / n_total * 100
        print(f"  CLV rate (ML): {n_moved}/{n_total} = {clv_rate:.1f}%")
        if clv_rate > 55:
            print(f"  Assessment: POSITIVE -- beating CLV on moneyline")
        elif clv_rate > 50:
            print(f"  Assessment: MARGINAL -- near breakeven on CLV")
        else:
            print(f"  Assessment: NEGATIVE -- not beating CLV on moneyline")
    else:
        print("  No ML CLV data available.")

    # --- Overall CLV ---
    all_clv_cols = ["spread_clv_moved_our_way", "total_clv_moved_our_way", "ml_clv_moved_our_way"]
    all_clv = []
    for col in all_clv_cols:
        if col in df.columns:
            vals = df[col].dropna()
            all_clv.extend(vals.tolist())

    if all_clv:
        overall_clv_rate = sum(all_clv) / len(all_clv) * 100
        print(f"\n  --- Overall CLV (all bet types combined) ---")
        print(f"  Total signals tracked: {len(all_clv)}")
        print(f"  Overall CLV rate: {overall_clv_rate:.1f}%")
        if overall_clv_rate > 55:
            print(f"\n  >>> GO-LIVE SIGNAL: Model is consistently beating CLV <<<")
        elif overall_clv_rate > 50:
            print(f"\n  >>> MARGINAL: Model is near-breakeven on CLV. More data needed. <<<")
        else:
            print(f"\n  >>> CAUTION: Model is not beating CLV. Reassess before live betting. <<<")

    # --- Per-date breakdown ---
    print(f"\n  --- Per-Date Summary ---")
    for date, grp in df.groupby("date"):
        signals = grp[
            (grp["spread_signal"] != "NO BET")
            | (grp["total_signal"] != "NO BET")
            | (grp["ml_signal"] != "NO BET")
        ]
        n_sig = len(signals)
        clv_vals = []
        for col in all_clv_cols:
            if col in grp.columns:
                clv_vals.extend(grp[col].dropna().tolist())
        if clv_vals:
            rate = sum(clv_vals) / len(clv_vals) * 100
            print(f"    {date}: {n_sig} signals, CLV rate = {rate:.0f}% ({int(sum(clv_vals))}/{len(clv_vals)})")
        else:
            print(f"    {date}: {n_sig} signals, no CLV data")

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Early line scanner: compare model predictions to opening lines and track CLV"
    )
    p.add_argument("--date", type=str, default=None,
                   help="Target date YYYYMMDD (default: today)")
    p.add_argument("--track", action="store_true",
                   help="Track closing lines and compute CLV for the date's signals")
    p.add_argument("--report", action="store_true",
                   help="Generate CLV summary report from all historical tracking data")
    p.add_argument("--spread-threshold", type=float, default=DEFAULT_SPREAD_EDGE_THRESHOLD,
                   help=f"Min spread edge (points) to flag (default: {DEFAULT_SPREAD_EDGE_THRESHOLD})")
    p.add_argument("--total-threshold", type=float, default=DEFAULT_TOTAL_EDGE_THRESHOLD,
                   help=f"Min total edge (points) to flag (default: {DEFAULT_TOTAL_EDGE_THRESHOLD})")
    p.add_argument("--ml-threshold", type=float, default=DEFAULT_ML_EDGE_THRESHOLD,
                   help=f"Min ML edge (probability) to flag (default: {DEFAULT_ML_EDGE_THRESHOLD})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target_date = args.date or datetime.now().strftime("%Y%m%d")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Report mode ---
    if args.report:
        generate_clv_report()
        return

    # --- Track mode ---
    if args.track:
        print(f"Tracking CLV for {target_date}...", flush=True)
        tracking = track_clv(target_date)
        if tracking.empty:
            print("No tracking results generated.", flush=True)
            return

        # Print summary
        n_games = len(tracking)
        for bet_type in ["spread", "total", "ml"]:
            sig_col = f"{bet_type}_signal"
            clv_col = f"{bet_type}_clv_moved_our_way"
            if sig_col not in tracking.columns or clv_col not in tracking.columns:
                continue
            signaled = tracking[tracking[sig_col] != "NO BET"]
            if signaled.empty:
                continue
            with_clv = signaled.dropna(subset=[clv_col])
            if with_clv.empty:
                continue
            n_moved = int(with_clv[clv_col].sum())
            n = len(with_clv)
            print(f"\n  {bet_type.upper()} CLV: {n_moved}/{n} signals saw line move our way ({n_moved/n*100:.0f}%)")

        print(f"\nTracking complete. Results in {CLV_TRACKING_PATH}")
        return

    # --- Scan mode (default) ---
    print(f"\n{'=' * 72}")
    print(f"  EARLY LINE SCANNER - {target_date}")
    print(f"{'=' * 72}\n")

    # Step 1: Take a snapshot of current lines
    print("Taking odds snapshot...", flush=True)
    try:
        snapshot = fetch_scoreboard_snapshot(target_date)
        filepath = save_snapshot(snapshot, target_date)
        print(f"  Saved snapshot: {filepath.name} ({snapshot['games_count']} games)", flush=True)
    except Exception as exc:
        print(f"  Warning: could not save snapshot: {exc}", flush=True)

    # Step 2: Get opening lines
    print("Fetching opening lines...", flush=True)
    opening_lines = get_opening_lines(target_date)
    if opening_lines.empty:
        print("No opening lines available. Exiting.", flush=True)
        return
    print(f"  Opening lines for {len(opening_lines)} games", flush=True)

    # Step 3: Generate model predictions
    print("\nGenerating model predictions...", flush=True)
    predictions = generate_predictions_for_date(target_date)
    if predictions.empty:
        print("No predictions generated. Exiting.", flush=True)
        return

    # Step 4: Compute edges
    print("\nComputing edges...", flush=True)
    edges = compute_edges(predictions, opening_lines)
    if edges.empty:
        print("No edges computed (no game matches). Exiting.", flush=True)
        return

    # Step 5: Flag value bets
    flagged = flag_value_bets(
        edges,
        spread_threshold=args.spread_threshold,
        total_threshold=args.total_threshold,
        ml_threshold=args.ml_threshold,
    )

    # Step 6: Save signals
    out_path = SIGNALS_DIR / f"early_signals_{target_date}.csv"
    flagged.to_csv(out_path, index=False)
    print(f"\nSignals saved to {out_path}", flush=True)

    # Step 7: Print summary
    n_total = len(flagged)
    n_with_signal = int(flagged["has_signal"].sum()) if "has_signal" in flagged.columns else 0

    print(f"\n{'=' * 72}")
    print(f"  EARLY SIGNALS SUMMARY - {target_date}")
    print(f"{'=' * 72}")
    print(f"\n  Games analyzed: {n_total}")
    print(f"  Games with value signals: {n_with_signal}")
    print(f"  Edge thresholds: spread={args.spread_threshold}pts, total={args.total_threshold}pts, ML={args.ml_threshold*100:.0f}%")

    for _, row in flagged.iterrows():
        away = row.get("away_team", "?")
        home = row.get("home_team", "?")
        has_any = row.get("has_signal", False)
        marker = " ***" if has_any else ""

        print(f"\n  {away} @ {home}{marker}")

        # Spread
        spread_open = row.get("market_spread_open")
        model_spread = row.get("model_spread")
        spread_edge = row.get("spread_edge")
        spread_sig = row.get("spread_signal", "NO BET")
        if pd.notna(spread_open) and pd.notna(model_spread):
            print(f"    Spread:  Market={spread_open:+.1f}  Model={model_spread:+.1f}  Edge={spread_edge:+.1f}  -> {spread_sig}")
        else:
            print(f"    Spread:  No data")

        # Total
        total_open = row.get("market_total_open")
        model_total = row.get("pred_total")
        total_edge = row.get("total_edge")
        total_sig = row.get("total_signal", "NO BET")
        if pd.notna(total_open) and pd.notna(model_total):
            print(f"    Total:   Market={total_open:.1f}  Model={model_total:.1f}  Edge={total_edge:+.1f}  -> {total_sig}")
        else:
            print(f"    Total:   No data")

        # ML
        ml_open = row.get("market_home_prob_open")
        model_ml = row.get("model_ml_prob")
        ml_edge = row.get("ml_edge")
        ml_sig = row.get("ml_signal", "NO BET")
        if pd.notna(ml_open) and pd.notna(model_ml):
            print(f"    ML:      Market={ml_open*100:.1f}%  Model={model_ml*100:.1f}%  Edge={ml_edge*100:+.1f}%  -> {ml_sig}")
        else:
            print(f"    ML:      No data")

    if n_with_signal > 0:
        print(f"\n  {'=' * 40}")
        print(f"  VALUE BETS ({n_with_signal} games):")
        print(f"  {'=' * 40}")
        value_games = flagged[flagged["has_signal"] == True]
        for _, row in value_games.iterrows():
            away = row.get("away_team", "?")
            home = row.get("home_team", "?")
            signals = []
            if row.get("spread_signal", "NO BET") != "NO BET":
                signals.append(f"{row['spread_signal']} (edge={row['spread_edge']:+.1f}pts)")
            if row.get("total_signal", "NO BET") != "NO BET":
                signals.append(f"{row['total_signal']} (edge={row['total_edge']:+.1f}pts)")
            if row.get("ml_signal", "NO BET") != "NO BET":
                signals.append(f"{row['ml_signal']} (edge={row['ml_edge']*100:+.1f}%)")
            print(f"    {away} @ {home}: {', '.join(signals)}")

    print(f"\n{'=' * 72}\n")


if __name__ == "__main__":
    main()
