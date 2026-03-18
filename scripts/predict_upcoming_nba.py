from __future__ import annotations

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb

from analyze_nba_2025_26_advanced import (
    BOXSCORE_URL_TMPL,
    INJURY_REPORT_CACHE,
    SCHEDULE_URL,
    SEASON,
    SEASONS,
    TEAM_COORDS,
    build_game_level,
    build_team_games_and_players,
    compute_elo_ratings,
    compute_injury_report_features,
    fetch_espn_injury_report,
    fetch_json,
    haversine_miles,
    join_espn_odds,
    load_odds_snapshots,
    match_injury_report_to_players,
    normalize_espn_abbr,
    time_series_cv_folds,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis" / "output"
PREDICTIONS_DIR = OUT_DIR / "predictions"
MODEL_DIR = OUT_DIR / "models"


WIN_FEATURES_BASE = [
    # Base differentials
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
    # Official injury report features (NaN when unavailable)
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

WIN_FEATURES_MARKET = WIN_FEATURES_BASE + [
    "market_home_spread_close",
    "market_total_close",
    "market_home_implied_prob_close",
    "market_spread_move_home",
    "market_total_move",
    "market_home_implied_prob_open",
    "market_ml_move",
    "market_spread_move_abs",
]

TOTAL_FEATURES_BASE = [
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
    # Official injury report features (NaN when unavailable)
    "home_injury_report_missing_minutes",
    "away_injury_report_missing_minutes",
    "home_injury_report_missing_impact",
    "away_injury_report_missing_impact",
    "home_injury_report_count_out",
    "away_injury_report_count_out",
    "home_injury_report_total_risk",
    "away_injury_report_total_risk",
    # Referee crew features (Item 1)
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

TOTAL_FEATURES_MARKET = TOTAL_FEATURES_BASE + [
    "market_total_close",
    "market_home_spread_close",
    "market_total_move",
    "market_total_move_abs",
]

MARGIN_FEATURES_BASE = WIN_FEATURES_BASE + [
    "home_pre_possessions_avg5",
    "away_pre_possessions_avg5",
    "home_pre_off_rating_avg5",
    "away_pre_off_rating_avg5",
    "home_pre_def_rating_avg5",
    "away_pre_def_rating_avg5",
]

MARGIN_FEATURES_MARKET = MARGIN_FEATURES_BASE + [
    "market_home_spread_close",
    "market_total_close",
    "market_spread_move_home",
    "market_spread_move_abs",
    "market_ml_move",
]

# Residual market-correction feature families (aligned to analysis artifact feature_lists when available).
MARGIN_FEATURES_MARKET_RESIDUAL = MARGIN_FEATURES_MARKET.copy()
TOTAL_FEATURES_MARKET_RESIDUAL = TOTAL_FEATURES_BASE + [
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

ROLLING_METRICS = [
    "net_rating",
    "off_rating",
    "def_rating",
    "efg",
    "tov_rate",
    "orb_rate",
    "ft_rate",
    "possessions",
    "margin",
]


def _overlay_prediction_snapshots(games: pd.DataFrame) -> pd.DataFrame:
    """Overlay snapshot-based line movement for upcoming games.

    If we have odds snapshots for the game dates, use them to enrich
    the open/close line data. For upcoming games where ESPN's close
    data falls back to current lines, snapshots provide time-series
    movement from earlier captures.
    """
    if "game_date_est" not in games.columns or "espn_event_id" not in games.columns:
        return games

    dates = games["game_date_est"].dropna().unique().tolist()
    all_snapshots: dict[str, dict[str, Any]] = {}
    for d in dates:
        date_movements = load_odds_snapshots(str(d))
        all_snapshots.update(date_movements)

    if not all_snapshots:
        return games

    snap_rows = []
    for eid, data in all_snapshots.items():
        row = {"espn_event_id": eid}
        row.update(data)
        snap_rows.append(row)
    snap_df = pd.DataFrame(snap_rows)
    if snap_df.empty:
        return games

    games = games.merge(snap_df, on="espn_event_id", how="left", suffixes=("", "_snap"))

    # Use snapshot spread/total as fallback for open values when ESPN data is NaN
    if "snapshot_spread_open" in games.columns:
        mask = games["market_home_spread_open"].isna() & games["snapshot_spread_open"].notna()
        games.loc[mask, "market_home_spread_open"] = games.loc[mask, "snapshot_spread_open"]

    if "snapshot_total_open" in games.columns:
        mask = games["market_total_open"].isna() & games["snapshot_total_open"].notna()
        games.loc[mask, "market_total_open"] = games.loc[mask, "snapshot_total_open"]

    # Recompute movement features
    games["market_spread_move_home"] = games["market_home_spread_close"] - games["market_home_spread_open"]
    games["market_total_move"] = games["market_total_close"] - games["market_total_open"]
    games["market_spread_move_abs"] = games["market_spread_move_home"].abs()
    games["market_total_move_abs"] = games["market_total_move"].abs()
    _hs_open = games["market_home_spread_open"]
    games["market_home_implied_prob_open"] = 1.0 / (1.0 + np.exp(_hs_open / 6.5))
    games["market_ml_move"] = games["market_home_implied_prob_close"] - games["market_home_implied_prob_open"]

    return games


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def optimal_allocation(ev: float, cap: float = 0.10) -> float:
    """Convert post-vig EV to a capped Allocation fraction."""
    if not np.isfinite(ev) or ev <= 0:
        return 0.0
    return clamp(float(ev) / VIG_FACTOR, 0.0, cap)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


VIG_FACTOR = 0.9524        # net payout per $1 at ~-105 juice (matches nba_evaluate default)
BREAKEVEN_PROB = 1.0 / (1.0 + VIG_FACTOR)  # ~0.5122
MIN_EDGE = 0.03            # 3% minimum edge for bet signals


def parse_iso_clock_to_seconds(clock: Any) -> float:
    if clock is None or clock == "":
        return 0.0
    s = str(clock)
    if s.startswith("PT"):
        s = s[2:]
    minutes = 0.0
    seconds = 0.0
    if "M" in s:
        m, s = s.split("M", 1)
        minutes = float(m or 0)
    if "S" in s:
        sec = s.replace("S", "")
        seconds = float(sec or 0)
    return minutes * 60.0 + seconds


def current_period_length_seconds(period_num: int, regulation_periods: int = 4) -> int:
    return 12 * 60 if period_num <= regulation_periods else 5 * 60


def live_time_features(period_num: int, game_clock: Any, regulation_periods: int = 4) -> dict[str, float]:
    period_num = int(period_num or 1)
    clock_sec = parse_iso_clock_to_seconds(game_clock)
    cur_period_len = current_period_length_seconds(period_num, regulation_periods)
    elapsed_before = 0
    if period_num > 1:
        reg_complete = min(period_num - 1, regulation_periods)
        ot_complete = max((period_num - 1) - regulation_periods, 0)
        elapsed_before = reg_complete * 12 * 60 + ot_complete * 5 * 60
    elapsed_sec = elapsed_before + max(0.0, cur_period_len - clock_sec)
    if period_num <= regulation_periods:
        rem_reg_sec = (regulation_periods - period_num) * 12 * 60 + clock_sec
    else:
        rem_reg_sec = clock_sec
    return {
        "live_clock_seconds_remaining_period": float(clock_sec),
        "live_elapsed_seconds": float(elapsed_sec),
        "live_elapsed_minutes": float(elapsed_sec / 60.0),
        "live_remaining_seconds_est": float(rem_reg_sec),
        "live_remaining_minutes_est": float(rem_reg_sec / 60.0),
        "live_remaining_regulation_fraction": float(rem_reg_sec / (48 * 60)),
    }


def build_latest_player_form_lookup(player_games: pd.DataFrame) -> pd.DataFrame:
    pg = player_games.copy().sort_values(["team", "player_id", "game_time_utc", "game_id"])
    # Use only completed historical games. Compute recent means from each player's latest completed appearances.
    latest_rows: list[dict[str, Any]] = []
    for (team, player_id), grp in pg.groupby(["team", "player_id"], sort=False):
        g = grp.tail(5)
        if g.empty:
            continue
        latest_rows.append(
            {
                "team": team,
                "player_id": int(player_id),
                "player_recent_minutes5": float(g["minutes"].mean()),
                "player_recent_points5": float(g["points"].mean()),
                "player_recent_assists5": float(g["assists"].mean()),
                "player_recent_rebounds5": float(g["rebounds"].mean()),
                "player_recent_played_rate5": float(g["played"].mean()),
                "player_recent_starter_rate5": float(g["starter"].mean()),
            }
        )
    return pd.DataFrame(latest_rows)


def fetch_live_boxscore_snapshot(game_id: str) -> dict[str, Any]:
    return fetch_json(BOXSCORE_URL_TMPL.format(game_id=game_id))


def parse_live_game_snapshot(game_id: str, player_form_lookup: pd.DataFrame) -> dict[str, Any]:
    payload = fetch_live_boxscore_snapshot(game_id)
    game = payload["game"]
    regulation_periods = int(game.get("regulationPeriods") or 4)
    period_num = int(game.get("period") or 1)
    game_clock = game.get("gameClock")
    home = game["homeTeam"]
    away = game["awayTeam"]

    def team_live_features(team_obj: dict[str, Any], prefix: str) -> dict[str, Any]:
        team_code = team_obj.get("teamTricode")
        rows = []
        for p in team_obj.get("players", []):
            pid = p.get("personId")
            if pid is None:
                continue
            rows.append(
                {
                    "team": team_code,
                    "player_id": int(pid),
                    "played_live": int(str(p.get("played", "0")) == "1"),
                    "oncourt_live": int(str(p.get("oncourt", "0")) == "1"),
                    "starter_live": int(str(p.get("starter", "0")) == "1"),
                    "status_text": p.get("status"),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return {
                f"{prefix}_live_roster_listed_count": 0,
                f"{prefix}_live_played_count": 0,
                f"{prefix}_live_oncourt_count": 0,
                f"{prefix}_live_oncourt_recent_minutes5_sum": np.nan,
                f"{prefix}_live_oncourt_recent_points5_sum": np.nan,
                f"{prefix}_live_played_recent_minutes5_sum": np.nan,
                f"{prefix}_live_played_recent_points5_sum": np.nan,
                f"{prefix}_live_inactive_listed_count": 0,
            }
        if not player_form_lookup.empty:
            df = df.merge(player_form_lookup, on=["team", "player_id"], how="left")
        for c in [
            "player_recent_minutes5",
            "player_recent_points5",
            "player_recent_assists5",
            "player_recent_rebounds5",
            "player_recent_played_rate5",
            "player_recent_starter_rate5",
        ]:
            if c not in df.columns:
                df[c] = np.nan
        inactive_mask = (df["played_live"] == 0) & (df["status_text"].astype(str).str.lower().isin(["inactive", "out"]))
        return {
            f"{prefix}_live_roster_listed_count": int(len(df)),
            f"{prefix}_live_played_count": int(df["played_live"].sum()),
            f"{prefix}_live_oncourt_count": int(df["oncourt_live"].sum()),
            f"{prefix}_live_oncourt_recent_minutes5_sum": float(df.loc[df["oncourt_live"] == 1, "player_recent_minutes5"].sum()),
            f"{prefix}_live_oncourt_recent_points5_sum": float(df.loc[df["oncourt_live"] == 1, "player_recent_points5"].sum()),
            f"{prefix}_live_played_recent_minutes5_sum": float(df.loc[df["played_live"] == 1, "player_recent_minutes5"].sum()),
            f"{prefix}_live_played_recent_points5_sum": float(df.loc[df["played_live"] == 1, "player_recent_points5"].sum()),
            f"{prefix}_live_inactive_listed_count": int(inactive_mask.sum()),
        }

    features = {
        "game_id": str(game_id),
        "live_game_status": int(game.get("gameStatus") or 0),
        "live_game_status_text": game.get("gameStatusText"),
        "live_period": period_num,
        "live_game_clock": str(game_clock),
        "live_home_score_current": float(home.get("score") or 0),
        "live_away_score_current": float(away.get("score") or 0),
    }
    features.update(live_time_features(period_num, game_clock, regulation_periods))
    features.update(team_live_features(home, "home"))
    features.update(team_live_features(away, "away"))
    features["live_home_margin_current"] = features["live_home_score_current"] - features["live_away_score_current"]
    features["live_total_current"] = features["live_home_score_current"] + features["live_away_score_current"]
    features["live_oncourt_recent_points5_diff"] = (
        features.get("home_live_oncourt_recent_points5_sum", np.nan)
        - features.get("away_live_oncourt_recent_points5_sum", np.nan)
    )
    features["live_oncourt_recent_minutes5_diff"] = (
        features.get("home_live_oncourt_recent_minutes5_sum", np.nan)
        - features.get("away_live_oncourt_recent_minutes5_sum", np.nan)
    )
    return features


def attach_live_snapshots(pred_df: pd.DataFrame, player_form_lookup: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    live_rows = out[out["game_status"] == 2]
    if live_rows.empty:
        out["prediction_mode"] = "pregame"
        return out
    snapshots = []
    for gid in live_rows["game_id"].astype(str).tolist():
        try:
            snapshots.append(parse_live_game_snapshot(gid, player_form_lookup))
        except Exception as exc:
            snapshots.append({"game_id": str(gid), "live_snapshot_error": str(exc)})
    snap_df = pd.DataFrame(snapshots)
    out = out.merge(snap_df, on="game_id", how="left")
    out["prediction_mode"] = np.where(out["game_status"] == 2, "live", "pregame")
    return out


def apply_live_adjustments(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["pregame_home_win_prob"] = out["home_win_prob"]
    out["pregame_pred_total"] = out["pred_total"]
    out["pregame_pred_home_margin"] = out["pred_home_margin"]
    out["pregame_pred_home_score"] = out["pred_home_score"]
    out["pregame_pred_away_score"] = out["pred_away_score"]

    live_home_score = out.get("live_home_score_current", pd.Series(np.nan, index=out.index))
    live_mask = out["prediction_mode"].eq("live") & live_home_score.notna()
    if not live_mask.any():
        return out

    for idx, row in out.loc[live_mask].iterrows():
        p0 = float(row["pregame_home_win_prob"])
        pre_margin = float(row["pregame_pred_home_margin"])
        pre_total = float(row["pregame_pred_total"])
        cur_margin = float(row["live_home_margin_current"])
        cur_total = float(row["live_total_current"])
        elapsed_min = float(row.get("live_elapsed_minutes", np.nan) or 0.0)
        rem_min = float(row.get("live_remaining_minutes_est", np.nan) or 0.0)
        rem_frac = clamp(float(row.get("live_remaining_regulation_fraction", np.nan) or (rem_min / 48.0)), 0.0, 1.0)

        # Active/on-court players influence only as a small heuristic tie-breaker.
        oncourt_pts_diff = float(row.get("live_oncourt_recent_points5_diff", 0.0) or 0.0)
        oncourt_min_diff = float(row.get("live_oncourt_recent_minutes5_diff", 0.0) or 0.0)
        oncourt_adj_margin = (0.06 * oncourt_pts_diff + 0.01 * oncourt_min_diff) * clamp(rem_frac, 0.1, 1.0)

        expected_future_margin = pre_margin * rem_frac + oncourt_adj_margin
        expected_final_margin = cur_margin + expected_future_margin
        sd_remaining = 1.8 + 12.0 * math.sqrt(max(rem_frac, 0.02))
        p_live = clamp(normal_cdf(expected_final_margin / sd_remaining), 0.001, 0.999)

        # Blend with pregame probability very early in the game to avoid overreacting.
        blend_live = clamp(elapsed_min / 10.0, 0.0, 1.0)
        p_adj = clamp((1 - blend_live) * p0 + blend_live * p_live, 0.001, 0.999)

        # Total projection: blend pregame total with current pace projection.
        if elapsed_min > 0.1:
            pace_proj_total = cur_total * (48.0 / max(elapsed_min, 1.0))
        else:
            pace_proj_total = pre_total
        total_blend = clamp(elapsed_min / 20.0, 0.0, 0.9)
        pred_total_live = max(cur_total, (1 - total_blend) * pre_total + total_blend * pace_proj_total)

        # Convert margin + total into team scores while respecting current score floor.
        home_final = 0.5 * (pred_total_live + expected_final_margin)
        away_final = 0.5 * (pred_total_live - expected_final_margin)
        home_final = max(home_final, float(row["live_home_score_current"]))
        away_final = max(away_final, float(row["live_away_score_current"]))
        pred_total_live = home_final + away_final
        pred_margin_live = home_final - away_final

        out.at[idx, "home_win_prob"] = p_adj
        out.at[idx, "pred_total"] = pred_total_live
        out.at[idx, "pred_home_margin"] = pred_margin_live
        out.at[idx, "pred_home_score"] = home_final
        out.at[idx, "pred_away_score"] = away_final

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict upcoming NBA games using season-to-date models.")
    parser.add_argument("--date", help="Target date in YYYYMMDD (default: today local time).")
    parser.add_argument("--days", type=int, default=1, help="How many consecutive dates to include (default: 1).")
    parser.add_argument(
        "--include-in-progress",
        action="store_true",
        help="Include games already in progress (gameStatus=2). Default includes only scheduled/not started.",
    )
    parser.add_argument(
        "--output",
        help="Optional CSV output path. Default: analysis/output/predictions/nba_predictions_<date>.csv",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest on held-out historical games instead of predicting upcoming games.",
    )
    return parser.parse_args()


def regular_season_schedule_all() -> pd.DataFrame:
    payload = fetch_json(SCHEDULE_URL)
    rows: list[dict[str, Any]] = []
    for day in payload["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            gid = str(g.get("gameId", ""))
            if not gid.startswith("002"):
                continue
            rows.append(
                {
                    "game_id": gid,
                    "game_status": int(g.get("gameStatus", 0)),
                    "game_status_text": g.get("gameStatusText"),
                    "game_date_est": pd.to_datetime(g.get("gameDateEst")).strftime("%Y%m%d")
                    if g.get("gameDateEst")
                    else None,
                    "game_time_utc": g.get("gameDateTimeUTC") or g.get("gameDateUTC"),
                    "home_team": g["homeTeam"]["teamTricode"],
                    "away_team": g["awayTeam"]["teamTricode"],
                    "arena_name": g.get("arenaName"),
                    "arena_city": g.get("arenaCity"),
                    "arena_state": g.get("arenaState"),
                }
            )
    df = pd.DataFrame(rows)
    df["game_time_utc"] = pd.to_datetime(df["game_time_utc"], utc=True)
    return df.sort_values(["game_time_utc", "game_id"]).reset_index(drop=True)


def compute_team_pregame_state(team_games: pd.DataFrame, team: str, opp: str, is_home: int, game_time_utc: pd.Timestamp) -> dict[str, Any]:
    hist = team_games[(team_games["team"] == team) & (team_games["game_time_utc"] < game_time_utc)].copy()
    hist = hist.sort_values(["game_time_utc", "game_id"])
    if hist.empty:
        # Season-start edge case; should be rare late in season.
        state = {
            "team": team,
            "opp": opp,
            "is_home": int(is_home),
            "game_time_utc": game_time_utc,
            "days_since_prev": np.nan,
            "b2b": 0,
            "three_in_four": 0,
            "four_in_six": 0,
            "five_in_seven": 0,
            "travel_miles_since_prev": np.nan,
            "travel_500_plus": 0,
            "travel_1000_plus": 0,
            "b2b_travel_500_plus": 0,
            "b2b_travel_heavy": 0,
            "timezone_change": 0,
            "eastbound_travel": 0,
            "games_played_season": 0,
            "late_season": 0,
            "home_stand_game_num": 1 if is_home else 0,
            "road_trip_game_num": 0 if is_home else 1,
            "road_trip_3_plus": 0 if is_home else 0,
            "pre_elo": 1500.0,  # default Elo for teams with no history
            "pre_pyth_win_pct": np.nan,
            "injury_proxy_absent_rotation_count": np.nan,
            "injury_proxy_absent_recent_starter_count": np.nan,
            "injury_proxy_missing_minutes5": np.nan,
            "injury_proxy_missing_points5": np.nan,
            "active_count": np.nan,
        }
        for m in ROLLING_METRICS:
            state[f"pre_{m}_avg5"] = np.nan
            state[f"pre_{m}_avg10"] = np.nan
            state[f"pre_{m}_season"] = np.nan
        return state

    last = hist.iloc[-1]
    times = hist["game_time_utc"].tolist()
    last_time = last["game_time_utc"]
    days_since_prev = (game_time_utc - last_time).total_seconds() / 86400.0
    b2b = int(days_since_prev > 0 and days_since_prev < 1.6)
    three_in_four = 0
    if len(times) >= 2:
        three_in_four = int((game_time_utc - times[-2]).total_seconds() / 86400.0 < 4.1)
    four_in_six = 0
    if len(times) >= 3:
        four_in_six = int((game_time_utc - times[-3]).total_seconds() / 86400.0 < 6.1)

    # Venue-based travel: previous venue is the home team's arena from the last game.
    prev_venue_team = last["team"] if int(last["is_home"]) == 1 else last["opp"]
    # Current game venue: if team is home, venue is team's own arena; if away, it's opponent's arena.
    next_venue_team = team if is_home == 1 else opp
    travel_miles = np.nan
    if prev_venue_team in TEAM_COORDS and next_venue_team in TEAM_COORDS:
        prev_lat, prev_lon = TEAM_COORDS[prev_venue_team]
        next_lat, next_lon = TEAM_COORDS[next_venue_team]
        travel_miles = haversine_miles(prev_lat, prev_lon, next_lat, next_lon)
    travel_500_plus = int(pd.notna(travel_miles) and travel_miles >= 500)
    travel_1000_plus = int(pd.notna(travel_miles) and travel_miles >= 1000)
    b2b_travel_500_plus = int(b2b == 1 and travel_500_plus == 1)
    b2b_travel_heavy = int(b2b == 1 and travel_1000_plus == 1)

    # Five-in-seven (Item 3)
    five_in_seven = 0
    if len(times) >= 4:
        five_in_seven = int((game_time_utc - times[-4]).total_seconds() / 86400.0 < 7.1)

    # Cross-timezone features (Item 3)
    timezone_change = 0
    eastbound_travel_flag = 0
    if prev_venue_team in TEAM_COORDS and next_venue_team in TEAM_COORDS:
        prev_lon = TEAM_COORDS[prev_venue_team][1]
        next_lon = TEAM_COORDS[next_venue_team][1]
        lon_diff = next_lon - prev_lon
        timezone_change = round(abs(lon_diff) / 15.0, 1)
        eastbound_travel_flag = int(lon_diff > 7.5)

    # Games played this season (Item 3)
    games_played = len(hist)
    late_season_flag = int(games_played > 65)

    # Pythagorean win expectation (Item 2)
    scored_total = hist["team_score"].sum() if "team_score" in hist.columns else np.nan
    allowed_total = hist["opp_score"].sum() if "opp_score" in hist.columns else np.nan
    PYTH_EXP = 13.91
    if pd.notna(scored_total) and pd.notna(allowed_total) and scored_total > 0 and allowed_total > 0 and len(hist) >= 5:
        se = scored_total ** PYTH_EXP
        ae = allowed_total ** PYTH_EXP
        pre_pyth_win_pct = se / (se + ae)
    else:
        pre_pyth_win_pct = np.nan

    if is_home == 1:
        home_stand_game_num = int(last["home_stand_game_num"] + 1) if int(last["is_home"]) == 1 else 1
        road_trip_game_num = 0
    else:
        road_trip_game_num = int(last["road_trip_game_num"] + 1) if int(last["is_home"]) == 0 else 1
        home_stand_game_num = 0

    # Current Elo: use post_elo from last game (i.e., Elo after last result).
    # Fall back to pre_elo from last game if post_elo is missing.
    current_elo = last.get("post_elo", np.nan)
    if pd.isna(current_elo):
        current_elo = last.get("pre_elo", np.nan)

    state: dict[str, Any] = {
        "team": team,
        "opp": opp,
        "is_home": int(is_home),
        "game_time_utc": game_time_utc,
        "days_since_prev": days_since_prev,
        "b2b": b2b,
        "three_in_four": three_in_four,
        "four_in_six": four_in_six,
        "five_in_seven": five_in_seven,
        "travel_miles_since_prev": travel_miles,
        "travel_500_plus": travel_500_plus,
        "travel_1000_plus": travel_1000_plus,
        "b2b_travel_500_plus": b2b_travel_500_plus,
        "b2b_travel_heavy": b2b_travel_heavy,
        "timezone_change": timezone_change,
        "eastbound_travel": eastbound_travel_flag,
        "games_played_season": games_played,
        "late_season": late_season_flag,
        "home_stand_game_num": home_stand_game_num,
        "road_trip_game_num": road_trip_game_num,
        "road_trip_3_plus": int(road_trip_game_num >= 3),
        # Elo rating entering the upcoming game
        "pre_elo": float(current_elo) if pd.notna(current_elo) else np.nan,
        # Pythagorean win expectation (Item 2)
        "pre_pyth_win_pct": pre_pyth_win_pct,
        # Carry forward latest observed availability proxy as a current-state proxy.
        "injury_proxy_absent_rotation_count": last.get("injury_proxy_absent_rotation_count", np.nan),
        "injury_proxy_absent_recent_starter_count": last.get("injury_proxy_absent_recent_starter_count", np.nan),
        "injury_proxy_missing_minutes5": last.get("injury_proxy_missing_minutes5", np.nan),
        "injury_proxy_missing_points5": last.get("injury_proxy_missing_points5", np.nan),
        "injury_proxy_weighted_availability": last.get("injury_proxy_weighted_availability", np.nan),
        "star_player_absent_flag": last.get("star_player_absent_flag", 0),
        "lineup_continuity": last.get("lineup_continuity", np.nan),
        "active_roster_plus_minus": last.get("active_roster_plus_minus", np.nan),
        "active_count": last.get("active_count", np.nan),
    }

    for m in ROLLING_METRICS:
        s = hist[m].dropna()
        state[f"pre_{m}_avg5"] = s.tail(5).mean() if len(s.tail(5)) >= 3 else np.nan
        state[f"pre_{m}_avg10"] = s.tail(10).mean() if len(s.tail(10)) >= 5 else np.nan
        state[f"pre_{m}_season"] = s.mean() if len(s) >= 5 else np.nan

    # Home/away venue-specific rolling averages (Item 2)
    for col in ["net_rating", "margin"]:
        if col in hist.columns:
            home_games = hist[hist["is_home"] == 1][col].dropna()
            away_games = hist[hist["is_home"] == 0][col].dropna()
            state[f"pre_{col}_home_avg5"] = home_games.tail(5).mean() if len(home_games.tail(5)) >= 2 else np.nan
            state[f"pre_{col}_away_avg5"] = away_games.tail(5).mean() if len(away_games.tail(5)) >= 2 else np.nan
        else:
            state[f"pre_{col}_home_avg5"] = np.nan
            state[f"pre_{col}_away_avg5"] = np.nan

    return state


def build_upcoming_team_states(upcoming_games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for g in upcoming_games.sort_values(["game_time_utc", "game_id"]).itertuples(index=False):
        game_time = pd.Timestamp(g.game_time_utc)
        rows.append(
            {
                "game_id": g.game_id,
                **compute_team_pregame_state(team_games, g.home_team, g.away_team, 1, game_time),
            }
        )
        rows.append(
            {
                "game_id": g.game_id,
                **compute_team_pregame_state(team_games, g.away_team, g.home_team, 0, game_time),
            }
        )
    return pd.DataFrame(rows)


def build_upcoming_game_features(upcoming_games: pd.DataFrame, team_states: pd.DataFrame, odds_joined: pd.DataFrame) -> pd.DataFrame:
    home = team_states[team_states["is_home"] == 1].copy()
    away = team_states[team_states["is_home"] == 0].copy()
    # 'team' becomes home_team/away_team after prefix, duplicating columns
    # already in upcoming_games.  Exclude it from the rename and drop it so
    # the merge doesn't produce _x/_y suffixed duplicates.
    _overlap = {"team"}
    base_cols = [c for c in team_states.columns if c not in {"game_id"} and c not in _overlap]
    home = home.rename(columns={c: f"home_{c}" for c in base_cols}).drop(columns=list(_overlap), errors="ignore")
    away = away.rename(columns={c: f"away_{c}" for c in base_cols}).drop(columns=list(_overlap), errors="ignore")
    games = upcoming_games.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")

    # Safety-net: drop any leftover _x/_y suffixed columns from the merge.
    suffix_cols = [c for c in games.columns if c.endswith(("_x", "_y"))]
    if suffix_cols:
        games = games.drop(columns=suffix_cols)

    # Derived diffs matching training schema.
    for metric in [
        "net_rating",
        "off_rating",
        "def_rating",
        "efg",
        "tov_rate",
        "orb_rate",
        "ft_rate",
        "possessions",
        "margin",
    ]:
        for w in ("avg5", "avg10", "season"):
            h = f"home_pre_{metric}_{w}"
            a = f"away_pre_{metric}_{w}"
            if h in games.columns and a in games.columns:
                games[f"diff_pre_{metric}_{w}"] = games[h] - games[a]

    for c in [
        "injury_proxy_missing_minutes5",
        "injury_proxy_missing_points5",
        "injury_proxy_absent_rotation_count",
        "injury_proxy_absent_recent_starter_count",
        "injury_proxy_weighted_availability",
        "star_player_absent_flag",
        "lineup_continuity",
        "active_roster_plus_minus",
        "active_count",
    ]:
        h = f"home_{c}"
        a = f"away_{c}"
        games[f"diff_{c}"] = games[h] - games[a]

    games["rest_diff"] = games["home_days_since_prev"] - games["away_days_since_prev"]
    games["home_b2b_adv"] = games["away_b2b"] - games["home_b2b"]
    games["travel_diff"] = games["home_travel_miles_since_prev"] - games["away_travel_miles_since_prev"]

    # Elo differential and expected margin
    if "home_pre_elo" in games.columns and "away_pre_elo" in games.columns:
        games["diff_pre_elo"] = games["home_pre_elo"] - games["away_pre_elo"]
        games["pre_elo_diff_expected_margin"] = games["diff_pre_elo"] / 28.0

    # --- Pythagorean win expectation differential (Item 2) ---
    if "home_pre_pyth_win_pct" in games.columns and "away_pre_pyth_win_pct" in games.columns:
        games["diff_pre_pyth_win_pct"] = games["home_pre_pyth_win_pct"] - games["away_pre_pyth_win_pct"]

    # --- Venue split differentials (Item 2) ---
    # For upcoming predictions, venue-specific splits come from the team's home/away-specific rolling averages
    # computed in compute_team_pregame_state (carried forward from team_games).
    # We manually wire: home team's home-split and away team's away-split.
    if "home_pre_net_rating_home_avg5" in games.columns:
        games["home_pre_net_rating_venue_split"] = games.get("home_pre_net_rating_home_avg5", np.nan)
    if "away_pre_net_rating_away_avg5" in games.columns:
        games["away_pre_net_rating_venue_split"] = games.get("away_pre_net_rating_away_avg5", np.nan)
    if "home_pre_margin_home_avg5" in games.columns:
        games["home_pre_margin_venue_split"] = games.get("home_pre_margin_home_avg5", np.nan)
    if "away_pre_margin_away_avg5" in games.columns:
        games["away_pre_margin_venue_split"] = games.get("away_pre_margin_away_avg5", np.nan)
    if "home_pre_net_rating_venue_split" in games.columns and "away_pre_net_rating_venue_split" in games.columns:
        games["diff_pre_net_rating_venue_split"] = games["home_pre_net_rating_venue_split"] - games["away_pre_net_rating_venue_split"]
    if "home_pre_margin_venue_split" in games.columns and "away_pre_margin_venue_split" in games.columns:
        games["diff_pre_margin_venue_split"] = games["home_pre_margin_venue_split"] - games["away_pre_margin_venue_split"]

    # --- Situational features (Item 3) ---
    # Altitude effect
    games["altitude_game"] = (games["home_team"] == "DEN").astype(int)
    games["altitude_short_rest"] = (
        (games["altitude_game"] == 1) & (games["away_days_since_prev"] <= 1.6)
    ).fillna(False).astype(int)

    # Situational diffs
    for c in ["five_in_seven", "b2b_travel_heavy", "games_played_season", "late_season"]:
        hc = f"home_{c}"
        ac = f"away_{c}"
        if hc in games.columns and ac in games.columns:
            games[f"diff_{c}"] = games[hc] - games[ac]

    odds_cols = [
        "game_id",
        "espn_event_id",
        "odds_provider_name",
        "market_home_spread_close",
        "market_home_spread_open",
        "market_total_close",
        "market_total_open",
        "market_home_implied_prob_close",
    ]
    games = games.merge(odds_joined[odds_cols].drop_duplicates("game_id"), on="game_id", how="left")
    # For upcoming games, close lines aren't available yet -- fall back to open lines
    games["market_home_spread_close"] = games["market_home_spread_close"].fillna(games["market_home_spread_open"])
    games["market_total_close"] = games["market_total_close"].fillna(games["market_total_open"])
    games["market_spread_move_home"] = games["market_home_spread_close"] - games["market_home_spread_open"]
    games["market_total_move"] = games["market_total_close"] - games["market_total_open"]
    games["market_spread_move_abs"] = games["market_spread_move_home"].abs()
    games["market_total_move_abs"] = games["market_total_move"].abs()

    # Opening implied probability derived from spread; ML movement
    _hs_open = games["market_home_spread_open"]
    games["market_home_implied_prob_open"] = 1.0 / (1.0 + np.exp(_hs_open / 6.5))
    games["market_ml_move"] = games["market_home_implied_prob_close"] - games["market_home_implied_prob_open"]

    # Overlay snapshot-based line movement if available
    games = _overlay_prediction_snapshots(games)

    return games.sort_values(["game_time_utc", "game_id"]).reset_index(drop=True)


def fit_xgb_classifier(
    df: pd.DataFrame, features: list[str], target: str, params: dict[str, Any] | None = None
) -> tuple[SimpleImputer, XGBClassifier]:
    train = df.dropna(subset=[target]).copy()
    feats = [f for f in features if f in train.columns]
    X = train[feats]
    y = train[target]
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    p = params or {}
    model = XGBClassifier(
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
    model.fit(X_imp, y)
    return imp, model


def fit_xgb_regressor(
    df: pd.DataFrame, features: list[str], target: str, params: dict[str, Any] | None = None
) -> tuple[SimpleImputer, XGBRegressor]:
    train = df.dropna(subset=[target]).copy()
    feats = [f for f in features if f in train.columns]
    X = train[feats]
    y = train[target]
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
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
    model.fit(X_imp, y)
    return imp, model


def fit_calibrated_classifier(
    df: pd.DataFrame, features: list[str], target: str, params: dict[str, Any] | None = None
) -> tuple[SimpleImputer, Any]:
    """Fit XGBoost + Platt scaling calibration (Phase 2D)."""
    train = df.dropna(subset=[target]).copy()
    feats = [f for f in features if f in train.columns]
    X = train[feats]
    y = train[target]
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
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
    cal_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    cal_model.fit(X_imp, y)
    return imp, cal_model


def fit_lgb_classifier(
    df: pd.DataFrame, features: list[str], target: str
) -> tuple[SimpleImputer, lgb.LGBMClassifier]:
    """LightGBM classifier for ensemble diversity (Phase 3D)."""
    train = df.dropna(subset=[target]).copy()
    feats = [f for f in features if f in train.columns]
    X = train[feats]
    y = train[target]
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbose=-1,
    )
    model.fit(X_imp, y)
    return imp, model


def fit_correction_model(
    games_hist: pd.DataFrame,
    base_win_imp: SimpleImputer,
    base_win_model: Any,
    base_total_imp: SimpleImputer,
    base_total_model: Any,
    win_features: list[str],
    total_features: list[str],
    shrinkage: float = 0.4,
    margin_residual_features: list[str] | None = None,
    total_residual_features: list[str] | None = None,
    margin_residual_params: dict[str, Any] | None = None,
    total_residual_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit residual market-correction models (XGBoost) for margin/total.

    Aligns prediction-time correction logic with analysis residual models:
    - margin residual target: actual_home_margin - implied_margin_from_spread
    - total residual target: actual_total - market_total_close
    """
    # Get games with market data
    market_games = games_hist.dropna(
        subset=["market_home_implied_prob_close", "market_total_close", "market_home_spread_close"]
    ).copy()
    if len(market_games) < 100:
        return {"fitted": False}
    # Keep the disagreement features for backward-compatible fallbacks and optional features.
    win_feats = [f for f in win_features if f in market_games.columns]
    total_feats = [f for f in total_features if f in market_games.columns]
    X_win = base_win_imp.transform(market_games[win_feats])
    p_model = base_win_model.predict_proba(X_win)[:, 1]
    X_total = base_total_imp.transform(market_games[total_feats])
    t_model = base_total_model.predict(X_total)
    p_market = market_games["market_home_implied_prob_close"].values
    t_market = market_games["market_total_close"].values
    market_games["model_market_win_disagreement"] = p_model - p_market
    market_games["model_market_total_disagreement"] = t_model - t_market

    market_games["market_home_margin_implied"] = -market_games["market_home_spread_close"]
    market_games["market_home_margin_residual"] = market_games["home_margin"] - market_games["market_home_margin_implied"]
    market_games["market_total_residual"] = market_games["total_points"] - market_games["market_total_close"]

    margin_feats = [f for f in (margin_residual_features or MARGIN_FEATURES_MARKET_RESIDUAL) if f in market_games.columns]
    total_resid_feats = [f for f in (total_residual_features or TOTAL_FEATURES_MARKET_RESIDUAL) if f in market_games.columns]
    if not margin_feats or not total_resid_feats:
        return {"fitted": False}

    margin_imp, margin_corr_model = fit_xgb_regressor(
        market_games, margin_feats, "market_home_margin_residual", margin_residual_params
    )
    total_corr_imp, total_corr_model = fit_xgb_regressor(
        market_games, total_resid_feats, "market_total_residual", total_residual_params
    )

    return {
        "fitted": True,
        "mode": "xgb_market_residual",
        "margin_corr_imp": margin_imp,
        "margin_corr_model": margin_corr_model,
        "margin_corr_features": margin_feats,
        "total_corr_imp": total_corr_imp,
        "total_corr_model": total_corr_model,
        "total_corr_features": total_resid_feats,
        # legacy keys retained for compatibility with older call sites if needed
        "shrinkage": 1.0 if not isinstance(shrinkage, (int, float)) else float(shrinkage),
    }


def predict_with_model(
    imp: SimpleImputer, model: Any, df: pd.DataFrame, features: list[str], proba: bool = False
) -> np.ndarray:
    feats = [f for f in features if f in df.columns]
    X = imp.transform(df[feats])
    # Preserve feature names as DataFrame to avoid sklearn warnings
    X = pd.DataFrame(X, columns=feats, index=df.index)
    if proba:
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def predict_bayesian_blend(
    row_df: pd.DataFrame,
    base_win_imp: SimpleImputer,
    base_win_model: Any,
    base_total_imp: SimpleImputer,
    base_total_model: Any,
    correction: dict[str, Any],
    win_features: list[str],
    total_features: list[str],
    margin_residual_std: float = 12.0,
) -> tuple[float, float, float, str]:
    """Bayesian market blending prediction (Phase 2B).

    Returns (win_prob, pred_total, pred_margin, method_used).
    """
    has_market = (
        pd.notna(row_df["market_home_implied_prob_close"].iloc[0])
        and pd.notna(row_df["market_total_close"].iloc[0])
        and pd.notna(row_df["market_home_spread_close"].iloc[0])
    )

    # Base model predictions (no market features)
    win_feats = [f for f in win_features if f in row_df.columns]
    total_feats = [f for f in total_features if f in row_df.columns]
    p_model = float(predict_with_model(base_win_imp, base_win_model, row_df, win_feats, proba=True)[0])
    t_model = float(predict_with_model(base_total_imp, base_total_model, row_df, total_feats)[0])

    if not has_market or not correction.get("fitted"):
        m_model = float("nan")
        return p_model, t_model, m_model, "base"

    p_market = float(row_df["market_home_implied_prob_close"].iloc[0])
    t_market = float(row_df["market_total_close"].iloc[0])
    m_market = float(-row_df["market_home_spread_close"].iloc[0])

    # Build residual correction features
    row_corr = row_df.copy()
    row_corr["model_market_win_disagreement"] = p_model - p_market
    row_corr["model_market_total_disagreement"] = t_model - t_market
    row_corr["market_home_margin_implied"] = m_market

    # Margin residual correction (market + predicted residual)
    corr_margin_feats = [f for f in correction["margin_corr_features"] if f in row_corr.columns]
    X_cm = correction["margin_corr_imp"].transform(row_corr[corr_margin_feats])
    margin_resid = float(correction["margin_corr_model"].predict(X_cm)[0])
    m_blend = m_market + margin_resid

    # Total residual correction (market + predicted residual)
    corr_total_feats = [f for f in correction["total_corr_features"] if f in row_corr.columns]
    X_ct = correction["total_corr_imp"].transform(row_corr[corr_total_feats])
    total_resid = float(correction["total_corr_model"].predict(X_ct)[0])
    t_blend = t_market + total_resid

    # Convert corrected margin into a win probability (keeps margin/total/win internally consistent with market correction).
    p_blend = margin_consistent_win_prob(m_blend, residual_std=margin_residual_std)
    return float(np.clip(p_blend, 0.001, 0.999)), float(t_blend), float(m_blend), "market_residual_blend"


def margin_consistent_win_prob(pred_margin: float, residual_std: float = 12.0) -> float:
    """Derive win probability from margin prediction (Phase 3C).

    P(win) = Phi(E[margin] / std) where Phi is standard normal CDF.
    """
    return float(0.5 * (1.0 + math.erf(pred_margin / (residual_std * math.sqrt(2.0)))))


def ensemble_predict(predictions: list[tuple[float, float]]) -> tuple[float, float]:
    """Simple ensemble: average predictions from multiple models (Phase 3D)."""
    if not predictions:
        return 0.5, 230.0
    win_probs = [p[0] for p in predictions]
    totals = [p[1] for p in predictions]
    return float(np.mean(win_probs)), float(np.mean(totals))


def load_tuned_params() -> dict[str, Any] | None:
    """Load tuned hyperparameters from persisted models (Phase 3E)."""
    models_path = MODEL_DIR / "advanced_models.joblib"
    if models_path.exists():
        try:
            models = joblib.load(models_path)
            tuned_params = models.get("tuned_params")
            if isinstance(tuned_params, dict):
                out = {
                    "win_params": tuned_params.get("win_enhanced"),
                    "total_params": tuned_params.get("total_enhanced"),
                    "margin_params": tuned_params.get("margin_enhanced"),
                    "win_market_params": tuned_params.get("win_market"),
                    "total_market_params": tuned_params.get("total_market"),
                    "margin_residual_params": tuned_params.get("margin_residual_market"),
                    "total_residual_params": tuned_params.get("total_residual_market"),
                    "feature_lists": models.get("feature_lists", {}),
                }
                return out
            return {
                "win_params": models.get("tuned_win_params"),
                "total_params": models.get("tuned_total_params"),
                "margin_params": models.get("tuned_margin_params"),
                "feature_lists": models.get("feature_lists", {}),
            }
        except Exception:
            pass
    return None


def build_training_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schedule_completed, team_games, player_games = build_team_games_and_players(include_historical=True)

    # Current season odds (live fetch)
    current_sched = schedule_completed[schedule_completed["season"] == SEASON].copy()
    current_with_odds = join_espn_odds(current_sched)

    # Historical odds from cache
    from analyze_nba_2025_26_advanced import load_historical_espn_odds, HIST_CACHE_DIR
    all_odds_dfs = [current_with_odds]
    for hist_season in SEASONS[:-1]:
        hist_odds = load_historical_espn_odds(hist_season)
        if not hist_odds.empty:
            hist_sched = schedule_completed[schedule_completed["season"] == hist_season].copy()
            merged = hist_sched.merge(
                hist_odds, on=["game_date_est", "home_team", "away_team"], how="left"
            )
            all_odds_dfs.append(merged)

    schedule_with_odds = pd.concat(all_odds_dfs, ignore_index=True)
    schedule_with_odds = schedule_with_odds.sort_values(["game_time_utc", "game_id"]).reset_index(drop=True)

    # Build referee features from cached boxscores
    from analyze_nba_2025_26_advanced import build_referee_game_features
    ref_features = build_referee_game_features(team_games)

    games_hist = build_game_level(team_games, schedule_with_odds, ref_features=ref_features)
    # Propagate season column
    if "season" in team_games.columns:
        season_map = team_games[team_games["is_home"] == 1][["game_id", "season"]].drop_duplicates("game_id")
        if "season" not in games_hist.columns:
            games_hist = games_hist.merge(season_map, on="game_id", how="left")

    return team_games, player_games, games_hist


def filter_upcoming(schedule_all: pd.DataFrame, start_date: str, days: int, include_in_progress: bool) -> pd.DataFrame:
    start_dt = datetime.strptime(start_date, "%Y%m%d").date()
    end_dt = start_dt + timedelta(days=days - 1)
    mask_date = schedule_all["game_date_est"].between(start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"))
    if include_in_progress:
        mask_status = schedule_all["game_status"].isin([1, 2])
    else:
        mask_status = schedule_all["game_status"].isin([1])
    return schedule_all[mask_date & mask_status].copy().sort_values(["game_time_utc", "game_id"])


def compute_team_records(team_games: pd.DataFrame, current_season: str) -> dict[str, dict[str, Any]]:
    """Compute W-L record and last-5 record for each team in the current season."""
    season_games = team_games[team_games["season"] == current_season].copy() if "season" in team_games.columns else team_games.copy()
    season_games = season_games.sort_values(["game_time_utc", "game_id"])
    records: dict[str, dict[str, Any]] = {}
    for team, grp in season_games.groupby("team"):
        wins = int((grp["margin"] > 0).sum())
        losses = int((grp["margin"] < 0).sum())
        last5 = grp.tail(5)
        l5_wins = int((last5["margin"] > 0).sum())
        l5_losses = int((last5["margin"] < 0).sum())
        last10 = grp.tail(10)
        l10_wins = int((last10["margin"] > 0).sum())
        l10_losses = int((last10["margin"] < 0).sum())
        # Streak: count consecutive W or L from most recent game
        streak_type = "W" if grp["margin"].iloc[-1] > 0 else "L"
        streak_count = 0
        for m in reversed(grp["margin"].tolist()):
            if (streak_type == "W" and m > 0) or (streak_type == "L" and m < 0):
                streak_count += 1
            else:
                break
        records[str(team)] = {
            "record": f"{wins}-{losses}",
            "last5": f"{l5_wins}-{l5_losses}",
            "last10": f"{l10_wins}-{l10_losses}",
            "streak": f"{streak_type}{streak_count}",
            "win_pct": round(wins / max(wins + losses, 1), 3),
        }
    return records


def recompute_market_signals(
    pred_df: pd.DataFrame,
    residual_std: float,
    total_residual_std: float,
) -> pd.DataFrame:
    """Recompute market-derived probabilities/signals from current predictions.

    This is required after live adjustments, which update home_win_prob/pred_total/pred_home_margin.
    """
    out = pred_df.copy()

    p_overs, p_unders = [], []
    spread_edges, spread_evs, spread_signals, spread_kelly_pcts = [], [], [], []
    ml_edges, ml_evs, ml_signals, ml_kelly_pcts = [], [], [], []
    total_edges, total_evs, total_signals, total_kelly_pcts = [], [], [], []

    for row in out.itertuples(index=False):
        one = row._asdict()
        has_market = (
            pd.notna(one.get("market_home_spread_close"))
            and pd.notna(one.get("market_total_close"))
            and pd.notna(one.get("market_home_implied_prob_close"))
        )

        p_final = float(one.get("home_win_prob", np.nan))
        t_final = float(one.get("pred_total", np.nan))
        m_pred = float(one.get("pred_home_margin", np.nan))

        # --- O/U probability from current projected total ---
        market_line = one.get("market_total_close", np.nan) if has_market else np.nan
        if (
            has_market
            and pd.notna(market_line)
            and np.isfinite(total_residual_std)
            and total_residual_std > 0
            and pd.notna(t_final)
        ):
            z = (float(market_line) - t_final) / total_residual_std
            p_over = 1.0 - normal_cdf(z)
            p_under = normal_cdf(z)
        else:
            p_over = np.nan
            p_under = np.nan
        p_overs.append(p_over)
        p_unders.append(p_under)

        # --- ATS edge from current projected margin ---
        spread_line = one.get("market_home_spread_close", np.nan) if has_market else np.nan
        if (
            has_market
            and pd.notna(spread_line)
            and np.isfinite(residual_std)
            and residual_std > 0
            and pd.notna(m_pred)
        ):
            implied_margin = -float(spread_line)
            margin_edge = m_pred - implied_margin
            p_home_cover = normal_cdf(margin_edge / residual_std)
            p_away_cover = 1.0 - p_home_cover
            if p_home_cover > BREAKEVEN_PROB + MIN_EDGE:
                spread_sig = "HOME ATS"
                p_bet_spread = p_home_cover
            elif p_away_cover > BREAKEVEN_PROB + MIN_EDGE:
                spread_sig = "AWAY ATS"
                p_bet_spread = p_away_cover
            else:
                spread_sig = "LOW CONFIDENCE"
                p_bet_spread = np.nan

            spread_ev = (
                (p_bet_spread * VIG_FACTOR - (1.0 - p_bet_spread))
                if pd.notna(p_bet_spread)
                else 0.0
            )
            if spread_ev <= 0:
                spread_sig = "LOW CONFIDENCE"
                spread_ev = 0.0
            spread_edge = margin_edge
        else:
            spread_edge = np.nan
            spread_sig = "LOW CONFIDENCE"
            spread_ev = 0.0
        spread_edges.append(spread_edge)
        spread_evs.append(spread_ev)
        spread_signals.append(spread_sig)
        spread_kelly_pcts.append(optimal_allocation(spread_ev) if spread_sig != "LOW CONFIDENCE" else 0.0)

        # --- Moneyline edge from current win probability ---
        fair_home = one.get("market_home_implied_prob_close", np.nan) if has_market else np.nan
        if has_market and pd.notna(fair_home) and pd.notna(p_final):
            ml_home_edge = p_final - float(fair_home)
            if abs(ml_home_edge) >= MIN_EDGE:
                if ml_home_edge > 0:
                    p_ml_bet = p_final
                    ml_sig = "HOME ML"
                else:
                    p_ml_bet = 1.0 - p_final
                    ml_sig = "AWAY ML"
                ml_ev = p_ml_bet * VIG_FACTOR - (1.0 - p_ml_bet)
                if ml_ev <= 0:
                    ml_sig = "LOW CONFIDENCE"
                    ml_ev = 0.0
            else:
                ml_sig = "LOW CONFIDENCE"
                ml_ev = 0.0
        else:
            ml_home_edge = np.nan
            ml_sig = "LOW CONFIDENCE"
            ml_ev = 0.0
        ml_edges.append(ml_home_edge)
        ml_evs.append(ml_ev)
        ml_signals.append(ml_sig)
        ml_kelly_pcts.append(optimal_allocation(ml_ev) if ml_sig != "LOW CONFIDENCE" else 0.0)

        # --- Total edge from current p_over ---
        if has_market and pd.notna(p_over):
            over_edge = p_over - 0.5
            if abs(over_edge) >= MIN_EDGE:
                if over_edge > 0:
                    p_bet_total = p_over
                    total_sig = "OVER"
                else:
                    p_bet_total = p_under
                    total_sig = "UNDER"
                total_ev = p_bet_total * VIG_FACTOR - (1.0 - p_bet_total)
                if total_ev <= 0:
                    total_sig = "LOW CONFIDENCE"
                    total_ev = 0.0
            else:
                total_sig = "LOW CONFIDENCE"
                total_ev = 0.0
        else:
            over_edge = np.nan
            total_sig = "LOW CONFIDENCE"
            total_ev = 0.0
        total_edges.append(over_edge)
        total_evs.append(total_ev)
        total_signals.append(total_sig)
        total_kelly_pcts.append(optimal_allocation(total_ev) if total_sig != "LOW CONFIDENCE" else 0.0)

    out["p_over"] = p_overs
    out["p_under"] = p_unders
    out["spread_edge_pts"] = spread_edges
    out["spread_ev_after_vig"] = spread_evs
    out["spread_bet_signal"] = spread_signals
    out["spread_kelly_pct"] = spread_kelly_pcts
    out["ml_edge"] = ml_edges
    out["ml_ev_after_vig"] = ml_evs
    out["ml_bet_signal"] = ml_signals
    out["ml_kelly_pct"] = ml_kelly_pcts
    out["total_edge"] = total_edges
    out["total_ev_after_vig"] = total_evs
    out["total_bet_signal"] = total_signals
    out["total_kelly_pct"] = total_kelly_pcts
    return out


def feature_intersection(features: list[str], *dfs: pd.DataFrame) -> list[str]:
    """Keep only features present in all provided dataframes, preserving order."""
    return [f for f in features if all(f in df.columns for df in dfs)]


def build_situation_notes(row: pd.Series) -> str:
    """Build a short situational note string highlighting rest/travel edges."""
    notes = []
    # Back-to-back
    home_b2b = int(row.get("home_b2b", 0) or 0)
    away_b2b = int(row.get("away_b2b", 0) or 0)
    if home_b2b and not away_b2b:
        notes.append("H:B2B")
    elif away_b2b and not home_b2b:
        notes.append("A:B2B")
    elif home_b2b and away_b2b:
        notes.append("BOTH:B2B")

    # Heavy travel
    home_travel = float(row.get("home_travel_miles_since_prev", 0) or 0)
    away_travel = float(row.get("away_travel_miles_since_prev", 0) or 0)
    if home_travel >= 1000:
        notes.append(f"H:travel{int(home_travel)}mi")
    elif home_travel >= 500:
        notes.append(f"H:travel{int(home_travel)}mi")
    if away_travel >= 1000:
        notes.append(f"A:travel{int(away_travel)}mi")
    elif away_travel >= 500:
        notes.append(f"A:travel{int(away_travel)}mi")

    # Road trip length
    home_road_trip = int(row.get("home_road_trip_game_num", 0) or 0)
    away_road_trip = int(row.get("away_road_trip_game_num", 0) or 0)
    if home_road_trip >= 3:
        notes.append(f"H:road#{home_road_trip}")
    if away_road_trip >= 3:
        notes.append(f"A:road#{away_road_trip}")

    # Rest advantage
    rest_diff = float(row.get("rest_diff", 0) or 0)
    if rest_diff >= 2.0:
        notes.append("H:rest+")
    elif rest_diff <= -2.0:
        notes.append("A:rest+")

    # Injury report highlights
    home_inj_out = int(row.get("home_injury_report_count_out", 0) or 0)
    away_inj_out = int(row.get("away_injury_report_count_out", 0) or 0)
    if home_inj_out >= 2:
        notes.append(f"H:{home_inj_out}out")
    if away_inj_out >= 2:
        notes.append(f"A:{away_inj_out}out")
    home_star = float(row.get("home_injury_report_star_status_top1", 1.0) or 1.0)
    away_star = float(row.get("away_injury_report_star_status_top1", 1.0) or 1.0)
    if home_star < 0.5:
        notes.append("H:star-out")
    if away_star < 0.5:
        notes.append("A:star-out")

    return "; ".join(notes) if notes else ""


def build_prediction_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    out["model_pick"] = np.where(out["home_win_prob"] >= 0.5, out["home_team"], out["away_team"])
    out["situation"] = out.apply(build_situation_notes, axis=1)
    out["home_win_prob_pct"] = (100 * out["home_win_prob"]).round(1)
    out["away_win_prob_pct"] = (100 * (1 - out["home_win_prob"])).round(1)
    out["pred_home_margin"] = out["pred_home_margin"].round(1)
    out["pred_total"] = out["pred_total"].round(1)
    out["pred_home_score"] = out["pred_home_score"].round(1)
    out["pred_away_score"] = out["pred_away_score"].round(1)
    out["market_home_spread_close"] = out["market_home_spread_close"].round(1)
    out["market_total_close"] = out["market_total_close"].round(1)
    out["edge_win_prob_vs_market_pp"] = (100 * (out["home_win_prob"] - out["market_home_implied_prob_close"])).round(1)
    out["edge_total_vs_market"] = (out["pred_total"] - out["market_total_close"]).round(1)
    out["edge_margin_vs_market"] = (out["pred_home_margin"] - (-out["market_home_spread_close"])).round(1)
    if "p_over" in out.columns:
        out["p_over_pct"] = (100 * out["p_over"]).round(1)
        out["p_under_pct"] = (100 * out["p_under"]).round(1)
        out["total_edge_pp"] = (100 * out["total_edge"]).round(1)
        out["spread_ev_pct"] = (100 * out["spread_ev_after_vig"]).round(1) if "spread_ev_after_vig" in out.columns else np.nan
        out["ml_ev_pct"] = (100 * out["ml_ev_after_vig"]).round(1) if "ml_ev_after_vig" in out.columns else np.nan
        out["total_ev_pct"] = (100 * out["total_ev_after_vig"]).round(1)
    out["start_local_guess"] = pd.to_datetime(out["game_time_utc"], utc=True).dt.tz_convert("America/New_York").dt.strftime(
        "%Y-%m-%d %I:%M %p ET"
    )
    if "prediction_mode" in out.columns:
        live_home_score = out.get("live_home_score_current", pd.Series(np.nan, index=out.index))
        live_away_score = out.get("live_away_score_current", pd.Series(np.nan, index=out.index))
        live_period = out.get("live_period", pd.Series("", index=out.index))
        live_clock = out.get("live_game_clock", pd.Series("", index=out.index))
        out["live_score"] = np.where(
            out["prediction_mode"].eq("live") & live_home_score.notna(),
            out["away_team"] + " " + live_away_score.fillna(0).astype(int).astype(str)
            + " - "
            + out["home_team"]
            + " "
            + live_home_score.fillna(0).astype(int).astype(str),
            "",
        )
        out["live_state"] = np.where(
            out["prediction_mode"].eq("live"),
            "P" + live_period.fillna("").astype(str) + " " + live_clock.fillna("").astype(str),
            "",
        )
    else:
        out["prediction_mode"] = "pregame"
        out["live_score"] = ""
        out["live_state"] = ""
    cols = [
        "start_local_guess",
        "away_team",
        "away_record",
        "away_streak",
        "home_team",
        "home_record",
        "home_streak",
        "prediction_mode",
        "live_score",
        "live_state",
        "model_pick",
        "home_win_prob_pct",
        "away_win_prob_pct",
        "pred_home_score",
        "pred_away_score",
        "pred_total",
        "pred_home_margin",
        "market_home_spread_close",
        "market_total_close",
        "edge_margin_vs_market",
        "edge_total_vs_market",
        "edge_win_prob_vs_market_pp",
        "p_over_pct",
        "p_under_pct",
        "spread_bet_signal",
        "ml_bet_signal",
        "total_bet_signal",
        "spread_ev_pct",
        "ml_ev_pct",
        "total_ev_pct",
        "model_confidence",
        "situation",
        "odds_provider_name",
        "game_status_text",
    ]
    present_cols = [c for c in cols if c in out.columns]
    return out[present_cols].sort_values(["start_local_guess", "away_team", "home_team"])


def _run_backtest(
    games_hist: pd.DataFrame,
    team_games: pd.DataFrame,
    win_params: dict[str, Any] | None,
    total_params: dict[str, Any] | None,
    margin_params: dict[str, Any] | None,
) -> None:
    """Run backtest on held-out last 20% of current season (Phase 3E)."""
    from nba_evaluate import (
        evaluate_win_model_comprehensive,
        evaluate_total_model_comprehensive,
        print_evaluation_report,
    )

    df = games_hist.sort_values("game_time_utc").reset_index(drop=True)
    if "season" in df.columns:
        current = df[df["season"] == SEASON]
        if len(current) < 50:
            print("Not enough current-season games for backtest.")
            return
        cut = current.index[int(len(current) * 0.8)]
        train = df.loc[:cut - 1].copy()
        test = df.loc[cut:].copy()
    else:
        cut = int(len(df) * 0.8)
        train = df.iloc[:cut].copy()
        test = df.iloc[cut:].copy()

    print(f"Backtest: {len(train)} train, {len(test)} test", flush=True)

    win_feats = [f for f in WIN_FEATURES_BASE if f in train.columns]
    total_feats = [f for f in TOTAL_FEATURES_BASE if f in train.columns]
    margin_feats = [f for f in MARGIN_FEATURES_BASE if f in train.columns]
    margin_train_params = margin_params

    win_imp, win_model = fit_xgb_classifier(train, win_feats, "home_win", win_params)
    total_imp, total_model = fit_xgb_regressor(train, total_feats, "total_points", total_params)
    margin_imp, margin_model = fit_xgb_regressor(train, margin_feats, "home_margin", margin_train_params)

    y_prob = predict_with_model(win_imp, win_model, test, win_feats, proba=True)
    y_total = predict_with_model(total_imp, total_model, test, total_feats)
    y_margin = predict_with_model(margin_imp, margin_model, test, margin_feats)

    market_prob = test["market_home_implied_prob_close"].values if "market_home_implied_prob_close" in test.columns else None
    market_total = test["market_total_close"].values if "market_total_close" in test.columns else None
    spread = test["market_home_spread_close"].values if "market_home_spread_close" in test.columns else None

    win_eval = evaluate_win_model_comprehensive(
        test["home_win"].values, y_prob,
        market_prob=market_prob, spread=spread,
        pred_margin=y_margin, y_margin=test["home_margin"].values,
    )
    total_eval = evaluate_total_model_comprehensive(
        test["total_points"].values, y_total,
        market_total=market_total,
    )

    print_evaluation_report(win_eval, total_eval, label="BACKTEST RESULTS")

    # Save backtest results
    backtest_path = PREDICTIONS_DIR / "backtest_results.json"
    import json
    backtest_path.write_text(json.dumps({"win": win_eval, "total": total_eval}, indent=2, default=str))
    print(f"\nSaved backtest: {backtest_path}")


def _attach_injury_report_features(
    pred_df: pd.DataFrame,
    player_games: pd.DataFrame,
    target_date: str,
) -> pd.DataFrame:
    """Fetch the ESPN injury report and attach team-level injury features to pred_df.

    For each game, the injury report provides pregame information about player
    availability that the backward-looking proxy system cannot capture (e.g.,
    game-day decisions, players returning from injury, load management).
    """
    INJURY_REPORT_CACHE.mkdir(parents=True, exist_ok=True)

    try:
        injury_rows = fetch_espn_injury_report(cache_key=target_date)
    except Exception as exc:
        print(f"  Warning: Could not fetch injury report: {exc}", flush=True)
        injury_rows = []

    injury_report_cols = [
        "injury_report_missing_minutes",
        "injury_report_missing_impact",
        "injury_report_star_status_top1",
        "injury_report_star_status_top2",
        "injury_report_count_out",
        "injury_report_count_questionable",
        "injury_report_total_risk",
        "injury_report_rotation_out",
    ]

    if not injury_rows:
        print("  No injury report data available -- features will be NaN", flush=True)
        for side in ("home", "away"):
            for col in injury_report_cols:
                pred_df[f"{side}_{col}"] = np.nan
        # Add diff features as NaN
        for col in injury_report_cols:
            pred_df[f"diff_{col}"] = np.nan
        return pred_df

    # Match injuries to our player roster
    injury_matched = match_injury_report_to_players(injury_rows, player_games)
    print(f"  Injury report: {len(injury_rows)} entries, {len(injury_matched)} matched to roster", flush=True)

    # Get all teams from the prediction dataframe
    all_teams = set()
    if "home_team" in pred_df.columns:
        all_teams.update(pred_df["home_team"].dropna().unique())
    if "away_team" in pred_df.columns:
        all_teams.update(pred_df["away_team"].dropna().unique())

    team_features = compute_injury_report_features(injury_matched, teams=list(all_teams))

    if team_features.empty:
        for side in ("home", "away"):
            for col in injury_report_cols:
                pred_df[f"{side}_{col}"] = np.nan
        for col in injury_report_cols:
            pred_df[f"diff_{col}"] = np.nan
        return pred_df

    # Merge for home team
    home_features = team_features.copy().rename(
        columns={c: f"home_{c}" for c in team_features.columns if c != "team"}
    ).rename(columns={"team": "home_team"})
    pred_df = pred_df.merge(home_features, on="home_team", how="left")

    # Merge for away team
    away_features = team_features.copy().rename(
        columns={c: f"away_{c}" for c in team_features.columns if c != "team"}
    ).rename(columns={"team": "away_team"})
    pred_df = pred_df.merge(away_features, on="away_team", how="left")

    # Compute differentials
    for col in injury_report_cols:
        hc = f"home_{col}"
        ac = f"away_{col}"
        if hc in pred_df.columns and ac in pred_df.columns:
            pred_df[f"diff_{col}"] = pred_df[hc] - pred_df[ac]

    # Log summary
    injured_out = sum(1 for r in injury_rows if r.get("status_lower") == "out")
    injured_dtd = sum(1 for r in injury_rows if r.get("status_lower") == "day-to-day")
    injured_q = sum(1 for r in injury_rows if r.get("status_lower") == "questionable")
    print(f"  Injury report summary: {injured_out} Out, {injured_dtd} Day-to-Day, {injured_q} Questionable", flush=True)

    return pred_df


def main() -> None:
    args = parse_args()
    target_date = args.date or datetime.now().strftime("%Y%m%d")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load tuned hyperparameters + feature lists if available
    tuned = load_tuned_params()
    win_params = tuned.get("win_params") if tuned else None
    total_params = tuned.get("total_params") if tuned else None
    margin_params = tuned.get("margin_params") if tuned else None
    margin_residual_params = tuned.get("margin_residual_params") if tuned else None
    total_residual_params = tuned.get("total_residual_params") if tuned else None
    feature_lists = (tuned or {}).get("feature_lists", {}) if tuned else {}

    win_base_features = feature_lists.get("enhanced_win", WIN_FEATURES_BASE)
    total_base_features = feature_lists.get("enhanced_total", TOTAL_FEATURES_BASE)
    market_margin_residual_features = feature_lists.get("market_margin_residual", MARGIN_FEATURES_MARKET_RESIDUAL)
    market_total_residual_features = feature_lists.get("market_total_residual", TOTAL_FEATURES_MARKET_RESIDUAL)

    print("Loading completed game history and training features...", flush=True)
    team_games, player_games, games_hist = build_training_history()

    # --- Backtest mode (Phase 3E) ---
    if args.backtest:
        print("Running backtest on held-out games...", flush=True)
        _run_backtest(games_hist, team_games, win_params, total_params, margin_params)
        return

    print("Loading full regular-season schedule and upcoming games...", flush=True)
    schedule_all = regular_season_schedule_all()
    upcoming = filter_upcoming(schedule_all, target_date, args.days, args.include_in_progress)
    if upcoming.empty:
        print(f"No upcoming scheduled games found for {target_date} (days={args.days}).")
        return

    print(f"Upcoming games selected: {len(upcoming)}", flush=True)
    print("Pulling current odds for upcoming slate (via ESPN)...", flush=True)
    upcoming_with_odds = join_espn_odds(upcoming)

    print("Building pregame features for upcoming games...", flush=True)
    team_states = build_upcoming_team_states(upcoming, team_games)
    pred_df = build_upcoming_game_features(upcoming, team_states, upcoming_with_odds)
    player_form_lookup = build_latest_player_form_lookup(player_games)
    pred_df = attach_live_snapshots(pred_df, player_form_lookup)

    # Fetch and merge official injury report for upcoming games
    print("Fetching official injury report (ESPN)...", flush=True)
    pred_df = _attach_injury_report_features(pred_df, player_games, target_date)

    # Enforce train/predict feature consistency for persisted feature_lists.
    def _sync_feature_list(name: str, feats: list[str]) -> list[str]:
        synced = feature_intersection(feats, games_hist, pred_df)
        dropped = [f for f in feats if f not in synced]
        if dropped:
            print(f"  {name}: dropped {len(dropped)} unavailable features", flush=True)
        if not synced:
            raise ValueError(f"No usable features available for {name}.")
        return synced

    win_base_features = _sync_feature_list("win_base_features", win_base_features)
    total_base_features = _sync_feature_list("total_base_features", total_base_features)
    margin_base_features = _sync_feature_list("margin_base_features", MARGIN_FEATURES_BASE)
    market_margin_residual_features = _sync_feature_list(
        "market_margin_residual_features", market_margin_residual_features
    )
    market_total_residual_features = _sync_feature_list(
        "market_total_residual_features", market_total_residual_features
    )

    # --- Train models on all completed games ---
    print("Training models...", flush=True)
    margin_train_params = margin_params

    # Base models (no market features) - used in Bayesian blending
    win_base_imp, win_base_model = fit_xgb_classifier(games_hist, win_base_features, "home_win", win_params)
    total_base_imp, total_base_model = fit_xgb_regressor(games_hist, total_base_features, "total_points", total_params)
    margin_base_imp, margin_base_model = fit_xgb_regressor(games_hist, margin_base_features, "home_margin", margin_train_params)

    # Calibrated win model (Phase 2D: Platt scaling)
    print("Fitting calibrated win model...", flush=True)
    win_cal_imp, win_cal_model = fit_calibrated_classifier(games_hist, win_base_features, "home_win", win_params)

    # LightGBM model for ensemble diversity (Phase 3D)
    print("Fitting LightGBM ensemble member...", flush=True)
    win_lgb_imp, win_lgb_model = fit_lgb_classifier(games_hist, win_base_features, "home_win")

    # Bayesian correction model (Phase 2B)
    print("Fitting Bayesian market correction model...", flush=True)
    correction = fit_correction_model(
        games_hist,
        win_base_imp,
        win_base_model,
        total_base_imp,
        total_base_model,
        win_base_features,
        total_base_features,
        shrinkage=1.0,
        margin_residual_features=market_margin_residual_features,
        total_residual_features=market_total_residual_features,
        margin_residual_params=margin_residual_params,
        total_residual_params=total_residual_params,
    )
    if correction.get("fitted"):
        print(f"  Correction model fitted successfully ({correction.get('mode', 'unknown')})", flush=True)
    else:
        print("  Correction model not fitted (insufficient market data)", flush=True)

    # Compute residual std for margin-consistent probability (Phase 3C)
    margin_preds = predict_with_model(margin_base_imp, margin_base_model, games_hist, margin_base_features)
    residual_std = float(np.std(games_hist["home_margin"].values - margin_preds))
    print(f"  Margin residual std: {residual_std:.1f}", flush=True)

    # Compute total residual std for O/U probability
    total_preds_hist = predict_with_model(total_base_imp, total_base_model, games_hist, total_base_features)
    total_residual_std = float(np.std(games_hist["total_points"].values - total_preds_hist))
    print(f"  Total residual std: {total_residual_std:.1f}", flush=True)

    # --- Predict row-by-row ---
    print("Generating predictions...", flush=True)
    win_probs = []
    totals = []
    margins = []
    win_model_used = []
    total_model_used = []
    margin_model_used = []
    confidence_intervals = []
    p_overs, p_unders = [], []
    win_edges, win_evs, spread_signals = [], [], []
    ml_edges, ml_evs, ml_signals = [], [], []
    total_edges, total_evs, total_signals = [], [], []
    spread_kelly_pcts, total_kelly_pcts, ml_kelly_pcts = [], [], []
    confidence_flags = []
    model_agreement_scores = []
    model_spread_details = []  # per-model picks for transparency

    for row in pred_df.itertuples(index=False):
        one = pd.DataFrame([row._asdict()])

        # XGBoost base prediction
        p_xgb = float(predict_with_model(win_base_imp, win_base_model, one, win_base_features, proba=True)[0])
        t_xgb = float(predict_with_model(total_base_imp, total_base_model, one, total_base_features)[0])

        # Calibrated prediction (Phase 2D)
        p_cal = float(predict_with_model(win_cal_imp, win_cal_model, one, win_base_features, proba=True)[0])

        # LightGBM prediction (Phase 3D)
        p_lgb = float(predict_with_model(win_lgb_imp, win_lgb_model, one, win_base_features, proba=True)[0])

        # Margin prediction
        m_pred = float(predict_with_model(margin_base_imp, margin_base_model, one, margin_base_features)[0])

        # Margin-consistent win probability (Phase 3C)
        p_margin = margin_consistent_win_prob(m_pred, residual_std)

        # Ensemble: average XGB calibrated + LightGBM + margin-consistent (Phase 3D)
        p_ensemble_base = float(np.mean([p_cal, p_lgb, p_margin]))

        # Bayesian market blending (Phase 2B)
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
            # Final: average ensemble base with market-residual margin-consistent probability.
            p_final = float(np.clip(0.5 * p_ensemble_base + 0.5 * p_blend, 0.001, 0.999))
            t_final = float(t_blend)
            m_pred = float(m_blend)
            win_model_used.append("ensemble_plus_market_residual")
            total_model_used.append("market_residual")
        elif has_market:
            # Fallback: just use market data as strong prior
            p_market = float(one["market_home_implied_prob_close"].iloc[0])
            p_final = float(np.clip(0.6 * p_market + 0.4 * p_ensemble_base, 0.001, 0.999))
            t_market = float(one["market_total_close"].iloc[0])
            t_final = 0.6 * t_market + 0.4 * t_xgb
            m_market = float(-one["market_home_spread_close"].iloc[0])
            m_pred = 0.6 * m_market + 0.4 * m_pred
            win_model_used.append("market_weighted")
            total_model_used.append("market_weighted")
        else:
            p_final = p_ensemble_base
            t_final = t_xgb
            win_model_used.append("ensemble_base")
            total_model_used.append("base")

        win_probs.append(p_final)
        totals.append(t_final)
        margins.append(m_pred)
        margin_model_used.append(
            "market_residual" if (has_market and correction.get("fitted")) else ("market_weighted" if has_market else "base")
        )

        # Confidence interval: pred_margin +/- 1.28 * residual_std (80% CI) (Phase 3E)
        confidence_intervals.append({
            "margin_ci_lo": round(m_pred - 1.28 * residual_std, 1),
            "margin_ci_hi": round(m_pred + 1.28 * residual_std, 1),
            "total_ci_lo": round(t_final - 1.28 * total_residual_std, 1),
            "total_ci_hi": round(t_final + 1.28 * total_residual_std, 1),
        })

        # --- O/U probability ---
        market_line = one["market_total_close"].iloc[0] if has_market else np.nan
        if has_market and pd.notna(market_line):
            z = (market_line - t_final) / total_residual_std
            p_over = 1.0 - normal_cdf(z)
            p_under = normal_cdf(z)
        else:
            p_over = np.nan
            p_under = np.nan
        p_overs.append(p_over)
        p_unders.append(p_under)

        # --- ATS (spread) edge: compare predicted margin to the spread ---
        spread_line = one["market_home_spread_close"].iloc[0] if has_market else np.nan
        if has_market and pd.notna(spread_line):
            # Spread is negative for home favorites. Model "covers" when
            # predicted margin exceeds the implied margin from the spread.
            implied_margin = -spread_line  # e.g. spread=-5 => implied margin=+5
            margin_edge = m_pred - implied_margin  # positive = model likes home more than market
            # Convert margin edge into a cover probability via normal CDF
            p_home_cover = normal_cdf(margin_edge / residual_std)
            p_away_cover = 1.0 - p_home_cover
            if p_home_cover > BREAKEVEN_PROB + MIN_EDGE:
                sig = "HOME ATS"
                p_bet = p_home_cover
                ev = p_bet * VIG_FACTOR - (1.0 - p_bet)
            elif p_away_cover > BREAKEVEN_PROB + MIN_EDGE:
                sig = "AWAY ATS"
                p_bet = p_away_cover
                ev = p_bet * VIG_FACTOR - (1.0 - p_bet)
            else:
                sig = "LOW CONFIDENCE"
                ev = 0.0
            # Gate: only signal if EV is actually positive after vig
            if ev <= 0:
                sig = "LOW CONFIDENCE"
                ev = 0.0
            home_edge = margin_edge  # raw margin edge in points
        else:
            home_edge = np.nan
            sig = "LOW CONFIDENCE"
            ev = 0.0
        win_edges.append(home_edge)
        win_evs.append(ev)
        spread_signals.append(sig)
        spread_kelly_pcts.append(optimal_allocation(ev) if sig != "LOW CONFIDENCE" else 0.0)

        # --- Moneyline edge: compare win probability to market implied prob ---
        fair_home = one["market_home_implied_prob_close"].iloc[0] if has_market else np.nan
        if has_market and pd.notna(fair_home):
            ml_home_edge = p_final - fair_home
            if abs(ml_home_edge) >= MIN_EDGE:
                if ml_home_edge > 0:
                    p_ml_bet = p_final
                    ml_sig = "HOME ML"
                else:
                    p_ml_bet = 1.0 - p_final
                    ml_sig = "AWAY ML"
                ml_ev = p_ml_bet * VIG_FACTOR - (1.0 - p_ml_bet)
                if ml_ev <= 0:
                    ml_sig = "LOW CONFIDENCE"
                    ml_ev = 0.0
            else:
                ml_sig = "LOW CONFIDENCE"
                ml_ev = 0.0
        else:
            ml_home_edge = np.nan
            ml_sig = "LOW CONFIDENCE"
            ml_ev = 0.0
        ml_edges.append(ml_home_edge)
        ml_evs.append(ml_ev)
        ml_signals.append(ml_sig)
        ml_kelly_pcts.append(optimal_allocation(ml_ev) if ml_sig != "LOW CONFIDENCE" else 0.0)

        # --- Total edge ---
        if has_market and pd.notna(p_over):
            over_edge = p_over - 0.5
            if abs(over_edge) >= MIN_EDGE:
                if over_edge > 0:
                    p_bet_total = p_over
                    t_sig = "OVER"
                else:
                    p_bet_total = p_under
                    t_sig = "UNDER"
                t_ev = p_bet_total * VIG_FACTOR - (1.0 - p_bet_total)
                # Gate: only signal if EV is actually positive after vig
                if t_ev <= 0:
                    t_sig = "LOW CONFIDENCE"
                    t_ev = 0.0
            else:
                t_sig = "LOW CONFIDENCE"
                t_ev = 0.0
        else:
            over_edge = np.nan
            t_sig = "LOW CONFIDENCE"
            t_ev = 0.0
        total_edges.append(over_edge)
        total_evs.append(t_ev)
        total_signals.append(t_sig)
        # Allocation criterion for total bet from post-vig EV, capped at 10%.
        total_kelly_pcts.append(optimal_allocation(t_ev) if t_sig != "LOW CONFIDENCE" else 0.0)

        # --- Model agreement and confidence ---
        # Collect individual model picks (home=1, away=0)
        component_probs = [p_xgb, p_cal, p_lgb, p_margin]
        component_picks = [int(p >= 0.5) for p in component_probs]
        n_home = sum(component_picks)
        n_models = len(component_picks)
        agreement_ratio = max(n_home, n_models - n_home) / n_models  # 0.5 to 1.0
        prob_spread = max(component_probs) - min(component_probs)  # how much models diverge

        # Higher agreement + narrower spread + market data = more confident
        if agreement_ratio == 1.0 and prob_spread < 0.10:
            conf_level = "HIGH"
        elif agreement_ratio >= 0.75 and prob_spread < 0.15:
            conf_level = "MED"
        else:
            conf_level = "LOW"

        if not (has_market and correction.get("fitted")):
            # Downgrade one level when no market correction available
            conf_level = "LOW" if conf_level != "LOW" else "VERY_LOW"

        confidence_flags.append(conf_level)
        model_agreement_scores.append(round(agreement_ratio, 2))
        model_spread_details.append(round(prob_spread, 3))

    pred_df["home_win_prob"] = win_probs
    pred_df["pred_total"] = totals
    pred_df["pred_home_margin"] = margins
    pred_df["pred_home_score"] = 0.5 * (pred_df["pred_total"] + pred_df["pred_home_margin"])
    pred_df["pred_away_score"] = 0.5 * (pred_df["pred_total"] - pred_df["pred_home_margin"])
    pred_df["win_model_used"] = win_model_used
    pred_df["total_model_used"] = total_model_used
    pred_df["margin_model_used"] = margin_model_used
    pred_df["p_over"] = p_overs
    pred_df["p_under"] = p_unders
    pred_df["spread_edge_pts"] = win_edges
    pred_df["spread_ev_after_vig"] = win_evs
    pred_df["spread_bet_signal"] = spread_signals
    pred_df["spread_kelly_pct"] = spread_kelly_pcts
    pred_df["ml_edge"] = ml_edges
    pred_df["ml_ev_after_vig"] = ml_evs
    pred_df["ml_bet_signal"] = ml_signals
    pred_df["ml_kelly_pct"] = ml_kelly_pcts
    pred_df["total_edge"] = total_edges
    pred_df["total_ev_after_vig"] = total_evs
    pred_df["total_bet_signal"] = total_signals
    pred_df["total_kelly_pct"] = total_kelly_pcts
    pred_df["model_confidence"] = confidence_flags
    pred_df["model_agreement"] = model_agreement_scores
    pred_df["model_prob_spread"] = model_spread_details

    # Add confidence intervals
    ci_df = pd.DataFrame(confidence_intervals)
    for col in ci_df.columns:
        pred_df[col] = ci_df[col].values

    pred_df = apply_live_adjustments(pred_df)
    live_changed = False
    if all(
        c in pred_df.columns
        for c in ["pregame_home_win_prob", "pregame_pred_total", "pregame_pred_home_margin"]
    ):
        delta_mask = (
            (pred_df["home_win_prob"] - pred_df["pregame_home_win_prob"]).abs() > 1e-9
        ) | (
            (pred_df["pred_total"] - pred_df["pregame_pred_total"]).abs() > 1e-9
        ) | (
            (pred_df["pred_home_margin"] - pred_df["pregame_pred_home_margin"]).abs() > 1e-9
        )
        live_changed = bool(delta_mask.any())
    if live_changed:
        pred_df = recompute_market_signals(
            pred_df, residual_std=residual_std, total_residual_std=total_residual_std
        )

    # --- Confidence-weighted Allocation sizing ---
    # Scale Allocation percentages by model confidence to reduce sizing on uncertain bets.
    KELLY_CONF_SCALE = {"HIGH": 1.0, "MED": 0.5, "LOW": 0.25, "VERY_LOW": 0.10}
    DAILY_ALLOCATION_CAP = 0.25  # 25% total portfolio cap per day
    for kcol in ["spread_kelly_pct", "ml_kelly_pct", "total_kelly_pct"]:
        if kcol in pred_df.columns:
            pred_df[kcol] = pred_df.apply(
                lambda r: r[kcol] * KELLY_CONF_SCALE.get(r.get("model_confidence", "LOW"), 0.25),
                axis=1,
            )
    # Apply daily Allocation cap: proportionally scale down if total exceeds cap
    total_kelly = (
        pred_df["spread_kelly_pct"].fillna(0).sum()
        + pred_df["ml_kelly_pct"].fillna(0).sum()
        + pred_df["total_kelly_pct"].fillna(0).sum()
    )
    if total_kelly > DAILY_ALLOCATION_CAP and total_kelly > 0:
        scale_factor = DAILY_ALLOCATION_CAP / total_kelly
        for kcol in ["spread_kelly_pct", "ml_kelly_pct", "total_kelly_pct"]:
            if kcol in pred_df.columns:
                pred_df[kcol] = pred_df[kcol] * scale_factor

    # --- Team records and form ---
    team_records = compute_team_records(team_games, SEASON)
    pred_df["home_record"] = pred_df["home_team"].map(lambda t: team_records.get(t, {}).get("record", "?"))
    pred_df["away_record"] = pred_df["away_team"].map(lambda t: team_records.get(t, {}).get("record", "?"))
    pred_df["home_last5"] = pred_df["home_team"].map(lambda t: team_records.get(t, {}).get("last5", "?"))
    pred_df["away_last5"] = pred_df["away_team"].map(lambda t: team_records.get(t, {}).get("last5", "?"))
    pred_df["home_streak"] = pred_df["home_team"].map(lambda t: team_records.get(t, {}).get("streak", "?"))
    pred_df["away_streak"] = pred_df["away_team"].map(lambda t: team_records.get(t, {}).get("streak", "?"))

    table = build_prediction_table(pred_df)

    # --- Per-game detailed breakdown ---
    print("\n" + "=" * 72)
    print("  UPCOMING NBA PREDICTIONS")
    print("=" * 72)
    for _, row in table.iterrows():
        away = row.get("away_team", "?")
        home = row.get("home_team", "?")
        away_rec = row.get("away_record", "")
        home_rec = row.get("home_record", "")
        away_strk = row.get("away_streak", "")
        home_strk = row.get("home_streak", "")
        start_time = row.get("start_local_guess", "")
        print(f"\n--- {away} ({away_rec}, {away_strk}) @ {home} ({home_rec}, {home_strk}) ---")
        print(f"  Time:       {start_time}")

        pick = row.get("model_pick", "?")
        hwp = row.get("home_win_prob_pct", "?")
        awp = row.get("away_win_prob_pct", "?")
        conf = row.get("model_confidence", "?")
        print(f"  Pick:       {pick}  ({home} {hwp}% / {away} {awp}%)  Confidence: {conf}")

        pred_hs = row.get("pred_home_score", "?")
        pred_as = row.get("pred_away_score", "?")
        pred_t = row.get("pred_total", "?")
        pred_m = row.get("pred_home_margin", "?")
        print(f"  Predicted:  {away} {pred_as} - {home} {pred_hs}  (Total: {pred_t}, Margin: {pred_m})")

        spread = row.get("market_home_spread_close", np.nan)
        mkt_total = row.get("market_total_close", np.nan)
        if pd.notna(spread) and pd.notna(mkt_total):
            em = row.get("edge_margin_vs_market", "?")
            et = row.get("edge_total_vs_market", "?")
            print(f"  Market:     Spread {spread:+.1f}, Total {mkt_total:.1f}")
            print(f"  Edge:       Margin {em:+.1f} pts, Total {et:+.1f} pts")

        # Bet signals
        sigs = []
        sp_sig = row.get("spread_bet_signal", "LOW CONFIDENCE")
        ml_sig = row.get("ml_bet_signal", "LOW CONFIDENCE")
        t_sig = row.get("total_bet_signal", "LOW CONFIDENCE")
        if sp_sig != "LOW CONFIDENCE":
            sp_ev = row.get("spread_ev_pct", 0)
            sigs.append(f"{sp_sig} (EV {sp_ev:+.1f}%)")
        if ml_sig != "LOW CONFIDENCE":
            ml_ev = row.get("ml_ev_pct", 0)
            sigs.append(f"{ml_sig} (EV {ml_ev:+.1f}%)")
        if t_sig != "LOW CONFIDENCE":
            t_ev = row.get("total_ev_pct", 0)
            sigs.append(f"{t_sig} (EV {t_ev:+.1f}%)")
        if sigs:
            print(f"  Signals:    {', '.join(sigs)}")

        # Situation notes
        sit = row.get("situation", "")
        if sit:
            print(f"  Situation:  {sit}")

    # Also print compact table for CSV-like viewing
    print(f"\n{'=' * 72}")
    print("  COMPACT TABLE VIEW")
    print("=" * 72)
    print(table.to_string(index=False))

    # --- Summary ---
    n_games = len(pred_df)
    spread_bets = pred_df[pred_df["spread_bet_signal"] != "LOW CONFIDENCE"]
    ml_bets = pred_df[pred_df["ml_bet_signal"] != "LOW CONFIDENCE"]
    total_bets = pred_df[pred_df["total_bet_signal"] != "LOW CONFIDENCE"]
    n_spread = len(spread_bets)
    n_ml = len(ml_bets)
    n_total = len(total_bets)

    print(f"\n=== PREDICTION SUMMARY ===")
    print(f"  Games analyzed:   {n_games}")
    print(f"  Spread (ATS):     {n_spread}/{n_games}")
    print(f"  Moneyline:        {n_ml}/{n_games}")
    print(f"  Total (O/U):      {n_total}/{n_games}")

    if n_spread > 0:
        best_spread = spread_bets.loc[spread_bets["spread_ev_after_vig"].idxmax()]
        best_side = best_spread["spread_bet_signal"]
        best_team = best_spread["home_team"] if "HOME" in best_side else best_spread["away_team"]
        best_ev_pct = 100 * best_spread["spread_ev_after_vig"]
        best_kelly = 100 * best_spread["spread_kelly_pct"]
        spread_line = best_spread["market_home_spread_close"]
        line_str = f"{spread_line:+.1f}" if pd.notna(spread_line) else "?"
        print(f"  Best spread:      {best_team} ({best_side}, line {line_str}) EV {best_ev_pct:+.1f}%  Allocation {best_kelly:.1f}%")

    if n_ml > 0:
        best_ml = ml_bets.loc[ml_bets["ml_ev_after_vig"].idxmax()]
        ml_side = best_ml["ml_bet_signal"]
        ml_team = best_ml["home_team"] if "HOME" in ml_side else best_ml["away_team"]
        ml_ev_pct = 100 * best_ml["ml_ev_after_vig"]
        ml_kelly = 100 * best_ml["ml_kelly_pct"]
        print(f"  Best moneyline:   {ml_team} ({ml_side}) EV {ml_ev_pct:+.1f}%  Allocation {ml_kelly:.1f}%")

    if n_total > 0:
        best_total = total_bets.loc[total_bets["total_ev_after_vig"].idxmax()]
        t_side = best_total["total_bet_signal"]
        matchup = f"{best_total['away_team']}@{best_total['home_team']}"
        t_ev_pct = 100 * best_total["total_ev_after_vig"]
        t_kelly = 100 * best_total["total_kelly_pct"]
        print(f"  Best total:       {matchup} ({t_side}) EV {t_ev_pct:+.1f}%  Allocation {t_kelly:.1f}%")

    if n_spread == 0 and n_ml == 0 and n_total == 0:
        print("  No actionable edges found.")

    # Summary line: total signals, positive EV count, total Allocation allocation
    n_total_signals = n_spread + n_ml + n_total
    n_positive_ev = (
        int((pred_df["spread_ev_after_vig"] > 0).sum())
        + int((pred_df["ml_ev_after_vig"] > 0).sum())
        + int((pred_df["total_ev_after_vig"] > 0).sum())
    )
    total_kelly_alloc = (
        pred_df["spread_kelly_pct"].fillna(0).sum()
        + pred_df["ml_kelly_pct"].fillna(0).sum()
        + pred_df["total_kelly_pct"].fillna(0).sum()
    )
    print(f"\n  {n_total_signals} signals today ({n_positive_ev} positive EV), total Allocation allocation: {100*total_kelly_alloc:.1f}%")

    # --- Recent model performance (out-of-sample: train on all but last 100, test on last 100) ---
    holdout_n = 100
    sorted_hist = games_hist.sort_values("game_time_utc").reset_index(drop=True)
    if len(sorted_hist) > holdout_n + 200:
        train_oos = sorted_hist.iloc[:-holdout_n].copy()
        test_oos = sorted_hist.iloc[-holdout_n:].copy()
        print(f"\n=== MODEL RECENT PERFORMANCE (out-of-sample, last {len(test_oos)} games) ===")
        # Fit quick models on the holdout-excluded training set
        oos_win_imp, oos_win_model = fit_xgb_classifier(train_oos, win_base_features, "home_win", win_params)
        oos_margin_imp, oos_margin_model = fit_xgb_regressor(train_oos, MARGIN_FEATURES_BASE, "home_margin", margin_train_params)
        oos_total_imp, oos_total_model = fit_xgb_regressor(train_oos, total_base_features, "total_points", total_params)

        oos_win_probs = predict_with_model(oos_win_imp, oos_win_model, test_oos, win_base_features, proba=True)
        oos_margins = predict_with_model(oos_margin_imp, oos_margin_model, test_oos, MARGIN_FEATURES_BASE)
        oos_totals = predict_with_model(oos_total_imp, oos_total_model, test_oos, total_base_features)

        # Straight-up accuracy
        oos_correct = ((oos_win_probs >= 0.5).astype(int) == test_oos["home_win"].values).sum()
        oos_acc = oos_correct / len(test_oos)
        print(f"  Straight-up:  {oos_correct}/{len(test_oos)} ({100*oos_acc:.1f}%)")

        # ATS accuracy (if spread data available)
        if "market_home_spread_close" in test_oos.columns:
            spread_home = test_oos["market_home_spread_close"].values
            valid_ats = ~np.isnan(spread_home)
            if valid_ats.sum() >= 10:
                ats_correct = (
                    (oos_margins[valid_ats] > -spread_home[valid_ats])
                    == (test_oos["home_margin"].values[valid_ats] > -spread_home[valid_ats])
                ).sum()
                ats_total = int(valid_ats.sum())
                ats_pct = ats_correct / ats_total
                print(f"  ATS:          {ats_correct}/{ats_total} ({100*ats_pct:.1f}%)")

        # O/U accuracy
        if "market_total_close" in test_oos.columns:
            mkt_total = test_oos["market_total_close"].values
            valid_ou = ~np.isnan(mkt_total)
            if valid_ou.sum() >= 10:
                ou_correct = ((oos_totals[valid_ou] > mkt_total[valid_ou]) == (test_oos["total_points"].values[valid_ou] > mkt_total[valid_ou])).sum()
                ou_total = int(valid_ou.sum())
                ou_pct = ou_correct / ou_total
                print(f"  O/U:          {ou_correct}/{ou_total} ({100*ou_pct:.1f}%)")

        # MAE for margin and total
        margin_mae = float(np.mean(np.abs(test_oos["home_margin"].values - oos_margins)))
        total_mae = float(np.mean(np.abs(test_oos["total_points"].values - oos_totals)))
        print(f"  Margin MAE:   {margin_mae:.1f} pts")
        print(f"  Total MAE:    {total_mae:.1f} pts")

    output_path = Path(args.output) if args.output else PREDICTIONS_DIR / f"nba_predictions_{target_date}.csv"
    pred_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions: {output_path}")


if __name__ == "__main__":
    main()
