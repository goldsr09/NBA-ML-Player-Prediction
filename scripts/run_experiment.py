#!/usr/bin/env python3
"""
Autoresearch experiment runner for NBA player prop predictions.

This is the SINGLE FILE the agent modifies. It contains:
  - EXPERIMENT_CONFIG: hyperparameters, feature groups, and settings
  - custom_features(): optional hook for new feature engineering
  - main(): loads data, runs walk-forward evaluation, prints standardized metrics

The agent modifies the config and/or adds custom feature engineering code,
then runs: python scripts/run_experiment.py

The output ends with a standardized metrics block that the agent parses.
Do NOT modify the evaluation harness or metric printing below the config section.

Usage: python scripts/run_experiment.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Add scripts dir to path and import from the props pipeline
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import predict_player_props as props

# ---------------------------------------------------------------------------
# ███████╗██╗  ██╗██████╗ ███████╗██████╗ ██╗███╗   ███╗███████╗███╗   ██╗████████╗
# ██╔════╝╚██╗██╔╝██╔══██╗██╔════╝██╔══██╗██║████╗ ████║██╔════╝████╗  ██║╚══██╔══╝
# █████╗   ╚███╔╝ ██████╔╝█████╗  ██████╔╝██║██╔████╔██║█████╗  ██╔██╗ ██║   ██║
# ██╔══╝   ██╔██╗ ██╔═══╝ ██╔══╝  ██╔══██╗██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║
# ███████╗██╔╝ ██╗██║     ███████╗██║  ██║██║██║ ╚═╝ ██║███████╗██║ ╚████║   ██║
# ╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝
#
# EVERYTHING ABOVE THIS LINE IS FIXED. MODIFY ONLY THE SECTION BELOW.
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIG: dict[str, Any] = {
    # --- XGBoost hyperparameters (applied to all stat targets) ---
    "xgb_params": {
        "n_estimators": 4000,
        "max_depth": 5,
        "learning_rate": 0.004,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "reg_lambda": 15.0,
        "reg_alpha": 0.1,
        "min_child_weight": 3,
    },

    # --- Override per-target (only target-specific differences) ---
    "xgb_params_by_target": {
        "points": {
            "reg_lambda": 20.0,
        },
        "fg3m": {
            "colsample_bytree": 0.7,
        },
        "assists": {
            "colsample_bytree": 0.7,
            "reg_lambda": 20.0,
        },
        "minutes": {
            "reg_lambda": 20.0,
        },
    },

    # --- Feature groups to EXCLUDE from the full feature list ---
    # Available groups (see FEATURE_GROUPS_FOR_ABLATION in predict_player_props.py):
    #   market_lines, boxscore_advanced, injury_context, matchup, referee,
    #   vegas_context, recency, distribution, shrinkage, tracking,
    #   implied_vegas, opp_3pt_defense, hustle, defensive_matchup,
    #   scoring_context, rotation, matchups_v3, bref_advanced
    "exclude_feature_groups": [],

    # --- Feature groups to INCLUDE (if set, only these groups' features are added) ---
    # Leave empty to use all groups (minus exclusions above).
    "include_feature_groups": [],

    # --- Additional features to ADD beyond the standard list ---
    # These are column names that must exist in the DataFrame after custom_features()
    "extra_features": ["cust_pts_per_min_avg5", "cust_reb_per_min_avg5", "cust_ast_per_min_avg5", "cust_reb_opportunity", "cust_blowout_risk", "cust_pace_adj_pts", "cust_pace_adj_reb", "cust_pace_adj_ast"],

    # --- Features to REMOVE from the standard list ---
    "remove_features": [],

    # --- Signal thresholds for betting evaluation ---
    "signal_buffer": 0.03,  # added to BREAKEVEN_PROB for signal threshold

    # --- Targets to evaluate ---
    "targets": ["points", "rebounds", "assists", "minutes"],

    # --- Minimum training rows per fold ---
    "min_train_rows": 500,
    "min_test_rows": 100,
}


def custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Optional hook: engineer new features on the player DataFrame.

    This function is called ONCE on the full player_df before walk-forward splits.
    Any new columns you create here can be referenced in EXPERIMENT_CONFIG["extra_features"].

    RULES:
    - You MUST NOT use future data. All features must be derived from pre-game
      (shifted) columns (prefixed with 'pre_') or static context.
    - You CAN create interaction features, ratios, polynomial terms, etc.
    - You CAN use columns from the existing DataFrame.

    Example:
        df["my_new_feature"] = df["pre_points_avg5"] * df["pre_minutes_avg5"]
    """
    # --- Add your custom features here ---

    # Per-minute rate features (avg5)
    mins5 = df["pre_minutes_avg5"].replace(0, np.nan)
    df["cust_pts_per_min_avg5"] = df["pre_points_avg5"] / mins5
    df["cust_reb_per_min_avg5"] = df["pre_rebounds_avg5"] / mins5
    df["cust_ast_per_min_avg5"] = df["pre_assists_avg5"] / mins5

    # Rest x production interactions
    df["cust_rest_x_pts"] = df["player_days_rest"] * df["pre_points_avg5"]
    df["cust_rest_x_min"] = df["player_days_rest"] * df["pre_minutes_avg5"]

    # Rebound opportunity: reb chance share x opponent possessions (pace-adjusted)
    df["cust_reb_opportunity"] = df["pre_player_reb_chance_share_avg5"] * df["opp_pre_possessions_avg5"]

    # Blowout risk: big spread × starter = minutes compression risk
    df["cust_blowout_risk"] = df["abs_spread"] * df["pre_starter_rate"]

    # Pace-adjusted production: per-minute rate × implied game pace
    df["cust_pace_adj_pts"] = (df["pre_points_avg5"] / mins5) * df["implied_pace"]
    df["cust_pace_adj_reb"] = (df["pre_rebounds_avg5"] / mins5) * df["implied_pace"]
    df["cust_pace_adj_ast"] = (df["pre_assists_avg5"] / mins5) * df["implied_pace"]

    return df


# ---------------------------------------------------------------------------
# ██████╗  ██████╗     ███╗   ██╗ ██████╗ ████████╗    ███████╗██████╗ ██╗████████╗
# ██╔══██╗██╔═══██╗    ████╗  ██║██╔═══██╗╚══██╔══╝    ██╔════╝██╔══██╗██║╚══██╔══╝
# ██║  ██║██║   ██║    ██╔██╗ ██║██║   ██║   ██║       █████╗  ██║  ██║██║   ██║
# ██║  ██║██║   ██║    ██║╚██╗██║██║   ██║   ██║       ██╔══╝  ██║  ██║██║   ██║
# ██████╔╝╚██████╔╝    ██║ ╚████║╚██████╔╝   ██║       ███████╗██████╔╝██║   ██║
# ╚═════╝  ╚═════╝     ╚═╝  ╚═══╝ ╚═════╝    ╚═╝       ╚══════╝╚═════╝ ╚═╝  ╚═╝
#
# EVERYTHING BELOW THIS LINE IS THE EVALUATION HARNESS. DO NOT MODIFY.
# ---------------------------------------------------------------------------


def _get_excluded_features(config: dict[str, Any]) -> set[str]:
    """Build set of features to exclude based on config."""
    excluded: set[str] = set()
    groups = props.FEATURE_GROUPS_FOR_ABLATION

    # Exclude specific groups
    for group_name in config.get("exclude_feature_groups", []):
        if group_name in groups:
            excluded.update(groups[group_name])

    # If include_feature_groups is set, exclude everything NOT in those groups
    include = config.get("include_feature_groups", [])
    if include:
        included_feats: set[str] = set()
        for group_name in include:
            if group_name in groups:
                included_feats.update(groups[group_name])
        # Exclude all group features not in the include list
        for group_name, feats in groups.items():
            if group_name not in include:
                excluded.update(feats)

    # Remove specific features
    excluded.update(config.get("remove_features", []))

    return excluded


def _get_features(target: str, config: dict[str, Any]) -> list[str]:
    """Get the feature list for a target, applying config exclusions/additions."""
    base_features = props.get_feature_list(target, two_stage=False, use_market_features=True)
    excluded = _get_excluded_features(config)
    features = [f for f in base_features if f not in excluded]
    extras = config.get("extra_features", [])
    features.extend(f for f in extras if f not in features)
    return features


def _get_params(target: str, config: dict[str, Any]) -> dict[str, Any]:
    """Get XGBoost params for a target, merging defaults with overrides."""
    params = dict(config.get("xgb_params", {}))
    per_target = config.get("xgb_params_by_target", {})
    if target in per_target:
        params.update(per_target[target])
    return params


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = wins / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def run_walk_forward(
    player_df: pd.DataFrame,
    config: dict[str, Any],
    bet_size: float = 100.0,
) -> dict[str, Any]:
    """Walk-forward backtest using the experiment config."""
    if "season" not in player_df.columns:
        print("ERROR: season column required", flush=True)
        return {}

    seasons = sorted(player_df["season"].unique())
    if len(seasons) < 3:
        print(f"ERROR: Need >=3 seasons, have {seasons}", flush=True)
        return {}

    targets = config.get("targets", list(props.PROP_TARGETS))
    # Add fg3m if available and not already in targets
    if "fg3m" not in targets and "fg3m" in player_df.columns and player_df["fg3m"].notna().sum() > 100:
        targets = list(targets) + ["fg3m"]

    signal_buffer = config.get("signal_buffer", 0.03)
    min_train = config.get("min_train_rows", 500)
    min_test = config.get("min_test_rows", 100)

    print(f"\nWalk-forward backtest: {len(seasons)} seasons, {len(targets)} targets", flush=True)
    print(f"Seasons: {seasons}", flush=True)

    fold_results: list[dict[str, Any]] = []

    for i in range(2, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]
        train = player_df[player_df["season"].isin(train_seasons)].copy()
        test = player_df[player_df["season"] == test_season].copy()

        if len(train) < min_train or len(test) < min_test:
            print(f"\n  Fold {i-1}: skip ({len(train)} train, {len(test)} test)", flush=True)
            continue

        print(f"\n  Fold {i-1}: {train_seasons} ({len(train)}) -> {test_season} ({len(test)})", flush=True)

        fold_metrics: dict[str, dict[str, Any]] = {}

        for target in targets:
            features = _get_features(target, config)
            feats = props.filter_features(features, train)
            if not feats:
                continue

            params = _get_params(target, config)
            try:
                imp, model, used_feats = props.train_prop_model(
                    train, features, target, params=params,
                )
            except ValueError:
                continue

            test_valid = test.dropna(subset=[target]).copy()
            if test_valid.empty:
                continue

            preds = props.predict_prop(imp, model, used_feats, test_valid)
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
                print(f"    {target:>10s}:  MAE={mae:.3f}  R2={r2:.3f}", flush=True)
                continue

            valid_mask = ~np.isnan(synthetic_lines)
            if valid_mask.sum() < 20:
                fold_metrics[target] = {"mae": mae, "rmse": rmse, "r2": r2, "n_test": len(test_valid)}
                print(f"    {target:>10s}:  MAE={mae:.3f}  R2={r2:.3f}", flush=True)
                continue

            preds_v = preds[valid_mask]
            actual_v = actual[valid_mask]
            lines_v = synthetic_lines[valid_mask]

            z_scores = (lines_v - preds_v) / max(residual_std, 0.01)
            p_overs = 1.0 - sp_stats.t.cdf(z_scores, df=7)

            over_mask = p_overs > (props.BREAKEVEN_PROB + signal_buffer)
            under_mask = p_overs < (1.0 - props.BREAKEVEN_PROB - signal_buffer)

            n_over = int(over_mask.sum())
            n_under = int(under_mask.sum())
            over_hit = int((actual_v[over_mask] > lines_v[over_mask]).sum()) if n_over > 0 else 0
            under_hit = int((actual_v[under_mask] < lines_v[under_mask]).sum()) if n_under > 0 else 0

            n_bets = n_over + n_under
            total_wins = over_hit + under_hit
            wr_low, wr_high = _wilson_ci(total_wins, n_bets)
            total_profit = (
                total_wins * bet_size * props.VIG_FACTOR
                - (n_bets - total_wins) * bet_size
            )

            fold_metrics[target] = {
                "mae": mae, "rmse": rmse, "r2": r2,
                "n_test": len(test_valid), "residual_std": residual_std,
                "n_over": n_over, "n_under": n_under, "n_bets": n_bets,
                "total_wins": total_wins,
                "over_hit_rate": round(over_hit / n_over, 4) if n_over > 0 else np.nan,
                "under_hit_rate": round(under_hit / n_under, 4) if n_under > 0 else np.nan,
                "total_win_rate": round(total_wins / n_bets, 4) if n_bets > 0 else np.nan,
                "profit": round(total_profit, 2),
                "roi_pct": round(100 * total_profit / (n_bets * bet_size), 2) if n_bets > 0 else np.nan,
            }

            wr = fold_metrics[target].get("total_win_rate", np.nan)
            wr_s = f"{wr:.1%}" if pd.notna(wr) else "N/A"
            print(
                f"    {target:>10s}:  MAE={mae:.3f}  R2={r2:.3f}  "
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

    # --- Aggregate ---
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
    total_mae_all: list[float] = []
    total_r2_all: list[float] = []

    for target, vals in agg.items():
        avg_mae = float(np.nanmean(vals["mae"]))
        avg_r2 = float(np.nanmean(vals["r2"]))
        sum_profit = sum(vals["profit"])
        sum_bets = int(sum(vals["n_bets"]))
        avg_wr = float(np.nanmean(vals["win_rate"])) if vals["win_rate"] else np.nan
        roi = (100 * sum_profit / (sum_bets * bet_size)) if sum_bets > 0 else np.nan

        total_profit_all += sum_profit
        total_bets_all += sum_bets
        total_mae_all.append(avg_mae)
        total_r2_all.append(avg_r2)

    overall_mae = float(np.nanmean(total_mae_all)) if total_mae_all else 999.0
    overall_r2 = float(np.nanmean(total_r2_all)) if total_r2_all else -999.0
    overall_roi = (100 * total_profit_all / (total_bets_all * bet_size)) if total_bets_all > 0 else 0.0

    return {
        "folds": fold_results,
        "aggregate": agg,
        "overall_mae": overall_mae,
        "overall_r2": overall_r2,
        "overall_profit": total_profit_all,
        "overall_bets": total_bets_all,
        "overall_roi": overall_roi,
        "per_target": {
            target: {
                "avg_mae": float(np.nanmean(v["mae"])),
                "avg_r2": float(np.nanmean(v["r2"])),
                "total_profit": sum(v["profit"]),
                "total_bets": int(sum(v["n_bets"])),
                "avg_win_rate": float(np.nanmean(v["win_rate"])) if v["win_rate"] else np.nan,
            }
            for target, v in agg.items()
        },
    }


def main() -> None:
    t0 = time.time()

    # Load feature cache
    print("Loading player feature cache...", flush=True)
    cache_file = props.PLAYER_FEATURE_CACHE_FILE
    if not cache_file.exists():
        # Try legacy paths
        for alt in [props.LEGACY_PLAYER_FEATURE_CACHE_FILE]:
            if alt.exists():
                cache_file = alt
                break
        else:
            print("ERROR: No player feature cache found. Run predict_player_props.py --weekly-retrain first.")
            sys.exit(1)

    player_df = pd.read_pickle(cache_file)
    print(f"  Loaded {len(player_df)} rows, {len(player_df.columns)} columns", flush=True)

    # Apply custom features
    player_df = custom_features(player_df)

    # Run walk-forward evaluation
    results = run_walk_forward(player_df, EXPERIMENT_CONFIG)

    elapsed = time.time() - t0

    if not results:
        print("\n---")
        print("status:           CRASH")
        print(f"elapsed_seconds:  {elapsed:.1f}")
        sys.exit(1)

    # --- Print standardized metrics block ---
    print(f"\n{'=' * 60}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 60}")
    for target, vals in results.get("per_target", {}).items():
        avg_mae = vals["avg_mae"]
        avg_r2 = vals["avg_r2"]
        total_bets = vals["total_bets"]
        avg_wr = vals.get("avg_win_rate", np.nan)
        total_profit = vals["total_profit"]
        roi = (100 * total_profit / (total_bets * 100)) if total_bets > 0 else 0.0
        wr_s = f"{avg_wr:.1%}" if pd.notna(avg_wr) else "N/A"
        print(
            f"  {target:>10s}:  MAE={avg_mae:.3f}  R2={avg_r2:.3f}  "
            f"Bets={total_bets}  WR={wr_s}  P/L=${total_profit:+.0f}  ROI={roi:.1f}%",
            flush=True,
        )

    overall_roi = results["overall_roi"]
    overall_bets = results["overall_bets"]
    if overall_bets > 0:
        if overall_roi > 2.0 and overall_bets >= 50:
            assessment = "GO"
        elif overall_roi > 0:
            assessment = "CAUTIOUS GO"
        else:
            assessment = "NO GO"
    else:
        assessment = "NO BETS"

    # --- Standardized output block (parsed by agent) ---
    print("\n---")
    print(f"overall_mae:      {results['overall_mae']:.6f}")
    print(f"overall_r2:       {results['overall_r2']:.6f}")
    print(f"overall_profit:   {results['overall_profit']:.2f}")
    print(f"overall_bets:     {results['overall_bets']}")
    print(f"overall_roi:      {results['overall_roi']:.2f}")
    print(f"assessment:       {assessment}")
    print(f"elapsed_seconds:  {elapsed:.1f}")
    for target, vals in results.get("per_target", {}).items():
        print(f"mae_{target}:{''.ljust(max(1, 14 - len(target)))}{vals['avg_mae']:.6f}")
    print("---")


if __name__ == "__main__":
    main()
