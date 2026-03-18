#!/usr/bin/env python3
"""Walk-forward backtesting for the NBA prediction system.

Implements true walk-forward validation to honestly assess whether the model
beats the market across multiple seasons. Each fold trains on all prior seasons
and tests on the next season -- no information leakage between folds.

Folds:
  Fold 1: Train on 2021-22,             test on 2022-23
  Fold 2: Train on 2021-22 + 2022-23,   test on 2023-24
  Fold 3: Train on 2021-22 .. 2023-24,  test on 2024-25
  Fold 4: Train on 2021-22 .. 2024-25,  test on 2025-26 (current season)

Usage:
    python3 scripts/walk_forward_backtest.py
    python3 scripts/walk_forward_backtest.py --folds 1,2,3   # run specific folds
    python3 scripts/walk_forward_backtest.py --no-market      # skip market comparison
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUT_DIR = PROJECT_ROOT / "analysis" / "output"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_nba_2025_26_advanced import (
    SEASON,
    SEASONS,
    add_player_availability_proxy,
    add_rest_and_rolling_team_features,
    add_travel_features,
    build_game_level,
    build_referee_game_features,
    build_team_games_and_players,
    compute_elo_ratings,
    join_espn_odds,
    load_historical_espn_odds,
    HIST_CACHE_DIR,
)

from nba_evaluate import (
    ats_accuracy,
    brier_score,
    calibration_error,
    evaluate_total_model_comprehensive,
    evaluate_win_model_comprehensive,
    over_under_accuracy,
    profit_loss_simulation,
    print_evaluation_report,
)

# Import feature lists from predict_upcoming_nba
from predict_upcoming_nba import (
    WIN_FEATURES_BASE,
    TOTAL_FEATURES_BASE,
    MARGIN_FEATURES_BASE,
    MARGIN_FEATURES_MARKET_RESIDUAL,
    TOTAL_FEATURES_MARKET_RESIDUAL,
    fit_xgb_classifier,
    fit_xgb_regressor,
    fit_calibrated_classifier,
    fit_lgb_classifier,
    predict_with_model,
    margin_consistent_win_prob,
    fit_correction_model,
)


# ---------------------------------------------------------------------------
# Walk-forward fold definitions
# ---------------------------------------------------------------------------

def define_folds() -> list[dict[str, Any]]:
    """Define walk-forward folds: train on past seasons, test on next season."""
    # SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    folds = []
    for i in range(1, len(SEASONS)):
        train_seasons = SEASONS[:i]
        test_season = SEASONS[i]
        folds.append({
            "fold": i,
            "train_seasons": train_seasons,
            "test_season": test_season,
            "description": f"Train {'+'.join(train_seasons)}, Test {test_season}",
        })
    return folds


# ---------------------------------------------------------------------------
# Model training and evaluation per fold
# ---------------------------------------------------------------------------

def run_single_fold(
    fold_def: dict[str, Any],
    games_all: pd.DataFrame,
    include_market: bool = True,
) -> dict[str, Any]:
    """Run a single walk-forward fold.

    Trains all models on train seasons, evaluates on test season.
    Returns comprehensive metrics.
    """
    fold_num = fold_def["fold"]
    train_seasons = fold_def["train_seasons"]
    test_season = fold_def["test_season"]

    print(f"\n{'=' * 72}")
    print(f"  FOLD {fold_num}: {fold_def['description']}")
    print(f"{'=' * 72}")

    # Split data by season
    train = games_all[games_all["season"].isin(train_seasons)].copy()
    test = games_all[games_all["season"] == test_season].copy()

    train = train.sort_values("game_time_utc").reset_index(drop=True)
    test = test.sort_values("game_time_utc").reset_index(drop=True)

    print(f"  Train: {len(train)} games ({', '.join(train_seasons)})")
    print(f"  Test:  {len(test)} games ({test_season})")

    if len(train) < 200:
        print(f"  WARNING: Very small training set ({len(train)} games). Results may be unreliable.")
    if len(test) < 50:
        print(f"  WARNING: Very small test set ({len(test)} games).")
        if len(test) == 0:
            return {"fold": fold_num, "error": "No test games", "train_size": len(train), "test_size": 0}

    # --- Filter features to those available ---
    win_feats = [f for f in WIN_FEATURES_BASE if f in train.columns and f in test.columns]
    total_feats = [f for f in TOTAL_FEATURES_BASE if f in train.columns and f in test.columns]
    margin_feats = [f for f in MARGIN_FEATURES_BASE if f in train.columns and f in test.columns]

    print(f"  Win features: {len(win_feats)}")
    print(f"  Total features: {len(total_feats)}")
    print(f"  Margin features: {len(margin_feats)}")

    if not win_feats or not total_feats or not margin_feats:
        return {"fold": fold_num, "error": "Insufficient features", "train_size": len(train), "test_size": len(test)}

    # --- Train models ---
    t0 = time.time()
    print(f"\n  Training XGBoost win classifier...", flush=True)
    win_imp, win_model = fit_xgb_classifier(train, win_feats, "home_win")

    print(f"  Training calibrated win classifier...", flush=True)
    win_cal_imp, win_cal_model = fit_calibrated_classifier(train, win_feats, "home_win")

    print(f"  Training LightGBM win classifier...", flush=True)
    win_lgb_imp, win_lgb_model = fit_lgb_classifier(train, win_feats, "home_win")

    print(f"  Training total regressor...", flush=True)
    total_imp, total_model = fit_xgb_regressor(train, total_feats, "total_points")

    print(f"  Training margin regressor...", flush=True)
    margin_imp, margin_model = fit_xgb_regressor(train, margin_feats, "home_margin")

    # Compute residual std from train set
    margin_preds_train = predict_with_model(margin_imp, margin_model, train, margin_feats)
    margin_residual_std = float(np.std(train["home_margin"].values - margin_preds_train))
    total_preds_train = predict_with_model(total_imp, total_model, train, total_feats)
    total_residual_std = float(np.std(train["total_points"].values - total_preds_train))

    # Market correction model (if market data available)
    correction = {"fitted": False}
    if include_market:
        market_margin_feats = [f for f in MARGIN_FEATURES_MARKET_RESIDUAL if f in train.columns and f in test.columns]
        market_total_feats = [f for f in TOTAL_FEATURES_MARKET_RESIDUAL if f in train.columns and f in test.columns]
        if market_margin_feats and market_total_feats:
            print(f"  Training market correction model...", flush=True)
            correction = fit_correction_model(
                train, win_imp, win_model, total_imp, total_model,
                win_feats, total_feats, shrinkage=1.0,
                margin_residual_features=market_margin_feats,
                total_residual_features=market_total_feats,
            )
            if correction.get("fitted"):
                print(f"    Market correction fitted successfully", flush=True)
            else:
                print(f"    Market correction not fitted (insufficient data)", flush=True)

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s", flush=True)

    # --- Predict on test set ---
    print(f"\n  Generating predictions on test set...", flush=True)

    # Individual model predictions
    p_xgb = predict_with_model(win_imp, win_model, test, win_feats, proba=True)
    p_cal = predict_with_model(win_cal_imp, win_cal_model, test, win_feats, proba=True)
    p_lgb = predict_with_model(win_lgb_imp, win_lgb_model, test, win_feats, proba=True)
    y_total = predict_with_model(total_imp, total_model, test, total_feats)
    y_margin = predict_with_model(margin_imp, margin_model, test, margin_feats)

    # Margin-consistent win probability
    p_margin = np.array([margin_consistent_win_prob(m, margin_residual_std) for m in y_margin])

    # Ensemble: average of calibrated, LightGBM, and margin-consistent
    p_ensemble = np.clip(np.mean([p_cal, p_lgb, p_margin], axis=0), 0.001, 0.999)

    # Final prediction: blend with market if available
    has_market = (
        test["market_home_implied_prob_close"].notna()
        & test["market_total_close"].notna()
        & test["market_home_spread_close"].notna()
    )

    p_final = p_ensemble.copy()
    t_final = y_total.copy()
    m_final = y_margin.copy()

    if include_market and correction.get("fitted") and has_market.any():
        print(f"  Applying market correction to {has_market.sum()} games...", flush=True)
        for idx in test.index[has_market]:
            one = test.loc[[idx]]
            try:
                from predict_upcoming_nba import predict_bayesian_blend
                p_blend, t_blend, m_blend, _ = predict_bayesian_blend(
                    one, win_imp, win_model, total_imp, total_model,
                    correction, win_feats, total_feats, margin_residual_std=margin_residual_std,
                )
                loc = test.index.get_loc(idx)
                p_final[loc] = float(np.clip(0.5 * p_ensemble[loc] + 0.5 * p_blend, 0.001, 0.999))
                t_final[loc] = float(t_blend)
                m_final[loc] = float(m_blend)
            except Exception:
                pass  # Keep ensemble prediction if blending fails

    # --- Evaluate ---
    print(f"\n  Evaluating...", flush=True)
    y_true_win = test["home_win"].values.astype(int)
    y_true_total = test["total_points"].values.astype(float)
    y_true_margin = test["home_margin"].values.astype(float)

    # Win model metrics
    win_pred = (p_final >= 0.5).astype(int)
    win_metrics = {
        "accuracy": float(accuracy_score(y_true_win, win_pred)),
        "auc": float(roc_auc_score(y_true_win, p_final)),
        "log_loss": float(log_loss(y_true_win, np.clip(p_final, 1e-6, 1 - 1e-6))),
        "brier_score": float(brier_score_loss(y_true_win, p_final)),
        "calibration_error": float(calibration_error(y_true_win, p_final)),
    }

    # Total model metrics
    total_metrics = {
        "mae": float(mean_absolute_error(y_true_total, t_final)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_total, t_final))),
        "r2": float(r2_score(y_true_total, t_final)),
    }

    # Margin model metrics
    margin_metrics = {
        "mae": float(mean_absolute_error(y_true_margin, m_final)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_margin, m_final))),
        "r2": float(r2_score(y_true_margin, m_final)),
    }

    # Individual model AUCs for comparison
    individual_aucs = {
        "xgb_auc": float(roc_auc_score(y_true_win, p_xgb)),
        "cal_auc": float(roc_auc_score(y_true_win, p_cal)),
        "lgb_auc": float(roc_auc_score(y_true_win, p_lgb)),
        "margin_consistent_auc": float(roc_auc_score(y_true_win, p_margin)),
        "ensemble_auc": float(roc_auc_score(y_true_win, p_ensemble)),
    }

    # Market comparison
    market_metrics = {}
    if include_market:
        market_prob = test["market_home_implied_prob_close"].values
        market_total = test["market_total_close"].values
        market_spread = test["market_home_spread_close"].values

        valid_market = ~np.isnan(market_prob)
        if valid_market.sum() > 50:
            mp = np.clip(market_prob[valid_market], 1e-6, 1 - 1e-6)
            y_mkt = y_true_win[valid_market]

            market_metrics["market_accuracy"] = float(accuracy_score(y_mkt, (mp >= 0.5).astype(int)))
            market_metrics["market_auc"] = float(roc_auc_score(y_mkt, mp))
            market_metrics["market_log_loss"] = float(log_loss(y_mkt, mp))
            market_metrics["market_brier_score"] = float(brier_score_loss(y_mkt, mp))

            # Model on same subset
            pf_mkt = np.clip(p_final[valid_market], 1e-6, 1 - 1e-6)
            market_metrics["model_log_loss_on_market_games"] = float(log_loss(y_mkt, pf_mkt))
            market_metrics["model_auc_on_market_games"] = float(roc_auc_score(y_mkt, pf_mkt))
            market_metrics["model_accuracy_on_market_games"] = float(accuracy_score(y_mkt, (pf_mkt >= 0.5).astype(int)))

            # Delta LL (model - market): negative = model better
            market_metrics["delta_log_loss"] = market_metrics["model_log_loss_on_market_games"] - market_metrics["market_log_loss"]
            market_metrics["model_beats_market_log_loss"] = market_metrics["delta_log_loss"] < 0

            # ATS accuracy
            valid_ats = ~(np.isnan(y_true_margin) | np.isnan(m_final) | np.isnan(market_spread))
            if valid_ats.sum() > 50:
                ats = ats_accuracy(y_true_margin[valid_ats], m_final[valid_ats], market_spread[valid_ats])
                market_metrics["ats_accuracy"] = ats.get("accuracy")
                market_metrics["ats_n"] = ats.get("n")

            # O/U accuracy
            valid_ou = ~(np.isnan(y_true_total) | np.isnan(t_final) | np.isnan(market_total))
            if valid_ou.sum() > 50:
                ou = over_under_accuracy(y_true_total[valid_ou], t_final[valid_ou], market_total[valid_ou])
                market_metrics["ou_accuracy"] = ou.get("accuracy")
                market_metrics["ou_n"] = ou.get("n")

            # P/L simulation
            valid_pl = valid_market
            if valid_pl.sum() > 50:
                pl_results = profit_loss_simulation(
                    y_true_win[valid_pl], p_final[valid_pl], market_prob[valid_pl]
                )
                market_metrics["profit_loss"] = pl_results

            # Total model vs market total
            valid_total_mkt = ~np.isnan(market_total)
            if valid_total_mkt.sum() > 50:
                market_metrics["market_total_mae"] = float(mean_absolute_error(
                    y_true_total[valid_total_mkt], market_total[valid_total_mkt]
                ))
                market_metrics["model_total_mae_on_market_games"] = float(mean_absolute_error(
                    y_true_total[valid_total_mkt], t_final[valid_total_mkt]
                ))
                market_metrics["model_beats_market_total_mae"] = (
                    market_metrics["model_total_mae_on_market_games"] < market_metrics["market_total_mae"]
                )

    result = {
        "fold": fold_num,
        "train_seasons": train_seasons,
        "test_season": test_season,
        "train_size": len(train),
        "test_size": len(test),
        "train_time_s": round(train_time, 1),
        "margin_residual_std": round(margin_residual_std, 2),
        "total_residual_std": round(total_residual_std, 2),
        "win_metrics": {k: round(v, 4) for k, v in win_metrics.items()},
        "total_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in total_metrics.items()},
        "margin_metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in margin_metrics.items()},
        "individual_aucs": {k: round(v, 4) for k, v in individual_aucs.items()},
        "market_metrics": market_metrics,
        "correction_fitted": correction.get("fitted", False),
    }

    # --- Print fold results ---
    print(f"\n  --- Fold {fold_num} Results ---")
    print(f"  Win model:    acc={win_metrics['accuracy']:.3f}  AUC={win_metrics['auc']:.3f}  "
          f"LL={win_metrics['log_loss']:.4f}  Brier={win_metrics['brier_score']:.4f}")
    print(f"  Total model:  MAE={total_metrics['mae']:.2f}  RMSE={total_metrics['rmse']:.2f}  "
          f"R2={total_metrics['r2']:.3f}")
    print(f"  Margin model: MAE={margin_metrics['mae']:.2f}  RMSE={margin_metrics['rmse']:.2f}  "
          f"R2={margin_metrics['r2']:.3f}")

    print(f"\n  Individual model AUCs:")
    for name, auc in individual_aucs.items():
        print(f"    {name}: {auc:.4f}")

    if market_metrics:
        print(f"\n  --- Market Comparison ---")
        if "market_log_loss" in market_metrics:
            print(f"  Market log loss:        {market_metrics['market_log_loss']:.4f}")
            print(f"  Model log loss:         {market_metrics['model_log_loss_on_market_games']:.4f}")
            dll = market_metrics['delta_log_loss']
            better = "MODEL" if dll < 0 else "MARKET"
            print(f"  Delta LL (model-market): {dll:+.4f} ({better} WINS)")

        if "market_auc" in market_metrics:
            print(f"  Market AUC:  {market_metrics['market_auc']:.4f}")
            print(f"  Model AUC:   {market_metrics['model_auc_on_market_games']:.4f}")

        if "ats_accuracy" in market_metrics and market_metrics["ats_accuracy"] is not None:
            print(f"  ATS accuracy: {market_metrics['ats_accuracy']:.3f} (n={market_metrics.get('ats_n', '?')})")

        if "ou_accuracy" in market_metrics and market_metrics["ou_accuracy"] is not None:
            print(f"  O/U accuracy: {market_metrics['ou_accuracy']:.3f} (n={market_metrics.get('ou_n', '?')})")

        if "market_total_mae" in market_metrics:
            print(f"  Market total MAE: {market_metrics['market_total_mae']:.2f}")
            print(f"  Model total MAE:  {market_metrics['model_total_mae_on_market_games']:.2f}")
            better = "MODEL" if market_metrics["model_beats_market_total_mae"] else "MARKET"
            print(f"  Total MAE winner: {better}")

        if "profit_loss" in market_metrics:
            print(f"  P/L simulation (flat $100 bets):")
            for pl in market_metrics["profit_loss"]:
                if pl["n_bets"] > 0:
                    print(f"    Edge>={pl['edge_threshold']:.0%}: "
                          f"{pl['n_bets']} bets, win={pl['win_rate']:.1%}, "
                          f"P/L=${pl['profit']:+.0f}, Accuracy={pl['accuracy_pct']:+.1f}%")

    # Store per-game predictions for further analysis
    result["per_game_predictions"] = {
        "game_ids": test["game_id"].tolist() if "game_id" in test.columns else [],
        "p_final": p_final.tolist(),
        "t_final": t_final.tolist(),
        "m_final": m_final.tolist(),
        "actual_win": y_true_win.tolist(),
        "actual_total": y_true_total.tolist(),
        "actual_margin": y_true_margin.tolist(),
    }

    return result


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def generate_aggregate_report(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate aggregate walk-forward report across all folds."""
    valid_folds = [f for f in fold_results if "error" not in f]
    if not valid_folds:
        return {"error": "No valid folds"}

    n_folds = len(valid_folds)

    # Collect metrics across folds
    win_accs = [f["win_metrics"]["accuracy"] for f in valid_folds]
    win_aucs = [f["win_metrics"]["auc"] for f in valid_folds]
    win_lls = [f["win_metrics"]["log_loss"] for f in valid_folds]
    win_briers = [f["win_metrics"]["brier_score"] for f in valid_folds]

    total_maes = [f["total_metrics"]["mae"] for f in valid_folds]
    total_rmses = [f["total_metrics"]["rmse"] for f in valid_folds]
    total_r2s = [f["total_metrics"]["r2"] for f in valid_folds]

    margin_maes = [f["margin_metrics"]["mae"] for f in valid_folds]
    margin_rmses = [f["margin_metrics"]["rmse"] for f in valid_folds]

    # Market comparison across folds
    folds_with_market = [f for f in valid_folds if f.get("market_metrics")]
    n_market_folds = len(folds_with_market)

    beats_ll = sum(1 for f in folds_with_market
                   if f["market_metrics"].get("model_beats_market_log_loss", False))
    beats_total_mae = sum(1 for f in folds_with_market
                         if f["market_metrics"].get("model_beats_market_total_mae", False))

    dll_values = [f["market_metrics"]["delta_log_loss"]
                  for f in folds_with_market if "delta_log_loss" in f["market_metrics"]]
    ats_values = [f["market_metrics"]["ats_accuracy"]
                  for f in folds_with_market
                  if f["market_metrics"].get("ats_accuracy") is not None]
    ou_values = [f["market_metrics"]["ou_accuracy"]
                 for f in folds_with_market
                 if f["market_metrics"].get("ou_accuracy") is not None]

    aggregate = {
        "n_folds": n_folds,
        "n_market_folds": n_market_folds,
        "total_train_games": sum(f["train_size"] for f in valid_folds),
        "total_test_games": sum(f["test_size"] for f in valid_folds),
        "win_model": {
            "accuracy": {"mean": np.mean(win_accs), "std": np.std(win_accs, ddof=1) if n_folds > 1 else 0},
            "auc": {"mean": np.mean(win_aucs), "std": np.std(win_aucs, ddof=1) if n_folds > 1 else 0},
            "log_loss": {"mean": np.mean(win_lls), "std": np.std(win_lls, ddof=1) if n_folds > 1 else 0},
            "brier_score": {"mean": np.mean(win_briers), "std": np.std(win_briers, ddof=1) if n_folds > 1 else 0},
        },
        "total_model": {
            "mae": {"mean": np.mean(total_maes), "std": np.std(total_maes, ddof=1) if n_folds > 1 else 0},
            "rmse": {"mean": np.mean(total_rmses), "std": np.std(total_rmses, ddof=1) if n_folds > 1 else 0},
            "r2": {"mean": np.mean(total_r2s), "std": np.std(total_r2s, ddof=1) if n_folds > 1 else 0},
        },
        "margin_model": {
            "mae": {"mean": np.mean(margin_maes), "std": np.std(margin_maes, ddof=1) if n_folds > 1 else 0},
            "rmse": {"mean": np.mean(margin_rmses), "std": np.std(margin_rmses, ddof=1) if n_folds > 1 else 0},
        },
        "market_comparison": {
            "folds_beating_market_log_loss": f"{beats_ll}/{n_market_folds}",
            "folds_beating_market_total_mae": f"{beats_total_mae}/{n_market_folds}",
            "avg_delta_log_loss": float(np.mean(dll_values)) if dll_values else None,
            "avg_ats_accuracy": float(np.mean(ats_values)) if ats_values else None,
            "avg_ou_accuracy": float(np.mean(ou_values)) if ou_values else None,
        },
        "per_fold": [],
    }

    for f in valid_folds:
        fold_summary = {
            "fold": f["fold"],
            "test_season": f["test_season"],
            "test_size": f["test_size"],
            "accuracy": f["win_metrics"]["accuracy"],
            "auc": f["win_metrics"]["auc"],
            "log_loss": f["win_metrics"]["log_loss"],
            "total_mae": f["total_metrics"]["mae"],
            "margin_mae": f["margin_metrics"]["mae"],
        }
        if f.get("market_metrics"):
            fold_summary["delta_ll"] = f["market_metrics"].get("delta_log_loss")
            fold_summary["ats_accuracy"] = f["market_metrics"].get("ats_accuracy")
            fold_summary["ou_accuracy"] = f["market_metrics"].get("ou_accuracy")
        aggregate["per_fold"].append(fold_summary)

    # Go/no-go assessment
    assessment = []
    if n_market_folds > 0:
        if beats_ll > n_market_folds / 2:
            assessment.append(f"PASS: Model beats market log loss in {beats_ll}/{n_market_folds} folds")
        else:
            assessment.append(f"FAIL: Model does not consistently beat market log loss ({beats_ll}/{n_market_folds} folds)")

        if beats_total_mae > n_market_folds / 2:
            assessment.append(f"PASS: Model beats market total MAE in {beats_total_mae}/{n_market_folds} folds")
        else:
            assessment.append(f"FAIL: Model does not consistently beat market total MAE ({beats_total_mae}/{n_market_folds} folds)")

        if dll_values and np.mean(dll_values) < 0:
            assessment.append(f"PASS: Average delta log loss is negative ({np.mean(dll_values):+.4f})")
        elif dll_values:
            assessment.append(f"FAIL: Average delta log loss is positive ({np.mean(dll_values):+.4f})")

        if ats_values and np.mean(ats_values) > 0.524:
            assessment.append(f"PASS: Average ATS accuracy ({np.mean(ats_values):.3f}) exceeds breakeven (0.524)")
        elif ats_values:
            assessment.append(f"FAIL: Average ATS accuracy ({np.mean(ats_values):.3f}) below breakeven (0.524)")
    else:
        assessment.append("NOTE: No market data available for comparison")

    # Overall quality checks (no market needed)
    if np.mean(win_aucs) > 0.70:
        assessment.append(f"PASS: Mean AUC ({np.mean(win_aucs):.3f}) exceeds 0.70 threshold")
    elif np.mean(win_aucs) > 0.65:
        assessment.append(f"MARGINAL: Mean AUC ({np.mean(win_aucs):.3f}) between 0.65-0.70")
    else:
        assessment.append(f"FAIL: Mean AUC ({np.mean(win_aucs):.3f}) below 0.65 threshold")

    if np.mean(win_accs) > 0.65:
        assessment.append(f"PASS: Mean accuracy ({np.mean(win_accs):.3f}) exceeds 0.65")
    else:
        assessment.append(f"MARGINAL: Mean accuracy ({np.mean(win_accs):.3f}) below 0.65")

    aggregate["assessment"] = assessment

    # Overall go/no-go
    pass_count = sum(1 for a in assessment if a.startswith("PASS"))
    fail_count = sum(1 for a in assessment if a.startswith("FAIL"))
    if fail_count == 0 and pass_count >= 2:
        aggregate["go_decision"] = "GO"
    elif fail_count <= 1 and pass_count >= 2:
        aggregate["go_decision"] = "CAUTIOUS GO"
    elif pass_count > fail_count:
        aggregate["go_decision"] = "MARGINAL"
    else:
        aggregate["go_decision"] = "NO GO"

    return aggregate


def print_aggregate_report(aggregate: dict[str, Any]) -> None:
    """Print formatted aggregate walk-forward report."""
    print(f"\n{'=' * 72}")
    print(f"  WALK-FORWARD BACKTEST -- AGGREGATE RESULTS")
    print(f"{'=' * 72}")

    print(f"\n  Folds evaluated: {aggregate['n_folds']}")
    print(f"  Folds with market data: {aggregate['n_market_folds']}")
    print(f"  Total test games: {aggregate['total_test_games']}")

    print(f"\n  --- Win Model (mean +/- std across folds) ---")
    for metric, vals in aggregate["win_model"].items():
        print(f"    {metric:20s}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

    print(f"\n  --- Total Model (mean +/- std across folds) ---")
    for metric, vals in aggregate["total_model"].items():
        print(f"    {metric:20s}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

    print(f"\n  --- Margin Model (mean +/- std across folds) ---")
    for metric, vals in aggregate["margin_model"].items():
        print(f"    {metric:20s}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

    if aggregate["market_comparison"]:
        print(f"\n  --- Market Comparison ---")
        mc = aggregate["market_comparison"]
        print(f"    Beats market log loss: {mc['folds_beating_market_log_loss']}")
        print(f"    Beats market total MAE: {mc['folds_beating_market_total_mae']}")
        if mc["avg_delta_log_loss"] is not None:
            print(f"    Avg delta LL (model-market): {mc['avg_delta_log_loss']:+.4f}")
        if mc["avg_ats_accuracy"] is not None:
            print(f"    Avg ATS accuracy: {mc['avg_ats_accuracy']:.3f}")
        if mc["avg_ou_accuracy"] is not None:
            print(f"    Avg O/U accuracy: {mc['avg_ou_accuracy']:.3f}")

    print(f"\n  --- Per-Fold Summary ---")
    print(f"  {'Fold':>4s} {'Season':>10s} {'N':>6s} {'Acc':>7s} {'AUC':>7s} {'LL':>8s} "
          f"{'T-MAE':>7s} {'M-MAE':>7s} {'dLL':>8s} {'ATS':>7s} {'O/U':>7s}")
    print(f"  {'-' * 80}")
    for f in aggregate["per_fold"]:
        dll_s = f"{f.get('delta_ll', np.nan):+.4f}" if pd.notna(f.get('delta_ll')) else "   N/A"
        ats_s = f"{f.get('ats_accuracy', np.nan):.3f}" if pd.notna(f.get('ats_accuracy')) else "  N/A"
        ou_s = f"{f.get('ou_accuracy', np.nan):.3f}" if pd.notna(f.get('ou_accuracy')) else "  N/A"
        print(f"  {f['fold']:4d} {f['test_season']:>10s} {f['test_size']:6d} "
              f"{f['accuracy']:7.3f} {f['auc']:7.3f} {f['log_loss']:8.4f} "
              f"{f['total_mae']:7.2f} {f['margin_mae']:7.2f} "
              f"{dll_s:>8s} {ats_s:>7s} {ou_s:>7s}")

    print(f"\n  --- Assessment ---")
    for a in aggregate.get("assessment", []):
        symbol = "  [+]" if a.startswith("PASS") else ("  [-]" if a.startswith("FAIL") else "  [~]")
        print(f"  {symbol} {a}")

    go = aggregate.get("go_decision", "UNKNOWN")
    print(f"\n  {'=' * 40}")
    if go == "GO":
        print(f"  >>> DECISION: GO -- Model shows consistent edge <<<")
    elif go == "CAUTIOUS GO":
        print(f"  >>> DECISION: CAUTIOUS GO -- Edge exists but not fully consistent <<<")
    elif go == "MARGINAL":
        print(f"  >>> DECISION: MARGINAL -- Results mixed, more data needed <<<")
    else:
        print(f"  >>> DECISION: NO GO -- Model does not show consistent edge <<<")
    print(f"  {'=' * 40}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward backtesting for NBA prediction models")
    p.add_argument("--folds", type=str, default=None,
                   help="Comma-separated fold numbers to run (default: all). E.g., '1,2,3'")
    p.add_argument("--no-market", action="store_true",
                   help="Skip market comparison (faster)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    include_market = not args.no_market

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  NBA WALK-FORWARD BACKTESTING")
    print("=" * 72)

    # Define folds
    all_folds = define_folds()
    print(f"\nDefined {len(all_folds)} walk-forward folds:")
    for f in all_folds:
        print(f"  Fold {f['fold']}: {f['description']}")

    # Filter to requested folds
    if args.folds:
        requested = [int(x.strip()) for x in args.folds.split(",")]
        all_folds = [f for f in all_folds if f["fold"] in requested]
        print(f"\nRunning folds: {[f['fold'] for f in all_folds]}")

    # Load all data
    print("\nLoading all game data and engineering features...", flush=True)
    t0 = time.time()
    schedule_df, team_games, player_games = build_team_games_and_players(include_historical=True)

    # Build odds-enriched schedule
    print("Joining ESPN odds...", flush=True)
    current_sched = schedule_df[schedule_df["season"] == SEASON].copy()
    current_with_odds = join_espn_odds(current_sched)

    all_odds_dfs = [current_with_odds]
    for hist_season in SEASONS[:-1]:
        hist_odds = load_historical_espn_odds(hist_season)
        if not hist_odds.empty:
            hist_sched = schedule_df[schedule_df["season"] == hist_season].copy()
            merged = hist_sched.merge(
                hist_odds, on=["game_date_est", "home_team", "away_team"], how="left"
            )
            all_odds_dfs.append(merged)

    schedule_with_odds = pd.concat(all_odds_dfs, ignore_index=True)
    schedule_with_odds = schedule_with_odds.sort_values(["game_time_utc", "game_id"]).reset_index(drop=True)

    # Build referee features
    print("Building referee features...", flush=True)
    ref_features = build_referee_game_features(team_games)

    # Build game-level DataFrame
    print("Building game-level features...", flush=True)
    games_all = build_game_level(team_games, schedule_with_odds, ref_features=ref_features)

    # Propagate season column
    if "season" in team_games.columns:
        season_map = team_games[team_games["is_home"] == 1][["game_id", "season"]].drop_duplicates("game_id")
        if "season" not in games_all.columns:
            games_all = games_all.merge(season_map, on="game_id", how="left")

    load_time = time.time() - t0
    print(f"\nData loaded in {load_time:.1f}s")
    print(f"Total game-level rows: {len(games_all)}")
    if "season" in games_all.columns:
        for s in SEASONS:
            n = (games_all["season"] == s).sum()
            if n > 0:
                print(f"  {s}: {n} games")

    # Run folds
    fold_results = []
    for fold_def in all_folds:
        try:
            result = run_single_fold(fold_def, games_all, include_market=include_market)
            fold_results.append(result)
        except Exception as exc:
            print(f"\n  ERROR in fold {fold_def['fold']}: {exc}")
            import traceback
            traceback.print_exc()
            fold_results.append({
                "fold": fold_def["fold"],
                "error": str(exc),
                "train_size": 0,
                "test_size": 0,
            })

    # Generate aggregate report
    aggregate = generate_aggregate_report(fold_results)
    print_aggregate_report(aggregate)

    # Save results
    output_path = OUT_DIR / "walk_forward_results.json"

    # Make per-game predictions serializable
    save_results = []
    for r in fold_results:
        r_copy = {k: v for k, v in r.items() if k != "per_game_predictions"}
        if "per_game_predictions" in r:
            # Store per-game predictions separately for size
            pgp = r["per_game_predictions"]
            r_copy["per_game_predictions_summary"] = {
                "n_games": len(pgp.get("game_ids", [])),
            }
        save_results.append(r_copy)

    output_data = {
        "run_timestamp": datetime.now().isoformat(),
        "aggregate": aggregate,
        "fold_results": save_results,
    }

    # Custom serializer for numpy types
    def _default_serializer(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if pd.isna(obj):
            return None
        return str(obj)

    output_path.write_text(json.dumps(output_data, indent=2, default=_default_serializer))
    print(f"\nResults saved to {output_path}")

    # Also save per-game predictions to a CSV for further analysis
    all_game_preds = []
    for r in fold_results:
        if "per_game_predictions" not in r:
            continue
        pgp = r["per_game_predictions"]
        n = len(pgp.get("game_ids", []))
        for i in range(n):
            all_game_preds.append({
                "fold": r["fold"],
                "test_season": r["test_season"],
                "game_id": pgp["game_ids"][i],
                "p_final": pgp["p_final"][i],
                "pred_total": pgp["t_final"][i],
                "pred_margin": pgp["m_final"][i],
                "actual_win": pgp["actual_win"][i],
                "actual_total": pgp["actual_total"][i],
                "actual_margin": pgp["actual_margin"][i],
            })

    if all_game_preds:
        pred_df = pd.DataFrame(all_game_preds)
        pred_path = OUT_DIR / "walk_forward_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Per-game predictions saved to {pred_path} ({len(pred_df)} games)")


if __name__ == "__main__":
    main()
