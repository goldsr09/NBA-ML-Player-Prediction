"""NBA model training, tuning, and calibration module.

Provides functions for XGBoost training, Optuna hyperparameter tuning,
time-series cross-validation, and model evaluation.

Usage:
    from nba_models import eval_win_model, optuna_tune_xgb_classifier, run_advanced_models
"""
from __future__ import annotations

from analyze_nba_2025_26_advanced import (
    # Model evaluation functions
    eval_win_model,
    eval_total_model,
    eval_win_model_cv,
    eval_total_model_cv,
    eval_market_baselines,
    run_advanced_models,
    # Tuning functions
    optuna_tune_xgb_classifier,
    optuna_tune_xgb_regressor,
    # Split functions
    chron_split,
    time_series_cv_folds,
)

__all__ = [
    "eval_win_model",
    "eval_total_model",
    "eval_win_model_cv",
    "eval_total_model_cv",
    "eval_market_baselines",
    "run_advanced_models",
    "optuna_tune_xgb_classifier",
    "optuna_tune_xgb_regressor",
    "chron_split",
    "time_series_cv_folds",
]
