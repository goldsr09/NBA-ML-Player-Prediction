# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA game prediction system: fetches boxscores and odds from NBA CDN + ESPN APIs, engineers features (rolling stats, travel, rest, player availability), trains XGBoost/LightGBM models, and generates daily spread/total/moneyline predictions.

## Running Scripts

All scripts run from the project root with `python scripts/<name>.py`. They must be run from the `scripts/` working directory context or the project root (scripts use `Path("analysis/output")` relative paths).

```bash
# Fetch historical boxscores + ESPN odds for training data (2021-22 through 2024-25)
python scripts/fetch_historical_seasons.py [--seasons 2021-22 2022-23] [--skip-odds]

# Run the advanced analysis pipeline (current season data + model training + evaluation)
cd scripts && python analyze_nba_2025_26_advanced.py [--tune] [--skip-espn]

# Generate predictions for upcoming games
cd scripts && python predict_upcoming_nba.py [--date YYYY-MM-DD] [--retrain] [--tune]

# Run the basic analysis (simpler logistic regression, no odds)
python scripts/analyze_nba_2025_26.py
```

## Architecture

### The Monolith: `analyze_nba_2025_26_advanced.py`

This is the core module (~1200 lines). All other modules import from it. It contains:
- **Data fetching**: NBA CDN schedule/boxscores, ESPN scoreboards/odds
- **Parsing**: team box stats, player box stats with minutes/plus-minus
- **Feature engineering**: rolling averages (5/10/season windows), travel distance (haversine), rest/B2B, player availability proxies, lineup continuity
- **Model training**: XGBoost + Optuna hyperparameter tuning, isotonic calibration, time-series CV
- **Evaluation**: accuracy, AUC, Brier score, ATS accuracy, O/U accuracy, P/L simulation

### Thin Re-export Modules

`nba_data.py`, `nba_features.py`, `nba_models.py` are re-export wrappers that import from `analyze_nba_2025_26_advanced.py`. They exist for cleaner import paths but contain no logic.

### Standalone Modules

- **`nba_evaluate.py`** — Standalone evaluation metrics (Brier score, calibration, ATS, O/U, profit/loss simulation, market comparison, prop calibration by bucket). Imported by the advanced script and predict_player_props.
- **`fetch_historical_seasons.py`** — Standalone historical data fetcher with its own caching. Fetches boxscores + ESPN odds for past seasons into `analysis/output/historical_cache/`.
- **`predict_upcoming_nba.py`** — Prediction pipeline: loads/retrains models, fetches upcoming schedule, generates daily CSV predictions to `analysis/output/predictions/`.
- **`predict_player_props.py`** — Player prop prediction pipeline with feedback loop (see below).

### Data Flow

```
fetch_historical_seasons.py → historical_cache/{season}/boxscores/*.json
                             → historical_cache/{season}/espn_odds/*.json

analyze_nba_2025_26_advanced.py:
  NBA CDN + ESPN APIs → nba_2025_26_advanced_cache/
  historical_cache/ + current season cache → feature engineering → model training
  → analysis/output/*.csv, *.json, models/*.joblib

predict_upcoming_nba.py:
  loads models/*.joblib (or retrains) → fetches upcoming schedule
  → analysis/output/predictions/nba_predictions_YYYYMMDD.csv

predict_player_props.py feedback loop:
  daily run → prop_results_history.csv (all predictions, NaN actuals)
  --grade-results → fills actuals from boxscores, computes hit/pnl
  --calibration-report → rolling metrics by stat/confidence/edge bucket
  --weekly-retrain → fresh models + calibration check + market backtest
```

### Player Props Feedback Loop

The prop prediction pipeline has a 4-phase feedback loop:

1. **Canonical Results** (`prop_results_history.csv`): Every prediction (signal + NO BET) is saved with decision-time features, line snapshots, and model outputs. Actuals filled post-game via `--grade-results`.
2. **Calibration Monitoring** (`--calibration-report`): Rolling metrics by stat_type, confidence, edge bucket. Alerts on miscalibration (gap > 8%), degraded Brier (> 0.30), or poor ROI (< -8%). Auto-suppresses signals for degraded stat types.
3. **Market Line Features**: Opening prop lines used as training features (NaN-safe for XGBoost). `--ablation-market-lines` validates improvement.
4. **OOF Residual Model**: Stage 3 model trained on out-of-fold residuals, corrections clipped to ±20% of base prediction.

#### Player Props CLI Flags

```bash
# Daily predictions (unchanged)
python scripts/predict_player_props.py --date 20260301

# Grade yesterday's predictions with actual results
python scripts/predict_player_props.py --grade-results [--date YYYYMMDD]

# Migrate old prop_tracking.csv into canonical history
python scripts/predict_player_props.py --migrate-tracking

# Generate standalone calibration report
python scripts/predict_player_props.py --calibration-report

# Market line feature ablation test
python scripts/predict_player_props.py --ablation-market-lines

# Weekly retrain (fresh models + calibration + market backtest)
python scripts/predict_player_props.py --weekly-retrain

# crontab: every Sunday 3 AM
# 0 3 * * 0 cd /path/to/NBA && python3 scripts/predict_player_props.py --weekly-retrain
```

### Key Conventions

- **Caching**: All API responses are cached to disk (JSON files). The advanced cache and historical cache are gitignored. Scripts are idempotent — re-running uses cached data.
- **Feature naming**: `pre_` prefix = pregame (shifted, no data leakage). `avg5`/`avg10`/`season` = rolling window. `diff_` prefix = home minus away differential.
- **Chronological splits**: All train/test splits are time-ordered (`chron_split`, `time_series_cv_folds`). Never random splits.
- **Models are persisted** as joblib files in `analysis/output/models/`.

## Dependencies

Python packages: pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, joblib, requests, shap (optional), scipy
