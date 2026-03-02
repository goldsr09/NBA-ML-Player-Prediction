#!/usr/bin/env python3
"""
Q4 Total Points Training Pipeline

Trains an XGBoost model to project final game totals from end-of-Q3 data.
Used by kalshi_q4_edge.py for live Kalshi edge scanning.

Usage:
    python3 scripts/train_q4_model.py              # Train with defaults
    python3 scripts/train_q4_model.py --tune        # With Optuna tuning
    python3 scripts/train_q4_model.py --eval-only   # Evaluate existing model
"""

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Paths ─────────────────────────────────────────────────────────────────

# Import shared base path from quarter_scoring_analysis
sys.path.insert(0, str(Path(__file__).parent))
from quarter_scoring_analysis import load_quarter_data, ANALYSIS_OUTPUT_BASE

MODEL_DIR = ANALYSIS_OUTPUT_BASE / "models"
MODEL_PATH = MODEL_DIR / "q4_total_model.joblib"

# ── Linear baseline (from quarter_scoring_analysis.py regression) ─────────
LINEAR_COEF = 1.0604
LINEAR_INTERCEPT = 44.34


# ── Feature Engineering ───────────────────────────────────────────────────

def compute_team_q4_rolling(qdf: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team rolling Q4 tendency features, shifted to prevent leakage.

    For each team-game, computes rolling averages of that team's Q4 scoring
    and Q4 vs pace differential. Shift(1) ensures we only use data available
    before the game starts.
    """
    # Build a long-form team-game table (each game appears twice: home + away)
    home = qdf[["game_id", "game_date", "home_team", "home_q4",
                 "home_q1", "home_q2", "home_q3"]].copy()
    home.columns = ["game_id", "game_date", "team", "q4", "q1", "q2", "q3"]
    home["venue"] = "home"

    away = qdf[["game_id", "game_date", "away_team", "away_q4",
                 "away_q1", "away_q2", "away_q3"]].copy()
    away.columns = ["game_id", "game_date", "team", "q4", "q1", "q2", "q3"]
    away["venue"] = "away"

    tg = pd.concat([home, away]).sort_values(["team", "game_date"]).reset_index(drop=True)
    tg["thru_3q_per_q"] = (tg["q1"] + tg["q2"] + tg["q3"]) / 3.0
    tg["q4_vs_pace"] = tg["q4"] - tg["thru_3q_per_q"]

    # Rolling averages per team, shifted 1 to prevent leakage
    for col, prefix in [("q4", "q4"), ("q4_vs_pace", "q4_vs_pace")]:
        for window, suffix in [(5, "avg5"), (10, "avg10")]:
            tg[f"{prefix}_{suffix}"] = (
                tg.groupby("team")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
            )

    # Split back into home and away
    home_feats = tg[tg["venue"] == "home"][
        ["game_id", "q4_avg5", "q4_avg10", "q4_vs_pace_avg5", "q4_vs_pace_avg10"]
    ].rename(columns=lambda c: f"home_{c}" if c != "game_id" else c)

    away_feats = tg[tg["venue"] == "away"][
        ["game_id", "q4_avg5", "q4_avg10", "q4_vs_pace_avg5", "q4_vs_pace_avg10"]
    ].rename(columns=lambda c: f"away_{c}" if c != "game_id" else c)

    return home_feats, away_feats


def build_team_q4_snapshot(qdf: pd.DataFrame) -> dict:
    """Build a snapshot of each team's latest Q4 tendency features.

    Returns dict: {team_tricode: {feature: value, ...}, ...}
    Used by the live scanner to look up team tendencies without re-running
    the full pipeline.
    """
    home = qdf[["game_date", "home_team", "home_q4",
                 "home_q1", "home_q2", "home_q3"]].copy()
    home.columns = ["game_date", "team", "q4", "q1", "q2", "q3"]

    away = qdf[["game_date", "away_team", "away_q4",
                 "away_q1", "away_q2", "away_q3"]].copy()
    away.columns = ["game_date", "team", "q4", "q1", "q2", "q3"]

    tg = pd.concat([home, away]).sort_values(["team", "game_date"]).reset_index(drop=True)
    tg["thru_3q_per_q"] = (tg["q1"] + tg["q2"] + tg["q3"]) / 3.0
    tg["q4_vs_pace"] = tg["q4"] - tg["thru_3q_per_q"]

    snapshot = {}
    for team, grp in tg.groupby("team"):
        last_games = grp.tail(10)
        snapshot[team] = {
            "q4_avg5": float(last_games.tail(5)["q4"].mean()),
            "q4_avg10": float(last_games["q4"].mean()),
            "q4_vs_pace_avg5": float(last_games.tail(5)["q4_vs_pace"].mean()),
            "q4_vs_pace_avg10": float(last_games["q4_vs_pace"].mean()),
        }
    return snapshot


def build_features(qdf: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix for Q4 total prediction.

    All features are available at end of Q3.
    """
    # Filter to regulation-only games
    reg = qdf[~qdf["has_ot"]].copy()
    print(f"  Regulation games: {len(reg):,}")

    # ── In-game scoring features ──────────────────────────────────────
    reg["thru_3q_per_q"] = reg["thru_3q_total"] / 3.0

    # Pace change: Q3 rate vs first-half per-Q average
    reg["h1_per_q"] = reg["h1_total"] / 2.0
    reg["q3_vs_h1_pace"] = reg["q3_total"] - reg["h1_per_q"]

    # Margin at end of 3Q
    reg["margin_3q"] = (
        (reg["home_q1"] + reg["home_q2"] + reg["home_q3"])
        - (reg["away_q1"] + reg["away_q2"] + reg["away_q3"])
    )
    reg["abs_margin_3q"] = reg["margin_3q"].abs()
    reg["is_blowout_15"] = (reg["abs_margin_3q"] >= 15).astype(int)
    reg["is_blowout_21"] = (reg["abs_margin_3q"] >= 21).astype(int)

    # ── Team Q4 tendencies (rolling, shifted) ─────────────────────────
    home_feats, away_feats = compute_team_q4_rolling(qdf)
    reg = reg.merge(home_feats, on="game_id", how="left")
    reg = reg.merge(away_feats, on="game_id", how="left")

    return reg


# ── Feature list ──────────────────────────────────────────────────────────

FEATURE_COLS = [
    # In-game scoring (all available from live scoreboard)
    "thru_3q_total", "q1_total", "q2_total", "q3_total", "h1_total", "thru_3q_per_q",
    # Pace change
    "q3_vs_h1_pace",
    # Blowout
    "margin_3q", "abs_margin_3q", "is_blowout_15", "is_blowout_21",
    # Team Q4 tendency (from team_q4_snapshot, available live)
    "home_q4_avg5", "home_q4_avg10", "away_q4_avg5", "away_q4_avg10",
    "home_q4_vs_pace_avg5", "away_q4_vs_pace_avg5",
]

TARGET_COL = "game_total"


# ── Chronological split ───────────────────────────────────────────────────

def chron_split(df: pd.DataFrame, frac: float = 0.8):
    """Time-ordered train/test split."""
    df = df.sort_values("game_date").reset_index(drop=True)
    cut = int(math.floor(len(df) * frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def time_series_cv_folds(df: pd.DataFrame, n_folds: int = 5):
    """Generate chronological CV fold indices."""
    df = df.sort_values("game_date").reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_folds + 1)

    folds = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        test_end = min(train_end + fold_size, n)
        train_idx = df.index[:train_end]
        test_idx = df.index[train_end:test_end]
        if len(test_idx) > 0:
            folds.append((train_idx.tolist(), test_idx.tolist()))
    return folds


# ── Training ──────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, tune: bool = False) -> dict:
    """Train XGBoost regressor for final total prediction.

    Returns artifact dict ready for joblib serialization.
    """
    # Ensure features exist
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"  WARNING: Missing features (will be NaN): {missing}")

    X = df[available_features].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # Imputer for NaN features (market data, early-season rolling)
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Chronological train/test split
    train_df, test_df = chron_split(df)
    X_train = imputer.transform(train_df[available_features].values.astype(np.float32))
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    X_test = imputer.transform(test_df[available_features].values.astype(np.float32))
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    # Model parameters
    if tune:
        params = _tune_hyperparams(X_train, y_train, df, available_features, imputer)
    else:
        params = {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
        }

    print(f"\n  Training XGBoost with params: {params}")
    model = XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)

    # ── Evaluate on held-out test set ─────────────────────────────────
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Linear baseline comparison
    baseline_pred = LINEAR_COEF * test_df["thru_3q_total"].values + LINEAR_INTERCEPT
    baseline_mae = mean_absolute_error(y_test, baseline_pred)

    print(f"\n  ── Test Set Evaluation ({len(test_df):,} games) ──")
    print(f"  XGBoost MAE:  {mae:.2f}")
    print(f"  XGBoost RMSE: {rmse:.2f}")
    print(f"  XGBoost R²:   {r2:.4f}")
    print(f"  Linear MAE:   {baseline_mae:.2f}")
    print(f"  Improvement:  {baseline_mae - mae:+.2f} pts MAE")

    # ── Bootstrap CI of MAE delta ─────────────────────────────────────
    ml_errors = np.abs(y_test - y_pred)
    lin_errors = np.abs(y_test - baseline_pred)
    rng = np.random.default_rng(42)
    n_boot = 2000
    deltas = np.empty(n_boot)
    n_test = len(y_test)
    for b in range(n_boot):
        idx = rng.integers(0, n_test, size=n_test)
        deltas[b] = lin_errors[idx].mean() - ml_errors[idx].mean()
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    ml_wins_bootstrap = (deltas > 0).mean()

    print(f"\n  ── Bootstrap CI (MAE delta: linear - XGBoost) ──")
    print(f"  Mean delta:   {deltas.mean():+.3f} pts (positive = ML better)")
    print(f"  95% CI:       [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"  P(ML better): {ml_wins_bootstrap:.1%}")

    ml_robust = ci_lo > 0  # CI entirely above zero = ML robustly better

    # ── Month-by-month stability ──────────────────────────────────────
    test_months = test_df["game_date"].dt.to_period("M")
    print(f"\n  ── Month-by-Month Stability ──")
    print(f"  {'Month':<10} {'N':>5} {'ML MAE':>8} {'Lin MAE':>8} {'Delta':>8}")
    print(f"  {'-'*42}")
    for month, idx in test_df.groupby(test_months).groups.items():
        m_ml = ml_errors[idx.values - test_df.index[0]].mean()
        m_lin = lin_errors[idx.values - test_df.index[0]].mean()
        print(f"  {str(month):<10} {len(idx):>5} {m_ml:>8.2f} {m_lin:>8.2f} {m_lin - m_ml:>+8.2f}")

    if not ml_robust:
        print(f"\n  WARNING: ML does not robustly beat linear (CI includes 0).")
        print(f"  Model will be saved but artifact flags use_ml=False.")

    # ── Conditional residual std by margin bucket ─────────────────────
    residuals = y_test - y_pred
    test_margins = test_df["abs_margin_3q"].values

    cond_std = {}
    for label, lo, hi in [("close_0_10", 0, 10), ("moderate_11_20", 11, 20), ("blowout_21_plus", 21, 200)]:
        mask = (test_margins >= lo) & (test_margins <= hi)
        bucket_resid = residuals[mask]
        if len(bucket_resid) >= 10:
            cond_std[label] = float(np.std(bucket_resid))
        else:
            cond_std[label] = float(np.std(residuals))  # fallback

    overall_std = float(np.std(residuals))
    print(f"\n  ── Conditional Std (by 3Q margin) ──")
    for label, std_val in cond_std.items():
        print(f"  {label:>20}: {std_val:.2f}")
    print(f"  {'overall':>20}: {overall_std:.2f}")

    # ── Retrain on full data for final model ──────────────────────────
    print(f"\n  Retraining on full dataset ({len(df):,} games)...")
    X_full = imputer.fit_transform(df[available_features].values.astype(np.float32))
    y_full = df[TARGET_COL].values.astype(np.float32)
    model.fit(X_full, y_full)

    # Recompute conditional std on full OOF residuals via time-series CV
    print("  Computing OOF residuals for conditional std...")
    oof_residuals = np.full(len(df), np.nan)
    oof_margins = df["abs_margin_3q"].values
    folds = time_series_cv_folds(df, n_folds=5)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        fold_X_tr = imputer.transform(df.iloc[train_idx][available_features].values.astype(np.float32))
        fold_y_tr = df.iloc[train_idx][TARGET_COL].values.astype(np.float32)
        fold_X_te = imputer.transform(df.iloc[test_idx][available_features].values.astype(np.float32))
        fold_y_te = df.iloc[test_idx][TARGET_COL].values.astype(np.float32)

        fold_model = XGBRegressor(**params, random_state=42)
        fold_model.fit(fold_X_tr, fold_y_tr)
        fold_pred = fold_model.predict(fold_X_te)
        oof_residuals[test_idx] = fold_y_te - fold_pred

    valid_mask = ~np.isnan(oof_residuals)
    oof_r = oof_residuals[valid_mask]
    oof_m = oof_margins[valid_mask]

    cond_std_final = {}
    for label, lo, hi in [("close_0_10", 0, 10), ("moderate_11_20", 11, 20), ("blowout_21_plus", 21, 200)]:
        mask = (oof_m >= lo) & (oof_m <= hi)
        bucket_resid = oof_r[mask]
        if len(bucket_resid) >= 20:
            cond_std_final[label] = float(np.std(bucket_resid))
        else:
            cond_std_final[label] = float(np.std(oof_r))

    overall_std_final = float(np.std(oof_r))

    print(f"\n  ── Final Conditional Std (OOF, {len(oof_r):,} games) ──")
    for label, std_val in cond_std_final.items():
        print(f"  {label:>20}: {std_val:.2f}")
    print(f"  {'overall':>20}: {overall_std_final:.2f}")

    # ── Feature importance ────────────────────────────────────────────
    importances = model.feature_importances_
    feat_imp = sorted(zip(available_features, importances), key=lambda x: -x[1])
    print(f"\n  ── Feature Importance (top 10) ──")
    for fname, imp in feat_imp[:10]:
        print(f"  {fname:>30}: {imp:.4f}")

    # ── Team Q4 snapshot ──────────────────────────────────────────────
    # Re-load full quarter data (including OT games) for snapshot
    qdf_full = load_quarter_data()
    team_snapshot = build_team_q4_snapshot(qdf_full)

    # ── Q4 elapsed-time uncertainty curve ─────────────────────────────
    # Empirically measure how much uncertainty remains at each fraction of Q4.
    # At fraction f, we know thru_3q + f*q4_actual; remaining is (1-f)*q4_actual.
    # Std of remaining scoring decreases as Q4 progresses.
    print(f"\n  ── Q4 Elapsed-Time Std Curve ──")
    q4_elapsed_std = _compute_q4_elapsed_std(df)

    # ── Probability calibration check ─────────────────────────────────
    print(f"\n  ── Probability Calibration (strikes 200-260) ──")
    _calibration_check(oof_residuals, oof_margins, df, cond_std_final, overall_std_final)

    artifact = {
        "model": model,
        "imputer": imputer,
        "features": available_features,
        "conditional_std": cond_std_final,
        "overall_std": overall_std_final,
        "q4_elapsed_std": q4_elapsed_std,
        "team_q4_snapshot": team_snapshot,
        "linear_fallback": {"coef": LINEAR_COEF, "intercept": LINEAR_INTERCEPT},
        "use_ml": ml_robust,
        "eval_metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mae_vs_baseline": float(baseline_mae - mae),
            "bootstrap_ci_lo": float(ci_lo),
            "bootstrap_ci_hi": float(ci_hi),
            "p_ml_better": float(ml_wins_bootstrap),
        },
        "trained_at": datetime.now().isoformat(),
        "n_games": len(df),
    }

    return artifact


def _calibration_check(oof_residuals, oof_margins, df, cond_std, overall_std):
    """Check probability calibration for Kalshi-style strikes."""
    valid_mask = ~np.isnan(oof_residuals)
    valid_df = df[valid_mask].copy()
    valid_r = oof_residuals[valid_mask]
    valid_m = oof_margins[valid_mask]

    # Reconstruct OOF predictions
    oof_pred = valid_df[TARGET_COL].values - valid_r

    # Assign conditional std to each game
    stds = np.full(len(valid_df), overall_std)
    for label, lo, hi in [("close_0_10", 0, 10), ("moderate_11_20", 11, 20), ("blowout_21_plus", 21, 200)]:
        mask = (valid_m >= lo) & (valid_m <= hi)
        stds[mask] = cond_std.get(label, overall_std)

    actuals = valid_df[TARGET_COL].values

    print(f"  {'Strike':>8} {'P(over)':>8} {'Empirical':>10} {'Gap':>8} {'N_over':>8}")
    print(f"  {'-'*46}")
    for strike in range(200, 265, 5):
        model_p_over = norm.sf(strike, loc=oof_pred, scale=stds).mean() * 100
        empirical_over = (actuals > strike).mean() * 100
        n_over = (actuals > strike).sum()
        gap = model_p_over - empirical_over
        marker = " !" if abs(gap) > 5 else ""
        print(f"  {strike:>8} {model_p_over:>7.1f}% {empirical_over:>9.1f}% "
              f"{gap:>+7.1f}% {n_over:>8}{marker}")


def _compute_q4_elapsed_std(df: pd.DataFrame) -> dict:
    """Compute empirical std of remaining scoring at each Q4 elapsed fraction.

    At fraction f of Q4, we know: thru_3q + f * q4_actual.
    The unknown remaining is: (1-f) * q4_actual.
    We measure the std of (game_total - known) across all games for each bucket.

    Returns dict mapping fraction labels to std values, e.g.:
        {"0.00": 9.7, "0.17": 8.1, "0.33": 6.5, ...}
    """
    q4_totals = df["q4_total"].values if "q4_total" in df.columns else (df["game_total"] - df["thru_3q_total"]).values
    game_totals = df[TARGET_COL].values
    thru_3q = df["thru_3q_total"].values

    fractions = [0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0]
    labels = ["0.00", "0.17", "0.33", "0.50", "0.67", "0.83", "1.00"]

    result = {}
    for frac, label in zip(fractions, labels):
        # Known at this point: thru_3q + frac * q4_actual
        known = thru_3q + frac * q4_totals
        remaining = game_totals - known  # what's still unknown
        result[label] = float(np.std(remaining))

    print(f"  {'Fraction':>10} {'Remaining Std':>14}")
    print(f"  {'-'*26}")
    for label in labels:
        print(f"  {label:>10} {result[label]:>13.2f}")

    return result


def _tune_hyperparams(X_train, y_train, df, features, imputer) -> dict:
    """Optuna hyperparameter tuning with time-series CV."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  WARNING: optuna not installed, using defaults")
        return {
            "n_estimators": 300, "max_depth": 4, "learning_rate": 0.04,
            "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0,
        }

    # Use the training portion for CV
    train_portion = df.iloc[:len(X_train)]
    folds = time_series_cv_folds(train_portion, n_folds=4)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        maes = []
        for train_idx, test_idx in folds:
            fold_X_tr = imputer.transform(
                train_portion.iloc[train_idx][features].values.astype(np.float32))
            fold_y_tr = train_portion.iloc[train_idx][TARGET_COL].values.astype(np.float32)
            fold_X_te = imputer.transform(
                train_portion.iloc[test_idx][features].values.astype(np.float32))
            fold_y_te = train_portion.iloc[test_idx][TARGET_COL].values.astype(np.float32)

            m = XGBRegressor(**params, random_state=42)
            m.fit(fold_X_tr, fold_y_tr)
            pred = m.predict(fold_X_te)
            maes.append(mean_absolute_error(fold_y_te, pred))
        return np.mean(maes)

    print(f"  Running Optuna tuning (50 trials, {len(folds)} CV folds)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best = study.best_params
    print(f"  Best MAE: {study.best_value:.3f}")
    print(f"  Best params: {best}")
    return best


# ── Evaluation-only mode ──────────────────────────────────────────────────

def evaluate_existing():
    """Load and evaluate an existing model artifact."""
    if not MODEL_PATH.exists():
        print(f"ERROR: No model found at {MODEL_PATH}")
        print("  Run: python3 scripts/train_q4_model.py  (to train first)")
        sys.exit(1)

    artifact = joblib.load(MODEL_PATH)
    print(f"  Model loaded from {MODEL_PATH}")
    print(f"  Trained at: {artifact['trained_at']}")
    print(f"  N games: {artifact['n_games']:,}")
    print(f"  Features: {len(artifact['features'])}")
    print(f"\n  ── Stored Metrics ──")
    for k, v in artifact["eval_metrics"].items():
        print(f"  {k:>20}: {v:.4f}")
    print(f"\n  ── Conditional Std ──")
    for k, v in artifact["conditional_std"].items():
        print(f"  {k:>20}: {v:.2f}")
    print(f"  {'overall':>20}: {artifact['overall_std']:.2f}")
    print(f"\n  ── Team Q4 Snapshot (sample) ──")
    teams = sorted(artifact["team_q4_snapshot"].keys())[:5]
    for t in teams:
        vals = artifact["team_q4_snapshot"][t]
        print(f"  {t}: q4_avg5={vals['q4_avg5']:.1f}, q4_avg10={vals['q4_avg10']:.1f}, "
              f"vs_pace_avg5={vals['q4_vs_pace_avg5']:+.1f}")


# ── Main entry point ──────────────────────────────────────────────────────

def train_and_save(tune: bool = False) -> dict:
    """Full training pipeline: load data, build features, train, save.

    Returns the saved artifact dict. Can be called from other scripts
    (e.g. kalshi_q4_edge.py --retrain).
    """
    print("=" * 70)
    print("  Q4 TOTAL POINTS TRAINING PIPELINE")
    print("=" * 70)

    # Load quarter-by-quarter data
    print("\n  Loading quarter data...")
    qdf = load_quarter_data()
    print(f"  Loaded {len(qdf):,} total games")

    # Build features
    print("\n  Building features...")
    df = build_features(qdf)
    print(f"  Feature matrix: {len(df):,} games × {len(FEATURE_COLS)} features")

    # Drop games with no target
    df = df.dropna(subset=[TARGET_COL])

    # Train
    artifact = train_model(df, tune=tune)

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}")
    print(f"  Artifact keys: {list(artifact.keys())}")

    return artifact


def main():
    parser = argparse.ArgumentParser(description="Q4 Total Points Training Pipeline")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate existing model")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_existing()
    else:
        train_and_save(tune=args.tune)


if __name__ == "__main__":
    main()
