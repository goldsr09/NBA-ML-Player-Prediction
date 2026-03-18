"""NBA model evaluation module.

Provides comprehensive evaluation metrics including:
- Brier score & calibration curves
- Against-the-spread (ATS) accuracy
- Over/under accuracy
- Confidence intervals from CV folds
- Market comparison tables
- Profit/loss simulation at various edge thresholds
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (lower is better, 0 = perfect)."""
    return float(brier_score_loss(y_true, y_prob))


def calibration_by_decile(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Compute calibration: predicted vs actual win rate by probability decile."""
    bins = np.linspace(0, 1, n_bins + 1)
    results = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        results.append(
            {
                "bin_lo": round(float(lo), 2),
                "bin_hi": round(float(hi), 2),
                "n_games": n,
                "predicted_mean": round(float(y_prob[mask].mean()), 4),
                "actual_mean": round(float(y_true[mask].mean()), 4),
                "abs_error": round(abs(float(y_prob[mask].mean()) - float(y_true[mask].mean())), 4),
            }
        )
    return results


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE): weighted average of bin calibration errors."""
    cal = calibration_by_decile(y_true, y_prob, n_bins)
    if not cal:
        return np.nan
    total_n = sum(b["n_games"] for b in cal)
    return sum(b["n_games"] * b["abs_error"] for b in cal) / total_n


def ats_accuracy(
    y_margin: np.ndarray, pred_margin: np.ndarray, spread: np.ndarray
) -> dict[str, Any]:
    """Against-the-spread accuracy: how often model picks beat the spread."""
    valid = ~(np.isnan(y_margin) | np.isnan(pred_margin) | np.isnan(spread))
    if valid.sum() == 0:
        return {"n": 0, "accuracy": np.nan}
    y_m = y_margin[valid]
    p_m = pred_margin[valid]
    s = spread[valid]

    # Model picks home to cover when predicted margin > -spread
    model_picks_home_cover = p_m > -s
    actual_home_covered = y_m > -s

    # Exclude pushes (exact spread match)
    pushes = np.isclose(y_m, -s, atol=0.25)
    non_push = ~pushes
    n_valid = int(non_push.sum())

    if n_valid == 0:
        return {"n": 0, "accuracy": np.nan, "pushes": int(pushes.sum())}

    correct = (model_picks_home_cover[non_push] == actual_home_covered[non_push]).sum()
    return {
        "n": n_valid,
        "accuracy": round(float(correct / n_valid), 4),
        "pushes": int(pushes.sum()),
        "breakeven": 0.524,
    }


def over_under_accuracy(
    y_total: np.ndarray, pred_total: np.ndarray, market_total: np.ndarray
) -> dict[str, Any]:
    """Over/under accuracy: how often model correctly predicts over or under market total."""
    valid = ~(np.isnan(y_total) | np.isnan(pred_total) | np.isnan(market_total))
    if valid.sum() == 0:
        return {"n": 0, "accuracy": np.nan}
    y_t = y_total[valid]
    p_t = pred_total[valid]
    m_t = market_total[valid]

    model_picks_over = p_t > m_t
    actual_over = y_t > m_t

    pushes = np.isclose(y_t, m_t, atol=0.25)
    non_push = ~pushes
    n_valid = int(non_push.sum())
    if n_valid == 0:
        return {"n": 0, "accuracy": np.nan, "pushes": int(pushes.sum())}

    correct = (model_picks_over[non_push] == actual_over[non_push]).sum()
    return {
        "n": n_valid,
        "accuracy": round(float(correct / n_valid), 4),
        "pushes": int(pushes.sum()),
        "breakeven": 0.524,
    }


def cv_confidence_interval(
    scores: list[float], confidence: float = 0.95
) -> dict[str, float]:
    """Compute mean, std, and confidence interval from CV fold scores."""
    arr = np.array(scores)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    n = len(arr)
    # Use t-distribution approximation for small n
    from scipy import stats as sp_stats

    if n > 1:
        t_val = sp_stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin = t_val * std / np.sqrt(n)
    else:
        margin = 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_lo": round(mean - margin, 4),
        "ci_hi": round(mean + margin, 4),
        "n_folds": n,
    }


def market_comparison_table(
    model_metrics: dict[str, Any],
    market_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compare model vs market across key metrics."""
    comparison = {}
    for metric in ["accuracy", "auc", "log_loss", "mae", "rmse", "r2"]:
        model_val = model_metrics.get(metric)
        market_val = market_metrics.get(metric)
        if model_val is not None and market_val is not None:
            higher_better = metric in ("accuracy", "auc", "r2")
            diff = model_val - market_val
            if higher_better:
                better = "model" if diff > 0 else "market"
            else:
                better = "model" if diff < 0 else "market"
            comparison[metric] = {
                "model": round(float(model_val), 4),
                "market": round(float(market_val), 4),
                "diff": round(float(diff), 4),
                "better": better,
            }
    return comparison


def profit_loss_simulation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    market_prob: np.ndarray,
    edge_thresholds: list[float] | None = None,
    bet_size: float = 100.0,
    vig_factor: float = 0.9524,  # -110 line = 1/1.05 ≈ 0.9524 net payout
) -> list[dict[str, Any]]:
    """Simulate profit/loss at various edge thresholds.

    For each threshold, bet on home when model_prob - market_prob > threshold,
    or bet on away when market_prob - model_prob > threshold.

    Args:
        y_true: actual binary outcomes (1 = home win)
        y_prob: model predicted probabilities
        market_prob: market implied probabilities
        edge_thresholds: list of minimum edges to trigger a bet
        bet_size: flat bet size per game
        vig_factor: payout factor accounting for vig (at -110, win pays 0.9524x)
    """
    if edge_thresholds is None:
        edge_thresholds = [0.0, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]

    valid = ~(np.isnan(y_true) | np.isnan(y_prob) | np.isnan(market_prob))
    y = y_true[valid]
    p_model = y_prob[valid]
    p_market = market_prob[valid]

    results = []
    for threshold in edge_thresholds:
        edge_home = p_model - p_market
        edge_away = (1 - p_model) - (1 - p_market)  # same magnitude, opposite sign

        # Bet on home when edge > threshold
        bet_home = edge_home > threshold
        # Bet on away when edge_away > threshold (i.e., edge_home < -threshold)
        bet_away = edge_home < -threshold

        n_bets = int(bet_home.sum() + bet_away.sum())
        if n_bets == 0:
            results.append(
                {
                    "edge_threshold": threshold,
                    "n_bets": 0,
                    "n_wins": 0,
                    "win_rate": 0.0,
                    "profit": 0.0,
                    "accuracy_pct": 0.0,
                }
            )
            continue

        # Home bets: win when y=1
        home_wins = int((bet_home & (y == 1)).sum())
        home_losses = int((bet_home & (y == 0)).sum())
        # Away bets: win when y=0
        away_wins = int((bet_away & (y == 0)).sum())
        away_losses = int((bet_away & (y == 1)).sum())

        total_wins = home_wins + away_wins
        total_losses = home_losses + away_losses
        profit = total_wins * bet_size * vig_factor - total_losses * bet_size

        results.append(
            {
                "edge_threshold": threshold,
                "n_bets": n_bets,
                "n_wins": total_wins,
                "win_rate": round(float(total_wins / n_bets), 4) if n_bets else 0.0,
                "profit": round(float(profit), 2),
                "accuracy_pct": round(float(100 * profit / (n_bets * bet_size)), 2) if n_bets else 0.0,
                "avg_edge": round(
                    float(
                        np.concatenate(
                            [edge_home[bet_home], -edge_home[bet_away]]
                        ).mean()
                    ),
                    4,
                )
                if n_bets
                else 0.0,
            }
        )
    return results


def evaluate_win_model_comprehensive(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    market_prob: np.ndarray | None = None,
    spread: np.ndarray | None = None,
    pred_margin: np.ndarray | None = None,
    y_margin: np.ndarray | None = None,
) -> dict[str, Any]:
    """Comprehensive evaluation for win probability model."""
    pred = (y_prob >= 0.5).astype(int)
    result: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
        "brier_score": round(brier_score(y_true, y_prob), 4),
        "calibration_error": round(calibration_error(y_true, y_prob), 4),
        "calibration_by_decile": calibration_by_decile(y_true, y_prob),
        "n_games": int(len(y_true)),
    }

    if market_prob is not None:
        valid_market = np.isfinite(market_prob) & np.isfinite(y_true.astype(float))
        if valid_market.sum() > 0:
            y_true_m = y_true[valid_market]
            y_prob_m = y_prob[valid_market]
            market_prob_m = market_prob[valid_market]
            market_pred = (market_prob_m >= 0.5).astype(int)
            result["market_comparison"] = market_comparison_table(
                result,
                {
                    "accuracy": accuracy_score(y_true_m, market_pred),
                    "auc": roc_auc_score(y_true_m, market_prob_m),
                    "log_loss": log_loss(y_true_m, np.clip(market_prob_m, 1e-6, 1 - 1e-6)),
                },
            )
            result["profit_loss"] = profit_loss_simulation(y_true_m, y_prob_m, market_prob_m)

    if spread is not None and pred_margin is not None and y_margin is not None:
        result["ats"] = ats_accuracy(y_margin, pred_margin, spread)

    return result


def evaluate_total_model_comprehensive(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    market_total: np.ndarray | None = None,
) -> dict[str, Any]:
    """Comprehensive evaluation for total points model."""
    result: dict[str, Any] = {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "n_games": int(len(y_true)),
    }

    if market_total is not None:
        valid_market = np.isfinite(market_total) & np.isfinite(y_true.astype(float)) & np.isfinite(y_pred.astype(float))
        if valid_market.sum() > 0:
            y_true_m = y_true[valid_market]
            y_pred_m = y_pred[valid_market]
            market_total_m = market_total[valid_market]
            result["over_under"] = over_under_accuracy(y_true_m, y_pred_m, market_total_m)
            result["market_comparison"] = market_comparison_table(
                result,
                {
                    "mae": mean_absolute_error(y_true_m, market_total_m),
                    "rmse": np.sqrt(mean_squared_error(y_true_m, market_total_m)),
                    "r2": r2_score(y_true_m, market_total_m),
                },
            )

    return result


def evaluate_cv_folds(
    fold_results: list[dict[str, Any]], metrics: list[str] | None = None
) -> dict[str, Any]:
    """Aggregate metrics across CV folds with confidence intervals."""
    if metrics is None:
        metrics = ["accuracy", "auc", "log_loss", "brier_score", "mae", "rmse", "r2"]

    summary: dict[str, Any] = {"n_folds": len(fold_results)}
    for metric in metrics:
        scores = [f[metric] for f in fold_results if metric in f and f[metric] is not None]
        if scores:
            summary[metric] = cv_confidence_interval(scores)
    return summary


def print_evaluation_report(
    win_eval: dict[str, Any] | None = None,
    total_eval: dict[str, Any] | None = None,
    cv_summary: dict[str, Any] | None = None,
    label: str = "",
) -> None:
    """Print formatted evaluation report."""
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    if win_eval:
        print("\n--- Win Model ---")
        print(f"  Accuracy: {win_eval['accuracy']:.4f}")
        print(f"  AUC: {win_eval['auc']:.4f}")
        print(f"  Log Loss: {win_eval['log_loss']:.4f}")
        print(f"  Brier Score: {win_eval['brier_score']:.4f}")
        print(f"  Calibration Error: {win_eval['calibration_error']:.4f}")
        print(f"  N games: {win_eval['n_games']}")

        if "ats" in win_eval and win_eval["ats"].get("n", 0) > 0:
            ats = win_eval["ats"]
            print(f"  ATS Accuracy: {ats['accuracy']:.4f} (n={ats['n']}, breakeven={ats['breakeven']})")

        if "market_comparison" in win_eval:
            print("  vs Market:")
            for k, v in win_eval["market_comparison"].items():
                print(f"    {k}: model={v['model']:.4f} market={v['market']:.4f} ({v['better']})")

        if "profit_loss" in win_eval:
            print("  P/L Simulation:")
            for pl in win_eval["profit_loss"]:
                if pl["n_bets"] > 0:
                    print(
                        f"    Edge>={pl['edge_threshold']:.0%}: {pl['n_bets']} bets, "
                        f"win={pl['win_rate']:.1%}, Accuracy={pl['accuracy_pct']:+.1f}%"
                    )

    if total_eval:
        print("\n--- Total Model ---")
        print(f"  MAE: {total_eval['mae']:.2f}")
        print(f"  RMSE: {total_eval['rmse']:.2f}")
        print(f"  R2: {total_eval['r2']:.4f}")
        print(f"  N games: {total_eval['n_games']}")

        if "over_under" in total_eval and total_eval["over_under"].get("n", 0) > 0:
            ou = total_eval["over_under"]
            print(f"  O/U Accuracy: {ou['accuracy']:.4f} (n={ou['n']}, breakeven={ou['breakeven']})")

        if "market_comparison" in total_eval:
            print("  vs Market:")
            for k, v in total_eval["market_comparison"].items():
                print(f"    {k}: model={v['model']:.4f} market={v['market']:.4f} ({v['better']})")

    if cv_summary:
        print("\n--- Cross-Validation Summary ---")
        print(f"  Folds: {cv_summary['n_folds']}")
        for metric, vals in cv_summary.items():
            if isinstance(vals, dict) and "mean" in vals:
                print(
                    f"  {metric}: {vals['mean']:.4f} +/- {vals['std']:.4f} "
                    f"[{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]"
                )


# ---------------------------------------------------------------------------
# Player Prop Calibration Utilities (Phase 2)
# ---------------------------------------------------------------------------


def prop_brier_score(
    p_hit: np.ndarray,
    hit: np.ndarray,
) -> float:
    """Brier score for player predictions: mean((p_hit - hit)^2).

    Args:
        p_hit: predicted probability of the outcome (P(over) for OVER bets, P(under) for UNDER).
        hit: binary actual outcome (1=hit, 0=miss).
    """
    valid = ~(np.isnan(p_hit) | np.isnan(hit))
    if valid.sum() == 0:
        return np.nan
    return float(np.mean((p_hit[valid] - hit[valid]) ** 2))


def prop_calibration_by_bucket(
    df: pd.DataFrame,
    group_col: str,
    min_sample: int = 50,
) -> list[dict[str, Any]]:
    """Compute calibration metrics for player predictions grouped by a column.

    ``df`` must have columns: ``p_hit``, ``hit``, ``pnl``, and ``group_col``.

    Returns a list of dicts, one per group, with:
      - group: the group value
      - n: sample size
      - hit_rate: actual hit rate
      - mean_p_hit: mean predicted probability
      - gap: |hit_rate - mean_p_hit| (miscalibration)
      - brier: Brier score for the group
      - log_loss: log-loss for the group
      - accuracy_pct: Accuracy%
    """
    results = []
    for group_val, grp in df.groupby(group_col):
        if len(grp) < min_sample:
            continue
        hit_arr = grp["hit"].to_numpy(dtype=float)
        p_arr = grp["p_hit"].to_numpy(dtype=float)
        valid = ~(np.isnan(hit_arr) | np.isnan(p_arr))
        if valid.sum() < min_sample:
            continue
        h = hit_arr[valid]
        p = p_arr[valid]
        hr = float(h.mean())
        mp = float(p.mean())
        brier = float(np.mean((p - h) ** 2))
        # log loss with clipping
        p_clip = np.clip(p, 1e-6, 1 - 1e-6)
        ll = float(-np.mean(h * np.log(p_clip) + (1 - h) * np.log(1 - p_clip)))
        if "signal" in grp.columns:
            allocation_mask = grp["signal"] != "LOW CONFIDENCE"
        else:
            allocation_mask = grp["pnl"].notna()
        n_allocations = int(allocation_mask.sum())
        total_allocated = n_allocations * 100.0
        pnl_sum = float(grp.loc[allocation_mask, "pnl"].sum()) if n_allocations > 0 else 0.0
        accuracy = float(pnl_sum / total_allocated * 100) if total_allocated > 0 else 0.0
        results.append({
            "group": group_val,
            "n": int(valid.sum()),
            "hit_rate": round(hr, 4),
            "mean_p_hit": round(mp, 4),
            "gap": round(abs(hr - mp), 4),
            "brier": round(brier, 4),
            "log_loss": round(ll, 4),
            "accuracy_pct": round(accuracy, 2),
        })
    return results
