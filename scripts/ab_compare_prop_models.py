#!/usr/bin/env python3
"""Fixed-window A/B evaluation for player prop model configs.

By default this compares:
  - baseline: current feature-cache version, full feature set, no tuned params
  - candidate: current feature-cache version, tuned params + selected groups (if valid)

It can also compare two explicit feature-cache versions (for example v15 vs v16)
on the same historical date window, which is the right way to isolate feature
engineering changes from tuning changes.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import predict_player_props as props


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis" / "output"
PREDICTIONS_DIR = OUT_DIR / "predictions"
MODEL_DIR = OUT_DIR / "models"


def _versioned_model_file(stem: str, suffix: str, version: str) -> Path:
    return MODEL_DIR / f"{stem}_{version}{suffix}"


def _load_feature_cache_for_version(version: str) -> pd.DataFrame:
    cache_file = _versioned_model_file("player_features_cache", ".pkl", version)
    meta_file = _versioned_model_file("player_features_cache_meta", ".json", version)
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing feature cache for version {version}: {cache_file}")
    df = pd.read_pickle(cache_file)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"Feature cache for version {version} is empty or invalid: {cache_file}")
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text())
            cache_version = str(meta.get("version", ""))
            if cache_version and cache_version != version:
                raise ValueError(
                    f"Feature cache metadata mismatch for {version}: meta says {cache_version}"
                )
        except Exception as exc:
            raise ValueError(f"Could not validate feature cache metadata for {version}: {exc}") from exc
    return df


def _resolve_targets(df: pd.DataFrame) -> list[str]:
    targets = list(props.PROP_TARGETS)
    if "fg3m" in df.columns and df["fg3m"].notna().sum() > 100:
        targets.append("fg3m")
    return targets


def _build_prediction_frame(pred_source: pd.DataFrame, predicted: pd.DataFrame) -> pd.DataFrame:
    pred_df = predicted.copy()
    base_cols = ["game_date_est", "home_team", "away_team", "team", "opp", "player_name", "player_id"]
    for col in base_cols:
        if col not in pred_df.columns and col in pred_source.columns:
            pred_df[col] = pred_source[col]
    if "game_date_est" not in pred_df.columns:
        pred_df["game_date_est"] = pd.to_datetime(
            pred_source["game_time_utc"], utc=True, errors="coerce"
        ).dt.strftime("%Y%m%d")
    pred_df["game_date_est"] = pred_df["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    return pred_df


def _date_key(series_df: pd.DataFrame) -> pd.Series:
    if "game_date_est" in series_df.columns:
        return series_df["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    if "game_time_utc" in series_df.columns:
        return pd.to_datetime(series_df["game_time_utc"], utc=True, errors="coerce").dt.strftime("%Y%m%d")
    return pd.Series("", index=series_df.index, dtype=object)


def _compute_residual_stds(train_df: pd.DataFrame, predicted_train: pd.DataFrame, targets: list[str]) -> dict[str, float]:
    residual_stds: dict[str, float] = {}
    for target in targets:
        pred_col = f"pred_{target}"
        if pred_col not in predicted_train.columns or target not in predicted_train.columns:
            continue
        valid = predicted_train.dropna(subset=[target, pred_col]).copy()
        if valid.empty:
            continue
        residual_stds[target] = float(
            np.std(valid[target].to_numpy(dtype=float) - valid[pred_col].to_numpy(dtype=float))
        )
    return residual_stds


def _match_actual_player_rows(
    date_slice: pd.DataFrame,
    player_name: str,
    team: str,
    player_id: Any,
) -> pd.DataFrame:
    matches = date_slice[date_slice["player_name_norm"].eq(props.normalize_player_name(player_name))]
    if team:
        team_matches = matches[matches["team"].astype(str).eq(str(team))]
        if not team_matches.empty:
            matches = team_matches
    if pd.notna(player_id) and "player_id" in matches.columns:
        id_matches = matches[matches["player_id"].astype(str).eq(str(int(player_id)) if str(player_id).replace(".", "", 1).isdigit() else str(player_id))]
        if not id_matches.empty:
            matches = id_matches
    return matches


def _grade_edges(edges: pd.DataFrame, actual_df: pd.DataFrame, bet_size: float) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    actual = actual_df.copy()
    if "game_date_est" in actual.columns:
        actual["game_date_est"] = actual["game_date_est"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)
    elif "game_time_utc" in actual.columns:
        actual["game_date_est"] = pd.to_datetime(actual["game_time_utc"], utc=True, errors="coerce").dt.strftime("%Y%m%d")
    else:
        actual["game_date_est"] = ""
    actual["player_name_norm"] = actual["player_name"].map(props.normalize_player_name)

    rows: list[dict[str, Any]] = []
    for _, r in edges.iterrows():
        stat = str(r.get("stat_type", ""))
        if stat not in actual.columns:
            continue
        date_slice = actual[actual["game_date_est"] == str(r.get("game_date_est", ""))]
        matches = _match_actual_player_rows(
            date_slice,
            player_name=str(r.get("player_name", "")),
            team=str(r.get("team", "")),
            player_id=r.get("player_id", np.nan),
        )
        if len(matches) != 1:
            continue
        a = matches.iloc[0]
        actual_val = a.get(stat, np.nan)
        if pd.isna(actual_val):
            continue

        line = float(r["prop_line"])
        side = str(r["signal"])
        if abs(float(actual_val) - line) < 1e-9:
            result = "PUSH"
            hit = np.nan
            pnl = 0.0
        elif side == "OVER":
            result = "WIN" if float(actual_val) > line else "LOSS"
            hit = 1 if result == "WIN" else 0
            payout = props._american_odds_to_decimal(r.get("over_odds", np.nan))
            payout = payout if pd.notna(payout) else props.VIG_FACTOR
            pnl = (payout * bet_size) if result == "WIN" else -bet_size
        elif side == "UNDER":
            result = "WIN" if float(actual_val) < line else "LOSS"
            hit = 1 if result == "WIN" else 0
            payout = props._american_odds_to_decimal(r.get("under_odds", np.nan))
            payout = payout if pd.notna(payout) else props.VIG_FACTOR
            pnl = (payout * bet_size) if result == "WIN" else -bet_size
        else:
            result = "LOW CONFIDENCE"
            hit = np.nan
            pnl = 0.0

        open_line = r.get("open_line", np.nan)
        mes_line = np.nan
        if pd.notna(open_line):
            mes_line = (line - float(open_line)) if side == "OVER" else (float(open_line) - line)

        rows.append(
            {
                **r.to_dict(),
                "actual_value": float(actual_val),
                "result": result,
                "hit": hit,
                "pnl": round(float(pnl), 2),
                "mes_line_pts": mes_line,
            }
        )
    return pd.DataFrame(rows)


def _summarize_signals(df: pd.DataFrame, bet_size: float) -> dict[str, Any]:
    if df.empty:
        return {"n_matches": 0, "n_signals": 0, "n_settled": 0}
    sig = df[df["signal"] != "LOW CONFIDENCE"].copy()
    settled = sig[sig["result"] != "PUSH"].copy()
    n_bets = int(len(settled))
    wins = int(settled["hit"].fillna(0).sum()) if n_bets > 0 else 0
    hit_rate = (wins / n_bets) if n_bets > 0 else np.nan
    pnl = float(settled["pnl"].sum()) if n_bets > 0 else 0.0
    accuracy = (100.0 * pnl / (n_bets * bet_size)) if n_bets > 0 else np.nan
    avg_mes = float(settled["mes_line_pts"].dropna().mean()) if settled["mes_line_pts"].notna().any() else np.nan
    per_stat: dict[str, Any] = {}
    for stat, g in settled.groupby("stat_type"):
        stat_bets = len(g)
        stat_wins = int(g["hit"].fillna(0).sum())
        stat_pnl = float(g["pnl"].sum())
        per_stat[stat] = {
            "n_bets": stat_bets,
            "wins": stat_wins,
            "hit_rate": (stat_wins / stat_bets) if stat_bets else np.nan,
            "accuracy_pct": (100.0 * stat_pnl / (stat_bets * bet_size)) if stat_bets else np.nan,
            "pnl": stat_pnl,
        }
    return {
        "n_matches": int(len(df)),
        "n_signals": int(len(sig)),
        "n_settled": n_bets,
        "wins": wins,
        "hit_rate": hit_rate,
        "accuracy_pct": accuracy,
        "pnl": pnl,
        "avg_mes_line_pts": avg_mes,
        "per_stat": per_stat,
    }


def _summarize_raw_opportunities(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    best_side = np.where(
        pd.to_numeric(df["ev_over"], errors="coerce").fillna(-np.inf)
        >= pd.to_numeric(df["ev_under"], errors="coerce").fillna(-np.inf),
        "OVER",
        "UNDER",
    )
    best_ev = np.where(best_side == "OVER", df["ev_over"], df["ev_under"])
    best_p = np.where(best_side == "OVER", df["p_over"], df["p_under"])
    advantage_pct = pd.to_numeric(df["advantage_pct"], errors="coerce").abs().to_numpy(dtype=float)
    best_ev = pd.to_numeric(pd.Series(best_ev), errors="coerce").to_numpy(dtype=float)
    best_p = pd.to_numeric(pd.Series(best_p), errors="coerce").to_numpy(dtype=float)
    over_mask = (
        (best_side == "OVER")
        & np.isfinite(best_ev)
        & (best_ev > props.MIN_EV_BY_SIDE["OVER"])
        & np.isfinite(advantage_pct)
        & (advantage_pct >= props.MIN_ADVANTAGE_PCT_BY_SIDE["OVER"])
        & np.isfinite(best_p)
        & (best_p > props.BREAKEVEN_PROB)
    )
    under_mask = (
        (best_side == "UNDER")
        & np.isfinite(best_ev)
        & (best_ev > props.MIN_EV_BY_SIDE["UNDER"])
        & np.isfinite(advantage_pct)
        & (advantage_pct >= props.MIN_ADVANTAGE_PCT_BY_SIDE["UNDER"])
        & np.isfinite(best_p)
        & (best_p > props.BREAKEVEN_PROB)
    )
    blocked = df["signal_blocked_reason"].fillna("none").value_counts().to_dict() if "signal_blocked_reason" in df.columns else {}
    return {
        "n_rows": int(len(df)),
        "n_positive_ev_over": int((pd.to_numeric(df["ev_over"], errors="coerce") > 0).sum()),
        "n_positive_ev_under": int((pd.to_numeric(df["ev_under"], errors="coerce") > 0).sum()),
        "n_threshold_candidates": int(over_mask.sum() + under_mask.sum()),
        "n_threshold_over": int(over_mask.sum()),
        "n_threshold_under": int(under_mask.sum()),
        "blocked_reasons": blocked,
    }


def _summarize_model_quality(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    per_stat: dict[str, Any] = {}
    for stat, g in df.groupby("stat_type"):
        valid = g.dropna(subset=["actual_value", "pred_value"]).copy()
        if valid.empty:
            continue
        y = valid["actual_value"].to_numpy(dtype=float)
        yhat = valid["pred_value"].to_numpy(dtype=float)
        err = yhat - y
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mean_err = float(np.mean(err))
        denom = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - (np.sum((y - yhat) ** 2) / denom)) if denom > 0 else np.nan
        per_stat[stat] = {
            "n_rows": int(len(valid)),
            "mae": mae,
            "rmse": rmse,
            "mean_error": mean_err,
            "r2": r2,
        }
    return per_stat


def _bootstrap_deltas(paired: pd.DataFrame, bet_size: float, n_bootstrap: int, seed: int) -> dict[str, Any]:
    if paired.empty:
        return {}
    rng = np.random.default_rng(seed)
    metrics = {
        "delta_accuracy_pct": [],
        "delta_hit_rate": [],
        "delta_avg_mes_line_pts": [],
        "delta_n_bets": [],
    }
    for _ in range(n_bootstrap):
        sample = paired.iloc[rng.integers(0, len(paired), len(paired))].copy()
        for side in ["baseline", "candidate"]:
            signal_col = f"{side}_signal"
            result_col = f"{side}_result"
            hit_col = f"{side}_hit"
            pnl_col = f"{side}_pnl"
            mes_col = f"{side}_mes_line_pts"
            settled = sample[(sample[signal_col] != "LOW CONFIDENCE") & (sample[result_col] != "PUSH")].copy()
            n_bets = len(settled)
            wins = float(settled[hit_col].fillna(0).sum()) if n_bets else 0.0
            sample[f"_{side}_n_bets"] = n_bets
            sample[f"_{side}_hit_rate"] = (wins / n_bets) if n_bets else np.nan
            sample[f"_{side}_accuracy"] = (100.0 * float(settled[pnl_col].sum()) / (n_bets * bet_size)) if n_bets else np.nan
            sample[f"_{side}_mes"] = float(settled[mes_col].dropna().mean()) if n_bets and settled[mes_col].notna().any() else np.nan
        metrics["delta_n_bets"].append(float(sample["_candidate_n_bets"].iloc[0] - sample["_baseline_n_bets"].iloc[0]))
        metrics["delta_hit_rate"].append(float(sample["_candidate_hit_rate"].iloc[0] - sample["_baseline_hit_rate"].iloc[0]))
        metrics["delta_accuracy_pct"].append(float(sample["_candidate_accuracy"].iloc[0] - sample["_baseline_accuracy"].iloc[0]))
        metrics["delta_avg_mes_line_pts"].append(float(sample["_candidate_mes"].iloc[0] - sample["_baseline_mes"].iloc[0]))

    out: dict[str, Any] = {}
    for name, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        out[name] = {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
        }
    return out


def _train_bundle(
    train_df: pd.DataFrame,
    player_df_for_signature: pd.DataFrame,
    max_dates: int,
    use_tuned: bool,
    use_selected_groups: bool,
) -> dict[str, Any]:
    targets = _resolve_targets(player_df_for_signature)
    selected_groups = (
        props.load_selected_feature_groups(player_df_for_signature, targets)
        if use_selected_groups else None
    )
    tuned_params = (
        props._load_tuned_params(player_df_for_signature, targets, selected_groups=selected_groups)
        if use_tuned else None
    )
    two_stage = props.train_two_stage_models(
        train_df,
        tuned_params=tuned_params,
        selected_groups=selected_groups,
    )
    single_models: dict[str, Any] = {}
    for target in targets:
        if target in two_stage:
            continue
        features = props.get_effective_feature_list(target, selected_groups=selected_groups)
        feats = props.filter_features(features, train_df)
        if not feats:
            continue
        try:
            imp, model, used_feats = props.train_prop_model(
                train_df, features, target, params=(tuned_params or {}).get(f"xgb_{target}"),
            )
            single_models[target] = (imp, model, used_feats)
        except ValueError:
            continue

    predicted_train = props.predict_two_stage(two_stage, train_df.copy()) if two_stage else train_df.copy()
    for target, (imp, model, feats) in single_models.items():
        predicted_train[f"pred_{target}"] = props.predict_prop(imp, model, feats, train_df)
    residual_stds = _compute_residual_stds(train_df, predicted_train, targets)

    market_residual_models, prob_calibrators, diagnostics = props.train_market_residual_models(
        train_df,
        max_dates=max_dates,
        pretrained_models=(two_stage, single_models),
    )
    return {
        "selected_groups": selected_groups,
        "tuned_params": tuned_params,
        "two_stage": two_stage,
        "single_models": single_models,
        "residual_stds": residual_stds,
        "market_residual_models": market_residual_models,
        "prob_calibrators": prob_calibrators,
        "market_diag": diagnostics,
    }


def _date_based_split(
    player_df: pd.DataFrame,
    test_frac: float,
    max_dates: int,
    fixed_test_dates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    df = player_df.sort_values("game_time_utc").reset_index(drop=True).copy()
    date_series = _date_key(df).astype(str)
    all_dates = sorted(d for d in date_series.dropna().unique() if d)
    if not all_dates:
        raise ValueError("No game dates available in feature cache.")

    if fixed_test_dates is None:
        n_test_dates = max(1, int(round(len(all_dates) * test_frac)))
        test_dates = all_dates[-n_test_dates:]
    else:
        test_dates = [d for d in fixed_test_dates if d in set(all_dates)]
        if not test_dates:
            raise ValueError("None of the requested fixed test dates exist in this feature cache.")

    test_dates = sorted(test_dates)
    if max_dates > 0 and len(test_dates) > max_dates:
        test_dates = test_dates[-max_dates:]

    first_test_date = test_dates[0]
    train = df[date_series < first_test_date].copy()
    test = df[date_series.isin(test_dates)].copy()
    if train.empty or test.empty:
        raise ValueError("Date-based split produced empty train or test set.")
    return train, test, test_dates


def _score_bundle(
    name: str,
    bundle: dict[str, Any],
    test_df: pd.DataFrame,
    prop_lines: pd.DataFrame,
    bet_size: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    predicted = props.predict_two_stage(bundle["two_stage"], test_df.copy()) if bundle["two_stage"] else test_df.copy()
    for target, (imp, model, feats) in bundle["single_models"].items():
        predicted[f"pred_{target}"] = props.predict_prop(imp, model, feats, test_df)

    pred_df = _build_prediction_frame(test_df, predicted)
    edges = props.compute_prediction_advantages(
        pred_df,
        prop_lines,
        bundle["residual_stds"],
        market_residual_models=bundle["market_residual_models"],
        prob_calibrators=bundle["prob_calibrators"],
        uncertainty_models=bundle["two_stage"].get("_uncertainty"),
        quantile_uncertainty_models=bundle["two_stage"].get("_uncertainty_quantile"),
    )
    graded = _grade_edges(edges, test_df, bet_size=bet_size)
    graded["config"] = name
    summary = _summarize_signals(graded, bet_size=bet_size)
    summary["raw_opportunities"] = _summarize_raw_opportunities(edges)
    summary["model_quality"] = _summarize_model_quality(graded)
    return graded, summary


def run_ab_evaluation(
    max_dates: int,
    test_frac: float,
    bet_size: float,
    n_bootstrap: int,
    seed: int,
    baseline_feature_version: str | None = None,
    candidate_feature_version: str | None = None,
    baseline_use_tuned: bool = False,
    candidate_use_tuned: bool = False,
    baseline_use_selected_groups: bool = False,
    candidate_use_selected_groups: bool = False,
) -> dict[str, Any]:
    explicit_versions = baseline_feature_version or candidate_feature_version

    if explicit_versions:
        baseline_version = baseline_feature_version or props.PLAYER_FEATURE_CACHE_VERSION
        candidate_version = candidate_feature_version or props.PLAYER_FEATURE_CACHE_VERSION
        baseline_player_df = _load_feature_cache_for_version(baseline_version)
        candidate_player_df = _load_feature_cache_for_version(candidate_version)
        train, test, test_dates = _date_based_split(
            baseline_player_df,
            test_frac=test_frac,
            max_dates=max_dates,
        )
        candidate_train, candidate_test, _ = _date_based_split(
            candidate_player_df,
            test_frac=test_frac,
            max_dates=max_dates,
            fixed_test_dates=test_dates,
        )
        baseline_label = f"baseline_{baseline_version}"
        candidate_label = f"candidate_{candidate_version}"
    else:
        schedule_df, team_games, player_games = props.build_team_games_and_players(include_historical=True)
        game_odds = props.load_game_odds_lookup(schedule_df)
        ref_features = props.build_referee_game_features(team_games)
        player_df = props.load_or_build_player_features(
            player_games,
            team_games,
            game_odds,
            min_games=props.DEFAULT_MIN_GAMES,
            ref_features=ref_features,
        )
        player_df = player_df.sort_values("game_time_utc").reset_index(drop=True)
        train, test, test_dates = _date_based_split(
            player_df,
            test_frac=test_frac,
            max_dates=max_dates,
        )
        baseline_player_df = player_df
        candidate_player_df = player_df
        candidate_train = train.copy()
        candidate_test = test.copy()
        baseline_label = "baseline"
        candidate_label = "candidate"

    prop_lines = props.load_prop_lines_for_dates(test_dates, fetch_missing=False)
    prop_lines = props._normalize_and_dedupe_prop_lines(prop_lines)

    baseline_bundle = _train_bundle(
        train,
        baseline_player_df,
        max_dates=max_dates,
        use_tuned=baseline_use_tuned,
        use_selected_groups=baseline_use_selected_groups,
    )
    candidate_bundle = _train_bundle(
        candidate_train,
        candidate_player_df,
        max_dates=max_dates,
        use_tuned=candidate_use_tuned,
        use_selected_groups=candidate_use_selected_groups,
    )

    baseline_df, baseline_summary = _score_bundle(baseline_label, baseline_bundle, test, prop_lines, bet_size)
    candidate_df, candidate_summary = _score_bundle(candidate_label, candidate_bundle, candidate_test, prop_lines, bet_size)

    key_cols = ["game_date_est", "team", "player_name", "stat_type", "prop_line"]
    baseline_core = baseline_df[key_cols + ["signal", "result", "hit", "pnl", "mes_line_pts", "pred_value", "ev_over", "ev_under", "confidence"]].copy()
    candidate_core = candidate_df[key_cols + ["signal", "result", "hit", "pnl", "mes_line_pts", "pred_value", "ev_over", "ev_under", "confidence"]].copy()
    paired = baseline_core.merge(
        candidate_core,
        on=key_cols,
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )

    paired = paired.rename(
        columns={
            "signal_baseline": "baseline_signal",
            "signal_candidate": "candidate_signal",
            "result_baseline": "baseline_result",
            "result_candidate": "candidate_result",
            "hit_baseline": "baseline_hit",
            "hit_candidate": "candidate_hit",
            "pnl_baseline": "baseline_pnl",
            "pnl_candidate": "candidate_pnl",
            "mes_line_pts_baseline": "baseline_mes_line_pts",
            "mes_line_pts_candidate": "candidate_mes_line_pts",
            "pred_value_baseline": "baseline_pred_value",
            "pred_value_candidate": "candidate_pred_value",
            "confidence_baseline": "baseline_confidence",
            "confidence_candidate": "candidate_confidence",
        }
    )
    paired["signal_changed"] = paired["baseline_signal"] != paired["candidate_signal"]
    paired["candidate_minus_baseline_pred"] = paired["candidate_pred_value"] - paired["baseline_pred_value"]

    bootstrap = _bootstrap_deltas(paired, bet_size=bet_size, n_bootstrap=n_bootstrap, seed=seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = PREDICTIONS_DIR / f"prop_ab_baseline_{ts}.csv"
    candidate_path = PREDICTIONS_DIR / f"prop_ab_candidate_{ts}.csv"
    paired_path = PREDICTIONS_DIR / f"prop_ab_paired_{ts}.csv"
    summary_path = PREDICTIONS_DIR / f"prop_ab_summary_{ts}.json"
    baseline_df.to_csv(baseline_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    paired.to_csv(paired_path, index=False)

    summary = {
        "config": {
            "max_dates": max_dates,
            "test_frac": test_frac,
            "bet_size": bet_size,
            "n_bootstrap": n_bootstrap,
            "n_eval_dates": len(test_dates),
            "start_date": test_dates[0] if test_dates else None,
            "end_date": test_dates[-1] if test_dates else None,
            "baseline_feature_version": baseline_feature_version,
            "candidate_feature_version": candidate_feature_version,
            "baseline_use_tuned": baseline_use_tuned,
            "candidate_use_tuned": candidate_use_tuned,
            "baseline_use_selected_groups": baseline_use_selected_groups,
            "candidate_use_selected_groups": candidate_use_selected_groups,
        },
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "paired": {
            "n_rows": int(len(paired)),
            "signal_changed": int(paired["signal_changed"].sum()) if not paired.empty else 0,
            "baseline_signals": int((paired["baseline_signal"] != "LOW CONFIDENCE").sum()) if not paired.empty else 0,
            "candidate_signals": int((paired["candidate_signal"] != "LOW CONFIDENCE").sum()) if not paired.empty else 0,
        },
        "bootstrap_deltas": bootstrap,
        "artifacts": {
            "baseline_csv": str(baseline_path),
            "candidate_csv": str(candidate_path),
            "paired_csv": str(paired_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    summary["artifacts"]["summary_json"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B compare player prop model configs on a fixed historical window.")
    p.add_argument("--max-dates", type=int, default=60, help="Number of evaluation dates from the test window.")
    p.add_argument("--test-frac", type=float, default=0.2, help="Chronological holdout fraction.")
    p.add_argument("--bet-size", type=float, default=100.0, help="Flat stake per settled signal.")
    p.add_argument("--bootstrap", type=int, default=500, help="Bootstrap iterations for paired deltas.")
    p.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    p.add_argument("--baseline-feature-version", type=str, default=None, help="Explicit baseline feature-cache version (e.g. v15).")
    p.add_argument("--candidate-feature-version", type=str, default=None, help="Explicit candidate feature-cache version (e.g. v16).")
    p.add_argument("--baseline-use-tuned", action="store_true", help="Use tuned params for the baseline config.")
    p.add_argument("--candidate-use-tuned", action="store_true", help="Use tuned params for the candidate config.")
    p.add_argument("--baseline-use-selected-groups", action="store_true", help="Use selected feature groups for the baseline config.")
    p.add_argument("--candidate-use-selected-groups", action="store_true", help="Use selected feature groups for the candidate config.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_ab_evaluation(
        max_dates=args.max_dates,
        test_frac=args.test_frac,
        bet_size=args.bet_size,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        baseline_feature_version=args.baseline_feature_version,
        candidate_feature_version=args.candidate_feature_version,
        baseline_use_tuned=args.baseline_use_tuned,
        candidate_use_tuned=args.candidate_use_tuned,
        baseline_use_selected_groups=args.baseline_use_selected_groups,
        candidate_use_selected_groups=args.candidate_use_selected_groups,
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
