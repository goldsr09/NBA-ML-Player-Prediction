from __future__ import annotations

import concurrent.futures as cf
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
BOXSCORE_URL_TMPL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
SEASON = "2025-26"
OUT_DIR = Path("analysis/output")
TEAM_GAMES_CSV = OUT_DIR / "nba_2025_26_team_games.csv"
GAMES_CSV = OUT_DIR / "nba_2025_26_games.csv"
SUMMARY_JSON = OUT_DIR / "nba_2025_26_summary.json"


def _to_float(v: Any) -> float:
    if v is None or v == "":
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace("%", ""))
        except ValueError:
            return np.nan
    return np.nan


def _safe_div(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b


def fetch_json(session: requests.Session, url: str, *, timeout: int = 30, retries: int = 3) -> dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # pragma: no cover - network-dependent
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def fetch_schedule() -> list[dict[str, Any]]:
    with requests.Session() as session:
        schedule = fetch_json(session, SCHEDULE_URL)["leagueSchedule"]["gameDates"]
    games: list[dict[str, Any]] = []
    for game_date in schedule:
        for game in game_date["games"]:
            gid = str(game.get("gameId", ""))
            if not gid.startswith("002"):  # regular season only
                continue
            if int(game.get("gameStatus", 0)) != 3:  # Final only
                continue
            games.append(
                {
                    "game_id": gid,
                    "game_date": game_date["gameDate"],
                    "game_date_utc": game.get("gameDateTimeUTC") or game.get("gameDateUTC"),
                    "home_team": game["homeTeam"]["teamTricode"],
                    "away_team": game["awayTeam"]["teamTricode"],
                }
            )
    games.sort(key=lambda g: (g["game_date_utc"] or "", g["game_id"]))
    return games


def fetch_boxscore_worker(game_id: str) -> dict[str, Any]:
    with requests.Session() as session:
        return fetch_json(session, BOXSCORE_URL_TMPL.format(game_id=game_id), timeout=30, retries=4)


def build_rows_from_boxscore(payload: dict[str, Any]) -> list[dict[str, Any]]:
    game = payload["game"]
    game_id = str(game["gameId"])
    game_time_utc = game.get("gameTimeUTC")
    home = game["homeTeam"]
    away = game["awayTeam"]

    def parse_side(team: dict[str, Any], opp: dict[str, Any], is_home: bool) -> dict[str, Any]:
        stats = team.get("statistics", {}) or {}
        opp_stats = opp.get("statistics", {}) or {}
        fga = _to_float(stats.get("fieldGoalsAttempted"))
        fgm = _to_float(stats.get("fieldGoalsMade"))
        fg3a = _to_float(stats.get("threePointersAttempted"))
        fg3m = _to_float(stats.get("threePointersMade"))
        fta = _to_float(stats.get("freeThrowsAttempted"))
        ftm = _to_float(stats.get("freeThrowsMade"))
        orb = _to_float(stats.get("reboundsOffensive"))
        drb = _to_float(stats.get("reboundsDefensive"))
        trb = _to_float(stats.get("reboundsTotal"))
        tov = _to_float(stats.get("turnoversTotal"))
        ast = _to_float(stats.get("assists"))
        stl = _to_float(stats.get("steals"))
        blk = _to_float(stats.get("blocks"))
        pf = _to_float(stats.get("foulsPersonal"))
        pts = _to_float(stats.get("points"))

        opp_fga = _to_float(opp_stats.get("fieldGoalsAttempted"))
        opp_fta = _to_float(opp_stats.get("freeThrowsAttempted"))
        opp_orb = _to_float(opp_stats.get("reboundsOffensive"))
        opp_drb = _to_float(opp_stats.get("reboundsDefensive"))
        opp_tov = _to_float(opp_stats.get("turnoversTotal"))
        opp_pts = _to_float(opp_stats.get("points"))

        poss_est = 0.5 * (
            (fga + 0.44 * fta - orb + tov) + (opp_fga + 0.44 * opp_fta - opp_orb + opp_tov)
        )
        efg = _safe_div(fgm + 0.5 * fg3m, fga)
        ft_rate = _safe_div(fta, fga)
        tov_rate = _safe_div(tov, poss_est)
        orb_rate = _safe_div(orb, orb + opp_drb)
        drb_rate = _safe_div(drb, drb + opp_orb)
        three_pa_rate = _safe_div(fg3a, fga)
        ts_attempts = fga + 0.44 * fta
        ts_pct = _safe_div(pts, 2 * ts_attempts)
        off_rating = _safe_div(100 * pts, poss_est)
        def_rating = _safe_div(100 * opp_pts, poss_est)
        net_rating = off_rating - def_rating if not (pd.isna(off_rating) or pd.isna(def_rating)) else np.nan

        return {
            "game_id": game_id,
            "game_time_utc": game_time_utc,
            "team_id": int(team["teamId"]),
            "team": team["teamTricode"],
            "opp_id": int(opp["teamId"]),
            "opp": opp["teamTricode"],
            "is_home": int(is_home),
            "team_score": pts,
            "opp_score": opp_pts,
            "win": int(pts > opp_pts),
            "margin": pts - opp_pts,
            "fga": fga,
            "fgm": fgm,
            "fg3a": fg3a,
            "fg3m": fg3m,
            "fta": fta,
            "ftm": ftm,
            "orb": orb,
            "drb": drb,
            "trb": trb,
            "ast": ast,
            "stl": stl,
            "blk": blk,
            "pf": pf,
            "tov": tov,
            "possessions": poss_est,
            "efg": efg,
            "ts_pct": ts_pct,
            "ft_rate": ft_rate,
            "tov_rate": tov_rate,
            "orb_rate": orb_rate,
            "drb_rate": drb_rate,
            "three_pa_rate": three_pa_rate,
            "off_rating": off_rating,
            "def_rating": def_rating,
            "net_rating": net_rating,
        }

    return [parse_side(home, away, True), parse_side(away, home, False)]


def fetch_all_boxscores(games: list[dict[str, Any]], max_workers: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    game_ids = [g["game_id"] for g in games]
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_boxscore_worker, gid): gid for gid in game_ids}
        completed = 0
        for fut in cf.as_completed(futures):
            gid = futures[fut]
            payload = fut.result()
            rows.extend(build_rows_from_boxscore(payload))
            completed += 1
            if completed % 100 == 0 or completed == len(game_ids):
                print(f"Fetched {completed}/{len(game_ids)} boxscores...", flush=True)
    return rows


def add_time_and_rest_features(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.copy()
    df["game_time_utc"] = pd.to_datetime(df["game_time_utc"], utc=True)
    df = df.sort_values(["team", "game_time_utc", "game_id"]).reset_index(drop=True)
    prev_time = df.groupby("team")["game_time_utc"].shift(1)
    df["days_since_prev"] = (df["game_time_utc"] - prev_time).dt.total_seconds() / 86400.0
    df["b2b"] = ((df["days_since_prev"] > 0) & (df["days_since_prev"] < 1.6)).astype(int)
    df["three_in_four"] = (
        (
            (df["game_time_utc"] - df.groupby("team")["game_time_utc"].shift(2)).dt.total_seconds() / 86400.0
        )
        < 4.1
    ).fillna(False).astype(int)
    return df


def add_rolling_features(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.copy().sort_values(["team", "game_time_utc", "game_id"]).reset_index(drop=True)
    base_cols = [
        "team_score",
        "opp_score",
        "margin",
        "possessions",
        "off_rating",
        "def_rating",
        "net_rating",
        "efg",
        "ts_pct",
        "ft_rate",
        "tov_rate",
        "orb_rate",
        "three_pa_rate",
        "ast",
        "tov",
    ]
    for col in base_cols:
        grp = df.groupby("team")[col]
        df[f"pre_{col}_avg5"] = grp.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        df[f"pre_{col}_avg10"] = grp.transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
        df[f"pre_{col}_season"] = grp.transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
    return df


def build_game_level(team_games: pd.DataFrame) -> pd.DataFrame:
    tg = team_games.copy()
    home = tg[tg["is_home"] == 1].copy()
    away = tg[tg["is_home"] == 0].copy()
    merge_cols = [
        "game_id",
        "game_time_utc",
        "team",
        "opp",
        "team_score",
        "opp_score",
        "win",
        "margin",
        "days_since_prev",
        "b2b",
        "three_in_four",
    ] + [c for c in tg.columns if c.startswith("pre_")]
    home = home[merge_cols].rename(columns={c: f"home_{c}" for c in merge_cols if c not in {"game_id", "game_time_utc"}})
    away = away[merge_cols].rename(columns={c: f"away_{c}" for c in merge_cols if c not in {"game_id", "game_time_utc"}})
    games = home.merge(away, on=["game_id", "game_time_utc"], how="inner")
    games["home_win"] = games["home_win"].astype(int)
    games["total_points"] = games["home_team_score"] + games["away_team_score"]
    games["rest_diff"] = games["home_days_since_prev"] - games["away_days_since_prev"]
    games["home_b2b_adv"] = games["away_b2b"] - games["home_b2b"]
    games["combined_pre_pace5"] = games["home_pre_possessions_avg5"] + games["away_pre_possessions_avg5"]

    # Differential features for models.
    diff_metrics = [
        "off_rating",
        "def_rating",
        "net_rating",
        "possessions",
        "efg",
        "ft_rate",
        "tov_rate",
        "orb_rate",
        "three_pa_rate",
        "margin",
        "team_score",
        "opp_score",
    ]
    for metric in diff_metrics:
        for window in ("avg5", "avg10", "season"):
            h = f"home_pre_{metric}_{window}"
            a = f"away_pre_{metric}_{window}"
            if h in games.columns and a in games.columns:
                games[f"diff_pre_{metric}_{window}"] = games[h] - games[a]

    return games.sort_values("game_time_utc").reset_index(drop=True)


def chron_split(df: pd.DataFrame, frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(math.floor(n * frac))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def run_models(games: pd.DataFrame, team_games: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # Win model (pregame features only).
    win_features = [
        "diff_pre_net_rating_avg5",
        "diff_pre_off_rating_avg5",
        "diff_pre_def_rating_avg5",
        "diff_pre_efg_avg5",
        "diff_pre_tov_rate_avg5",
        "diff_pre_orb_rate_avg5",
        "diff_pre_ft_rate_avg5",
        "diff_pre_possessions_avg5",
        "diff_pre_margin_avg10",
        "diff_pre_net_rating_season",
        "rest_diff",
        "home_b2b_adv",
        "home_b2b",
        "away_b2b",
    ]
    win_df = games.dropna(subset=["home_win"]).copy()
    train_g, test_g = chron_split(win_df, 0.8)

    X_train = train_g[win_features]
    X_test = test_g[win_features]
    y_train = train_g["home_win"]
    y_test = test_g["home_win"]

    logit = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    logit.fit(X_train, y_train)
    proba = logit.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    coef_names = win_features
    coef_vals = logit.named_steps["model"].coef_[0]
    coef_ranked = sorted(
        [{"feature": f, "coef": float(c)} for f, c in zip(coef_names, coef_vals)],
        key=lambda x: abs(x["coef"]),
        reverse=True,
    )
    results["win_model"] = {
        "n_games_train": int(len(train_g)),
        "n_games_test": int(len(test_g)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "baseline_home_win_rate_test": float(y_test.mean()),
        "top_coefficients": coef_ranked[:8],
    }

    # Team score model: predict each team's points using team + opponent pregame rolling stats.
    score_features = [
        "is_home",
        "days_since_prev",
        "b2b",
        "three_in_four",
        "pre_team_score_avg5",
        "pre_team_score_avg10",
        "pre_possessions_avg5",
        "pre_possessions_avg10",
        "pre_off_rating_avg5",
        "pre_off_rating_avg10",
        "pre_efg_avg5",
        "pre_tov_rate_avg5",
        "pre_ft_rate_avg5",
        "pre_orb_rate_avg5",
        # opponent context (merged below)
        "opp_pre_opp_score_avg5",  # opp defensive points allowed recent
        "opp_pre_def_rating_avg5",
        "opp_pre_possessions_avg5",
        "opp_pre_efg_avg5",
        "opp_pre_tov_rate_avg5",
        "opp_pre_orb_rate_avg5",
        "opp_b2b",
    ]
    opp_cols = [
        "game_id",
        "team",
        "pre_opp_score_avg5",
        "pre_def_rating_avg5",
        "pre_possessions_avg5",
        "pre_efg_avg5",
        "pre_tov_rate_avg5",
        "pre_orb_rate_avg5",
        "b2b",
    ]
    opp_view = team_games[opp_cols].rename(
        columns={
            "team": "opp",
            "pre_opp_score_avg5": "opp_pre_opp_score_avg5",
            "pre_def_rating_avg5": "opp_pre_def_rating_avg5",
            "pre_possessions_avg5": "opp_pre_possessions_avg5",
            "pre_efg_avg5": "opp_pre_efg_avg5",
            "pre_tov_rate_avg5": "opp_pre_tov_rate_avg5",
            "pre_orb_rate_avg5": "opp_pre_orb_rate_avg5",
            "b2b": "opp_b2b",
        }
    )
    team_score_df = team_games.merge(opp_view, on=["game_id", "opp"], how="left")
    team_score_df = team_score_df.sort_values("game_time_utc").reset_index(drop=True)

    train_t, test_t = chron_split(team_score_df, 0.8)
    Xtr = train_t[score_features]
    Xte = test_t[score_features]
    ytr = train_t["team_score"]
    yte = test_t["team_score"]

    lin = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    )
    lin.fit(Xtr, ytr)
    yhat = lin.predict(Xte)
    results["score_model"] = {
        "n_team_games_train": int(len(train_t)),
        "n_team_games_test": int(len(test_t)),
        "mae": float(mean_absolute_error(yte, yhat)),
        "rmse": float(np.sqrt(mean_squared_error(yte, yhat))),
        "r2": float(r2_score(yte, yhat)),
        "test_avg_points": float(yte.mean()),
    }

    # Total points model from game-level pregame features.
    total_features = [
        "home_pre_possessions_avg5",
        "away_pre_possessions_avg5",
        "home_pre_off_rating_avg5",
        "away_pre_off_rating_avg5",
        "home_pre_def_rating_avg5",
        "away_pre_def_rating_avg5",
        "home_pre_efg_avg5",
        "away_pre_efg_avg5",
        "rest_diff",
        "home_b2b",
        "away_b2b",
    ]
    total_df = games.copy().sort_values("game_time_utc")
    train_tot, test_tot = chron_split(total_df, 0.8)
    tot_model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())])
    tot_model.fit(train_tot[total_features], train_tot["total_points"])
    tot_pred = tot_model.predict(test_tot[total_features])
    results["total_model"] = {
        "n_games_train": int(len(train_tot)),
        "n_games_test": int(len(test_tot)),
        "mae": float(mean_absolute_error(test_tot["total_points"], tot_pred)),
        "rmse": float(np.sqrt(mean_squared_error(test_tot["total_points"], tot_pred))),
        "r2": float(r2_score(test_tot["total_points"], tot_pred)),
        "test_avg_total": float(test_tot["total_points"].mean()),
    }

    return results


def summarize_trends(games: pd.DataFrame, team_games: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}

    out["coverage"] = {
        "season": SEASON,
        "games_completed_regular_season": int(len(games)),
        "team_games": int(len(team_games)),
        "date_start_utc": str(team_games["game_time_utc"].min()),
        "date_end_utc": str(team_games["game_time_utc"].max()),
    }

    out["league_baseline"] = {
        "home_win_rate": float(games["home_win"].mean()),
        "avg_total_points": float(games["total_points"].mean()),
        "median_total_points": float(games["total_points"].median()),
        "avg_home_points": float(games["home_team_score"].mean()),
        "avg_away_points": float(games["away_team_score"].mean()),
        "avg_home_margin": float(games["home_team_score"].sub(games["away_team_score"]).mean()),
        "avg_team_possessions": float(team_games["possessions"].mean()),
        "avg_team_off_rating": float(team_games["off_rating"].mean()),
        "avg_team_efg": float(team_games["efg"].mean()),
        "avg_team_3pa_rate": float(team_games["three_pa_rate"].mean()),
    }

    # Month trends.
    month_df = team_games.copy()
    month_df["month"] = month_df["game_time_utc"].dt.strftime("%Y-%m")
    monthly = (
        month_df.groupby("month")
        .agg(
            games=("game_id", "nunique"),
            avg_points=("team_score", "mean"),
            avg_possessions=("possessions", "mean"),
            avg_off_rating=("off_rating", "mean"),
            avg_efg=("efg", "mean"),
            avg_3pa_rate=("three_pa_rate", "mean"),
            avg_ft_rate=("ft_rate", "mean"),
        )
        .reset_index()
    )
    out["monthly_trends"] = monthly.to_dict(orient="records")

    # Win rates when winning key stat battles (descriptive, postgame).
    home = team_games[team_games["is_home"] == 1][["game_id", "efg", "tov_rate", "orb_rate", "ft_rate", "win"]]
    away = team_games[team_games["is_home"] == 0][["game_id", "efg", "tov_rate", "orb_rate", "ft_rate", "win"]]
    battle = home.merge(away, on="game_id", suffixes=("_home", "_away"))
    battle["home_win"] = battle["win_home"]
    battle["home_won_efg"] = (battle["efg_home"] > battle["efg_away"]).astype(int)
    battle["home_won_tov"] = (battle["tov_rate_home"] < battle["tov_rate_away"]).astype(int)
    battle["home_won_orb"] = (battle["orb_rate_home"] > battle["orb_rate_away"]).astype(int)
    battle["home_won_ft"] = (battle["ft_rate_home"] > battle["ft_rate_away"]).astype(int)

    stat_battles = {}
    for flag in ["home_won_efg", "home_won_tov", "home_won_orb", "home_won_ft"]:
        mask = battle[flag] == 1
        stat_battles[flag] = {
            "games": int(mask.sum()),
            "home_win_rate_when_true": float(battle.loc[mask, "home_win"].mean()),
            "home_win_rate_when_false": float(battle.loc[~mask, "home_win"].mean()),
        }
    out["stat_battle_win_rates"] = stat_battles

    # Margin correlation with factor differentials.
    game_pairs = (
        team_games[["game_id", "is_home", "margin", "efg", "tov_rate", "orb_rate", "ft_rate", "possessions"]]
        .copy()
        .sort_values(["game_id", "is_home"], ascending=[True, False])
    )
    home_side = game_pairs[game_pairs["is_home"] == 1].rename(
        columns={
            "margin": "home_margin",
            "efg": "home_efg",
            "tov_rate": "home_tov_rate",
            "orb_rate": "home_orb_rate",
            "ft_rate": "home_ft_rate",
            "possessions": "home_possessions",
        }
    )
    away_side = game_pairs[game_pairs["is_home"] == 0].rename(
        columns={
            "margin": "away_margin",
            "efg": "away_efg",
            "tov_rate": "away_tov_rate",
            "orb_rate": "away_orb_rate",
            "ft_rate": "away_ft_rate",
            "possessions": "away_possessions",
        }
    )
    corr_df = home_side.merge(away_side, on="game_id")
    corr_df["margin"] = corr_df["home_margin"]
    corr_df["efg_diff"] = corr_df["home_efg"] - corr_df["away_efg"]
    corr_df["tov_rate_diff"] = corr_df["home_tov_rate"] - corr_df["away_tov_rate"]
    corr_df["orb_rate_diff"] = corr_df["home_orb_rate"] - corr_df["away_orb_rate"]
    corr_df["ft_rate_diff"] = corr_df["home_ft_rate"] - corr_df["away_ft_rate"]
    corr_df["pace_avg"] = 0.5 * (corr_df["home_possessions"] + corr_df["away_possessions"])
    corr_metrics = {}
    for c in ["efg_diff", "tov_rate_diff", "orb_rate_diff", "ft_rate_diff", "pace_avg"]:
        corr_metrics[c] = float(corr_df["margin"].corr(corr_df[c]))
    out["margin_correlations_home_perspective"] = corr_metrics

    # Thresholds for descriptive predictive heuristics.
    thresholds = {}
    thresholds["efg_edge_3pp"] = float((corr_df.loc[corr_df["efg_diff"] >= 0.03, "margin"] > 0).mean())
    thresholds["efg_edge_5pp"] = float((corr_df.loc[corr_df["efg_diff"] >= 0.05, "margin"] > 0).mean())
    thresholds["lose_tov_rate_by_3pp"] = float((corr_df.loc[corr_df["tov_rate_diff"] >= 0.03, "margin"] < 0).mean())
    thresholds["orb_edge_8pp"] = float((corr_df.loc[corr_df["orb_rate_diff"] >= 0.08, "margin"] > 0).mean())
    out["descriptive_threshold_win_rates"] = thresholds

    # Rest effects.
    rest = team_games.copy()
    rest["rest_bucket"] = pd.cut(
        rest["days_since_prev"],
        bins=[-np.inf, 1.6, 2.6, 3.6, np.inf],
        labels=["b2b", "1_day_rest", "2_days_rest", "3+_days_rest"],
    )
    rest_summary = (
        rest.groupby("rest_bucket", observed=True)
        .agg(
            games=("game_id", "count"),
            win_rate=("win", "mean"),
            avg_points=("team_score", "mean"),
            avg_off_rating=("off_rating", "mean"),
            avg_possessions=("possessions", "mean"),
        )
        .reset_index()
    )
    out["team_rest_effects"] = rest_summary.to_dict(orient="records")

    # B2B split by home/away.
    b2b_split = (
        rest.groupby(["is_home", "b2b"])
        .agg(
            games=("game_id", "count"),
            win_rate=("win", "mean"),
            avg_points=("team_score", "mean"),
            avg_off_rating=("off_rating", "mean"),
        )
        .reset_index()
    )
    out["b2b_home_away_split"] = b2b_split.to_dict(orient="records")

    return out


def print_key_findings(summary: dict[str, Any], model_results: dict[str, Any]) -> None:
    cov = summary["coverage"]
    base = summary["league_baseline"]
    print("\n=== COVERAGE ===")
    print(
        f"Season {cov['season']} regular season games completed: {cov['games_completed_regular_season']} "
        f"({cov['team_games']} team-games)"
    )
    print(f"Date range (UTC): {cov['date_start_utc']} to {cov['date_end_utc']}")

    print("\n=== LEAGUE BASELINE ===")
    print(
        f"Home win rate: {base['home_win_rate']:.3f} | Avg total: {base['avg_total_points']:.1f} | "
        f"Avg home/away points: {base['avg_home_points']:.1f}/{base['avg_away_points']:.1f}"
    )
    print(
        f"Avg team possessions: {base['avg_team_possessions']:.1f} | Avg ORtg: {base['avg_team_off_rating']:.1f} | "
        f"Avg eFG%: {base['avg_team_efg']:.3f} | Avg 3PA rate: {base['avg_team_3pa_rate']:.3f}"
    )

    print("\n=== MONTHLY TREND SNAPSHOT ===")
    for row in summary["monthly_trends"]:
        print(
            f"{row['month']}: games={int(row['games'])}, pts={row['avg_points']:.1f}, "
            f"pace={row['avg_possessions']:.1f}, ORtg={row['avg_off_rating']:.1f}, "
            f"eFG={row['avg_efg']:.3f}, 3PA rate={row['avg_3pa_rate']:.3f}"
        )

    print("\n=== STAT BATTLE WIN RATES (HOME TEAM, DESCRIPTIVE) ===")
    for k, v in summary["stat_battle_win_rates"].items():
        print(
            f"{k}: when true win%={v['home_win_rate_when_true']:.3f} vs when false={v['home_win_rate_when_false']:.3f}"
        )

    print("\n=== MARGIN CORRELATIONS (HOME MARGIN) ===")
    for k, v in summary["margin_correlations_home_perspective"].items():
        print(f"{k}: {v:.3f}")

    print("\n=== SIMPLE THRESHOLD HEURISTICS (DESCRIPTIVE) ===")
    for k, v in summary["descriptive_threshold_win_rates"].items():
        print(f"{k}: {v:.3f}")

    print("\n=== PREDICTIVE MODELS (OUT-OF-SAMPLE, CHRONO SPLIT) ===")
    wm = model_results["win_model"]
    print(
        f"Win model: acc={wm['accuracy']:.3f}, AUC={wm['auc']:.3f}, logloss={wm['log_loss']:.3f}, "
        f"baseline_home_win_rate={wm['baseline_home_win_rate_test']:.3f}"
    )
    print("Top win-model coefficients (scaled features):")
    for coef in wm["top_coefficients"]:
        print(f"  {coef['feature']}: {coef['coef']:+.3f}")

    sm = model_results["score_model"]
    print(
        f"Team score model: MAE={sm['mae']:.2f}, RMSE={sm['rmse']:.2f}, R2={sm['r2']:.3f}, "
        f"test avg points={sm['test_avg_points']:.2f}"
    )
    tm = model_results["total_model"]
    print(
        f"Total points model: MAE={tm['mae']:.2f}, RMSE={tm['rmse']:.2f}, R2={tm['r2']:.3f}, "
        f"test avg total={tm['test_avg_total']:.2f}"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {SEASON} schedule...", flush=True)
    schedule_games = fetch_schedule()
    print(f"Completed regular season games found: {len(schedule_games)}", flush=True)

    print("Downloading box scores...", flush=True)
    rows = fetch_all_boxscores(schedule_games, max_workers=12)
    team_games = pd.DataFrame(rows).drop_duplicates(subset=["game_id", "team"]).reset_index(drop=True)
    team_games = add_time_and_rest_features(team_games)
    team_games = add_rolling_features(team_games)
    games = build_game_level(team_games)

    team_games.to_csv(TEAM_GAMES_CSV, index=False)
    games.to_csv(GAMES_CSV, index=False)

    print("Running summaries and models...", flush=True)
    summary = summarize_trends(games, team_games)
    model_results = run_models(games, team_games)
    payload = {"summary": summary, "models": model_results}
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2))

    print_key_findings(summary, model_results)
    print(f"\nSaved: {TEAM_GAMES_CSV}")
    print(f"Saved: {GAMES_CSV}")
    print(f"Saved: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
