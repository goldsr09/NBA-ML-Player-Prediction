#!/usr/bin/env python3
"""Backtest Feb 27 2026 player performance predictions against actual results.

Loads:
  - Cached prediction advantages (with signal thresholds already computed)
  - Actual box scores from NBA CDN
Grades each signal, reports win rate, Performance, Accuracy, breakdown by confidence tier.
"""
from __future__ import annotations

import json
import sys
import unicodedata
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_nba_2025_26_advanced import (
    BOXSCORE_CACHE,
    CACHE_DIR,
    fetch_json,
    _minutes_to_float,
    _to_float,
)
from predict_player_props import (
    VIG_FACTOR,
    MIN_ADVANTAGE_PCT,
    MIN_EV,
    HIGH_CONF_EV,
    MIN_ADVANTAGE_PCT_BY_SIDE,
    MIN_EV_BY_SIDE,
    SIGNAL_POINTS_ONLY,
    MIN_SIGNAL_PRED_MINUTES,
    MIN_SIGNAL_PRE_MINUTES_AVG10,
    MIN_ABS_EDGE,
    normalize_player_name,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "analysis" / "output"
PREDICTIONS_DIR = OUT_DIR / "predictions"

TARGET_DATE = "20260227"


# ---------------------------------------------------------------------------
# Fetch actual box scores for a date
# ---------------------------------------------------------------------------
def fetch_actuals_for_date(date_str: str) -> pd.DataFrame:
    """Fetch actual player stats from NBA CDN boxscores for a given date."""
    # NBA CDN schedule URL for the date
    # Date format: YYYY-MM-DD
    formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    schedule_url = f"https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

    # Instead of parsing the full schedule, load from cached boxscores
    # The monolith caches boxscores in BOXSCORE_CACHE = CACHE_DIR / "boxscores"
    BOXSCORE_CACHE.mkdir(parents=True, exist_ok=True)

    # First, find game IDs for this date from the existing cache or schedule
    from analyze_nba_2025_26_advanced import (
        SCHEDULE_URL,
        SEASON,
        parse_player_box_rows,
    )

    # Load schedule to find game IDs
    sched = fetch_json(SCHEDULE_URL, cache_path=CACHE_DIR / "schedule.json")
    game_dates = sched.get("leagueSchedule", {}).get("gameDates", [])

    # Build target date in MM/DD/YYYY format for matching schedule's gameDate field
    # date_str is YYYYMMDD, schedule uses "MM/DD/YYYY 00:00:00"
    target_mmddyyyy = f"{date_str[4:6]}/{date_str[6:8]}/{date_str[:4]}"

    target_games = []
    for gd in game_dates:
        gd_date = str(gd.get("gameDate", ""))
        if gd_date.startswith(target_mmddyyyy):
            for g in gd.get("games", []):
                game_id = g.get("gameId", "")
                home = g.get("homeTeam", {}).get("teamTricode", "")
                away = g.get("awayTeam", {}).get("teamTricode", "")
                status = g.get("gameStatus", 0)
                if game_id and status == 3:  # 3 = Final
                    target_games.append({
                        "game_id": game_id,
                        "home_team": home,
                        "away_team": away,
                    })
            break

    if not target_games:
        print(f"No completed games found for {date_str} in schedule")
        return pd.DataFrame()

    print(f"Found {len(target_games)} completed games for {date_str}")

    all_players = []
    for game_info in target_games:
        game_id = game_info["game_id"]
        home_team = game_info["home_team"]
        away_team = game_info["away_team"]

        # Fetch boxscore
        box_url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
        cache_path = BOXSCORE_CACHE / f"{game_id}.json"
        box_data = fetch_json(box_url, cache_path=cache_path)

        if not box_data:
            print(f"  Could not fetch boxscore for {game_id}")
            continue

        game = box_data.get("game", {})

        for side in ["homeTeam", "awayTeam"]:
            team_data = game.get(side, {})
            team_tri = team_data.get("teamTricode", "")
            players = team_data.get("players", [])

            for p in players:
                stats = p.get("statistics", {})
                name = f"{p.get('firstName', '')} {p.get('familyName', '')}".strip()
                if not name or name == " ":
                    name = p.get("name", "")

                mins_str = stats.get("minutes", "PT00M00.00S")
                mins = _parse_iso_minutes(mins_str)

                all_players.append({
                    "game_id": game_id,
                    "game_date_est": date_str,
                    "home_team": home_team,
                    "away_team": away_team,
                    "team": team_tri,
                    "opp": away_team if side == "homeTeam" else home_team,
                    "player_name": name,
                    "player_id": p.get("personId", ""),
                    "points": _to_float(stats.get("points", 0)),
                    "rebounds": _to_float(stats.get("reboundsTotal", 0)),
                    "assists": _to_float(stats.get("assists", 0)),
                    "minutes": mins,
                    "fg3m": _to_float(stats.get("threePointersMade", 0)),
                    "steals": _to_float(stats.get("steals", 0)),
                    "blocks": _to_float(stats.get("blocks", 0)),
                    "turnovers": _to_float(stats.get("turnovers", 0)),
                    "starter": 1 if p.get("starter", "") == "1" else 0,
                })

    if not all_players:
        return pd.DataFrame()

    df = pd.DataFrame(all_players)
    print(f"  Loaded {len(df)} player box rows across {len(target_games)} games")
    return df


def _parse_iso_minutes(s: str) -> float:
    """Parse ISO 8601 duration like 'PT32M15.00S' to float minutes."""
    if not s or s == "PT00M00.00S":
        return 0.0
    s = str(s)
    if s.startswith("PT"):
        s = s[2:]
    mins = 0.0
    if "M" in s:
        parts = s.split("M")
        try:
            mins += float(parts[0])
        except (ValueError, IndexError):
            pass
        s = parts[1] if len(parts) > 1 else ""
    if s.endswith("S"):
        s = s[:-1]
    if s:
        try:
            mins += float(s) / 60.0
        except ValueError:
            pass
    return round(mins, 1)


# ---------------------------------------------------------------------------
# Grade signals
# ---------------------------------------------------------------------------
def grade_signals(edges: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Match edge signals to actual results and grade them."""
    edges = edges.copy()
    if "signal" in edges.columns and (edges["signal"] != "LOW CONFIDENCE").any():
        edges["rederived_signal"] = edges["signal"]
        edges["rederived_confidence"] = edges.get("confidence", "")
        actionable = edges[edges["rederived_signal"] != "LOW CONFIDENCE"].copy()
        print(f"\nUsing saved signals from edge file: {len(actionable)}")
    else:
        # Fallback: re-derive with the same policy thresholds used by live prediction code.
        signals = []
        for _, row in edges.iterrows():
            signal = "LOW CONFIDENCE"
            confidence = ""
            stat_type = str(row.get("stat_type", ""))
            ev_over = row.get("ev_over", np.nan)
            ev_under = row.get("ev_under", np.nan)
            edge = row.get("edge", 0)
            advantage_pct = row.get("advantage_pct", 0)
            p_over = row.get("p_over", np.nan)
            p_under = row.get("p_under", np.nan)

            min_abs = MIN_ABS_EDGE.get(stat_type, 2.0)
            pred_minutes = pd.to_numeric(row.get("pred_minutes"), errors="coerce")
            pre_minutes_avg10 = pd.to_numeric(row.get("pre_minutes_avg10"), errors="coerce")
            stat_gate_ok = (not SIGNAL_POINTS_ONLY) or (stat_type == "points")
            minutes_gate_ok = (
                pd.notna(pred_minutes) and pred_minutes >= MIN_SIGNAL_PRED_MINUTES
                and pd.notna(pre_minutes_avg10) and pre_minutes_avg10 >= MIN_SIGNAL_PRE_MINUTES_AVG10
            )

            over_odds = row.get("over_odds", np.nan)
            under_odds = row.get("under_odds", np.nan)
            if stat_gate_ok and minutes_gate_ok and pd.notna(ev_over) and pd.notna(ev_under) and abs(edge) >= min_abs:
                over_implied = _american_to_implied(over_odds) if pd.notna(over_odds) else 0.5122
                under_implied = _american_to_implied(under_odds) if pd.notna(under_odds) else 0.5122
                total_implied = over_implied + under_implied
                over_breakeven = over_implied / total_implied if total_implied > 0 else 0.5
                under_breakeven = under_implied / total_implied if total_implied > 0 else 0.5

                if (
                    ev_over > MIN_EV_BY_SIDE["OVER"]
                    and pd.notna(p_over)
                    and p_over > over_breakeven
                    and abs(advantage_pct) >= MIN_ADVANTAGE_PCT_BY_SIDE["OVER"]
                ):
                    signal = "OVER"
                    confidence = "HIGH CONFIDENCE" if ev_over >= HIGH_CONF_EV else "MODERATE CONFIDENCE"
                elif (
                    ev_under > MIN_EV_BY_SIDE["UNDER"]
                    and pd.notna(p_under)
                    and p_under > under_breakeven
                    and abs(advantage_pct) >= MIN_ADVANTAGE_PCT_BY_SIDE["UNDER"]
                ):
                    signal = "UNDER"
                    confidence = "HIGH CONFIDENCE" if ev_under >= HIGH_CONF_EV else "MODERATE CONFIDENCE"

            # Line movement upgrade
            if signal != "LOW CONFIDENCE" and confidence == "MODERATE CONFIDENCE":
                open_line = row.get("open_line", np.nan)
                prop_line = row.get("prop_line", np.nan)
                pred_value = row.get("pred_value", np.nan)
                if pd.notna(open_line) and float(open_line) > 0 and pd.notna(prop_line):
                    line_move = prop_line - float(open_line)
                    line_move_pct = (line_move / float(open_line)) * 100
                    model_says_over = pred_value > prop_line
                    line_moved_up = line_move > 0
                    line_confirms = (model_says_over and line_moved_up) or (not model_says_over and not line_moved_up)
                    if line_confirms and abs(line_move_pct) >= 2.0:
                        confidence = "HIGH CONFIDENCE"

            signals.append({"rederived_signal": signal, "rederived_confidence": confidence})

        sig_df = pd.DataFrame(signals)
        edges["rederived_signal"] = sig_df["rederived_signal"].values
        edges["rederived_confidence"] = sig_df["rederived_confidence"].values
        actionable = edges[edges["rederived_signal"] != "LOW CONFIDENCE"].copy()
        print(f"\nRe-derived signals (no saved signals available): {len(actionable)}")

    print(f"  HIGH CONFIDENCE: {(actionable['rederived_confidence'] == 'HIGH CONFIDENCE').sum()}")
    print(f"  LEAN:     {(actionable['rederived_confidence'] == 'LEAN').sum()}")

    if actionable.empty:
        return pd.DataFrame()

    # Normalize player names for matching
    actuals["norm_name"] = actuals["player_name"].apply(normalize_player_name)

    results = []
    unmatched = 0

    for _, row in actionable.iterrows():
        player = row["player_name"]
        stat_type = row["stat_type"]
        prop_line = row["prop_line"]
        pred_value = row["pred_value"]
        signal = row["rederived_signal"]
        confidence = row["rederived_confidence"]
        team = row.get("team", "")

        norm_player = normalize_player_name(player)

        # Match to actuals
        mask = actuals["norm_name"] == norm_player
        if team:
            team_mask = mask & (actuals["team"] == team)
            if team_mask.any():
                mask = team_mask

        if not mask.any():
            # Try fuzzy: last name match
            last_name = norm_player.split()[-1] if norm_player.split() else norm_player
            mask = actuals["norm_name"].str.contains(last_name, na=False)
            if team:
                team_mask = mask & (actuals["team"] == team)
                if team_mask.any():
                    mask = team_mask

        matched = actuals[mask]
        if matched.empty:
            unmatched += 1
            continue

        actual_row = matched.iloc[0]
        actual_val = actual_row.get(stat_type, np.nan)
        if pd.isna(actual_val):
            unmatched += 1
            continue

        actual_minutes = actual_row.get("minutes", 0)

        # Grade
        if signal == "OVER":
            hit = 1 if actual_val > prop_line else 0
            push = actual_val == prop_line
            ev_chosen = row.get("ev_over", np.nan)
            odds_chosen = row.get("over_odds", -110)
        else:
            hit = 1 if actual_val < prop_line else 0
            push = actual_val == prop_line
            ev_chosen = row.get("ev_under", np.nan)
            odds_chosen = row.get("under_odds", -110)

        # P/L at flat $100 per bet
        if push:
            pnl = 0.0
        elif hit:
            decimal_odds = _american_to_decimal(odds_chosen) if pd.notna(odds_chosen) else VIG_FACTOR
            pnl = decimal_odds * 100
        else:
            pnl = -100.0

        results.append({
            "player_name": player,
            "team": team,
            "opp": row.get("opp", ""),
            "stat_type": stat_type,
            "prop_line": prop_line,
            "pred_value": pred_value,
            "actual_value": float(actual_val),
            "actual_minutes": float(actual_minutes),
            "signal": signal,
            "confidence": confidence,
            "edge": row.get("edge", 0),
            "advantage_pct": row.get("advantage_pct", 0),
            "ev_at_signal": float(ev_chosen) if pd.notna(ev_chosen) else np.nan,
            "p_over": row.get("p_over", np.nan),
            "p_under": row.get("p_under", np.nan),
            "over_odds": row.get("over_odds", np.nan),
            "under_odds": row.get("under_odds", np.nan),
            "hit": hit,
            "push": int(push),
            "pnl": round(pnl, 2),
            "pred_error": round(pred_value - float(actual_val), 1),
        })

    if unmatched:
        print(f"  Unmatched signals: {unmatched}")

    return pd.DataFrame(results)


def _american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability."""
    if pd.isna(odds):
        return 0.5
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return 0.5


def _american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal performance per prediction."""
    if pd.isna(odds):
        return VIG_FACTOR
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    elif odds < 0:
        return 100.0 / abs(odds)
    return 1.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(graded: pd.DataFrame) -> None:
    """Print detailed backtest report."""

    if graded.empty:
        print("\nNo graded signals to report.")
        return

    n_bets = len(graded)
    n_hits = int(graded["hit"].sum())
    n_pushes = int(graded["push"].sum())
    total_pnl = graded["pnl"].sum()
    total_allocated = (n_bets - n_pushes) * 100
    hit_rate = n_hits / (n_bets - n_pushes) if (n_bets - n_pushes) > 0 else 0
    accuracy = (total_pnl / (n_bets * 100)) * 100 if n_bets > 0 else 0

    print("\n" + "=" * 72)
    print(f"  PROP BACKTEST RESULTS: Feb 27, 2026")
    print("=" * 72)
    print(f"\n  Total signals:     {n_bets}")
    print(f"  Hits:              {n_hits}")
    print(f"  Misses:            {n_bets - n_hits - n_pushes}")
    print(f"  Pushes:            {n_pushes}")
    print(f"  Hit rate:          {hit_rate:.1%}")
    print(f"  Total P/L:         ${total_pnl:+.0f}")
    print(f"  Total allocated:     ${n_bets * 100:,.0f}")
    print(f"  Accuracy:               {accuracy:+.1f}%")

    # --- Breakdown by confidence tier ---
    print(f"\n  --- By Confidence Tier ---")
    for tier in ["HIGH CONFIDENCE", "MODERATE CONFIDENCE"]:
        subset = graded[graded["confidence"] == tier]
        if subset.empty:
            continue
        t_bets = len(subset)
        t_hits = int(subset["hit"].sum())
        t_pushes = int(subset["push"].sum())
        t_pnl = subset["pnl"].sum()
        t_hr = t_hits / (t_bets - t_pushes) if (t_bets - t_pushes) > 0 else 0
        t_accuracy = (t_pnl / (t_bets * 100)) * 100 if t_bets > 0 else 0
        avg_ev = subset["ev_at_signal"].dropna().mean()
        print(f"\n  {tier}:")
        print(f"    Bets: {t_bets}  Hits: {t_hits}  Rate: {t_hr:.1%}  P/L: ${t_pnl:+.0f}  Accuracy: {t_accuracy:+.1f}%  Avg EV: {avg_ev:+.3f}")

    # --- Breakdown by stat type ---
    print(f"\n  --- By Stat Type ---")
    for st in sorted(graded["stat_type"].unique()):
        subset = graded[graded["stat_type"] == st]
        t_bets = len(subset)
        t_hits = int(subset["hit"].sum())
        t_pushes = int(subset["push"].sum())
        t_pnl = subset["pnl"].sum()
        t_hr = t_hits / (t_bets - t_pushes) if (t_bets - t_pushes) > 0 else 0
        t_accuracy = (t_pnl / (t_bets * 100)) * 100 if t_bets > 0 else 0
        print(f"    {st:10s}: {t_bets:>3d} bets, {t_hr:.1%} hit rate, ${t_pnl:+.0f} P/L ({t_accuracy:+.1f}% Accuracy)")

    # --- Breakdown by signal direction ---
    print(f"\n  --- By Direction ---")
    for direction in ["OVER", "UNDER"]:
        subset = graded[graded["signal"] == direction]
        if subset.empty:
            continue
        t_bets = len(subset)
        t_hits = int(subset["hit"].sum())
        t_pushes = int(subset["push"].sum())
        t_pnl = subset["pnl"].sum()
        t_hr = t_hits / (t_bets - t_pushes) if (t_bets - t_pushes) > 0 else 0
        print(f"    {direction:6s}: {t_bets:>3d} bets, {t_hr:.1%} hit rate, ${t_pnl:+.0f} P/L")

    # --- Individual bet detail ---
    print(f"\n  --- Individual Bets (sorted by |EV|) ---")
    sorted_bets = graded.sort_values("ev_at_signal", ascending=False)
    for _, r in sorted_bets.iterrows():
        result = "HIT " if r["hit"] else "PUSH" if r["push"] else "MISS"
        print(
            f"    {r['player_name']:25s} {r['stat_type']:10s} "
            f"{r['signal']:6s} Line={r['prop_line']:5.1f}  "
            f"Pred={r['pred_value']:5.1f}  Actual={r['actual_value']:5.1f}  "
            f"EV={r['ev_at_signal']:+.3f}  {result}  ${r['pnl']:+.0f}"
        )

    # --- Prediction accuracy on signalled bets ---
    print(f"\n  --- Prediction Accuracy (signalled bets only) ---")
    mae = graded["pred_error"].abs().mean()
    directional_correct = 0
    for _, r in graded.iterrows():
        if r["signal"] == "OVER" and r["actual_value"] > r["pred_value"] * 0.85:
            directional_correct += 1
        elif r["signal"] == "UNDER" and r["actual_value"] < r["pred_value"] * 1.15:
            directional_correct += 1
    print(f"    MAE (pred vs actual): {mae:.1f}")

    # Minutes played check: did any misses have low minutes?
    misses = graded[(graded["hit"] == 0) & (graded["push"] == 0)]
    if not misses.empty:
        low_min_misses = misses[misses["actual_minutes"] < 15]
        if not low_min_misses.empty:
            print(f"\n  --- Low-Minutes Misses (DNP/blowout) ---")
            for _, r in low_min_misses.iterrows():
                print(
                    f"    {r['player_name']:25s} {r['stat_type']:10s} "
                    f"played {r['actual_minutes']:.0f} min (pred miss likely due to minutes)"
                )

    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# EV threshold sweep
# ---------------------------------------------------------------------------
def threshold_sweep(edges: pd.DataFrame, actuals: pd.DataFrame) -> None:
    """Test different EV thresholds to find optimal cutoff."""
    print("\n  --- Threshold Sensitivity Analysis ---")
    print(f"  {'Min EV':>8s} {'Min Edge%':>10s} {'Bets':>6s} {'Hits':>6s} {'Rate':>7s} {'P/L':>8s} {'Accuracy':>7s}")

    for min_ev in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for min_advantage in [10.0, 15.0, 20.0]:
            # Re-derive with these thresholds
            count = 0
            hits = 0
            pnl = 0.0

            actuals_norm = actuals.copy()
            actuals_norm["norm_name"] = actuals_norm["player_name"].apply(normalize_player_name)

            for _, row in edges.iterrows():
                stat_type = row["stat_type"]
                ev_over = row.get("ev_over", np.nan)
                ev_under = row.get("ev_under", np.nan)
                edge = row.get("edge", 0)
                advantage_pct = row.get("advantage_pct", 0)
                p_over = row.get("p_over", np.nan)
                p_under = row.get("p_under", np.nan)

                min_abs = MIN_ABS_EDGE.get(stat_type, 2.0)
                signal = None
                ev_chosen = np.nan
                odds_chosen = np.nan

                if pd.notna(ev_over) and pd.notna(ev_under) and abs(edge) >= min_abs:
                    if ev_over > min_ev and pd.notna(p_over) and p_over > 0.5 and abs(advantage_pct) >= min_advantage:
                        signal = "OVER"
                        ev_chosen = ev_over
                        odds_chosen = row.get("over_odds", -110)
                    elif ev_under > min_ev and pd.notna(p_under) and p_under > 0.5 and abs(advantage_pct) >= min_advantage:
                        signal = "UNDER"
                        ev_chosen = ev_under
                        odds_chosen = row.get("under_odds", -110)

                if signal is None:
                    continue

                # Match actual
                player = row["player_name"]
                team = row.get("team", "")
                norm_player = normalize_player_name(player)
                mask = actuals_norm["norm_name"] == norm_player
                if team:
                    team_mask = mask & (actuals_norm["team"] == team)
                    if team_mask.any():
                        mask = team_mask

                if not mask.any():
                    continue

                actual_val = actuals_norm[mask].iloc[0].get(stat_type, np.nan)
                if pd.isna(actual_val):
                    continue

                count += 1
                if signal == "OVER":
                    hit = actual_val > row["prop_line"]
                else:
                    hit = actual_val < row["prop_line"]

                if hit:
                    hits += 1
                    dec = _american_to_decimal(odds_chosen) if pd.notna(odds_chosen) else VIG_FACTOR
                    pnl += dec * 100
                else:
                    pnl -= 100

            if count > 0:
                rate = hits / count
                accuracy = (pnl / (count * 100)) * 100
                print(f"  {min_ev:8.2f} {min_advantage:10.1f} {count:6d} {hits:6d} {rate:7.1%} ${pnl:>+7.0f} {accuracy:>+6.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  NBA PLAYER PROP BACKTEST - February 27, 2026")
    print("  Real market lines from ESPN vs actual box scores")
    print("=" * 72)

    # Load edges
    edge_file = PREDICTIONS_DIR / f"player_prop_edges_{TARGET_DATE}.csv"
    if not edge_file.exists():
        print(f"ERROR: Edge file not found: {edge_file}")
        return

    edges = pd.read_csv(edge_file)
    print(f"\nLoaded {len(edges)} prediction advantage rows from {edge_file}")

    # Fetch actuals
    print(f"\nFetching actual box scores for {TARGET_DATE}...")
    actuals = fetch_actuals_for_date(TARGET_DATE)

    if actuals.empty:
        print("ERROR: No actuals found")
        return

    print(f"\nActuals summary:")
    for team in sorted(actuals["team"].unique()):
        team_df = actuals[actuals["team"] == team]
        print(f"  {team}: {len(team_df)} players")

    # Grade signals
    print("\nGrading signals...")
    graded = grade_signals(edges, actuals)

    if graded.empty:
        print("No signals could be graded.")
        return

    # Save graded results
    graded_file = PREDICTIONS_DIR / f"prop_backtest_graded_{TARGET_DATE}.csv"
    graded.to_csv(graded_file, index=False)
    print(f"Saved graded results to {graded_file}")

    # Print report
    print_report(graded)

    # Threshold sweep
    threshold_sweep(edges, actuals)

    print("\nDone.")


if __name__ == "__main__":
    main()
