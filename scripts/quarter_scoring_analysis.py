"""
Quarter-by-Quarter Scoring Analysis for Live Total Prediction Edge

Goal: Understand if knowing the score through 3 quarters gives a predictive
advantage on the final total (4th quarter scoring).

Key questions:
1. How predictable is Q4 scoring given Q1-Q3 pace/scoring?
2. Do high-scoring games through 3Q tend to stay high or regress?
3. Do low-scoring games through 3Q tend to stay low or regress?
4. Is there a systematic bias in how Q4 totals relate to pregame totals?
"""

import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# ── Shared cache base path ────────────────────────────────────────────────
# Resolve from script location -> project root -> analysis/output.
# Respects NBA_OUTPUT_DIR env var override for non-standard layouts.
# Other scripts (train_q4_model.py) import this constant.
import os as _os
ANALYSIS_OUTPUT_BASE = Path(
    _os.environ.get("NBA_OUTPUT_DIR",
                     str(Path(__file__).resolve().parent.parent / "analysis" / "output"))
)

# ── Load all boxscores across seasons ─────────────────────────────────────

def load_quarter_data():
    """Extract quarter-by-quarter scoring from all cached boxscores."""
    rows = []

    base = ANALYSIS_OUTPUT_BASE

    # Historical seasons
    hist_dir = base / "historical_cache"
    patterns = []
    if hist_dir.exists():
        for season_dir in sorted(hist_dir.iterdir()):
            box_dir = season_dir / "boxscores"
            if box_dir.exists():
                patterns.append((season_dir.name, box_dir))

    # Current season
    curr_dir = base / "nba_2025_26_advanced_cache" / "boxscores"
    if curr_dir.exists():
        patterns.append(("2025-26", curr_dir))

    for season, box_dir in patterns:
        for fpath in sorted(box_dir.glob("*.json")):
            try:
                with open(fpath) as f:
                    data = json.load(f)

                game = data.get("game", data)
                game_id = game.get("gameId", fpath.stem)
                game_date = game.get("gameTimeUTC", "")[:10]

                home = game.get("homeTeam", {})
                away = game.get("awayTeam", {})

                home_periods = home.get("periods", [])
                away_periods = away.get("periods", [])

                if not home_periods or not away_periods:
                    continue

                # Only analyze regulation games (exactly 4 quarters, no OT)
                home_reg = [p for p in home_periods if p.get("periodType") == "REGULAR"]
                away_reg = [p for p in away_periods if p.get("periodType") == "REGULAR"]
                has_ot = any(p.get("periodType") == "OVERTIME" for p in home_periods)

                if len(home_reg) < 4 or len(away_reg) < 4:
                    continue

                home_q = [home_reg[i]["score"] for i in range(4)]
                away_q = [away_reg[i]["score"] for i in range(4)]

                row = {
                    "season": season,
                    "game_id": game_id,
                    "game_date": game_date,
                    "has_ot": has_ot,
                    "home_team": home.get("teamTricode", ""),
                    "away_team": away.get("teamTricode", ""),
                    "home_q1": home_q[0], "home_q2": home_q[1],
                    "home_q3": home_q[2], "home_q4": home_q[3],
                    "away_q1": away_q[0], "away_q2": away_q[1],
                    "away_q3": away_q[2], "away_q4": away_q[3],
                }

                # Derived totals
                row["home_total"] = sum(home_q)
                row["away_total"] = sum(away_q)
                row["game_total"] = row["home_total"] + row["away_total"]

                # Quarter combined totals
                for q in range(1, 5):
                    row[f"q{q}_total"] = row[f"home_q{q}"] + row[f"away_q{q}"]

                # Through-3Q total
                row["thru_3q_total"] = row["q1_total"] + row["q2_total"] + row["q3_total"]
                row["q4_total"] = row["home_q4"] + row["away_q4"]

                # Halftime
                row["h1_total"] = row["q1_total"] + row["q2_total"]
                row["h2_total"] = row["q3_total"] + row["q4_total"]

                rows.append(row)

            except Exception as e:
                continue

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.sort_values("game_date").reset_index(drop=True)
    return df


def main():
    print("=" * 70)
    print("NBA QUARTER-BY-QUARTER SCORING ANALYSIS")
    print("Goal: Find prediction advantage on totals entering Q4")
    print("=" * 70)

    df = load_quarter_data()
    print(f"\nLoaded {len(df):,} games across seasons: {sorted(df['season'].unique())}")

    # Separate regulation-only for clean analysis
    reg = df[~df["has_ot"]].copy()
    print(f"Regulation-only games: {len(reg):,} (excluded {len(df) - len(reg):,} OT games)")

    # ── 1. Basic Quarter Scoring Distributions ────────────────────────────
    print("\n" + "=" * 70)
    print("1. QUARTER SCORING DISTRIBUTIONS (Combined both teams)")
    print("=" * 70)

    for q in range(1, 5):
        col = f"q{q}_total"
        print(f"\n  Q{q}: mean={reg[col].mean():.1f}  std={reg[col].std():.1f}  "
              f"median={reg[col].median():.1f}  "
              f"range=[{reg[col].min()}-{reg[col].max()}]")

    print(f"\n  Through 3Q: mean={reg['thru_3q_total'].mean():.1f}  "
          f"std={reg['thru_3q_total'].std():.1f}")
    print(f"  Q4 alone:   mean={reg['q4_total'].mean():.1f}  "
          f"std={reg['q4_total'].std():.1f}")
    print(f"  Full game:  mean={reg['game_total'].mean():.1f}  "
          f"std={reg['game_total'].std():.1f}")

    # ── 2. Q4 as % of game total ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. Q4 AS PROPORTION OF GAME TOTAL")
    print("=" * 70)

    reg["q4_pct"] = reg["q4_total"] / reg["game_total"]
    for q in range(1, 5):
        reg[f"q{q}_pct"] = reg[f"q{q}_total"] / reg["game_total"]
        print(f"  Q{q} share: {reg[f'q{q}_pct'].mean():.1%} (std={reg[f'q{q}_pct'].std():.1%})")

    # ── 3. Correlation: Through-3Q total vs Q4 total ─────────────────────
    print("\n" + "=" * 70)
    print("3. CORRELATION: THROUGH-3Q TOTAL vs Q4 SCORING")
    print("=" * 70)

    r, p = stats.pearsonr(reg["thru_3q_total"], reg["q4_total"])
    print(f"\n  Pearson r = {r:.4f} (p = {p:.2e})")

    r_s, p_s = stats.spearmanr(reg["thru_3q_total"], reg["q4_total"])
    print(f"  Spearman r = {r_s:.4f} (p = {p_s:.2e})")

    # Correlation of Q1-Q3 pace with Q4
    for q in range(1, 4):
        r_q, _ = stats.pearsonr(reg[f"q{q}_total"], reg["q4_total"])
        print(f"  Q{q} vs Q4 correlation: r = {r_q:.4f}")

    # ── 4. Regression to mean analysis ────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. REGRESSION TO MEAN: Q4 SCORING BY THROUGH-3Q PACE")
    print("=" * 70)

    # Bucket games by through-3Q total
    reg["thru_3q_bucket"] = pd.qcut(reg["thru_3q_total"], q=5,
                                      labels=["Very Low", "Low", "Medium", "High", "Very High"])

    bucket_stats = reg.groupby("thru_3q_bucket", observed=True).agg(
        n_games=("game_id", "count"),
        avg_thru_3q=("thru_3q_total", "mean"),
        avg_q4=("q4_total", "mean"),
        std_q4=("q4_total", "std"),
        avg_final=("game_total", "mean"),
        implied_q4=("thru_3q_total", lambda x: reg["game_total"].mean() - x.mean()),
    ).round(1)

    print(f"\n  {'Pace Bucket':<12} {'Games':>6} {'Avg 3Q':>8} {'Avg Q4':>8} {'Q4 Std':>8} {'Avg Final':>10} {'Naive Q4*':>10}")
    print("  " + "-" * 66)

    overall_q4_mean = reg["q4_total"].mean()
    for idx, row in bucket_stats.iterrows():
        naive_q4 = row["avg_thru_3q"] / 3  # if Q4 matched per-quarter pace
        print(f"  {idx:<12} {int(row['n_games']):>6} {row['avg_thru_3q']:>8.1f} "
              f"{row['avg_q4']:>8.1f} {row['std_q4']:>8.1f} {row['avg_final']:>10.1f} {naive_q4:>10.1f}")

    print(f"\n  * Naive Q4 = what Q4 would be if it matched per-quarter pace from Q1-Q3")
    print(f"  * Overall Q4 mean = {overall_q4_mean:.1f}")
    print(f"\n  KEY INSIGHT: Compare 'Avg Q4' across buckets.")
    print(f"  If Q4 is similar across all buckets → strong regression to mean → prediction advantage exists.")
    print(f"  If Q4 tracks pace → less regression → harder to exploit.")

    # ── 5. Implied vs actual final total ──────────────────────────────────
    print("\n" + "=" * 70)
    print("5. PROJECTED FINAL TOTAL vs ACTUAL (from 3Q extrapolation)")
    print("=" * 70)

    # Simple projection: (thru_3q / 3) * 4
    reg["projected_total"] = (reg["thru_3q_total"] / 3) * 4
    reg["projection_error"] = reg["game_total"] - reg["projected_total"]

    print(f"\n  Projection method: (3Q total / 3) × 4")
    print(f"  Mean error (actual - projected): {reg['projection_error'].mean():.2f}")
    print(f"  Std of error: {reg['projection_error'].std():.1f}")
    print(f"  MAE: {reg['projection_error'].abs().mean():.1f}")

    # Better projection using regression
    from sklearn.linear_model import LinearRegression
    X = reg[["thru_3q_total"]].values
    y = reg["game_total"].values
    lr = LinearRegression().fit(X, y)
    print(f"\n  Linear regression: Final = {lr.coef_[0]:.4f} × 3Q_total + {lr.intercept_:.2f}")
    print(f"  R² = {lr.score(X, y):.4f}")

    reg["lr_projected"] = lr.predict(X)
    reg["lr_error"] = reg["game_total"] - reg["lr_projected"]
    print(f"  Regression MAE: {reg['lr_error'].abs().mean():.1f}")

    shrinkage = lr.coef_[0]
    print(f"\n  Shrinkage coefficient: {shrinkage:.4f}")
    print(f"  (1.333 = no regression, 1.0 = full regression to mean)")
    print(f"  Amount of regression: {(4/3 - shrinkage) / (4/3 - 1) * 100:.1f}%")

    # ── 6. Prediction simulation: over/under on projected 4Q total ───────────
    print("\n" + "=" * 70)
    print("6. PREDICTION SIMULATION: OVER/UNDER ON Q4 TOTAL")
    print("=" * 70)

    # Simulate: if the live total line at end of Q3 is set at (pregame_total - thru_3q),
    # but we know Q4 scoring regresses to the mean...

    # Strategy: When thru_3q pace is HIGH, bet UNDER on remaining total
    #           When thru_3q pace is LOW, bet OVER on remaining total

    # Use the naive Q4 projection (pace-based) as proxy for live line
    reg["naive_q4_line"] = reg["thru_3q_total"] / 3
    reg["q4_over_naive"] = reg["q4_total"] > reg["naive_q4_line"]

    print(f"\n  If live Q4 line = (3Q total / 3):")
    print(f"  Over hits: {reg['q4_over_naive'].mean():.1%}")
    print(f"  Under hits: {(~reg['q4_over_naive']).mean():.1%}")

    # By pace bucket
    print(f"\n  {'Pace Bucket':<12} {'Over %':>8} {'Under %':>8} {'Avg Q4':>8} {'Line':>8} {'Edge':>8}")
    print("  " + "-" * 56)

    for bucket in ["Very Low", "Low", "Medium", "High", "Very High"]:
        sub = reg[reg["thru_3q_bucket"] == bucket]
        over_pct = sub["q4_over_naive"].mean()
        avg_q4 = sub["q4_total"].mean()
        avg_line = sub["naive_q4_line"].mean()
        edge = avg_q4 - avg_line
        print(f"  {bucket:<12} {over_pct:>8.1%} {1-over_pct:>8.1%} {avg_q4:>8.1f} {avg_line:>8.1f} {edge:>+8.1f}")

    # ── 7. Team-level Q4 tendencies ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("7. TEAM Q4 SCORING TENDENCIES (min 50 games)")
    print("=" * 70)

    # Combine home and away perspectives
    home_q4 = reg[["home_team", "home_q4", "home_q1", "home_q2", "home_q3"]].rename(
        columns={"home_team": "team", "home_q4": "q4", "home_q1": "q1", "home_q2": "q2", "home_q3": "q3"})
    away_q4 = reg[["away_team", "away_q4", "away_q1", "away_q2", "away_q3"]].rename(
        columns={"away_team": "team", "away_q4": "q4", "away_q1": "q1", "away_q2": "q2", "away_q3": "q3"})
    team_df = pd.concat([home_q4, away_q4])

    team_df["thru_3q_per_q"] = (team_df["q1"] + team_df["q2"] + team_df["q3"]) / 3
    team_df["q4_vs_pace"] = team_df["q4"] - team_df["thru_3q_per_q"]

    team_stats = team_df.groupby("team").agg(
        n_games=("q4", "count"),
        avg_q4=("q4", "mean"),
        q4_std=("q4", "std"),
        avg_q4_vs_pace=("q4_vs_pace", "mean"),
    ).round(2)

    team_stats = team_stats[team_stats["n_games"] >= 50].sort_values("avg_q4_vs_pace")

    print(f"\n  Teams that UNDERPERFORM their pace in Q4 (bet UNDER):")
    print(f"  {'Team':<6} {'Games':>6} {'Avg Q4':>8} {'Q4 vs Pace':>12}")
    for idx, row in team_stats.head(10).iterrows():
        print(f"  {idx:<6} {int(row['n_games']):>6} {row['avg_q4']:>8.1f} {row['avg_q4_vs_pace']:>+12.2f}")

    print(f"\n  Teams that OUTPERFORM their pace in Q4 (bet OVER):")
    for idx, row in team_stats.tail(10).iterrows():
        print(f"  {idx:<6} {int(row['n_games']):>6} {row['avg_q4']:>8.1f} {row['avg_q4_vs_pace']:>+12.2f}")

    # ── 8. Season-by-season stability ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("8. SEASON-BY-SEASON Q4 PATTERNS")
    print("=" * 70)

    season_stats = reg.groupby("season").agg(
        n_games=("game_id", "count"),
        avg_total=("game_total", "mean"),
        avg_q4=("q4_total", "mean"),
        q4_std=("q4_total", "std"),
        q4_pct=("q4_pct", "mean"),
        corr_3q_q4=("thru_3q_total", lambda x: x.corr(reg.loc[x.index, "q4_total"])),
    ).round(3)

    print(f"\n  {'Season':<10} {'Games':>6} {'Avg Total':>10} {'Avg Q4':>8} {'Q4 Std':>8} {'Q4 %':>8} {'3Q-Q4 Corr':>12}")
    print("  " + "-" * 66)
    for idx, row in season_stats.iterrows():
        print(f"  {idx:<10} {int(row['n_games']):>6} {row['avg_total']:>10.1f} {row['avg_q4']:>8.1f} "
              f"{row['q4_std']:>8.1f} {row['q4_pct']:>8.1%} {row['corr_3q_q4']:>12.3f}")

    # ── 9. Blowout effect ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("9. BLOWOUT EFFECT: MARGIN THROUGH 3Q vs Q4 SCORING")
    print("=" * 70)

    reg["margin_3q"] = (reg["home_q1"] + reg["home_q2"] + reg["home_q3"]) - \
                        (reg["away_q1"] + reg["away_q2"] + reg["away_q3"])
    reg["abs_margin_3q"] = reg["margin_3q"].abs()

    reg["margin_bucket"] = pd.cut(reg["abs_margin_3q"],
                                   bins=[0, 5, 10, 15, 20, 50],
                                   labels=["0-5", "6-10", "11-15", "16-20", "21+"])

    margin_stats = reg.groupby("margin_bucket", observed=True).agg(
        n_games=("game_id", "count"),
        avg_q4=("q4_total", "mean"),
        avg_thru_3q=("thru_3q_total", "mean"),
        avg_final=("game_total", "mean"),
    ).round(1)

    print(f"\n  {'3Q Margin':<12} {'Games':>6} {'Avg 3Q':>8} {'Avg Q4':>8} {'Avg Final':>10}")
    print("  " + "-" * 48)
    for idx, row in margin_stats.iterrows():
        print(f"  {idx:<12} {int(row['n_games']):>6} {row['avg_thru_3q']:>8.1f} "
              f"{row['avg_q4']:>8.1f} {row['avg_final']:>10.1f}")

    print(f"\n  KEY: Blowouts (large 3Q margin) tend to have {'LOWER' if margin_stats.iloc[-1]['avg_q4'] < margin_stats.iloc[0]['avg_q4'] else 'HIGHER'} Q4 scoring")
    print(f"  → Garbage time / starters pulled affects Q4 totals")

    # ── 10. Summary and recommendations ──────────────────────────────────
    print("\n" + "=" * 70)
    print("10. SUMMARY & PREDICTION IMPLICATIONS")
    print("=" * 70)

    regression_pct = (4/3 - shrinkage) / (4/3 - 1) * 100

    print(f"""
  FINDINGS:

  1. Q4 scoring averages {overall_q4_mean:.1f} pts (combined), accounting for
     {reg['q4_pct'].mean():.1%} of game totals.

  2. Regression to mean: {regression_pct:.0f}% — Q4 scoring partially regresses
     toward the overall mean regardless of Q1-Q3 pace.
     - Linear model: Final = {lr.coef_[0]:.3f} × 3Q_total + {lr.intercept_:.1f}
     - This means for every 10 extra points scored through 3Q,
       the final total only increases by ~{lr.coef_[0] * 10:.1f} points (not 13.3).

  3. 3Q-to-Q4 correlation: r = {r:.3f} — {'weak' if abs(r) < 0.15 else 'moderate' if abs(r) < 0.3 else 'strong'}.

  4. Blowout effect: Games with large 3Q margins see depressed Q4 scoring
     (starters pulled, pace slows).

  PREDICTION IMPLICATIONS:

  - If live lines at end of Q3 simply extrapolate pace, there IS an edge:
    • High-scoring 3Q games → bet UNDER on Q4/final total
    • Low-scoring 3Q games → bet OVER on Q4/final total

  - The edge is largest for extreme pace games (very high or very low 3Q totals).

  - Blowouts compound the effect — large margins + high totals = strongest UNDER signal.

  - CAVEAT: Market sources know about regression to the mean. The real edge depends
    on whether live lines fully price in this regression. Track actual live lines
    vs. these projections to quantify the exploitable edge.
""")

    # Save detailed data
    out_dir = Path(__file__).parent.parent / "analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    reg.to_csv(out_dir / "quarter_scoring_analysis.csv", index=False)
    print(f"  Detailed data saved to analysis/output/quarter_scoring_analysis.csv")


if __name__ == "__main__":
    main()
