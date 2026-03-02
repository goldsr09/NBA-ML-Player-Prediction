#!/usr/bin/env python3
"""
Grade Kalshi Q4 Snapshot Logs

Reads logged snapshots from kalshi_q4_edge.py --log, identifies decision
points (end of Q3 / early Q4), grades against final outcomes, and computes
realized P/L and CLV.

Usage:
    python3 scripts/grade_kalshi_snapshots.py                    # Grade all logged days
    python3 scripts/grade_kalshi_snapshots.py --date 20260301    # Grade specific day
    python3 scripts/grade_kalshi_snapshots.py --summary          # Aggregate P/L summary
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ANALYSIS_OUTPUT = Path("/Users/ryangoldstein/NBA/analysis/output")
LOG_DIR = ANALYSIS_OUTPUT / "kalshi_logs"
RESULTS_CSV = ANALYSIS_OUTPUT / "kalshi_graded_results.csv"

# ── Helpers ───────────────────────────────────────────────────────────────

def load_ml_artifact():
    """Load the trained model artifact for re-projecting."""
    model_path = ANALYSIS_OUTPUT / "models" / "q4_total_model.joblib"
    if not model_path.exists():
        return None
    try:
        import joblib
        return joblib.load(model_path)
    except Exception:
        return None


def find_decision_snapshot(snapshots):
    """From a list of snapshots for one game, find the best decision point.

    Preference: end of Q3 (period=3, clock near 0) or earliest Q4 snapshot.
    Returns the snapshot dict and the game entry within it, or (None, None).
    """
    candidates = []
    for snap in snapshots:
        for g in snap.get("games", []):
            period = g.get("period", 0)
            clock = g.get("clock_seconds")

            # End of Q3: period=3, clock <= 30s (or None = between quarters)
            if period == 3 and (clock is None or clock <= 30):
                candidates.append((0, snap, g))  # priority 0 = best
            # Early Q4: period=4, clock > 600 (less than 2 min played)
            elif period == 4 and clock is not None and clock > 600:
                candidates.append((1, snap, g))
            # Mid Q4 fallback
            elif period == 4:
                candidates.append((2, snap, g))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0])
    _, snap, game = candidates[0]
    return snap, game


def find_final_snapshot(snapshots, game_id):
    """Find the snapshot where this game is Final."""
    for snap in reversed(snapshots):
        for g in snap.get("games", []):
            if g.get("game_id") == game_id and "Final" in g.get("status", ""):
                return snap, g
    return None, None


def compute_projection(game, artifact):
    """Compute model projection for a game at decision time."""
    home_qs = {int(k): v for k, v in game.get("home_qs", {}).items()}
    away_qs = {int(k): v for k, v in game.get("away_qs", {}).items()}

    q1 = home_qs.get(1, 0) + away_qs.get(1, 0)
    q2 = home_qs.get(2, 0) + away_qs.get(2, 0)
    q3 = home_qs.get(3, 0) + away_qs.get(3, 0)
    thru_3q = q1 + q2 + q3
    margin_3q = sum(home_qs.get(q, 0) for q in [1, 2, 3]) - sum(away_qs.get(q, 0) for q in [1, 2, 3])

    if artifact is not None:
        sys.path.insert(0, str(Path(__file__).parent))
        from kalshi_q4_edge import project_final_total_ml
        proj, std = project_final_total_ml(
            thru_3q, q1, q2, q3, margin_3q,
            game.get("home", ""), game.get("away", ""), artifact)
    else:
        proj = 1.0604 * thru_3q + 44.34
        std = 9.2

    return proj, std, thru_3q


def grade_day(day_dir, artifact):
    """Grade all games for one day's snapshots.

    Returns list of graded trade dicts.
    """
    snapshot_files = sorted(day_dir.glob("snapshot_*.json"))
    if not snapshot_files:
        return []

    # Load all snapshots
    snapshots = []
    for f in snapshot_files:
        with open(f) as fh:
            snapshots.append(json.load(fh))

    # Collect unique game_ids across all snapshots
    game_ids = set()
    for snap in snapshots:
        for g in snap.get("games", []):
            game_ids.add(g.get("game_id"))

    trades = []
    for game_id in game_ids:
        # Get all snapshots containing this game
        game_snapshots = []
        for snap in snapshots:
            for g in snap.get("games", []):
                if g.get("game_id") == game_id:
                    game_snapshots.append({"timestamp": snap["timestamp"], **g,
                                           "kalshi_markets": g.get("kalshi_markets", [])})

        if not game_snapshots:
            continue

        # Find decision point
        dec_snap, dec_game = find_decision_snapshot(
            [{"timestamp": gs["timestamp"], "games": [gs]} for gs in game_snapshots])
        if dec_snap is None:
            continue

        # Find final result
        final_snap, final_game = find_final_snapshot(
            [{"timestamp": gs["timestamp"], "games": [gs]} for gs in game_snapshots],
            game_id)

        if final_game is None:
            continue

        actual_total = final_game.get("home_score", 0) + final_game.get("away_score", 0)
        if actual_total == 0:
            continue

        # Compute projection at decision time
        proj, std, thru_3q = compute_projection(dec_game, artifact)

        # Grade each market strike at decision time
        dec_markets = dec_game.get("kalshi_markets", [])
        if not dec_markets:
            continue

        from scipy.stats import norm

        for m in dec_markets:
            strike = m.get("strike", 0)
            yes_ask = m.get("yes_ask", 0)
            no_ask = m.get("no_ask", 0)
            volume = m.get("volume", 0)

            if yes_ask <= 0 or no_ask <= 0:
                continue

            model_over_pct = norm.sf(strike, loc=proj, scale=max(std, 1.0)) * 100
            model_under_pct = 100 - model_over_pct

            edge_over = model_over_pct - yes_ask
            edge_under = model_under_pct - no_ask

            # Determine which side (if any) we'd trade
            side = None
            cost = 0
            if edge_over >= 5 and yes_ask > 0:
                side = "OVER"
                cost = yes_ask
            elif edge_under >= 5 and no_ask > 0:
                side = "UNDER"
                cost = no_ask

            if side is None:
                continue

            # Grade outcome
            went_over = actual_total > strike
            if side == "OVER":
                won = went_over
                payout = 100 - cost if won else -cost
            else:
                won = not went_over
                payout = 100 - cost if won else -cost

            # CLV: compare entry price to what fair value was at decision time
            fair_price = model_over_pct if side == "OVER" else model_under_pct
            clv = fair_price - cost

            trades.append({
                "date": day_dir.name,
                "game_id": game_id,
                "home": dec_game.get("home", ""),
                "away": dec_game.get("away", ""),
                "decision_time": dec_snap.get("timestamp", ""),
                "period": dec_game.get("period", 0),
                "thru_3q": thru_3q,
                "projection": round(proj, 1),
                "std": round(std, 2),
                "strike": strike,
                "side": side,
                "cost": cost,
                "edge": round(edge_over if side == "OVER" else edge_under, 1),
                "model_prob": round(model_over_pct if side == "OVER" else model_under_pct, 1),
                "actual_total": actual_total,
                "won": won,
                "pnl": round(payout, 2),
                "clv": round(clv, 1),
                "volume": volume,
                "ticker": m.get("ticker", ""),
            })

    return trades


# ── Output ────────────────────────────────────────────────────────────────

def print_trades(trades):
    """Print graded trades in a table."""
    if not trades:
        print("  No trades to grade.")
        return

    print(f"\n  {'Date':<10} {'Game':<12} {'Strike':>7} {'Side':<6} {'Cost':>5} "
          f"{'Edge':>6} {'Actual':>7} {'Won':>4} {'P/L':>7} {'CLV':>6}")
    print(f"  {'-'*78}")

    for t in trades:
        label = f"{t['away']}@{t['home']}"
        won_str = "Y" if t["won"] else "N"
        print(f"  {t['date']:<10} {label:<12} {t['strike']:>7} {t['side']:<6} "
              f"{t['cost']:>4}c {t['edge']:>+5.1f} {t['actual_total']:>7} "
              f"{won_str:>4} {t['pnl']:>+6.1f}c {t['clv']:>+5.1f}")


def print_summary(trades):
    """Print aggregate P/L summary."""
    if not trades:
        print("  No trades to summarize.")
        return

    n = len(trades)
    wins = sum(1 for t in trades if t["won"])
    total_pnl = sum(t["pnl"] for t in trades)
    avg_pnl = total_pnl / n
    avg_edge = sum(t["edge"] for t in trades) / n
    avg_clv = sum(t["clv"] for t in trades) / n
    avg_cost = sum(t["cost"] for t in trades) / n

    roi = total_pnl / sum(t["cost"] for t in trades) * 100 if n > 0 else 0

    print(f"\n  ── AGGREGATE SUMMARY ──")
    print(f"  Total trades:    {n}")
    print(f"  Win rate:        {wins}/{n} ({wins/n:.1%})")
    print(f"  Total P/L:       {total_pnl:+.1f}c")
    print(f"  Avg P/L:         {avg_pnl:+.1f}c per trade")
    print(f"  ROI:             {roi:+.1f}%")
    print(f"  Avg edge:        {avg_edge:+.1f}%")
    print(f"  Avg CLV:         {avg_clv:+.1f}%")
    print(f"  Avg cost:        {avg_cost:.0f}c")

    # By side
    for side in ["OVER", "UNDER"]:
        side_trades = [t for t in trades if t["side"] == side]
        if not side_trades:
            continue
        sn = len(side_trades)
        sw = sum(1 for t in side_trades if t["won"])
        spnl = sum(t["pnl"] for t in side_trades)
        print(f"\n  {side}:  {sn} trades, {sw}/{sn} wins ({sw/sn:.1%}), "
              f"P/L={spnl:+.1f}c")

    # By date
    dates = sorted(set(t["date"] for t in trades))
    if len(dates) > 1:
        print(f"\n  ── BY DATE ──")
        for d in dates:
            dt = [t for t in trades if t["date"] == d]
            dn = len(dt)
            dpnl = sum(t["pnl"] for t in dt)
            dw = sum(1 for t in dt if t["won"])
            print(f"  {d}: {dn} trades, {dw}/{dn} wins, P/L={dpnl:+.1f}c")


def save_results(trades):
    """Append graded trades to CSV for ongoing tracking."""
    import csv

    fields = ["date", "game_id", "home", "away", "decision_time", "period",
              "thru_3q", "projection", "std", "strike", "side", "cost", "edge",
              "model_prob", "actual_total", "won", "pnl", "clv", "volume", "ticker"]

    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for t in trades:
            writer.writerow({k: t[k] for k in fields})
    print(f"\n  Results appended to {RESULTS_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grade Kalshi Q4 Snapshot Logs")
    parser.add_argument("--date", type=str, help="Grade specific day (YYYYMMDD)")
    parser.add_argument("--summary", action="store_true", help="Show aggregate summary only")
    parser.add_argument("--save", action="store_true", help="Append results to CSV")
    args = parser.parse_args()

    if not LOG_DIR.exists():
        print(f"No snapshot logs found at {LOG_DIR}")
        print("Run: python3 scripts/kalshi_q4_edge.py --log --watch 60")
        sys.exit(1)

    artifact = load_ml_artifact()
    if artifact:
        print(f"Using ML model (trained {artifact.get('trained_at', '?')[:10]})")
    else:
        print("No ML model found, using linear regression")

    # Determine which days to grade
    if args.date:
        day_dirs = [LOG_DIR / args.date]
        if not day_dirs[0].exists():
            print(f"No snapshots for {args.date}")
            sys.exit(1)
    else:
        day_dirs = sorted(d for d in LOG_DIR.iterdir() if d.is_dir())

    if not day_dirs:
        print("No snapshot directories found.")
        sys.exit(1)

    all_trades = []
    for day_dir in day_dirs:
        print(f"\n{'='*60}")
        print(f"  Grading {day_dir.name}")
        print(f"{'='*60}")

        n_snapshots = len(list(day_dir.glob("snapshot_*.json")))
        print(f"  {n_snapshots} snapshots")

        trades = grade_day(day_dir, artifact)
        print(f"  {len(trades)} actionable trades found")

        if trades and not args.summary:
            print_trades(trades)

        all_trades.extend(trades)

    if all_trades:
        print_summary(all_trades)
        if args.save:
            save_results(all_trades)
    else:
        print("\n  No actionable trades across all logged days.")
        print("  (Trades require edge >= 5% at end of Q3 / early Q4)")


if __name__ == "__main__":
    main()
