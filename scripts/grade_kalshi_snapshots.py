#!/usr/bin/env python3
"""
Grade Kalshi Q4 Snapshot Logs

Reads logged snapshots from kalshi_q4_edge.py --log, identifies decision
points (end of Q3 / early Q4), grades against final outcomes, and computes
realized P/L and closing-line value (CLV).

Mirrors the scanner's exact projection mode and gating policy so that
reported P/L matches what live signals would have produced.

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

import os as _os
ANALYSIS_OUTPUT = Path(
    _os.environ.get("NBA_OUTPUT_DIR",
                     str(Path(__file__).resolve().parent.parent / "analysis" / "output"))
)
LOG_DIR = ANALYSIS_OUTPUT / "kalshi_logs"
RESULTS_CSV = ANALYSIS_OUTPUT / "kalshi_graded_results.csv"

# Import gating constants from the scanner so grader policy is identical.
sys.path.insert(0, str(Path(__file__).parent))
from kalshi_q4_edge import (
    MIN_VOLUME, MAX_SPREAD, MIN_EV_AFTER_FEES, MIN_EDGE_PCT,
    project_final_total, project_final_total_ml,
)

# Linear fallback constants (duplicated here to avoid circular state issues)
_LINEAR_COEF = 1.0604
_LINEAR_INTERCEPT = 44.34


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


def find_decision_snapshot(game_snapshots):
    """From a list of per-game snapshot dicts, find the best decision point.

    Preference: end of Q3 (period=3, clock near 0) or earliest Q4 snapshot.
    Returns the game snapshot dict, or None.
    """
    candidates = []
    for gs in game_snapshots:
        period = gs.get("period", 0)
        clock = gs.get("clock_seconds")

        # End of Q3: period=3, clock <= 30s (or None = between quarters)
        if period == 3 and (clock is None or clock <= 30):
            candidates.append((0, gs))  # priority 0 = best
        # Early Q4: period=4, clock > 600 (less than 2 min played)
        elif period == 4 and clock is not None and clock > 600:
            candidates.append((1, gs))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def find_final_snapshot(game_snapshots):
    """Find the snapshot where this game is Final."""
    for gs in reversed(game_snapshots):
        if "Final" in gs.get("status", ""):
            return gs
    return None


def find_closing_snapshot(game_snapshots, decision_ts):
    """Find the last snapshot before Final for closing-line prices.

    CLV = closing_price - entry_price. The closing snapshot is the last
    one captured before the game ended.
    """
    pre_final = []
    for gs in game_snapshots:
        if "Final" not in gs.get("status", ""):
            ts = gs.get("timestamp", "")
            if ts > decision_ts:
                pre_final.append(gs)

    if not pre_final:
        return None
    return pre_final[-1]


def compute_projection(game, artifact):
    """Compute projection mirroring the scanner's exact logic.

    Respects use_ml flag: uses ML only when artifact says it's robust,
    otherwise uses linear projection with conditional std from artifact.
    """
    home_qs = {int(k): v for k, v in game.get("home_qs", {}).items()}
    away_qs = {int(k): v for k, v in game.get("away_qs", {}).items()}

    q1 = home_qs.get(1, 0) + away_qs.get(1, 0)
    q2 = home_qs.get(2, 0) + away_qs.get(2, 0)
    q3 = home_qs.get(3, 0) + away_qs.get(3, 0)
    thru_3q = q1 + q2 + q3
    margin_3q = sum(home_qs.get(q, 0) for q in [1, 2, 3]) - sum(away_qs.get(q, 0) for q in [1, 2, 3])
    abs_margin = abs(margin_3q)

    use_ml = (artifact is not None and artifact.get("use_ml", True))

    if use_ml:
        proj, std = project_final_total_ml(
            thru_3q, q1, q2, q3, margin_3q,
            game.get("home", ""), game.get("away", ""), artifact)
    elif artifact is not None:
        # Linear projection but with conditional std from artifact
        proj = _LINEAR_COEF * thru_3q + _LINEAR_INTERCEPT
        cond_std = artifact.get("conditional_std", {})
        overall_std = artifact.get("overall_std", 9.2)
        if abs_margin <= 10:
            std = cond_std.get("close_0_10", overall_std)
        elif abs_margin <= 20:
            std = cond_std.get("moderate_11_20", overall_std)
        else:
            std = cond_std.get("blowout_21_plus", overall_std)
    else:
        proj = _LINEAR_COEF * thru_3q + _LINEAR_INTERCEPT
        std = 9.2

    return proj, std, thru_3q


def game_period_eligible(game):
    """Check if game period meets decision threshold (mirrors scanner gate)."""
    period = game.get("period", 0)
    clock = game.get("clock_seconds")
    return (
        (period >= 4) or
        (period == 3 and (clock is None or clock <= 30))
    )


def market_eligible(m):
    """Check if a single market passes liquidity/quality gates."""
    yes_ask = m.get("yes_ask", 0)
    no_ask = m.get("no_ask", 0)
    yes_bid = m.get("yes_bid", 0)
    volume = m.get("volume", 0)

    if yes_ask <= 0 or no_ask <= 0:
        return False

    spread = yes_ask - yes_bid if yes_bid > 0 else 99
    return volume >= MIN_VOLUME and spread <= MAX_SPREAD


# ── Core grading ──────────────────────────────────────────────────────────

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

    # Collect per-game snapshot timelines
    game_timelines = {}  # game_id -> [snapshot_dicts]
    for snap in snapshots:
        for g in snap.get("games", []):
            gid = g.get("game_id")
            if gid:
                entry = {"timestamp": snap["timestamp"], **g,
                         "kalshi_markets": g.get("kalshi_markets", [])}
                game_timelines.setdefault(gid, []).append(entry)

    from scipy.stats import norm

    trades = []
    for game_id, timeline in game_timelines.items():
        # Find decision point
        dec_game = find_decision_snapshot(timeline)
        if dec_game is None:
            continue

        # Game-level period gate
        if not game_period_eligible(dec_game):
            continue

        # Find final result
        final_game = find_final_snapshot(timeline)
        if final_game is None:
            continue

        actual_total = final_game.get("home_score", 0) + final_game.get("away_score", 0)
        if actual_total == 0:
            continue

        # Compute projection at decision time (mirrors scanner mode)
        proj, std, thru_3q = compute_projection(dec_game, artifact)

        # Find closing snapshot for CLV
        closing_game = find_closing_snapshot(timeline, dec_game.get("timestamp", ""))
        closing_markets = {}
        if closing_game:
            for cm in closing_game.get("kalshi_markets", []):
                closing_markets[cm.get("strike")] = cm

        # Grade each market strike at decision time
        dec_markets = dec_game.get("kalshi_markets", [])
        if not dec_markets:
            continue

        for m in dec_markets:
            strike = m.get("strike", 0)
            yes_ask = m.get("yes_ask", 0)
            no_ask = m.get("no_ask", 0)
            yes_bid = m.get("yes_bid", 0)
            volume = m.get("volume", 0)

            # Market-level gate (identical to scanner)
            if not market_eligible(m):
                continue

            model_over_pct = norm.sf(strike, loc=proj, scale=max(std, 1.0)) * 100
            model_under_pct = 100 - model_over_pct

            # Edge against executable prices (mirrors scanner)
            edge_over = model_over_pct - yes_ask
            edge_under = model_under_pct - no_ask

            # EV per contract
            ev_buy_yes = (model_over_pct / 100) * (100 - yes_ask) - (model_under_pct / 100) * yes_ask
            ev_buy_no = (model_under_pct / 100) * (100 - no_ask) - (model_over_pct / 100) * no_ask

            # Determine side using scanner's exact gating logic
            side = None
            cost = 0
            edge = 0
            ev = 0
            if edge_over >= MIN_EDGE_PCT and ev_buy_yes >= MIN_EV_AFTER_FEES:
                side = "OVER"
                cost = yes_ask
                edge = edge_over
                ev = ev_buy_yes
            elif edge_under >= MIN_EDGE_PCT and ev_buy_no >= MIN_EV_AFTER_FEES:
                side = "UNDER"
                cost = no_ask
                edge = edge_under
                ev = ev_buy_no

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

            # Model edge (what the model thinks the edge is at entry)
            model_edge = edge

            # True CLV: compare entry price to closing (last pre-final) price.
            # CLV > 0 means the market moved toward our position after entry.
            clv = np.nan
            closing_m = closing_markets.get(strike)
            if closing_m is not None:
                if side == "OVER":
                    closing_price = closing_m.get("yes_ask", 0)
                    if closing_price > 0:
                        clv = float(closing_price - cost)
                else:
                    closing_price = closing_m.get("no_ask", 0)
                    if closing_price > 0:
                        clv = float(closing_price - cost)

            trades.append({
                "date": day_dir.name,
                "game_id": game_id,
                "home": dec_game.get("home", ""),
                "away": dec_game.get("away", ""),
                "decision_time": dec_game.get("timestamp", ""),
                "period": dec_game.get("period", 0),
                "thru_3q": thru_3q,
                "projection": round(proj, 1),
                "std": round(std, 2),
                "strike": strike,
                "side": side,
                "cost": cost,
                "model_edge": round(model_edge, 1),
                "ev": round(ev, 1),
                "model_prob": round(model_over_pct if side == "OVER" else model_under_pct, 1),
                "actual_total": actual_total,
                "won": won,
                "pnl": round(payout, 2),
                "clv": round(clv, 1) if not np.isnan(clv) else None,
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
          f"{'Edge':>6} {'EV':>6} {'Actual':>7} {'Won':>4} {'P/L':>7} {'CLV':>6}")
    print(f"  {'-'*84}")

    for t in trades:
        label = f"{t['away']}@{t['home']}"
        won_str = "Y" if t["won"] else "N"
        clv_str = f"{t['clv']:>+5.1f}" if t["clv"] is not None else "   N/A"
        print(f"  {t['date']:<10} {label:<12} {t['strike']:>7} {t['side']:<6} "
              f"{t['cost']:>4}c {t['model_edge']:>+5.1f} {t['ev']:>+5.1f} {t['actual_total']:>7} "
              f"{won_str:>4} {t['pnl']:>+6.1f}c {clv_str}")


def print_summary(trades):
    """Print aggregate P/L summary."""
    if not trades:
        print("  No trades to summarize.")
        return

    n = len(trades)
    wins = sum(1 for t in trades if t["won"])
    total_pnl = sum(t["pnl"] for t in trades)
    avg_pnl = total_pnl / n
    avg_edge = sum(t["model_edge"] for t in trades) / n
    avg_cost = sum(t["cost"] for t in trades) / n

    roi = total_pnl / sum(t["cost"] for t in trades) * 100 if n > 0 else 0

    # CLV stats (only where available)
    clv_trades = [t for t in trades if t["clv"] is not None]
    avg_clv = sum(t["clv"] for t in clv_trades) / len(clv_trades) if clv_trades else 0

    print(f"\n  ── AGGREGATE SUMMARY ──")
    print(f"  Total trades:    {n}")
    print(f"  Win rate:        {wins}/{n} ({wins/n:.1%})")
    print(f"  Total P/L:       {total_pnl:+.1f}c")
    print(f"  Avg P/L:         {avg_pnl:+.1f}c per trade")
    print(f"  ROI:             {roi:+.1f}%")
    print(f"  Avg model edge:  {avg_edge:+.1f}%")
    print(f"  Avg cost:        {avg_cost:.0f}c")
    if clv_trades:
        print(f"  Avg CLV:         {avg_clv:+.1f}c ({len(clv_trades)}/{n} with closing data)")
    else:
        print(f"  Avg CLV:         N/A (no closing snapshots)")
    print(f"\n  Gates: vol>={MIN_VOLUME}, spread<={MAX_SPREAD}c, "
          f"edge>={MIN_EDGE_PCT}%, EV>={MIN_EV_AFTER_FEES}c")

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
              "thru_3q", "projection", "std", "strike", "side", "cost", "model_edge",
              "ev", "model_prob", "actual_total", "won", "pnl", "clv", "volume", "ticker"]

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
        use_ml = artifact.get("use_ml", True)
        mode = "ML projection" if use_ml else "linear + ML std/snapshot"
        print(f"Model: {mode} (trained {artifact.get('trained_at', '?')[:10]})")
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
        print(f"  (Gates: period>=3 end, vol>={MIN_VOLUME}, spread<={MAX_SPREAD}c, "
              f"edge>={MIN_EDGE_PCT}%, EV>={MIN_EV_AFTER_FEES}c)")


if __name__ == "__main__":
    main()
