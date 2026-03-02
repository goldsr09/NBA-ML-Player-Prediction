#!/usr/bin/env python3
"""
Kalshi Q4 Total Points Edge Finder

Pulls live NBA scores + Kalshi over/under markets to identify 4th-quarter
betting edges. Uses a trained XGBoost model (from train_q4_model.py) for
projections, with automatic fallback to linear regression.

Usage:
    python3 scripts/kalshi_q4_edge.py              # Live scan
    python3 scripts/kalshi_q4_edge.py --watch       # Re-scan every 60s
    python3 scripts/kalshi_q4_edge.py --watch 30    # Re-scan every 30s
    python3 scripts/kalshi_q4_edge.py --log         # Log Kalshi market snapshots
    python3 scripts/kalshi_q4_edge.py --retrain     # Retrain model then scan
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
ANALYSIS_OUTPUT = Path("/Users/ryangoldstein/NBA/analysis/output")
MODEL_PATH = ANALYSIS_OUTPUT / "models" / "q4_total_model.joblib"
LOG_DIR = ANALYSIS_OUTPUT / "kalshi_logs"

# ── Linear regression fallback ────────────────────────────────────────────
# Final = 1.0604 × thru_3q_total + 44.34  (R² = 0.787, 5,516 games)
REG_COEF = 1.0604
REG_INTERCEPT = 44.34
OVERALL_Q4_MEAN = 54.7

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
NBA_SCOREBOARD = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"

# ── Signal gating thresholds ──────────────────────────────────────────────
# Only fire signals when market conditions meet all of these.
MIN_PERIOD = 3           # Must be end-of-Q3 or later (period >= 3 with clock <= 30s, or period >= 4)
MIN_VOLUME = 10          # Minimum contract volume on the strike
MAX_SPREAD = 12          # Max bid-ask spread (cents) to consider a market liquid
MIN_EV_AFTER_FEES = 2.0  # Minimum EV (cents) after Kalshi fee (currently 0)
MIN_EDGE_PCT = 3.0       # Minimum edge % to show a signal at all

# Team tricode mappings (NBA CDN → Kalshi event ticker abbreviations)
TRICODE_MAP = {
    "PHX": "PHX", "PHO": "PHX",
    "GS": "GSW", "GSW": "GSW",
    "SA": "SAS", "SAS": "SAS",
    "NY": "NYK", "NYK": "NYK",
    "NO": "NOP", "NOP": "NOP",
    "LAL": "LAL", "LAC": "LAC",
}

# ── Global model state (loaded once at startup) ──────────────────────────

ML_ARTIFACT = None  # Set by load_ml_model()


# ── Model loading ─────────────────────────────────────────────────────────

def load_ml_model():
    """Try to load the trained ML model artifact. Returns None on failure."""
    if not MODEL_PATH.exists():
        return None
    try:
        import joblib
        artifact = joblib.load(MODEL_PATH)
        # Validate required keys
        for key in ("model", "imputer", "features", "conditional_std", "overall_std"):
            if key not in artifact:
                print(f"  WARNING: Model artifact missing key '{key}', falling back to linear")
                return None
        return artifact
    except Exception as e:
        print(f"  WARNING: Failed to load ML model: {e}")
        return None


def check_model_staleness(artifact):
    """Print a warning if the model is more than 14 days old."""
    if artifact is None:
        return
    trained_at = artifact.get("trained_at", "")
    if not trained_at:
        return
    try:
        trained_dt = datetime.fromisoformat(trained_at)
        age_days = (datetime.now() - trained_dt).days
        if age_days > 14:
            print(f"  WARNING: Model is {age_days} days old (trained {trained_at[:10]})")
            print(f"  Consider re-training: python3 scripts/kalshi_q4_edge.py --retrain")
    except (ValueError, TypeError):
        pass


# ── ML projection ─────────────────────────────────────────────────────────

def project_final_total_ml(thru_3q, q1, q2, q3, margin_3q, home, away, artifact):
    """Project final total using the trained XGBoost model.

    Builds a feature vector from live game data + team Q4 snapshot,
    runs through imputer + model, and returns (projected_total, std).
    """
    model = artifact["model"]
    imputer = artifact["imputer"]
    features = artifact["features"]
    cond_std = artifact["conditional_std"]
    overall_std = artifact["overall_std"]
    snapshot = artifact.get("team_q4_snapshot", {})

    h1_total = q1 + q2
    thru_3q_per_q = thru_3q / 3.0
    h1_per_q = h1_total / 2.0

    abs_margin = abs(margin_3q)

    # Look up team Q4 tendencies from snapshot
    home_snap = snapshot.get(home, {})
    away_snap = snapshot.get(away, {})

    # All features here are available at inference time
    feat_vals = {
        "thru_3q_total": thru_3q,
        "q1_total": q1,
        "q2_total": q2,
        "q3_total": q3,
        "h1_total": h1_total,
        "thru_3q_per_q": thru_3q_per_q,
        "q3_vs_h1_pace": q3 - h1_per_q,
        "margin_3q": margin_3q,
        "abs_margin_3q": abs_margin,
        "is_blowout_15": int(abs_margin >= 15),
        "is_blowout_21": int(abs_margin >= 21),
        "home_q4_avg5": home_snap.get("q4_avg5", np.nan),
        "home_q4_avg10": home_snap.get("q4_avg10", np.nan),
        "away_q4_avg5": away_snap.get("q4_avg5", np.nan),
        "away_q4_avg10": away_snap.get("q4_avg10", np.nan),
        "home_q4_vs_pace_avg5": home_snap.get("q4_vs_pace_avg5", np.nan),
        "away_q4_vs_pace_avg5": away_snap.get("q4_vs_pace_avg5", np.nan),
    }

    # Build feature vector in the correct order
    X = np.array([[feat_vals.get(f, np.nan) for f in features]], dtype=np.float32)
    X = imputer.transform(X)

    projected = float(model.predict(X)[0])

    # Conditional std based on margin bucket
    if abs_margin <= 10:
        std = cond_std.get("close_0_10", overall_std)
    elif abs_margin <= 20:
        std = cond_std.get("moderate_11_20", overall_std)
    else:
        std = cond_std.get("blowout_21_plus", overall_std)

    return projected, std


# ── Linear fallback ───────────────────────────────────────────────────────

def project_final_total(thru_3q):
    """Project final total using linear regression (fallback)."""
    return REG_COEF * thru_3q + REG_INTERCEPT


def _lookup_q4_elapsed_std(fraction_done, base_std):
    """Look up empirical std for remaining scoring given Q4 elapsed fraction.

    Uses the q4_elapsed_std curve from the trained model artifact.
    Falls back to square-root decay if artifact not available.
    """
    if ML_ARTIFACT is None or "q4_elapsed_std" not in ML_ARTIFACT:
        # Fallback: square-root decay (less aggressive than linear)
        return base_std * max(0.1, (1 - fraction_done) ** 0.5)

    curve = ML_ARTIFACT["q4_elapsed_std"]
    # Curve keys: "0.00", "0.17", "0.33", "0.50", "0.67", "0.83", "1.00"
    knots = sorted((float(k), v) for k, v in curve.items())

    # Linear interpolation between nearest knots
    for i in range(len(knots) - 1):
        f0, s0 = knots[i]
        f1, s1 = knots[i + 1]
        if f0 <= fraction_done <= f1:
            t = (fraction_done - f0) / (f1 - f0) if f1 > f0 else 0
            return s0 + t * (s1 - s0)

    # Beyond last knot
    return max(knots[-1][1], 1.0)


# ── Network helpers ───────────────────────────────────────────────────────

def fetch_json(url):
    """Fetch JSON from a URL."""
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"  ERROR fetching {url}: {e}")
        return None


def get_live_games():
    """Get live NBA scores with quarter breakdowns."""
    data = fetch_json(NBA_SCOREBOARD)
    if not data:
        return []

    games = []
    for g in data.get("scoreboard", {}).get("games", []):
        ht = g["homeTeam"]
        at = g["awayTeam"]

        home_periods = ht.get("periods", [])
        away_periods = at.get("periods", [])

        home_qs = {p["period"]: p["score"] for p in home_periods}
        away_qs = {p["period"]: p["score"] for p in away_periods}

        period = g.get("period", 0)
        status = g.get("gameStatusText", "")

        game_clock = ""
        clock_seconds = None
        if status and ":" in status and not status.startswith("Final"):
            parts = status.split()
            for p in parts:
                if ":" in p and p.replace(":", "").replace(".", "").isdigit():
                    game_clock = p
                    try:
                        mins, secs = p.split(":")
                        clock_seconds = int(mins) * 60 + float(secs)
                    except ValueError:
                        pass

        games.append({
            "game_id": g.get("gameId", ""),
            "home": ht["teamTricode"],
            "away": at["teamTricode"],
            "home_score": ht.get("score", 0),
            "away_score": at.get("score", 0),
            "period": period,
            "status": status,
            "game_clock": game_clock,
            "clock_seconds": clock_seconds,
            "home_qs": home_qs,
            "away_qs": away_qs,
        })

    return games


def get_kalshi_nba_totals():
    """Fetch all open KXNBATOTAL markets from Kalshi."""
    url = f"{KALSHI_API}/markets?series_ticker=KXNBATOTAL&status=open&limit=1000"
    data = fetch_json(url)
    if not data:
        return {}

    events = {}
    for m in data.get("markets", []):
        et = m["event_ticker"]
        if et not in events:
            events[et] = []
        events[et].append({
            "ticker": m["ticker"],
            "strike": m.get("floor_strike", 0),
            "yes_bid": m.get("yes_bid", 0),
            "yes_ask": m.get("yes_ask", 0),
            "no_bid": m.get("no_bid", 0),
            "no_ask": m.get("no_ask", 0),
            "volume": m.get("volume", 0),
            "last_price": m.get("last_price", 0),
        })

    for et in events:
        events[et].sort(key=lambda x: x["strike"])

    return events


def match_game_to_kalshi(game, kalshi_events):
    """Match an NBA game to its Kalshi event ticker."""
    away = game["away"]
    home = game["home"]

    for et in kalshi_events:
        et_upper = et.upper()
        if away in et_upper and home in et_upper:
            return et

    away_alt = TRICODE_MAP.get(away, away)
    home_alt = TRICODE_MAP.get(home, home)
    for et in kalshi_events:
        et_upper = et.upper()
        if away_alt in et_upper and home_alt in et_upper:
            return et

    return None


def compute_thru_3q(game):
    """Compute through-3Q total from quarter scores."""
    home_qs = game["home_qs"]
    away_qs = game["away_qs"]
    thru_3q = 0
    for q in [1, 2, 3]:
        thru_3q += home_qs.get(q, 0) + away_qs.get(q, 0)
    return thru_3q


def implied_probability_to_line(markets):
    """Estimate the market-implied total from Kalshi strike prices."""
    best_strike = None
    best_diff = float("inf")

    for m in markets:
        yes_mid = (m["yes_bid"] + m["yes_ask"]) / 2
        diff = abs(yes_mid - 50)
        if diff < best_diff:
            best_diff = diff
            best_strike = m["strike"]

    return best_strike


# ── Logging ───────────────────────────────────────────────────────────────

def log_snapshot(games, kalshi_events):
    """Write a JSON snapshot of all game scores + Kalshi prices to disk."""
    today = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%H%M%S")
    day_dir = LOG_DIR / today
    day_dir.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "games": [],
    }

    for game in games:
        event = match_game_to_kalshi(game, kalshi_events)
        markets = kalshi_events.get(event, []) if event else []

        snapshot["games"].append({
            "game_id": game["game_id"],
            "home": game["home"],
            "away": game["away"],
            "home_score": game["home_score"],
            "away_score": game["away_score"],
            "period": game["period"],
            "status": game["status"],
            "game_clock": game["game_clock"],
            "clock_seconds": game["clock_seconds"],
            "home_qs": {str(k): v for k, v in game["home_qs"].items()},
            "away_qs": {str(k): v for k, v in game["away_qs"].items()},
            "kalshi_event": event,
            "kalshi_markets": markets,
        })

    out_path = day_dir / f"snapshot_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return out_path


# ── Game display ──────────────────────────────────────────────────────────

def display_game(game, kalshi_event, markets):
    """Display analysis for a single game."""
    global ML_ARTIFACT

    away = game["away"]
    home = game["home"]
    status = game["status"]
    period = game["period"]

    print(f"\n{'='*70}")
    print(f"  {away} @ {home}  |  {status}  |  Kalshi: {kalshi_event or 'NOT FOUND'}")
    print(f"{'='*70}")

    # Quarter scores
    home_qs = game["home_qs"]
    away_qs = game["away_qs"]

    q_header = "     "
    for q in range(1, 5):
        q_header += f"  Q{q}"
    q_header += "  Total"
    print(q_header)

    away_line = f"  {away}"
    for q in range(1, 5):
        away_line += f"  {away_qs.get(q, '-'):>3}"
    away_line += f"  {game['away_score']:>5}"
    print(away_line)

    home_line = f"  {home}"
    for q in range(1, 5):
        home_line += f"  {home_qs.get(q, '-'):>3}"
    home_line += f"  {game['home_score']:>5}"
    print(home_line)

    current_total = game["home_score"] + game["away_score"]
    print(f"\n  Current combined score: {current_total}")

    # Determine game phase and compute projection
    clock_secs = game.get("clock_seconds")

    if period < 3 or (period == 3 and clock_secs and clock_secs > 600):
        thru_complete = sum(home_qs.get(q, 0) + away_qs.get(q, 0) for q in range(1, period + 1))
        completed_total = sum(home_qs.get(q, 0) + away_qs.get(q, 0)
                              for q in range(1, period))
        in_progress_pts = current_total - completed_total

        print(f"\n  Game in Q{period} ({game.get('game_clock', '')}) — waiting for Q3 to complete.")
        print(f"  Current total: {current_total}")

        if period >= 2 or (period == 2 and (clock_secs is None or clock_secs < 60)):
            h1_total = (home_qs.get(1, 0) + away_qs.get(1, 0) +
                        home_qs.get(2, 0) + away_qs.get(2, 0))
            if h1_total > 0:
                naive_final = h1_total * 2
                print(f"  Halftime total: {h1_total}")
                print(f"  Naive 2x projection: {naive_final}")
        return

    # Q3 complete or Q4+ — can use our model
    thru_3q = compute_thru_3q(game)

    # For mid-Q3 games, estimate 3Q total from current scoring rate
    if period == 3 and clock_secs is not None and clock_secs > 0:
        q3_elapsed = 720 - clock_secs
        q3_so_far = (home_qs.get(3, 0) + away_qs.get(3, 0))
        if q3_elapsed > 0:
            q3_projected = q3_so_far * (720 / q3_elapsed)
        else:
            q3_projected = q3_so_far
        h1h2 = sum(home_qs.get(q, 0) + away_qs.get(q, 0) for q in [1, 2])
        thru_3q = h1h2 + q3_projected
        print(f"\n  Q3 in progress ({game.get('game_clock', '')} remaining)")
        print(f"  Q3 so far: {q3_so_far} ({q3_elapsed:.0f}s elapsed)")
        print(f"  Projected Q3: {q3_projected:.0f}")
    elif period == 3 and (clock_secs is None or clock_secs == 0):
        pass

    # ── Compute projection (ML or linear fallback) ────────────────────
    q1 = home_qs.get(1, 0) + away_qs.get(1, 0)
    q2 = home_qs.get(2, 0) + away_qs.get(2, 0)
    q3 = home_qs.get(3, 0) + away_qs.get(3, 0)
    home_3q = sum(home_qs.get(q, 0) for q in [1, 2, 3])
    away_3q = sum(away_qs.get(q, 0) for q in [1, 2, 3])
    margin_3q = home_3q - away_3q

    # Use ML only if artifact says it's robust; otherwise use linear
    # but still use artifact for conditional_std and team snapshot
    use_ml = (ML_ARTIFACT is not None and ML_ARTIFACT.get("use_ml", True))
    if use_ml:
        projected, projection_std = project_final_total_ml(
            thru_3q, q1, q2, q3, margin_3q, home, away, ML_ARTIFACT)
    elif ML_ARTIFACT is not None:
        # Linear projection but with conditional std from artifact
        projected = project_final_total(thru_3q)
        abs_margin = abs(margin_3q)
        cond_std = ML_ARTIFACT.get("conditional_std", {})
        overall_std = ML_ARTIFACT.get("overall_std", 9.2)
        if abs_margin <= 10:
            projection_std = cond_std.get("close_0_10", overall_std)
        elif abs_margin <= 20:
            projection_std = cond_std.get("moderate_11_20", overall_std)
        else:
            projection_std = cond_std.get("blowout_21_plus", overall_std)
    else:
        projected = project_final_total(thru_3q)
        projection_std = 9.2

    naive_projected = (thru_3q / 3) * 4
    q4_projected = projected - thru_3q
    pace_q4 = thru_3q / 3

    # If in Q4, adjust projection based on actual Q4 scoring + remaining time.
    # Use empirical elapsed-time std curve from training (if available).
    time_adjusted_projection = projected

    if period == 4 and clock_secs is not None:
        q4_so_far = current_total - compute_thru_3q(game)
        q4_elapsed = 720 - clock_secs
        q4_fraction_done = q4_elapsed / 720 if q4_elapsed > 0 else 0

        if q4_fraction_done > 0.1:
            q4_remaining_model = q4_projected * (1 - q4_fraction_done)
            time_adjusted_projection = compute_thru_3q(game) + q4_so_far + q4_remaining_model
            projection_std = _lookup_q4_elapsed_std(q4_fraction_done, projection_std)

    print(f"\n  Through 3Q total:       {thru_3q:.0f}")
    print(f"  Per-quarter pace:       {thru_3q/3:.1f}")
    print(f"  Naive projection (4x):  {naive_projected:.1f}")
    print(f"  Model projection:       {projected:.1f}")
    print(f"  Projected Q4:           {q4_projected:.1f}  (pace would be {pace_q4:.1f})")
    print(f"  Regression savings:     {naive_projected - projected:+.1f} pts vs naive")

    if period == 4 and clock_secs is not None:
        actual_3q = compute_thru_3q(game)
        q4_so_far = current_total - actual_3q
        q4_elapsed = 720 - clock_secs
        q4_fraction_done = q4_elapsed / 720

        print(f"\n  Q4 CLOCK ADJUSTMENT:")
        print(f"  Q4 time elapsed:        {q4_elapsed:.0f}s / 720s ({q4_fraction_done:.0%})")
        print(f"  Q4 actual so far:       {q4_so_far}")
        print(f"  Time-adjusted proj:     {time_adjusted_projection:.1f}")
        print(f"  Adjusted std:           {projection_std:.1f}")

    if not markets:
        print(f"\n  No Kalshi markets found for this game.")
        return

    # Implied Kalshi line
    implied_line = implied_probability_to_line(markets)
    print(f"\n  Kalshi implied line:    ~{implied_line}")
    if implied_line:
        model_vs_market = time_adjusted_projection - implied_line
        print(f"  Model vs Kalshi:        {model_vs_market:+.1f} pts")

    # Edge analysis on each strike using time-adjusted projection.
    # Edge is computed against executable prices (ask), not midpoint.
    from scipy.stats import norm
    edges = []
    for m in markets:
        strike = m["strike"]

        model_over_prob = norm.sf(strike, loc=time_adjusted_projection, scale=max(projection_std, 1.0)) * 100
        model_under_prob = 100 - model_over_prob

        # Executable implied probs: what you pay to enter the position
        over_cost = m["yes_ask"]    # cost to buy YES (over)
        under_cost = m["no_ask"]    # cost to buy NO (under)

        # Edge = model prob - price you pay (positive = +EV)
        edge_over = model_over_prob - over_cost
        edge_under = model_under_prob - under_cost

        # EV per contract (in cents): P(win)*payout - P(lose)*cost
        ev_buy_yes = (model_over_prob / 100) * (100 - over_cost) - (model_under_prob / 100) * over_cost
        ev_buy_no = (model_under_prob / 100) * (100 - under_cost) - (model_over_prob / 100) * under_cost

        # Gating: check if this market meets liquidity/quality thresholds
        spread = m["yes_ask"] - m["yes_bid"] if m["yes_bid"] > 0 else 99
        liquid = (m["volume"] >= MIN_VOLUME and spread <= MAX_SPREAD)

        edges.append({
            "strike": strike,
            "yes_ask": m["yes_ask"],
            "no_ask": m["no_ask"],
            "yes_bid": m["yes_bid"],
            "no_bid": m["no_bid"],
            "spread": spread,
            "volume": m["volume"],
            "over_cost": over_cost,
            "under_cost": under_cost,
            "model_over%": round(model_over_prob, 1),
            "edge_over": round(edge_over, 1),
            "edge_under": round(edge_under, 1),
            "ev_buy_yes": round(ev_buy_yes, 2),
            "ev_buy_no": round(ev_buy_no, 2),
            "liquid": liquid,
            "ticker": m["ticker"],
        })

    # Game-level gate: only signal if period meets threshold
    # (period=3 clock<=30s or period>=4 is fine; mid-Q3 extrapolation is not)
    game_eligible = (
        (period >= 4) or
        (period == 3 and (clock_secs is None or clock_secs <= 30))
    )

    print(f"\n  {'Strike':>8} {'Ask O':>6} {'Ask U':>6} {'Mdl O%':>7} {'Edge O':>7} {'Edge U':>7} "
          f"{'Signal':>8} {'EV':>7} {'Vol':>8}")
    print(f"  {'-'*70}")

    for e in edges:
        # Signal based on which side has better edge
        best_edge = max(e["edge_over"], e["edge_under"])
        signal = ""

        # Only emit signals if game + market pass gates
        if game_eligible and e["liquid"]:
            if e["edge_over"] >= MIN_EDGE_PCT and e["ev_buy_yes"] >= MIN_EV_AFTER_FEES:
                signal = "OVER" if e["edge_over"] >= 5 else "over?"
            elif e["edge_under"] >= MIN_EDGE_PCT and e["ev_buy_no"] >= MIN_EV_AFTER_FEES:
                signal = "UNDER" if e["edge_under"] >= 5 else "under?"

        marker = ""
        if signal in ("OVER", "UNDER") and best_edge >= 10:
            marker = " **"
        elif signal in ("OVER", "UNDER"):
            marker = " *"

        best_ev = e["ev_buy_yes"] if e["edge_over"] >= e["edge_under"] else e["ev_buy_no"]

        print(f"  {e['strike']:>8} {e['over_cost']:>5}c {e['under_cost']:>5}c {e['model_over%']:>6.1f}% "
              f"{e['edge_over']:>+6.1f} {e['edge_under']:>+6.1f} "
              f"{signal:>8} {best_ev:>+6.1f} {e['volume']:>8,}{marker}")

    if not game_eligible:
        print(f"\n  SIGNALS GATED: game in Q{period} — waiting for end of Q3")
        return

    # Filter to gated edges for best-bet selection
    gated = [e for e in edges if e["liquid"]]
    if not gated:
        print(f"\n  SIGNALS GATED: no markets meet liquidity thresholds "
              f"(vol>={MIN_VOLUME}, spread<={MAX_SPREAD}c)")
        return

    best_over = max(gated, key=lambda x: x["ev_buy_yes"])
    best_under = max(gated, key=lambda x: x["ev_buy_no"])

    print(f"\n  BEST BETS (vol>={MIN_VOLUME}, spread<={MAX_SPREAD}c, EV>={MIN_EV_AFTER_FEES}c):")
    if best_over["ev_buy_yes"] >= MIN_EV_AFTER_FEES and best_over["edge_over"] >= MIN_EDGE_PCT:
        print(f"    OVER  {best_over['strike']}: buy YES at {best_over['yes_ask']}c "
              f"(EV={best_over['ev_buy_yes']:+.1f}c, edge={best_over['edge_over']:+.1f}%)")
    else:
        print(f"    OVER  — no qualifying over bets")

    if best_under["ev_buy_no"] >= MIN_EV_AFTER_FEES and best_under["edge_under"] >= MIN_EDGE_PCT:
        print(f"    UNDER {best_under['strike']}: buy NO at {best_under['no_ask']}c "
              f"(EV={best_under['ev_buy_no']:+.1f}c, edge={best_under['edge_under']:+.1f}%)")
    else:
        print(f"    UNDER — no qualifying under bets")


# ── Scan loop ─────────────────────────────────────────────────────────────

def run_scan(log_enabled=False):
    """Run a single scan of all live games."""
    global ML_ARTIFACT

    # Header
    print(f"\n{'#'*70}")
    print(f"  KALSHI Q4 EDGE SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if ML_ARTIFACT is not None:
        metrics = ML_ARTIFACT.get("eval_metrics", {})
        n_games = ML_ARTIFACT.get("n_games", "?")
        mae = metrics.get("mae", 0)
        use_ml = ML_ARTIFACT.get("use_ml", True)
        mode = "XGBoost ML" if use_ml else "Linear + ML std/snapshot"
        print(f"  Model: {mode} ({n_games:,} games, MAE={mae:.1f})")
    else:
        print(f"  Model: Linear fallback (Final = {REG_COEF:.4f} x 3Q_total + {REG_INTERCEPT:.1f})")
    print(f"{'#'*70}")

    games = get_live_games()
    if not games:
        print("\n  No games found on today's scoreboard.")
        return

    kalshi = get_kalshi_nba_totals()
    print(f"\n  Found {len(games)} NBA games, {len(kalshi)} Kalshi total events")

    # Log snapshot if enabled
    if log_enabled and games:
        log_path = log_snapshot(games, kalshi)
        print(f"  Snapshot logged to {log_path}")

    # Separate by status
    live = [g for g in games if g["period"] > 0 and "Final" not in g["status"]]
    final = [g for g in games if "Final" in g["status"]]
    pregame = [g for g in games if g["period"] == 0 and "Final" not in g["status"]]

    if live:
        print(f"\n  LIVE GAMES ({len(live)}):")
        for game in live:
            event = match_game_to_kalshi(game, kalshi)
            markets = kalshi.get(event, []) if event else []
            display_game(game, event, markets)
    else:
        print(f"\n  No live games currently.")

    if final:
        print(f"\n\n{'='*70}")
        print(f"  COMPLETED GAMES — MODEL ACCURACY CHECK")
        print(f"{'='*70}")

        errors = []
        ml_errors = []
        for game in final:
            thru_3q = compute_thru_3q(game)
            actual = game["home_score"] + game["away_score"]

            # Linear baseline
            linear_proj = project_final_total(thru_3q)
            naive = (thru_3q / 3) * 4
            error = actual - linear_proj
            naive_error = actual - naive

            # ML model if available
            ml_label = ""
            if ML_ARTIFACT is not None:
                q1 = game["home_qs"].get(1, 0) + game["away_qs"].get(1, 0)
                q2 = game["home_qs"].get(2, 0) + game["away_qs"].get(2, 0)
                q3 = game["home_qs"].get(3, 0) + game["away_qs"].get(3, 0)
                home_3q = sum(game["home_qs"].get(q, 0) for q in [1, 2, 3])
                away_3q = sum(game["away_qs"].get(q, 0) for q in [1, 2, 3])
                margin_3q = home_3q - away_3q
                ml_proj, _ = project_final_total_ml(
                    thru_3q, q1, q2, q3, margin_3q,
                    game["home"], game["away"], ML_ARTIFACT)
                ml_err = actual - ml_proj
                ml_errors.append(abs(ml_err))
                ml_label = f"  ML={ml_proj:.0f} (err={ml_err:+.0f})"

            errors.append(abs(error))

            print(f"\n  {game['away']} @ {game['home']}: "
                  f"3Q={thru_3q}  Linear={linear_proj:.0f}  Actual={actual}  "
                  f"Error={error:+.0f}  (Naive={naive_error:+.0f}){ml_label}")

        if errors:
            msg = f"\n  Today's linear MAE: {sum(errors)/len(errors):.1f} pts"
            if ml_errors:
                msg += f"  |  ML MAE: {sum(ml_errors)/len(ml_errors):.1f} pts"
            print(msg)

    if pregame:
        labels = [f'{g["away"]}@{g["home"]}' for g in pregame]
        print(f"\n  Pregame: {', '.join(labels)}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    global ML_ARTIFACT

    parser = argparse.ArgumentParser(description="Kalshi Q4 Total Points Edge Finder")
    parser.add_argument("--watch", nargs="?", const=60, type=int, metavar="SECS",
                        help="Re-scan every N seconds (default: 60)")
    parser.add_argument("--log", action="store_true",
                        help="Log Kalshi market snapshots each scan cycle")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain Q4 model before scanning")
    args = parser.parse_args()

    try:
        from scipy.stats import norm  # noqa: F401
    except ImportError:
        print("ERROR: scipy required. Install with: pip install scipy")
        sys.exit(1)

    # Retrain if requested
    if args.retrain:
        print("Retraining Q4 model...")
        sys.path.insert(0, str(SCRIPT_DIR))
        from train_q4_model import train_and_save
        train_and_save()
        print()

    # Load ML model (or fall back to linear)
    ML_ARTIFACT = load_ml_model()
    if ML_ARTIFACT is not None:
        n_feats = len(ML_ARTIFACT.get("features", []))
        n_games = ML_ARTIFACT.get("n_games", "?")
        print(f"Loaded ML model: {n_feats} features, {n_games} games")
        check_model_staleness(ML_ARTIFACT)
    else:
        print("No ML model found — using linear regression fallback")
        print(f"  Train with: python3 scripts/train_q4_model.py")

    if args.watch:
        print(f"\nWatching every {args.watch}s — press Ctrl+C to stop\n")
        while True:
            try:
                run_scan(log_enabled=args.log)
                print(f"\n  Next scan in {args.watch}s...")
                time.sleep(args.watch)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        run_scan(log_enabled=args.log)


if __name__ == "__main__":
    main()
