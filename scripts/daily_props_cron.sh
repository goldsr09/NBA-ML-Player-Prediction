#!/usr/bin/env bash
# daily_props_cron.sh - Daily NBA player props pipeline
#
# Designed to run at ~3 PM ET (before most 7 PM ET tips).
# Fetches ESPN prop lines, runs player prop predictions, saves outputs.
# Idempotent: safe to re-run (cached data reused, outputs overwritten).
#
# crontab entry (3:00 PM ET = 20:00 UTC, or 15:00 America/New_York):
#   0 15 * * * /Users/ryangoldstein/NBA/scripts/daily_props_cron.sh
#
# Or if your system crontab honours CRON_TZ:
#   CRON_TZ=America/New_York
#   0 15 * * * /Users/ryangoldstein/NBA/scripts/daily_props_cron.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT="/Users/ryangoldstein/NBA"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
LOG_DIR="${PROJECT_ROOT}/analysis/output/prop_logs"
PREDICTIONS_DIR="${PROJECT_ROOT}/analysis/output/predictions"
PYTHON="python3"

# Target date (today in YYYYMMDD). Override with: ./daily_props_cron.sh 20260301
TARGET_DATE="${1:-$(date +%Y%m%d)}"
LOG_FILE="${LOG_DIR}/props_${TARGET_DATE}.log"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "${LOG_DIR}"
mkdir -p "${PREDICTIONS_DIR}"

# Redirect all output to the dated log file (and also to stdout if interactive)
if [ -t 1 ]; then
    # Interactive terminal: tee to both log and console
    exec > >(tee -a "${LOG_FILE}") 2>&1
else
    # Cron: log only
    exec >> "${LOG_FILE}" 2>&1
fi

echo "============================================================"
echo "NBA Daily Props Pipeline"
echo "Date:    ${TARGET_DATE}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Fetch and cache ESPN prop lines for today
# ---------------------------------------------------------------------------
echo "--- Step 1: Fetching prop lines for ${TARGET_DATE} ---"
cd "${SCRIPTS_DIR}"
${PYTHON} -c "
import sys
sys.path.insert(0, '.')
from predict_player_props import fetch_player_prop_lines, PROP_CACHE_DIR
PROP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
lines = fetch_player_prop_lines('${TARGET_DATE}')
if lines.empty:
    print('WARNING: No prop lines available for ${TARGET_DATE}')
else:
    print(f'Cached {len(lines)} prop lines for ${TARGET_DATE}')
"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Run game-level predictions (spread/total/ML)
# ---------------------------------------------------------------------------
echo "--- Step 2: Running game-level predictions ---"
cd "${SCRIPTS_DIR}"
${PYTHON} predict_upcoming_nba.py --date "${TARGET_DATE}" || {
    echo "WARNING: Game-level predictions failed (non-fatal, continuing)"
}
echo ""

# ---------------------------------------------------------------------------
# Step 3: Run player prop predictions with edge signals
# ---------------------------------------------------------------------------
echo "--- Step 3: Running player prop predictions ---"
cd "${SCRIPTS_DIR}"
${PYTHON} predict_player_props.py --date "${TARGET_DATE}"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Grade yesterday's predictions with actual results
# ---------------------------------------------------------------------------
YESTERDAY=$(date -v-1d +%Y%m%d 2>/dev/null || date -d "yesterday" +%Y%m%d 2>/dev/null || echo "")
if [ -n "${YESTERDAY}" ]; then
    echo "--- Step 4: Grading yesterday's results (${YESTERDAY}) ---"
    cd "${SCRIPTS_DIR}"
    ${PYTHON} predict_player_props.py --grade-results --date "${YESTERDAY}" || {
        echo "WARNING: Grading for ${YESTERDAY} failed (games may not be final)"
    }
    echo ""
else
    echo "--- Step 4: Skipped (could not compute yesterday's date) ---"
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 5: Weekly actionable market-line backtest snapshot (Sundays)
# ---------------------------------------------------------------------------
DOW=$(date +%u 2>/dev/null || echo "0")  # 1=Mon ... 7=Sun
if [ "${DOW}" = "7" ]; then
    echo "--- Step 5: Weekly actionable market-line backtest snapshot ---"
    cd "${SCRIPTS_DIR}"
    ${PYTHON} predict_player_props.py \
        --backtest-market-props \
        --market-backtest-max-dates 60 \
        --record-weekly-market-check \
        --date "${TARGET_DATE}" || {
        echo "WARNING: Weekly actionable market backtest snapshot failed (non-fatal)"
    }
    echo ""
else
    echo "--- Step 5: Skipped (weekly snapshot runs on Sunday) ---"
    echo ""
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Pipeline complete."
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""
echo "Outputs:"
echo "  Prop lines:  ${PROJECT_ROOT}/analysis/output/prop_cache/prop_lines_${TARGET_DATE}.csv"
echo "  Predictions: ${PREDICTIONS_DIR}/player_props_${TARGET_DATE}.csv"
echo "  Edges:       ${PREDICTIONS_DIR}/player_prop_edges_${TARGET_DATE}.csv"
echo "  Game preds:  ${PREDICTIONS_DIR}/nba_predictions_${TARGET_DATE}.csv"
echo "  Progress:    ${PROJECT_ROOT}/analysis/output/prop_logs/market_data_progress.csv"
echo "  Weekly chk:  ${PROJECT_ROOT}/analysis/output/prop_logs/market_weekly_actionable_backtest.csv"
echo "  This log:    ${LOG_FILE}"
echo "============================================================"
