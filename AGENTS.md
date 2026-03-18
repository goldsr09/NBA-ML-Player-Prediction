# AGENTS.md

## Workflow Commands (verified from script CLI)

Run from repo root (`/Users/ryangoldstein/NBA`) so relative output paths resolve under `analysis/output`.

### Opening line snapshots
```bash
python3 scripts/fetch_opening_lines.py --date YYYYMMDD --days-ahead 1
```

### Early line scanner / Market efficiency tracking
```bash
python3 scripts/early_line_scanner.py --date YYYYMMDD
python3 scripts/early_line_scanner.py --date YYYYMMDD --track
python3 scripts/early_line_scanner.py --report
python3 scripts/early_line_scanner.py --date YYYYMMDD --spread-threshold 2.5 --total-threshold 3.5 --ml-threshold 0.06
```

### Walk-forward model backtest
```bash
python3 scripts/walk_forward_backtest.py
python3 scripts/walk_forward_backtest.py --folds 1,2,3 --no-market
```

### Upcoming game predictions
```bash
python3 scripts/predict_upcoming_nba.py --date YYYYMMDD --days 1
python3 scripts/predict_upcoming_nba.py --date YYYYMMDD --include-in-progress
python3 scripts/predict_upcoming_nba.py --date YYYYMMDD --output analysis/output/predictions/nba_predictions_YYYYMMDD.csv
python3 scripts/predict_upcoming_nba.py --backtest
```

### Full-season analysis pipelines
```bash
python3 scripts/analyze_nba_2025_26.py
python3 scripts/analyze_nba_2025_26_advanced.py
```

### Player props workflows
```bash
python3 scripts/predict_player_props.py --date YYYYMMDD --days 1
python3 scripts/predict_player_props.py --backtest
python3 scripts/predict_player_props.py --backtest-props
python3 scripts/predict_player_props.py --backtest-market-props
python3 scripts/predict_player_props.py --walk-forward
python3 scripts/predict_player_props.py --date YYYYMMDD --track-efficiency
python3 scripts/predict_player_props.py --date YYYYMMDD --grade-results
python3 scripts/predict_player_props.py --calibration-report
python3 scripts/predict_player_props.py --date YYYYMMDD --daily-report
python3 scripts/predict_player_props.py --date YYYYMMDD --deploy-status
python3 scripts/predict_player_props.py --date YYYYMMDD --weekly-retrain
python3 scripts/predict_player_props.py --date YYYYMMDD --enforce-lineup-lock --lineup-lock-minutes 30
python3 scripts/predict_player_props.py --date YYYYMMDD --asof-utc YYYY-MM-DDTHH:MM:SSZ
python3 scripts/predict_player_props.py --date YYYYMMDD --force-starter-refresh
python3 scripts/predict_player_props.py --backtest-market-props --market-backtest-max-dates 60 --record-weekly-market-check --date YYYYMMDD
python3 scripts/predict_player_props.py --backtest-market-props --market-backtest-fetch-missing --market-backtest-max-dates 60
python3 scripts/predict_player_props.py --ablation-box-adv --ablation-max-dates 45
python3 scripts/predict_player_props.py --ablation-market-lines
python3 scripts/predict_player_props.py --ablation-features
python3 scripts/predict_player_props.py --date YYYYMMDD --enable-experimental-market-models --market-model-max-dates 120
python3 scripts/predict_player_props.py --date YYYYMMDD --box-adv-fetch-missing --box-adv-max-fetch 100
python3 scripts/predict_player_props.py --date YYYYMMDD --keep-doubtful
python3 scripts/predict_player_props.py --date YYYYMMDD --enable-player-encoding
python3 scripts/predict_player_props.py --date YYYYMMDD --min-games 25
python3 scripts/predict_player_props.py --date YYYYMMDD --prop-lines analysis/output/props/lines_YYYYMMDD.csv
python3 scripts/predict_player_props.py --tune
python3 scripts/predict_player_props.py --tune --tune-trials 100
python3 scripts/predict_player_props.py --auto-feature-select
python3 scripts/predict_player_props.py --migrate-tracking
```

### Q4 model training
```bash
python3 scripts/train_q4_model.py
python3 scripts/train_q4_model.py --tune
python3 scripts/train_q4_model.py --eval-only
```

### Quarter scoring analysis
```bash
python3 scripts/quarter_scoring_analysis.py
```

### Historical season data fetch
```bash
python3 scripts/fetch_historical_seasons.py
python3 scripts/fetch_historical_seasons.py --seasons 2023-24 2024-25 --workers 8
python3 scripts/fetch_historical_seasons.py --seasons 2024-25 --skip-odds
```

### Supplemental data fetch
```bash
python3 scripts/fetch_bdl_tracking.py --seasons 2024 2025
python3 scripts/fetch_bdl_tracking.py --max-dates 30
python3 scripts/fetch_bdl_tracking.py --max-dates 30 --dry-run
python3 scripts/fetch_nba_rotation_matchups.py --max-games 200
python3 scripts/fetch_nba_defensive_scoring.py
python3 scripts/fetch_nba_defensive_scoring.py --endpoint defensive --max-games 200
python3 scripts/fetch_nba_defensive_scoring.py --endpoint scoring --max-games 200
```

### Player prop model A/B comparison
```bash
python3 scripts/ab_compare_prop_models.py
python3 scripts/ab_compare_prop_models.py --max-dates 45 --bootstrap 2000 --seed 7
python3 scripts/ab_compare_prop_models.py --max-dates 60 --test-frac 0.3 --bet-size 100
```

### Daily props cron pipeline
```bash
bash scripts/daily_props_cron.sh
bash scripts/daily_props_cron.sh YYYYMMDD
```

## TODO
- Confirm whether cron-based snapshot scheduling should be documented here or kept only in script output text.
- Reconfirm Kalshi Q4 workflow docs if `scripts/kalshi_q4_edge.py` and `scripts/grade_kalshi_snapshots.py` are reintroduced.
- Decide whether `scripts/backtest_feb27_props.py` should be documented as a standard workflow or kept as a one-date archival utility.
