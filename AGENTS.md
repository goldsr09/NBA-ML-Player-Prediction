# AGENTS.md

## Workflow Commands (verified from script CLI)

Run from repo root (`/Users/ryangoldstein/NBA`) so relative output paths resolve under `analysis/output`.

### Opening line snapshots
```bash
python3 scripts/fetch_opening_lines.py --date YYYYMMDD --days-ahead 1
```

### Early line scanner / CLV tracking
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

### Player props workflows
```bash
python3 scripts/predict_player_props.py --date YYYYMMDD --days 1
python3 scripts/predict_player_props.py --backtest
python3 scripts/predict_player_props.py --backtest-props
python3 scripts/predict_player_props.py --backtest-market-props
python3 scripts/predict_player_props.py --walk-forward
python3 scripts/predict_player_props.py --date YYYYMMDD --track-clv
python3 scripts/predict_player_props.py --date YYYYMMDD --grade-results
python3 scripts/predict_player_props.py --calibration-report
python3 scripts/predict_player_props.py --date YYYYMMDD --daily-report
python3 scripts/predict_player_props.py --date YYYYMMDD --deploy-status
python3 scripts/predict_player_props.py --date YYYYMMDD --weekly-retrain
python3 scripts/predict_player_props.py --date YYYYMMDD --enforce-lineup-lock --lineup-lock-minutes 30
python3 scripts/predict_player_props.py --date YYYYMMDD --asof-utc YYYY-MM-DDTHH:MM:SSZ
python3 scripts/predict_player_props.py --date YYYYMMDD --force-starter-refresh
python3 scripts/predict_player_props.py --backtest-market-props --market-backtest-max-dates 60 --record-weekly-market-check --date YYYYMMDD
```

### Kalshi Q4 workflows
```bash
python3 scripts/kalshi_q4_edge.py
python3 scripts/kalshi_q4_edge.py --watch 60 --log
python3 scripts/kalshi_q4_edge.py --retrain
python3 scripts/grade_kalshi_snapshots.py --date YYYYMMDD --save
python3 scripts/grade_kalshi_snapshots.py --summary
```

### Q4 model training
```bash
python3 scripts/train_q4_model.py
python3 scripts/train_q4_model.py --tune
python3 scripts/train_q4_model.py --eval-only
```

### Historical season data fetch
```bash
python3 scripts/fetch_historical_seasons.py
python3 scripts/fetch_historical_seasons.py --seasons 2023-24 2024-25 --workers 8
python3 scripts/fetch_historical_seasons.py --seasons 2024-25 --skip-odds
```

### Daily props cron pipeline
```bash
bash scripts/daily_props_cron.sh
bash scripts/daily_props_cron.sh YYYYMMDD
```

## TODO
- Confirm whether cron-based snapshot scheduling should be documented here or kept only in script output text.
