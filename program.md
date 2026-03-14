# autoresearch — NBA Player Props

Autonomous experiment loop for improving NBA player prop prediction models.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar13`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files** for full context:
   - `program.md` — this file (your instructions)
   - `scripts/run_experiment.py` — the file you modify (config + custom features)
   - `CLAUDE.md` — project context and architecture
4. **Verify data exists**: Check that `analysis/output/models/player_features_cache_v16.pkl` (or current version) exists. If not, tell the human to run `python scripts/predict_player_props.py --weekly-retrain` first.
5. **Initialize results.tsv**: Create `results.tsv` in the project root with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

**What you CAN do:**
- Modify `scripts/run_experiment.py` — this is the only file you edit. Fair game:
  - Change XGBoost hyperparameters in `EXPERIMENT_CONFIG["xgb_params"]`
  - Change per-target overrides in `EXPERIMENT_CONFIG["xgb_params_by_target"]`
  - Add/remove feature groups via `exclude_feature_groups` and `include_feature_groups`
  - Add/remove individual features via `extra_features` and `remove_features`
  - Change signal thresholds (`signal_buffer`)
  - Write new feature engineering code in `custom_features()`
  - Any combination of the above

**What you CANNOT do:**
- Modify `scripts/predict_player_props.py` — it is the evaluation ground truth.
- Modify any other scripts or data files.
- Install new packages.
- Modify the evaluation harness in `run_experiment.py` (everything below the "DO NOT EDIT" banner).

**The goal: get the lowest overall_mae.** Secondary goals are higher overall_roi and win_rate, but MAE is the primary metric — it directly measures prediction quality.

**Simplicity criterion**: All else being equal, simpler is better. A tiny MAE improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh complexity cost against improvement magnitude. A 0.001 MAE improvement from adding 5 features? Probably keep. A 0.0005 MAE improvement from 30 lines of hacky feature engineering? Probably not.

**The first run**: Always establish the baseline first — run with the default config unmodified.

## Running an Experiment

```bash
python scripts/run_experiment.py > run.log 2>&1
```

Always redirect output to `run.log`. Do NOT let output flood your context.

## Output Format

The script prints a standardized block at the end:

```
---
overall_mae:      4.123456
overall_r2:       0.234567
overall_profit:   150.00
overall_bets:     342
overall_roi:      4.39
assessment:       GO
elapsed_seconds:  180.5
mae_points:       5.678901
mae_rebounds:     2.345678
mae_assists:      1.890123
mae_minutes:      6.789012
---
```

Extract the key metrics:
```
grep "^overall_mae:\|^overall_roi:\|^assessment:" run.log
```

If grep returns empty, the run crashed. Run `tail -n 50 run.log` to read the error.

## Logging Results

When an experiment is done, log it to `results.tsv` (tab-separated).

Header and columns:

```
commit	overall_mae	overall_roi	status	description
```

1. git commit hash (short, 7 chars)
2. overall_mae achieved — use 0.000000 for crashes
3. overall_roi (percentage) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	overall_mae	overall_roi	status	description
a1b2c3d	4.123456	4.39	keep	baseline
b2c3d4e	4.098765	5.12	keep	increase learning_rate to 0.03
c3d4e5f	4.234567	2.10	discard	exclude injury_context features
d4e5f6g	0.000000	0.0	crash	bad custom feature (KeyError)
```

## The Experiment Loop

LOOP FOREVER:

1. Look at the git state and results.tsv for context on what's been tried.
2. Modify `scripts/run_experiment.py` with an experimental idea.
3. `git commit -am "experiment: <short description>"`
4. Run the experiment: `python scripts/run_experiment.py > run.log 2>&1`
5. Read out the results: `grep "^overall_mae:\|^overall_roi:\|^assessment:\|^mae_" run.log`
6. If grep is empty, the run crashed. `tail -n 50 run.log` to debug. Fix if trivial, else skip.
7. Record the results in `results.tsv` (do NOT commit results.tsv — leave it untracked).
8. **If overall_mae improved** (lower): keep the commit, this is the new baseline.
9. **If overall_mae is equal or worse**: `git reset --hard HEAD~1` to revert.

## Experiment Ideas (starting points)

These are ordered roughly by expected impact. Try them in any order, combine ideas, go deeper on what works.

### Hyperparameter Tuning
- Increase/decrease `learning_rate` (try 0.01, 0.015, 0.03, 0.04, 0.05)
- Change `max_depth` (3, 4, 5, 6, 7)
- Change `n_estimators` (200, 300, 400, 500, 600, 800)
- Adjust regularization: `reg_lambda` (0.5, 1.0, 2.0, 5.0, 10.0), `reg_alpha` (0, 0.05, 0.1, 0.5, 1.0)
- Change `subsample` (0.7, 0.8, 0.85, 0.9, 1.0)
- Change `colsample_bytree` (0.6, 0.7, 0.8, 0.9, 1.0)
- Change `min_child_weight` (1, 3, 5, 10)
- Per-target tuning: different params for points vs rebounds vs assists

### Feature Selection
- Exclude noisy feature groups: try removing `referee`, `rotation`, `matchups_v3`, `defensive_matchup`, `scoring_context` one at a time
- Include only the strongest groups: keep only `market_lines`, `injury_context`, `recency`, `vegas_context`
- Remove individual noisy features
- Try with only rolling averages (exclude everything but base features)

### Feature Engineering (in `custom_features()`)
- Interaction features: `pre_points_avg5 * pre_minutes_avg5`
- Ratio features: `pre_points_avg5 / pre_minutes_avg5` (per-minute rates)
- Momentum: difference between avg3 and avg10 (trend)
- Matchup interactions: `matchup_pace_avg * pre_points_avg5`
- Vegas-player interactions: `implied_total * pre_usage_proxy`
- Injury-adjusted projections: `pre_points_avg5 * (1 + usage_boost_proxy)`
- Opponent defense context: `opp_pre_def_rating_avg5 * pre_points_avg5`
- Home/away splits: `player_is_home * pre_points_avg5`
- Rest effects: `player_days_rest * pre_minutes_avg5`
- B2B fatigue modeling: `is_b2b * pre_points_avg5 * pre_minutes_avg5`
- Ceiling/floor features: `pre_{target}_max10 - pre_{target}_avg10` (upside potential)
- Consistency features: `pre_{target}_std10 / pre_{target}_avg10` (coefficient of variation)

### Signal Tuning
- Change `signal_buffer` (0.01, 0.02, 0.03, 0.05, 0.07)
- Note: this affects ROI/win_rate but NOT MAE

### Advanced Ideas
- Separate configs for each target (e.g., deeper trees for points, shallower for assists)
- Non-linear feature transforms (log, sqrt of count stats)
- Binned features (discretize continuous features)
- Target-specific feature lists (remove minutes-irrelevant features from minutes model)
- Combine multiple ideas that individually showed small improvements

## Tips

- **One change at a time**: Isolate what works. Don't change 5 things simultaneously.
- **Record your reasoning**: The description in results.tsv should say what you tried and why.
- **Build on winners**: After finding something that improves MAE, try variations of that idea.
- **Don't be afraid to go back to basics**: Sometimes the best move is simplifying.
- **Check per-target MAE**: An overall improvement might come from just one target. The `mae_*` lines in the output show per-target performance.
- **Watch for overfitting signals**: If MAE improves dramatically, check that R2 also improves and ROI doesn't tank. Suspicious: MAE drops a lot but ROI goes negative.
- **Feature engineering is high-leverage**: The config knobs have diminishing returns. Creative features can unlock bigger improvements.

## Timeout

Each experiment should take 3-15 minutes depending on data size. If a run exceeds 30 minutes, kill it and treat it as a crash.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away. You are autonomous. If you run out of ideas, think harder — re-read the feature list, look at what nearly worked, try combining approaches, try more radical changes. The loop runs until the human interrupts you, period.
