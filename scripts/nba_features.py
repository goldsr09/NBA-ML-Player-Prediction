"""NBA feature engineering module.

Provides functions for computing rolling stats, travel features,
injury/player availability proxies, and game-level differentials.

Usage:
    from nba_features import add_rest_and_rolling_team_features, build_game_level
"""
from __future__ import annotations

from analyze_nba_2025_26_advanced import (
    # Feature engineering functions
    add_rest_and_rolling_team_features,
    add_travel_features,
    add_player_availability_proxy,
    build_game_level,
    build_team_games_and_players,
    # CV/selection functions
    time_series_cv_folds,
    shap_feature_importance,
    select_features_by_shap,
)

__all__ = [
    "add_rest_and_rolling_team_features",
    "add_travel_features",
    "add_player_availability_proxy",
    "build_game_level",
    "build_team_games_and_players",
    "time_series_cv_folds",
    "shap_feature_importance",
    "select_features_by_shap",
]
