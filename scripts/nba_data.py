"""NBA data fetching, caching, and parsing module.

Provides functions for fetching game data from the NBA CDN and ESPN APIs.
This module is a refactored extraction from analyze_nba_2025_26_advanced.py.

Usage:
    from nba_data import fetch_json, fetch_schedule_df, fetch_all_boxscores
"""
from __future__ import annotations

# Re-export all data functions from the main advanced script.
# This module serves as the canonical import path for data functions.
# The implementations remain in the advanced script for backward compatibility.
from analyze_nba_2025_26_advanced import (
    # Constants
    SEASON,
    SEASONS,
    SCHEDULE_URL,
    BOXSCORE_URL_TMPL,
    ESPN_SCOREBOARD_URL,
    ESPN_ODDS_LIST_URL,
    OUT_DIR,
    CACHE_DIR,
    HIST_CACHE_DIR,
    BOXSCORE_CACHE,
    ESPN_SB_CACHE,
    ESPN_ODDS_CACHE,
    MODEL_DIR,
    TEAM_COORDS,
    ESPN_ABBR_MAP,
    # Utility functions
    _to_float,
    _safe_div,
    _minutes_to_float,
    american_to_prob,
    normalize_prob_pair,
    haversine_miles,
    fetch_json,
    normalize_espn_abbr,
    # Schedule / boxscore functions
    fetch_schedule_df,
    parse_team_box_rows,
    parse_player_box_rows,
    fetch_boxscore_payload,
    fetch_all_boxscores,
    # ESPN odds functions
    fetch_espn_scoreboard_for_date,
    fetch_espn_odds_for_event,
    extract_espn_events,
    fetch_all_espn_odds,
    join_espn_odds,
    # Historical data
    load_historical_season,
    load_historical_espn_odds,
)

__all__ = [
    "SEASON", "SEASONS", "SCHEDULE_URL", "BOXSCORE_URL_TMPL",
    "ESPN_SCOREBOARD_URL", "ESPN_ODDS_LIST_URL",
    "OUT_DIR", "CACHE_DIR", "HIST_CACHE_DIR", "BOXSCORE_CACHE",
    "ESPN_SB_CACHE", "ESPN_ODDS_CACHE", "MODEL_DIR",
    "TEAM_COORDS", "ESPN_ABBR_MAP",
    "_to_float", "_safe_div", "_minutes_to_float",
    "american_to_prob", "normalize_prob_pair", "haversine_miles",
    "fetch_json", "normalize_espn_abbr",
    "fetch_schedule_df", "parse_team_box_rows", "parse_player_box_rows",
    "fetch_boxscore_payload", "fetch_all_boxscores",
    "fetch_espn_scoreboard_for_date", "fetch_espn_odds_for_event",
    "extract_espn_events", "fetch_all_espn_odds", "join_espn_odds",
    "load_historical_season", "load_historical_espn_odds",
]
