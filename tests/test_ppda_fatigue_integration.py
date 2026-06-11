"""Integration tests for PPDA and fatigue feature group wiring."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from polymbappe.data.tables import Table, TABLE_COLUMNS


def test_team_ppda_table_registered():
    assert hasattr(Table, "TEAM_PPDA")
    assert Table.TEAM_PPDA == "team_ppda"
    assert TABLE_COLUMNS[Table.TEAM_PPDA] == ("team", "date", "ppda")


def test_season_minutes_table_registered():
    assert hasattr(Table, "SEASON_MINUTES")
    assert Table.SEASON_MINUTES == "season_minutes"
    assert TABLE_COLUMNS[Table.SEASON_MINUTES] == ("team", "tournament", "season_minutes")


from pathlib import Path

import polars as pl

from polymbappe.config import Settings
from polymbappe.context.runtime import (
    FEATURE_GROUPS,
    SIM_CONTEXT_FEATURES,
    fixture_feature_row,
    latest_ppda,
    latest_season_load,
    build_tournament_context_features,
)
from polymbappe.data.ingest import ingest_team_ppda, ingest_season_minutes
from polymbappe.data.store import read_table, table_exists
from polymbappe.eval.backtest import Tournament


def test_ingest_team_ppda(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv = "team,date,ppda\nBrazil,2022-11-24,8.5\nBrazil,2022-11-28,9.1\n"
    (raw_dir / "team_ppda.csv").write_text(csv)

    settings = Settings(data_dir=tmp_path)
    (tmp_path / "processed").mkdir()
    count = ingest_team_ppda(settings)
    assert count == 2
    assert table_exists(Table.TEAM_PPDA, settings)
    df = read_table(Table.TEAM_PPDA, settings)
    assert df.height == 2
    assert set(df.columns) == {"team", "date", "ppda"}


def test_ingest_team_ppda_missing_file(tmp_path: Path):
    settings = Settings(data_dir=tmp_path)
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    assert ingest_team_ppda(settings) == 0


def test_ingest_season_minutes(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv = "team,tournament,season_minutes\nBrazil,WC2022,3200\nArgentina,WC2022,2800\n"
    (raw_dir / "season_minutes.csv").write_text(csv)

    settings = Settings(data_dir=tmp_path)
    (tmp_path / "processed").mkdir()
    count = ingest_season_minutes(settings)
    assert count == 2
    assert table_exists(Table.SEASON_MINUTES, settings)
    df = read_table(Table.SEASON_MINUTES, settings)
    assert df.height == 2
    assert set(df.columns) == {"team", "tournament", "season_minutes"}


def test_ppda_group_in_feature_groups():
    assert "ppda" in FEATURE_GROUPS
    assert FEATURE_GROUPS["ppda"] == ["home_ppda", "away_ppda", "ppda_diff"]


def test_ppda_in_sim_context_features():
    assert "home_ppda" in SIM_CONTEXT_FEATURES
    assert "away_ppda" in SIM_CONTEXT_FEATURES
    assert "ppda_diff" in SIM_CONTEXT_FEATURES


def test_season_load_in_feature_groups():
    assert "season_load" in FEATURE_GROUPS
    assert FEATURE_GROUPS["season_load"] == ["home_season_load", "away_season_load"]


def test_travel_in_feature_groups():
    assert "travel" in FEATURE_GROUPS
    assert FEATURE_GROUPS["travel"] == ["home_travel_km", "away_travel_km", "home_fatigued", "away_fatigued"]


def test_latest_ppda_returns_dict():
    # Brazil needs at least 2 matches so shift(1) in the rolling window yields a non-null value.
    matches = pl.DataFrame({
        "match_id": ["m0", "m1", "m2"],
        "date": [date(2022, 11, 16), date(2022, 11, 20), date(2022, 11, 24)],
        "home_team": ["Brazil", "Brazil", "Argentina"],
        "away_team": ["Japan", "Serbia", "Mexico"],
        "home_goals": [1, 2, 2],
        "away_goals": [0, 0, 0],
        "competition": ["FIFA World Cup", "FIFA World Cup", "FIFA World Cup"],
        "is_knockout": [False, False, False],
        "neutral_site": [True, True, True],
        "group": ["G", "G", "C"],
    })
    team_ppda = pl.DataFrame({
        "team": ["Brazil", "Brazil", "Brazil"],
        "date": [date(2022, 11, 16), date(2022, 11, 20), date(2022, 11, 24)],
        "ppda": [7.5, 8.5, 9.0],
    })
    result = latest_ppda(matches, team_ppda, as_of_date=date(2022, 11, 25))
    assert isinstance(result, dict)
    assert "Brazil" in result


def test_latest_season_load():
    minutes = pl.DataFrame({
        "team": ["Brazil", "Argentina", "Mexico"],
        "tournament": ["WC2022", "WC2022", "WC2022"],
        "season_minutes": [3200.0, 2800.0, 2000.0],
    })
    result = latest_season_load(minutes, tournament="WC2022")
    assert isinstance(result, dict)
    assert "Brazil" in result
    assert result["Brazil"] > 0
    assert result["Mexico"] < 0


def test_fixture_feature_row_includes_all_columns():
    row = fixture_feature_row(
        "Brazil", "Argentina",
        overperf={"Brazil": 0.1, "Argentina": -0.05},
        elo={"Brazil": 2100, "Argentina": 2050},
        ppda={"Brazil": 8.5, "Argentina": 11.0},
        season_load={"Brazil": 1.2, "Argentina": -0.3},
    )
    assert row["home_ppda"] == 8.5
    assert row["away_ppda"] == 11.0
    assert row["ppda_diff"] == pytest.approx(-2.5)
    assert row["home_season_load"] == 1.2
    assert row["away_season_load"] == -0.3
    assert row["home_travel_km"] == 0.0
    assert row["away_travel_km"] == 0.0
    assert row["home_fatigued"] == 0.0
    assert row["away_fatigued"] == 0.0


def test_build_tournament_context_features_includes_all_columns():
    matches = pl.DataFrame({
        "match_id": [f"m{i}" for i in range(20)],
        "date": [date(2018, 6, 1) + timedelta(days=i) for i in range(20)],
        "home_team": ["France"] * 10 + ["Brazil"] * 10,
        "away_team": ["Germany"] * 10 + ["Argentina"] * 10,
        "home_goals": [2] * 20,
        "away_goals": [1] * 20,
        "competition": ["FIFA World Cup"] * 20,
        "is_knockout": [False] * 20,
        "neutral_site": [True] * 20,
        "group": ["A"] * 20,
    })
    tournaments = (
        Tournament("WC2018", "FIFA World Cup", date(2018, 6, 15), date(2018, 7, 15)),
    )
    result = build_tournament_context_features(matches, tournaments)
    for col in SIM_CONTEXT_FEATURES:
        assert col in result.columns, f"Missing column: {col}"


def test_sim_context_features_matches_feature_groups():
    """SIM_CONTEXT_FEATURES and FEATURE_GROUPS columns must be in sync."""
    all_group_cols = set(c for cols in FEATURE_GROUPS.values() for c in cols)
    sim_set = set(SIM_CONTEXT_FEATURES)
    assert sim_set == all_group_cols
