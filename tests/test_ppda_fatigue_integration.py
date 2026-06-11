"""Integration tests for PPDA and fatigue feature group wiring."""

from __future__ import annotations

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
from polymbappe.data.ingest import ingest_team_ppda, ingest_season_minutes
from polymbappe.data.store import read_table, table_exists


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
