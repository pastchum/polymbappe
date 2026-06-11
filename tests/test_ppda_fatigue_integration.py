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
