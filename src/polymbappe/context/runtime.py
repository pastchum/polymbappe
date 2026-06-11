"""Runtime contextual features shared by training and simulation (spec 2.2, 4.1).

The contextual adjuster must see the *same* feature columns when it is fit (on historical
matches) and when it is applied per simulated match. This module is the single source of
truth for that minimal, data-light feature set — the signals derivable from match results
and Elo alone, so the contextual layer works without extra ingestion:

* ``home_xg_overperf`` / ``away_xg_overperf`` — rolling goals-minus-xG overperformance
  (proxy from goals when real xG is absent; spec 2.2 Group E permanent signal).
* ``draw_pressure`` — group-stage Elo-gap draw signal (spec 2.2 Group F).
* ``home_ppda`` / ``away_ppda`` / ``ppda_diff`` — pressing intensity (spec 2.2 Group A).
* ``home_season_load`` / ``away_season_load`` — club season minutes z-scored (spec 2.2 Group D).
* ``home_travel_km`` / ``away_travel_km`` / ``home_fatigued`` / ``away_fatigued`` — travel
  distance and rest-day fatigue flags (spec 2.2 Group D).

The adjuster's per-group toggles map onto :data:`FEATURE_GROUPS`.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from polymbappe.context.draw_pressure import stage_elo_interaction

#: Contextual feature columns, in fixed order, used by the simulation-time adjuster.
SIM_CONTEXT_FEATURES: tuple[str, ...] = (
    "home_xg_overperf", "away_xg_overperf", "draw_pressure",
    "home_ppda", "away_ppda", "ppda_diff",
    "home_season_load", "away_season_load",
    "home_travel_km", "away_travel_km", "home_fatigued", "away_fatigued",
)

#: Group -> columns mapping for the adjuster's toggle gating.
FEATURE_GROUPS: dict[str, list[str]] = {
    "xg_overperformance": ["home_xg_overperf", "away_xg_overperf"],
    "draw_pressure": ["draw_pressure"],
    "ppda": ["home_ppda", "away_ppda", "ppda_diff"],
    "season_load": ["home_season_load", "away_season_load"],
    "travel": ["home_travel_km", "away_travel_km", "home_fatigued", "away_fatigued"],
}


def latest_overperformance(
    matches: pl.DataFrame, team_xg: pl.DataFrame | None = None, as_of_date: date | None = None
) -> dict[str, float]:
    """Latest rolling xG-overperformance per team from history (0.0 if none)."""

    from polymbappe.context.sentiment import build_xg_overperformance

    overperf = build_xg_overperformance(matches, team_xg, as_of_date)
    latest = (
        overperf.drop_nulls("xg_overperformance")
        .sort(["team", "date"])
        .group_by("team")
        .agg(pl.col("xg_overperformance").last())
    )
    return {r["team"]: float(r["xg_overperformance"]) for r in latest.iter_rows(named=True)}


def latest_ppda(
    matches: pl.DataFrame, team_ppda: pl.DataFrame | None = None, as_of_date: date | None = None
) -> dict[str, float]:
    """Latest rolling PPDA per team from history (0.0 if none)."""

    from polymbappe.context.ppda import build_ppda_features

    ppda_df = build_ppda_features(matches, team_ppda, as_of_date)
    latest = (
        ppda_df.filter(pl.col("ppda_available"))
        .drop_nulls("ppda")
        .sort(["team", "date"])
        .group_by("team")
        .agg(pl.col("ppda").last())
    )
    return {r["team"]: float(r["ppda"]) for r in latest.iter_rows(named=True)}


def latest_season_load(
    season_minutes: pl.DataFrame | None = None, tournament: str | None = None
) -> dict[str, float]:
    """Latest season-load (z-scored) per team for a given tournament (0.0 if none)."""

    from polymbappe.context.fatigue import build_season_load_features

    if season_minutes is None or season_minutes.is_empty():
        return {}
    load_df = build_season_load_features(season_minutes)
    if tournament is not None:
        load_df = load_df.filter(pl.col("tournament") == tournament)
    return {
        r["team"]: float(r["season_load"])
        for r in load_df.iter_rows(named=True)
    }


def fixture_feature_row(
    home: str,
    away: str,
    overperf: dict[str, float],
    elo: dict[str, float],
    *,
    is_knockout: bool = False,
    ppda: dict[str, float] | None = None,
    season_load: dict[str, float] | None = None,
    travel: dict[str, float] | None = None,
    fatigued: dict[str, bool] | None = None,
) -> dict[str, float]:
    """Build the contextual feature row for one fixture."""

    gap = elo.get(home, 1500.0) - elo.get(away, 1500.0)
    ppda = ppda or {}
    season_load = season_load or {}
    travel = travel or {}
    fatigued = fatigued or {}
    home_ppda = ppda.get(home, 0.0)
    away_ppda = ppda.get(away, 0.0)
    return {
        "home_xg_overperf": overperf.get(home, 0.0),
        "away_xg_overperf": overperf.get(away, 0.0),
        "draw_pressure": stage_elo_interaction(is_knockout, gap),
        "home_ppda": home_ppda,
        "away_ppda": away_ppda,
        "ppda_diff": home_ppda - away_ppda,
        "home_season_load": season_load.get(home, 0.0),
        "away_season_load": season_load.get(away, 0.0),
        "home_travel_km": travel.get(home, 0.0),
        "away_travel_km": travel.get(away, 0.0),
        "home_fatigued": float(fatigued.get(home, False)),
        "away_fatigued": float(fatigued.get(away, False)),
    }


def build_tournament_context_features(
    matches: pl.DataFrame,
    tournaments: object,
    *,
    team_ppda: pl.DataFrame | None = None,
    season_minutes: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-fixture contextual features (keyed by ``match_id``) for a set of tournaments.

    For each tournament, computes xG-overperformance, PPDA, season load and Elo as of its
    start (history only), then the per-fixture feature row — the same
    :data:`SIM_CONTEXT_FEATURES` columns the simulation builds at prediction time, so the
    contextual adjuster sees an identical feature set when fit (here) and when applied live.
    Shared by training (:mod:`polymbappe.models.train`) and the backtest objective.
    """

    from polymbappe.eval.backtest import select_fixtures
    from polymbappe.features.elo import build_elo_snapshots

    rows: list[dict[str, object]] = []
    for tournament in tournaments:  # type: ignore[attr-defined]
        fixtures = select_fixtures(matches, tournament)
        if fixtures.is_empty():
            continue
        history = matches.filter(pl.col("date") < tournament.start)
        if history.is_empty():
            continue
        overperf = latest_overperformance(history)
        ppda_map = latest_ppda(history, team_ppda, as_of_date=tournament.start)
        sl_map = latest_season_load(season_minutes, tournament=tournament.name)
        snaps = (
            build_elo_snapshots(history)
            .sort(["team", "date"])
            .group_by("team")
            .agg(pl.col("rating").last())
        )
        elo = {r["team"]: float(r["rating"]) for r in snaps.iter_rows(named=True)}
        for fx in fixtures.iter_rows(named=True):
            feats = fixture_feature_row(
                fx["home_team"], fx["away_team"], overperf, elo,
                ppda=ppda_map, season_load=sl_map,
            )
            rows.append({"match_id": fx["match_id"], **feats})
    cols = {"match_id": pl.Utf8, **{c: pl.Float64 for c in SIM_CONTEXT_FEATURES}}
    return pl.DataFrame(rows, schema=cols)
