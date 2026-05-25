"""Unit tests for rerandomstats.case_crossover.

Ported from ThermoStrife v0.1.1 (DOI 10.5281/zenodo.20371612) with the
following adaptations:

- ``benjamini_hochberg`` tests are excluded; that function lives in the
  ``rerandomstats.model_comparison`` submodule (added separately in
  the same v0.2.0 cycle) and has its own test file.
- Default ``rng`` is now ``None`` (non-deterministic) rather than a
  seeded constant; tests that need reproducibility pass an explicit
  seeded ``np.random.default_rng(N)``.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from rerandomstats.case_crossover import (
    build_case_crossover_frame,
    case_crossover_conditional_logit,
    daylight_hours,
    h3_within_event_contrast,
    hsiang_sigma_rescaled,
    stratified_permutation,
    stratify_case_crossover,
)

# ─────────────────────────────────────────────────────────────────
#  Daylight-hours closed form
# ─────────────────────────────────────────────────────────────────


class TestDaylightHours:
    def test_equator_is_roughly_twelve_hours(self):
        for d in (date(2020, 3, 20), date(2020, 6, 21), date(2020, 12, 21)):
            assert 11.5 <= daylight_hours(0.0, d) <= 12.5, d

    def test_summer_solstice_in_paris_is_long(self):
        assert 15.5 <= daylight_hours(48.86, date(2020, 6, 21)) <= 16.5

    def test_winter_solstice_in_paris_is_short(self):
        assert 8.0 <= daylight_hours(48.86, date(2020, 12, 21)) <= 9.0

    def test_polar_day_and_night(self):
        assert daylight_hours(85.0, date(2020, 6, 21)) == 24.0
        assert daylight_hours(85.0, date(2020, 12, 21)) == 0.0


# ─────────────────────────────────────────────────────────────────
#  Synthetic-event factory used across case-crossover tests
# ─────────────────────────────────────────────────────────────────


def _synthetic_event(
    event_id: str,
    lat: float,
    when: date,
    event_tmax: float,
    baseline_mean: float,
    baseline_std: float,
    n_baseline: int = 250,
    rng_seed: int = 0,
) -> dict:
    rng = np.random.default_rng(rng_seed)
    baseline_vals = rng.normal(baseline_mean, baseline_std, n_baseline)
    dates = pd.date_range(
        end=when - pd.Timedelta(days=8), periods=n_baseline, freq="D"
    ).date
    baseline = pd.DataFrame({"tmax": baseline_vals}, index=pd.Index(dates, name="date"))
    return {
        "event_id": event_id,
        "lat": lat,
        "lon": 0.0,
        "when": when,
        "tmax_event_c": event_tmax,
        "baseline": baseline,
    }


class TestCaseCrossover:
    def test_null_case_returns_nonsignificant(self):
        rng = np.random.default_rng(42)
        events = []
        for i in range(50):
            baseline_mean = float(rng.normal(15, 3))
            event_tmax = float(rng.normal(baseline_mean, 5))
            events.append(_synthetic_event(
                f"null_{i}", 45.0, date(2000, 6, 15),
                event_tmax, baseline_mean, 5.0,
                rng_seed=i,
            ))
        frame = build_case_crossover_frame(events)
        result = case_crossover_conditional_logit(frame, covariates=[])
        assert not result.get("skipped")
        assert result["pvalue_two_sided"] > 0.05, result

    def test_strong_positive_signal_is_detected(self):
        events = [
            _synthetic_event(
                f"hot_{i}", 45.0, date(2000, 6, 15),
                event_tmax=20.0,
                baseline_mean=15.0,
                baseline_std=3.0,
                rng_seed=i,
            )
            for i in range(40)
        ]
        frame = build_case_crossover_frame(events)
        result = case_crossover_conditional_logit(frame, covariates=[])
        assert not result.get("skipped")
        assert result["beta_per_C"] > 0
        assert result["or_per_C"] > 1.0
        assert result["pvalue_one_sided"] < 0.05, result


class TestStratifiedPermutation:
    def test_strong_positive_signal_p_is_small(self):
        events = [
            _synthetic_event(
                f"hot_{i}", 45.0, date(2000, 6, 15),
                event_tmax=20.0, baseline_mean=15.0, baseline_std=3.0,
                rng_seed=i,
            )
            for i in range(30)
        ]
        frame = build_case_crossover_frame(events)
        result = stratified_permutation(
            frame, n_perm=2000, rng=np.random.default_rng(0),
        )
        assert result["observed_diff_C"] > 0
        assert result["pvalue_one_sided"] < 0.05


class TestSigmaRescaled:
    def test_z_scores_track_expected(self):
        events = [
            _synthetic_event(
                f"sigma_{i}", 45.0, date(2000, 6, 15),
                event_tmax=15.0 + 2.0 * 3.0,  # baseline_mean + 2σ
                baseline_mean=15.0, baseline_std=3.0,
                n_baseline=500, rng_seed=i,
            )
            for i in range(20)
        ]
        result = hsiang_sigma_rescaled(events, rng=np.random.default_rng(0))
        assert result["n_events"] == 20
        assert 1.7 <= result["mean_z"] <= 2.3, result
        assert result["fraction_positive"] >= 0.9

    def test_skipped_on_empty_input(self):
        result = hsiang_sigma_rescaled([])
        assert result.get("skipped")


class TestH3WithinEventContrast:
    """Synthetic events with a programmable fetcher exercise H3.

    The fake fetcher returns ``surround_tmax`` for offsets in
    ``surround_offsets`` and ``window_tmax`` for offsets in
    ``window_offsets`` (other than offset 0, which comes from the
    event's ``tmax_event_c``).
    """

    @staticmethod
    def _fake_fetcher(window_tmax: float, surround_tmax: float):
        def _fetch(provenance, lat, lon, when, *, station_id=None):
            return _fetch.window if abs(when.day - 15) <= 1 else _fetch.surround
        _fetch.window = window_tmax
        _fetch.surround = surround_tmax
        return _fetch

    def _make(self, n, event_tmax, baseline_mean, baseline_std, seed=0):
        return [
            _synthetic_event(
                f"h3_{i}", lat=45.0, when=date(2000, 6, 15),
                event_tmax=event_tmax,
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                rng_seed=seed + i,
            )
            for i in range(n)
        ]

    def test_strong_concentration_is_detected(self):
        events = self._make(n=40, event_tmax=22.0, baseline_mean=15.0, baseline_std=3.0)
        for e in events:
            e["provenance"] = "tier1_ghcn"
            e["station_id"] = "FAKE"
        fetcher = self._fake_fetcher(window_tmax=22.0, surround_tmax=15.0)
        result = h3_within_event_contrast(
            events, fetch_fn=fetcher, rng=np.random.default_rng(0),
        )
        assert result["n_events_used"] == 40
        assert result["mean_diff_C"] > 5.0
        assert result["pvalue_one_sided"] < 0.001, result

    def test_flat_profile_gives_nonsignificant_p(self):
        events = self._make(n=40, event_tmax=15.0, baseline_mean=15.0, baseline_std=3.0)
        for e in events:
            e["provenance"] = "tier1_ghcn"
            e["station_id"] = "FAKE"
        fetcher = self._fake_fetcher(window_tmax=15.0, surround_tmax=15.0)
        result = h3_within_event_contrast(events, fetch_fn=fetcher)
        if result.get("skipped"):
            return
        assert result["pvalue_one_sided"] > 0.05, result

    def test_skipped_when_fetcher_returns_none(self):
        events = self._make(n=20, event_tmax=22.0, baseline_mean=15.0, baseline_std=3.0)
        for e in events:
            e["provenance"] = "tier1_ghcn"
            e["station_id"] = "FAKE"

        def none_fetcher(*args, **kwargs):
            return None

        result = h3_within_event_contrast(events, fetch_fn=none_fetcher)
        assert result.get("skipped")
        assert "only 0 events" in result["reason"]


@pytest.mark.parametrize("missing_field", ["baseline", "tmax_event_c"])
def test_build_frame_skips_unresolved(missing_field):
    e = _synthetic_event(
        "ok", 45.0, date(2000, 6, 15), 20.0, 15.0, 3.0, rng_seed=1,
    )
    bad = dict(e)
    bad["event_id"] = "bad"
    if missing_field == "baseline":
        bad["baseline"] = pd.DataFrame(columns=["tmax"])
    else:
        bad["tmax_event_c"] = None
    frame = build_case_crossover_frame([e, bad])
    assert "ok" in frame["event_id"].values


def test_build_frame_empty_input_returns_empty_columns():
    frame = build_case_crossover_frame([])
    assert frame.empty
    assert list(frame.columns) == ["event_id", "day", "is_case", "tmax_c", "daylight_h"]


def test_case_crossover_on_empty_frame_is_skipped():
    empty = build_case_crossover_frame([])
    result = case_crossover_conditional_logit(empty)
    assert result.get("skipped")


class TestStratifyCaseCrossover:
    def test_splits_and_skips_small(self):
        events_a = [
            _synthetic_event(
                f"a_{i}", lat=45.0, when=date(2000, 6, 15),
                event_tmax=20.0, baseline_mean=15.0, baseline_std=3.0,
                rng_seed=i,
            )
            for i in range(20)
        ]
        events_b = [
            _synthetic_event(
                f"b_{i}", lat=45.0, when=date(2000, 6, 15),
                event_tmax=20.0, baseline_mean=15.0, baseline_std=3.0,
                rng_seed=100 + i,
            )
            for i in range(3)
        ]
        for e in events_a:
            e["stratum"] = "A"
        for e in events_b:
            e["stratum"] = "B"
        events = events_a + events_b

        results = stratify_case_crossover(
            events, key_fn=lambda e: e["stratum"], min_events=5, covariates=[],
        )
        assert set(results.keys()) == {"A", "B"}
        assert not results["A"].get("skipped"), results["A"]
        assert results["A"]["n_events"] == 20
        assert results["A"]["beta_per_C"] > 0
        assert results["B"].get("skipped")
        assert results["B"]["n_events"] == 3

    def test_none_key_filters_event(self):
        events = [
            _synthetic_event(
                f"k_{i}", lat=45.0, when=date(2000, 6, 15),
                event_tmax=20.0, baseline_mean=15.0, baseline_std=3.0,
                rng_seed=i,
            )
            for i in range(20)
        ]
        key_fn = lambda e: None if e["event_id"].endswith("0") else "kept"  # noqa: E731
        results = stratify_case_crossover(events, key_fn, min_events=5, covariates=[])
        assert results["kept"]["n_events"] == 18
