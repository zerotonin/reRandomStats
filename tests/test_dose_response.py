"""Unit tests for rerandomstats.dose_response.

Synthetic data; no real-world DigiMuh or ThermoFooty data is touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rerandomstats.dose_response import (
    broken_stick_fit,
    davies_test,
    hill_fit,
    per_subject_segmented,
    pscore_test,
)

# ─────────────────────────────────────────────────────────────────
#  Synthetic-data helpers
# ─────────────────────────────────────────────────────────────────


def _broken_stick_data(
    bp: float = 20.0,
    slope_below: float = 0.05,
    slope_above: float = 0.8,
    intercept: float = 38.0,
    n: int = 300,
    noise_sd: float = 0.2,
    x_min: float = 10.0,
    x_max: float = 30.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise (x, y) from a broken-stick model + Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, n)
    y = np.where(
        x <= bp,
        intercept + slope_below * x,
        intercept + slope_below * bp + slope_above * (x - bp),
    )
    y = y + rng.normal(0, noise_sd, n)
    return x, y


def _hill_data(
    ec50: float = 20.0,
    hill_n: float = 4.0,
    y_min: float = 38.0,
    y_max: float = 42.0,
    n: int = 300,
    noise_sd: float = 0.15,
    x_min: float = 10.0,
    x_max: float = 30.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise (x, y) from a Hill curve + Gaussian noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, n)
    ratio = ec50 / np.maximum(x, 1e-10)
    y = y_min + (y_max - y_min) / (1.0 + np.power(ratio, hill_n)) + rng.normal(0, noise_sd, n)
    return x, y


def _linear_data(
    slope: float = 0.1,
    intercept: float = 38.0,
    n: int = 300,
    noise_sd: float = 0.2,
    x_min: float = 10.0,
    x_max: float = 30.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesise (x, y) from a single straight line + noise."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(x_min, x_max, n)
    y = intercept + slope * x + rng.normal(0, noise_sd, n)
    return x, y


# ─────────────────────────────────────────────────────────────────
#  broken_stick_fit
# ─────────────────────────────────────────────────────────────────


class TestBrokenStickFit:
    def test_recovers_planted_breakpoint(self):
        x, y = _broken_stick_data(bp=22.0)
        res = broken_stick_fit(x, y)
        assert res["converged"]
        # Breakpoint should be within ±1.5 °C of truth at this signal-to-noise
        assert abs(res["breakpoint"] - 22.0) < 1.5, res["breakpoint"]

    def test_slope_constraint_recovered(self):
        x, y = _broken_stick_data(slope_below=0.05, slope_above=0.8)
        res = broken_stick_fit(x, y)
        assert res["slope_above"] > res["slope_below"]
        assert res["slope_above"] > 0

    def test_too_few_observations_skipped(self):
        res = broken_stick_fit(np.arange(20.0), np.arange(20.0))
        assert not res["converged"]
        assert np.isnan(res["breakpoint"])
        assert res["n"] == 20

    def test_linear_data_rejects_constraint(self):
        """Pure straight line should fail the slope_above > slope_below test."""
        x, y = _linear_data(slope=0.1)
        res = broken_stick_fit(x, y)
        # Either the model rejects (converged=False with rejected_reason)
        # or fits with similar slopes — both acceptable for a pure line.
        if res["converged"]:
            assert abs(res["slope_above"] - res["slope_below"]) < 0.05
        else:
            assert res["rejected_reason"] is not None

    def test_returns_profile_rss_ci(self):
        x, y = _broken_stick_data(bp=22.0)
        res = broken_stick_fit(x, y)
        assert res["converged"]
        if not res["breakpoint_ci_truncated"]:
            assert res["breakpoint_ci_lo"] < res["breakpoint"] < res["breakpoint_ci_hi"]
        assert res["breakpoint_se"] > 0

    def test_r_squared_in_unit_interval(self):
        x, y = _broken_stick_data(bp=20.0)
        res = broken_stick_fit(x, y)
        assert 0.0 <= res["r_squared"] <= 1.0


# ─────────────────────────────────────────────────────────────────
#  davies_test
# ─────────────────────────────────────────────────────────────────


class TestDaviesTest:
    def test_detects_planted_breakpoint(self):
        x, y = _broken_stick_data(bp=22.0, slope_above=1.2)
        res = davies_test(x, y)
        assert res["pvalue"] < 0.05, res

    def test_no_signal_in_linear_data(self):
        x, y = _linear_data(slope=0.1, n=300, noise_sd=0.4, seed=42)
        res = davies_test(x, y)
        # No breakpoint planted; p should not be tiny.
        assert res["pvalue"] > 0.01, res

    def test_too_few_observations_returns_nan(self):
        res = davies_test(np.arange(10.0), np.arange(10.0))
        assert np.isnan(res["pvalue"])
        assert res["n"] == 10

    def test_k_parameter_controls_evaluation_count(self):
        x, y = _broken_stick_data(bp=22.0)
        res_default = davies_test(x, y, k=10)
        res_dense = davies_test(x, y, k=30)
        # Both should detect the breakpoint
        assert res_default["pvalue"] < 0.05
        assert res_dense["pvalue"] < 0.05


# ─────────────────────────────────────────────────────────────────
#  pscore_test
# ─────────────────────────────────────────────────────────────────


class TestPscoreTest:
    def test_detects_planted_breakpoint(self):
        x, y = _broken_stick_data(bp=22.0, slope_above=1.2)
        res = pscore_test(x, y)
        assert res["pvalue"] < 0.05, res

    def test_no_signal_in_linear_data(self):
        x, y = _linear_data(slope=0.1, n=300, noise_sd=0.4, seed=42)
        res = pscore_test(x, y)
        assert res["pvalue"] > 0.01, res

    def test_too_few_observations_returns_nan(self):
        res = pscore_test(np.arange(10.0), np.arange(10.0))
        assert np.isnan(res["pvalue"])

    def test_pscore_at_least_as_powerful_as_davies_on_clear_signal(self):
        """Muggeo (2016) claim: pscore typically more powerful than Davies."""
        x, y = _broken_stick_data(bp=22.0, slope_above=0.4, noise_sd=0.5, seed=7)
        davies = davies_test(x, y)
        pscore = pscore_test(x, y)
        # On reasonably clean signal both should detect; pscore typically smaller p.
        assert pscore["pvalue"] <= davies["pvalue"] * 1.5  # generous tolerance


# ─────────────────────────────────────────────────────────────────
#  hill_fit
# ─────────────────────────────────────────────────────────────────


class TestHillFit:
    def test_recovers_planted_ec50(self):
        x, y = _hill_data(ec50=20.0, hill_n=4.0)
        res = hill_fit(x, y)
        assert res["converged"]
        assert abs(res["ec50"] - 20.0) < 2.0, res["ec50"]

    def test_recovers_planted_hill_coefficient(self):
        x, y = _hill_data(ec50=20.0, hill_n=4.0)
        res = hill_fit(x, y)
        assert res["converged"]
        # Hill coefficient should be in the right ballpark
        assert 2.0 <= res["hill_n"] <= 10.0, res["hill_n"]

    def test_lower_bend_below_ec50_for_sharp_curve(self):
        """Sharp Hill curves (n large) have a lower bend close to but below EC50."""
        x, y = _hill_data(ec50=20.0, hill_n=8.0, noise_sd=0.1)
        res = hill_fit(x, y)
        assert res["converged"]
        if res["bend_plausible"]:
            assert res["lower_bend"] < res["ec50"]

    def test_too_few_observations_skipped(self):
        res = hill_fit(np.arange(20.0), np.arange(20.0))
        assert not res["converged"]
        assert np.isnan(res["ec50"])

    def test_r_squared_in_unit_interval(self):
        x, y = _hill_data(ec50=20.0, hill_n=4.0)
        res = hill_fit(x, y)
        if res["converged"]:
            assert 0.0 <= res["r_squared"] <= 1.0


# ─────────────────────────────────────────────────────────────────
#  per_subject_segmented
# ─────────────────────────────────────────────────────────────────


class TestPerSubjectSegmented:
    def _panel(self, n_subjects: int, n_per_subject: int = 100, seed: int = 0) -> pd.DataFrame:
        """Build a long-format panel of n_subjects each with a planted breakpoint."""
        rows = []
        for s in range(n_subjects):
            # Vary the planted breakpoint across subjects.
            bp = 20.0 + (s % 5) * 0.5
            x, y = _broken_stick_data(
                bp=bp, n=n_per_subject, noise_sd=0.2, seed=seed + s,
            )
            for xi, yi in zip(x, y):
                rows.append({"subject_id": f"subj_{s}", "x": xi, "y": yi})
        return pd.DataFrame(rows)

    def test_returns_one_row_per_subject(self):
        panel = self._panel(n_subjects=8)
        out = per_subject_segmented(panel, "subject_id", "x", "y")
        assert len(out) == 8
        assert set(out["subject_id"]) == {f"subj_{s}" for s in range(8)}

    def test_columns_include_model_result_keys(self):
        panel = self._panel(n_subjects=4)
        out = per_subject_segmented(panel, "subject_id", "x", "y")
        # broken_stick_fit returns breakpoint, slope_below, slope_above, etc.
        assert "breakpoint" in out.columns
        assert "slope_below" in out.columns
        assert "slope_above" in out.columns
        assert "converged" in out.columns
        assert "n" in out.columns

    def test_subjects_under_min_n_are_marked_unconverged(self):
        # Build a panel where 2 subjects have <50 observations.
        panel_big = self._panel(n_subjects=4, n_per_subject=100)
        panel_small = self._panel(n_subjects=2, n_per_subject=20, seed=99)
        panel_small["subject_id"] = panel_small["subject_id"].str.replace("subj", "small")
        panel = pd.concat([panel_big, panel_small], ignore_index=True)

        out = per_subject_segmented(panel, "subject_id", "x", "y", min_n=50)
        small_rows = out[out["subject_id"].str.startswith("small")]
        assert len(small_rows) == 2
        assert (small_rows["converged"] == False).all()  # noqa: E712

    def test_swappable_model(self):
        """Same iterator should work with hill_fit as the model."""
        # Build a panel from Hill curves for this test
        rows = []
        for s in range(4):
            x, y = _hill_data(ec50=20.0 + s, n=100, seed=s)
            for xi, yi in zip(x, y):
                rows.append({"subject_id": f"hill_{s}", "x": xi, "y": yi})
        panel = pd.DataFrame(rows)

        out = per_subject_segmented(
            panel, "subject_id", "x", "y", model=hill_fit,
        )
        assert "ec50" in out.columns
        assert "hill_n" in out.columns

    def test_composite_subject_key(self):
        """List of column names as subject_col should work for composite keys."""
        panel = self._panel(n_subjects=4)
        panel["year"] = [2024 + (i % 2) for i in range(len(panel))]

        out = per_subject_segmented(panel, ["subject_id", "year"], "x", "y")
        assert "subject_id" in out.columns
        assert "year" in out.columns
        # 4 subjects × 2 years = 8 groups
        assert len(out) == 8

    def test_model_kwargs_forwarded(self):
        panel = self._panel(n_subjects=4)
        out = per_subject_segmented(
            panel, "subject_id", "x", "y", model_kwargs={"n_grid": 50},
        )
        assert len(out) == 4
