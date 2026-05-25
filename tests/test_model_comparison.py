"""Unit tests for rerandomstats.model_comparison."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rerandomstats.model_comparison import (
    benjamini_hochberg,
    correct_pvalues,
    correct_pvalues_array,
    likelihood_ratio_test,
    wald_two_sample_beta,
)

# ─────────────────────────────────────────────────────────────────
#  wald_two_sample_beta
# ─────────────────────────────────────────────────────────────────


class TestWaldTwoSampleBeta:
    def test_identical_estimates_give_p_near_one(self):
        res = wald_two_sample_beta(0.10, 0.02, 0.10, 0.02)
        assert res["diff"] == pytest.approx(0.0)
        assert res["z_statistic"] == pytest.approx(0.0)
        assert res["pvalue"] == pytest.approx(1.0)

    def test_large_separation_gives_p_near_zero_two_sided(self):
        # Two betas 5 SE apart -> z ≈ 5/√2 ≈ 3.54 -> p ≈ 0.0004
        res = wald_two_sample_beta(0.50, 0.10, 0.00, 0.10)
        assert res["diff"] == pytest.approx(0.50)
        assert res["pvalue"] < 0.001

    def test_one_sided_greater(self):
        # β_a > β_b should give a small p for alternative='greater'
        res = wald_two_sample_beta(
            0.30, 0.05, 0.10, 0.05, alternative="greater"
        )
        assert res["pvalue"] < 0.01

    def test_one_sided_greater_with_reversed_signs(self):
        # β_a < β_b should give a LARGE p for alternative='greater'
        res = wald_two_sample_beta(
            0.10, 0.05, 0.30, 0.05, alternative="greater"
        )
        assert res["pvalue"] > 0.5

    def test_one_sided_less_mirrors_greater(self):
        res_g = wald_two_sample_beta(0.10, 0.05, 0.30, 0.05, alternative="greater")
        res_l = wald_two_sample_beta(0.10, 0.05, 0.30, 0.05, alternative="less")
        assert res_g["pvalue"] + res_l["pvalue"] == pytest.approx(1.0, abs=1e-6)

    def test_ci95_brackets_the_diff(self):
        res = wald_two_sample_beta(0.40, 0.10, 0.20, 0.10)
        # CI should be symmetric about the diff (0.20)
        assert res["ci95_low_diff"] < res["diff"] < res["ci95_high_diff"]
        midpoint = (res["ci95_low_diff"] + res["ci95_high_diff"]) / 2
        assert midpoint == pytest.approx(res["diff"])

    def test_zero_se_gives_nan_z(self):
        res = wald_two_sample_beta(0.10, 0.0, 0.20, 0.0)
        assert np.isnan(res["z_statistic"])
        assert np.isnan(res["pvalue"])

    def test_named_labels_propagate(self):
        res = wald_two_sample_beta(
            0.10, 0.02, 0.20, 0.02,
            name_a="hot_host", name_b="cool_host",
        )
        assert res["name_a"] == "hot_host"
        assert res["name_b"] == "cool_host"

    def test_invalid_alternative_raises(self):
        with pytest.raises(ValueError, match="alternative"):
            wald_two_sample_beta(0.1, 0.02, 0.2, 0.02, alternative="bogus")


# ─────────────────────────────────────────────────────────────────
#  likelihood_ratio_test
# ─────────────────────────────────────────────────────────────────


class TestLikelihoodRatioTest:
    """Uses SimpleNamespace stand-ins for statsmodels result objects.

    The real test surface is `.llf` and `.df_model`; SimpleNamespace
    objects with those attributes exercise the function exactly as
    statsmodels result objects would.
    """

    def test_strong_improvement_gives_small_p(self):
        # Δlog-likelihood = 50 over 2 extra params → LR = 100, df = 2 → p ≈ 0
        full = SimpleNamespace(llf=-100.0, df_model=5)
        reduced = SimpleNamespace(llf=-150.0, df_model=3)
        res = likelihood_ratio_test(full, reduced)
        assert res["lr_statistic"] == pytest.approx(100.0)
        assert res["df"] == 2
        assert res["pvalue"] < 0.001

    def test_no_improvement_gives_p_one(self):
        # Identical log-likelihoods → LR = 0 → p = 1
        full = SimpleNamespace(llf=-150.0, df_model=5)
        reduced = SimpleNamespace(llf=-150.0, df_model=3)
        res = likelihood_ratio_test(full, reduced)
        assert res["lr_statistic"] == pytest.approx(0.0)
        assert res["pvalue"] == pytest.approx(1.0)

    def test_modest_improvement_gives_intermediate_p(self):
        # ΔLL = 2 over 1 param → LR = 4, df = 1 → p ≈ 0.0455
        full = SimpleNamespace(llf=-100.0, df_model=4)
        reduced = SimpleNamespace(llf=-102.0, df_model=3)
        res = likelihood_ratio_test(full, reduced)
        assert res["lr_statistic"] == pytest.approx(4.0)
        assert 0.04 < res["pvalue"] < 0.05

    def test_log_likelihoods_carried_through(self):
        full = SimpleNamespace(llf=-100.0, df_model=4)
        reduced = SimpleNamespace(llf=-102.0, df_model=3)
        res = likelihood_ratio_test(full, reduced)
        assert res["log_likelihood_full"] == -100.0
        assert res["log_likelihood_reduced"] == -102.0

    def test_df_mismatch_raises(self):
        # Reduced has MORE parameters than full → invalid
        full = SimpleNamespace(llf=-100.0, df_model=3)
        reduced = SimpleNamespace(llf=-100.0, df_model=5)
        with pytest.raises(ValueError, match="more parameters"):
            likelihood_ratio_test(full, reduced)

    def test_df_equal_raises(self):
        full = SimpleNamespace(llf=-100.0, df_model=4)
        reduced = SimpleNamespace(llf=-100.0, df_model=4)
        with pytest.raises(ValueError, match="more parameters"):
            likelihood_ratio_test(full, reduced)


# ─────────────────────────────────────────────────────────────────
#  benjamini_hochberg
# ─────────────────────────────────────────────────────────────────


class TestBenjaminiHochberg:
    def test_known_battery(self):
        """Ported from the ThermoStrife test for byte-identical behaviour."""
        res = benjamini_hochberg({
            "A": 0.0015, "B": 0.0016, "C": 0.0052, "D": 0.037, "E": 0.070,
        }, alpha=0.05)
        assert res["family_size"] == 5
        assert res["bonferroni_threshold"] == pytest.approx(0.010)
        assert res["bh_cutoff_rank"] == 4  # BH rescues D
        assert res["n_bonferroni_rejected"] == 3
        assert res["results"]["D"]["bh_reject"] is True
        assert res["results"]["D"]["bonferroni_reject"] is False

    def test_empty_family(self):
        res = benjamini_hochberg({}, alpha=0.05)
        assert res["family_size"] == 0
        assert res["results"] == {}

    def test_all_pass(self):
        res = benjamini_hochberg({"x": 0.001, "y": 0.002, "z": 0.003}, alpha=0.05)
        assert res["n_bh_rejected"] == 3
        assert all(row["bh_reject"] for row in res["results"].values())

    def test_all_fail(self):
        res = benjamini_hochberg({"x": 0.5, "y": 0.6, "z": 0.7}, alpha=0.05)
        assert res["n_bh_rejected"] == 0
        assert not any(row["bh_reject"] for row in res["results"].values())

    def test_bh_q_values_are_monotone(self):
        """Adjusted q-values must be monotonically non-decreasing in raw-p rank."""
        res = benjamini_hochberg({
            "t1": 0.001, "t2": 0.01, "t3": 0.02, "t4": 0.04, "t5": 0.30,
        }, alpha=0.05)
        ranked = sorted(res["results"].items(), key=lambda kv: kv[1]["rank"])
        qs = [row["bh_adjusted_p"] for _, row in ranked]
        assert qs == sorted(qs), qs

    def test_alpha_passes_through(self):
        res = benjamini_hochberg({"x": 0.001}, alpha=0.10)
        assert res["alpha"] == 0.10
        assert res["bonferroni_threshold"] == pytest.approx(0.10)

    def test_input_dict_order_does_not_affect_output(self):
        """Names should map back to the same per-test results regardless of insertion order."""
        a = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.001})
        b = benjamini_hochberg({"c": 0.001, "b": 0.04, "a": 0.01})
        for k in ("a", "b", "c"):
            assert a["results"][k]["raw_p"] == b["results"][k]["raw_p"]
            assert a["results"][k]["bh_adjusted_p"] == b["results"][k]["bh_adjusted_p"]
            assert a["results"][k]["bh_reject"] == b["results"][k]["bh_reject"]


# ─────────────────────────────────────────────────────────────────
#  correct_pvalues — single-method correction, statsmodels-backed
# ─────────────────────────────────────────────────────────────────


class TestCorrectPvalues:
    def test_empty_input(self):
        res = correct_pvalues({})
        assert res["family_size"] == 0
        assert res["n_rejected"] == 0
        assert res["results"] == {}

    def test_default_method_is_bh(self):
        res = correct_pvalues({"a": 0.01, "b": 0.04})
        assert res["method"] == "fdr_bh"

    def test_bonferroni_method(self):
        res = correct_pvalues(
            {"a": 0.001, "b": 0.02, "c": 0.5}, method="bonferroni", alpha=0.05,
        )
        # Bonferroni adjusted: raw_p * k, clipped to 1.
        assert res["results"]["a"]["adjusted_p"] == pytest.approx(0.003)
        assert res["results"]["b"]["adjusted_p"] == pytest.approx(0.06)
        assert res["results"]["c"]["adjusted_p"] == pytest.approx(1.0)
        assert res["results"]["a"]["reject"] is True
        assert res["results"]["b"]["reject"] is False

    def test_holm_method(self):
        res = correct_pvalues(
            {"a": 0.001, "b": 0.01, "c": 0.04}, method="holm", alpha=0.05,
        )
        # Holm: step-down. Default in statsmodels works without error.
        assert res["method"] == "holm"
        assert res["n_rejected"] >= 1

    def test_rank_matches_raw_p_order(self):
        res = correct_pvalues({"c": 0.5, "a": 0.001, "b": 0.04})
        # rank 1 = smallest raw_p (a); rank 2 = b; rank 3 = c
        assert res["results"]["a"]["rank"] == 1
        assert res["results"]["b"]["rank"] == 2
        assert res["results"]["c"]["rank"] == 3

    def test_insertion_order_preserved_in_results(self):
        # results dict insertion order should match input insertion order
        res = correct_pvalues({"z": 0.5, "a": 0.001, "m": 0.04})
        assert list(res["results"].keys()) == ["z", "a", "m"]


# ─────────────────────────────────────────────────────────────────
#  Algorithmic-source consistency: benjamini_hochberg delegates
# ─────────────────────────────────────────────────────────────────


class TestSharedCoreConsistency:
    """Verify benjamini_hochberg's results match correct_pvalues directly.

    This is the guard rail that proves the "single algorithmic source"
    architecture: benjamini_hochberg is implemented on top of
    correct_pvalues, so its BH adjusted_p must equal correct_pvalues'
    BH adjusted_p, and likewise for Bonferroni.
    """

    @pytest.fixture
    def battery(self):
        return {"A": 0.0015, "B": 0.0016, "C": 0.0052, "D": 0.037, "E": 0.070}

    def test_bh_adjusted_p_matches(self, battery):
        bh_direct = correct_pvalues(battery, method="fdr_bh", alpha=0.05)
        dual = benjamini_hochberg(battery, alpha=0.05)
        for name in battery:
            assert (
                bh_direct["results"][name]["adjusted_p"]
                == dual["results"][name]["bh_adjusted_p"]
            ), name

    def test_bonferroni_adjusted_p_matches(self, battery):
        bf_direct = correct_pvalues(battery, method="bonferroni", alpha=0.05)
        dual = benjamini_hochberg(battery, alpha=0.05)
        for name in battery:
            assert (
                bf_direct["results"][name]["adjusted_p"]
                == dual["results"][name]["bonferroni_adjusted_p"]
            ), name

    def test_reject_flags_match(self, battery):
        bh_direct = correct_pvalues(battery, method="fdr_bh")
        bf_direct = correct_pvalues(battery, method="bonferroni")
        dual = benjamini_hochberg(battery)
        for name in battery:
            assert (
                dual["results"][name]["bh_reject"]
                == bh_direct["results"][name]["reject"]
            )
            assert (
                dual["results"][name]["bonferroni_reject"]
                == bf_direct["results"][name]["reject"]
            )

    def test_counts_match(self, battery):
        bh_direct = correct_pvalues(battery, method="fdr_bh")
        bf_direct = correct_pvalues(battery, method="bonferroni")
        dual = benjamini_hochberg(battery)
        assert dual["n_bh_rejected"] == bh_direct["n_rejected"]
        assert dual["n_bonferroni_rejected"] == bf_direct["n_rejected"]


# ─────────────────────────────────────────────────────────────────
#  correct_pvalues_array — array-in / array-out helper
# ─────────────────────────────────────────────────────────────────


class TestCorrectPvaluesArray:
    def test_empty_input(self):
        out = correct_pvalues_array(np.array([]))
        assert out.size == 0

    def test_known_bh_battery_matches_dict_form(self):
        """Array helper must produce same adjusted p-values as the dict form."""
        ps = np.array([0.0015, 0.0016, 0.0052, 0.037, 0.070])
        arr = correct_pvalues_array(ps, method="fdr_bh")

        # Compare against the dict form on the same data.
        dict_result = correct_pvalues(
            {f"t{i}": p for i, p in enumerate(ps)}, method="fdr_bh",
        )
        # Recover the dict-form adjusted_p in the original order.
        dict_adj = np.array([dict_result["results"][f"t{i}"]["adjusted_p"] for i in range(5)])
        np.testing.assert_allclose(arr, dict_adj)

    def test_nan_entries_preserved(self):
        ps = np.array([0.001, np.nan, 0.02, np.nan, 0.5])
        out = correct_pvalues_array(ps)
        assert np.isnan(out[1])
        assert np.isnan(out[3])
        # Finite entries get adjusted
        assert np.isfinite(out[0])
        assert np.isfinite(out[2])
        assert np.isfinite(out[4])

    def test_all_nan_input(self):
        ps = np.array([np.nan, np.nan, np.nan])
        out = correct_pvalues_array(ps)
        assert np.all(np.isnan(out))

    def test_bonferroni_method(self):
        ps = np.array([0.001, 0.02, 0.5])
        out = correct_pvalues_array(ps, method="bonferroni")
        np.testing.assert_allclose(out, [0.003, 0.06, 1.0])

    def test_output_shape_matches_input(self):
        ps = np.random.default_rng(0).uniform(0, 1, 17)
        out = correct_pvalues_array(ps)
        assert out.shape == ps.shape

    def test_routes_through_shared_core_same_as_dict(self):
        """The array helper and the dict helper must agree element-wise.

        This is the guard rail proving the shared-algorithmic-source
        invariant: correct_pvalues_array and correct_pvalues both
        delegate to statsmodels.stats.multitest.multipletests.
        """
        rng = np.random.default_rng(7)
        ps = rng.uniform(0, 1, 25)
        # Array form
        arr_out = correct_pvalues_array(ps, method="fdr_bh")
        # Dict form
        dict_out = correct_pvalues(
            {f"t{i}": p for i, p in enumerate(ps)}, method="fdr_bh",
        )
        dict_arr = np.array([dict_out["results"][f"t{i}"]["adjusted_p"] for i in range(25)])
        np.testing.assert_allclose(arr_out, dict_arr)
