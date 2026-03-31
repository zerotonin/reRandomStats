"""
┌──────────────────────────────────────────────────────────────────────┐
│               tests/test_rerandomstats.py                            │
│                                                                      │
│  pytest suite for the reRandomStats package.  Tests cover all        │
│  public classes and key edge cases.                                  │
│                                                                      │
│  Run:  pytest -v                                                     │
└──────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import pandas as pd
import pytest

from rerandomstats import (
    BinomialStats,
    DataIO,
    FisherExactTest,
    FisherResamplingTest,
    GetNofK,
    HypothesisTests,
    MultipleBinomialTests,
    MultiGroupTest,
    write_pretty_table,
)


# ┌──────────────────────────────────────────────────────────────────┐
# │                  FisherExactTest                                 │
# └──────────────────────────────────────────────────────────────────┘


class TestFisherExactTest:
    """Tests for :class:`FisherExactTest`."""

    def test_significant_result(self):
        """A clearly skewed 2×2 table should yield p < 0.05."""
        test = FisherExactTest((8, 2), (1, 5))
        p = test.main()
        assert p < 0.05
        assert test.p_value == p

    def test_non_significant_result(self):
        """Balanced counts should not be significant."""
        test = FisherExactTest((5, 5), (5, 5))
        p = test.main()
        assert p > 0.05

    def test_alternative_less(self):
        """One-sided 'less' should produce a valid p-value."""
        test = FisherExactTest((8, 2), (1, 5), alternative="less")
        p = test.main()
        assert 0 <= p <= 1

    def test_alternative_greater(self):
        """One-sided 'greater' should produce a valid p-value."""
        test = FisherExactTest((8, 2), (1, 5), alternative="greater")
        p = test.main()
        assert 0 <= p <= 1

    def test_p_value_stored(self):
        """p_value attribute is None before and float after main()."""
        test = FisherExactTest((3, 7), (7, 3))
        assert test.p_value is None
        test.main()
        assert isinstance(test.p_value, float)


# ┌──────────────────────────────────────────────────────────────────┐
# │                  FisherResamplingTest                            │
# └──────────────────────────────────────────────────────────────────┘


class TestFisherResamplingTest:
    """Tests for :class:`FisherResamplingTest`."""

    def test_significant_separation(self):
        """Well-separated groups should give p < 0.05."""
        np.random.seed(42)
        a = [1, 2, 3, 4, 5]
        b = [10, 11, 12, 13, 14]
        test = FisherResamplingTest(a, b, "meanDiff", 5000)
        p = test.main()
        assert p < 0.05

    def test_identical_groups(self):
        """Identical data should give p close to 1."""
        a = [5, 5, 5, 5]
        b = [5, 5, 5, 5]
        test = FisherResamplingTest(a, b, "medianDiff", 1000)
        p = test.main()
        assert p > 0.5

    def test_median_diff(self):
        """medianDiff statistic should work."""
        test = FisherResamplingTest([1, 2, 3], [7, 8, 9], "medianDiff", 5000)
        p = test.main()
        assert 0 < p <= 1

    def test_sum_diff(self):
        """sumDiff statistic should work."""
        test = FisherResamplingTest([1, 2, 3], [7, 8, 9], "sumDiff", 5000)
        p = test.main()
        assert 0 < p <= 1

    def test_invalid_func_raises(self):
        """Unknown statistic name should raise ValueError."""
        test = FisherResamplingTest([1], [2], "bogus", 100)
        test.get_shuffled_indices()
        with pytest.raises(ValueError, match="bogus"):
            test.calculate_test([1], [2])

    def test_exhaustive_mode(self):
        """combination_n='all' should use exhaustive enumeration for small n."""
        test = FisherResamplingTest([1, 2], [3, 4], "meanDiff", "all")
        p = test.main()
        assert 0 < p <= 1

    def test_attributes_populated(self):
        """After main(), key attributes should be populated."""
        test = FisherResamplingTest([1, 2, 3], [4, 5, 6], "meanDiff", 1000)
        test.main()
        assert test.original_test_result is not None
        assert len(test.shuffled_results) > 0
        assert test.p_value is not None


# ┌──────────────────────────────────────────────────────────────────┐
# │                  GetNofK                                         │
# └──────────────────────────────────────────────────────────────────┘


class TestGetNofK:
    """Tests for :class:`GetNofK`."""

    def test_all_combinations(self):
        """Exhaustive mode should produce C(4,2) = 6 combos for [a,b]+[c,d]."""
        nk = GetNofK([1, 2], [3, 4], "all")
        nk.main()
        assert nk.combination_n == 6
        assert len(nk.data_indices) == 6

    def test_resampling_count(self):
        """Resampling mode should produce the requested number of draws."""
        nk = GetNofK([1, 2, 3], [4, 5, 6], 500)
        nk.main()
        assert nk.combination_n == 500

    def test_shuffled_set_lengths(self):
        """Shuffled sets should preserve original group sizes."""
        nk = GetNofK([1, 2, 3], [4, 5, 6, 7], 100)
        nk.main()
        a, b = nk.get_shuffled_set(0)
        assert len(a) == 3  # shorter set
        assert len(b) == 4

    def test_complement_covers_all(self):
        """Each (A, B) pair should cover all indices exactly once."""
        nk = GetNofK([10, 20], [30, 40, 50], "all")
        nk.main()
        for idx_a, idx_b in nk.data_indices:
            assert set(idx_a) | set(idx_b) == set(range(5))
            assert set(idx_a) & set(idx_b) == set()

    def test_invalid_mode_raises(self):
        """An unknown mode string should raise ValueError."""
        nk = GetNofK([1], [2], 10)
        nk.mode = "invalid_mode"
        with pytest.raises(ValueError, match="invalid_mode"):
            nk.main()


# ┌──────────────────────────────────────────────────────────────────┐
# │                  HypothesisTests                                 │
# └──────────────────────────────────────────────────────────────────┘


class TestHypothesisTests:
    """Tests for :class:`HypothesisTests`."""

    @pytest.fixture
    def separated_data(self):
        return [1, 2, 3, 4, 5], [10, 11, 12, 13, 14]

    @pytest.fixture
    def similar_data(self):
        return [1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5]

    def test_mann_whitney_u(self, separated_data):
        a, b = separated_data
        ht = HypothesisTests(a, b, "MannWhitneyU")
        assert ht.main() < 0.05

    def test_kruskal_wallis(self, separated_data):
        a, b = separated_data
        ht = HypothesisTests(a, b, "KruskalWallis")
        assert ht.main() < 0.05

    def test_wilcoxon_rank_sum(self, separated_data):
        a, b = separated_data
        ht = HypothesisTests(a, b, "WilcoxonRankSum")
        assert ht.main() < 0.05

    def test_independent_t(self, separated_data):
        a, b = separated_data
        ht = HypothesisTests(a, b, "IndependentT")
        assert ht.main() < 0.05

    def test_mood_median(self, separated_data):
        a, b = separated_data
        ht = HypothesisTests(a, b, "MoodMedian")
        p = ht.main()
        assert 0 <= p <= 1

    def test_non_significant(self, similar_data):
        a, b = similar_data
        ht = HypothesisTests(a, b, "MannWhitneyU")
        assert ht.main() > 0.05

    def test_invalid_test_raises(self):
        ht = HypothesisTests([1], [2], "NonExistentTest")
        with pytest.raises(ValueError, match="NonExistentTest"):
            ht.main()


# ┌──────────────────────────────────────────────────────────────────┐
# │                  BinomialStats                                   │
# └──────────────────────────────────────────────────────────────────┘


class TestBinomialStats:
    """Tests for :class:`BinomialStats`."""

    def test_fair_coin(self):
        """50/100 heads should not reject H₀ of p=0.5."""
        bs = BinomialStats(50, 100)
        result = bs.binomial_test(base_rate=0.5)
        assert result.pvalue > 0.05

    def test_biased_coin(self):
        """90/100 heads should reject H₀ of p=0.5."""
        bs = BinomialStats(90, 100)
        result = bs.binomial_test(base_rate=0.5)
        assert result.pvalue < 0.05

    def test_exact_ci_keys(self):
        """CI dict should have the right keys."""
        bs = BinomialStats(60, 100)
        ci = bs.exact_ci()
        assert set(ci.keys()) == {"Proportion", "Lower CI", "Upper CI"}

    def test_exact_ci_ordering(self):
        """Lower CI < Proportion < Upper CI."""
        bs = BinomialStats(60, 100)
        ci = bs.exact_ci()
        assert ci["Lower CI"] < ci["Proportion"] < ci["Upper CI"]

    def test_exact_ci_bounds(self):
        """CI should be within [0, 100]."""
        bs = BinomialStats(1, 1000)
        ci = bs.exact_ci()
        assert ci["Lower CI"] >= 0
        assert ci["Upper CI"] <= 100


# ┌──────────────────────────────────────────────────────────────────┐
# │                  MultipleBinomialTests                           │
# └──────────────────────────────────────────────────────────────────┘


class TestMultipleBinomialTests:
    """Tests for :class:`MultipleBinomialTests`."""

    def test_ztest_different_proportions(self):
        """Very different proportions should be significant."""
        mbt = MultipleBinomialTests((90, 10), (10, 90), "ztest")
        p = mbt.main()
        assert p < 0.05

    def test_chi2_similar_proportions(self):
        """Similar proportions should not be significant."""
        mbt = MultipleBinomialTests((50, 50), (48, 52), "chi2")
        p = mbt.main()
        assert p > 0.05

    def test_invalid_func_raises(self):
        mbt = MultipleBinomialTests((10, 10), (10, 10), "bogus")
        with pytest.raises(ValueError, match="bogus"):
            mbt.main()

    def test_identical_returns_one(self):
        """Identical groups may produce NaN → should return 1.0."""
        mbt = MultipleBinomialTests((0, 0), (0, 0), "ztest")
        p = mbt.main()
        assert p == 1.0


# ┌──────────────────────────────────────────────────────────────────┐
# │                  MultiGroupTest                                  │
# └──────────────────────────────────────────────────────────────────┘


class TestMultiGroupTest:
    """Tests for :class:`MultiGroupTest`."""

    @pytest.fixture
    def three_groups(self):
        np.random.seed(0)
        data = list(np.concatenate([
            np.random.normal(0, 1, 10),
            np.random.normal(5, 1, 10),
            np.random.normal(10, 1, 10),
        ]))
        groups = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
        return data, groups

    def test_fisher_resampling(self, three_groups):
        data, groups = three_groups
        mgt = MultiGroupTest(data, groups, "Fisher:meanDiff", 2000)
        df = mgt.main()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # C(3,2) = 3 pairs

    def test_hypo_mann_whitney(self, three_groups):
        data, groups = three_groups
        mgt = MultiGroupTest(data, groups, "hypo:MannWhitneyU", 1000)
        df = mgt.main()
        assert "p value corrected" in df.columns
        assert all(df["h"])  # all pairs should be significant

    def test_custom_combination_set(self, three_groups):
        data, groups = three_groups
        mgt = MultiGroupTest(
            data, groups, "hypo:IndependentT", 1000,
            combination_set=[("A", "C")],
        )
        df = mgt.main()
        assert len(df) == 1
        assert df.iloc[0]["groupA"] == "A"
        assert df.iloc[0]["groupB"] == "C"

    def test_output_columns(self, three_groups):
        data, groups = three_groups
        mgt = MultiGroupTest(data, groups, "hypo:IndependentT", 1000)
        df = mgt.main()
        expected = {
            "groupA", "groupA_n", "groupB", "groupB_n",
            "p value", "p value corrected", "h", "sig. level",
        }
        assert set(df.columns) == expected

    def test_significance_levels(self):
        assert MultiGroupTest.get_significance_level(0.1) == "n.s."
        assert MultiGroupTest.get_significance_level(0.04) == "*"
        assert MultiGroupTest.get_significance_level(0.005) == "**"
        assert MultiGroupTest.get_significance_level(0.0001) == "***"

    def test_invalid_family_raises(self):
        mgt = MultiGroupTest([1, 2], ["A", "B"], "Bogus:test", 100)
        with pytest.raises(ValueError, match="Bogus"):
            mgt.main()


# ┌──────────────────────────────────────────────────────────────────┐
# │                  DataIO                                          │
# └──────────────────────────────────────────────────────────────────┘


class TestDataIO:
    """Tests for :class:`DataIO`."""

    def test_wide_table_csv(self, tmp_path):
        """Round-trip: write a CSV, read it, convert to long format."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("A,B,C\n1,2,3\n4,,6\n")
        dio = DataIO()
        ids, vals = dio.wide_table_csv_to_long_table(str(csv_file))
        assert len(ids) == 5  # one NaN dropped
        assert len(vals) == 5

    def test_subset(self):
        ids = ["a", "b", "c", "d"]
        vals = [1.0, 2.0, 3.0, 4.0]
        sub_ids, sub_vals = DataIO.get_subset_of_data(ids, vals, ["a", "d"])
        assert sub_ids == ["a", "d"]
        assert sub_vals == [1.0, 4.0]

    def test_make_square_np_matrix(self):
        dio = DataIO()
        rows = [["1", "2", "3"], ["4", "", "6"]]
        mat = dio.make_square_np_matrix(rows)
        assert mat.shape == (2, 3)
        assert np.isnan(mat[1, 1])
        assert mat[0, 0] == 1.0

    def test_split_csv_headers(self, tmp_path):
        csv_file = tmp_path / "h.csv"
        csv_file.write_text("ColA,ColB\n1,2\n")
        dio = DataIO()
        dio.read_csv(str(csv_file))
        headers = dio.split_csv_headers()
        assert headers == ["ColA", "ColB"]


# ┌──────────────────────────────────────────────────────────────────┐
# │                  write_pretty_table                              │
# └──────────────────────────────────────────────────────────────────┘


class TestWritePrettyTable:
    """Tests for :func:`write_pretty_table`."""

    def test_returns_table(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        tbl = write_pretty_table(df, show=False)
        assert "x" in tbl.get_string()

    def test_file_write(self, tmp_path):
        df = pd.DataFrame({"a": [10]})
        out = tmp_path / "out.txt"
        write_pretty_table(df, str(out), write=True, show=False)
        content = out.read_text()
        assert "10" in content


# ┌──────────────────────────────────────────────────────────────────┐
# │                  Package-level smoke test                        │
# └──────────────────────────────────────────────────────────────────┘


class TestPackageImport:
    """Verify the package imports and version string exist."""

    def test_version_string(self):
        import rerandomstats
        assert isinstance(rerandomstats.__version__, str)

    def test_all_exports(self):
        import rerandomstats
        for name in rerandomstats.__all__:
            assert hasattr(rerandomstats, name)
