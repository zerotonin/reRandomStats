"""
┌──────────────────────────────────────────────────────────────────────┐
│               tests/test_paired_resampling.py                        │
│                                                                      │
│  pytest suite for the partially paired resampling test.  Synthetic   │
│  data only.                                                          │
│                                                                      │
│  Run:  pytest -v                                                     │
└──────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import pytest

from rerandomstats import (
    FisherResamplingTest,
    MultiGroupPairedTest,
    PartiallyPairedResamplingTest,
)


def _herd(n_paired=40, n_only=15, shift=0.0, subject_sd=3.0, noise_sd=0.5,
          seed=0):
    """Two 'summers' of a herd: some cows in both, some in one only."""
    rng = np.random.default_rng(seed)
    base = rng.normal(75.0, subject_sd, n_paired)
    a = np.r_[base + rng.normal(0, noise_sd, n_paired),
              rng.normal(75.0, subject_sd, n_only)]
    b = np.r_[base + shift + rng.normal(0, noise_sd, n_paired),
              rng.normal(75.0 + shift, subject_sd, n_only)]
    ids_a = [f"p{i}" for i in range(n_paired)] + [f"a{i}" for i in range(n_only)]
    ids_b = [f"p{i}" for i in range(n_paired)] + [f"b{i}" for i in range(n_only)]
    return a, b, ids_a, ids_b


# ┌──────────────────────────────────────────────────────────────────┐
# │                PartiallyPairedResamplingTest                      │
# └──────────────────────────────────────────────────────────────────┘


class TestPartiallyPairedResamplingTest:
    """Tests for :class:`PartiallyPairedResamplingTest`."""

    def test_counts_paired_and_unpaired_subjects(self):
        a, b, ids_a, ids_b = _herd(n_paired=12, n_only=5)
        test = PartiallyPairedResamplingTest(a, b, ids_a, ids_b, "medianDiff",
                                             500, seed=1)
        test.main()
        assert (test.n_paired, test.n_only_a, test.n_only_b) == (12, 5, 5)

    def test_pairing_recovers_a_shift_the_unpaired_test_misses(self):
        """Large spread between subjects, small consistent shift within."""
        a, b, ids_a, ids_b = _herd(n_paired=40, n_only=0, shift=0.8,
                                   subject_sd=4.0, noise_sd=0.3, seed=2)
        paired = PartiallyPairedResamplingTest(a, b, ids_a, ids_b, "meanDiff",
                                               5_000, seed=1).main()
        unpaired = FisherResamplingTest(list(a), list(b), "meanDiff",
                                        5_000, seed=1).main()
        assert paired < 0.01
        assert unpaired > 0.05

    def test_no_shared_subjects_matches_the_ordinary_resampling_test(self):
        rng = np.random.default_rng(3)
        a, b = rng.normal(0.0, 1.0, 40), rng.normal(0.7, 1.0, 40)
        ids_a = [f"a{i}" for i in range(40)]
        ids_b = [f"b{i}" for i in range(40)]
        test = PartiallyPairedResamplingTest(a, b, ids_a, ids_b, "meanDiff",
                                             10_000, seed=1)
        p_new = test.main()
        p_old = FisherResamplingTest(list(a), list(b), "meanDiff",
                                     10_000, seed=1).main()
        assert test.n_paired == 0
        assert p_new == pytest.approx(p_old, abs=0.01)

    def test_identical_groups_give_p_one(self):
        a, _, ids_a, _ = _herd(n_paired=20, n_only=0)
        p = PartiallyPairedResamplingTest(a, a, ids_a, ids_a, "medianDiff",
                                          500, seed=1).main()
        assert p == 1.0

    def test_false_positive_rate_is_controlled_with_correlated_subjects(self):
        """Under the null, with strong subject effects, about 5 % reject."""
        rejections = 0
        n_sim = 200
        for sim in range(n_sim):
            a, b, ids_a, ids_b = _herd(n_paired=30, n_only=10, shift=0.0,
                                       subject_sd=4.0, noise_sd=1.0,
                                       seed=1_000 + sim)
            p = PartiallyPairedResamplingTest(a, b, ids_a, ids_b, "medianDiff",
                                              400, seed=sim).main()
            rejections += p < 0.05
        assert 0.01 <= rejections / n_sim <= 0.10

    def test_seed_makes_the_p_value_reproducible(self):
        a, b, ids_a, ids_b = _herd(shift=0.4, seed=4)
        args = (a, b, ids_a, ids_b, "medianDiff", 1_000)
        assert (PartiallyPairedResamplingTest(*args, seed=7).main()
                == PartiallyPairedResamplingTest(*args, seed=7).main())

    def test_p_value_bounds_and_null_distribution(self):
        a, b, ids_a, ids_b = _herd(shift=5.0, seed=5)
        test = PartiallyPairedResamplingTest(a, b, ids_a, ids_b, "meanDiff",
                                             999, seed=1)
        p = test.main()
        assert p == pytest.approx(1 / 1_000)          # nothing as extreme
        assert test.shuffled_results.size == 999
        assert np.all(np.diff(test.shuffled_results) >= 0)
        assert test.original_test_result == pytest.approx(a.mean() - b.mean())

    def test_missing_value_turns_a_pair_into_an_unpaired_subject(self):
        a, b, ids_a, ids_b = _herd(n_paired=10, n_only=3)
        a = a.copy()
        a[0] = np.nan
        test = PartiallyPairedResamplingTest(a, b, ids_a, ids_b, "medianDiff",
                                             300, seed=1)
        test.main()
        assert (test.n_paired, test.n_only_a, test.n_only_b) == (9, 3, 4)

    def test_order_of_subjects_does_not_matter(self):
        a, b, ids_a, ids_b = _herd(n_paired=15, n_only=5, shift=0.5, seed=6)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(b))
        p1 = PartiallyPairedResamplingTest(a, b, ids_a, ids_b, "meanDiff",
                                           300, seed=1)
        p2 = PartiallyPairedResamplingTest(a, b[perm], ids_a,
                                           [ids_b[i] for i in perm],
                                           "meanDiff", 300, seed=1)
        p1.main()
        p2.main()
        assert p1.n_paired == p2.n_paired
        assert p1.original_test_result == pytest.approx(p2.original_test_result)

    def test_sum_diff_is_supported(self):
        a, b, ids_a, ids_b = _herd(shift=3.0, seed=8)
        p = PartiallyPairedResamplingTest(a, b, ids_a, ids_b, "sumDiff",
                                          1_000, seed=1).main()
        assert p < 0.05

    def test_repeated_subject_within_a_group_raises(self):
        with pytest.raises(ValueError, match="more than once"):
            PartiallyPairedResamplingTest([1.0, 2.0], [3.0, 4.0],
                                          ["x", "x"], ["y", "z"],
                                          "meanDiff", 100).main()

    def test_unknown_statistic_raises(self):
        with pytest.raises(ValueError, match="unknown statistic"):
            PartiallyPairedResamplingTest([1.0], [2.0], ["x"], ["y"],
                                          "maxDiff", 100).main()

    def test_exhaustive_mode_is_refused(self):
        with pytest.raises(ValueError, match="positive integer"):
            PartiallyPairedResamplingTest([1.0], [2.0], ["x"], ["y"],
                                          "meanDiff", "all").main()

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="subjects"):
            PartiallyPairedResamplingTest([1.0, 2.0], [3.0], ["x"], ["y"],
                                          "meanDiff", 100).main()


# ┌──────────────────────────────────────────────────────────────────┐
# │                     MultiGroupPairedTest                          │
# └──────────────────────────────────────────────────────────────────┘


def _three_years(seed=0):
    """Thirty cows in all three years; the third year sits higher."""
    rng = np.random.default_rng(seed)
    base = rng.normal(75.0, 3.0, 30)
    data, group, subject = [], [], []
    for year, shift in (("2021", 0.0), ("2022", 0.0), ("2023", 2.0)):
        data += list(base + shift + rng.normal(0, 0.5, 30))
        group += [year] * 30
        subject += [f"cow{i}" for i in range(30)]
    return data, group, subject


class TestMultiGroupPairedTest:
    """Tests for :class:`MultiGroupPairedTest`."""

    def test_table_layout(self):
        data, group, subject = _three_years()
        table = MultiGroupPairedTest(data, group, subject, "medianDiff",
                                     500, seed=1).main()
        assert list(table.columns) == [
            "groupA", "groupA_n", "groupB", "groupB_n", "n_paired",
            "n_only_A", "n_only_B", "statistic", "p value",
            "p value corrected", "h", "sig. level"]
        assert len(table) == 3
        assert (table["n_paired"] == 30).all()
        assert (table["p value corrected"] >= table["p value"] - 1e-12).all()

    def test_finds_the_shifted_year(self):
        data, group, subject = _three_years()
        table = MultiGroupPairedTest(data, group, subject, "medianDiff",
                                     2_000, seed=1).main()
        by_pair = table.set_index(["groupA", "groupB"])
        assert not by_pair.loc[("2021", "2022"), "h"]
        assert by_pair.loc[("2021", "2023"), "h"]
        assert by_pair.loc[("2022", "2023"), "h"]
        assert by_pair.loc[("2021", "2023"), "statistic"] < 0

    def test_combination_set_restricts_the_pairs(self):
        data, group, subject = _three_years()
        table = MultiGroupPairedTest(
            data, group, subject, "meanDiff", 300,
            combination_set=[("2021", "2023")], seed=1).main()
        assert len(table) == 1
        assert table.loc[0, "groupB"] == "2023"
