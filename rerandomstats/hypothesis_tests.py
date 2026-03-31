"""
┌──────────────────────────────────────────────────────────────────────┐
│     hypothesis_tests.py « Classical Hypothesis Test Wrapper »        │
│                                                                      │
│  Thin dispatcher around scipy.stats for two-sample hypothesis       │
│  tests.  Supported tests: Mann-Whitney U, Kruskal-Wallis,          │
│  Chi-square, Kolmogorov-Smirnov, Mood's Median, Wilcoxon           │
│  Rank-Sum, and the independent t-test.                              │
│                                                                      │
│  Author : Bart R.H. Geurten                                         │
│  Licence: MIT                                                        │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Literal, Sequence

from scipy.stats import (
    chisquare,
    kruskal,
    kstest,
    mannwhitneyu,
    mood,
    ranksums,
    ttest_ind,
)


# Accepted test names (for documentation / validation)
_VALID_TESTS = {
    "MannWhitneyU",
    "KruskalWallis",
    "ChiSquare",
    "Kolmogorov",
    "MoodMedian",
    "WilcoxonRankSum",
    "IndependentT",
}


class HypothesisTests:
    """Unified interface for two-sample hypothesis tests.

    This class wraps several :mod:`scipy.stats` tests behind a
    single ``func`` string selector.  It is consumed primarily by
    :class:`~rerandomstats.multi_group_test.MultiGroupTest` when the
    test family is ``'hypo'``.

    Supported *func* values:

    ============== ======================================================
    Name           Description
    ============== ======================================================
    MannWhitneyU   Non-parametric comparison of two independent groups.
    KruskalWallis  Non-parametric comparison (>2 groups supported).
    ChiSquare      Test for independence on frequency data.
    Kolmogorov     Two-sample Kolmogorov-Smirnov distribution test.
    MoodMedian     Non-parametric median comparison.
    WilcoxonRankSum Wilcoxon rank-sum (equivalent to Mann-Whitney).
    IndependentT   Parametric t-test for independent samples.
    ============== ======================================================

    Args:
        data_a: First sample.
        data_b: Second sample.
        func: Name of the test to perform.
        alternative: ``'two-sided'``, ``'less'``, or ``'greater'``.

    Example:
        >>> ht = HypothesisTests([1, 2, 3], [5, 6, 7], 'MannWhitneyU')
        >>> p = ht.main()
    """

    def __init__(
        self,
        data_a: Sequence[float],
        data_b: Sequence[float],
        func: str,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> None:
        self.data_a = data_a
        self.data_b = data_b
        self.func = func
        self.alternative = alternative

    # ── main entry point ─────────────────────────────────────────────

    def main(self) -> float:
        """Run the selected hypothesis test and return its p-value.

        Returns:
            The p-value produced by the chosen test.

        Raises:
            ValueError: If :attr:`func` is not a recognised test name.
        """
        if self.func == "MannWhitneyU":
            p_value = mannwhitneyu(
                self.data_a, self.data_b, alternative=self.alternative
            )[1]

        elif self.func == "KruskalWallis":
            p_value = kruskal(self.data_a, self.data_b)[1]

        elif self.func == "ChiSquare":
            _, p_value, _, _ = chisquare(self.data_a, self.data_b)

        elif self.func == "WilcoxonRankSum":
            _, p_value = ranksums(
                self.data_a, self.data_b, alternative=self.alternative
            )

        elif self.func == "Kolmogorov":
            _, p_value = kstest(self.data_a, self.data_b)

        elif self.func == "MoodMedian":
            p_value = mood(
                self.data_a, self.data_b, alternative=self.alternative
            )[1]

        elif self.func == "IndependentT":
            _, p_value = ttest_ind(self.data_a, self.data_b)

        else:
            raise ValueError(
                f"HypothesisTests: unknown test function '{self.func}'. "
                f"Choose from: {', '.join(sorted(_VALID_TESTS))}"
            )

        return float(p_value)
