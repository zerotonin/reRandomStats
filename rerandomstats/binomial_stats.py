"""
┌──────────────────────────────────────────────────────────────────────┐
│        binomial_stats.py « Binomial Proportion Tests »               │
│                                                                      │
│  Single-sample binomial test with Wilson confidence intervals,      │
│  and a two-sample proportions test (z-test or chi-square) for       │
│  comparing success rates across groups.                             │
│                                                                      │
│  Author : Bart R.H. Geurten                                         │
│  Licence: MIT                                                        │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Dict, Literal, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint, proportions_chisquare, proportions_ztest


class BinomialStats:
    """Single-sample binomial test with confidence intervals.

    Wraps :func:`scipy.stats.binomtest` and
    :func:`statsmodels.stats.proportion.proportion_confint` (Wilson
    method) for quick proportion analysis.

    Args:
        heads: Number of successes observed.
        total_flips: Total number of trials.
        alpha: Significance level (used for the CI).
        alternative: ``'two-sided'``, ``'greater'``, or ``'less'``.

    Example:
        >>> bs = BinomialStats(heads=80, total_flips=100)
        >>> result = bs.binomial_test(base_rate=0.5)
        >>> result.pvalue < 0.05
        True
    """

    def __init__(
        self,
        heads: int,
        total_flips: int,
        alpha: float = 0.05,
        alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    ) -> None:
        self.heads = heads
        self.total_flips = total_flips
        self.alpha = alpha
        self.alternative = alternative

    # ── binomial test ────────────────────────────────────────────────

    def binomial_test(self, base_rate: float = 0.5):
        """Perform a binomial test against *base_rate*.

        Args:
            base_rate: Expected success probability under H₀.

        Returns:
            :class:`scipy.stats.BinomTestResult` object (access
            ``.pvalue`` for the p-value).
        """
        return binomtest(
            self.heads,
            self.total_flips,
            p=base_rate,
            alternative=self.alternative,
        )

    # ── Wilson confidence interval ───────────────────────────────────

    def exact_ci(self) -> Dict[str, float]:
        """Compute a Wilson confidence interval for the proportion.

        Returns:
            Dictionary with keys ``'Proportion'``, ``'Lower CI'``,
            and ``'Upper CI'`` (all as percentages).
        """
        x = float(self.heads)
        n = float(self.total_flips)
        proportion = round((x / n) * 100, 2)

        lower, upper = proportion_confint(
            count=x, nobs=n, alpha=self.alpha, method="wilson"
        )
        return {
            "Proportion": proportion,
            "Lower CI": max(0.0, round(lower * 100, 4)),
            "Upper CI": min(100.0, round(upper * 100, 4)),
        }


class MultipleBinomialTests:
    """Two-sample proportions test (z-test or chi-square).

    Compares the success proportions in two groups.  Each group is
    specified as a tuple ``(successes, failures)``.

    Args:
        data_a: ``(successes, failures)`` for group A.
        data_b: ``(successes, failures)`` for group B.
        func: ``'ztest'`` or ``'chi2'``.
        alternative: ``'two-sided'``, ``'smaller'``, or ``'larger'``.

    Example:
        >>> mbt = MultipleBinomialTests((30, 70), (50, 50), 'ztest')
        >>> p = mbt.main()
    """

    def __init__(
        self,
        data_a: Tuple[int, int],
        data_b: Tuple[int, int],
        func: Literal["ztest", "chi2"],
        alternative: Literal["two-sided", "smaller", "larger"] = "two-sided",
    ) -> None:
        self.data_a = data_a
        self.data_b = data_b
        self.func = func
        self.alternative = alternative

    # ── main entry point ─────────────────────────────────────────────

    def main(self) -> float:
        """Run the proportions comparison and return the p-value.

        Returns:
            The p-value.  Returns ``1.0`` if the result is *NaN*
            (identical samples or missing data).

        Raises:
            ValueError: If :attr:`func` is unrecognised.
        """
        counts = np.array((self.data_a[0], self.data_b[0]))
        observations = np.array((np.sum(self.data_a), np.sum(self.data_b)))

        if self.func == "ztest":
            p_value = proportions_ztest(
                count=counts, nobs=observations, alternative=self.alternative
            )[1]
        elif self.func == "chi2":
            p_value = proportions_chisquare(count=counts, nobs=observations)[1]
        else:
            raise ValueError(
                f"MultipleBinomialTests: unknown test function '{self.func}'"
            )

        if np.isnan(p_value):
            p_value = 1.0
            print(
                "MultipleBinomialTests: p-value is NaN (identical samples "
                "or NaN in data) — set to 1.0"
            )
        return float(p_value)
