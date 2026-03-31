"""
┌──────────────────────────────────────────────────────────────────────┐
│      fisher_resampling.py « Fisher's Resampling Test »               │
│                                                                      │
│  Implements Sir Ronald Fisher's re-randomisation test for            │
│  comparing two independent samples.  The observed test statistic    │
│  (mean, median, or sum difference) is ranked against a null         │
│  distribution built by combinatorial or random reshuffling.         │
│                                                                      │
│  Author : Bart R.H. Geurten                                         │
│  Licence: MIT                                                        │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Callable, List, Literal, Sequence, Tuple, Union

import numpy as np

from rerandomstats.resample_n_of_k import GetNofK


class FisherResamplingTest:
    """Two-sample resampling test in the tradition of R.A. Fisher.

    The observed difference between two groups (by mean, median, or
    sum) is ranked against a null distribution constructed by
    reshuffling group labels.

    Args:
        data_a: First sample.
        data_b: Second sample.
        func: Test-statistic identifier — ``'meanDiff'``,
            ``'medianDiff'``, or ``'sumDiff'``.
        combination_n: ``'all'`` for exhaustive permutation or an
            integer for random resampling draws.

    Attributes:
        p_value: Two-sided p-value after :meth:`main` has been called.
        original_test_result: Observed test statistic.
        shuffled_results: Sorted null distribution of the statistic.

    Example:
        >>> test = FisherResamplingTest([1, 2, 3], [7, 8, 9], 'meanDiff', 10000)
        >>> p = test.main()
        >>> p < 0.05
        True
    """

    def __init__(
        self,
        data_a: Sequence[float],
        data_b: Sequence[float],
        func: Literal["meanDiff", "medianDiff", "sumDiff"],
        combination_n: Union[int, str] = 10_000,
    ) -> None:
        self.data_a = data_a
        self.data_b = data_b
        self.func = func
        self.combination_n = combination_n

        self.p_value: float | None = None
        self.original_test_result: float | None = None
        self.shuffled_results: List[float] = []
        self.n_of_k: GetNofK | None = None
        self.resample_n: int = 0

    # ── shuffled-index generation ────────────────────────────────────

    def get_shuffled_indices(self) -> None:
        """Build the combinatorial index sets via :class:`GetNofK`."""
        self.n_of_k = GetNofK(self.data_a, self.data_b, self.combination_n)
        self.n_of_k.main()
        self.resample_n = self.n_of_k.combination_n

    # ── main entry point ─────────────────────────────────────────────

    def main(self) -> float:
        """Execute the full resampling pipeline and return the p-value.

        Steps:
            1. Generate shuffled index sets.
            2. Compute the observed test statistic.
            3. Build the null distribution via bootstrap resampling.
            4. Rank the observed statistic and derive a two-sided
               p-value.

        Returns:
            Two-sided p-value.
        """
        self.get_shuffled_indices()
        self.original_test_result = self.calculate_test(self.data_a, self.data_b)
        self.shuffled_results = sorted(self.bootstrap_resampling())

        self.index_of_original_in_shuffled = self._get_index_of_closest_value(
            self.shuffled_results, self.original_test_result
        )
        self.index_normalized = self.index_of_original_in_shuffled / self.resample_n

        if self.index_normalized > 0.5:
            self.index_normalized = abs(self.index_normalized - 1)
        if self.index_normalized == 0.0:
            self.index_normalized = 1.0 / self.resample_n

        self.p_value = self.index_normalized * 2
        return self.p_value

    # ── closest-value ranking ────────────────────────────────────────

    @staticmethod
    def _get_index_of_closest_value(
        values_list: Sequence[float], value_to_match: float
    ) -> float:
        """Return the median index of the closest value(s) in a sorted list.

        When multiple values share the minimum absolute difference
        (e.g. when both samples are identical), the *median* of their
        indices is returned to avoid a first-index bias.

        Args:
            values_list: Sorted numeric sequence.
            value_to_match: Target value.

        Returns:
            Median index of the closest matching value(s).
        """
        arr = np.sort(np.asarray(values_list, dtype=float))
        abs_diff = np.abs(arr - value_to_match)
        min_diff = np.min(abs_diff)
        min_indices = np.where(abs_diff == min_diff)[0]
        return float(np.median(min_indices))

    # ── null-distribution construction ───────────────────────────────

    def bootstrap_resampling(self) -> List[float]:
        """Compute the test statistic for every reshuffled split.

        Returns:
            List of resampled test-statistic values.
        """
        assert self.n_of_k is not None
        results: List[float] = []
        for i in range(self.resample_n):
            shuf_a, shuf_b = self.n_of_k.get_shuffled_set(i)
            results.append(self.calculate_test(shuf_a, shuf_b))
        return results

    # ── test-statistic dispatch ──────────────────────────────────────

    def calculate_test(
        self, data_a: Sequence[float], data_b: Sequence[float]
    ) -> float:
        """Dispatch to the chosen test-statistic function.

        Args:
            data_a: First sample.
            data_b: Second sample.

        Returns:
            Scalar test statistic.

        Raises:
            ValueError: If :attr:`func` is not recognised.
        """
        if self.func == "medianDiff":
            return float(np.median(data_a) - np.median(data_b))
        elif self.func == "meanDiff":
            return float(np.mean(data_a) - np.mean(data_b))
        elif self.func == "sumDiff":
            return float(np.sum(data_a) - np.sum(data_b))
        else:
            raise ValueError(
                f"FisherResamplingTest.calculate_test: "
                f"unknown statistic '{self.func}'"
            )
