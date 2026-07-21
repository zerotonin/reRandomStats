"""
┌──────────────────────────────────────────────────────────────────────┐
│     multi_group_test.py « Pairwise Multi-Group Comparisons »         │
│                                                                      │
│  Runs all pairwise statistical tests across >2 groups, then         │
│  applies multiple-testing correction (default: Benjamini-Hochberg   │
│  FDR).  Supports Fisher resampling, Fisher exact, classical         │
│  hypothesis tests, and binomial proportion tests.                   │
│                                                                      │
│  Author : Bart R.H. Geurten                                         │
│  Licence: MIT                                                        │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from rerandomstats.binomial_stats import MultipleBinomialTests
from rerandomstats.fisher_exact import FisherExactTest
from rerandomstats.fisher_resampling import FisherResamplingTest
from rerandomstats.hypothesis_tests import HypothesisTests


class MultiGroupTest:
    """Pairwise multi-group testing with FDR correction.

    Given parallel lists of *data* values and *group* labels the class
    performs all (or a user-specified subset of) pairwise comparisons,
    then adjusts p-values for multiplicity.

    The test is selected via a ``'family:name'`` string:

    =================== ==============================================
    Family              Available names
    =================== ==============================================
    ``Fisher``          ``medianDiff``, ``meanDiff``, ``sumDiff``,
                        ``exact``
    ``Binomial``        ``ztest``, ``chi2``
    ``hypo``            ``MannWhitneyU``, ``KruskalWallis``,
                        ``ChiSquare``, ``Kolmogorov``, ``MoodMedian``,
                        ``WilcoxonRankSum``, ``IndependentT``
    =================== ==============================================

    Args:
        data: Flat list of observed values.
        group: Corresponding group labels (same length as *data*).
        test: ``'family:name'`` string.
        combination_n: Passed to :class:`FisherResamplingTest` when
            the Fisher family is selected.
        correction_type: Any method accepted by
            :func:`statsmodels.stats.multipletests`.
        combination_set: Optional list of ``(groupA, groupB)`` tuples
            restricting which pairs are tested.
        seed: Passed to :class:`FisherResamplingTest` so a whole
            pairwise sweep is reproducible.  ``None`` (the default)
            keeps the historical run-to-run variation.

    Attributes:
        df: :class:`pandas.DataFrame` with results (available after
            :meth:`main`).

    Example:
        >>> import numpy as np
        >>> data = list(np.random.randn(30))
        >>> groups = ['A']*10 + ['B']*10 + ['C']*10
        >>> mgt = MultiGroupTest(data, groups, 'hypo:MannWhitneyU', 1000)
        >>> result = mgt.main()
        >>> isinstance(result, pd.DataFrame)
        True
    """

    def __init__(
        self,
        data: Sequence[float],
        group: Sequence[str],
        test: str,
        combination_n: Union[int, str] = 10_000,
        correction_type: str = "fdr_bh",
        combination_set: Sequence[Tuple[str, str]] = (),
        seed: int | None = None,
    ) -> None:
        self.data = data
        self.group = group
        self.test = test
        self.correction_type = correction_type
        self.combination_n = combination_n
        self.combination_set = combination_set
        self.seed = seed

        # populated by main()
        self.group_names: Tuple[str, ...] = ()
        self.grouped_data: Tuple[List[float], ...] = ()
        self.group_combinations: List[Tuple[int, int]] = []
        self.p_values: List[float] = []
        self.p_values_corrected: List[float] = []
        self.sig: List[bool] = []
        self.df: Optional[pd.DataFrame] = None

    # ── data rearrangement ───────────────────────────────────────────

    def rearrange_data(self) -> None:
        """Group flat *data* / *group* lists into per-group sublists."""
        names: List[str] = []
        grouped: List[List[float]] = []
        for name in dict.fromkeys(self.group):  # preserves insertion order
            indices = [i for i, g in enumerate(self.group) if g == name]
            grouped.append([self.data[i] for i in indices])
            names.append(name)
        self.grouped_data = tuple(grouped)
        self.group_names = tuple(names)

    # ── main entry point ─────────────────────────────────────────────

    def main(self) -> pd.DataFrame:
        """Run all pairwise tests and return a summary DataFrame.

        Returns:
            DataFrame with columns *groupA*, *groupA_n*, *groupB*,
            *groupB_n*, *p value*, *p value corrected*, *h*, and
            *sig. level*.
        """
        self.rearrange_data()
        self.group_combinations = self._get_combinations()
        self.p_values = self._run_tests()

        reject, corrected, _, _ = sm.stats.multipletests(
            self.p_values, alpha=0.05, method=self.correction_type
        )
        self.sig = list(reject)
        self.p_values_corrected = list(corrected)
        self.df = self._create_output()
        return self.df

    # ── combination generation ───────────────────────────────────────

    def _get_combinations(self) -> List[Tuple[int, int]]:
        """Return index pairs for all group pairings to test."""
        if self.combination_set:
            return [
                (self.group_names.index(a), self.group_names.index(b))
                for a, b in self.combination_set
            ]
        return list(combinations(range(len(self.group_names)), 2))

    # ── significance stars ───────────────────────────────────────────

    @staticmethod
    def get_significance_level(p_value: float) -> str:
        """Convert a p-value to the conventional star notation.

        Args:
            p_value: Corrected p-value.

        Returns:
            ``'n.s.'``, ``'*'``, ``'**'``, or ``'***'``.
        """
        if p_value > 0.05:
            return "n.s."
        if p_value > 0.01:
            return "*"
        if p_value > 0.001:
            return "**"
        return "***"

    # ── output table construction ────────────────────────────────────

    def _create_output(self) -> pd.DataFrame:
        """Build the results DataFrame."""
        group_list = list(self.group)
        first = [self.group_names[c[0]] for c in self.group_combinations]
        n1 = [group_list.count(self.group_names[c[0]]) for c in self.group_combinations]
        second = [self.group_names[c[1]] for c in self.group_combinations]
        n2 = [group_list.count(self.group_names[c[1]]) for c in self.group_combinations]
        stars = [self.get_significance_level(p) for p in self.p_values_corrected]
        return pd.DataFrame(
            {
                "groupA": first,
                "groupA_n": n1,
                "groupB": second,
                "groupB_n": n2,
                "p value": self.p_values,
                "p value corrected": self.p_values_corrected,
                "h": self.sig,
                "sig. level": stars,
            }
        )

    # ── test dispatch ────────────────────────────────────────────────

    def _run_tests(self) -> List[float]:
        """Execute the chosen test for every group pair."""
        test_obj = self._choose_test()
        p_values: List[float] = []
        for idx_a, idx_b in tqdm(self.group_combinations, desc="testing group combinations"):
            test_obj.data_a = self.grouped_data[idx_a]
            test_obj.data_b = self.grouped_data[idx_b]
            p_values.append(test_obj.main())
        return p_values

    def _choose_test(self):
        """Instantiate the appropriate test object from :attr:`test`.

        The *test* string has the form ``'family:name'``.

        Returns:
            An uninitialised test object whose ``data_a`` / ``data_b``
            attributes will be set per comparison.

        Raises:
            ValueError: If the test family is unrecognised.
        """
        family, name = self.test.split(":")

        if family == "Fisher":
            if name == "exact":
                return FisherExactTest((), ())
            return FisherResamplingTest([], [], name, self.combination_n,
                                        seed=self.seed)
        elif family == "Binomial":
            return MultipleBinomialTests((), (), name)
        elif family == "hypo":
            return HypothesisTests([], [], name)
        else:
            raise ValueError(
                f"MultiGroupTest: unknown test family '{family}'"
            )
