"""
┌──────────────────────────────────────────────────────────────────────┐
│              fisher_exact.py « Fisher's Exact Test »                 │
│                                                                      │
│  Wrapper around scipy.stats.fisher_exact for 2×2 contingency        │
│  tables.  Returns the p-value for the null hypothesis that the      │
│  odds ratio of the underlying population equals one.                │
│                                                                      │
│  Author : Bart R.H. Geurten                                         │
│  Licence: MIT                                                        │
└──────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
from scipy.stats import fisher_exact


class FisherExactTest:
    """Perform Fisher's exact test on a 2×2 contingency table.

    The two input tuples each contain two counts (option_A, option_B).
    For example ``(alive, dead)`` for treatment and control groups.

    Args:
        data_a: Counts for group A as ``(count_optionA, count_optionB)``.
        data_b: Counts for group B as ``(count_optionA, count_optionB)``.
        alternative: Sidedness of the test — ``'two-sided'``,
            ``'less'``, or ``'greater'``.

    Attributes:
        p_value: The p-value after :meth:`main` has been called.

    Example:
        >>> test = FisherExactTest((8, 2), (1, 5))
        >>> p = test.main()
        >>> p < 0.05
        True
    """

    def __init__(
        self,
        data_a: Tuple[int, int],
        data_b: Tuple[int, int],
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> None:
        self.data_a = data_a
        self.data_b = data_b
        self.alternative = alternative
        self.p_value: float | None = None

    # ── main entry point ─────────────────────────────────────────────
    def main(self) -> float:
        """Run Fisher's exact test and return the p-value.

        Constructs the 2×2 contingency table from *data_a* and *data_b*,
        delegates to :func:`scipy.stats.fisher_exact`, and stores /
        returns the resulting p-value.

        Returns:
            The p-value of Fisher's exact test.
        """
        table = np.array([list(self.data_a), list(self.data_b)])
        _, self.p_value = fisher_exact(table, alternative=self.alternative)
        return self.p_value
