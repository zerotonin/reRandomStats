"""
┌──────────────────────────────────────────────────────────────────────┐
│  paired_resampling.py « Resampling Test for Partially Paired Groups »│
│                                                                      │
│  Compares two groups that share some of their subjects — a herd      │
│  measured in two summers, a cohort before and after.  The statistic  │
│  is the difference of the group means / medians / sums, so the       │
│  question stays at group level; the null distribution respects the   │
│  pairing, so repeated subjects are not counted as independent.       │
│                                                                      │
│  Author : Bart R.H. Geurten                                          │
│  Licence: MIT                                                        │
└──────────────────────────────────────────────────────────────────────┘

Two classes: :class:`PartiallyPairedResamplingTest` compares two groups,
:class:`MultiGroupPairedTest` runs every pairwise comparison among
several groups and corrects for multiplicity.

**What is borrowed and what is not.**  The rearrangement scheme — swap
the two values of a complete pair with probability one half, re-partition
the unpaired values at random between the groups — is that of Einsporn &
Habtzghi (2013).  Their test statistic is a weighted combination of the
paired and the unpaired mean difference.  The statistic here is the
difference of the group means, medians or sums over all observations, so
that the reported number is the difference between the two groups as
they are plotted.  A permutation test is valid for any statistic once
the rearrangements respect the design.

**References.**

- Einsporn RL, Habtzghi D (2013) Combining paired and two-sample data
  using a permutation test.  Journal of Data Science 11(4):767-779.
  doi:10.6339/JDS.2013.11(4).1164 — the rearrangement scheme.
- Phipson B, Smyth GK (2010) Permutation P-values should never be zero:
  calculating exact P-values when permutations are randomly drawn.
  Statistical Applications in Genetics and Molecular Biology 9(1).
  doi:10.2202/1544-6115.1585 — the p-value ``(k + 1) / (N + 1)``.
- Benjamini Y, Hochberg Y (1995) Controlling the false discovery rate:
  a practical and powerful approach to multiple testing.  Journal of the
  Royal Statistical Society Series B 57(1):289-300.
  doi:10.1111/j.2517-6161.1995.tb02031.x — the default correction of
  :class:`MultiGroupPairedTest`.
- Derrick B, White P (2022) Review of the partially overlapping samples
  framework: paired observations and independent observations in two
  samples.  The Quantitative Methods for Psychology 18(1):55-65.
  doi:10.20982/tqmp.18.1.p055 — review of the design and of the tests
  proposed for it.

**Related permutation approaches** (other statistics, other schemes).

- Amro L, Pauly M (2017) Permuting incomplete paired data: a novel exact
  and asymptotic correct randomization test.  Journal of Statistical
  Computation and Simulation 87(6):1148-1159.
  doi:10.1080/00949655.2016.1249871
- Amro L, Konietschke F, Pauly M (2019) Multiplication-combination tests
  for incomplete paired data.  Statistics in Medicine 38(17):3243-3255.
  doi:10.1002/sim.8178
- Johnson EN, Richter SJ (2022) Permutation tests for mixed paired and
  two-sample designs.  Computational Statistics 37(2):739-750.
  doi:10.1007/s00180-021-01137-9
"""

from __future__ import annotations

from itertools import combinations
from typing import Hashable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from rerandomstats.multi_group_test import MultiGroupTest

_STATISTICS = {"medianDiff": np.median, "meanDiff": np.mean, "sumDiff": np.sum}
_CHUNK = 2_000  # permutations evaluated per vectorised block


class PartiallyPairedResamplingTest:
    """Two-group resampling test for groups that share subjects.

    Fisher's resampling test shuffles every observation freely between
    the two groups, which assumes the observations are independent.  When
    some subjects appear in both groups that assumption fails: their two
    values are correlated, and free shuffling misjudges the spread of the
    null distribution.

    Here the test statistic is unchanged — ``statistic(A) − statistic(B)``
    over *all* observations of each group — but the null distribution is
    built from the rearrangements the design allows:

    * a subject measured in **both** groups keeps both values and only
      swaps which one counts for which group (probability one half);
    * subjects measured in **one** group only are shuffled among
      themselves between the groups.

    Group sizes are therefore the same in every rearrangement.  With no
    shared subjects the test is an ordinary two-sample resampling test;
    with only shared subjects it is a paired test of the group statistic.

    Args:
        data_a: Values of the first group.
        data_b: Values of the second group.
        subject_a: Subject identifier of each value in ``data_a``.
        subject_b: Subject identifier of each value in ``data_b``.
        func: Test-statistic identifier — ``'meanDiff'``,
            ``'medianDiff'``, or ``'sumDiff'``.
        combination_n: Number of random rearrangements.  Exhaustive
            enumeration is not offered; the space grows as
            ``2**n_paired`` times a binomial coefficient.
        seed: Seed for the rearrangements.  Pass an integer when the
            p-value goes into a manuscript.

    Attributes:
        p_value: Two-sided p-value, ``(k + 1) / (combination_n + 1)``
            with ``k`` the number of rearrangements at least as extreme
            as the observed statistic.
        original_test_result: Observed test statistic.
        shuffled_results: Sorted null distribution of the statistic.
        n_paired: Subjects present in both groups.
        n_only_a: Subjects present in the first group only.
        n_only_b: Subjects present in the second group only.

    Non-finite values are dropped before pairing, so a subject with a
    missing value in one group counts as unpaired in the other.

    The rearrangement scheme is that of Einsporn & Habtzghi (2013), who
    permute complete pairs and re-partition the unpaired observations "in
    the standard ways associated with randomization testing".  Their
    statistic is a weighted combination of the paired and the unpaired
    mean difference; here it is the difference of the group statistics
    over all observations, which keeps the estimate directly readable as
    the difference between the two groups.  The p-value follows Phipson &
    Smyth (2010).  Derrick & White (2022) review the wider family of
    tests for partially overlapping samples.

    References:
        Einsporn RL, Habtzghi D (2013) Combining paired and two-sample
        data using a permutation test.  Journal of Data Science
        11(4):767-779.  doi:10.6339/JDS.2013.11(4).1164

        Phipson B, Smyth GK (2010) Permutation P-values should never be
        zero: calculating exact P-values when permutations are randomly
        drawn.  Statistical Applications in Genetics and Molecular
        Biology 9(1).  doi:10.2202/1544-6115.1585

        Derrick B, White P (2022) Review of the partially overlapping
        samples framework: paired observations and independent
        observations in two samples.  The Quantitative Methods for
        Psychology 18(1):55-65.  doi:10.20982/tqmp.18.1.p055

    Example:
        >>> a = [10.0, 12.0, 14.0, 16.0, 18.0, 11.0]
        >>> b = [11.5, 13.4, 15.6, 17.5, 19.4, 30.0]
        >>> ids_a = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
        >>> ids_b = ['c1', 'c2', 'c3', 'c4', 'c5', 'c7']
        >>> test = PartiallyPairedResamplingTest(a, b, ids_a, ids_b,
        ...                                      'meanDiff', 2000, seed=1)
        >>> 0.0 < test.main() <= 1.0
        True
        >>> test.n_paired, test.n_only_a, test.n_only_b
        (5, 1, 1)
    """

    def __init__(
        self,
        data_a: Sequence[float],
        data_b: Sequence[float],
        subject_a: Sequence[Hashable],
        subject_b: Sequence[Hashable],
        func: Literal["meanDiff", "medianDiff", "sumDiff"],
        combination_n: int = 10_000,
        seed: int | None = None,
    ) -> None:
        self.data_a = data_a
        self.data_b = data_b
        self.subject_a = subject_a
        self.subject_b = subject_b
        self.func = func
        self.combination_n = combination_n
        self.seed = seed

        self.p_value: float | None = None
        self.original_test_result: float | None = None
        self.shuffled_results: np.ndarray = np.empty(0)
        self.n_paired: int = 0
        self.n_only_a: int = 0
        self.n_only_b: int = 0

    # ── main entry point ─────────────────────────────────────────────

    def main(self) -> float:
        """Run the test and return the two-sided p-value.

        Raises:
            ValueError: On an unknown statistic, a non-integer
                ``combination_n``, values and subjects of different
                length, a subject repeated within one group, or an
                empty group.
        """
        if self.func not in _STATISTICS:
            raise ValueError(
                f"PartiallyPairedResamplingTest: unknown statistic '{self.func}'")
        if isinstance(self.combination_n, bool) \
                or not isinstance(self.combination_n, (int, np.integer)) \
                or self.combination_n < 1:
            raise ValueError(
                "PartiallyPairedResamplingTest: combination_n must be a "
                "positive integer (exhaustive enumeration is not supported)")

        a = self._as_series(self.data_a, self.subject_a, "first")
        b = self._as_series(self.data_b, self.subject_b, "second")
        if a.empty or b.empty:
            raise ValueError(
                "PartiallyPairedResamplingTest: both groups need at least "
                "one finite value")

        shared = a.index.intersection(b.index)
        a_paired, b_paired = a.loc[shared].to_numpy(), b.loc[shared].to_numpy()
        a_only = a.drop(index=shared).to_numpy()
        b_only = b.drop(index=shared).to_numpy()
        self.n_paired = int(len(shared))
        self.n_only_a, self.n_only_b = int(a_only.size), int(b_only.size)

        statistic = _STATISTICS[self.func]
        observed = float(statistic(a.to_numpy()) - statistic(b.to_numpy()))
        self.original_test_result = observed

        null = self._null_distribution(a_paired, b_paired, a_only, b_only)
        self.shuffled_results = np.sort(null)
        tolerance = 1e-12 * max(1.0, abs(observed))
        k = int(np.sum(np.abs(null) >= abs(observed) - tolerance))
        self.p_value = (k + 1) / (self.combination_n + 1)
        return self.p_value

    # ── input handling ───────────────────────────────────────────────

    @staticmethod
    def _as_series(values: Sequence[float], subjects: Sequence[Hashable],
                   which: str) -> pd.Series:
        """Finite values indexed by subject; one value per subject."""
        values = np.asarray(values, dtype=float)
        subjects = np.asarray(subjects, dtype=object)
        if values.shape != subjects.shape:
            raise ValueError(
                f"PartiallyPairedResamplingTest: {which} group has "
                f"{values.size} values but {subjects.size} subjects")
        keep = np.isfinite(values)
        series = pd.Series(values[keep], index=pd.Index(subjects[keep]))
        if series.index.has_duplicates:
            repeated = series.index[series.index.duplicated()].unique().tolist()
            raise ValueError(
                f"PartiallyPairedResamplingTest: subject(s) {repeated[:5]} "
                f"occur more than once in the {which} group")
        return series

    # ── null-distribution construction ───────────────────────────────

    def _null_distribution(self, a_paired: np.ndarray, b_paired: np.ndarray,
                           a_only: np.ndarray, b_only: np.ndarray) -> np.ndarray:
        """Statistic under the rearrangements the design allows."""
        statistic = _STATISTICS[self.func]
        rng = np.random.default_rng(self.seed)
        pool = np.concatenate([a_only, b_only])
        null = np.empty(self.combination_n)
        for start in range(0, self.combination_n, _CHUNK):
            m = min(_CHUNK, self.combination_n - start)
            parts_a: List[np.ndarray] = []
            parts_b: List[np.ndarray] = []
            if a_paired.size:
                swap = rng.random((m, a_paired.size)) < 0.5
                parts_a.append(np.where(swap, b_paired, a_paired))
                parts_b.append(np.where(swap, a_paired, b_paired))
            if pool.size:
                order = np.argsort(rng.random((m, pool.size)), axis=1)
                shuffled = pool[order]
                parts_a.append(shuffled[:, :a_only.size])
                parts_b.append(shuffled[:, a_only.size:])
            group_a = np.concatenate(parts_a, axis=1)
            group_b = np.concatenate(parts_b, axis=1)
            null[start:start + m] = (statistic(group_a, axis=1)
                                     - statistic(group_b, axis=1))
        return null


class MultiGroupPairedTest:
    """All pairwise :class:`PartiallyPairedResamplingTest` comparisons.

    The counterpart of :class:`~rerandomstats.MultiGroupTest` for groups
    that share subjects: every pair of groups is compared at group level
    with a null distribution that respects the pairing, and the p-values
    are corrected for multiplicity.

    Args:
        data: Flat list of observed values.
        group: Group label of each value.
        subject: Subject identifier of each value.
        func: ``'meanDiff'``, ``'medianDiff'``, or ``'sumDiff'``.
        combination_n: Rearrangements per comparison.
        correction_type: Any method accepted by
            :func:`statsmodels.stats.multipletests`.
        combination_set: Optional list of ``(groupA, groupB)`` tuples
            restricting which pairs are tested.
        seed: Seed for the rearrangements of every comparison.

    Attributes:
        df: Result table (available after :meth:`main`).

    Each comparison uses the rearrangement scheme of Einsporn & Habtzghi
    (2013) with the group-level statistic described in
    :class:`PartiallyPairedResamplingTest`; the default correction is the
    false-discovery-rate procedure of Benjamini & Hochberg (1995).

    References:
        Einsporn RL, Habtzghi D (2013) Combining paired and two-sample
        data using a permutation test.  Journal of Data Science
        11(4):767-779.  doi:10.6339/JDS.2013.11(4).1164

        Benjamini Y, Hochberg Y (1995) Controlling the false discovery
        rate: a practical and powerful approach to multiple testing.
        Journal of the Royal Statistical Society Series B 57(1):289-300.
        doi:10.1111/j.2517-6161.1995.tb02031.x

        Derrick B, White P (2022) Review of the partially overlapping
        samples framework: paired observations and independent
        observations in two samples.  The Quantitative Methods for
        Psychology 18(1):55-65.  doi:10.20982/tqmp.18.1.p055

    Example:
        >>> data = [1.0, 2.0, 3.0, 4.0, 2.1, 3.2, 4.1, 5.3, 9.0, 9.5, 8.7, 9.9]
        >>> group = ['y1'] * 4 + ['y2'] * 4 + ['y3'] * 4
        >>> subject = ['a', 'b', 'c', 'd'] * 3
        >>> MultiGroupPairedTest(data, group, subject, 'meanDiff',
        ...                      500, seed=1).main().shape[0]
        3
    """

    def __init__(
        self,
        data: Sequence[float],
        group: Sequence[Hashable],
        subject: Sequence[Hashable],
        func: Literal["meanDiff", "medianDiff", "sumDiff"] = "medianDiff",
        combination_n: int = 10_000,
        correction_type: str = "fdr_bh",
        combination_set: Sequence[Tuple[Hashable, Hashable]] = (),
        seed: int | None = None,
    ) -> None:
        self.data = data
        self.group = group
        self.subject = subject
        self.func = func
        self.combination_n = combination_n
        self.correction_type = correction_type
        self.combination_set = combination_set
        self.seed = seed
        self.df: Optional[pd.DataFrame] = None

    def main(self) -> pd.DataFrame:
        """Run all pairwise tests and return a summary DataFrame.

        Returns:
            DataFrame with the columns of :class:`MultiGroupTest`
            (*groupA*, *groupA_n*, *groupB*, *groupB_n*, *p value*,
            *p value corrected*, *h*, *sig. level*) plus *n_paired*,
            *n_only_A*, *n_only_B* and the observed *statistic*.
        """
        frame = pd.DataFrame({
            "value": np.asarray(self.data, dtype=float),
            "group": list(self.group),
            "subject": list(self.subject),
        })
        names = list(dict.fromkeys(frame["group"]))
        pairs = (list(self.combination_set) if self.combination_set
                 else list(combinations(names, 2)))

        rows = []
        for name_a, name_b in tqdm(pairs, desc="testing group combinations"):
            part_a = frame[frame["group"] == name_a]
            part_b = frame[frame["group"] == name_b]
            test = PartiallyPairedResamplingTest(
                part_a["value"].to_numpy(), part_b["value"].to_numpy(),
                part_a["subject"].to_numpy(), part_b["subject"].to_numpy(),
                self.func, self.combination_n, seed=self.seed)
            p_value = test.main()
            rows.append({
                "groupA": name_a,
                "groupA_n": test.n_paired + test.n_only_a,
                "groupB": name_b,
                "groupB_n": test.n_paired + test.n_only_b,
                "n_paired": test.n_paired,
                "n_only_A": test.n_only_a,
                "n_only_B": test.n_only_b,
                "statistic": test.original_test_result,
                "p value": p_value,
            })
        table = pd.DataFrame(rows)
        reject, corrected, _, _ = sm.stats.multipletests(
            table["p value"], alpha=0.05, method=self.correction_type)
        table["p value corrected"] = corrected
        table["h"] = reject
        table["sig. level"] = [MultiGroupTest.get_significance_level(p)
                               for p in corrected]
        self.df = table
        return table
