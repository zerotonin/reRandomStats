#!/usr/bin/env python3
"""
┌──────────────────────────────────────────────────────────────────────┐
│          example_analysis.py « Quick-Start Demo »                    │
│                                                                      │
│  Demonstrates the core reRandomStats workflow:                      │
│    1. Fisher resampling test (two-sample)                           │
│    2. Multi-group pairwise comparisons with FDR correction          │
│    3. Binomial proportion test                                      │
│    4. Classical hypothesis test via the unified interface            │
└──────────────────────────────────────────────────────────────────────┘
"""

import numpy as np

from rerandomstats import (
    BinomialStats,
    FisherExactTest,
    FisherResamplingTest,
    HypothesisTests,
    MultiGroupTest,
)


def main() -> None:
    np.random.seed(42)

    # ── 1. Two-sample Fisher resampling ──────────────────────────────
    print("=" * 60)
    print("1. Fisher Resampling Test (median difference)")
    print("=" * 60)

    control = [2.1, 3.5, 1.8, 4.2, 3.0, 2.7, 1.9, 3.8]
    treatment = [5.4, 6.1, 7.3, 5.9, 6.8, 7.0, 5.5, 6.4]

    test = FisherResamplingTest(control, treatment, "medianDiff", 20_000)
    p = test.main()
    print(f"  Control   median: {np.median(control):.2f}")
    print(f"  Treatment median: {np.median(treatment):.2f}")
    print(f"  p-value:          {p:.4f}\n")

    # ── 2. Multi-group pairwise comparisons ──────────────────────────
    print("=" * 60)
    print("2. Multi-Group Test (Fisher resampling, BH-FDR)")
    print("=" * 60)

    data = list(
        np.concatenate(
            [
                np.random.normal(0, 1, 12),
                np.random.normal(3, 1, 12),
                np.random.normal(6, 1, 12),
            ]
        )
    )
    groups = ["wildtype"] * 12 + ["mutant_A"] * 12 + ["mutant_B"] * 12

    mgt = MultiGroupTest(data, groups, "Fisher:medianDiff", 10_000)
    result = mgt.main()
    print(result.to_string(index=False))
    print()

    # ── 3. Fisher's exact test ───────────────────────────────────────
    print("=" * 60)
    print("3. Fisher's Exact Test")
    print("=" * 60)

    fet = FisherExactTest((45, 5), (30, 20))
    print(f"  p-value: {fet.main():.4f}\n")

    # ── 4. Binomial proportion ───────────────────────────────────────
    print("=" * 60)
    print("4. Binomial Test + Wilson CI")
    print("=" * 60)

    bs = BinomialStats(heads=73, total_flips=100)
    binom_result = bs.binomial_test(base_rate=0.5)
    ci = bs.exact_ci()
    print(f"  p-value:    {binom_result.pvalue:.4f}")
    print(f"  Proportion: {ci['Proportion']}%")
    print(f"  Wilson CI:  [{ci['Lower CI']}, {ci['Upper CI']}]%\n")

    # ── 5. Classical hypothesis test ─────────────────────────────────
    print("=" * 60)
    print("5. Mann-Whitney U Test")
    print("=" * 60)

    ht = HypothesisTests(control, treatment, "MannWhitneyU")
    print(f"  p-value: {ht.main():.4f}\n")


if __name__ == "__main__":
    main()
