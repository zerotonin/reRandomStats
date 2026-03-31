#!/usr/bin/env python3
"""
┌──────────────────────────────────────────────────────────────────────┐
│       monitoring_analysis.py « Method-Type Comparison »               │
│                                                                      │
│  Compares lizard detection rates across three monitoring methods     │
│  (ACO, BTT, STT) using Fisher's resampling test with                │
│  Benjamini-Hochberg FDR correction via the reRandomStats package.   │
│                                                                      │
│  Design:                                                             │
│    - 80 clusters, each containing one of each method                │
│    - 4 integer sessions (0.5 sessions excluded)                     │
│    - Paired within cluster; sessions pooled                         │
│    - Response: Tukutuku, Oligosoma, total lizard count              │
│                                                                      │
│  Note on test statistic:                                             │
│    Data is heavily zero-inflated (85–99% zeros). All group          │
│    medians = 0, so medianDiff produces a test statistic of 0       │
│    for every pair and the permutation test has no power.            │
│    meanDiff is used instead. medianDiff is run for completeness     │
│    but will return non-significant p-values by construction.        │
│                                                                      │
│  Author : Bart R.H. Geurten                                         │
└──────────────────────────────────────────────────────────────────────┘
"""

from pathlib import Path

import pandas as pd

from rerandomstats import MultiGroupTest, write_pretty_table

# ┌──────────────────────────────────────────────────────────────────┐
# │                     « Configuration »                            │
# └──────────────────────────────────────────────────────────────────┘

CSV_PATH = Path("/home/geuba03p/PyProjects/reRandomStats/data/Monitoring.csv")
RESAMPLE_N = 20_000
CORRECTION = "fdr_bh"
SESSIONS = [1, 2, 3, 4]  # ignore 0.5 sessions
RESPONSE_VARS = ["Tukutuku", "Oligosoma", "total_lizard"]
TEST_STATISTICS = ["meanDiff", "medianDiff"]

# ┌──────────────────────────────────────────────────────────────────┐
# │                     « Data Loading »                             │
# └──────────────────────────────────────────────────────────────────┘

print("=" * 70)
print("  MONITORING DATA — METHOD-TYPE COMPARISON")
print("  Fisher resampling test | Benjamini-Hochberg FDR")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
df = df[df["session"].isin(SESSIONS)].copy()
df["total_lizard"] = df["Tukutuku"] + df["Oligosoma"]

print(f"\nFiltered to sessions {SESSIONS}: {len(df)} observations")
print(f"Clusters: {df['cluster'].nunique()}")
print(f"Methods:  {sorted(df['method_type'].unique())}")
print(f"Resamples: {RESAMPLE_N:,}")
print(f"Correction: {CORRECTION}")

# ┌──────────────────────────────────────────────────────────────────┐
# │                  « Descriptive Statistics »                      │
# └──────────────────────────────────────────────────────────────────┘

print("\n" + "─" * 70)
print("  DESCRIPTIVE STATISTICS")
print("─" * 70)

for var in RESPONSE_VARS:
    print(f"\n  {var}:")
    for mt in sorted(df["method_type"].unique()):
        vals = df.loc[df["method_type"] == mt, var].dropna()
        n_zero = (vals == 0).sum()
        print(
            f"    {mt}: n={len(vals):3d}  sum={vals.sum():5.0f}  "
            f"mean={vals.mean():.4f}  median={vals.median():.1f}  "
            f"zeros={n_zero}/{len(vals)} ({100*n_zero/len(vals):.1f}%)"
        )

# ┌──────────────────────────────────────────────────────────────────┐
# │                  « Statistical Analysis »                        │
# └──────────────────────────────────────────────────────────────────┘

all_results = []

for stat in TEST_STATISTICS:
    print("\n" + "=" * 70)
    print(f"  FISHER RESAMPLING — {stat}")
    print("=" * 70)

    for var in RESPONSE_VARS:
        sub = df[["method_type", var]].dropna(subset=[var])

        mgt = MultiGroupTest(
            data=sub[var].tolist(),
            group=sub["method_type"].tolist(),
            test=f"Fisher:{stat}",
            combination_n=RESAMPLE_N,
            correction_type=CORRECTION,
        )
        result = mgt.main()

        # tag for combined output
        result.insert(0, "response", var)
        result.insert(1, "statistic", stat)
        all_results.append(result)

        print(f"\n  ── {var} ({stat}) ──")
        write_pretty_table(result, show=True)

# ┌──────────────────────────────────────────────────────────────────┐
# │                  « Combined Output »                             │
# └──────────────────────────────────────────────────────────────────┘

combined = pd.concat(all_results, ignore_index=True)
out_path = Path("monitoring_method_comparison.csv")
combined.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")

# ┌──────────────────────────────────────────────────────────────────┐
# │                  « Summary »                                     │
# └──────────────────────────────────────────────────────────────────┘

print("\n" + "=" * 70)
print("  SUMMARY OF SIGNIFICANT RESULTS (meanDiff only)")
print("=" * 70)

mean_results = combined[combined["statistic"] == "meanDiff"]
sig = mean_results[mean_results["h"] == True]

if len(sig) == 0:
    print("\n  No significant pairwise differences found.")
else:
    for _, row in sig.iterrows():
        print(
            f"\n  {row['response']}: {row['groupA']} vs {row['groupB']}"
            f"  p_corr={row['p value corrected']:.4f} {row['sig. level']}"
        )

print("\n" + "─" * 70)
print("  NOTE: medianDiff results are non-informative for this dataset.")
print("  All group medians = 0 due to heavy zero-inflation (85–99%).")
print("  meanDiff results should be used for interpretation.")
print("─" * 70)