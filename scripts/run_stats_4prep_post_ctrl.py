#!/usr/bin/env python3
"""
┌──────────────────────────────────────────────────────────────────────┐
│    monitoring_pre_post_ctrl.py « Pre vs Post Predator Control »       │
│                                                                      │
│  Compares detection rates before and after predator control drops   │
│  using Fisher's resampling test with Benjamini-Hochberg FDR         │
│  correction via the reRandomStats package.                          │
│                                                                      │
│  Design:                                                             │
│    - post_ctrl = 0 (sessions 1 & 2, pre-control)                   │
│    - post_ctrl = 1 (sessions 3 & 4, post-control)                  │
│    - 0.5 sessions excluded                                          │
│    - 80 clusters × 3 methods × 2 sessions per condition            │
│    - Response: Tukutuku, Oligosoma, total_lizard, Rattus,          │
│                Lizard, Nothing                                      │
│                                                                      │
│  Note on test statistic:                                             │
│    Data is heavily zero-inflated — all group medians = 0.           │
│    meanDiff is used as the primary test statistic.                  │
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
RESPONSE_VARS = ["Tukutuku", "Oligosoma", "total_lizard", "Rattus", "Lizard", "Nothing"]
TEST_STATISTICS = ["meanDiff"]

# ┌──────────────────────────────────────────────────────────────────┐
# │                     « Data Loading »                             │
# └──────────────────────────────────────────────────────────────────┘

print("=" * 70)
print("  MONITORING DATA — PRE vs POST PREDATOR CONTROL")
print("  Fisher resampling test | Benjamini-Hochberg FDR")
print("=" * 70)

df = pd.read_csv(CSV_PATH)
df = df[df["session"].isin(SESSIONS)].copy()
df["total_lizard"] = df["Tukutuku"] + df["Oligosoma"]
df["ctrl_label"] = df["post_ctrl"].map({0: "pre_ctrl", 1: "post_ctrl"})

print(f"\nFiltered to sessions {SESSIONS}: {len(df)} observations")
print(f"Clusters: {df['cluster'].nunique()}")
print(f"Methods:  {sorted(df['method_type'].unique())}")
print(f"Pre-control  (sessions 1,2): n={len(df[df['post_ctrl']==0])}")
print(f"Post-control (sessions 3,4): n={len(df[df['post_ctrl']==1])}")
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
    for label in ["pre_ctrl", "post_ctrl"]:
        vals = df.loc[df["ctrl_label"] == label, var].dropna()
        n_zero = (vals == 0).sum()
        print(
            f"    {label}: n={len(vals):3d}  sum={vals.sum():5.0f}  "
            f"mean={vals.mean():.4f}  median={vals.median():.1f}  "
            f"zeros={n_zero}/{len(vals)} ({100*n_zero/len(vals):.1f}%)"
        )

# ┌──────────────────────────────────────────────────────────────────┐
# │                  « Statistical Analysis — Pooled »               │
# └──────────────────────────────────────────────────────────────────┘

all_results = []

for stat in TEST_STATISTICS:
    print("\n" + "=" * 70)
    print(f"  FISHER RESAMPLING — {stat} — ALL METHODS POOLED")
    print("=" * 70)

    for var in RESPONSE_VARS:
        sub = df[["ctrl_label", var]].dropna(subset=[var])

        mgt = MultiGroupTest(
            data=sub[var].tolist(),
            group=sub["ctrl_label"].tolist(),
            test=f"Fisher:{stat}",
            combination_n=RESAMPLE_N,
            correction_type=CORRECTION,
        )
        result = mgt.main()

        result.insert(0, "response", var)
        result.insert(1, "statistic", stat)
        result.insert(2, "method", "pooled")
        all_results.append(result)

        print(f"\n  ── {var} ({stat}) ──")
        write_pretty_table(result, show=True)

# ┌──────────────────────────────────────────────────────────────────┐
# │            « Statistical Analysis — Per Method »                 │
# └──────────────────────────────────────────────────────────────────┘

for stat in TEST_STATISTICS:
    print("\n" + "=" * 70)
    print(f"  FISHER RESAMPLING — {stat} — PER METHOD TYPE")
    print("=" * 70)

    for method in sorted(df["method_type"].unique()):
        df_m = df[df["method_type"] == method]

        for var in RESPONSE_VARS:
            sub = df_m[["ctrl_label", var]].dropna(subset=[var])

            if sub["ctrl_label"].nunique() < 2:
                continue

            mgt = MultiGroupTest(
                data=sub[var].tolist(),
                group=sub["ctrl_label"].tolist(),
                test=f"Fisher:{stat}",
                combination_n=RESAMPLE_N,
                correction_type=CORRECTION,
            )
            result = mgt.main()

            result.insert(0, "response", var)
            result.insert(1, "statistic", stat)
            result.insert(2, "method", method)
            all_results.append(result)

            print(f"\n  ── {method} | {var} ({stat}) ──")
            write_pretty_table(result, show=True)

# ┌──────────────────────────────────────────────────────────────────┐
# │                  « Combined Output »                             │
# └──────────────────────────────────────────────────────────────────┘

combined = pd.concat(all_results, ignore_index=True)
out_path = Path("monitoring_pre_post_ctrl.csv")
combined.to_csv(out_path, index=False)
print(f"\nResults saved to: {out_path}")

# ┌──────────────────────────────────────────────────────────────────┐
# │                  « Summary »                                     │
# └──────────────────────────────────────────────────────────────────┘

print("\n" + "=" * 70)
print("  SUMMARY OF SIGNIFICANT RESULTS")
print("=" * 70)

sig = combined[combined["h"] == True]

if len(sig) == 0:
    print("\n  No significant pre vs post differences found.")
else:
    for _, row in sig.iterrows():
        print(
            f"\n  [{row['method']:6s}] {row['response']}: "
            f"{row['groupA']} vs {row['groupB']}"
            f"  p_corr={row['p value corrected']:.4f} {row['sig. level']}"
        )

print("\n" + "─" * 70)
print("  NOTE: meanDiff used due to heavy zero-inflation (all medians = 0).")
print("─" * 70)