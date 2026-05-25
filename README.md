# reRandomStats

[![Tests](https://github.com/zerotonin/rerandomstats/actions/workflows/tests.yml/badge.svg)](https://github.com/zerotonin/rerandomstats/actions/workflows/tests.yml)
[![Docs](https://github.com/zerotonin/rerandomstats/actions/workflows/docs.yml/badge.svg)](https://zerotonin.github.io/reRandomStats/)
[![Release](https://github.com/zerotonin/rerandomstats/actions/workflows/release.yml/badge.svg)](https://github.com/zerotonin/rerandomstats/releases)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PENDING.svg)](https://zenodo.org/)

```
╔══════════════════════════════════════════════════════════════════╗
║  reRandomStats                                                   ║
║  « Re-randomisation statistics in the spirit of Fisher »         ║
╚══════════════════════════════════════════════════════════════════╝
```

A comprehensive Python toolkit for **re-randomisation statistics** in the tradition of [Sir Ronald A. Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher). The package provides Fisher's resampling test with flexible test statistics, pairwise multi-group comparisons with multiple-testing correction, binomial proportion tests, and a unified interface to classical parametric and non-parametric hypothesis tests.

## Features

### Core resampling and hypothesis-test framework

- **Fisher's Resampling Test** — permutation-based two-sample test using mean, median, or sum differences as the test statistic. Supports exhaustive enumeration for small samples and random resampling for large ones.
- **Fisher's Exact Test** — wrapper for 2×2 contingency table analysis.
- **Multi-Group Pairwise Testing** — runs all (or user-specified) pairwise comparisons with automatic multiple-testing correction (Benjamini-Hochberg FDR, Bonferroni, Holm, and others via `statsmodels`).
- **Binomial Proportion Tests** — `BinomialStats` for single-sample binomial test with Wilson confidence intervals plus two-sample z-test and chi-square comparisons, and `MultipleBinomialTests` for batched binomial comparisons with multiple-testing correction.
- **Classical Hypothesis Tests** — unified dispatcher for Mann-Whitney U, Kruskal-Wallis, Kolmogorov-Smirnov, Mood's Median, Wilcoxon Rank-Sum, independent t-test, and chi-square.
- **Data I/O** — CSV reader supporting German-locale semicolon-delimited files, with wide→long table conversion.
- **Pretty Tables** — `write_pretty_table` helper that takes any results DataFrame and renders a publication-ready ASCII / Markdown table for inclusion in manuscripts and logs.
- **Combinatoric Resampling Utility** — `GetNofK` exhaustively enumerates n-of-k partitions for small-sample exact resampling.

### New in v0.2.0 — three application-oriented submodules

- **Case-crossover estimators** (`rerandomstats.case_crossover`) — time-stratified case-crossover conditional logit (Maclure 1991; Lee et al. 2023) with stratified-permutation backup, closed-form daylight-hours covariate, within-event temporal-contrast test (hot-day-vs-hot-week), and a Burke-2015 σ-rescaled effect translator for cross-study comparability. Promoted from ThermoStrife v0.1.1.
- **Model comparison** (`rerandomstats.model_comparison`) — two-sample Wald z-test on independently-estimated coefficients (`wald_two_sample_beta`), nested-model likelihood-ratio test (`likelihood_ratio_test`), single-method correction (`correct_pvalues`) and array helper (`correct_pvalues_array`), and dual-method BH + Bonferroni report (`benjamini_hochberg`). **All four correction helpers route through one `statsmodels.stats.multitest.multipletests` call** — the package's shared-algorithmic-source invariant prevents drift between BH implementations.
- **Dose-response and breakpoint analysis** (`rerandomstats.dose_response`) — broken-stick segmented regression with profile-RSS 95 % CI on the breakpoint (`broken_stick_fit`), the Davies (1987 / 2002) and Muggeo (2016) Pseudo-Score breakpoint-existence tests (`davies_test`, `pscore_test`), 4-parameter Hill / logistic fit with Sebaugh–McCray (2003) lower-bend point (`hill_fit`), and a per-subject iterator (`per_subject_segmented`) that applies any of the four fitters across a panel of subjects. Pickle-safe for `concurrent.futures.ProcessPoolExecutor` parallelism. Ported verbatim from the DigiMuh dairy-cow heat-stress pipeline.

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/zerotonin/rerandomstats.git
cd rerandomstats
pip install -e ".[dev]"
```

### Via conda environment

```bash
conda env create -f environment.yml
conda activate rerandomstats
pip install -e .
```

### Dependencies

Core: `numpy`, `scipy`, `pandas`, `statsmodels`, `prettytable`, `tqdm`

## Quick Start

### Two-sample Fisher resampling test

```python
from rerandomstats import FisherResamplingTest

# Compare two groups using median differences
test = FisherResamplingTest(
    data_a=[1.2, 3.4, 2.1, 4.5, 3.3],
    data_b=[5.6, 7.8, 6.5, 8.9, 7.2],
    func='medianDiff',
    combination_n=20_000,
)
p_value = test.main()
print(f"p = {p_value:.4f}")
```

### Multi-group pairwise comparisons with FDR correction

```python
import numpy as np
from rerandomstats import MultiGroupTest

data   = list(np.random.randn(30))
groups = ['control'] * 10 + ['treatment_A'] * 10 + ['treatment_B'] * 10

mgt = MultiGroupTest(
    data=data,
    group=groups,
    test='Fisher:medianDiff',
    combination_n=20_000,
    correction_type='fdr_bh',
)
results_df = mgt.main()
print(results_df)
```

### Fisher's exact test

```python
from rerandomstats import FisherExactTest

test = FisherExactTest(data_a=(8, 2), data_b=(1, 5))
print(f"p = {test.main():.4f}")
```

### Binomial proportion test

```python
from rerandomstats import BinomialStats

bs = BinomialStats(heads=73, total_flips=100)
result = bs.binomial_test(base_rate=0.5)
print(f"p = {result.pvalue:.4f}")
print(bs.exact_ci())
```

### Classical hypothesis tests via the unified interface

```python
from rerandomstats import HypothesisTests

ht = HypothesisTests(
    data_a=[1, 2, 3, 4, 5],
    data_b=[6, 7, 8, 9, 10],
    func='MannWhitneyU',
)
print(f"p = {ht.main():.4f}")
```

### Case-crossover conditional logit (v0.2.0)

```python
from rerandomstats import build_case_crossover_frame, case_crossover_conditional_logit

# events: list of dicts; each event has event_id, lat, lon, when (date),
# tmax_event_c (float), baseline (DataFrame index=date, column 'tmax').
frame = build_case_crossover_frame(events)
result = case_crossover_conditional_logit(frame)
print(f"OR per +1 °C: {result['or_per_C']:.3f} "
      f"(95% CI {result['or_ci95_low']:.3f}–{result['or_ci95_high']:.3f}), "
      f"p_one_sided = {result['pvalue_one_sided']:.4f}")
```

### Wald comparison of two independently-fitted coefficients (v0.2.0)

```python
from rerandomstats import wald_two_sample_beta

# E.g. ThermoFooty H7: hot-host pool β vs cool-host pool β
result = wald_two_sample_beta(
    beta_a=0.082, se_a=0.025,    # hot-host pool
    beta_b=0.041, se_b=0.022,    # cool-host pool
    alternative='two-sided',
    name_a='hot_host', name_b='cool_host',
)
print(f"Δβ = {result['diff']:+.4f}, z = {result['z_statistic']:.2f}, "
      f"p = {result['pvalue']:.4f}")
```

### Heterogeneous battery correction (BH + Bonferroni side-by-side, v0.2.0)

```python
from rerandomstats import benjamini_hochberg

# Pre-computed p-values from a heterogeneous battery (case-crossover,
# Wald, Poisson, LRT — anything that produced a p-value).
battery = {
    "H2": 0.0008, "H3": 0.012, "H4": 0.039,
    "H5": 0.001, "H0_spec": 0.0047,
}
result = benjamini_hochberg(battery, alpha=0.05)
for name, row in result["results"].items():
    print(f"{name}: raw p = {row['raw_p']:.4f}  "
          f"BH q = {row['bh_adjusted_p']:.4f}  BH-reject = {row['bh_reject']}")
```

### Breakpoint detection on dose-response data (v0.2.0)

```python
import numpy as np
from rerandomstats import broken_stick_fit, davies_test, pscore_test, hill_fit

x = ...  # predictor (e.g. THI, ambient temperature, anomaly)
y = ...  # response (e.g. core temperature, card rate)

# Primary: broken-stick segmented regression with profile-RSS CI on breakpoint
bs = broken_stick_fit(x, y)
if bs['converged']:
    print(f"Breakpoint = {bs['breakpoint']:.2f} "
          f"[{bs['breakpoint_ci_lo']:.2f}, {bs['breakpoint_ci_hi']:.2f}], "
          f"R² = {bs['r_squared']:.3f}")

# Existence tests (Davies upper-bound + Muggeo Pseudo-Score, typically more powerful)
print(f"Davies p = {davies_test(x, y)['pvalue']:.4f}")
print(f"Pscore p = {pscore_test(x, y)['pvalue']:.4f}")

# Rescue / alternative: 4-parameter Hill with Sebaugh–McCray lower bend
hf = hill_fit(x, y)
if hf['converged']:
    print(f"EC50 = {hf['ec50']:.2f}, Hill n = {hf['hill_n']:.2f}, "
          f"lower bend = {hf['lower_bend']:.2f}")
```

### Per-subject (per-animal / per-player / per-station) breakpoint distribution (v0.2.0)

```python
from rerandomstats import per_subject_segmented, broken_stick_fit

# df: long-format panel with one row per (subject, observation).
result_df = per_subject_segmented(
    df, subject_col="animal_id", x_col="thi", y_col="rumen_temp",
    model=broken_stick_fit, min_n=50,
)
# result_df has one row per subject with columns: animal_id + all keys
# from broken_stick_fit (breakpoint, slope_below, slope_above, …).
```

## Available Tests

| Family | Test String | Description |
|--------|------------|-------------|
| Fisher | `Fisher:medianDiff` | Resampling test — median difference |
| Fisher | `Fisher:meanDiff` | Resampling test — mean difference |
| Fisher | `Fisher:sumDiff` | Resampling test — sum difference |
| Fisher | `Fisher:exact` | Fisher's exact test (2×2 table) |
| Binomial | `Binomial:ztest` | Two-sample proportions z-test |
| Binomial | `Binomial:chi2` | Two-sample proportions chi-square |
| hypo | `hypo:MannWhitneyU` | Mann-Whitney U test |
| hypo | `hypo:KruskalWallis` | Kruskal-Wallis H test |
| hypo | `hypo:ChiSquare` | Chi-square goodness of fit |
| hypo | `hypo:Kolmogorov` | Kolmogorov-Smirnov test |
| hypo | `hypo:MoodMedian` | Mood's median test |
| hypo | `hypo:WilcoxonRankSum` | Wilcoxon rank-sum test |
| hypo | `hypo:IndependentT` | Independent samples t-test |

## Documentation

Full API documentation is built with Sphinx and hosted at:
**[https://zerotonin.github.io/reRandomStats/](https://zerotonin.github.io/reRandomStats/)**

To build locally:

```bash
cd docs
make html
open _build/html/index.html
```

## Running Tests

```bash
pytest
```

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/zerotonin/rerandomstats/issues).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this software in your research, please cite the **version
you used**. Citation metadata is in [`CITATION.cff`](CITATION.cff)
and on the GitHub repo's "Cite this repository" button.

```bibtex
@software{geurten_rerandomstats,
  author    = {Geurten, Bart R. H.},
  title     = {{reRandomStats}: Re-randomisation Statistics Toolkit},
  year      = {2025},
  version   = {0.1.0},
  url       = {https://github.com/zerotonin/rerandomstats},
  license   = {MIT},
  note      = {Zenodo DOI to appear on first GitHub Release; replace
               this block with the version-DOI you used.},
}
```

> **Note for Elsevier submissions:** Elsevier Editorial Manager does
> not parse `@software`. Convert to `@misc` at submission time per
> the lab BibTeX convention.

## Reproducing the analyses in published papers

This repository preserves **per-paper code snapshots as permanent
git tags** under the `paper-*` namespace, so any reader of the
associated paper can check out the exact code state that produced
the published results:

```bash
git clone https://github.com/zerotonin/rerandomstats.git
cd rerandomstats
git tag -l 'paper-*'                            # browse available snapshots
git checkout paper-Berger_Senthilan_2024        # e.g. for Berger & Senthilan (2024)
pip install -e .
```

Browse all snapshots at
[github.com/zerotonin/rerandomstats/tags](https://github.com/zerotonin/rerandomstats/tags).
Tags are created at the tip of the per-paper feature branch when the
paper is released; the branches are then removed to keep the active
branch list clean while the snapshot remains permanently citable.

## Used by

Downstream lab projects that depend on reRandomStats:

- **[ThermoStrife](https://github.com/zerotonin/thermostrife)**
  ([Zenodo DOI 10.5281/zenodo.20371612](https://doi.org/10.5281/zenodo.20371612)) —
  historical-uprisings temperature companion to the ThermoKourt
  *Drosophila* heat-aggression pipeline. Case-crossover conditional
  logit + stratified permutation + σ-rescaled effect machinery
  currently lives in `thermostrife.inference`; will migrate to
  `rerandomstats.case_crossover` from v0.2.0 onwards.
- **ThermoFooty** (pre-registered at
  [OSF DOI 10.17605/OSF.IO/YZVAK](https://doi.org/10.17605/OSF.IO/YZVAK),
  repo TBD) — pre-registered natural-experiment test of heat-aggression
  on European soccer. Will consume reRandomStats v0.2.0 as the
  canonical stats backend.

## Acknowledgements

We acknowledge [Sir Ronald Aylmer Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher) for his pioneering work on the re-randomisation test and his foundational contributions to the field of statistics.

## Maintainer

[Bart R.H. Geurten](https://orcid.org/0000-0002-1816-3241) — Department of Zoology, University of Otago, Dunedin, New Zealand.
