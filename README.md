# reRandomStats

[![Tests](https://github.com/zerotonin/rerandomstats/actions/workflows/tests.yml/badge.svg)](https://github.com/zerotonin/rerandomstats/actions/workflows/tests.yml)
[![Docs](https://github.com/zerotonin/rerandomstats/actions/workflows/docs.yml/badge.svg)](https://zerotonin.github.io/rerandomstats/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

```
╔══════════════════════════════════════════════════════════════════╗
║  reRandomStats                                                   ║
║  « Re-randomisation statistics in the spirit of Fisher »         ║
╚══════════════════════════════════════════════════════════════════╝
```

A comprehensive Python toolkit for **re-randomisation statistics** in the tradition of [Sir Ronald A. Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher). The package provides Fisher's resampling test with flexible test statistics, pairwise multi-group comparisons with multiple-testing correction, binomial proportion tests, and a unified interface to classical parametric and non-parametric hypothesis tests.

## Features

- **Fisher's Resampling Test** — permutation-based two-sample test using mean, median, or sum differences as the test statistic. Supports exhaustive enumeration for small samples and random resampling for large ones.
- **Fisher's Exact Test** — wrapper for 2×2 contingency table analysis.
- **Multi-Group Pairwise Testing** — runs all (or user-specified) pairwise comparisons with automatic multiple-testing correction (Benjamini-Hochberg FDR, Bonferroni, Holm, and others via `statsmodels`).
- **Binomial Proportion Tests** — single-sample binomial test with Wilson confidence intervals, plus two-sample z-test and chi-square comparisons.
- **Classical Hypothesis Tests** — unified dispatcher for Mann-Whitney U, Kruskal-Wallis, Kolmogorov-Smirnov, Mood's Median, Wilcoxon Rank-Sum, independent t-test, and chi-square.
- **Data I/O** — CSV reader supporting German-locale semicolon-delimited files, with wide→long table conversion.

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
**[https://zerotonin.github.io/rerandomstats/](https://zerotonin.github.io/rerandomstats/)**

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

If you use this software in your research, please cite it:

```bibtex
@software{geurten_rerandomstats,
  author    = {Geurten, Bart R.H.},
  title     = {{reRandomStats}: Re-randomisation Statistics Toolkit},
  url       = {https://github.com/zerotonin/rerandomstats},
  license   = {MIT},
}
```

## Acknowledgements

We acknowledge [Sir Ronald Aylmer Fisher](https://en.wikipedia.org/wiki/Ronald_Fisher) for his pioneering work on the re-randomisation test and his foundational contributions to the field of statistics.

## Maintainer

[Bart R.H. Geurten](https://orcid.org/0000-0002-1816-3241) — Department of Zoology, University of Otago, Dunedin, New Zealand.
