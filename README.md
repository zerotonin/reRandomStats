# reRandomStats

[![Tests](https://github.com/zerotonin/rerandomstats/actions/workflows/tests.yml/badge.svg)](https://github.com/zerotonin/rerandomstats/actions/workflows/tests.yml)
[![Docs](https://github.com/zerotonin/rerandomstats/actions/workflows/docs.yml/badge.svg)](https://zerotonin.github.io/rerandomstats/)
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

- **Fisher's Resampling Test** — permutation-based two-sample test using mean, median, or sum differences as the test statistic. Supports exhaustive enumeration for small samples and random resampling for large ones.
- **Fisher's Exact Test** — wrapper for 2×2 contingency table analysis.
- **Multi-Group Pairwise Testing** — runs all (or user-specified) pairwise comparisons with automatic multiple-testing correction (Benjamini-Hochberg FDR, Bonferroni, Holm, and others via `statsmodels`).
- **Binomial Proportion Tests** — `BinomialStats` for single-sample binomial test with Wilson confidence intervals plus two-sample z-test and chi-square comparisons, and `MultipleBinomialTests` for batched binomial comparisons with multiple-testing correction.
- **Classical Hypothesis Tests** — unified dispatcher for Mann-Whitney U, Kruskal-Wallis, Kolmogorov-Smirnov, Mood's Median, Wilcoxon Rank-Sum, independent t-test, and chi-square.
- **Data I/O** — CSV reader supporting German-locale semicolon-delimited files, with wide→long table conversion.
- **Pretty Tables** — `write_pretty_table` helper that takes any results DataFrame and renders a publication-ready ASCII / Markdown table for inclusion in manuscripts and logs.
- **Combinatoric Resampling Utility** — `GetNofK` exhaustively enumerates n-of-k partitions for small-sample exact resampling.

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
