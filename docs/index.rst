reRandomStats Documentation
============================

.. image:: https://github.com/zerotonin/reRandomStats/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/zerotonin/reRandomStats/actions/workflows/tests.yml
   :alt: Tests

.. image:: https://github.com/zerotonin/reRandomStats/actions/workflows/docs.yml/badge.svg
   :target: https://zerotonin.github.io/reRandomStats/
   :alt: Docs

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

A comprehensive Python toolkit for **re-randomisation statistics** in
the tradition of Sir Ronald A. Fisher, extended in v0.2.0 with
time-stratified case-crossover estimators, model-comparison helpers
that share a single algorithmic source for multiple-comparisons
correction, and a dose-response / breakpoint-analysis toolkit
(broken-stick + Davies + Muggeo Pseudo-Score + 4-parameter Hill).

Features
--------

Core (v0.1.0+):

- **Fisher's Resampling Test** — flexible test statistics
  (mean / median / sum differences)
- **Fisher's Exact Test** — 2×2 contingency table
- **Multi-Group Pairwise Testing** — with automatic multiple-testing
  correction
- **Binomial Proportion Tests** — single- and two-sample variants
  with Wilson confidence intervals
- **Classical Hypothesis Tests** — unified dispatcher for
  Mann-Whitney U, Kruskal-Wallis, Kolmogorov-Smirnov, Mood's
  Median, Wilcoxon Rank-Sum, independent t-test, chi-square
- **Data I/O** — CSV reader with German-locale support and
  wide → long table conversion
- **Pretty-Table Output** — manuscript-ready ASCII / Markdown tables

New in v0.2.0 — three submodules:

- **case_crossover** — time-stratified case-crossover conditional
  logit (Maclure 1991, Lee et al. 2023) with stratified-permutation
  backup, daylight-hours covariate, within-event temporal-contrast
  test (hot-day-vs-hot-week), and Burke-2015 σ-rescaled effect
  translator
- **model_comparison** — two-sample Wald z-test on
  independently-estimated coefficients, nested-model likelihood-
  ratio test, single-method correction (`correct_pvalues`), array
  helper (`correct_pvalues_array`), and dual-method BH + Bonferroni
  report (`benjamini_hochberg`); all four correction helpers route
  through one `statsmodels.stats.multitest.multipletests` call
- **dose_response** — broken-stick segmented regression with
  profile-RSS 95 % CI on the breakpoint, Davies (1987 / 2002) and
  Muggeo (2016) Pseudo-Score tests for breakpoint existence,
  4-parameter Hill fit with Sebaugh–McCray (2003) lower-bend point,
  and a per-subject iterator that applies any of the four to a panel
  of subjects (pickle-safe for ``concurrent.futures.ProcessPoolExecutor``)

Quick Start
-----------

Core resampling test:

.. code-block:: python

   from rerandomstats import FisherResamplingTest

   test = FisherResamplingTest(
       data_a=[1.2, 3.4, 2.1, 4.5, 3.3],
       data_b=[5.6, 7.8, 6.5, 8.9, 7.2],
       func='medianDiff',
       combination_n=20_000,
   )
   p_value = test.main()
   print(f"p = {p_value:.4f}")

Multi-comparisons correction on a heterogeneous battery:

.. code-block:: python

   from rerandomstats import benjamini_hochberg

   # p-values from arbitrary tests (case-crossover, Wald, Poisson, LRT, …)
   result = benjamini_hochberg({
       "H1": 0.001, "H2": 0.012, "H3": 0.040, "H4": 0.080,
   })
   for name, row in result["results"].items():
       print(f"{name}: q = {row['bh_adjusted_p']:.4f}  "
             f"BH-reject = {row['bh_reject']}")

Breakpoint detection on dose-response data:

.. code-block:: python

   import numpy as np
   from rerandomstats import broken_stick_fit, davies_test, hill_fit

   x = np.random.uniform(10, 30, 300)
   y = np.where(x <= 22, 38 + 0.05 * x,
                38 + 0.05 * 22 + 0.8 * (x - 22))
   y += np.random.normal(0, 0.2, 300)

   bs = broken_stick_fit(x, y)
   print(f"Breakpoint = {bs['breakpoint']:.2f} "
         f"[{bs['breakpoint_ci_lo']:.2f}, {bs['breakpoint_ci_hi']:.2f}]")

   davies = davies_test(x, y)
   print(f"Davies p = {davies['pvalue']:.4f}")

   hill = hill_fit(x, y)
   print(f"EC50 = {hill['ec50']:.2f}, Hill n = {hill['hill_n']:.2f}, "
         f"lower bend = {hill['lower_bend']:.2f}")

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   api
   examples

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
