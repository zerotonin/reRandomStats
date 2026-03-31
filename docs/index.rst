reRandomStats Documentation
============================

.. image:: https://github.com/zerotonin/rerandomstats/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/zerotonin/rerandomstats/actions/workflows/tests.yml
   :alt: Tests

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

A comprehensive Python toolkit for **re-randomisation statistics** in the
tradition of Sir Ronald A. Fisher.

Features
--------

- Fisher's Resampling Test with flexible test statistics (mean, median, sum)
- Fisher's Exact Test for 2×2 contingency tables
- Multi-group pairwise comparisons with FDR correction
- Binomial proportion tests with Wilson confidence intervals
- Classical hypothesis test dispatcher (Mann-Whitney U, Kruskal-Wallis, etc.)
- CSV data I/O with German-locale support and wide→long conversion

Quick Start
-----------

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
