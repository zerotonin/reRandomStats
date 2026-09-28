Examples
========

Two-Sample Resampling Test
--------------------------

Compare two independent groups using Fisher's resampling test with the median
difference as the test statistic:

.. code-block:: python

   from rerandomstats import FisherResamplingTest

   control   = [2.1, 3.5, 1.8, 4.2, 3.0, 2.7]
   treatment = [5.4, 6.1, 7.3, 5.9, 6.8, 7.0]

   test = FisherResamplingTest(control, treatment, 'medianDiff', 20_000)
   p = test.main()
   print(f"Median difference p-value: {p:.4f}")

Multi-Group Comparison with FDR Correction
------------------------------------------

Run all pairwise comparisons across three genotypes and correct for multiple
testing using the Benjamini-Hochberg procedure:

.. code-block:: python

   import numpy as np
   from rerandomstats import MultiGroupTest

   np.random.seed(42)
   data   = list(np.concatenate([
       np.random.normal(0, 1, 15),
       np.random.normal(3, 1, 15),
       np.random.normal(6, 1, 15),
   ]))
   groups = ['wildtype'] * 15 + ['mutant_A'] * 15 + ['mutant_B'] * 15

   mgt = MultiGroupTest(data, groups, 'Fisher:medianDiff', 20_000)
   result = mgt.main()
   print(result.to_string(index=False))

Groups That Share Subjects
--------------------------

When the same subjects appear in more than one group — a herd followed
over several summers, a cohort before and after — the two values of a
repeat subject are correlated, and shuffling every value freely between
the groups is not a valid null distribution.  Pass the subject
identifiers: the question stays at group level (here, the difference of
the two medians), while repeat subjects are kept paired.

.. code-block:: python

   import numpy as np
   from rerandomstats import MultiGroupPairedTest

   rng = np.random.default_rng(1)
   level = rng.normal(75, 3, 40)                  # each cow's own level

   values, years, cows = [], [], []
   for year, shift in (('2021', 0.0), ('2022', 0.0), ('2023', 1.5)):
       # 40 cows present every year ...
       values += list(level + shift + rng.normal(0, 0.5, 40))
       years  += [year] * 40
       cows   += [f'cow{i}' for i in range(40)]
       # ... and 15 cows seen in this year only
       values += list(rng.normal(75 + shift, 3, 15))
       years  += [year] * 15
       cows   += [f'{year}-only{i}' for i in range(15)]

   table = MultiGroupPairedTest(
       data=values, group=years, subject=cows,
       func='medianDiff', combination_n=20_000,
       correction_type='fdr_bh', seed=1,
   ).main()
   print(table[['groupA', 'groupB', 'n_paired', 'n_only_A', 'n_only_B',
                'statistic', 'p value corrected', 'sig. level']]
         .to_string(index=False))

For a single comparison use ``PartiallyPairedResamplingTest`` with
``data_a``, ``data_b``, ``subject_a`` and ``subject_b``.

In each rearrangement a cow present in both years swaps her two values
with probability one half, and the cows present in one year only are
re-partitioned at random between the two years.  This is the scheme of
Einsporn & Habtzghi (2013).  Their statistic weights a paired and an
unpaired mean difference; the statistic here is the plain difference
between the two groups.

References:

- Einsporn RL, Habtzghi D (2013) Combining paired and two-sample data
  using a permutation test.  *Journal of Data Science* 11(4):767–779.
  `doi:10.6339/JDS.2013.11(4).1164 <https://doi.org/10.6339/JDS.2013.11(4).1164>`_
- Phipson B, Smyth GK (2010) Permutation P-values should never be zero:
  calculating exact P-values when permutations are randomly drawn.
  *Statistical Applications in Genetics and Molecular Biology* 9(1).
  `doi:10.2202/1544-6115.1585 <https://doi.org/10.2202/1544-6115.1585>`_
- Benjamini Y, Hochberg Y (1995) Controlling the false discovery rate: a
  practical and powerful approach to multiple testing.  *Journal of the
  Royal Statistical Society Series B* 57(1):289–300.
  `doi:10.1111/j.2517-6161.1995.tb02031.x <https://doi.org/10.1111/j.2517-6161.1995.tb02031.x>`_
- Derrick B, White P (2022) Review of the partially overlapping samples
  framework: paired observations and independent observations in two
  samples.  *The Quantitative Methods for Psychology* 18(1):55–65.
  `doi:10.20982/tqmp.18.1.p055 <https://doi.org/10.20982/tqmp.18.1.p055>`_

Related permutation approaches with other statistics: Amro & Pauly
(2017, *Journal of Statistical Computation and Simulation*
87(6):1148–1159), Amro, Konietschke & Pauly (2019, *Statistics in
Medicine* 38(17):3243–3255) and Johnson & Richter (2022, *Computational
Statistics* 37(2):739–750).

Fisher's Exact Test
-------------------

Test whether survival differs between treated and control groups:

.. code-block:: python

   from rerandomstats import FisherExactTest

   #             (alive, dead)
   treated = (45, 5)
   control = (30, 20)

   test = FisherExactTest(treated, control)
   print(f"p = {test.main():.4f}")

Binomial Proportion with Confidence Interval
---------------------------------------------

Test whether an observed proportion differs from a base rate and compute
the Wilson confidence interval:

.. code-block:: python

   from rerandomstats import BinomialStats

   bs = BinomialStats(heads=73, total_flips=100)
   result = bs.binomial_test(base_rate=0.5)
   print(f"Binomial test p = {result.pvalue:.4f}")
   print(f"Wilson CI: {bs.exact_ci()}")

Variance Ratio Permutation Test
-------------------------------

A standalone permutation test for comparing variances (as used in
bouton vs inter-bouton analysis):

.. code-block:: python

   import numpy as np

   inter_bouton = np.array([-15.86, -17.54, -4.08, -0.48, 7.21, 12.97, 15.38, 18.02])
   bouton       = np.array([-74.23, -73.51, -12.73, 13.45, 16.10, 21.86, 28.35, 83.36])

   observed_ratio = np.var(bouton, ddof=1) / np.var(inter_bouton, ddof=1)

   np.random.seed(42)
   pooled = np.concatenate([inter_bouton, bouton])
   n = len(inter_bouton)
   ratios = []
   for _ in range(100_000):
       np.random.shuffle(pooled)
       v_inter = np.var(pooled[:n], ddof=1)
       v_bout  = np.var(pooled[n:], ddof=1)
       if v_inter > 0:
           ratios.append(v_bout / v_inter)

   p = np.mean(np.array(ratios) >= observed_ratio)
   print(f"Variance ratio = {observed_ratio:.2f}, p = {p:.6f}")
