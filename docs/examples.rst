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
