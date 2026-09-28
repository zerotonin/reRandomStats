API Reference
=============

Fisher's Exact Test
-------------------

.. automodule:: rerandomstats.fisher_exact
   :members:
   :undoc-members:
   :show-inheritance:

Fisher's Resampling Test
------------------------

.. automodule:: rerandomstats.fisher_resampling
   :members:
   :undoc-members:
   :show-inheritance:

Hypothesis Tests
----------------

.. automodule:: rerandomstats.hypothesis_tests
   :members:
   :undoc-members:
   :show-inheritance:

Binomial Statistics
-------------------

.. automodule:: rerandomstats.binomial_stats
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Group Testing
-------------------

.. automodule:: rerandomstats.multi_group_test
   :members:
   :undoc-members:
   :show-inheritance:

Resampling for Partially Paired Groups (v0.4.0)
-----------------------------------------------

Group-level resampling test for two groups that share some of their
subjects, and its pairwise multi-group counterpart with multiplicity
correction.  The statistic is the difference of the group means,
medians or sums; the null distribution swaps the two values of a shared
subject together and shuffles unshared subjects between the groups.

The rearrangement scheme is that of Einsporn & Habtzghi (2013); the
statistic is not theirs (they weight a paired and an unpaired mean
difference), it is the plain difference between the two groups.  The
p-value is computed as in Phipson & Smyth (2010), and the default
multiplicity correction is that of Benjamini & Hochberg (1995).  Full
references are given in the module documentation below.

.. automodule:: rerandomstats.paired_resampling
   :members:
   :undoc-members:
   :show-inheritance:

Case-Crossover Estimators (v0.2.0)
----------------------------------

Time-stratified case-crossover conditional logit with stratified-
permutation backup and Burke-2015 σ-rescaled effect translator.

.. automodule:: rerandomstats.case_crossover
   :members:
   :undoc-members:
   :show-inheritance:

Model Comparison (v0.2.0)
-------------------------

Wald two-sample-β test, nested-model likelihood-ratio test, and the
shared-algorithmic-source p-value correction helpers
(``correct_pvalues`` / ``correct_pvalues_array`` /
``benjamini_hochberg``) routing through ``statsmodels.stats.multitest.multipletests``.

.. automodule:: rerandomstats.model_comparison
   :members:
   :undoc-members:
   :show-inheritance:

Dose-Response and Breakpoint Analysis (v0.2.0)
----------------------------------------------

Broken-stick segmented regression with profile-RSS 95 % CI on the
breakpoint, the Davies (1987 / 2002) and Muggeo (2016) Pseudo-Score
breakpoint-existence tests, the 4-parameter Hill / logistic fit with
Sebaugh–McCray lower-bend point, and a per-subject iterator that
applies any of the four to a panel of subjects.

.. automodule:: rerandomstats.dose_response
   :members:
   :undoc-members:
   :show-inheritance:

Resampling Index Generator
--------------------------

.. automodule:: rerandomstats.resample_n_of_k
   :members:
   :undoc-members:
   :show-inheritance:

Data I/O
--------

.. automodule:: rerandomstats.data_io
   :members:
   :undoc-members:
   :show-inheritance:

Pretty Table Output
-------------------

.. automodule:: rerandomstats.pretty_table
   :members:
   :undoc-members:
   :show-inheritance:
