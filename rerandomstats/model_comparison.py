# ╔══════════════════════════════════════════════════════════════════╗
# ║  reRandomStats — model_comparison                                ║
# ║  « comparing fitted coefficients + correcting test batteries »   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Four tools for the after-the-fit stage of any analysis where    ║
# ║  multiple estimators are compared or multiple tests are pooled:  ║
# ║                                                                  ║
# ║    wald_two_sample_beta   — two-sample Wald z-test on the        ║
# ║                              difference between two independent  ║
# ║                              estimated coefficients              ║
# ║    likelihood_ratio_test  — LRT of a full vs reduced statsmodels ║
# ║                              fit (nested models)                 ║
# ║    correct_pvalues        — single-method correction of a dict   ║
# ║                              of pre-computed p-values; thin      ║
# ║                              wrapper around statsmodels.stats.   ║
# ║                              multipletests — the SHARED          ║
# ║                              ALGORITHMIC CORE for all multi-     ║
# ║                              comparison correction in the        ║
# ║                              package                             ║
# ║    benjamini_hochberg     — dual-method (BH + Bonferroni) report ║
# ║                              built on correct_pvalues; the       ║
# ║                              convenience API for the typical     ║
# ║                              "show both corrections side-by-     ║
# ║                              side" use case                      ║
# ║                                                                  ║
# ║  Design rule: every multi-comparison correction in reRandomStats ║
# ║  ultimately routes through statsmodels.stats.multipletests, so   ║
# ║  the actual correction math has exactly one implementation.      ║
# ║  Different I/O wrappers (dict-in/dict-out here, DataFrame-in/    ║
# ║  DataFrame-out in MultiGroupTest) are fine; the algorithm is     ║
# ║  not duplicated.                                                 ║
# ║                                                                  ║
# ║  Generalised from the ThermoStrife pipeline (Geurten 2026,       ║
# ║  ThermoStrife v0.1.1, DOI 10.5281/zenodo.20371612).              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Comparing fitted coefficients and correcting heterogeneous test batteries.

The four functions in this module operate on already-fitted models or
already-computed p-values — they do NOT take raw data.  For raw-data
workflows that combine data ingestion, pairwise testing, and
correction into one call, use :class:`rerandomstats.MultiGroupTest`.

This separation matters when the test battery is heterogeneous (e.g.
a case-crossover conditional logit, a Poisson IRR, a Wald two-sample-β
contrast, and a likelihood-ratio test, all in the same auxiliary
battery) and the corresponding p-values cannot all be produced by a
single pairwise-comparisons pipeline.  In that case the caller runs
each test individually, collects the p-values into a dict, and passes
the dict to :func:`correct_pvalues` (for a single method) or
:func:`benjamini_hochberg` (for the BH + Bonferroni dual-method
report) for correction.

**Algorithmic-source guarantee.** Both :func:`correct_pvalues` and
:func:`benjamini_hochberg` delegate the underlying correction
arithmetic to ``statsmodels.stats.multipletests``; the same call is
used internally by :class:`rerandomstats.MultiGroupTest`.  There is
no parallel implementation of BH / Bonferroni / Holm / Sidak / FDR-BY
in this package — every code path that adjusts p-values converges
on statsmodels' canonical implementation.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import statsmodels.stats.multitest as sm_multitest
from scipy import stats

# ┌────────────────────────────────────────────────────────────┐
# │ Wald two-sample test on two estimated coefficients         │
# └────────────────────────────────────────────────────────────┘


def wald_two_sample_beta(
    beta_a: float,
    se_a: float,
    beta_b: float,
    se_b: float,
    *,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    name_a: str = "a",
    name_b: str = "b",
) -> dict:
    """Wald z-test on the difference between two estimated coefficients.

    Tests ``H0: β_a = β_b`` (or directional variants) using

        z = (β_a − β_b) / sqrt(SE(β_a)² + SE(β_b)²)

    This is the right test when you have two independently-fitted
    estimates of related coefficients — e.g. a case-crossover OR
    fitted on a hot-host subsample vs the same estimator fitted on a
    cool-host subsample, or a Poisson IRR for an aggression-set
    outcome vs the same estimator on a non-aggression outcome.  The
    formula assumes the two estimates are independent (no overlap in
    the data they were fit on); for **nested** models on the **same**
    data, use :func:`likelihood_ratio_test` instead.

    Args:
        beta_a:      Estimated coefficient from subsample / model A.
        se_a:        Standard error of ``beta_a``.
        beta_b:      Estimated coefficient from subsample / model B.
        se_b:        Standard error of ``beta_b``.
        alternative: ``"two-sided"`` (default) tests ``β_a ≠ β_b``.
                     ``"greater"`` tests ``β_a > β_b``.
                     ``"less"`` tests ``β_a < β_b``.
        name_a:      Optional label for the A side, surfaced in the
                     output dict.  Defaults to ``"a"``.
        name_b:      Optional label for the B side.  Defaults to ``"b"``.

    Returns:
        Dict with the input estimates, the difference and its SE, the
        Wald z statistic, the requested-tail p-value, and a 95 %
        confidence interval on the difference.

    Raises:
        ValueError: if ``alternative`` is not one of the three accepted
                    strings.
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less'; got {alternative!r}"
        )

    diff = float(beta_a - beta_b)
    se_diff = float(math.sqrt(se_a**2 + se_b**2))
    z = diff / se_diff if se_diff > 0 else float("nan")

    if math.isnan(z):
        p = float("nan")
    elif alternative == "two-sided":
        p = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    elif alternative == "greater":
        p = float(1.0 - stats.norm.cdf(z))
    else:  # "less"
        p = float(stats.norm.cdf(z))

    return {
        "name_a": name_a,
        "beta_a": float(beta_a),
        "se_a": float(se_a),
        "name_b": name_b,
        "beta_b": float(beta_b),
        "se_b": float(se_b),
        "diff": diff,
        "se_diff": se_diff,
        "z_statistic": float(z),
        "alternative": alternative,
        "pvalue": p,
        "ci95_low_diff": diff - 1.96 * se_diff,
        "ci95_high_diff": diff + 1.96 * se_diff,
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Likelihood-ratio test on nested models                     │
# └────────────────────────────────────────────────────────────┘


def likelihood_ratio_test(
    model_full: Any,
    model_reduced: Any,
) -> dict:
    """Likelihood-ratio test of a full vs reduced nested fit.

    Computes::

        LR = 2 · (log L_full − log L_reduced)
        p  = 1 − F_chi²(LR; df_full − df_reduced)

    Both arguments may be any objects exposing ``.llf`` (log-likelihood)
    and ``.df_model`` (number of estimated parameters) attributes —
    the convention used by ``statsmodels`` result objects, which
    covers ``ConditionalLogit``, ``Poisson``, ``Logit``, ``OLS``, and
    most others.

    Use this for **nested** model comparisons on the **same** data:
    e.g. testing whether adding an interaction term to a
    case-crossover model significantly improves fit, or whether a
    tournament-edition × anomaly interaction is needed beyond the
    main-effect model.  For comparing two estimates of the SAME
    coefficient fitted on DIFFERENT data, use
    :func:`wald_two_sample_beta` instead.

    Args:
        model_full:    Fitted result object for the larger (nesting) model.
        model_reduced: Fitted result object for the smaller (nested) model.

    Returns:
        Dict with the LR statistic, the degrees-of-freedom difference,
        the chi² p-value, and both log-likelihoods.

    Raises:
        ValueError: if ``model_full`` does not have strictly more
                    parameters than ``model_reduced``.
    """
    llf_full = float(model_full.llf)
    llf_red = float(model_reduced.llf)
    df_diff = int(model_full.df_model - model_reduced.df_model)
    if df_diff <= 0:
        raise ValueError(
            f"model_full must have more parameters than model_reduced; "
            f"got df_full={model_full.df_model}, df_reduced={model_reduced.df_model}"
        )

    lr_stat = 2.0 * (llf_full - llf_red)
    p = float(1.0 - stats.chi2.cdf(lr_stat, df_diff))

    return {
        "lr_statistic": float(lr_stat),
        "df": df_diff,
        "pvalue": p,
        "log_likelihood_full": llf_full,
        "log_likelihood_reduced": llf_red,
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Single-method correction  « shared algorithmic core »      │
# └────────────────────────────────────────────────────────────┘


def correct_pvalues(
    pvalues: dict[str, float],
    *,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> dict:
    """Adjust a heterogeneous battery of p-values via one correction method.

    Thin wrapper around ``statsmodels.stats.multitest.multipletests``.
    This is the **shared algorithmic core** for every multi-comparison
    correction in reRandomStats: both :func:`benjamini_hochberg` (the
    dual-method convenience helper) and the
    :class:`rerandomstats.MultiGroupTest` pipeline ultimately route
    through the same statsmodels call.  No parallel implementation of
    the underlying math exists in this package.

    Use this directly when you want a single correction method (e.g.
    just BH FDR or just Holm) and need a tidy per-test result dict
    rather than DataFrame columns.

    Args:
        pvalues: Dict mapping test name → raw p-value.  Keys are
                 carried verbatim through to the output ``results``
                 dict.  Insertion order is preserved.
        alpha:   Family-wise significance level.  Default 0.05.
        method:  Any method accepted by
                 ``statsmodels.stats.multitest.multipletests``: e.g.
                 ``"fdr_bh"`` (Benjamini–Hochberg step-up, the
                 default), ``"fdr_by"`` (Benjamini–Yekutieli),
                 ``"bonferroni"``, ``"holm"``, ``"hommel"``,
                 ``"sidak"``, ``"holm-sidak"``, ``"simes-hochberg"``,
                 ``"hommel"``, ``"fdr_tsbh"``, ``"fdr_tsbky"``.

    Returns:
        Dict with top-level keys ``family_size``, ``alpha``,
        ``method``, ``n_rejected``, and ``results``.  ``results`` is
        a per-test dict carrying ``raw_p``, ``rank`` (1 = smallest
        raw p-value), ``adjusted_p``, and ``reject`` (bool).  Empty
        input returns ``{"family_size": 0, "alpha": alpha, "method":
        method, "n_rejected": 0, "results": {}}``.
    """
    items = list(pvalues.items())
    k = len(items)
    if k == 0:
        return {
            "family_size": 0, "alpha": alpha, "method": method,
            "n_rejected": 0, "results": {},
        }

    names = [n for n, _ in items]
    raw_ps = np.array([p for _, p in items], dtype=float)

    # The single algorithmic source for the correction math.
    reject, adjusted, _, _ = sm_multitest.multipletests(
        raw_ps, alpha=alpha, method=method,
    )

    # Rank in raw-p ascending order (1 = smallest).  Used for
    # downstream callers that want to enumerate tests in increasing
    # p-order without re-sorting.
    sort_idx = np.argsort(raw_ps, kind="stable")
    ranks = np.empty(k, dtype=int)
    ranks[sort_idx] = np.arange(1, k + 1)

    results = {}
    for i, name in enumerate(names):
        results[name] = {
            "raw_p": float(raw_ps[i]),
            "rank": int(ranks[i]),
            "adjusted_p": float(adjusted[i]),
            "reject": bool(reject[i]),
        }
    return {
        "family_size": k,
        "alpha": alpha,
        "method": method,
        "n_rejected": int(reject.sum()),
        "results": results,
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Dual-method (BH + Bonferroni) battery report               │
# └────────────────────────────────────────────────────────────┘


def benjamini_hochberg(
    pvalues: dict[str, float],
    alpha: float = 0.05,
) -> dict:
    """Dual-method BH + Bonferroni report for a heterogeneous battery.

    Convenience wrapper around :func:`correct_pvalues`: runs the
    correction twice (``method="fdr_bh"`` and ``method="bonferroni"``)
    and merges the per-test results so reports can show BH q-values
    AND Bonferroni-adjusted p-values side-by-side.  The BH FDR is
    typically the primary (less conservative for correlated batteries)
    with Bonferroni as a conservative reference.

    Both corrections are computed via the same
    ``statsmodels.stats.multitest.multipletests`` call (one per
    method), so the underlying math has exactly one implementation.

    Args:
        pvalues: Dict mapping test name → raw p-value.
        alpha:   Target false-discovery rate.  Default 0.05.

    Returns:
        Dict with per-test ``raw_p``, ``rank``, ``bh_adjusted_p``,
        ``bh_threshold`` (the per-rank BH cut ``(rank/k)·alpha``),
        ``bh_reject``, ``bonferroni_adjusted_p``, and
        ``bonferroni_reject``.  Top-level keys also expose
        ``family_size``, ``alpha``, ``bonferroni_threshold`` (the
        per-test alpha/k cut), ``bh_cutoff_rank`` (largest rank that
        passes BH), and ``n_bh_rejected`` / ``n_bonferroni_rejected``
        counts.  Empty input returns
        ``{"family_size": 0, "alpha": alpha, "results": {}}``.
    """
    bh = correct_pvalues(pvalues, alpha=alpha, method="fdr_bh")
    if bh["family_size"] == 0:
        return {"family_size": 0, "alpha": alpha, "results": {}}

    bf = correct_pvalues(pvalues, alpha=alpha, method="bonferroni")
    k = bh["family_size"]
    bonf_threshold = alpha / k

    results: dict[str, dict] = {}
    for name in bh["results"]:
        bh_r = bh["results"][name]
        bf_r = bf["results"][name]
        rank = bh_r["rank"]
        results[name] = {
            "raw_p": bh_r["raw_p"],
            "rank": rank,
            "bh_adjusted_p": bh_r["adjusted_p"],
            "bh_threshold": float(rank / k * alpha),
            "bh_reject": bh_r["reject"],
            "bonferroni_adjusted_p": bf_r["adjusted_p"],
            "bonferroni_reject": bf_r["reject"],
        }
    n_bh_rejected = bh["n_rejected"]
    return {
        "family_size": k,
        "alpha": alpha,
        "bonferroni_threshold": float(bonf_threshold),
        "bh_cutoff_rank": n_bh_rejected,
        "n_bh_rejected": n_bh_rejected,
        "n_bonferroni_rejected": bf["n_rejected"],
        "results": results,
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Array-in / array-out BH helper                             │
# │ « drop-in replacement for project-specific NumPy BH »      │
# └────────────────────────────────────────────────────────────┘


def correct_pvalues_array(
    pvalues: np.ndarray,
    *,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> np.ndarray:
    """Adjust a NumPy array of p-values, preserving NaN entries.

    Array-in / array-out wrapper around the same shared
    ``statsmodels.stats.multitest.multipletests`` call that
    :func:`correct_pvalues` uses.  Designed as a one-line drop-in
    replacement for project-specific BH/Bonferroni helpers that take
    and return arrays — e.g. ``digimuh.stats_core.benjamini_hochberg``
    — so refactors converging on reRandomStats are a single import
    change.

    NaN entries in the input are preserved at the same positions in
    the output and are excluded from the correction (the family size
    used by statsmodels is the count of finite p-values, matching the
    DigiMuh convention).

    Args:
        pvalues: 1-D array of raw p-values.  May contain NaN.
        alpha:   Family-wise significance level.  Default 0.05.
        method:  Any method accepted by
                 ``statsmodels.stats.multitest.multipletests``.
                 Default ``"fdr_bh"``.

    Returns:
        Array of adjusted p-values, same shape and dtype as input,
        with NaN entries preserved.  Empty input returns an empty
        array.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p

    notnan = ~np.isnan(p)
    p_valid = p[notnan]
    if p_valid.size == 0:
        return p.copy()

    _reject, adjusted, _, _ = sm_multitest.multipletests(
        p_valid, alpha=alpha, method=method,
    )

    out = np.full(n, np.nan)
    out[notnan] = adjusted
    return out


__all__ = [
    "wald_two_sample_beta",
    "likelihood_ratio_test",
    "correct_pvalues",
    "correct_pvalues_array",
    "benjamini_hochberg",
]
