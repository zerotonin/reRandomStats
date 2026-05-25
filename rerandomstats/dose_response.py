# ╔══════════════════════════════════════════════════════════════════╗
# ║  reRandomStats — dose_response                                   ║
# ║  « breakpoint + sigmoid dose-response fitting »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Threshold-detection toolkit for one-predictor / one-response    ║
# ║  data where the relationship is suspected to be non-linear:      ║
# ║                                                                  ║
# ║    broken_stick_fit  — two-segment piecewise linear fit, with    ║
# ║                         profile-RSS 95 % CI on the breakpoint    ║
# ║                         (Muggeo 2003 segmented regression)       ║
# ║    davies_test       — Davies (1987 / 2002) test for breakpoint  ║
# ║                         existence                                ║
# ║    pscore_test       — Muggeo (2016) Pseudo-Score test for       ║
# ║                         breakpoint existence; typically more     ║
# ║                         powerful than Davies for a single        ║
# ║                         changepoint                              ║
# ║    hill_fit          — 4-parameter logistic / Hill dose-response ║
# ║                         fit (Hill 1910; Goutelle et al. 2008)    ║
# ║                         with Sebaugh & McCray (2003) lower-bend  ║
# ║                         point as a sigmoid-equivalent of the     ║
# ║                         broken-stick breakpoint                  ║
# ║    per_subject_segmented — iterate any of the above fitters      ║
# ║                         over a panel of subjects; pickle-safe    ║
# ║                         for ProcessPoolExecutor parallelism      ║
# ║                                                                  ║
# ║  Generalised from the DigiMuh dairy-cow heat-stress pipeline.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Breakpoint and sigmoid dose-response fitting for two-variable data.

All five functions in this module take ``x`` / ``y`` arrays (or a
DataFrame for :func:`per_subject_segmented`) and return either a
``dict`` (the four primary fitters) or a ``DataFrame`` (the per-subject
iterator).  No I/O, no side effects, no project-specific assumptions.

**Skip-marker convention.** When the input is too small to fit
(``n < 30`` for the primary fitters, ``n < min_n`` for the per-subject
iterator), the result dict carries ``converged=False`` and ``n=<actual
sample size>``, plus the primary parameter (e.g. ``breakpoint`` or
``ec50``) as ``np.nan``.  This is the DigiMuh-derived convention;
callers should test ``converged`` before accessing fit-specific fields.

**References.**

- Davies RB (1987) Biometrika 74:33-43.
- Davies RB (2002) Biometrika 89:484-489.
- Goutelle S, Maurin M, Rougier F, Barbaut X, Bourguignon L, Ducher M,
  Maire P (2008) Fundamental & Clinical Pharmacology 22:633-648.
- Hill AV (1910) Journal of Physiology 40:iv-vii.
- Muggeo VMR (2003) Statistics in Medicine 22:3055-3071.
- Muggeo VMR (2016) Journal of Statistical Computation and Simulation
  86:3059-3067.
- Sebaugh JL, McCray PD (2003) Pharmaceutical Statistics 2:167-174.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

log = logging.getLogger("rerandomstats.dose_response")


# ┌────────────────────────────────────────────────────────────┐
# │ Broken-stick (segmented) regression with breakpoint CI     │
# └────────────────────────────────────────────────────────────┘


def broken_stick_fit(
    x: np.ndarray,
    y: np.ndarray,
    x_range: tuple[float, float] | None = None,
    n_grid: int = 200,
) -> dict:
    """Fit a two-segment piecewise-linear model with one breakpoint.

    Model: ``y = α_below + β_below·x`` for ``x ≤ ψ``; ``y = α_above +
    β_above·x`` for ``x > ψ``.  Biological constraint enforced:
    ``slope_above > slope_below`` and ``slope_above > 0`` (the
    "flat-ish below, steeper above" pattern typical of threshold
    physiological responses).  Fits that violate this constraint are
    returned with ``converged=False`` and a ``rejected_reason`` string.

    The breakpoint is located by grid search over ``n_grid`` candidates
    in ``x_range``, refined by ``scipy.optimize.minimize_scalar`` on
    the residual sum of squares.  A **profile-RSS 95 % confidence
    interval** on the breakpoint is computed from the grid-evaluated
    RSS profile using the F-distribution threshold for a
    one-parameter constraint::

        CI = {ψ : RSS(ψ) ≤ RSS_min · (1 + F_{1, df_resid; 0.95} / df_resid)}

    Threshold crossings are linearly interpolated between adjacent
    grid points; CIs that hit the search bounds are flagged with
    ``breakpoint_ci_truncated=True``.

    Args:
        x:       Predictor array.  NaN entries dropped.
        y:       Response array.  Must align with ``x``.
        x_range: Search bounds for the breakpoint.  Default
                 ``(percentile_10, percentile_90)`` of ``x``.
        n_grid:  Grid resolution for the initial breakpoint search.
                 Default 200.

    Returns:
        Dict with the breakpoint, both segment slopes and intercepts,
        ``r_squared``, ``n``, ``n_below``, ``n_above``, ``converged``,
        ``breakpoint_ci_lo``, ``breakpoint_ci_hi``, ``breakpoint_se``
        (derived from half-CI-width / 1.96), ``breakpoint_ci_truncated``,
        and ``rejected_reason`` (``None`` if the fit converged).

    References:
        Muggeo VMR (2003) Estimating regression models with unknown
        break-points. Statistics in Medicine 22:3055-3071.
    """
    from scipy.stats import f as _f_dist

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 30:
        return {"breakpoint": np.nan, "converged": False, "n": n}

    if x_range is None:
        x_range = (np.percentile(x, 10), np.percentile(x, 90))
    if x_range[0] >= x_range[1]:
        return {"breakpoint": np.nan, "converged": False, "n": n}

    def rss_at_bp(bp: float) -> float:
        left, right = x <= bp, x > bp
        if left.sum() < 10 or right.sum() < 10:
            return np.inf
        try:
            Xl = np.column_stack([np.ones(left.sum()), x[left]])
            cl, *_ = np.linalg.lstsq(Xl, y[left], rcond=None)
            Xr = np.column_stack([np.ones(right.sum()), x[right]])
            cr, *_ = np.linalg.lstsq(Xr, y[right], rcond=None)
        except np.linalg.LinAlgError:
            return np.inf
        return np.sum((y[left] - Xl @ cl) ** 2) + np.sum((y[right] - Xr @ cr) ** 2)

    grid = np.linspace(x_range[0], x_range[1], n_grid)
    rss_vals = np.array([rss_at_bp(bp) for bp in grid])
    best_idx = int(np.argmin(rss_vals))
    if not np.isfinite(rss_vals[best_idx]):
        return {"breakpoint": np.nan, "converged": False, "n": n}

    lo = grid[max(0, best_idx - 2)]
    hi = grid[min(len(grid) - 1, best_idx + 2)]
    result = minimize_scalar(rss_at_bp, bounds=(lo, hi), method="bounded")
    bp = float(result.x)
    final_rss = float(result.fun)

    left, right = x <= bp, x > bp
    Xl = np.column_stack([np.ones(left.sum()), x[left]])
    cl, *_ = np.linalg.lstsq(Xl, y[left], rcond=None)
    Xr = np.column_stack([np.ones(right.sum()), x[right]])
    cr, *_ = np.linalg.lstsq(Xr, y[right], rcond=None)

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_sq = 1.0 - final_rss / ss_tot if ss_tot > 0 else np.nan

    slope_below, slope_above = float(cl[1]), float(cr[1])
    valid = slope_above > slope_below and slope_above > 0

    # ── Profile-RSS 95 % CI on the breakpoint ───────────────────
    p_linear = 4  # two intercepts, two slopes
    df_resid = n - p_linear
    ci_lo = ci_hi = bp_se = np.nan
    truncated = False
    if df_resid > 0 and final_rss > 0 and valid:
        f_crit = float(_f_dist.ppf(0.95, 1, df_resid))
        rss_thresh = final_rss * (1.0 + f_crit / df_resid)
        ok = rss_vals <= rss_thresh
        if ok.any():
            lo_idx = int(np.argmax(ok))
            hi_idx = int(len(ok) - 1 - np.argmax(ok[::-1]))

            def _interp_cross(i_out: int, i_in: int) -> float:
                ri, rj = rss_vals[i_out], rss_vals[i_in]
                if not (np.isfinite(ri) and np.isfinite(rj)) or ri == rj:
                    return float(grid[i_in])
                t = (rss_thresh - ri) / (rj - ri)
                return float(grid[i_out] + t * (grid[i_in] - grid[i_out]))

            if lo_idx > 0:
                ci_lo = _interp_cross(lo_idx - 1, lo_idx)
            else:
                ci_lo = float(grid[0])
                truncated = True
            if hi_idx < len(grid) - 1:
                ci_hi = _interp_cross(hi_idx + 1, hi_idx)
            else:
                ci_hi = float(grid[-1])
                truncated = True
            ci_lo, ci_hi = min(ci_lo, ci_hi), max(ci_lo, ci_hi)
            bp_se = (ci_hi - ci_lo) / (2.0 * 1.96)

    return {
        "breakpoint": bp if valid else np.nan,
        "slope_below": slope_below,
        "intercept_below": float(cl[0]),
        "slope_above": slope_above,
        "intercept_above": float(cr[0]),
        "r_squared": float(r_sq) if np.isfinite(r_sq) else np.nan,
        "n": n,
        "n_below": int(left.sum()),
        "n_above": int(right.sum()),
        "converged": bool(valid),
        "breakpoint_ci_lo": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
        "breakpoint_ci_hi": float(ci_hi) if np.isfinite(ci_hi) else np.nan,
        "breakpoint_se": float(bp_se) if np.isfinite(bp_se) else np.nan,
        "breakpoint_ci_truncated": bool(truncated),
        "rejected_reason": (
            None if valid
            else f"slope_above ({slope_above:.4f}) <= slope_below ({slope_below:.4f})"
        ),
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Davies (1987 / 2002) test for breakpoint existence         │
# └────────────────────────────────────────────────────────────┘


def davies_test(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    x_range: tuple[float, float] | None = None,
) -> dict:
    """Davies (1987 / 2002) test for the existence of a breakpoint.

    Tests ``H0`` (the relationship is a single straight line, no
    breakpoint) vs ``H1`` (there is an unknown breakpoint where the
    slope changes).  The procedure evaluates ``k`` candidate
    breakpoints, computes a naive t-statistic for the
    slope-difference at each, takes the maximum, and corrects the
    p-value for the search using Davies' upper bound::

        p ≤ Pr(|T| > M) + V · φ(M)

    where ``V`` is the number of sign changes in the t-statistic
    sequence and ``φ`` is the standard-normal (or t) density.

    Args:
        x:       Predictor array.  NaN entries dropped.
        y:       Response array.  Must align with ``x``.
        k:       Number of evaluation points.  Default 10 (matches the
                 R ``segmented`` package default).
        x_range: Search range for candidate breakpoints.  Default
                 ``(percentile_5, percentile_95)`` of ``x``.

    Returns:
        Dict with ``pvalue``, ``best_at`` (candidate with strongest
        signal), ``max_statistic``, ``n_sign_changes``, and ``n``.

    References:
        Davies RB (1987) Biometrika 74:33-43.
        Davies RB (2002) Biometrika 89:484-489.
    """
    from scipy.stats import norm
    from scipy.stats import t as t_dist

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 30:
        return {"pvalue": np.nan, "best_at": np.nan, "max_statistic": np.nan, "n": n}

    if x_range is None:
        x_range = (np.percentile(x, 5), np.percentile(x, 95))

    psi_candidates = np.linspace(x_range[0], x_range[1], k)
    t_stats = np.full(k, np.nan)
    for j, psi in enumerate(psi_candidates):
        seg_var = np.maximum(x - psi, 0.0)
        X_design = np.column_stack([np.ones(n), x, seg_var])
        try:
            beta, _residuals, rank, _sv = np.linalg.lstsq(X_design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank < 3:
            continue
        y_hat = X_design @ beta
        resid = y - y_hat
        df = n - 3
        if df <= 0:
            continue
        sigma2 = float(np.sum(resid ** 2)) / df
        try:
            XtX_inv = np.linalg.inv(X_design.T @ X_design)
        except np.linalg.LinAlgError:
            continue
        se_delta = float(np.sqrt(max(sigma2 * XtX_inv[2, 2], 0.0)))
        if se_delta <= 0:
            continue
        t_stats[j] = beta[2] / se_delta

    valid_t = t_stats[np.isfinite(t_stats)]
    if len(valid_t) < 2:
        return {"pvalue": np.nan, "best_at": np.nan, "max_statistic": np.nan, "n": n}

    abs_t = np.abs(valid_t)
    M = float(np.max(abs_t))
    best_idx = int(np.nanargmax(np.abs(t_stats)))
    best_psi = float(psi_candidates[best_idx])

    signs = np.sign(valid_t)
    sign_changes = int(np.sum(np.abs(np.diff(signs)) > 0))

    df = n - 3
    if df > 300:
        # Large-sample normal approximation (exact gamma overflows).
        p_naive = 2.0 * (1.0 - norm.cdf(M))
        phi_M = norm.pdf(M)
    else:
        p_naive = 2.0 * (1.0 - t_dist.cdf(M, df))
        phi_M = t_dist.pdf(M, df)

    p_davies = float(min(p_naive + sign_changes * phi_M, 1.0))

    return {
        "pvalue": p_davies,
        "best_at": best_psi,
        "max_statistic": M,
        "n_sign_changes": sign_changes,
        "n": n,
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Pseudo-Score test (Muggeo 2016)                            │
# └────────────────────────────────────────────────────────────┘


def pscore_test(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    x_range: tuple[float, float] | None = None,
) -> dict:
    """Pseudo-Score test for breakpoint existence (Muggeo 2016).

    Typically **more powerful than the Davies test** for detecting a
    single changepoint.  Averages the segmented variable over ``k``
    candidate breakpoints::

        φ̄(x) = (1/k) · Σ_j max(x - ψ_j, 0)

    and tests whether ``γ`` is significant in the augmented linear
    model ``y = α + β·x + γ·φ̄``.  The key insight: under H0 (no
    breakpoint) ``γ = 0``; under H1 ``γ`` picks up the slope change
    regardless of the true breakpoint location, sidestepping the
    nuisance-parameter problem that Davies' procedure handles via
    upper-bound correction.

    Args:
        x:       Predictor array.  NaN entries dropped.
        y:       Response array.  Must align with ``x``.
        k:       Number of evaluation points.  Default 10.
        x_range: Search range for candidate breakpoints.  Default
                 ``(percentile_5, percentile_95)`` of ``x``.

    Returns:
        Dict with ``pvalue``, ``t_statistic``, ``best_at`` (candidate
        ψ with the strongest INDIVIDUAL signal, for reporting only —
        not the test statistic itself), ``df``, and ``n``.

    References:
        Muggeo VMR (2016) Testing with a nuisance parameter present
        only under the alternative: a score-based approach with
        application to segmented modelling. J Stat Comput Simul
        86:3059-3067.
    """
    from scipy.stats import t as t_dist

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 30:
        return {"pvalue": np.nan, "t_statistic": np.nan, "n": n}

    if x_range is None:
        x_range = (np.percentile(x, 5), np.percentile(x, 95))

    psi_candidates = np.linspace(x_range[0], x_range[1], k)

    # Averaged segmented variable.
    phi_bar = np.zeros(n)
    for psi in psi_candidates:
        phi_bar += np.maximum(x - psi, 0.0)
    phi_bar /= k

    X_design = np.column_stack([np.ones(n), x, phi_bar])
    try:
        beta, _residuals, rank, _sv = np.linalg.lstsq(X_design, y, rcond=None)
    except np.linalg.LinAlgError:
        return {"pvalue": np.nan, "t_statistic": np.nan, "n": n}
    if rank < 3:
        return {"pvalue": np.nan, "t_statistic": np.nan, "n": n}

    y_hat = X_design @ beta
    resid = y - y_hat
    df = n - 3
    if df <= 0:
        return {"pvalue": np.nan, "t_statistic": np.nan, "n": n}

    sigma2 = float(np.sum(resid ** 2)) / df
    try:
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
    except np.linalg.LinAlgError:
        return {"pvalue": np.nan, "t_statistic": np.nan, "n": n}

    se_gamma = float(np.sqrt(max(sigma2 * XtX_inv[2, 2], 0.0)))
    if se_gamma <= 0:
        return {"pvalue": np.nan, "t_statistic": np.nan, "n": n}

    t_stat = float(beta[2] / se_gamma)
    pvalue = float(2.0 * (1.0 - t_dist.cdf(abs(t_stat), df)))

    # Reporting-only: which individual candidate ψ gave the strongest signal.
    best_t = 0.0
    best_psi = np.nan
    for psi in psi_candidates:
        seg = np.maximum(x - psi, 0.0)
        Xd = np.column_stack([np.ones(n), x, seg])
        try:
            b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            yh = Xd @ b
            r = y - yh
            s2 = float(np.sum(r ** 2)) / (n - 3)
            inv_xx = np.linalg.inv(Xd.T @ Xd)
            se = float(np.sqrt(max(s2 * inv_xx[2, 2], 0.0)))
            if se > 0 and abs(b[2] / se) > abs(best_t):
                best_t = b[2] / se
                best_psi = float(psi)
        except (np.linalg.LinAlgError, ValueError):
            continue

    return {
        "pvalue": pvalue,
        "t_statistic": t_stat,
        "best_at": best_psi,
        "df": df,
        "n": n,
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Hill / 4-parameter logistic fit  « sigmoid dose-response » │
# └────────────────────────────────────────────────────────────┘


def hill_fit(
    x: np.ndarray,
    y: np.ndarray,
    x_range: tuple[float, float] | None = None,
) -> dict:
    """Fit a four-parameter logistic (Hill) dose-response model.

    Model::

        y = y_min + (y_max - y_min) / (1 + (EC50 / x)^n)

    EC50 is the predictor value at half-maximal response.  The Hill
    coefficient ``n`` (``hill_n`` in the return dict) captures
    steepness: large ``n`` = sharp threshold-like switch, small ``n`` =
    gradual transition.

    In addition to EC50, the **lower bend point** following
    Sebaugh & McCray (2003) is returned: this is the x-value where the
    second derivative of the Hill curve equals zero on the lower
    side, marking the transition from the baseline plateau into the
    rising phase::

        x_bend_lower = EC50 · ((n - 1) / (n + 1))^(1/n)    for n > 1

    For ``n ≤ 1`` (no inflection in the lower portion), falls back to
    EC10 as a conventional onset marker.  The lower bend is directly
    comparable to a broken-stick breakpoint — both mark the onset of
    the rising regime — and is the recommended single-number "where
    does the response start" summary for sigmoid dose-response data.

    Args:
        x:       Predictor array.  NaN entries dropped.
        y:       Response array.  Must align with ``x``.
        x_range: Plausible range for the EC50 initial guess and
                 bounds.  Default ``(percentile_10, percentile_90)``
                 of ``x``.

    Returns:
        Dict with ``ec50``, ``hill_n``, ``y_min``, ``y_max``,
        ``lower_bend`` (``np.nan`` if outside the data range),
        ``r_squared``, ``aic``, ``n``, ``converged``, and
        ``bend_plausible``.

    References:
        Goutelle S et al. (2008) The Hill equation: a review of its
        capabilities in pharmacological modelling. Fundamental &
        Clinical Pharmacology 22:633-648.
        Hill AV (1910) The possible effects of the aggregation of the
        molecules of haemoglobin on its dissociation curves. Journal
        of Physiology 40:iv-vii.
        Sebaugh JL, McCray PD (2003) Defining the linear portion of a
        sigmoid-shaped curve: bend points. Pharmaceutical Statistics
        2:167-174.
    """
    from scipy.optimize import curve_fit

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 30:
        return {"ec50": np.nan, "converged": False, "n": n}

    if x_range is None:
        x_range = (np.percentile(x, 10), np.percentile(x, 90))

    def hill_func(xv, y_min, y_max, ec50, hill_n):
        ratio = np.clip(ec50 / np.maximum(xv, 1e-10), 1e-10, 1e10)
        return y_min + (y_max - y_min) / (1.0 + np.power(ratio, hill_n))

    y_lo = float(np.percentile(y, 10))
    y_hi = float(np.percentile(y, 90))
    ec50_init = float(np.mean(x_range))

    bounds_lo = [np.min(y) - 5, np.median(y), x_range[0] * 0.5, 0.5]
    bounds_hi = [np.median(y), np.max(y) + 5, x_range[1] * 1.5, 30.0]

    try:
        popt, _pcov = curve_fit(
            hill_func, x, y,
            p0=[y_lo, y_hi, ec50_init, 2.0],
            bounds=(bounds_lo, bounds_hi),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return {"ec50": np.nan, "converged": False, "n": n}

    y_min_fit, y_max_fit, ec50, hill_n = (float(p) for p in popt)

    y_pred = hill_func(x, *popt)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    k_params = 5  # 4 Hill params + sigma
    aic = n * np.log(ss_res / n) + 2 * k_params if n > k_params else np.nan

    if hill_n > 1.0:
        lower_bend = ec50 * ((hill_n - 1.0) / (hill_n + 1.0)) ** (1.0 / hill_n)
    else:
        lower_bend = ec50 * (0.10 / 0.90) ** (1.0 / hill_n)

    x_lo, x_hi = float(np.min(x)), float(np.max(x))
    plausible = bool((x_lo - 10) <= lower_bend <= x_hi)

    return {
        "ec50": ec50,
        "hill_n": hill_n,
        "y_min": y_min_fit,
        "y_max": y_max_fit,
        "lower_bend": float(lower_bend) if plausible else np.nan,
        "r_squared": float(r_sq) if np.isfinite(r_sq) else np.nan,
        "aic": float(aic) if np.isfinite(aic) else np.nan,
        "n": n,
        "converged": True,
        "bend_plausible": plausible,
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Per-subject iterator  « DigiMuh-style cohort fitting »     │
# └────────────────────────────────────────────────────────────┘


def per_subject_segmented(
    df: pd.DataFrame,
    subject_col: str | list[str],
    x_col: str,
    y_col: str,
    *,
    model: Callable[..., dict] = broken_stick_fit,
    min_n: int = 50,
    model_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Apply a dose-response fitter to each subject's data.

    For every unique value (or tuple, for composite keys) in
    ``df[subject_col]``, fit ``model(x, y, **model_kwargs)`` where
    ``x`` and ``y`` are the subject's rows of ``df[x_col]`` and
    ``df[y_col]``.  Subjects with fewer than ``min_n`` observations
    are returned with the model's standard skip marker
    (``converged=False`` plus the primary parameter as ``np.nan``).

    Pickle-safe: the ``model`` callable must be a module-level
    function so this iterator can be wrapped in a
    ``concurrent.futures.ProcessPoolExecutor`` for parallel fitting
    over large panels.  All four primary fitters in this module
    (``broken_stick_fit``, ``davies_test``, ``pscore_test``,
    ``hill_fit``) qualify.

    Args:
        df:           Long-format DataFrame with one row per
                      observation.
        subject_col:  Column name (or list of column names for a
                      composite key) identifying subjects.
        x_col:        Column holding the predictor.
        y_col:        Column holding the response.
        model:        Fitter callable taking ``(x, y, **kwargs)`` and
                      returning a result dict.  Defaults to
                      :func:`broken_stick_fit`.
        min_n:        Minimum observations per subject.  Subjects
                      below this threshold get a skip-marker row.
                      Default 50.
        model_kwargs: Extra keyword arguments forwarded to ``model``.

    Returns:
        DataFrame with one row per subject.  Columns: the subject key
        column(s) plus all keys returned by ``model``.  Subjects with
        ``n < min_n`` carry ``converged=False`` and ``n=<actual>``.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if isinstance(subject_col, str):
        group_keys = [subject_col]
    else:
        group_keys = list(subject_col)

    rows: list[dict] = []
    for key, grp in df.groupby(group_keys, dropna=False):
        x = grp[x_col].to_numpy(dtype=float)
        y = grp[y_col].to_numpy(dtype=float)

        # pandas 2.x: groupby on a LIST of columns always yields a tuple key
        # (even when the list has length 1).  Normalising once handles both
        # the single-column and composite-key cases uniformly.
        key_tuple = key if isinstance(key, tuple) else (key,)
        subject_id = dict(zip(group_keys, key_tuple, strict=False))

        if len(x) < min_n:
            row = {**subject_id, "n": len(x), "converged": False}
        else:
            result = model(x, y, **model_kwargs)
            row = {**subject_id, **result}
        rows.append(row)

    return pd.DataFrame(rows)


__all__ = [
    "broken_stick_fit",
    "davies_test",
    "pscore_test",
    "hill_fit",
    "per_subject_segmented",
]
