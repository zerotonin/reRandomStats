# ╔══════════════════════════════════════════════════════════════════╗
# ║  reRandomStats — case_crossover                                  ║
# ║  « time-stratified case-crossover + permutation backup »         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Time-stratified case-crossover with conditional logit (Maclure  ║
# ║  1991; Lee et al. 2023) for binary outcomes with within-stratum  ║
# ║  matched controls.  Each event carries its own pre-matched       ║
# ║  control set; the conditional likelihood integrates out the      ║
# ║  stratum intercepts.  Stratified permutation provides a          ║
# ║  distribution-free backup.                                       ║
# ║                                                                  ║
# ║  Plus three companion utilities: a closed-form daylight-hours    ║
# ║  covariate, a Burke-2015 σ-rescaled effect-size translator, and  ║
# ║  a within-event temporal-contrast test (hot-day vs hot-week).    ║
# ║                                                                  ║
# ║  Generalised from the ThermoStrife pipeline (Geurten 2026,       ║
# ║  ThermoStrife v0.1.1, DOI 10.5281/zenodo.20371612).              ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Time-stratified case-crossover estimators with permutation backup.

A *case-crossover* design (Maclure 1991, Am J Epidemiol 133:144–153)
estimates the within-stratum association between a transient exposure
and a binary outcome by matching each case observation against a set
of control observations drawn from the same subject / stratum at
other times.  The stratum intercepts are integrated out in the
conditional likelihood, so all time-invariant subject characteristics
are absorbed by construction.

The events-list interchange convention used throughout this module
mirrors the ThermoStrife pipeline (Geurten 2026,
https://doi.org/10.5281/zenodo.20371612).  Each event is a dict with
the following keys:

- ``event_id`` (str)                — stratum identifier.
- ``lat`` (float), ``lon`` (float)  — only used by the daylight
  covariate and the within-event-contrast fetcher; pass dummy values
  if these are not needed.
- ``when`` (datetime.date)          — event-day timestamp.
- ``tmax_event_c`` (float)          — case-row exposure value.
- ``baseline`` (pandas.DataFrame)   — one row per control day, index
  = ``date``, with a column named ``tmax`` carrying the control-day
  exposure value.  The column is named ``tmax`` for historical reasons
  in the ThermoStrife data model; the value may be any continuous
  exposure variable in the caller's units.

The case-crossover machinery is exposure-agnostic: ``tmax_c`` is
literally just the column name.  The functions in this module make
no biological assumption about what that column represents.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy import stats

# Sensible defaults; all functions accept explicit overrides.
_DEFAULT_N_PERM: int = 10_000
_DEFAULT_N_BOOT: int = 10_000


# ┌────────────────────────────────────────────────────────────┐
# │ Daylight-hours covariate  « closed-form solar geometry »   │
# └────────────────────────────────────────────────────────────┘


def daylight_hours(lat_deg: float, when: date) -> float:
    """Closed-form solar daylight hours for ``when`` at latitude ``lat_deg``.

    Uses the standard astronomical approximation (sunrise / sunset
    by solar declination, ignoring atmospheric refraction and
    astronomical twilight).  Accurate to ~10 min outside the polar
    circles, which is well inside the noise of typical daily exposure
    inference and avoids a hard dependency on ``astral`` or
    ``pysolar``.

    Args:
        lat_deg: Latitude in decimal degrees (positive north).
        when:    Calendar date.

    Returns:
        Daylight in hours; clamped to ``0.0`` at polar night and
        ``24.0`` at polar day.
    """
    doy = when.timetuple().tm_yday
    decl = 23.44 * math.sin(math.radians(360 / 365 * (doy - 81)))
    lat = math.radians(lat_deg)
    decl_r = math.radians(decl)
    cos_h = -math.tan(lat) * math.tan(decl_r)
    if cos_h <= -1.0:
        return 24.0
    if cos_h >= 1.0:
        return 0.0
    return (math.degrees(math.acos(cos_h)) * 2.0) / 15.0


# ┌────────────────────────────────────────────────────────────┐
# │ Case-crossover frame  « (case + controls) per event »      │
# └────────────────────────────────────────────────────────────┘


def build_case_crossover_frame(events: list[dict]) -> pd.DataFrame:
    """Stack per-event case + control rows into one long DataFrame.

    Events with no baseline rows are silently skipped.  Events with
    a ``None`` ``tmax_event_c`` produce a case row whose ``tmax_c``
    is ``NaN``; downstream estimators will drop those.

    Args:
        events: List of event dicts conforming to the module-level
                interchange convention (see module docstring).

    Returns:
        Long DataFrame with columns
        ``[event_id, day, is_case, tmax_c, daylight_h]``, one row per
        (event, day) pair.  Cases get ``is_case = 1``, controls get
        ``is_case = 0``.
    """
    frames = []
    for e in events:
        eid = e["event_id"]
        lat = e["lat"]
        when = e["when"]
        baseline = e["baseline"]
        if baseline is None or len(baseline) == 0:
            continue
        ctrl = pd.DataFrame({
            "event_id": eid,
            "day": baseline.index,
            "is_case": 0,
            "tmax_c": baseline["tmax"].values,
        })
        ctrl["daylight_h"] = [daylight_hours(lat, d) for d in ctrl["day"]]
        case = pd.DataFrame({
            "event_id": [eid],
            "day": [when],
            "is_case": [1],
            "tmax_c": [e["tmax_event_c"]],
            "daylight_h": [daylight_hours(lat, when)],
        })
        frames.append(pd.concat([case, ctrl], ignore_index=True))
    if not frames:
        return pd.DataFrame(
            columns=["event_id", "day", "is_case", "tmax_c", "daylight_h"]
        )
    return pd.concat(frames, ignore_index=True)


# ┌────────────────────────────────────────────────────────────┐
# │ Conditional logit  « primary case-crossover estimator »    │
# └────────────────────────────────────────────────────────────┘


def case_crossover_conditional_logit(
    frame: pd.DataFrame,
    *,
    covariates: list[str] | None = None,
) -> dict:
    """Conditional logistic regression on matched case-control sets.

    Fits ``logit P(case | event_id) = β · tmax_c + γ · covariates`` via
    ``statsmodels.discrete.conditional_models.ConditionalLogit``.  The
    stratum intercepts are integrated out by the conditional
    likelihood, so all time-invariant within-stratum characteristics
    are absorbed by construction.  ``exp(β)`` is the odds ratio of
    case-vs-control per +1 unit of the exposure variable.

    Args:
        frame:      Output of :func:`build_case_crossover_frame` (or a
                    DataFrame with the same column conventions).
        covariates: Extra columns to include alongside ``tmax_c``.
                    Defaults to ``["daylight_h"]``.

    Returns:
        Dict with the headline estimates and metadata.  Carries a
        ``{"skipped": True, "reason": str}`` marker if the frame is
        empty or no stratum retains a case row after NA filtering.
    """
    from statsmodels.discrete.conditional_models import ConditionalLogit

    if frame.empty:
        return {"skipped": True, "reason": "empty case-crossover frame"}
    if covariates is None:
        covariates = ["daylight_h"]

    cols = ["tmax_c", *covariates]
    work = frame.dropna(subset=[*cols, "is_case", "event_id"]).copy()
    # Drop strata that lost their case row after NA filtering.
    keep = work.groupby("event_id")["is_case"].sum() > 0
    work = work[work["event_id"].isin(keep[keep].index)]
    if work.empty:
        return {"skipped": True, "reason": "no event_id retains a case row"}

    model = ConditionalLogit(
        endog=work["is_case"].astype(int).values,
        exog=work[cols].astype(float).values,
        groups=work["event_id"].astype(str).values,
    )
    result = model.fit(disp=0)
    beta = float(result.params[0])
    se = float(result.bse[0])
    z = beta / se if se > 0 else float("nan")
    p_one_sided = 1.0 - stats.norm.cdf(z) if not math.isnan(z) else float("nan")
    p_two_sided = (
        2.0 * (1.0 - stats.norm.cdf(abs(z))) if not math.isnan(z) else float("nan")
    )
    return {
        "n_events": int(work["event_id"].nunique()),
        "n_rows": int(len(work)),
        "covariates": covariates,
        "beta_per_C": beta,
        "se_per_C": se,
        "or_per_C": float(math.exp(beta)),
        "or_ci95_low": float(math.exp(beta - 1.96 * se)),
        "or_ci95_high": float(math.exp(beta + 1.96 * se)),
        "pvalue_one_sided": float(p_one_sided),
        "pvalue_two_sided": float(p_two_sided),
        "covariate_betas": {
            name: float(b)
            for name, b in zip(cols[1:], result.params[1:])
        },
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Stratified permutation  « non-parametric backup test »     │
# └────────────────────────────────────────────────────────────┘


def stratified_permutation(
    frame: pd.DataFrame,
    n_perm: int = _DEFAULT_N_PERM,
    rng: np.random.Generator | None = None,
) -> dict:
    """Within-event label-shuffle test on the mean case-vs-control gap.

    Under the null, the case-day exposure is exchangeable with the
    baseline-day exposures within the same stratum.  For each event,
    the test shuffles the case label across (case + controls),
    computes (case-exposure) minus (mean of control-exposures), and
    averages over events.  The empirical permutation distribution of
    this statistic gives a distribution-free backup for
    :func:`case_crossover_conditional_logit`.

    Args:
        frame:  Output of :func:`build_case_crossover_frame`.
        n_perm: Number of permutation draws.  Default 10 000.
        rng:    Optional NumPy ``Generator``.  Default ``None`` →
                fresh non-deterministic RNG.  Pass an explicit seeded
                generator for reproducibility.

    Returns:
        Dict carrying ``observed_diff_C``, ``pvalue_one_sided``,
        ``pvalue_two_sided``, ``n_events``, and ``n_perm``.  Returns a
        ``{"skipped": True}`` marker on empty frames or when no event
        has exactly one case row.
    """
    if rng is None:
        rng = np.random.default_rng()
    work = frame.dropna(subset=["tmax_c", "is_case", "event_id"]).copy()
    if work.empty:
        return {"skipped": True, "reason": "empty frame"}

    per_event = []
    for _eid, grp in work.groupby("event_id"):
        if grp["is_case"].sum() != 1:
            continue
        per_event.append(
            (grp["tmax_c"].to_numpy(), grp["is_case"].to_numpy().astype(int))
        )
    if not per_event:
        return {"skipped": True, "reason": "no event has exactly one case row"}

    def _stat(case_labels_per_event):
        diffs = []
        for (tmaxs, labels) in case_labels_per_event:
            ev = tmaxs[labels == 1].mean()
            ct = tmaxs[labels == 0].mean()
            diffs.append(ev - ct)
        return float(np.mean(diffs))

    observed = _stat(per_event)
    perm_stats = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = []
        for (tmaxs, _orig) in per_event:
            new_labels = np.zeros_like(_orig)
            new_labels[rng.integers(0, len(tmaxs))] = 1
            shuffled.append((tmaxs, new_labels))
        perm_stats[i] = _stat(shuffled)
    p_two_sided = float(np.mean(np.abs(perm_stats) >= abs(observed)))
    p_one_sided = float(np.mean(perm_stats >= observed))
    return {
        "n_events": len(per_event),
        "n_perm": n_perm,
        "observed_diff_C": observed,
        "pvalue_one_sided": p_one_sided,
        "pvalue_two_sided": p_two_sided,
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Stratified refit  « per-subgroup case-crossover »          │
# └────────────────────────────────────────────────────────────┘


def stratify_case_crossover(
    events: list[dict],
    key_fn: Callable[[dict], str | None],
    *,
    min_events: int = 5,
    covariates: list[str] | None = None,
) -> dict:
    """Split events into subgroups and re-fit the case-crossover per subgroup.

    Args:
        events:     List of event dicts (interchange convention per
                    module docstring).
        key_fn:     Callable mapping an event to a stratum label.
                    Events for which ``key_fn`` returns ``None`` are
                    filtered out rather than bucketed under a
                    ``"None"`` label.
        min_events: Strata with fewer events than this threshold are
                    not fit; the result dict carries a ``{"skipped":
                    True, "reason": ...}`` marker for them so the
                    caller can still surface them in reports.
        covariates: Forwarded to
                    :func:`case_crossover_conditional_logit`.
                    Defaults to ``[]`` rather than ``["daylight_h"]``
                    because the daylight covariate often becomes
                    collinear within small strata.

    Returns:
        Dict mapping stratum label to that stratum's
        :func:`case_crossover_conditional_logit` result, with an extra
        ``n_events`` key on every entry.
    """
    if covariates is None:
        covariates = []
    buckets: dict[str, list[dict]] = {}
    for e in events:
        label = key_fn(e)
        if label is None:
            continue
        buckets.setdefault(str(label), []).append(e)

    results: dict[str, dict] = {}
    for label, evs in sorted(buckets.items()):
        if len(evs) < min_events:
            results[label] = {
                "skipped": True,
                "reason": f"only {len(evs)} events (< {min_events})",
                "n_events": len(evs),
            }
            continue
        frame = build_case_crossover_frame(evs)
        results[label] = case_crossover_conditional_logit(
            frame, covariates=covariates
        )
        results[label]["n_events"] = len(evs)
    return results


# ┌────────────────────────────────────────────────────────────┐
# │ Within-event temporal contrast  « hot-day vs hot-week »    │
# └────────────────────────────────────────────────────────────┘


def event_anomaly_profile(
    events: list[dict],
    fetch_fn: Callable[..., float | None],
    offsets: tuple[int, ...] = (-7, -2, -1, 0, 1, 2, 7),
) -> pd.DataFrame:
    """Fetch per-event exposure at each ``offset`` and return as a long frame.

    For each event, looks up the exposure on day ``when + offset`` via
    the caller-supplied ``fetch_fn``, subtracts the event's
    baseline-window mean to get an anomaly, and stacks the rows.
    Offsets that the fetcher returns ``None`` for are silently dropped
    — downstream consumers (e.g. superposed-epoch plots) handle the
    resulting ragged ``n`` per offset.

    Args:
        events:   List of event dicts.  Each must additionally provide
                  ``provenance`` and optionally ``station_id``, which
                  are forwarded to ``fetch_fn``.
        fetch_fn: Callable with signature
                  ``fetch_fn(provenance, lat, lon, day, *, station_id)``
                  returning either a float (the exposure value) or
                  ``None`` (no data for that day).  This is the seam
                  through which the caller's own data backend is
                  plugged in.
        offsets:  Day offsets relative to the event day.  Default
                  ``(-7, -2, -1, 0, 1, 2, 7)`` mirrors the
                  superposed-epoch composite used in
                  Geurten (2026) ThermoStrife v0.1.1.

    Returns:
        Long DataFrame with columns ``event_id``, ``offset_days``,
        ``anomaly_C``.  Empty DataFrame if no event resolves any offset.
    """
    rows: list[dict] = []
    for e in events:
        baseline = e.get("baseline")
        if baseline is None or len(baseline) == 0:
            continue
        baseline_mean = float(baseline["tmax"].mean())
        for off in offsets:
            if off == 0:
                v = e.get("tmax_event_c")
            else:
                day = e["when"] + timedelta(days=off)
                v = fetch_fn(
                    e["provenance"], e["lat"], e["lon"], day,
                    station_id=e.get("station_id"),
                )
            if v is None:
                continue
            rows.append({
                "event_id": e["event_id"],
                "offset_days": int(off),
                "anomaly_C": float(v) - baseline_mean,
            })
    return pd.DataFrame(rows)


def h3_within_event_contrast(
    events: list[dict],
    fetch_fn: Callable[..., float | None],
    *,
    window_offsets: tuple[int, ...] = (0, -1),
    surround_offsets: tuple[int, ...] = (-7, 7),
    n_boot: int = _DEFAULT_N_BOOT,
    rng: np.random.Generator | None = None,
) -> dict:
    """Per-event paired contrast: mean anomaly in window vs surround offsets.

    For each event, fetches exposure on day ``t + offset`` for every
    offset in ``window_offsets ∪ surround_offsets`` via ``fetch_fn``,
    converts to an anomaly by subtracting the existing baseline mean,
    then computes::

        diff = mean(anomaly @ window_offsets) − mean(anomaly @ surround_offsets)

    Across-event distribution of ``diff`` is tested with a one-sided
    Wilcoxon signed-rank against H₁: median > 0.  A bootstrap 95 % CI
    on the mean is also returned.

    Args:
        events:           Event dicts (interchange convention; must
                          provide ``provenance`` and optionally
                          ``station_id`` for the fetcher).
        fetch_fn:         Same signature as in
                          :func:`event_anomaly_profile`.
        window_offsets:   Offsets that should sit AT or NEAR the event
                          day under the hypothesis.  Default
                          ``(0, -1)`` (event day + day-before).
        surround_offsets: Offsets that should sit AWAY from the event
                          day.  Default ``(-7, 7)``.
        n_boot:           Bootstrap draws for the mean CI.
        rng:              Optional ``np.random.Generator``.

    Returns:
        Dict with ``mean_diff_C``, ``median_diff_C``, ``ci95_low_C``,
        ``ci95_high_C``, ``wilcoxon_statistic``, ``pvalue_one_sided``,
        ``n_events_used``, and ``n_events_skipped``.  Returns
        ``{"skipped": True, ...}`` if fewer than 5 events resolve all
        required offsets, or if every diff is exactly zero (Wilcoxon
        undefined).
    """
    if rng is None:
        rng = np.random.default_rng()
    diffs = []
    n_skipped_missing_day = 0
    needed = tuple(set(window_offsets) | set(surround_offsets))

    for e in events:
        baseline = e.get("baseline")
        if baseline is None or len(baseline) == 0:
            n_skipped_missing_day += 1
            continue
        baseline_mean = float(baseline["tmax"].mean())

        vals: dict[int, float] = {}
        ok = True
        for off in needed:
            if off == 0:
                v = e.get("tmax_event_c")
            else:
                day = e["when"] + timedelta(days=off)
                v = fetch_fn(
                    e["provenance"], e["lat"], e["lon"], day,
                    station_id=e.get("station_id"),
                )
            if v is None:
                ok = False
                break
            vals[off] = float(v)
        if not ok:
            n_skipped_missing_day += 1
            continue

        window_anom = float(np.mean([vals[off] - baseline_mean for off in window_offsets]))
        surround_anom = float(np.mean([vals[off] - baseline_mean for off in surround_offsets]))
        diffs.append(window_anom - surround_anom)

    if len(diffs) < 5:
        return {
            "skipped": True,
            "reason": f"only {len(diffs)} events had all required offsets",
        }

    arr = np.asarray(diffs)
    if np.all(arr == 0):
        return {
            "skipped": True,
            "reason": "all event diffs are exactly zero — Wilcoxon undefined",
            "n_events_used": int(len(arr)),
        }
    res = stats.wilcoxon(arr, alternative="greater", zero_method="wilcox")
    boot = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])

    return {
        "n_events_used": int(len(arr)),
        "n_events_skipped": int(n_skipped_missing_day),
        "window_offsets_days": list(window_offsets),
        "surround_offsets_days": list(surround_offsets),
        "mean_diff_C": float(arr.mean()),
        "median_diff_C": float(np.median(arr)),
        "ci95_low_C": float(ci_lo),
        "ci95_high_C": float(ci_hi),
        "wilcoxon_statistic": float(res.statistic),
        "pvalue_one_sided": float(res.pvalue),
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Burke-2015 σ rescaling  « cross-study effect comparison »  │
# └────────────────────────────────────────────────────────────┘


def hsiang_sigma_rescaled(
    events: list[dict],
    n_boot: int = _DEFAULT_N_BOOT,
    rng: np.random.Generator | None = None,
) -> dict:
    """Per-event z-score (event − baseline_mean) / baseline_std, averaged.

    Following Burke, Hsiang & Miguel (2015, Ann Rev Econ 7:577–617),
    expressing each event's exposure anomaly as a multiple of its own
    baseline-window standard deviation makes effects comparable
    across stations, climates, and study designs.  Their headline
    meta-analytic number for interpersonal violence is +2.4 % per 1 σ
    contemporaneous warming.

    Args:
        events: Event dicts (interchange convention).  Events without
                a usable baseline (< 2 rows or zero standard
                deviation) are silently dropped.
        n_boot: Bootstrap draws for the CI.
        rng:    Optional ``np.random.Generator``.

    Returns:
        Dict with ``mean_z``, ``median_z``, ``ci95_low_z``,
        ``ci95_high_z``, ``fraction_positive``, and ``n_events``.
        Returns ``{"skipped": True}`` if no event has a usable baseline.
    """
    if rng is None:
        rng = np.random.default_rng()
    zs = []
    for e in events:
        baseline = e.get("baseline")
        if baseline is None or len(baseline) < 2:
            continue
        bmean = float(baseline["tmax"].mean())
        bstd = float(baseline["tmax"].std(ddof=1))
        if bstd <= 0:
            continue
        ev = e.get("tmax_event_c")
        if ev is None:
            continue
        zs.append((float(ev) - bmean) / bstd)
    if not zs:
        return {"skipped": True, "reason": "no events with usable baseline"}
    z_arr = np.array(zs)
    n = len(z_arr)
    boot = rng.choice(z_arr, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {
        "n_events": n,
        "mean_z": float(z_arr.mean()),
        "median_z": float(np.median(z_arr)),
        "ci95_low_z": float(lo),
        "ci95_high_z": float(hi),
        "fraction_positive": float((z_arr > 0).mean()),
    }


__all__ = [
    "daylight_hours",
    "build_case_crossover_frame",
    "case_crossover_conditional_logit",
    "stratified_permutation",
    "stratify_case_crossover",
    "event_anomaly_profile",
    "h3_within_event_contrast",
    "hsiang_sigma_rescaled",
]
