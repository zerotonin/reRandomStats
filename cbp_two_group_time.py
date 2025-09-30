import numpy as np
from scipy.stats import ttest_ind
from itertools import combinations
from frites.stats import cluster_threshold, cluster_correction_mcp
from frites.workflow import WfStats
import pandas as pd
import math 
from pathlib import Path
from scipy.stats import mannwhitneyu

def rbc_curve(ctrl, ko):
    """
    ctrl, ko: arrays (n_flies, n_times)
    returns: RBC per time (shape: n_times), robust & signed
    """
    n_times = ctrl.shape[1]
    rbc = np.zeros(n_times, dtype=float)
    for t in range(n_times):
        x = ctrl[:, t]
        y = ko[:, t]
        x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
        if len(x) == 0 or len(y) == 0:
            rbc[t] = 0.0
            continue
        # U for x vs y (ties counted as 0.5 internally)
        U = mannwhitneyu(x, y, alternative='two-sided', method='auto').statistic
        rbc[t] = 2.0 * U / (len(x) * len(y)) - 1.0
    return rbc

def cbp_two_group_time(ctrl, ko, alpha=0.05, tail=0, tfce=True,
                       n_perm='auto', random_state=0):
    """
    Cluster-based permutation test (two independent groups) along time.
    Returns:
      p_time  : (n_times,) FWER-corrected p-values per time
      sig_int : list of (start_idx, end_idx) intervals with p<alpha
      t_obs   : (n_times,) observed Welch t-statistic
      thr_spec: 'tfce' or float threshold used
    """
    rng = np.random.default_rng(random_state)
    ctrl = np.asarray(ctrl, float); ko = np.asarray(ko, float)
    assert ctrl.ndim == ko.ndim == 2 and ctrl.shape[1] == ko.shape[1]
    n_c, n_t = ctrl.shape
    X_all = np.vstack([ctrl, ko]); n_all = X_all.shape[0]

    # 1) observed Welch t per time (NaN-safe)
    t_obs, _ = ttest_ind(ctrl, ko, axis=0, equal_var=False, nan_policy='omit')
    t_obs = np.nan_to_num(t_obs, nan=0.0, posinf=0.0, neginf=0.0)
    


    # 2) permutations (exact if small, else sample)
    total = math.comb(n_all, n_c)
    if n_perm == 'auto':
        n_perm = total if total <= 50000 else 5000

    t_perm = np.empty((n_perm, n_t), float)
    if n_perm == total:  # exact
        from itertools import combinations
        all_idx = np.arange(n_all)
        for i, idx_c in enumerate(combinations(all_idx, n_c)):
            idx_c = np.asarray(idx_c)
            idx_k = np.setdiff1d(all_idx, idx_c, assume_unique=True)
            t_perm[i], _ = ttest_ind(X_all[idx_c], X_all[idx_k], axis=0, equal_var=False, nan_policy='omit')
    else:  # Monte Carlo
        for i in range(n_perm):
            perm = rng.permutation(n_all)
            idx_c = perm[:n_c]; idx_k = perm[n_c:]
            t_perm[i], _ = ttest_ind(X_all[idx_c], X_all[idx_k], axis=0, equal_var=False, nan_policy='omit')

    t_perm = np.nan_to_num(t_perm, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) reshape to (n_roi, n_times) & (n_perm, n_roi, n_times)
    x   = t_obs[None, :]            # (1, n_times)
    x_p = t_perm[:, None, :]        # (n_perm, 1, n_times)

    p_time = None
    thr_spec = None

    if tfce:
        # Primary: WfStats with TFCE; Fallback: numeric threshold if it fails
        try:
            wf = WfStats()
            pvals, _ = wf.fit(
                effect=[x], perms=[x_p],
                inference='rfx', mcp='cluster', tail=tail,
                cluster_th='tfce', cluster_alpha=alpha, ttested=True
            )
            p_time = pvals[:, 0]
            thr_spec = 'tfce'
        except Exception as e:
            # Fallback to numeric threshold (no TFCE)
            thr = cluster_threshold(x, x_p, alpha=alpha, tail=tail, tfce=False)  # float
            p = cluster_correction_mcp(x, x_p, th=thr, tail=tail)               # (1, n_times)
            p_time = p[0]
            thr_spec = float(thr)
    else:
        # Pure numeric threshold path
        thr = cluster_threshold(x, x_p, alpha=alpha, tail=tail, tfce=False)      # float
        p  = cluster_correction_mcp(x, x_p, th=thr, tail=tail)                   # (1, n_times)
        p_time = p[0]
        thr_spec = float(thr)

    # 4) contiguous significant intervals
    sig = p_time < alpha
    sig_int = []
    if sig.any():
        edges = np.diff(np.concatenate(([False], sig, [False])).astype(int))
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]
        sig_int = list(zip(starts, ends))  # end is exclusive

    return p_time, sig_int, t_obs, thr_spec

def cbp_two_group_time_ranksum(ctrl, ko, alpha=0.05, tail=0, tfce=True,
                               n_perm='auto', random_state=0):
    """
    As before, but uses Mann–Whitney ranksum (RBC) per time instead of Welch t.
    Returns:
      p_time  : (n_times,) FWER-corrected p per time
      sig_int : list of (start_idx, end_idx)
      stat_obs: (n_times,) observed RBC curve
      thr_spec: 'tfce' or float threshold used
    """
    rng = np.random.default_rng(random_state)
    ctrl = np.asarray(ctrl, float); ko = np.asarray(ko, float)
    assert ctrl.ndim == ko.ndim == 2 and ctrl.shape[1] == ko.shape[1]
    n_c, n_t = ctrl.shape
    X_all = np.vstack([ctrl, ko]); n_all = X_all.shape[0]

    # 1) observed robust stat per time (RBC)
    stat_obs = rbc_curve(ctrl, ko)
    stat_obs = np.nan_to_num(stat_obs, nan=0.0)

    # 2) permutations (exact if small, else sample)
    import math
    total = math.comb(n_all, n_c)
    if n_perm == 'auto':
        n_perm = total if total <= 50000 else 5000

    stat_perm = np.empty((n_perm, n_t), float)
    if n_perm == total:
        from itertools import combinations
        all_idx = np.arange(n_all)
        for i, idx_c in enumerate(combinations(all_idx, n_c)):
            idx_c = np.asarray(idx_c)
            idx_k = np.setdiff1d(all_idx, idx_c, assume_unique=True)
            stat_perm[i] = rbc_curve(X_all[idx_c], X_all[idx_k])
    else:
        for i in range(n_perm):
            perm = rng.permutation(n_all)
            idx_c = perm[:n_c]; idx_k = perm[n_c:]
            stat_perm[i] = rbc_curve(X_all[idx_c], X_all[idx_k])

    stat_perm = np.nan_to_num(stat_perm, nan=0.0)

    # 3) reshape for frites
    x   = stat_obs[None, :]           # (1, n_times)
    x_p = stat_perm[:, None, :]       # (n_perm, 1, n_times)

    # 4) TFCE (preferred) with robust fallback, exactly like before
    from frites.workflow import WfStats
    from frites.stats import cluster_threshold, cluster_correction_mcp

    p_time = None; thr_spec = None
    if tfce:
        try:
            wf = WfStats()
            pvals, _ = wf.fit(
                effect=[x], perms=[x_p],
                inference='rfx', mcp='cluster', tail=tail,
                cluster_th='tfce', cluster_alpha=alpha,
                ttested=True   # means "we pass a precomputed stat curve"
            )
            p_time = pvals[:, 0]
            thr_spec = 'tfce'
        except Exception:
            thr = cluster_threshold(x, x_p, alpha=alpha, tail=tail, tfce=False)
            p = cluster_correction_mcp(x, x_p, th=thr, tail=tail)
            p_time = p[0]; thr_spec = float(thr)
    else:
        thr = cluster_threshold(x, x_p, alpha=alpha, tail=tail, tfce=False)
        p  = cluster_correction_mcp(x, x_p, th=thr, tail=tail)
        p_time = p[0]; thr_spec = float(thr)

    # 5) contiguous significant windows
    sig = p_time < alpha
    sig_int = []
    if sig.any():
        edges = np.diff(np.concatenate(([False], sig, [False])).astype(int))
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]
        sig_int = list(zip(starts, ends))

    return p_time, sig_int, stat_obs, thr_spec

# ---------- helpers to prep matrices ----------

def common_time_axis(df, structure, compartment, window=None):
    """Intersection of available times across genotypes (optionally cropped)."""
    m = (df['structure'] == structure) & (df['compartment'] == compartment)
    wt = df[m & (df['genotype'] == 'dnc-wt')]['time_s'].unique()
    kd = df[m & (df['genotype'] == 'dnc-KD')]['time_s'].unique()
    t = np.sort(np.intersect1d(wt, kd))
    if window is not None:
        tmin, tmax = window
        if tmin is not None: t = t[t >= tmin]
        if tmax is not None: t = t[t <= tmax]
    return t

def fly_matrix(df, genotype, structure, compartment, time_axis):
    """
    Build (n_flies, n_times) by averaging all ROIs of a fly at each time.
    """
    d = df[(df['genotype'] == genotype) &
           (df['structure'] == structure) &
           (df['compartment'] == compartment)][['fly_id', 'time_s', 'dff']]

    # average across ROIs for a fly at each time
    g = (d.groupby(['fly_id', 'time_s'], as_index=False)['dff']
           .mean())

    fly_ids = np.sort(g['fly_id'].unique())
    X = np.full((len(fly_ids), len(time_axis)), np.nan, float)
    for i, fid in enumerate(fly_ids):
        s = (g[g['fly_id'] == fid]
             .set_index('time_s')
             .reindex(time_axis)['dff'])
        X[i, :] = s.to_numpy()
    return X, fly_ids

# ---------- load your data ----------
df = pd.read_csv("data/fig_1_camp_long.csv")

# parameters for Figure 2
structure    = 'Bouton'                
compartments = ['g1','g2','g3','g4','g5']
time_window  = (-10.0, 90.0)           # seconds around shock (adjust; or set to None)

alpha        = 0.05
tfce         = True                   # good default for broad effects
tail         = 0                      # two-sided
n_perm       = "auto"              # exact if <= 50k, else 5000
rng_seed     = 0

rows = []
for comp in compartments:
    # 1) common time grid for WT & KD
    t = common_time_axis(df, structure=structure, compartment=comp,
                         window=time_window)
    if t.size == 0:
        print(f"[{comp}] No overlapping times between genotypes in the selected window.")
        continue

    # 2) build fly-level matrices
    ctrl, ctrl_ids = fly_matrix(df, 'dnc-wt', structure, comp, t)
    ko,   ko_ids   = fly_matrix(df, 'dnc-KD', structure, comp, t)

    # sanity: need >= 2 flies per group at most time points
    if ctrl.shape[0] < 2 or ko.shape[0] < 2:
        print(f"[{comp}] Not enough flies after filtering (WT={ctrl.shape[0]}, KD={ko.shape[0]}).")
        continue

    # 3) run CBP
    p_time, sig_int, t_obs, thr = cbp_two_group_time(
        ctrl, ko, alpha=alpha, tail=tail, tfce=True,  # keep True; it will fallback if needed
        n_perm=n_perm, random_state=rng_seed
    )
    # p_time, sig_int, stat_obs, thr = cbp_two_group_time_ranksum(
    #     ctrl, ko, alpha=alpha, tail=0, tfce=True,
    #     n_perm=n_perm, random_state=rng_seed
    # )

    # 4) summarize clusters as time intervals with a cluster p
    for (s, e) in sig_int:
        # cluster p is constant within the span; take min to be explicit
        cluster_p = float(np.nanmin(p_time[s:e]))
        rows.append({
            'structure': structure,
            'compartment': comp,
            't_start_s': float(t[s]),
            't_end_s':   float(t[e-1]),
            'cluster_p': cluster_p,
            'direction': np.sign(np.nanmean(np.nanmean(ctrl[:, s:e], axis=1) -
                                            np.nanmean(ko[:, s:e],   axis=1)))  # + WT>KD, - KD>WT
        })

    # quick console output
    if sig_int:
        print(f"[{comp}] significant windows @ alpha={alpha}:")
        for (s, e) in sig_int:
            print(f"  {t[s]:7.3f} s  →  {t[e-1]:7.3f} s   (p≈{np.nanmin(p_time[s:e]):.4g})")
    else:
        print(f"[{comp}] no significant windows (alpha={alpha})")

# 5) save a tidy results table for the figure legend / methods
out = pd.DataFrame(rows)
Path("out").mkdir(exist_ok=True)
out_path = Path("out/fig2_cbp_results.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved cluster table: {out_path.resolve()}")

# Optionally: also save per-time p-curves per compartment
# (so you can plot shaded significant spans later)
per_time = []
for comp in compartments:
    # rebuild to get p_time; skip repetition by caching if you prefer
    t = common_time_axis(df, structure, comp, time_window)
    if t.size == 0: continue
    ctrl, _ = fly_matrix(df, 'dnc-wt', structure, comp, t)
    ko,   _ = fly_matrix(df, 'dnc-KD', structure, comp, t)
    p_time, sig_int, *_ = cbp_two_group_time(ctrl, ko, alpha=alpha, tail=tail,
                                             tfce=tfce, n_perm=n_perm, random_state=rng_seed)
    per_time.append(pd.DataFrame({'structure': structure,
                                  'compartment': comp,
                                  'time_s': t,
                                  'p_fwer': p_time}))
if per_time:
    pd.concat(per_time, ignore_index=True).to_csv("out/fig2_per_time_p.csv", index=False)