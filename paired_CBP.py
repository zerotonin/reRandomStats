# ---------- Paired CBP (Bouton vs Axon within wildtype) ----------

import numpy as np
import math
from scipy.stats import ttest_1samp
from itertools import product
from frites.workflow import WfStats
from frites.stats import cluster_threshold, cluster_correction_mcp

def cbp_two_group_time_paired(X_bouton, X_axon, alpha=0.05, tail=0, tfce=True,
                              n_perm='auto', random_state=0):
    """
    Paired cluster-based permutation test along time
    (Bouton vs Axon within the same flies).

    X_bouton, X_axon: (n_flies, n_times) aligned to the SAME fly order
    Returns:
      p_time  : (n_times,) FWER-corrected p-values per time
      sig_int : list of (start_idx, end_idx) contiguous significant spans
      t_obs   : (n_times,) paired t-statistic of differences
      thr_spec: 'tfce' or float threshold used
    """
    rng = np.random.default_rng(random_state)
    X_bouton = np.asarray(X_bouton, float)
    X_axon   = np.asarray(X_axon,   float)
    assert X_bouton.shape == X_axon.shape
    n_flies, n_times = X_bouton.shape

    # 1) observed paired differences & paired t per time
    D = X_bouton - X_axon                                  # (n_flies, n_times)
    t_obs = ttest_1samp(D, popmean=0.0, axis=0, nan_policy='omit').statistic
    t_obs = np.nan_to_num(t_obs, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) sign-flip permutations on paired differences
    total = 2 ** n_flies
    if n_perm == 'auto':
        n_perm = total if total <= 50000 else 5000

    t_perm = np.empty((n_perm, n_times), float)
    if n_perm == total:  # enumerate all Rademacher sign patterns
        flips = np.array(list(product([-1, 1], repeat=n_flies)), dtype=float)  # (2^n, n_flies)
        for i in range(n_perm):
            Di = flips[i][:, None] * D
            t_perm[i] = ttest_1samp(Di, 0.0, axis=0, nan_policy='omit').statistic
    else:  # Monte Carlo sign flips
        for i in range(n_perm):
            s = rng.choice((-1.0, 1.0), size=(n_flies, 1))
            Di = s * D
            t_perm[i] = ttest_1samp(Di, 0.0, axis=0, nan_policy='omit').statistic

    t_perm = np.nan_to_num(t_perm, nan=0.0, posinf=0.0, neginf=0.0)

    # 3) reshape for frites (ROI axis first)
    x   = t_obs[None, :]          # (1, n_times)
    x_p = t_perm[:, None, :]      # (n_perm, 1, n_times)

    # 4) TFCE (preferred) with safe fallback to numeric threshold
    p_time = None; thr_spec = None
    if tfce:
        try:
            wf = WfStats()
            pvals, _ = wf.fit(
                effect=[x], perms=[x_p],
                inference='rfx', mcp='cluster', tail=tail,
                cluster_th='tfce', cluster_alpha=alpha,
                ttested=True
            )
            p_time  = pvals[:, 0]
            thr_spec = 'tfce'
        except Exception:
            thr = cluster_threshold(x, x_p, alpha=alpha, tail=tail, tfce=False)  # float
            p   = cluster_correction_mcp(x, x_p, th=thr, tail=tail)              # (1, n_times)
            p_time  = p[0]
            thr_spec = float(thr)
    else:
        thr = cluster_threshold(x, x_p, alpha=alpha, tail=tail, tfce=False)
        p   = cluster_correction_mcp(x, x_p, th=thr, tail=tail)
        p_time  = p[0]
        thr_spec = float(thr)

    # 5) contiguous significant intervals
    sig = p_time < alpha
    sig_int = []
    if sig.any():
        edges  = np.diff(np.concatenate(([False], sig, [False])).astype(int))
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]
        sig_int = list(zip(starts, ends))  # end exclusive

    return p_time, sig_int, t_obs, thr_spec


# ---------- helpers to prep matrices (WT Bouton vs Axon) ----------

def common_time_axis_structs(df, genotype, compartment, structures=('Bouton', 'Axon'), window=None):
    """Intersection of times across the given structures within one genotype."""
    times = None
    for s in structures:
        t_s = np.sort(df[(df['genotype'] == genotype) &
                         (df['structure'] == s) &
                         (df['compartment'] == compartment)]['time_s'].unique())
        times = t_s if times is None else np.intersect1d(times, t_s)
    if window is not None:
        tmin, tmax = window
        if tmin is not None: times = times[times >= tmin]
        if tmax is not None: times = times[times <= tmax]
    return np.asarray(times)

def fly_matrix(df, genotype, structure, compartment, time_axis):
    """
    Build (n_flies, n_times) by averaging all ROIs of a fly at each time.
    """
    d = df[(df['genotype'] == genotype) &
           (df['structure'] == structure) &
           (df['compartment'] == compartment)][['fly_id', 'time_s', 'dff']]
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

def align_by_fly(X, ids, target_ids):
    """Reorder rows of X to match target_ids (intersection already taken)."""
    pos = {fid: i for i, fid in enumerate(ids)}
    order = [pos[fid] for fid in target_ids]
    return X[order, :]


# ---------- DRIVER: Figure 1 (WT Bouton vs Axon) ----------

import pandas as pd
from pathlib import Path

df = pd.read_csv("data/fig_1_camp_long.csv")

genotype     = 'dnc-wt'
compartments = ['g1','g2','g3','g4','g5']     # test all γ-lobes
time_window  = (-10.0, 90.0)                  # adjust as needed

alpha    = 0.05
tail     = 0
n_perm   = 'auto'
rng_seed = 0

rows = []
for comp in compartments:
    # 1) common time grid across Bouton & Axon (WT only)
    t = common_time_axis_structs(df, genotype=genotype, compartment=comp,
                                 structures=('Bouton','Axon'), window=time_window)
    if t.size == 0:
        print(f"[{comp}] No overlapping times between Bouton and Axon in WT within the window.")
        continue

    # 2) fly-level matrices per structure
    X_b, ids_b = fly_matrix(df, genotype, 'Bouton', comp, t)
    X_a, ids_a = fly_matrix(df, genotype, 'Axon',   comp, t)

    # 3) restrict to flies present in BOTH structures (paired design)
    common_ids = np.intersect1d(ids_b, ids_a)
    if common_ids.size < 2:
        print(f"[{comp}] Not enough matched flies with both Bouton & Axon (n={common_ids.size}).")
        continue
    Xb = align_by_fly(X_b, ids_b, common_ids)
    Xa = align_by_fly(X_a, ids_a, common_ids)

    # 4) paired CBP (TFCE, sign-flip permutations)
    p_time, sig_int, t_obs, thr = cbp_two_group_time_paired(
        Xb, Xa, alpha=alpha, tail=tail, tfce=True,
        n_perm=n_perm, random_state=rng_seed
    )

    # 5) summarize clusters
    if sig_int:
        print(f"[{comp}] significant WT Bouton vs Axon windows (alpha={alpha}):")
    else:
        print(f"[{comp}] no significant WT Bouton vs Axon windows (alpha={alpha})")
    for (s, e) in sig_int:
        cluster_p = float(np.nanmin(p_time[s:e]))
        rows.append({
            'genotype': genotype,
            'contrast': 'Bouton_vs_Axon',
            'compartment': comp,
            't_start_s': float(t[s]),
            't_end_s':   float(t[e-1]),
            'cluster_p': cluster_p,
            # direction: + means Bouton > Axon, - means Axon > Bouton
            'direction': np.sign(np.nanmean(Xb[:, s:e]) - np.nanmean(Xa[:, s:e]))
        })
        print(f"  {t[s]:7.3f}s → {t[e-1]:7.3f}s   (p≈{cluster_p:.4g})")

# 6) save results
Path("out").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("out/fig1_cbp_results.csv", index=False)

# Optional: per-time p-curves
pt = []
for comp in compartments:
    t = common_time_axis_structs(df, genotype, comp, ('Bouton','Axon'), time_window)
    if t.size == 0: continue
    X_b, ids_b = fly_matrix(df, genotype, 'Bouton', comp, t)
    X_a, ids_a = fly_matrix(df, genotype, 'Axon',   comp, t)
    common_ids = np.intersect1d(ids_b, ids_a)
    if common_ids.size < 2: continue
    Xb = align_by_fly(X_b, ids_b, common_ids)
    Xa = align_by_fly(X_a, ids_a, common_ids)
    p_time, sig_int, *_ = cbp_two_group_time_paired(Xb, Xa, alpha=alpha, tail=tail,
                                                    tfce=True, n_perm=n_perm, random_state=rng_seed)
    pt.append(pd.DataFrame({'genotype': genotype, 'contrast': 'Bouton_vs_Axon',
                            'compartment': comp, 'time_s': t, 'p_fwer': p_time}))
if pt:
    pd.concat(pt, ignore_index=True).to_csv("out/fig1_per_time_p.csv", index=False)
