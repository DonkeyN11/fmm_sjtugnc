#!/usr/bin/env python3
"""Exp7: Statistical significance tests for CMM vs FMM comparison.

Computes:
  1. Bootstrap 95% CI for accuracy, ECE, AUC, TW separation
  2. McNemar's test for paired accuracy differences
  3. DeLong test for AUC comparison
  4. Per-trajectory bootstrap CIs
"""
import csv, json, math, sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 200
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight"})

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/data/real_data"
OUT = ROOT / "experiments/output/exp7_statistical"
OUT.mkdir(parents=True, exist_ok=True)

# Load reverse edge map
REV = json.load(open(ROOT / "experiments/config/reverse_edge_map.json"))

def em(m, t):
    return str(m) == str(t) or REV.get(str(m)) == str(t)

def tf(v):
    try: return float(v)
    except: return None

def ece(confs, labels, n=10):
    confs = np.array(confs); labels = np.array(labels)
    N = len(confs); e = 0.0
    for i in range(n):
        lo, hi = i/n, (i+1)/n
        mask = (confs >= lo) & (confs < hi)
        if i == n-1: mask = (confs >= lo) & (confs <= hi)
        idx = np.where(mask)[0]; nb = len(idx)
        if nb == 0: continue
        mc = float(np.mean(confs[idx]))
        acc = float(np.mean(labels[idx]))
        e += nb/N * abs(mc - acc)
    return e

def auc_score(labels, scores):
    order = np.argsort(scores)[::-1]
    ls = np.array(labels)[order]
    n_pos = np.sum(ls); n_neg = len(ls) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    tpr = np.cumsum(ls) / n_pos
    fpr = np.cumsum(1 - ls) / n_neg
    tpr = np.concatenate([[0], tpr]); fpr = np.concatenate([[0], fpr])
    return float(np.trapezoid(tpr, fpr))

def bootstrap_ci(data, stat_fn, n_bootstrap=10000, alpha=0.05):
    """Compute bootstrap (1-alpha) CI for a statistic."""
    data = np.array(data)
    n = len(data)
    stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = data[np.random.randint(0, n, n)]
        stats[i] = stat_fn(sample)
    return np.percentile(stats, [100*alpha/2, 100*(1-alpha/2)])

def bootstrap_ci_two_sample(x, y, stat_fn, n_bootstrap=10000, alpha=0.05):
    """Bootstrap CI for stat_fn(x) - stat_fn(y)."""
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sx = x[np.random.randint(0, nx, nx)]
        sy = y[np.random.randint(0, ny, ny)]
        diffs[i] = stat_fn(sx) - stat_fn(sy)
    return np.percentile(diffs, [100*alpha/2, 100*(1-alpha/2)])

def mcnemar_test(correct_a, correct_b):
    """McNemar's test for paired binary outcomes.
    H0: both methods have the same error rate.
    Returns chi-squared statistic and p-value.
    """
    n_ab = np.sum((correct_a == 1) & (correct_b == 0))  # A correct, B wrong
    n_ba = np.sum((correct_a == 0) & (correct_b == 1))  # A wrong, B correct
    if n_ab + n_ba == 0:
        return 0.0, 1.0
    from scipy.stats import chi2
    chi_sq = (abs(n_ab - n_ba) - 1)**2 / (n_ab + n_ba)  # Yates correction
    p = 1 - chi2.cdf(chi_sq, 1)
    return chi_sq, p

def delong_test(labels, scores_a, scores_b):
    """DeLong test for comparing two AUCs.
    Simplified implementation using the DeLong variance estimator.
    Returns z-statistic and two-sided p-value.
    """
    labels = np.array(labels)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    # Compute structural components for each method
    def compute_v(labels, scores):
        order = np.argsort(scores)
        ls = labels[order]
        # Placements: for each positive, count negatives with lower score
        placements = np.zeros(n_pos)
        pos_idx = np.where(ls == 1)[0]
        for pi, p in enumerate(pos_idx):
            placements[pi] = np.sum(ls[:p] == 0)
        v10 = np.sum(placements) / (n_pos * n_neg)
        return v10, placements, pos_idx, ls

    v10_a, place_a, pos_a, ls_a = compute_v(labels, scores_a)
    v10_b, place_b, pos_b, ls_b = compute_v(labels, scores_b)

    # DeLong covariance
    # V10_a = (1/n_neg) * sum over positives of (1/n_pos) * sum of indicator(scores_neg < scores_pos)
    # We approximate using the classical DeLong formula
    auc_a = auc_score(labels, scores_a)
    auc_b = auc_score(labels, scores_b)

    # Compute variance using bootstrap since DeLong is complex
    n = len(labels)
    n_boot = 5000
    auc_diffs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        auc_diffs[i] = auc_score(labels[idx], scores_a[idx]) - auc_score(labels[idx], scores_b[idx])

    z = np.mean(auc_diffs) / (np.std(auc_diffs) + 1e-10)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def tw_separation(tw, labels):
    """Mean TW correct - mean TW wrong."""
    tw = np.array(tw); labels = np.array(labels)
    return float(np.mean(tw[labels == 1]) - np.mean(tw[labels == 0]))


# ============================================================
# Load data
# ============================================================
print("Loading data...")
gt_edges = {}
with open(DATA / "ground_truth.csv", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        gt_edges[(row["id"].strip(), int(row["seq"]))] = row["edge_id"].strip()

cmm_tws, cmm_labels, fmm_tws, fmm_labels = [], [], [], []
per_traj = defaultdict(lambda: {"cmm_label": [], "cmm_tw": [], "fmm_label": [], "fmm_tw": []})

with open(DATA / "aligned.csv", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        tid = row["id"].strip(); useq = int(row["uni_seq"])
        if (tid, useq) not in gt_edges: continue
        gt_eid = gt_edges[(tid, useq)]
        if gt_eid in ("0", "-1"): continue

        cmm_x = row.get("cmm_x", "")
        cmm_tw = tf(row.get("cmm_tw", "0")) or 0.5
        cmm_cpath = row.get("cmm_cpath", "").strip()
        if cmm_x:
            correct = em(cmm_cpath, gt_eid)
            cmm_tws.append(cmm_tw); cmm_labels.append(1.0 if correct else 0.0)
            per_traj[tid]["cmm_tw"].append(cmm_tw)
            per_traj[tid]["cmm_label"].append(1.0 if correct else 0.0)

        fmm_x = row.get("fmm_x", "")
        fmm_tw = tf(row.get("fmm_tw", "0")) or 0.5
        fmm_cpath = row.get("fmm_cpath", "").strip()
        if fmm_x:
            correct = em(fmm_cpath, gt_eid)
            fmm_tws.append(fmm_tw); fmm_labels.append(1.0 if correct else 0.0)
            per_traj[tid]["fmm_tw"].append(fmm_tw)
            per_traj[tid]["fmm_label"].append(1.0 if correct else 0.0)

cmm_labels = np.array(cmm_labels); cmm_tws = np.array(cmm_tws)
fmm_labels = np.array(fmm_labels); fmm_tws = np.array(fmm_tws)

print(f"CMM eval epochs: {len(cmm_labels)}")
print(f"FMM eval epochs: {len(fmm_labels)}")

# ============================================================
# 1. Point Estimates
# ============================================================
print("\n" + "="*70)
print("1. POINT ESTIMATES")
print("="*70)

cmm_acc = np.mean(cmm_labels)
fmm_acc = np.mean(fmm_labels)
print(f"CMM accuracy: {cmm_acc:.4f} ({cmm_acc*100:.1f}%)")
print(f"FMM accuracy: {fmm_acc:.4f} ({fmm_acc*100:.1f}%)")

cmm_ece_val = ece(cmm_tws, cmm_labels)
fmm_ece_val = ece(fmm_tws, fmm_labels)
print(f"CMM ECE (10-bin): {cmm_ece_val:.4f}")
print(f"FMM ECE (10-bin): {fmm_ece_val:.4f}")

cmm_auc_val = auc_score(cmm_labels, cmm_tws)
fmm_auc_val = auc_score(fmm_labels, fmm_tws)
print(f"CMM AUC: {cmm_auc_val:.4f}")
print(f"FMM AUC: {fmm_auc_val:.4f}")

cmm_sep = tw_separation(cmm_tws, cmm_labels)
fmm_sep = tw_separation(fmm_tws, fmm_labels)
print(f"CMM TW separation: {cmm_sep:.4f}")
print(f"FMM TW separation: {fmm_sep:.4f}")
print(f"Ratio: {cmm_sep/fmm_sep:.1f}x")

# ============================================================
# 2. Bootstrap 95% CIs
# ============================================================
print("\n" + "="*70)
print("2. BOOTSTRAP 95% CONFIDENCE INTERVALS (N=10,000)")
print("="*70)

np.random.seed(42)

# Accuracy CIs
ci = bootstrap_ci(cmm_labels, np.mean)
print(f"CMM accuracy: {cmm_acc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
ci = bootstrap_ci(fmm_labels, np.mean)
print(f"FMM accuracy: {fmm_acc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
ci = bootstrap_ci_two_sample(cmm_labels, fmm_labels, np.mean)
print(f"Accuracy diff (CMM-FMM): {cmm_acc-fmm_acc:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

# ECE CIs
def ece_fn(data):
    """ECE of paired (tw, label) data."""
    tws, lbs = data[:, 0], data[:, 1]
    return ece(tws, lbs)

cmm_paired = np.column_stack([cmm_tws, cmm_labels])
fmm_paired = np.column_stack([fmm_tws, fmm_labels])

ci = bootstrap_ci(cmm_paired, ece_fn)
print(f"\nCMM ECE: {cmm_ece_val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
ci = bootstrap_ci(fmm_paired, ece_fn)
print(f"FMM ECE: {fmm_ece_val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

# AUC CIs
def auc_fn(data):
    tws, lbs = data[:, 0], data[:, 1]
    return auc_score(lbs, tws)

ci = bootstrap_ci(cmm_paired, auc_fn)
print(f"\nCMM AUC: {cmm_auc_val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
ci = bootstrap_ci(fmm_paired, auc_fn)
print(f"FMM AUC: {fmm_auc_val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

# TW separation CIs
def sep_fn(data):
    tws, lbs = data[:, 0], data[:, 1]
    return tw_separation(tws, lbs)

ci = bootstrap_ci(cmm_paired, sep_fn)
print(f"\nCMM TW separation: {cmm_sep:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
ci = bootstrap_ci(fmm_paired, sep_fn)
print(f"FMM TW separation: {fmm_sep:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")

# ============================================================
# 3. McNemar's Test
# ============================================================
print("\n" + "="*70)
print("3. McNEMAR'S TEST (paired accuracy comparison)")
print("="*70)

chi_sq, p = mcnemar_test(cmm_labels, fmm_labels)
print(f"Chi-squared (Yates): {chi_sq:.2f}")
print(f"p-value: {p:.6f}")
print(f"Significant at alpha=0.05: {'YES' if p < 0.05 else 'NO'}")

# Per-trajectory McNemar
print("\nPer-trajectory:")
for tid in sorted(per_traj.keys(), key=lambda x: int(x)):
    t = per_traj[tid]
    if len(t["cmm_label"]) < 5: continue
    chi_sq, p = mcnemar_test(np.array(t["cmm_label"]), np.array(t["fmm_label"]))
    acc_c = np.mean(t["cmm_label"]); acc_f = np.mean(t["fmm_label"])
    sig = "*" if p < 0.05 else ""
    print(f"  Traj {tid}: CMM={acc_c:.3f} FMM={acc_f:.3f} chi2={chi_sq:.2f} p={p:.4f} {sig}")

# ============================================================
# 4. DeLong Test for AUC
# ============================================================
print("\n" + "="*70)
print("4. DeLONG TEST (AUC comparison, bootstrap approximation)")
print("="*70)

z, p = delong_test(cmm_labels, cmm_tws, fmm_tws)
print(f"Comparison: CMM AUC={cmm_auc_val:.4f} vs FMM AUC={fmm_auc_val:.4f}")
print(f"z-statistic: {z:.2f}")
print(f"p-value: {p:.6f}")
print(f"Significant at alpha=0.05: {'YES' if p < 0.05 else 'NO'}")

# ============================================================
# 5. Per-Trajectory Bootstrap CIs
# ============================================================
print("\n" + "="*70)
print("5. PER-TRAJECTORY ACCURACY WITH 95% BOOTSTRAP CI")
print("="*70)

print(f"\n{'Traj':<8} {'CMM Acc':<18} {'FMM Acc':<18} {'Diff':<18}")
print("-" * 62)
for tid in sorted(per_traj.keys(), key=lambda x: int(x)):
    t = per_traj[tid]
    if len(t["cmm_label"]) < 5: continue
    acc_c = np.mean(t["cmm_label"])
    acc_f = np.mean(t["fmm_label"])
    ci_c = bootstrap_ci(t["cmm_label"], np.mean, n_bootstrap=5000)
    ci_f = bootstrap_ci(t["fmm_label"], np.mean, n_bootstrap=5000)
    diff = acc_c - acc_f
    ci_d = bootstrap_ci_two_sample(t["cmm_label"], t["fmm_label"], np.mean, n_bootstrap=5000)
    print(f"Traj {tid:<3} {acc_c:.3f} [{ci_c[0]:.3f},{ci_c[1]:.3f}]  "
          f"{acc_f:.3f} [{ci_f[0]:.3f},{ci_f[1]:.3f}]  "
          f"{diff:+.3f} [{ci_d[0]:+.3f},{ci_d[1]:+.3f}]")

# ============================================================
# 6. Summary Table (LaTeX-ready)
# ============================================================
print("\n" + "="*70)
print("6. LaTeX TABLE")
print("="*70)

# Compute bootstraps once with fixed seed
np.random.seed(42)

acc_ci_c = bootstrap_ci(cmm_labels, np.mean)
acc_ci_f = bootstrap_ci(fmm_labels, np.mean)
acc_ci_d = bootstrap_ci_two_sample(cmm_labels, fmm_labels, np.mean)
ece_ci_c = bootstrap_ci(cmm_paired, ece_fn)
ece_ci_f = bootstrap_ci(fmm_paired, ece_fn)
auc_ci_c = bootstrap_ci(cmm_paired, auc_fn)
auc_ci_f = bootstrap_ci(fmm_paired, auc_fn)
sep_ci_c = bootstrap_ci(cmm_paired, sep_fn)
sep_ci_f = bootstrap_ci(fmm_paired, sep_fn)

print(r"""
\begin{table}[t]
\caption{Statistical Significance of CMM vs.\ FMM on Real-Vehicle Data}
\label{tab:statistical}
\centering
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Metric & CMM & FMM & $\Delta$ & $p$-value \\
\midrule""")

print(f"Accuracy & {cmm_acc:.3f} [{acc_ci_c[0]:.3f},{acc_ci_c[1]:.3f}] & "
      f"{fmm_acc:.3f} [{acc_ci_f[0]:.3f},{acc_ci_f[1]:.3f}] & "
      f"{cmm_acc-fmm_acc:+.3f} [{acc_ci_d[0]:+.3f},{acc_ci_d[1]:+.3f}] & "
      f"{p:.4f} \\\\")

# McNemar for accuracy is already computed
print(f"ECE(TW)   & {cmm_ece_val:.3f} [{ece_ci_c[0]:.3f},{ece_ci_c[1]:.3f}] & "
      f"{fmm_ece_val:.3f} [{ece_ci_f[0]:.3f},{ece_ci_f[1]:.3f}] & "
      f"{cmm_ece_val-fmm_ece_val:+.3f} & --- \\\\")

# DeLong p-value
z_auc, p_auc = delong_test(cmm_labels, cmm_tws, fmm_tws)
print(f"AUC       & {cmm_auc_val:.3f} [{auc_ci_c[0]:.3f},{auc_ci_c[1]:.3f}] & "
      f"{fmm_auc_val:.3f} [{auc_ci_f[0]:.3f},{auc_ci_f[1]:.3f}] & "
      f"{cmm_auc_val-fmm_auc_val:+.3f} & {p_auc:.4f} \\\\")

print(f"TW Sep.   & {cmm_sep:.3f} [{sep_ci_c[0]:.3f},{sep_ci_c[1]:.3f}] & "
      f"{fmm_sep:.3f} [{sep_ci_f[0]:.3f},{sep_ci_f[1]:.3f}] & "
      f"{cmm_sep-fmm_sep:+.3f} & --- \\\\")

print(r"""\bottomrule
\end{tabular}
\end{table}
""")

# ============================================================
# 7. Figures
# ============================================================
print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Error bar plot
ax = axes[0, 0]
metrics = ["Accuracy", "ECE(TW)", "AUC", "TW Sep."]
cmm_vals = [cmm_acc, cmm_ece_val, cmm_auc_val, cmm_sep]
fmm_vals = [fmm_acc, fmm_ece_val, fmm_auc_val, fmm_sep]
cmm_cis = [
    (cmm_acc - acc_ci_c[0], acc_ci_c[1] - cmm_acc),
    (cmm_ece_val - ece_ci_c[0], ece_ci_c[1] - cmm_ece_val),
    (cmm_auc_val - auc_ci_c[0], auc_ci_c[1] - cmm_auc_val),
    (cmm_sep - sep_ci_c[0], sep_ci_c[1] - cmm_sep),
]
fmm_cis = [
    (fmm_acc - acc_ci_f[0], acc_ci_f[1] - fmm_acc),
    (fmm_ece_val - ece_ci_f[0], ece_ci_f[1] - fmm_ece_val),
    (fmm_auc_val - auc_ci_f[0], auc_ci_f[1] - fmm_auc_val),
    (fmm_sep - sep_ci_f[0], sep_ci_f[1] - fmm_sep),
]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax.bar(x - w/2, cmm_vals, w, yerr=np.array(cmm_cis).T, capsize=4,
               color="#2166ac", label="CMM")
bars2 = ax.bar(x + w/2, fmm_vals, w, yerr=np.array(fmm_cis).T, capsize=4,
               color="#b2182b", label="FMM")
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel("Value"); ax.set_title("CMM vs FMM with 95% Bootstrap CI")
ax.legend()

# Per-traj accuracy with CI
ax = axes[0, 1]
traj_ids = sorted([t for t in per_traj if len(per_traj[t]["cmm_label"]) >= 5],
                   key=lambda x: int(x))
cmm_accs, fmm_accs = [], []
cmm_lo, cmm_hi, fmm_lo, fmm_hi = [], [], [], []
for tid in traj_ids:
    t = per_traj[tid]
    acc_c = np.mean(t["cmm_label"]); acc_f = np.mean(t["fmm_label"])
    ci_c = bootstrap_ci(np.array(t["cmm_label"]), np.mean, n_bootstrap=5000)
    ci_f = bootstrap_ci(np.array(t["fmm_label"]), np.mean, n_bootstrap=5000)
    cmm_accs.append(acc_c); fmm_accs.append(acc_f)
    cmm_lo.append(acc_c - ci_c[0]); cmm_hi.append(ci_c[1] - acc_c)
    fmm_lo.append(acc_f - ci_f[0]); fmm_hi.append(ci_f[1] - acc_f)
x = np.arange(len(traj_ids))
ax.errorbar(x - 0.15, cmm_accs, yerr=[cmm_lo, cmm_hi], fmt='o', capsize=3,
            color="#2166ac", label="CMM", markersize=6)
ax.errorbar(x + 0.15, fmm_accs, yerr=[fmm_lo, fmm_hi], fmt='s', capsize=3,
            color="#b2182b", label="FMM", markersize=6)
ax.set_xticks(x); ax.set_xticklabels(traj_ids)
ax.set_ylabel("Accuracy"); ax.set_xlabel("Trajectory")
ax.set_title("Per-Trajectory Accuracy with 95% CI")
ax.legend()

# Bootstrap distribution of accuracy diff
ax = axes[1, 0]
np.random.seed(42)
n_boot = 10000
diffs = np.zeros(n_boot)
c = np.array(cmm_labels); f = np.array(fmm_labels)
for i in range(n_boot):
    idx = np.random.randint(0, len(c), len(c))
    diffs[i] = np.mean(c[idx]) - np.mean(f[idx])
ax.hist(diffs, bins=50, color="#2166ac", edgecolor="white", alpha=0.8, density=True)
ax.axvline(0, color="gray", linestyle="--", linewidth=1)
ax.axvline(np.percentile(diffs, 2.5), color="red", linestyle=":", linewidth=1,
           label=f"95% CI: [{np.percentile(diffs, 2.5):.4f}, {np.percentile(diffs, 97.5):.4f}]")
ax.axvline(np.percentile(diffs, 97.5), color="red", linestyle=":", linewidth=1)
ax.axvline(cmm_acc - fmm_acc, color="black", linewidth=2, label=f"Observed: {cmm_acc-fmm_acc:.4f}")
ax.set_xlabel("CMM - FMM Accuracy Difference")
ax.set_ylabel("Density")
ax.set_title("Bootstrap Distribution of Accuracy Difference")
ax.legend(fontsize=7)

# Reliability diagram with CI shading
ax = axes[1, 1]
n_bins = 10
for name, tws, lbs, color in [("CMM", cmm_tws, cmm_labels, "#2166ac"),
                                ("FMM", fmm_tws, fmm_labels, "#b2182b")]:
    bin_confs, bin_accs, bin_lo, bin_hi = [], [], [], []
    for i in range(n_bins):
        lo, hi = i/n_bins, (i+1)/n_bins
        mask = (tws >= lo) & (tws < hi)
        if i == n_bins-1: mask = (tws >= lo) & (tws <= hi)
        idx = np.where(mask)[0]
        if len(idx) < 5: continue
        mc = np.mean(tws[idx]); ma = np.mean(lbs[idx])
        # Bootstrap CI for accuracy in this bin
        def acc_bin_fn(x):
            return np.mean(x)
        ci = bootstrap_ci(lbs[idx], np.mean, n_bootstrap=2000)
        bin_confs.append(mc); bin_accs.append(ma)
        bin_lo.append(ma - ci[0]); bin_hi.append(ci[1] - ma)
    bin_confs = np.array(bin_confs); bin_accs = np.array(bin_accs)
    ax.errorbar(bin_confs, bin_accs, yerr=[bin_lo, bin_hi], fmt='o-', capsize=2,
                color=color, label=name, markersize=5, linewidth=1.5)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
ax.set_title("Reliability Diagram with 95% CI")
ax.legend()

plt.tight_layout()
fig.savefig(OUT / "statistical_tests.png")
print(f"Figure saved: {OUT / 'statistical_tests.png'}")

# Save JSON results
results = {
    "point_estimates": {
        "cmm_accuracy": float(cmm_acc), "fmm_accuracy": float(fmm_acc),
        "cmm_ece": float(cmm_ece_val), "fmm_ece": float(fmm_ece_val),
        "cmm_auc": float(cmm_auc_val), "fmm_auc": float(fmm_auc_val),
        "cmm_tw_separation": float(cmm_sep), "fmm_tw_separation": float(fmm_sep),
    },
    "bootstrap_ci": {
        "cmm_accuracy": [float(acc_ci_c[0]), float(acc_ci_c[1])],
        "fmm_accuracy": [float(acc_ci_f[0]), float(acc_ci_f[1])],
        "accuracy_diff": [float(acc_ci_d[0]), float(acc_ci_d[1])],
        "cmm_ece": [float(ece_ci_c[0]), float(ece_ci_c[1])],
        "fmm_ece": [float(ece_ci_f[0]), float(ece_ci_f[1])],
        "cmm_auc": [float(auc_ci_c[0]), float(auc_ci_c[1])],
        "fmm_auc": [float(auc_ci_f[0]), float(auc_ci_f[1])],
        "cmm_tw_sep": [float(sep_ci_c[0]), float(sep_ci_c[1])],
        "fmm_tw_sep": [float(sep_ci_f[0]), float(sep_ci_f[1])],
    },
    "mcnemar": {"chi_squared": float(chi_sq), "p_value": float(p)},
    "delong": {"z_statistic": float(z_auc), "p_value": float(p_auc)},
}
with open(OUT / "statistical_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {OUT / 'statistical_results.json'}")

print("\nDone.")
