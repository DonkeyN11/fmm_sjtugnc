#!/usr/bin/env python3
"""Redraw all Exp6 figures from updated aligned.csv."""
import csv, json, math, numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/real_data"
OUT = ROOT / "output/exp6_real"
OUT.mkdir(parents=True, exist_ok=True)

DPI = 200
COLOR_CMM = "#2166ac"
COLOR_FMM = "#b2182b"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight"})

REV = json.load(open(ROOT / "config/reverse_edge_map.json"))
REV = {str(k): str(v) for k, v in REV.items()}
def em(m, t):
    return str(m) == str(t) or REV.get(str(m)) == str(t)

def hdist(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2)
    dx = (lon1 - lon2) * 111320 * math.cos(mlat)
    dy = (lat1 - lat2) * 111320
    return math.sqrt(dx * dx + dy * dy)

def tf(v):
    try:
        return float(v)
    except:
        return None

def ece(confs, labels, n=10):
    N = len(confs)
    e = 0.0
    m = 0.0
    bins = []
    for i in range(n):
        lo, hi = i / n, (i + 1) / n
        mask = (confs >= lo) & (confs < hi) if i < n - 1 else (confs >= lo) & (confs <= hi)
        idx = np.where(mask)[0]
        nb = len(idx)
        if nb == 0: continue
        mc = float(np.mean(confs[idx]))
        acc = float(np.mean(labels[idx]))
        gap = abs(mc - acc)
        e += nb / N * gap
        m = max(m, gap)
        bins.append({"n": nb, "mean_conf": mc, "accuracy": acc, "lo": lo, "hi": hi})
    return e, m, bins

def auc_score(labels, scores):
    order = np.argsort(scores)[::-1]
    ls = labels[order]
    n_pos = np.sum(ls)
    n_neg = len(ls) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5, None, None
    tpr = np.cumsum(ls) / n_pos
    fpr = np.cumsum(1 - ls) / n_neg
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return float(np.trapz(tpr, fpr)), fpr, tpr

# ── Load data ──
print("Loading data...")
gt_edges = {}
with open(DATA / "ground_truth.csv", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        gt_edges[(row["id"].strip(), int(row["seq"]))] = row["edge_id"].strip()

rtk_pos = {}
with open(DATA / "ground_truth_points.csv", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        ts = str(round(float(row["timestamp"])))
        rtk_pos[(row["id"].strip(), ts)] = (float(row["x"]), float(row["y"]))

cmm_errs, cmm_tws, cmm_lbls = [], [], []
fmm_errs, fmm_tws, fmm_lbls = [], [], []
per_traj = defaultdict(lambda: {"cmm_ok": 0, "cmm_n": 0, "cmm_errs": [], "fmm_ok": 0, "fmm_n": 0, "fmm_errs": []})
traj_data = defaultdict(lambda: {"cmm": [], "fmm": [], "gt": []})  # per-traj detailed

with open(DATA / "aligned.csv", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        tid = row["id"].strip()
        useq = int(row["uni_seq"])
        ts_norm = str(round(float(row["timestamp"])))
        if (tid, useq) not in gt_edges: continue
        gt_eid = gt_edges[(tid, useq)]
        if gt_eid in ("0", "-1"): continue

        # CMM
        cmm_x = row.get("cmm_x", "")
        cmm_tw = tf(row.get("cmm_tw", "0"))
        cmm_cpath = row.get("cmm_cpath", "").strip()
        if cmm_x:
            correct = em(cmm_cpath, gt_eid)
            cmm_lbls.append(1.0 if correct else 0.0)
            cmm_tws.append(cmm_tw or 0.5)
            per_traj[tid]["cmm_ok"] += int(correct)
            per_traj[tid]["cmm_n"] += 1
            if (tid, ts_norm) in rtk_pos:
                rtk_lon, rtk_lat = rtk_pos[(tid, ts_norm)]
                e = hdist(float(cmm_x), float(row["cmm_y"]), rtk_lon, rtk_lat)
                cmm_errs.append(e)
                per_traj[tid]["cmm_errs"].append(e)
            traj_data[tid]["cmm"].append({"seq": useq, "tw": cmm_tw, "correct": correct, "cpath": cmm_cpath, "gt": gt_eid})

        # FMM
        fmm_x = row.get("fmm_x", "")
        fmm_tw = tf(row.get("fmm_tw", "0"))
        fmm_cpath = row.get("fmm_cpath", "").strip()
        if fmm_x:
            correct = em(fmm_cpath, gt_eid)
            fmm_lbls.append(1.0 if correct else 0.0)
            fmm_tws.append(fmm_tw or 0.5)
            per_traj[tid]["fmm_ok"] += int(correct)
            per_traj[tid]["fmm_n"] += 1
            if (tid, ts_norm) in rtk_pos:
                rtk_lon, rtk_lat = rtk_pos[(tid, ts_norm)]
                fmm_errs.append(hdist(float(fmm_x), float(row["fmm_y"]), rtk_lon, rtk_lat))
                per_traj[tid]["fmm_errs"].append(fmm_errs[-1])

cL = np.array(cmm_lbls)
fL = np.array(fmm_lbls)
cT = np.array(cmm_tws)
fT = np.array(fmm_tws)
cE = np.array(cmm_errs)
fE = np.array(fmm_errs)

print(f"Loaded: {len(cL)} eval epochs")

# ── Figure 1: Per-Trajectory Accuracy ──
print("Drawing fig_accuracy.png...")
fig, ax = plt.subplots(figsize=(8, 4))
tids = sorted(per_traj.keys(), key=int)
x = np.arange(len(tids))
w = 0.35
cmm_accs = [per_traj[t]["cmm_ok"] / per_traj[t]["cmm_n"] * 100 for t in tids]
fmm_accs = [per_traj[t]["fmm_ok"] / per_traj[t]["fmm_n"] * 100 for t in tids]
bars1 = ax.bar(x - w / 2, cmm_accs, w, color=COLOR_CMM, label="CMM", edgecolor="white", lw=0.5)
bars2 = ax.bar(x + w / 2, fmm_accs, w, color=COLOR_FMM, label="FMM", edgecolor="white", lw=0.5)
for i, (cv, fv) in enumerate(zip(cmm_accs, fmm_accs)):
    ax.text(i - w / 2, cv + 1, f"{cv:.0f}", ha="center", fontsize=7, fontweight="bold", color=COLOR_CMM)
    ax.text(i + w / 2, fv + 1, f"{fv:.0f}", ha="center", fontsize=7, fontweight="bold", color=COLOR_FMM)
ax.set_xticks(x)
ax.set_xticklabels([f"Traj {t}" for t in tids])
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Per-Trajectory Segment Accuracy (n={len(cL)}, CMM={np.mean(cL)*100:.1f}% vs FMM={np.mean(fL)*100:.1f}%)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
ax.set_ylim(0, 108)
fig.tight_layout()
fig.savefig(OUT / "fig_accuracy.png", dpi=DPI)
plt.close()

# ── Figure 2: Reliability Diagram ──
print("Drawing fig_calibration.png...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
# 2a: Reliability diagram
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
cmm_ece_val, cmm_mce, cmm_bins = ece(cT, cL)
fmm_ece_val, fmm_mce, fmm_bins = ece(fT, fL)
for label, bins, color, ece_v in [("CMM", cmm_bins, COLOR_CMM, cmm_ece_val), ("FMM", fmm_bins, COLOR_FMM, fmm_ece_val)]:
    confs = [b["mean_conf"] for b in bins if b["n"] > 0]
    accs = [b["accuracy"] for b in bins if b["n"] > 0]
    ns = [b["n"] for b in bins if b["n"] > 0]
    if confs:
        ax.scatter(confs, accs, s=[n * 2 for n in ns], color=color, alpha=0.7, edgecolors="white", lw=0.3, zorder=3)
        ax.plot(confs, accs, "-", color=color, lw=1.2, alpha=0.5)
ax.set_xlabel("Trustworthiness")
ax.set_ylabel("Accuracy")
ax.set_title(f"(a) Reliability (ECE: CMM={cmm_ece_val:.3f}, FMM={fmm_ece_val:.3f})")
ax.legend()
ax.grid(alpha=0.3)

# 2b: TW histogram (correct vs wrong)
ax = axes[1]
bins = np.linspace(0, 1, 31)
cmm_tw_c = cT[cL == 1]
cmm_tw_w = cT[cL == 0]
ax.hist(cmm_tw_c, bins=bins, alpha=0.6, color=COLOR_CMM, label=f"CMM correct (n={len(cmm_tw_c)})")
ax.hist(cmm_tw_w, bins=bins, alpha=0.6, color="red", label=f"CMM wrong (n={len(cmm_tw_w)})")
ax.set_xlabel("Trustworthiness")
ax.set_ylabel("Count")
ax.set_title(f"(b) TW Distribution (sep={np.mean(cmm_tw_c)-np.mean(cmm_tw_w):.3f})")
ax.legend(fontsize=6)
ax.grid(alpha=0.3)

# 2c: ECE per-bin
ax = axes[2]
labels_bin = [f"{b['lo']:.1f}" for b in cmm_bins if b["n"] > 0]
gaps = [abs(b["mean_conf"] - b["accuracy"]) for b in cmm_bins if b["n"] > 0]
ns = [b["n"] for b in cmm_bins if b["n"] > 0]
colors = [COLOR_CMM if g < 0.1 else "orange" if g < 0.2 else "red" for g in gaps]
ax.bar(range(len(gaps)), gaps, color=colors, edgecolor="white", lw=0.5)
ax.set_xticks(range(len(gaps)))
ax.set_xticklabels(labels_bin, rotation=45, fontsize=6)
ax.set_ylabel("|Conf - Acc|")
ax.set_title(f"(c) ECE per Bin (CMM)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "fig_calibration.png", dpi=DPI)
plt.close()

# ── Figure 3: ROC Comparison ──
print("Drawing fig_roc_comparison.png...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
cmm_auc, cmm_fpr, cmm_tpr = auc_score(cL, cT)
fmm_auc, fmm_fpr, fmm_tpr = auc_score(fL, fT)
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
if cmm_fpr is not None:
    ax.plot(cmm_fpr, cmm_tpr, color=COLOR_CMM, lw=1.5, label=f"CMM AUC={cmm_auc:.3f}")
if fmm_fpr is not None:
    ax.plot(fmm_fpr, fmm_tpr, color=COLOR_FMM, lw=1.5, label=f"FMM AUC={fmm_auc:.3f}")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("(a) ROC Curves")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
thresholds = [0.999, 0.99, 0.95, 0.90, 0.50]
cmm_reject = []
fmm_reject = []
cmm_keep = []
fmm_keep = []
for th in thresholds:
    cmm_pred_wrong = cT < th
    fmm_pred_wrong = fT < th
    cmm_reject.append(np.mean(cmm_pred_wrong[cL == 0]) * 100 if np.sum(cL == 0) > 0 else 0)
    fmm_reject.append(np.mean(fmm_pred_wrong[fL == 0]) * 100 if np.sum(fL == 0) > 0 else 0)
    cmm_keep.append(np.mean(~cmm_pred_wrong[cL == 1]) * 100 if np.sum(cL == 1) > 0 else 0)
    fmm_keep.append(np.mean(~fmm_pred_wrong[fL == 1]) * 100 if np.sum(fL == 1) > 0 else 0)
x = np.arange(len(thresholds))
w = 0.2
ax.bar(x - w, cmm_reject, w, color=COLOR_CMM, label="CMM: reject wrong", edgecolor="white", lw=0.3)
ax.bar(x, fmm_reject, w, color=COLOR_FMM, label="FMM: reject wrong", edgecolor="white", lw=0.3)
ax.bar(x + w, cmm_keep, w, color=COLOR_CMM, alpha=0.4, label="CMM: keep correct", edgecolor="white", lw=0.3)
ax.set_xticks(x)
ax.set_xticklabels([str(t) for t in thresholds])
ax.set_xlabel("TW Threshold")
ax.set_ylabel("%")
ax.set_title("(b) Rejection Rate by Threshold")
ax.legend(fontsize=6)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "fig_roc_comparison.png", dpi=DPI)
plt.close()

# ── Figure 4: Traj 22 TW timeline ──
print("Drawing fig_traj22_tw.png...")
td22 = traj_data.get("22", {"cmm": []})
if td22["cmm"]:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    seqs = [d["seq"] for d in td22["cmm"]]
    tws = [d["tw"] for d in td22["cmm"]]
    corrects = [d["correct"] for d in td22["cmm"]]
    ax = axes[0]
    colors = [COLOR_CMM if c else "red" for c in corrects]
    ax.scatter(seqs, tws, c=colors, s=3, alpha=0.6)
    ax.axhline(y=0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Trustworthiness")
    ax.set_title(f"Traj 22: Trustworthiness Timeline (acc={np.mean(corrects)*100:.1f}%)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    window = 50
    rolling_acc = [np.mean(corrects[max(0, i - window):i + window]) * 100 for i in range(len(corrects))]
    ax.fill_between(seqs, 0, 100, color="red", alpha=0.1)
    ax.fill_between(seqs, rolling_acc, 100, color=COLOR_CMM, alpha=0.3)
    ax.plot(seqs, rolling_acc, color=COLOR_CMM, lw=1)
    ax.set_xlabel("Sequence")
    ax.set_ylabel(f"Rolling Acc ({window}-epoch)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_traj22_tw.png", dpi=DPI)
    plt.close()

# ── Figure 5: Error CDF ──
print("Drawing reliability.png (Error CDF)...")
fig, ax = plt.subplots(figsize=(7, 4))
for errs, color, label in [(cE, COLOR_CMM, "CMM"), (fE, COLOR_FMM, "FMM")]:
    se = np.sort(errs)
    cdf = np.arange(1, len(se) + 1) / len(se)
    ax.plot(se, cdf, color=color, lw=1.5, label=f"{label} (μ={np.mean(errs):.1f}m, P95={np.percentile(errs, 95):.1f}m)")
ax.axhline(0.5, color="gray", lw=0.8, ls="--")
ax.axhline(0.95, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Position Error (m)")
ax.set_ylabel("CDF")
ax.set_title("Position Error CDF: CMM vs FMM")
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(0, np.percentile(np.concatenate([cE, fE]), 99))
fig.tight_layout()
fig.savefig(OUT / "reliability.png", dpi=DPI)
plt.close()

# ── Figure 6: TW Histogram ──
print("Drawing tw_histogram.png...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
bins = np.linspace(0, 1, 41)
ax.hist(cT, bins=bins, alpha=0.7, color=COLOR_CMM, label=f"CMM (μ={np.mean(cT):.3f}, σ={np.std(cT):.3f})")
ax.set_xlabel("Trustworthiness")
ax.set_ylabel("Count")
ax.set_title("CMM Trustworthiness Distribution")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax = axes[1]
ax.hist(fT, bins=bins, alpha=0.7, color=COLOR_FMM, label=f"FMM (μ={np.mean(fT):.3f}, σ={np.std(fT):.3f})")
ax.set_xlabel("Trustworthiness")
ax.set_ylabel("Count")
ax.set_title("FMM Trustworthiness Distribution")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "tw_histogram.png", dpi=DPI)
plt.close()

# ── Figure 7: ROC curve (single) ──
print("Drawing roc_curve.png...")
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
if cmm_fpr is not None:
    ax.plot(cmm_fpr, cmm_tpr, color=COLOR_CMM, lw=2, label=f"CMM (AUC={cmm_auc:.3f}, n={len(cL)})")
if fmm_fpr is not None:
    ax.plot(fmm_fpr, fmm_tpr, color=COLOR_FMM, lw=2, label=f"FMM (AUC={fmm_auc:.3f}, n={len(fL)})")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC: Trustworthiness as Mismatch Detector")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "roc_curve.png", dpi=DPI)
plt.close()

# ── Figure 8: Traj 11 detailed ──
print("Drawing traj11_accuracy.png...")
td11 = traj_data.get("11", {"cmm": [], "fmm": []})
if td11["cmm"]:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # Traj 11 TW timeline
    ax = axes[0, 0]
    seqs11 = [d["seq"] for d in td11["cmm"]]
    tws11 = [d["tw"] for d in td11["cmm"]]
    corrects11 = [d["correct"] for d in td11["cmm"]]
    c11 = [COLOR_CMM if c else "red" for c in corrects11]
    ax.scatter(seqs11, tws11, c=c11, s=2, alpha=0.5)
    ax.axhline(y=0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("TW")
    ax.set_title(f"Traj 11 TW (acc={np.mean(corrects11)*100:.1f}%)")
    ax.grid(alpha=0.3)
    # Error scatter
    ax = axes[0, 1]
    e11 = per_traj["11"]["cmm_errs"]
    if e11:
        ax.hist(e11, bins=40, alpha=0.7, color=COLOR_CMM)
        ax.set_xlabel("Position Error (m)")
        ax.set_title(f"CMM Error (μ={np.mean(e11):.1f}m)")
        ax.grid(alpha=0.3)
    # Per-traj summary
    ax = axes[1, 0]
    tids_all = sorted(per_traj.keys(), key=int)
    x_all = np.arange(len(tids_all))
    cmm_a = [per_traj[t]["cmm_ok"] / max(per_traj[t]["cmm_n"], 1) * 100 for t in tids_all]
    fmm_a = [per_traj[t]["fmm_ok"] / max(per_traj[t]["fmm_n"], 1) * 100 for t in tids_all]
    ax.bar(x_all - 0.2, cmm_a, 0.35, color=COLOR_CMM, label="CMM")
    ax.bar(x_all + 0.2, fmm_a, 0.35, color=COLOR_FMM, label="FMM")
    for i, (cv, fv) in enumerate(zip(cmm_a, fmm_a)):
        ax.text(i - 0.2, cv + 1, f"{cv:.0f}", ha="center", fontsize=6, color=COLOR_CMM)
        ax.text(i + 0.2, fv + 1, f"{fv:.0f}", ha="center", fontsize=6, color=COLOR_FMM)
    ax.set_xticks(x_all)
    ax.set_xticklabels([f"T{t}" for t in tids_all])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("All Trajectories")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 108)
    # Position error boxplot
    ax = axes[1, 1]
    errs_by_traj_cmm = [per_traj[t]["cmm_errs"] for t in tids_all if per_traj[t]["cmm_errs"]]
    errs_by_traj_fmm = [per_traj[t]["fmm_errs"] for t in tids_all if per_traj[t]["fmm_errs"]]
    bp1 = ax.boxplot(errs_by_traj_cmm, positions=np.arange(len(errs_by_traj_cmm)) - 0.15, widths=0.25,
                      patch_artist=True, boxprops=dict(facecolor=COLOR_CMM, alpha=0.6))
    bp2 = ax.boxplot(errs_by_traj_fmm, positions=np.arange(len(errs_by_traj_fmm)) + 0.15, widths=0.25,
                      patch_artist=True, boxprops=dict(facecolor=COLOR_FMM, alpha=0.6))
    ax.set_xticks(range(len(tids_all)))
    ax.set_xticklabels([f"T{t}" for t in tids_all])
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Position Error Distribution")
    ax.grid(alpha=0.3, axis="y")
    # TW separation summary
    ax = axes[1, 2]
    ax.axis("off")
    tw_c_cmm = np.mean(cT[cL == 1]) if np.sum(cL == 1) > 0 else 0
    tw_w_cmm = np.mean(cT[cL == 0]) if np.sum(cL == 0) > 0 else 0
    tw_c_fmm = np.mean(fT[fL == 1]) if np.sum(fL == 1) > 0 else 0
    tw_w_fmm = np.mean(fT[fL == 0]) if np.sum(fL == 0) > 0 else 0
    lines = [
        f"CMM: {np.mean(cL)*100:.1f}% accuracy",
        f"  ECE={cmm_ece_val:.4f}  AUC={cmm_auc:.3f}",
        f"  TW: correct μ={tw_c_cmm:.3f}  wrong μ={tw_w_cmm:.3f}",
        f"  TW separation: {tw_c_cmm-tw_w_cmm:.3f}",
        f"",
        f"FMM: {np.mean(fL)*100:.1f}% accuracy",
        f"  ECE={fmm_ece_val:.4f}  AUC={fmm_auc:.3f}",
        f"  TW: correct μ={tw_c_fmm:.3f}  wrong μ={tw_w_fmm:.3f}",
        f"  TW separation: {tw_c_fmm-tw_w_fmm:.3f}",
        f"",
        f"Eval epochs: {len(cL)}",
    ]
    for i, line in enumerate(lines):
        ax.text(0.1, 0.95 - i * 0.08, line, transform=ax.transAxes, fontsize=9, family="monospace")
    ax.set_title("Summary")
    # Hide empty subplot
    axes[0, 2].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "traj11_accuracy.png", dpi=DPI)
    plt.close()

# ── All done ──
print(f"\nAll figures redrawn in {OUT}/")
for p in sorted(OUT.glob("*.png")):
    print(f"  {p.name}  ({p.stat().st_size//1024} KB)")
