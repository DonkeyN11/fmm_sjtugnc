#!/usr/bin/env python3
"""
Exp6: Real-vehicle matching accuracy — CMM vs FMM on traj 11.

Uses GT road segments (ground_truth.csv) and RTK positions (ground_truth_points.csv).

Metrics per epoch:
  - Edge match: CMM/FMM cpath == GT edge_id (reverse-edge aware)
  - Position error: Haversine distance between matched pgeom and RTK position
  - Epochs with edge_id=0 (no-road) or missing RTK are excluded

Usage:
  python experiments/scripts/exp6_real_accuracy.py
"""

import csv, json, math, re, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 200
COLOR_CMM = "#2166ac"
COLOR_FMM = "#b2182b"
plt.rcParams.update({"font.size":8, "axes.labelsize":9, "axes.titlesize":10,
    "legend.fontsize":7, "xtick.labelsize":7, "ytick.labelsize":7,
    "figure.dpi":DPI, "savefig.dpi":DPI, "savefig.bbox":"tight"})

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/real_data"
OUT_DIR = PROJECT_ROOT / "output/exp6_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load reverse edge map ──
REVERSE_MAP = {}
rev_path = PROJECT_ROOT / "config/reverse_edge_map.json"
if rev_path.exists():
    with open(rev_path) as f:
        REVERSE_MAP = json.load(f)


def edge_match(matched, truth):
    return str(matched) == str(truth) or REVERSE_MAP.get(str(matched)) == str(truth)


def parse_point(wkt):
    m = re.search(r"POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)", str(wkt), re.I)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def haversine_m(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2)
    dx = (lon1 - lon2) * 111320 * math.cos(mlat)
    dy = (lat1 - lat2) * 111320
    return math.sqrt(dx * dx + dy * dy)


def to_float(v):
    try: return float(v)
    except: return None


# ── Load GT edge IDs by unified seq (from aligned.csv which has uni_seq) ──
print("Loading aligned data...")
gt_by_uniseq = {}  # (id, uni_seq) -> edge_id
rtk_by_uniseq = {}  # (id, uni_seq) -> (rtk_lon, rtk_lat)
with open(DATA_DIR / "aligned.csv", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        tid = row["id"].strip()
        useq = int(row["uni_seq"])
        # GT edge (from ground_truth.csv, matched by timestamp in align script)
        # We need to load GT separately and match by timestamp
        pass

# ── Load GT edges (ground_truth.csv) matched by (id, seq) ──
# Note: GT seq matches observation seq (monotonic, no sub-trajectory reset)
print("Loading GT edges...")
gt_edges = {}
with open(DATA_DIR / "ground_truth.csv", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        key = (row["id"].strip(), int(row["seq"]))
        gt_edges[key] = row["edge_id"].strip()

# ── Load RTK positions (ground_truth_points.csv) by (id, timestamp_norm) ──
print("Loading RTK positions...")
rtk_pos = {}
with open(DATA_DIR / "ground_truth_points.csv", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        ts = str(round(float(row["timestamp"])))
        rtk_pos[(row["id"].strip(), ts)] = (float(row["x"]), float(row["y"]))

# ── Process aligned.csv (timestamp-joined, correct seq for all sources) ──
print("Processing aligned data...")
cmm_errors, cmm_correct = [], []
cmm_trusts, cmm_labels = [], []
fmm_errors, fmm_correct = [], []
fmm_trusts, fmm_labels = [], []

with open(DATA_DIR / "aligned.csv", newline="") as f:
    for row in csv.DictReader(f, delimiter=";"):
        tid = row["id"].strip()
        useq = int(row["uni_seq"])
        ts_norm = str(round(float(row.get("timestamp", "0"))))

        # ── CMM ──
        cmm_x = row.get("cmm_x", "")
        cmm_tw = to_float(row.get("cmm_tw", "0"))
        cmm_cpath = row.get("cmm_cpath", "").strip()

        if (tid, useq) in gt_edges and gt_edges[(tid, useq)] != "0" and cmm_x:
            gt_eid = gt_edges[(tid, useq)]
            correct = edge_match(cmm_cpath, gt_eid)
            cmm_correct.append(1.0 if correct else 0.0)
            cmm_trusts.append(cmm_tw if cmm_tw is not None else 0.5)
            cmm_labels.append(1.0 if correct else 0.0)

            # Position error vs RTK
            rtk_key = (tid, ts_norm)
            if rtk_key in rtk_pos:
                rtk_lon, rtk_lat = rtk_pos[rtk_key]
                cmm_lon, cmm_lat = float(cmm_x), float(row["cmm_y"])
                err = haversine_m(cmm_lon, cmm_lat, rtk_lon, rtk_lat)
                cmm_errors.append(err)

        # ── FMM ──
        fmm_x = row.get("fmm_x", "")
        fmm_tw = to_float(row.get("fmm_tw", "0"))
        fmm_cpath = row.get("fmm_cpath", "").strip()

        if (tid, useq) in gt_edges and gt_edges[(tid, useq)] != "0" and fmm_x:
            gt_eid = gt_edges[(tid, useq)]
            correct = edge_match(fmm_cpath, gt_eid)
            fmm_correct.append(1.0 if correct else 0.0)
            fmm_trusts.append(fmm_tw if fmm_tw is not None else 0.5)
            fmm_labels.append(1.0 if correct else 0.0)

            rtk_key = (tid, ts_norm)
            if rtk_key in rtk_pos:
                rtk_lon, rtk_lat = rtk_pos[rtk_key]
                fmm_lon, fmm_lat = float(fmm_x), float(row["fmm_y"])
                err = haversine_m(fmm_lon, fmm_lat, rtk_lon, rtk_lat)
                fmm_errors.append(err)

# ── Compute metrics ──
def compute_ece(confs, labels, n_bins=10):
    N = len(confs); ece = 0.0; mce = 0.0; per_bin = []
    for i in range(n_bins):
        lo, hi = i/n_bins, (i+1)/n_bins
        mask = (confs >= lo) & (confs < hi)
        if i == n_bins-1: mask = (confs >= lo) & (confs <= hi)
        idxs = np.where(mask)[0]
        nb = len(idxs)
        if nb == 0: continue
        mc = float(np.mean(confs[idxs])); acc = float(np.mean(labels[idxs]))
        gap = abs(mc - acc); ece += nb/N * gap; mce = max(mce, gap)
        per_bin.append({"n":nb,"mean_conf":mc,"accuracy":acc,"lo":lo,"hi":hi})
    return ece, mce, per_bin

def compute_roc_auc(labels, scores):
    order = np.argsort(scores)[::-1]; ls = labels[order]
    n_pos = np.sum(ls); n_neg = len(ls) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5, None, None
    tpr = np.cumsum(ls) / n_pos; fpr = np.cumsum(1 - ls) / n_neg
    tpr = np.concatenate([[0], tpr]); fpr = np.concatenate([[0], fpr])
    return float(np.trapz(tpr, fpr)), fpr, tpr

cmm_acc = np.mean(cmm_correct) if cmm_correct else 0
fmm_acc = np.mean(fmm_correct) if fmm_correct else 0
cmm_ece, cmm_mce, cmm_bins = compute_ece(np.array(cmm_trusts), np.array(cmm_labels))
fmm_ece, fmm_mce, fmm_bins = compute_ece(np.array(fmm_trusts), np.array(fmm_labels))
cmm_auc, cmm_fpr, cmm_tpr = compute_roc_auc(np.array(cmm_labels), np.array(cmm_trusts))
fmm_auc, fmm_fpr, fmm_tpr = compute_roc_auc(np.array(fmm_labels), np.array(fmm_trusts))

cmm_errs = np.array(cmm_errors)
fmm_errs = np.array(fmm_errors)

print(f"\n{'='*60}")
print(f"Traj 11 Real-Vehicle Results")
print(f"{'='*60}")
print(f"  Edge accuracy:        CMM={cmm_acc*100:.1f}%,  FMM={fmm_acc*100:.1f}%")
print(f"  ECE (trustworthiness): CMM={cmm_ece:.4f},       FMM={fmm_ece:.4f}")
print(f"  ROC AUC:              CMM={cmm_auc:.3f},       FMM={fmm_auc:.3f}")
print(f"  Position error (m):   CMM mean={np.mean(cmm_errs):.1f}, med={np.median(cmm_errs):.1f}, P95={np.percentile(cmm_errs,95):.1f}")
print(f"                         FMM mean={np.mean(fmm_errs):.1f}, med={np.median(fmm_errs):.1f}, P95={np.percentile(fmm_errs,95):.1f}")
print(f"  Epochs evaluated:     edge={len(cmm_correct)}, pos={len(cmm_errors)}")

# ── Plot ──
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# (a) Edge accuracy bar
ax = axes[0,0]
ax.bar(["CMM","FMM"], [cmm_acc*100, fmm_acc*100], color=[COLOR_CMM, COLOR_FMM],
       edgecolor="white", lw=0.5)
for i, v in enumerate([cmm_acc*100, fmm_acc*100]):
    ax.text(i, v+1, f"{v:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Accuracy (%)"); ax.set_title(f"(a) Segment Accuracy (n={len(cmm_correct)})")
ax.set_ylim(0, 105); ax.grid(alpha=0.3, axis="y")

# (b) Position error histogram
ax = axes[0,1]
bins = np.linspace(0, min(np.percentile(np.concatenate([cmm_errs,fmm_errs]), 98), 50), 40)
ax.hist(cmm_errs, bins=bins, alpha=0.5, color=COLOR_CMM, label=f"CMM (μ={np.mean(cmm_errs):.1f}m)")
ax.hist(fmm_errs, bins=bins, alpha=0.5, color=COLOR_FMM, label=f"FMM (μ={np.mean(fmm_errs):.1f}m)")
ax.set_xlabel("Position error (m)"); ax.set_ylabel("Count")
ax.set_title("(b) Position Error Distribution"); ax.legend(); ax.grid(alpha=0.3)

# (c) ECE bar
ax = axes[0,2]
ax.bar(["CMM","FMM"], [cmm_ece, fmm_ece], color=[COLOR_CMM, COLOR_FMM],
       edgecolor="white", lw=0.5)
for i, v in enumerate([cmm_ece, fmm_ece]):
    ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_ylabel("ECE"); ax.set_title("(c) Expected Calibration Error"); ax.grid(alpha=0.3, axis="y")

# (d) Reliability diagram
ax = axes[1,0]
ax.plot([0,1],[0,1],"k--",lw=0.8)
for label, bins, color in [("CMM",cmm_bins,COLOR_CMM),("FMM",fmm_bins,COLOR_FMM)]:
    confs=[b["mean_conf"] for b in bins if b["n"]>0]
    accs=[b["accuracy"] for b in bins if b["n"]>0]
    if confs: ax.plot(confs, accs, "o-", color=color, lw=1.2, ms=5, label=f"{label} (ECE={cmm_ece if label=='CMM' else fmm_ece:.3f})")
ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
ax.set_title("(d) Reliability Diagram"); ax.legend(); ax.grid(alpha=0.3)

# (e) ROC curves
ax = axes[1,1]
ax.plot([0,1],[0,1],"k--",lw=0.8)
if cmm_fpr is not None: ax.plot(cmm_fpr, cmm_tpr, color=COLOR_CMM, lw=1.5, label=f"CMM (AUC={cmm_auc:.3f})")
if fmm_fpr is not None: ax.plot(fmm_fpr, fmm_tpr, color=COLOR_FMM, lw=1.5, label=f"FMM (AUC={fmm_auc:.3f})")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("(e) ROC Curves"); ax.legend(); ax.grid(alpha=0.3)

# (f) Error CDF
ax = axes[1,2]
for errs, color, label in [(cmm_errs,COLOR_CMM,"CMM"),(fmm_errs,COLOR_FMM,"FMM")]:
    se = np.sort(errs); cdf = np.arange(1,len(se)+1)/len(se)
    ax.plot(se, cdf, color=color, lw=1.5, label=f"{label}")
ax.axhline(0.5, color="gray", lw=0.8, ls="--")
ax.axhline(0.95, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Position error (m)"); ax.set_ylabel("CDF")
ax.set_title("(f) Error CDF"); ax.legend(); ax.grid(alpha=0.3)
ax.set_xlim(0, np.percentile(np.concatenate([cmm_errs,fmm_errs]), 99))

fig.suptitle("Traj 11 Real-Vehicle: CMM vs FMM (k=16)", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "traj11_accuracy.png", dpi=DPI)
plt.close(fig)
print(f"\nSaved: {OUT_DIR}/traj11_accuracy.png")

# ── Save stats CSV ──
with open(OUT_DIR / "traj11_stats.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric","CMM","FMM"])
    w.writerow(["edge_accuracy", f"{cmm_acc*100:.2f}%", f"{fmm_acc*100:.2f}%"])
    w.writerow(["ece", f"{cmm_ece:.4f}", f"{fmm_ece:.4f}"])
    w.writerow(["mce", f"{cmm_mce:.4f}", f"{fmm_mce:.4f}"])
    w.writerow(["roc_auc", f"{cmm_auc:.4f}", f"{fmm_auc:.4f}"])
    w.writerow(["pos_err_mean_m", f"{np.mean(cmm_errs):.2f}", f"{np.mean(fmm_errs):.2f}"])
    w.writerow(["pos_err_median_m", f"{np.median(cmm_errs):.2f}", f"{np.median(fmm_errs):.2f}"])
    w.writerow(["pos_err_p95_m", f"{np.percentile(cmm_errs,95):.2f}", f"{np.percentile(fmm_errs,95):.2f}"])
    w.writerow(["n_edge_eval", str(len(cmm_correct)), str(len(fmm_correct))])
    w.writerow(["n_pos_eval", str(len(cmm_errors)), str(len(fmm_errors))])
print(f"Saved: {OUT_DIR}/traj11_stats.csv")
