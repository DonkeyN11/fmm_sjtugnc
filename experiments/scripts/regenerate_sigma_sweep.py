#!/usr/bin/env python3
"""Regenerate sigma_sweep.png (Figure 6) from available data sources.

Data sources:
  - sigma_sweep_full.json: full metrics (summary + bins/ROC) for all sigma
    levels (primary source; feeds panels a-f)
  - sigma_sweep_table.csv: summary metrics only (fallback for panels a-d)

Changes from previous version:
  - FMM → HMM in all legend labels
  - Reads sigma_sweep_full.json first, falls back to sigma_sweep_table.csv
  - Panels (e-f) fallback order: sigma_10 → sigma_10_sr1 → sigma_15_sr1
  - Handles n=0 entries gracefully
  - Panel (a) plots median error with IQR (P25-P75) error bars; quartiles are
    read from point_error_p25/point_error_p75 in the JSON when present, else
    recomputed from the raw result CSVs + ground truth in data/simulation/
  - Saves both PNG and SVG versions
"""
import csv, json, math, re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT / "experiments/output/3_full_matching"
FULL_JSON = OUTPUT_DIR / "sigma_sweep_full.json"
SUMMARY_CSV = OUTPUT_DIR / "sigma_sweep_table.csv"
FIGS_DIR = PROJECT / "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
COLOR_CMM = "#2166ac"
COLOR_HMM = "#b2182b"
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

# ── Load data: sigma_sweep_full.json first, fall back to sigma_sweep_table.csv ──
cmm_data = {}    # label -> {metric: value}  (panels a-d)
hmm_data = {}
detail_cmm = {}  # label -> full metrics dict with bins/ROC (panels e-f)
detail_hmm = {}

loaded_source = None
if FULL_JSON.exists():
    with open(FULL_JSON) as f:
        full_data = json.load(f)
    for m in full_data.get("cmm", []):
        detail_cmm[m["label"]] = m
        cmm_data[m["label"]] = m
    for m in full_data.get("fmm", []):
        detail_hmm[m["label"]] = m
        hmm_data[m["label"]] = m
    if cmm_data or hmm_data:
        loaded_source = FULL_JSON.name
        print(f"Loaded {FULL_JSON.name}: {len(cmm_data)} CMM, {len(hmm_data)} HMM entries")
    else:
        print(f"WARNING: {FULL_JSON.name} has no entries — falling back to {SUMMARY_CSV.name}")

if loaded_source is None:
    if not SUMMARY_CSV.exists():
        print(f"ERROR: neither {FULL_JSON.name} nor {SUMMARY_CSV.name} found.")
        sys.exit(1)

    with open(SUMMARY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            algo = row["algorithm"].strip()
            label = row["label"].strip()
            target = cmm_data if algo == "CMM" else hmm_data
            target[label] = {
                "n": int(row["n"]) if row["n"] else 0,
                "point_error_mean": float(row["point_error_mean"]) if row["point_error_mean"] else None,
                "point_error_median": float(row["point_error_median"]) if row["point_error_median"] else None,
                "point_error_rmse": float(row["point_error_rmse"]) if row["point_error_rmse"] else None,
                "point_error_p95": float(row["point_error_p95"]) if row["point_error_p95"] else None,
                "seg_accuracy": float(row["seg_accuracy"]) if row["seg_accuracy"] else None,
                "ece_tw": float(row["ece_tw"]) if row["ece_tw"] else None,
                "ece_ep": float(row["ece_ep"]) if row["ece_ep"] else None,
                "roc_auc": float(row["roc_auc"]) if row["roc_auc"] else None,
            }

    print(f"Loaded {SUMMARY_CSV.name}: {len(cmm_data)} CMM, {len(hmm_data)} HMM entries")

# Determine sigma levels (sorted by numeric value, filter out sr variants)
sigma_labels_cmm = sorted(
    [k for k in cmm_data.keys() if not "_sr" in k and k.startswith("sigma_")],
    key=lambda s: int(s.replace("sigma_", ""))
)
sigma_labels_hmm = sorted(
    [k for k in hmm_data.keys() if not "_sr" in k and k.startswith("sigma_")],
    key=lambda s: int(s.replace("sigma_", ""))
)
sigma_labels = sorted(set(sigma_labels_cmm) | set(sigma_labels_hmm),
                      key=lambda s: int(s.replace("sigma_", "")))
sigma_vals = [int(s.replace("sigma_", "")) for s in sigma_labels]
print(f"Sigma levels: {sigma_labels}")

# Helper: get a metric from loaded data
def csv_get(data_dict, sigma_label, key, default=np.nan):
    entry = data_dict.get(sigma_label, {})
    val = entry.get(key)
    return val if val is not None else default

# ── IQR (P25-P75) for panel (a) error bars ──
# sigma_sweep_full.json only stores mean/median/rmse/p95, so the quartiles are
# recomputed from the raw per-point errors (result CSV + ground truth) using the
# same logic as compute_metrics() in exp3_full_matching.py.
RAW_DATA_ROOT = PROJECT / "data/simulation"
_IQR_CACHE = {}


def _find_sigma_data_dir(sigma_label):
    """Locate raw dataset dir for a sigma label (mirrors exp3_full_matching.py)."""
    candidates = [
        RAW_DATA_ROOT / sigma_label / "no_occlusion" / "no_fault",
        RAW_DATA_ROOT / sigma_label,
    ]
    for d in candidates:
        if d.is_dir() and (d / "observations.csv").exists():
            return d
    return None


def _load_gt_points_by_ts(data_dir):
    """(id, ts) and (id, round(ts)) -> (x, y) from ground_truth_points.csv."""
    gt = {}
    gt_path = data_dir / "ground_truth_points.csv"
    if not gt_path.exists():
        return gt
    with open(gt_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row["id"].strip()
            ts_str = row.get("timestamp", "").strip()
            if not ts_str:
                continue
            ts = float(ts_str)
            x, y = float(row["x"]), float(row["y"])
            gt[(tid, ts)] = (x, y)
            gt[(tid, int(round(ts)))] = (x, y)
    return gt


def _parse_point(wkt):
    m = re.search(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', str(wkt), re.I)
    return (float(m.group(1)), float(m.group(2))) if m else None


def _haversine_deg_to_m(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111320.0 * math.cos(mlat)
    dy = (lat1 - lat2) * 111320.0
    return math.sqrt(dx * dx + dy * dy)


def _get_iqr(sigma_label, algo):
    """Return (p25, p75) in meters for a sigma level, or None.

    algo is the JSON dict key: 'cmm' or 'fmm'. Prefers point_error_p25/p75 from
    the loaded data; otherwise recomputes from the raw result CSV + ground truth.
    """
    key = (sigma_label, algo)
    if key in _IQR_CACHE:
        return _IQR_CACHE[key]

    entry = (cmm_data if algo == "cmm" else hmm_data).get(sigma_label, {})
    p25, p75 = entry.get("point_error_p25"), entry.get("point_error_p75")
    if p25 is not None and p75 is not None:
        _IQR_CACHE[key] = (float(p25), float(p75))
        return _IQR_CACHE[key]

    data_dir = _find_sigma_data_dir(sigma_label)
    mr_name = "cmm_result.csv" if algo == "cmm" else "fmm_result.csv"
    if data_dir is not None and (data_dir / mr_name).exists():
        gt = _load_gt_points_by_ts(data_dir)
        errors = []
        with open(data_dir / mr_name, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                tid = row.get("id", "").strip()
                ts_str = row.get("timestamp", "").strip()
                pt = _parse_point(row.get("pgeom", ""))
                if pt and ts_str:
                    try:
                        ts = float(ts_str)
                    except ValueError:
                        continue
                    g = gt.get((tid, ts)) or gt.get((tid, int(round(ts))))
                    if g is not None:
                        errors.append(_haversine_deg_to_m(pt[0], pt[1], g[0], g[1]))
        if errors:
            err = np.array(errors)
            iqr = (float(np.percentile(err, 25)), float(np.percentile(err, 75)))
            _IQR_CACHE[key] = iqr
            print(f"  IQR from raw data: {sigma_label}/{algo} "
                  f"P25={iqr[0]:.4f} m, P75={iqr[1]:.4f} m (n={len(err)})")
            return iqr

    _IQR_CACHE[key] = None
    print(f"  WARNING: no P25/P75 for {sigma_label}/{algo} — drawing without error bars")
    return None


# ── Build figure ──
fig, axes = plt.subplots(3, 2, figsize=(7.5, 10.5))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

# (a) Point error — median with IQR (P25-P75) error bars
for algo, fmt, color, data in [
    ("cmm", "o-", COLOR_CMM, cmm_data),
    ("fmm", "s--", COLOR_HMM, hmm_data),
]:
    meds = [csv_get(data, s, "point_error_median") for s in sigma_labels]
    iqrs = [_get_iqr(s, algo) for s in sigma_labels]
    # Asymmetric yerr = [median-P25, P75-median]; mask levels without data
    yerr = np.ma.masked_invalid(np.array([
        [m - i[0] if i else np.nan for m, i in zip(meds, iqrs)],
        [i[1] - m if i else np.nan for m, i in zip(meds, iqrs)],
    ]))
    ax1.errorbar(sigma_vals, meds, yerr=yerr, fmt=fmt, color=color,
                 lw=1.2, ms=5, capsize=2.5, ecolor=color, elinewidth=1.0,
                 label="CaMM" if algo == "cmm" else "HMM")
ax1.set_xlabel(r"$\sigma_{\rho}$ (m)")
ax1.set_ylabel("Median error (m)")
ax1.set_title("(a) Point Error (Median ± IQR)")
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 32)

# (b) Segment accuracy
ax2.plot(sigma_vals, [csv_get(cmm_data, s, "seg_accuracy", 0) * 100 for s in sigma_labels],
         "o-", color=COLOR_CMM, lw=1.2, ms=5, label="CaMM")
ax2.plot(sigma_vals, [csv_get(hmm_data, s, "seg_accuracy", 0) * 100 for s in sigma_labels],
         "s--", color=COLOR_HMM, lw=1.2, ms=5, label="HMM")
ax2.set_xlabel(r"$\sigma_{\rho}$ (m)")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("(b) Segment Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 32)
ax2.set_ylim(0, 105)

# (c) ECE
ax3.plot(sigma_vals, [csv_get(cmm_data, s, "ece_tw") for s in sigma_labels],
         "o-", color=COLOR_CMM, lw=1.2, ms=5, label="CaMM")
ax3.plot(sigma_vals, [csv_get(hmm_data, s, "ece_tw") for s in sigma_labels],
         "s--", color=COLOR_HMM, lw=1.2, ms=5, label="HMM")
ax3.set_xlabel(r"$\sigma_{\rho}$ (m)")
ax3.set_ylabel("ECE")
ax3.set_title("(c) ECE (Trustworthiness)")
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_xlim(0, 32)
ax3.set_ylim(0, 0.6)

# (d) ROC AUC
ax4.plot(sigma_vals, [csv_get(cmm_data, s, "roc_auc") for s in sigma_labels],
         "o-", color=COLOR_CMM, lw=1.2, ms=5, label="CaMM")
ax4.plot(sigma_vals, [csv_get(hmm_data, s, "roc_auc") for s in sigma_labels],
         "s--", color=COLOR_HMM, lw=1.2, ms=5, label="HMM")
ax4.axhline(0.5, color="gray", lw=0.8, ls="--")
ax4.set_xlabel(r"$\sigma_{\rho}$ (m)")
ax4.set_ylabel("AUC")
ax4.set_title("(d) ROC AUC")
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_xlim(0, 32)
ax4.set_ylim(0.4, 1.0)

# (e) Reliability diagram — use sigma_10 if available in detailed data,
#     fall back to sigma_10_sr1, then sigma_15_sr1
RELIABILITY_SIGMA = "sigma_10"
def _find_detail_data(detail_dict, preferred_label):
    """Find detailed data, trying preferred label first, then fallbacks."""
    if preferred_label in detail_dict:
        return detail_dict[preferred_label]
    # Try sr1 variants
    for fallback in [f"{preferred_label}_sr1", "sigma_15_sr1"]:
        if fallback in detail_dict:
            return detail_dict[fallback]
    # Try any key starting with preferred prefix
    for k, v in detail_dict.items():
        if k.startswith(preferred_label):
            return v
    return None

ax5.plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect calibration")
# Try detailed JSON first, then CSV fallback
cmm_detail = _find_detail_data(detail_cmm, RELIABILITY_SIGMA)
hmm_detail = _find_detail_data(detail_hmm, RELIABILITY_SIGMA)

reliability_plotted = False
all_ns = []
for label, detail, color in [("CaMM", cmm_detail, COLOR_CMM), ("HMM", hmm_detail, COLOR_HMM)]:
    if detail is None:
        continue
    bins = detail.get("ece_tw_bins", [])
    if bins:
        confs = [b["mean_conf"] for b in bins if b.get("n", 0) > 0]
        accs = [b["accuracy"] for b in bins if b.get("n", 0) > 0]
        ns = [b["n"] for b in bins if b.get("n", 0) > 0]
        all_ns.extend(ns)
        if confs:
            ax5.scatter(confs, accs, s=[max(n * 0.5, 10) for n in ns],
                       color=color, label=label, edgecolors="white", lw=0.5, zorder=3, alpha=0.8)
            ax5.plot(confs, accs, "-", color=color, lw=1.0, alpha=0.4)
            reliability_plotted = True

# Add sample-size (bubble size) legend
if all_ns:
    max_n = max(all_ns)
    small_samp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                               markersize=5, label=f'< {max_n//4:.0f}')
    medium_samp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                                markersize=8, label=f'{max_n//4:.0f} – {max_n//2:.0f}')
    large_samp = mlines.Line2D([], [], color='gray', marker='o', linestyle='None',
                               markersize=12, label=f'> {max_n//2:.0f}')
    legend_size = ax5.legend(handles=[small_samp, medium_samp, large_samp],
                             title="Sample Size (n)", loc='lower right',
                             fontsize=5.5, title_fontsize=6)
    ax5.add_artist(legend_size)

# Fallback: if no detailed bins, show note
if not reliability_plotted:
    ax5.text(0.5, 0.5, "No detailed bin data available", ha="center", va="center",
             transform=ax5.transAxes, fontsize=10, color="gray")

# Determine which sigma was actually used for the panel title
used_sigma_label = RELIABILITY_SIGMA
if cmm_detail:
    used_sigma_label = cmm_detail.get("label", RELIABILITY_SIGMA)
ax5.set_xlabel("Confidence")
ax5.set_ylabel("Accuracy")
ax5.set_title(f"(e) Reliability ({used_sigma_label})")
ax5.legend(fontsize=7)
ax5.grid(alpha=0.3)

# (f) ROC curves
ax6.plot([0, 1], [0, 1], "k--", lw=0.8)
roc_plotted = False
for label, detail, color in [("CaMM", cmm_detail, COLOR_CMM), ("HMM", hmm_detail, COLOR_HMM)]:
    if detail is None:
        continue
    fpr = detail.get("fpr")
    tpr = detail.get("tpr")
    auc = detail.get("roc_auc", np.nan)
    if fpr and tpr:
        ax6.plot(fpr, tpr, color=color, lw=1.2, label=f"{label} (AUC={auc:.3f})")
        roc_plotted = True
    elif auc is not None and not np.isnan(auc):
        # At least note the AUC even without the curve
        pass

if not roc_plotted:
    ax6.text(0.5, 0.5, "No ROC curve data available", ha="center", va="center",
             transform=ax6.transAxes, fontsize=10, color="gray")
ax6.set_xlabel("FPR")
ax6.set_ylabel("TPR")
ax6.set_title(f"(f) ROC ({used_sigma_label})")
ax6.legend(fontsize=7)
ax6.grid(alpha=0.3)

fig.suptitle("CaMM vs HMM: Sigma Sensitivity ($k=16$)", fontsize=13, fontweight="bold")
fig.tight_layout()
out = FIGS_DIR / "sigma_sweep.png"
out_svg = FIGS_DIR / "sigma_sweep.svg"
fig.savefig(out, dpi=DPI)
fig.savefig(out_svg)
plt.close(fig)
print(f"Saved {out}")
print(f"Saved {out_svg}")
print(f"  PNG file size: {out.stat().st_size // 1024} KB")
print(f"  SVG file size: {out_svg.stat().st_size // 1024} KB")
