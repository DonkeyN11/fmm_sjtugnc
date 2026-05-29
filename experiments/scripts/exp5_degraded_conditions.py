#!/usr/bin/env python3
"""
Experiment 5: CMM vs FMM reliability under degraded conditions at sigma=30m.

Compares 4 conditions:
  1. Clean:         no occlusion, no fault
  2. Fault only:    no occlusion, with step fault (P=0.3, U(100,500)m)
  3. Occlusion:     cross-road occlusion (45 deg half-width), no fault
  4. Both:          occlusion + fault combined

Metrics: point error, segment accuracy, ECE, ROC/AUC, reliability diagram,
         trustworthiness separation (correct vs wrong edge)

Usage:
  python experiments/scripts/exp5_degraded_conditions.py \
    --cmm-bin build/cmm --fmm-bin build/fmm \
    --output-dir experiments/output/5_degraded \
    --jobs 4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})

# Condition config
CONDITIONS = {
    "clean":    ("no_occlusion", "no_fault"),
    "fault":    ("no_occlusion", "with_fault"),
    "occlusion": ("with_occlusion", "no_fault"),
    "both":     ("with_occlusion", "with_fault"),
}
CONDITION_LABELS = {
    "clean":     "Clean",
    "fault":     "Fault",
    "occlusion": "Occlusion",
    "both":      "Occlusion+Fault",
}
CONDITION_COLORS = {
    "clean":     "#4dac26",
    "fault":     "#e66101",
    "occlusion": "#5e3c99",
    "both":      "#b2182b",
}

DATA_ROOT = Path("experiments/data/sigma_30")


# ══════════════════════════════════════════════════════════════════════════════
# Config (k=16 per README)
# ══════════════════════════════════════════════════════════════════════════════

def build_cmm_xml(gps_csv: str, mr_out: str, network_shp: str, ubodt: str) -> Path:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network><file>{network_shp}</file><id>key</id><source>u</source><target>v</target></network>
    <ubodt><file>{ubodt}</file></ubodt>
    <gps>
      <file>{gps_csv}</file><id>id</id><x>x</x><y>y</y><timestamp>timestamp</timestamp>
      <sde>sde</sde><sdn>sdn</sdn><sdu>sdu</sdu>
      <sdne>sdne</sdne><sdeu>sdeu</sdeu><sdun>sdun</sdun>
      <protection_level>protection_level</protection_level>
    </gps>
    <gps_point>true</gps_point>
  </input>
  <output>
    <file>{mr_out}</file><point_mode>true</point_mode>
    <fields><seq/><timestamp/><ogeom/><cpath/><tpath/><opath/><pgeom/>
      <ep/><tp/><trustworthiness/><n_best_trustworthiness/><candidates/>
      <status/><delta_entropy/><posterior_entropy/><h0_lambda/><cumu_prob/></fields>
  </output>
  <parameters>
    <k>16</k><min_candidates>1</min_candidates><protection_level_multiplier>3</protection_level_multiplier>
    <reverse_tolerance>0.0</reverse_tolerance><normalized>false</normalized>
    <use_mahalanobis>true</use_mahalanobis><filtered>false</filtered>
    <window_length>100</window_length>
    <max_interval>180.0</max_interval><trustworthiness_threshold>0.0</trustworthiness_threshold>
    <phmi>0.00001</phmi><lag_steps>5</lag_steps>
    <phmi_pl_multiplier>1</phmi_pl_multiplier><h0_prior_log_odds>0</h0_prior_log_odds>
  </parameters>
  <other><log_level>2</log_level><use_omp>true</use_omp><step>500</step>
    <convert_to_projected>false</convert_to_projected></other>
</config>"""
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    tmp.write(xml); tmp.close()
    return Path(tmp.name)


def build_fmm_xml(gps_csv: str, mr_out: str, network_shp: str, ubodt: str) -> Path:
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network><file>{network_shp}</file><id>key</id><source>u</source><target>v</target></network>
    <ubodt><file>{ubodt}</file></ubodt>
    <gps>
      <file>{gps_csv}</file><id>id</id><geom>geom</geom><timestamp>timestamp</timestamp>
    </gps>
  </input>
  <output>
    <file>{mr_out}</file><point_mode>true</point_mode>
    <fields><seq/><timestamp/><ogeom/><cpath/><tpath/><opath/><pgeom/>
      <ep/><tp/><trustworthiness/></fields>
  </output>
  <parameters>
    <k>16</k><r>0.03</r><pf>0</pf><gps_error>0.001</gps_error>
    <reverse_tolerance>0.0</reverse_tolerance>
  </parameters>
  <other><log_level>2</log_level><use_omp>true</use_omp><step>500</step></other>
</config>"""
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    tmp.write(xml); tmp.close()
    return Path(tmp.name)


# ══════════════════════════════════════════════════════════════════════════════
# Matching runner
# ══════════════════════════════════════════════════════════════════════════════

def run_matcher(bin_path: Path, xml_path: Path, cwd: Path, timeout: int = 600) -> bool:
    try:
        subprocess.run([str(bin_path), str(xml_path)], check=True,
                       capture_output=True, text=True, cwd=str(cwd), timeout=timeout)
        return True
    except subprocess.CalledProcessError:
        return False
    finally:
        try: os.unlink(xml_path.name)
        except OSError: pass


def convert_to_fmm_trajectory(obs_csv: Path, traj_csv: Path):
    from collections import defaultdict
    trajs = defaultdict(list)
    with open(obs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            trajs[row["id"]].append(row)
    with open(traj_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "geom", "timestamp"])
        for tid, rows in sorted(trajs.items(), key=lambda x: int(x[0])):
            points = [(float(r["x"]), float(r["y"])) for r in rows]
            geom = "LINESTRING(" + ",".join(f"{x} {y}" for x, y in points) + ")"
            tss = ",".join(r["timestamp"] for r in rows)
            w.writerow([tid, geom, tss])


def match_condition(data_dir: Path, cmm_bin: Path, fmm_bin: Path,
                    network_shp: str, ubodt: str, project_root: Path
                    ) -> Dict[str, Optional[Path]]:
    obs_csv = data_dir / "observations.csv"
    if not obs_csv.exists():
        return {"cmm": None, "fmm": None}

    results = {"cmm": None, "fmm": None}

    cmm_out = str((data_dir / "cmm_result.csv").resolve())
    if not (data_dir / "cmm_result.csv").exists():
        xml = build_cmm_xml(str(obs_csv.resolve()), cmm_out, network_shp, ubodt)
        if run_matcher(cmm_bin, xml, project_root):
            results["cmm"] = data_dir / "cmm_result.csv"

    fmm_out = str((data_dir / "fmm_result.csv").resolve())
    if not (data_dir / "fmm_result.csv").exists():
        traj_csv = data_dir / "fmm_trajectory.csv"
        if not traj_csv.exists():
            convert_to_fmm_trajectory(obs_csv, traj_csv)
        xml = build_fmm_xml(str(traj_csv.resolve()), fmm_out, network_shp, ubodt)
        if run_matcher(fmm_bin, xml, project_root):
            results["fmm"] = data_dir / "fmm_result.csv"

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Ground truth (timestamp-based)
# ══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(data_dir: Path) -> Tuple[Dict, Dict, Dict, Dict]:
    gt_points_by_seq = {}
    gt_edges_by_seq = {}
    gt_points_by_ts = {}
    gt_seq_by_ts = {}

    gt_path = data_dir / "ground_truth_points.csv"
    if gt_path.exists():
        with open(gt_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                tid = row["id"].strip()
                seq = int(row["seq"])
                ts_val = row.get("timestamp", "").strip()
                x, y = float(row["x"]), float(row["y"])
                gt_points_by_seq[(tid, seq)] = (x, y)
                if ts_val:
                    ts = float(ts_val)
                    ts_int = int(round(ts))
                    gt_points_by_ts[(tid, ts)] = (x, y)
                    gt_points_by_ts[(tid, ts_int)] = (x, y)
                    gt_seq_by_ts[(tid, ts)] = seq
                    gt_seq_by_ts[(tid, ts_int)] = seq

    edge_path = data_dir / "ground_truth.csv"
    if edge_path.exists():
        with open(edge_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                tid = row["id"].strip()
                if "point_edge_ids" in row:
                    try:
                        edges = json.loads(row["point_edge_ids"])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    for seq, eid in enumerate(edges):
                        gt_edges_by_seq[(tid, seq)] = str(eid)

    return gt_points_by_seq, gt_edges_by_seq, gt_points_by_ts, gt_seq_by_ts


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def parse_point(wkt):
    m = re.search(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', str(wkt), re.I)
    return (float(m.group(1)), float(m.group(2))) if m else None


def haversine_deg_to_m(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111320.0 * math.cos(mlat)
    dy = (lat1 - lat2) * 111320.0
    return math.sqrt(dx * dx + dy * dy)


def load_reverse_map():
    map_path = Path(__file__).resolve().parents[1] / "config" / "reverse_edge_map.json"
    if map_path.exists():
        with open(map_path) as f: return json.load(f)
    return {}


_REV = None


def edge_match(matched, truth):
    global _REV
    if _REV is None: _REV = load_reverse_map()
    return str(matched) == str(truth) or _REV.get(str(matched)) == str(truth)


def _to_float(val):
    try: return float(val)
    except: return None


def compute_ece(confidences, labels, n_bins=10):
    N = len(confidences)
    per_bin = []
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1: mask = (confidences >= lo) & (confidences <= hi)
        idxs = np.where(mask)[0]
        nb = len(idxs)
        if nb == 0: continue
        mc = float(np.mean(confidences[idxs]))
        acc = float(np.mean(labels[idxs]))
        gap = abs(mc - acc)
        ece += nb / N * gap; mce = max(mce, gap)
        per_bin.append({"n": nb, "mean_conf": mc, "accuracy": acc, "lo": lo, "hi": hi})
    return ece, mce, per_bin


def compute_roc_auc(labels, scores):
    order = np.argsort(scores)[::-1]
    ls = labels[order]
    n_pos = np.sum(ls); n_neg = len(ls) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5, None, None
    tpr = np.cumsum(ls) / n_pos; fpr = np.cumsum(1 - ls) / n_neg
    tpr = np.concatenate([[0], tpr]); fpr = np.concatenate([[0], fpr])
    return float(np.trapz(tpr, fpr)), fpr, tpr


def compute_metrics(data_dir, mr_csv, gt_points_by_seq, gt_edges_by_seq,
                    gt_points_by_ts, gt_seq_by_ts, label):
    if not mr_csv or not mr_csv.exists():
        return {"label": label, "error": "no result", "n": 0}

    errors, trusts, eps_list = [], [], []
    n_total = 0
    edge_labels, edge_trusts = [], []
    corr_trusts, mis_trusts = [], []

    with open(mr_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row.get("id", "").strip()
            ts_str = row.get("timestamp", "").strip()
            pgeom = row.get("pgeom", "")
            cpath = row.get("cpath", "").strip()
            trust = _to_float(row.get("trustworthiness", "0"))
            ep = _to_float(row.get("ep", "0"))

            n_total += 1
            if trust is not None: trusts.append(trust)
            if ep is not None: eps_list.append(ep)

            ts = None; ts_int = None
            if ts_str:
                try:
                    ts = float(ts_str); ts_int = int(round(ts))
                except ValueError: ts = ts_str

            pt = parse_point(pgeom)
            if pt and ts is not None:
                ts_key = (tid, ts)
                if ts_key in gt_points_by_ts:
                    gt_x, gt_y = gt_points_by_ts[ts_key]
                elif ts_int is not None and (tid, ts_int) in gt_points_by_ts:
                    gt_x, gt_y = gt_points_by_ts[(tid, ts_int)]
                else:
                    gt_x = gt_y = None
                if gt_x is not None:
                    err = haversine_deg_to_m(pt[0], pt[1], gt_x, gt_y)
                    errors.append(err)

            if cpath and ts is not None:
                ts_key = (tid, ts)
                if ts_key in gt_seq_by_ts:
                    orig_seq = gt_seq_by_ts[ts_key]
                elif ts_int is not None and (tid, ts_int) in gt_seq_by_ts:
                    orig_seq = gt_seq_by_ts[(tid, ts_int)]
                else:
                    orig_seq = None
                if orig_seq is not None:
                    ek = (tid, orig_seq)
                    if ek in gt_edges_by_seq:
                        is_correct = edge_match(str(cpath), str(gt_edges_by_seq[ek]))
                        edge_labels.append(1.0 if is_correct else 0.0)
                        edge_trusts.append(trust if trust is not None else 0.5)
                        (corr_trusts if is_correct else mis_trusts).append(trust if trust is not None else 0.5)

    if n_total == 0:
        return {"label": label, "error": "empty", "n": 0}

    err_arr = np.array(errors) if errors else np.array([0])
    trust_arr = np.array(trusts) if trusts else np.array([0.5])
    eps_arr = np.array(eps_list) if eps_list else np.array([0.5])
    edge_corr = np.array(edge_labels) if edge_labels else np.array([0.0])
    edge_trust_arr = np.array(edge_trusts) if edge_trusts else np.array([0.5])

    seg_acc = float(np.mean(edge_corr)) if len(edge_corr) > 0 else None
    ece_tw, mce_tw, ece_bins = compute_ece(edge_trust_arr, edge_corr) if len(edge_corr) > 0 else (1.0, 1.0, [])
    auc, fpr, tpr = compute_roc_auc(edge_corr, edge_trust_arr) if len(edge_corr) > 0 else (0.5, None, None)

    corr_mean = float(np.mean(corr_trusts)) if corr_trusts else None
    mis_mean = float(np.mean(mis_trusts)) if mis_trusts else None
    sep = (corr_mean - mis_mean) if (corr_mean and mis_mean) else None

    return {
        "label": label, "n": n_total,
        "point_error_mean": float(np.mean(err_arr)),
        "point_error_median": float(np.median(err_arr)),
        "point_error_rmse": float(np.sqrt(np.mean(err_arr ** 2))),
        "point_error_p95": float(np.percentile(err_arr, 95)),
        "seg_accuracy": seg_acc,
        "ece_tw": ece_tw, "mce_tw": mce_tw,
        "ece_tw_bins": ece_bins,
        "roc_auc": auc,
        "fpr": fpr.tolist() if fpr is not None else None,
        "tpr": tpr.tolist() if tpr is not None else None,
        "corr_trust_mean": corr_mean, "mis_trust_mean": mis_mean,
        "trust_separation": sep,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_degraded_comparison(cmm_all, fmm_all, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    cond_order = ["clean", "fault", "occlusion", "both"]
    x = np.arange(len(cond_order))
    x_labels = [CONDITION_LABELS[c] for c in cond_order]
    bar_w = 0.35

    def _get(mlist, cond, key, default=0):
        for m in mlist:
            if m["label"] == cond: return m.get(key, default) or default
        return default

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    # (a) Point error
    ax1.bar(x - bar_w/2, [_get(cmm_all, c, "point_error_mean") for c in cond_order],
            bar_w, color="#2166ac", label="CMM", edgecolor="white", lw=0.5)
    ax1.bar(x + bar_w/2, [_get(fmm_all, c, "point_error_mean") for c in cond_order],
            bar_w, color="#b2182b", label="FMM", edgecolor="white", lw=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(x_labels, rotation=15, ha="right")
    ax1.set_ylabel("Mean error (m)"); ax1.set_title("(a) Point Error")
    ax1.legend(); ax1.grid(alpha=0.3, axis="y")

    # (b) Segment accuracy
    ax2.bar(x - bar_w/2, [(_get(cmm_all, c, "seg_accuracy", 0))*100 for c in cond_order],
            bar_w, color="#2166ac", edgecolor="white", lw=0.5)
    ax2.bar(x + bar_w/2, [(_get(fmm_all, c, "seg_accuracy", 0))*100 for c in cond_order],
            bar_w, color="#b2182b", edgecolor="white", lw=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(x_labels, rotation=15, ha="right")
    ax2.set_ylabel("Accuracy (%)"); ax2.set_title("(b) Segment Accuracy")
    ax2.set_ylim(0, 105); ax2.grid(alpha=0.3, axis="y")

    # (c) ECE
    ax3.bar(x - bar_w/2, [_get(cmm_all, c, "ece_tw") for c in cond_order],
            bar_w, color="#2166ac", edgecolor="white", lw=0.5)
    ax3.bar(x + bar_w/2, [_get(fmm_all, c, "ece_tw") for c in cond_order],
            bar_w, color="#b2182b", edgecolor="white", lw=0.5)
    ax3.set_xticks(x); ax3.set_xticklabels(x_labels, rotation=15, ha="right")
    ax3.set_ylabel("ECE"); ax3.set_title("(c) Expected Calibration Error")
    ax3.grid(alpha=0.3, axis="y")

    # (d) ROC AUC
    ax4.bar(x - bar_w/2, [_get(cmm_all, c, "roc_auc") for c in cond_order],
            bar_w, color="#2166ac", edgecolor="white", lw=0.5)
    ax4.bar(x + bar_w/2, [_get(fmm_all, c, "roc_auc") for c in cond_order],
            bar_w, color="#b2182b", edgecolor="white", lw=0.5)
    ax4.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax4.set_xticks(x); ax4.set_xticklabels(x_labels, rotation=15, ha="right")
    ax4.set_ylabel("AUC"); ax4.set_title("(d) ROC AUC")
    ax4.set_ylim(0.3, 1.0); ax4.grid(alpha=0.3, axis="y")

    # (e) Reliability diagram — CMM only, all 4 conditions overlaid
    ax5.plot([0, 1], [0, 1], "k--", lw=0.6, label="Perfect")
    for cond in cond_order:
        m = next((m for m in cmm_all if m["label"] == cond), None)
        if m is None: continue
        bins = m.get("ece_tw_bins", [])
        if not bins: continue
        confs = [b["mean_conf"] for b in bins if b["n"] > 0]
        accs = [b["accuracy"] for b in bins if b["n"] > 0]
        if not confs: continue
        ece = m.get("ece_tw", 0)
        ax5.plot(confs, accs, "o-", color=CONDITION_COLORS.get(cond, "gray"),
                 lw=1.0, ms=5, label=f"{CONDITION_LABELS[cond]} (ECE={ece:.3f})")
    ax5.set_xlabel("Confidence"); ax5.set_ylabel("Accuracy")
    ax5.set_title("(e) Reliability Diagram — CMM"); ax5.legend(fontsize=5)
    ax5.grid(alpha=0.3)

    # (f) ROC curves — CMM only
    ax6.plot([0, 1], [0, 1], "k--", lw=0.6)
    for cond in cond_order:
        m = next((m for m in cmm_all if m["label"] == cond), None)
        if m is None: continue
        fpr = m.get("fpr"); tpr = m.get("tpr"); auc = m.get("roc_auc", 0.5)
        if fpr and tpr:
            ax6.plot(fpr, tpr, lw=1.2, color=CONDITION_COLORS.get(cond, "gray"),
                     label=f"{CONDITION_LABELS[cond]} (AUC={auc:.3f})")
    ax6.set_xlabel("FPR"); ax6.set_ylabel("TPR")
    ax6.set_title("(f) ROC Curves — CMM"); ax6.legend(fontsize=5)
    ax6.grid(alpha=0.3)

    fig.suptitle("CMM vs FMM Under Degraded Conditions (σ=30m)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "degraded_comparison.png", dpi=DPI)
    plt.close(fig)

    # ── Second figure: stacked bar of seg accuracy breakdown ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax_idx, (algo, color) in enumerate([("CMM", "#2166ac"), ("FMM", "#b2182b")]):
        ax = axes[ax_idx]
        mlist = cmm_all if algo == "CMM" else fmm_all
        vals = [(_get(mlist, c, "seg_accuracy", 0) or 0) * 100 for c in cond_order]
        wrong_vals = [100 - v for v in vals]
        ax.bar(x, vals, bar_w * 2, color=color, alpha=0.85, edgecolor="white", lw=0.5, label="Correct")
        ax.bar(x, wrong_vals, bar_w * 2, bottom=vals, color="lightgray", alpha=0.5,
               edgecolor="white", lw=0.5, label="Wrong")
        for i, (v, c) in enumerate(zip(vals, cond_order)):
            ax.text(i, v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=7, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(x_labels, rotation=15, ha="right")
        ax.set_ylabel("%"); ax.set_title(f"{algo} Segment Accuracy"); ax.legend(fontsize=6)
        ax.set_ylim(0, 105)
    fig.suptitle("Segment Correctness Breakdown", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_breakdown.png", dpi=DPI)
    plt.close(fig)

    # ── Third figure: trust distribution overlay ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for idx, cond in enumerate(cond_order):
        ax = axes.flat[idx]
        data_dir = DATA_ROOT / CONDITIONS[cond][0] / CONDITIONS[cond][1]
        for algo, mr_name, color in [("CMM", "cmm_result.csv", "#2166ac"), ("FMM", "fmm_result.csv", "#b2182b")]:
            mr_csv = data_dir / mr_name
            if not mr_csv.exists(): continue
            gt_pts_seq, gt_edg_seq, gt_pts_ts, gt_seq_ts = load_ground_truth(data_dir)
            trusts_hist, labels_hist = [], []
            with open(mr_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    tid = row.get("id", "").strip()
                    ts_str = row.get("timestamp", "").strip()
                    cpath = row.get("cpath", "").strip()
                    trust = _to_float(row.get("trustworthiness", "0"))
                    ts = None; ts_int = None
                    if ts_str:
                        try: ts = float(ts_str); ts_int = int(round(ts))
                        except ValueError: ts = ts_str
                    if not cpath or ts is None or trust is None: continue
                    ts_key = (tid, ts)
                    if ts_key not in gt_seq_ts and ts_int is not None:
                        ts_key = (tid, ts_int)
                    if ts_key not in gt_seq_ts: continue
                    orig_seq = gt_seq_ts[ts_key]
                    ek = (tid, orig_seq)
                    if ek not in gt_edg_seq: continue
                    correct = edge_match(str(cpath), str(gt_edg_seq[ek]))
                    trusts_hist.append(trust)
                    labels_hist.append(1.0 if correct else 0.0)
            if not trusts_hist: continue
            t_arr = np.array(trusts_hist); l_arr = np.array(labels_hist)
            corr = t_arr[l_arr == 1]; mis = t_arr[l_arr == 0]
            bins = np.linspace(0, 1, 21)
            # Compute mean trust difference for this panel
            cmean = np.mean(corr) if len(corr) > 0 else 0
            mmean = np.mean(mis) if len(mis) > 0 else 0
            ax.axvline(cmean, color=color, lw=1.5, ls="--", alpha=0.6)
            ax.axvline(mmean, color="gray", lw=1.5, ls=":", alpha=0.6)
            if len(corr) > 0:
                ax.hist(corr, bins=bins, alpha=0.5, color=color, label=f"{algo}Correct(μ={cmean:.2f})",
                        density=True, edgecolor="white", lw=0.3)
            if len(mis) > 0:
                ax.hist(mis, bins=bins, alpha=0.2, color=color, label=f"{algo}Wrong(μ={mmean:.2f})",
                        density=True, edgecolor="white", lw=0.3, hatch="//")
        ax.set_title(CONDITION_LABELS[cond], fontsize=9); ax.legend(fontsize=5)
        ax.set_xlabel("Trustworthiness")
    fig.suptitle("Trustworthiness Distribution — Correct vs Wrong Edge", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "trust_distribution.png", dpi=DPI)
    plt.close(fig)


def write_summary(cmm_all, fmm_all, output_dir):
    rows = []
    for m in cmm_all:
        rows.append({"algorithm": "CMM", **{k: v for k, v in m.items()
            if k not in ("fpr", "tpr", "ece_tw_bins")}})
    for m in fmm_all:
        rows.append({"algorithm": "FMM", **{k: v for k, v in m.items()
            if k not in ("fpr", "tpr", "ece_tw_bins")}})
    out = output_dir / "degraded_summary.csv"
    with open(out, "w", newline="") as f:
        fields = ["algorithm", "label", "n", "point_error_mean", "point_error_median",
                  "point_error_rmse", "point_error_p95", "seg_accuracy",
                  "ece_tw", "mce_tw", "roc_auc", "trust_separation"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  Summary: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmm-bin", type=Path, default=Path("build/cmm"))
    parser.add_argument("--fmm-bin", type=Path, default=Path("build/fmm"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/output/5_degraded"))
    parser.add_argument("--network-shp", default="input/map/hainan/edges.shp")
    parser.add_argument("--ubodt", default="input/map/hainan_ubodt_indexed.bin")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-match", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    network_shp = str((project_root / args.network_shp).resolve())
    ubodt = str((project_root / args.ubodt).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cond_order = ["clean", "fault", "occlusion", "both"]
    data_dirs = {c: DATA_ROOT / CONDITIONS[c][0] / CONDITIONS[c][1] for c in cond_order}

    # Verify all data exists
    for c, d in data_dirs.items():
        obs = d / "observations.csv"
        if not obs.exists():
            print(f"  MISSING: {c} -> {obs}")
            return

    # ── Phase 1: Matching ──
    if not args.skip_match:
        if args.force:
            for c, d in data_dirs.items():
                for f in d.glob("cmm_result*.csv"): f.unlink()
                for f in d.glob("fmm_result*.csv"): f.unlink()

        print("\n=== Phase 1: Running CMM and FMM ===")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {}
            for c, d in data_dirs.items():
                future = pool.submit(match_condition, d, args.cmm_bin, args.fmm_bin,
                                     network_shp, ubodt, project_root)
                futures[future] = (c, d)
            for future in as_completed(futures):
                c, d = futures[future]
                try:
                    result = future.result()
                    cmm_ok = "OK" if result["cmm"] else "FAIL"
                    fmm_ok = "OK" if result["fmm"] else "FAIL"
                    print(f"  {c}: CMM={cmm_ok}, FMM={fmm_ok}")
                except Exception as e:
                    print(f"  {c}: ERROR {e}")

    # ── Phase 2: Metrics ──
    print("\n=== Phase 2: Computing metrics ===")
    cmm_all, fmm_all = [], []

    for c in cond_order:
        d = data_dirs[c]
        gt_pts_seq, gt_edg_seq, gt_pts_ts, gt_seq_ts = load_ground_truth(d)

        for algo, mr_name, mlist in [("CMM", "cmm_result.csv", cmm_all),
                                      ("FMM", "fmm_result.csv", fmm_all)]:
            mr = d / mr_name
            if mr.exists():
                m = compute_metrics(d, mr, gt_pts_seq, gt_edg_seq, gt_pts_ts, gt_seq_ts, c)
                mlist.append(m)
                seg = m.get("seg_accuracy", 0) or 0
                print(f"  {c}/{algo}: err={m.get('point_error_mean',0):.1f}m, "
                      f"acc={seg*100:.1f}%, ECE={m.get('ece_tw',1):.4f}, "
                      f"AUC={m.get('roc_auc',0.5):.3f}, sep={m.get('trust_separation',0) or 0:.3f}")
            else:
                print(f"  {c}/{algo}: no result")

    # ── Phase 3: Figures ──
    print("\n=== Phase 3: Generating figures ===")
    plot_degraded_comparison(cmm_all, fmm_all, args.output_dir)
    write_summary(cmm_all, fmm_all, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
