#!/usr/bin/env python3
"""
Experiment 3: Full matching pipeline — CMM vs FMM on all datasets.

Performs:
  1. Sigma sensitivity: CMM vs FMM across sigma = 1,5,10,15,20,25,30 m
  2. Sample rate sensitivity: CMM vs FMM at subsample intervals 1s,2s,5s,10s
     for representative sigma levels (5,15,25m)

Metrics per dataset:
  - Point error statistics (vs ground_truth_points)
  - Segment-level accuracy (cpath vs ground_truth point_edge_ids, reverse-aware)
  - Calibration (ECE) of trustworthiness
  - ROC + AUC for mismatch detection
  - Reliability diagrams per sigma

Config per experiments/README.md:
  CMM: k=16, PL_mult=3, window_length=100, tw_threshold=0.0, phmi_pl_mult=1, lag=5
  FMM: k=16, r=0.03, pf=0, gps_error=0.001

Usage:
  python experiments/scripts/exp3_full_matching.py \
    --data-root experiments/data \
    --cmm-bin build/cmm \
    --fmm-bin build/fmm \
    --output-dir experiments/output/3_full_matching \
    --jobs 8
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
COLOR_CMM = "#2166ac"
COLOR_FMM = "#b2182b"
COLORS_SR = ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0"]  # sample rate CMM
COLORS_SR_F = ["#b2182b", "#d6604d", "#f4a582", "#fddbc7"]  # sample rate FMM
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})


# ══════════════════════════════════════════════════════════════════════════════
# Config builders — k=16 per README
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
    """FMM config — trajectory-mode input, r/gps_error in degrees (k=16 per README)."""
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
    """Convert point-format observations to FMM trajectory CSV."""
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


def create_subsampled_csv(obs_csv: Path, out_csv: Path, interval: int):
    """Subsample observations by keeping every `interval`-th point per trajectory.

    interval=1: keep all (original), interval=2: keep every other, etc.
    """
    from collections import defaultdict
    trajs = defaultdict(list)
    with open(obs_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = reader.fieldnames
        for row in reader:
            trajs[row["id"]].append(row)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        for tid in sorted(trajs.keys(), key=int):
            rows = trajs[tid]
            for i in range(0, len(rows), interval):
                w.writerow(rows[i])


def match_dataset(data_dir: Path, cmm_bin: Path, fmm_bin: Path,
                  network_shp: str, ubodt: str, project_root: Path,
                  suffix: str = ""
                  ) -> Dict[str, Optional[Path]]:
    """Run CMM and FMM on one dataset. suffix is appended to result filenames."""
    obs_csv = data_dir / "observations.csv"
    if not obs_csv.exists():
        return {"cmm": None, "fmm": None}

    results = {"cmm": None, "fmm": None}

    # CMM (point-mode)
    cmm_name = f"cmm_result{suffix}.csv"
    cmm_out = str((data_dir / cmm_name).resolve())
    if not (data_dir / cmm_name).exists():
        xml = build_cmm_xml(str(obs_csv.resolve()), cmm_out, network_shp, ubodt)
        if run_matcher(cmm_bin, xml, project_root):
            results["cmm"] = data_dir / cmm_name

    # FMM (trajectory-mode)
    fmm_name = f"fmm_result{suffix}.csv"
    fmm_out = str((data_dir / fmm_name).resolve())
    if not (data_dir / fmm_name).exists():
        traj_name = f"fmm_trajectory{suffix}.csv"
        traj_csv = data_dir / traj_name
        if not traj_csv.exists():
            convert_to_fmm_trajectory(obs_csv, traj_csv)
        xml = build_fmm_xml(str(traj_csv.resolve()), fmm_out, network_shp, ubodt)
        if run_matcher(fmm_bin, xml, project_root):
            results["fmm"] = data_dir / fmm_name

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Metric computation
# ══════════════════════════════════════════════════════════════════════════════

def load_ground_truth(data_dir: Path) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load ground truth.

    Returns:
      gt_points_by_seq:  (id, seq) -> (x, y)
      gt_edges_by_seq:   (id, seq) -> edge_id
      gt_points_by_ts:   (id, timestamp) -> (x, y)
      gt_seq_by_ts:      (id, timestamp) -> seq
    """
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
                    # Store both exact and rounded-int keys (CMM rounds to int)
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


def parse_point(wkt: str) -> Tuple[float, float] | None:
    m = re.search(r'POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)', str(wkt), re.I)
    return (float(m.group(1)), float(m.group(2))) if m else None


def haversine_deg_to_m(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111320.0 * math.cos(mlat)
    dy = (lat1 - lat2) * 111320.0
    return math.sqrt(dx * dx + dy * dy)


def load_reverse_map() -> Dict[str, str]:
    map_path = Path(__file__).resolve().parents[1] / "config" / "reverse_edge_map.json"
    if map_path.exists():
        with open(map_path) as f:
            return json.load(f)
    return {}


REVERSE_MAP = None


def edge_match(matched: str, truth: str) -> bool:
    global REVERSE_MAP
    if REVERSE_MAP is None:
        REVERSE_MAP = load_reverse_map()
    return str(matched) == str(truth) or REVERSE_MAP.get(str(matched)) == str(truth)


def compute_metrics(data_dir: Path, mr_csv: Path,
                    gt_points_by_seq: Dict, gt_edges_by_seq: Dict,
                    gt_points_by_ts: Dict, gt_seq_by_ts: Dict,
                    label: str) -> Dict:
    """Compute metrics using timestamp-based matching (robust to subsampling)."""
    if not mr_csv or not mr_csv.exists():
        return {"label": label, "error": "no result", "n": 0}

    errors = []
    trusts = []
    eps_list = []
    edge_correct = []
    n_total = 0

    with open(mr_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row.get("id", "").strip()
            ts_str = row.get("timestamp", "").strip()
            pgeom = row.get("pgeom", "")
            cpath = row.get("cpath", "").strip()
            trust = to_float(row.get("trustworthiness", "0"))
            ep = to_float(row.get("ep", "0"))

            n_total += 1
            if trust is not None: trusts.append(trust)
            if ep is not None: eps_list.append(ep)

            # Timestamp-based matching (CMM rounds to int, GT has fractional)
            ts = None
            ts_int = None
            if ts_str:
                try:
                    ts = float(ts_str)
                    ts_int = int(round(ts))
                except ValueError:
                    ts = ts_str

            # Point error — try exact ts first, then rounded int
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

            # Edge accuracy — map timestamp → seq → edge
            if cpath and ts is not None:
                ts_key = (tid, ts)
                if ts_key in gt_seq_by_ts:
                    orig_seq = gt_seq_by_ts[ts_key]
                elif ts_int is not None and (tid, ts_int) in gt_seq_by_ts:
                    orig_seq = gt_seq_by_ts[(tid, ts_int)]
                else:
                    orig_seq = None
                if orig_seq is not None:
                    edge_key = (tid, orig_seq)
                    if edge_key in gt_edges_by_seq:
                        edge_correct.append(edge_match(str(cpath), str(gt_edges_by_seq[edge_key])))

    if n_total == 0:
        return {"label": label, "error": "empty", "n": 0}

    err_arr = np.array(errors) if errors else np.array([0])
    trust_arr = np.array(trusts) if trusts else np.array([0.5])
    eps_arr = np.array(eps_list) if eps_list else np.array([0.5])
    edge_corr = np.array(edge_correct, dtype=float)

    seg_acc = float(np.mean(edge_corr)) if len(edge_corr) > 0 else None

    if len(edge_corr) > 0:
        ece_tw, mce_tw, ece_bins = compute_ece(trust_arr, edge_corr)
    else:
        ece_tw, mce_tw, ece_bins = 1.0, 1.0, []

    if len(edge_corr) > 0 and len(trusts) > 0:
        auc, fpr, tpr = compute_roc_auc(edge_corr, trust_arr)
    else:
        auc, fpr, tpr = 0.5, None, None

    return {
        "label": label,
        "n": n_total,
        "point_error_mean": float(np.mean(err_arr)),
        "point_error_median": float(np.median(err_arr)),
        "point_error_rmse": float(np.sqrt(np.mean(err_arr ** 2))),
        "point_error_p95": float(np.percentile(err_arr, 95)),
        "seg_accuracy": seg_acc,
        "seg_n_correct": int(np.sum(edge_corr)) if len(edge_corr) > 0 else 0,
        "seg_n_total": len(edge_corr),
        "ece_tw": ece_tw,
        "mce_tw": mce_tw,
        "ece_tw_bins": ece_bins,
        "ece_ep": compute_ece(eps_arr, edge_corr)[0] if len(edge_corr) > 0 else 1.0,
        "roc_auc": auc,
        "fpr": fpr.tolist() if fpr is not None else None,
        "tpr": tpr.tolist() if tpr is not None else None,
    }


def to_float(val):
    try: return float(val)
    except (ValueError, TypeError): return None


def compute_ece(confidences, labels, n_bins=10):
    N = len(confidences)
    per_bin = []
    ece = 0.0; mce = 0.0
    for i in range(n_bins):
        lo, hi = i/n_bins, (i+1)/n_bins
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins-1: mask = (confidences >= lo) & (confidences <= hi)
        idxs = np.where(mask)[0]
        nb = len(idxs)
        if nb == 0: continue
        mc = float(np.mean(confidences[idxs]))
        acc = float(np.mean(labels[idxs]))
        gap = abs(mc - acc)
        ece += nb/N * gap; mce = max(mce, gap)
        per_bin.append({"n": nb, "mean_conf": mc, "accuracy": acc, "lo": lo, "hi": hi})
    return ece, mce, per_bin


def compute_roc_auc(labels, scores):
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    n_pos = np.sum(labels_sorted); n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5, None, None
    tpr = np.cumsum(labels_sorted) / n_pos
    fpr = np.cumsum(1 - labels_sorted) / n_neg
    tpr = np.concatenate([[0], tpr]); fpr = np.concatenate([[0], fpr])
    auc = float(np.trapz(tpr, fpr))
    return auc, fpr, tpr


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_sigma_comparison(cmm_metrics: List[Dict], fmm_metrics: List[Dict], output_dir: Path):
    """6-panel sigma sweep comparison figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to clean sigma labels only
    sigmas = sorted(
        set(m["label"] for m in cmm_metrics
            if "fault" not in m["label"] and "occ" not in m["label"] and "_sr" not in m["label"]),
        key=lambda s: int(s.replace("sigma_", ""))
    )

    def get(m_list, s, key, default=0):
        for m in m_list:
            if m["label"] == s: return m.get(key, default)
        return default

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.5))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    sigma_vals = [int(s.replace("sigma_", "")) for s in sigmas]

    # (a) Point error vs sigma
    ax1.plot(sigma_vals, [get(cmm_metrics, s, "point_error_mean") for s in sigmas],
             "o-", color=COLOR_CMM, lw=1.2, ms=5, label="CMM")
    ax1.plot(sigma_vals, [get(fmm_metrics, s, "point_error_mean") for s in sigmas],
             "s-", color=COLOR_FMM, lw=1.2, ms=5, label="FMM")
    ax1.set_xlabel(r"$\sigma_{\rho}$ (m)"); ax1.set_ylabel("Mean error (m)")
    ax1.set_title("(a) Point Error"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 32)

    # (b) Segment accuracy
    def _acc_pct(m_list, s):
        v = get(m_list, s, "seg_accuracy", 0)
        return (v or 0) * 100
    ax2.plot(sigma_vals, [_acc_pct(cmm_metrics, s) for s in sigmas],
             "o-", color=COLOR_CMM, lw=1.2, ms=5)
    ax2.plot(sigma_vals, [_acc_pct(fmm_metrics, s) for s in sigmas],
             "s-", color=COLOR_FMM, lw=1.2, ms=5)
    ax2.set_xlabel(r"$\sigma_{\rho}$ (m)"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("(b) Segment Accuracy"); ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 32); ax2.set_ylim(0, 105)

    # (c) ECE (trustworthiness)
    ax3.plot(sigma_vals, [get(cmm_metrics, s, "ece_tw") for s in sigmas],
             "o-", color=COLOR_CMM, lw=1.2, ms=5, label="CMM")
    ax3.plot(sigma_vals, [get(fmm_metrics, s, "ece_tw") for s in sigmas],
             "s-", color=COLOR_FMM, lw=1.2, ms=5, label="FMM")
    ax3.set_xlabel(r"$\sigma_{\rho}$ (m)"); ax3.set_ylabel("ECE")
    ax3.set_title("(c) ECE (Trustworthiness)"); ax3.legend(); ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 32); ax3.set_ylim(0, 0.6)

    # (d) ROC AUC
    ax4.plot(sigma_vals, [get(cmm_metrics, s, "roc_auc") for s in sigmas],
             "o-", color=COLOR_CMM, lw=1.2, ms=5)
    ax4.plot(sigma_vals, [get(fmm_metrics, s, "roc_auc") for s in sigmas],
             "s-", color=COLOR_FMM, lw=1.2, ms=5)
    ax4.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax4.set_xlabel(r"$\sigma_{\rho}$ (m)"); ax4.set_ylabel("AUC")
    ax4.set_title("(d) ROC AUC"); ax4.grid(alpha=0.3)
    ax4.set_xlim(0, 32); ax4.set_ylim(0.4, 1.0)

    # (e) Reliability diagram for sigma=10
    mid_sigma = "sigma_10"
    for label, metrics, color in [("CMM", cmm_metrics, COLOR_CMM), ("FMM", fmm_metrics, COLOR_FMM)]:
        bins = get(metrics, mid_sigma, "ece_tw_bins", [])
        if not bins:
            # Try closest available
            for s in sigmas:
                bins = get(metrics, s, "ece_tw_bins", [])
                if bins: break
        if not bins: continue
        confs = [b["mean_conf"] for b in bins if b["n"] > 0]
        accs = [b["accuracy"] for b in bins if b["n"] > 0]
        if confs:
            ax5.scatter(confs, accs, s=20, color=color, label=label, edgecolors="white", lw=0.5)
    ax5.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax5.set_xlabel("Confidence"); ax5.set_ylabel("Accuracy")
    ax5.set_title("(e) Reliability (σ=10m)"); ax5.legend(); ax5.grid(alpha=0.3)

    # (f) ROC curves for sigma=10
    for label, metrics, color in [("CMM", cmm_metrics, COLOR_CMM), ("FMM", fmm_metrics, COLOR_FMM)]:
        fpr = get(metrics, mid_sigma, "fpr")
        tpr = get(metrics, mid_sigma, "tpr")
        auc = get(metrics, mid_sigma, "roc_auc")
        if not fpr:
            for s in sigmas:
                fpr = get(metrics, s, "fpr")
                tpr = get(metrics, s, "tpr")
                auc = get(metrics, s, "roc_auc")
                if fpr: break
        if fpr and tpr:
            ax6.plot(fpr, tpr, color=color, lw=1.2, label=f"{label} (AUC={auc:.3f})")
    ax6.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax6.set_xlabel("FPR"); ax6.set_ylabel("TPR")
    ax6.set_title("(f) ROC (σ=10m)"); ax6.legend(); ax6.grid(alpha=0.3)

    fig.suptitle("CMM vs FMM: Matching Performance (k=16)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_overview.png", dpi=DPI)
    plt.close(fig)

    # Reliability diagrams for all sigmas
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for idx, s in enumerate(sigmas[:8]):
        ax = axes.flat[idx]
        ax.plot([0, 1], [0, 1], "k--", lw=0.6)
        for label, metrics, color in [("CMM", cmm_metrics, COLOR_CMM), ("FMM", fmm_metrics, COLOR_FMM)]:
            bins = get(metrics, s, "ece_tw_bins", [])
            if not bins: continue
            confs = [b["mean_conf"] for b in bins if b["n"] > 0]
            accs = [b["accuracy"] for b in bins if b["n"] > 0]
            if confs: ax.scatter(confs, accs, s=12, color=color, label=label, edgecolors="white", lw=0.3, alpha=0.8)
        ece_c = get(cmm_metrics, s, "ece_tw", 0)
        ece_f = get(fmm_metrics, s, "ece_tw", 0)
        ax.set_title(rf"$\sigma$={sigma_vals[idx]}m (C:{ece_c:.3f},F:{ece_f:.3f})")
        ax.grid(alpha=0.3)
    for idx in range(len(sigmas), 8): axes.flat[idx].axis("off")
    fig.suptitle("Reliability Diagrams (k=16)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_all_sigmas.png", dpi=DPI)
    plt.close(fig)


def plot_sample_rate_sensitivity(sr_metrics: Dict[str, List[Dict]], output_dir: Path):
    """Plot sample rate sensitivity: accuracy/E CE/AUC vs sample interval."""
    if not sr_metrics:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    intervals = sorted({int(m["label"].split("sr")[-1]) for ml in sr_metrics.values() for m in ml})

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax1, ax2, ax3, ax4 = axes.flat

    for sigma_key, metrics_list in sr_metrics.items():
        sigma_label = sigma_key.replace("_sr_samples", "")
        cmm_m = [m for m in metrics_list if m.get("algorithm") == "CMM"]
        fmm_m = [m for m in metrics_list if m.get("algorithm") == "FMM"]

        cmm_by_int = {int(m["label"].split("sr")[-1]): m for m in cmm_m}
        fmm_by_int = {int(m["label"].split("sr")[-1]): m for m in fmm_m}

        sr_vals = sorted(cmm_by_int.keys())
        if not sr_vals:
            continue

        cmm_acc = [cmm_by_int[v].get("seg_accuracy", 0) * 100 for v in sr_vals]
        fmm_acc = [fmm_by_int[v].get("seg_accuracy", 0) * 100 if v in fmm_by_int else None for v in sr_vals]
        cmm_ece = [cmm_by_int[v].get("ece_tw", 1) for v in sr_vals]
        fmm_ece = [fmm_by_int[v].get("ece_tw", 1) if v in fmm_by_int else None for v in sr_vals]
        cmm_auc = [cmm_by_int[v].get("roc_auc", 0.5) for v in sr_vals]
        fmm_auc = [fmm_by_int[v].get("roc_auc", 0.5) if v in fmm_by_int else None for v in sr_vals]
        cmm_err = [cmm_by_int[v].get("point_error_mean", 0) for v in sr_vals]
        fmm_err = [fmm_by_int[v].get("point_error_mean", 0) if v in fmm_by_int else None for v in sr_vals]

        idx = list(sr_metrics.keys()).index(sigma_key)
        c = COLORS_SR[idx % len(COLORS_SR)] if idx < len(COLORS_SR) else f"C{idx}"
        cf = COLORS_SR_F[idx % len(COLORS_SR_F)] if idx < len(COLORS_SR_F) else f"C{idx+1}"

        ax1.plot(sr_vals, cmm_acc, "o-", color=c, lw=1.2, ms=5, label=f"CMM {sigma_label}")
        ax1.plot(sr_vals, fmm_acc, "s--", color=cf, lw=1.2, ms=5, label=f"FMM {sigma_label}")

        ax2.plot(sr_vals, cmm_ece, "o-", color=c, lw=1.2, ms=5)
        ax2.plot(sr_vals, fmm_ece, "s--", color=cf, lw=1.2, ms=5)

        ax3.plot(sr_vals, cmm_auc, "o-", color=c, lw=1.2, ms=5)
        ax3.plot(sr_vals, fmm_auc, "s--", color=cf, lw=1.2, ms=5)

        ax4.plot(sr_vals, cmm_err, "o-", color=c, lw=1.2, ms=5)
        ax4.plot(sr_vals, fmm_err, "s--", color=cf, lw=1.2, ms=5)

    ax1.set_xlabel("Sample interval (s)"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("(a) Segment Accuracy"); ax1.legend(fontsize=5); ax1.grid(alpha=0.3)
    ax2.set_xlabel("Sample interval (s)"); ax2.set_ylabel("ECE")
    ax2.set_title("(b) ECE (Trustworthiness)"); ax2.grid(alpha=0.3)
    ax3.set_xlabel("Sample interval (s)"); ax3.set_ylabel("AUC")
    ax3.set_title("(c) ROC AUC"); ax3.grid(alpha=0.3)
    ax3.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax4.set_xlabel("Sample interval (s)"); ax4.set_ylabel("Mean error (m)")
    ax4.set_title("(d) Point Error"); ax4.grid(alpha=0.3)

    fig.suptitle("Sample Rate Sensitivity: CMM vs FMM (k=16)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "sample_rate_sensitivity.png", dpi=DPI)
    plt.close(fig)


def write_summary(cmm_metrics, fmm_metrics, output_dir, prefix="summary"):
    rows = []
    for m in cmm_metrics:
        rows.append({"algorithm": "CMM", **{k: v for k, v in m.items() if k not in ("fpr", "tpr", "ece_tw_bins")}})
    for m in fmm_metrics:
        rows.append({"algorithm": "FMM", **{k: v for k, v in m.items() if k not in ("fpr", "tpr", "ece_tw_bins")}})

    out = output_dir / f"{prefix}_table.csv"
    with open(out, "w", newline="") as f:
        fieldnames = ["algorithm", "label", "n", "point_error_mean", "point_error_median",
                      "point_error_rmse", "point_error_p95", "seg_accuracy", "ece_tw", "ece_ep", "roc_auc"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  Summary: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data"))
    parser.add_argument("--cmm-bin", type=Path, default=Path("build/cmm"))
    parser.add_argument("--fmm-bin", type=Path, default=Path("build/fmm"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/output/3_full_matching"))
    parser.add_argument("--network-shp", default="input/map/hainan/edges.shp")
    parser.add_argument("--ubodt", default="input/map/hainan_ubodt_indexed.bin")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-match", action="store_true",
                        help="Skip matching, only compute metrics from existing results")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run matching even if results exist")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    network_shp = str((project_root / args.network_shp).resolve())
    ubodt = str((project_root / args.ubodt).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sigma-level datasets ──
    dataset_dirs = sorted(args.data_root.glob("sigma_*/no_occlusion/no_fault"))
    if not dataset_dirs:
        dataset_dirs = sorted([d for d in args.data_root.glob("sigma_*")
                               if d.is_dir() and (d / "observations.csv").exists()])
    print(f"Found {len(dataset_dirs)} sigma-level datasets")

    # ── Sample rate config ──
    SAMPLE_INTERVALS = [1, 2, 5, 10]  # seconds (original=1s)
    SR_SIGMA_LABELS = ["sigma_05", "sigma_15", "sigma_25"]  # representative levels

    # ── Phase 1: Create subsampled datasets ──
    sr_dataset_dirs = {}  # sigma_key -> {interval: data_dir}
    for d in dataset_dirs:
        label = next((p for p in d.parts if p.startswith("sigma_")), d.name)
        if label in SR_SIGMA_LABELS:
            sr_dataset_dirs[label] = {}
            for interval in SAMPLE_INTERVALS:
                if interval == 1:
                    sr_dataset_dirs[label][interval] = d  # original
                else:
                    sr_dir = d / f"subsample_{interval}s"
                    sr_dir.mkdir(parents=True, exist_ok=True)
                    # Symlink ground truth files, create subsampled observations
                    obs_out = sr_dir / "observations.csv"
                    if not obs_out.exists():
                        create_subsampled_csv(d / "observations.csv", obs_out, interval)
                    for fname in ["ground_truth.csv", "ground_truth_points.csv", "metadata.json"]:
                        src = d / fname
                        dst = sr_dir / fname
                        if src.exists() and not dst.exists():
                            dst.symlink_to(os.path.relpath(src, sr_dir))
                    sr_dataset_dirs[label][interval] = sr_dir

    # ── Phase 2: Run matching ──
    if not args.skip_match:
        print("\n=== Phase 2: Running CMM and FMM ===")

        # Collect all matching tasks
        all_dirs = list(dataset_dirs)  # sigma sweep
        for sigma_key, sr_dict in sr_dataset_dirs.items():
            for interval, d in sr_dict.items():
                if interval > 1:  # interval=1 already in dataset_dirs
                    all_dirs.append(d)

        if args.force:
            for d in all_dirs:
                for f in d.glob("cmm_result*.csv"):
                    f.unlink()
                    print(f"  Removed {f}")
                for f in d.glob("fmm_result*.csv"):
                    f.unlink()
                    print(f"  Removed {f}")

        match_results = {}
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {}
            for d in all_dirs:
                future = pool.submit(match_dataset, d, args.cmm_bin, args.fmm_bin,
                                     network_shp, ubodt, project_root)
                futures[future] = d
            for future in as_completed(futures):
                d = futures[future]
                try:
                    result = future.result()
                    match_results[d] = result
                    label = str(d.relative_to(args.data_root))
                    cmm_ok = "OK" if result["cmm"] else "FAIL"
                    fmm_ok = "OK" if result["fmm"] else "FAIL"
                    print(f"  {label}: CMM={cmm_ok}, FMM={fmm_ok}")
                except Exception as e:
                    print(f"  {d.name}: ERROR {e}")

    # ── Phase 3: Compute sigma sweep metrics ──
    print("\n=== Phase 3: Computing sigma sweep metrics ===")
    cmm_all, fmm_all = [], []

    for d in dataset_dirs:
        label = next((p for p in d.parts if p.startswith("sigma_")), d.name)
        gt_points_by_seq, gt_edges_by_seq, gt_points_by_ts, gt_seq_by_ts = load_ground_truth(d)

        for algo, mr_name, metrics_list in [
            ("CMM", "cmm_result.csv", cmm_all),
            ("FMM", "fmm_result.csv", fmm_all),
        ]:
            mr_csv = d / mr_name
            if mr_csv.exists():
                m = compute_metrics(d, mr_csv, gt_points_by_seq, gt_edges_by_seq,
                                    gt_points_by_ts, gt_seq_by_ts, label)
                metrics_list.append(m)
                seg = m.get('seg_accuracy', 0) or 0
                print(f"  {label}/{algo}: err={m.get('point_error_mean',0):.1f}m, "
                      f"seg_acc={seg*100:.1f}%, ECE={m.get('ece_tw',1.0):.4f}, "
                      f"AUC={m.get('roc_auc',0.5):.3f}, n={m.get('n',0)}")
            else:
                print(f"  {label}/{algo}: no result")

    # ── Phase 4: Compute sample rate metrics ──
    print("\n=== Phase 4: Computing sample rate sensitivity metrics ===")
    sr_metrics = {}  # sigma_key -> list of metrics dicts

    for sigma_key, sr_dict in sr_dataset_dirs.items():
        sr_metrics[sigma_key] = []
        for interval in SAMPLE_INTERVALS:
            d = sr_dict[interval]
            gt_points_by_seq, gt_edges_by_seq, gt_points_by_ts, gt_seq_by_ts = load_ground_truth(d)
            sr_label = f"{sigma_key}_sr{interval}"

            for algo, mr_name in [("CMM", "cmm_result.csv"), ("FMM", "fmm_result.csv")]:
                mr_csv = d / mr_name
                if mr_csv.exists():
                    m = compute_metrics(d, mr_csv, gt_points_by_seq, gt_edges_by_seq,
                                        gt_points_by_ts, gt_seq_by_ts, sr_label)
                    m["algorithm"] = algo
                    m["sample_interval"] = interval
                    sr_metrics[sigma_key].append(m)
                    seg = m.get('seg_accuracy', 0) or 0
                    print(f"  {sr_label}/{algo}: err={m.get('point_error_mean',0):.1f}m, "
                          f"seg_acc={seg*100:.1f}%, ECE={m.get('ece_tw',1.0):.4f}, "
                          f"AUC={m.get('roc_auc',0.5):.3f}, n={m.get('n',0)}")

    # ── Phase 5: Generate figures ──
    print("\n=== Phase 5: Generating figures ===")
    plot_sigma_comparison(cmm_all, fmm_all, args.output_dir)
    write_summary(cmm_all, fmm_all, args.output_dir, "sigma_sweep")

    # Sample rate summary
    if sr_metrics:
        plot_sample_rate_sensitivity(sr_metrics, args.output_dir)
        sr_all = [m for ml in sr_metrics.values() for m in ml]
        write_summary([m for m in sr_all if m.get("algorithm") == "CMM"],
                      [m for m in sr_all if m.get("algorithm") == "FMM"],
                      args.output_dir, "sample_rate")

    print("\nDone.")


if __name__ == "__main__":
    main()
