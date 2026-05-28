#!/usr/bin/env python3
"""
Experiment 4: Effects of wrong emission model (sigma mismatch) on CMM.

Scenario: WLS/RAIM always assumes sigma_rho=20m for positioning, but the true
pseudorange noise varies from 10m to 30m. This creates a systematic bias in
the reported covariance matrix, leading to:
  - Over-conservative emission:  sigma_true < sigma_assumed (pr10, pr15)
    Covariance ellipse is too large -> emission probabilities too flat
  - Correct emission:            sigma_true = sigma_assumed (pr20)
  - Over-confident emission:     sigma_true > sigma_assumed (pr25, pr30)
    Covariance ellipse is too small -> emission probabilities too peaked

Hypothesis:
  - Over-confident: CMM accuracy may be high (sharp emission), but ECE will
    degrade (trustworthiness over-estimated)
  - Over-conservative: CMM ECE may be better, but accuracy may drop marginally
    (flat emission becomes less discriminative)

Usage:
  python experiments/scripts/exp4_sigma_mismatch.py \\
    --data-root experiments/data/sigma_mismatch \\
    --cmm-bin build/cmm --fmm-bin build/fmm \\
    --output-dir experiments/output/4_sigma_mismatch \\
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
plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 6, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})


# ══════════════════════════════════════════════════════════════════════════════
# Config builders (k=16, same as exp3)
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


def match_one_dataset(data_dir: Path, cmm_bin: Path, fmm_bin: Path,
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
# Ground truth loading (timestamp-based, same as exp3)
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


_REVERSE_MAP = None


def edge_match(matched: str, truth: str) -> bool:
    global _REVERSE_MAP
    if _REVERSE_MAP is None:
        _REVERSE_MAP = load_reverse_map()
    return str(matched) == str(truth) or _REVERSE_MAP.get(str(matched)) == str(truth)


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


def compute_metrics(data_dir: Path, mr_csv: Path,
                    gt_points_by_seq: Dict, gt_edges_by_seq: Dict,
                    gt_points_by_ts: Dict, gt_seq_by_ts: Dict,
                    label: str) -> Dict:
    if not mr_csv or not mr_csv.exists():
        return {"label": label, "error": "no result", "n": 0}

    errors = []
    trusts = []
    eps_list = []
    mis_trusts = []   # trustworthiness at wrong-edge points
    corr_trusts = []  # trustworthiness at correct-edge points
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

            ts = None; ts_int = None
            if ts_str:
                try:
                    ts = float(ts_str)
                    ts_int = int(round(ts))
                except ValueError:
                    ts = ts_str

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
                    edge_key = (tid, orig_seq)
                    if edge_key in gt_edges_by_seq:
                        is_correct = edge_match(str(cpath), str(gt_edges_by_seq[edge_key]))
                        if is_correct:
                            corr_trusts.append(trust)
                        else:
                            mis_trusts.append(trust)

            # For ECE, need all edge-label pairs
            # We'll recompute below

    if n_total == 0:
        return {"label": label, "error": "empty", "n": 0}

    err_arr = np.array(errors) if errors else np.array([0])
    trust_arr = np.array(trusts) if trusts else np.array([0.5])
    eps_arr = np.array(eps_list) if eps_list else np.array([0.5])

    # Compute edge labels for ECE (re-read with edge labels)
    edge_labels = []
    edge_trusts = []
    with open(mr_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row.get("id", "").strip()
            ts_str = row.get("timestamp", "").strip()
            cpath = row.get("cpath", "").strip()
            trust = to_float(row.get("trustworthiness", "0"))
            ts = None; ts_int = None
            if ts_str:
                try:
                    ts = float(ts_str)
                    ts_int = int(round(ts))
                except ValueError:
                    ts = ts_str
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
                        edge_labels.append(1.0 if edge_match(str(cpath), str(gt_edges_by_seq[edge_key])) else 0.0)
                        edge_trusts.append(trust if trust is not None else 0.5)

    edge_corr = np.array(edge_labels) if edge_labels else np.array([0.0])
    edge_trust_arr = np.array(edge_trusts) if edge_trusts else np.array([0.5])

    seg_acc = float(np.mean(edge_corr)) if len(edge_corr) > 0 else None
    ece_tw, mce_tw, ece_bins = compute_ece(edge_trust_arr, edge_corr) if len(edge_corr) > 0 else (1.0, 1.0, [])
    auc, fpr, tpr = compute_roc_auc(edge_corr, edge_trust_arr) if len(edge_corr) > 0 else (0.5, None, None)

    # Trustworthiness distribution split
    corr_trust_mean = float(np.mean(corr_trusts)) if corr_trusts else None
    mis_trust_mean = float(np.mean(mis_trusts)) if mis_trusts else None
    trust_separation = (corr_trust_mean - mis_trust_mean) if (corr_trusts and mis_trusts) else None

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
        "roc_auc": auc,
        "fpr": fpr.tolist() if fpr is not None else None,
        "tpr": tpr.tolist() if tpr is not None else None,
        "corr_trust_mean": corr_trust_mean,
        "mis_trust_mean": mis_trust_mean,
        "trust_separation": trust_separation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

# Color map: blue (over-conservative) -> green (correct) -> red (over-confident)
COLOR_MAP = {
    "pr10": "#2166ac",   # blue — very conservative
    "pr15": "#92c5de",   # light blue — slightly conservative
    "pr20": "#4dac26",   # green — correct
    "pr25": "#f4a582",   # light red — slightly over-confident
    "pr30": "#b2182b",   # red — very over-confident
}

SIGMA_PSEUDO = {"pr10": 10, "pr15": 15, "pr20": 20, "pr25": 25, "pr30": 30}


def _sigma_pr_from_label(label: str) -> int:
    """Extract sigma_pseudorange from label like 'pr10_wls20'."""
    import re
    m = re.search(r'pr(\d+)', label)
    return int(m.group(1)) if m else 20


def plot_mismatch_analysis(cmm_all: List[Dict], fmm_all: List[Dict], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(cmm_all, key=lambda m: _sigma_pr_from_label(m["label"]))
    lbls = [m["label"] for m in labels]
    sigma_pr = [_sigma_pr_from_label(l) for l in lbls]
    mismatch = [s - 20 for s in sigma_pr]  # -10..+10

    def get_cmm(key, default=0):
        return [m.get(key, default) for m in labels]
    def get_fmm(key, default=0):
        result = []
        for l in lbls:
            for m in fmm_all:
                if m["label"] == l: result.append(m.get(key, default)); break
            else: result.append(default)
        return result

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

    # (a) Point error vs sigma_pseudorange
    ax1.plot(sigma_pr, get_cmm("point_error_mean"), "o-", color="#2166ac", lw=1.5, ms=7, label="CMM")
    ax1.plot(sigma_pr, get_fmm("point_error_mean"), "s--", color="#b2182b", lw=1.5, ms=7, label="FMM")
    ax1.axvline(20, color="gray", ls=":", lw=1, alpha=0.5, label=r"$\sigma_{wls}=20$m")
    ax1.set_xlabel(r"True $\sigma_{\rho}$ (m)"); ax1.set_ylabel("Mean error (m)")
    ax1.set_title("(a) Point Error vs True Noise"); ax1.legend(); ax1.grid(alpha=0.3)

    # (b) Segment accuracy vs sigma_pseudorange
    ax2.plot(sigma_pr, [(v or 0)*100 for v in get_cmm("seg_accuracy")], "o-", color="#2166ac", lw=1.5, ms=7)
    ax2.plot(sigma_pr, [(v or 0)*100 for v in get_fmm("seg_accuracy")], "s--", color="#b2182b", lw=1.5, ms=7)
    ax2.axvline(20, color="gray", ls=":", lw=1, alpha=0.5)
    ax2.set_xlabel(r"True $\sigma_{\rho}$ (m)"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("(b) Segment Accuracy"); ax2.grid(alpha=0.3)

    # (c) ECE vs sigma_pseudorange (KEY PLOT)
    ax3.plot(sigma_pr, get_cmm("ece_tw"), "o-", color="#2166ac", lw=1.5, ms=7, label="CMM")
    ax3.plot(sigma_pr, get_fmm("ece_tw"), "s--", color="#b2182b", lw=1.5, ms=7, label="FMM")
    ax3.axvline(20, color="gray", ls=":", lw=1, alpha=0.5)
    ax3.set_xlabel(r"True $\sigma_{\rho}$ (m)"); ax3.set_ylabel("ECE")
    ax3.set_title("(c) Calibration Error (ECE) — Key Metric"); ax3.legend(); ax3.grid(alpha=0.3)

    # (d) Trustworthiness separation (correct - wrong)
    ax4.plot(sigma_pr, get_cmm("trust_separation", None), "o-", color="#2166ac", lw=1.5, ms=7, label="CMM")
    ax4.plot(sigma_pr, get_fmm("trust_separation", None), "s--", color="#b2182b", lw=1.5, ms=7, label="FMM")
    ax4.axhline(0, color="gray", lw=0.8, ls="--")
    ax4.axvline(20, color="gray", ls=":", lw=1, alpha=0.5)
    ax4.set_xlabel(r"True $\sigma_{\rho}$ (m)"); ax4.set_ylabel(r"$\Delta$Trust (correct $-$ wrong)")
    ax4.set_title("(d) Trust Separation"); ax4.legend(); ax4.grid(alpha=0.3)

    # (e) Reliability diagrams — per condition, CMM
    ax5.plot([0, 1], [0, 1], "k--", lw=0.6)
    for m in labels:
        lbl = m["label"]
        sigma_key = f"σ_pr={_sigma_pr_from_label(lbl)}m"
        bins = m.get("ece_tw_bins", [])
        if not bins: continue
        confs = [b["mean_conf"] for b in bins if b["n"] > 0]
        accs = [b["accuracy"] for b in bins if b["n"] > 0]
        color = COLOR_MAP.get(lbl, "gray")
        marker = "o" if lbl == "pr20" else ("^" if _sigma_pr_from_label(lbl) < 20 else "v")
        ax5.scatter(confs, accs, s=25, color=color, label=sigma_key, marker=marker,
                    edgecolors="white", lw=0.5)
    ax5.set_xlabel("Confidence"); ax5.set_ylabel("Accuracy")
    ax5.set_title("(e) Reliability Diagram — CMM"); ax5.legend(fontsize=5); ax5.grid(alpha=0.3)

    # (f) ECE vs mismatch (sigma_true - sigma_assumed)
    ax6.plot(mismatch, get_cmm("ece_tw"), "o-", color="#2166ac", lw=1.5, ms=7, label="CMM")
    ax6.plot(mismatch, get_fmm("ece_tw"), "s--", color="#b2182b", lw=1.5, ms=7, label="FMM")
    ax6.axvline(0, color="gray", ls=":", lw=1, alpha=0.5)
    ax6.set_xlabel(r"$\sigma_{true} - \sigma_{assumed}$ (m)")
    ax6.set_ylabel("ECE")
    ax6.set_title(r"(f) ECE vs Mismatch ($\sigma_{true}-\sigma_{wls}$)")
    ax6.legend(); ax6.grid(alpha=0.3)
    # Annotate over-/under-confidence regions
    y_lim = ax6.get_ylim()
    ax6.annotate("Over-\nconservative", xy=(-5, y_lim[1]*0.9), fontsize=6, ha="center", color="#2166ac")
    ax6.annotate("Over-\nconfident", xy=(5, y_lim[1]*0.9), fontsize=6, ha="center", color="#b2182b")

    fig.suptitle("Effect of Wrong Emission Model (σ_mismatch): CMM vs FMM", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "mismatch_analysis.png", dpi=DPI)
    plt.close(fig)

    # ── Second figure: Trustworthiness distribution shift ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for idx, (lbl_key, title) in enumerate([
        ("pr10", "Over-Conservative (σ_pr=10, σ_wls=20)"),
        ("pr20", "Correct (σ_pr=20, σ_wls=20)"),
        ("pr30", "Over-Confident (σ_pr=30, σ_wls=20)"),
    ]):
        ax = axes[idx]
        for algo, mr_name, color, hatch in [
            ("CMM", "cmm_result.csv", "#2166ac", ""),
            ("FMM", "fmm_result.csv", "#b2182b", "//"),
        ]:
            # Find the data dir for this mismatch level
            data_dir = next(Path("experiments/data/sigma_mismatch").glob(f"{lbl_key}*"))
            mr_csv = data_dir / mr_name
            if not mr_csv.exists():
                continue
            trusts, labels_list = [], []
            _, gt_edges_by_seq, _, gt_seq_by_ts = load_ground_truth(data_dir)
            with open(mr_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    tid = row.get("id", "").strip()
                    ts_str = row.get("timestamp", "").strip()
                    cpath = row.get("cpath", "").strip()
                    trust = to_float(row.get("trustworthiness", "0"))
                    ts = None; ts_int = None
                    if ts_str:
                        try:
                            ts = float(ts_str); ts_int = int(round(ts))
                        except ValueError: ts = ts_str
                    if not cpath or ts is None or trust is None:
                        continue
                    ts_key = (tid, ts)
                    if ts_key not in gt_seq_by_ts and ts_int is not None:
                        ts_key = (tid, ts_int)
                    if ts_key not in gt_seq_by_ts:
                        continue
                    orig_seq = gt_seq_by_ts[ts_key]
                    edge_key = (tid, orig_seq)
                    if edge_key not in gt_edges_by_seq:
                        continue
                    correct = edge_match(str(cpath), str(gt_edges_by_seq[edge_key]))
                    trusts.append(trust)
                    labels_list.append(1.0 if correct else 0.0)
            if not trusts: continue
            trusts_arr = np.array(trusts)
            labels_arr = np.array(labels_list)
            corr_t = trusts_arr[labels_arr == 1]
            mis_t = trusts_arr[labels_arr == 0]
            bins = np.linspace(0, 1, 21)
            if len(corr_t) > 0:
                ax.hist(corr_t, bins=bins, alpha=0.5, color=color, label=f"{algo} Correct",
                        density=True, edgecolor="white", lw=0.3)
            if len(mis_t) > 0:
                ax.hist(mis_t, bins=bins, alpha=0.3, color=color, label=f"{algo} Wrong",
                        density=True, edgecolor="white", lw=0.3, hatch=hatch)
        ax.set_xlabel("Trustworthiness"); ax.set_ylabel("Density")
        ax.set_title(title, fontsize=8); ax.legend(fontsize=5)
    fig.suptitle("Trustworthiness Distribution: Correct vs Wrong Edge", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "trustworthiness_distribution.png", dpi=DPI)
    plt.close(fig)

    # ── Third figure: ECE decomposition per bin ──
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), sharey=True)
    for idx, m in enumerate(labels):
        ax = axes[idx]
        lbl = m["label"]
        sigma_key = f"σ_pr={_sigma_pr_from_label(lbl)}m"
        bins = m.get("ece_tw_bins", [])
        if not bins: continue
        confs = [b["mean_conf"] for b in bins if b["n"] > 0]
        accs = [b["accuracy"] for b in bins if b["n"] > 0]
        gap = [abs(c - a) for c, a in zip(confs, accs)]
        x = range(len(confs))
        ax.bar(x, gap, color=COLOR_MAP.get(lbl, "gray"), alpha=0.7, edgecolor="white", lw=0.3)
        ax.set_title(sigma_key, fontsize=8)
        ax.set_xlabel("Conf. bin"); ax.set_xticks(x[::2])
        ax.set_xticklabels([f"{bins[i]['lo']:.1f}" for i in x[::2]], fontsize=5, rotation=45)
        ax.grid(alpha=0.2, axis="y")
    axes[0].set_ylabel("|Conf - Acc|")
    fig.suptitle("ECE Per-Bin Decomposition — CMM", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "ece_per_bin.png", dpi=DPI)
    plt.close(fig)


def write_summary(cmm_all, fmm_all, output_dir):
    rows = []
    for m in cmm_all:
        rows.append({"algorithm": "CMM", **{k: v for k, v in m.items()
                     if k not in ("fpr", "tpr", "ece_tw_bins", "corr_trust_mean", "mis_trust_mean")}})
    for m in fmm_all:
        rows.append({"algorithm": "FMM", **{k: v for k, v in m.items()
                     if k not in ("fpr", "tpr", "ece_tw_bins", "corr_trust_mean", "mis_trust_mean")}})
    out = output_dir / "mismatch_summary.csv"
    with open(out, "w", newline="") as f:
        fieldnames = ["algorithm", "label", "n", "point_error_mean", "point_error_median",
                      "point_error_rmse", "point_error_p95", "seg_accuracy",
                      "ece_tw", "mce_tw", "roc_auc", "trust_separation"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"  Summary: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("experiments/data/sigma_mismatch"))
    parser.add_argument("--cmm-bin", type=Path, default=Path("build/cmm"))
    parser.add_argument("--fmm-bin", type=Path, default=Path("build/fmm"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/output/4_sigma_mismatch"))
    parser.add_argument("--network-shp", default="input/map/hainan/edges.shp")
    parser.add_argument("--ubodt", default="input/map/hainan_ubodt_indexed.bin")
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--skip-match", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    network_shp = str((project_root / args.network_shp).resolve())
    ubodt = str((project_root / args.ubodt).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted(args.data_root.glob("pr*"))
    print(f"Found {len(dataset_dirs)} mismatch datasets: {[d.name for d in dataset_dirs]}")

    # ── Phase 1: Matching ──
    if not args.skip_match:
        if args.force:
            for d in dataset_dirs:
                for f in d.glob("cmm_result*.csv"): f.unlink()
                for f in d.glob("fmm_result*.csv"): f.unlink()
        print("\n=== Phase 1: Running CMM and FMM ===")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {}
            for d in dataset_dirs:
                future = pool.submit(match_one_dataset, d, args.cmm_bin, args.fmm_bin,
                                     network_shp, ubodt, project_root)
                futures[future] = d
            for future in as_completed(futures):
                d = futures[future]
                try:
                    result = future.result()
                    cmm_ok = "OK" if result["cmm"] else "FAIL"
                    fmm_ok = "OK" if result["fmm"] else "FAIL"
                    print(f"  {d.name}: CMM={cmm_ok}, FMM={fmm_ok}")
                except Exception as e:
                    print(f"  {d.name}: ERROR {e}")

    # ── Phase 2: Metrics ──
    print("\n=== Phase 2: Computing metrics ===")
    cmm_all, fmm_all = [], []

    for d in dataset_dirs:
        label = d.name  # pr10_wls20, etc.
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
                      f"AUC={m.get('roc_auc',0.5):.3f}, "
                      f"sep={m.get('trust_separation',0) or 0:.3f}")
            else:
                print(f"  {label}/{algo}: no result")

    # ── Phase 3: Figures ──
    print("\n=== Phase 3: Generating figures ===")
    plot_mismatch_analysis(cmm_all, fmm_all, args.output_dir)
    write_summary(cmm_all, fmm_all, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
