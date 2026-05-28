#!/usr/bin/env python3
"""
Shared utilities for the experiments suite.

Sources of adapted functions:
  _read_csv_auto       — adapted from python/plot_sigma_error_pl_cdf.py:22
  compute_mahalanobis_d2 — adapted from python/check_error_gaussian.py:37-53
  _whiten_error        — adapted from python/plot_methodology_evidence.py:123-128
  _rayleigh_cdf        — adapted from python/plot_methodology_evidence.py:42-45
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Unified style (matching tests/python/gen_figures.py) ────────────────────
DPI = 300
COLOR_CMM = "#2166ac"
COLOR_FMM = "#b2182b"
COLOR_FILT = "#4393c3"
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _read_csv_auto(path: Path) -> pd.DataFrame:
    """Read CSV with semicolon or comma delimiter."""
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path, sep=",")
    return df


def load_dataset(dir_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load observations.csv and ground_truth_points.csv from a dataset directory.

    Returns (obs_df, truth_df, metadata).
    """
    obs_path = dir_path / "observations_fixed.csv"
    if not obs_path.exists():
        obs_path = dir_path / "observations.csv"
    gt_path = dir_path / "ground_truth_points.csv"
    meta_path = dir_path / "metadata.json"

    obs = _read_csv_auto(obs_path)
    truth = None
    if gt_path.exists():
        truth = _read_csv_auto(gt_path)

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return obs, truth, meta


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ═══════════════════════════════════════════════════════════════════════════════
# Covariance / Mahalanobis helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _cov_from_row(row: pd.Series) -> np.ndarray:
    sde = float(row["sde"])
    sdn = float(row["sdn"])
    sdne = float(row.get("sdne", 0.0))
    return np.array([[sde * sde, sdne], [sdne, sdn * sdn]], dtype=float)


def compute_mahalanobis_d2(errors: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """Compute squared Mahalanobis distance for each error-covariance pair.

    errors: (N, 2), covs: (N, 2, 2)
    """
    N = errors.shape[0]
    d2 = np.zeros(N)
    for i in range(N):
        e = errors[i]
        cov = covs[i]
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(cov)):
            d2[i] = np.nan
            continue
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            d2[i] = np.nan
            continue
        d2[i] = float(e.T @ inv @ e)
    return d2


def _whiten_error(err: np.ndarray, cov: np.ndarray) -> np.ndarray | None:
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return None
    return np.linalg.solve(L, err)


def _rayleigh_cdf(r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    return 1.0 - np.exp(-(r ** 2) / (2.0 * sigma ** 2))


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10
                ) -> Tuple[float, float, list]:
    """Compute Expected Calibration Error and per-bin data.

    Returns (ECE, MCE, per_bin_list).
    """
    N = len(confidences)
    per_bin = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if i == n_bins - 1:
            idxs = np.where((confidences >= lo) & (confidences <= hi))[0]
        else:
            idxs = np.where((confidences >= lo) & (confidences < hi))[0]
        nb = len(idxs)
        if nb == 0:
            per_bin.append({"n": 0, "mean_conf": float("nan"), "accuracy": float("nan"),
                          "lo": lo, "hi": hi})
            continue
        mc = float(np.mean(confidences[idxs]))
        acc = float(np.mean(labels[idxs]))
        gap = abs(mc - acc)
        ece += nb / N * gap
        mce = max(mce, gap)
        per_bin.append({"n": nb, "mean_conf": mc, "accuracy": acc, "lo": lo, "hi": hi})

    return ece, mce, per_bin


def compute_brier_logloss(confidences: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    eps = 1e-15
    brier = float(np.mean((confidences - labels) ** 2))
    logloss = -np.mean(labels * np.log(np.maximum(confidences, eps)) +
                       (1 - labels) * np.log(np.maximum(1 - confidences, eps)))
    return brier, logloss


# ═══════════════════════════════════════════════════════════════════════════════
# Segment-level accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def segment_level_accuracy(matched_edge_ids: List[str],
                           truth_edge_ids: List[str]) -> Tuple[float, int, int]:
    """Return (accuracy, n_correct, n_total) where correctness is edge-level match."""
    n_total = len(truth_edge_ids)
    if n_total == 0:
        return 0.0, 0, 0
    n_correct = sum(1 for m, t in zip(matched_edge_ids, truth_edge_ids) if str(m) == str(t))
    return n_correct / n_total, n_correct, n_total


# ═══════════════════════════════════════════════════════════════════════════════
# Stanford / PL metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stanford_metrics(errors: np.ndarray, protection_levels: np.ndarray,
                             fault_flags: np.ndarray | None = None
                             ) -> dict:
    """Compute P_md (missed detection) and P_fa (false alarm) from error vs PL.

    HMI: error > PL (hazardous misleading information).

    Returns dict with: P_md, P_fa, n_HMI, n_FA, n_total, mean_PL, mean_error.
    """
    mask = np.isfinite(errors) & np.isfinite(protection_levels)
    err = errors[mask]
    pl = protection_levels[mask]
    n_total = len(err)
    if n_total == 0:
        return {"P_md": 0, "P_fa": 0, "n_HMI": 0, "n_FA": 0,
                "n_total": 0, "mean_PL": 0, "mean_error": 0}

    n_HMI = int(np.sum(err > pl))
    n_safe = n_total - n_HMI

    # P_md = HMI / total (overall integrity failure rate)
    P_md = n_HMI / n_total

    # P_fa: fraction of safe epochs where PL indicates alarm
    # (reported PL far exceeds actual error)
    alarm_threshold = np.percentile(err, 95)  # 95th percentile as alarm ref
    if fault_flags is not None:
        # For fault datasets, compute per-fault P_fa
        mask_fault = fault_flags[mask]
        n_FA = int(np.sum((err > alarm_threshold) & ~mask_fault))
    else:
        n_FA = int(np.sum(err > alarm_threshold))

    P_fa = n_FA / n_total if n_total > 0 else 0.0

    return {"P_md": round(P_md, 6), "P_fa": round(P_fa, 6),
            "n_HMI": n_HMI, "n_FA": n_FA, "n_total": n_total,
            "mean_PL": float(np.mean(pl)), "mean_error": float(np.mean(err))}


# ═══════════════════════════════════════════════════════════════════════════════
# ROC computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve and AUC. Higher scores → more likely positive.

    Returns (fpr, tpr, auc).
    """
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    n_pos = np.sum(labels_sorted)
    n_neg = len(labels_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5

    tpr = np.cumsum(labels_sorted) / n_pos
    fpr = np.cumsum(1 - labels_sorted) / n_neg

    # Prepend (0,0)
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc
