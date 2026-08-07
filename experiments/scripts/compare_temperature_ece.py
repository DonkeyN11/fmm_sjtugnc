#!/usr/bin/env python3
"""Compare old (pre-temperature-adapt) vs new (temperature_adapt=true) CMM results
across the sample-rate datasets: sigma in {5, 15, 25} m x sample interval {1,2,5,10} s.

Result files per dataset (sr=1 -> base dir, sr>1 -> subsample_{sr}s dir):
  OLD = cmm_result.csv.bak0806   (regenerated Aug 6 10:42, no adaptive temperature)
  NEW = cmm_result.csv           (regenerated Aug 6 20:12, temperature_adapt=true)

Metrics per result file (per-row aligned: trustworthiness <-> edge correctness label
of the same epoch):
  - ECE       : 10-bin expected calibration error; confidence = trustworthiness,
                label = edge_match(cpath, gt edge) with reverse-edge-map awareness
  - seg_acc   : mean edge correctness
  - TW mean/std: over valid trustworthiness probabilities in [0,1] (overflow rows,
                i.e. inf/huge values produced by the tempered softmax at the
                terminal epoch, are excluded and counted)
  - med err   : median point error (haversine m vs ground_truth_points.csv)

Output: experiments/output/3_full_matching/temperature_comparison.csv
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT / "data/simulation"
OUT_CSV = PROJECT / "experiments/output/3_full_matching/temperature_comparison.csv"
REVERSE_MAP_PATH = PROJECT / "experiments/config/reverse_edge_map.json"

SIGMAS = [5, 15, 25]          # sigma_05 / sigma_15 / sigma_25 (m)
INTERVALS = [1, 2, 5, 10]      # sample intervals (s)
OLD_SUFFIX = "cmm_result.csv.bak0806"   # pre-temperature result
NEW_SUFFIX = "cmm_result.csv"           # temperature_adapt=true result

COLUMNS = ["sigma", "sr",
           "old_ECE", "new_ECE", "old_acc", "new_acc",
           "TW_mean_old", "TW_mean_new", "TW_std_old", "TW_std_new",
           "old_med_err", "new_med_err"]


# ──────────────────────────────────────────────────────────────────────────────
# Ground truth loading
# ──────────────────────────────────────────────────────────────────────────────

def load_ground_truth(data_dir: Path):
    """Return gt points/edges indexed by (id, timestamp).

    CMM result CSVs renumber `seq` to the per-file point index (0..N-1), so for
    subsampled datasets (interval > 1) seq no longer matches the full-resolution
    ground truth. Matching is therefore done on timestamps, exactly like
    exp3_full_matching.compute_metrics: gt keys are stored under both the exact
    fractional timestamp and its rounded-int value (CMM writes int timestamps).
    """
    gt_points_by_ts = {}
    gt_edges_by_ts = {}
    seq_to_ts = {}

    gt_points = data_dir / "ground_truth_points.csv"
    if gt_points.exists():
        with open(gt_points, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                tid = row["id"].strip()
                ts = float(row["timestamp"])
                seq_to_ts[(tid, int(row["seq"]))] = ts
                gt_points_by_ts[(tid, ts)] = (float(row["x"]), float(row["y"]))
                gt_points_by_ts[(tid, int(round(ts)))] = (float(row["x"]), float(row["y"]))

    gt_edges = data_dir / "ground_truth.csv"
    if gt_edges.exists() and seq_to_ts:
        with open(gt_edges, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                tid = row["id"].strip()
                try:
                    edges = json.loads(row["point_edge_ids"])
                except (json.JSONDecodeError, TypeError):
                    continue
                # point_edge_ids is indexed by the full-resolution seq; map each
                # seq back to its timestamp so subsampled results can be matched.
                for seq, eid in enumerate(edges):
                    ts = seq_to_ts.get((tid, seq))
                    if ts is None:
                        continue
                    gt_edges_by_ts[(tid, ts)] = str(eid)
                    gt_edges_by_ts[(tid, int(round(ts)))] = str(eid)

    return gt_points_by_ts, gt_edges_by_ts


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_point(wkt: str):
    m = re.search(r"POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)", str(wkt), re.I)
    return (float(m.group(1)), float(m.group(2))) if m else None


def haversine_deg_to_m(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111320.0 * math.cos(mlat)
    dy = (lat1 - lat2) * 111320.0
    return math.sqrt(dx * dx + dy * dy)


def load_reverse_map():
    if REVERSE_MAP_PATH.exists():
        with open(REVERSE_MAP_PATH) as f:
            return json.load(f)
    return {}


REVERSE_MAP = load_reverse_map()


def edge_match(matched: str, truth: str) -> bool:
    return str(matched) == str(truth) or REVERSE_MAP.get(str(matched)) == str(truth)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_ece(confidences, labels, n_bins=10):
    """10-bin expected calibration error; confidence vs binary label, per-row aligned.

    Follows the convention of exp3_full_matching.compute_ece: values outside the
    bin domain [0,1] fall into no bin and are dropped from the ECE (reported
    separately via the overflow count).
    """
    confidences = np.asarray(confidences, dtype=float)
    labels = np.asarray(labels, dtype=float)
    N = len(confidences)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        mc = float(np.mean(confidences[idxs]))
        acc = float(np.mean(labels[idxs]))
        gap = abs(mc - acc)
        ece += len(idxs) / N * gap
        mce = max(mce, gap)
    return ece, mce


def lookup_by_ts(index, tid, ts_str):
    """Look up (tid, ts) in a dict that holds both exact and rounded-int keys."""
    if not ts_str:
        return None
    try:
        ts = float(ts_str)
    except ValueError:
        return None
    val = index.get((tid, ts))
    if val is None:
        val = index.get((tid, int(round(ts))))
    return val


def compute_metrics(data_dir: Path, mr_csv: Path, gt_points_by_ts, gt_edges_by_ts):
    """Per-row aligned metrics from one CMM result file (timestamp-based matching)."""
    if not mr_csv.exists():
        return None

    trusts = []
    labels = []
    errors = []
    overflow = 0  # rows whose trustworthiness is not a valid probability in [0,1]

    with open(mr_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row.get("id", "").strip()
            ts_str = row.get("timestamp", "").strip()

            try:
                tw = float(row.get("trustworthiness", "0"))
            except ValueError:
                tw = float("nan")

            if not (0.0 <= tw <= 1.0):
                overflow += 1

            gt_edge = lookup_by_ts(gt_edges_by_ts, tid, ts_str)
            if gt_edge is not None:
                trusts.append(tw)
                labels.append(float(edge_match(str(row.get("cpath", "")), gt_edge)))

            pt = parse_point(row.get("pgeom", ""))
            gt_pt = lookup_by_ts(gt_points_by_ts, tid, ts_str)
            if pt is not None and gt_pt is not None:
                errors.append(haversine_deg_to_m(pt[0], pt[1], gt_pt[0], gt_pt[1]))

    if not labels:
        return {"n": len(trusts), "error": "no labels", "overflow": overflow}

    trusts = np.asarray(trusts)
    labels = np.asarray(labels)
    valid = (trusts >= 0.0) & (trusts <= 1.0) & np.isfinite(trusts)

    ece, mce = compute_ece(trusts, labels)
    return {
        "n": len(labels),
        "seg_accuracy": float(np.mean(labels)),
        "ece_tw": ece,
        "mce_tw": mce,
        "tw_mean": float(np.mean(trusts[valid])) if valid.any() else float("nan"),
        "tw_std": float(np.std(trusts[valid])) if valid.any() else float("nan"),
        "overflow": overflow,
        "point_error_median": float(np.median(errors)) if errors else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def dataset_dir(sigma: int, interval: int) -> Path:
    base = DATA_ROOT / f"sigma_{sigma:02d}/no_occlusion/no_fault"
    return base if interval == 1 else base / f"subsample_{interval}s"


def main():
    rows = []
    print(f"{'sigma':>5} {'sr':>3} | {'old_ECE':>8} {'new_ECE':>8} | "
          f"{'old_acc':>8} {'new_acc':>8} | {'TW_mean_old':>11} {'TW_mean_new':>11} "
          f"{'TW_std_old':>10} {'TW_std_new':>10} | {'old_med':>8} {'new_med':>8}")
    print("-" * 118)

    for sigma in SIGMAS:
        for sr in INTERVALS:
            d = dataset_dir(sigma, sr)
            gt_points_by_ts, gt_edges_by_ts = load_ground_truth(d)

            old = compute_metrics(d, d / OLD_SUFFIX, gt_points_by_ts, gt_edges_by_ts)
            new = compute_metrics(d, d / NEW_SUFFIX, gt_points_by_ts, gt_edges_by_ts)
            if old is None or new is None or "error" in old or "error" in new:
                print(f"{sigma:>5} {sr:>3} | missing result file", file=sys.stderr)
                continue

            rows.append({
                "sigma": sigma,
                "sr": sr,
                "old_ECE": old["ece_tw"],
                "new_ECE": new["ece_tw"],
                "old_acc": old["seg_accuracy"],
                "new_acc": new["seg_accuracy"],
                "TW_mean_old": old["tw_mean"],
                "TW_mean_new": new["tw_mean"],
                "TW_std_old": old["tw_std"],
                "TW_std_new": new["tw_std"],
                "old_med_err": old["point_error_median"],
                "new_med_err": new["point_error_median"],
            })

            print(f"{sigma:>5} {sr:>3} | {old['ece_tw']:8.4f} {new['ece_tw']:8.4f} | "
                  f"{old['seg_accuracy']*100:7.2f}% {new['seg_accuracy']*100:7.2f}% | "
                  f"{old['tw_mean']:11.4f} {new['tw_mean']:11.4f} "
                  f"{old['tw_std']:10.4f} {new['tw_std']:10.4f} | "
                  f"{old['point_error_median']:8.2f} {new['point_error_median']:8.2f}")

            for tag, m in (("OLD", old), ("NEW", new)):
                if m["overflow"]:
                    print(f"    note: {tag} sigma={sigma} sr={sr}: "
                          f"{m['overflow']} rows with non-[0,1] trustworthiness "
                          f"(tempered-softmax overflow), excluded from TW stats/ECE")

    # ── Save CSV ──
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT_CSV}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
