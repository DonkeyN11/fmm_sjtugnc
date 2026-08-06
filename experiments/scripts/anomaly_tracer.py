#!/usr/bin/env python3
"""
Anomaly tracer for simulation map-matching results.

Iterates over:
  data/simulation/sigma_*/no_occlusion/no_fault/
  data/simulation/sigma_mismatch/pr*/
  data/simulation/sigma_30/*/

For each dataset directory, reads cmm_result.csv, fmm_result.csv,
ground_truth.csv and ground_truth_points.csv, computes per-epoch
point error (matched point vs ground truth point, metres) and edge
correctness (matched cpath edge vs ground-truth point_edge_ids,
direction-aware via the reverse edge map), then flags four anomaly
types per epoch:

  large_error          : point error > 1000 m
  confident_wrong_edge : trustworthiness > 0.9 but matched edge != ground truth
  outlier_10x_median   : point error > 10 x group median
  above_p95            : point error > group 95th percentile

(median / p95 thresholds are computed per group defined by
experiment + sigma + condition + algorithm)

Outputs:
  experiments/output/anomaly_report.csv   - one row per (epoch, anomaly type)
  experiments/output/anomaly_summary.csv  - anomaly-type counts per
    (experiment, sigma, condition, algorithm, traj_id)

Usage:
  python experiments/scripts/anomaly_tracer.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = ROOT / "data" / "simulation"
CONFIG_DIR = ROOT / "experiments" / "config"
OUT_DIR = ROOT / "experiments" / "output"

REPORT_COLUMNS = [
    "experiment", "sigma", "condition", "algorithm", "traj_id", "epoch_seq",
    "timestamp", "point_error_m", "trustworthiness", "edge_correct", "anomaly_type",
]
SUMMARY_COLUMNS = [
    "experiment", "sigma", "condition", "algorithm", "traj_id", "n_epochs",
    "n_wrong_edge", "n_large_error", "n_confident_wrong_edge",
    "n_outlier_10x_median", "n_above_p95", "total_anomalies",
]

ANOMALY_LARGE_ERROR = "large_error"
ANOMALY_CONFIDENT_WRONG = "confident_wrong_edge"
ANOMALY_10X_MEDIAN = "outlier_10x_median"
ANOMALY_ABOVE_P95 = "above_p95"

LARGE_ERROR_M = 1000.0
CONFIDENT_TW = 0.9
MEDIAN_MULT = 10.0

_POINT_RE = re.compile(r"POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)", re.I)


def haversine_deg_to_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Distance in metres between two lon/lat points (local flat-Earth approx)."""
    mlat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon1 - lon2) * 111320.0 * math.cos(mlat)
    dy = (lat1 - lat2) * 111320.0
    return math.sqrt(dx * dx + dy * dy)


def parse_point(wkt: str) -> Optional[Tuple[float, float]]:
    m = _POINT_RE.search(str(wkt))
    return (float(m.group(1)), float(m.group(2))) if m else None


def load_reverse_map() -> Dict[str, str]:
    path = CONFIG_DIR / "reverse_edge_map.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def discover_datasets() -> List[Tuple[Path, str, float, str]]:
    """Return [(dir, experiment, sigma, condition)] with duplicates removed.

    Pattern 1: sigma_*/no_occlusion/no_fault          -> experiment 'sigma_sweep'
    Pattern 2: sigma_mismatch/pr*                     -> experiment 'sigma_mismatch'
    Pattern 3: sigma_30/*/* (all four conditions)     -> experiment 'sigma_30'
    """
    found: Dict[Path, Tuple[str, float, str]] = {}

    def add(dirs, experiment):
        for d in dirs:
            if not d.is_dir():
                continue
            key = d.resolve()
            if key in found:
                continue
            if experiment == "sigma_sweep":
                # path layout: sim_root / sigma_XX / no_occlusion / no_fault
                sigma = float(d.parent.parent.name.replace("sigma_", ""))
                cond = "no_occlusion_no_fault"
            elif experiment == "sigma_mismatch":
                # dir name like pr10_wls20
                m = re.match(r"pr(\d+)", d.name)
                sigma = float(m.group(1)) if m else float("nan")
                cond = d.name
            else:  # sigma_30
                rel = d.relative_to(SIM_ROOT / "sigma_30")
                sigma = 30.0
                cond = "_".join(rel.parts)  # e.g. no_occlusion_with_fault
            found[key] = (experiment, sigma, cond)

    add(sorted(SIM_ROOT.glob("sigma_*/no_occlusion/no_fault")), "sigma_sweep")
    add(sorted(SIM_ROOT.glob("sigma_mismatch/pr*")), "sigma_mismatch")
    add(sorted(SIM_ROOT.glob("sigma_30/*/*")), "sigma_30")

    return [(d, exp, sigma, cond) for d, (exp, sigma, cond) in sorted(found.items())]


def load_gt_points(data_dir: Path) -> Dict[Tuple[str, int], Tuple[float, float]]:
    out: Dict[Tuple[str, int], Tuple[float, float]] = {}
    path = data_dir / "ground_truth_points.csv"
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row["id"].strip()
            seq = int(row["seq"])
            try:
                out[(tid, seq)] = (float(row["x"]), float(row["y"]))
            except (KeyError, ValueError):
                continue
    return out


def load_gt_edges(data_dir: Path) -> Dict[Tuple[str, int], str]:
    """Ground-truth edge id per (tid, seq) from ground_truth.csv point_edge_ids."""
    out: Dict[Tuple[str, int], str] = {}
    path = data_dir / "ground_truth.csv"
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row["id"].strip()
            raw = row.get("point_edge_ids", "").strip()
            if not raw:
                continue
            try:
                edges = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            for seq, eid in enumerate(edges):
                out[(tid, seq)] = str(eid)
    return out


def edge_matches(matched: str, truth: str, rev_map: Dict[str, str]) -> bool:
    matched = str(matched)
    return matched == truth or rev_map.get(matched) == truth


def process_dataset(
    data_dir: Path, experiment: str, sigma: float, condition: str,
    rev_map: Dict[str, str],
) -> List[dict]:
    """Per-epoch records for both algorithms in one dataset directory."""
    gt_points = load_gt_points(data_dir)
    gt_edges = load_gt_edges(data_dir)
    records: List[dict] = []

    for algo in ("cmm", "fmm"):
        result_path = data_dir / f"{algo}_result.csv"
        if not result_path.exists():
            print(f"  SKIP {experiment}/{condition} {algo}: no result file")
            continue
        with open(result_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter=";"):
                if row.get("status", "").strip() != "SUCCESS":
                    continue
                tid = row.get("id", "").strip()
                try:
                    seq = int(row["seq"])
                except (KeyError, ValueError):
                    continue
                gt_pt = gt_points.get((tid, seq))
                if gt_pt is None:
                    continue
                pgeom = parse_point(row.get("pgeom", ""))
                if pgeom is None:
                    continue
                try:
                    tw = float(row.get("trustworthiness", ""))
                except ValueError:
                    tw = float("nan")
                cpath = row.get("cpath", "").strip()
                ts_raw = row.get("timestamp", "").strip()
                try:
                    ts = float(ts_raw) if ts_raw else None
                except ValueError:
                    ts = None

                point_err = haversine_deg_to_m(pgeom[0], pgeom[1], gt_pt[0], gt_pt[1])

                gt_edge = gt_edges.get((tid, seq))
                if cpath and gt_edge is not None:
                    correct = edge_matches(cpath, gt_edge, rev_map)
                else:
                    correct = None

                records.append({
                    "experiment": experiment,
                    "sigma": sigma,
                    "condition": condition,
                    "algorithm": algo,
                    "traj_id": tid,
                    "epoch_seq": seq,
                    "timestamp": ts,
                    "point_error_m": point_err,
                    "trustworthiness": tw,
                    "edge_correct": correct,
                })
    return records


def flag_anomalies(records: List[dict]) -> List[dict]:
    """Append per-record anomaly types; one report row per (record, type)."""
    groups: Dict[Tuple, List[float]] = {}
    for rec in records:
        key = (rec["experiment"], rec["sigma"], rec["condition"], rec["algorithm"])
        groups.setdefault(key, []).append(rec["point_error_m"])

    stats: Dict[Tuple, Tuple[float, float]] = {}
    for key, errs in groups.items():
        arr = np.asarray(errs, dtype=float)
        median = float(np.median(arr))
        p95 = float(np.percentile(arr, 95))
        stats[key] = (median, p95)

    report_rows: List[dict] = []
    for rec in records:
        key = (rec["experiment"], rec["sigma"], rec["condition"], rec["algorithm"])
        median, p95 = stats[key]
        err = rec["point_error_m"]
        tw = rec["trustworthiness"]
        correct = rec["edge_correct"]

        types: List[str] = []
        if err > LARGE_ERROR_M:
            types.append(ANOMALY_LARGE_ERROR)
        if correct is False and tw > CONFIDENT_TW:
            types.append(ANOMALY_CONFIDENT_WRONG)
        if median > 0.0 and err > MEDIAN_MULT * median:
            types.append(ANOMALY_10X_MEDIAN)
        if err > p95:
            types.append(ANOMALY_ABOVE_P95)

        for atype in types:
            row = dict(rec)
            row["anomaly_type"] = atype
            report_rows.append(row)
    return report_rows


def build_summary(report_rows: List[dict], all_records: List[dict]) -> List[dict]:
    """Single-pass aggregation of anomaly-type counts per traj group."""
    per_epoch_types: Dict[Tuple, Dict[str, int]] = {}
    for row in report_rows:
        key = (row["experiment"], row["sigma"], row["condition"],
               row["algorithm"], row["traj_id"])
        counter = per_epoch_types.setdefault(key, {})
        counter[row["anomaly_type"]] = counter.get(row["anomaly_type"], 0) + 1

    epoch_counts: Dict[Tuple, int] = {}
    wrong_counts: Dict[Tuple, int] = {}
    for rec in all_records:
        key = (rec["experiment"], rec["sigma"], rec["condition"],
               rec["algorithm"], rec["traj_id"])
        epoch_counts[key] = epoch_counts.get(key, 0) + 1
        if rec["edge_correct"] is False:
            wrong_counts[key] = wrong_counts.get(key, 0) + 1

    summary: List[dict] = []
    for key in sorted(set(epoch_counts) | set(per_epoch_types)):
        counter = per_epoch_types.get(key, {})
        summary.append({
            "experiment": key[0],
            "sigma": key[1],
            "condition": key[2],
            "algorithm": key[3],
            "traj_id": key[4],
            "n_epochs": epoch_counts.get(key, 0),
            "n_wrong_edge": wrong_counts.get(key, 0),
            "n_large_error": counter.get(ANOMALY_LARGE_ERROR, 0),
            "n_confident_wrong_edge": counter.get(ANOMALY_CONFIDENT_WRONG, 0),
            "n_outlier_10x_median": counter.get(ANOMALY_10X_MEDIAN, 0),
            "n_above_p95": counter.get(ANOMALY_ABOVE_P95, 0),
            "total_anomalies": sum(counter.values()),
        })
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-root", type=Path, default=SIM_ROOT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    sim_root: Path = args.sim_root
    out_dir: Path = args.out_dir
    rev_map = load_reverse_map()
    print(f"reverse_edge_map entries: {len(rev_map)}")

    datasets = discover_datasets()
    print(f"found {len(datasets)} dataset directories:")
    for d, exp, sigma, cond in datasets:
        print(f"  [{exp}] sigma={sigma} condition={cond} -> {d}")

    all_records: List[dict] = []
    for d, exp, sigma, cond in datasets:
        print(f"processing {d}")
        recs = process_dataset(d, exp, sigma, cond, rev_map)
        print(f"  {len(recs)} epochs")
        all_records.extend(recs)

    if not all_records:
        print("no records processed, aborting")
        return 1

    print(f"total epochs analysed: {len(all_records)}")
    report_rows = flag_anomalies(all_records)
    print(f"total anomaly rows (epoch x type): {len(report_rows)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "anomaly_report.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(report_rows)
    print(f"wrote {report_path} ({len(report_rows)} rows)")

    summary = build_summary(report_rows, all_records)
    summary_path = out_dir / "anomaly_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for s in summary:
            w.writerow(s)
    print(f"wrote {summary_path} ({len(summary)} rows)")

    # ── top-N worst anomalies by point error ─────────────────────────────
    top = sorted(report_rows, key=lambda r: r["point_error_m"], reverse=True)[:args.top_n]
    print(f"\nTop {args.top_n} worst anomalies (by point_error_m):")
    print(f"{'#':>3} {'experiment':<14} {'sigma':>5} {'condition':<22} {'algo':<4} "
          f"{'traj_id':>7} {'epoch_seq':>9} {'point_err_m':>11} {'tw':>8} "
          f"{'edge_ok':>7} {'type':<22}")
    for i, r in enumerate(top, 1):
        print(f"{i:>3} {r['experiment']:<14} {r['sigma']:>5g} {r['condition']:<22} "
              f"{r['algorithm']:<4} {r['traj_id']:>7} {r['epoch_seq']:>9} "
              f"{r['point_error_m']:>11.2f} {r['trustworthiness']:>8.4f} "
              f"{r['edge_correct'] if r['edge_correct'] is not None else 'NA':>7} "
              f"{r['anomaly_type']:<22}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
