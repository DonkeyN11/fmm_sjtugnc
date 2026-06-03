#!/usr/bin/env python3
"""
Align all real-data sources by (trajectory, timestamp) into a unified CSV.

Inputs (in experiments/data/real_data/):
  cmm_input_points.csv   — SPP observations (id, timestamp, x, y, sde, sdn, ...)
  cmm_result.csv         — CMM match result (id, timestamp, pgeom, cpath, trustworthiness, ...)
  fmm_result.csv         — FMM match result (id, timestamp, pgeom, cpath, trustworthiness, ...)
  ground_truth_points.csv — RTK ground truth (id, timestamp, x, y)

Output:
  experiments/data/real_data/aligned.csv
    id, uni_seq, timestamp,
    obs_x, obs_y, obs_sde, obs_sdn, obs_sdne,
    cmm_x, cmm_y, cmm_tw, cmm_trustworthiness, cmm_cpath,
    fmm_x, fmm_y, fmm_tw, fmm_trustworthiness, fmm_cpath,
    gt_x, gt_y

Usage:
  python experiments/scripts/align_real_data.py
"""

import csv, re, math
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/real_data"


def parse_point(wkt):
    """Extract (lon, lat) from WKT POINT(lon lat)."""
    m = re.search(r"POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)", str(wkt), re.I)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def load_csv(path, key_cols, val_cols):
    """Load CSV rows into dict[(id, timestamp)] = dict of values."""
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row["id"].strip()
            try:
                ts = round(float(row["timestamp"]))
            except (KeyError, ValueError):
                continue
            key = (tid, ts)
            vals = {}
            for c in val_cols:
                if c in row:
                    vals[c] = row[c]
            # Parse pgeom if present
            if "pgeom" in row:
                x, y = parse_point(row["pgeom"])
                vals["x"] = x
                vals["y"] = y
            result[key] = vals
    return result


def main():
    # ── Load all data sources keyed by (id, timestamp) ──
    print("Loading SPP observations...")
    obs = load_csv(DATA_DIR / "cmm_input_points.csv",
                   ["id", "timestamp"],
                   ["x", "y", "sde", "sdn", "sdne", "sdu", "protection_level"])

    print("Loading CMM results...")
    cmm = load_csv(DATA_DIR / "cmm_result.csv",
                   ["id", "timestamp"],
                   ["pgeom", "cpath", "trustworthiness", "ep", "tp", "n_best_trustworthiness",
                    "delta_entropy", "posterior_entropy", "h0_lambda", "status"])

    print("Loading FMM results...")
    fmm = load_csv(DATA_DIR / "fmm_result.csv",
                   ["id", "timestamp"],
                   ["pgeom", "cpath", "trustworthiness", "ep", "tp", "sp_dist", "eu_dist"])

    print("Loading RTK ground truth...")
    gt = load_csv(DATA_DIR / "ground_truth_points.csv",
                  ["id", "timestamp"],
                  ["x", "y"])

    # Load GT road segments (keyed by id, seq — same as uni_seq ordering)
    print("Loading GT edges...")
    gt_edges = {}  # (id, uni_seq) -> edge_id
    gt_path = DATA_DIR / "ground_truth.csv"
    if gt_path.exists():
        with open(gt_path, newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                gt_edges[(row["id"].strip(), int(row["seq"]))] = row["edge_id"].strip()

    # ── Collect all unique (id, timestamp) keys ──
    all_keys = set(obs.keys()) | set(cmm.keys()) | set(fmm.keys()) | set(gt.keys())
    print(f"\nTotal unique (id, ts) keys: {len(all_keys)}")

    # ── Build aligned rows ──
    rows = []
    for tid in sorted(set(k[0] for k in all_keys)):
        traj_keys = sorted([k for k in all_keys if k[0] == tid], key=lambda k: int(k[1]))
        uni_seq = 0
        for key in traj_keys:
            _, ts = key
            row = {"id": tid, "uni_seq": uni_seq, "timestamp": ts}

            # SPP observation
            o = obs.get(key, {})
            row["obs_x"] = o.get("x", "")
            row["obs_y"] = o.get("y", "")
            row["obs_sde"] = o.get("sde", "")
            row["obs_sdn"] = o.get("sdn", "")
            row["obs_sdne"] = o.get("sdne", "")

            # CMM
            c = cmm.get(key, {})
            row["cmm_x"] = c.get("x", "")
            row["cmm_y"] = c.get("y", "")
            row["cmm_tw"] = c.get("trustworthiness", "")
            row["cmm_cpath"] = c.get("cpath", "")
            row["cmm_status"] = c.get("status", "")
            row["cmm_delta_entropy"] = c.get("delta_entropy", "")
            row["cmm_posterior_entropy"] = c.get("posterior_entropy", "")

            # FMM
            f = fmm.get(key, {})
            row["fmm_x"] = f.get("x", "")
            row["fmm_y"] = f.get("y", "")
            row["fmm_tw"] = f.get("trustworthiness", "")
            row["fmm_cpath"] = f.get("cpath", "")

            # Ground truth
            g = gt.get(key, {})
            row["gt_x"] = g.get("x", "")
            row["gt_y"] = g.get("y", "")
            row["gt_edge"] = gt_edges.get((tid, uni_seq), "")

            rows.append(row)
            uni_seq += 1

    # ── Write aligned CSV ──
    out = DATA_DIR / "aligned.csv"
    fields = ["id", "uni_seq", "timestamp",
              "obs_x", "obs_y", "obs_sde", "obs_sdn", "obs_sdne",
              "cmm_x", "cmm_y", "cmm_tw", "cmm_cpath",
              "cmm_status", "cmm_delta_entropy", "cmm_posterior_entropy",
              "fmm_x", "fmm_y", "fmm_tw", "fmm_cpath",
              "gt_x", "gt_y", "gt_edge"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
