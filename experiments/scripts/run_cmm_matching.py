#!/usr/bin/env python3
"""
Run CMM matching on synthetic datasets and compute segment-level accuracy.

Pipeline:
  1. Reproject UTM observations to WGS84 (with covariance Jacobian rotation)
  2. Run CMM on reprojected data
  3. Compare cpath with ground truth point_edge_ids → segment accuracy

Usage:
  python experiments/scripts/run_cmm_matching.py \
    --data-root experiments/data \
    --cmm-bin build/cmm \
    --output-dir experiments/output/cmm_matching
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

PROJ_DATA = os.environ.get("PROJ_DATA", "")

def reproject_utm_to_wgs84(src_path: Path, dst_path: Path) -> Path:
    """Reproject UTM observations + covariance to WGS84 (lon, lat) degrees.

    Covariance Jacobian at latitude lat:
      dx_deg/dx_m = 1 / (111320 * cos(lat_rad))
      dy_deg/dy_m = 1 / 111320
    """
    import pyproj
    utm_epsg = 32649  # UTM zone 49N for Hainan
    transformer = pyproj.Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    METERS_PER_DEG_LAT = 111320.0

    with open(src_path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter=";")
        new_header = ["id", "timestamp", "x", "y",
                      "sde", "sdn", "sdu", "sdne", "sdeu", "sdun",
                      "protection_level"]
        with open(dst_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=new_header, delimiter=";")
            writer.writeheader()
            for row in reader:
                x_m = float(row["x"])
                y_m = float(row["y"])
                lon, lat = transformer.transform(x_m, y_m)

                cos_lat = math.cos(math.radians(lat))
                jx = 1.0 / (METERS_PER_DEG_LAT * cos_lat)
                jy = 1.0 / METERS_PER_DEG_LAT

                sde_m = float(row["sde"])
                sdn_m = float(row["sdn"])
                sdne_m = float(row.get("sdne", 0))

                writer.writerow({
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "x": f"{lon:.8f}",
                    "y": f"{lat:.8f}",
                    "sde": f"{abs(jx) * sde_m:.12f}",
                    "sdn": f"{abs(jy) * sdn_m:.12f}",
                    "sdu": row.get("sdu", "0.0"),
                    "sdne": f"{jx * jy * sdne_m:.16f}",
                    "sdeu": row.get("sdeu", "0.0"),
                    "sdun": row.get("sdun", "0.0"),
                    "protection_level": f"{float(row['protection_level']) * max(abs(jx), abs(jy)):.12f}",
                })
    return dst_path


def build_cmm_xml(gps_csv: str, out_file: str, network_shp: str,
                  lag_steps: int = 5, k: int = 8, radius: float = 300.0) -> Path:
    """Create a temp CMM config XML pointing at the reprojected GPS CSV."""
    xml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<config>
    <parameters>
        <k>{k}</k>
        <radius>{radius}</radius>
        <gps_error>50.0</gps_error>
        <reverse_tolerance>0.0</reverse_tolerance>
        <normalized>false</normalized>
        <use_mahalanobis>true</use_mahalanobis>
        <filtered>false</filtered>
        <max_interval>180.0</max_interval>
        <lag_steps>{lag_steps}</lag_steps>
        <phmi>0.00001</phmi>
        <protection_level_multiplier>10</protection_level_multiplier>
        <phmi_pl_multiplier>5</phmi_pl_multiplier>
        <h0_prior_log_odds>0</h0_prior_log_odds>
    </parameters>
    <input>
        <gps>
            <file>{gps_csv}</file>
            <id_column>id</id_column>
            <geom_column>test</geom_column>
            <timestamp_column>timestamp</timestamp_column>
            <gps_format>cmm</gps_format>
        </gps>
        <network>
            <file>{network_shp}</file>
            <id_column>fid</id_column>
            <source_column>u</source_column>
            <target_column>v</target_column>
        </network>
        <ubodt>
            <file></file>
        </ubodt>
        <gps_point>true</gps_point>
    </input>
    <output>
        <file>{out_file}</file>
    </output>
</config>"""
    tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
    tmp.write(xml_content)
    tmp.close()
    return Path(tmp.name)


def run_cmm(cmm_bin: Path, xml_path: Path, cwd: Path) -> str:
    """Run CMM, return output CSV path."""
    subprocess.run([str(cmm_bin), str(xml_path)], check=True,
                   capture_output=True, text=True, cwd=str(cwd), timeout=300)
    tree = ET.parse(xml_path)
    return tree.getroot().find("output").find("file").text


def compute_segment_accuracy(mr_csv: Path, gt_csv: Path) -> dict:
    """Compare CMM cpath with ground truth point_edge_ids per epoch."""
    # Load ground truth edge IDs per (traj_id, seq)
    gt_edges = {}
    with open(gt_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row["id"].strip()
            if "point_edge_ids" in row:
                try:
                    edges = json.loads(row["point_edge_ids"])
                except (json.JSONDecodeError, TypeError):
                    continue
                gt_edges[tid] = edges

    if not gt_edges:
        # Try ground_truth.csv instead
        gt_csv2 = gt_csv.parent / "ground_truth.csv"
        if gt_csv2.exists():
            with open(gt_csv2, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    tid = row["id"].strip()
                    if "point_edge_ids" in row:
                        try:
                            edges = json.loads(row["point_edge_ids"])
                        except (json.JSONDecodeError, TypeError):
                            continue
                        gt_edges[tid] = edges

    # Load CMM matched cpath per epoch
    matched = {}
    with open(mr_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            tid = row.get("id", "").strip()
            cpath = row.get("cpath", "").strip()
            seq = int(row.get("seq", 0))
            if tid not in matched:
                matched[tid] = {}
            matched[tid][seq] = cpath

    # Per-trajectory accuracy
    results = {}
    total_correct = 0
    total_epochs = 0
    for tid, truth_list in gt_edges.items():
        if tid not in matched:
            continue
        n_correct = 0
        n_total = 0
        for seq, true_edge in enumerate(truth_list):
            if seq in matched[tid]:
                n_total += 1
                if str(matched[tid][seq]) == str(true_edge):
                    n_correct += 1
        if n_total > 0:
            results[tid] = {"correct": n_correct, "total": n_total,
                            "accuracy": n_correct / n_total}
            total_correct += n_correct
            total_epochs += n_total

    return {
        "per_trajectory": results,
        "overall_accuracy": total_correct / total_epochs if total_epochs > 0 else 0,
        "total_correct": total_correct,
        "total_epochs": total_epochs,
    }


def process_dataset(data_dir: Path, cmm_bin: Path, output_dir: Path,
                    network_shp: str, project_root: Path) -> dict | None:
    """Process one dataset directory: reproject, run CMM, compute accuracy."""
    obs_csv = data_dir / "observations.csv"
    if not obs_csv.exists():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Observations are already in WGS84 (generate_data_cmm.py outputs WGS84 now)
    # Run CMM directly
    mr_out = str(output_dir / "cmm_match_result.csv")
    xml_path = build_cmm_xml(str(obs_csv.resolve()), mr_out, network_shp)

    try:
        run_cmm(cmm_bin, xml_path, project_root)
    except subprocess.CalledProcessError as e:
        print(f"    CMM FAILED: {e.stderr[:200] if e.stderr else str(e)}")
        os.unlink(xml_path)
        return None
    finally:
        if os.path.exists(xml_path.name):
            os.unlink(xml_path.name)

    # Compute accuracy
    mr_file = Path(mr_out)
    if not mr_file.exists():
        return None

    gt_csv = data_dir / "ground_truth.csv"
    acc = compute_segment_accuracy(mr_file, gt_csv)
    return acc


def main():
    parser = argparse.ArgumentParser(description="Run CMM matching and evaluate segment accuracy.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cmm-bin", type=Path, default=Path("build/cmm"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/output/cmm_matching"))
    parser.add_argument("--network-shp", type=str, default="input/map/hainan/edges.shp")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific dataset names (default: all sigma_*/no_occlusion/no_fault)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    if args.datasets:
        dataset_dirs = [args.data_root / d for d in args.datasets]
    else:
        dataset_dirs = sorted(args.data_root.glob("sigma_*/no_occlusion/no_fault"))

    print(f"Processing {len(dataset_dirs)} datasets...")
    all_results = {}

    for data_dir in dataset_dirs:
        label = next((p for p in data_dir.parts if p.startswith("sigma_")), data_dir.name)
        print(f"\n  {label} ...", end=" ", flush=True)

        out_dir = args.output_dir / label
        acc = process_dataset(data_dir, args.cmm_bin, out_dir, args.network_shp, project_root)

        if acc:
            overall = acc["overall_accuracy"]
            print(f"accuracy={overall*100:.1f}% ({acc['total_correct']}/{acc['total_epochs']})")
            all_results[label] = {"accuracy": overall, **acc}
        else:
            print("SKIPPED")

    # Summary
    if all_results:
        out_csv = args.output_dir / "segment_accuracy_summary.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "accuracy", "correct", "total"])
            for label, r in sorted(all_results.items()):
                w.writerow([label, f"{r['accuracy']:.4f}", r["total_correct"], r["total_epochs"]])
        print(f"\n  Summary: {out_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
