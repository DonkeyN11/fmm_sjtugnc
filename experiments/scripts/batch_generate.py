#!/usr/bin/env python3
"""
Batch data generation for all experiment scenarios.

Generates datasets for:
  1. Per-sigma-level: sigma = 1, 5, 10, 15, 20, 25, 30 m (no occlusion, no fault)
  2. sigma=30 with cross-road occlusion (no fault)
  3. sigma=30 with step fault injection (no occlusion)
  4. sigma=30 with both occlusion and fault
  5. Full sigma sweep: 1→30 m, 10 trajectories per level (for parameter sensitivity)

Usage:
  python experiments/scripts/batch_generate.py \
    --output-dir experiments/data \
    --shapefile input/map/haikou/edges.shp \
    --jobs 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
GENERATOR = SCRIPT_DIR / "generate_data_cmm.py"
PROJECT_ROOT = SCRIPT_DIR.parents[1]


def run_generation(output_dir: Path, extra_args: List[str], description: str):
    """Run generate_data_cmm.py with given args."""
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")

    cmd = [
        sys.executable, str(GENERATOR),
        "--output-dir", str(output_dir),
        "-j", "1",  # Single process to avoid issues
    ] + extra_args

    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:] if result.stderr else 'unknown error'}")
        return False

    # Print last few lines of stdout for summary
    lines = result.stdout.strip().split("\n")
    for line in lines[-5:]:
        print(f"  {line}")
    print(f"  OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch data generation for all experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/data"),
                        help="Root data output directory.")
    parser.add_argument("--shapefile", type=Path,
                        default=Path("input/map/haikou/edges.shp"),
                        help="Road network shapefile.")
    parser.add_argument("--jobs", type=int, default=8, help="Number of parallel workers.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--step", nargs="*", choices=["sigma_levels", "occlusion", "fault",
                        "occlusion_fault", "full_sweep"],
                        default=None,
                        help="Which generation steps to run (default: all)")
    args = parser.parse_args()

    base_dir = args.output_dir.resolve()
    shapefile = (PROJECT_ROOT / args.shapefile).resolve()
    if not shapefile.exists():
        raise SystemExit(f"Shapefile not found: {shapefile}")

    base_args = [
        "--shapefile", str(shapefile),
        "--speed", "12.0",
        "--sample-rate", "1.0",
        "--num-sats", "8",
        "--seed", str(args.seed + 1),
    ]

    steps = args.step or ["sigma_levels", "occlusion", "fault", "occlusion_fault", "full_sweep"]

    # ── Step 1: Per-sigma-level datasets ────────────────────────────────────
    sigma_levels = [1, 5, 10, 15, 20, 25, 30]
    n_traj_per_level = 20

    if "sigma_levels" in steps:
        for sigma in sigma_levels:
            sigma_dir = base_dir / f"sigma_{sigma:02d}" / "no_occlusion" / "no_fault"
            desc = f"sigma={sigma}m, count={n_traj_per_level}, no occlusion, no fault"

            success = run_generation(sigma_dir, base_args + [
                "--min-sigma-pr", str(sigma),
                "--max-sigma-pr", str(sigma),
                "--count", str(n_traj_per_level),
                "--points", "1000",
            ], desc)
            if not success:
                print(f"  WARNING: Generation failed for sigma={sigma}")

    # ── Step 2: sigma=30 with occlusion ─────────────────────────────────────
    if "occlusion" in steps:
        occ_dir = base_dir / "sigma_30" / "with_occlusion" / "no_fault"
        run_generation(occ_dir, base_args + [
            "--min-sigma-pr", "30",
            "--max-sigma-pr", "30",
            "--count", str(n_traj_per_level),
            "--points", "1000",
            "--occlusion-angle", "45",
            "--occlusion-elevation-cutoff", "30",
        ], "sigma=30m, occlusion=45°, no fault")

    # ── Step 3: sigma=30 with fault ─────────────────────────────────────────
    if "fault" in steps:
        fault_dir = base_dir / "sigma_30" / "no_occlusion" / "with_fault"
        run_generation(fault_dir, base_args + [
            "--min-sigma-pr", "30",
            "--max-sigma-pr", "30",
            "--count", str(n_traj_per_level),
            "--points", "1000",
            "--fault-probability", "0.3",
            "--fault-magnitude-min", "100",
            "--fault-magnitude-max", "500",
        ], "sigma=30m, fault P=0.3, magnitude U(100,500)m")

    # ── Step 4: sigma=30 with both occlusion and fault ──────────────────────
    if "occlusion_fault" in steps:
        both_dir = base_dir / "sigma_30" / "with_occlusion" / "with_fault"
        run_generation(both_dir, base_args + [
            "--min-sigma-pr", "30",
            "--max-sigma-pr", "30",
            "--count", str(n_traj_per_level),
            "--points", "1000",
            "--occlusion-angle", "45",
            "--occlusion-elevation-cutoff", "30",
            "--fault-probability", "0.3",
            "--fault-magnitude-min", "100",
            "--fault-magnitude-max", "500",
        ], "sigma=30m, occlusion=45°, fault P=0.3, magnitude U(100,500)m")

    # ── Step 5: Full sigma sweep (10 trajs per level) ───────────────────────
    if "full_sweep" in steps:
        sweep_dir = base_dir / "sigma_01_to_30"
        n_levels = len(sigma_levels)
        run_generation(sweep_dir, base_args + [
            "--min-sigma-pr", "1",
            "--max-sigma-pr", "30",
            "--count", str(n_levels * 10),  # 10 per level
            "--points", "1000",
        ], "Full sigma sweep 1→30m, 10 trajectories/level")

    # ── Write sweep summary ─────────────────────────────────────────────────
    summary_path = base_dir / "sweep_summary.json"
    summary: Dict[str, List[str]] = {}
    for d in sorted(base_dir.glob("**")):
        if d.is_dir() and (d / "observations.csv").exists():
            rel = d.relative_to(base_dir)
            key = str(rel)
            files = [f.name for f in d.iterdir() if f.is_file()]
            summary[key] = files
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary written to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
