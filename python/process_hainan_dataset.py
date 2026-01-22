#!/usr/bin/env python3
"""
Hainan Dataset Processing Pipeline
====================================

This script:
1. Reads spp_solution.txt from directories 1.3, 1.4, 1.5, 1.6
2. Converts to cmm_trajectory.csv (projected coordinates)
3. Performs FMM and CMM map matching
4. Generates Mapbox visualization

Usage:
    python process_hainan_dataset.py [--dataset-dir DIR] [--network FILE] [--ubodt FILE]
"""

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Optional dependencies
try:
    from pyproj import CRS, Transformer, Geod
except ImportError:
    print("Warning: pyproj not available. Install with: pip install pyproj")
    Transformer = None
    Geod = None

# NMEA regex pattern
NMEA_RE = re.compile(r"\$[A-Z]{2}[A-Z0-9]{3}[^\r\n$]*")


def dm_to_dd(dm_str: str) -> float:
    """Convert NMEA DDMM.MMMM to decimal degrees."""
    if not dm_str:
        return 0.0
    try:
        val = float(dm_str)
    except ValueError:
        return 0.0
    degrees = int(val / 100)
    minutes = val - degrees * 100
    return degrees + minutes / 60


def iter_nmea_sentences(path: Path) -> Iterable[str]:
    """Iterate over NMEA sentences in a file."""
    try:
        with path.open("rb") as fh:
            data = fh.read()
            text = data.decode("ascii", errors="ignore")
            for match in NMEA_RE.finditer(text):
                line = match.group(0).strip()
                if "*" in line:
                    star = line.find("*")
                    line = line[: min(star + 3, len(line))]
                yield line
    except Exception as e:
        print(f"Warning: Error reading {path}: {e}")
        return


def parse_time_fields(time_str: str) -> Optional[Tuple[int, int, int, int]]:
    """Parse time string into hh, mm, ss, microseconds."""
    if not time_str or len(time_str) < 6:
        return None
    try:
        hh = int(time_str[0:2])
        mm = int(time_str[2:4])
        ss = int(time_str[4:6])
    except ValueError:
        return None
    frac = ""
    if "." in time_str:
        frac = time_str.split(".", 1)[1]
    micro = int((frac + "000000")[:6])
    return hh, mm, ss, micro


def parse_rmc_date(date_str: str) -> Optional[date]:
    """Parse RMC date string."""
    if not date_str or len(date_str) != 6:
        return None
    try:
        day = int(date_str[0:2])
        month = int(date_str[2:4])
        year = int(date_str[4:6]) + 2000
        return date(year, month, day)
    except ValueError:
        return None


def parse_zda_date(parts: List[str]) -> Optional[date]:
    """Parse ZDA date string."""
    if len(parts) < 5:
        return None
    try:
        day = int(parts[2])
        month = int(parts[3])
        year = int(parts[4])
        return date(year, month, day)
    except ValueError:
        return None


def ellipse_to_covariance(smaj: float, smin: float, orient_deg: float) -> Tuple[float, float, float]:
    """Convert ellipse parameters to covariance matrix."""
    theta = math.radians(orient_deg)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    vmaj = smaj * smaj
    vmin = smin * smin
    var_e = vmaj * sin_t * sin_t + vmin * cos_t * cos_t
    var_n = vmaj * cos_t * cos_t + vmin * sin_t * sin_t
    cov_en = (vmaj - vmin) * sin_t * cos_t
    return math.sqrt(var_e), math.sqrt(var_n), cov_en


def determine_utm_epsg(lon_deg: float, lat_deg: float) -> Optional[int]:
    """Determine UTM EPSG code from coordinates."""
    if not (math.isfinite(lon_deg) and math.isfinite(lat_deg)):
        return None
    if lat_deg <= -80.0 or lat_deg >= 84.0:
        return None
    zone = int(math.floor((lon_deg + 180.0) / 6.0)) + 1
    zone = max(1, min(zone, 60))
    base = 32600 if lat_deg >= 0.0 else 32700
    return base + zone


def parse_spp_solution(
    input_path: Path,
    output_path: Path,
    traj_id: int = 1,
    project_utm: bool = True,
    utm_epsg: Optional[int] = None,
    protection_level_scale: float = 2.0,
        default_sigma: float = 1.0,
) -> None:
    """
    Parse spp_solution.txt and convert to cmm_trajectory.csv format.

    Args:
        input_path: Path to spp_solution.txt
        output_path: Path to output cmm_trajectory.csv
        traj_id: Trajectory ID
        project_utm: Whether to project to UTM coordinates
        utm_epsg: UTM EPSG code (auto-detect if None)
        protection_level_scale: Scale factor for protection level
        default_sigma: Default sigma when GST is missing
    """
    records: Dict[str, Dict[str, Optional[object]]] = {}
    time_order: List[str] = []
    dates: List[date] = []

    print(f"Parsing NMEA from: {input_path}")

    for line in iter_nmea_sentences(input_path):
        content = line.split("*", 1)[0]
        parts = [p.strip() for p in content.split(",")]
        if not parts or not parts[0]:
            continue

        msg_type = parts[0][-3:]
        time_str: Optional[str] = None

        # Extract time from various message types
        if msg_type in {"GGA", "GST", "RMC", "ZDA"}:
            if len(parts) > 1:
                time_str = parts[1]
        elif msg_type == "GLL":
            if len(parts) > 5:
                time_str = parts[5]

        if not time_str:
            continue

        if time_str not in records:
            records[time_str] = {
                "lat": None,
                "lon": None,
                "smaj": None,
                "smin": None,
                "orient": None,
                "lat_err": None,
                "lon_err": None,
                "alt_err": None,
                "date": None,
            }
            time_order.append(time_str)

        rec = records[time_str]

        # Parse GGA/GLL/RMC for position
        if msg_type == "GGA":
            if len(parts) > 5 and parts[2] and parts[4]:
                lat = dm_to_dd(parts[2])
                if parts[3] == "S":
                    lat = -lat
                lon = dm_to_dd(parts[4])
                if parts[5] == "W":
                    lon = -lon
                rec["lat"] = lat
                rec["lon"] = lon

        elif msg_type == "GLL":
            if len(parts) > 4 and parts[1] and parts[3]:
                lat = dm_to_dd(parts[1])
                if parts[2] == "S":
                    lat = -lat
                lon = dm_to_dd(parts[3])
                if parts[4] == "W":
                    lon = -lon
                if rec["lat"] is None:
                    rec["lat"] = lat
                if rec["lon"] is None:
                    rec["lon"] = lon

        elif msg_type == "RMC":
            if len(parts) > 6 and parts[3] and parts[5]:
                lat = dm_to_dd(parts[3])
                if parts[4] == "S":
                    lat = -lat
                lon = dm_to_dd(parts[5])
                if parts[6] == "W":
                    lon = -lon
                if rec["lat"] is None:
                    rec["lat"] = lat
                if rec["lon"] is None:
                    rec["lon"] = lon
            if len(parts) > 9 and parts[9]:
                date_obj = parse_rmc_date(parts[9])
                if date_obj:
                    rec["date"] = date_obj
                    dates.append(date_obj)

        elif msg_type == "ZDA":
            date_obj = parse_zda_date(parts)
            if date_obj:
                rec["date"] = date_obj
                dates.append(date_obj)

        # Parse GST for covariance
        elif msg_type == "GST":
            if len(parts) > 7:
                try:
                    rec["smaj"] = float(parts[3]) if parts[3] else None
                    rec["smin"] = float(parts[4]) if parts[4] else None
                    rec["orient"] = float(parts[5]) if parts[5] else None
                    rec["lat_err"] = float(parts[6]) if parts[6] else None
                    rec["lon_err"] = float(parts[7]) if parts[7] else None
                    if len(parts) > 8:
                        rec["alt_err"] = float(parts[8]) if parts[8] else None
                except ValueError:
                    continue

    if not records:
        raise RuntimeError(f"No NMEA sentences found in {input_path}")

    fallback_date = min(dates) if dates else None

    # Build trajectory data
    points: List[Tuple[float, float, float, Dict[str, Optional[float]]]] = []
    for time_str in time_order:
        rec = records[time_str]
        if rec["lat"] is None or rec["lon"] is None:
            continue

        time_fields = parse_time_fields(time_str)
        if not time_fields:
            continue

        date_obj = rec["date"] or fallback_date
        if date_obj is None:
            continue

        hh, mm, ss, micro = time_fields
        dt = datetime(
            date_obj.year,
            date_obj.month,
            date_obj.day,
            hh,
            mm,
            ss,
            micro,
            tzinfo=timezone.utc,
        )
        timestamp = dt.timestamp()
        points.append((timestamp, rec["lat"], rec["lon"], rec))

    if not points:
        raise RuntimeError("No valid position fixes with timestamps were found.")

    # Determine UTM zone
    if project_utm:
        if Transformer is None:
            raise RuntimeError("pyproj is required for UTM projection but is not available.")
        if utm_epsg is None:
            mean_lon = sum(p[2] for p in points) / len(points)
            mean_lat = sum(p[1] for p in points) / len(points)
            utm_epsg = determine_utm_epsg(mean_lon, mean_lat)
        if utm_epsg is None:
            raise RuntimeError("Unable to determine UTM EPSG; use --utm-epsg to set it explicitly.")
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
        print(f"Using UTM EPSG:{utm_epsg} for projection.")
    else:
        transformer = None

    timestamps: List[float] = []
    coords: List[Tuple[float, float]] = []
    covariances: List[List[float]] = []
    protection_levels: List[float] = []

    for timestamp, lat, lon, rec in sorted(points, key=lambda item: item[0]):
        # Project coordinates
        if transformer:
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat

        # Extract covariance
        smaj = rec["smaj"]
        smin = rec["smin"]
        orient = rec["orient"] if rec["orient"] is not None else 0.0

        if smaj is not None and smin is not None:
            sde, sdn, sdne = ellipse_to_covariance(smaj, smin, orient)
        else:
            if rec["lat_err"] is not None and rec["lon_err"] is not None:
                sde, sdn = rec["lon_err"], rec["lat_err"]
            else:
                sde = default_sigma
                sdn = default_sigma
            sdne = 0.0

        # Convert to meters if needed (projected coordinates)
        if not transformer:
            sde, sdn, sdne = sde / 111320.0, sdn / 111320.0, sdne / (111320.0 * 111320.0)

        # Altitude error
        sdu = rec["alt_err"] if rec["alt_err"] is not None else 0.0
        sdeu = 0.0
        sdun = 0.0

        # Calculate protection level
        base_sigma = max(sde, sdn)
        protection_level = protection_level_scale * base_sigma

        coords.append((x, y))
        timestamps.append(round(timestamp, 6))
        covariances.append(
            [
                round(sde, 8),
                round(sdn, 8),
                round(sdu, 8),
                round(sdne, 10),
                round(sdeu, 10),
                round(sdun, 10),
            ]
        )
        protection_levels.append(round(protection_level, 8))

    # Create WKT
    wkt = "LINESTRING (" + ", ".join(f"{x:.8f} {y:.8f}" for x, y in coords) + ")"

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["id", "geom", "timestamps", "covariances", "protection_levels"])
        writer.writerow(
            [
                traj_id,
                wkt,
                json.dumps(timestamps, separators=(",", ":"), ensure_ascii=False),
                json.dumps(covariances, separators=(",", ":"), ensure_ascii=False),
                json.dumps(protection_levels, separators=(",", ":"), ensure_ascii=False),
            ]
        )

    print(f"  Wrote {len(coords)} points to: {output_path}")


def run_fmm_matching(
    network_file: str,
    ubodt_file: str,
    trajectory_file: str,
    output_file: str,
    radius: float = 300.0,
) -> bool:
    """Run FMM map matching."""
    print(f"\n{'='*60}")
    print("Running FMM Map Matching")
    print('='*60)

    fmm_exe = "build/fmm"
    if not Path(fmm_exe).exists():
        print(f"Error: {fmm_exe} not found. Please build the project first.")
        return False

    cmd = [
        fmm_exe,
        "--network", network_file,
        "--ubodt", ubodt_file,
        "--input", trajectory_file,
        "--output", output_file,
        "--radius", str(radius),
        "--output-matched", "matched.csv"
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"FMM matching completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running FMM: {e}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False


def run_cmm_matching(
    network_file: str,
    ubodt_file: str,
    trajectory_file: str,
    output_file: str,
    radius: float = 300.0,
) -> bool:
    """Run CMM map matching."""
    print(f"\n{'='*60}")
    print("Running CMM Map Matching")
    print('='*60)

    cmm_exe = "build/cmm"
    if not Path(cmm_exe).exists():
        print(f"Error: {cmm_exe} not found. Please build the project first.")
        return False

    cmd = [
        cmm_exe,
        "--network", network_file,
        "--ubodt", ubodt_file,
        "--input", trajectory_file,
        "--output", output_file,
        "--search-radius", str(radius),
        "--output-matched", "matched.csv"
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"CMM matching completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running CMM: {e}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False


def generate_mapbox_visualization(
    cmm_traj_file: str,
    cmm_results_file: str,
    fmm_results_file: str,
    output_html: str,
    token: Optional[str] = None,
) -> bool:
    """Generate Mapbox GL visualization."""
    print(f"\n{'='*60}")
    print("Generating Mapbox Visualization")
    print('='*60)

    draw_script = "python/draw_hainan_map.py"
    if not Path(draw_script).exists():
        print(f"Error: {draw_script} not found.")
        return False

    # Extract trajectory ID from cmm_results filename
    traj_id = Path(cmm_traj_file).parent.name  # e.g., "1.3"

    cmd = ["python3", draw_script]

    if token:
        cmd.extend(["--token", token])

    cmd.extend([
        "--cmm-trajectory", cmm_traj_file,
        "--cmm", cmm_results_file,
        "--fmm", fmm_results_file,
        "--ids", traj_id,
        "--output", output_html
    ])

    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"Visualization generated: {output_html}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating visualization: {e}")
        if e.stdout:
            print(f"STDOUT:\n{e.stdout}")
        if e.stderr:
            print(f"STDERR:\n{e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process Hainan dataset: convert trajectories, run FMM/CMM, generate visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all directories (1.3, 1.4, 1.5, 1.6)
  python process_hainan_dataset.py \\
      --dataset-dir dataset_hainan_06 \\
      --network dataset_hainan_06/hainan.shp \\
      --ubodt dataset_hainan_06/hainan.ubodt \\
      --radius 300

  # Process single directory
  python process_hainan_dataset.py \\
      --dataset-dir dataset_hainan_06/1.3 \\
      --network dataset_hainan_06/hainan.shp \\
      --ubodt dataset_hainan_06/hainan.ubodt

  # Skip matching, only convert trajectories
  python process_hainan_dataset.py --no-matching --dataset-dir dataset_hainan_06/1.3
        """
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset_hainan_06",
        help="Root directory containing 1.3, 1.4, 1.5, 1.6 folders"
    )

    parser.add_argument(
        "--network",
        type=str,
        default="dataset_hainan_06/hainan.shp",
        help="Path to road network shapefile"
    )

    parser.add_argument(
        "--ubodt",
        type=str,
        default="dataset_hainan_06/hainan.ubodt",
        help="Path to UBODT file"
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=300.0,
        help="Search radius for map matching (default: 300m)"
    )

    parser.add_argument(
        "--no-matching",
        action="store_true",
        help="Skip map matching, only convert trajectories"
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip map matching, only convert trajectories"
    )

    parser.add_argument(
        "--mapbox-token",
        type=str,
        default=None,
        help="Mapbox access token (default: use built-in token)"
    )

    args = parser.parse_args()

    # Directories to process
    subdirs = ["1.3", "1.4", "1.5", "1.6"]
    base_dir = Path(args.dataset_dir)

    print("="*70)
    print("Hainan Dataset Processing Pipeline")
    print("="*70)
    print(f"Dataset directory: {base_dir}")
    print(f"Network: {args.network}")
    print(f"UBODT: {args.ubodt}")
    print(f"Search radius: {args.radius}m")
    print(f"Subdirectories: {', subdirs}")
    print("="*70)

    # Check required files exist
    if not Path(args.network).exists():
        print(f"\nError: Network file not found: {args.network}")
        return 1

    if not Path(args.ubodt).exists():
        print(f"\nError: UBODT file not found: {args.ubodt}")
        return 1

    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = base_dir / subdir
        if not subdir_path.exists():
            print(f"\nWarning: Directory not found: {subdir_path}, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"Processing: {subdir}")
        print('='*70)

        # Input and output paths
        spp_file = subdir_path / "实时定位结果" / "spp_solution.txt"
        mr_dir = subdir_path / "mr"
        mr_dir.mkdir(parents=True, exist_ok=True)

        cmm_traj_file = mr_dir / "cmm_trajectory.csv"
        cmm_results_file = mr_dir / "cmm_results.csv"
        fmm_results_file = mr_dir / "fmm_results.csv"
        output_html = mr_dir / "mapbox_view.html"

        if not spp_file.exists():
            print(f"Warning: spp_solution.txt not found in {subdir_path}, skipping...")
            continue

        # Step 1: Convert to cmm_trajectory.csv
        print("\nStep 1/4: Converting to cmm_trajectory.csv")
        print("-" * 50)
        try:
            parse_spp_solution(
                input_path=spp_file,
                output_path=cmm_traj_file,
                traj_id=int(subdir.replace(".", "")),  # Use subdir number as ID
                project_utm=True,
                utm_epsg=None,  # Auto-detect
                protection_level_scale=2.0,
                default_sigma=1.0
            )
        except Exception as e:
            print(f"Error converting {spp_file}: {e}")
            continue

        # Step 2: Run FMM matching
        if not args.no_matching:
            print("\nStep 2/4: Running FMM map matching")
            print("-" * 50)
            fmm_success = run_fmm_matching(
                network_file=args.network,
                ubodt_file=args.ubodt,
                trajectory_file=str(cmm_traj_file),
                output_file=str(fmm_results_file),
                radius=args.radius
            )
            if not fmm_success:
                print("FMM matching failed, continuing...")

        # Step 3: Run CMM matching
        if not args.no_matching:
            print("\nStep 3/4: Running CMM map matching")
            print("-" * 50)
            cmm_success = run_cmm_matching(
                network_file=args.network,
                ubodt_file=args.ubodt,
                trajectory_file=str(cmm_traj_file),
                output_file=str(cmm_results_file),
                radius=args.radius
            )
            if not cmm_success:
                print("CMM matching failed, continuing...")

        # Step 4: Generate visualization
        if not args.no_visualization and not args.no_matching:
            print("\nStep 4/4: Generating Mapbox visualization")
            print("-" * 50)

            # Check if result files exist
            if not Path(cmm_results_file).exists():
                print(f"Warning: CMM results not found: {cmm_results_file}, skipping visualization...")
            elif not Path(fmm_results_file).exists():
                print(f"Warning: FMM results not found: {fmm_results_file}, skipping visualization...")
            else:
                generate_mapbox_visualization(
                    cmm_traj_file=str(cmm_traj_file),
                    cmm_results_file=str(cmm_results_file),
                    fmm_results_file=str(fmm_results_file),
                    output_html=str(output_html),
                    token=args.mapbox_token
                )

    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)

    print("\nSummary:")
    print("--------")
    print(f"Processed directories: {', subdirs}")
    print(f"Output location: Each subdir/mr/ directory contains:")
    print("  - cmm_trajectory.csv   (converted trajectory)")

    if not args.no_matching:
        print("  - cmm_results.csv      (CMM matching results)")
        print("  - fmm_results.csv      (FMM matching results)")

    if not args.no_visualization and not args.no_matching:
        print("  - mapbox_view.html      (interactive map visualization)")

    print("\nNext steps:")
    print("  1. Open the generated HTML files in a web browser")
    print("  2. Use the checkboxes to toggle CMM/FMM observation layers")
    print("  3. Hover over points to see detailed information")
    print("  4. Use the search box to find specific sequence numbers")

    return 0


if __name__ == "__main__":
    main()
