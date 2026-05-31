#!/usr/bin/env python3
"""
Hainan Dataset Processing Pipeline (LLA Version with Rigorous Covariance Transform)
====================================================================================

This script:
1. Reads spp_solution.txt from directories
2. Converts to cmm_trajectory.csv (LLA coordinates with rigorous covariance transform)
3. Performs FMM and CMM map matching
4. Generates Mapbox visualization

Key Features:
- Uses WGS84 ellipsoid parameters for coordinate conversion
- Computes Jacobian matrix for rigorous covariance transformation: P_lla = J * P_utm * J^T
- Converts LLA protection level to meters for physical interpretation
- Double precision (float64) throughout

Usage:
    python process_hainan_dataset_lla_rigid.py [--dataset-dir DIR] [--network FILE] [--ubodt FILE]
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

# Increase CSV field size limit to handle large trajectories
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

# Add build path for Python module
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
build_path = project_root / 'build' / 'python'

if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
else:
    sys.path.remove(str(script_dir))
    sys.path.insert(0, str(script_dir))

if (build_path / "fmm.py").exists():
    if str(build_path) not in sys.path:
        sys.path.insert(0, str(build_path))

try:
    from fmm import *
    import fmm as fmm_module
    print(f"Loaded fmm from: {fmm_module.__file__}")
except ImportError:
    print(f"Error: fmm Python module not found. sys.path: {sys.path}")
    sys.exit(1)

# Optional: pyproj for precise ellipsoid calculations
# If not available, use simplified approximation
try:
    from pyproj import Geod
    PYPROJ_AVAILABLE = True
except ImportError:
    print("Info: pyproj not available. Using WGS84 approximation.")
    PYPROJ_AVAILABLE = False

# NMEA regex pattern
NMEA_RE = re.compile(r"\$[A-Z]{2}[A-Z]{3}[^\r\n$]*")


# ============================================================================
# WGS84 Ellipsoid Parameters
# ============================================================================
WGS84_A = 6378137.0  # Semi-major axis (m)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # First eccentricity squared
WGS84_EP2 = WGS84_E2 / (1 - WGS84_E2)  # Second eccentricity squared


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
    """Iterate over NMEA sentences in a file line by line."""
    try:
        with path.open("r", encoding="ascii", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                for match in NMEA_RE.finditer(line):
                    content = match.group(0)
                    if "*" in content:
                        star = content.find("*")
                        content = content[: min(star + 3, len(content))]
                    yield content
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


def ellipse_to_covariance_degrees(
    smaj_m: float,
    smin_m: float,
    orient_deg: float,
    lat_deg: float
) -> Tuple[float, float, float]:
    """
    Convert GST ellipse parameters (meters) to LLA covariance (degrees).

    Args:
        smaj_m: Semi-major axis in meters (from GST)
        smin_m: Semi-minor axis in meters (from GST)
        orient_deg: Orientation in degrees (from North, clockwise)
        lat_deg: Latitude in degrees (for meters-to-degrees conversion)

    Returns:
        (sigma_lon, sigma_lat, sigma_lon_lat) in degrees
    """
    # Step 1: Convert ellipse to covariance in meters
    theta = math.radians(orient_deg)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    vmaj_m = smaj_m * smaj_m
    vmin_m = smin_m * smin_m

    # Variance in East/North directions (meters^2)
    var_e_m2 = vmaj_m * sin_t * sin_t + vmin_m * cos_t * cos_t
    var_n_m2 = vmaj_m * cos_t * cos_t + vmin_m * sin_t * sin_t
    cov_en_m2 = (vmaj_m - vmin_m) * sin_t * cos_t

    # Step 2: Convert meters^2 to degrees^2
    # Use WGS84 to get precise conversion at this latitude
    lat_rad = math.radians(lat_deg)
    sin_lat = math.sin(lat_rad)
    W = math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)

    # Radius of curvature in prime vertical (East-West)
    R_e = WGS84_A / W
    # Radius of curvature in meridian (North-South)
    R_n = WGS84_A * (1 - WGS84_E2) / (W ** 3)

    # Meters to degrees conversion factors
    # 1 degree longitude ≈ R_e * cos(lat) * π / 180 meters
    # 1 degree latitude ≈ R_n * π / 180 meters
    meters_per_deg_lon = R_e * math.cos(lat_rad) * math.pi / 180.0
    meters_per_deg_lat = R_n * math.pi / 180.0

    # Convert variance: var_deg^2 = var_m^2 / (meters_per_deg)^2
    var_lon_deg2 = var_e_m2 / (meters_per_deg_lon ** 2)
    var_lat_deg2 = var_n_m2 / (meters_per_deg_lat ** 2)
    cov_lon_lat_deg2 = cov_en_m2 / (meters_per_deg_lon * meters_per_deg_lat)

    # Return standard deviations (sqrt of variance) and covariance
    sigma_lon = math.sqrt(var_lon_deg2) if var_lon_deg2 > 0 else 0.0
    sigma_lat = math.sqrt(var_lat_deg2) if var_lat_deg2 > 0 else 0.0

    return sigma_lon, sigma_lat, cov_lon_lat_deg2


def meters_to_degrees_err(sigma_m: float, lat_deg: float, is_longitude: bool = True) -> float:
    """
    Convert error from meters to degrees at a given latitude.

    Args:
        sigma_m: Standard deviation in meters
        lat_deg: Latitude in degrees
        is_longitude: True for longitude, False for latitude

    Returns:
        Standard deviation in degrees
    """
    lat_rad = math.radians(lat_deg)
    sin_lat = math.sin(lat_rad)
    W = math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)

    if is_longitude:
        # Longitude: use prime vertical radius
        R_e = WGS84_A / W
        meters_per_deg = R_e * math.cos(lat_rad) * math.pi / 180.0
    else:
        # Latitude: use meridian radius
        R_n = WGS84_A * (1 - WGS84_E2) / (W ** 3)
        meters_per_deg = R_n * math.pi / 180.0

    return sigma_m / meters_per_deg


def compute_protection_level_lla(
    cov_lla: List[List[float]],
    lat_deg: float,
    k_factor: float = 3.72,
    output_unit: str = "degrees"
) -> float:
    """
    Compute Horizontal Protection Level (HPL) in LLA coordinates.

    Direct computation in degrees (consistent with map coordinates).
    Can optionally convert to meters for interpretation.

    Args:
        cov_lla: 3x3 covariance matrix in LLA (degrees^2 for lon/lat, meters^2 for h)
        lat_deg: Latitude in degrees
        k_factor: K factor for integrity risk (default 3.72 for 10^-4)
        output_unit: "degrees" or "meters" - unit for HPL output

    Returns:
        HPL in specified unit (degrees or meters)
    """
    # Extract 2x2 horizontal covariance (in degrees^2)
    var_lon = cov_lla[0][0]  # σ²_λ (longitude variance in degrees^2)
    var_lat = cov_lla[1][1]  # σ²_φ (latitude variance in degrees^2)
    cov_lon_lat = cov_lla[0][1]  # σ_λφ (lon-lat covariance in degrees^2)

    # HPL formula in degrees: HPL_deg = K * sqrt(σ²_max_deg)
    # σ²_max = (σ²_lon + σ²_lat)/2 + sqrt(((σ²_lon - σ²_lat)/2)² + σ²_lon_lat²)
    var_sum = var_lon + var_lat
    var_diff = var_lon - var_lat
    var_max_deg = 0.5 * var_sum + math.sqrt(0.25 * var_diff * var_diff + cov_lon_lat * cov_lon_lat)

    hpl_deg = k_factor * math.sqrt(var_max_deg)

    # Convert to meters if requested
    if output_unit == "meters":
        # Use WGS84 radii to convert degrees to meters at this latitude
        lat_rad = math.radians(lat_deg)
        sin_lat = math.sin(lat_rad)
        W = math.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
        R_n = WGS84_A * (1 - WGS84_E2) / (W ** 3)  # Meridian radius
        R_e = WGS84_A / W  # Prime vertical radius

        # Approximate: 1 degree lon ≈ R_e * cos(lat) * π / 180 meters
        #              1 degree lat ≈ R_n * π / 180 meters
        # Use average for HPL conversion
        meters_per_degree_avg = (R_n + R_e * math.cos(lat_rad)) / 2 * math.pi / 180.0
        hpl_meters = hpl_deg * meters_per_degree_avg
        return hpl_meters

    return hpl_deg


def parse_spp_solution(
    input_path: Path,
    output_path: Path,
    traj_id: int = 1,
    k_factor: float = 3.72,
    default_sigma: float = 1.0,
    hpl_unit: str = "degrees",
) -> None:
    """
    Parse spp_solution.txt and convert to cmm_trajectory.csv format (LLA in degrees).

    Process:
    1. Parse NMEA to get LLA coordinates (degrees)
    2. Parse GST ellipse parameters (degrees - non-standard format)
    3. Convert ellipse to covariance matrix (degrees^2)
    4. Compute HPL directly in degrees (consistent with map coordinates)
    5. Output: coordinates(deg), covariance(deg^2), HPL(deg or meters)

    Args:
        input_path: Path to spp_solution.txt
        output_path: Path to output cmm_trajectory.csv
        traj_id: Trajectory ID
        k_factor: K factor for protection level (default 3.72 for 10^-4 integrity risk)
        default_sigma: Default sigma when GST is missing (degrees)
        hpl_unit: HPL output unit: "degrees" or "meters" (default: "degrees")
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

    print(f"  Processing {len(points)} GPS points...")
    print(f"  Converting covariance from meters to degrees (WGS84)")

    timestamps: List[float] = []
    coords: List[Tuple[float, float]] = []
    covariances: List[List[float]] = []
    protection_levels: List[float] = []

    for timestamp, lat, lon, rec in sorted(points, key=lambda item: item[0]):
        # Use LLA coordinates directly (degrees)
        x, y = lon, lat

        # Extract GST ellipse parameters (in meters)
        smaj = rec["smaj"]
        smin = rec["smin"]
        orient = rec["orient"] if rec["orient"] is not None else 0.0

        # Convert to degrees
        if smaj is not None and smin is not None:
            # Use ellipse parameters (preferred)
            sigma_lon, sigma_lat, sigma_lon_lat = ellipse_to_covariance_degrees(
                smaj, smin, orient, lat
            )
            sigma_h = rec["alt_err"] if rec["alt_err"] is not None else 0.0  # Keep altitude error in meters
        else:
            # Fallback to lat_err/lon_err (in meters from GST)
            if rec["lat_err"] is not None and rec["lon_err"] is not None:
                sigma_lon = meters_to_degrees_err(rec["lon_err"], lat, is_longitude=True)
                sigma_lat = meters_to_degrees_err(rec["lat_err"], lat, is_longitude=False)
            else:
                # Use default value
                sigma_lon = meters_to_degrees_err(default_sigma, lat, is_longitude=True)
                sigma_lat = meters_to_degrees_err(default_sigma, lat, is_longitude=False)
            sigma_lon_lat = 0.0
            sigma_h = rec["alt_err"] if rec["alt_err"] is not None else 0.0

        # Build covariance matrix for HPL calculation (3x3 in degrees^2)
        P_lla = [
            [sigma_lon * sigma_lon,    sigma_lon_lat,           0.0],
            [sigma_lon_lat,            sigma_lat * sigma_lat,   0.0],
            [0.0,                      0.0,                     sigma_h * sigma_h]
        ]

        # Compute HPL (in degrees by default, or meters if specified)
        hpl = compute_protection_level_lla(P_lla, lat, k_factor, output_unit=hpl_unit)

        coords.append((x, y))
        timestamps.append(round(timestamp, 6))
        covariances.append(
            [
                round(sigma_lon, 12),      # σ_lon (degrees) - high precision
                round(sigma_lat, 12),      # σ_lat (degrees) - high precision
                round(sigma_h, 8),         # σ_h (meters)
                round(sigma_lon_lat, 14),  # σ_lon_lat (degrees^2)
                round(0.0, 14),            # σ_lon_h (degrees*m) - assumed 0
                round(0.0, 14),            # σ_lat_h (degrees*m) - assumed 0
            ]
        )
        protection_levels.append(round(hpl, 10 if hpl_unit == "degrees" else 8))

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
    print(f"  Covariance unit: degrees (lon/lat), meters (altitude)")
    print(f"  HPL unit: {hpl_unit}")


def read_cmm_trajectory_csv(file_path: str) -> List[Tuple[int, LineString, List[float], List[List[float]], List[float]]]:
    """Read cmm_trajectory.csv file and return list of trajectories."""
    trajectories = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            traj_id = int(row['id'])
            geom = wkt2linestring(row['geom'])
            timestamps = json.loads(row['timestamps'])
            covariances = json.loads(row['covariances'])
            protection_levels = json.loads(row['protection_levels'])
            trajectories.append((traj_id, geom, timestamps, covariances, protection_levels))
    return trajectories


def run_fmm_matching_python(
    fmm: FastMapMatch,
    fmm_config: FastMapMatchConfig,
    trajectory_file: str,
    output_file: str,
) -> bool:
    """Run FMM map matching using Python API."""
    print(f"\n{'='*60}")
    print("Running FMM Map Matching (Python API)")
    print('='*60)

    try:
        trajectories = read_cmm_trajectory_csv(trajectory_file)

        with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=';')
            writer.writerow([
                'id', 'cpath', 'pgeom', 'ep', 'tp', 'trustworthiness', 'timestamp'
            ])

            for traj_id, geom, timestamps, covariances, protection_levels in trajectories:
                traj = Trajectory(traj_id, geom, timestamps)
                result = fmm.match_traj(traj, fmm_config)

                ep_list = []
                tp_list = []
                tw_list = []

                pgeom_line = LineString()
                for i in range(len(result.opt_candidate_path)):
                    mc = result.opt_candidate_path[i]
                    ep_list.append(mc.ep)
                    tp_list.append(mc.tp)
                    tw_list.append(mc.trustworthiness)
                    x = mc.c.point.get_x(0)
                    y = mc.c.point.get_y(0)
                    pgeom_line.add_point(x, y)

                writer.writerow([
                    traj_id,
                    json.dumps(list(result.cpath)),
                    pgeom_line.export_wkt(),
                    json.dumps([f"{ep:.6e}" for ep in ep_list]),
                    json.dumps([f"{tp:.6e}" for tp in tp_list]),
                    json.dumps([f"{tw:.6f}" for tw in tw_list]),
                    json.dumps(timestamps)
                ])

                print(f"  Matched trajectory {traj_id}: {len(result.cpath)} edges matched")

        print(f"FMM matching completed successfully")
        return True
    except Exception as e:
        print(f"Error running FMM: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_cmm_matching_python(
    cmm: CovarianceMapMatch,
    cmm_config: CovarianceMapMatchConfig,
    trajectory_file: str,
    output_file: str,
) -> bool:
    """Run CMM map matching using Python API."""
    print(f"\n{'='*60}")
    print("Running CMM Map Matching (Python API)")
    print('='*60)

    try:
        trajectories = read_cmm_trajectory_csv(trajectory_file)

        with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=';')
            writer.writerow([
                'id', 'pgeom', 'eu_dist', 'sp_dist',
                'ep', 'tp', 'trustworthiness', 'timestamp'
            ])

            for traj_id, geom, timestamps, covariances, protection_levels in trajectories:
                traj = CMMTrajectory()
                traj.id = traj_id
                traj.geom = geom

                ts_vec = DoubleVector()
                for ts in timestamps:
                    ts_vec.append(ts)
                traj.timestamps = ts_vec

                cov_vec = CovarianceMatrixVector()
                for cov_data in covariances:
                    cov = CovarianceMatrix()
                    cov.sde = cov_data[0]
                    cov.sdn = cov_data[1]
                    cov.sdu = cov_data[2]
                    cov.sdne = cov_data[3]
                    cov.sdeu = cov_data[4]
                    cov.sdun = cov_data[5]
                    cov_vec.append(cov)
                traj.covariances = cov_vec

                pl_vec = DoubleVector()
                for pl in protection_levels:
                    pl_vec.append(pl)
                traj.protection_levels = pl_vec

                result = cmm.match_traj(traj, cmm_config)

                ep_list = []
                tp_list = []
                tw_list = []

                pgeom_line = LineString()
                for i in range(len(result.opt_candidate_path)):
                    mc = result.opt_candidate_path[i]
                    ep_list.append(mc.ep)
                    tp_list.append(mc.tp)
                    tw_list.append(mc.trustworthiness)
                    x = mc.c.point.get_x(0)
                    y = mc.c.point.get_y(0)
                    pgeom_line.add_point(x, y)

                sp_dist_str = ",".join(f"{d:.6f}" for d in result.sp_distances)
                eu_dist_str = ",".join(f"{d:.6f}" for d in result.eu_distances)

                writer.writerow([
                    traj_id,
                    pgeom_line.export_wkt(),
                    eu_dist_str,
                    sp_dist_str,
                    json.dumps([f"{ep:.6e}" for ep in ep_list]),
                    json.dumps([f"{tp:.6e}" for tp in tp_list]),
                    json.dumps([f"{tw:.6f}" for tw in tw_list]),
                    json.dumps(timestamps)
                ])

                total_eu_dist = sum(result.eu_distances) if result.eu_distances else 0.0
                total_sp_dist = sum(result.sp_distances) if result.sp_distances else 0.0
                print(f"  Matched trajectory {traj_id}: total eu_dist={total_eu_dist:.6f}deg, total sp_dist={total_sp_dist:.6f}deg")

        print(f"CMM matching completed successfully")
        return True
    except Exception as e:
        print(f"Error running CMM: {e}")
        import traceback
        traceback.print_exc()
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

    traj_id = Path(cmm_traj_file).parent.name

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
        description="Process Hainan dataset with rigorous LLA covariance transformation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific subdirectories
  python process_hainan_dataset_lla_rigid.py \\
      --subdirs 1.3 1.4 \\
      --dataset-dir dataset_hainan_06 \\
      --network input/map/hainan/edges.shp \\
      --ubodt input/map/hainan/hainan_ubodt.bin

Key Features:
  - Rigorous covariance transform: P_lla = J * P_utm * J^T
  - Jacobian computed via numerical differentiation (epsilon=1mm)
  - HPL computed using WGS84 ellipsoid radii
  - All calculations in double precision (float64)
        """
    )

    parser.add_argument(
        "--subdirs",
        nargs='+',
        default=["1.1", "1.2",  "1.3", "1.4","2.1", "2.2", "2.3"],
        help="Subdirectories to process"
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset_hainan_06",
        help="Root directory containing subdirectories"
    )

    parser.add_argument(
        "--network",
        type=str,
        default="input/map/hainan/edges.shp",
        help="Path to road network shapefile"
    )

    parser.add_argument(
        "--network-id",
        type=str,
        default="key",
        help="Network edge ID field"
    )

    parser.add_argument(
        "--network-source",
        type=str,
        default="u",
        help="Network source field"
    )

    parser.add_argument(
        "--network-target",
        type=str,
        default="v",
        help="Network target field"
    )

    parser.add_argument(
        "--ubodt",
        type=str,
        default="input/map/hainan/hainan_ubodt_indexed.bin",
        help="Path to UBODT file"
    )

    parser.add_argument(
        "--k-factor",
        type=float,
        default=3.72,
        help="K factor for protection level (default 3.72 for 10^-4 integrity risk)"
    )

    parser.add_argument(
        "--default-sigma",
        type=float,
        default=1.0,
        help="Default sigma when GST is missing (meters)"
    )

    parser.add_argument(
        "--hpl-unit",
        type=str,
        default="degrees",
        choices=["degrees", "meters"],
        help="HPL output unit: 'degrees' (default, consistent with map) or 'meters' (for interpretation)"
    )

    # CMM parameters (in degrees for LLA)
    parser.add_argument(
        "--cmm-k",
        type=int,
        default=16,
        help="CMM: number of candidates"
    )

    parser.add_argument(
        "--cmm-min-candidates",
        type=int,
        default=1,
        help="CMM: minimum candidates to keep"
    )

    parser.add_argument(
        "--cmm-protection-level-multiplier",
        type=float,
        default=10.0,
        help="CMM: protection level multiplier"
    )

    parser.add_argument(
        "--cmm-reverse-tolerance",
        type=float,
        default=0.0001,
        help="CMM: reverse tolerance (degrees, ~0.0001 deg ≈ 10m)"
    )

    parser.add_argument(
        "--cmm-window-length",
        type=int,
        default=100,
        help="CMM: window length for trustworthiness"
    )

    # FMM parameters (in degrees for LLA)
    parser.add_argument(
        "--fmm-k",
        type=int,
        default=16,
        help="FMM: number of candidates"
    )

    parser.add_argument(
        "--fmm-radius",
        type=float,
        default=0.001,
        help="FMM: search radius (degrees, ~0.001 deg ≈ 100m)"
    )

    parser.add_argument(
        "--fmm-gps-error",
        type=float,
        default=0.0001,
        help="FMM: GPS error (degrees, ~0.0001 deg ≈ 10m)"
    )

    parser.add_argument(
        "--no-matching",
        action="store_true",
        help="Skip map matching, only convert trajectories"
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Skip visualization generation"
    )

    parser.add_argument(
        "--mapbox-token",
        type=str,
        default=None,
        help="Mapbox access token"
    )

    args = parser.parse_args()

    base_dir = Path(args.dataset_dir)
    subdirs = args.subdirs

    print("="*70)
    print("Hainan Dataset Processing - LLA with Rigorous Covariance Transform")
    print("="*70)
    print(f"Dataset directory: {base_dir}")
    print(f"Subdirectories: {', '.join(subdirs)}")
    print(f"Network: {args.network}")
    print(f"UBODT: {args.ubodt}")
    print("="*70)

    print("\nCoordinate System: LLA (WGS84)")
    print(f"  Input: GST covariance in meters")
    print(f"  Output: Covariance in degrees (lon/lat), meters (altitude)")
    print(f"  HPL unit: {args.hpl_unit}")
    print(f"  FMM radius: {args.fmm_radius} deg (~{args.fmm_radius * 111320:.1f}m)")
    print(f"  FMM GPS error: {args.fmm_gps_error} deg (~{args.fmm_gps_error * 111320:.1f}m)")
    print(f"  CMM reverse tolerance: {args.cmm_reverse_tolerance} deg (~{args.cmm_reverse_tolerance * 111320:.1f}m)")

    if not Path(args.network).exists():
        print(f"\nError: Network file not found: {args.network}")
        return 1

    if not Path(args.ubodt).exists():
        print(f"\nError: UBODT file not found: {args.ubodt}")
        return 1

    print("\n" + "="*70)
    print("Loading Network and UBODT")
    print("="*70)
    print("WARNING: Ensure your Network and UBODT are in LLA (WGS84) coordinates!")

    print(f"Loading network from: {args.network}")
    network = Network(args.network, args.network_id, args.network_source, args.network_target)
    print(f"  Nodes: {network.get_node_count()}")
    print(f"  Edges: {network.get_edge_count()}")

    print("Creating network graph...")
    graph = NetworkGraph(network)
    print(f"  Vertices: {graph.get_num_vertices()}")

    print(f"Loading UBODT from: {args.ubodt}")
    ubodt = UBODT.read_ubodt_file(args.ubodt)
    print(f"  Rows: {ubodt.get_num_rows()}")
    print("Network and UBODT loaded successfully!\n")

    fmm = None
    cmm = None
    if not args.no_matching:
        print("Creating map matching instances...")
        fmm_config = FastMapMatchConfig(
            k_arg=args.fmm_k,
            r_arg=args.fmm_radius,
            gps_error=args.fmm_gps_error
        )
        fmm = FastMapMatch(network, graph, ubodt)
        print("  FMM instance created")

        cmm_config = CovarianceMapMatchConfig(
            k_arg=args.cmm_k,
            min_candidates_arg=args.cmm_min_candidates,
            protection_level_multiplier_arg=args.cmm_protection_level_multiplier,
            reverse_tolerance=args.cmm_reverse_tolerance,
            normalized_arg=True,
            use_mahalanobis_candidates_arg=True,
            window_length_arg=args.cmm_window_length,
            margin_used_trustworthiness_arg=False
        )
        cmm = CovarianceMapMatch(network, graph, ubodt)
        print("  CMM instance created\n")

    try:
        for subdir in subdirs:
            subdir_path = base_dir / subdir
            if not subdir_path.exists():
                print(f"\nWarning: Directory not found: {subdir_path}, skipping...")
                continue

            print(f"\n{'='*70}")
            print(f"Processing: {subdir}")
            print('='*70)

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

            print("\nStep 1/4: Converting to cmm_trajectory.csv (LLA with covariance in degrees)")
            print("-" * 50)
            try:
                parse_spp_solution(
                    input_path=spp_file,
                    output_path=cmm_traj_file,
                    traj_id=int(subdir.replace(".", "")),
                    k_factor=args.k_factor,
                    default_sigma=args.default_sigma,
                    hpl_unit=args.hpl_unit
                )
            except Exception as e:
                print(f"Error converting {spp_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if not args.no_matching:
                print("\nStep 2/4: Running FMM map matching")
                print("-" * 50)
                fmm_success = run_fmm_matching_python(
                    fmm=fmm,
                    fmm_config=fmm_config,
                    trajectory_file=str(cmm_traj_file),
                    output_file=str(fmm_results_file),
                )
                if not fmm_success:
                    print("FMM matching failed, continuing...")

                print("\nStep 3/4: Running CMM map matching")
                print("-" * 50)
                cmm_success = run_cmm_matching_python(
                    cmm=cmm,
                    cmm_config=cmm_config,
                    trajectory_file=str(cmm_traj_file),
                    output_file=str(cmm_results_file),
                )
                if not cmm_success:
                    print("CMM matching failed, continuing...")

            if not args.no_visualization and not args.no_matching:
                print("\nStep 4/4: Generating Mapbox visualization")
                print("-" * 50)

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

    finally:
        print("\n" + "="*70)
        print("Processing Complete!")
        print("="*70)

        print("\nSummary:")
        print("--------")
        print(f"Processed directories: {', '.join(subdirs)}")
        print(f"Output location: Each subdir/mr/ directory contains:")
        print("  - cmm_trajectory.csv   (LLA coordinates with rigorous covariance)")

        if not args.no_matching:
            print("  - cmm_results.csv      (CMM matching results)")
            print("  - fmm_results.csv      (FMM matching results)")

        if not args.no_visualization and not args.no_matching:
            print("  - mapbox_view.html      (interactive map visualization)")

        print("\nTechnical Details:")
        print("  - WGS84 ellipsoid parameters used for meters-to-degrees conversion")
        print("  - Input: GST covariance in meters")
        print("  - Output: Covariance in degrees (lon/lat), meters (altitude)")
        print(f"  - HPL unit: {args.hpl_unit}")
        print("  - All calculations in double precision (float64)")

    return 0


if __name__ == "__main__":
    main()
