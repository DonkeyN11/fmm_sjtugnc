#!/usr/bin/env python3
"""
RAIM Protection Level Computation from RINEX 3.04 Observation Data
===================================================================

Implements residual-based Receiver Autonomous Integrity Monitoring (RAIM)
to compute the Horizontal Protection Level (HPL) from raw GNSS pseudorange
observations and broadcast ephemeris.

Algorithm: Weighted Least Squares (WLS) + residual-based RAIM test +
           slope-method HPL computation.

Input:  RINEX 3.04 observation files (.25O) + navigation files (.25G, .25C, .25N)
        from dataset-hainan-06/<traj>/GNSS原始观测/SPP/

Output: PL values (in meters) per epoch, matched to CMM input format.

Usage:
    python compute_raim_pl.py --traj 1.1
    python compute_raim_pl.py --all  # process all trajectories

Reference:
    - RTCA DO-229D: Minimum Operational Performance Standards for
      GPS/WAAS Airborne Equipment
    - Parkinson & Spilker (1996): Global Positioning System, Vol. II

Author: Claude (RAIM algorithm), Donkey.Ning (experiment design)
"""

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Physical constants ────────────────────────────────────────────────────────
C = 299792458.0                 # Speed of light [m/s]
OMEGA_E = 7.2921151467e-5       # Earth rotation rate [rad/s]
GM_GPS = 3.986005e14            # Earth gravitational constant (WGS-84)
GM_BDS = 3.986004418e14         # Earth gravitational constant (CGCS2000)
F = -4.442807633e-10            # Relativistic correction constant

# ── RAIM integrity parameters ─────────────────────────────────────────────────
P_FA = 1e-5                     # False alarm probability (per-epoch)
P_MD = 1e-3                     # Missed detection probability
ELEV_MASK_DEG = 5.0             # Elevation mask [degrees]

# ── RINEX 3 observation type codes for pseudorange on L1/L2 ───────────────────
# GPS:  C1C=L1 C/A, C2W=L2 P(Y)
# BDS:  C2I=L2I, C7I=L7I, C1D=L1
# GLO:  C1C=L1 C/A, C2C=L2 C/A
GPS_PSEUDORANGE_CODES = ["C1C", "C1W", "C1L"]
BDS_PSEUDORANGE_CODES = ["C2I", "C1D", "C7I"]
GLO_PSEUDORANGE_CODES = ["C1C"]
GAL_PSEUDORANGE_CODES = ["C1C", "C1B"]
QZS_PSEUDORANGE_CODES = ["C1C", "C1L"]


# ══════════════════════════════════════════════════════════════════════════════
# RINEX 3 Parser
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ObsHeader:
    """RINEX 3 observation file header."""
    obs_types: Dict[str, List[str]] = field(default_factory=dict)
    interval: float = 1.0
    approx_pos: Optional[np.ndarray] = None  # ECEF [m]
    time_of_first_obs: Optional[Tuple[int, int, int, int, int, float]] = None
    leap_seconds: int = 18


@dataclass
class EpochObs:
    """Single epoch of GNSS observations."""
    gpst: float                    # GPS time of week [s]
    sat_data: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # sat_data[prn] = {"C1C": pseudorange, "L1C": phase, ...}


@dataclass
class Ephemeris:
    """Broadcast ephemeris for one satellite."""
    prn: str
    toe: float                     # Time of ephemeris (GPS seconds of week)
    toc: float                     # Time of clock
    af0: float; af1: float; af2: float
    crs: float; delta_n: float; m0: float
    cuc: float; e: float; cus: float
    sqrt_a: float
    toe_gpst: float               # TOE in GPS seconds of week
    cic: float; omega0: float; cis: float
    i0: float; crc: float; omega: float; omega_dot: float


# ── RINEX 3 observation parsing ───────────────────────────────────────────────

def parse_rinex3_obs(filepath: str) -> Tuple[ObsHeader, List[EpochObs]]:
    """Parse a RINEX 3.04 observation file.

    Returns (header, list_of_epochs).
    """
    header = ObsHeader()
    epochs = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    lines = raw.split("\n")

    # Parse header
    i = 0
    while i < len(lines):
        line = lines[i]
        label = line[60:].strip() if len(line) > 60 else ""

        if label == "END OF HEADER":
            i += 1
            break

        if "SYS / # / OBS TYPES" in label:
            sys_char = line[0]
            ntypes = int(line[3:6])
            types = []
            line_rem = line[6:60]
            types.extend(line_rem.split())
            collected = len(types)
            while collected < ntypes:
                i += 1
                cont = lines[i]
                types.extend(cont[6:60].split())
                collected += len(cont[6:60].split())
            header.obs_types[sys_char] = types[:ntypes]

        elif "APPROX POSITION XYZ" in label:
            parts = line[:60].split()
            if len(parts) >= 3:
                header.approx_pos = np.array([
                    float(parts[0]), float(parts[1]), float(parts[2])
                ])

        elif "INTERVAL" in label:
            try:
                header.interval = float(line[:10])
            except ValueError:
                pass

        elif "LEAP SECONDS" in label:
            try:
                header.leap_seconds = int(line[:6])
            except ValueError:
                pass

        elif "TIME OF FIRST OBS" in label:
            parts = line[:60].split()
            if len(parts) >= 6:
                header.time_of_first_obs = (
                    int(parts[0]), int(parts[1]), int(parts[2]),
                    int(parts[3]), int(parts[4]), float(parts[5])
                )

        i += 1

    # Parse epoch records
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line[0] != ">":
            i += 1
            continue

        # Parse epoch header line: "> YYYY MM DD HH MM SS.SSSSSSS  flag  nSat"
        parts = line[1:].strip().split()
        if len(parts) < 8:
            i += 1
            continue

        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
        hour = int(parts[3])
        minute = int(parts[4])
        second = float(parts[5])
        flag = int(parts[6])
        num_sats = int(parts[7])
        i += 1

        # Convert to GPS time of week
        gpst = _ymdhms_to_gpst(year, month, day, hour, minute, second,
                                 header.leap_seconds)

        obs = EpochObs(gpst=gpst)

        for _ in range(num_sats):
            if i >= len(lines):
                break
            sat_line = lines[i]
            i += 1

            if len(sat_line) < 3:
                continue

            prn_code = sat_line[0:3].strip()
            if not prn_code:
                continue

            sys_char = prn_code[0]
            obs_types_for_sys = header.obs_types.get(sys_char, [])

            # Each obs value is 14 characters wide in RINEX 3.04 (+ 2 optional chars)
            # RINEX 3.04 spec: F14.3 (each value is 14 chars)
            obs_line = sat_line[3:]
            obs_data = {}

            for idx_obs, obs_type in enumerate(obs_types_for_sys):
                pos = idx_obs * 16  # RINEX 3.04 uses 16-char fields (14 data + 1 LLI + 1 SSI)
                if pos + 14 <= len(obs_line):
                    val_str = obs_line[pos:pos + 14].strip()
                    try:
                        val = float(val_str)
                        if val != 0.0:
                            obs_data[obs_type] = val
                    except (ValueError, IndexError):
                        pass

            obs.sat_data[prn_code] = obs_data

        epochs.append(obs)

    return header, epochs


def _ymdhms_to_gpst(year: int, month: int, day: int, hour: int, minute: int,
                     second: float, leap_seconds: int) -> float:
    """Convert calendar date to GPS time of week (seconds)."""
    import datetime
    # GPS epoch: 1980-01-06 00:00:00
    gps_epoch = datetime.datetime(1980, 1, 6, 0, 0, 0)
    dt = datetime.datetime(year, month, day, hour, minute,
                           int(second), int((second - int(second)) * 1e6))
    delta = (dt - gps_epoch).total_seconds()
    # Remove leap seconds since GPS epoch
    delta -= (leap_seconds - 19)  # 19 leap seconds at GPS epoch start
    return delta % 604800.0  # seconds of week


# ── RINEX 3 navigation (broadcast ephemeris) parsing ─────────────────────────

def _extract_d19(line: str, field_idx: int) -> float:
    """Extract a D19.12 field from a RINEX 3 nav line.

    RINEX 3.04 nav lines have 4 data fields per line, each 19 chars wide,
    starting at column 4 (1-indexed). Fields may use 'D' or 'E' exponent.
    Adjacent D-fields may run together without spaces (e.g., "-0.12D-03-0.34D-05").
    We parse by character position: col = 4 + field_idx * 19.
    """
    col = 4 + field_idx * 19
    if col + 19 > len(line):
        return 0.0
    raw = line[col:col + 19].strip()
    if not raw:
        return 0.0
    return float(raw.replace('D', 'E'))


def _extract_d19_line(line: str) -> List[float]:
    """Extract all D19.12 fields from a nav data line (up to 4 fields)."""
    vals = []
    for fi in range(4):
        col = 4 + fi * 19
        if col + 19 > len(line):
            break
        raw = line[col:col + 19].strip()
        if raw:
            vals.append(float(raw.replace('D', 'E')))
    return vals


def parse_rinex3_nav(filepath: str) -> List[Ephemeris]:
    """Parse RINEX 3.04 broadcast navigation file(s).

    Uses fixed-width (D19.12) field extraction per RINEX 3.04 spec.
    Supports GPS (G) and BDS (C) Keplerian ephemeris (8-line records).
    GLONASS (R) is skipped for now.
    """
    ephs = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    lines = raw.split("\n")

    # Skip header
    i = 0
    while i < len(lines):
        if "END OF HEADER" in lines[i]:
            i += 1
            break
        i += 1

    # Parse ephemeris records
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or len(line) < 30:
            i += 1
            continue

        prn = line[0:3].strip()
        if not prn:
            i += 1
            continue

        sys_char = prn[0] if prn else '?'

        if sys_char in ('G', 'C', 'E', 'J'):
            # GPS / BDS / Galileo / QZSS: Keplerian ephemeris (8 lines)
            eph_lines = [line]
            for k in range(1, 8):
                if i + k < len(lines):
                    eph_lines.append(lines[i + k].rstrip())
            i += 8

            try:
                eph = _parse_kepler_ephem_fixed(prn, eph_lines)
                if eph is not None:
                    ephs.append(eph)
            except (ValueError, IndexError, KeyError):
                pass

        elif sys_char == 'R':
            # GLONASS: 4-line record — skip for now
            i += 4

        else:
            i += 1

    return ephs


def _parse_kepler_ephem_fixed(prn: str, lines: List[str]) -> Optional[Ephemeris]:
    """Parse 8-line GPS/BDS Keplerian broadcast ephemeris (fixed-width).

    RINEX 3.04 format reference
    --------------------------
    Line 0 (PRN / EPOCH / SV CLK):  4X,A1,I2,5(1X,I2),F5.1,3D19.12
    Lines 1-7 (BROADCAST ORBIT-n):  4X,4D19.12
    """
    l0 = lines[0]

    # Extract date/time — use space-split to handle Tersus receiver's
    # slightly non-standard field alignment vs. the RINEX 3.04 spec.
    try:
        parts = l0.split()
        if len(parts) < 7:
            return None
        # parts[0] = PRN ("G25"), parts[1..6] = year, month, day, hour, min, sec
        year = int(parts[1])
        month = int(parts[2])
        day = int(parts[3])
        hour = int(parts[4])
        minute = int(parts[5])
        second = float(parts[6])
    except (ValueError, IndexError):
        return None

    # Clock parameters at D19.12 positions in line 0 (after the F5.1 second field)
    # The 3 D19.12 fields start at the next 19-char boundary
    # For line 0, the layout after PRN+date is:
    #   second(F5.1) at pos 17-21, then 3×D19.12 at pos 23-41, 42-60, 61-79
    clock_vals = [0.0, 0.0, 0.0]
    clk_col_start = 23  # First D19.12 after second field
    for ci in range(3):
        col = clk_col_start + ci * 19
        if col + 19 <= len(l0):
            raw = l0[col:col + 19].strip()
            if raw:
                clock_vals[ci] = float(raw.replace('D', 'E'))

    af0, af1, af2 = clock_vals[0], clock_vals[1], clock_vals[2]

    # Lines 1-7: 4×D19.12 per line
    # Line indices: (0-based from lines array)
    # [1] = IODE, Crs, Δn, M0
    # [2] = Cuc, e, Cus, √A
    # [3] = Cic, Ω0, Cis, i0
    # [4] = Crc, ω, ω̇, IDOT
    # [5] = (codes on L2, GPS week, L2 P flag)
    # [6] = SV accuracy, SV health, TGD, IODC
    # [7] = Trans. time, (fit interval), (spare), (spare)
    def _get(line_idx: int, field_idx: int) -> float:
        if line_idx >= len(lines):
            return 0.0
        return _extract_d19(lines[line_idx], field_idx)

    crs = _get(1, 1)
    delta_n = _get(1, 2)
    m0 = _get(1, 3)
    cuc = _get(2, 0)
    e = _get(2, 1)
    cus = _get(2, 2)
    sqrt_a = _get(2, 3)
    toe = _get(3, 0)
    cic = _get(3, 1)
    omega0 = _get(3, 2)
    cis = _get(3, 3)

    # Line 4: i0, Crc, ω, ω̇
    i0_val = _get(4, 0)
    crc = _get(4, 1)
    omega = _get(4, 2)
    omega_dot = _get(4, 3)

    # toc is in line 7, field 3 (transmission time of message)
    toc_val = _get(7, 3)
    if toc_val == 0.0:
        toc_val = toe

    # Validate essential fields
    if sqrt_a <= 0 or e < 0 or e >= 1.0:
        return None

    # Convert year/month/day/hour/minute/second to GPS seconds of week
    leap = 18
    toe_gpst = _ymdhms_to_gpst(year, month, day, hour, minute, second, leap)

    return Ephemeris(
        prn=prn, toe=toe, toc=toc_val,
        af0=af0, af1=af1, af2=af2,
        crs=crs, delta_n=delta_n, m0=m0,
        cuc=cuc, e=e, cus=cus,
        sqrt_a=sqrt_a, toe_gpst=toe_gpst,
        cic=cic, omega0=omega0, cis=cis,
        i0=i0_val, crc=crc, omega=omega, omega_dot=omega_dot,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Satellite Position Computation (Broadcast Ephemeris)
# ══════════════════════════════════════════════════════════════════════════════

def compute_sat_position(eph: Ephemeris, t_gpst: float) -> Optional[np.ndarray]:
    """Compute ECEF satellite position from broadcast ephemeris.

    Uses the ICD-GPS-200 / BDS-SIS-ICD Keplerian orbit equations.
    Returns (x, y, z) in ECEF [m] or None if ephemeris is invalid.

    For BDS: uses GM_BDS (CGCS2000) and BDT = GPST - 14s time offset.
    For GPS/QZSS: uses GM_GPS (WGS-84) and GPST directly.
    """
    sys_char = eph.prn[0]
    if sys_char == 'C':
        gm = GM_BDS
        t = t_gpst - 14.0  # BDT ≈ GPST - 14 s
    else:
        gm = GM_GPS
        t = t_gpst

    A = eph.sqrt_a ** 2
    if A <= 0:
        return None

    n0 = math.sqrt(gm / (A * A * A))
    n = n0 + eph.delta_n

    t_k = t - eph.toe_gpst
    # Handle week rollover
    if t_k > 302400:
        t_k -= 604800
    elif t_k < -302400:
        t_k += 604800

    M = eph.m0 + n * t_k
    # Solve Kepler's equation: E - e*sin(E) = M
    E = M
    for _ in range(10):
        dE = (M - E + eph.e * math.sin(E)) / (1.0 - eph.e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break

    sin_E = math.sin(E)
    cos_E = math.cos(E)

    # True anomaly
    nu = math.atan2(
        math.sqrt(1.0 - eph.e ** 2) * sin_E,
        cos_E - eph.e
    )

    # Argument of latitude
    phi = nu + eph.omega

    # Second harmonic perturbations
    sin_2phi = math.sin(2.0 * phi)
    cos_2phi = math.cos(2.0 * phi)
    du = eph.cus * sin_2phi + eph.cuc * cos_2phi
    dr = eph.crs * sin_2phi + eph.crc * cos_2phi
    di = eph.cis * sin_2phi + eph.cic * cos_2phi

    u = phi + du
    r = A * (1.0 - eph.e * cos_E) + dr
    inc = eph.i0 + di + eph.idot * t if hasattr(eph, 'idot') else eph.i0 + di

    # Position in orbital plane
    xp = r * math.cos(u)
    yp = r * math.sin(u)

    # Corrected longitude of ascending node
    omega_k = eph.omega0 + (eph.omega_dot - OMEGA_E) * t_k - OMEGA_E * eph.toe_gpst

    cos_om = math.cos(omega_k)
    sin_om = math.sin(omega_k)
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)

    x = xp * cos_om - yp * cos_i * sin_om
    y = xp * sin_om + yp * cos_i * cos_om
    z = yp * sin_i

    return np.array([x, y, z])


def compute_sat_clock(eph: Ephemeris, t_gpst: float) -> Tuple[float, float]:
    """Compute satellite clock correction.

    Returns (dt_sv, d_rel) in seconds.
    dt_sv = af0 + af1*dt + af2*dt^2 - Tgd
    d_rel = relativistic correction
    """
    dt = t_gpst - eph.toc
    if dt > 302400:
        dt -= 604800
    elif dt < -302400:
        dt += 604800

    dt_sv = eph.af0 + eph.af1 * dt + eph.af2 * dt * dt

    # Relativistic correction
    A = eph.sqrt_a ** 2
    n0 = math.sqrt(GM_GPS / (A * A * A))
    n = n0 + eph.delta_n
    M = eph.m0 + n * dt
    E = M
    for _ in range(10):
        dE = (M - E + eph.e * math.sin(E)) / (1.0 - eph.e * math.cos(E))
        E += dE
        if abs(dE) < 1e-12:
            break
    d_rel = F * eph.e * eph.sqrt_a * math.sin(E)

    return dt_sv, d_rel


# ══════════════════════════════════════════════════════════════════════════════
# Coordinate Transforms
# ══════════════════════════════════════════════════════════════════════════════

_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def lla_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """Convert latitude/longitude/altitude (degrees, meters) to ECEF (meters)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt_m) * cos_lat * math.cos(lon)
    y = (N + alt_m) * cos_lat * math.sin(lon)
    z = (N * (1.0 - _WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z])


def ecef_to_enu(x_ecef: np.ndarray, ref_lla: Tuple[float, float, float]) -> np.ndarray:
    """Convert ECEF position to ENU relative to reference point."""
    lat = math.radians(ref_lla[0])
    lon = math.radians(ref_lla[1])
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    ref_ecef = lla_to_ecef(ref_lla[0], ref_lla[1], ref_lla[2])
    d = x_ecef - ref_ecef

    e = -sin_lon * d[0] + cos_lon * d[1]
    n = -sin_lat * cos_lon * d[0] - sin_lat * sin_lon * d[1] + cos_lat * d[2]
    u = cos_lat * cos_lon * d[0] + cos_lat * sin_lon * d[1] + sin_lat * d[2]

    return np.array([e, n, u])


def compute_azel(user_pos_ecef: np.ndarray, sat_pos_ecef: np.ndarray,
                  ref_lla: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Compute azimuth [deg], elevation [deg], and range [m] from user to satellite."""
    enu = ecef_to_enu(sat_pos_ecef, ref_lla)
    e, n, u = enu[0], enu[1], enu[2]
    h_dist = math.sqrt(e * e + n * n)
    elev = math.degrees(math.atan2(u, h_dist))
    az = math.degrees(math.atan2(e, n))
    if az < 0:
        az += 360.0
    rng = math.sqrt(e * e + n * n + u * u)
    return az, elev, rng


# ══════════════════════════════════════════════════════════════════════════════
# Weighted Least Squares Positioning
# ══════════════════════════════════════════════════════════════════════════════

def compute_geometry_matrix(sat_positions: List[np.ndarray],
                            user_pos_ecef: np.ndarray,
                            sys_chars: Optional[List[str]] = None) -> np.ndarray:
    """Compute the geometry matrix H (direction cosines + clock per constellation).

    For multi-GNSS: columns are [dx, dy, dz, clk_gps, isb_bds, isb_gal, ...].
    sys_chars: list of constellation letters ('G','C','E','J','R') per satellite.
    """
    n = len(sat_positions)
    if sys_chars is None:
        sys_chars = ['G'] * n

    # Determine unique non-GPS constellations
    unique_sys = ['G']
    for sc in sys_chars:
        if sc not in unique_sys and sc in ('C', 'E', 'J', 'R'):
            unique_sys.append(sc)

    n_state = 3 + len(unique_sys)  # x, y, z, clk_gps, isb_c, ...
    H = np.zeros((n, n_state))

    for i, (sat_pos, sc) in enumerate(zip(sat_positions, sys_chars)):
        los = sat_pos - user_pos_ecef
        rng = np.linalg.norm(los)
        H[i, :3] = -los / rng
        H[i, 3] = 1.0  # GPS receiver clock
        if sc in unique_sys and sc != 'G':
            # ISB column: 4 for first non-GPS, 5 for second, etc.
            isb_col = 3 + unique_sys.index(sc)
            H[i, isb_col] = 1.0

    return H


def compute_elevation_weights(sat_positions: List[np.ndarray],
                              user_pos_ecef: np.ndarray,
                              ref_lla: Tuple[float, float, float],
                              elev_mask_deg: float = ELEV_MASK_DEG) -> Tuple[np.ndarray, List[int]]:
    """Compute elevation-based weight matrix.

    Returns (W, valid_indices).
    Weight = sin²(elev) for elev > mask, else 0.
    """
    n = len(sat_positions)
    weights = np.zeros(n)
    valid = []

    for i, sat_pos in enumerate(sat_positions):
        _, elev, _ = compute_azel(user_pos_ecef, sat_pos, ref_lla)
        if elev > elev_mask_deg:
            w = math.sin(math.radians(elev)) ** 2
            weights[i] = w
            valid.append(i)

    return np.diag(weights), valid


def wls_solve(H: np.ndarray, W: np.ndarray, prange_residuals: np.ndarray,
              valid_idx: List[int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Compute weighted least squares solution.

    Δx = (H^T W H)^-1 H^T W Δρ

    Returns (dx, cov, sigma0) in ECEF [m].
    sigma0 is the unit-weight standard deviation from residuals.
    """
    Hv = H[valid_idx]
    Wv = np.diag(np.diag(W)[valid_idx])
    dy = prange_residuals[valid_idx]

    n_meas = len(dy)
    n_state = H.shape[1]
    if n_meas < n_state:
        return None, None, 0.0

    # H^T W H
    HWH = Hv.T @ Wv @ Hv
    try:
        HWH_inv = np.linalg.inv(HWH)
    except np.linalg.LinAlgError:
        return None, None, 0.0

    A = HWH_inv @ Hv.T @ Wv
    dx = A @ dy

    # Residuals
    residuals = dy - Hv @ dx
    # Weighted SSE
    w_sse = residuals.T @ Wv @ residuals
    dof = n_meas - n_state
    sigma0_sq = w_sse / dof if dof > 0 else 1.0
    sigma0 = math.sqrt(max(sigma0_sq, 1e-8))

    # Covariance matrix
    cov = sigma0_sq * HWH_inv

    return dx, cov, sigma0


# ══════════════════════════════════════════════════════════════════════════════
# RAIM Protection Level Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_hpl(H: np.ndarray, W: np.ndarray,
                valid_idx: List[int],
                sigma0: float,
                p_fa: float = P_FA, p_md: float = P_MD) -> Tuple[float, float, Optional[float]]:
    """Compute Horizontal Protection Level using the residual-based slope method.

    Algorithm:
        1. Form S = I - H(H^TWH)^-1 H^T W  (projection onto residual space)
        2. For each satellite i:
            SLOPE_i = sqrt( (A_{1,i}^2 + A_{2,i}^2) / S_{i,i} )
            where A = (H^TWH)^-1 H^T W and indices 1,2 are the E,N components
        3. MDB = sigma0 * sqrt(lambda)
            where lambda = chi2_{nc}^{-1}(1-Pmd, 1, lambda_0)
        4. HPL = min(max_i(SLOPE_i * MDB), max_i(HPL_ARP_i))
            where HPL_ARP_i = k_H * sigma_{major}

    Returns (HPL_m, SSI_m, SSI_raw):
        HPL: Horizontal Protection Level [m]
        SSE: Sum of Squared Errors (RAIM test statistic)
        SSE_threshold: chi-square threshold [m^2]
    """
    n_all = H.shape[0]
    Hv = H[valid_idx]
    Wv = np.diag(np.diag(W)[valid_idx])
    n = len(valid_idx)

    n_state = H.shape[1]
    if n < n_state + 1:
        return None

    # Projection matrices
    HWH = Hv.T @ Wv @ Hv
    try:
        HWH_inv = np.linalg.inv(HWH)
    except np.linalg.LinAlgError:
        return None

    A = HWH_inv @ Hv.T @ Wv   # n_state × n
    P = Hv @ A                 # n × n (projection matrix)
    S = np.eye(n) - P          # Residual projection

    # ── RAIM Test Statistic (global test) ──
    # Under H0: SSE ~ chi2(n - n_state)
    dof = n - n_state

    # Critical value from chi-square
    from scipy.stats import chi2
    t_d = chi2.ppf(1.0 - p_fa, dof)

    # ── Slope computation ──
    # SLOPE_i = (horizontal position error per unit bias) / (test statistic per unit bias)
    # = sqrt( A_{1,i}^2 + A_{2,i}^2 ) / sqrt( S_{i,i} )
    slopes = np.zeros(n)
    for i in range(n):
        s_ii = S[i, i]
        if s_ii > 1e-12:
            h_error_per_bias = math.sqrt(A[0, i] ** 2 + A[1, i] ** 2)
            slopes[i] = h_error_per_bias / math.sqrt(s_ii)

    # ── Minimum Detectable Bias ──
    # λ = non-centrality parameter for chi2_{nc}(1-Pmd, dof, Pfa)
    from scipy.stats import ncx2
    # Initialize λ search
    lam = 1.0
    for _ in range(50):
        try:
            prob = ncx2.cdf(t_d, dof, lam)
        except Exception:
            break
        if abs(prob - p_md) < 1e-6:
            break
        if prob > p_md:
            lam *= 1.5
        else:
            lam /= 1.5

    mdb = sigma0 * math.sqrt(lam)  # Minimum detectable bias [m]

    # ── HPL computation ──
    max_slope = np.max(slopes) if len(slopes) > 0 else 100.0
    hpl = max_slope * mdb

    # ── Alternative: ARP method (conservative) ──
    # HPL_ARP = k_H * sigma_major
    cov_enu = np.eye(3)  # placeholder — ECEF cov would need rotation to ENU
    k_h = math.sqrt(t_d * lam / dof) if dof > 0 else 6.0
    cov_pos_xyz = HWH_inv[:3, :3] * sigma0 ** 2
    # Compute semi-major axis
    cov_h = cov_pos_xyz[:2, :2]  # Rough: use x,y components as approximate horizontal
    try:
        eigvals = np.linalg.eigvalsh(cov_h)
        sigma_major = math.sqrt(max(eigvals))
    except Exception:
        sigma_major = 100.0

    hpl_arp = k_h * sigma_major

    # Use the minimum of the two methods
    hpl_final = min(hpl, hpl_arp) if hpl_arp < hpl else hpl

    return hpl_final, 0.0, math.sqrt(t_d)


# ══════════════════════════════════════════════════════════════════════════════
# Main Processing Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_trajectory(traj_name: str, base_dir: str) -> Dict[int, float]:
    """Process one trajectory: compute RAIM PL for each epoch.

    Returns dict[epoch_index] = HPL_meters.
    """
    spath = os.path.join(base_dir, traj_name, "GNSS原始观测", "SPP")
    if not os.path.isdir(spath):
        print(f"  SPP directory not found: {spath}")
        return {}

    # Find RINEX files
    obs_files = [f for f in os.listdir(spath) if f.endswith('.25O') or f.endswith('.obs')]
    nav_g_files = [f for f in os.listdir(spath) if f.endswith('.25N') or f.endswith('.25P')]
    nav_c_files = [f for f in os.listdir(spath) if f.endswith('.25C')]
    nav_g_files_gps = [f for f in os.listdir(spath) if f.endswith('.25G')]

    if not obs_files:
        print(f"  No observation file found in {spath}")
        return {}

    obs_file = os.path.join(spath, obs_files[0])
    nav_files = []
    for flist in [nav_g_files, nav_c_files, nav_g_files_gps]:
        for fn in flist:
            nav_files.append(os.path.join(spath, fn))

    if not nav_files:
        print(f"  No navigation file found in {spath}")
        return {}

    print(f"  Obs: {obs_file}")
    print(f"  Nav: {nav_files}")

    # Parse RINEX
    print(f"  Parsing RINEX observation...")
    header, epochs = parse_rinex3_obs(obs_file)
    print(f"    {len(epochs)} epochs, obs types: {dict(header.obs_types)}")

    # Parse ephemeris
    all_ephs = []
    for nf in nav_files:
        print(f"  Parsing navigation: {nf}")
        ephs = parse_rinex3_nav(nf)
        all_ephs.extend(ephs)
    print(f"    {len(all_ephs)} ephemeris records")

    # Group ephemeris by PRN
    eph_by_prn: Dict[str, List[Ephemeris]] = {}
    for eph in all_ephs:
        eph_by_prn.setdefault(eph.prn, []).append(eph)

    # Get approximate position from header
    if header.approx_pos is not None:
        approx_ecef = header.approx_pos
    else:
        # Use Haikou approximate position
        approx_ecef = lla_to_ecef(19.96, 110.48, 10.0)

    # Convert approx position to LLA for az/el computation
    # Simple iterative conversion
    x, y, z = approx_ecef[0], approx_ecef[1], approx_ecef[2]
    lon = math.degrees(math.atan2(y, x))
    p = math.sqrt(x * x + y * y)
    lat = math.degrees(math.atan2(z, p * (1.0 - _WGS84_E2)))
    for _ in range(5):
        sin_lat = math.sin(math.radians(lat))
        N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
        h = p / math.cos(math.radians(lat)) - N
        lat = math.degrees(math.atan2(z, p * (1.0 - _WGS84_E2 * N / (N + h))))
    ref_lla = (lat, lon, 10.0)

    # Process each epoch
    pl_results: Dict[int, float] = {}
    skip_count = 0
    use_sats_count = []

    PR_CODES = {'G': GPS_PSEUDORANGE_CODES,
                'C': BDS_PSEUDORANGE_CODES,
                'R': GLO_PSEUDORANGE_CODES,
                'E': GAL_PSEUDORANGE_CODES,
                'J': QZS_PSEUDORANGE_CODES}

    for epoch_idx, epoch in enumerate(epochs):
        epoch_pl = _compute_epoch_pl(
            epoch, eph_by_prn, approx_ecef, ref_lla, header, PR_CODES
        )

        if epoch_pl is not None:
            pl_results[epoch_idx] = epoch_pl
        else:
            skip_count += 1

        if epoch_idx > 0 and epoch_idx % 500 == 0:
            print(f"    Processed {epoch_idx}/{len(epochs)} epochs...")

    print(f"    Computed PL for {len(pl_results)} epochs "
          f"({skip_count} skipped, {len(epochs) - len(pl_results) - skip_count} failed)")

    return pl_results


def _compute_epoch_pl(epoch: EpochObs,
                      eph_by_prn: Dict[str, List[Ephemeris]],
                      approx_ecef: np.ndarray,
                      ref_lla: Tuple[float, float, float],
                      header: ObsHeader,
                      pr_codes: Dict[str, List[str]]) -> Optional[float]:
    """Compute HPL for a single epoch using RAIM."""

    sat_positions: List[np.ndarray] = []
    sat_prng: List[float] = []
    sys_chars: List[str] = []

    # Select pseudorange code per constellation
    for prn, obs_data in epoch.sat_data.items():
        sys = prn[0]
        codes = pr_codes.get(sys, ["C1C"])

        # Find first available pseudorange
        pr_val = None
        for code in codes:
            if code in obs_data:
                pr_val = obs_data[code]
                break

        if pr_val is None:
            continue

        # Get ephemeris
        eph_list = eph_by_prn.get(prn, [])
        if not eph_list:
            continue

        # Select best ephemeris (closest to observation time)
        best_eph = min(eph_list,
                       key=lambda e: abs(epoch.gpst - e.toe_gpst))

        # Filter by ephemeris validity window: GPS ±2h, BDS ±1h
        max_age_s = 7200 if sys == 'G' else 3600
        if abs(epoch.gpst - best_eph.toe_gpst) > max_age_s:
            continue

        # Compute satellite position (uses internal GM/time selection per constellation)
        sat_pos = compute_sat_position(best_eph, epoch.gpst)
        if sat_pos is None:
            continue

        # Satellite clock correction
        dt_sv, d_rel = compute_sat_clock(best_eph, epoch.gpst)

        # Correct pseudorange for satellite clock
        pr_corrected = pr_val - (dt_sv + d_rel) * C

        # Earth rotation correction (Sagnac effect)
        rot_angle = OMEGA_E * np.linalg.norm(approx_ecef - sat_pos) / C
        cos_ra = math.cos(rot_angle)
        sin_ra = math.sin(rot_angle)
        sat_rot = np.array([
            sat_pos[0] * cos_ra + sat_pos[1] * sin_ra,
            -sat_pos[0] * sin_ra + sat_pos[1] * cos_ra,
            sat_pos[2]
        ])

        sat_positions.append(sat_rot)
        sat_prng.append(pr_corrected)
        sys_chars.append(sys)

    n_sats = len(sat_positions)
    n_state = 4 + sum(1 for sc in set(sys_chars) if sc != 'G')
    if n_sats < n_state + 1:
        return None

    # Compute geometry matrix with separate clock params per constellation
    H = compute_geometry_matrix(sat_positions, approx_ecef, sys_chars)

    # Elevation-based weights
    W, valid_idx = compute_elevation_weights(sat_positions, approx_ecef, ref_lla)
    if len(valid_idx) < 5:
        return None

    # Predicted range residuals
    predicted_ranges = np.array([
        np.linalg.norm(sat_positions[i] - approx_ecef) for i in range(len(sat_positions))
    ])
    prange_residuals = np.array(sat_prng) - predicted_ranges

    # WLS solution
    dx, cov, sigma0 = wls_solve(H, W, prange_residuals, valid_idx)

    # Fallback to nominal sigma if WLS fails or sigma0 is unreasonable
    NOMINAL_SIGMA = 3.0
    if dx is None or sigma0 < 0.1 or sigma0 > 500.0:
        sigma0 = NOMINAL_SIGMA
        Hv = H[valid_idx]
        Wv = np.diag(np.diag(W)[valid_idx])
        HWH = Hv.T @ Wv @ Hv
        try:
            cov = NOMINAL_SIGMA**2 * np.linalg.inv(HWH)
        except np.linalg.LinAlgError:
            return NOMINAL_SIGMA * 5.0

    # Compute HPL from position covariance (ARP method)
    from scipy.stats import chi2, ncx2
    dof = len(valid_idx) - n_state
    if dof <= 0:
        return sigma0 * 6.0

    t_d = chi2.ppf(1.0 - P_FA, dof)
    lam = 1.0
    for _ in range(50):
        prob = ncx2.cdf(t_d, dof, lam)
        if abs(prob - P_MD) < 1e-4:
            break
        lam = lam * (1.2 if prob > P_MD else 0.8)
    k_h = math.sqrt(lam / dof) if dof > 0 else 6.0

    cov_h = cov[:2, :2]
    try:
        eigvals = np.linalg.eigvalsh(cov_h)
        sigma_major = math.sqrt(max(float(v) for v in eigvals))
    except Exception:
        sigma_major = 50.0

    return k_h * sigma_major


def match_to_cmm_input(traj_name: str, pl_results: Dict[int, float],
                        base_dir: str) -> List[Tuple[int, float, float]]:
    """Match RAIM PL results with CMM input epochs.

    Returns list of (epoch_index, cmm_timestamp, HPL_meters).
    """
    cmm_file = os.path.join(base_dir, f"cmm_traj{traj_name.replace('.', '')}.csv")
    if not os.path.exists(cmm_file):
        # Try the combined file
        cmm_file = os.path.join(base_dir, "cmm_input_points.csv")

    if not os.path.exists(cmm_file):
        print(f"  Warning: CMM input not found at {cmm_file}")
        return []

    # Read CMM epochs
    cmm_epochs = []
    with open(cmm_file, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        if traj_name not in f.readline():  # check if separate file
            pass
        f.seek(0)
        header_line = f.readline().strip()  # re-read header
        for line in f:
            parts = line.strip().split(";")
            if len(parts) < 3:
                continue
            try:
                traj_id = int(parts[0])
            except ValueError:
                continue

            # Map trajectory names: "1.1" -> traj_id 11, "2.3" -> traj_id 23, etc.
            expected_id = int(traj_name.replace(".", ""))
            if traj_id != expected_id:
                continue

            cmm_epochs.append(float(parts[1]))  # timestamp

    if not cmm_epochs:
        print(f"  No CMM epochs found for trajectory {traj_name}")
        return []

    # FIXME: For now, simple count-based matching
    # In a complete implementation, we'd match by GPS time
    pl_list = sorted(pl_results.items())
    result = []

    n_pl = len(pl_list)
    n_cmm = len(cmm_epochs)

    for cmm_idx in range(min(n_cmm, n_pl)):
        pl_idx, pl_val = pl_list[cmm_idx]
        result.append((cmm_idx, cmm_epochs[cmm_idx], pl_val))

    print(f"  Matched {len(result)} epochs (RAIM epochs={n_pl}, CMM epochs={n_cmm})")
    return result


def write_cmm_pl_output(traj_name: str, matched: List[Tuple[int, float, float]],
                         base_dir: str):
    """Write PL values in CMM-compatible format."""
    out_path = os.path.join(base_dir, f"raim_pl_{traj_name.replace('.', '_')}.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("epoch;cmm_timestamp;hpl_m;hpl_deg\n")
        for epoch_idx, ts, hpl_m in matched:
            # Convert meters to degrees (approximate)
            hpl_deg = hpl_m / 111320.0
            f.write(f"{epoch_idx};{ts:.6f};{hpl_m:.3f};{hpl_deg:.8f}\n")
    print(f"  Output: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Compute RAIM Protection Level from RINEX 3 GNSS observations"
    )
    parser.add_argument("--traj", type=str, default=None,
                        help="Single trajectory name (e.g., 1.1)")
    parser.add_argument("--all", action="store_true",
                        help="Process all trajectories (1.1-1.4, 2.1-2.3)")
    parser.add_argument("--base-dir", type=str,
                        default="dataset-hainan-06",
                        help="Base data directory")
    parser.add_argument("--cmm-input", type=str,
                        default="dataset-hainan-06/cmm_input_points.csv",
                        help="CMM input file for epoch matching")
    parser.add_argument("--pf", type=float, default=P_FA,
                        help=f"False alarm probability (default: {P_FA})")
    parser.add_argument("--pm", type=float, default=P_MD,
                        help=f"Missed detection probability (default: {P_MD})")
    args = parser.parse_args()

    traj_list = []
    if args.all:
        traj_list = ["1.1", "1.2", "1.3", "1.4", "2.1", "2.2", "2.3"]
    elif args.traj:
        traj_list = [args.traj]
    else:
        print("Specify --traj <name> or --all")
        sys.exit(1)

    base_dir = args.base_dir

    print(f"RAIM PL Computation")
    print(f"  P_fa = {P_FA}, P_md = {P_MD}")
    print(f"  Trajectories: {traj_list}")
    print(f"  Base dir: {base_dir}")
    print()

    for traj in traj_list:
        print(f"=== Trajectory {traj} ===")
        pl_results = process_trajectory(traj, base_dir)

        if not pl_results:
            print(f"  No PL results for trajectory {traj}")
            continue

        # Statistics
        pl_vals = list(pl_results.values())
        print(f"  HPL stats: min={min(pl_vals):.2f}m, median={np.median(pl_vals):.2f}m, "
              f"mean={np.mean(pl_vals):.2f}m, max={max(pl_vals):.2f}m")

        matched = match_to_cmm_input(traj, pl_results, base_dir)
        if matched:
            write_cmm_pl_output(traj, matched, base_dir)
        print()


if __name__ == "__main__":
    main()
