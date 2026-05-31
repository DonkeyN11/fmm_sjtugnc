import argparse
import csv
import json
import math
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover - optional dependency
    CRS = None
    Transformer = None

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
    with path.open("rb") as fh:
        data = fh.read()
    text = data.decode("ascii", errors="ignore")
    for match in NMEA_RE.finditer(text):
        line = match.group(0).strip()
        if "*" in line:
            star = line.find("*")
            line = line[: min(star + 3, len(line))]
        yield line


def parse_time_fields(time_str: str) -> Optional[Tuple[int, int, int, int]]:
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
    theta = math.radians(orient_deg)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)
    vmaj = smaj * smaj
    vmin = smin * smin
    var_e = vmaj * sin_t * sin_t + vmin * cos_t * cos_t
    var_n = vmaj * cos_t * cos_t + vmin * sin_t * sin_t
    cov_en = (vmaj - vmin) * sin_t * cos_t
    return math.sqrt(var_e), math.sqrt(var_n), cov_en


def meters_to_degrees(
    sde: float,
    sdn: float,
    sdne: float,
    lat_deg: float,
) -> Tuple[float, float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat_deg))
    if meters_per_deg_lon <= 0:
        return sde, sdn, sdne
    sde_deg = sde / meters_per_deg_lon
    sdn_deg = sdn / meters_per_deg_lat
    sdne_deg = sdne / (meters_per_deg_lon * meters_per_deg_lat)
    return sde_deg, sdn_deg, sdne_deg


def determine_utm_epsg(lon_deg: float, lat_deg: float) -> Optional[int]:
    if not (math.isfinite(lon_deg) and math.isfinite(lat_deg)):
        return None
    if lat_deg <= -80.0 or lat_deg >= 84.0:
        return None
    zone = int(math.floor((lon_deg + 180.0) / 6.0)) + 1
    zone = max(1, min(zone, 60))
    base = 32600 if lat_deg >= 0.0 else 32700
    return base + zone


def parse_nmea_file(
    input_path: Path,
    output_path: Path,
    traj_id: int = 1,
    project_utm: bool = True,
    utm_epsg: Optional[int] = None,
    protection_level_scale: float = 2.0,
    default_sigma: float = 1.0,
) -> None:
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
        if transformer:
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat

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

        if not transformer:
            sde, sdn, sdne = meters_to_degrees(sde, sdn, sdne, lat)

        sdu = rec["alt_err"] if rec["alt_err"] is not None else 0.0
        sdeu = 0.0
        sdun = 0.0

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

    wkt = "LINESTRING (" + ", ".join(f"{x:.8f} {y:.8f}" for x, y in coords) + ")"

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

    print(f"Wrote {len(coords)} points to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NMEA/GST logs to cmm_trajectory.csv format.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset_hainan_06/1.1/实时定位结果/spp_solution.txt"),
        help="Input NMEA file (text or binary).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_hainan_06/1.1/mr/cmm_trajectory.csv"),
        help="Output cmm_trajectory.csv path.",
    )
    parser.add_argument("--traj-id", type=int, default=1, help="Trajectory id.")
    parser.add_argument(
        "--no-project",
        action="store_true",
        help="Keep lon/lat coordinates (covariance converted to degrees).",
    )
    parser.add_argument(
        "--utm-epsg",
        type=int,
        default=None,
        help="Force UTM EPSG code (e.g. 32649).",
    )
    parser.add_argument(
        "--protection-level-scale",
        type=float,
        default=2.0,
        help="Scale factor for protection level derived from GST.",
    )
    parser.add_argument(
        "--default-sigma",
        type=float,
        default=1.0,
        help="Fallback sigma (meters) when GST is missing.",
    )

    args = parser.parse_args()
    parse_nmea_file(
        input_path=args.input,
        output_path=args.output,
        traj_id=args.traj_id,
        project_utm=not args.no_project,
        utm_epsg=args.utm_epsg,
        protection_level_scale=args.protection_level_scale,
        default_sigma=args.default_sigma,
    )


if __name__ == "__main__":
    main()
