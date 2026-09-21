#!/usr/bin/env python3
"""Rewrite the `protection_level` column of a CMM input table in place.

Why this is exact and not an approximation
------------------------------------------
`generate_data_cmm.py` holds the protection level in UTM metres and writes it in
degrees, because the CMM app consumes the column verbatim as a search radius
(cmm_algorithm.cpp:787) and the network is EPSG:4326.

The scalar PL is a *bound*, so the degree-space radius has to cover the metric
radius in every direction. A degree step in longitude covers |d(x,y)/dlon|
metres and one in latitude |d(x,y)/dlat|; to cover all directions the division
must use the SMALLER of the two, i.e. the largest degrees-per-metre. The
previous revision divided by the larger one, leaving the east-west radius 5.8%
short. Both revisions use the same two column norms of the forward Jacobian and
the same `_get_jacobian_inv`, so the correction is a per-row factor

    new_pl = old_pl * max(scale_lon, scale_lat) / min(scale_lon, scale_lat)

Nothing else in the file depends on the change: the covariance columns
(sde/sdn/sdne/sdu/sdeu/sdun) are untouched, and they are what the emission model
consumes. So repairing the column is equivalent to regenerating the dataset --
and is verifiable against a freshly generated one, which `--verify` does.

Usage
-----
    python experiments/scripts/repair_protection_level_degrees.py FILE [FILE ...]
    python experiments/scripts/repair_protection_level_degrees.py --verify REFERENCE FILE
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pyproj

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from generate_data_cmm import _get_jacobian_inv  # noqa: E402  the same one the generator uses

UTM_EPSG = 32649
PL_COL = "protection_level"

_FWD = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{UTM_EPSG}", always_xy=True)


def scales(lon: float, lat: float) -> tuple[float, float]:
    """(metres per degree longitude, metres per degree latitude) at (lon, lat)."""
    j_inv = _get_jacobian_inv(lon, lat, _FWD)
    j_fwd = np.linalg.inv(j_inv)
    scale_lon = math.hypot(float(j_fwd[0, 0]), float(j_fwd[1, 0]))
    scale_lat = math.hypot(float(j_fwd[0, 1]), float(j_fwd[1, 1]))
    return scale_lon, scale_lat


def repair(path: Path, dry_run: bool = False) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if PL_COL not in fieldnames:
        raise SystemExit(f"{path}: no '{PL_COL}' column (has {fieldnames})")

    ratios = []
    for r in rows:
        lon, lat = float(r["x"]), float(r["y"])
        s_lon, s_lat = scales(lon, lat)
        lo, hi = min(s_lon, s_lat), max(s_lon, s_lat)
        old = float(r[PL_COL])
        # old was written as pl_m / hi; recover pl_m, then divide by lo
        new = old * hi / lo
        r[PL_COL] = f"{new:.12f}"
        ratios.append(hi / lo)

    stats = {
        "path": str(path),
        "rows": len(rows),
        "ratio_min": min(ratios) if ratios else float("nan"),
        "ratio_max": max(ratios) if ratios else float("nan"),
        "ratio_mean": float(np.mean(ratios)) if ratios else float("nan"),
    }
    if not dry_run:
        shutil.copy2(path, path.with_suffix(path.suffix + ".prepl"))
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
            w.writeheader()
            w.writerows(rows)
        tmp.replace(path)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the rescale factors without writing.")
    ap.add_argument("--verify", type=Path, default=None,
                    help="With one FILE: assert FILE's repaired column equals the "
                         "column in this freshly generated reference.")
    args = ap.parse_args()

    for p in args.files:
        s = repair(p, dry_run=args.dry_run)
        print(f"{s['path']}: {s['rows']} rows  factor {s['ratio_min']:.12f}"
              f"..{s['ratio_max']:.12f} (mean {s['ratio_mean']:.12f})")

    if args.verify is not None:
        if len(args.files) != 1:
            raise SystemExit("--verify takes exactly one FILE")
        a = {r["id"] + "|" + r["seq"]: r[PL_COL]
             for r in csv.DictReader(args.files[0].open(newline=""), delimiter=";")}
        b = {r["id"] + "|" + r["seq"]: r[PL_COL]
             for r in csv.DictReader(args.verify.open(newline=""), delimiter=";")}
        if set(a) != set(b):
            print(f"verify: KEY SETS DIFFER ({len(a)} vs {len(b)})")
            return 1
        worst, worst_k = 0.0, None
        for k in a:
            d = abs(float(a[k]) - float(b[k]))
            if d > worst:
                worst, worst_k = d, k
        print(f"verify vs {args.verify}: max |diff| = {worst:.3e} over {len(a)} rows"
              + ("" if worst == 0.0 else f"  (at {worst_k})"))
        return 0 if worst < 1e-9 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
