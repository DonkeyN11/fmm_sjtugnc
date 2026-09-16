#!/usr/bin/env python3
"""Rescale the GST covariance columns of a CMM input CSV to the measured SPP error.

Why this exists. ``extract_spp_for_cmm.py`` writes the receiver's NMEA GST
1-sigma ellipse as the per-epoch covariance. That ellipse describes the
receiver's RTK solution class, not the SPP solution it is actually emitting, so
the covariance handed to the emission model is far too small -- measured against
RTK it is short by a factor of 3.0008 in sigma on the Haikou set. The model is
then over-sharp and the filtering posterior saturates.

Normally the fix would be to re-run the generator. It cannot be: the hainan_06
raw receiver logs (``实时定位结果/spp_solution.txt``) are no longer on disk, and
the CSV is the only surviving copy of the data they contained. So this script
applies the same calibration constant, imported from the generator so there is
exactly one definition of it, directly to an existing CSV.

What it does NOT touch. ``sdu``, ``sdeu`` and ``sdun`` are left alone, including
the fact that ``sdu`` is in metres while ``sde``/``sdn`` are in degrees -- that
unit mix is a separate known issue and is not this script's business. The
``protection_level`` column is left alone too: it comes from a different
pipeline (``merge_raim_pl.py``) and rescaling the search radius is a separate
change.

The transform is line-oriented, not a pandas round-trip, so that every column
this script does not intend to change keeps its original *text* and not merely
its original value. ``diff`` on the output then shows only the three covariance
columns, which is what makes the change reviewable.

    python experiments/scripts/apply_gst_calibration.py \
        --in  <pristine.csv> \
        --out data/real_vehicle/hainan_06/cmm_input_points.csv

@author: Chenzhang Ning
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_spp_for_cmm import GST_COVARIANCE_SCALE  # noqa: E402

# 0-based positions in the semicolon-separated row. Header is
# id;timestamp;x;y;sde;sdn;sdu;sdne;sdeu;sdun;protection_level
COL_SDE = 4
COL_SDN = 5
COL_SDNE = 7
EXPECTED_FIELDS = 11


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True,
                    help="pristine CMM input CSV (raw, uncalibrated GST columns)")
    ap.add_argument("--out", dest="dst", required=True,
                    help="where to write the calibrated CSV")
    args = ap.parse_args()

    if Path(args.src).resolve() == Path(args.dst).resolve():
        ap.error("--in and --out are the same file; the pristine input would be "
                 "destroyed and the script would no longer be re-runnable")

    # sigma scales by k, the cross-COVARIANCE by k**2. Getting this wrong is the
    # classic way to silently change the correlation of every epoch, so the two
    # factors are named rather than inlined.
    k = GST_COVARIANCE_SCALE
    k_var = k * k

    n = 0
    with open(args.src, newline="") as fin, open(args.dst, "w", newline="") as fout:
        header = fin.readline()
        fout.write(header)
        if header.strip().split(";") != [
            "id", "timestamp", "x", "y", "sde", "sdn", "sdu", "sdne",
            "sdeu", "sdun", "protection_level",
        ]:
            raise SystemExit(f"unexpected header in {args.src}: {header!r}")

        for lineno, line in enumerate(fin, start=2):
            parts = line.rstrip("\n").split(";")
            if len(parts) != EXPECTED_FIELDS:
                raise SystemExit(
                    f"{args.src}:{lineno}: expected {EXPECTED_FIELDS} fields, "
                    f"got {len(parts)}")
            parts[COL_SDE] = repr(float(parts[COL_SDE]) * k)
            parts[COL_SDN] = repr(float(parts[COL_SDN]) * k)
            parts[COL_SDNE] = repr(float(parts[COL_SDNE]) * k_var)
            fout.write(";".join(parts) + "\n")
            n += 1

    print(f"{args.src} -> {args.dst}")
    print(f"  {n} rows, sigma x {k}, cross-covariance x {k_var:.4f} "
          f"(k**2 = {k_var:.6f})")


if __name__ == "__main__":
    main()
