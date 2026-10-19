#!/usr/bin/env python3
"""WP1a step 0 (gate): does the patched binary reproduce the archived run?

The patched writer adds one output field. If the added field changed any existing
column the patch is not inert and nothing downstream can be trusted, so this runs
before every other measurement: it diffs the per-epoch values of the fresh run
against data/archive_20260929/scalar/{cmm,fmm}.csv.

THE KEY IS PER MATCHER, for the same reason real_final_tables.load_gt() documents:
CaMM's timestamp is unique per epoch but its seq is not (it restarts at every
sub-segment), and FMM's timestamp is not per-epoch at all (137 distinct values over
16,155 rows) while its seq is a clean 0-based index. Comparing FMM on timestamp
would silently check 137 epochs and report a pass, which is the failure mode this
gate exists to prevent.

Usage: python3 experiments/scripts/online_check_unmodified.py
"""
import csv
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
NEW = BASE / "data/archive_20261019/online"
OLD = BASE / "data/archive_20260929/scalar"
COLS = ["trustworthiness", "trustworthiness_smoothed", "cpath"]


def load(path, cols, keycol):
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            key = (r["id"].strip(), (r.get(keycol) or "").strip())
            d[key] = {c: (r.get(c) or "").strip() for c in cols}
    return d


def main():
    rc = 0
    for name, keycol in (("cmm", "timestamp"), ("fmm", "seq")):
        a = load(NEW / ("raw_%s.csv" % name), COLS, keycol)
        b = load(OLD / ("%s.csv" % name), COLS, keycol)
        ka, kb = set(a), set(b)
        print("%s (key=%s): new keys=%d old keys=%d only_new=%d only_old=%d"
              % (name, keycol, len(ka), len(kb), len(ka - kb), len(kb - ka)))
        if ka != kb:
            rc = 1
            print("   KEY MISMATCH -- patched run does not cover the archived epochs")
        for c in COLS:
            worst, wk, nbad = 0.0, None, 0
            for k in ka & kb:
                x, y = a[k][c], b[k][c]
                if x == y:
                    continue
                try:
                    dx = abs(float(x) - float(y))
                except ValueError:
                    nbad += 1
                    if wk is None:
                        wk = (k, x, y)
                    continue
                if dx > worst:
                    worst, wk = dx, (k, x, y)
            if nbad or worst > 0:
                rc = 1
                print("   %-24s MISMATCH: %d string-diff, max |delta| = %.3e at %s"
                      % (c, nbad, worst, wk))
            else:
                print("   %-24s identical on all %d epochs" % (c, len(ka & kb)))
    print("GATE:", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    sys.exit(main())

