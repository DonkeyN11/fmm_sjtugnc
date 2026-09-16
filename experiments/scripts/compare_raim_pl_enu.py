#!/usr/bin/env python3
"""Compare RAIM protection levels computed before vs. after the ECEF->ENU fix.

The legacy `compute_raim_pl.py` built the geometry matrix H directly in ECEF
x/y/z and then took `cov[:2, :2]` as the "horizontal" block. That block is the
x-y plane of the ECEF frame, which is tilted by the geodetic latitude relative
to the local horizontal plane, so `sqrt(max_eigval(cov[:2,:2]))` over-estimates
the true horizontal semi-major axis. The fixed version resolves the
line-of-sight unit vectors in ENU, so `cov[:2, :2]` is the true East-North
block.

Usage:
    python experiments/scripts/compare_raim_pl_enu.py \
        --old data/real_vehicle/hainan_06/backup_raim_pl_ecef_20260915 \
        --new data/real_vehicle/hainan_06
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

TRAJS = ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3"]


def load_pl(base_dir, traj):
    path = os.path.join(base_dir, f"raim_pl_{traj}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=";")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="directory with legacy (ECEF) PL files")
    ap.add_argument("--new", required=True, help="directory with fixed (ENU) PL files")
    args = ap.parse_args()

    rows = []
    old_all, new_all = [], []
    for traj in TRAJS:
        do, dn = load_pl(args.old, traj), load_pl(args.new, traj)
        if do is None or dn is None:
            print(f"  [skip] {traj}: missing file", file=sys.stderr)
            continue
        n = min(len(do), len(dn))
        o, w = do["hpl_m"].to_numpy()[:n], dn["hpl_m"].to_numpy()[:n]
        old_all.append(o)
        new_all.append(w)
        rows.append(dict(
            traj=traj, n=n,
            old_med=np.median(o), new_med=np.median(w),
            old_mean=o.mean(), new_mean=w.mean(),
            old_min=o.min(), new_min=w.min(),
            old_max=o.max(), new_max=w.max(),
            shrink=(np.median(w) / np.median(o) - 1.0) * 100.0,
        ))

    if not rows:
        print("no overlapping files found", file=sys.stderr)
        return 1

    t = pd.DataFrame(rows)
    o_all = np.concatenate(old_all)
    w_all = np.concatenate(new_all)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 50)

    print("=" * 100)
    print("RAIM HPL per trajectory: legacy (ECEF x/y block)  vs  fixed (ENU East-North block)")
    print("=" * 100)
    print(t[["traj", "n", "old_med", "new_med", "shrink",
             "old_mean", "new_mean", "old_min", "new_min",
             "old_max", "new_max"]].to_string(
        index=False,
        float_format=lambda x: f"{x:,.1f}"))
    print()
    print("=" * 100)
    print(f"ALL EPOCHS (n = {len(o_all)})")
    print("=" * 100)
    for name, arr in (("legacy", o_all), ("fixed", w_all)):
        print(f"  {name:6s}  min={arr.min():9.3f}  "
              f"p25={np.percentile(arr, 25):9.3f}  "
              f"median={np.median(arr):9.3f}  "
              f"mean={arr.mean():9.3f}  "
              f"p75={np.percentile(arr, 75):9.3f}  "
              f"p95={np.percentile(arr, 95):9.3f}  "
              f"max={arr.max():9.3f}   [m]")
    print(f"  ratio (fixed/legacy): median={np.median(w_all / o_all):.4f}  "
          f"mean={(w_all / o_all).mean():.4f}  "
          f"p05={np.percentile(w_all / o_all, 5):.4f}  "
          f"p95={np.percentile(w_all / o_all, 95):.4f}")
    print(f"  median HPL reduction: {(1 - np.median(w_all) / np.median(o_all)) * 100:.2f} %")
    print(f"  epochs with PL unchanged (|ratio-1| < 1e-6): "
          f"{int((np.abs(w_all / o_all - 1) < 1e-6).sum())} / {len(o_all)}")

    # The candidate-search radius IS the PL (r_i = HPL_i, no multiplier); the
    # only enlargement is the bounded doubling fallback that fires on a
    # zero-candidate epoch. Report the median HPL itself as the entry radius.
    print()
    print("  Candidate search radius = HPL (no multiplier; r_i = HPL_i):")
    print(f"    legacy median entry radius = {np.median(o_all):8.2f} m")
    print(f"    fixed  median entry radius = {np.median(w_all):8.2f} m")

    return 0


if __name__ == "__main__":
    sys.exit(main())
