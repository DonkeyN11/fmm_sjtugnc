#!/usr/bin/env python3
"""Merge RAIM-computed HPL values into the CMM aggregated input table.

Replaces the 'protection_level' column with RAIM HPL values (in degrees),
preserving all other columns. Backs up the original file to .bak.

The trajectories and the raim_pl_*.csv files live under --base-dir, e.g.
data/real_vehicle/hainan_06. The historical default (data/real_vehicle) no
longer contains them, so pass --base-dir explicitly for that dataset.

Usage:
    python experiments/scripts/merge_raim_pl.py \
        --base-dir data/real_vehicle/hainan_06
    # write to a new file, leaving the original untouched:
    python experiments/scripts/merge_raim_pl.py \
        --base-dir data/real_vehicle/hainan_06 \
        --out data/real_vehicle/hainan_06/cmm_input_points_enu.csv
"""

import argparse
import os
import shutil
import sys

import numpy as np

DEFAULT_BASE_DIR = "data/real_vehicle"


TS_DECIMALS = 6  # matches write_cmm_pl_output's "{ts:.6f}" and the CSV's own precision


def _ts_key(value) -> float:
    """Normalise a timestamp so the two files' text forms compare equal.

    The CMM table writes '1750306259.0'; raim_pl_<traj>.csv writes
    '1750306259.000000'. Both parse to the same float, but rounding pins that
    down rather than relying on it.
    """
    return round(float(value), TS_DECIMALS)


def load_raim_pl(traj_name: str, raim_pattern: str) -> dict:
    """Load RAIM PL values for a trajectory, keyed by CMM timestamp.

    File format is 'epoch;cmm_timestamp;hpl_m;hpl_deg'.

    The key is cmm_timestamp, NOT the epoch column. The epoch column is a RINEX
    epoch index: the RINEX file runs at 10 Hz and the CMM table at 1 Hz over the
    same span, so pairing the two files by index stretches a tenth of the RINEX
    file across the whole trajectory. cmm_timestamp is the CMM table's own
    timestamp for the epoch this PL belongs to, which is the only key that
    survives a change of sampling rate.

    Returns dict[timestamp_key] = hpl_deg (float).
    """
    fname = traj_name.replace('.', '_')
    raim_file = raim_pattern.format(traj=fname)
    if not os.path.exists(raim_file):
        print(f"  WARNING: RAIM file not found: {raim_file}")
        return {}

    pl_dict = {}
    with open(raim_file, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        expected = "epoch", "cmm_timestamp", "hpl_m", "hpl_deg"
        if tuple(header.split(";"))[:4] != expected:
            print(f"  WARNING: {raim_file} has unexpected columns: {header!r}")
        for line in f:
            parts = line.strip().split(";")
            if len(parts) >= 4:
                pl_dict[_ts_key(parts[1])] = float(parts[3])
    return pl_dict


def merge_pl(base_dir: str = DEFAULT_BASE_DIR,
             cmm_file: str = None,
             out_file: str = None) -> None:
    """Main merge function: replace protection_level with RAIM HPL.

    base_dir  : directory holding raim_pl_<traj>.csv and, by default,
                cmm_input_points.csv.
    cmm_file  : input aggregated CMM table (default <base_dir>/cmm_input_points.csv).
    out_file  : destination (default: overwrite cmm_file in place, after backing
                it up to <cmm_file>.bak). Pass an explicit path to keep the
                original untouched.
    """
    if cmm_file is None:
        cmm_file = os.path.join(base_dir, "cmm_input_points.csv")
    raim_pattern = os.path.join(base_dir, "raim_pl_{traj}.csv")
    CMM_FILE = cmm_file
    write_file = out_file or cmm_file

    if not os.path.exists(CMM_FILE):
        print(f"ERROR: CMM input file not found: {CMM_FILE}")
        sys.exit(1)

    if out_file is not None:
        print(f"  Writing to a separate file; {CMM_FILE} will NOT be modified.")
    else:
        print(f"  NOTE: {CMM_FILE} will be overwritten in place.")

    # Backup original
    bak_file = CMM_FILE + ".bak"
    if not os.path.exists(bak_file):
        shutil.copy2(CMM_FILE, bak_file)
        print(f"Backed up to {bak_file}")

    # Load all RAIM PL data
    all_pl = {}
    trajectory_names = ["1.1", "1.2", "1.3", "1.4", "2.1", "2.2", "2.3"]
    for traj in trajectory_names:
        pl_dict = load_raim_pl(traj, raim_pattern)
        traj_id = int(traj.replace(".", ""))  # "1.1" -> 11
        all_pl[traj_id] = pl_dict
        print(f"  Trajectory {traj}: {len(pl_dict)} PL epochs loaded")

    # Process CMM file line by line
    output_lines = []
    per_traj_total = {}   # traj_id -> rows seen
    per_traj_repl = {}    # traj_id -> rows whose PL came from RAIM
    used = {tid: set() for tid in all_pl}  # traj_id -> RAIM timestamps consumed
    replaced_count = 0
    kept_count = 0

    with open(CMM_FILE, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        output_lines.append(header)
        cols = header.split(";")
        ncols = len(cols)
        # Column index for protection_level (last column in standard format)
        pl_col_idx = ncols - 1  # protection_level is last
        try:
            ts_col_idx = cols.index("timestamp")
        except ValueError:
            print(f"ERROR: no 'timestamp' column in {CMM_FILE}; cannot key the merge")
            sys.exit(1)

        for line in f:
            line = line.strip()
            if not line:
                output_lines.append(line)
                continue

            parts = line.split(";")
            if len(parts) < ncols:
                output_lines.append(line)
                continue

            try:
                traj_id = int(parts[0])
                ts = _ts_key(parts[ts_col_idx])
            except ValueError:
                output_lines.append(line)
                continue

            per_traj_total[traj_id] = per_traj_total.get(traj_id, 0) + 1

            new_pl = all_pl.get(traj_id, {}).get(ts)
            if new_pl is not None:
                parts[pl_col_idx] = f"{new_pl:.10f}"
                replaced_count += 1
                used[traj_id].add(ts)
                per_traj_repl[traj_id] = per_traj_repl.get(traj_id, 0) + 1
            else:
                kept_count += 1

            output_lines.append(";".join(parts))

    # Write output
    with open(write_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\n  Wrote {write_file}")

    # Print statistics
    print(f"\n=== Merge Summary ===")
    print(f"  Replaced PL: {replaced_count} epochs")
    print(f"  Kept original PL: {kept_count} epochs (no RAIM data for that epoch)")
    unmatched_worst = []
    for traj_id, total in sorted(per_traj_total.items()):
        repl = per_traj_repl.get(traj_id, 0)
        avail = len(all_pl.get(traj_id, {}))
        unused = avail - len(used.get(traj_id, ()))
        print(f"  Traj {traj_id}: {total} epochs, {repl} PL from RAIM "
              f"({avail} available, {unused} unused)")
        # A trajectory that has RAIM data for roughly every epoch but matches
        # almost none of them is the signature of a key mismatch, not of missing
        # data. Report it loudly instead of letting it look like a clean merge.
        if avail and repl < 0.5 * min(total, avail):
            unmatched_worst.append(traj_id)

    if unmatched_worst:
        print(f"\n  *** WARNING: trajectories {unmatched_worst} matched fewer than "
              f"half the RAIM epochs available to them.")
        print(f"  *** The RAIM timestamps and the CMM timestamps are probably not "
              f"in the same time base.")
        print(f"  *** (RINEX epoch tags are GPS time; the CMM table is UTC. If "
              f"compute_raim_pl.py ever stops propagating the CMM timestamp "
              f"verbatim, check the leap-second offset before trusting this run.)")

    # Statistics of new PL values
    if replaced_count > 0:
        new_pls = []
        for line in output_lines[1:]:
            parts = line.split(";")
            if len(parts) >= ncols and parts[0].isdigit():
                new_pls.append(float(parts[pl_col_idx]))

        if new_pls:
            pl_arr = np.array(new_pls)
            print(f"\n  New PL stats (deg):")
            print(f"    min   = {pl_arr.min():.8f}")
            print(f"    mean  = {pl_arr.mean():.8f}")
            print(f"    median= {np.median(pl_arr):.8f}")
            print(f"    max   = {pl_arr.max():.8f}")
            # Convert to meters
            pl_m = pl_arr * 111320.0
            print(f"  New PL stats (m):")
            print(f"    min   = {pl_m.min():.1f}")
            print(f"    mean  = {pl_m.mean():.1f}")
            print(f"    median= {np.median(pl_m):.1f}")
            print(f"    max   = {pl_m.max():.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR,
                    help="directory holding raim_pl_<traj>.csv (default: %(default)s)")
    ap.add_argument("--cmm-input", default=None,
                    help="input CMM table (default: <base-dir>/cmm_input_points.csv)")
    ap.add_argument("--out", default=None,
                    help="output path; omit to overwrite --cmm-input in place")
    args = ap.parse_args()
    merge_pl(args.base_dir, args.cmm_input, args.out)


if __name__ == "__main__":
    main()
