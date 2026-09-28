#!/usr/bin/env python3
"""
Align the real-data sources by (id, timestamp) into one wide CSV.

The real-data table is the union of four sources, all keyed on
(trajectory id, epoch timestamp):

  --obs        SPP observations    id, timestamp, x, y, sde, sdn, sdne, sdu, protection_level
  --cmm        CMM match result    id, timestamp, pgeom, cpath, trustworthiness, status, ...
  --fmm        FMM match result    id, timestamp, pgeom, cpath, trustworthiness, ...
  --gt-points  RTK ground truth    id, timestamp, x, y
  --gt-edges   RTK edge per epoch  id, seq, timestamp, x, y, edge_id

Output (--out):

  id, uni_seq, timestamp,
  obs_x, obs_y, obs_sde, obs_sdn, obs_sdne,
  cmm_x, cmm_y, cmm_tw, cmm_cpath,
  cmm_status, cmm_delta_entropy, cmm_posterior_entropy,
  fmm_x, fmm_y, fmm_tw, fmm_cpath,
  gt_x, gt_y, gt_edge

Two things this script deliberately does NOT do.

1. It does not drop any trajectory.  The output is the complete factual record
   of the dataset -- 7 trajectories / 16155 epochs, trajectory 12 included.
   A trajectory that has to be left out of a reported figure is excluded in the
   EVALUATION layer instead (EXCLUDE_TRAJ in verify_paper_numbers.py), where
   the exclusion is a stated policy that can be read, checked and argued with,
   rather than an absence in a data file.  The difference matters: the
   evaluators key off gt_edge, so a missing trajectory silently shrinks the
   denominator of every metric computed from this file, with nothing in the
   evaluation path to say that it happened.  See the EXCLUDE_TRAJ comment in
   verify_paper_numbers.py for why trajectory 12 is the one such case.

2. It does not join the ground-truth edge positionally.  Earlier revisions
   mapped gt_edge through a running per-trajectory counter ("uni_seq") into
   ground_truth.csv's `seq` column.  That join is right only as long as the
   union of the four sources holds exactly the timestamps ground_truth.csv was
   built from; add or drop one epoch anywhere and every later uni_seq shifts
   by one, silently mislabelling the rest of that trajectory -- which then
   reads as a matching error.  ground_truth.csv carries its own `timestamp`
   column, and joining on (id, timestamp) was verified to agree with the
   positional join on all 16155 epochs (0 mismatches; the two key sets are
   exactly equal), so the fragile join is gone.  uni_seq survives as an output
   column because seven plotting scripts use it as a per-trajectory sequence
   index.  It restarts at 0 for every id, so it is stable when a trajectory is
   added.

Usage:
  python experiments/scripts/align_real_data.py
  python experiments/scripts/align_real_data.py --cmm .../cmm_result.csv --out /tmp/try.csv
"""

import argparse
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAINAN_DIR = PROJECT_ROOT / "data/real_vehicle/hainan_06"
PROCESSED_DIR = HAINAN_DIR / "processed"

# Defaults point at the CURRENT inputs, not at the copies in processed/ that
# earlier revisions used.  The two that actually changed:
#   obs: processed/cmm_input_points.csv is the pre-GST-calibration May 31 table;
#        hainan_06/cmm_input_points.csv is the calibrated one.
#   cmm: processed/cmm_result.csv predates the ENU protection-level correction
#        and the pruned parameter set; cmm_result_0729.csv is the output of the
#        current code on input/config/cmm_real_0729.xml.
DEFAULTS = {
    "obs": HAINAN_DIR / "cmm_input_points.csv",
    "cmm": PROCESSED_DIR / "cmm_result_0729.csv",
    "fmm": PROCESSED_DIR / "fmm_result.csv",
    "gt_points": PROCESSED_DIR / "ground_truth_points.csv",
    "gt_edges": PROCESSED_DIR / "ground_truth.csv",
    "out": PROCESSED_DIR / "aligned.csv",
}

FIELDS = ["id", "uni_seq", "timestamp",
          "obs_x", "obs_y", "obs_sde", "obs_sdn", "obs_sdne",
          "cmm_x", "cmm_y", "cmm_tw", "cmm_tw_smoothed", "cmm_cpath",
          "cmm_status", "cmm_delta_entropy", "cmm_posterior_entropy",
          "fmm_x", "fmm_y", "fmm_tw", "fmm_cpath",
          "gt_x", "gt_y", "gt_edge"]


def parse_point(wkt):
    """Extract (lon, lat) from WKT POINT(lon lat)."""
    m = re.search(r"POINT\s*\(\s*([\d.\-]+)\s+([\d.\-]+)\s*\)", str(wkt), re.I)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def key_of(row):
    """(id, timestamp) key of a source row, or None if the epoch is unreadable.

    The tables disagree on timestamp formatting ('1750306260' vs
    '1750306260.0'), so the join must be numeric; round() lands them on the same
    integer.  Do NOT normalise by stripping trailing zeros -- that maps
    1750306260 and 175030626 to one key and silently merges two epochs.
    """
    try:
        return (row["id"].strip(), round(float(row["timestamp"])))
    except (KeyError, ValueError):
        return None


def load_csv(path, val_cols):
    """Load a source CSV into dict[(id, timestamp)] -> {column: value}.

    Missing columns are skipped rather than fatal, so one loader serves all four
    sources: they share the key columns but not the value columns.
    """
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            key = key_of(row)
            if key is None:
                continue
            vals = {c: row[c] for c in val_cols if c in row}
            if "pgeom" in row:  # match results carry the position as WKT
                vals["x"], vals["y"] = parse_point(row["pgeom"])
            result[key] = vals
    return result


def load_gt_edges(path):
    """dict[(id, timestamp)] -> edge_id, from the RTK per-epoch edge table.

    key_of() drops the `seq` column entirely -- see the module docstring.
    """
    edges = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            key = key_of(row)
            if key is not None:
                edges[key] = row["edge_id"].strip()
    return edges


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, path in DEFAULTS.items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=Path, default=path,
                        help=f"default: {path}")
    args = ap.parse_args()

    print("Loading SPP observations...")
    obs = load_csv(args.obs, ["x", "y", "sde", "sdn", "sdne", "sdu",
                              "protection_level"])
    print("Loading CMM results...")
    cmm = load_csv(args.cmm, ["pgeom", "cpath", "trustworthiness",
                              "trustworthiness_smoothed", "ep", "tp",
                              "n_best_trustworthiness", "delta_entropy",
                              "posterior_entropy", "status"])
    print("Loading FMM results...")
    fmm = load_csv(args.fmm, ["pgeom", "cpath", "trustworthiness", "ep", "tp",
                              "sp_dist", "eu_dist"])
    print("Loading RTK ground truth...")
    gt = load_csv(args.gt_points, ["x", "y"])
    gt_edges = load_gt_edges(args.gt_edges)

    # ── Union of every epoch any source knows about ──
    all_keys = set(obs) | set(cmm) | set(fmm) | set(gt)
    trajs = sorted({k[0] for k in all_keys}, key=lambda t: (len(t), t))
    print(f"\n{len(all_keys)} unique (id, timestamp) keys over "
          f"{len(trajs)} trajectories: {trajs}")

    # ── Build aligned rows ──
    rows = []
    for tid in trajs:
        traj_keys = sorted((k for k in all_keys if k[0] == tid), key=lambda k: k[1])
        for uni_seq, key in enumerate(traj_keys):
            _, ts = key
            row = {"id": tid, "uni_seq": uni_seq, "timestamp": ts}

            o = obs.get(key, {})
            row["obs_x"] = o.get("x", "")
            row["obs_y"] = o.get("y", "")
            row["obs_sde"] = o.get("sde", "")
            row["obs_sdn"] = o.get("sdn", "")
            row["obs_sdne"] = o.get("sdne", "")

            c = cmm.get(key, {})
            row["cmm_x"] = c.get("x") if c.get("x") is not None else ""
            row["cmm_y"] = c.get("y") if c.get("y") is not None else ""
            row["cmm_tw"] = c.get("trustworthiness", "")
            # The smoothed posterior of the same Viterbi candidate. It is the arm
            # the paper's distance-based charge is computed from, and it is what
            # the Case-2 window illustrates: a sustained filtering collapse that
            # the whole-trajectory evidence largely removes.
            row["cmm_tw_smoothed"] = c.get("trustworthiness_smoothed", "")
            row["cmm_cpath"] = c.get("cpath", "")
            row["cmm_status"] = c.get("status", "")
            row["cmm_delta_entropy"] = c.get("delta_entropy", "")
            row["cmm_posterior_entropy"] = c.get("posterior_entropy", "")

            f = fmm.get(key, {})
            row["fmm_x"] = f.get("x") if f.get("x") is not None else ""
            row["fmm_y"] = f.get("y") if f.get("y") is not None else ""
            row["fmm_tw"] = f.get("trustworthiness", "")
            row["fmm_cpath"] = f.get("cpath", "")

            g = gt.get(key, {})
            row["gt_x"] = g.get("x") if g.get("x") is not None else ""
            row["gt_y"] = g.get("y") if g.get("y") is not None else ""
            row["gt_edge"] = gt_edges.get(key, "")

            rows.append(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter=";")
        w.writeheader()
        w.writerows(rows)

    n_eval = sum(1 for r in rows if r["gt_edge"] not in ("0", "-1", ""))
    print(f"\nSaved: {args.out} ({len(rows)} rows)")
    print(f"  trajectories:      {len(trajs)}")
    print(f"  evaluable epochs:  {n_eval} (gt_edge not in {{0, -1, ''}})")
    print("  per-trajectory:    " + "  ".join(
        f"{t}={sum(1 for r in rows if r['id'] == t)}" for t in trajs))


if __name__ == "__main__":
    main()
