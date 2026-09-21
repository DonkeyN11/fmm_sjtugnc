#!/usr/bin/env python3
"""One table: every real-data artifact this project has, on one ground truth.

Written for the paper-alignment review, where the question is not "what does the
current code score" but "which of the files on disk still supports which
manuscript row". Each artifact on disk was produced under a different
combination of (ground truth, covariance scale, candidate radius), so a table
that scores them all against a single GT is the only way to see what moved.

Metric definitions are IMPORTED from verify_paper_numbers.py -- that file is the
canonical evaluator (see the note in eval_result_csv.py) and this script exists
only to widen its input surface, not to restate its arithmetic.

Two input shapes are accepted:

  result   the matcher's own output: `cpath`, `trustworthiness`, `pgeom`
  merged   an `aligned*.csv`, which glues observations, results and ground truth
           into one row: `cmm_cpath` / `cmm_tw` / `cmm_x` / `cmm_y`, and the same
           for `fmm_`

Usage:
    python experiments/scripts/inventory_real_metrics.py \
        --gt data/real_vehicle/hainan_06/processed/aligned_0729.csv \
        --row paper_CaMM=processed/aligned_0729.csv:cmm \
        --row new_CaMM=processed/cmm_result_0729.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import verify_paper_numbers as v  # noqa: E402  (path set above)

BASE = SCRIPT_DIR.parents[1]
REAL = BASE / "data/real_vehicle/hainan_06/processed"
TW_SENTINEL = -900.0

# A merged file names its result columns by prefix; a raw result file does not.
MERGED = {
    "cmm": ("cmm_cpath", "cmm_tw", "cmm_x", "cmm_y"),
    "fmm": ("fmm_cpath", "fmm_tw", "fmm_x", "fmm_y"),
}


def load_gt(path: Path):
    """Two views of the same ground truth, both over evaluable epochs only.

    Same population rule as eval_result_csv.py: EXCLUDE_TRAJ first, then drop
    gt_edge in {'0','-1',''} because the epoch has no ground-truth road and so
    cannot enter a denominator. The GT *point* is carried alongside but is
    allowed to be absent -- a few epochs have an edge label and no fix.

    Returned as (by_timestamp, by_seq) because the two result families need
    different keys. CaMM's output carries a per-epoch timestamp and a `seq` that
    restarts at every sub-segment boundary, so timestamp is its key; FMM's
    output carries a `seq` that counts points within the trajectory but a
    *constant* timestamp per trajectory, so seq is its key. `uni_seq` in the
    aligned files is that same per-trajectory point index -- the row counts
    agree exactly (2696/734/2400/1407/3630/2551/2737). See load_run().
    """
    by_ts, by_seq = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["id"].strip() in v.EXCLUDE_TRAJ:
                continue
            e = r["gt_edge"].strip()
            if e in ("0", "-1", ""):
                continue
            gx, gy = r["gt_x"].strip(), r["gt_y"].strip()
            pt = (float(gx), float(gy)) if gx and gy else None
            tid = r["id"].strip()
            by_ts[(tid, v.norm_ts(r["timestamp"]))] = (e, pt)
            by_seq[(tid, int(r["uni_seq"]))] = (e, pt)
    return by_ts, by_seq


def load_run(path: Path, prefix: str | None):
    """Result rows, keyed on whichever of (id,timestamp) / (id,seq) is injective.

    Row order is not a key (use_omp), and neither candidate key is universal, so
    the choice is made from the data rather than assumed: a key that maps two
    rows to the same slot has silently collapsed the population, which is how a
    rank statistic ends up computed over 137 epochs instead of 16155.
    """
    rows = []
    p = MERGED.get(prefix) if prefix else None
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if p:
                cpath, tw_s, x_s, y_s = (r[c].strip() for c in p)
            else:
                cpath, tw_s = r["cpath"].strip(), r["trustworthiness"].strip()
                x, y = v.parse_point(r["pgeom"])
                x_s, y_s = ("" if x is None else str(x)), ("" if y is None else str(y))
            if tw_s in ("", "-999", "-999.0"):
                tw = None
            else:
                tw = float(tw_s)
                if tw <= TW_SENTINEL:
                    tw = None
            try:
                pt = (float(x_s), float(y_s)) if x_s and y_s else None
            except ValueError:
                pt = None
            rows.append((r["id"].strip(), v.norm_ts(r["timestamp"]),
                         int(r["seq"]) if "seq" in r and r["seq"].strip() else None,
                         (cpath, tw, pt)))

    for label, idx in (("timestamp", 1), ("seq", 2)):
        if any(row[idx] is None for row in rows):
            continue
        if len({(row[0], row[idx]) for row in rows}) == len(rows):
            return {(row[0], row[idx]): row[3] for row in rows}, label
    raise SystemExit(f"{path.name}: neither (id,timestamp) nor (id,seq) is unique")


def evaluate(path: Path, prefix: str | None, gt):
    by_ts, by_seq = gt
    run, key_name = load_run(path, prefix)
    gt = by_ts if key_name == "timestamp" else by_seq
    labels, scores, tw_c, tw_w = [], [], [], []
    perr, n, ok, n_abs, n_failed = [], 0, 0, 0, 0
    for key, (gt_e, gt_pt) in gt.items():
        hit = run.get(key)
        if hit is None:
            continue  # absent from the output: not counted, as in eval_result_csv
        cpath, tw, pt = hit
        n += 1
        correct = v.is_edge_match(cpath, gt_e)
        ok += int(correct)
        if tw is not None:
            labels.append(1.0 if correct else 0.0)
            scores.append(tw)
            (tw_c if correct else tw_w).append(tw)
        if pt and gt_pt:
            perr.append(v.haversine_m(pt[0], pt[1], gt_pt[0], gt_pt[1]))
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    assert len(labels) == len(scores), "label/score pairing broken"
    return {
        "run": path.name + (f"[{prefix}]" if prefix else ""),
        "rows": len(run),
        "n": n,
        "acc": 100.0 * ok / n if n else float("nan"),
        "ece": v.ece(scores, labels) if len(scores) else float("nan"),
        "auc": v.auc(labels, scores) if len(scores) else float("nan"),
        "tw_c": float(np.mean(tw_c)) if tw_c else float("nan"),
        "tw_w": float(np.mean(tw_w)) if tw_w else float("nan"),
        "sep": (float(np.mean(tw_c)) - float(np.mean(tw_w))) if tw_c and tw_w else float("nan"),
        "perr": float(np.mean(perr)) if perr else float("nan"),
        "perr95": float(np.percentile(perr, 95)) if perr else float("nan"),
        "miss": n - len(scores),  # evaluable epochs with no trustworthiness
    }


def parse_row(spec: str):
    lhs, rhs = spec.split("=", 1)
    path, _, prefix = rhs.partition(":")
    p = Path(path)
    if not p.is_absolute():
        p = REAL / p
    if not p.exists():
        raise SystemExit(f"no such file: {p}")
    return lhs, p, (prefix or None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, default=REAL / "aligned_0729.csv")
    ap.add_argument("--row", action="append", required=True,
                    help="NAME=PATH[:cmm|fmm]")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    print(f"GT : {args.gt.name}  ({len(gt[0])} evaluable epochs, "
          f"trajectories {sorted({k[0] for k in gt[0]})})")
    print()
    cols = ("run", "rows", "n", "acc", "ece", "auc", "tw_c", "tw_w", "sep",
            "perr", "perr95", "miss")
    hdr = f"{'run':<30}{'rows':>7}{'n':>7}{'acc%':>9}{'ECE':>8}{'AUC':>8}" \
          f"{'TW_cor':>9}{'TW_wrg':>9}{'sep':>8}{'err_m':>8}{'p95_m':>8}{'miss':>6}"
    print(hdr)
    print("-" * len(hdr))
    for spec in args.row:
        name, path, prefix = parse_row(spec)
        m = evaluate(path, prefix, gt)
        print(f"{name:<30}{m['rows']:>7}{m['n']:>7}{m['acc']:>9.4f}{m['ece']:>8.4f}"
              f"{m['auc']:>8.4f}{m['tw_c']:>9.4f}{m['tw_w']:>9.4f}{m['sep']:>8.4f}"
              f"{m['perr']:>8.2f}{m['perr95']:>8.2f}{m['miss']:>6}")
    del cols
    return 0


if __name__ == "__main__":
    sys.exit(main())
