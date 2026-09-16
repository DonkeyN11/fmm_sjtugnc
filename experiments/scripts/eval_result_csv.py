#!/usr/bin/env python3
"""Evaluate CaMM/CMM result CSVs, and diff two of them epoch by epoch.

The metric definitions are IMPORTED from verify_paper_numbers.py rather than
copied. That file's helpers were corrected in commit f728584 (AUC as the
tie-corrected Mann-Whitney rank statistic instead of trapz on the ROC steps,
and the -999 trustworthiness sentinel treated as missing), so it is the
canonical evaluator in this repo. commit f728584's message records that with
those fixes the CMM row reproduces the manuscript's 0.721 / 0.040 / 96.0.
Several other scripts under experiments/scripts and python/experiments still
carry pre-f728584 copies of the metric code; do NOT take numbers from them, and
do not add another copy here.

Usage:
    # metrics for one or more runs
    python experiments/scripts/eval_result_csv.py runA.csv runB.csv

    # epoch-by-epoch comparison of two runs
    python experiments/scripts/eval_result_csv.py --diff runA.csv runB.csv

The diff is keyed on (id, timestamp) and compares column values, because
use_omp makes the ROW ORDER of the output file nondeterministic while the cell
values stay deterministic. Never compare two runs line by line.

The (id, timestamp) key is also why ``seq`` is not part of it: the writer
restarts seq at 0 at every sub-segment boundary, so it is not unique.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import verify_paper_numbers as v  # noqa: E402  (path set above)

BASE = SCRIPT_DIR.parents[1]
GT_CSV = BASE / "data/real_vehicle/hainan_06/processed/aligned.csv"
TW_SENTINEL = -900.0  # writer emits -999.0; anything at or below this is "no value"


def load_gt():
    """(id, timestamp) -> gt_edge, for epochs that have a road edge.

    Mirrors verify_paper_numbers.main(): gt_edge in {'0','-1',''} means the
    epoch has no ground-truth road and is not evaluable, so it is dropped from
    the denominator entirely.
    """
    gt = {}
    with open(GT_CSV, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            e = r["gt_edge"].strip()
            if e in ("0", "-1", ""):
                continue
            gt[(r["id"].strip(), v.norm_ts(r["timestamp"]))] = e
    return gt


def load_run(path):
    """(id, timestamp) -> row dict, with trustworthiness None where sentinel."""
    d = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            try:
                tw = float(r["trustworthiness"])
            except (ValueError, KeyError):
                tw = TW_SENTINEL
            if tw <= TW_SENTINEL:
                tw = None
            r["_tw"] = tw
            d[(r["id"].strip(), v.norm_ts(r["timestamp"]))] = r
    return d


def metrics(path, gt):
    run = load_run(path)
    labels, scores, n, ok = [], [], 0, 0
    n_failed = 0
    for key, gt_e in gt.items():
        row = run.get(key)
        if row is None:
            continue  # epoch absent from the output: not counted, as in verify_*
        n += 1
        correct = v.is_edge_match(row["cpath"].strip(), gt_e)
        ok += int(correct)
        # Accuracy counts every matched epoch, including a FAILED one: its cpath
        # is empty, so it is a miss. But the calibration metrics below need
        # (label, score) pairs, and a FAILED row carries the TW sentinel, i.e.
        # no score. So label and score must be appended together or not at all --
        # keeping them in separate loops silently misaligns them the first time a
        # run contains a sentinel-TW epoch, which is exactly what happened when
        # the protection level was corrected and 112 such epochs appeared.
        # verify_paper_numbers.main() splits into tw_c/tw_w and appends only when
        # tw is not None, which is the same rule expressed as two lists.
        if row["_tw"] is not None:
            labels.append(1.0 if correct else 0.0)
            scores.append(row["_tw"])
        if row.get("status", "").startswith("FAILED"):
            n_failed += 1
    labels = np.array(labels)
    scores = np.array(scores)
    assert len(labels) == len(scores), "label/score pairing broken"
    return {
        "run": Path(path).name,
        "rows": len(run),
        "n": n,
        "acc": 100.0 * ok / n if n else float("nan"),
        "ece": v.ece(scores, labels) if len(scores) else float("nan"),
        "auc": v.auc(labels, scores) if len(scores) else float("nan"),
        "meanTW": scores.mean() if len(scores) else float("nan"),
        "failed": n_failed,
    }


def diff(path_a, path_b):
    a, b = load_run(path_a), load_run(path_b)
    ka, kb = set(a), set(b)
    if ka != kb:
        only_a, only_b = ka - kb, kb - ka
        print(f"KEY SETS DIFFER: {len(only_a)} only in A, {len(only_b)} only in B")
        for k in sorted(only_b)[:5]:
            print(f"  only in B (new row): id={k[0]} ts={k[1]}")
        for k in sorted(only_a)[:5]:
            print(f"  only in A (lost row): id={k[0]} ts={k[1]}")
    cols = [c for c in next(iter(a.values())).keys() if not c.startswith("_")]
    counts = defaultdict(int)
    for k in ka & kb:
        for c in cols:
            if a[k].get(c) != b[k].get(c):
                counts[c] += 1
    if not counts:
        print(f"IDENTICAL on all {len(cols)} columns over {len(ka & kb)} shared epochs")
        return True
    print(f"DIFFER on {len(ka & kb)} shared epochs:")
    for c, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {c:26s} {n}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--diff", action="store_true",
                    help="compare exactly two runs epoch by epoch")
    args = ap.parse_args()

    if args.diff:
        if len(args.csvs) != 2:
            ap.error("--diff takes exactly two CSVs")
        sys.exit(0 if diff(*args.csvs) else 1)

    gt = load_gt()
    print(f"GT: {len(gt)} evaluable epochs, "
          f"trajectories {sorted({k[0] for k in gt})}")
    hdr = f"{'run':<34}{'rows':>7}{'n':>7}{'acc%':>10}{'ECE':>10}{'AUC':>9}{'meanTW':>9}{'failed':>7}"
    print(hdr)
    print("-" * len(hdr))
    for p in args.csvs:
        m = metrics(p, gt)
        print(f"{m['run']:<34}{m['rows']:>7}{m['n']:>7}{m['acc']:>10.4f}"
              f"{m['ece']:>10.4f}{m['auc']:>9.4f}{m['meanTW']:>9.4f}{m['failed']:>7}")


if __name__ == "__main__":
    main()
