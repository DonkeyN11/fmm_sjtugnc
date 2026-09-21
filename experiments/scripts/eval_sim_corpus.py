#!/usr/bin/env python3
"""Segment accuracy / ECE / ROC-AUC over a simulation corpus.

Edge identity is decided by `verify_paper_numbers.is_edge_match`, the project's
canonical definition, imported rather than reimplemented so a number computed
here cannot silently disagree with one computed elsewhere.

How an epoch is joined to its ground-truth edge
-----------------------------------------------
`exp3_full_matching.load_ground_truth` resolves seq through a timestamp map built
only from `ground_truth_points.csv`, which the sweep family does not have -- so
all 70 000 of its epochs resolve to nothing and the dataset silently scores as
empty rather than failing.

The join used here is the timestamp, taken from `observations.csv` (present in
every family, and the very table the matcher consumed). `(id, timestamp)` is
unique across all 247 030 observation rows, so the map is lossless; the matcher
rounds the timestamp to an integer second, so the lookup uses `int(round(ts))`.

The output's `seq` column is deliberately NOT the join key. It is the input index
only while a trajectory stays in one piece: when the matcher splits one, `seq`
restarts at 0 while the timestamps keep running, so the same `(id, seq)` names
two different epochs. That happens here -- `sigma_10/no_occlusion/no_fault@id=2`
emits 950 rows whose `seq` starts at 0 with timestamps 50 s into the observation.
`seq` is therefore only a cross-check, and `--cross-check` re-derives every
dataset's labels through exp3's independent loader and asserts they agree.

What this does NOT take from exp3_full_matching.compute_metrics:

1. **AUC.** `exp3.compute_roc_auc` integrates the ROC step curve with
   `np.trapz`. `verify_paper_numbers.auc` is the tie-corrected Mann-Whitney rank
   statistic, and its docstring records why the difference matters: trapz
   interpolates linearly across a step that is vertical, which reversed the
   CMM > FMM ordering the manuscript reports. The rank statistic is the one the
   paper quotes, so it is the one used here.

2. **ECE.** `exp3.compute_metrics` builds `trusts` over *every* row but
   `edge_correct` only over rows that have a ground-truth edge, then calls
   `compute_ece(trust_arr, edge_corr)`. Those arrays have different lengths, so
   the per-bin `labels[idxs]` lookup reads the wrong epochs -- silently, since
   the index stays in range until the very end. Here the label and the score are
   appended together, which is the same rule `eval_result_csv.py` states for the
   real-data side.

Usage
-----
    python experiments/scripts/eval_sim_corpus.py --root data/simulation_plfix \
        --suffix _plfix --baseline-suffix _prepl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments/scripts"))

import exp3_full_matching as exp3  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

TW_SENTINEL = -900.0  # the writer emits -999.0 for "no value"


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _seq(row: dict) -> int:
    """The seq column as an int.

    seq is the join key on both sides, so it must have one representation: the
    ground-truth array is indexed with `enumerate` (int) while the CSV column is
    text. Mixing the two silently misses every lookup.
    """
    return int(float(row["seq"]))


def load_gt_edges(data_dir: Path) -> dict:
    """(id, seq) -> edge_id, from ground_truth.csv's point_edge_ids JSON array."""
    out = {}
    for row in _read_csv(data_dir / "ground_truth.csv"):
        try:
            edges = json.loads(row["point_edge_ids"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        tid = row["id"].strip()
        for seq, eid in enumerate(edges):  # the array is indexed by seq
            out[(tid, seq)] = str(eid)
    return out


def load_obs_index(data_dir: Path) -> tuple[dict, dict]:
    """(id, ts_rounded) -> seq, and (id, seq) -> ts_rounded from observations.csv.

    The first is the join key (lossless: (id, timestamp) is unique across the
    corpus); the second only exists to detect a `seq` column that has restarted.
    """
    by_ts, by_seq = {}, {}
    for row in _read_csv(data_dir / "observations.csv"):
        tid = row["id"].strip()
        seq = _seq(row)
        ts = int(round(float(row["timestamp"])))
        by_ts[(tid, ts)] = seq
        by_seq[(tid, seq)] = ts
    return by_ts, by_seq


def metrics_for(data_dir: Path, mr_csv: Path, cross_check: bool = False) -> dict:
    gt_edges = load_gt_edges(data_dir)
    ts_to_seq, seq_to_ts = load_obs_index(data_dir)

    labels, scores, resolved = [], [], []
    status = Counter()
    n_rows = n_evaluable = n_seq_agrees = n_seq_diverged = n_unresolved = 0
    with mr_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            n_rows += 1
            id_ = row["id"].strip()
            status[row.get("status", "?")] += 1
            cpath = row.get("cpath", "").strip()

            # The matcher rounds the timestamp to an integer second.
            ts = int(round(float(row["timestamp"])))
            seq = ts_to_seq.get((id_, ts))

            if seq is None:
                n_unresolved += 1
                continue  # not an epoch this dataset ever observed
            if seq == _seq(row):
                n_seq_agrees += 1
            else:
                # A split trajectory: seq restarted, so the column is stale.
                n_seq_diverged += 1
            if (id_, seq) not in gt_edges:
                n_unresolved += 1
                continue  # no ground-truth road here: outside the denominator

            n_evaluable += 1
            resolved.append((id_, seq))
            correct = float(v.is_edge_match(cpath, gt_edges[(id_, seq)]))

            tw = float(row.get("trustworthiness", TW_SENTINEL) or TW_SENTINEL)
            if tw <= TW_SENTINEL:
                continue  # sentinel: the epoch has no score, so it pairs with nothing
            labels.append(correct)
            scores.append(tw)

    labels = np.array(labels)
    scores = np.array(scores)
    assert len(labels) == len(scores)
    if cross_check:
        _assert_agrees_with_exp3(data_dir, mr_csv, resolved)
    return {
        "n_rows": n_rows,
        "n_evaluable": n_evaluable,
        "n_seq_agrees": n_seq_agrees,
        "n_seq_diverged": n_seq_diverged,
        "n_unresolved": n_unresolved,
        "n_scored": len(labels),
        "seg_accuracy": 100.0 * float(labels.mean()) if len(labels) else float("nan"),
        "ece": float(v.ece(scores, labels)) if len(scores) else float("nan"),
        "auc": float(v.auc(labels, scores)) if len(scores) else float("nan"),
        "mean_tw": float(scores.mean()) if len(scores) else float("nan"),
        "status": dict(status),
    }


def _assert_agrees_with_exp3(data_dir: Path, mr_csv: Path, ours: list) -> None:
    """Re-derive the (id, seq) join through exp3's independent loader.

    exp3 reads the ground truth from a different file through a different
    mechanism, so agreement is evidence and not a tautology. Comparing the
    resolved (id, seq) is a far sharper test than comparing labels: the corpus is
    94-99.9% correct, so two different joins would still agree on most labels.
    A dataset exp3 cannot resolve at all (no ground_truth_points.csv) is reported
    rather than failed -- that is the known gap being worked around.
    """
    _, edg_seq, _, seq_ts = exp3.load_ground_truth(data_dir)
    if not seq_ts:
        print(f"    cross-check: exp3 resolves nothing ({data_dir.name}), skipped")
        return
    theirs = []
    for row in _read_csv(mr_csv):
        id_ = row["id"].strip()
        ts = float(row["timestamp"])
        seq = seq_ts.get((id_, ts))
        if seq is None:
            seq = seq_ts.get((id_, int(round(ts))))
        if seq is None or (id_, seq) not in edg_seq:
            continue
        tw = float(row.get("trustworthiness", TW_SENTINEL) or TW_SENTINEL)
        if tw <= TW_SENTINEL:
            continue
        theirs.append((id_, seq))
    assert theirs == ours, (
        f"{data_dir}: cross-check join differs "
        f"({len(ours)} vs {len(theirs)} epochs; "
        f"first mismatch {next((a for a, b in zip(ours, theirs) if a != b), None)})")
    print(f"    cross-check: {len(ours)} epochs, identical join")


def family_of(rel: str) -> str:
    if rel.startswith("sigma_mismatch/"):
        return "mismatch"
    if "subsample_" in rel:
        return "subsample"
    if rel.startswith("sigma_01_to_30"):
        return "sweep"
    return "sigma"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/simulation_plfix"))
    ap.add_argument("--suffix", default="_plfix")
    ap.add_argument("--baseline-suffix", default="_prepl",
                    help="Suffix of the comparison run, or '' to skip.")
    ap.add_argument("--baseline-root", type=Path, default=None,
                    help="Tree holding the baseline run when it is not the same "
                         "tree as --root (relative paths are matched under it).")
    ap.add_argument("--cross-check", action="store_true",
                    help="Re-derive every dataset's labels through exp3's "
                         "independent timestamp-based loader and assert equality.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write the JSON summary. Defaults to "
                         "<root>/metrics_summary.json, which OVERWRITES an existing "
                         "file from an earlier corpus; prefer --out into /tmp.")
    ap.add_argument("--no-write", action="store_true",
                    help="Evaluate and print without writing any JSON summary.")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwriting an existing <root>/metrics_summary.json. "
                         "Without this, a run that would clobber one stops and asks "
                         "for --out or --force instead.")
    args = ap.parse_args()

    root = (BASE / args.root).resolve()
    broot = (BASE / args.baseline_root).resolve() if args.baseline_root else root
    dirs = sorted(p.parent for p in root.rglob("observations.csv"))

    # Decide the write target BEFORE the (slow) evaluation, so a run that is going
    # to refuse does not first spend minutes recomputing. This guard exists because
    # an evaluation run without --out silently overwrote data/simulation/
    # metrics_summary.json, which is untracked and gitignored, so neither git nor a
    # backup could restore it. Nothing else in the repository reads that file -- it
    # is a write-only run log -- but "nothing reads it" is not a property the script
    # was entitled to assume.
    out = None
    if not args.no_write:
        out = args.out or (root / "metrics_summary.json")
        if out.exists() and not args.force and args.out is None:
            print(f"refusing to overwrite {out}\n"
                  f"  it holds an earlier corpus's evaluation and is not under version "
                  f"control.\n"
                  f"  pass --out /tmp/… to write elsewhere, --force to overwrite, or "
                  f"--no-write to skip.\n", file=sys.stderr)
            return 2

    rows = []
    print(f"corpus: {root}   ({len(dirs)} datasets)\n")
    hdr = (f"{'dataset':44s} {'family':9s} {'n_eval':>7s} "
           f"{'acc%':>8s} {'ECE':>8s} {'AUC':>7s} {'meanTW':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for d in dirs:
        rel = str(d.relative_to(root))
        fam = family_of(rel)
        cur = d / f"cmm_result{args.suffix}.csv"
        if not cur.exists():
            print(f"{rel:44s} {fam:9s}   (no result)")
            continue
        m = metrics_for(d, cur, cross_check=args.cross_check)
        rec = {"dataset": rel, "family": fam, **m}

        base = None
        if args.baseline_suffix:
            # The baseline run lives beside its own observations/ground_truth,
            # so resolve it under --baseline-root at the same relative path.
            bdir = broot / d.relative_to(root)
            b = bdir / f"cmm_result{args.baseline_suffix}.csv"
            if b.exists():
                base = metrics_for(bdir, b)
                rec["baseline"] = base
                rec["delta"] = {
                    "seg_accuracy": m["seg_accuracy"] - base["seg_accuracy"],
                    "ece": m["ece"] - base["ece"],
                    "auc": m["auc"] - base["auc"],
                }
        rows.append(rec)

        warn = ""
        if m["n_seq_diverged"]:
            warn += f"  [split-traj {m['n_seq_diverged']}]"
        if m["n_unresolved"]:
            warn += f"  [unresolved {m['n_unresolved']}]"
        print(f"{rel:44s} {fam:9s} {m['n_evaluable']:7d} "
              f"{m['seg_accuracy']:8.4f} {m['ece']:8.4f} {m['auc']:7.4f} "
              f"{m['mean_tw']:7.4f}{warn}")
        if base is not None:
            print(f"{'  (old PL)':44s} {'':9s} {base['n_evaluable']:7d} "
                  f"{base['seg_accuracy']:8.4f} {base['ece']:8.4f} {base['auc']:7.4f} "
                  f"{base['mean_tw']:7.4f}   Δ {rec['delta']['seg_accuracy']:+.4f}pp "
                  f"{rec['delta']['ece']:+.4f} {rec['delta']['auc']:+.4f}")

    if out is None:
        print("\n--no-write: no summary written")
    else:
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
