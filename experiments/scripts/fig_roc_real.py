#!/usr/bin/env python3
"""ROC of trustworthiness as a mismatch detector on the real-vehicle dataset.

Regenerates figs/roc_direction_aware_fixed_en.png from the two runs the paper's
Section V reports: the final CaMM configuration (k_search = 1, delta_road =
13.5 m, elliptical covariance, Mahalanobis projection, direction consistency,
PHMI grouping, forward posterior) and the HMM baseline scored with its own
native Viterbi score delta_t / sum_j delta_t(j) at r = 300 m, sigma = 100 m,
reverse_tolerance = 0.1.

The previous version of this figure was never produced by a checked-in script,
so it could not be regenerated when the baseline's score definition changed; the
AUCs it printed (0.721 / 0.583) belong to a baseline that is no longer the one
reported.

Usage:
    python experiments/scripts/fig_roc_real.py --cmm /tmp/cmm_final.csv \
        --hmm /tmp/fmm_final.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import inventory_real_metrics as I  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

BASE = SCRIPT_DIR.parents[1]
FIG = (BASE / "docs/calib/figs")


def pairs(path: Path, gt):
    """(score, label) over the evaluable epochs the run answers."""
    run, key = I.load_run(path, None)
    G = gt[0] if key == "timestamp" else gt[1]
    s, y = [], []
    for k, (gt_edge, _) in G.items():
        hit = run.get(k)
        if hit is None:
            continue
        cpath, tw, _pt = hit
        if tw is None or tw <= -900:
            continue
        s.append(tw)
        y.append(1.0 if v.is_edge_match(cpath, gt_edge) else 0.0)
    return np.asarray(s), np.asarray(y)


def roc(scores, labels):
    """Step-wise ROC; ties handled by grouping equal scores."""
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1.0 - y)
    keep = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tpr = np.r_[0.0, tps[keep] / tps[-1]]
    fpr = np.r_[0.0, fps[keep] / fps[-1]]
    return fpr, tpr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cmm", type=Path, required=True)
    ap.add_argument("--hmm", type=Path, required=True)
    ap.add_argument("--gt", type=Path,
                    default=BASE / "data/real_vehicle/hainan_06/processed/aligned_0729.csv")
    ap.add_argument("--out", type=Path, default=FIG / "roc_direction_aware_fixed_en.png")
    args = ap.parse_args()

    gt = I.load_gt(args.gt)
    cs, cy = pairs(args.cmm, gt)
    hs, hy = pairs(args.hmm, gt)
    ca, ha = v.auc(cy, cs), v.auc(hy, hs)
    print(f"CaMM  n={len(cs)}  mismatches={int((cy == 0).sum())}  AUC={ca:.4f}")
    print(f"HMM   n={len(hs)}  mismatches={int((hy == 0).sum())}  AUC={ha:.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    for s, y, a, lab, col, ls in ((cs, cy, ca, "CaMM", "#c0392b", "-"),
                                  (hs, hy, ha, "HMM baseline", "#7f8c8d", "--")):
        f, t = roc(s, y)
        ax.plot(f, t, color=col, ls=ls, lw=2.2, label=f"{lab} (AUC={a:.3f})", zorder=3)
    ax.plot([0, 1], [0, 1], "k:", lw=1.2, label="Chance (AUC=0.5)", zorder=2)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("Real-vehicle ROC: Mismatch Detection", fontsize=16)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=13)
    ax.legend(loc="lower right", fontsize=13, frameon=True)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
