#!/usr/bin/env python3
"""Sample-rate sensitivity figure for the paper (CaMM vs HMM).

Reads the metrics written by exp3_full_matching.py's sample-rate stage and draws
the 2x2 panel the manuscript uses: (a) segment accuracy, (b) ECE of the
trustworthiness score, (c) ROC AUC, (d) mean point error, against the sampling
interval.

The interval set is not hard-coded here; it is whatever the metrics file
contains, because the manuscript's text and this figure previously disagreed on
it. `exp3_full_matching.py` currently decimates to SAMPLE_INTERVALS = [1, 2, 5,
10] s; if that list changes, the figure and the text have to change with it.

Usage:
    python experiments/scripts/fig_sample_rate.py \
        --metrics experiments/output/3_full_matching/sample_rate_full.json \
        --out <paper>/figs/sample_rate_sensitivity_fixed_en.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
FIGS = (BASE / "docs/calib/figs")

# One colour per noise level, as in the rest of the paper's figures.
SIGMA_STYLE = {
    "sigma_05": ("#2c7fb8", r"$\sigma_\rho=5$m"),
    "sigma_15": ("#e6550d", r"$\sigma_\rho=15$m"),
    "sigma_25": ("#756bb1", r"$\sigma_\rho=25$m"),
}


def series(entries):
    """label -> list of (interval, metric dict)."""
    out = defaultdict(list)
    for m in entries:
        out[m["label"].split("_sr")[0]].append((int(m["sample_interval"]), m))
    for k in out:
        out[k].sort()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path,
                    default=BASE / "experiments/output/3_full_matching/sample_rate_full.json")
    ap.add_argument("--out", type=Path,
                    default=FIGS / "sample_rate_sensitivity_fixed_en.png")
    args = ap.parse_args()

    data = json.loads(args.metrics.read_text())
    cmm, hmm = series(data["cmm"]), series(data["fmm"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("seg_accuracy", lambda v: v * 100.0, "(a) Segment Accuracy", "Accuracy (%)"),
        ("ece_tw", lambda v: v, "(b) ECE (Trustworthiness)", "ECE"),
        ("roc_auc", lambda v: v, "(c) ROC AUC", "AUC"),
        ("point_error_mean", lambda v: v, "(d) Point Error", "Mean Error (m)"),
    ]
    for ax, (key, scale, title, ylab) in zip(axes.flat, panels):
        for sig, (col, _lab) in SIGMA_STYLE.items():
            if sig in cmm:
                xs = [i for i, _ in cmm[sig]]
                ax.plot(xs, [scale(m[key]) for _, m in cmm[sig]],
                        "o-", color=col, lw=1.6, ms=6, zorder=3)
            if sig in hmm:
                xs = [i for i, _ in hmm[sig]]
                ax.plot(xs, [scale(m[key]) for _, m in hmm[sig]],
                        "s--", color=col, lw=1.6, ms=6, alpha=0.75, zorder=2)
        ax.set_xlabel("Sample Interval (s)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks(sorted({i for s in (cmm, hmm) for i, _ in s.get(list(SIGMA_STYLE)[0], [])}))
        ax.grid(alpha=0.3)
        if key == "roc_auc":
            ax.axhline(0.5, color="gray", lw=1.0, ls="--")

    handles = [Line2D([], [], color="k", marker="o", ls="-", ms=7, label="CaMM (solid)"),
               Line2D([], [], color="k", marker="s", ls="--", ms=7, alpha=0.75,
                      label="HMM (dashed)")]
    handles += [Line2D([], [], color=c, lw=2.5, label=lab) for c, lab in SIGMA_STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Sample Rate Sensitivity: CaMM vs HMM ($k=16$)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")

    for key, _s, _t, _y in panels:
        print(f"\n{key}")
        for sig in SIGMA_STYLE:
            c = ", ".join(f"{i}s={scale(m[key]):.4f}" for i, m in cmm.get(sig, []))
            h = ", ".join(f"{i}s={scale(m[key]):.4f}" for i, m in hmm.get(sig, []))
            print(f"  {sig}  CaMM[{c}]")
            print(f"  {sig}  HMM [{h}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
