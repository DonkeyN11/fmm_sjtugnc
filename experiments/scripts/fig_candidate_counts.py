#!/usr/bin/env python3
"""Candidate count comparison: CaMM vs the fixed-radius HMM baseline.

Both series are measured, not assumed. The HMM series used to be drawn as a
hard-coded `[16] * n` on the reasoning that a fixed radius saturates the k = 16
cap; it does not. At r = 300 m the baseline averages 2.6-4.5 candidates per
epoch on this corpus, because k is a ceiling and most epochs have far fewer
than k road segments inside the disk. The figure therefore runs `build/fmm`
itself and caches the per-epoch counts next to the family it measured.

The cache is `fmm_candidates.csv` in each family directory, written once and
reused; delete it to force a re-measurement. It is written to its own file
rather than appended to `fmm_result.csv` so that re-running this figure cannot
change any artifact the paper's other numbers are computed from.

Usage:
    python experiments/scripts/fig_candidate_counts.py
"""

import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
SIM_DATA = PROJECT / "data/simulation"
REAL_DATA = PROJECT / "data/real_vehicle/hainan_06/processed"
OUT_DIR = (PROJECT / "docs/calib/figs")
FMM_BIN = PROJECT / "build/fmm"
DPI, C_CaMM, C_HMM = 300, "#2166ac", "#b2182b"

plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
                     "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight"})

# The baseline the paper reports: r = 300 m, sigma = 100 m (0.001 deg), rt = 0.1,
# native Viterbi score. Same numbers as build_fmm_xml() in exp3_full_matching.py.
FMM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network><file>input/map/hainan/edges.shp</file><id>key</id><source>u</source><target>v</target></network>
    <ubodt><file>input/map/hainan_ubodt_indexed.bin</file></ubodt>
    <gps><file>{gps}</file><id>id</id><geom>geom</geom><timestamp>timestamp</timestamp></gps>
  </input>
  <output><file>{out}</file>
    <fields><seq/><timestamp/><cpath/><trustworthiness/><candidates/></fields></output>
  <parameters>
    <k>16</k><r>0.0026949335</r><gps_error>0.001</gps_error>
    <reverse_tolerance>0.1</reverse_tolerance>
    <tw_use_viterbi>true</tw_use_viterbi>
  </parameters>
  <other><log_level>2</log_level><use_omp>true</use_omp><step>500</step></other>
</config>"""


def count_candidates(path):
    """Per-epoch candidate counts from a result CSV's `candidates` column.

    The column holds a parenthesised list of (x, y, ep) triples, one per
    candidate. Counting the separators and adding one -- which is what this
    used to do -- reports an epoch with NO candidate, written as an empty
    "()", as having one, so every epoch with no candidate inflates the mean.
    Count opening parens followed by a number instead, which gives 0 for "()".
    """
    counts = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            cand = row.get("candidates", "").strip()
            if cand:
                counts.append(len(re.findall(r"\([-0-9.]", cand)))
    return np.array(counts) if counts else np.array([0])


def measure_hmm(data_dir):
    """Mean candidates per epoch for the HMM baseline, measured and cached."""
    cache = data_dir / "fmm_candidates.csv"
    if cache.exists():
        with open(cache, newline="") as f:
            return np.array([int(r["candidates"]) for r in csv.DictReader(f)])

    traj = data_dir / "fmm_trajectory.csv"
    if not traj.exists() or not FMM_BIN.exists():
        return None
    out = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
    cfg = Path(tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w").name)
    cfg.write_text(FMM_XML.format(gps=traj.resolve(), out=out))
    try:
        r = subprocess.run([str(FMM_BIN), str(cfg)], cwd=PROJECT,
                           capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            print(f"  HMM measurement failed in {data_dir.name}: {r.stderr.strip()[-120:]}")
            return None
        vals = count_candidates(out)
    finally:
        cfg.unlink(missing_ok=True)
        out.unlink(missing_ok=True)

    with open(cache, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidates"])
        w.writerows([[int(v)] for v in vals])
    return vals


def main() -> int:
    sigma_dirs = sorted(SIM_DATA.glob("sigma_*/no_occlusion/no_fault"),
                        key=lambda d: int(d.parent.parent.name.replace("sigma_", "")))
    labels, cmm_means, hmm_means = [], [], []
    for d in sigma_dirs:
        cmm_file = d / "cmm_result.csv"
        if not cmm_file.exists():
            continue
        sv = int(d.parent.parent.name.replace("sigma_", ""))
        c = count_candidates(cmm_file)
        h = measure_hmm(d)
        if h is None:
            continue
        labels.append(f"$\\sigma_\\rho$={sv}")
        cmm_means.append(float(np.mean(c)))
        hmm_means.append(float(np.mean(h)))
        print(f"  sigma={sv:<3} CaMM {cmm_means[-1]:.2f}   HMM {hmm_means[-1]:.2f}")

    if not labels:
        print("no families measured", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(labels))
    ax.bar(x - 0.175, cmm_means, 0.35, color=C_CaMM, label="CaMM (HPL-adaptive)",
           edgecolor="white", lw=0.5)
    ax.bar(x + 0.175, hmm_means, 0.35, color=C_HMM, label="HMM (fixed $r$ = 300 m, $k$ = 16)",
           edgecolor="white", lw=0.5)
    for i, (a, b) in enumerate(zip(cmm_means, hmm_means)):
        ax.text(i - 0.175, a + 0.08, f"{a:.1f}", ha="center", fontsize=7.5,
                fontweight="bold", color=C_CaMM)
        ax.text(i + 0.175, b + 0.08, f"{b:.1f}", ha="center", fontsize=7.5,
                fontweight="bold", color=C_HMM)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Pseudorange noise $\\sigma_\\rho$ (m)")
    ax.set_ylabel("Mean candidates per epoch")
    ax.set_title("Candidate Count: CaMM (HPL-adaptive) vs HMM (fixed-radius)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(cmm_means + hmm_means) * 1.35)
    fig.tight_layout()
    out = OUT_DIR / "candidate_count_simulation.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
