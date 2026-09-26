#!/usr/bin/env python3
"""ECE binning robustness for the calibration audit.

WHY THIS EXISTS
---------------
The audit in docs/calib/calib.tex reports an expected calibration error computed
on ten equal-width bins. Two things are known about that choice and neither was
stated in the paper:

  * Equal-width bins let the population of a bin vary freely, and the estimator's
    bias is largest for correctly calibrated forecasts, depending on bin count and
    sample size (Roelofs et al., AISTATS 2022; Futami & Fujisawa, NeurIPS 2024;
    Kumar et al., NeurIPS 2019). Equal-mass bins are less biased.

  * The three-term Murphy decomposition that the paper prints alongside the ECE is
    exact only when one stratifies on every distinct issued probability. Once the
    probabilities are binned, Stephenson, Coelho & Jolliffe (Weather and
    Forecasting 23(4):752-757, 2008) show two further within-bin components are
    required. The refinement the paper omits is a consequence of the same binning
    choice.

So the audit is repeated here under equal-mass binning and under a range of bin
counts. The pass criterion is not that the ECE values agree -- they are different
estimators of the same quantity and need not -- but that the ORDERING of the five
score arms is stable.

USAGE
    python experiments/scripts/ece_binning_robustness.py [--n 10 15 20]

Ground truth and result files are the ones named in verify_paper_numbers.py:
    data/real_vehicle/hainan_06/processed/aligned_0729.csv
    data/archive_20260922/real_final/{cmm_final,fmm_final}.csv
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path("/home/ncz/fmm_sjtugnc")
sys.path.insert(0, str(BASE / "experiments" / "scripts"))
import verify_paper_numbers as v          # noqa: E402
import inventory_real_metrics as I        # noqa: E402

GT = BASE / "data/real_vehicle/hainan_06/processed/aligned_0729.csv"
CMM = BASE / "data/archive_20260922/real_final/cmm_final.csv"
FMM = BASE / "data/archive_20260922/real_final/fmm_final.csv"


# ----------------------------------------------------------------------------- estimators
def ece_equal_width(conf, corr, n=10):
    """The paper's estimator. Identical arithmetic to verify_paper_numbers.ece()."""
    e, N = 0.0, len(conf)
    for i in range(n):
        lo, hi = i / n, (i + 1) / n
        mask = (conf >= lo) & (conf < hi) if i < n - 1 else (conf >= lo) & (conf <= hi)
        nb = int(mask.sum())
        if nb == 0:
            continue
        e += nb / N * abs(conf[mask].mean() - corr[mask].mean())
    return e


def ece_equal_mass(conf, corr, n=10):
    """Quantile-binned ECE, plus the number of non-empty bins.

    The second return value is not decoration. A score with a large point mass has
    degenerate quantiles: np.quantile collapses onto the mass value, searchsorted
    routes every tied score into one bin, and the remaining bins come out empty.
    Equal-mass binning is then not an estimator of anything, and a caller that
    reports its value without the non-empty count is reporting an artefact. CaMM's
    score puts 88 % of its epochs at exactly 1.0, so this happens to it; FMM's does
    not, so the scheme is usable on the baseline and not on the proposal.
    """
    N = len(conf)
    edges = np.quantile(conf, np.linspace(0, 1, n + 1))
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n - 1)
    e, nonempty = 0.0, 0
    for b in range(n):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            continue
        nonempty += 1
        e += nb / N * abs(conf[mask].mean() - corr[mask].mean())
    return e, nonempty


def max_point_mass(conf):
    """Largest fraction of epochs sharing one score value."""
    _, cnt = np.unique(conf, return_counts=True)
    return float(cnt.max() / len(conf))


def mce(conf, corr, n=10, mode="width"):
    if mode == "mass":
        edges = np.quantile(conf, np.linspace(0, 1, n + 1))
        idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n - 1)
    else:
        idx = np.clip(np.digitize(conf, np.linspace(0, 1, n + 1)[1:-1]), 0, n - 1)
    gaps = [abs(conf[idx == b].mean() - corr[idx == b].mean())
            for b in range(n) if (idx == b).sum()]
    return max(gaps) if gaps else float("nan")


# ----------------------------------------------------------------------------- loading
def load_run(path):
    """Same key/value extraction as data/archive_20260922/real_final/score.py."""
    raw = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            if r["id"].strip() in v.EXCLUDE_TRAJ:
                continue
            tws = r["trustworthiness"].strip()
            tw = None if tws in ("", "-999", "-999.0") or float(tws) <= -900 else float(tws)
            x, y = v.parse_point(r["pgeom"])
            seq = int(r["seq"]) if r.get("seq", "").strip() else None
            raw.append((r["id"].strip(), v.norm_ts(r["timestamp"]), seq,
                        r["cpath"].strip(), tw, (x, y) if x is not None else None))
    for label, idx in (("timestamp", 1), ("seq", 2)):
        if any(row[idx] is None for row in raw):
            continue
        if len({(row[0], row[idx]) for row in raw}) == len(raw):
            return {(r[0], r[idx]): r[3:] for r in raw}, label
    raise SystemExit(f"{path}: no unique key")


def pairs(run, gt_by_ts, gt_by_seq, key):
    """Join a run onto the ground truth on whichever key the run is injective on.

    The two result files need not agree: score.py picks the GT index per run for
    the same reason, and hard-wiring one index silently produced an empty sample
    for the FMM arm on the first attempt.
    """
    gt = gt_by_ts if key == "timestamp" else gt_by_seq
    out = []
    for k, (gt_e, _) in gt.items():
        hit = run.get(k)
        if hit is None:
            continue
        cpath, tw, _ = hit
        if tw is None:
            continue
        out.append((float(v.is_edge_match(cpath, gt_e)), float(tw)))
    return np.array([p[1] for p in out]), np.array([p[0] for p in out])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[10, 15, 20])
    args = ap.parse_args()

    by_ts, by_seq = I.load_gt(GT)
    print(f"ground truth {GT.name}: {len(by_ts)} epochs (the audit population)")

    arms = []
    for name, path in (("S1 FMM", FMM), ("S3/S4 CaMM", CMM)):
        run, key = load_run(path)
        conf, corr = pairs(run, by_ts, by_seq, key)
        print(f"  {name:<12} rows={len(run):>6}  key={key:<9} joined n={len(conf)}"
              f"  acc={100*corr.mean():.4f}")
        if len(conf) == 0:
            raise SystemExit(f"{name}: joined no epochs -- refusing to report")
        arms.append(dict(name=name, conf=conf, corr=corr, base=corr.mean()))
    # S0: a constant at the base rate. Carried as an arm so the degeneracy the
    # paper argues for is visible in the same table rather than asserted in prose.
    base = arms[-1]["corr"].mean()
    arms.insert(0, dict(name="S0 base rate", conf=np.full_like(arms[-1]["conf"], base),
                        corr=arms[-1]["corr"], base=base))

    for n in args.n:
        print(f"\n=== B = {n} bins")
        print(f"{'arm':>14}{'n':>7}{'acc%':>9}"
              f"{'ECE_width':>11}{'MCE_width':>11}{'ECE_mass':>10}{'mass_nb':>9}"
              f"{'Brier':>9}{'BSS':>8}")
        for a in arms:
            conf, corr = a["conf"], a["corr"]
            bs = float(np.mean((conf - corr) ** 2))
            bs_ref = float(np.mean((a["base"] - corr) ** 2))
            mass_ece, nb = ece_equal_mass(conf, corr, n)
            shown = f"{mass_ece:>10.4f}" if nb == n else f"{'--':>10}"
            print(f"{a['name']:>14}{len(conf):>7}{100*corr.mean():>9.4f}"
                  f"{ece_equal_width(conf, corr, n):>11.4f}"
                  f"{mce(conf, corr, n, 'width'):>11.4f}"
                  f"{shown}{nb:>4}/{n:<4}"
                  f"{bs:>9.4f}{1 - bs / bs_ref:>8.4f}")

    print("\n=== score degeneracy (why equal-mass may be unavailable)")
    for a in arms:
        print(f"  {a['name']:>14}: {len(np.unique(a['conf'])):>6} distinct values, "
              f"max point mass {100*max_point_mass(a['conf']):>5.1f}%")

    # declared pass criterion: the ORDERING of the arms must not depend on the binning
    print("\n=== ordering stability (the declared pass criterion)")
    for n in args.n:
        order = sorted(arms, key=lambda a: ece_equal_width(a["conf"], a["corr"], n))
        print(f"  B={n:>2} equal-width: " + "  <  ".join(a["name"] for a in order))
    usable = all(ece_equal_mass(a["conf"], a["corr"], args.n[0])[1] == args.n[0] for a in arms)
    print(f"  equal-mass applied to all arms: {'yes' if usable else 'NO -- degenerate on at least one arm'}")


if __name__ == "__main__":
    main()
