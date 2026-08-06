#!/usr/bin/env python3
"""Reliability diagram: trustworthiness calibration against segment-level ground truth.
Reads aligned.csv (CaMM + HMM results with ground truth edge IDs).
Outputs: exp1_reliability.json (consumed by gen_figures.py).
"""

import csv, json, math, sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ALIGNED = PROJECT / "data/real_vehicle/processed/aligned.csv"
REV_MAP = PROJECT / "experiments/config/reverse_edge_map.json"
OUT = PROJECT / "data/real_vehicle/mr/exp1_reliability.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_aligned(path):
    """Load aligned.csv, return CaMM and HMM rows with segment-level correctness."""
    REV = {}
    if REV_MAP.exists():
        with open(REV_MAP) as f:
            REV = json.load(f)
        REV = {str(k): str(v) for k, v in REV.items()}

    def em(m, t):
        return str(m) == str(t) or REV.get(str(m)) == str(t)

    tmm_rows, hmm_rows = [], []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            gt = row.get("gt_edge", "").strip()
            if gt in ("", "0", "-1"):
                skipped += 1
                continue
            # CaMM
            tw_s = row.get("cmm_tw", "0").strip()
            tw = float(tw_s) if tw_s else 0.0
            cp = row.get("cmm_cpath", "").strip()
            if cp:
                tmm_rows.append({"tw": tw, "correct": em(cp, gt)})
            # HMM
            tw_s = row.get("fmm_tw", "0").strip()
            tw = float(tw_s) if tw_s else 0.0
            cp = row.get("fmm_cpath", "").strip()
            if cp:
                hmm_rows.append({"tw": tw, "correct": em(cp, gt)})
    return tmm_rows, hmm_rows, skipped


def compute_ece(confs, labels, n_bins=10):
    """Compute ECE, MCE, and per-bin stats."""
    N = len(confs)
    per_bin = []
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (confs >= lo) & (confs < hi) if i < n_bins - 1 else (confs >= lo) & (confs <= hi)
        idx = np.where(mask)[0]
        nb = len(idx)
        if nb == 0:
            per_bin.append({"bin_center": round((lo + hi) / 2, 2),
                            "mean_conf": float("nan"), "accuracy": float("nan"),
                            "count": 0, "lo": lo, "hi": hi})
            continue
        mc = float(np.mean(confs[idx]))
        acc = float(np.mean(labels[idx]))
        gap = abs(mc - acc)
        ece += nb / N * gap
        mce = max(mce, gap)
        per_bin.append({"bin_center": round((lo + hi) / 2, 2),
                        "mean_conf": round(mc, 4), "accuracy": round(acc, 4),
                        "count": nb, "lo": lo, "hi": hi})
    return {"ece": round(ece, 4), "mce": round(mce, 4), "n_total": N, "per_bin": per_bin}


def compute_brier_logloss(confs, labels):
    brier = sum((c - l) ** 2 for c, l in zip(confs, labels)) / len(confs)
    eps = 1e-15
    logloss = -sum(l * math.log(max(c, eps)) + (1 - l) * math.log(max(1 - c, eps))
                   for c, l in zip(confs, labels)) / len(confs)
    return round(brier, 6), round(logloss, 6)


def main():
    print("Loading aligned.csv (segment-level ground truth)...")
    tmm, hmm, skipped = load_aligned(ALIGNED)

    tmm_confs = np.array([r["tw"] for r in tmm])
    tmm_labels = np.array([1.0 if r["correct"] else 0.0 for r in tmm])
    hmm_confs = np.array([r["tw"] for r in hmm])
    hmm_labels = np.array([1.0 if r["correct"] else 0.0 for r in hmm])

    tmm_acc = float(np.mean(tmm_labels))
    hmm_acc = float(np.mean(hmm_labels))

    print(f"  CaMM: {len(tmm)} epochs, TW ∈ [{tmm_confs.min():.4f}, {tmm_confs.max():.4f}], "
          f"seg accuracy={tmm_acc*100:.1f}%")
    print(f"  HMM: {len(hmm)} epochs, TW ∈ [{hmm_confs.min():.4f}, {hmm_confs.max():.4f}], "
          f"seg accuracy={hmm_acc*100:.1f}%")
    print(f"  Skipped (no GT): {skipped}")

    tmm_cal = compute_ece(tmm_confs, tmm_labels)
    hmm_cal = compute_ece(hmm_confs, hmm_labels)
    brier_t, ll_t = compute_brier_logloss(tmm_confs, tmm_labels)
    brier_h, ll_h = compute_brier_logloss(hmm_confs, hmm_labels)

    # ── Bin table ──
    print(f"\n  {'Bin':>4s}  {'Range':>8s}  {'CaMM n':>7s}  {'CaMM TW':>7s}  {'CaMM acc':>8s}  "
          f"{'HMM n':>7s}  {'HMM TW':>7s}  {'HMM acc':>8s}")
    print("  " + "-" * 70)
    for i in range(10):
        tb = tmm_cal["per_bin"][i]
        hb = hmm_cal["per_bin"][i]
        def s(v, w=7):
            return f"{v:>{w}.4f}" if not (isinstance(v, float) and math.isnan(v)) else " " * (w - 3) + "nan"
        print(f"  {i:4d}  {tb['lo']:.1f}-{tb['hi']:.1f}  {tb['count']:7d}  {s(tb['mean_conf']):>7s}  "
              f"{s(tb['accuracy'], 8):>8s}  {hb['count']:7d}  {s(hb['mean_conf']):>7s}  "
              f"{s(hb['accuracy'], 8):>8s}")

    # ── Summary ──
    n_corr_t = sum(tmm_labels)
    n_corr_h = sum(hmm_labels)
    print(f"\n  CaMM: {int(n_corr_t)}/{len(tmm)} correct ({tmm_acc*100:.1f}%) | "
          f"ECE={tmm_cal['ece']:.4f}  MCE={tmm_cal['mce']:.4f}  "
          f"Brier={brier_t:.4f}  LogLoss={ll_t:.4f}")
    print(f"  HMM: {int(n_corr_h)}/{len(hmm)} correct ({hmm_acc*100:.1f}%) | "
          f"ECE={hmm_cal['ece']:.4f}  MCE={hmm_cal['mce']:.4f}  "
          f"Brier={brier_h:.4f}  LogLoss={ll_h:.4f}")

    # ── Export JSON (compatible with gen_figures.py) ──
    output = {
        "experiment": "exp1_reliability_diagram",
        "metric": "TW (trustworthiness / filtering posterior)",
        "label": "segment-level correctness (matched edge == GT edge)",
        "methods": {"cmm": "CaMM (anisotropic Mahalanobis)", "fmm": "HMM (isotropic Euclidean)"},
        "per_threshold": {
            "5": {  # gen_figures.py uses key="5"
                "cmm": {
                    "per_bin": tmm_cal["per_bin"],
                    "ece": tmm_cal["ece"], "mce": tmm_cal["mce"],
                    "brier": brier_t, "logloss": ll_t,
                    "n_correct": int(n_corr_t), "n_total": len(tmm),
                },
                "fmm": {
                    "per_bin": hmm_cal["per_bin"],
                    "ece": hmm_cal["ece"], "mce": hmm_cal["mce"],
                    "brier": brier_h, "logloss": ll_h,
                    "n_correct": int(n_corr_h), "n_total": len(hmm),
                },
            }
        }
    }

    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
