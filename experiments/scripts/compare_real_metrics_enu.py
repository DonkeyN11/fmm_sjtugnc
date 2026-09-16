#!/usr/bin/env python3
"""Real-data CaMM metrics: legacy (ECEF x/y) PL vs ENU-corrected PL.

The baseline column uses cmm_result_fixed_en.csv, produced with the legacy
protection levels (cov[:2,:2] taken from the ECEF x-y block). The candidate
column uses cmm_result_enu.csv, produced with the ENU East-North block — see
compute_raim_pl.py and compare_raim_pl_enu.py for how those PL values differ.

The metric definitions are copied verbatim from verify_paper_numbers.py so the
two columns are directly comparable:
  * ground truth from processed/aligned.csv, keyed by (id, timestamp)
  * reverse edge map applied via is_edge_match()
  * epochs with gt_edge in {0, -1, ''} are dropped
  * ECE over 10 equal-width confidence bins
  * ROC AUC by the trapezoidal rule on the TPR/FPR curve

Trajectory 12 is excluded explicitly.

Usage:
    python experiments/scripts/compare_real_metrics_enu.py
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
PROC = BASE / "data/real_vehicle/hainan_06/processed"
REV_MAP = json.load(open(BASE / "experiments/config/reverse_edge_map.json"))
OUT = BASE / "experiments/output/exp6_real_enu"

EXCLUDE_TRAJ = {"12"}
TW_SENTINEL = -999.0  # writer's trustworthiness value on FAILED rows

BASELINE = PROC / "cmm_result_fixed_en.csv"
CANDIDATE = PROC / "cmm_result_enu.csv"
FMM = PROC / "fmm_result.csv"

# Extra ENU variants at other protection_level_multiplier settings, used to tell
# "the covariance frame was wrong" apart from "the search radius now needs a
# larger multiplier". Missing files are skipped.
EXTRA_VARIANTS = [("enu_m1", "cmm_result_enu_mul1.csv"),
                  ("enu_m5", "cmm_result_enu_mul5.csv"),
                  ("enu_m10", "cmm_result_enu_mul10.csv")]


# ── metric helpers (verbatim from verify_paper_numbers.py) ──────────────────
def is_edge_match(m, g):
    m = m.strip(); g = g.strip()
    if m == g:
        return True
    if REV_MAP.get(m, '') == g:
        return True
    if REV_MAP.get(g, '') == m:
        return True
    return False


def haversine_m(lon1, lat1, lon2, lat2):
    mlat = math.radians((lat1 + lat2) / 2)
    return math.hypot((lon1 - lon2) * 111320 * math.cos(mlat), (lat1 - lat2) * 111320)


def ece(confs, corrects, n=10):
    e = 0.0
    N = len(confs)
    for i in range(n):
        lo, hi = i / n, (i + 1) / n
        mask = (confs >= lo) & (confs < hi) if i < n - 1 else (confs >= lo) & (confs <= hi)
        nb = mask.sum()
        if nb == 0:
            continue
        e += nb / N * abs(confs[mask].mean() - corrects[mask].mean())
    return e


def auc(labels, scores):
    order = np.argsort(scores)[::-1]
    ls = labels[order]
    n_pos = ls.sum()
    n_neg = len(ls) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, None, None
    tpr = np.concatenate([[0], np.cumsum(ls) / n_pos])
    fpr = np.concatenate([[0], np.cumsum(1 - ls) / n_neg])
    return float(np.trapz(tpr, fpr)), fpr, tpr


def parse_point(wkt):
    import re
    m = re.match(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', str(wkt))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


# ── data loading ────────────────────────────────────────────────────────────
def load_gt():
    """(id, ts) -> (gt_edge, gt_lon, gt_lat); traj 12 and no-road epochs dropped."""
    gt = {}
    with open(PROC / "aligned.csv", newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            tid = r['id'].strip()
            if tid in EXCLUDE_TRAJ:
                continue
            gt_e = r['gt_edge'].strip()
            if gt_e in ('0', '-1', ''):
                continue
            try:
                gx, gy = float(r['gt_x']), float(r['gt_y'])
            except (ValueError, KeyError):
                continue
            ts = r['timestamp'].strip().rstrip('0').rstrip('.')
            gt[(tid, ts)] = (gt_e, gx, gy)
    return gt


def load_mr(path):
    """(id, ts) -> (cpath, tw, pgeom, status).

    Rows written for a FAILED match carry trustworthiness = -999, the writer's
    sentinel for "no score". Map it to 0.0 so that a failed epoch enters the
    calibration as a zero-confidence prediction, which is what it is.
    """
    d = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f, delimiter=';'):
            tid = r['id'].strip()
            if tid in EXCLUDE_TRAJ:
                continue
            ts = r['timestamp'].strip().rstrip('0').rstrip('.')
            try:
                tw = float(r['trustworthiness'])
            except (ValueError, KeyError):
                tw = float('nan')
            if tw == TW_SENTINEL or math.isnan(tw):
                tw = 0.0
            d[(tid, ts)] = (r['cpath'].strip(), tw, r.get('pgeom', ''), r.get('status', ''))
    return d


def evaluate(mr, gt):
    per = defaultdict(lambda: {"n": 0, "ok": 0, "tw_c": [], "tw_w": [], "perr": []})
    labels, scores, errs = [], [], []
    matched_flags = []
    n_missing = 0
    n_failed = 0
    for key, (gt_e, gx, gy) in gt.items():
        tid = key[0]
        entry = mr.get(key)
        if entry is None:
            n_missing += 1
            continue
        cpath, tw, pgeom, status = entry
        is_matched = status in ("SUCCESS", "PARTIAL")
        if not is_matched:
            n_failed += 1
        correct = is_edge_match(cpath, gt_e) if cpath else False
        labels.append(1.0 if correct else 0.0)
        scores.append(tw)
        matched_flags.append(is_matched)
        per[tid]["n"] += 1
        per[tid]["ok"] += int(correct)
        (per[tid]["tw_c"] if correct else per[tid]["tw_w"]).append(tw)
        px, py = parse_point(pgeom)
        if px is not None:
            e = haversine_m(px, py, gx, gy)
            per[tid]["perr"].append(e)
            errs.append(e)
    labels = np.array(labels); scores = np.array(scores); errs = np.array(errs)
    a, fpr, tpr = auc(labels, scores)

    # AUC restricted to epochs that actually produced a match. Failed epochs get
    # confidence 0 and are (almost always) wrong, which inflates the AUC by pure
    # failure detection; this isolates the discrimination among matched epochs.
    matched_mask = np.array(matched_flags, dtype=bool)
    if matched_mask.sum() > 0 and len(set(labels[matched_mask])) > 1:
        auc_matched = auc(labels[matched_mask], scores[matched_mask])[0]
    else:
        auc_matched = float('nan')

    return {
        "per_traj": per,
        "n": len(labels),
        "n_missing": n_missing,
        "n_failed": n_failed,
        "auc_matched": auc_matched,
        "accuracy": float(labels.mean()) if len(labels) else float('nan'),
        "ece": ece(scores, labels),
        "auc": a,
        "fpr": fpr, "tpr": tpr,
        "tw_corr": float(np.mean([t for s in per.values() for t in s["tw_c"]])) if per else float('nan'),
        "tw_wrong": float(np.mean([t for s in per.values() for t in s["tw_w"]])) if per else float('nan'),
        "perr_mean": float(errs.mean()) if len(errs) else float('nan'),
        "perr_p95": float(np.percentile(errs, 95)) if len(errs) else float('nan'),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    print(f"Ground-truth epochs (traj 12 excluded, no-road dropped): {len(gt)}")

    variants = [("legacy", BASELINE), ("enu", CANDIDATE)]
    variants += [(nm, PROC / fn) for nm, fn in EXTRA_VARIANTS]
    variants += [("fmm", FMM)]

    results = {}
    for name, path in variants:
        if not path.exists():
            print(f"  [skip] {name}: {path.name} not found")
            continue
        mr = load_mr(path)
        results[name] = evaluate(mr, gt)
        print(f"  loaded {name:8s} from {path.name}: {len(mr)} epochs, "
              f"{results[name]['n_missing']} without a result row")

    order = [n for n, _ in variants if n in results]
    rows = []
    for name in order:
        m = results[name]
        rows.append((name, m["n"], m["accuracy"] * 100, m["ece"], m["auc"],
                     m.get("auc_matched", float('nan')),
                     m["tw_corr"], m["tw_wrong"], m["perr_mean"], m["perr_p95"],
                     m.get("n_failed", 0)))

    print()
    print("=" * 118)
    print("Real-vehicle metrics, traj 12 excluded")
    print("=" * 118)
    print(f"{'variant':>9} {'epochs':>7} {'failed':>7} {'acc %':>8} {'ECE':>8} "
          f"{'AUC':>7} {'AUC*':>7} {'TW corr':>8} {'TW wrong':>9} {'pos err':>8} {'P95':>7}")
    print("-" * 118)
    for name, n, acc, e, a, am, tc, tw_, pm, p95, nf in rows:
        print(f"{name:>9} {n:>7} {nf:>7} {acc:>7.2f}% {e:>8.4f} {a:>7.3f} {am:>7.3f} "
              f"{tc:>8.4f} {tw_:>9.4f} {pm:>8.2f} {p95:>7.2f}")
    print("  AUC* = AUC restricted to epochs that produced a match (failed epochs removed)")

    print()
    print("Per-trajectory segment accuracy (%):")
    print(f"{'traj':>6} " + " ".join(f"{n:>10}" for n, *_ in rows))
    tids = sorted({t for n in results for t in results[n]["per_traj"]}, key=lambda x: int(x))
    for tid in tids:
        cells = []
        for name in order:
            s = results[name]["per_traj"].get(tid)
            cells.append(f"{s['ok'] / s['n'] * 100:>9.2f}%" if s and s["n"] else f"{'-':>10}")
        print(f"{tid:>6} " + " ".join(cells))

    if "legacy" in results and "enu" in results:
        lo, ne = results["legacy"], results["enu"]
        print()
        print("Delta (ENU - legacy):")
        print(f"  segment accuracy : {(ne['accuracy'] - lo['accuracy']) * 100:+.2f} pp")
        print(f"  ECE              : {ne['ece'] - lo['ece']:+.4f}")
        print(f"  AUC              : {ne['auc'] - lo['auc']:+.3f}")

    # persist (drop the bulky ROC arrays into their own file)
    summary = {k: {kk: vv for kk, vv in v.items()
                   if kk not in ("per_traj", "fpr", "tpr")} for k, v in results.items()}
    for k, v in results.items():
        summary[k]["per_traj"] = {t: {"n": s["n"], "ok": s["ok"],
                                      "acc": s["ok"] / s["n"] if s["n"] else None}
                                  for t, s in v["per_traj"].items()}
    with open(OUT / "metrics_enu_vs_legacy.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT / "roc_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "fpr", "tpr"])
        for k, v in results.items():
            if v["fpr"] is None:
                continue
            for x, y in zip(v["fpr"], v["tpr"]):
                w.writerow([k, f"{x:.8f}", f"{y:.8f}"])
    print(f"\nWrote {OUT}/metrics_enu_vs_legacy.json and {OUT}/roc_curves.csv")


if __name__ == "__main__":
    main()
