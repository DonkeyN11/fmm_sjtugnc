#!/usr/bin/env python3
"""WP1a steps 2-5: online accuracy, the six-arm audit table, uncertainty, the tail.

The join and the labels are NOT re-derived here: `real_final_tables.load_gt()`
carries the repo's canonical rule (the two matchers need different join keys --
CaMM on its timestamp, FMM on seq resolved through the input table) and
`verify_paper_numbers.is_edge_match` carries the label. `real_final_tables.evaluate`
is the arithmetic the manuscript's table was produced with; this script imports
`v.ece`, `rft.mce`, `v.auc`, `v.log_score` and computes only Brier/BSS/separation
inline, then CHECKS that its own batch rows equal `rft.evaluate`'s before writing
anything (see `validate_against_evaluate`).

ARMS: three per model. `online_delta` and `online_alpha` come from the `online`
column of the fresh run (i_on = argmax_j delta_t(j), the causal decision);
`batch_alphabeta` is `trustworthiness_smoothed` of the archived run, i.e. the
paper's own reported arm.

OUTPUT (data/archive_20261019/online/)
    audit_table.csv, online_accuracy.json, intervals.csv, tail.json

Usage:
    python3 experiments/scripts/online_audit.py
"""
import csv
import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments/scripts"))
import verify_paper_numbers as v          # noqa: E402
import real_final_tables as rft           # noqa: E402

SEED = 20261019
NBOOT = 2000
AR = BASE / "data/archive_20261019/online"
SCALAR = BASE / "data/archive_20260929/scalar"

# model, arm, path, score source, key, tuple index (None -> plain column)
ARMS = [
    ("cmm", "online_delta",     AR / "raw_cmm.csv", "online", None, 1),
    ("cmm", "online_alpha",     AR / "raw_cmm.csv", "online", None, 2),
    ("cmm", "batch_alphabeta",  SCALAR / "cmm.csv", "trustworthiness_smoothed", None, None),
    ("fmm", "online_delta",     AR / "raw_fmm.csv", "online", "seq", 1),
    ("fmm", "online_alpha",     AR / "raw_fmm.csv", "online", "seq", 2),
    ("fmm", "batch_alphabeta",  SCALAR / "fmm.csv", "trustworthiness_smoothed", "seq", None),
]


def pairs(path, col, key, idx):
    """(traj, label, score) arrays on the evaluable population, canonical join."""
    run = rft.load_run(path, ts_by_seq if key == "seq" else None)
    traj, labels, scores = [], [], []
    for k, (gt_edge, _gx, _gy) in gt.items():
        r = run.get(k)
        if r is None:
            continue
        correct = v.is_edge_match(r["cpath"].strip(), gt_edge)
        if idx is None:
            try:
                s = float(r[col])
            except (ValueError, KeyError):
                s = None
            if s is None or s <= -900.0:
                continue
        else:
            cell = (r.get(col) or "").strip()
            if not (cell.startswith("(") and cell.endswith(")")):
                continue
            parts = cell[1:-1].split(",")
            s = float(parts[idx])
        traj.append(k[0])
        labels.append(1.0 if correct else 0.0)
        scores.append(s)
    return np.array(traj), np.array(labels), np.array(scores)


def metrics(labels, scores):
    if len(scores) == 0:
        return {}
    p = labels.mean()
    brier = float(((scores - labels) ** 2).mean())
    return dict(
        n=int(len(labels)),
        acc_pct=100.0 * float(labels.mean()),
        ece=v.ece(scores, labels),
        mce=rft.mce(scores, labels),
        brier=brier,
        bss=1.0 - brier / (p * (1.0 - p)),
        log=v.log_score(labels, scores),
        auc=v.auc(labels, scores),
        sep=float(scores[labels == 1].mean() - scores[labels == 0].mean()),
    )


def validate_against_evaluate():
    """My six-arm arithmetic must equal the manuscript's own table code."""
    bad = []
    for col, key, path in (("trustworthiness_smoothed", None, SCALAR / "cmm.csv"),
                           ("trustworthiness_smoothed", ts_by_seq, SCALAR / "fmm.csv")):
        ref = rft.evaluate(path, gt, col, key)
        _t, lab, sc = pairs(path, col, "seq" if key is not None else None, None)
        mine = metrics(lab, sc)
        for k in ("ece", "mce", "brier", "bss", "auc", "sep"):
            if abs(mine[k] - ref[k]) > 1e-9:
                bad.append((path.name, k, mine[k], ref[k]))
    if bad:
        for b in bad:
            print("  MISMATCH", b)
        sys.exit("FATAL: my metrics disagree with real_final_tables.evaluate")
    print("validate: six-arm arithmetic == real_final_tables.evaluate (ece/mce/brier/bss/auc/sep)")


def boot_marginal(traj, labels, scores, metric, rng):
    uniq = sorted(set(traj))
    groups = {t: np.where(traj == t)[0] for t in uniq}
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        idx = np.concatenate([groups[uniq[i]] for i in pick])
        out[b] = metric(labels[idx], scores[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_paired(traj, lab, sca, lab2, sca2, metric, rng):
    """Same trajectories in every replicate (paired), percentile CI of the difference."""
    uniq = sorted(set(traj))
    groups = {t: np.where(traj == t)[0] for t in uniq}
    out = np.empty(NBOOT)
    for b in range(NBOOT):
        pick = rng.choice(len(uniq), len(uniq), replace=True)
        idx = np.concatenate([groups[uniq[i]] for i in pick])
        out[b] = metric(lab[idx], sca[idx]) - metric(lab2[idx], sca2[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def signflip(traj, lab, sca, lab2, sca2, metric):
    """Exact 2^6 sign-flip test on the per-trajectory metric differences."""
    uniq = sorted(set(traj))
    d = []
    for t in uniq:
        m = traj == t
        d.append(metric(lab[m], sca[m]) - metric(lab2[m], sca2[m]))
    d = np.array(d)
    obs = d.mean()
    signs = np.array(list(product([-1.0, 1.0], repeat=len(uniq))))
    stats = (signs * d).mean(axis=1)
    p = float((np.abs(stats) >= abs(obs) - 1e-15).mean())
    return float(obs), p, len(uniq)


def main():
    global gt, ts_by_seq
    gt, ts_by_seq = rft.load_gt()
    print("evaluable GT epochs:", len(gt))
    validate_against_evaluate()

    data = {}
    rows = []
    for model, arm, path, col, key, idx in ARMS:
        traj, lab, sc = pairs(path, col, key, idx)
        m = metrics(lab, sc)
        data[(model, arm)] = (traj, lab, sc, m)
        for stat in ("n", "acc_pct", "ece", "mce", "brier", "bss", "log", "auc", "sep"):
            rows.append(dict(model=model, arm=arm, statistic=stat,
                             value=("%d" % m[stat]) if stat == "n" else "%.6f" % m[stat],
                             n=m["n"]))
        print("%-4s %-16s n=%-6d acc=%6.2f ece=%.4f mce=%.3f brier=%.4f bss=%+.3f "
              "log=%.4f auc=%.3f sep=%.3f"
              % (model, arm, m["n"], m["acc_pct"], m["ece"], m["mce"], m["brier"],
                 m["bss"], m["log"], m["auc"], m["sep"]))

    with open(AR / "audit_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "arm", "statistic", "value", "n"])
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------ intervals
    rng = np.random.default_rng(SEED)
    ic_rows = []
    for (model, arm), (traj, lab, sc, m) in data.items():
        lo, hi = boot_marginal(traj, lab, sc, lambda a, b: v.ece(b, a), rng)
        ic_rows.append(dict(contrast="%s/%s" % (model, arm), statistic="ece",
                            point="%.6f" % m["ece"], lo="%.6f" % lo, hi="%.6f" % hi,
                            kind="marginal", test="trajectory-clustered bootstrap 95%",
                            n=m["n"]))
        lo, hi = boot_marginal(traj, lab, sc, lambda a, b: v.auc(a, b), rng)
        ic_rows.append(dict(contrast="%s/%s" % (model, arm), statistic="auc",
                            point="%.6f" % m["auc"], lo="%.6f" % lo, hi="%.6f" % hi,
                            kind="marginal", test="trajectory-clustered bootstrap 95%",
                            n=m["n"]))

    contrasts = []
    for model in ("cmm", "fmm"):
        contrasts += [(model, "online_delta", "batch_alphabeta"),
                      (model, "online_alpha", "batch_alphabeta"),
                      (model, "online_delta", "online_alpha")]
    for model, a, b in contrasts:
        ta, la, sa, ma = data[(model, a)]
        tb, lb, sb, mb = data[(model, b)]
        # same population and same order by construction (both keyed off gt), but assert
        assert len(la) == len(lb) and (la == lb).all(), "pairing broken"
        rng2 = np.random.default_rng(SEED)
        for stat, fn in (("ece", lambda y, s: v.ece(s, y)),
                         ("auc", lambda y, s: v.auc(y, s))):
            lo, hi = boot_paired(ta, la, sa, lb, sb, fn, rng2)
            ic_rows.append(dict(contrast="%s - %s (%s)" % (a, b, model), statistic=stat,
                                point="%.6f" % (ma[stat] - mb[stat]),
                                lo="%.6f" % lo, hi="%.6f" % hi, kind="paired",
                                test="trajectory-clustered paired bootstrap 95%",
                                n=ma["n"]))
            obs, p, k = signflip(ta, la, sa, lb, sb, fn)
            ic_rows.append(dict(contrast="%s - %s (%s)" % (a, b, model), statistic=stat,
                                point="%.6f" % obs, lo="", hi="",
                                kind="paired-signflip",
                                test="exact 2^%d sign-flip p=%.4f" % (k, p), n=ma["n"]))
    with open(AR / "intervals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["contrast", "statistic", "point", "lo", "hi",
                                          "kind", "test", "n"])
        w.writeheader()
        w.writerows(ic_rows)

    # --------------------------------------------------------------- online accuracy
    acc = {}
    for model, raw, key in (("cmm", AR / "raw_cmm.csv", None),
                            ("fmm", AR / "raw_fmm.csv", "seq")):
        run = rft.load_run(raw, ts_by_seq if key == "seq" else None)
        per = defaultdict(lambda: dict(n=0, on_ok=0, batch_ok=0, diverged=0))
        for k, (gt_edge, _x, _y) in gt.items():
            r = run.get(k)
            if r is None:
                continue
            cell = (r.get("online") or "").strip()
            if not (cell.startswith("(") and cell.endswith(")")):
                continue
            i_on = int(float(cell[1:-1].split(",")[0]))
            try:
                i_star = int(float((r.get("opath") or "").strip()))
            except ValueError:
                i_star = None
            d = per[k[0]]
            d["n"] += 1
            d["on_ok"] += int(v.is_edge_match(str(i_on), gt_edge))
            d["batch_ok"] += int(v.is_edge_match(r["cpath"].strip(), gt_edge))
            d["diverged"] += int(i_star is not None and i_on != i_star)
        tot = dict(n=0, on_ok=0, batch_ok=0, diverged=0)
        acc[model] = {}
        for t in sorted(per):
            d = per[t]
            acc[model][t] = dict(
                n=d["n"],
                online_acc_pct=100.0 * d["on_ok"] / d["n"],
                batch_acc_pct=100.0 * d["batch_ok"] / d["n"],
                diverged_epochs=d["diverged"],
                diverged_pct=100.0 * d["diverged"] / d["n"])
            for kk in ("n", "on_ok", "batch_ok", "diverged"):
                tot[kk] += d[kk]
        acc[model]["ALL"] = dict(
            n=tot["n"],
            online_acc_pct=100.0 * tot["on_ok"] / tot["n"],
            batch_acc_pct=100.0 * tot["batch_ok"] / tot["n"],
            diverged_epochs=tot["diverged"],
            diverged_pct=100.0 * tot["diverged"] / tot["n"])
        print("%s ALL online_acc=%.2f batch_acc=%.2f diverged=%d (%.2f%%)"
              % (model, acc[model]["ALL"]["online_acc_pct"],
                 acc[model]["ALL"]["batch_acc_pct"],
                 acc[model]["ALL"]["diverged_epochs"],
                 acc[model]["ALL"]["diverged_pct"]))
    with open(AR / "online_accuracy.json", "w") as f:
        json.dump(acc, f, indent=2, sort_keys=True)

    # ---------------------------------------------------------------------- tail
    tail = {}
    for (model, arm), (traj, lab, sc, m) in data.items():
        one = np.isclose(sc, 1.0, atol=0.0, rtol=0.0)
        wrong = one & (lab == 0.0)
        by_traj = defaultdict(int)
        for t in traj[wrong]:
            by_traj[t] += 1
        sub = sc < 1.0
        subm = metrics(lab[sub], sc[sub]) if sub.sum() else {}
        tail["%s/%s" % (model, arm)] = dict(
            n=int(len(sc)),
            n_at_one=int(one.sum()),
            n_at_one_wrong=int(wrong.sum()),
            n_at_one_wrong_by_traj=dict(sorted(by_traj.items())),
            n_not_saturated=int(sub.sum()),
            non_saturated=dict(
                n=subm.get("n", 0),
                ece=subm.get("ece"), auc=subm.get("auc"), bss=subm.get("bss")),
            n_one_minus_eps_falsified=int((sc >= 1.0 - 1e-12).sum()))
    with open(AR / "tail.json", "w") as f:
        json.dump(tail, f, indent=2, sort_keys=True)

    # ------------------------------------------- online arms, own-decision labels
    sc_rows = []
    for model, raw, key in (("cmm", AR / "raw_cmm.csv", None),
                            ("fmm", AR / "raw_fmm.csv", "seq")):
        run = rft.load_run(raw, ts_by_seq if key == "seq" else None)
        for arm, idx in (("online_delta", 1), ("online_alpha", 2)):
            lab2, sc2 = [], []
            for k, (gt_edge, _x, _y) in gt.items():
                r = run.get(k)
                if r is None:
                    continue
                cell = (r.get("online") or "").strip()
                if not (cell.startswith("(") and cell.endswith(")")):
                    continue
                parts = cell[1:-1].split(",")
                i_on = int(float(parts[0]))
                lab2.append(1.0 if v.is_edge_match(str(i_on), gt_edge) else 0.0)
                sc2.append(float(parts[idx]))
            m2 = metrics(np.array(lab2), np.array(sc2))
            for stat in ("n", "acc_pct", "ece", "mce", "brier", "bss", "log", "auc", "sep"):
                sc_rows.append(dict(model=model, arm=arm, statistic=stat,
                                    value=("%d" % m2[stat]) if stat == "n" else "%.6f" % m2[stat],
                                    n=m2["n"]))
            print("%-4s %-16s OWN-LABEL acc=%6.2f ece=%.4f mce=%.3f bss=%+.3f log=%.4f auc=%.3f"
                  % (model, arm, m2["acc_pct"], m2["ece"], m2["mce"], m2["bss"],
                     m2["log"], m2["auc"]))
    with open(AR / "audit_table_selfconsistent.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "arm", "statistic", "value", "n"])
        w.writeheader()
        w.writerows(sc_rows)

    for k in ("cmm/batch_alphabeta", "fmm/batch_alphabeta",
              "cmm/online_delta", "cmm/online_alpha",
              "fmm/online_delta", "fmm/online_alpha"):
        t = tail[k]
        print("%-22s at1=%-5d wrong=%-4d by_traj=%-28s non-sat=%-5d (ece=%.4f auc=%.3f bss=%+.2f)"
              % (k, t["n_at_one"], t["n_at_one_wrong"], t["n_at_one_wrong_by_traj"],
                 t["n_not_saturated"], t["non_saturated"]["ece"] or 0,
                 t["non_saturated"]["auc"] or 0, t["non_saturated"]["bss"] or 0))


if __name__ == "__main__":
    main()

