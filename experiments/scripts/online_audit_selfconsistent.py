#!/usr/bin/env python3
"""WP1a follow-up: the SELF-CONSISTENT audit table, full metrics, intervals, tail.

THE RULING THIS IMPLEMENTS. An online score describes the online decision
i_on_t = argmax_j delta_t(j), so its outcome variable is whether THAT decision is
right, not whether the batch Viterbi path is right. audit_table.csv (batch labels)
is kept as a control table only; this file is the one that may enter the manuscript.

CONSEQUENCE THAT MUST BE READ WITH THE NUMBERS. The three arms no longer share one
label column:
    online delta / online alpha  ->  label = is_edge_match(i_on_t, gt_t)
    batch alpha-beta             ->  label = is_edge_match(cpath_t, gt_t)
Each arm is therefore scored against its own decision, which is the self-consistent
reading. A paired contrast (online - batch) is still taken on the SAME 13,069 epochs
with the SAME trajectory resamples, but it now mixes two effects: a better score and
a different (more often wrong) decision. It is a difference between two forecasting
systems, not an estimator ablation. The one contrast that IS a clean ablation is
online_delta - online_alpha, where both arms share the online label.

The bootstrap and the sign-flip are imported from online_audit.py, not re-implemented,
so the seed (20261019), the 2,000 replicates and the 6-trajectory resampling are
identical to intervals.csv and the two files are directly comparable.

OUTPUT (data/archive_20261019/online/)
    audit_table_selfconsistent_full.csv  3 arms x 2 models x 8 statistics, with a
                                         `labels` column naming the outcome variable
    intervals_selfconsistent.csv         marginal 95% CI per arm, paired 95% CI and
                                         exact 2^6 sign-flip p per contrast
    tail_selfconsistent.json             epochs reported at exactly 1.000 under the
                                         self-consistent label, and how many are wrong

Usage:
    python3 experiments/scripts/online_audit_selfconsistent.py
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments/scripts"))
import verify_paper_numbers as v           # noqa: E402
import real_final_tables as rft            # noqa: E402
import online_audit as oa                  # noqa: E402  (seed / bootstrap / sign-flip)

AR = BASE / "data/archive_20261019/online"
SCALAR = BASE / "data/archive_20260929/scalar"
LABEL_ONLINE = "self-consistent(online decision)"
LABEL_BATCH = "self-consistent(batch decision)"
STATS = ("n", "acc_pct", "ece", "mce", "brier", "bss", "log", "auc", "sep")


def load_model(name, online_path, batch_path, key):
    """One pass over the evaluable epochs; every array shares the same epoch order.

    The population is the INTERSECTION: an epoch enters only if it has both a batch
    score and an online record, so the three arms are paired epoch for epoch.
    """
    gt, ts_by_seq = oa.gt, oa.ts_by_seq
    run_o = rft.load_run(online_path, ts_by_seq if key == "seq" else None)
    run_b = rft.load_run(batch_path, ts_by_seq if key == "seq" else None)
    traj, lab_b, lab_o, s_b, s_d, s_a = [], [], [], [], [], []
    for k, (gt_edge, _x, _y) in gt.items():
        ro, rb = run_o.get(k), run_b.get(k)
        if ro is None or rb is None:
            continue
        try:
            sb = float(rb["trustworthiness_smoothed"])
        except (ValueError, KeyError):
            continue
        if sb <= -900.0:
            continue
        cell = (ro.get("online") or "").strip()
        if not (cell.startswith("(") and cell.endswith(")")):
            continue
        p = cell[1:-1].split(",")
        if len(p) != 6:
            continue
        i_on = int(float(p[0]))
        traj.append(k[0])
        lab_b.append(1.0 if v.is_edge_match(rb["cpath"].strip(), gt_edge) else 0.0)
        lab_o.append(1.0 if v.is_edge_match(str(i_on), gt_edge) else 0.0)
        s_b.append(sb)
        s_d.append(float(p[1]))
        s_a.append(float(p[2]))
    return dict(model=name,
                traj=np.array(traj), lab_b=np.array(lab_b), lab_o=np.array(lab_o),
                batch=np.array(s_b), delta=np.array(s_d), alpha=np.array(s_a))


def arm_metrics(d, which):
    """(labels, scores) for one arm of one model, both already aligned."""
    if which == "batch_alphabeta":
        return d["lab_b"], d["batch"]
    if which == "online_delta":
        return d["lab_o"], d["delta"]
    return d["lab_o"], d["alpha"]


def main():
    oa.gt, oa.ts_by_seq = rft.load_gt()
    print("evaluable GT epochs:", len(oa.gt))

    models = {
        "cmm": load_model("cmm", AR / "raw_cmm.csv", SCALAR / "cmm.csv", "timestamp"),
        "fmm": load_model("fmm", AR / "raw_fmm.csv", SCALAR / "fmm.csv", "seq"),
    }

    ARMS = [("online_delta", LABEL_ONLINE), ("online_alpha", LABEL_ONLINE),
            ("batch_alphabeta", LABEL_BATCH)]

    # ------------------------------------------------------------------ main table
    rows = []
    store = {}
    for model, d in models.items():
        print("\n%s: n=%d  online_acc=%.2f%%  batch_acc=%.2f%%"
              % (model, len(d["traj"]), 100 * d["lab_o"].mean(), 100 * d["lab_b"].mean()))
        for arm, labname in ARMS:
            lab, sc = arm_metrics(d, arm)
            m = oa.metrics(lab, sc)
            store[(model, arm)] = (d["traj"], lab, sc, m)
            for st in STATS:
                rows.append(dict(model=model, arm=arm, labels=labname, statistic=st,
                                 value=("%d" % m[st]) if st == "n" else "%.6f" % m[st],
                                 n=m["n"]))
            print("  %-16s labels=%-34s acc=%6.2f ece=%.4f mce=%.3f brier=%.4f "
                  "bss=%+.3f log=%.4f auc=%.3f sep=%.3f"
                  % (arm, labname, m["acc_pct"], m["ece"], m["mce"], m["brier"],
                     m["bss"], m["log"], m["auc"], m["sep"]))

    with open(AR / "audit_table_selfconsistent_full.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "arm", "labels", "statistic", "value", "n"])
        w.writeheader()
        w.writerows(rows)

    # -------------------------------------------------------------------- intervals
    ic = []
    for (model, arm), (traj, lab, sc, m) in store.items():
        for stat, fn in (("ece", lambda y, s: v.ece(s, y)),
                         ("auc", lambda y, s: v.auc(y, s))):
            rng = np.random.default_rng(oa.SEED)
            lo, hi = oa.boot_marginal(traj, lab, sc, fn, rng)
            ic.append(dict(contrast="%s/%s" % (model, arm), statistic=stat,
                           point="%.6f" % m[stat], lo="%.6f" % lo, hi="%.6f" % hi,
                           kind="marginal", p_signflip="",
                           test="trajectory-clustered bootstrap 95%", n=m["n"]))

    for model in ("cmm", "fmm"):
        for a, b in (("online_delta", "batch_alphabeta"),
                     ("online_alpha", "batch_alphabeta"),
                     ("online_delta", "online_alpha")):
            ta, la, sa, ma = store[(model, a)]
            tb, lb, sb, mb = store[(model, b)]
            assert len(la) == len(lb) and len(sa) == len(sb), "pairing broken"
            for stat, fn in (("ece", lambda y, s: v.ece(s, y)),
                             ("auc", lambda y, s: v.auc(y, s))):
                rng = np.random.default_rng(oa.SEED)
                lo, hi = oa.boot_paired(ta, la, sa, lb, sb, fn, rng)
                ic.append(dict(contrast="%s - %s (%s)" % (a, b, model), statistic=stat,
                               point="%.6f" % (ma[stat] - mb[stat]),
                               lo="%.6f" % lo, hi="%.6f" % hi, kind="paired",
                               p_signflip="",
                               test="trajectory-clustered paired bootstrap 95%", n=ma["n"]))
                obs, p, k = oa.signflip(ta, la, sa, lb, sb, fn)
                ic.append(dict(contrast="%s - %s (%s)" % (a, b, model), statistic=stat,
                               point="%.6f" % obs, lo="", hi="", kind="paired-signflip",
                               p_signflip="%.4f" % p,
                               test="exact 2^%d sign-flip" % k, n=ma["n"]))

    with open(AR / "intervals_selfconsistent.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["contrast", "statistic", "point", "lo", "hi",
                                          "kind", "p_signflip", "test", "n"])
        w.writeheader()
        w.writerows(ic)

    # ------------------------------------------------------------------------- tail
    tail = {}
    for (model, arm), (traj, lab, sc, m) in store.items():
        one = sc == 1.0
        wrong = one & (lab == 0.0)
        by = defaultdict(int)
        for t in traj[wrong]:
            by[t] += 1
        tail["%s/%s" % (model, arm)] = dict(
            labels=LABEL_ONLINE if arm.startswith("online") else LABEL_BATCH,
            n=int(len(sc)), n_at_one=int(one.sum()), n_at_one_wrong=int(wrong.sum()),
            n_at_one_wrong_by_traj=dict(sorted(by.items())),
            n_at_one_correct=int((one & (lab == 1.0)).sum()))
    with open(AR / "tail_selfconsistent.json", "w") as f:
        json.dump(tail, f, indent=2, sort_keys=True)

    print("\ntail (self-consistent labels):")
    for k in sorted(tail):
        t = tail[k]
        print("  %-22s at1=%-6d wrong=%-4d by_traj=%-24s" %
              (k, t["n_at_one"], t["n_at_one_wrong"], t["n_at_one_wrong_by_traj"]))


if __name__ == "__main__":
    main()

