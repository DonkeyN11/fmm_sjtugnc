#!/usr/bin/env python3
"""WP1a step 1: export the ONLINE decision record for both matchers.

WHAT THIS ADDS. The paper's arms read the confidence at the *backtraced* Viterbi
node i*_t, which is allowed to see the whole track. A deployed monitor does not
have that: it must decide at epoch t with z_{1:t} only. The online decision is

    i_on_t = argmax_j delta_t(j),

the Viterbi argmax read at layer t of the forward recursion -- causal by
construction, and generally NOT i*_t. This script writes that node, and the two
per-layer softmaxes read at it, for the proposed matcher (CaMM scalar config) and
the stock-FMM baseline.

The C++ side emits the record as one extra output field `online`; the field
defaults off, so no existing config, archive or number moves (see
online_check_unmodified.py, which diffs the patched binary against the archived
run and is the gate for everything downstream).

OUTPUT (data/archive_20261019/online/)
    cmm_online.xml, fmm_online.xml  the configs actually run, derived from
                                    input/config/paper1_{cmm_scalar,fmm_baseline}.xml
                                    by adding <online/> (and <opath/> where absent)
                                    and moving the output path
    raw_cmm.csv, raw_fmm.csv        the matcher's own CSV, including `online`
    cmm.log, fmm.log                matcher logs
    cmm_online.csv, fmm_online.csv  the deliverable, 9 columns:
        traj, seq, timestamp, i_on_id, delta_on, alpha_on, delta_sum, alpha_sum, n_cand
        (delta_sum / alpha_sum are LOG-space layer normalisers, log sum_j delta_t(j)
         and log sum_j alpha_t(j); the linear sums underflow to 0)

Usage:
    python3 experiments/scripts/online_export.py
"""
import csv
import datetime
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
ARCHIVE = BASE / "data/archive_20261019/online"
BUILD = BASE / "build"

CMM_SRC = BASE / "input/config/paper1_cmm_scalar.xml"
FMM_SRC = BASE / "input/config/paper1_fmm_baseline.xml"

SPEC = ["traj", "seq", "timestamp", "i_on_id", "delta_on", "alpha_on",
        "delta_sum", "alpha_sum", "n_cand"]


def derive_config(src: Path, dst: Path, out_csv: Path) -> None:
    """Copy a paper config, add <online/> (+ <opath/> if absent), retarget <output><file>."""
    text = src.read_text(encoding="utf-8")
    inj = ""
    if "<online/>" not in text:
        inj += "      <online/>\n"
    if "<opath/>" not in text:
        inj += "      <opath/>\n"
    if inj:
        text = text.replace("    <fields>\n", "    <fields>\n" + inj, 1)
    text = re.sub(r"(<output>.*?<file>)[^<]*(</file>)",
                  lambda m: m.group(1) + str(out_csv) + m.group(2),
                  text, count=1, flags=re.S)
    dst.write_text(text, encoding="utf-8")


def run(binary: Path, cfg: Path, log: Path) -> None:
    with open(log, "w") as fh:
        p = subprocess.run([str(binary), str(cfg)], stdout=fh, stderr=subprocess.STDOUT,
                           cwd=str(BASE))
    if p.returncode != 0:
        sys.exit("FAILED: %s %s (see %s)" % (binary.name, cfg.name, log))


def reshape(raw: Path, out: Path) -> int:
    """raw matcher CSV -> the 9-column online table."""
    n = 0
    with open(raw, newline="") as fi, open(out, "w", newline="") as fo:
        rd = csv.DictReader(fi, delimiter=";")
        if "online" not in rd.fieldnames:
            sys.exit("FATAL: %s has no `online` column; header=%s" % (raw.name, rd.fieldnames))
        wr = csv.writer(fo)
        wr.writerow(SPEC)
        for r in rd:
            cell = (r["online"] or "").strip()
            if not (cell.startswith("(") and cell.endswith(")")):
                # epoch carries no online record (failure row): keep the key, drop values
                wr.writerow([r["id"], r.get("seq", ""), r.get("timestamp", ""),
                             "", "", "", "", "", ""])
                n += 1
                continue
            v = cell[1:-1].split(",")
            if len(v) != 6:
                sys.exit("FATAL: bad online cell %r in %s" % (cell, raw.name))
            wr.writerow([r["id"], r.get("seq", ""), r.get("timestamp", ""),
                         int(float(v[0])), v[1], v[2], v[3], v[4], int(float(v[5]))])
            n += 1
    return n


def main() -> None:
    print("run date", datetime.date.today().isoformat(), "->", ARCHIVE)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    for name, src in (("cmm", CMM_SRC), ("fmm", FMM_SRC)):
        cfg = ARCHIVE / ("%s_online.xml" % name)
        raw = ARCHIVE / ("raw_%s.csv" % name)
        derive_config(src, cfg, raw)
        print("config ->", cfg.relative_to(BASE))

    run(BUILD / "cmm", ARCHIVE / "cmm_online.xml", ARCHIVE / "cmm.log")
    run(BUILD / "fmm", ARCHIVE / "fmm_online.xml", ARCHIVE / "fmm.log")

    for name in ("cmm", "fmm"):
        n = reshape(ARCHIVE / ("raw_%s.csv" % name), ARCHIVE / ("%s_online.csv" % name))
        print("%s_online.csv rows=%d" % (name, n))


if __name__ == "__main__":
    main()

