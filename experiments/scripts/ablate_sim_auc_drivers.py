#!/usr/bin/env python3
"""Which change cost the calibration? Vary one input at a time on a fixed dataset.

The 2026-09 rewrite moved three things at once on the simulation side:

  1. the covariance frame            (generator fixes A/B, a new corpus)
  2. the covariance magnitude        (the MIN_SIGMA floor was deleted)
  3. the work the direction penalty does
     (`kappa = v^2 / (sde*sdn)`, inversely proportional to the variance, so
     deleting a floor that rescaled every small covariance inflates it)

(1) cannot be undone without regenerating the corpus, but (2) and (3) can be
varied on an *existing* corpus without touching the geometry at all, because
both are pure functions of columns the matcher reads. That is what this does --
one dataset, one column rewritten per run, ground truth and trajectory geometry
bit-identical throughout. Any change in accuracy/ECE/AUC is therefore attributable
to that column and nothing else.

Variants
--------
  base          unmodified
  dir_off       `direction_penalty=false`; the covariance untouched
  scale_A       every standard deviation times A (covariances times A^2), the
                protection level untouched. This is the exact signature of the
                deleted floor: it rescaled the covariance column and left
                `protection_level` alone.
  scale_A_pl    the same, plus the protection level, i.e. a self-consistent
                inflation of the whole stochastic model -- what the GST
                calibration `k = 3.0008` does on the real data.

Usage
-----
    python experiments/scripts/ablate_sim_auc_drivers.py \
        --dataset data/simulation_plfix/sigma_10/no_occlusion/no_fault \
        --scales 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import csv

import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments/scripts"))

from run_cmm_on_corpus import build_cmm_xml, CMM_BIN  # noqa: E402

SD_COLS = ("sde", "sdn", "sdu")
COV_COLS = ("sdne", "sdeu", "sdun")

# Every variant writes the same result filename, so one pass of the corpus
# evaluator over the work tree returns all of them in its per-dataset table.
RESULT_SUFFIX = "_v"


def write_variant(src: Path, dst: Path, scale: float, scale_pl: bool,
                  dir_penalty: bool, tmp_root: Path) -> Path:
    """One dataset directory: symlinks except for the two columns that change.

    Refuses to reuse a directory. Rebuilding in place would mean deleting, and a
    recursive delete is not this script's call to make; pass a fresh --work.
    """
    if dst.exists():
        raise SystemExit(f"{dst} already exists -- pass a fresh --work directory")
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.name in ("observations.csv",) or f.name.startswith("cmm_result"):
            continue
        (dst / f.name).symlink_to(f.resolve())

    with (src / "observations.csv").open(newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh, delimiter=";")
        cols = list(rdr.fieldnames)
        rows = list(rdr)
    with (dst / "observations.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
        w.writeheader()
        for r in rows:
            if scale != 1.0:
                for c in SD_COLS:
                    r[c] = repr(float(r[c]) * scale)
                for c in COV_COLS:
                    r[c] = repr(float(r[c]) * scale * scale)
                if scale_pl:
                    r["protection_level"] = repr(float(r["protection_level"]) * scale)
            w.writerow(r)

    xml = build_cmm_xml(str((dst / "observations.csv").resolve()),
                        str((dst / f"cmm_result{RESULT_SUFFIX}.csv").resolve()))
    if not dir_penalty:
        xml = xml.replace("<direction_penalty>true</direction_penalty>",
                          "<direction_penalty>false</direction_penalty>")
    xp = tmp_root / f"{dst.name}.xml"
    xp.write_text(xml)
    return xp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path,
                    default=Path("data/simulation_plfix/sigma_10/no_occlusion/no_fault"))
    ap.add_argument("--scales", type=float, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--work", type=Path, default=Path("/tmp/auc_ablation"))
    args = ap.parse_args()

    if not CMM_BIN.exists():
        raise SystemExit(f"matcher not built: {CMM_BIN}")
    src = (BASE / args.dataset).resolve()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)

    variants = [("base", 1.0, False, True), ("dir_off", 1.0, False, False)]
    for s in args.scales:
        if s != 1.0:
            variants.append((f"scale_{s:g}", s, False, True))
            variants.append((f"scale_{s:g}_pl", s, True, True))

    print(f"dataset: {src.relative_to(BASE)}")
    print(f"work   : {work}\n")
    for name, scale, spl, dpen in variants:
        d = work / name
        xp = write_variant(src, d, scale, spl, dpen, work)
        t0 = time.time()
        r = subprocess.run([str(CMM_BIN), str(xp)], capture_output=True, text=True)
        dt = time.time() - t0
        status = "ok" if r.returncode == 0 else f"FAIL {r.stderr[-200:]}"
        print(f"  {name:14s} scale={scale:<5g} pl={'y' if spl else 'n'} "
              f"dir={'y' if dpen else 'n'}  {dt:6.1f}s  {status}", flush=True)
        if r.returncode != 0:
            return 1

    print()
    rc = subprocess.run(
        [sys.executable, str(BASE / "experiments/scripts/eval_sim_corpus.py"),
         "--root", str(work), "--suffix", RESULT_SUFFIX,
         "--out", str(work / "eval.csv")],
        capture_output=True, text=True)
    print(rc.stdout[-4000:])
    if rc.returncode != 0:
        print(rc.stderr[-2000:])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
