#!/usr/bin/env python3
"""Rebuild the 33-family simulation corpus with the corrected protection level.

Why this exists
---------------
`generate_data_cmm.py` writes the CMM input table in degrees. Two conversions
happen there, both of which were wrong in the corpus at `data/simulation_fixAB/`
and were corrected on 2026-09-16 (fixes A and B):

  A. the covariance m^2 -> deg^2 conversion used a *diagonal* Jacobian, which
     drops the meridian convergence term. It is now the full congruence
     transform `J_inv @ Sigma_m @ J_inv.T`.
  B. the satellites were placed at `sats_unit * ORBIT_RADIUS_M`, i.e. around the
     UTM origin rather than around the vehicle, so the WLS line-of-sight
     directions deviated from the sampled ones by a median 6.14 deg.

A third change is *not* a defect fix but a deliberate口径 choice: the scalar
protection level (a bound, consumed verbatim as the search radius at
cmm_algorithm.cpp:787) is now divided by the SMALLER of the two metric scales so
the degree-space radius covers the metric bound in every direction. The previous
revision divided by the larger, leaving the east-west radius 5.8% short.

Fixes A and B were already applied to the corpus at `data/simulation_fixAB/`;
only the protection-level column differs from what this script produces. That is
the point: run this, then diff column-by-column against fixAB. Every column
except `protection_level` must be byte-identical, which is a much stronger check
than "the numbers look reasonable".

Recipe provenance
-----------------
The family list is read off the existing corpus so the two trees are comparable
by construction, and each family's generator arguments are the ones recorded in
`experiments/README.md` (the sigma/occlusion/fault table and the
`--sigma-pr-wls 20` note) and in `batch_generate.py`. Subsampled families are not
generated: they are decimations of a base observations.csv and are produced with
exp3_full_matching.create_subsampled_csv, the same function the experiment uses.

Usage
-----
    python experiments/scripts/regenerate_fixAB_corpus.py \
        --reference data/simulation_fixAB \
        --out-root  data/simulation_fixAB_v2 \
        --jobs 8 --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
GENERATOR = SCRIPT_DIR / "generate_data_cmm.py"

sys.path.insert(0, str(SCRIPT_DIR))
from exp3_full_matching import create_subsampled_csv  # noqa: E402

SEED = 43  # batch_generate.py passes --seed (default 42) + 1
SHAPEFILE = "input/map/haikou/edges.shp"
SUBSAMPLE_INTERVALS = [2, 5, 10, 20, 30, 60]


def family_list(reference: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Return [(relative_dir, path_parts)] for every dataset in the reference tree."""
    out = []
    for obs in sorted(reference.rglob("observations.csv")):
        rel = obs.parent.relative_to(reference)
        out.append((str(rel), rel.parts))
    return out


def classify(parts: tuple[str, ...]) -> str:
    """Which generator produced this family?"""
    if parts[0] == "sigma_mismatch":
        return "mismatch"
    if "subsample_" in parts[-1]:
        return "subsample"
    if parts[0] == "sigma_01_to_30":
        return "sweep"
    return "sigma"


def base_args() -> list[str]:
    return [
        "--shapefile", str((PROJECT_ROOT / SHAPEFILE).resolve()),
        "--speed", "12.0",
        "--sample-rate", "1.0",
        "--num-sats", "8",
        "--seed", str(SEED),
    ]


def recipe_for(rel: str, parts: tuple[str, ...]) -> list[str] | None:
    """The exact generate_data_cmm.py arguments for one family.

    Returns None for families that are derived, not generated.
    """
    kind = classify(parts)
    tail = parts[-1]

    if kind == "subsample":
        return None

    if kind == "mismatch":
        # sigma_mismatch/prNN_wls20 : true pseudorange noise NN, WLS assuming 20 m
        pr = int(tail.split("_")[0][2:])
        assert tail == f"pr{pr}_wls20", tail
        return base_args() + [
            "--min-sigma-pr", str(pr), "--max-sigma-pr", str(pr),
            "--sigma-pr-wls", "20",
            "--count", "10", "--points", "1000",
        ]

    if kind == "sweep":
        # sigma_01_to_30 : 70 trajectories, sigma uniform in [1, 30] m
        return base_args() + [
            "--min-sigma-pr", "1", "--max-sigma-pr", "30",
            "--count", "70", "--points", "1000",
        ]

    # sigma_NN/... : 20 trajectories per level; sigma_30 also carries the
    # occlusion / fault / both degraded conditions
    sigma = int(parts[0].split("_")[1])
    args = base_args() + [
        "--min-sigma-pr", str(sigma), "--max-sigma-pr", str(sigma),
        "--count", "20", "--points", "1000",
    ]
    if "with_occlusion" in parts:
        args += ["--occlusion-angle", "45", "--occlusion-elevation-cutoff", "30"]
    if "with_fault" in parts:
        args += ["--fault-probability", "0.3",
                 "--fault-magnitude-min", "100", "--fault-magnitude-max", "500"]
    return args


def run_one(out_dir: Path, args: list[str], jobs: int) -> tuple[bool, str, float]:
    t0 = time.time()
    cmd = [sys.executable, str(GENERATOR), "--output-dir", str(out_dir), "--jobs", str(jobs)] + args
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        return False, "\n".join(tail), dt
    return True, "", dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, default=Path("data/simulation_fixAB"),
                    help="Corpus whose family list is reproduced (read-only).")
    ap.add_argument("--out-root", type=Path, default=Path("data/simulation_fixAB_v2"))
    ap.add_argument("--jobs", type=int, default=8,
                    help="Workers inside each generate_data_cmm.py process.")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Datasets generated at once. jobs*concurrency should stay "
                         "below the core count.")
    args = ap.parse_args()

    reference = (PROJECT_ROOT / args.reference).resolve()
    out_root = (PROJECT_ROOT / args.out_root).resolve()
    if not reference.is_dir():
        raise SystemExit(f"reference corpus not found: {reference}")
    if not GENERATOR.exists():
        raise SystemExit(f"generator not found: {GENERATOR}")

    families = family_list(reference)
    generated = [(r, p) for r, p in families if classify(p) != "subsample"]
    derived = [(r, p) for r, p in families if classify(p) == "subsample"]

    print(f"reference : {reference}")
    print(f"out-root  : {out_root}")
    print(f"families  : {len(families)}  ({len(generated)} generated + {len(derived)} derived)")
    print(f"parallel  : {args.concurrency} datasets x {args.jobs} workers", flush=True)

    # ── Phase 1: generated families ────────────────────────────────────────
    t0 = time.time()
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for rel, parts in generated:
            rec = recipe_for(rel, parts)
            if rec is None:
                failed.append(f"{rel}: no recipe")
                continue
            futures[pool.submit(run_one, out_root / rel, rec, args.jobs)] = rel
        for fut in as_completed(futures):
            rel = futures[fut]
            ok, err, dt = fut.result()
            print(f"  [{'ok ' if ok else 'FAIL'}] {rel}  {dt:6.1f}s"
                  + ("" if ok else f"\n{err}"), flush=True)
            if not ok:
                failed.append(rel)

    print(f"\nphase 1 done in {time.time() - t0:.1f}s", flush=True)
    if failed:
        print(f"  {len(failed)} failures: {failed}", flush=True)

    # ── Phase 2: subsampled families (decimations of a base table) ─────────
    # Decimate the NEWLY generated base, not the reference one: the decimation
    # must carry the corrected protection_level column, and the ground-truth
    # symlinks must resolve inside the new tree so the family is self-contained.
    t1 = time.time()
    for rel, parts in derived:
        interval = int(parts[-1].replace("subsample_", "").replace("s", ""))
        base_dir = out_root / Path(*parts[:-1])
        src = base_dir / "observations.csv"
        if not src.exists():
            failed.append(f"{rel}: base {src} missing")
            continue
        dst_dir = out_root / rel
        dst_dir.mkdir(parents=True, exist_ok=True)
        create_subsampled_csv(src, dst_dir / "observations.csv", interval)
        for fname in ("ground_truth.csv", "ground_truth_points.csv", "metadata.json"):
            f = base_dir / fname
            if f.exists():
                link = dst_dir / fname
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(os.path.relpath(f, dst_dir))
        print(f"  [ok ] {rel}  (every {interval}s)", flush=True)

    print(f"phase 2 done in {time.time() - t1:.1f}s", flush=True)

    # ── Phase 3: manifest ─────────────────────────────────────────────────
    manifest = {
        "reference": str(reference.relative_to(PROJECT_ROOT)),
        "seed": SEED,
        "shapefile": SHAPEFILE,
        "generated": {rel: recipe_for(rel, parts) for rel, parts in generated},
        "derived": {rel: {"source": str(out_root / Path(*parts[:-1])),
                          "interval_s": int(parts[-1].replace("subsample_", "").replace("s", ""))}
                    for rel, parts in derived},
        "failures": failed,
    }
    (out_root / "regeneration_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nmanifest: {out_root / 'regeneration_manifest.json'}")

    if failed:
        print(f"\nFAILED ({len(failed)}): {failed}")
        return 1
    print("\nAll families regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
