#!/usr/bin/env python3
"""Attribute the CaMM calibration result to individual components.

The TITS review asked (K-3 / DA-C2 / DA-C1) for the gain over the HMM baseline
to be attributed rather than asserted, and named an existing ablation
(exp8_ablation_components.py + ablation_results.json) that the manuscript never
cited. That artifact cannot be cited: it was produced before the implementation
was pruned, it carries an `h0_lambda` column that no longer exists, and it is
organised around six configuration keys that have since been deleted
(`temperature_adapt`, `protection_level_multiplier`, `phmi_pl_multiplier`,
`normalized`, `background_prob`, `lag_steps`, `map_error_std`). Two of its four
arms are now the same arm, and six of its nine condition directories were never
filled. So the ablation is rebuilt here.

Once the FMM trustworthiness is scored with the same estimand as CaMM -- the
filtering posterior a_t(i*)/sum_j a_t(j), which is what the paper's Eq. (16)
defines and what CaMM reports -- the two methods differ in exactly three
mechanisms. PHMI-grouped emission normalisation is a fourth, CaMM-only. This
script varies all four independently (2^4 = 16 arms) at a matched operating
point, then sweeps the two continuous parameters to show the envelope:

    A  search radius      constant r          vs  r_i = HPL_i
    B  projection metric  Euclidean           vs  Mahalanobis
    C  emission shape     isotropic sigma^2 I vs  anisotropic Sigma
    D  PHMI grouping      phmi = 0            vs  phmi = 1e-5

A and C had no way to be varied before. A is now the model itself -- the radius
is r_i = k_search * HPL_i + delta_road, and the constant-radius level is
k_search = 0 with delta_road carrying the radius. C is the one ablation-only key
left in CovarianceMapMatchConfig. The script asserts up front that every key it
uses is actually wired into the binary, because the config loaders ignore
unknown keys silently and exit 0 -- a mistyped key produces a flat line that
reads like a real negative result rather than an error. That check is what
caught the rename when fixed_search_radius became k_search + delta_road. See
experiments/obsolete/README.md.

Metric definitions are imported, not restated, from inventory_real_metrics,
which takes them from verify_paper_numbers.py -- the same code path that scores
the manuscript's numbers.

Usage:
    python experiments/scripts/ablate_cmm_factors.py            # 16 arms + sweeps
    python experiments/scripts/ablate_cmm_factors.py --stage1   # factorial only
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from collections import Counter
from itertools import product
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import inventory_real_metrics as inv  # noqa: E402  (path set above)

BASE = SCRIPT_DIR.parents[1]
CMM_BIN = BASE / "build/cmm"
FMM_BIN = BASE / "build/fmm"
# The implementation lives in the shared library, not in the thin executable:
# `grep -c <key> build/cmm` is 0 for every key, including ones that work.
LIB = BASE / "build/libFMMLIB.so"
INPUT = BASE / "data/real_vehicle/hainan_06/cmm_input_points_k1.csv"
GT = BASE / "data/real_vehicle/hainan_06/processed/aligned_0729.csv"
PROD = BASE / "data/real_vehicle/hainan_06/processed/cmm_result_k1.csv"

# Fixed for every arm; these are the live real-data values from
# input/config/cmm_real_0729.xml, so the all-"1" corner is the production run.
FIXED = dict(k=16, min_candidates=1, reverse_tolerance=0.1, filtered="true",
             max_interval=180.0, trustworthiness_threshold=0.0,
             cumulative_reverse_pct=0.03)
PHMI_ON = 1.0e-5

# The ablation-only keys this script needs, for the capability probe.
ABLATION_KEYS = ("k_search", "delta_road", "isotropic_emission_scale")

XML = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network><file>input/map/hainan/edges.shp</file><id>key</id><source>u</source><target>v</target></network>
    <ubodt><file>input/map/hainan_ubodt_indexed.bin</file></ubodt>
    <gps>
      <file>{gps}</file>
      <id>id</id><x>x</x><y>y</y><timestamp>timestamp</timestamp>
      <sde>sde</sde><sdn>sdn</sdn><sdu>sdu</sdu>
      <sdne>sdne</sdne><sdeu>sdeu</sdeu><sdun>sdun</sdun>
      <protection_level>protection_level</protection_level>
    </gps>
  </input>
  <output>
    <file>{out}</file>
    <fields><point_mode/><seq/><timestamp/><ogeom/><cpath/><tpath/><opath/><pgeom/>
      <ep/><tp/><trustworthiness/><n_best_trustworthiness/><candidates/>
      <status/><delta_entropy/><posterior_entropy/><cumu_prob/></fields>
  </output>
  <parameters>
    <k>16</k>
    <min_candidates>1</min_candidates>
    <reverse_tolerance>0.1</reverse_tolerance>
    <use_mahalanobis>{mahal}</use_mahalanobis>
    <filtered>true</filtered>
    <max_interval>180.0</max_interval>
    <trustworthiness_threshold>0.0</trustworthiness_threshold>
    <phmi>{phmi}</phmi>
    <cumulative_reverse_pct>0.03</cumulative_reverse_pct>
    <direction_penalty>{dp}</direction_penalty>
    <k_search>{k_search}</k_search>
    <delta_road>{delta_road}</delta_road>
    <isotropic_emission_scale>{iso}</isotropic_emission_scale>
  </parameters>
  <other>
    <log_level>{log_level}</log_level>
    <use_omp>true</use_omp>
    <input_epsg>4326</input_epsg>
  </other>
</config>
"""


def median_pl(path: Path) -> float:
    """Median protection level of the input table, in network units.

    This is where the constant-radius arm searches from, so that the two levels
    of the radius factor enclose the same typical area and the factor measures
    adaptivity rather than size.
    """
    pls = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter=";"):
            try:
                pls.append(float(r["protection_level"]))
            except (TypeError, ValueError):
                continue
    pls.sort()
    return pls[len(pls) // 2]


def run_arm(work: Path, name: str, *, k_search: float, delta_road: float,
            iso: float, mahal: bool, phmi: float, dp: bool = True) -> Path:
    xml, out = work / f"{name}.xml", work / f"{name}.csv"
    xml.write_text(XML.format(
        gps=INPUT.resolve(), out=out, mahal="true" if mahal else "false",
        phmi=phmi, dp="true" if dp else "false", k_search=k_search,
        delta_road=delta_road, iso=iso,
        # Quiet unless the caller is probing: the ablation-only warning fires on
        # every arm and would swamp the log. Level 4 keeps only errors.
        log_level=4 if iso == 0.0 and k_search == 1.0 and delta_road == 0.0 else 2,
    ))
    with open(work / f"{name}.log", "w") as log:
        subprocess.run([str(CMM_BIN), str(xml)], stdout=log,
                       stderr=subprocess.STDOUT, cwd=BASE, check=False)
    return out


def diff_cells(a: Path, b: Path) -> int:
    """Count epochs whose matched path differs.

    Keyed on (id, timestamp), NOT on (id, seq). CMM's `seq` restarts at every
    sub-segment boundary, so it is not unique within a trajectory and a
    seq-keyed comparison silently collapses rows -- an early draft of this
    script reported 12834 rows for a 16155-row file that way. FMM's `seq` does
    count points within a trajectory, which is why fair_fmm_sweep.diff_cells
    can use it; this is the CMM counterpart.
    """
    def load(p):
        d = {}
        with open(p, newline="") as f:
            for r in csv.DictReader(f, delimiter=";"):
                d[(r["id"].strip(), float(r["timestamp"]))] = r["cpath"].strip()
        return d
    da, db = load(a), load(b)
    return sum(1 for k in set(da) | set(db) if da.get(k) != db.get(k))


def probe_keys() -> None:
    """Fail loudly if the ablation-only keys are not in the built library.

    The loaders read every key with `get(key, default)` and have no else
    branch, so a key the binary does not know is ignored without a warning and
    without a non-zero exit code. A sweep over such a key returns identical
    results for every value and looks like a real negative result. Checking the
    library first is what makes that failure impossible to miss.
    """
    if not LIB.exists():
        raise SystemExit(f"not built: {LIB}")
    for key in ABLATION_KEYS:
        n = subprocess.run(["grep", "-ac", key, str(LIB)],
                           capture_output=True, text=True).stdout.strip()
        if n in ("", "0"):
            raise SystemExit(
                f"{key} is not in {LIB.name}. Either the binary is stale -- "
                f"rebuild -- or the key is not wired up, in which case every "
                f"arm that sets it would silently be the same run."
            )
        print(f"  key wired: {key} ({n} hits in {LIB.name})")


def score(path: Path, gt) -> dict:
    return inv.evaluate(path, None, gt)


def correctness_map(path: Path, gt) -> dict:
    """{(id, key): bool} over the evaluable epochs this run produced.

    Reuses the evaluator's own key choice and match rule, so this cannot drift
    from the headline metrics.
    """
    by_ts, by_seq = gt
    run, key_name = inv.load_run(path, None)
    g = by_ts if key_name == "timestamp" else by_seq
    out = {}
    for key, (gt_e, _pt) in g.items():
        hit = run.get(key)
        if hit is None:
            continue
        out[key] = inv.v.is_edge_match(hit[0], gt_e)
    return out


def fmt_row(label: str, m: dict) -> str:
    return (f"{label:<22}{m['n']:>7}{m['acc']:>9.4f}{m['ece']:>8.4f}"
            f"{m['auc']:>8.4f}{m['tw_c']:>9.4f}{m['tw_w']:>9.4f}"
            f"{m['sep']:>9.4f}{m['perr']:>8.2f}{m['miss']:>8}")


HEADER = (f"{'arm':<22}{'n':>7}{'acc%':>9}{'ECE':>8}{'AUC':>8}"
          f"{'TW_cor':>9}{'TW_wrg':>9}{'sep':>9}{'err_m':>8}{'skipped':>8}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path("/tmp/cmm_ablation"))
    ap.add_argument("--stage1", action="store_true", help="factorial only")
    ap.add_argument("--direction-penalty", default="true",
                    choices=["true", "false"])
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    dp = args.direction_penalty == "true"

    print("Preflight")
    probe_keys()
    r_fixed = median_pl(INPUT)
    print(f"  constant-radius arm searches from the median PL = {r_fixed:.8f} "
          f"({r_fixed * 111320:.2f} m)")

    gt = inv.load_gt(GT)
    print(f"  GT: {GT.name}, {len(gt[0])} evaluable epochs\n")

    # --- positive controls -------------------------------------------------
    # Each new key must change the output at an extreme value. A key that is
    # silently ignored produces a flat line across all its levels; this is what
    # turns that into an error instead of a finding.
    print("Positive controls (an extreme value must move the output)")
    ctrl = run_arm(args.work, "ctrl_default", k_search=1.0, delta_road=0.0,
                   iso=0.0, mahal=True, phmi=PHMI_ON, dp=dp)
    for name, kw in (("ctrl_fixed_tiny", dict(k_search=0.0, delta_road=1e-6, iso=0.0)),
                     ("ctrl_iso_huge", dict(k_search=1.0, delta_road=0.0, iso=100.0))):
        p = run_arm(args.work, name, mahal=True, phmi=PHMI_ON, dp=dp, **kw)
        d = diff_cells(p, ctrl)
        status = "ok" if d > 0 else "NOT WIRED"
        print(f"  {name:<18} {d:>6} epochs differ from the default   {status}")
        if d == 0:
            raise SystemExit(f"{name} changed nothing: that key is not reaching "
                             f"the algorithm.")

    # --- stage 1: 2^4 factorial -------------------------------------------
    # Level 0/1 per factor. A: 0 = constant radius, 1 = r_i = HPL_i.
    # B: 0 = Euclidean projection, 1 = Mahalanobis. C: 0 = isotropic emission
    # (variance-matched, scale 1), 1 = the reported anisotropic covariance.
    # D: 0 = phmi off, 1 = phmi on.
    print(f"\nStage 1 -- 2^4 factorial, direction_penalty={args.direction_penalty}")
    print(HEADER)
    print("-" * len(HEADER))
    rows, maps = [], {}
    for a, b, c, d in product((0, 1), repeat=4):
        name = f"A{a}_B{b}_C{c}_D{d}"
        # A=0 pins the radius to a constant: k_search = 0 leaves only the
        # offset term, which is the ablation's "constant vs adaptive" contrast.
        path = run_arm(args.work, name,
                       k_search=1.0 if a else 0.0,
                       delta_road=0.0 if a else r_fixed,
                       iso=0.0 if c else 1.0,
                       mahal=bool(b), phmi=PHMI_ON if d else 0.0, dp=dp)
        m = score(path, gt)
        rows.append((name, (a, b, c, d), m))
        maps[name] = correctness_map(path, gt)
        print(fmt_row(name, m))

    # The production run, for orientation.
    if PROD.exists():
        print(fmt_row("(production k1)", score(PROD, gt)))

    # --- common-epoch-set accuracy ----------------------------------------
    # Arms can differ in row count, so an accuracy computed per arm can be
    # compared across different denominators. Repeat it on the epochs every arm
    # produced, which is the comparison the factorial's main effects rest on.
    print("\nAccuracy on the epochs common to all 16 arms")
    common = set.intersection(*(set(m) for m in maps.values()))
    for name, _, m in rows:
        cm = maps[name]
        acc = 100.0 * sum(cm[k] for k in common) / len(common)
        print(f"  {name:<22}{len(common):>7} epochs   acc {acc:>8.4f} %"
              f"   (per-arm {m['acc']:.4f} %)")

    # --- main effects ------------------------------------------------------
    print("\nMain effects (mean over the other three factors)")
    print(f"{'factor':<14}{'level 0':<28}{'level 1':<28}{'delta':>10}")
    print("-" * 80)
    labels = [("A radius", "constant", "HPL"),
              ("B projection", "Euclidean", "Mahalanobis"),
              ("C emission", "isotropic", "anisotropic"),
              ("D PHMI", "off", "on")]
    for fi, (flabel, l0, l1) in enumerate(labels):
        for metric in ("ece", "auc", "acc"):
            v0 = [m[metric] for _, lv, m in rows if lv[fi] == 0]
            v1 = [m[metric] for _, lv, m in rows if lv[fi] == 1]
            m0, m1 = statistics.fmean(v0), statistics.fmean(v1)
            print(f"{flabel if metric == 'ece' else '':<14}"
                  f"{l0 + ' ' + metric:<28}{l1 + ' ' + metric:<28}"
                  f"{m1 - m0:>+10.4f}")

    # --- stage 2: envelopes ------------------------------------------------
    if not args.stage1:
        print("\nStage 2a -- constant-radius envelope at the FMM-like corner "
              "(B=0 Euclidean, C=0 isotropic, D=0)")
        print(HEADER)
        print("-" * len(HEADER))
        for metres in (5, 10, 20, 22.77, 30, 50, 100):
            p = run_arm(args.work, f"r{metres}", k_search=0.0,
                        delta_road=metres / 111320.0,
                        iso=1.0, mahal=False, phmi=0.0, dp=dp)
            print(fmt_row(f"r = {metres} m", score(p, gt)))

        print("\nStage 2b -- isotropic-emission scale envelope "
              "(A=1 HPL, B=1 Mahalanobis, D=1)")
        print(HEADER)
        print("-" * len(HEADER))
        for sc in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            p = run_arm(args.work, f"iso{sc}", k_search=1.0, delta_road=0.0,
                        iso=sc, mahal=True, phmi=PHMI_ON, dp=dp)
            print(fmt_row(f"scale = {sc}", score(p, gt)))
        print(fmt_row("scale = off (aniso)", score(PROD, gt)))

    print(f"\nArtifacts in {args.work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
