#!/usr/bin/env python3
"""Rebuild tab:ablation -- the real-data 2^3 over the observation model's components.

WHY THIS FILE EXISTS. The eight rows of tab:ablation entered the manuscript during the
2026-09-28 review round with NO producer anywhere in the repository. A grep over
experiments/scripts/ for the A/D/M design, for the string 'anisotrop', and for the
table's own accuracy values (98.13 / 91.07) finds nothing that writes them; the older
ablate_cmm_factors.py is a different design (its letters are radius/projection/emission/
PHMI, 2^4 = 16 arms) and cannot generate this table. That is the defect class this
project has already been bitten by more than once -- the numbers reproduce but their
generator is not tracked, so nobody can tell a re-run from a stale copy. This script is
that provenance, and it re-derives the table from the shipped configuration rather than
restating it.

THE DESIGN. A 2^3 over the three components of the observation model, holding the
operating point (input/config/paper1_cmm.xml) fixed in everything else:

    A  emission shape   isotropic sigma_i^2 I   vs  anisotropic Sigma_i
    D  direction term   off                     vs  on (one-sided von Mises)
    M  map term         off                     vs  10.6 m

A=0 uses the ablation-only key `isotropic_emission_scale = 1`, which replaces the
emission covariance by sigma_i = sqrt((sde_i^2 + sdn_i^2)/2) -- computed PER EPOCH, so
the total variance is matched and only the SHAPE changes. It cannot be emulated by
rewriting the input table: that would rotate the projection metric to Euclidean at the
same time and confound the emission factor with the projection factor. The projection
therefore remains Mahalanobis in every row of this table, which is why this table does
NOT answer whether the 2x2 matrix can be removed from the model -- see
exp_projection_metric.py, which varies that axis.

THE SUBSTITUTION IS A REPLACEMENT, NOT AN APPEND. Boost's property_tree returns the
FIRST match for a key, so appending a second <key> leaves the original in force and the
"ablation" reproduces the baseline row exactly while looking like a real negative
result. Every substitution below asserts that it changed the text and that exactly one
occurrence of the key survives; that is what caught the same bug in
exp_sim_sigma_sweep.py.

THE PAPER ROW IS BUILT IN and every cell is compared at the manuscript's displayed
precision. A row that no longer reproduces prints FAIL rather than being quietly
re-reported, because a table whose generator disagrees with it is worse than no table.

A=0,D=1,M=1 is the row that motivated the design: it equals A=1,D=1,M=1 to the
displayed precision, flipping 3 of 16,155 matched paths. Anisotropy is inert HERE
because the 10.6 m map term dominates a ~0.8 m ellipse -- a good-condition statement,
not a general one.

Usage: python3 experiments/scripts/ablate_real_components.py [--keep]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import real_final_tables as R  # noqa: E402

OUT = Path(os.environ.get('CMM_ABL_OUT') or (BASE / 'data/archive_20260929/abl'))
CMM = BASE / 'build/cmm'

# The manuscript's tab:ablation at the SCALAR operating point, as typeset.
# (D, M) -> (acc %, ECE, Brier, BSS, AUC, sep, smoothed ECE)
#
# The design is 2^2, not 2^3. The third axis was the emission's anisotropy, and the
# model no longer has it: input/config/paper1_cmm_scalar.xml uses an isotropic
# per-epoch emission, so there is nothing to switch. The evidence that settled that
# axis is the simulation sweep (exp_sim_sigma_sweep.py, arms `euclid-only` and
# `iso-emis-only`), which separates the projection metric from the emission shape over
# 15 degraded cells; it is cited there rather than duplicated here.
#
# Two cells of the PREVIOUS (2^3) table were found wrong by this script on 2026-09-28
# and corrected in the manuscript: Brier read 0.055 where the measurement rounds to
# 0.054, and sep read 0.000 where it rounds to -0.004.
PAPER = {
    (1, 1): (98.11, 0.038, 0.032, -0.75, 0.838, 0.429, 0.007),
    (1, 0): (96.40, 0.054, 0.054, -0.57, 0.621, 0.046, 0.034),
    (0, 1): (97.47, 0.071, 0.059, -1.38, 0.682, 0.338, 0.030),
    (0, 0): (91.08, 0.131, 0.130, -0.59, 0.564, -0.004, 0.093),
}

COLUMNS = (('acc', 2), ('ECE', 3), ('Brier', 3), ('BSS', 2),
           ('AUC', 3), ('sep', 3), ('smECE', 3))


def _set(text: str, key: str, value: str) -> str:
    """Set <key>value</key> in the <parameters> block, replacing any existing element.

    Replacement rather than append is load-bearing -- see the module docstring.

    The post-condition is "exactly one occurrence, holding the requested value", NOT
    "the text changed". An arm is allowed to request the value the config already
    carries -- M=1 asks for map_error_sigma = 10.6, which is what paper1_cmm.xml
    already says -- and asserting on text inequality would reject that legitimate case.
    The invariant that matters is uniqueness: Boost takes the FIRST match, so a second
    occurrence is the failure this guards against.
    """
    want = f'<{key}>{value}</{key}>'
    pat = re.compile(rf'<{key}>[^<]*</{key}>')
    if pat.search(text):
        new = pat.sub(want, text, count=1)
    else:
        anchor = '  </parameters>'
        assert anchor in text, f'{key}: no <parameters> block to insert into'
        new = text.replace(anchor, f'    {want}\n{anchor}', 1)
    n = len(re.findall(rf'<{key}>', new))
    assert n == 1, f'{key}: {n} occurrences after substitution'
    assert want in new, f'{key}: substitution did not produce {want}'
    return new


def arm_config(d: int, m: int) -> Path:
    """Write the config for one corner of the design and return its path.

    Derived from the delivered operating point (R.CMM_CONFIG), so the ablation cannot
    drift from it -- the failure this project has hit repeatedly is a second copy of an
    operating point going stale in silence.
    """
    text = R._config(R.CMM_CONFIG, OUT / f'D{d}M{m}.csv')
    # Both keys are stated explicitly, so that no arm inherits a value by omission and
    # silently becomes a different experiment.
    text = _set(text, 'direction_penalty', 'true' if d else 'false')
    text = _set(text, 'map_error_sigma', '10.6' if m else '0.0')
    cfg = OUT / f'D{d}M{m}.xml'
    cfg.write_text(text)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keep', action='store_true',
                    help='Keep the per-arm configs and result CSVs (default: keep both; '
                         'the flag exists so a caller can assert they were written).')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    gt, _ = R.load_gt()
    print(f'{len(gt)} evaluable epochs ({R.CMM_CONFIG})\n')
    hdr = (f'{"D":>2}{"M":>2}  {"Acc.%":>7}{"ECE":>8}{"Brier":>8}{"BSS":>8}'
           f'{"AUC":>8}{"sep":>8}{"smECE":>8}')
    print(hdr)
    print('-' * len(hdr))

    failures = []
    for d, m in sorted(PAPER, reverse=True):
        r = subprocess.run([str(CMM), str(arm_config(d, m))], cwd=BASE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f'{d:>2}{m:>2}  FAILED TO RUN (exit {r.returncode})')
            failures.append((d, m, 'run failed'))
            continue
        csv = OUT / f'D{d}M{m}.csv'
        filt = R.evaluate(csv, gt, 'trustworthiness')
        sm = R.evaluate(csv, gt, 'trustworthiness_smoothed')
        got = (filt['acc'], filt['ece'], filt['brier'], filt['bss'],
               filt['auc'], filt['sep'], sm['ece'])
        print(f'{d:>2}{m:>2}  {got[0]:7.2f}{got[1]:8.3f}{got[2]:8.3f}'
              f'{got[3]:+8.2f}{got[4]:8.3f}{got[5]:8.3f}{got[6]:8.3f}')
        want = PAPER[(d, m)]
        for (name, dec), g, w in zip(COLUMNS, got, want):
            # half a unit in the last printed place, plus a float-representation epsilon
            if abs(g - w) > 0.5 * 10.0 ** (-dec) + 1e-9:
                failures.append((d, m, f'{name}: measured {g!r}, typeset {w} '
                                       f'-- a correct rounding would print '
                                       f'{round(g, dec)}'))

    print('\n=== self-check against the typeset table (manuscript precision) ===')
    if not failures:
        print(f'  all {len(PAPER)} rows x 7 metrics reproduce the manuscript')
        return 0
    for d, m, why in failures:
        print(f'  D{d}M{m}: {why}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
