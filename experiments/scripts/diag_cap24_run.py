#!/usr/bin/env python3
"""D6b (manuscript R20): rerun the delivered matcher with the candidate cap at 24.

The manuscript infers "the cap binds nowhere" from a mean of 11.2 candidates against a cap
of 16. D6 shows the archived run actually sits at the cap on a quarter of its epochs, so the
only way to price the cap is to raise it and re-measure. Two runs are produced, both from
input/config/paper1_cmm_scalar.xml with exactly one parameter changed:

    k 16 -> 24          and the output file redirected into the diagnostic archive
    + tw_use_viterbi=true for the delta arm (the scalar config does not carry the element;
      exp_tp_attribution.py inserts it the same way)

Everything else -- the 300 m disk, map_error_sigma 10.6, kappa 0.227/0.0084, the 2 s gap
guard, reverse tolerance -- is unchanged, so the contrast isolates the cap.

Usage: python3 experiments/scripts/diag_cap24_run.py
"""
import subprocess
import sys
from pathlib import Path

BASE = Path('/home/ncz/fmm_sjtugnc')
DIAG = BASE / 'data/archive_20261019/diag'
SRC = BASE / 'input/config/paper1_cmm_scalar.xml'
OLD_OUT = '<file>data/archive_20260929/scalar/cmm.csv</file>'
VIT_ANCHOR = '<filtered>true</filtered>'
VIT_LINE = '<filtered>true</filtered>\n    <tw_use_viterbi>true</tw_use_viterbi>'


def build(name, viterbi):
    text = SRC.read_text(encoding='utf-8')
    assert text.count('<k>16</k>') == 1, 'k not found exactly once'
    text = text.replace('<k>16</k>', '<k>24</k>')
    assert text.count(OLD_OUT) == 1, 'output element not found exactly once'
    out = DIAG / f'{name}.csv'
    text = text.replace(OLD_OUT, f'<file>{out.relative_to(BASE)}</file>')
    if viterbi:
        assert text.count(VIT_ANCHOR) == 1
        text = text.replace(VIT_ANCHOR, VIT_LINE)
    cfg = DIAG / f'{name}.xml'
    cfg.write_text(text, encoding='utf-8')
    return cfg, out


def main():
    DIAG.mkdir(parents=True, exist_ok=True)
    for name, vit in (('cmm_cap24', False), ('cmm_cap24_viterbi', True)):
        cfg, out = build(name, vit)
        if out.exists() and out.stat().st_size > 0:
            print(f'{name}: exists ({out.stat().st_size} B)')
            continue
        log = DIAG / f'{name}.log'
        with open(log, 'w') as lf:
            rc = subprocess.run([str(BASE / 'build/cmm'), str(cfg)], cwd=BASE,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        n = sum(1 for _ in open(out)) - 1 if out.exists() else 0
        print(f'{name}: rc={rc} rows={n} -> {out}')
        if rc != 0:
            print(log.read_text()[-800:])
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
