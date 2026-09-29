"""The sigma sweep of Section IV: calibration against measurement noise, both matchers.

WHY. The real-vehicle experiment shows the score is calibrated at one noise level. It
cannot show where that stops being true, because it has one receiver on one day. The
simulation corpus sweeps the pseudorange noise sigma_rho over {1, 5, 10, 15, 20, 25, 30} m
with the error model known, so it is the place where "calibration degrades, and here is
the boundary" can be shown rather than asserted -- and where the degraded-condition cells
(occlusion, faults, a deliberately misspecified covariance) live.

WHY A BASELINE RUN AND NOT JUST THE CORPUS'S OWN. Each corpus cell carries an
`fmm_result.csv`, but nothing in the tree records the configuration it was produced with,
and this project has already been burned once by a run whose provenance was a stale
config (see the operating-point note in CLAUDE.md). So the baseline is re-run here at a
configuration stated in this file, and the corpus's own file is reported beside it. Where
they agree the corpus's file is the documented baseline and may be used; where they
differ, only the re-run is used and the discrepancy is printed.

THE BASELINE CONFIG, and the two mistakes it took to find it. `build_fmm_xml` below is
the `exp3_full_matching.build_fmm_xml` config verbatim, which is what generated the
corpus:
    trajectory mode (`<geom>geom</geom>` on the cell's `fmm_trajectory.csv`),
    k=16, r=0.0026949335 (= 300 m at 111320 m/deg), gps_error=0.001 (= 111.32 m),
    reverse_tolerance=0.1, tw_use_viterbi=true, step=500 (progress report only).
With it the re-run is **bit-identical** to the corpus's `fmm_result.csv` on every cell
that has one -- all 10 sweep/degraded cells, every column, all 100 000 rows. The two
deviations an earlier version of this script made, both recorded here because both are
easy to repeat:
  1. gps_error=0.0008983112 -- the REAL-DATA sigma (100 m). The corpus was generated at
     0.001, and the difference moves ECE at every sigma.
  2. point mode on `observations.csv`. Measured, not assumed: `CSVPointReader` parses the
     x/y/timestamp columns with `std::stof`, i.e. into float32. 110.05620886 becomes
     110.05620575 (0.33 m) and the integer-rounded timestamp 1779957201 becomes
     1779957248, so the point-mode run does not even address the same epochs. Point mode
     is not a choice a corpus runner may silently make.

THE CMM SIDE. `--cmm-arm paper1` is the previous operating point, at the parameter block of
`input/config/paper1_cmm.xml` verbatim: the fixed 300 m disk both matchers search
(k_search=0, delta_road=0.0026949335), phmi=0, k=16, min_candidates=1, Mahalanobis
projection, the 10.6 m map-registration term in the emission, and the two-parameter heading
model at its converged constants (direction_kappa_scale=0.227, direction_kappa_floor=
0.0084 rad^2), with direction_max_interval=2 s and direction_symmetric=false at their
defaults. `--cmm-arm scalar` is the candidate operating point,
`input/config/paper1_cmm_scalar.xml`: the same block with the covariance's SHAPE removed
from both places it entered -- Euclidean projection and an isotropic per-epoch emission --
so the 2x2 matrix is used nowhere.

THE HEADING CONSTANTS WERE CORRECTED ON 2026-09-29. They read 0.254 / 0.0093 -- the
FIRST ITERATION of a fixed-point fit -- until then, and the shipped configuration had
carried the converged 0.227 / 0.0084 for a day. Every `paper1` number this script produced
before that date is at the wrong constants and is not comparable with the ones after it.
The template is now the shipped block, and the substitution arms below take their values
from it rather than restating them.

`--cmm-arm direction-off` is the same model with the heading term disabled; that
is the scope ablation Section IV-C reports, not a configuration the paper runs.
`euclid-only` and `iso-emis-only` split `scalar` into its two halves, because the combined
arm cannot say which of them moves a number.

WHY THE HEADING TERM IS OUT OF SCOPE ON THIS CORPUS, which is a result rather than a
concession. The term scores the displacement against the candidate's tangent, so it needs
the displacement to have a measurable direction. Its signal-to-noise on the corpus is
|v| / sqrt(n^T Sigma n) = 7.6, 2.3, 1.8, 1.5, 1.8, 1.8, 1.6 over sigma_rho = 1..30 m,
against 27.9 on the Haikou data. The corpus is 1 Hz at 12 m/s so the step is ~12 m while
its fabricated covariance is ~1.5 sigma_rho in position; at sigma_rho = 10 m the position
uncertainty alone is 15 m. The fitted intercept b of sigma_theta^2 = c n^T Sigma n / |v|^2
+ b is 0.7-2.9 rad^2 (48-97 deg) on the corpus against 0.008 rad^2 (5.2 deg) on the real
data: that is the signature of a sample with no heading in it, not of a floor. Measured
consequence, and both directions belong in the paper: with the term the ECE is better at
six of the seven levels (0.002 against 0.102 at sigma = 1 m) and worse at sigma = 30 m
(0.222 against 0.168), while accuracy is worse at six of the seven (up to -16.23 pp) and
AUC worse at six of the seven; and the number of epochs that are wrong AND reported above
0.9 goes from 66 to 1298 at sigma = 30 m. A corpus able to test the term would need a
higher sample rate or a smaller position noise, which is a property of the generator.
`experiments/scripts/fit_direction_kappa.py` reproduces the ratios.

The corpus's own archived CaMM `cmm_result.csv` is NOT used and is not reproduced: those
files were written 2026-09-21, before the heading term was recalibrated and before the
09-22/09-26 commits. The script still reports the difference, which is the provenance check
that the FMM side passes exactly and the CaMM side does not.

METRICS. The same definitions as Section V. ECE and the tie-corrected rank AUC come from
`verify_paper_numbers` (imported, not reimplemented); MCE, Brier, the Brier skill score
and the TW separation come from `real_final_tables.evaluate`, so a simulation number and
a real-data number are the same functional of the same arrays. Accuracy is over every
evaluable epoch; the score-based metrics are over the epochs that carry a score (the
writer emits -999 for "no value").

THE MISMATCH FAMILY, which is where these cells stop being comparable. `sigma_mismatch/
prNN_wls20` holds the actual pseudorange sigma_rho = NN m while the WLS covariance was
computed with an assumed sigma of 20 m, so pr10 is over-conservative and pr30 is
over-confident by the same construction. But those five cells are 07-31 artefacts with no
`fmm_trajectory.csv` at all, and their stored `fmm_result.csv` does not reproduce under
ANY config tried (gps_error in {0.0008983112, 0.001, 0.01, 0.05, 0.0026949335} x k in
{8,16} x tw_use_viterbi in {true,false}, all differing on 10 000 of 10 000 rows). The
baseline there is therefore re-run on a `fmm_trajectory.csv` synthesised from
`observations.csv` by `exp3_full_matching.convert_to_fmm_trajectory` -- the same converter
the corpus runner uses -- and written into OUTDIR rather than into the corpus tree.

Usage: python3 experiments/scripts/exp_sim_sigma_sweep.py [--skip-rerun]
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path('/home/ncz/fmm_sjtugnc')
sys.path.insert(0, str(BASE / 'experiments/scripts'))
import eval_sim_corpus as E  # noqa: E402
import exp3_full_matching as exp3  # noqa: E402
import real_final_tables as R  # noqa: E402
import verify_paper_numbers as v  # noqa: E402

SIM = BASE / 'data/simulation'
# Overridable, so a candidate operating point's sweep can be written beside the
# delivered one instead of on top of it.
OUTDIR = Path(os.environ.get('CMM_SIM_OUTDIR') or (BASE / 'data/archive_20260928/sim'))
LEVELS = ['sigma_01', 'sigma_05', 'sigma_10', 'sigma_15', 'sigma_20', 'sigma_25', 'sigma_30']

NET = 'input/map/hainan/edges.shp'
UBODT = 'input/map/hainan_ubodt_indexed.bin'

# The corpus's FMM configuration, reproduced bit-exactly. See the module docstring.
FMM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network><file>{net}</file><id>key</id><source>u</source><target>v</target></network>
    <ubodt><file>{ubodt}</file></ubodt>
    <gps><file>{gps}</file><id>id</id><geom>geom</geom><timestamp>timestamp</timestamp></gps>
  </input>
  <output>
    <file>{out}</file>
    <fields><point_mode/><seq/><timestamp/><ogeom/><cpath/><tpath/><opath/><pgeom/>
      <ep/><tp/><trustworthiness/></fields>
  </output>
  <parameters>
    <k>16</k><r>0.0026949335</r><gps_error>0.001</gps_error>
    <reverse_tolerance>0.1</reverse_tolerance><tw_use_viterbi>true</tw_use_viterbi>
  </parameters>
  <other><log_level>2</log_level><use_omp>true</use_omp><step>500</step></other>
</config>
"""

TW_SENTINEL = -900.0

# Section IV's CaMM configuration: the parameter block of input/config/paper1_cmm.xml,
# verbatim, with the corpus's own input file. The two direction constants are the model's
# calibrated heading parameters and are both written out, as in the shipped file.
CMM_PAPER1 = """<?xml version="1.0" encoding="UTF-8"?>
<config>
  <input>
    <network><file>{net}</file><id>key</id><source>u</source><target>v</target></network>
    <ubodt><file>{ubodt}</file></ubodt>
    <gps>
      <file>{gps}</file><id>id</id><x>x</x><y>y</y><timestamp>timestamp</timestamp>
      <sde>sde</sde><sdn>sdn</sdn><sdu>sdu</sdu>
      <sdne>sdne</sdne><sdeu>sdeu</sdeu><sdun>sdun</sdun>
      <protection_level>protection_level</protection_level>
    </gps>
  </input>
  <output>
    <file>{out}</file>
    <fields><point_mode/><seq/><timestamp/><ogeom/><cpath/><tpath/><opath/><pgeom/>
      <ep/><tp/><trustworthiness/><trustworthiness_smoothed/><n_best_trustworthiness/><candidates/>
      <status/><delta_entropy/><posterior_entropy/><cumu_prob/></fields>
  </output>
  <parameters>
    <k>16</k><min_candidates>1</min_candidates>
    <reverse_tolerance>0.1</reverse_tolerance>
    <use_mahalanobis>true</use_mahalanobis><filtered>true</filtered>
    <max_interval>180.0</max_interval><trustworthiness_threshold>0.0</trustworthiness_threshold>
    <phmi>0.0</phmi><cumulative_reverse_pct>0.03</cumulative_reverse_pct>
    <direction_penalty>true</direction_penalty><enable_gap_bridging>true</enable_gap_bridging>
    <k_search>0.0</k_search><delta_road>0.0026949335</delta_road>
    <map_error_sigma>10.6</map_error_sigma>
    <direction_kappa_scale>0.227</direction_kappa_scale>
    <direction_kappa_floor>0.0084</direction_kappa_floor>
  </parameters>
  <other><log_level>2</log_level><use_omp>true</use_omp><input_epsg>4326</input_epsg></other>
</config>
"""


def cells():
    """(label, sigma_rho, directory) for every cell this section reports.

    sigma_rho is the generator's `min_sigma_pr`; for the mismatch family it is the
    ACTUAL pseudorange noise, whose assumed value is 20 m (the cell's name says so).
    """
    out = []
    for lvl in LEVELS:
        d = SIM / lvl / 'no_occlusion' / 'no_fault'
        if (d / 'metadata.json').exists():
            out.append((lvl, json.load(open(d / 'metadata.json'))['arguments']['min_sigma_pr'], d))
    for sub in ['no_occlusion/with_fault', 'with_occlusion/no_fault', 'with_occlusion/with_fault']:
        d = SIM / 'sigma_30' / sub
        if (d / 'metadata.json').exists():
            out.append((f'sigma_30/{sub}', 30.0, d))
    for name in ['pr10_wls20', 'pr15_wls20', 'pr20_wls20', 'pr25_wls20', 'pr30_wls20']:
        d = SIM / 'sigma_mismatch' / name
        if (d / 'metadata.json').exists():
            out.append((f'sigma_mismatch/{name}',
                        json.load(open(d / 'metadata.json'))['arguments']['min_sigma_pr'], d))
    return out


def pairs(data_dir, run_csv, col='trustworthiness'):
    """(label, score, n_evaluable) over the epochs this run answers.

    The join and the label are `eval_sim_corpus`'s canonical ones: ground-truth edge from
    `ground_truth.csv`'s point_edge_ids indexed by seq, reached through the (id,
    rounded timestamp) map built from `observations.csv`. `seq` is not the key -- a split
    trajectory restarts it.

    `col` selects the estimator, not the run: one CMM run writes the filtering posterior
    to `trustworthiness` and the smoothed one to `trustworthiness_smoothed`, and the two
    arms must be read off the SAME file so that the candidate and the Viterbi path are
    identical and only the evidence set differs. The default keeps every existing caller
    on the filtering column.
    """
    gt_edges = E.load_gt_edges(data_dir)
    ts_to_seq, _ = E.load_obs_index(data_dir)
    y, s, n_eval = [], [], 0
    with run_csv.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            tid = row['id'].strip()
            seq = ts_to_seq.get((tid, int(round(float(row['timestamp'])))))
            if seq is None or (tid, seq) not in gt_edges:
                continue
            n_eval += 1
            try:
                tw = float(row[col])
            except (ValueError, KeyError):
                continue
            if tw <= TW_SENTINEL:
                continue
            y.append(1.0 if v.is_edge_match(row.get('cpath', '').strip(),
                                            gt_edges[(tid, seq)]) else 0.0)
            s.append(tw)
    return np.array(y), np.array(s), n_eval


def _point_err(data_dir, run_csv):
    """Mean and P95 distance from the matched point to the truth point, in metres."""
    gt = {}
    with (data_dir / 'ground_truth_points.csv').open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            gt[(row['id'].strip(), int(row['seq']))] = (float(row['x']), float(row['y']))
    ts_to_seq, _ = E.load_obs_index(data_dir)
    errs = []
    with run_csv.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            tid = row['id'].strip()
            seq = ts_to_seq.get((tid, int(round(float(row['timestamp'])))))
            g = gt.get((tid, seq)) if seq is not None else None
            p = _wkt_point(row.get('pgeom', ''))
            if g and p:
                errs.append(v.haversine_m(p[0], p[1], g[0], g[1]))
    if not errs:
        return float('nan'), float('nan')
    a = np.array(errs)
    return float(a.mean()), float(np.percentile(a, 95))


def _wkt_point(s):
    import re
    m = re.match(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', str(s).strip())
    return (float(m.group(1)), float(m.group(2))) if m else None


def stats(y, s, n_eval):
    """The Section V metric set, on the definitions Section V uses."""
    if len(y) == 0:
        return dict(n_eval=n_eval, n=0, acc=float('nan'), ece=float('nan'),
                    mce=float('nan'), brier=float('nan'), bss=float('nan'),
                    unc=float('nan'), res=float('nan'), rel=float('nan'),
                    auc=0.5, sep=float('nan'), mean_tw=float('nan'))
    n_ok = int(round(y.sum()))
    p = float(y.mean())
    brier = float(((s - y) ** 2).mean())
    unc_res_rel = _decompose(y, s)
    return dict(
        n_eval=n_eval, n=len(y),
        acc=100.0 * n_ok / n_eval if n_eval else float('nan'),
        ece=float(v.ece(s, y)), mce=float(R.mce(s, y)),
        brier=brier, bss=1.0 - brier / (p * (1.0 - p)),
        unc=unc_res_rel[0], res=unc_res_rel[1], rel=unc_res_rel[2],
        auc=float(v.auc(y, s)),
        sep=float(s[y == 1].mean() - s[y == 0].mean()) if 0 < n_ok < len(y) else float('nan'),
        mean_tw=float(s.mean()),
        sat=100.0 * float((s >= 1.0).mean()),
    )


def _decompose(y, s, nbins=10):
    """Brier = UNC - RES + REL, the Murphy decomposition on the same 10 equal-width bins.

    Reported alongside BSS because BSS is their ratio over UNC and therefore says
    nothing once UNC collapses. UNC depends only on the base rate, so a matcher has no
    influence on the denominator it is being divided by.
    """
    n = len(y)
    p_bar = float(y.mean())
    unc = p_bar * (1.0 - p_bar)
    res = rel = 0.0
    edges = np.linspace(0.0, 1.0, nbins + 1)
    for b in range(nbins):
        lo, hi = edges[b], edges[b + 1]
        m = (s >= lo) & (s <= hi) if b == nbins - 1 else (s >= lo) & (s < hi)
        if not m.any():
            continue
        o_k = float(y[m].mean())
        p_k = float(s[m].mean())
        res += m.sum() / n * (o_k - p_bar) ** 2
        rel += m.sum() / n * (o_k - p_k) ** 2
    return unc, res, rel


def run_fmm(data_dir, tag, skip_rerun):
    """The baseline at the corpus's own config; the stored file if --skip-rerun."""
    traj = data_dir / 'fmm_trajectory.csv'
    synthesised = False
    if not traj.exists():
        traj = OUTDIR / f'{tag}_fmm_trajectory.csv'
        if not traj.exists():
            exp3.convert_to_fmm_trajectory(data_dir / 'observations.csv', traj)
        synthesised = True
    if skip_rerun:
        stored = data_dir / 'fmm_result.csv'
        return (stored if stored.exists() else None), synthesised
    out = OUTDIR / f'{tag}_fmm.csv'
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w') as fh:
        fh.write(FMM_XML.format(gps=str(traj), out=str(out), net=NET, ubodt=UBODT))
        cfg = fh.name
    r = subprocess.run([str(BASE / 'build' / 'fmm'), cfg], cwd=BASE,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    Path(cfg).unlink()
    return (out if r.returncode == 0 else None), synthesised


def _set_key(text, key, value):
    """Set <key>value</key> in the <parameters> block, replacing any existing element.

    Replacement rather than append is load-bearing, and it is the bug the direction-off
    arm was written to avoid: Boost's property_tree returns the FIRST match for a key, so
    appending a second <key> silently leaves the original in force and the "ablation"
    reproduces the baseline arm exactly while looking like a real result. The
    post-condition is therefore uniqueness plus the requested value -- not text
    inequality, which would reject an arm that legitimately asks for a value the
    template already carries.
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


# Each arm is the set of (key, value) settings applied to CMM_PAPER1.
#
#   paper1         the shipped model, i.e. Section IV's CaMM arm. It must reproduce the
#                  corpus's stored cmm_result.csv BIT-EXACTLY; that is the invariant the
#                  verdict section checks, and it is why every other arm is reported
#                  against this one rather than against the stored file.
#   direction-off  Section IV-C's scope ablation: the same model with the heading term
#                  disabled. The constants go to 0 as well as the switch, because a
#                  disabled term that still carries fitted constants is a different
#                  statement than a disabled term.
#   no-cov-shape   The question of whether the 2x2 covariance is needed at all: Euclidean
#                  candidate projection and an isotropic per-epoch emission, i.e. the
#                  matrix is used NOWHERE. On the real data this reproduces the operating
#                  point (98.11/0.0382/0.8380 against 98.13/0.0386/0.8368). It is run here
#                  because the real data is a GOOD-CONDITION regime: the 10.6 m map term
#                  dominates a ~0.8 m ellipse, so the shape has nowhere to show. On this
#                  corpus sigma_rho reaches 30 m and the fabricated position sigma is
#                  ~1.5 sigma_rho, where the ellipse is comparable to the map error -- if
#                  the shape matters anywhere in this paper, it matters here.
#   euclid-only    no-cov-shape's two halves taken separately, because the combined arm
#   iso-emis-only  cannot say WHICH of them moves the number. `no-cov-shape` = both.
ARMS = {
    'paper1': [],
    'direction-off': [('direction_penalty', 'false'),
                      ('direction_kappa_scale', '0.0'),
                      ('direction_kappa_floor', '0.0')],
    'scalar': [('use_mahalanobis', 'false'),
               ('isotropic_emission_scale', '1.0')],
    # The Viterbi-score estimator on the same path: the third arm of the estimator
    # comparison. Same candidates, same Viterbi path, different score.
    'scalar-viterbi': [('use_mahalanobis', 'false'),
                       ('isotropic_emission_scale', '1.0'),
                       ('tw_use_viterbi', 'true')],
    # Section IV-C's scope ablation, at the SCALAR operating point. The direction term is
    # the axis the corpus cannot evaluate (see the docstring), so this arm exists to show
    # what turning it off does here, not to select a configuration.
    'scalar-direction-off': [('use_mahalanobis', 'false'),
                             ('isotropic_emission_scale', '1.0'),
                             ('direction_penalty', 'false'),
                             ('direction_kappa_scale', '0.0'),
                             ('direction_kappa_floor', '0.0')],
    'euclid-only': [('use_mahalanobis', 'false')],
    'iso-emis-only': [('isotropic_emission_scale', '1.0')],
}


def run_cmm(data_dir, tag, skip_rerun, arm, cmm_bin):
    if skip_rerun:
        return None
    # The arm is part of the filename so that two arms can coexist on disk. They used to
    # share one name, which made a comparison impossible without saving the first table
    # by hand -- and a hand-saved table is the provenance this script exists to avoid.
    out = OUTDIR / f'{tag}_cmm_{arm}.csv'
    text = CMM_PAPER1
    for key, value in ARMS[arm]:
        text = _set_key(text, key, value)
    if arm != 'paper1':
        assert text != CMM_PAPER1, f'{arm}: no substitution applied'
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False, mode='w') as fh:
        fh.write(text.format(gps=str(data_dir / 'observations.csv'), out=str(out),
                             net=NET, ubodt=UBODT))
        cfg = Path(fh.name)
    r = subprocess.run([str(cmm_bin), str(cfg)], cwd=BASE,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    cfg.unlink()
    return out if r.returncode == 0 else None


def agree_lines(rerun_csv, corpus_csv):
    """Differing rows between a re-run and the corpus file, or None if not comparable.

    Keyed on (id, seq), not on line order: the writer runs under OpenMP and buffers, so
    two runs of the same config emit the same rows in a different order. Comparing raw
    lines reports 100 % disagreement for two identical runs.
    """
    if rerun_csv is None or not corpus_csv.exists():
        return None

    def keyed(p):
        out = {}
        with p.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f, delimiter=';'):
                out[(row['id'].strip(), int(float(row['seq']))) ] = row
        return out

    a, b = keyed(rerun_csv), keyed(corpus_csv)
    # `cpath` is the decisive column: a different matched edge is a different result, and
    # every score follows from it. The float columns are not compared -- a re-run of the
    # identical config reproduces them to the printed 8 decimals but not always bitwise.
    shared = set(a) & set(b)
    diff = sum(1 for k in shared if a[k]['cpath'] != b[k]['cpath']) + len(set(a) ^ set(b))
    return diff, len(a)


HDR = (f'{"cell":32s}{"n":>7s}{"acc %":>9s}{"ECE":>9s}{"MCE":>8s}{"Brier":>9s}'
       f'{"BSS":>9s}{"UNC":>9s}{"AUC":>8s}{"sep":>9s}')


def show(tag, cm, fm):
    print(f'{tag:32s}{cm["n"]:7d}{cm["acc"]:9.2f}{cm["ece"]:9.4f}{cm["mce"]:8.4f}'
          f'{cm["brier"]:9.4f}{cm["bss"]:+9.3f}{cm["unc"]:9.4f}{cm["auc"]:8.4f}{cm["sep"]:9.4f}')
    print(f'{"  " + "FMM":32s}{fm["n"]:7d}{fm["acc"]:9.2f}{fm["ece"]:9.4f}{fm["mce"]:8.4f}'
          f'{fm["brier"]:9.4f}{fm["bss"]:+9.3f}{fm["unc"]:9.4f}{fm["auc"]:8.4f}{fm["sep"]:9.4f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-rerun', action='store_true',
                    help='Evaluate the stored corpus files only; run no matcher.')
    ap.add_argument('--cmm-bin', type=Path, default=BASE / 'build' / 'cmm',
                    help='The CaMM binary to run. The default is the shipped tree, which '
                         'is the arm Section IV reports.')
    ap.add_argument('--cmm-arm', choices=sorted(ARMS), default='paper1',
                    help='paper1 (default) = the shipped model, i.e. the parameter block '
                         'of input/config/paper1_cmm.xml; it must reproduce the stored '
                         'cmm_result.csv bit-exactly. direction-off = the same model with '
                         'the heading term disabled, the scope ablation of Section IV-C. '
                         'no-cov-shape = Euclidean projection + isotropic per-epoch '
                         'emission, i.e. the 2x2 covariance used nowhere. Only paper1 is '
                         'a configuration the paper runs; the other two are ablations and '
                         'are expected to differ from the stored corpus file.')
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(HDR)
    print('-' * len(HDR))
    rows, verdicts = [], []
    for tag, sigma, d in cells():
        safe = tag.replace('/', '_')
        cmm_csv = ((d / 'cmm_result.csv') if args.skip_rerun
                   else run_cmm(d, safe, args.skip_rerun, args.cmm_arm, args.cmm_bin))
        fmm_csv, synth = run_fmm(d, safe, args.skip_rerun)
        yc, sc, nc = pairs(d, cmm_csv)
        cm = stats(yc, sc, nc)
        yf, sf, nf = pairs(d, fmm_csv)
        fm = stats(yf, sf, nf)
        show(tag, cm, fm)

        corpus_cmm = d / 'cmm_result.csv' if not args.skip_rerun else None
        cmm_agree = agree_lines(cmm_csv, corpus_cmm) if (corpus_cmm and corpus_cmm.exists()) else None
        fmm_agree = agree_lines(fmm_csv, d / 'fmm_result.csv')
        if fmm_agree:
            verdicts.append((tag, 'FMM', fmm_agree[0], fmm_agree[1], synth))
        if cmm_agree:
            verdicts.append((tag, f'CMM[{args.cmm_arm}]', cmm_agree[0], cmm_agree[1], False))
        for m in (cm, fm):
            m['err_mean'], m['err_p95'] = _point_err(d, cmm_csv if m is cm else fmm_csv)
        rows.append(dict(cell=tag, sigma=sigma, dir=str(d.relative_to(BASE)),
                         synth_traj=synth, cmm=cm, fmm=fm,
                         cmm_diff_lines=cmm_agree[0] if cmm_agree else None,
                         fmm_diff_lines=fmm_agree[0] if fmm_agree else None,
                         fmm_rerun_rows=fmm_agree[1] if fmm_agree else None))

    print('\n=== does the re-run reproduce the corpus file? ===')
    if args.cmm_arm != 'paper1':
        print(f'  (the stored cmm_result.csv is the paper1 model, so CMM[{args.cmm_arm}] is '
              f'EXPECTED to differ; only CMM[paper1] is a bit-exactness check)')
    for tag, arm, diff, nrows, synth in verdicts:
        extra = '  [traj input synthesised from observations.csv]' if synth else ''
        print(f'  {tag:32s} {arm}: ' +
              ('BIT-EXACT' if diff == 0 else f'DIFFERS on {diff}/{nrows} lines') + extra)

    # BSS degeneracy: the accuracy at which the constant's variance equals the matcher's
    # own Brier. Above it BSS is negative for any matcher, however good its ranking.
    print('\n=== Brier skill score vs the base-rate constant ===')
    print(f'{"cell":32s}{"acc %":>9s}{"p(1-p)":>10s}{"Brier":>9s}{"BSS":>9s}'
          f'{"crit acc %":>12s}{"AUC":>8s}')
    for r in rows:
        c = r['cmm']
        if not np.isfinite(c['brier']) or not np.isfinite(c['unc']) or c['unc'] <= 0:
            continue
        # p(1-p) = Brier has the root p* = (1 + sqrt(1 - 4*Brier)) / 2 in [0.5, 1].
        disc = 1.0 - 4.0 * c['brier']
        crit = 100.0 * (1.0 + np.sqrt(disc)) / 2.0 if disc >= 0 else float('nan')
        print(f'{r["cell"]:32s}{c["acc"]:9.2f}{c["unc"]:10.5f}{c["brier"]:9.4f}'
              f'{c["bss"]:+9.3f}{crit:12.2f}{c["auc"]:8.4f}')

    json.dump(rows, open(OUTDIR / 'sigma_sweep.json', 'w'), indent=1)
    print(f'\nwrote {OUTDIR / "sigma_sweep.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
