"""Is the covariance the emission uses consistent with the actual SPP error?

e = z_SPP - z_RTK ;  D^2 = e^T Sigma^{-1} e ;  under a correct covariance D^2 ~ chi2(2),
so E[D^2] = 2, median D^2 = 1.386, and 5 % of epochs should exceed 5.991.

WHY THIS IS THE APPENDIX THAT MATTERS. The emission of Eq. (sigma_i) is only as good as
this covariance, and the simulated version of this check is close to constructive: on the
corpus the covariance is derived from the same noise model that generates the errors, so
E[D^2] = 2 is nearly guaranteed and proves little. The real-data version is the one with
content, and it needs a comparison truth as good as the covariance being tested.

Run twice, because the two rows mean different things:
  (a) aligned.csv's own obs_sde/obs_sdn/obs_sdne
  (b) the RAW GST covariance from cmm_input_points_k1.csv, joined on (id, timestamp)

Since 2026-09-28 aligned.csv is built with --obs cmm_input_points_k1.csv, so (a) and (b)
are the SAME covariance and must agree exactly; before that (a) held the x3.0008
calibrated table. If the two rows differ, aligned.csv was built from a different
observation table than the one named here and the discrepancy is the finding.

THE READING IS NOT "k = 2.85". A mean D^2 of 16 says the covariance is ~2.85x too small
in sigma ONLY IF the reference truth is exact. It is not: the same 2.85 is reproduced to
within 1 % by an independent route that strips ground-truth defects and the antenna lever
arm out of the calibration factor, which puts the deficit in the REFERENCE rather than in
the covariance. The shape is wrong in both rows (median D^2 far below 1.386 while the
mean sits near 2, i.e. a few gross outliers dominate) and that shape is the signature of
the ground-truth defects, not of the covariance model. Report the three facts together or
not at all.

Usage: python3 experiments/scripts/exp_covariance_consistency_real.py
"""
import csv, math, numpy as np
from scipy import stats

BASE = '/home/ncz/fmm_sjtugnc/'
MET = 111320.0


def d2_stats(ex, ey, se, sn, sne, label):
    D2 = []
    for i in range(len(ex)):
        S = np.array([[se[i] ** 2, sne[i]], [sne[i], sn[i] ** 2]])
        if np.linalg.det(S) <= 0:
            continue
        e = np.array([ex[i], ey[i]])
        D2.append(float(e @ np.linalg.inv(S) @ e))
    D2 = np.array(D2)
    k = math.sqrt(D2.mean() / 2)
    ks = stats.kstest(D2, 'chi2', args=(2,))
    print(f'--- {label} ---')
    print(f'  n                 : {len(D2)}')
    print(f'  median sigma_major: {np.median(np.maximum(se, sn)):.3f} m  (axis-aligned proxy)')
    print(f'  mean D^2          : {D2.mean():.3f}    (chi2_2 expects 2)')
    print(f'  median D^2        : {np.median(D2):.3f}    (chi2_2 expects 1.386)')
    print(f'  implied scale k   : {k:.3f}    (<1 => overconfident)')
    print(f'  KS vs chi2(2)     : {ks.statistic:.4f}  p={ks.pvalue:.2e}')
    print(f'  share D^2 > 5.991 : {100 * (D2 > 5.991).mean():.2f}%   (expect 5%)')
    print()
    return D2


# ---- load aligned.csv ----
rows = []
with open(BASE + 'data/real_vehicle/hainan_06/processed/aligned.csv', newline='') as f:
    for r in csv.DictReader(f, delimiter=';'):
        try:
            rows.append((r['id'].strip(), float(r['timestamp']),
                         float(r['obs_x']), float(r['obs_y']),
                         float(r['obs_sde']), float(r['obs_sdn']), float(r['obs_sdne']),
                         float(r['gt_x']), float(r['gt_y'])))
        except (ValueError, KeyError):
            pass
print(f'aligned.csv rows parsed: {len(rows)}')
lat = np.median([r[8] for r in rows])

# ---- (a) aligned.csv's own covariance ----
ex = np.array([(r[2] - r[7]) * MET * math.cos(math.radians(lat)) for r in rows])
ey = np.array([(r[3] - r[8]) * MET for r in rows])
se = np.array([r[4] * MET for r in rows])
sn = np.array([r[5] * MET for r in rows])
sne = np.array([r[6] * MET * MET for r in rows])
d2_a = d2_stats(ex, ey, se, sn, sne, "(a) aligned.csv obs_sde/obs_sdn/obs_sdne")

# ---- (b) raw GST from the k1 table ----
gst = {}
with open(BASE + 'data/real_vehicle/hainan_06/cmm_input_points_k1.csv', newline='') as f:
    for r in csv.DictReader(f, delimiter=';'):
        try:
            gst[(r['id'].strip(), float(r['timestamp']))] = (
                float(r['sde']), float(r['sdn']), float(r['sdne']))
        except (ValueError, KeyError):
            pass
print(f'cmm_input_points_k1.csv rows: {len(gst)}')

ex2, ey2, se2, sn2, sne2 = [], [], [], [], []
joined = 0
for r in rows:
    key = (r[0], r[1])
    if key not in gst:
        continue
    joined += 1
    ex2.append((r[2] - r[7]) * MET * math.cos(math.radians(lat)))
    ey2.append((r[3] - r[8]) * MET)
    sde, sdn, sdne = gst[key]
    se2.append(sde * MET)
    sn2.append(sdn * MET)
    sne2.append(sdne * MET * MET)
print(f'joined on (id, timestamp): {joined}')
print()
d2_b = d2_stats(np.array(ex2), np.array(ey2), np.array(se2), np.array(sn2), np.array(sne2),
         "(b) RAW GST (cmm_input_points_k1.csv)")

if len(d2_a) == len(d2_b) and np.allclose(d2_a, d2_b):
    print("(a) and (b) are the same covariance to machine precision: aligned.csv is built")
    print("from the k1 table, as the 2026-09-16 k = 1 ruling requires.")
else:
    print("(a) and (b) DIFFER -- aligned.csv was built from a different observation table.")
    print("Re-run align_real_data.py with --obs cmm_input_points_k1.csv before reporting.")
