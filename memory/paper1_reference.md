# paper1_reference.md — on-demand detail moved out of CLAUDE.md

Moved 2026-10-12 per the read-on-demand rule. CLAUDE.md keeps the rules, the operating
point and the pointers; the detail below is preserved **verbatim** from CLAUDE.md as of
2026-10-12 (line ranges cited per section). Read the relevant section before quoting or
recomputing any number it covers.

---

## 1. Empirical reference detail (was CLAUDE.md §Empirical Performance Reference, lines 376–499)

**Six trajectories resolve the calibration contrasts and nothing else**
(`bootstrap_real_metrics.py`: unit = trajectory, 2000 replicates, seed 20260928, shared
draws; the exact $2^6$ sign-flip test runs alongside):

| contrast | point | 95 % CI | verdict |
|---|---|---|---|
| $\alpha\beta$ − baseline $\delta$, ECE | −0.186 | [−0.230, −0.152] | **resolved** (p = 1/64) |
| $\delta$ − baseline $\delta$, ECE | −0.152 | [−0.185, −0.127] | **resolved** (p = 1/64) |
| $\alpha$ − baseline $\alpha$, ECE | −0.326 | [−0.398, −0.222] | resolved |
| $\alpha\beta$ − baseline $\alpha\beta$, ECE | −0.190 | [−0.273, −0.074] | resolved — the fair pair |
| $\alpha\beta$ − $\delta$, ECE | −0.035 | [−0.055, −0.017] | resolved — the ablation |
| $\alpha\beta$ − $\delta$, BSS | +1.08 | [+0.52, +1.63] | resolved |
| $\alpha$ − $\delta$, ECE | −0.003 | [−0.005, +0.000] | unresolved |
| $\alpha\beta$ − baseline $\delta$, accuracy | +2.69 pp | [−0.62, +8.44] | unresolved (p = 14/64) |
| $\alpha\beta$ − baseline $\delta$, AUC | −0.043 | [−0.177, +0.207] | unresolved |
| $\alpha\beta$ − baseline $\delta$, pos. error | −0.34 m | [−0.88, −0.03] | resolved, marginal |

**The 2×3 grid reads as a grid.** Column by column the proposed observation model beats
the baseline on ECE at all three estimators. Row by row at the proposed model the trade is
monotone: $\delta \to \alpha \to \alpha\beta$ moves ECE $.042 \to .038 \to .007$ and
BSS $-0.81 \to -0.75 \to +0.27$ while AUC $.840 \to .838 \to .822$ and separation
$.465 \to .429 \to .261$ fall — every step toward calibration is a step away from
discrimination. At the baseline's model no such trade exists: smoothing repairs the
posterior but not the native score.

**Three findings travel with the numbers.** (1) The log score is negative-skill for every
arm against its own base-rate constant ($-0.08$ to $-0.67$ relative): $31\,\%$ of the
proposed arms' errors are reported above $0.9$ (baseline native $2\,\%$), so the mass is
calibrated and the tail is not. **That verdict is clip-dependent and the clip is
$10^{-12}$.** `verify_paper_numbers.log_score` clips saturated scores to
$[\epsilon, 1-\epsilon]$ with $\epsilon = 10^{-12}$, and until 2026-10-10 the manuscript
never said so. Recomputed: the smoothed arm scores $-0.173$, $-0.137$, $-0.101$, $-0.088$,
$-0.076$ at $\epsilon = 10^{-12}, 10^{-9}, 10^{-6}, 10^{-5}, 10^{-4}$, so it **crosses its
own constant between $10^{-6}$ and $10^{-5}$**; the filtering and Viterbi arms stay below
theirs at every clip, and the baseline's is clip-insensitive. The $67$ saturated-and-wrong
epochs each contribute $\ln(10^{-12}) = -27.6$ nats and carry the whole deficit. So the
clip-free evidence is the **count**: $67$ falsified certainty claims, $40$ of them on
traj 11 ($2.4\,\%$ of its pinned epochs), $14$ on 13 ($1.0\,\%$), $13$ on 22 ($0.85\,\%$),
and $0$ in traj 14/21/23's $5{,}132$ pinned epochs. The manuscript now discloses the clip
and reports the sensitivity row; do not quote the clipped mean as a property of the score.
(2) Post-hoc trajectory-LOO recalibration of the baseline
was measured (`audit_tail_and_recal.py`: isotonic reaches ECE 0.040 with BSS still
$-0.14$) and is deliberately not an arm of the paper — it lands at the filtering arm's
level, not the smoothed arm's. (3) On the non-saturated subset (score < 1, the $3{,}303$ epochs, $25.3\,\%$ of the
population) the filtering arm's ECE is $0.1116$ and the smoothed arm's $0.0135$ with AUC
$0.945$, BSS $+0.36$; there the smoothed arm also **beats that subset's own base-rate
constant on the log score, $-0.122$ against $-0.212$**, while the filtering arm stays below
it at $-0.368$. That subset is the paper's strongest single cell. The headline
$\text{ECE} = 0.007$ must be read against it: $95.8\,\%$ of the population sits in the one
bin $[0.9, 1]$, which carries $63\,\%$ of the headline, and inside it the $9{,}766$ pinned
epochs are over-confident by $0.69$ pp while the $2{,}755$ in $[0.9,1)$ are under-confident
by $0.44$ pp — two opposite errors, each comparable to the $0.007$ headline itself and
opposite in sign, so the two sub-populations partly cancel in the aggregate. The
smoothed arm's MCE $0.29$ is a bin holding **one epoch**. The smoothed arm's
false-alarm rate on correctly-matched epochs is $0.09\,\%$ at `tw < 0.5` and $0.00\,\%$
at `tw < 0.1`, against $3.4\,\%$ and $1.2\,\%$ for the filtering arm; the baseline's
native score alarms on $21.8\,\%$ and $2.6\,\%$ of its own correct epochs. Runtime:
cmm 1.80 s vs fmm 1.24 s on the corpus; the backward pass adds nothing measurable.

**Per-trajectory, the baseline's pooled AUC win is a pooling artefact**: it leads on
traj 11 (0.918 against 0.662) and 13 (0.854 against 0.759) and trails on 21, 22 and 23
(0.952 / 0.651 / 0.868 against 0.996 / 0.820 / 0.997). The smoothed posterior is still
saturated (68–75 % of epochs at exactly 1.000), which is why its low ECE is bought partly
by accuracy rather than by resolving residual error; it is the only audited score with a
positive Brier skill against the constant at its own base rate.

**CaMM output is content-deterministic but row-order nondeterministic** (OpenMP trajectory-block
order differs per run), so an `md5` check on `cmm.csv` fails spuriously and any positional or
`seq`-based join on it is unsafe; the FMM apps are byte-deterministic.

**The candidate-count row no longer differentiates.** Both matchers search the same 300 m
disk and retain the same 16 candidates, so the "3× candidate reduction" claimed by an
earlier formulation of this work is gone. That is the price of the row-5 choice, and it is
the reason the comparison is now a clean one: it isolates the observation model.

**The heading term's constants are a fixed point, and are fitted not assumed.**
`direction_kappa_scale` (c = 2(1−ρ)) and `direction_kappa_floor` (b, rad²) come from one
regression, σ_θ² = c·(nᵀΣn)/|v|² + b, over correctly-matched gap-free steps
(`experiments/scripts/fit_direction_kappa.py`). Because that sample is "epochs the matcher
got right", it depends on the matcher's parameters, so the constants are a fixed point of
match-then-fit; iterating converges in one step. Current values: **c = 0.227, b = 0.0084
rad² (5.2°), z(b) = 9.0, n = 11,792, ρ = 0.89**. The earlier c = 0.36 came from regressing
on 1/|v|² alone, which omits the per-epoch regressor and is biased — not a second estimate
of the same quantity.

**The term has a scope, and it is written into §III as Eq. (dir_snr).** The derivation
linearises θ ≈ nᵀv/|v|, which needs |v| ≫ √(nᵀΣn). Haikou's median ratio is 27.9; the
simulation corpus is 1.5–7.6 at every level (1 Hz at 12 m/s gives a ~12 m step while its
fabricated position σ is ~1.5σ_ρ), so **the corpus cannot evaluate the direction term** and
§IV says so rather than treating it as an ablation. Any corpus meant to test it needs a
higher sample rate or a smaller position noise.

**Three traps recorded here so they are not re-entered.**

1. *The previous table in this file was a debug configuration.* It carried
   `96.90 / 0.0437 / 0.6493` and `2.92` candidates/epoch, which is ladder row **7**
   (HPL-adaptive `k_search=1`, `delta_road=13.5` m, `phmi=1e-5`) — a configuration §III of
   the manuscript has not described since 2026-09-23, when that material moved to paper 2.
   The candidate count was the only visible symptom. A related episode in the same week
   recorded an off-network-state debug run (`bg_radius = 10000 m`) as the delivered
   configuration; that state has since been removed from the code entirely. When a number
   in this file changes, check `CovarianceMapMatchConfig::print()` in the run log before
   believing it.

2. *The join keys differ between the matchers.* CaMM result CSVs must be joined to
   `processed/aligned.csv` on `(id, timestamp)`; their `seq` column restarts at 0 for each
   sub-segment (traj 11 emits 2,695 rows whose `seq` stops at 1,239), so a `seq` join maps
   traj 11 onto the wrong epochs. FMM is the mirror image — its `seq` is a valid index into
   `uni_seq` and its `timestamp` is not per-epoch (traj 11: 2,696 rows, 25 distinct
   values). Both are implemented in `real_final_tables.py:load_gt/load_run` with the
   measurements that establish them.

3. *A check file has an anchor, and a wrong anchor passes.* `verify_paper_numbers.py`
   defaulted to `data/archive_20260928/final/` — the **anisotropic** run
   (`use_mahalanobis=true`, `isotropic_emission_scale=0`, per its `cmm.log`) — while the
   manuscript reports the scalar one. Every claim still passed, because the two runs differ
   by less than `check()`'s relative tolerance; the per-trajectory figures in `PAPER` were
   the anisotropic run's (96.69 / 98.74 / 97.74, All 98.13) where `tab:per_traj` prints the
   scalar run's (96.65 / 98.69 / 97.79, All 98.11). A regression target that is *one run
   off* is invisible in exactly the way a stale one is not: nothing is red, and the numbers
   it defends are not the numbers in the paper. Anchored 2026-09-30 on
   `data/archive_20260929/scalar/`, and `main()` now prints the **full** paths of the two
   runs it reads, so the anchor is visible in every run's output rather than inferred from
   the metric names. `exp_tp_attribution.py` had the same default and was fixed with it.

---

## 2. Calibration-scale derivation (was CLAUDE.md Level-2, lines 213–215)

This reverses an earlier calibration, and the reversal is the load-bearing part. The model
consumes only the quadratic form $\mathbf{d}^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}\mathbf{d}$
— $\log|\boldsymbol{\Sigma}|$ cancels in the same-epoch softmax — so the natural scale
criterion is $k=\sqrt{\mathbb{E}[\mathbf{d}^{\mathrm{T}}\boldsymbol{\Sigma}_{\mathrm{gst}}^{-1}\mathbf{d}]/2}$,
which over 15,034 epochs measured $3.0008$. That factor was a **compensation, not a
calibration**. Stripped in stages over the same population: $1.8683$ after removing two
ground-truth defects in `build_real_data_gt.py` (a $\pm1$ s RTK substitution, and the
11.3 % of "RTK truth" rows that are not fixed solutions), then $0.9430$ after also
removing the antenna lever arm (the SPP and RTK antenna phase centres are not co-located on
this vehicle) and a 1.0--2.7 m fixed-azimuth SPP bias. What is left is ~1, and an
independent route agrees: raw GST median $\sigma_\mathrm{major}$ 0.815 m against a
post-removal residual of ~0.707 m gives 0.87, i.e. the raw GST is already right and
slightly conservative. Donkey.Ning's 2026-09-16 ruling is $k = 1$. The constant is retained
as the single definition that `apply_gst_calibration.py` imports, and at $k = 1$ that
script is the identity.

Do not restore $3.0008$ without re-deriving it: with the ground-truth defects and the lever
arm removed, it inflates the covariance by $9\times$ and moves the reported ECE and AUC
with it.

---

## 3. Background-state removal measurement (was CLAUDE.md Level-3, line 224)

An off-road **background state** $p_{\text{bg}} = 0.1$ used to be listed here as a further
normalization step. It has been removed from the code: because both the emission and the
transition normalisation happen before the softmax that produces trustworthiness, the
constant factor cancels, and the only place it did not cancel was layer initialisation,
where it was miscounted into $K$. Measured on the Haikou set it changed the reported
trustworthiness of exactly 8 of 16155 epochs — the first epoch of each trajectory and
sub-segment — and no matched path. It was also actively harmful: appending it made a
zero-candidate epoch look non-empty, which let an off-road epoch enter the Viterbi layer
and permanently stall the sub-segment.

---

## 4. Mechanisms deliberately absent — full table (was CLAUDE.md lines 245–255)

The CMM implementation is a direct translation of the paper's §III formulas. The following
have been removed so that no behaviour needs to be justified outside those formulas; do not
reintroduce them without a corresponding paper change.

| Removed | Why |
|---|---|
| Adaptive softmax temperature | Was a no-op on the real data (ECE $0.0410 \to 0.0400$, AUC $0.7204 \to 0.7203$, both measured on an older ground truth); never described in the paper. |
| Fixed-lag smoothing (`lag_steps`, `apply_lag_smoothing`, `flush_lag_buffer`) | Never described in the paper, and it *degrades* TW calibration on real data with tight HPL (ECE $0.04 \to 0.26$ at $L=20$). Use $L=0$ always. |
| Bounded radius fallback (`should_expand_search_radius`, `MAX_SEARCH_RADIUS_DOUBLINGS`, `radius_doublings`, `fixed_search_radius`) | Was a second radius rule that doubled the disk until `min_candidates` was reached, so it fired on exactly the epochs the covariance model was least able to justify, and the candidates it recovered needed a PHMI weight the model had no principled way to assign. An epoch with no candidate in the disk is now dropped instead. |
| Protection-level multiplier (search and PHMI boundary) | $r_i = k_\text{search}\mathrm{HPL}_i + \delta_\text{road}$ with $k_\text{search} = 1$ and $\delta_\text{road}$ the documented lane/map allowance. A further multiplicative knob on the radius, or a second independent multiplier on the PHMI classification boundary, is what the paper does not have. |
| `MIN_SIGMA` floor / `map_error_std` / `min_gps_error_degrees` | Replaced by the covariance validity test plus the isotropic fallback. |
| Off-road background state | See §3 above. |
| Sequential H0 hypothesis test (`h0_prior_log_odds`, `h0_lambda`) | Never described in the paper. Removed in full: the recursion, `MatchedCandidate::h0_lambda`, `ResultConfig::write_h0_lambda`, the `mm_writer` column and the `<h0_lambda/>` field in the configs. The column was constant 1.0 for all 16155 real-data epochs; removing it leaves the other 16 columns bit-identical. |

---

## 5. Build guards (was CLAUDE.md §Paper build check, lines 519–557)

Guards, all worth running before a build is called good:

```bash
python3 - <<'EOF'
import re
p = 'docs/calib/calib.tex'
s = open(p, encoding='utf-8').read()
labels = set(re.findall(r'\\label\{([^}]*)\}', s))
refs = set(re.findall(r'\\ref\{([^}]*)\}', s)) | set(re.findall(r'\\eqref\{([^}]*)\}', s))
print('unresolved refs:', sorted(r for r in refs if r not in labels) or 'none')
for env in ('table', 'table*', 'figure', 'figure*', 'equation', 'align', 'algorithm'):
    o = len(re.findall(r'\\begin\{' + re.escape(env) + r'\}', s))
    c = len(re.findall(r'\\end\{' + re.escape(env) + r'\}', s))
    if o != c:
        print(f'UNBALANCED {env}: {o} begin, {c} end')
# a real TAB character -- \t in a regex is not the same thing
for i, l in enumerate(s.split('\n'), 1):
    if '\t' in l:
        print(f'TAB on line {i}: {l.strip()[:70]}')
# a doubled backslash before a macro name (\\mathbf, \\Sigma, ...)
for i, l in enumerate(s.split('\n'), 1):
    if re.search(r'\\\\[A-Za-z]', l):
        print(f'DOUBLE BACKSLASH on line {i}: {l.strip()[:70]}')
# a line that repeats the tail of the line before it
ls = s.split('\n')
for i in range(1, len(ls)):
    a, b = ls[i - 1].strip(), ls[i].strip()
    if b and a.endswith(b.split()[0]) and len(b.split()) > 2 and b[0].isupper():
        print(f'possible duplicated fragment at line {i + 1}: {b[:60]}')
# equations outside Section III is a standing requirement of this paper
a, b = s.index(r'\section{Trustworthiness for Map Matching}'), s.index(r'\section{Simulation Study}')
print('equations outside Section III:',
      len(re.findall(r'\\begin\{(equation|align)\}', s[:a] + s[b:])))
EOF
```

An unresolved-reference count of zero from the log means the *existing* `\ref`s resolved; it
says nothing about a `\ref` that is no longer there. The same logic covers everything in the
table: a clean log says the source is *parseable*, not that it is *correct*.
