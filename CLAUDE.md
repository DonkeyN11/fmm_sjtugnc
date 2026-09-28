# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FMM (Fast Map Matching) is an open-source C++17 framework for matching noisy GPS trajectories to road networks using Hidden Markov Models with precomputation. This fork extends the original [FMM](https://github.com/cyang-kth/fmm) with **Covariance Map Matching (CMM)** — integrating GNSS covariance matrices and Protection Levels into the matching objective.

## Build System

### Conda Environment (Required)

The build takes GDAL and Boost from the conda **base** environment:

```bash
conda activate base      # /home/ncz/anaconda3 — this is the one that has GDAL
```

There is no `fmm_env`. This file named one until 2026-09-22; the name does not
resolve, and `conda activate fmm_env` fails before the build starts. The four other
envs on this machine (`ox`, `sam_road`, `sam_road2`, `sat2graph`) have neither GDAL
nor Boost, so base is not a preference among options — it is the only one that
provides the dependencies. The existing `build/CMakeCache.txt` records
`CONDA_PREFIX=/home/ncz/anaconda3`, `GDAL_LIBRARY=/home/ncz/anaconda3/lib/libgdal.so`,
confirming which environment the checked-in build was made against.

**SWIG does not come from conda here.** It is the system package at `/usr/bin/swig`
(4.0.2), so the Python bindings build without any conda env contribution.

CMakeLists.txt detects `CONDA_PREFIX` and uses conda-provided headers/libraries with `-isystem`.

### Build Commands

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**No `make install` needed for development** — binaries go to `build/` directly via `CMAKE_RUNTIME_OUTPUT_DIRECTORY`.

### Key Build Targets

| Binary | Source | Purpose |
|--------|--------|---------|
| `fmm` | `src/app/fmm.cpp` | FMM command-line tool |
| `cmm` | `src/app/cmm.cpp` | CMM command-line tool |
| `stmatch` | `src/app/stmatch.cpp` | Spatio-temporal matching |
| `ubodt_gen` | `src/app/ubodt_gen_app.cpp` | UBODT precomputation |
| `ubodt_converter` | `src/app/ubodt_converter_app.cpp` | UBODT format converter |
| `ubodt_daemon` | `src/app/ubodt_daemon.cpp` | UBODT memory-resident daemon |
| `h3mm` | `src/app/h3mm.cpp` | H3 hexagon matching |
| `libFMMLIB.so` | All object libs | Shared library + Python bindings base |

### Python Bindings

Built automatically via SWIG into `build/python/` — produces `fmm.py` and `_fmm.so`. Import with:

```python
import sys; sys.path.insert(0, 'build/python')
from fmm import Network, NetworkGraph, UBODT, FastMapMatch, FastMapMatchConfig
from fmm import CovarianceMapMatch, CovarianceMapMatchConfig
```

SWIG renames the constructor arguments with an `_arg` suffix, so the keyword form is `CovarianceMapMatchConfig(k_arg=16, min_candidates_arg=1)` — `k=16` raises `TypeError`. See [CMM_README.md](CMM_README.md) for the CMM C++ API.

### Testing

There is no automated `make test` target. C++ tests go in `tests/`, Python tests in `python/experiments/`:

```bash
# Build and run a C++ test (add to CMakeLists.txt manually)
cd build && make performance_test && ./performance_test

# Run Python tests
cd python/experiments && python exp1_lag_sweep.py
```

## Source Architecture

### Layered Structure

```
src/
├── core/          Basic geometry types (Point, LineString, Trajectory)
├── config/        Configuration classes (GPSConfig, NetworkConfig, ResultConfig)
├── network/       Road network (Network), graph (NetworkGraph), RTree index
├── io/            GPS readers (CSV point/trajectory, Shapefile), result writer
├── algorithm/     Geometric algorithms (shortest path, linestring ops)
├── util/          Logging, string utilities
├── mm/
│   ├── mm_type.hpp           Core MM types (Candidate, MatchedCandidate, MatchResult)
│   ├── transition_graph.hpp  HMM transition graph + Viterbi inference
│   ├── composite_graph.hpp   Composite graph for combined networks
│   ├── fmm/                  Fast Map Matching (UBODT-based HMM)
│   ├── cmm/                  Covariance Map Matching (uncertainty-aware)
│   ├── stmatch/              Spatio-Temporal Matching (no precomputation)
│   └── h3mm/                 H3 hexagon-based matching
├── app/           CLI entry points (fmm.cpp, cmm.cpp, stmatch.cpp, etc.)
└── python/        SWIG interface definition (fmm.i) and pyfmm.hpp types
```

### Key Data Flow

1. **Network** loads shapefile → builds RTree spatial index, maps edge/node IDs ↔ indices
2. **GPSReader** reads trajectories from CSV/Shapefile (or CMM aggregated CSV with covariance JSON)
3. **UBODT** (precomputed) provides O(1) shortest-path-distance lookups between any two network nodes
4. **Candidate search** uses RTree KNN within a radius (both matchers fixed at 300 m at paper 1's operating point; CMM *also* supports the HPL-adaptive $k_\text{search}\mathrm{HPL}_i + \delta_\text{road}$ disk, which is paper 2's)
5. **TransitionGraph** builds layered HMM graph; **Viterbi** finds optimal path
6. **MatchResult** is returned with full path (opath, cpath, mgeom, trustworthiness scores)

## Theoretical Framework: GNSS-Consistent HMM Map Matching

This project implements the **Trustworthy Map Matching (TMM)** framework described in the accompanying paper ([docs/](docs/)). The core idea is a three-level integration of GNSS stochastic information into the HMM pipeline: (1) HPL-based candidate search, (2) covariance-based emission probability, and (3) filtering posterior as trustworthiness.

### GNSS Stochastic Model (Foundation)

**Position covariance from WLS.** GNSS pseudorange observations are processed via Weighted Least-Squares:
$$
\widehat{\Delta\mathbf{x}} = (\mathbf{H}^{\mathrm{T}}\mathbf{W}\mathbf{H})^{-1}\mathbf{H}^{\mathrm{T}}\mathbf{W}\mathbf{y},
\qquad
\boldsymbol{\Sigma}_{x} = (\mathbf{H}^{\mathrm{T}}\mathbf{W}\mathbf{H})^{-1}
$$
The $2\times 2$ horizontal block $\boldsymbol{\Sigma}_i = \begin{bmatrix} \sigma_E^2 & \sigma_{EN} \\ \sigma_{EN} & \sigma_N^2 \end{bmatrix}$ drives the anisotropic emission model. In practice, $\boldsymbol{\Sigma}_i$ is parsed from NMEA `GST` messages (via `experiments/scripts/extract_spp_for_cmm.py`, which applies the calibration described below) or computed from RINEX observations in `experiments/scripts/compute_raim_pl.py`.

**Protection Level from RAIM.** The Horizontal Protection Level (HPL) bounds position error at integrity risk $10^{-5}$:
$$
\mathrm{HPL} = K \cdot \sigma_{\mathrm{major}}, \qquad
\sigma_{\mathrm{major}} = \sqrt{ \frac{\sigma_E^2 + \sigma_N^2}{2} + \sqrt{ \left( \frac{\sigma_E^2 - \sigma_N^2}{2} \right)^2 + \sigma_{EN}^2 } }
$$
$K$ is the containment factor for the stated integrity risk, and it depends on which tail the risk is apportioned over:

| risk | 1-D one-sided $Q(K)$ | 1-D two-sided $2Q(K) = \chi^2_1$ | $\chi^2_2$ (horizontal, circular) | $\chi^2_3$ |
|---|---|---|---|---|
| $10^{-5}$ | 4.2649 | 4.4172 | **4.7985** | 5.0894 |
| $10^{-7}$ | 5.1993 | 5.3267 | 5.6777 | 5.9503 |

HPL is a horizontal containment radius, so the $\chi^2_2$ column is the one it uses: **$K = 4.7985$ at $10^{-5}$**. The value $K = 5.33$ that used to be quoted here is the 1-D two-sided factor for $10^{-7}$ (a $9.82\times10^{-8}$ tail), not for $10^{-5}$ — two decades of risk away. For an anisotropic ellipse the $1-10^{-5}$ containment radius relative to $\sigma_{\mathrm{major}}$ is *smaller* than the circular case, falling to 3.31 at an axis ratio of 1:100, so 4.7985 is conservative there. RAIM performs a global $\chi^2$ consistency test with iterative Fault Detection and Exclusion (FDE), requiring a minimum of 5 remaining satellites. **Do NOT use simple bounding boxes** (`max(sde, sdn)`) — the eigenvalue-based semi-major axis is the correct formula.

### Level 1: Candidate Search

CMM supports two search disks and **paper 1 uses the fixed one**. The adaptive,
geometry-driven disk is
$$
r_i = k_\text{search} \, \mathrm{HPL}_i + \delta_\text{road},
$$
with $k_\text{search}=1$ and $\delta_\text{road}=13.5$ m at the HPL-adaptive operating
point, i.e. ablation-ladder row 7. **The manuscript's §V is row 5: $k_\text{search}=0$
with $\delta_\text{road}=300$ m, a fixed 300 m disk, the same one the FMM baseline
searches.** The adaptive form belongs to paper 2 (integrity); the fixed disk is what makes
the real-data comparison isolate the observation model. Candidate road segments are all
edges intersecting the disk. Under good satellite geometry HPL is small (~14 m at
$\sigma=1$ m); under degraded geometry HPL grows proportionally. $k_\text{search}$ and
$\delta_\text{road}$ are the two global constants that replace what used to be a per-epoch
tuning loop; $\delta_\text{road}$ is an allowance for the two quantities the HPL does not bound — the lateral offset of the driven lane from the mapped centreline, and map georeferencing error. The paper's operating points are $k_\text{search} = 1$ with $\delta_\text{road} = 3.5$ m in simulation (the national urban lane width, CJJ 37--2012) and $\delta_\text{road} = 13.5$ m on the real data (that lane width plus a 10 m map-error allowance, converted to degrees by the $111{,}320$ m/deg convention). The implementation is the pure static `effective_search_radius()` ([src/mm/cmm/cmm_algorithm.cpp:1840](src/mm/cmm/cmm_algorithm.cpp#L1840)).

**An epoch with no candidate is dropped.** The construction above presumes the network intersects the search disk at all, which can fail (an optimistic $\mathrm{HPL}_i$, a multipath-displaced fix, or a road absent from the network). A bounded radius fallback used to double the radius here until a floor was reached (`should_expand_search_radius`, `MAX_SEARCH_RADIUS_DOUBLINGS = 8`, `fixed_search_radius`); **it has been removed in full**, because it was a second, undocumented radius rule firing on exactly the epochs the model was least able to justify. An epoch whose disk admits no road segment is now simply not matched: no row is emitted for it and the remaining epochs are matched as before. On the Haikou set this drops 401 rows, none of which carries a ground-truth label, so the evaluable population is unchanged at 13,123 epochs.

`min_candidates` survives with a narrower meaning than it had under the fallback. It is now a floor on how many candidates are *kept* from the disk, not a trigger for enlarging it: if fewer than `min_candidates` candidates survive the metric-based selection, the nearest remaining candidates from the pool are appended by Euclidean distance ([src/mm/cmm/cmm_algorithm.cpp:1063](src/mm/cmm/cmm_algorithm.cpp#L1063)). The search radius is never changed by it.

### Level 2: Covariance-Based Emission Probability

**Mahalanobis projection.** Candidate points are generated by projecting the GNSS fix onto each candidate road segment using the Mahalanobis metric (not Euclidean):
$$
x_{i,j} = \arg\min_{x \in s_j} (z_i - x)^{\mathrm{T}} \boldsymbol{\Sigma}_i^{-1} (z_i - x)
$$
Geometric interpretation via Cholesky factorization $\boldsymbol{\Sigma}_i = \mathbf{L}_i \mathbf{L}_i^{\mathrm{T}}$: the whitening transformation $\tilde{z} = \mathbf{L}^{-1} z$ maps the anisotropic error ellipse to a unit circle, so Mahalanobis projection in the original space is exactly Euclidean projection in the whitened space.

**Emission probability.** Modeled as a bivariate Gaussian governed by the GNSS covariance:
$$
p(z_i \mid x_{i,j}) = \frac{1}{2\pi \sqrt{|\boldsymbol{\Sigma}_i|}} \exp\!\left[ -\frac{1}{2} \mathbf{d}_{i,j}^{\mathrm{T}} \boldsymbol{\Sigma}_i^{-1} \mathbf{d}_{i,j} \right]
$$
where $\mathbf{d}_{i,j} = z_i - x_{i,j}$. Computed in log space as $-\tfrac{1}{2}(\log 2\pi + \log |\boldsymbol{\Sigma}_i| + \mathbf{d}_{i,j}^{\mathrm{T}}\boldsymbol{\Sigma}_i^{-1}\mathbf{d}_{i,j})$ at [src/mm/cmm/cmm_algorithm.cpp:920](src/mm/cmm/cmm_algorithm.cpp#L920). There is **no floor on the standard deviation** — that was `MIN_SIGMA`, replaced by the validity test plus the isotropic fallback above. There **is** one inflating term, and it is off by default: `map_error_sigma` adds $s^2\mathbf{I}$ in the *variance* to account for map registration, and the paper's operating point sets it to 10.6 m, so at that operating point the emission covariance is $\boldsymbol{\Sigma}_i + s^2\mathbf{I}$ rather than $\boldsymbol{\Sigma}_i$. An unadorned read of the formula above describes the default configuration, not the reported one.

**Covariance validity.** The covariance is used exactly as reported, however small. If it is not a valid $2\times2$ Gaussian — `sde`/`sdn` non-finite or non-positive, or $s_{de}^2 s_{dn}^2 - s_{dne}^2 \le 0$ — then and only then is it replaced wholesale by a documented isotropic 5 m fallback (`fallback_covariance()`), converted to degrees when the network is geographic. The test is `is_covariance_usable()`; the two cases are mutually exclusive and there is no third path. This replaced a hard-coded `MIN_SIGMA` floor that did not check validity but rescaled *every* small covariance up to a fixed minimum — it fired on 99.98 % of the Haikou epochs with a median scale factor of 7.22, so it was the model rather than a guard rail, and it propagated into the direction penalty, which is inversely proportional to the standard deviation.

**Data-side calibration: none is applied.** `extract_spp_for_cmm.py` carries a `GST_COVARIANCE_SCALE` whose value is now **1.0**, so the raw NMEA `GST` covariance is written through unscaled into the CMM input table (`cmm_input_points_k1.csv`, the table every current real-data result consumes).

This reverses an earlier calibration, and the reversal is the load-bearing part. The model consumes only the quadratic form $\mathbf{d}^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}\mathbf{d}$ — $\log|\boldsymbol{\Sigma}|$ cancels in the same-epoch softmax — so the natural scale criterion is $k=\sqrt{\mathbb{E}[\mathbf{d}^{\mathrm{T}}\boldsymbol{\Sigma}_{\mathrm{gst}}^{-1}\mathbf{d}]/2}$, which over 15,034 epochs measured $3.0008$. That factor was a **compensation, not a calibration**. Stripped in stages over the same population: $1.8683$ after removing two ground-truth defects in `build_real_data_gt.py` (a $\pm1$ s RTK substitution, and the 11.3 % of "RTK truth" rows that are not fixed solutions), then $0.9430$ after also removing the antenna lever arm (the SPP and RTK antenna phase centres are not co-located on this vehicle) and a 1.0--2.7 m fixed-azimuth SPP bias. What is left is ~1, and an independent route agrees: raw GST median $\sigma_\mathrm{major}$ 0.815 m against a post-removal residual of ~0.707 m gives 0.87, i.e. the raw GST is already right and slightly conservative. Donkey.Ning's 2026-09-16 ruling is $k = 1$. The constant is retained as the single definition that `apply_gst_calibration.py` imports, and at $k = 1$ that script is the identity.

Do not restore $3.0008$ without re-deriving it: with the ground-truth defects and the lever arm removed, it inflates the covariance by $9\times$ and moves the reported ECE and AUC with it.

### Level 3: Trustworthiness as Calibrated Posterior

**Probabilistic normalization (critical for calibration).** Three normalization steps ensure valid probabilities:
1. **PHMI-grouped emission normalization**: candidates are partitioned at the protection level into $\mathcal{C}^{\mathrm{in}}_i = \{j : d_{i,j} \le \mathrm{HPL}_i\}$ and its complement; each group is normalized over its own members and scaled by the probability that it contains the truth, $(1-\mathrm{PHMI})$ and $\mathrm{PHMI}$ respectively, so the layer sums to 1 when both groups are non-empty (or when only the inside group is) and to $\mathrm{PHMI}$ when only the outside group is. That last case is not exotic: whenever $\delta_\text{road}$ puts an annulus outside $\mathrm{HPL}_i$ inside the disk, an epoch can end up with candidates that are all outside the bound, and it then carries total emission mass $\mathrm{PHMI}$ — the epoch is kept, but it is not claimed to be explained. The per-epoch trustworthiness is a ratio at a single epoch, so this mass cancels there and acts only on the choice of path. With `phmi = 0` the grouping degenerates to a plain layer-wise softmax.
2. **Row-normalized transitions**: $t_{a \to b} = w_{a \to b} / \sum_j w_{a \to j}$ where $w_{a \to b} = \min(1, d_{\text{gnss}} / d_{\text{road}})$, ensuring $\sum_j t_{a \to j} = 1$.
3. **Uniform initial prior**: $\pi(i) = 1/K$ for $K$ road candidates.

An off-road **background state** $p_{\text{bg}} = 0.1$ used to be listed here as a further normalization step. It has been removed from the code: because both the emission and the transition normalisation happen before the softmax that produces trustworthiness, the constant factor cancels, and the only place it did not cancel was layer initialisation, where it was miscounted into $K$. Measured on the Haikou set it changed the reported trustworthiness of exactly 8 of 16155 epochs — the first epoch of each trajectory and sub-segment — and no matched path. It was also actively harmful: appending it made a zero-candidate epoch look non-empty, which let an off-road epoch enter the Viterbi layer and permanently stall the sub-segment. An epoch with no road candidate is now simply skipped.

**Forward algorithm (trustworthiness computation).** The per-epoch trustworthiness is the filtering posterior of the Viterbi-optimal candidate $i^*$:
$$
\text{tw}_t = P(x_t = i^* \mid z_{1:t}) = \frac{\alpha_t(i^*)}{\sum_j \alpha_t(j)} = \text{softmax}(\log \alpha_t)_{i^*}
$$
where $\alpha_t$ is the forward probability computed via the standard HMM log-space recursion:
$$
\log \alpha_t(b) = \log \sum_a \exp(\log \alpha_{t-1}(a) + \log t_{a \to b}) + \log p(z_t \mid x_t^{(b)})
$$
Both forward ($\alpha_t$) and Viterbi ($\delta_t$) recursions are computed. The forward sum accounts for all competing paths; the Viterbi max tracks the single best path. Trustworthiness uses the forward sum in the denominator for proper probabilistic semantics.

**Information-theoretic metrics (per-epoch).** Two quantities complement trustworthiness:
- **Posterior entropy** $H_t = -\sum_i p_i \log_2 p_i$ (remaining ambiguity)
- **Information gain** $\Delta H_t = H_{\text{prior}}^t - H_t$ (uncertainty resolved by the current observation)
These are stored in `MatchedCandidate::posterior_entropy` and `MatchedCandidate::delta_entropy`.

Note from the paper (README.md §Trustworthiness Evaluation): $\Delta H$ is information gain, not directly "matching confidence." Small $\Delta H$ + small posterior entropy → high confidence. Small $\Delta H$ + large posterior entropy → noisy GNSS, low confidence. $\Delta H$ alone cannot distinguish these cases — always use both metrics together.

### Mechanisms deliberately absent

The CMM implementation is a direct translation of the paper's §III formulas. The following have been removed so that no behaviour needs to be justified outside those formulas; do not reintroduce them without a corresponding paper change.

| Removed | Why |
|---|---|
| Adaptive softmax temperature | Was a no-op on the real data (ECE $0.0410 \to 0.0400$, AUC $0.7204 \to 0.7203$, both measured on an older ground truth); never described in the paper. |
| Fixed-lag smoothing (`lag_steps`, `apply_lag_smoothing`, `flush_lag_buffer`) | Never described in the paper, and it *degrades* TW calibration on real data with tight HPL (ECE $0.04 \to 0.26$ at $L=20$). Use $L=0$ always. |
| Bounded radius fallback (`should_expand_search_radius`, `MAX_SEARCH_RADIUS_DOUBLINGS`, `radius_doublings`, `fixed_search_radius`) | Was a second radius rule that doubled the disk until `min_candidates` was reached, so it fired on exactly the epochs the covariance model was least able to justify, and the candidates it recovered needed a PHMI weight the model had no principled way to assign. An epoch with no candidate in the disk is now dropped instead. |
| Protection-level multiplier (search and PHMI boundary) | $r_i = k_\text{search}\mathrm{HPL}_i + \delta_\text{road}$ with $k_\text{search} = 1$ and $\delta_\text{road}$ the documented lane/map allowance. A further multiplicative knob on the radius, or a second independent multiplier on the PHMI classification boundary, is what the paper does not have. |
| `MIN_SIGMA` floor / `map_error_std` / `min_gps_error_degrees` | Replaced by the covariance validity test plus the isotropic fallback above. |
| Off-road background state | See the note under Level 3. |
| Sequential H0 hypothesis test (`h0_prior_log_odds`, `h0_lambda`) | Never described in the paper. Removed in full: the recursion, `MatchedCandidate::h0_lambda`, `ResultConfig::write_h0_lambda`, the `mm_writer` column and the `<h0_lambda/>` field in the configs. The column was constant 1.0 for all 16155 real-data epochs; removing it leaves the other 16 columns bit-identical. |

---

## FMM vs CMM: Key Differences

| Aspect | FMM | CMM |
|--------|-----|-----|
| Emission model | Isotropic Gaussian $\mathcal{N}(0,\sigma^2 I)$ | Anisotropic Mahalanobis (covariance-consistent) |
| Candidate search | Fixed radius | HPL-based adaptive region |
| Candidate projection | Orthogonal (Euclidean) | Mahalanobis (statistically optimal) |
| GPS error | Constant scalar $\sigma$ | Per-epoch covariance $\boldsymbol{\Sigma}_i$ |
| Trustworthiness | Filtering posterior via forward algorithm (same estimand as CMM by default; `tw_use_viterbi` switches it to the Viterbi score) | Filtering posterior via forward algorithm |
| Prerequisites | UBODT only | UBODT + covariance + protection level per epoch |
| Calibration (ECE) | 0.190 (under-confident) | 0.044 |
| Real accuracy | 95.0% segment | 96.9% segment |
| ROC AUC (mismatch) | **0.859** | 0.649 |

## Key Configuration Parameters

### CMM (CovarianceMapMatchConfig)

This is the complete set of keys `CovarianceMapMatchConfig::load_from_xml` reads ([src/mm/cmm/cmm_algorithm.cpp:618](src/mm/cmm/cmm_algorithm.cpp#L618)). Any other key under `<parameters>` is silently ignored.

**The "Real-data value" column is ablation-ladder row 5** — the paper's operating point, `input/config/paper1_cmm.xml`. It was row 7 until 2026-09-28 and some cells still carried row-7 numbers afterwards, which is the defect that memory `cmm-operating-point-is-row5-fixed-300m` records; the three that differ between the rows ($k_\text{search}$, $\delta_\text{road}$, $\mathrm{phmi}$) are marked below. When a cell here and `paper1_cmm.xml` disagree, the config is right.

| Parameter | Default | Real-data value | Meaning |
|-----------|---------|-----------------|---------|
| `k` | 8 | 16 | Max candidates retained per epoch |
| `min_candidates` | 3 | 1 | Floor on the candidates *kept* from the search disk. If fewer survive the metric-based selection, the nearest remaining ones are appended by Euclidean distance; the disk is never enlarged. 1 ⇒ the floor is inert |
| `k_search` | 1.0 | **0.0** ⚠row5 | Multiplier on the protection level in $r_i = k_\text{search}\mathrm{HPL}_i + \delta_\text{road}$. 0 ⇒ the radius is a constant $\delta_\text{road}$. $k_\text{search}=1$ with $\delta_\text{road}=13.5$ m is row 7 and belongs to paper 2 |
| `delta_road` | 0.0 | **0.0026949335** ⚠row5 | Additive search-radius allowance, in **network units** (degrees on this dataset). Real value is 300 m = 300/111320 — the *same* fixed 300 m disk the FMM baseline searches. Simulation uses 3.5 m = 3.14407e-05; row 7's adaptive disk uses 13.5 m = 0.000121272 |
| `reverse_tolerance` | 0.0 | 0.1 | **Ratio of edge length** — 0.1 = 10% extra reverse travel allowed on top of the fixed-15%-of-edge-length guard in `get_sp_dist` |
| `use_mahalanobis` | true | true | Mahalanobis rather than Euclidean candidate projection |
| `filtered` | true | true | Drop points whose TW falls below `trustworthiness_threshold` |
| `enable_gap_bridging` | true | (default) | Skip invalid points to bridge gaps |
| `phmi` | 1.0e-5 | **0.0** ⚠row5 | Integrity risk: the inside/outside weight of the grouped emission normalization. 0 disables the grouping, so the layer degenerates to the plain softmax of Eq. (layer_norm) and nothing more. The two cannot be varied independently of the radius: PHMI only has a boundary to group at because the disk is HPL-derived, so $r$ and the grouping are one choice, and paper 1 declines both. Paper 2 carries 1.0e-5 |
| `cumulative_reverse_pct` | 0.03 | 0.03 | Max cumulative reverse as fraction of edge length (one-way edges only). Reduced from 15% to fix Traj 22 false lock. |
| `direction_penalty` | true | (default) | von Mises direction-consistency penalty on reversed candidates. Turning it off costs ≈2.05 pp accuracy, so it carries weight in every reported number. $\kappa = \|v\|^2/(c\,n^{\mathsf T}\Sigma n + b\|v\|^2)$, so it is *not* invariant to a change of covariance scale |
| `direction_kappa_scale` | 0.36 | **0.227** | $c = 2(1-\rho)$, the factor converting a one-epoch covariance into a two-epoch *difference* covariance; $\rho = 0.89$ here. Fitted by regressing $\sigma_\theta^2$ on $n^{\mathsf T}\Sigma n/\|v\|^2$. **The fit is a fixed point:** its sample is "epochs the matcher got right", which depends on these constants. Regressing against the previous configuration's output gives 0.254; re-running at that and regressing again gives 0.227, which reproduces itself. 0.227 is the converged value and the definition; **0.254 is the first iteration, not a second estimate of the same quantity**, and an earlier 0.36 came from regressing on $1/\|v\|^2$ alone, which omits the per-epoch regressor and is biased |
| `direction_kappa_floor` | 0.0 | **0.0084** | $b$ in rad²: the heading variance the covariance cannot contain (vehicle heading change within the epoch, digitised-centreline disagreement, reference-truth noise). 5.2°, significant at $z = 9.0$ over 11,792 steps. At the median step it is **29× the $c$ term**, so on Haikou the term is largely a constant angular tolerance. With $b>0$ $\kappa$ saturates at $1/b$ instead of growing without bound. Removing the $c$ term and keeping only the floor costs 2.5 pp of accuracy, so the term's value is its **speed dependence**, not its magnitude |
| `max_interval` | 180 | 180 | Max time interval (s) before splitting trajectory |
| `trustworthiness_threshold` | 0.0 | 0.0 | Min TW to retain (0 = keep all) |
| `map_error_sigma` | 0.0 | **10.6** | Isotropic map-registration standard deviation in **metres**, added to the emission covariance in quadrature. The operating point's value is $\sqrt{10^2+3.5^2}$: the paper's own decomposition of $\delta_\text{road}$ into 3.5 m of lane width and a 10 m map allowance, added in **variance** because the two sources are independent (a radius allowance is a worst case and adds linearly; a variance does not). The direction penalty deliberately does **not** see this term — a misregistered map shifts where the road is, not which way the vehicle points, and letting $\kappa$ follow the widened covariance would confound the emission change with a direction change |
| `direction_max_interval` | 2.0 | 2.0 | Gap guard (s): a displacement spanning longer is not a heading, so the term is skipped. A **time** bound, not a speed bound — the four offending Haikou epochs carried 2450 m over 118 s, an ordinary 20.8 m/s that any motorway-admitting speed ceiling would let through. 0 disables the guard |

**Ablation-only keys (not part of the paper's model).** Four further keys are read by the same loader and are listed here so that the "complete set" claim above stays true. None appears in the paper's §III formulas, none is set by any config in `input/config/`, and all default to a value that disables them, reproducing the pre-existing behaviour bit for bit. `isotropic_emission_scale` and `tw_use_viterbi` exist for the component-attribution experiment ([experiments/scripts/ablate_cmm_factors.py](experiments/scripts/ablate_cmm_factors.py)); the two `direction_*` keys exist so that a **measured negative result** stays reproducible rather than becoming a claim ([tests/test_cmm_direction_penalty.cpp](tests/test_cmm_direction_penalty.cpp)). `isotropic_emission_scale` raises a `SPDLOG_WARN` prefixed `ABLATION-ONLY` whenever it is not at its neutral value; the others do not warn and are only reported by `print()`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `isotropic_emission_scale` | 0.0 | > 0 replaces the emission covariance by an isotropic one, $\sigma_i = \text{scale}\cdot\sqrt{(s_{de}^2+s_{dn}^2)/2}$, computed **per epoch** so total variance is matched; `scale = 1` is the matched operating point. It cannot be emulated by rewriting the input table, because that would move `use_mahalanobis` at the same time. The direction penalty is switched to the same covariance, since $\kappa \propto 1/\sigma^2$ is not scale-invariant |
| `tw_use_viterbi` | false | false = $\alpha_t(i^*)/\sum_j\alpha_t(j)$, the filtering posterior of Eq. (16); true = $\delta_t(i^*)/\sum_j\delta_t(j)$, the Viterbi score. The same key exists in `FastMapMatchConfig` for the baseline side of the comparison |
| `direction_symmetric` | false | true makes the von Mises term symmetric — $\kappa(\cos\theta-1)$, penalising any misalignment rather than only a reversal. **Measured net loss** (2026-09-27), which is why it is off: accuracy 97.87 → 96.44 (−1.43 pp, −1.14 pp excluding the 61 GT-defect epochs, so not a labelling artefact), ECE 0.0303 → 0.0452, BSS −0.176 → −0.510, separation 0.435 → 0.305, false alarm at `tw < 0.5` 2.04 % → 3.29 %. It also fails its own goal — Traj 11's window stays 44/44 wrong under it, because the road the vehicle was on is not a candidate at all, so no re-ranking among candidates can recover it |
| `direction_lambda_max` | 0.0 | Cap on the per-epoch heading penalty in nats; 0 = no cap. Only meaningful with `direction_symmetric`: the one-sided form's penalty at a reversal reaches ~2000 on Haikou, and capping *that* would gut the wrong-way detection the shipped term exists for |

App-level keys outside `<parameters>`, i.e. the complete set [`CMMAppConfig::load_from_xml`](src/mm/cmm/cmm_app_config.cpp) reads from `<other>`: `<use_omp>`, `<log_level>`, `<input_epsg>`, `<network_bbox_from_gps>`, `<network_bbox_padding>`. Note that `<step>` is **not** among them — the CMM app never reads it; only `fmm_app_config.cpp` and `stmatch_app_config.cpp` do, as a fallback step distance. `<convert_to_projected>` is likewise read by no app: the CMM app takes the CRS from `<input_epsg>` instead.

A key that no loader recognises is ignored without any warning, so an inert element in a config file reads as an active setting to anyone who has not checked the loader. Two such elements were removed from the configs in this repository: `<gps_point>` (the CMM app parses its input with its own per-epoch column reader and never calls `GPSConfig::get_gps_format()`, whose only caller is `src/io/gps_reader.cpp`; and the configs placed it at `config.input.gps_point`, while the loader that does read it looks at `config.input.gps.gps_point`), and `<point_mode>` directly under `<output>` (the parser reads it only inside `<fields>`, and it defaults to true).

### FMM (FastMapMatchConfig)

| Parameter | Typical | Meaning |
|-----------|---------|---------|
| `k` | 8 | Candidates per epoch |
| `radius` | 300 (map units) | Fixed search radius. The source default is `300.0` in **network units**, which on an EPSG:4326 network means 300 degrees; there is no usable default and the radius must be converted (300 m = 0.0026949335, at 111320 m/deg) |
| `gps_error` | 50 (map units) | Fixed isotropic GPS error $\sigma$, same unit caveat (100 m = 0.0008983112) |
| `reverse_tolerance` | 0.0 | Reverse movement tolerance. **Ratio of edge length**, not a distance |
| `tw_use_viterbi` | false | false = forward posterior $\alpha_t(i^*)/\sum_j\alpha_t(j)$, the same estimand CaMM reports; true = Viterbi score $\delta_t(i^*)/\sum_j\delta_t(j)$ |

## Known Limitations and Failure Modes

1. **Parallel-edge emission ambiguity** (Traj 22, §V-B of the paper): the Mahalanobis emission cannot reliably distinguish parallel carriageway edges separated by ~12 m when SPP accuracy is ~2–5 m; it favours the geometrically closer edge even when that edge is the wrong direction. This is the one trajectory on which CaMM loses to the baseline, and loses badly — 91.8 % against 99.0 % at the paper's operating point. The baseline's broad isotropic emission ($\sigma = 100$ m) is nearly flat across the two carriageways, so path continuity decides; CaMM's tightly scaled covariance makes the emission local and leaves the decision to the cumulative reverse guard (3 % of edge length, one-way edges only). This is the clearest case in this dataset in which a tighter covariance is worse rather than better.

2. **Cumulative reverse guard CRS sensitivity**: The guard was originally calibrated for metric coordinate systems. When applied in EPSG:4326 (degree-based), the hard cap becomes ~111,000× too large, effectively disabling the guard. Always verify the guard threshold is dimensionally consistent with the CRS.

3. **The real-data protection level is geometry-only**: the RAIM generator's $\sigma_0$ is $10^5$ too large, so its guard trips and every epoch gets the `3.0² · (HᵀWH)⁻¹` fallback (measured: 14,620/14,620 epochs on trajectory 1.4). The `protection_level` column in the dataset is therefore a function of satellite geometry alone with a hard-coded $\sigma=3$ m, not of the actual measurement noise. Everything downstream still works — the PL is a valid bounding radius — but it cannot be read as "this epoch's observation quality". See the `compute_raim_pl.py` row in the validation pipeline.

4. **RAIM requires ≥5 visible satellites**: Performance in urban canyons with frequent blockage is untested. ARAIM MHSS (multi-hypothesis solution separation) would be needed for multi-fault integrity guarantees.

5. **Off-road epochs are skipped, not modelled**: an epoch whose search radius admits no road candidate now contributes no candidate at all, so it is excluded from the evaluation and matching continues with the remaining epochs. The alternative -- an explicit off-road state in the HMM -- would emit a row for such an epoch, at the cost of the Viterbi-layer bookkeeping that previously stalled the sub-segment. Which is preferable depends on whether the downstream consumer needs a row per input epoch.

6. **Single-city validation**: Real experiments are limited to Haikou, Hainan (152,547 edges, 6 trajectories, 15,421 epochs). External validity for different cities and receiver classes is not yet established.

7. **Emission model misspecification**: When the WLS solver's assumed $\sigma_{\rho}$ differs from the true pseudorange noise, ECE degrades asymmetrically — over-confidence ($\sigma_{\text{wls}} < \sigma_{\rho}^{\text{true}}$) degrades calibration more severely than over-conservatism. The RAIM-FDE module is designed to prevent severe over-confidence.

## Empirical Performance Reference (from Paper)

Real-vehicle results over the $13{,}069$ evaluable Haikou epochs, at the paper's
operating point. **The operating point is ablation-ladder row 5**
(`data/archive_20260922/stair/run2.py`, and `EXPERIMENT_DESIGN.md` §0.1 registers it as
"the paper's model"): CaMM with $k_\text{search}=0$ and $\delta_\text{road}=300$ m, i.e.
the *same fixed 300 m disk the baseline searches*, `phmi = 0`, elliptical covariance,
Mahalanobis projection, the one-sided von Mises direction term with
`direction_kappa_scale = 0.227` (converged; 0.36 was the superseded biased fit — see
the parameter table) and `direction_kappa_floor = 0.0084` rad², a 2 s gap guard, and
the map-registration term `map_error_sigma = 10.6` m. The baseline runs at $r=300$ m, $\sigma=100$ m,
`reverse_tolerance` 0.1, scored with its own native Viterbi score
$\delta_t/\sum_j\delta_t(j)$.

Reproduce both runs plus every number below with
`python3 experiments/scripts/real_final_tables.py`. That script is the provenance: before
it existed these numbers were produced by ad-hoc scripts under `/tmp`, which is how a
debug configuration came to be recorded as the delivered one (see below).

| Metric | CaMM (smoothed) | CaMM (filtering) | FMM baseline |
|--------|-----------------|------------------|--------------|
| Segment accuracy (real) | **98.13 %** | **98.13 %** | 95.42 % |
| Mean position error (real) | **5.66 m** | **5.66 m** | 5.99 m |
| Position error P95 | **13.72 m** | **13.72 m** | 14.96 m |
| ECE (TW calibration) | **0.0067** | 0.0386 | 0.1931 |
| MCE (TW) | **0.2932** | 0.9359 | 0.7299 |
| Brier | **0.0135** | 0.0326 | 0.1346 |
| Brier skill score | **+0.266** | $-0.774$ | $-2.082$ |
| Log score (nats) | **$-0.173$** | $-0.233$ | $-0.410$ |
| ROC AUC (mismatch detection) | 0.8199 | 0.8368 | **0.8650** |
| TW separation (correct−wrong) | 0.2578 | 0.4262 | **0.4417** |
| Mean candidates/epoch | 8.82 | 8.82 | 8.82 |
| Epochs at TW = 1.000 | 74.6 % | 67.9 % | 2.0 % |

**Six trajectories resolve ONE of these differences, and it is the calibration one.**
Everything above is a point estimate; the paper reports trajectory-clustered intervals
beside it (`experiments/scripts/bootstrap_real_metrics.py`, resampling unit = trajectory,
2000 replicates, seed 20260928). Of the six between-matcher contrasts the narrative rests
on, five have intervals containing zero:

| contrast | point | 95 % CI | verdict |
|---|---|---|---|
| CaMM − FMM, ECE | −0.154 | [−0.187, −0.129] | **resolved** (permutation p = 1/64) |
| CaMM smoothed − FMM smoothed, ECE | −0.190 | [−0.273, −0.073] | **resolved** — the fair pair (same estimator, same evidence set) |
| CaMM − FMM, position error | −0.34 m | [−0.90, −0.02] | resolved, marginal |
| CaMM − FMM, accuracy | +2.72 pp | [−0.59, +8.51] | unresolved |
| FMM − CaMM, AUC | +0.029 | [−0.201, +0.151] | unresolved |
| FMM − CaMM, separation | +0.018 | [−0.483, +0.226] | unresolved |
| FMM − CaMM, BSS | −1.29 | [−11.38, +0.46] | unresolved |

Within the proposed matcher the estimator ordering IS resolved — smoothed beats filtering
on ECE (−0.032 [−0.050, −0.017]) and BSS (+1.06 [+0.55, +1.57]) and loses on separation
(−0.164 [−0.253, −0.088]) — and on the baseline's isotropic model the Viterbi score beats
its own posterior on both properties. So the paper does not claim an accuracy or a
discrimination win: the dataset cannot rank them on those axes, while it can on
calibration.

**The review round of 2026-09-28 added an FMM smoothed arm (S6), and the audit is now a
complete 2×3 grid** — two observation models, each scored by δ, α and αβ. FMM smoothed
(implemented as `FastMapMatch::smooth_tg`, forward pass bit-identical) attains ECE 0.1970 /
BSS −2.71 / AUC 0.7876: smoothing repairs the baseline's posterior but not its native
score, and the fair pair CaMM-smoothed − FMM-smoothed excludes zero. Per-trajectory, FMM's
AUC win is carried by traj 11 alone, and its smoothed arm's pooled ECE is a near-
cancellation of wildly different per-trajectory values.

**Three further findings from that round travel with the numbers.** (1) The log score is
negative-skill for every arm vs its own constant (−0.85 to −3.6 relative): 31 % of CaMM's
errors are reported above 0.9 (FMM native 2 %), so the mass is calibrated and the tail is
not — reported in `tab:tail`. (2) Post-hoc recalibration of FMM native by trajectory-LOO
(temperature / isotonic) reaches ECE 0.034 / 0.040 with BSS still negative: it matches the
CaMM filtering arm, not the smoothed arm, and CaMM needs no post-hoc step (`tab:recal`).
(3) On the non-saturated subset (score < 1) the filtering arm's ECE rises to 0.110 while
the smoothed arm's stays at 0.014 and its AUC rises to 0.944. Runtime: cmm 1.79 s vs
fmm 1.11 s on the corpus (1.6×); the backward pass adds nothing measurable.

CaMM's posterior is still saturated (68–75 % of epochs at exactly 1.000), which is why its
low ECE is bought partly by being accurate rather than by resolving residual errors. The
smoothed arm is the **only** score of the five in `tab:audit` with a positive Brier skill
against the constant at its own base rate, and its false-alarm rate on correctly-matched
epochs is 3.41 % → 0.10 % at `tw < 0.5` and 1.24 % → 0.00 % at `tw < 0.1` relative to the
filtering arm.

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

**Two traps recorded here so they are not re-entered.**

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

Evaluate with `experiments/scripts/real_final_tables.py` for the real data and
`experiments/scripts/eval_sim_corpus.py` for the simulation corpus; all three take their
metric definitions from `verify_paper_numbers.py`.

## Paper build check

`pdflatex calib.tex` reports 0 errors / 0 undefined / 0 overfull, and **that is not a
sufficient check**. A `\ref` that loses its backslash typesets as literal text and produces
no warning at all — `Fig.~\ref{fig:audit}` silently became `Fig.~` + newline + `ef{fig:audit}`
during a 2026-09-28 edit and survived several clean compiles. Also uncaught by the log:
`\begin{table}` left on a line that begins with `%`, which comments it out and makes the
`\caption` below it fall outside any float.

Two cheap guards, both worth running before a build is called good:

```bash
python3 - <<'EOF'
import re
s = open('docs/calib/calib.tex', encoding='utf-8').read()
labels = set(re.findall(r'\\label\{([^}]*)\}', s))
refs = set(re.findall(r'\\ref\{([^}]*)\}', s)) | set(re.findall(r'\\eqref\{([^}]*)\}', s))
bad = sorted(r for r in refs if r not in labels)
print('unresolved refs:', bad or 'none')
for env in ('table', 'table*', 'figure', 'equation', 'algorithm'):
    o = len(re.findall(r'\\begin\{' + re.escape(env) + r'\}', s))
    c = len(re.findall(r'\\end\{' + re.escape(env) + r'\}', s))
    if o != c:
        print(f'UNBALANCED {env}: {o} begin, {c} end')
EOF
```

An unresolved-reference count of zero from the log means the *existing* `\ref`s resolved; it
says nothing about a `\ref` that is no longer there.

## UBODT System

UBODT (Upper Bounded Origin Destination Table) maps (source_node, target_node) → shortest path distance. It's critical for FMM/CMM performance.

### Formats

| Format | Extension | Reader | Notes |
|--------|-----------|--------|-------|
| CSV | `.csv` | `read_ubodt_csv` | Human-readable, slow |
| Binary (Boost) | `.bin` | `read_ubodt_binary` | Boost serialization |
| Memory-mapped | `.bin` | `read_ubodt_mmap_binary` | Fast, zero-copy (mmap) |
| Indexed binary | `.bin` | `read_ubodt_indexed_binary` | Seekable, fast random access |
| Shared memory | `.bin` | `load_shm_file` | Baked binary, zero-copy via `ubodt_converter` |

Use `ubodt_converter` to convert between formats. Use `ubodt_daemon` to keep UBODT resident in page cache for faster startup across FMM/CMM invocations.

## Coordinate System Handling

- Controlled by `<input_epsg>` in XML config (e.g., `4326` for WGS84). `input_epsg` is the only CRS key the CMM config reads; the `Network` constructor's `convert_to_projected` argument still exists but defaults to false and nothing sets it.
- System reads network CRS from `.prj` file, automatically reprojects input trajectories if EPSG differs
- Covariance matrices are rotated via Jacobian transformation during reprojection
- Grid convergence angle is applied to covariance rotation when network is projected (e.g., UTM)

## CMM Input Format

CMM uses an aggregated CSV format with JSON-encoded per-point data:

```
id;geom;timestamps;covariances;protection_levels
1;"LINESTRING(121.0 31.0, 121.1 31.1)";"[1000.0,1001.0,...]";"[[0.68,0.69,0.81,0.033,0,0],[0.67,0.69,0.81,0.032,0,0],...]";"[2.5,2.6,...]"
```

Covariance JSON array per point: `[sde, sdn, sdu, sdne, sdeu, sdun]`.

See [input/config/](input/config/) for example config XML files. The real-vehicle CMM table lives at `data/real_vehicle/hainan_06/cmm_input_points.csv` (gitignored).

## Validation Pipeline (Python)

**`experiments/scripts/` is the live pipeline.** `python/experiments/` is an older copy of it: the two directories share five filenames (`compute_raim_pl.py`, `merge_raim_pl.py`, `mapbox_spp_rtk.py`, `plot_spp_error.py`, `plot_spp_per_traj.py`) and the `experiments/scripts/` versions are the maintained ones. Prefer them, and check which copy a script actually is before trusting its output.

| Script | Purpose |
|--------|---------|
| `extract_spp_for_cmm.py` | Build the CMM input table from raw SPP; applies `GST_COVARIANCE_SCALE` |
| `apply_gst_calibration.py` | Apply the same scale to an existing input table, line-oriented and idempotent |
| `compute_raim_pl.py` | Compute RAIM-derived HPL from raw RINEX observations and broadcast ephemeris. **Known defect**: the unit-weight $\sigma_0$ comes out at $1.6\times10^5$–$2.2\times10^5$ (median $1.79\times10^5$) on trajectory 1.4, so the `sigma0 > 500` guard trips and the fallback `cov = 3.0² · (HᵀWH)⁻¹` fires on **100 %** of the 14,620 epochs. The resulting `protection_level` column therefore carries no observation-noise information: it is a pure function of satellite geometry with a fixed $\sigma=3$ m, i.e. a geometry-driven HPL rather than a measurement-driven one. The ephemeris parse is *not* the cause — it is correct (line-2 field 3 parses to √A = 5153.669, A = 26,560 km, the true GPS semi-major axis). |
| `merge_raim_pl.py` | Merge computed PL into the CMM input table, keyed on `(traj, timestamp)` — never on the RINEX epoch index, which is a 10 Hz counter against a 1 Hz table |
| `eval_result_csv.py` | **Canonical evaluator**: accuracy, ECE, ROC AUC (tie-corrected Mann–Whitney), mean TW. Metric definitions are imported from `verify_paper_numbers.py` |
| `verify_paper_numbers.py` | Reproduce the manuscript's reported figures |
| `exp3_parameter_sensitivity.py`, `exp4_sigma_mismatch.py`, `exp5_degraded_conditions.py` | Paper simulation studies |
| `fig_*.py` | Paper figures |

The Monte Carlo simulation framework is in `python/experiments/monte_carlo*/` directories.

### Dataset

The real-vehicle dataset is `data/real_vehicle/` — 6 trajectories, 15,421 epochs, collected in Haikou, Hainan with Tersus BX50C receiver (SPP + RTK ground truth). Road network: `input/map/hainan/edges.shp` (152,547 edges). Precomputed UBODT: `input/map/hainan/hainan_ubodt_indexed.bin`.

## Dependencies

- **GDAL** ≥ 2.2 — Shapefile I/O, spatial reference handling
- **Boost** ≥ 1.56 — Graph (routing), Geometry (RTree), Serialization (UBODT binary)
- **OpenMP** — parallel batch matching
- **SWIG** — Python bindings generation
- **h3** (vendored in `third_party/h3/`) — hexagonal grid indexing
- **spdlog** (vendored in `third_party/spdlog/`) — logging
- **cxxopts** (vendored in `third_party/cxxopts/`) — CLI argument parsing

CMM no longer uses Eigen3 — it uses a simple `Matrix2d` struct defined in `src/mm/cmm/cmm_algorithm.hpp`.

## Important Constraints

- **Branch policy**: Never commit to `master`. Use `feature/<name>` or `exp/<name>` branches.
- **Test placement**: C++ tests → `tests/`, Python tests → `python/experiments/`. NEVER put tests in `src/` or `python/`.
- **No `rm -rf`**: Never use recursive deletion commands without explicit confirmation.
- **Large files**: Data/results/caches > 10 MB must be in `.gitignore`. Confirm `.gitignore` coverage before `git add`.
- **Optimization**: Compile with `-O3`. Avoid heavy STL overhead in hot loops.
- **`reverse_tolerance` is a ratio of edge length** — not an absolute distance. 0.1 means 10% of edge length.
