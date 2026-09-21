# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FMM (Fast Map Matching) is an open-source C++17 framework for matching noisy GPS trajectories to road networks using Hidden Markov Models with precomputation. This fork extends the original [FMM](https://github.com/cyang-kth/fmm) with **Covariance Map Matching (CMM)** — integrating GNSS covariance matrices and Protection Levels into the matching objective.

## Build System

### Conda Environment (Required)

The build expects a conda environment providing GDAL, Boost, SWIG, and other dependencies:

```bash
conda activate fmm_env   # or the project-specific conda env
```

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
4. **Candidate search** uses RTree KNN within a radius (FMM: fixed; CMM: adaptive, $k_\text{search}\mathrm{HPL}_i + \delta_\text{road}$ — no multiplier)
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

### Level 1: HPL-Based Candidate Search

Instead of a fixed-radius search, CMM uses an adaptive, geometry-driven search disk whose radius is
$$
r_i = k_\text{search} \, \mathrm{HPL}_i + \delta_\text{road}
$$
Candidate road segments are all edges intersecting that disk. Under good satellite geometry HPL is small (~14 m at $\sigma=1$ m); under degraded geometry HPL grows proportionally. $k_\text{search}$ (default 1.0) and $\delta_\text{road}$ (default 0.0) are the two global constants that replace what used to be a per-epoch tuning loop; $\delta_\text{road}$ is an allowance for the two quantities the HPL does not bound — the lateral offset of the driven lane from the mapped centreline, and map georeferencing error. The paper's operating points are $k_\text{search} = 1$ with $\delta_\text{road} = 3.5$ m in simulation (the national urban lane width, CJJ 37--2012) and $\delta_\text{road} = 13.5$ m on the real data (that lane width plus a 10 m map-error allowance, converted to degrees by the $111{,}320$ m/deg convention). The implementation is the pure static `effective_search_radius()` ([src/mm/cmm/cmm_algorithm.cpp:1840](src/mm/cmm/cmm_algorithm.cpp#L1840)).

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
where $\mathbf{d}_{i,j} = z_i - x_{i,j}$. Computed in log space as $-\tfrac{1}{2}(\log 2\pi + \log |\boldsymbol{\Sigma}_i| + \mathbf{d}_{i,j}^{\mathrm{T}}\boldsymbol{\Sigma}_i^{-1}\mathbf{d}_{i,j})$ at [src/mm/cmm/cmm_algorithm.cpp:920](src/mm/cmm/cmm_algorithm.cpp#L920), with no inflating term: there is no map-error variance and no floor on the standard deviation.

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

This is the complete set of keys `CovarianceMapMatchConfig::load_from_xml` reads ([src/mm/cmm/cmm_algorithm.cpp:584](src/mm/cmm/cmm_algorithm.cpp#L584)). Any other key under `<parameters>` is silently ignored.

| Parameter | Default | Real-data value | Meaning |
|-----------|---------|-----------------|---------|
| `k` | 8 | 16 | Max candidates retained per epoch |
| `min_candidates` | 3 | 1 | Floor on the candidates *kept* from the search disk. If fewer survive the metric-based selection, the nearest remaining ones are appended by Euclidean distance; the disk is never enlarged. 1 ⇒ the floor is inert |
| `k_search` | 1.0 | 1.0 | Multiplier on the protection level in $r_i = k_\text{search}\mathrm{HPL}_i + \delta_\text{road}$. 0 ⇒ the radius is a constant $\delta_\text{road}$ |
| `delta_road` | 0.0 | 0.000121272 | Additive search-radius allowance, in **network units** (degrees on this dataset). Real value is 13.5 m = 13.5/111320; simulation uses 3.5 m = 3.14407e-05 |
| `reverse_tolerance` | 0.0 | 0.1 | **Ratio of edge length** — 0.1 = 10% extra reverse travel allowed on top of the fixed-15%-of-edge-length guard in `get_sp_dist` |
| `use_mahalanobis` | true | true | Mahalanobis rather than Euclidean candidate projection |
| `filtered` | true | true | Drop points whose TW falls below `trustworthiness_threshold` |
| `enable_gap_bridging` | true | (default) | Skip invalid points to bridge gaps |
| `phmi` | 1.0e-5 | 1.0e-5 | Integrity risk: the inside/outside weight of the grouped emission normalization. 0 disables the grouping |
| `cumulative_reverse_pct` | 0.03 | 0.03 | Max cumulative reverse as fraction of edge length (one-way edges only). Reduced from 15% to fix Traj 22 false lock. |
| `direction_penalty` | true | (default) | von Mises direction-consistency penalty on reversed candidates. Turning it off costs ≈2.05 pp accuracy, so it carries weight in every reported number. Its $\kappa = v^2/(s_{de} s_{dn})$ is inversely proportional to the variance, so it is *not* invariant to a change of covariance scale |
| `max_interval` | 180 | 180 | Max time interval (s) before splitting trajectory |
| `trustworthiness_threshold` | 0.0 | 0.0 | Min TW to retain (0 = keep all) |

**Ablation-only keys (not part of the paper's model).** Two further keys are read by the same loader and are listed here so that the "complete set" claim above stays true. Neither appears in the paper's §III formulas, neither is set by any config in `input/config/`, and both default to a value that disables them, reproducing the pre-existing behaviour bit for bit. They exist only for the component-attribution experiment ([experiments/scripts/ablate_cmm_factors.py](experiments/scripts/ablate_cmm_factors.py)). `isotropic_emission_scale` raises a `SPDLOG_WARN` prefixed `ABLATION-ONLY` whenever it is not at its neutral value; `tw_use_viterbi` does not warn and is only reported by `print()`.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `isotropic_emission_scale` | 0.0 | > 0 replaces the emission covariance by an isotropic one, $\sigma_i = \text{scale}\cdot\sqrt{(s_{de}^2+s_{dn}^2)/2}$, computed **per epoch** so total variance is matched; `scale = 1` is the matched operating point. It cannot be emulated by rewriting the input table, because that would move `use_mahalanobis` at the same time. The direction penalty is switched to the same covariance, since $\kappa \propto 1/\sigma^2$ is not scale-invariant |
| `tw_use_viterbi` | false | false = $\alpha_t(i^*)/\sum_j\alpha_t(j)$, the filtering posterior of Eq. (16); true = $\delta_t(i^*)/\sum_j\delta_t(j)$, the Viterbi score. The same key exists in `FastMapMatchConfig` for the baseline side of the comparison |

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

Real-vehicle results over the $13{,}123$ evaluable Haikou epochs, at the paper's
operating point: CaMM with $k_\text{search}=1$, $\delta_\text{road}=13.5$ m,
elliptical covariance, Mahalanobis projection, direction consistency and PHMI
grouping; the baseline at $r=300$ m, $\sigma=100$ m, `reverse_tolerance` 0.1,
scored with its own native Viterbi score $\delta_t/\sum_j\delta_t(j)$.

| Metric | CaMM | FMM baseline |
|--------|------|--------------|
| Segment accuracy (real) | **96.90%** | 95.03% |
| Mean position error (real) | **5.47 m** | 6.02 m |
| Position error P95 | **13.53 m** | 15.01 m |
| ECE (TW calibration) | **0.0437** | 0.1903 |
| MCE (TW) | 0.888 | 0.697 |
| Brier | **0.0444** | 0.1355 |
| ROC AUC (mismatch detection) | 0.6493 | **0.8589** |
| TW separation (correct−wrong) | 0.0767 | **0.4312** |
| Mean candidates/epoch | **2.92** | 8.82 |

**Neither method dominates.** CaMM wins calibration, position error and the
candidate count; the baseline wins both rank statistics. The reason is that
CaMM's posterior is saturated — 97.5 % of its epochs fall in the top confidence
bin, where the mean score is 1.000 against an accuracy of 0.972 — so its low ECE
is bought by being accurate rather than by resolving the residual errors, and it
has little ranking range left. The baseline's score is under-confident over its
whole range (mean 0.781 correct, 0.350 wrong) but orders the epochs better. The
paper reports both, and attributes calibration and discrimination separately; do
not quote the ECE row alone as evidence that CaMM's score is the more useful one.

The old table here carried 96.0 % / 88.9 % / 0.040 / 0.876 / 0.721 / 0.583 /
0.262 / 0.014. Three of those are now traced: the baseline's 88.9 % accuracy,
8.9 m error and 0.583 AUC came from a 3.3 km search radius (the config's
$r = 0.03^\circ$, a 10× conversion error from the 300 m it was meant to be), and
its 0.876 ECE from a hybrid $\delta_t/\sum_j\alpha_t(j)$ score that is neither
the Viterbi score nor the filtering posterior. The remainder were carried over
from artifacts this repository no longer holds and could not be reproduced from
the current code and data when this file was last revised. All are superseded by
the table above.

Evaluate with `experiments/scripts/eval_result_csv.py` for the real data and
`experiments/scripts/eval_sim_corpus.py` for the simulation corpus; both take
their metric definitions from `verify_paper_numbers.py`.

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
