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
4. **Candidate search** uses RTree KNN within a radius (FMM: fixed; CMM: adaptive based on PL × multiplier)
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
$K \approx 5.33$ for $10^{-5}$ integrity risk under Gaussian error assumption. RAIM performs a global $\chi^2$ consistency test with iterative Fault Detection and Exclusion (FDE), requiring a minimum of 5 remaining satellites. **Do NOT use simple bounding boxes** (`max(sde, sdn)`) — the eigenvalue-based semi-major axis is the correct formula.

### Level 1: HPL-Based Candidate Search

Instead of a fixed-radius search, CMM uses an adaptive, geometry-driven search disk:
$$
\Omega_i = \{ x \in \mathbb{R}^2 \mid \| x - z_i \|_2 \le \mathrm{HPL}_i \}
$$
Candidate road segments are all edges intersecting $\Omega_i$. Under good satellite geometry HPL is small (~14 m at $\sigma=1$ m); under degraded geometry HPL grows proportionally. The search radius **is** the protection level — there is no multiplier parameter.

**Bounded radius fallback.** The construction above presumes the network intersects $\Omega_i$ at all, which can fail (an optimistic $\mathrm{HPL}_i$, a multipath-displaced fix, or a road absent from the network). Instead of discarding the epoch, the *search radius alone* is enlarged and the query retried:
$$
r_i^{(m)} = 2^m \, \mathrm{HPL}_i, \qquad m = 0, 1, \dots, M
$$
doubling until the candidate set reaches the floor `min_candidates` (default 3; 1 in the real-data configs, where the retry therefore fires exactly on a zero-candidate epoch), and skipping the epoch only if it is still empty at $m = M$. `MAX_SEARCH_RADIUS_DOUBLINGS = 8`, so the fallback reaches at most $256\,\mathrm{HPL}_i$. The predicate is the pure static `should_expand_search_radius()` ([src/mm/cmm/cmm_algorithm.cpp](src/mm/cmm/cmm_algorithm.cpp)).

This is deliberately narrower than the heuristic radius expansion criticised in the paper: it only ever *enlarges* and only until a floor is reached (never re-shrinks to hold a target count), it is bounded by a fixed $M$ rather than iterated until a count is met, and a segment admitted at $m > 0$ lies outside $\mathrm{HPL}_i$ by construction, so the PHMI-grouped normalization below gives it the integrity-risk weight $\mathrm{PHMI}$ — it enters the HMM as a candidate the covariance does *not* support.

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

**Data-side calibration.** NMEA `GST` describes the receiver's *RTK* solution class, not the SPP solution actually emitted, so the raw GST covariance understates the SPP error. `experiments/scripts/extract_spp_for_cmm.py` scales it by `GST_COVARIANCE_SCALE = 3.0008` before building the CMM input table. The factor is not an $\sigma$-ratio: the model consumes only the quadratic form $\mathbf{d}^{\mathrm{T}}\boldsymbol{\Sigma}^{-1}\mathbf{d}$, and $\log|\boldsymbol{\Sigma}|$ cancels in the same-epoch softmax, so the matched-scale criterion is $k=\sqrt{\mathbb{E}[\mathbf{d}^{\mathrm{T}}\boldsymbol{\Sigma}_{\mathrm{gst}}^{-1}\mathbf{d}]/2}$ (measured 3.0008 over 15034 epochs; median $q = 5.39$). Covariance scales by $k^2$, so $\sigma$ scales by $k$ and the cross-covariance by $k^2$.

### Level 3: Trustworthiness as Calibrated Posterior

**Probabilistic normalization (critical for calibration).** Three normalization steps ensure valid probabilities:
1. **PHMI-grouped emission normalization**: candidates are partitioned at the protection level into $\mathcal{C}^{\mathrm{in}}_i = \{j : d_{i,j} \le \mathrm{HPL}_i\}$ and its complement; each group is normalized over its own members and scaled by the probability that it contains the truth, $(1-\mathrm{PHMI})$ and $\mathrm{PHMI}$ respectively, so the layer sums to 1 when both groups are non-empty (or when only the inside group is) and to $\mathrm{PHMI}$ when only the outside group is. That last case is not exotic: with `min_candidates = 1` the radius fallback fires exactly when nothing lies inside $\mathrm{HPL}_i$, so every epoch it rescues carries emission mass $\mathrm{PHMI}$ — the epoch is kept, but it is not claimed to be explained. The per-epoch trustworthiness is a ratio at a single epoch, so this mass cancels there and acts only on the choice of path. With `phmi = 0` the grouping degenerates to a plain layer-wise softmax. This is what gives the bounded radius fallback its meaning: candidates recovered at $m>0$ are all outside $\mathrm{HPL}_i$, so they are exactly the ones that receive the $\mathrm{PHMI}$ weight.
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
| Adaptive softmax temperature | Was a no-op on the real data (ECE $0.0410 \to 0.0400$, AUC $0.7204 \to 0.7203$); never described in the paper. |
| Fixed-lag smoothing (`lag_steps`, `apply_lag_smoothing`, `flush_lag_buffer`) | Never described in the paper, and the paper reports it *degrades* TW calibration on real data with tight HPL (ECE $0.040 \to 0.26$ at $L=20$). Use $L=0$ always. |
| Protection-level multiplier (search and PHMI boundary) | $r_i = \mathrm{HPL}_i$ exactly. A multiplier on the search radius is a tuning knob the paper does not have; a separate multiplier on the PHMI boundary would have classified candidates found by radius doubling as "inside PL", which is the opposite of the intent. |
| `MIN_SIGMA` floor / `map_error_std` / `min_gps_error_degrees` | Replaced by the covariance validity test plus the isotropic fallback above. |
| Off-road background state | See the note under Level 3. |
| Sequential H0 hypothesis test (`h0_prior_log_odds`) | The recursion is gone. `MatchedCandidate::h0_lambda` and the `h0_lambda` output column survive as inert plumbing that always emits 1.0 — see "Known Limitations". |

---

## FMM vs CMM: Key Differences

| Aspect | FMM | CMM |
|--------|-----|-----|
| Emission model | Isotropic Gaussian $\mathcal{N}(0,\sigma^2 I)$ | Anisotropic Mahalanobis (covariance-consistent) |
| Candidate search | Fixed radius | HPL-based adaptive region |
| Candidate projection | Orthogonal (Euclidean) | Mahalanobis (statistically optimal) |
| GPS error | Constant scalar $\sigma$ | Per-epoch covariance $\boldsymbol{\Sigma}_i$ |
| Trustworthiness | Raw Viterbi score (uncalibrated) | Filtering posterior via forward algorithm (calibrated) |
| Prerequisites | UBODT only | UBODT + covariance + protection level per epoch |
| Calibration (ECE) | 0.876 (under-confident) | 0.040 (95% reduction) |
| Real accuracy | 88.9% segment | 96.0% segment |

## Key Configuration Parameters

### CMM (CovarianceMapMatchConfig)

This is the complete set of keys `CovarianceMapMatchConfig::load_from_xml` reads ([src/mm/cmm/cmm_algorithm.cpp:584](src/mm/cmm/cmm_algorithm.cpp#L584)). Any other key under `<parameters>` is silently ignored.

| Parameter | Default | Real-data value | Meaning |
|-----------|---------|-----------------|---------|
| `k` | 8 | 16 | Max candidates retained per epoch |
| `min_candidates` | 3 | 1 | Candidate floor the radius fallback doubles toward. 1 ⇒ the fallback fires exactly on a zero-candidate epoch |
| `reverse_tolerance` | 0.0 | 0.1 | **Ratio of edge length** — 0.1 = 10% extra reverse travel allowed on top of the fixed-15%-of-edge-length guard in `get_sp_dist` |
| `use_mahalanobis` | true | true | Mahalanobis rather than Euclidean candidate projection |
| `filtered` | true | true | Drop points whose TW falls below `trustworthiness_threshold` |
| `enable_gap_bridging` | true | (default) | Skip invalid points to bridge gaps |
| `phmi` | 1.0e-5 | 1.0e-5 | Integrity risk: the inside/outside weight of the grouped emission normalization. 0 disables the grouping |
| `cumulative_reverse_pct` | 0.03 | 0.03 | Max cumulative reverse as fraction of edge length (one-way edges only). Reduced from 15% to fix Traj 22 false lock. |
| `direction_penalty` | true | (default) | von Mises direction-consistency penalty on reversed candidates. Turning it off costs ≈2.05 pp accuracy, so it carries weight in every reported number. Its $\kappa = v^2/(s_{de} s_{dn})$ is inversely proportional to the variance, so it is *not* invariant to a change of covariance scale |
| `max_interval` | 180 | 180 | Max time interval (s) before splitting trajectory |
| `trustworthiness_threshold` | 0.0 | 0.0 | Min TW to retain (0 = keep all) |

App-level keys outside `<parameters>`: `<input_epsg>`, `<log_level>`, `<use_omp>`, `<step>`.

### FMM (FastMapMatchConfig)

| Parameter | Typical | Meaning |
|-----------|---------|---------|
| `k` | 8 | Candidates per epoch |
| `radius` | 300 (map units) | Fixed search radius |
| `gps_error` | 50 (map units) | Fixed isotropic GPS error $\sigma$ |
| `reverse_tolerance` | 0.0 | Reverse movement tolerance |

## Known Limitations and Failure Modes

1. **Parallel-edge emission ambiguity** (Traj 22 failure case, §V-G of paper): The Mahalanobis emission cannot reliably distinguish parallel carriageway edges separated by ~12 m when SPP accuracy is ~2–5 m. The emission systematically favors the geometrically closer edge even when it's the wrong direction. The cumulative reverse guard (3% of edge length, one-way edges only) partially mitigates this (Traj 22 accuracy: 71.6% → 91.6%), but the underlying ambiguity remains for closely-spaced parallel roads.

2. **Cumulative reverse guard CRS sensitivity**: The guard was originally calibrated for metric coordinate systems. When applied in EPSG:4326 (degree-based), the hard cap becomes ~111,000× too large, effectively disabling the guard. Always verify the guard threshold is dimensionally consistent with the CRS.

3. **The real-data protection level is geometry-only**: the RAIM generator's $\sigma_0$ is $10^5$ too large, so its guard trips and every epoch gets the `3.0² · (HᵀWH)⁻¹` fallback (measured: 14,620/14,620 epochs on trajectory 1.4). The `protection_level` column in the dataset is therefore a function of satellite geometry alone with a hard-coded $\sigma=3$ m, not of the actual measurement noise. Everything downstream still works — the PL is a valid bounding radius — but it cannot be read as "this epoch's observation quality". See the `compute_raim_pl.py` row in the validation pipeline.

4. **`h0_lambda` is inert plumbing**: The sequential H0 recursion was removed, but `MatchedCandidate::h0_lambda`, `ResultConfig::write_h0_lambda`, the `mm_writer` column and the `<h0_lambda/>` field in `cmm_real_0729.xml` / `cmm_test_sigma_05.xml` remain. All three `process_sub_segment` call sites pass `nullptr`, so the column is constant 1.0 and `write_h0_lambda` defaults to false. Either finish the removal or delete the column from the two configs' `<fields>`.

5. **RAIM requires ≥5 visible satellites**: Performance in urban canyons with frequent blockage is untested. ARAIM MHSS (multi-hypothesis solution separation) would be needed for multi-fault integrity guarantees.

6. **Off-road epochs are skipped, not modelled**: an epoch whose search radius admits no road candidate now contributes no candidate at all, so it is excluded from the evaluation and matching continues with the remaining epochs. The alternative -- an explicit off-road state in the HMM -- would emit a row for such an epoch, at the cost of the Viterbi-layer bookkeeping that previously stalled the sub-segment. Which is preferable depends on whether the downstream consumer needs a row per input epoch.

7. **Single-city validation**: Real experiments are limited to Haikou, Hainan (152,547 edges, 6 trajectories, 15,421 epochs). External validity for different cities and receiver classes is not yet established.

8. **Emission model misspecification**: When the WLS solver's assumed $\sigma_{\rho}$ differs from the true pseudorange noise, ECE degrades asymmetrically — over-confidence ($\sigma_{\text{wls}} < \sigma_{\rho}^{\text{true}}$) degrades calibration more severely than over-conservatism. The RAIM-FDE module is designed to prevent severe over-confidence.

## Empirical Performance Reference (from Paper)

| Metric | CMM | FMM |
|--------|-----|-----|
| Segment accuracy (real) | 96.0% | 88.9% |
| Mean position error (real) | 5.6 m | 8.9 m |
| ECE (TW calibration) | 0.040 | 0.876 |
| TW separation (correct−wrong) | 0.262 | 0.014 |
| ROC AUC (mismatch detection) | 0.721 | 0.583 |
| Acc. at $\sigma_{\rho}=30$ m (sim) | 90.6% | 56.2% |

\*FMM's normalized Viterbi scores are severely under-confident (mean TW 0.015), near zero for both correct and wrong matches, leaving almost no discriminative signal. CMM's TW drops from 0.972 (correct) to 0.710 (wrong), providing actionable separation.

**Reproduction status.** The ECE and AUC rows reproduce under the pruned implementation; the accuracy row does not. On `cmm_real_0729.xml` over the calibrated 15,421-epoch Haikou set the current code gives **97.23 % segment accuracy / ECE 0.0391 / AUC 0.7592** with zero failed epochs. 22 of the 29 manuscript rows reproduce; the 96.0 % CMM accuracy figure is not reproducible from the current data and code and is under investigation. Evaluate with `experiments/scripts/eval_result_csv.py`, which takes its metric definitions from `verify_paper_numbers.py`.

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
