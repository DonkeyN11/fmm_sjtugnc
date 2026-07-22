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

See [python/CMM_PYTHON_API.md](python/CMM_PYTHON_API.md) for CMM Python API details.

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
The $2\times 2$ horizontal block $\boldsymbol{\Sigma}_i = \begin{bmatrix} \sigma_E^2 & \sigma_{EN} \\ \sigma_{EN} & \sigma_N^2 \end{bmatrix}$ drives the anisotropic emission model. In practice, $\boldsymbol{\Sigma}_i$ is parsed from NMEA `GST` messages or computed from RINEX observations in `python/experiments/compute_raim_pl.py`.

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
Candidate road segments are all edges intersecting $\Omega_i$. Under good satellite geometry HPL is small (~14 m at $\sigma=1$ m); under degraded geometry HPL grows proportionally. The `protection_level_multiplier` parameter allows scaling the HPL (e.g., 3.0 in simulations, 10.0 in real experiments).

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
where $\mathbf{d}_{i,j} = z_i - x_{i,j}$. This is implemented in `calculate_emission_log_prob()` in [src/mm/cmm/cmm_algorithm.hpp](src/mm/cmm/cmm_algorithm.hpp).

### Level 3: Trustworthiness as Calibrated Posterior

**Probabilistic normalization (critical for calibration).** Three normalization steps ensure valid probabilities:
1. **Background state**: A pseudo-candidate with fixed emission $p_{\text{bg}}$ (default 0.1) represents the off-road hypothesis. Effective emission: $p'(z_i \mid x_{i,j}) = (1-p_{\text{bg}}) \cdot p(z_i \mid x_{i,j})$. This prevents overconfidence when all road candidates have low likelihood.
2. **Row-normalized transitions**: $t_{a \to b} = w_{a \to b} / \sum_j w_{a \to j}$ where $w_{a \to b} = \min(1, d_{\text{gnss}} / d_{\text{road}})$, ensuring $\sum_j t_{a \to j} = 1$.
3. **Uniform initial prior**: $\pi(i) = 1/K$ for $K$ real candidates.

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

### Fixed-Lag Smoothing

When `lag_steps > 0`, CMM buffers $L+1$ transition graph layers and re-evaluates posterior probabilities using future evidence before finalizing trustworthiness scores. This is implemented in `apply_lag_smoothing()` and `flush_lag_buffer()`. However, the paper reports that on real data with tight RAIM-derived HPL, lag smoothing degrades TW calibration (ECE increases from 0.069 at $L=0$ to 0.26 at $L=20$), suggesting it is most beneficial when the receiver does NOT provide covariance outputs. For receivers providing full covariance, $L=0$ is recommended.

### PHMI (Integrity Monitoring Mode)

Sequential Bayesian H0 hypothesis test accumulated across the trajectory via `h0_prior_log_odds` and cumulative likelihood ratios. The cumulative ratio $\lambda_t = \prod_{\tau=1}^t \text{LR}_\tau$ is stored in `MatchedCandidate::h0_lambda`.

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
| Calibration (ECE) | 0.107 (over-confident) | 0.069 (36% reduction) |
| Real accuracy | 88.1% segment | 96.9% segment |

## Key Configuration Parameters

### CMM (CovarianceMapMatchConfig)

| Parameter | Typical Value | Meaning |
|-----------|--------------|---------|
| `k` | 16 | Max candidates per epoch |
| `min_candidates` | 1 | Min candidates retained |
| `protection_level_multiplier` | 3.0 (sim) / 10.0 (real) | Scales HPL for candidate search radius |
| `phmi_pl_multiplier` | 5.0 | Separate scaling for integrity check (decoupled from search) |
| `reverse_tolerance` | 0.1 | **Ratio of edge length** — 0.1 = 10% max reverse travel |
| `cumulative_reverse_pct` | 0.03 | Max cumulative reverse as fraction of edge length (one-way edges only). 3% in paper, reduced from 15% to fix Traj 22 false lock. |
| `lag_steps` | 0 (real) / 5 (sim) | Fixed-lag smoothing steps. 0 = real-time filtering, N = N-step delay |
| `background_prob` | 0.1 | Off-road background state probability (Laplace smoothing) |
| `map_error_std` | 5.0e-6 deg (~0.5 m) | Map error added in quadrature to GPS variance |
| `min_gps_error_degrees` | 1.0e-6 (~0.1 m) | Floor on GPS error to prevent over-confidence |
| `phmi` | 1.0e-5 | Integrity risk for PHMI mode |
| `h0_prior_log_odds` | 0.0 | Log-odds of null hypothesis prior ($\lambda_0 = 1$) |
| `max_gap_distance` | 2000 m | Max physical distance for gap bridging |
| `max_interval` | 180 s | Max time interval before splitting trajectory |
| `trustworthiness_threshold` | 0.0 | Min TW to retain (0 = keep all) |

### FMM (FastMapMatchConfig)

| Parameter | Typical | Meaning |
|-----------|---------|---------|
| `k` | 8 | Candidates per epoch |
| `radius` | 300 (map units) | Fixed search radius |
| `gps_error` | 50 (map units) | Fixed isotropic GPS error $\sigma$ |
| `reverse_tolerance` | 0.0 | Reverse movement tolerance |

## Known Limitations and Failure Modes

1. **Parallel-edge emission ambiguity** (Traj 22 failure case, §V-G of paper): The Mahalanobis emission cannot reliably distinguish parallel carriageway edges separated by ~12 m when SPP accuracy is ~2–5 m. The emission systematically favors the geometrically closer edge even when it's the wrong direction. The cumulative reverse guard (3% of edge length, one-way edges only) partially mitigates this (Traj 22 accuracy: 71.6% → 93.4%), but the underlying ambiguity remains for closely-spaced parallel roads.

2. **Cumulative reverse guard CRS sensitivity**: The guard was originally calibrated for metric coordinate systems. When applied in EPSG:4326 (degree-based), the hard cap becomes ~111,000× too large, effectively disabling the guard. Always verify the guard threshold is dimensionally consistent with the CRS.

3. **Fixed-lag smoothing degrades with tight PL**: When the RAIM-derived HPL is already tight (median ~22.8 m), fixed-lag smoothing can degrade TW calibration (ECE 0.069 → 0.26 at L=20). Use L=0 for covariance-equipped receivers; L > 0 may help for receivers without covariance output.

4. **RAIM requires ≥5 visible satellites**: Performance in urban canyons with frequent blockage is untested. ARAIM MHSS (multi-hypothesis solution separation) would be needed for multi-fault integrity guarantees.

5. **Background state $p_{\text{bg}} = 0.1$** acts as Laplace smoothing; the optimal value likely depends on road network density and GNSS quality. A data-driven calibration is future work.

6. **Single-city validation**: Real experiments are limited to Haikou, Hainan (152,547 edges, 7 trajectories, 16,155 epochs). External validity for different cities and receiver classes is not yet established.

7. **Emission model misspecification**: When the WLS solver's assumed $\sigma_{\rho}$ differs from the true pseudorange noise, ECE degrades asymmetrically — over-confidence ($\sigma_{\text{wls}} < \sigma_{\rho}^{\text{true}}$) degrades calibration more severely than over-conservatism. The RAIM-FDE module is designed to prevent severe over-confidence.

## Empirical Performance Reference (from Paper)

| Metric | CMM | FMM |
|--------|-----|-----|
| Segment accuracy (real) | 96.9% | 88.1% |
| Mean position error (real) | 5.6 m | 9.4 m |
| ECE (TW calibration) | 0.069 | 0.107 |
| TW separation (correct−wrong) | 0.291 | 0.085 |
| ROC AUC (mismatch detection) | 0.600 | 0.965* |
| Acc. at $\sigma_{\rho}=30$ m (sim) | 76.8% | 56.1% |

\*FMM's high AUC is an artifact of near-binary TW scores (s.d. 0.039) — the scores are compressed near 1.0 regardless of correctness, inflating AUC while providing poor practical discriminative power. CMM's TW drops from 0.925 (correct) to 0.633 (wrong), providing actionable separation.

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

- Controlled by `<input_epsg>` in XML config (e.g., `4326` for WGS84)
- **`convert_to_projected` is DEPRECATED** — always use `input_epsg`
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

See [input/](input/) for example config XML files, [input_cmm_100/](input_cmm_100/) for CMM input data.

## Validation Pipeline (Python)

The paper's experiments are implemented as Python scripts in `python/experiments/`. Key entry points:

| Script | Purpose |
|--------|---------|
| `compute_raim_pl.py` | Compute RAIM-derived HPL from raw RINEX observations and broadcast ephemeris |
| `merge_raim_pl.py` | Merge computed PL with trajectory data to create CMM input |
| `exp1_lag_sweep.py` | Lag parameter sweep for fixed-lag smoothing sensitivity |
| `exp1_reliability_diagram.py` | Generate reliability diagrams and ECE metrics |
| `exp2_synthetic_validation.py` | Monte Carlo simulation with synthetic GNSS constellation |
| `exp3_phmi_analysis.py` | PHMI integrity monitoring analysis |
| `exp4_ablation_ece.py` | Ablation study: per-component ECE contribution |
| `gen_figures.py` | Generate paper figures from experiment outputs |
| `evaluate_match_metrics.py` | Compute accuracy, ECE, ROC AUC from matching results |
| `analyze_spp_error.py` | SPP error distribution analysis |
| `mapbox_spp_rtk.py` | Mapbox visualization of SPP vs RTK trajectories |

The Monte Carlo simulation framework is in `monte_carlo/`, `monte_carlo_1050/`, and `monte_carlo_enu/` directories.

### Dataset

The real-vehicle dataset is `data/real_vehicle/` — 7 trajectories, 16,155 epochs, collected in Haikou, Hainan with Tersus BX50C receiver (SPP + RTK ground truth). Road network: `input/map/hainan/edges.shp` (152,547 edges). Precomputed UBODT: `input/map/hainan/hainan_ubodt_indexed.bin`.

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
