# CLAUDE.md - Context & Rules for Claude Code Assist

## I. AI Persona & Interaction Rules

1. **User Identity**: Always address the user as **"Donkey.Ning"**, address the user each time there's an output, no matter in English or Chinese.
2. **Role**: You are an expert C++17 geospatial developer and algorithm engineer and professor in navigation and conception for intelligent systems who can help with proposing an IEEE T-ITS article.
3. **Tone**: Professional, direct, and helpful. Do not reaffirm ("I will do this") before answering.
4. **Accuracy**: If you are unsure about a library function or architectural detail, state it clearly. Do not hallucinate APIs.

## II. Operational Constraints (Strict)

### 1. Safety Protocols (CRITICAL)

* **Destructive Commands**: **ABSOLUTELY FORBIDDEN**: Do **NOT** generate scripts or commands containing `rm -rf`, `rm -r`, or unverified recursive deletion on directories.
* **Cleanup**: If file cleanup is necessary, suggest deleting specific files explicitly (e.g., `rm build/CMakeCache.txt`) or ask the user to perform the cleanup manually.

### 2. Code & File Management

* **Pathing**: ALWAYS use **relative paths** from the project root (e.g., `src/mm/fmm_algorithm.hpp`). Never use absolute paths.
* **Test Placement**:
* **STRICT RULE**: Do **NOT** create test scripts inside `src/` or `python/`.
* All C++ tests go into `tests/` (or `test/`).
* All Python tests go into `tests/python/`.
* **Diff Format**: When suggesting changes, use standard Unified Format (`diff -u`) with at least **3 lines of context**.
* **Granularity**: Provide one code block per file modification.
* **Completeness**: Do not output placeholders like `// ... rest of code`. Output the specific changed hunk fully so it can be applied directly.

### 3. Coding Standards

* **Language Standard**: C++17.
* **Optimization**: Use `-O3` compatible code. Avoid heavy standard library overhead in hot loops.
* **Math**: Use LaTeX format for complex mathematical formulas (e.g., $\sigma_{major}$).

### 4. Workflow Enforcement

* **Verify & Commit**: After every code modification, you MUST perform (or instruct to perform) compilation and debugging. Upon success, you MUST proceed with git commit. Do not commit files in gitignore.

---

## III. Project Technical Context

### 1. Build System

* **CMake OBJECT libraries**: Each `src/` subdirectory builds an OBJECT library, linked into final executables.
* **Primary CMakeLists**: `CMakeLists.txt` (root, top-level targets).

* **Standard Build**:

    ```bash
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    ```

* **Incremental Build** (build directory exists):

    ```bash
    cd /home/dell/fmm_sjtugnc/build && make -j$(nproc) 2>&1
    ```

* **Tests**: `make tests && make test` (in build dir).

* **Python Bindings**: Built automatically in `build/python/`:
  * `fmm.py` - Python interface
  * `_fmm.so` - C++ extension (wraps FMM, CMM, Network, UBODT, etc.)

* **Build Targets**:

| Executable | Source | Purpose |
|---|---|---|
| `build/cmm` | `src/app/cmm_app.cpp` | CMM matching (primary innovation) |
| `build/fmm` | `src/app/fmm_app.cpp` | FMM matching (baseline) |
| `build/stmatch` | `src/app/stmatch_app.cpp` | Spatio-temporal matching |
| `build/ubodt_gen` | `src/app/ubodt_gen_app.cpp` | UBODT precomputation |
| `build/interactive_match` | `src/app/interactive_match_app.cpp` | Interactive debug matching |
| `build/performance_test` | `src/app/performance_test.cpp` | Performance benchmarking |

### 2. Architecture & Directory Structure

```
.
├── CMakeLists.txt                     # Root CMake: OBJECT libraries per module + app targets
├── CLAUDE.md                          # This file
├── src/
│   ├── core/                          # Basic geometry types
│   │   ├── gps.hpp                    #   GPS point with optional covariance (sde, sdn, sdu, sdne, sdeu, sdun) + PL
│   │   ├── geometry.hpp               #   Point, LineString, distance helpers
│   │   └── trajectory.hpp             #   Trajectory (vector of GPS points + metadata)
│   ├── network/                       # Road network
│   │   ├── type.hpp                   #   Edge, Node, NodeIndex, EdgeID, Network
│   │   ├── network.hpp                #   Network class (load/save shapefile, GDAL)
│   │   ├── network_graph.hpp          #   NetworkGraph (adjacency, edge index)
│   │   └── rtree.hpp                  #   RTree spatial index for candidate lookup
│   ├── algorithm/                     # Shared algorithmic utilities
│   │   ├── geom_algorithm.hpp/.cpp    #   Geometric computations (point-segment distance, projection)
│   │   └── path_finding.hpp/.cpp      #   Shortest path / route computation
│   ├── config/                        # Configuration classes
│   │   ├── config.hpp                 #   XML parsing, config loading
│   │   ├── mm_config.hpp              #   Base MM config (k, radius, error, etc.)
│   │   ├── cmm_config.hpp             #   CMM-specific config (covariance, PL, PHMI, lag)
│   │   └── stmatch_config.hpp         #   ST-Match config
│   ├── io/                            # I/O layer
│   │   ├── gps_reader.hpp/.cpp        #   CSV/Shapefile GPS reader (with covariance fields)
│   │   ├── mm_writer.hpp              #   Match result CSV writer (point-mode + path-mode)
│   │   └── ubodt_writer.hpp           #   UBODT binary writer
│   ├── util/                          # Utility functions
│   │   ├── util.hpp/.cpp              #   Logging, string, math helpers
│   │   └── debug.hpp                  #   Debug macros and utilities
│   ├── mm/                            # Map matching algorithms
│   │   ├── mm_type.hpp                #   Core MM types: Candidate, MatchedCandidate, MatchResult, MatchStatus
│   │   ├── mm_algorithm.hpp           #   Base MM interface
│   │   ├── fmm/                       #   Fast Map Matching (UBODT-based, baseline)
│   │   │   ├── fmm_algorithm.hpp/.cpp #   Viterbi on UBODT
│   │   │   └── ubodt.hpp/.cpp         #   UBODT data structure + builder
│   │   ├── cmm/                       #   Covariance Map Matching (our innovation)
│   │   │   ├── cmm_algorithm.hpp/.cpp #   Full CMM engine (~2200+ lines)
│   │   │   └── cmm_log_wrapper.hpp    #   Log-level rating summary per result
│   │   ├── stmatch/                   #   Spatio-Temporal Matching
│   │   │   ├── stmatch_algorithm.hpp/.cpp
│   │   │   └── viability.hpp
│   │   └── h3mm/                      #   H3 hexagonal grid Map Matching (experimental)
│   │       └── h3mm_algorithm.hpp/.cpp
│   └── app/                           # Application entry points
│       ├── cmm_app.cpp                #   CMM CLI app (parallel via OpenMP)
│       ├── fmm_app.cpp                #   FMM CLI app (parallel via OpenMP)
│       ├── stmatch_app.cpp            #   ST-Match CLI app
│       ├── ubodt_gen_app.cpp          #   UBODT generator
│       ├── interactive_match_app.cpp  #   Console interactive matching
│       └── performance_test.cpp       #   Performance benchmark
├── input/                             # Input data
│   ├── config/                        #   XML config files
│   │   ├── cmm_config_omp.xml         #   Primary CMM config (k=8, PL_mult=10, PHMI=1e-5, lag=5)
│   │   ├── cmm_config_omp_debug.xml   #   Debug variant
│   │   ├── cmm_monte_carlo.xml        #   Monte Carlo CMM config
│   │   ├── fmm_config_omp.xml         #   FMM baseline config
│   │   └── stmatch_config.xml         #   ST-Match config
│   └── map/                           #   Road network shapefiles
│       ├── hainan/edges.shp           #   Haikou road network
│       └── hainan_ubodt_indexed.bin   #   Precomputed UBODT
├── python/                            # Analysis, plotting, data generation scripts
│   ├── generate_data_cmm.py           #   Synthetic CMM test data from virtual GNSS
│   ├── generate_monte_carlo_setup.py  #   MC simulation setup generator
│   ├── Monte_Carlo.py                 #   MC simulation runner
│   ├── process_hainan_dataset*.py     #   Dataset preprocessing (SPP extraction, NMEA→CMM)
│   ├── extract_spp_for_cmm.py         #   SPP solution → CMM trajectory
│   ├── NMEA2cmm.py                    #   NMEA log → CMM CSV
│   ├── plot_*.py                      #   Visualization scripts
│   ├── analyze_*.py                   #   Analysis scripts
│   ├── draw_*.py                      #   Map drawing scripts
│   ├── final_cmm_analysis.py          #   Comprehensive CMM analysis
│   ├── compare_cmm_filters.py         #   Filter comparison analysis
│   └── rearrange_mr.py                #   Match result post-processing
├── tests/
│   ├── CMakeLists.txt                 #   Test build (Catch2)
│   ├── test_geometry.cpp              #   Geometry unit tests
│   ├── test_network.cpp              #   Network unit tests
│   ├── test_network_graph.cpp         #   NetworkGraph unit tests
│   ├── test_fmm.cpp                   #   FMM unit tests
│   ├── test_cmm.cpp                   #   CMM unit tests
│   ├── test_stmatch.cpp               #   ST-Match unit tests
│   ├── test_ubodt.cpp                 #   UBODT unit tests
│   ├── test_reader.cpp                #   Reader unit tests
│   ├── test_interpolation.cpp         #   Interpolation unit tests
│   ├── test_candidate_search.hpp      #   Candidate search test helper
│   ├── test_interactive.cpp           #   Interactive match tests
│   ├── analyze_trust_roc.py           #   Trustworthiness ROC analysis
│   └── python/                        #   Experiment scripts (see §III.8)
│       ├── exp1_reliability_diagram.py
│       ├── exp1_lag_sweep.py
│       ├── exp1_multitraj_lag_sweep.py
│       ├── exp2_synthetic_validation.py
│       ├── exp3_phmi_analysis.py
│       ├── exp3_multiplier_sweep.py
│       ├── exp3_deep_dive.py
│       ├── exp4_ablation_ece.py
│       └── evaluate_match_metrics.py
├── dataset-hainan-06/                 # Primary real-world GNSS dataset
│   ├── cmm_traj11.csv                 #   CMM-format trajectories (traj11..traj14, traj21..traj23)
│   ├── cmm_traj12.csv
│   ├── cmm_traj13.csv
│   ├── cmm_traj14.csv
│   ├── cmm_traj21.csv
│   ├── cmm_traj22.csv
│   ├── cmm_traj23.csv
│   ├── mr/                            #   Match results + experiment outputs
│   │   ├── cmm_results*.csv           #     Various CMM result versions
│   │   ├── fmm_results*.csv           #     FMM baseline results
│   │   ├── cmm_0428_k=8_PL=100_*.csv  #     Per-trajectory results
│   │   ├── lag_sweep/                 #     Lag-steps sweep results (16 files, lag=0..100)
│   │   ├── multitraj_sweep/           #     Multi-trajectory sweep (10 files, lag=0..50)
│   │   ├── h0_test/                   #     Sequential Bayesian H0 test (8 files, lag=0..30)
│   │   ├── h0_global/                 #     Trajectory-global H0 lambda (7 files, lag=0..20)
│   │   ├── phmi_analysis/             #     PHMI=1e-5 lag sweep (10 files, lag=0..50)
│   │   ├── multiplier_sweep/          #     PHMI multiplier sweep (11 files, mult=5..15)
│   │   ├── synthetic/                 #     Synthetic data validation (18 files)
│   │   ├── ablation/                  #     Ablation study (4 files)
│   │   └── exp1_reliability.json      #     Reliability diagram results
│   └── tools/                         #   Dataset utilities
├── simulation/                        # Monte Carlo simulation data
│   ├── small_sigma/                   #   Low-noise scenario
│   │   ├── raw_data/                  #     Ground truth, observations, calibration
│   │   └── mm_result/                 #     cmm_result.csv, fmm_result.csv
│   ├── huge_sigma/                    #   High-noise scenario
│   │   ├── raw_data/                  #     Ground truth, observations, constellation
│   │   └── mm_result/                 #     cmm_result.csv, fmm_result.csv
│   ├── simulation_bak/                #   Backup of simulation data
│   └── statistic/                     #   Error stats figures
├── output/                            # Analysis output figures
│   ├── cmm_roc_curve.png              #   CMM ROC
│   ├── final_roc_comparison.png       #   ROC comparison CMM vs FMM
│   ├── final_error_dist.png           #   Error distribution
│   ├── cmm_filter_comparison.png      #   Filter comparison
│   └── monte_carlo_setup/             #   MC sky plots, hexbin
├── output_trust/                      # Trustworthiness analysis output
│   ├── cmm_roc_curve.png              #   Trust ROC
│   └── final_error_dist.png           #   Error distribution
├── python_dataset/                    # Dataset-level analysis scripts + output
│   ├── analyze_trust_roc.py
│   ├── analysis_error.py
│   └── output/                        #   ROC, error/trust comparison figures
├── tmp/                               # Temporary experiment artifacts
├── scripts/                           # PPTX generation scripts
├── example/                           # Jupyter notebook examples
│   ├── notebook/                      #   FMM, STMatch, post-processing examples
│   ├── h3/                            #   H3 hexagon demo
│   └── osmnx_example/                 #   OSMnx network + matching examples
└── docs/                              # Documentation & article
    ├── literature/                    #   MD notes on related literature
    └── Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/
                                       #   LaTeX source for IEEE T-ITS article
```

### 3. Key Algorithm: CMM (Covariance Map Matching) — Our Innovation

**Core innovation**: Introduces GNSS isotropic probabilistic model into HMM-based map matching. Unlike FMM which uses a scalar error radius, CMM uses the full 2$\times$2 covariance matrix to define anisotropic emission probabilities via Mahalanobis distance, yielding calibrated posterior probabilities.

#### 3.1 CMM Algorithm Components

The CMM engine (`src/mm/cmm/cmm_algorithm.cpp`, ~2200+ lines) implements:

1. **Anisotropic Emission Probability** — Multivariate Gaussian using the full EN-covariance matrix $\Sigma = [\sigma_E^2, \sigma_{EN}; \sigma_{EN}, \sigma_N^2]$. The emission log-probability for candidate $c$ given observation $o$ is:
   $$ \log P(o|c) = -\frac{1}{2} \log(2\pi|\Sigma|) - \frac{1}{2} (o-c)^T \Sigma^{-1} (o-c) $$

2. **Log-Space HMM** — All probability computations in log-space for numerical stability with very small probabilities. Viterbi decoding operates on log-probabilities.

3. **Fixed-Lag Smoothing** (window_length, lag_steps) — Computes posterior distribution $P(\text{state}_t | \text{obs}_{1:t+L})$ over states at step $t$ using observations up to $t + \text{lag\_steps}$ ahead. This is the forward-backward (Baum-Welch) algorithm restricted to a sliding window:
   - Forward pass accumulates log-probabilities
   - Backward pass within a limited lag window computes the smoothing posterior
   - Output layer-normalized trustworthiness, delta entropy, posterior entropy

4. **Trustworthiness** — Normalized smoothing posterior $P(\text{state}_t | \text{obs}_{1:t+L})$. This is a per-point quality metric indicating how confident the model is about each match given all observed evidence.

5. **Delta Entropy** — Information gain from prior to posterior in bits:
   $$ \Delta H_t = H(\text{prior}) - H(\text{posterior}_t) $$
   Quantifies how much uncertainty was resolved at each epoch.

6. **Posterior Entropy** — Layer posterior entropy $H(\text{posterior}_t)$ in bits, measuring remaining uncertainty.

7. **PHMI Integrity Monitoring** — Protection Level based on eigen-decomposition of the covariance matrix:
   - Error Ellipse Semi-Major Axis:
     $$ \sigma_{\text{major}} = \sqrt{\frac{\sigma_E^2 + \sigma_N^2}{2} + \sqrt{\left(\frac{\sigma_E^2 - \sigma_N^2}{2}\right)^2 + \sigma_{EN}^2}} $$
   - Scaled by configurable $K$ factor (engineering value $K \approx 5.33$ for $10^{-5}$ integrity risk)
   - $\text{PL} = K \cdot \sigma_{\text{major}} \cdot \text{PHMI multiplier}$
   - Candidates outside the PL ellipse are excluded from the search

8. **Bayesian Sequential H0 Test** — Accumulates log-likelihood ratio $\lambda_t$ over the trajectory:
   $$ \lambda_t = \prod_{\tau=1}^{t} \frac{P(o_\tau | H_1)}{P(o_\tau | H_0)} $$
   where $H_1$ assumes observation comes from a road and $H_0$ assumes GNSS-only prior. Clamped at configurable thresholds. Exported as `h0_lambda` per epoch.

9. **Gap Bridging** — Handles missing epoch observations by interpolating across gaps using the road network topology. Configurable `max_interval` parameter.

10. **Reverse Tolerance** — Allows matching to edges in the reverse direction (as a ratio of edge length). Set to `0.0` to strictly enforce one-way matching.

#### 3.2 CMM Configuration XML Parameters

Full parameter set (`input/config/cmm_config_omp.xml`):

```xml
<parameters>
    <k>8</k>                                <!-- Candidates per GPS point -->
    <min_candidates>1</min_candidates>       <!-- Minimum candidates required -->
    <protection_level_multiplier>10</protection_level_multiplier>  <!-- K factor for PL -->
    <reverse_tolerance>0.0</reverse_tolerance>  <!-- Reverse matching tolerance -->
    <normalized>false</normalized>           <!-- Use normalized Mahalanobis -->
    <use_mahalanobis>true</use_mahalanobis>  <!-- Enable anisotropic emission -->
    <window_length>100</window_length>       <!-- Fixed-lag window size -->
    <filtered>false</filtered>              <!-- Enable filtering (keep strongest trajectory) -->
    <max_interval>180.0</max_interval>       <!-- Max gap seconds for gap bridging -->
    <trustworthiness_threshold>10.0</trustworthiness_threshold>  <!-- Trust threshold -->
    <phmi>0.00001</phmi>                     <!-- PHMI target (1e-5) -->
    <lag_steps>5</lag_steps>                 <!-- Smoothing lag steps -->
    <phmi_pl_multiplier>5</phmi_pl_multiplier>  <!-- PL multiplier under PHMI mode -->
    <h0_prior_log_odds>0</h0_prior_log_odds> <!-- H0 prior log-odds -->
</parameters>
```

#### 3.3 CMM Input Format

GPS observations are CSV files with these columns:
```
id, timestamp, x, y, sde, sdn, sdu, sdne, sdeu, sdun, protection_level
```
Where `sde`, `sdn`, `sdne` form the EN covariance matrix.

### 4. Key Algorithm: FMM (Fast Map Matching) — Baseline

* **Dependency**: Requires a precomputed UBODT file (`ubodt_gen` tool).
* **Algorithm**: Standard Viterbi HMM with isotropic Gaussian emission (scalar error radius $\sigma$).
* **Performance**: Extremely fast for batch processing due to UBODT lookup tables.
* **Emission**: $P(o|c) = \frac{1}{\sqrt{2\pi}\sigma} \exp(-d^2 / 2\sigma^2)$ where $d$ is Euclidean distance.
* **Use Case**: Best for small to medium networks, baseline for comparison.

### 5. Key Algorithm: ST-Match (Spatio-Temporal Matching)

* **No precomputation** required. Uses RTree for online candidate search.
* Combines spatial and temporal features in the emission/transition model.
* Slower but more flexible than FMM.

### 6. Key Algorithm: H3MM (H3 Hexagonal Map Matching) — Experimental

* Uses Uber's H3 hexagonal grid system for spatial indexing.
* Experimental module, not yet integrated into the main analysis pipeline.

### 7. Coordinate Systems

* **Config**: Controlled by `<input_epsg>` in XML.
* **Do NOT use `convert_to_projected`** (deprecated).
* **Behavior**: The system automatically reprojects input trajectories to match the road network's EPSG (read from `.prj`).
* **Math**: Covariance matrices are rotated via Jacobian transformation during reprojection.

### 8. Data Structures

**Core Types** (`src/mm/mm_type.hpp`):

* `Candidate`: `{ index, offset, dist, edge*, point }` — A single candidate match on a road edge.
* `CandidateEmission`: `{ x, y, ep }` — Candidate coordinates + emission probability.
* `MatchedCandidate`: `{ candidate, ep, tp, cumu_prob, sp_dist, trustworthiness, delta_entropy, posterior_entropy, h0_lambda }` — Per-point matched result with all quality metrics.
* `MatchStatus`: Enum — `SUCCESS`, `PARTIAL`, `FAILED_NO_CANDIDATE`, `FAILED_DISCONNECTED`.
* `MatchResult`: `{ id, status, opt_candidate_path, opath, cpath, indices, mgeom, candidate_details, nbest_trustworthiness, sp_distances, eu_distances, original_indices }` — Complete trajectory match output.

### 9. Experiments

All experiments live under `tests/python/` as numerical exp1–4 plus supporting analyses. Results output to `dataset-hainan-06/mr/<subdir>/`.

#### Exp 1: Reliability Diagram — Calibration of Emission Probabilities
* **Script**: `tests/python/exp1_reliability_diagram.py`
* **Purpose**: Compares CMM (anisotropic Mahalanobis) vs FMM (isotropic Euclidean) emission probability calibration.
* **Metrics**: ECE (Expected Calibration Error), MCE (Maximum Calibration Error), Brier Score, LogLoss at 3m/5m/10m error thresholds.
* **Output**: `dataset-hainan-06/mr/exp1_reliability.json`

#### Exp 1a: Single-Trajectory Lag Sweep
* **Script**: `tests/python/exp1_lag_sweep.py`
* **Purpose**: Runs CMM on traj11 at lag=0,1,2,3,5,7,10,15,20,25,30,40,50,75,100 to characterize calibration vs delay trade-off.
* **Output**: `dataset-hainan-06/mr/lag_sweep/` (16 CSV files)

#### Exp 1b: Multi-Trajectory Lag Sweep
* **Script**: `tests/python/exp1_multitraj_lag_sweep.py`
* **Purpose**: Extends lag sweep to all 7 trajectories (traj11,12,13,14,21,22,23) at lag=0,5,10,15,20,25,30,35,40,50.
* **Output**: `dataset-hainan-06/mr/multitraj_sweep/` (10 CSV files, ~5.5MB each)

#### Exp 2: Synthetic Data Validation
* **Script**: `tests/python/exp2_synthetic_validation.py`
* **Purpose**: Generates synthetic trajectories on the Haikou road network (vehicle ON road, perfect ground truth), runs CMM+FMM+lag sweep, compares calibration metrics against real-data results. Validates internal validity of the probabilistic model.
* **Output**: `dataset-hainan-06/mr/synthetic/` (18 files: CSVs, JSON, metadata)

#### Exp 3: PHMI-Enhanced Emission + Protection Level Coverage
* **Script**: `tests/python/exp3_phmi_analysis.py`
* **Purpose**: Evaluates GNSS Protection Level (PL) coverage under PHMI=1e-5. Computes PL coverage statistics and ECE comparison with/without PHMI enhancement.
* **Output**: `dataset-hainan-06/mr/phmi_analysis/` (10 CSV files)

#### Exp 3a: PHMI Multiplier Sweep
* **Script**: `tests/python/exp3_multiplier_sweep.py`
* **Purpose**: Sweeps PHMI PL multiplier from 5 to 15 to find optimal value balancing coverage and precision.
* **Output**: `dataset-hainan-06/mr/multiplier_sweep/` (11 CSV files)

#### Exp 3b: PHMI Deep Dive
* **Script**: `tests/python/exp3_deep_dive.py`
* **Purpose**: Bin-level ECE analysis, traj21 first 100 epochs, all-outside-PL frequency analysis, warning trigger effectiveness evaluation.
* **Output**: Console + figures

#### Exp 4: ECE Ablation Study
* **Script**: `tests/python/exp4_ablation_ece.py`
* **Purpose**: 100-bin ECE ablation across configurations: FMM baseline → CMM lag=0 → CMM+lag → CMM+lag+PHMI, on both synthetic and real data. Quantifies per-component contribution to calibration.
* **Output**: `dataset-hainan-06/mr/ablation/` (4 CSV files: real_lag0, real_lag20, synth_lag40)

#### Supporting Analyses

* **ROC Curves** (`tests/analyze_trust_roc.py`, `python_dataset/analyze_trust_roc.py`) — Trustworthiness as binary error detector, CMM vs FMM at 3m/5m/10m/15m thresholds.
* **Error Distributions** (`python/analyze_cmm_error.py`, `python/analyze_likelihood.py`, `python/final_cmm_analysis.py`) — Error histograms, CDFs, filter comparisons.
* **Delta Entropy Analysis** — Information gain per epoch, posterior entropy tracking.
* **H0 Bayesian Test** (`dataset-hainan-06/mr/h0_test/`, `dataset-hainan-06/mr/h0_global/`) — Sequential and trajectory-global Bayesian hypothesis testing.
* **PPTX Generation** (`scripts/gen_cmm_section4_pptx.py`, `scripts/gen_cmm_full_deck.py`) — Generate presentation slide decks for Section 4 trustworthiness evaluation (14 slides).

### 10. Dataset: Hainan-06

* **Location**: Haikou, Hainan, China
* **Road network**: `input/map/hainan/edges.shp` (EPSG projected)
* **UBODT**: `input/map/hainan_ubodt_indexed.bin`
* **7 GNSS trajectories**: Collected via low-cost GNSS receiver with SPP + covariance output
  * traj11, traj12, traj13, traj14, traj21, traj22, traj23
* **Input format**: CSV with columns `id, timestamp, x, y, sde, sdn, sdu, sdne, sdeu, sdun, protection_level`
* **Preprocessing**: `python/extract_spp_for_cmm.py` extracts SPP solutions; `python/NMEA2cmm.py` converts NMEA logs; `python/process_hainan_dataset*.py` handles rigid registration and LLA processing

### 11. Simulation

Monte Carlo simulation validating CMM under controlled noise conditions:

* **Small sigma**: Low GNSS uncertainty — validates baseline performance
* **Huge sigma**: High GNSS uncertainty — validates CMM advantage over FMM
* **Setup**: `python/generate_monte_carlo_setup.py` generates test points + constellation geometry; `python/Monte_Carlo.py` runs the simulation
* **Results**: `simulation/small_sigma/mm_result/` and `simulation/huge_sigma/mm_result/` contain `cmm_result.csv` and `fmm_result.csv`

### 12. Python API

```python
import sys
sys.path.insert(0, 'build/python')
from fmm import Network, NetworkGraph, UBODT, FastMapMatch, FastMapMatchConfig

network = Network("edges.shp", "id", "u", "v", False)
graph = NetworkGraph(network)
ubodt = UBODT.read_ubodt_file("ubodt.bin")
config = FastMapMatchConfig(k=8, radius=300, error=50)
fmm = FastMapMatch(network, graph, ubodt)
result = fmm.match_traj(trajectory, config)
```

### 13. LaTeX Article

* **Path**: `docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/`
* **Target**: IEEE Transactions on Intelligent Transportation Systems (T-ITS)
* **Core contribution**: CMM algorithm with anisotropic error ellipse modeling, trustworthiness framework via fixed-lag smoothing posterior, PHMI integrity monitoring via Bayesian sequential testing
* **Literature notes**: `docs/literature/` contains MD summaries of related work (some may be outdated — verify against current codebase before citing)

---

## IV. Prompt Suggestions

At the end of your response, suggest 2 follow-up prompts.

```bash

```

```bash

```
