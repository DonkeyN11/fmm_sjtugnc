# Experiments Suite for IEEE T-ITS Article

Reorganized experiment scripts and synthetic data generation pipeline for
"Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse."

## Directory Structure

```
experiments/
  README.md                        # This file
  config/
    reverse_edge_map.json          # 135K reverse-edge pair mappings for Hainan network
  data/                            # Generated synthetic datasets
    sigma_01/ ~ sigma_30/          # Per-sigma-level (20 trajs each, 1000 epochs/traj)
      no_occlusion/no_fault/       #   Nominal (clean) condition
    sigma_30/
      no_occlusion/with_fault/     #   Step fault injection only
      with_occlusion/no_fault/     #   Cross-road occlusion only
      with_occlusion/with_fault/   #   Occlusion + fault combined
    sigma_mismatch/                # Sigma mismatch datasets for Exp4
      pr10_wls20/ ~ pr30_wls20/    #   pr = true pseudorange noise, wls = assumed
  scripts/                         # All experiment & analysis scripts (22 files)
  output/
    1_covariance_validation/       # Per-sigma 4-panel validation figures + KS summary
    2_stanford/                    # Per-condition Stanford plots + P_md/P_fa table
    3_full_matching/               # Sigma sweep + sample rate figures + CSV summaries
    4_sigma_mismatch/              # Mismatch analysis figures + CSV
    5_degraded/                    # Degraded condition comparison figures + CSV
    spp_error/                     # SPP vs RTK error analysis figures + CSV
```

## Scripts Reference

### Data Generation
| Script | Purpose |
|--------|---------|
| `generate_data_cmm.py` | Enhanced synthetic data generator (WLS+RAIM, occlusion, fault injection) |
| `batch_generate.py` | Orchestrates all data generation steps in parallel |
| `extract_spp_for_cmm.py` | Parse NMEA SPP output → CMM-format CSV with covariance |
| `NMEA2cmm.py` | Alternative NMEA-to-CMM converter |
| `process_hainan_dataset_lla_rigid.py` | Full Hainan dataset processing pipeline (SPP→CMM→FMM/CMM matching→Mapbox) |

### Synthetic Experiments (Exp1–5)
| Script | Experiment | Purpose |
|--------|-----------|---------|
| `exp1_covariance_validation.py` | Exp1 | χ² histogram, P-P, whitened errors, Rayleigh CDF |
| `exp2_stanford_pl.py` | Exp2 | Stanford plots + $P_{\text{md}}$/$P_{\text{fa}}$ across conditions |
| `exp3_full_matching.py` | Exp3 | Sigma sweep + sample rate sensitivity (CMM vs FMM) |
| `exp4_sigma_mismatch.py` | Exp4 | Wrong emission model effects (over-confident vs over-conservative) |
| `exp5_degraded_conditions.py` | Exp5 | Robustness under fault/occlusion at σ=30m |

### Real-Vehicle SPP/RTK Analysis (Exp6)
| Script | Purpose |
|--------|---------|
| `exp6_compare_spp_rtk.py` | Time-align SPP positions vs RTK ground truth, compute error statistics |
| `exp6_stanford_spp_rtk.py` | Stanford plot: SPP error vs RAIM PL from `cmm_input_points.csv` |
| `plot_spp_error.py` | 6-panel SPP error summary figure |
| `plot_spp_per_traj.py` | Per-trajectory SPP error time-series + histogram figures |

### Infrastructure & Visualization
| Script | Purpose |
|--------|---------|
| `utils.py` | Shared I/O, metrics (ECE, ROC/AUC), plotting helpers |
| `compute_raim_pl.py` | RAIM PL from RINEX 3.04 observations (canonical copy) |
| `merge_raim_pl.py` | Merge RAIM PL into CMM input CSV |
| `run_cmm_matching.py` | Standalone CMM batch runner |
| `mapbox_viz.py` | Mapbox 3D visualization: CMM/FMM results + GT + road network |
| `mapbox_spp_rtk.py` | Mapbox visualization: SPP + covariance ellipses vs RTK ground truth |
| `fig_stanford_combined.py` | Combined 4-panel Stanford figure for paper |

---

## Data Generation

### Generator: `scripts/generate_data_cmm.py`

The core synthetic data generator extends `python/generate_data_cmm.py` with:

1. **MAX_SIGMA_PR**: extended from 10.0 to 30.0
2. **Perpendicular road occlusion**: satellites within $\pm 45^\circ$ of cross-track direction and below $30^\circ$ elevation are removed, simulating building/tree blockage
3. **Step fault injection**: with probability $P_{\text{fault}}$, a random satellite receives a constant bias $\Delta\rho \sim \mathcal{U}(100,500)$ m on all its pseudorange observations
4. **RAIM-FDE module**: global $\chi^2$ residual test ($P_{\text{FA}}=10^{-5}$) + local $w$-test + iterative satellite exclusion (max 3, min 5)
5. **Per-point edge IDs**: `ground_truth.csv` includes `point_edge_ids` (JSON array) for segment-level accuracy computation
6. **WGS84 output**: observations written in lon/lat degrees matching `dataset-hainan-06/cmm_input_points.csv` format

**Output files per dataset:**
| File | Description |
|------|-------------|
| `observations.csv` | GNSS observations: id, seq, timestamp, x, y, sde, sdn, sdu, sdne, sdeu, sdun, protection_level, fde_detected, fde_excluded, fde_sse, fde_threshold |
| `ground_truth.csv` | Ground truth: id, geom (UTM), timestamp, edge_ids, point_edge_ids |
| `ground_truth_points.csv` | Per-epoch GT points (WGS84): id, seq, timestamp, x, y |
| `metadata.json` | Trajectory metadata: sigma_pr, constellation, edge_ids |

**New CLI parameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--occlusion-angle` | 0.0 | Half-width of cross-track occlusion wedge (deg) |
| `--occlusion-elevation-cutoff` | 30.0 | Elevation below which occlusion applies (deg) |
| `--fault-probability` | 0.0 | Per-trajectory fault injection probability |
| `--fault-magnitude-min` | 100.0 | Minimum fault bias magnitude (m) |
| `--fault-magnitude-max` | 500.0 | Maximum fault bias magnitude (m) |
| `--sigma-pr-wls` | (same as sigma) | Pseudorange noise assumed by WLS solver (for mismatch exp) |

### Batch Generator: `scripts/batch_generate.py`

Generates all datasets in parallel via `ThreadPoolExecutor`:

```bash
python experiments/scripts/batch_generate.py \
    --output-dir experiments/data \
    --shapefile input/map/hainan/edges.shp \
    --seed 42 \
    --jobs 8
```

**Generation steps** (use `--step` to select subsets):
| Step | Description | Datasets |
|------|-------------|----------|
| `sigma_levels` | 20 trajectories per $\sigma \in \{1,5,10,15,20,25,30\}$ m | `sigma_01/` ~ `sigma_30/` |
| `occlusion` | $\sigma=30$ m + cross-road occlusion | `sigma_30/with_occlusion/no_fault/` |
| `fault` | $\sigma=30$ m + step fault | `sigma_30/no_occlusion/with_fault/` |
| `occlusion_fault` | $\sigma=30$ m + occlusion + fault | `sigma_30/with_occlusion/with_fault/` |
| `full_sweep` | 10 trajectories per level, $\sigma$ uniform in $[1,30]$ m | `sigma_01_to_30/` |

**Note:** Sigma mismatch datasets (Exp4, `sigma_mismatch/pr*_wls20/`) are generated separately with `--sigma-pr-wls 20` and varying `--min-sigma-pr`/`--max-sigma-pr`.

---

## Common Configuration (k=16, per README specification)

All experiments use the same HMM and matching parameters:

### CMM Configuration
```xml
<k>16</k>
<min_candidates>1</min_candidates>
<protection_level_multiplier>3</protection_level_multiplier>
<reverse_tolerance>0.0</reverse_tolerance>
<normalized>false</normalized>
<use_mahalanobis>true</use_mahalanobis>
<window_length>100</window_length>
<filtered>false</filtered>
<max_interval>180.0</max_interval>
<trustworthiness_threshold>0.0</trustworthiness_threshold>
<phmi>0.00001</phmi>
<lag_steps>0</lag_steps>
<phmi_pl_multiplier>1</phmi_pl_multiplier>
<h0_prior_log_odds>0</h0_prior_log_odds>
```

### FMM Configuration
```xml
<k>16</k>
<r>0.03</r>
<pf>0</pf>
<gps_error>0.001</gps_error>
```

Both use `input/map/hainan/edges.shp` (road network) and `input/map/hainan_ubodt_indexed.bin` (UBODT).

---

## Experiment 1: Covariance Model Validation

**Objective:** Verify that the WLS-derived GNSS covariance matrix $\boldsymbol{\Sigma}_i$ is statistically consistent with the actual positioning errors. Under correct covariance, the squared Mahalanobis distance $D_i^2 = \mathbf{e}_i^{\text{T}} \boldsymbol{\Sigma}_i^{-1} \mathbf{e}_i$ follows $\chi^2(2)$.

**Script:** `experiments/scripts/exp1_covariance_validation.py`

**Input data:** All `sigma_01/` ~ `sigma_30/` (no_occlusion/no_fault), 20 trajectories each, 20,000 epochs per sigma level.

**Diagnostics (4-panel per sigma):**
1. $\chi^2(2)$ histogram: empirical $D_i^2$ distribution vs. theoretical PDF
2. P-P plot: empirical CDF vs. theoretical $\chi^2(2)$ CDF
3. Whitened error scatter: $\boldsymbol{\Sigma}_i^{-1/2}\mathbf{e}_i$ in plane with $1\sigma$/$2\sigma$ circles
4. Rayleigh CDF: radial error CDF vs. Rayleigh prediction from covariance

**Metrics:** Kolmogorov-Smirnov (KS) statistic for proposed (covariance-based) vs. isotropic model.

```bash
python experiments/scripts/exp1_covariance_validation.py \
    --data-root experiments/data \
    --output-dir experiments/output/1_covariance_validation
```

**Results summary:**

| $\sigma_{\rho}$ | KS (Covariance) | KS (Isotropic) | N |
|:---:|:---:|:---:|:---:|
| 1 m | 0.0088 | 0.0702 | 10,000 |
| 5 m | 0.0060 | 0.1377 | 10,000 |
| 10 m | 0.0084 | 0.1326 | 10,000 |
| 15 m | 0.0083 | 0.0738 | 10,000 |
| 20 m | 0.0094 | 0.0806 | 10,000 |
| 25 m | 0.0140 | 0.1128 | 10,000 |
| 30 m | 0.0082 | 0.1169 | 10,000 |

**Key finding:** The covariance-based model achieves KS $< 0.015$ across all $\sigma_{\rho}$ levels (close to the theoretical $\chi^2(2)$), while the isotropic model shows KS $0.07$--$0.14$ (significant deviation). This validates that the WLS-derived $\boldsymbol{\Sigma}_i$ correctly captures the stochastic structure of GNSS positioning errors.

**Output:** `experiments/output/1_covariance_validation/validation_sigma_XX.png` (7 files), `summary_table.csv`

---

## Experiment 2: Stanford Plot and Protection Level Performance

**Objective:** Evaluate (i) whether the RAIM-derived HPL bounds the true horizontal error at the specified integrity risk ($10^{-5}$), and (ii) whether the FDE module correctly identifies and excludes faulty satellites.

**Script:** `experiments/scripts/exp2_stanford_pl.py`

**Input data:** All `sigma_01/` ~ `sigma_30/` (nominal) plus sigma_30 degradation conditions (fault, occlusion, occlusion+fault).

**Degradation modes tested:**
- **Clean:** no occlusion, no fault ($\sigma = 1,5,10,15,20,25,30$ m)
- **Fault:** $\sigma=30$ m, step fault $P=0.3$, $U(100,500)$ m
- **Occlusion:** $\sigma=30$ m, cross-road occlusion $45^\circ$ wedge, elevation cutoff $30^\circ$
- **Occlusion+Fault:** $\sigma=30$ m, both combined

**Metrics:**
- $P_{\text{md}}$: missed-detection rate (HMI events where error $>$ PL with fault present)
- $P_{\text{fa}}$: false-alarm rate (error $>$ PL under fault-free condition)
- Stanford plot: true horizontal error vs. HPL per epoch, color-coded by HMI status

**RAIM parameters:** $P_{\text{FA}} = 10^{-5}$, $P_{\text{MD}} = 10^{-3}$, max 3 FDE exclusions, min 5 satellites.

```bash
python experiments/scripts/exp2_stanford_pl.py \
    --data-root experiments/data \
    --output-dir experiments/output/2_stanford
```

**Results summary:**

| Condition | $\sigma_{\rho}$ | $P_{\text{md}}$ | $P_{\text{fa}}$ | Mean PL (m) | Mean Error (m) |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Clean | 1 m | 0 | 0.05 | 14.4 | 1.9 |
| Clean | 10 m | 0 | 0.05 | 139.4 | 22.0 |
| Clean | 20 m | 0 | 0.05 | 268.6 | 35.7 |
| Clean | 30 m | 0 | 0.05 | 510.0 | 61.8 |
| Fault | 30 m | 0.0075 | 0.05 | 434.0 | 76.6 |
| Occlusion | 30 m | 0.0002 | 0.05 | 259.0 | 44.3 |
| Occlusion+Fault | 30 m | 0.0089 | 0.05 | 288.9 | 54.0 |

**Key finding:** Zero HMI events under nominal conditions ($P_{\text{md}}=0$). FDE reduces $P_{\text{md}}$ to $<1\%$ under fault injection. The occultation condition reduces mean PL (259 m vs. 510 m clean) because cross-track satellite removal degrades geometry but the remaining along-track satellites constrain HPL.

**Output:** `experiments/output/2_stanford/stanford_sigma_XX.png` (10 files), `summary_table.csv`

---

## Experiment 3: Parameter Sensitivity — CMM vs. FMM

**Objective:** Comprehensive comparison of CMM vs. FMM across (a) pseudorange noise levels $\sigma_{\rho} \in [1,30]$ m and (b) sample rates $f_s \in [0.1, 1.0]$ Hz.

**Script:** `experiments/scripts/exp3_full_matching.py`

### 3a. Sigma Sensitivity

**Input data:** `sigma_01/` ~ `sigma_30/no_occlusion/no_fault/`, 20 trajectories × 1000 epochs each, 1 Hz.

**Metrics (per sigma, per algorithm):**
- Point error (mean, median, RMSE, P95)
- Segment-level accuracy (matched edge ID == ground truth, reverse-edge aware via `reverse_edge_map.json`)
- ECE (10-bin) and MCE of trustworthiness score
- ROC AUC for mismatch detection (correct edge vs. wrong edge)
- Reliability diagram per sigma
- ROC curves per sigma

### 3b. Sample Rate Sensitivity

For representative sigma levels $\sigma_{\rho} \in \{5, 15, 25\}$ m, the original 1 Hz observations are decimated to intervals of 2 s, 5 s, and 10 s by retaining every $N$-th epoch. CMM and FMM are re-run on each subsampled dataset.

**Subsampled datasets** are created in-place under each condition directory (e.g., `sigma_05/no_occlusion/no_fault/subsample_2s/`).

**Timestamp-based matching** is used for metric computation because CMM reassigns sequential `seq` values (0,1,2,...) regardless of input `seq`, which would break matching under decimation. Ground truth is keyed by `(id, timestamp)` with both exact-float and rounded-int lookups to handle CMM's integer timestamp rounding.

```bash
python experiments/scripts/exp3_full_matching.py \
    --data-root experiments/data \
    --cmm-bin build/cmm \
    --fmm-bin build/fmm \
    --output-dir experiments/output/3_full_matching \
    --jobs 8
```

**Results — Sigma sweep (1 Hz):**

| $\sigma_{\rho}$ | CMM Acc | CMM ECE | CMM AUC | FMM Acc | FMM ECE | FMM AUC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 m | 0.999 | 0.243 | 0.803 | 0.998 | 0.003 | 0.594 |
| 5 m | 0.999 | 0.238 | 0.842 | 0.894 | 0.224 | 0.816 |
| 10 m | 0.960 | 0.163 | 0.711 | 0.721 | 0.428 | 0.681 |
| 15 m | 0.989 | 0.262 | 0.664 | 0.743 | 0.466 | 0.694 |
| 20 m | 0.986 | 0.299 | 0.702 | 0.758 | 0.496 | 0.564 |
| 25 m | 0.774 | 0.294 | 0.731 | 0.722 | 0.540 | 0.410 |
| 30 m | 0.768 | 0.134 | 0.828 | 0.562 | 0.479 | 0.480 |

**Results — Sample rate (selected levels):**

| $\sigma_{\rho}$ | Rate | CMM Acc | CMM ECE | FMM Acc | FMM ECE |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 m | 1 s | 0.999 | 0.238 | 0.894 | 0.224 |
| 5 m | 2 s | 0.999 | 0.257 | 0.975 | 0.045 |
| 5 m | 5 s | 0.999 | 0.282 | 0.997 | 0.010 |
| 5 m | 10 s | 0.997 | 0.294 | 0.996 | 0.019 |
| 15 m | 1 s | 0.989 | 0.262 | 0.743 | 0.466 |
| 15 m | 2 s | 0.991 | 0.240 | 0.795 | 0.342 |
| 15 m | 5 s | 0.936 | 0.214 | 0.919 | 0.138 |
| 15 m | 10 s | 0.931 | 0.197 | 0.993 | 0.077 |
| 25 m | 1 s | 0.774 | 0.294 | 0.722 | 0.540 |
| 25 m | 2 s | 0.775 | 0.306 | 0.732 | 0.481 |
| 25 m | 5 s | 0.781 | 0.325 | 0.791 | 0.299 |
| 25 m | 10 s | 0.881 | 0.353 | 0.914 | 0.140 |

**Key findings:**
1. CMM dominates FMM in segment accuracy across all sigma levels; the gap widens at $\sigma_{\rho} \ge 10$ m.
2. CMM ECE remains below 0.30 across all $\sigma_{\rho}$, while FMM ECE exceeds 0.40 at moderate-to-high noise.
3. CMM is robust to sample rate (accuracy stable at 1--10 s intervals); FMM improves at lower rates because temporal decimation acts as implicit noise filtering, partially compensating for the lack of anisotropic emission.

**Output:**
- `experiments/output/3_full_matching/comparison_overview.png` (6-panel)
- `experiments/output/3_full_matching/sample_rate_sensitivity.png` (4-panel)
- `experiments/output/3_full_matching/reliability_all_sigmas.png` (8-panel)
- `experiments/output/3_full_matching/sigma_sweep_table.csv`
- `experiments/output/3_full_matching/sample_rate_table.csv`

**Raw match results:** `experiments/data/sigma_XX/no_occlusion/no_fault/cmm_result.csv`, `fmm_result.csv` (14 files + 18 subsample files)

---

## Experiment 4: Wrong Emission Model — Sigma Mismatch Effects

**Objective:** Analyze the effect of a misspecified emission model where the WLS solver assumes $\sigma_{\text{wls}} = 20$ m but the true pseudorange noise $\sigma_{\rho}^{\text{true}}$ varies from 10 to 30 m. This creates:
- **Over-conservative** emission ($\sigma_{\rho}^{\text{true}} < \sigma_{\text{wls}}$): covariance ellipse too large, emission too flat
- **Over-confident** emission ($\sigma_{\rho}^{\text{true}} > \sigma_{\text{wls}}$): covariance ellipse too small, emission too peaked

**Script:** `experiments/scripts/exp4_sigma_mismatch.py`

**Input data:** `sigma_mismatch/pr10_wls20/` ~ `pr30_wls20/`, 10 trajectories × 1000 epochs each.

**Metrics:** Point error, segment accuracy, ECE, ROC AUC, trustworthiness separation ($\mu_{\text{correct}} - \mu_{\text{wrong}}$), per-bin ECE decomposition, trustworthiness distribution histograms (correct vs. wrong edges).

```bash
python experiments/scripts/exp4_sigma_mismatch.py \
    --data-root experiments/data/sigma_mismatch \
    --cmm-bin build/cmm \
    --fmm-bin build/fmm \
    --output-dir experiments/output/4_sigma_mismatch \
    --jobs 5
```

**Results summary:**

| $\sigma_{\rho}^{\text{true}}$ | Mismatch | CMM Acc | CMM ECE | CMM Trust Sep | FMM Acc | FMM ECE |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 10 m | $-10$ (over-cons.) | 0.999 | 0.082 | 0.439 | 0.525 | 0.378 |
| 15 m | $-5$ | 0.878 | 0.192 | 0.252 | 0.632 | 0.477 |
| 20 m | $0$ (correct) | 0.782 | **0.048** | **0.496** | 0.446 | 0.410 |
| 25 m | $+5$ | 0.899 | 0.234 | 0.478 | 0.673 | 0.478 |
| 30 m | $+10$ (over-conf.) | 0.931 | 0.215 | 0.188 | 0.587 | 0.464 |

**Key findings:**
1. CMM achieves minimum ECE (0.048) when $\sigma_{\text{wls}} = \sigma_{\rho}^{\text{true}}$ — the emission model is best calibrated when correctly specified.
2. ECE degradation is **asymmetric**: over-confident emission ($\sigma_{\rho}^{\text{true}} > \sigma_{\text{wls}}$) degrades ECE more severely (0.22--0.23) than over-conservative emission (0.08 at $\sigma_{\rho}^{\text{true}}=10$ m).
3. Trustworthiness separation between correct and wrong edges peaks at correct specification (0.496) and drops to 0.188 under severe over-confidence.
4. **Caveat:** The 5 mismatch datasets have different ground truth paths (different random seeds). Cross-comparison of absolute accuracy/error is confounded by path difficulty; the ECE trend is directionally informative.

**Output:**
- `experiments/output/4_sigma_mismatch/mismatch_analysis.png` (6-panel)
- `experiments/output/4_sigma_mismatch/trustworthiness_distribution.png`
- `experiments/output/4_sigma_mismatch/ece_per_bin.png`
- `experiments/output/4_sigma_mismatch/mismatch_summary.csv`

---

## Experiment 5: Robustness Under Degraded Conditions

**Objective:** Evaluate CMM vs. FMM reliability under four degradation scenarios at $\sigma_{\rho}=30$ m to test whether the proposed method maintains advantage under realistic GNSS impairments.

**Script:** `experiments/scripts/exp5_degraded_conditions.py`

**Conditions tested:**
| Key | Occlusion | Fault | Description |
|-----|:---:|:---:|-------------|
| `clean` | no | no | Nominal, open-sky |
| `fault` | no | yes | Single-satellite step bias $U(100,500)$ m, $P=0.3$ |
| `occlusion` | yes | no | Cross-road wedge $\pm 45^\circ$, elev. cutoff $30^\circ$ |
| `both` | yes | yes | Occlusion + fault combined |

**Metrics:** Point error, segment accuracy, ECE, ROC AUC, trustworthiness separation, reliability diagrams (CMM, all conditions overlaid), ROC curves (CMM, all conditions overlaid), trustworthiness distribution histograms (correct vs. wrong per condition).

```bash
python experiments/scripts/exp5_degraded_conditions.py \
    --cmm-bin build/cmm \
    --fmm-bin build/fmm \
    --output-dir experiments/output/5_degraded \
    --jobs 4
```

**Results summary:**

| Condition | CMM Acc | CMM ECE | CMM AUC | FMM Acc | FMM ECE | FMM AUC |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Clean | 0.768 | 0.134 | 0.828 | 0.562 | 0.479 | 0.480 |
| Fault | 0.717 | 0.247 | 0.706 | 0.591 | 0.476 | 0.489 |
| Occlusion | 0.897 | 0.181 | 0.835 | 0.610 | 0.464 | 0.533 |
| Occlusion+Fault | 0.780 | 0.147 | 0.764 | 0.610 | 0.458 | 0.528 |

**Key findings:**
1. CMM maintains 71.7--89.7% segment accuracy across all conditions vs. FMM's 56.2--61.0%.
2. Fault injection causes the largest ECE increase for CMM (0.134 $\to$ 0.247), indicating that undetected faults inflate posterior entropy.
3. Cross-road occlusion does not significantly harm CMM — the Mahalanobis emission adapts to the elongated cross-track covariance.
4. **Caveat:** The 4 conditions have different ground truth paths. Cross-condition comparison of absolute accuracy is confounded by path difficulty.

**Output:**
- `experiments/output/5_degraded/degraded_comparison.png` (6-panel)
- `experiments/output/5_degraded/accuracy_breakdown.png`
- `experiments/output/5_degraded/trust_distribution.png` (4-panel)
- `experiments/output/5_degraded/degraded_summary.csv`

---

## Quick Reference: Run All Experiments

```bash
cd /home/ncz/fmm_sjtugnc

# Step 0: Build CMM and FMM binaries
cd build && make -j$(nproc) cmm fmm && cd ..

# Step 1: Generate all data (if not already present)
python experiments/scripts/batch_generate.py \
    --output-dir experiments/data \
    --shapefile input/map/hainan/edges.shp \
    --seed 42

# Step 2: Run experiments sequentially
python experiments/scripts/exp1_covariance_validation.py \
    --data-root experiments/data \
    --output-dir experiments/output/1_covariance_validation

python experiments/scripts/exp2_stanford_pl.py \
    --data-root experiments/data \
    --output-dir experiments/output/2_stanford

python experiments/scripts/exp3_full_matching.py \
    --data-root experiments/data \
    --cmm-bin build/cmm --fmm-bin build/fmm \
    --output-dir experiments/output/3_full_matching \
    --jobs 8

python experiments/scripts/exp4_sigma_mismatch.py \
    --data-root experiments/data/sigma_mismatch \
    --cmm-bin build/cmm --fmm-bin build/fmm \
    --output-dir experiments/output/4_sigma_mismatch \
    --jobs 5

python experiments/scripts/exp5_degraded_conditions.py \
    --cmm-bin build/cmm --fmm-bin build/fmm \
    --output-dir experiments/output/5_degraded \
    --jobs 4

# Step 3: Copy figures to paper directory (optional)
cp experiments/output/1_covariance_validation/validation_sigma_10.png \
   "docs/../figs/fig2_consistency_overview.png"
cp experiments/output/3_full_matching/comparison_overview.png \
   "docs/../figs/sigma_sweep.png"
# ... (see Copy experiment output figures section above)
```

---

## Known Issues and Caveats

1. **sigma_10 FMM empty output:** A known FMM binary bug causes empty output (header only) at $\sigma=10$ m despite matching ~8000 points. All other sigma levels produce valid FMM results.

2. **sigma_mismatch uncontrolled comparison:** The 5 mismatch datasets (Exp4) use different random paths (different `ground_truth.csv` MD5 hashes). For a properly controlled experiment, regenerate with a shared seed so all pr levels share identical trajectories and satellite geometries.

3. **sigma_30 degraded uncontrolled comparison:** Similarly, the 4 conditions in Exp5 (clean/fault/occlusion/both) have different ground truth paths. Regenerate with shared seed for controlled A/B testing.

4. **Timestamp rounding:** CMM rounds timestamps to integers while GT points have fractional seconds. The metric computation in exp3-5 handles this by storing both exact-float and rounded-int keys in the GT lookup tables.

5. **Reverse edge matching:** The Haikou road network contains 135K bidirectional edge pairs (e.g., 36014 $\leftrightarrow$ 36016). The `reverse_edge_map.json` in `experiments/config/` maps each edge to its reverse counterpart. Segment accuracy treats a match to a reverse-direction edge as correct.
