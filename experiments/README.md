# Experiments Suite for IEEE T-ITS Article

Reorganized experiment scripts and synthetic data generation pipeline for
"Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse."

## Directory Structure

```
experiments/
  config/
    experiment_params.json       # (TBD) Shared parameter defaults
  data/                          # Generated synthetic datasets
    sigma_01/ ~ sigma_30/        # Per-sigma-level (20 trajs each)
    sigma_30/with_occlusion/     # Cross-road occlusion only
    sigma_30/with_fault/         # Fault injection only
    sigma_30/with_occlusion/with_fault/  # Both occlusion + fault
    sigma_01_to_30/              # Full sweep (10 trajs/level, 1-30m)
  scripts/
    generate_data_cmm.py         # Enhanced data generator
    batch_generate.py            # Orchestrates all data generation
    utils.py                     # Shared I/O, metrics, plotting
    exp1_covariance_validation.py
    exp2_stanford_pl.py
    exp3_parameter_sensitivity.py
  output/
    1_covariance_validation/     # Chi2 histogram, P-P, whitened, Rayleigh CDF
    2_stanford/                  # Stanford plots + P_md/P_fa table
    3_parameter_sensitivity/     # Point error vs sigma, accuracy vs sigma
```

## Key Changes from Original (python/generate_data_cmm.py)

1. **MAX_SIGMA_PR**: extended from 10.0 → 30.0
2. **Perpendicular road occlusion**: satellites in cross-track direction ±45° below 30° elevation are removed (simulates buildings/trees)
3. **Step fault injection**: random satellite gets U(100,500)m bias with configurable probability
4. **Per-point edge IDs**: `ground_truth.csv` now includes `point_edge_ids` (JSON array) for segment-level accuracy computation

## Usage

### 1. Generate all data
```bash
cd /home/ncz/fmm_sjtugnc
python experiments/scripts/batch_generate.py \
    --output-dir experiments/data \
    --shapefile input/map/haikou/edges.shp \
    --seed 42
```

### 2. Run experiments
```bash
# Experiment 1: Covariance model validation
python experiments/scripts/exp1_covariance_validation.py \
    --data-root experiments/data \
    --output-dir experiments/output/1_covariance_validation

# Experiment 2: Stanford PL analysis
python experiments/scripts/exp2_stanford_pl.py \
    --data-root experiments/data \
    --output-dir experiments/output/2_stanford

# Experiment 3: Parameter sensitivity
python experiments/scripts/exp3_parameter_sensitivity.py \
    --data-root experiments/data \
    --output-dir experiments/output/3_parameter_sensitivity
```

## New CLI Parameters (generate_data_cmm.py)

| Flag | Default | Description |
|------|---------|-------------|
| `--occlusion-angle` | 0.0 | Half-width of cross-track occlusion wedge (deg) |
| `--occlusion-elevation-cutoff` | 30.0 | Elevation below which occlusion applies (deg) |
| `--fault-probability` | 0.0 | Per-trajectory fault injection probability |
| `--fault-magnitude-min` | 100.0 | Minimum fault bias magnitude (m) |
| `--fault-magnitude-max` | 500.0 | Maximum fault bias magnitude (m) |

## Experiment setup
1. **Trajectory Number**: for each set of sigma, at least 10 trajectories is needed and 1000 epochs for each trajectory. When generate the synthesic data, the ground_truth point in lla, the ground_truth road segment in the id/key of .shp should be both included. To analyze the effect of wrongly estimated sigma_rho on the matching results, a special set of data is needed: use the sigma=20m in the RAIM and WLS, but generate a series of psuedorange observation from 10m to 30m, thus the effects of wrong emission model due to over-confidence or over-conservative of sigma_rho on CMM can be obtained. 
2. **Experiment Objectives**: experiment objectives is listed respectively: exp1 is to valid the model about covariance is correct rather than isotropic emission model; exp2 is to valid the performance of RAIM including its PL results in stanford plot and P_fa, P_ma for the fde part, remember PL is calculated with the fault-free observations-use the sets of pseudorange observation without the ones recognized as fault; exp3 is sensitive analysis of CMM vs. FMM on different sigma_rou and different sample rate-by drop gnss results. exp4 is to analyze the effect of wrong emission model due to wrong sigma_rho on CMM.

