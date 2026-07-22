# TMM: Trustworthy Map Matching

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.TXT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](CMakeLists.txt)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()

**Covariance-consistent HMM map matching with calibrated trustworthiness evaluation.**

TMM extends the [Fast Map Matching (FMM)](https://github.com/cyang-kth/fmm) framework by integrating GNSS positioning covariance matrices and Receiver Autonomous Integrity Monitoring (RAIM) protection levels into a probabilistically normalized Hidden Markov Model. The result is a **calibrated per-epoch trustworthiness score** — a filtering posterior probability that the matched road segment is correct — enabling safety-critical integrity monitoring in autonomous driving, road pricing, and high-precision navigation.

> **Paper**: Ning, C., Yang, R., Zhan, X., Zhai, Y., Sun, Y. "Trustworthy Map Matching: Calibrated Posterior Confidence via GNSS-consistent Probabilistic Model." *IEEE Transactions on Intelligent Transportation Systems* (submitted, 2026).

## Key Features

- **GNSS-consistent emission model** — anisotropic Mahalanobis distance replaces isotropic Euclidean projection
- **HPL-adaptive candidate search** — protection level dynamically scales the search radius, reducing candidate count by up to 7.6×
- **Calibrated trustworthiness (TW)** — filtering posterior with proper probabilistic normalization (background state, row-normalized transitions, uniform prior)
- **96.9% segment accuracy** on real-vehicle data (16,155 epochs, Haikou, Hainan) vs. 88.1% for classical HMM
- **ECE = 0.069** (36% reduction over HMM baseline ECE 0.107)
- C++17 core with Python bindings via SWIG; Monte Carlo simulation framework included

## Quick Start

### Prerequisites

- C++17 compiler (GCC ≥ 9, Clang ≥ 10)
- CMake ≥ 3.5
- GDAL ≥ 2.2, Boost ≥ 1.56 (graph, geometry, serialization)
- OpenMP (optional, for parallel batch matching)
- SWIG (for Python bindings)
- Conda environment recommended

### Build

```bash
conda activate fmm_env   # or your conda environment with GDAL/Boost/SWIG

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Binaries are placed in `build/`:
| Binary | Purpose |
|--------|---------|
| `cmm` | TMM (Covariance Map Matching) CLI |
| `fmm` | FMM (Fast Map Matching) CLI |
| `ubodt_gen` | UBODT precomputation |
| `stmatch` | Spatio-temporal matching |

### Python API

```python
import sys; sys.path.insert(0, 'build/python')
from fmm import Network, NetworkGraph, UBODT
from fmm import CovarianceMapMatch, CovarianceMapMatchConfig

network = Network("input/map/hainan/edges.shp", "key", "u", "v")
graph = NetworkGraph(network)
ubodt = UBODT.read_ubodt_file("input/map/hainan_ubodt_indexed.bin")
config = CovarianceMapMatchConfig(k=16, protection_level_multiplier=10.0)
cmm = CovarianceMapMatch(network, graph, ubodt)
result = cmm.match_traj(trajectory, config)
```

## Project Structure

```
├── src/                    # C++17 source (core, network, mm/cmm, mm/fmm, io, app)
├── python/                 # SWIG bindings + paper experiment scripts
│   ├── fmm.i               # SWIG interface
│   └── experiments/        # Paper experiment pipeline (exp1–exp4, figures)
├── experiments/
│   ├── scripts/            # Experiment orchestration + figure generation
│   ├── config/             # Experiment configuration (JSON)
│   └── output/             # Experiment results (CSV, JSON)
├── input/
│   ├── config/             # CMM/FMM XML configuration templates
│   └── map/                # Road network shapefile + UBODT (Haikou, Hainan)
├── data/                   # Datasets (excluded from git — see below)
│   ├── real_vehicle/       # Haikou SPP GNSS trajectories (16,155 epochs, RTK GT)
│   └── simulation/         # Monte Carlo simulation data (σ = 1–30 m)
├── docs/                   # Manuscript + figures (excluded from git)
├── third_party/            # Vendored dependencies (h3, spdlog, cxxopts)
├── cmake/                  # CMake modules
├── example/                # Usage examples
├── docker/                 # Docker support
└── _archive_review/        # Archived legacy scripts & data (excluded from git)
```

## Reproducing Paper Experiments

### 1. Data Preparation

Real-vehicle and simulation datasets are in `data/` (**excluded from git** — contact authors for access).
- Real data: `data/real_vehicle/` — 7 SPP trajectories with RTK ground truth
- Simulation data: `data/simulation/sigma_*/` — Monte Carlo datasets per noise level

### 2. Run CMM/FMM Matching

```bash
# Real-vehicle CMM matching (example)
./build/cmm --config input/config/cmm_config.xml

# Simulation sweep (Exp 2–3)
python experiments/scripts/exp3_full_matching.py --skip-match --jobs 8
python experiments/scripts/exp5_degraded_conditions.py --skip-match --jobs 8
```

### 3. Compute Metrics & Generate Figures

```bash
# Reliability diagram
python python/experiments/exp1_reliability_diagram.py
python python/experiments/gen_figures.py

# Sigma sensitivity & degraded conditions
python experiments/scripts/regenerate_sigma_sweep.py
python experiments/scripts/regenerate_degraded.py

# Candidate count comparison
python experiments/scripts/fig_candidate_counts.py

# Per-trajectory TW visualization
python experiments/scripts/fig_traj_tw_individual.py
```

## Theoretical Framework

TMM integrates GNSS stochastic information at three levels:

1. **HPL-based candidate search** — adaptive radius $r_i = \mathrm{HPL}_i$ replaces fixed-radius search
2. **Covariance-based emission** — anisotropic multivariate Gaussian $p(z_i|x) \propto \exp(-\frac{1}{2}\mathbf{d}^\top\mathbf{\Sigma}_i^{-1}\mathbf{d})$ via Mahalanobis projection
3. **Trustworthiness as calibrated posterior** — filtering posterior $\mathrm{tw}_t = P(x_t=i^*\mid z_{1:t}) = \alpha_t(i^*)/\sum_j\alpha_t(j)$ with proper normalization

See [CLAUDE.md](CLAUDE.md) for detailed algorithmic documentation and [CMM_README.md](CMM_README.md) for the CMM C++ API.

## FMM vs TMM

| Aspect | FMM (Classical HMM) | TMM (This Work) |
|--------|---------------------|-----------------|
| Emission model | Isotropic $\mathcal{N}(0,\sigma^2 I)$ | Anisotropic Mahalanobis (covariance) |
| Candidate search | Fixed radius $r$ | HPL-adaptive $r_i = \mathrm{HPL}_i$ |
| Candidate projection | Orthogonal (Euclidean) | Mahalanobis (statistically optimal) |
| Trustworthiness | Raw Viterbi score | Filtering posterior (calibrated) |
| ECE | 0.107 | **0.069** (36% ↓) |
| Real accuracy | 88.1% | **96.9%** |

## Citation

```bibtex
@article{ning2026tmm,
  title={Trustworthy Map Matching: Calibrated Posterior Confidence via GNSS-consistent Probabilistic Model},
  author={Ning, Chenzhang and Yang, Rong and Zhan, Xingqun and Zhai, Yawei and Sun, Yulong},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026},
  note={submitted}
}
```

The original FMM algorithm is described in:

```bibtex
@article{Yang2018FastMM,
  title={Fast map matching, an algorithm integrating hidden Markov model with precomputation},
  author={Yang, Can and Gidofalvi, Gyozo},
  journal={International Journal of Geographical Information Science},
  volume={32}, number={3}, pages={547--570}, year={2018}
}
```

## License

MIT License. See [LICENSE.TXT](LICENSE.TXT).

## Contact

- **Chenzhang Ning** (Donkey.Ning) — Ph.D. student, School of Aeronautics and Astronautics, Shanghai Jiao Tong University
- Advisor: Prof. Xingqun Zhan
- Repository: [https://github.com/DonkeyN11/fmm_sjtugnc](https://github.com/DonkeyN11/fmm_sjtugnc)
