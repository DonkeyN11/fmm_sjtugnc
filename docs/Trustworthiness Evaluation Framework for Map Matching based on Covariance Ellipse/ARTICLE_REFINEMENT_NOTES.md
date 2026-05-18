# CMM Article Refinement Notes

> **Article**: `Trustworthiness Evaluation Framework.tex`
> **Target**: IEEE Transactions on Intelligent Transportation Systems (T-ITS)
> **Author**: Chenzhang Ning, Shanghai Jiao Tong University
> **Last refined**: 2025-06

---

## 1. Article Structure (Post-Refinement)

| Section | Content | Key Changes |
|---------|---------|-------------|
| Abstract | Anisotropic emission + fixed-lag smoothing + RAIM PL; ECE 0.978→0.121; AUC 0.720 | Replaced "300K trips" with "7 trajectories, 16,155 epochs" |
| §I Introduction | Problem statement, 3 contributions | Contributions rewritten to match actual code |
| **§II Related Work** | HMM Map Matching, GNSS Integrity, Probabilistic Calibration | **New section** |
| §III Classical HMM | Problem formulation, emission, transition, Viterbi, limitations | Kept from original |
| §IV Proposed Method | WLS covariance, HPL, Mahalanobis emission, fixed-lag smoothing | **Trustworthiness derivation rewritten** (was sliding-window sum; now softmax posterior) |
| §V Simulation | MC setup, probabilistic validation, accuracy enhancement | Commented content cleaned; kept core results |
| §VI Real Vehicle Experiment | Dataset, calibration, ablation, lag sweep, ROC | **Completely rewritten** with Hainan-06 data |
| §VII Conclusion | 3 contributions, ablation results, future work | Rewritten with actual numbers |
| References | 11 entries | Expanded from 2 |

---

## 2. Trustworthiness Derivation (Post-Refinement)

### Old Formula (REMOVED)
```
T = sum_{i=n}^{n+l} [ log(p_tilde(z_i | x_i^*)) + log(p(x_i^* | x_{i-1}^*)) ]
```
This was a sliding-window sum of log-likelihoods — does NOT match code.

### New Formula (CORRECT — matches `apply_lag_smoothing()`)
```
Part 1 — Forward variable (filtering):
  log α_t(b) = log Σ_a exp(log α_{t-1}(a) + log t_{a→b}) + log p(z_t | s_t^(b))

Part 2 — Viterbi future propagation:
  β_{t+L}(i) = max_{path t+1..t+L} Σ [log t + log ep]

Part 3 — Smoothing posterior (softmax):
  score_t(i) = log α_t(i) + β_{t+L}(i)
  tw_t(i) = exp(score_t(i)) / Σ_j exp(score_t(j))

Part 4 — Entropy:
  H_t = -Σ p_i log₂(p_i)
  ΔH_t = H_prior - H_t
```
Code location: `src/mm/cmm/cmm_algorithm.cpp:2011-2114`

---

## 3. Key Quantitative Results

### Experiment Data
| Dataset | Hainan-06, Haikou, China |
|---------|--------------------------|
| Receiver | Tersus BX50C (GPS+BDS+GLO+GAL+QZS) |
| Trajectories | 7 (traj11-14, traj21-23) |
| Total epochs | 16,155 |
| Ground truth | RTK post-processing (cm-level) |
| RAIM HPL (median) | 22.8 m |

### Calibration Results (ECE at 5m threshold)
| Metric | CMM | FMM |
|--------|-----|-----|
| ECE (3m) | 0.122 | 0.989 |
| ECE (5m) | 0.121 | 0.978 |
| ECE (10m) | 0.099 | 0.942 |
| MCE (5m) | 0.576 | 0.989 |
| Brier (5m) | 0.165 | 0.420 |

### Ablation Study (ECE at 5m)
| Configuration | ECE(TW) | Delta ECE |
|---|---|---|
| FMM (isotropic) | 0.978 | -- |
| + CMM anisotropic, L=0 | 0.121 | -0.857 |
| + Fixed-lag (L=20) | 0.261 | +0.140 |
| + PHMI integrity | 0.261 | +0.000 |

**Key finding**: Anisotropic Mahalanobis emission contributes 87.6% of the calibration improvement. Fixed-lag smoothing degrades calibration with RAIM PL.

### Lag Sweep
- Cross-trajectory optimal lag = 0 (smoothing does NOT improve calibration with RAIM PL)
- Per-trajectory median error: 3.0m (traj11) to 11.9m (traj14)
- Per-trajectory %correct ≤5m: 9.9% (traj12) to 75.2% (traj23)

### ROC
- AUC = 0.720 (trustworthiness as mismatch detector at 10m)
- High-trust epochs (tw > 0.95): only 2.3% mismatches

---

## 4. Figure & Table Inventory

### Figures (14 total)
| # | File | Content | Section |
|---|------|---------|---------|
| 1 | `abstract figure.png` | Traditional vs proposed comparison | §I |
| 2 | `example.png` | Map matching problem example | §III |
| 3 | `traditionHMM.png` | Traditional HMM process | §III |
| 4 | `covariance.png` | GNSS covariance calculation | §IV-B |
| 5 | `PL.png` | Protection level illustration | §IV-B |
| 6 | `PLsearch.png` | HPL-based candidate search | §IV-C |
| 7 | `maha_vs_eu.png` | Euclidean vs Mahalanobis projection | §IV-C |
| 8 | `measprob.png` | Emission probability illustration | §IV-C |
| 9 | `trustworthiness.png` | Trustworthiness in road segments | §IV-D |
| 10 | `reliability_diagram.png` | CMM vs FMM calibration (NEW) | §VI-C |
| 11 | `ece_ablation.png` | ECE ablation bar chart (NEW) | §VI-D |
| 12 | `lag_sweep_multitraj.png` | Lag sweep for 7 trajectories (NEW) | §VI-E |
| 13 | `trust_roc_curve.png` | ROC curve (AUC=0.720) | §VI-F |
| 14 | `hainan_map.png` | Dataset overview | §VI-A |

### Tables (3 total)
| Table | Content | Section |
|-------|---------|---------|
| `tab:dataset` | 7 trajectories: epochs, HPL, error | §VI-A |
| `tab:calibration` | ECE/MCE/Brier: CMM vs FMM | §VI-C |
| `tab:ablation` | ECE decomposition: FMM→CMM→+lag→+PHMI | §VI-D |
| `tab:mc_setup` | Monte Carlo simulation configuration | §V-A |

---

## 5. LaTeX Compilation

```bash
# Install dependencies
sudo apt install texlive-fonts-recommended

# Compile
cd "docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/"
pdflatex "Trustworthiness Evaluation Framework.tex"
bibtex "Trustworthiness Evaluation Framework"
pdflatex "Trustworthiness Evaluation Framework.tex"
pdflatex "Trustworthiness Evaluation Framework.tex"
```

**Note**: `IEEEtran.cls` is already in the article directory (bundled with the project).

---

## 6. Code-to-Article Mapping

| Article Concept | Code Location |
|----------------|---------------|
| Anisotropic emission (Mahalanobis) | `cmm_algorithm.cpp:776-815` (`calculate_emission_log_prob`) |
| PHMI emission normalization | `cmm_algorithm.cpp:1090-1156` |
| Candidate search with PL | `cmm_algorithm.cpp:819-1008` (`search_candidates_with_protection_level`) |
| Forward recursion (cumu_prob) | `cmm_algorithm.cpp:1742-1807` (`initialize_first_layer`) + `1898-1917` (`update_layer_cmm`) |
| Fixed-lag smoothing (trustworthiness) | `cmm_algorithm.cpp:2011-2114` (`apply_lag_smoothing`) |
| Buffer management | `cmm_algorithm.cpp:1620-1652` (main loop) + `2117-2124` (`flush_lag_buffer`) |
| Filtering posterior entropy | `cmm_algorithm.cpp:1920-1968` (update_layer_cmm entropy section) |
| H0 Bayesian test (h0_lambda) | `cmm_algorithm.cpp:1626-1640` |
| nbest_trustworthiness | Extracted from layer top-3 softmax values (post-refactoring) |
| Gap bridging | `cmm_algorithm.cpp:1488-1525` |
| Viterbi backtrack | `TransitionGraph::backtrack()` (transition_graph impl) |
| Config: lag_steps | `cmm_config_omp.xml:69` |
| Config: phmi | `cmm_config_omp.xml:68` |
| RAIM PL computation | `tests/python/compute_raim_pl.py` |
| PL merge into CMM input | `tests/python/merge_raim_pl.py` |

### Removed Code Paths
- `compute_window_trustworthiness()` — removed (Path B sliding-window top-K)
- `push_top_k()` — removed (helper for above)
- `window_length` config parameter — removed
- `margin_used_trustworthiness` config parameter — removed

---

## 7. Removed/Replaced Content

| Original | Replaced With |
|----------|---------------|
| "30万车次 / 300K+ vehicle trips" | "7 trajectories, 16,155 epochs" |
| "David 30 GNSS receiver" | "Tersus BX50C" |
| MPF + Medark baselines | FMM baseline only |
| "60% higher availability" | "ECE reduced from 0.978 to 0.121" |
| "50% higher accuracy" | Ablation: anisotropic Mahalanobis = 87.6% improvement |
| "274.75 km, 286 min" | Trajectory table with per-trajectory stats |
| Sliding window T statistic | Fixed-lag softmax posterior |
| MPF/Medark comparison paragraphs | Calibration analysis + ablation + lag sweep |
| "Michael Shell" biography | Removed for submission |
| IEEE template placeholders | Author: Chenzhang Ning, SJTU |

---

## 8. References (11 entries in `references.bib`)

| Key | Citation |
|-----|----------|
| `HMM` | Newson & Krumm, ACM SIGSPATIAL 2009 (seminal HMM map matching) |
| `FMM` | Yang & Gidófalvi, IJGIS 2018 (UBODT precomputation) |
| `Guo2017` | Guo et al., ICML 2017 (ECE, calibration of neural networks) |
| `Woltche2023` | Wöltche, Trans. in GIS 2023 (MDP map matching benchmark) |
| `Wang2024` | Wang et al., Trans. in GIS 2024 (low-frequency trajectory MM) |
| `Li2023` | Li et al., 2023 (improved HMM map matching) |
| `Zhang2018` | Zhang et al., Math. Prob. Eng. 2018 (RAIM M-estimation) |
| `Zhang2019` | Zhang et al., Math. Prob. Eng. 2019 (ARAIM MHSS thresholds) |
| `Kim2019` | Kim et al., J. Sensors 2019 (PL for vehicle localization) |
| `Park2024` | Park & Chung, J. Field Robotics 2024 (uncertainty-aware LiDAR) |
| `Phillips2016` | Phillips et al., J. Field Robotics 2016 (Bayesian verification) |

### Local Literature PDFs (in `docs/literatures/`)
25 PDFs available for additional citations. Key ones already used or partially used:
- `li-2023-an-improved-hidden-markov-model...pdf` → cited as Li2023
- `MPF_A_Multi-Noise_Perception_Framework...pdf` → MPF (mentioned in original, removed)
- `Medark...pdf` → Medark (mentioned in original, removed)
- `Localization_Integrity_for_Intelligent_Vehicles...pdf` → GNSS integrity
- `bayesian-fault-tolerant-position-estimator...pdf` → Bayesian GNSS

---

## 9. Experiment Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `tests/python/exp1_reliability_diagram.py` | ECE/MCE/Brier/LogLoss CMM vs FMM | `exp1_reliability.json` |
| `tests/python/exp1_lag_sweep.py` | Single-traj lag sweep (traj11) | `mr/lag_sweep/*.csv` |
| `tests/python/exp1_multitraj_lag_sweep.py` | Multi-traj lag sweep (7 trajs) | `mr/multitraj_sweep/*.csv` |
| `tests/python/exp2_synthetic_validation.py` | Synthetic data validation | `mr/synthetic/*.csv` |
| `tests/python/exp3_phmi_analysis.py` | PHMI=1e-5 lag sweep | `mr/phmi_analysis/*.csv` |
| `tests/python/exp4_ablation_ece.py` | ECE ablation (100-bin) | `mr/ablation/*.csv` |
| `tests/python/compute_raim_pl.py` | RAIM PL from RINEX observations | `raim_pl_X_X.csv` |
| `tests/python/merge_raim_pl.py` | Merge RAIM PL into CMM input | Updates `cmm_input_points.csv` |
| `tests/python/gen_figures.py` | Generate article figures | `figs/*.png` |
| `tests/python/evaluate_match_metrics.py` | ROC-AUC per metric | Console output |

---

## 10. Remaining TODOs for Future Iterations

1. **Install `texlive-fonts-recommended`** and compile the LaTeX to verify clean output
2. **Regenerate legacy figures** (`figs/error_distribution.png`, `figs/error_analysis.png`, `figs/trust_roc_curve.png`, `figs/trust_vs_error.png`) with RAIM PL data
3. **Add more citations** from the 25 local PDFs in `docs/literatures/`
4. **Clean commented-out content** in Section V (~400 lines of old simulation experiment descriptions)
5. **Add acknowledgments** (funding, institutions)
6. **Consider adding a delta-entropy figure** (complementary metric to trustworthiness)
7. **Verify the `\ref{}` cross-references** after final LaTeX compilation

---

## 11. Key Design Decisions Made

1. **nbest_trustworthiness**: Now sourced from lag-smoothing softmax top-3 (Path A), NOT from a separate sliding-window computation (Path B — removed)
2. **window_length** and **margin_used_trustworthiness**: Removed entirely from config and code
3. **trustworthiness_threshold**: Fixed from inverted logic (`trust <= threshold` → `trust >= threshold`), threshold changed from entropy domain (10.0 bits) to probability domain (0.5)
4. **enable_gap_bridging**: On by default (even though not in XML config; defaults to `true`)
5. **filtered**: Currently `false` in config (no epoch filtering — all points pass)
6. **lag_steps**: Set to 0 in current config (filtering-only mode) — this is optimal for RAIM PL
7. **RAIM PL**: Replaced random PL values with residual-based RAIM computation from raw RINEX 3.04 observations (median HPL = 22.8m)

---

## 12. Quick Re-run Commands

```bash
# Re-run all experiments with RAIM PL
cd /home/dell/fmm_sjtugnc

# 1. Compute RAIM PL for all trajectories
python3 tests/python/compute_raim_pl.py --all

# 2. Merge PL into CMM input
python3 tests/python/merge_raim_pl.py

# 3. Run experiments
python3 tests/python/exp1_multitraj_lag_sweep.py
python3 tests/python/exp4_ablation_ece.py
python3 tests/python/exp1_reliability_diagram.py

# 4. Generate figures
python3 tests/python/gen_figures.py

# 5. Build CMM
cd build && cmake .. && make -j$(nproc)

# 6. Test CMM
cd .. && ./build/cmm input/config/cmm_config_omp.xml
```
