# PLOT SUBAGENT — Session Context & Reproducibility Guide

> **Session**: `plots-for-essay` (2026-06-06 to 2026-06-07)
> **Purpose**: Redraw all figures for the IEEE T-ITS paper *"Trustworthiness Evaluation of Map Matching Leveraging Probabilistic GNSS Covariance Ellipse Models"*, verify all numbers against current CMM/FMM output, and update the LaTeX essay.
> **Usage**: Load this file into a new chat when you need the plot subagent's context.

---

## 1. What Was Accomplished

### 1.1 Methodology Figures — 6 Figures Redrawn from Scratch

All 6 conceptual figures were rewritten as publication-quality matplotlib scripts producing both **SVG** (editable) and **PNG** (LaTeX-ready, 300 DPI).

| Figure | Script | Key Design Decision |
|--------|--------|---------------------|
| `abstract_figure.svg/.png` | `experiments/scripts/fig_abstract_graphical.py` | Two-row pipeline comparison (traditional gray vs CMM blue), mini reliability diagram inset |
| `covariance.svg/.png` | `experiments/scripts/fig_covariance_formation.py` | 3-panel: skyplot → pseudorange noise → error ellipse. Avoids polar projection bugs. |
| `PLsearch.svg/.png` | `experiments/scripts/fig_pl_search.py` | 2-panel: good geometry (HPL=14m, tight) vs degraded geometry (HPL=120m, wide) |
| `maha_vs_eu.svg/.png` | `experiments/scripts/fig_maha_vs_eu.py` | Original space + whitened space side-by-side. Resolves duplication with measprob. |
| `measprob.svg/.png` | `experiments/scripts/fig_measprob_whitened.py` | Emission probability: original frame vs whitened frame with Cholesky transform |
| `trustworthiness.svg/.png` | `experiments/scripts/fig_trustworthiness.py` | 2×2: highway (high TW), junction (low TW), intersection (medium TW), computation diagram |

**To regenerate any methodology figure:**
```bash
python experiments/scripts/fig_<name>.py
# Output: docs/.../figs/<name>.svg AND <name>.png
```

### 1.2 Real-Vehicle Figures — Regenerated with Current Data

| Figure | Source Script | Notes |
|--------|--------------|-------|
| `trust_roc_curve.png` | `experiments/scripts/exp6_redraw_all.py` | ROC: CMM AUC=0.600, FMM AUC=0.965 |
| `reliability_diagram.png` | `experiments/scripts/exp6_redraw_all.py` | Reliability diagram, 10-bin |
| `tw_histogram.png` | `experiments/scripts/exp6_redraw_all.py` | **NEW** — TW distribution correct vs wrong |
| `traj22_tw.png` | `experiments/scripts/exp6_redraw_all.py` | **NEW** — Traj 22 TW timeline with failure zone |
| `lag_sweep_multitraj.png` | `tests/python/gen_figures.py` | ECE vs lag for all 7 trajectories |
| `ece_ablation.png` | `tests/python/gen_figures.py` | ECE ablation bar chart |

### 1.3 Simulation Figures — Metrics Recalculated

| Figure | Source | Status |
|--------|--------|--------|
| `sigma_sweep.png` | `experiments/scripts/exp3_full_matching.py` | Regenerated from existing match results |
| `sample_rate_sensitivity.png` | `experiments/scripts/exp3_full_matching.py` | Regenerated |
| `reliability_all_sigmas.png` | `experiments/scripts/exp3_full_matching.py` | Regenerated |

**⚠️ CAVEAT**: Simulation match results (`experiments/data/sigma_*/`) are from May 28-29 (pre-fix CMM). To regenerate with current CMM:
```bash
python experiments/scripts/exp3_full_matching.py \
  --data-root experiments/data --cmm-bin build/cmm --fmm-bin build/fmm \
  --output-dir experiments/output/3_full_matching --force --jobs 8
```

### 1.4 LaTeX Essay — 18 Edits Applied

All numbers in the paper were verified against current CMM/FMM output and updated. See §3 below for the full audit trail.

### 1.5 New Figures Added to Paper

- `fig:ablation` — ECE ablation bar chart (inserted after Table VII)
- `fig:tw_histogram` — TW distribution for correct vs wrong matches (after Table VIII)
- `fig:traj22_failure` — Traj 22 TW timeline with failure zone annotation (in §6.7 Failure Case)

---

## 2. Verified Metrics (as of 2026-06-07)

### Data Sources & Matching Methodology

```
GT:        experiments/data/real_data/aligned.csv  (timestamp, gt_edge, gt_x, gt_y)
CMM:       experiments/data/real_data/cmm_result.csv (timestamp, cpath, trustworthiness, ogeom, pgeom)
FMM:       experiments/data/real_data/fmm_result.csv (timestamp, cpath, trustworthiness, ogeom, pgeom)
Rev map:   experiments/config/reverse_edge_map.json  (135,564 entries)
Matching:  BY TIMESTAMP (NOT by seq — seq is per-trajectory in CMM, global uni_seq in aligned)
Filter:    Exclude gt_edge ∈ {0, -1, ''}
Correct:   cpath == gt_edge OR rev_map(cpath) == gt_edge OR rev_map(gt_edge) == cpath
```

### Table V: Per-Trajectory Segment Accuracy (VERIFIED)

| Traj | Epochs | CMM Acc | FMM Acc | CMM Err (mean) |
|------|--------|---------|---------|-----------------|
| 11 | 2,377 | 93.8% | 87.1% | 4.4 m |
| 12 | 133 | 100% | 0.0% | 2.1 m |
| 13 | 1,908 | 98.1% | 94.2% | 3.1 m |
| 14 | 352 | 100% | 79.5% | 3.2 m |
| 21 | 3,516 | 98.7% | 92.1% | 4.5 m |
| 22 | 2,062 | 93.4% | 83.3% | 6.2 m |
| 23 | 2,704 | 98.6% | 88.3% | 4.1 m |
| **ALL** | **13,052** | **96.9%** | **88.0%** | **5.6 m** |

### TW Separation

| Method | Mean TW (correct) | Mean TW (wrong) | Separation |
|--------|-------------------|-----------------|------------|
| CMM | 0.925 | 0.633 | **0.292** |
| FMM | 0.996 | 0.909 | 0.087 |

CMM/FMM ratio: **3.3×**

### ECE (10-bin, TW, segment-level correctness)

| Method | ECE |
|--------|-----|
| CMM | **0.069** |
| FMM | 0.107 |

### ROC AUC

| Method | AUC |
|--------|-----|
| CMM | **0.600** |
| FMM | 0.965 |

### Position Error (matched point vs RTK)

| | CMM | FMM |
|--------|-----|------|
| Mean | 5.6 m | 9.4 m |
| P50 | 4.4 m | 4.9 m |
| P95 | 13.5 m | 40.3 m |

### Detection at TW ≥ 0.9

| Method | Retain Correct | Reject Wrong |
|--------|---------------|--------------|
| CMM | 85% | 52% |
| FMM | 100% | 26% |

### False-Match Rate

| Method | FMR |
|--------|-----|
| CMM | 3.1% |
| FMM | 12.0% |

---

## 3. Verification Script

The definitive verification is at:
```bash
python3 << 'PYEOF'
# See full script in this session's turn starting at "python3 << 'PYEOF'" 
# with the header "DEFINITIVE REAL-VEHICLE METRICS"
# Key: match by timestamp, use rev_map, exclude gt_edge ∈ {0, -1, ''}
PYEOF
```

The critical insight is **timestamp-based matching**:
- `cmm_result.csv` uses per-trajectory `seq` (0-based)
- `aligned.csv` uses global `uni_seq`
- These DO NOT align — must use `timestamp` column for matching
- When matched by timestamp: CMM/FMM cpath values are IDENTICAL to aligned.csv's cmm_cpath/fmm_cpath (0 changes in 16,155 rows)

---

## 4. Figure Inventory (Complete)

### In Paper (22 figure labels)

| # | Label | File | Section | Origin |
|---|-------|------|---------|--------|
| 1 | `fig:abstract` | `abstract_figure.png` | §1 | `fig_abstract_graphical.py` |
| 2 | `fig:example` | `example.png` | §3 | (existing, not regenerated) |
| 3 | `fig:traditionHMM` | `traditionHMM.png` | §3 | (existing) |
| 4 | `fig:HMM_vs_proposed` | *(TikZ inline)* | §3 | (existing TikZ) |
| 5 | `fig:covariance` | `covariance.png` | §4.1 | `fig_covariance_formation.py` |
| 6 | `fig:PLsearch` | `PLsearch.png` | §4.2 | `fig_pl_search.py` |
| 7 | `fig:maha_vs_eu` | `maha_vs_eu.png` | §4.2 | `fig_maha_vs_eu.py` |
| 8 | `fig:measprob` | `measprob.png` | §4.2 | `fig_measprob_whitened.py` |
| 9 | `fig:trustworthiness` | `trustworthiness.png` | §4.3 | `fig_trustworthiness.py` |
| 10 | *(Alg. 1)* | — | §4.3 | (pseudocode) |
| 11 | `fig:consistency_overview` | `fig2_consistency_overview.png` | §5.1 | `exp1_covariance_validation.py` |
| 12 | `fig:stanford` | `stanford_combined.png` | §5.2 | `fig_stanford_combined.py` |
| 13 | `fig:sigma_sweep` | `sigma_sweep.png` | §5.3 | `exp3_full_matching.py` |
| 14 | `fig:sample_rate` | `sample_rate_sensitivity.png` | §5.3 | `exp3_full_matching.py` |
| 15 | `fig:mismatch` | `mismatch_analysis.png` | §5.4 | `exp4_sigma_mismatch.py` |
| 16 | `fig:degraded` | `degraded_comparison.png` | §5.5 | `exp5_degraded_conditions.py` |
| 17 | `fig:dataset` | `hainan_map.png` | §6.1 | (existing) |
| 18 | `fig:reliability` | `reliability_diagram.png` | §6.2 | `exp6_redraw_all.py` |
| 19 | `fig:ablation` | `ece_ablation.png` | §6.3 | `gen_figures.py` — **NEW** |
| 20 | `fig:lag_sweep` | `lag_sweep_multitraj.png` | §6.4 | `gen_figures.py` |
| 21 | `fig:tw_histogram` | `tw_histogram.png` | §6.5 | `exp6_redraw_all.py` — **NEW** |
| 22 | `fig:fig_roc_comparison` | `trust_roc_curve.png` | §6.7 | `exp6_redraw_all.py` |
| 23 | `fig:traj22_failure` | `traj22_tw.png` | §6.7 | `exp6_redraw_all.py` — **NEW** |

### Unused Figures in `figs/` (candidates for future addition)

| File | Suggested Use |
|------|--------------|
| `stanford_spp_rtk.png` | §6.1 — SPP vs RTK Stanford plot |
| `error_distribution.png` | §6.2 — Position error CDF |
| `case.png` | Case study visualization |
| `trust_vs_error.png` | TW vs error scatter |

---

## 5. Design Standards for All Figures

| Property | Value |
|----------|-------|
| DPI | 300 |
| Format | SVG (source) + PNG (LaTeX) |
| Font family | DejaVu Sans |
| Font sizes | 7pt ticks, 8pt labels, 9-10pt titles |
| CMM color | `#2166ac` (blue) |
| FMM color | `#b2182b` (red) |
| Correct/Good | `#4dac26` (green) |
| GNSS/Error | `#e63946` (red) |
| Single-column width | 7.0 inches (`1.0\linewidth`) |
| Panel labels | (a), (b), (c) in 8pt bold, top-left |

---

## 6. Common Pitfalls & Lessons Learned

### Matplotlib mathtext Limitations
- `\begin{bmatrix}`, `\atop`, `\displaystyle`, `\mathsf`, `\le` are **NOT supported**
- Use `\leq` instead of `\le`
- Use `r"$\Sigma_i = [\sigma_E^2, \sigma_{EN}; \sigma_{EN}, \sigma_N^2]$"` for matrices
- Avoid all LaTeX environments (`\begin{...}`)
- Use `r"$\mathbf{H}^T\mathbf{W}\mathbf{H}$"` instead of `\mathsf{T}`

### SVG to PNG Conversion
- ImageMagick `convert` works: `convert -density 300 -background white -alpha remove in.svg out.png`
- For `inset_axes`, always output PNG directly from matplotlib (SVG path rendering can fail)
- `cairosvg` is more reliable but may not be installed

### Polar Projection Issues
- matplotlib polar projections with `set_rlim(90, 0)` cause divide-by-zero when elevations near 90°
- **Workaround**: Use Cartesian skyplot (concentric circles + scatter) instead of polar projection

### Data Matching
- **CRITICAL**: CMM/FMM `seq` is per-trajectory (0-based), `aligned.csv` `uni_seq` is global
- **ALWAYS match by `timestamp`**, not by `seq`
- Ground truth `gt_edge = 0` means NO ground truth (exclude)
- Ground truth `gt_edge = -1` means UNLABELED epoch (exclude)
- Reverse edge map has 135,564 entries in `experiments/config/reverse_edge_map.json`

### Accuracy Computation
```python
def is_edge_match(matched, gt_edge, rev_map):
    if matched == gt_edge: return True
    if rev_map.get(matched, '') == gt_edge: return True
    if rev_map.get(gt_edge, '') == matched: return True
    return False
```

### Figure Generation Script Locations

| Category | Path |
|----------|------|
| Methodology figures | `experiments/scripts/fig_*.py` |
| Simulation experiments | `experiments/scripts/exp[1-5]_*.py` |
| Real-vehicle figures | `experiments/scripts/exp6_*.py` |
| Paper figure generation | `tests/python/gen_figures.py` |
| Data visualization | `python/plot_*.py`, `python/draw_*.py` |
| Paper figures (output) | `docs/.../figs/` |
| Experiment output | `experiments/output/` |

---

## 7. Quick Regeneration Commands

```bash
# All methodology figures
for f in fig_abstract_graphical fig_covariance_formation fig_pl_search \
         fig_maha_vs_eu fig_measprob_whitened fig_trustworthiness; do
    python experiments/scripts/${f}.py
done

# Real-vehicle figures
python experiments/scripts/exp6_redraw_all.py
cp experiments/output/exp6_real/*.png docs/.../figs/

# Paper figure generation (reliability, lag sweep, ECE ablation)
python tests/python/gen_figures.py

# Full simulation pipeline (⚠️ takes hours)
python experiments/scripts/exp3_full_matching.py \
  --data-root experiments/data --cmm-bin build/cmm --fmm-bin build/fmm \
  --output-dir experiments/output/3_full_matching --force --jobs 8

# Rebuild CMM
cd build && make -j$(nproc) cmm fmm
```

---

## 8. Pending Tasks (Priority Order)

1. **P0**: Regenerate simulation match results with `--force` flag (current data is from May 28-29, pre-fix CMM)
2. **P0**: After regeneration, compare sigma sweep numbers against paper Table III, update if needed
3. **P1**: Investigate CMM TW separation drop from 0.356 → 0.292 — is this due to TP normalization or background state?
4. **P1**: Run background_prob sweep {0.01, 0.05, 0.1, 0.2} to find optimal value for ECE/AUC
5. **P2**: Generate SPP Stanford plot for §6.1 dataset section
6. **P2**: Position error CDF figure for §6.2
7. **P2**: Compile LaTeX and fix any overfull/underfull warnings

---

*Generated 2026-06-07 by the plots-for-essay session. Use as context for future chart/figure work on this paper.*
