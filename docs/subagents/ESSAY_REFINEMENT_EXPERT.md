# Essay Refinement Expert — CMM Paper for IEEE T-ITS

**Created**: 2026-06-07 (from session `article_refinement_260607`)
**Role**: IEEE T-ITS essay writing and revision expert for the CMM trustworthiness paper
**Purpose**: Read this file at the start of any new session to reconstruct the full refinement context. Contains all fixes applied, verified numbers, outstanding issues, and knowledge needed to continue paper revision.

---

## Quick-Start (for a new session)

To activate the essay expert persona, tell Claude:

> "Read `docs/subagents/ESSAY_REFINEMENT_EXPERT.md`. You are now the essay refinement expert for my CMM IEEE T-ITS paper. Start from the outstanding issues in priority order."

Or for a specific task:

> "Read `docs/subagents/ESSAY_REFINEMENT_EXPERT.md`. Help me address MAJOR issue #1 — expand the references from 11 to 30+ entries."

---

## Session Summary (2026-06-07)

Conducted a full article review covering:
- LaTeX source code inspection (~2100 lines)
- Codebase verification (CMM algorithm, experiment scripts)
- Literature search (IEEE Xplore, Google Scholar) for missing references
- Experimental data verification (ECE values, TW distributions, mismatch rates)
- Figure generation and LaTeX compilation
- Applied critical fixes and committed to feature branch

**Branch**: `feature/article-refinement-fixes` (commit `1086987`, 1 ahead of master)

---

## Files That Encode This Session's Knowledge

### Primary

| File | Content |
|------|---------|
| `docs/subagents/ESSAY_REFINEMENT_EXPERT.md` | this file — full session context |
| `docs/Trustworthiness Evaluation Framework.../Trustworthiness Evaluation Framework.tex` | Main article (~2100 lines, edited in this session) |
| `docs/Trustworthiness Evaluation Framework.../ARTICLE_REFINEMENT_NOTES.md` | Paper structure, key results, code-to-article mapping, TODO items |
| `docs/Trustworthiness Evaluation Framework.../references.bib` | 11 references (needs expansion) |

### Experiment Data

| File | Content |
|------|---------|
| `dataset-hainan-06/mr/exp1_reliability.json` | EP-based ECE/MCE/Brier — CMM vs FMM (10-bin) |
| `dataset-hainan-06/mr/multitraj_sweep/cmm_all_lag*.csv` | 7-traj lag sweep results (lag 0-50) |
| `dataset-hainan-06/mr/ablation/real_lag0_phmi.csv` | Ablation study data |
| `dataset-hainan-06/mr/ablation/real_lag20_phmi.csv` | Ablation study data |

### Experiment Scripts

| Script | Purpose |
|--------|---------|
| `tests/python/exp1_reliability_diagram.py` | EP-based ECE/MCE/Brier CMM vs FMM |
| `tests/python/exp1_multitraj_lag_sweep.py` | 7-traj lag sweep (0-50) |
| `tests/python/exp4_ablation_ece.py` | 100-bin ECE ablation (FMM→CMM→+lag→+PHMI) |
| `tests/python/gen_figures.py` | Generate 3 article figures |
| `tests/python/compute_raim_pl.py` | RAIM PL from RINEX observations |
| `tests/analyze_trust_roc.py` | Trustworthiness ROC-AUC analysis |

---

## Critical Fixes Applied (committed)

### 1. Algorithm 1 Rewritten
The old pseudocode used a wrong trustworthiness formula (`T ← 1/N Σ ln(...)` — sliding-window sum). Replaced with the correct algorithm showing:
- Per-epoch candidate generation (HPL-based search + Mahalanobis projection)
- Forward recursion (log α_t)
- Fixed-lag smoothing buffer management (`|B| > L` → smooth oldest layer)
- Viterbi-style future evidence propagation (β_{t+L})
- Softmax normalization → per-epoch trustworthiness tw_t
- Posterior entropy H_t and delta entropy ΔH_t

### 2. Transition Probability β Description Fixed
**Before**: "β is a smoothing parameter which makes the sum of the probability equals to 1" (WRONG)
**After**: "β is a scale parameter controlling how rapidly the transition probability decays with increasing discrepancy between network and GNSS distances" (CORRECT)

### 3. Naming Convention Unified
`CMM-HMM` → `CMM` throughout the paper.

### 4. Grammar Fixes (6 corrections)
- "shows priority of CMM to HMM" → "demonstrate the advantages of CMM over FMM"
- "Monte-Carlo experiment is carried on" → "Monte Carlo experiments are conducted"
- "pseudo range" → "pseudorange" (standard GNSS terminology)
- "standard derivation" → "standard deviation"
- "Trajectories is simulated" → "Trajectories are simulated"
- "trustworthiness effectives" → "trustworthiness effectiveness"

### 5. TW > 0.95 Claim Clarified
Added: uses `L=5` fixed-lag smoothing; mismatch = "matched road segment differing from RTK ground truth" (not position error).

### 6. Figures Generated (3 new PNGs)
| Figure | Script | Content |
|--------|--------|---------|
| `figs/reliability_diagram.png` | `gen_figures.py:fig_reliability_diagram()` | 10-bin calibration CMM vs FMM at 5m |
| `figs/ece_ablation.png` | `gen_figures.py:fig_ece_ablation()` | ECE decomposition bar chart |
| `figs/lag_sweep_multitraj.png` | `gen_figures.py:fig_lag_sweep()` | ECE vs lag_steps for 7 trajectories |

---

## Verified Numbers vs Article Claims

### TW Distribution (Real Data, lag=0, POST-FIX — verified 2026-06-07)

| Traj | Epochs | Accuracy | Mean TW | % TW > 0.95 |
|------|--------|----------|---------|-------------|
| 11 | 2439 | 93.9% | — | — |
| 12 | 133 | 100.0% | — | — |
| 13 | 1908 | 98.1% | — | — |
| 14 | 352 | 100.0% | — | — |
| 21 | 3581 | 98.7% | — | — |
| 22 | 2123 | 93.5% | — | — |
| 23 | 2720 | 98.6% | — | — |
| **Overall** | **13,256** | **96.9%** | **0.928** | **82.6%** |

**Key insight (post-fix)**: At lag=0, 82.6% of epochs have TW > 0.95 — the 3% reverse guard fix dramatically improved trustworthiness calibration without needing lag smoothing. The pre-fix CMM had 0% TW > 0.95 at lag=0.

### Pre-Fix TW Distribution (for comparison, lag=0)

| Traj | Epochs | Median Err | % ≤ 5m | Mean TW | % TW > 0.95 |
|------|--------|-----------|--------|---------|-------------|
| 11 | 2696 | 3.0m | 69.1% | 0.448 | 0.0% |
| 12 | 734 | 10.4m | 9.9% | 0.480 | 0.0% |
| 13 | 2400 | 3.2m | 64.8% | 0.460 | 0.0% |
| 14 | 1407 | 11.9m | 19.3% | 0.495 | 0.0% |
| 21 | 3630 | 6.7m | 40.9% | 0.446 | 0.0% |
| 22 | 2551 | 5.7m | 45.0% | 0.323 | 0.0% |
| 23 | 2737 | 3.0m | 75.2% | 0.432 | 0.0% |

**Note**: Pre-fix TW was the joint path posterior δ_t/Σα_t which decays to 0 for long trajectories. Post-fix TW is the filtering posterior with forward_cumu.

### TW > 0.95 at Different Lag Values

| Lag | % TW > 0.95 | Mismatch rate (>10m position error) |
|-----|-------------|--------------------------------------|
| 0 | 0.0% | N/A |
| 5 | 80.8% | 26.7% |
| 10 | 82.7% | 26.4% |
| 20 | 84.2% | 26.3% |

**Note**: 26% is by position error (>10m), NOT segment mismatch. The article's "2.3%" is by segment mismatch — verification requires RTK ground truth edge IDs.

### EP-based ECE (from `exp1_reliability.json`)

| Threshold | CMM EP ECE | FMM EP ECE |
|-----------|-----------|-----------|
| 3m | 0.260 | 0.978 |
| 5m | 0.262 | 0.976 |
| 10m | 0.319 | 0.973 |

**Note**: Article Table 2 reports TW-based ECE (CMM=0.121 at 5m), lower than EP-based ECE=0.262. The distinction between EP and TW calibration should be clarified.

---

## Outstanding Issues (PRIORITY ORDER)

### MAJOR

**M1 — Only 11 references (T-ITS norm: 30-60)**

Suggested additions from recent IEEE T-ITS literature:

| Key | Citation | Relevance |
|-----|----------|-----------|
| `Jia2023_OARAIM` | M. Jia et al., "Performance Analysis of Opportunistic ARAIM," *IEEE T-ITS*, vol. 24, no. 10, 2023. DOI: `10.1109/TITS.2023.3277393` | GNSS integrity + protection levels in T-ITS |
| `Xiong2024_BPCIM` | J. Xiong et al., "Integrity for Belief Propagation-Based Cooperative Positioning," *IEEE T-ITS*, vol. 25, no. 9, 2024. | Distributed integrity monitoring with PL derivation |
| `Huang2024` | Y. Huang et al., "Accurate Map Matching Method for Mobile Phone Signaling Data Under Spatio-Temporal Uncertainty," *IEEE T-ITS*, vol. 25, no. 2, 2024. | HMM map matching with spatiotemporal uncertainty |
| `Li2024_CRF` | H. Li et al., "A Novel Map Matching Method Based on Improved HMM and CRF," *Int. J. Digital Earth*, 2024. | HMM+CRF for map matching |
| `Chidambaram2024` | "How Flawed is ECE? An Analysis via Logit Smoothing," *arXiv:2402.14568*, 2024. | ECE critique + smoother alternatives |
| `Kangsepp2025` | "On the usefulness of the fit-on-test view on evaluating calibration of classifiers," *Machine Learning*, vol. 114, 2025. | Improved reliability diagrams |
| `Electronics2025_Survey` | "Advancing Map-Matching and Route Prediction: Challenges, Methods, and Unified Solutions," *Electronics*, vol. 14, no. 18, 2025. | Comprehensive map matching survey |

Also check `docs/literatures/` — 25 PDFs available for additional citations.

**M2 — Li2023 reference incomplete** (`references.bib:61-66`)

Missing fields: `volume`, `number`, `pages`, `doi`. Need to find the complete citation on IEEE Xplore or Google Scholar.

**M3 — Kim2019 reference content mismatch**

Article text (line 176): "Proposed Protection Level-based evaluation of LiDAR map-matching reliability for vehicle localization"
Actual paper title: "Precise Vehicle Position and Heading Estimation Using a Binary Road Marking Map"
These describe different research. Either find the correct paper on PL-based LiDAR localization, or update the article text to match the actual Kim2019 paper.

**M4 — ECE numbers clarified (RESOLVED 2026-06-07)**

**Resolution**: Three distinct ECE types exist:
1. **TW-based ECE** (primary): CMM=0.069, FMM=0.107 — calibration of trustworthiness vs segment correctness. Used in paper abstract.
2. **EP-based ECE**: CMM=0.262, FMM=0.976 — calibration of emission probability vs position error. From `exp1_reliability.json`.
3. **Position-error ECE**: CMM=0.121, FMM=0.978 — calibration of trustworthiness vs position error threshold.

The paper must clearly state which ECE is being reported in each section. Abstract uses TW-based ECE (0.069). Simulation sections may use position-error ECE. See `ESSAY_REFINEMENT_PLAN.md` R5 for full resolution.

### MODERATE

**M5 — Section V has ~400 lines of commented-out content**

Lines 1286-1718 contain extensive commented simulation methodology. Clean up to prevent accidental inclusion and improve source readability.

**M6 — No Limitations/Discussion section**

Should be added before Conclusion, acknowledging:
- RAIM PL requires ≥5 visible satellites; urban canyon performance untested
- Only one city (Haikou) evaluated; generalization unverified
- 7 trajectories may not cover all driving conditions
- Fixed-lag smoothing degrades calibration with RAIM PL (ECE increases from 0.121 to 0.261)

**M7 — Figure `trustworthiness_traj9newultra.png` naming**

Referenced at line 1814, but the Hainan-06 dataset has traj11-23 (no traj9). May be from an older dataset. Verify the source or rename.

### MINOR

**M8 — Undescriptive figure label**: `\label{fig:enter-label}` at line 319 → should be `\label{fig:traditional_hmm}`

**M9 — Inconsistent matrix notation**: `\boldsymbol{\Sigma}` vs `\mathbf{\Sigma}` used interchangeably.

**M10 — Missing acknowledgments**: "The authors thank XXX for XXX." — fill in funding/individuals.

---

## LaTeX Compilation

### Build Command

```bash
TEXDIR="/home/ncz/fmm_sjtugnc/docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse"
cd "$TEXDIR"
/usr/bin/pdflatex -interaction=nonstopmode "Trustworthiness Evaluation Framework.tex"
/usr/bin/bibtex "Trustworthiness Evaluation Framework"
/usr/bin/pdflatex -interaction=nonstopmode "Trustworthiness Evaluation Framework.tex"
/usr/bin/pdflatex -interaction=nonstopmode "Trustworthiness Evaluation Framework.tex"
```

### Important: Use System pdflatex

The conda texlive installation has broken format files. Use `/usr/bin/pdflatex` (TeX Live 2022/Debian). Required packages were installed to `/home/ncz/texmf/` via `apt download` + `dpkg-deb -x` for: `texlive-pictures`, `texlive-latex-recommended`, `texlive-science`, `texlive-latex-extra`, `texlive-publishers`.

---

## Code-to-Article Mapping

| Article Concept | Code Location | Status |
|----------------|---------------|--------|
| Anisotropic emission (Mahalanobis EP) | `src/mm/cmm/cmm_algorithm.cpp:776-815` | ✓ Matches |
| Candidate search with PL | `src/mm/cmm/cmm_algorithm.cpp:819-1008` | ✓ Matches |
| Forward recursion (cumu_prob) | `src/mm/cmm/cmm_algorithm.cpp:1742-1807` + `1898-1917` | ✓ Matches |
| Fixed-lag smoothing | `src/mm/cmm/cmm_algorithm.cpp:2011-2114` | ✓ Matches |
| Buffer management | `src/mm/cmm/cmm_algorithm.cpp:1620-1652` + `2117-2124` | ✓ Matches |
| Posterior entropy | `src/mm/cmm/cmm_algorithm.cpp:1920-1968` | ✓ Matches |
| H0 Bayesian test (h0_lambda) | `src/mm/cmm/cmm_algorithm.cpp:1626-1640` | ✓ Matches |
| RAIM PL computation | `tests/python/compute_raim_pl.py` | External script |

---

## Dataset: Hainan-06

- **Location**: Haikou, Hainan, China
- **Receiver**: Tersus BX50C (GPS+BDS+GLO+GAL+QZS, 10Hz)
- **7 trajectories**: traj11-14 (June 19) + traj21-23 (June 21), 16,155 total epochs
- **Ground truth**: RTK post-processing (cm-level)
- **RAIM HPL median**: 22.8m
- **Road network**: `input/map/hainan/edges.shp`
- **UBODT**: `input/map/hainan_ubodt_indexed.bin`

---

## Related Subagent Files

- `docs/subagents/REVIEW_AGENT_BRIEFING.md` — IEEE T-ITS reviewer persona (2026-06-06)
- `docs/subagents/LITERATURE_AGENT.md` — Literature search and analysis
- `docs/subagents/EXPERIMENT_EXPERT.md` — Experiment design and data analysis
- `docs/subagents/HMM_THEORY_AGENT.md` — HMM mathematical derivations
- `docs/subagents/PLOT_SUBAGENT.md` — Figure generation standards
