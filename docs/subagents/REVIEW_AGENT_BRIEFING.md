# Review Agent Briefing — CMM Paper for IEEE T-ITS

**Created**: 2026-06-06 (from session `review_on_article260605`)
**Role**: IEEE T-ITS reviewer persona for the CMM paper
**Purpose**: Read this file at the start of any new session to reconstruct the reviewer context. All detailed findings are in the linked files below.

---

## Quick-Start (for a new session)

To activate the reviewer persona, tell Claude:

> "Adopt the IEEE T-ITS reviewer persona defined in `docs/REVIEW_AGENT_BRIEFING.md`. First read all linked files in that document, then help me revise the paper."

Or more explicitly:

> "Read `docs/REVIEW_AGENT_BRIEFING.md`, then read `docs/reviews260605.md` and `docs/traj22_deep_dive_260606.md`. You are now the review agent for my CMM paper. Help me address the P0 issues in priority order."

---

## Files That Encode This Session's Knowledge

### Primary (read these first)

| File | Lines | Content |
|------|:-----|---------|
| `docs/REVIEW_AGENT_BRIEFING.md` | this file | Agent persona, methodology, file index |
| `docs/reviews260605.md` | 637 | **Full paper review**: novelty, math rigor, experiments, 6 P0 + 7 P1 + 5 P2 issues |
| `docs/traj22_deep_dive_260606.md` | 350+ | **Traj 22 root cause analysis**: Viterbi arithmetic, 4 hypotheses, emission model limitation |

### Context (read if deeper understanding needed)

| File | Content |
|------|---------|
| `experiments/RECORDS.md` (last 3 sections) | 2026-06-05/06 entries: review summary + traj 22 deep dive status |
| `docs/Trustworthiness Evaluation Framework.../ARTICLE_REFINEMENT_NOTES.md` | Paper structure, key results, code-to-article mapping, TODO items |
| `experiments/math_theory.md` | HMM mathematical derivation (Chinese) — forward algorithm, Viterbi, posterior confidence |

### Source Code (for bug investigation)

| File | Key Lines | Relevance |
|------|:----------|-----------|
| `src/mm/cmm/cmm_algorithm.cpp` | 265–344 | `create_edge_candidate()` — Mahalanobis projection |
| `src/mm/cmm/cmm_algorithm.cpp` | 925–978 | Per-edge dedup (keep best metric) |
| `src/mm/cmm/cmm_algorithm.cpp` | 1742–1800 | `get_sp_dist()` — shortest path + reverse guard |
| `src/mm/cmm/cmm_algorithm.cpp` | 1892–1990 | `update_layer_cmm()` — TP row norm + Viterbi forward |
| `src/mm/cmm/cmm_algorithm.cpp` | 1927–1940 | **Cumulative reverse travel guard** (bug suspect) |
| `src/mm/fmm/fmm_algorithm.cpp` | 374–461 | FMM update_layer for comparison |

### Experiment Scripts

| File | Experiment |
|------|-----------|
| `experiments/scripts/exp6_real_accuracy.py` | Real-vehicle matching accuracy (7 trajectories) |
| `experiments/scripts/exp4_sigma_mismatch.py` | Sigma mismatch analysis |
| `experiments/scripts/exp5_degraded_conditions.py` | Degraded condition evaluation |
| `experiments/scripts/utils.py` | Shared metrics: ECE, ROC/AUC, segment accuracy, Stanford plots |

### Data Files

| File | Content |
|------|---------|
| `experiments/data/real_data/cmm_result.csv` | CMM match results (16,155 epochs, semicolon-delimited) |
| `experiments/data/real_data/fmm_result.csv` | FMM baseline results |
| `experiments/data/real_data/aligned.csv` | Timestamp-aligned SPP + CMM + FMM + GT |
| `experiments/data/real_data/ground_truth.csv` | GT edge segments (16,155 rows, point-format) |
| `input/map/hainan/edges.shp` | Haikou road network (field `key` = edge ID) |

---

## Review Methodology Used

This session conducted a **5-dimensional review**:

1. **Paper text analysis**: Read full 1887-line LaTeX manuscript, checked consistency, English, structure
2. **Code-to-paper alignment**: Verified that every equation and algorithm in the paper matches the C++ implementation
3. **Mathematical derivation check**: Traced Viterbi arithmetic, TP normalization, forward algorithm
4. **Experimental data audit**: Loaded actual CSV results, verified reported numbers against raw data
5. **Failure case deep-dive**: Exhaustive per-epoch analysis of traj 22 seq 1680–1935

### Reviewer Persona

- **Role**: IEEE T-ITS reviewer with GNSS integrity + HMM map matching expertise
- **Stance**: Supportive but rigorous — wants the paper to succeed, will not accept sloppy math or unsupported claims
- **Red lines**: (a) equations that don't match the code, (b) claims without statistical evidence, (c) novelty overstatement

---

## Summary of All Findings

### Novelty Assessment
- **Core contribution is the SYSTEMATIC INTEGRATION**, not individual components
- Paper should reposition as a systems paper for T-ITS
- WLS derivation, RAIM/ARAIM description, Viterbi/forward algorithms are NOT novel — condense or cite

### Critical Issues (P0 — 6 items)

| # | Issue | Location | Fix |
|:--|:------|:---------|:----|
| 1 | PL formula describes ARAIM but code implements RAIM | §IV-A.2, `compute_raim_pl.py` | Rewrite §IV-A.2 to describe RAIM only |
| 2 | No statistical significance tests anywhere | §V, §VI | Add bootstrap CI, McNemar, DeLong tests |
| 3 | "Exponential decay" claim about forward denominator is wrong | §IV-C.2 | Revise derivation |
| 4 | Ablation study is one sentence, not a table | §VI-E | Run 4-config ablation, add table |
| 5 | ECE inconsistent: abstract says 0.078, conclusion says 0.072 | Abstract vs §VII | Use 0.072 throughout |
| 6 | Missing limitations paragraph | §VII | Single city, single receiver, 7 trajectories |

### Should-Fix Issues (P1 — 7 items)

| # | Issue |
|:--|:------|
| 7 | Restructure Related Work around 3 gaps, not topics |
| 8 | Compress WLS derivation to 5 lines |
| 9 | Define "trustworthiness" formally in §I |
| 10 | Add partial AUC (pAUC) to defend against FMM's inflated AUC |
| 11 | Discuss FMM configuration fairness (r=0.03° may disadvantage FMM) |
| 12 | Connect background state p_bg=0.1 to Laplace smoothing literature |
| 13 | Expand traj 22 failure case with Viterbi arithmetic from deep dive |

### Traj 22 Deep Dive — Key Conclusions

1. **Viterbi IS mathematically correct** — not a software bug
2. **Emission model systematically favors wrong edge**: EP(76260)≈0.6 vs EP(33989)≈0.3 at most epochs
3. **TP glue isn't strong enough**: EP advantage (+245 nats over 256 epochs) overwhelms TP penalties (-83 nats)
4. **Root cause**: Mahalanobis emission cannot distinguish parallel edges 12m apart with 2–5m SPP noise
5. **Three hypotheses for seq 1680 transition failure** (need debug confirmation):
   - H1: Cumulative reverse travel guard blocks 33989 self-TP
   - H2: Edge ID mismatch — cand[1] assigned to wrong edge
   - H3: Row normalization dilutes self-TP (rejected)
6. **Long-term fix**: α_geom + α_vel discount factors (see RECORDS.md 2025-06-03)

### Experiment Status — ⚠️ NOTE: Review was conducted on PRE-FIX CMM (before cumulative_reverse_pct=0.03 fix)

**Pre-fix numbers (review baseline, 2026-06-05)**:

| Metric | CMM (pre-fix) | FMM | Notes |
|--------|:---:|:---:|:------|
| Edge accuracy | 91.4% | 88.1% | +3.3pp |
| Position error (mean) | 5.3m | 9.4m | 44% lower |
| ECE | 0.072 | 0.107 | CMM better calibrated |
| AUC | 0.764 | 0.965 | FMM inflated (TW compression) |
| TW separation | 0.329 | 0.085 | 3.9× better |

**Post-fix numbers (current, 2026-06-07, 3% reverse guard, verified against aligned.csv)**:

| Metric | CMM (post-fix) | FMM | Notes |
|--------|:---:|:---:|:------|
| Edge accuracy | **96.9%** | 88.1% | +8.8pp |
| ECE (TW, 10-bin) | **0.069** | 0.107 | 36% reduction |
| AUC | **0.606** | 0.965 | CMM AUC dropped (less TW compression = more realistic) |
| TW separation | **0.291** | 0.085 | 3.4× better |
| Mean TW | 0.928 | 0.944 | |
| TW > 0.95 | 82.6% | — | At lag=0, without smoothing |
| Traj 22 accuracy | **93.5%** | 83.8% | +20.6pp from pre-fix (was 72.9%) |

**P0 Re-evaluation against post-fix state**:
- P0-5 (ECE inconsistency): **RESOLVED** — paper now uses 0.069 consistently
- P0-4 (ablation study): **PARTIALLY RESOLVED** — ECE ablation figure exists, but full multi-metric table still needed
- P0-1, P0-2, P0-3, P0-6: **STILL UNRESOLVED** — methodological issues independent of code fixes
- All P1 issues: **STILL UNRESOLVED**

---

## Suggested Prompts for Future Sessions

### For paper revision
> "Adopt the reviewer persona from `docs/REVIEW_AGENT_BRIEFING.md`. Read all linked files. Start with P0-1: rewrite Section IV-A.2 to describe RAIM only, removing the ARAIM MHSS description."

### For traj 22 debugging
> "Read `docs/traj22_deep_dive_260606.md`. Add debug fprintf at cmm_algorithm.cpp:1927 to trace the reverse guard for edge 33989. Rebuild CMM and re-run traj 22."

### For ablation study
> "Read `docs/reviews260605.md` §3.2.4. Design and run the 4-configuration ablation study: FMM → CMM+unnormTP → CMM+normTP → CMM+normTP+bgState+uniformPrior. Measure accuracy, ECE, AUC, TW separation for each."

### For statistical tests
> "Read `docs/reviews260605.md` §3.2.2. Add bootstrap 95% CI, McNemar's test, and DeLong test to all key metric comparisons. Update all tables."

### For related work restructuring
> "Read `docs/reviews260605.md` §4.2.1. Restructure Section II around three gaps: (1) isotropic emission, (2) GNSS integrity never in map matching, (3) no calibration analysis of confidence scores."

---

*This briefing encodes the full review session `review_on_article260605` (2026-06-05/06).*
*All linked files exist in the repository and should be read in full by any session adopting this persona.*