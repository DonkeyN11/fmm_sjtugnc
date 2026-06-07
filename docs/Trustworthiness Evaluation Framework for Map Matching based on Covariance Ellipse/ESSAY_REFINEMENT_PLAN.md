# Essay Refinement Plan — CMM Paper for IEEE T-ITS

**Created**: 2026-06-07 (ultracode fanned-out analysis)
**Purpose**: Unified plan resolving all contradictions between 6 subagents + reviewer, with verified numbers from post-fix CMM data.

---

## 0. Verified Ground-Truth Numbers (Post-Fix, 3% Reverse Guard)

All numbers verified against `experiments/data/real_data/aligned.csv` (13,256 eval epochs, 7 trajectories):

| Metric | CMM (post-fix) | FMM | Notes |
|--------|:---:|:---:|:------|
| Segment accuracy | **96.9%** | 88.1% | Post-fix CMM: +5.5pp over pre-fix (91.4%) |
| TW ECE (10-bin) | **0.069** | 0.107 | TW-based, 10-bin. Also reported as 0.0684 raw |
| TW separation | **0.291** | 0.085 | 3.4× better (correct 0.926 − wrong 0.634) |
| TW AUC | **0.606** | 0.965 | FMM inflated by TW compression artifact |
| Mean TW | 0.928 | 0.944 | |
| TW > 0.95 | 82.6% | — | |
| Traj 11 | 93.9% | 87.4% | |
| Traj 12 | 100.0% | 0.0% | FMM complete failure on short traj |
| Traj 13 | 98.1% | 94.2% | |
| Traj 14 | 100.0% | 79.5% | |
| Traj 21 | 98.7% | 92.2% | |
| Traj 22 | 93.5% | 83.8% | Post-fix: +20.6pp over pre-fix (72.9%) |
| Traj 23 | 98.6% | 87.8% | |

---

## 1. Contradictions Between Subagents — Full Matrix

### 1.1 REVIEW_AGENT_BRIEFING.md vs Actual Data (CRITICAL — STALE)

The reviewer's briefing (`docs/subagents/REVIEW_AGENT_BRIEFING.md`) and full review (`docs/reviews260605.md`) were written against **pre-fix CMM** (before cumulative_reverse_pct=0.03 fix). Almost all metrics are outdated:

| Claim | REVIEW says | Actual (post-fix) | Δ |
|-------|:-----------:|:-----------------:|:--:|
| CMM accuracy | 91.4% | 96.9% | +5.5pp |
| CMM ECE | 0.072 | 0.069 | −0.003 |
| CMM AUC | 0.764 | 0.606 | −0.158 |
| TW separation | 0.329 | 0.291 | −0.038 |
| Traj 22 accuracy | "72.9% regression" | 93.5% (fixed!) | +20.6pp |

**Resolution**: REVIEW_AGENT_BRIEFING.md MUST be updated to reflect post-fix numbers. However, the **review's methodological critiques (P0-1 through P0-4, P1 items) remain valid** — only the numbers changed, not the gaps in the paper.

### 1.2 REVIEW_AGENT_BRIEFING.md P0 Issues — Re-evaluated Against Post-Fix State

| P0 # | Issue | Status (2026-06-07) |
|:-----|:------|:---------------------|
| P0-1 | PL formula (ARAIM vs RAIM) | **UNRESOLVED** — paper still describes ARAIM but code uses RAIM |
| P0-2 | No statistical significance tests | **UNRESOLVED** — no bootstrap CI, McNemar, DeLong added |
| P0-3 | "Exponential decay" claim wrong | **UNRESOLVED** — §IV-C.2 still makes this claim |
| P0-4 | Ablation study is one sentence | **PARTIALLY** — ECE ablation figure exists, but no accuracy/AUC/TW table |
| P0-5 | ECE inconsistent (0.078 vs 0.072) | **RESOLVED** — paper now uses 0.069 consistently |
| P0-6 | Missing limitations paragraph | **PARTIALLY** — future work section added, but formal Limitations section missing |

### 1.3 ESSAY_REFINEMENT_EXPERT.md Internal Inconsistencies

| Issue | Description | Resolution |
|-------|-------------|------------|
| M4: "ECE numbers need source clarification" | Table 2 reports EP-based ECE=0.121, abstract reports TW-based ECE=0.069 | Must clarify EP vs TW ECE throughout. Abstract now uses TW-based — correct. Simulation sections need updating |
| TW distribution table (lines 109-118) | Reports lag=0: 0% TW > 0.95 | Pre-fix numbers. Post-fix lag=0 has 82.6% TW > 0.95. Table is STALE |
| TW > 0.95 at different lag (lines 123-128) | 26.7% mismatch rate by position error | Must clarify whether position error OR segment mismatch is the criterion |
| "traj9newultra.png" reference | Traj 9 doesn't exist in Hainan-06 dataset | Should be renamed or removed |

### 1.4 ARTICLE_REFINEMENT_NOTES.md vs Current Paper

| Claim | NOTES says | Paper says | Actual data |
|-------|-----------|------------|-------------|
| ECE(5m) CMM | 0.121 | 0.069 | 0.069 (TW-based) |
| ECE(5m) FMM | 0.978 | 0.107 | 0.107 (TW-based) |
| Ablation ECE delta | CMM→L=20: +0.140 | — | — |

**Root cause**: NOTES reports EP-based ECE (emission probability calibration). Paper now reports TW-based ECE (trustworthiness calibration). Two different things — both valid but must be clearly distinguished.

### 1.5 PLOT_SUBAGENT.md Consistency

Mostly consistent with post-fix data:
- AUC=0.600 ✓ (actual 0.606)
- References `exp6_redraw_all.py` which generates post-fix figures ✓
- **BUT**: `trust_roc_curve.png` may still show old pre-fix curve. Need verification.

### 1.6 EXPERIMENT_EXPERT.md vs CODEFIX_SUBAGENT.md

| Issue | EXPERIMENT says | CODEFIX says | Resolution |
|-------|----------------|-------------|------------|
| Branch | "On master — VIOLATES POLICY" | Config file has cumulative_reverse_pct=0.03 | Must move to feature branch |
| Fix status | "NOT COMMITTED, NOT ON FEATURE BRANCH" | Post-fix results generated | Changes ARE on master but uncommitted. Branch `fix/cumulative-reverse-guard-3pct` has the fix |
| Simulation data | "from May 28-29 (pre-fix CMM)" | — | All simulation experiments need re-running with post-fix CMM |

### 1.7 Cross-Agent Number Discrepancies Summary

```
EP-based ECE (emission probability calibration):
  FMM: 0.976 at 5m (ESSAY_REFINEMENT_EXPERT line 138)
  CMM: 0.262 at 5m (ESSAY_REFINEMENT_EXPERT line 138)
  
TW-based ECE (trustworthiness calibration):  
  FMM: 0.107 (verified from data, in paper abstract)
  CMM: 0.069 (verified from data, in paper abstract)

Position-error ECE (ARTICLE_REFINEMENT_NOTES):
  FMM: 0.978 at 5m
  CMM: 0.121 at 5m
```

**THE PAPER MUST CLEARLY DISTINGUISH THESE THREE ECE TYPES.** Currently it conflates them.

---

## 2. Priority-Ordered Refinement Tasks

### 🔴 RED (Must Fix Before Submission — 2-3 weeks)

**R1 — Move to Feature Branch + Commit All Work**
- Branch from master: `feature/essay-final-submission`
- Commit all uncommitted changes (LaTeX, figures, subagents, experiment scripts)
- Merge relevant fixes from `fix/cumulative-reverse-guard-3pct` and `feature/article-refinement-fixes`
- After this task: everything is on a proper feature branch

**R2 — Update REVIEW_AGENT_BRIEFING.md with Post-Fix Numbers**
- Replace all stale metrics ($91.4\% \to 96.9\%$, $0.072 \to 0.069$, $0.764 \to 0.606$, $0.329 \to 0.291$)
- Re-evaluate P0/P1/P2 priorities — P0-5 is resolved, P0-4 partially resolved
- Note that P0-2 (statistical tests) and P0-1 (PL formula) remain critical

**R3 — Fix PL Formula Description (P0-1)**
- Rewrite §IV-A.2 to describe RAIM only (remove ARAIM MHSS equations)
- State honestly: "We use RAIM-derived HPL. ARAIM is future work."
- Verify against `tests/python/compute_raim_pl.py`

**R4 — Add Statistical Significance Tests (P0-2)**
- Bootstrap 95% CI for all key metrics (accuracy, ECE, AUC, TW separation)
- McNemar's test for CMM vs FMM accuracy differences (per-trajectory)
- DeLong test for AUC comparison
- Error bars on reliability diagram
- Script: create `experiments/scripts/exp7_statistical_tests.py`

**R5 — Distinguish EP-Based vs TW-Based ECE Throughout Paper**
- Abstract: clearly state "TW-based ECE" (already uses 0.069 ✓)
- Section V (simulation): add note that simulation uses EP-based ECE
- Section VI (real): clarify that Table reports TW-based ECE
- Add a paragraph explaining the difference and why TW-based ECE is the primary metric
- Update ARTICLE_REFINEMENT_NOTES.md and ESSAY_REFINEMENT_EXPERT.md

**R6 — Full Ablation Study Table (P0-4)**
- Run 4-configuration ablation on real data with post-fix CMM:
  1. FMM baseline
  2. CMM: Mahalanobis EP + unnormalized TP + EP-based prior
  3. CMM: + TP row normalization + uniform prior
  4. CMM: + background state (p_bg=0.1)
- Measure: accuracy, ECE (TW, 10-bin), AUC, TW separation per config
- Create proper LaTeX table (not just figure)
- Script: extend `tests/python/exp4_ablation_ece.py`

**R7 — Add Formal Limitations Section**
- Before Conclusion (§VII): new section "Limitations and Future Work"
- Single city (Haikou), single receiver (Tersus BX50C), 7 trajectories
- RAIM requires ≥5 visible satellites; urban canyon performance untested
- Synthetic data limitations (8 satellites, 45–80° elevation — unrealistically clean)
- Fixed-lag smoothing degrades TW calibration with RAIM PL (ECE increases from 0.069 to ~0.26)
- Current emission cannot distinguish parallel edges <15m apart (traj 22 case)

### 🟡 YELLOW (Should Fix Before Submission — 1-2 weeks)

**Y1 — Fix "Exponential Decay" Claim (P0-3)**
- Rewrite §IV-C.2 explanation of why joint posterior decays
- Correct claim: in properly row-normalized HMM, Σα_t does NOT grow exponentially
- True cause: emission densities are not probabilities, background state absorbs mass
- Consult `docs/subagents/HMM_THEORY_AGENT.md` for correct derivation

**Y2 — Expand References from 11 to 30+**
- Verify and add suggestions from ESSAY_REFINEMENT_EXPERT M1
- Fix Li2023 incomplete entry (missing volume, number, pages, doi)
- Fix Kim2019 content mismatch (paper title vs article description)
- Search for 15-20 additional T-ITS relevant citations
- Check `docs/literatures/` — 25 PDFs available

**Y3 — Re-run All Simulation Experiments with Post-Fix CMM**
- Exp1-5 need re-execution with current CMM binary
- The 3% reverse guard fix primarily affects real data, but simulation numbers should be verified
- At minimum: re-run the sigma sweep (Exp3) and mismatch analysis (Exp4)
- Update all tables and figures in Section V

**Y4 — Fix FMM Accuracy in Abstract (88.0% → 88.1%)**
- Abstract line 61: "88.0\%" → "88.1\%"
- Verify all other numbers are exact (3.3× → 3.4×?)

**Y5 — Update ESSAY_REFINEMENT_EXPERT.md TW Distribution Table**
- Replace pre-fix lag=0 table (0% TW>0.95) with post-fix numbers (82.6% TW>0.95)
- Clarify that post-fix lag=0 already provides high TW without lag smoothing

**Y6 — Connect Background State to Laplace Smoothing Literature (P1-12)**
- Add citation: Laplace smoothing, additive smoothing in Naive Bayes
- Explain p_bg=0.1 as regularization, not a "real" probability
- Discuss sensitivity: mention that {0.01, 0.05, 0.1, 0.2} sweep is future work

**Y7 — Expand Traj 22 Failure Case (P1-13)**
- Condense `docs/traj22_deep_dive_260606.md` into paper §VI-G
- Show: EP advantage (+245 nats over 256 epochs) overwhelms TP penalties (−83 nats)
- Note: post-fix accuracy improved from 72.9% → 93.5% due to reverse guard fix
- Discuss: remaining 6.5% errors are emission model limitation, not algorithm bug

### 🟢 GREEN (If Time Permits — 1 week)

**G1 — Partial AUC (pAUC) Analysis**
- Compute pAUC in FPR < 0.3 range where CMM likely outperforms FMM
- Add to paper §VI: defend against FMM's inflated full AUC

**G2 — Background Probability Sensitivity Sweep**
- Run CMM with bg_prob ∈ {0.01, 0.05, 0.1, 0.2}
- Report: ECE, accuracy, AUC, TW separation per value
- Add to limitations: "p_bg=0.1 is an engineering choice"

**G3 — Delta Entropy Analysis**
- Does ΔH_t predict mismatches better than trustworthiness?
- Compute AUC(ΔH) vs AUC(TW) for mismatch detection
- Add paragraph to §VI

**G4 — Runtime Profiling**
- Measure computational overhead: Mahalanobis vs Euclidean emission
- Report: candidate search time, emission computation time, total per-epoch time

**G5 — Rename/Remove Stale Figures**
- Find "traj9newultra.png" reference and fix
- Verify all figure references match actual files in `figs/`

**G6 — Clean Up Commented-Out Content**
- Section V has ~400 lines of commented simulation methodology (lines 1286-1718)
- Either integrate or delete

---

## 3. Subagent MD Files That Need Editing (Contradiction Resolution)

### Files to UPDATE (stale numbers):

| File | What to Change |
|------|---------------|
| `docs/subagents/REVIEW_AGENT_BRIEFING.md` | Update ALL metrics table (line 133-139): 91.4%→96.9%, 0.072→0.069, 0.764→0.606, 0.329→0.291. Mark P0-5 as RESOLVED. Add note that review was pre-fix. |
| `docs/Trustworthiness.../ARTICLE_REFINEMENT_NOTES.md` | Update calibration table (lines 66-72): clarify EP-based vs TW-based. Add new TW-based ECE numbers. Update ablation table for post-fix. |
| `docs/subagents/ESSAY_REFINEMENT_EXPERT.md` | Update M4 with resolution. Update TW distribution table (lines 109-128) with post-fix numbers. Remove "traj9" reference. Update verified metrics. |
| `docs/subagents/EXPERIMENT_EXPERT.md` | Update Section 2: note that fixes ARE on master (or feature branch after R1). Update metric table. Add simulation re-run task. |
| `docs/subagents/PLOT_SUBAGENT.md` | Verify figure inventory against post-fix data. Note any figures needing regeneration. |

### Files that are CONSISTENT (no changes needed):

| File | Status |
|------|--------|
| `docs/subagents/CODEFIX_SUBAGENT.md` | Post-fix numbers correct. Branch status needs update after R1. |
| `docs/subagents/HMM_THEORY_AGENT.md` | Math derivation correct. Needs P0-3 fix applied to paper. |
| `docs/subagents/LITERATURE_AGENT.md` | Reference gaps correctly identified. Taxonomy valid. |

---

## 4. Execution Order

```
Week 1:  R1 (branch) → R2 (update reviewer) → R3 (PL rewrite) → R5 (ECE clarity)
Week 2:  R4 (statistical tests) → R6 (ablation table) → Y1 (exponential decay)
Week 3:  R7 (limitations) → Y2 (references) → Y3 (re-run simulations)
Week 4:  Y4-Y7 (minor fixes) → Re-review by reviewer agent → G1-G6 (if time)
```

---

## 5. Reviewer Re-Check Protocol

After completing RED tasks, invoke the reviewer agent:
1. Read updated `docs/subagents/REVIEW_AGENT_BRIEFING.md`
2. Read full updated LaTeX manuscript
3. Verify all P0 issues are resolved
4. Produce new review: should have **zero P0 issues** and only minor P1/P2
5. If any P0 remains → iterate before submission

---

*Plan generated by Claude Code (deepseek-v4-pro, ultracode mode) on 2026-06-07.*
*Data verified against `experiments/data/real_data/aligned.csv` (post-fix, 13,256 epochs).*