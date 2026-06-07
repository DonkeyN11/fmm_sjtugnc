# Experiment Expert Knowledge Base

Generated from full conversation history. Read this before any experiment work.

## Current Role
You are the experiment expert for the IEEE T-ITS paper "Trustworthiness Evaluation of Map Matching Leveraging Probabilistic GNSS Covariance Ellipse Models." Your knowledge covers: synthetic experiments (Exp1-5), real-vehicle experiments (Exp6), CMM C++ codebase, data pipeline, and all bugs/fixes.

---

## 1. Project Architecture

### Directory Layout
```
experiments/
  RECORDS.md              # Daily work log (read first!)
  README.md               # Full experiment documentation
  EXPERIMENT_EXPERT.md    # THIS FILE — comprehensive knowledge base
  config/
    reverse_edge_map.json # 135K bidirectional edge pairs (36014↔36016 etc.)
  data/
    sigma_01/ ~ sigma_30/ # Synthetic data (20 trajs/level, 1000 epochs/traj)
    sigma_mismatch/       # Exp4 data (pr10_wls20 ~ pr30_wls20)
    real_data/            # Real-vehicle data
  scripts/                # All 22+ experiment scripts
  output/
    exp6_real/            # Real-vehicle accuracy results
    spp_error/            # SPP vs RTK analysis + mapbox HTML
```

### Key C++ Files
- `src/mm/cmm/cmm_algorithm.cpp` — Main CMM engine (~2200+ lines)
- `src/mm/transition_graph.hpp` — TGNode struct, TransitionGraph
- `src/mm/transition_graph.cpp` — Backtrack function

### TGNode Struct (transition_graph.hpp line 29)
```cpp
struct TGNode {
  const Candidate *c;
  TGNode *prev;
  double ep;           // emission log-probability (after PHMI redistribution)
  double tp;           // transition probability from best prev
  double cumu_prob;    // Viterbi max (for backtrack) — PATCHED from forward-sum
  double forward_cumu; // Forward sum (for softmax/trust/entropy) — ADDED
  double reverse_dist; // Cumulative reverse on same edge — ADDED
  double sp_dist;
  double trustworthiness;
  double delta_entropy;
  double posterior_entropy;
};
```

---

## 2. C++ Fixes Applied (ON master — VIOLATES CLAUDE.md policy. MUST move to feature branch per ESSAY_REFINEMENT_PLAN.md R1)

All changes are on `master` — VIOLATES CLAUDE.md policy. The `fix/cumulative-reverse-guard-3pct` branch contains the same fixes. Post-fix results were generated from master and verified against `experiments/data/real_data/aligned.csv`.

### Post-Fix CMM Metrics (verified 2026-06-07)
| Metric | CMM (post-fix) | FMM |
|--------|:---:|:---:|
| Segment accuracy | 96.9% | 88.1% |
| TW ECE (10-bin) | 0.069 | 0.107 |
| TW AUC | 0.606 | 0.965 |
| TW separation | 0.291 | 0.085 |

### Fix 1: Viterbi-max cumu + forward_cumu (cmm_algorithm.cpp ~line 1971)
**Problem**: Forward-sum cumu inflated junction edges (seq 2128 bug)
**Change**: 
```cpp
node_b.cumu_prob = best_log_branch_prob + node_b.ep;     // Viterbi for path
node_b.forward_cumu = log_sum_prev_probs + node_b.ep;    // Forward for trust
```
**Impact**: Seq 2128 now stays on 149100 (was switching to 8048 due to forward-sum inflation)

### Fix 2: Per-edge candidate dedup (cmm_algorithm.cpp ~line 960)
**Problem**: Same edge generates up to 3 candidates (projection + source node + target node), inflating k
**Change**: Keep only best-metric candidate per edge index
**Impact**: Avg candidates reduced from 5.1 to 3.9

### Fix 3: Cumulative reverse guard (cmm_algorithm.cpp ~line 1882)
**Problem**: 15% per-step guard allows unlimited cumulative reverse on long edges
**Change**: Track cumulative reverse across same-edge epochs; block when > min(30m, 15% edge length)
**Impact**: Guards short edges (<200m); less effective for 3.8km edges

### Fix 4: H0 lambda outside lag_steps gate (cmm_algorithm.cpp ~line 1632)
**Problem**: H0 lambda accumulation was inside `if (lag_steps > 0)` — never ran with lag=0
**Change**: Moved accumulation outside the gate
**Impact**: TW mean improved from 0.46 → 0.92; ECE dropped from 0.49 → 0.11

### Fix 5: h0_prior_log_odds wired (cmm_algorithm.cpp lines 1495, 1549, 1612)
**Problem**: Config parameter was parsed but never used (hardcoded to 0.0)
**Change**: `h0_log_lambda = config.h0_prior_log_odds` instead of `= 0.0`
**Impact**: With `h0_prior_log_odds=10`, α₀≈1.0 — no initial trust penalty

### DEBUG CODE STILL ACTIVE — MUST REMOVE:
- `cmm_algorithm.cpp`: fprintf at lines ~1987, ~2092 (EPOCH_1676, FINAL_WINNER)
- Search for all `fprintf(stderr, ...)` in both .cpp files and remove

---

## 3. Real-Vehicle Data Pipeline

### Data Flow
```
cmm_input_points.csv (SPP obs) ──┐
ground_truth_points.csv (RTK GT) ─┤
cmm_result.csv (CMM output) ──────┼──→ align_real_data.py → aligned.csv
fmm_result.csv (FMM output) ──────┤
ground_truth.csv (GT edges) ──────┘
                                      │
                                      └──→ exp6_real_accuracy.py → accuracy/ECE/AUC
```

### Key Data Files
| File | Description | Rows |
|------|-------------|:---:|
| `experiments/data/real_data/cmm_input_points.csv` | SPP observations | 16,155 |
| `experiments/data/real_data/ground_truth_points.csv` | RTK positions (timestamp-matched) | 15,758 |
| `experiments/data/real_data/ground_truth.csv` | GT edge IDs (point-format) | 16,155 |
| `experiments/data/real_data/cmm_result.csv` | Current CMM output (k=16, lag=0, all fixes) | 16,155 |
| `experiments/data/real_data/fmm_result.csv` | FMM output (k=16, point-mode) | 16,155 |
| `experiments/data/real_data/aligned.csv` | All sources joined by timestamp | 16,155 |
| `experiments/data/real_data/GT_segments.txt` | Human-readable GT edge ranges | — |
| `experiments/data/real_data/cmm_result.bak` | Backup before latest CMM run | — |

### Trajectory Summary
| Traj | ID | Epochs | CMM Acc | Unique GT Edges | Notes |
|:---:|:---:|:---:|:---:|:---:|------|
| 11 | 11 | 2,696 | 94.5% | 30 | 3× loop 8088→1890, gas station, under bridge |
| 12 | 12 | 734 | 100% | 1 | All on 8088, mostly gas station loops |
| 13 | 13 | 2,400 | 99.1% | 26 | Same loop as traj 11 |
| 14 | 14 | 1,407 | 100% | 2 | Near 20099/72291, mostly parked |
| 21 | 21 | 3,630 | 98.6% | 64 | Downtown Haikou, longest |
| 22 | 22 | 2,551 | 83.1% | 53 | Dense urban, worst accuracy |
| 23 | 23 | 2,737 | 95.4% | 58 | Mixed route |

### Overall Results (All 7 Trajectories)
| Metric | CMM | FMM |
|--------|:---:|:---:|
| Edge accuracy | **94.8%** | 88.1% |
| ECE | **0.087** | 0.107 |
| AUC | 0.646 | 0.965 |
| Position error (mean) | **5.4m** | 9.4m |
| Position error (P95) | **13.2m** | 40.3m |
| Eval epochs | 13,256 | (2,899 excluded as no-road) |

---

## 4. Known Bugs (Active Investigation)

### BUG 1: Sub-Trajectory Boundary — Wrong Candidate in CSV (P0)
**Location**: traj 22, seq 1675→1676 (sub-trajectory boundary)
**Symptom**: Viterbi layer correctly selects 33989 (cumu=-6189.71), but CSV outputs 76260 (cumu=-6195.40)
**Evidence**: 
- `FINAL_WINNER edge=33989 cumu=-6189.71` at the correct layer
- `EPOCH_1676` debug shows 33989 with better cumu than 76260
- Seq 1675 has non-empty tpath (end of sub-trajectory A) — boundary indicator
**Hypothesis**: `process_sub_segment` or gap-bridging uses different node selection than layer Viterbi
**To fix**: Investigate `process_sub_segment` at ~line 1300; check how `tg_opath` nodes are selected at boundary layers

### BUG 2: 76260↔33989 Flip-Flop — Bidirectional Pair (P1)
**Location**: traj 22, seq 1670-1812 area
**Symptom**: CMM oscillates between 76260 and 33989 every ~20 epochs
**Root cause**: Both edges share similar geometry (nearby but not identical). 76260 enters candidate pool from edge 29612 (TP=0.17), then self-sustains. 33989's self-transition blocked by offset regression near junction.
**Impact**: ~200 mismatch epochs in traj 22

### BUG 3: Label Bias — Softmax Overconfidence (P1)
**Location**: traj 13 seq 2399
**Symptom**: max EP=1e-5 but TW=1.0 — softmax picks best of bad lot
**Fix planned**: Wire `background_log_prob` into softmax as pseudo-candidate

---

## 5. Key Algorithm Details

### Emission Probability Pipeline
1. Raw EP computed from Mahalanobis distance
2. PHMI redistribution: inside-PL get `(1-PHMI) × (ep/sum_ep_in)`; outside-PL get `PHMI × (ep/sum_ep_out)`
3. Result stored as `node->ep` (log space, used in Viterbi)
4. **CSV `ep` column** = `exp(node->ep)` — linear space, different from Viterbi value!
5. **CSV candidates column** = raw EP BEFORE PHMI redistribution

### Cumu Probability (After Viterbi Fix)
```
cumu_prob = max_a(cumu_a + log_tp(a→b)) + ep_b    // Viterbi: for backtrack
forward_cumu = log_sum_exp_a(cumu_a + log_tp(a→b)) + ep_b  // Forward: for trust
```

### Trustworthiness Computation
```
n_best_trustworthiness = softmax(cumu_prob over layer candidates)  // from node trustworthiness
output TW = n_best_tw * alpha_t  // where alpha_t = lambda_t/(1+lambda_t)
```
Note: After H0 fix, `alpha_t ≈ 1.0` after epoch 1 (with `h0_prior_log_odds=10`)

### Same-Edge Transition (get_sp_dist, line ~1720)
- Forward (offset increases): SP = diff, TP = calc_tp(SP, eu_dist)
- Small regression (<15% edge): SP = 0 (projection noise)
- Larger regression: UBODT reverse lookup → if found, SP = diff; else -1 (blocked)
- Cumulative guard: reverse_dist accumulates across consecutive same-edge epochs

---

## 6. Common Operations

### Regenerate aligned.csv
```bash
python experiments/scripts/align_real_data.py
```

### Compute full accuracy
```bash
python experiments/scripts/exp6_real_accuracy.py
```

### Build GT from GT_segments.txt
```bash
python experiments/scripts/build_gt_segments.py
```

### Run CMM on real data
```bash
# Build XML with:
#   k=16, lag_steps=0, h0_prior_log_odds=10, phmi=0.00001
#   protection_level_multiplier=3, phmi_pl_multiplier=1
# Then run:
/path/to/build/cmm config.xml
```

### Generate mapbox visualization
```bash
python experiments/scripts/mapbox_viz.py \
  --input experiments/data/real_data/cmm_input_points.csv \
  --ground-truth experiments/data/real_data/ground_truth_points.csv \
  --cmm experiments/data/real_data/cmm_result.csv \
  --fmm experiments/data/real_data/fmm_result.csv \
  --edges input/map/hainan/edges.shp \
  --output experiments/output/spp_error/mapbox_real_viz.html
```

---

## 7. Critical Conventions

1. **Seq alignment**: `cmm_result.csv` seq resets at sub-trajectories. `aligned.csv` uses `uni_seq` (monotonic). Always use `uni_seq` for analysis.
2. **Edge matching**: Use `reverse_edge_map.json` for bidirectional pair matching (`36014↔36016`).
3. **GT format**: `ground_truth.csv` is point-format: `id;seq;timestamp;x;y;edge_id`. edge_id=0 means no road; edge_id=-1 means not moved (traj 14 only).
4. **Timestamp matching**: CMM rounds to integer; GT has fractional. Normalize via `round(float(ts))`.
5. **No-road exclusion**: edge_id in ("0", "-1") excluded from accuracy computation.
6. **Build**: `cd /home/ncz/fmm_sjtugnc/build && make -j$(nproc) cmm`
7. **LaTeX compile**: `/usr/bin/pdflatex` (not conda version)
