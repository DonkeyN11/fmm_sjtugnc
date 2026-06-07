# HMM Theory & Algorithm Agent — Knowledge Transfer File

**Created**: 2026-06-05
**Source session**: CMM math refactor, TW formula debugging, essay alignment
**Purpose**: Enable a new chat session to resume HMM mathematical reasoning, C++ algorithm debugging, and essay-code consistency checks.

---

## 1. What This Agent Does (That Others Can't)

This agent operates at the intersection of three domains that other subagents treat separately:

| Capability | Other Subagents | This Agent |
|------------|:---:|:---:|
| Trace HMM probability formulas through ~2200 lines of C++ | No | **Yes** |
| Identify mathematical bugs in forward/Viterbi recurrence | No | **Yes** |
| Verify essay formulas match actual code implementation | No | **Yes** |
| Understand degeneracy of joint path posterior for long trajectories | No | **Yes** |
| Propose probabilistic fixes with mathematical justification | No | **Yes** |
| Edit LaTeX to align with C++ code changes | Plot subagent can edit LaTeX | **Yes** |
| Run CMM/FMM, analyze results, generate figures | Experiment expert can | **Yes** |
| Read and extend math_theory.md | No | **Yes** |

## 2. Key Mathematical Framework

### HMM Probability Model (as of 2026-06-05)

**Emission Probability** (EP):
```
log p(z_t | s_t) = -0.5 * [log(2π) + log|Σ_eff| + Δx^T Σ_eff^{-1} Δx]
Σ_eff = Σ_GNSS + σ_map^2 * I  (anisotropic covariance + isotropic map noise)
```
- EP is PHMI-conditioned: inside-PL group gets (1-PHMI) mass, outside-PL gets PHMI mass
- Background pseudo-candidate added with fixed probability `background_prob` (linear, default 0.1)
- Real candidates scaled by (1 - bg_prob); bg gets bg_prob

**Transition Probability** (TP):
```
tp_raw = min(1.0, eu_dist / sp_dist)   // same as FMM
tp_norm = tp_raw / Σ_b tp_raw[a][b]     // row-normalized: Σ_j a(i,j) = 1
```

**Forward Recurrence** (α):
```
log α_t(j) = log Σ_i exp(log α_{t-1}(i) + log tp_norm(i,j)) + log p(z_t | s_t)
Initial: log α_1(i) = log(1/K) + log p(z_1 | s_1)
```
- `forward_cumu` stores log α_t in TGNode
- α_t sums over ALL paths — converges to a distribution over states

**Viterbi Recurrence** (δ):
```
log δ_t(j) = max_i [log δ_{t-1}(i) + log tp_norm(i,j)] + log p(z_t | s_t)
Initial: log δ_1(i) = log(1/K) + log p(z_1 | s_1)
```
- `cumu_prob` stores log δ_t in TGNode
- δ_t picks the SINGLE best path

**Filtering Posterior** (TW — current implementation):
```
TW_t = P(x_t = i* | z_{1:t}) = exp(forward_cumu_{i*}) / Σ_j exp(forward_cumu_j)
     = softmax(forward_cumu) at Viterbi winner
```
- Per-epoch probability ∈ [0,1]
- Uses per-layer normalization — avoids degeneracy

**Joint Path Posterior** (degenerate — NOT used):
```
P(H*_{1:t} | z_{1:t}) = exp(cumu_prob_{i*}) / Σ_j exp(forward_cumu_j)
```
- Mathematically correct but decays exponentially with trajectory length
- With row-normalized TP, forward sum grows ~0.02 nats/epoch faster than Viterbi max
- After 1675 epochs: exp(-34) ≈ 1.7e-15 — practically zero

**n_best_trustworthiness**:
- Top-3 filtering posteriors in the layer (softmax of forward_cumu)
- Background pseudo-candidate (edge=nullptr) is EXCLUDED from n_best

## 3. Critical Code Locations

### Key Files
| File | Role |
|------|------|
| `src/mm/cmm/cmm_algorithm.cpp` | Main CMM engine (~2200 lines) |
| `src/mm/fmm/fmm_algorithm.cpp` | FMM baseline |
| `src/mm/transition_graph.cpp` | TGNode, calc_tp, calc_ep, reset_layer |
| `src/mm/transition_graph.hpp` | TGNode struct (cumu_prob, forward_cumu, trustworthiness) |
| `experiments/math_theory.md` | Mathematical derivation |

### Critical Functions
| Function | File:Line | What It Does |
|----------|-----------|-------------|
| `update_layer_cmm` | cmm_algorithm.cpp:~1983 | Forward α + Viterbi δ recursion. **Line ~1998**: incoming_log_probs MUST use `node_a.forward_cumu` not `node_a.cumu_prob` |
| `initialize_first_layer` | cmm_algorithm.cpp:~1807 | Sets cumu_prob = forward_cumu = log(1/K) + ep for road, ep for bg |
| `process_sub_segment` | cmm_algorithm.cpp:~1412 | Extracts TW from TGNode, builds output. n_best excludes bg (c==nullptr) |
| `reset_layer` | transition_graph.cpp:~126 | FMM first layer. Now initializes forward_cumu + softmax |
| `update_layer` | fmm_algorithm.cpp:~388 | FMM Viterbi + forward. **Bug fixed**: incoming_logs uses forward_cumu |

### Known Bugs Fixed (2026-06-05)
| Bug | File | Fix |
|-----|------|-----|
| Forward α used cumu_prob (Viterbi max) instead of forward_cumu | CMM + FMM | `incoming_log_probs.push_back(node_a.forward_cumu + log_tp)` |
| n_best dominated by background candidate (62% of epochs) | CMM | `if (node.c != nullptr)` filter |
| EP-only fallback for unreachable candidates inflated log_Z | FMM | Removed fallback |
| CLI default `background_prob` was -20.0 (log-space), should be 0.1 (linear) | CMM config | Fixed `default_value("0.1")` |

## 4. Current Exp6 Results (2026-06-05)

```
Config: k=16, lag=0, protection_level_multiplier=10, phmi_pl_multiplier=1,
        background_prob=0.1, map_error_std=5e-5

CMM: accuracy=91.4%, ECE=0.078, AUC=0.764, TW sep=0.332, pos err=5.3m
FMM: accuracy=88.1%, ECE=0.107, AUC=0.965, TW sep=0.085, pos err=9.4m
```

## 5. Build & Run

```bash
cd /home/ncz/fmm_sjtugnc/build && make -j$(nproc) cmm fmm
cd /home/ncz/fmm_sjtugnc && ./build/cmm input/config/cmm_config_omp.xml
python3 experiments/scripts/align_real_data.py
python3 experiments/scripts/exp6_real_accuracy.py
```

## 6. Essay

- LaTeX: `docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/Trustworthiness Evaluation Framework.tex`
- PDF: same directory, `Trustworthiness Evaluation Framework.pdf`
- Build: `pdflatex -interaction=nonstopmode "Trustworthiness Evaluation Framework.tex"` (run 3×)
- TW is described as **filtering posterior** P(x_t | z_{1:t}), NOT joint path posterior
- Algorithm 1 line 1242: `tw_i ← α_i(i*) / Σ α_i` (filtering posterior)
- Key equation label: `eq:filtering_posterior_tw`

## 7. When to Use This Agent

Invoke this agent when:
- A C++ HMM probability bug needs mathematical root-cause analysis
- The essay's formulas need verification against the C++ implementation
- TW behavior seems wrong and needs tracing through forward/Viterbi recurrence
- New probabilistic features need to be designed with HMM theory consistency
- A reviewer asks about the mathematical rigor of the trustworthiness metric
