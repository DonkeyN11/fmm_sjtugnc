# Experiment Records

## 2025-05-31

### Completed: Real-Data Visualization Pipeline

**Data sources aligned** in `experiments/data/real_data/`:

| File | Source | Description |
|------|--------|-------------|
| `cmm_input_points.csv` | SPP receiver output | 16,155 epochs, 7 trajectories (id=11,12,13,14,21,22,23) |
| `ground_truth_points.csv` | RTK NMEA | 15,758 RTK ground truth positions, timestamp-matched to SPP |
| `cmm_result.csv` | CMM (k=16, point mode) | Match results with trustworthiness, posterior entropy |
| `fmm_result.csv` | FMM (k=16, point mode, regenerated) | Match results for baseline comparison |
| `aligned.csv` | Timestamp-joined | All 4 sources joined on `(id, timestamp)` with unified `uni_seq` |

**Visualization** at `experiments/output/spp_error/mapbox_real_viz.html` (171 MB):
- 6 layers: SPP obs (blue), RTK GT (green), CMM path (orange), FMM path (red), trustworthiness colormap (toggle with 't'), road network (gray)
- All layers share consistent seq via monotonic per-trajectory counter
- GT points use observation's seq (matched by normalized timestamp)

**Scripts created/modified:**
| Script | Purpose |
|--------|---------|
| `experiments/scripts/build_real_data_gt.py` | Two-pass RTK NMEA parser → `ground_truth_points.csv` (timestamp-based matching) |
| `experiments/scripts/align_real_data.py` | Timestamp-aligned CSV joining all 4 sources |
| `experiments/scripts/mapbox_viz.py` | Modified: monotonic seq for CMM/FMM; GT seq from obs mapping; timestamp normalization |
| `experiments/scripts/mapbox_spp_rtk.py` | Mapbox: SPP + covariance ellipses vs RTK |
| `experiments/scripts/mapbox_real_data.py` | Mapbox: using aligned.csv |

**Bugs fixed:**
1. RTK NMEA parser: GGA appears before RMC → first point skipped → 1-epoch lag. Fixed: two-pass parser with RMC self-registration.
2. CMM sub-trajectory seq reset → CMM seq=0 for each sub-trajectory. Fixed: monotonic per-trajectory counter.
3. GT seq mismatch → GT used its own seq (starting at 0). Fixed: `(id, timestamp_normalized)` → obs seq lookup.
4. FMM trajectory-mode output → only 136 unique timestamps. Fixed: re-ran FMM in point-mode.
5. Timestamp format mismatch (`"1750306259.0"` vs `"1750306592"`). Fixed: `round(float(ts))` normalization.
6. **Seq-based alignment breaks at SPP gaps** → SPP seq 419→420 has 110s gap, all subsequent RTK positions got wrong timestamps (~110s fore-leap). Fixed: replaced seq-offset with timestamp-based matching (`build_real_data_gt.py`).

---

## 2025-06-01

### Completed: GT Road Segment Labeling & OCR Attempt

1. **Manual GT labeling**: Donkey.Ning manually labeled GT road segments for traj 11 by visually inspecting the Mapbox visualization. Notes saved as:
   - `experiments/data/real_data/traj11_GT.pdf` (5 pages, handwritten)
   - `experiments/data/real_data/traj11_GT1.jpg` through `traj11_GT5.jpg`
   - Format: edge_id with seq range and timestamps, one road segment per line

2. **OCR attempt failed**: 
   - tesseract (conda): segfault, then library path issues, then handwriting completely unreadable
   - easyocr: numpy compatibility crash
   - paddleocr: OOM killed
   - Model vision: images unsupported in this environment
   - Grid paper background and cursive handwriting defeated all OCR approaches

3. **Resolution**: User will enter GT edge data as a minimal text file (`gt_edges_traj11.txt`) with format:
   ```
   edge_id: seq_start seq_end
   ```
   Timestamps are auto-derived from `cmm_input_points.csv`.

### Status at End of Day

**Working:**
- SPP observations ↔ RTK GT position alignment (timestamp-based, verified no >5m error outside genuine SPP faults)
- Mapbox visualization with correct seq across all layers
- CMM and FMM match results for all 7 trajectories

**Pending:**
- GT road segment CSV (`ground_truth.csv`) — traj 11 labeled (handwritten), not yet digitized; traj 12-23 not started
- Segment-level accuracy computation for real-vehicle data
- Paper Section 6 (Real Vehicle Experiment) rewrite

---

## 2025-06-02 (Tomorrow)

### Priority 1: Build All GT Road Segments

- [ ] Digitize traj 11 GT edges from handwritten notes → `ground_truth.csv`
- [ ] Label GT road segments for traj 12, 13, 14, 21, 22, 23
- [ ] Format: `id; geom; timestamp; edge_ids; point_edge_ids` (matching synthetic data format)
- [ ] Write script to auto-derive timestamps from seq ranges + `cmm_input_points.csv`

### Priority 2: Compute Real-Vehicle Matching Accuracy

- [ ] Run segment accuracy: compare CMM `cpath` and FMM `cpath` against GT edge IDs
- [ ] Compute ECE, ROC/AUC for trustworthiness on real data
- [ ] Compare CMM vs FMM on real SPP data (7 trajectories, ~16K epochs)
- [ ] Generate figures: accuracy bar chart, reliability diagram, ROC curves

### Priority 3: Update Paper

- [ ] Rewrite Section 6 (Real Vehicle Experiment) with actual results
- [ ] Add SPP error statistics (mean 2.0m, P95 4.1m)
- [ ] Add Stanford plot (SPP error vs RAIM PL, P_md=0)
- [ ] Compare synthetic (Exp1-5) vs real-vehicle results
- [ ] Recompile PDF

### Related Files

```
experiments/data/real_data/
  traj11_GT.pdf / traj11_GT*.jpg   # Handwritten GT edge labels (traj 11 only)
  cmm_input_points.csv              # SPP observations
  ground_truth_points.csv           # RTK positions (GT, timestamp-matched)
  cmm_result.csv                    # CMM match results (point-mode, k=16)
  fmm_result.csv                    # FMM match results (point-mode, k=16)
  aligned.csv                       # Timestamp-aligned unified data
```

---

## 2025-06-02 Morning

### Completed: Traj 11 GT Segments + Real-Vehicle Accuracy

1. **GT road segments digitized** from handwritten notes → `ground_truth.csv` (point-format). Script: `experiments/scripts/build_gt_segments.py`. Coverage: 2,696 epochs, 30 unique edges, 70 transitions, 151 no-road (5.6%).

2. **Real-vehicle matching accuracy computed** (script: `exp6_real_accuracy.py`). Used `aligned.csv` for correct timestamp-based seq matching:

| Metric | CMM | FMM |
|--------|:---:|:---:|
| Edge accuracy | **94.5%** | 83.7% |
| Position error (mean) | 4.6 m | 7.4 m |
| Position error (P95) | 12.9 m | 29.5 m |
| ECE (10-bin) | 0.491 | 0.152 |
| ROC AUC | 0.251 | 0.951 |
| Epochs evaluated | 2,545 | 2,545 |
| No-road excluded | 151 (5.6%) | same |

**CMM wins on accuracy (+10.8pp) and position error. ECE/AUC are worse.**

### Why CMM's ECE Failed to Meet Expectations

**Expected (from synthetic Exp3 at σ=30m):** CMM ECE≈0.13, AUC≈0.83 — trust should be well-calibrated.

**Observed (real data):** CMM ECE=0.49, AUC=0.25 — trust is anti-correlated with correctness.

**Root cause — CMM trust is compressed at ~0.5, driven by 3 factors:**

**1. Sub-trajectory restart destroys HMM memory.**
CMM splits traj 11 into 8 sub-trajectories at gaps (110s, 222s, etc.). At each split, the HMM prior resets to uniform over k=16 candidates. With L=5 smoothing, the first few epochs of each sub-trajectory have only ~5 future observations to accumulate evidence — not enough to break symmetry on a dense road network. Softmax stays near 1/K_eff ≈ 0.5.

**2. Dense urban road network with high candidate ambiguity.**
The Haikou network has parallel roads, bidirectional edges, and closely-spaced intersections. With SPP noise at only ~2m, 8+ satellites at >45° elevation, the position error is often within 1σ of multiple road edges. All plausible candidates get similar emission scores → softmax evenly distributes mass → max trust capped at ~0.3–0.5.

**3. k=16 with L=5 gives insufficient evidence accumulation.**
In synthetic experiments (σ=30m, sparse roads), L=5 was enough because large position errors made emission probabilities more differentiated — wrong edges were strongly penalized. On real data (σ≈2m SPP), emission differences between candidates are tiny. The Viterbi forward pass needs more future evidence to accumulate discriminating power.

**Evidence from trust distribution:**
```
CMM trust: mean=0.454, median=0.500, P25=0.498, P75=0.500 (IQR=0.002)
FMM trust: mean=0.979, median=0.999, almost all >0.9

CMM bins: <0.1=158, 0.1-0.3=120, 0.3-0.5=1626, 0.5-0.7=792, >0.7=0
FMM bins: <0.1=6,   0.5-0.7=51,     >0.7=2638
```

90% of CMM trust values are in [0.3, 0.7]. Zero epochs >0.7. Trust is nearly constant — useless as a discriminator regardless of correctness. CMM trusts a correct match at ~0.5 the same as a wrong match at ~0.5.

**Why FMM looks better on ECE/AUC (but isn't):**
- FMM trust is trivially binary: ~0.94 for correct, ~0.999 for also-correct
- This tiny spread happens to weakly correlate with correctness → high AUC
- ECE≈0.15 only because most epochs land in the [0.9, 1.0] bin where accuracy is 0.837, so |0.95-0.837|=0.11 — a single-bin artifact
- FMM is practically uncalibrated: claims ~100% confidence but is wrong 16.3% of the time

**Conclusion: The softmax posterior trustworthiness with L=0, k=16 does NOT function as a mismatch detector on dense urban networks with good GNSS (SPP ~2m).**

**Update:** Re-ran CMM with explicit `lag=0` — results unchanged (ECE=0.491, AUC=0.315). Lag is NOT the cause of trust compression.

### Output Files
```
experiments/data/real_data/ground_truth.csv     # GT edges (2,696 rows, point-format)
experiments/scripts/build_gt_segments.py        # GT edge → CSV builder
experiments/scripts/exp6_real_accuracy.py       # Real-vehicle accuracy analysis
experiments/output/exp6_real/traj11_accuracy.png # 6-panel comparison figure
experiments/output/exp6_real/traj11_stats.csv    # Numeric summary
```

---

## 2025-06-02 Afternoon Schedule

### Schedule Item 1: Improve CMM Trustworthiness Calibration on Real Data

#### 1. Lag Steps Sweep
- **Status**: Lag=0 tested — no improvement. Larger lag unlikely to help (trust compression is from softmax, not evidence window). **Skip.**

#### 2. H0 Lambda as Alternative Trust Metric
- **Hypothesis**: h0_lambda accumulates log-likelihood ratio over the full trajectory, providing a trajectory-global confidence score that may be better calibrated than per-epoch softmax
- **Test**: Extract h0_lambda from CMM `cmm_result.csv`; compute ECE/AUC using h0_lambda as the score
- **Note**: h0_lambda needs normalization (sigmoid or clamping) to map from [0, ∞) to [0, 1]

#### 3. Alternative Normalizations of Candidate Scores
- **Hypothesis**: Softmax over k=16 compresses trust when candidates are equally plausible. Alternative normalizations:
  - **Top-2 ratio**: tw = p_1 / (p_1 + p_2) — measures winner's dominance
  - **Entropy-based**: tw = 1 - H/H_max — normalized information gain (already in delta_entropy column)
  - **Raw log-likelihood**: tw = sigmoid(score_max - score_mean)
- **Test**: Parse `candidates` column from cmm_result.csv to reconstruct per-epoch candidate scores, recompute trust under each normalization
- **Metrics**: ECE, AUC, trust distribution shape for each variant

### Schedule Item 2: Complete GT Road Segments for All 7 Trajectories

- [ ] Traj 12 (id=12): 724 epochs — label road edges in Mapbox
- [ ] Traj 13 (id=13): 2,391 epochs — label road edges
- [ ] Traj 14 (id=14): 1,370 epochs — label road edges
- [ ] Traj 21 (id=21): 3,565 epochs — label road edges
- [ ] Traj 22 (id=22): 2,490 epochs — label road edges
- [ ] Traj 23 (id=23): 2,721 epochs — label road edges
- [ ] Type edge IDs into `build_gt_segments.py` (same format as traj 11)
- [ ] Re-run `exp6_real_accuracy.py` on all 7 trajectories combined
- [ ] Generate full real-vehicle comparison figure (CMM vs FMM across all trajectories)

#### 2. H0 Lambda as Alternative Trust Metric
- **Hypothesis**: h0_lambda accumulates log-likelihood ratio over the full trajectory, providing a trajectory-global confidence score that may be better calibrated than per-epoch softmax
- **Test**: Extract h0_lambda from CMM `cmm_result.csv`; compute ECE/AUC using h0_lambda as the score
- **Note**: h0_lambda needs normalization (sigmoid or clamping) to map from [0, ∞) to [0, 1]

#### 3. Alternative Normalizations of Candidate Scores
- **Hypothesis**: Softmax over k=16 compresses trust when candidates are equally plausible. Alternative normalizations:
  - **Top-2 ratio**: tw = p_1 / (p_1 + p_2) — measures winner's dominance
  - **Entropy-based**: tw = 1 - H/H_max — normalized information gain (already in delta_entropy column)
  - **Raw log-likelihood**: tw = sigmoid(score_max - score_mean)
- **Test**: Parse `candidates` column from cmm_result.csv to reconstruct per-epoch candidate scores, recompute trust under each normalization
- **Metrics**: ECE, AUC, trust distribution shape for each variant

