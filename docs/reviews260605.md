---
  Comprehensive Review: "Trustworthiness Evaluation of Map 
  Matching Leveraging Probabilistic GNSS Covariance Ellipse 
  Models"
  
  Donkey.Ning, I have now read every relevant file: your full
  LaTeX paper (1887 lines), the CMM source code (~2782 lines),
  all 30 experiment scripts, RECORDS.md (804 lines of
  development diary), math_theory.md (HMM derivation),
  literature notes, and configuration files. Below is a
  structured, multi-perspective review addressing your three
  core questions.

  ---
  0. Overall Verdict
  
  ┌───────────────┬──────────────────────────┬────────────┐
  │   Criterion   │        Assessment        │ Confidence │
  ├───────────────┼──────────────────────────┼────────────┤
  │ Novelty for   │ Borderline-acceptable,   │    High    │
  │ IEEE T-ITS    │   needs repositioning    │            │
  ├───────────────┼──────────────────────────┼────────────┤
  │ Mathematical  │  Mostly sound, several   │    High    │
  │ rigor         │  claims need correction  │            │
  ├───────────────┼──────────────────────────┼────────────┤
  │ Experimental  │ Adequate scope, missing  │    High    │
  │ sufficiency   │       key analyses       │            │
  ├───────────────┼──────────────────────────┼────────────┤
  │ Ready to      │ No — requires 2–4 weeks  │    High    │
  │ submit?       │       of revision        │            │
  └───────────────┴──────────────────────────┴────────────┘

  The work is publishable at T-ITS or TVT after revisions. The
  core idea — propagating GNSS covariance through HMM map
  matching and using the filtering posterior as a calibrated
  trustworthiness metric — is genuinely novel in the map
  matching literature. However, the paper's current framing
  overclaims in several areas, some mathematical derivations
  need correction, and the experimental analysis requires
  additional rigor.

  ---
  1. Novelty Assessment
  
  1.1 What Is Genuinely Novel

  Contribution 1 — Systematic integration of GNSS stochastic 
  model into HMM map matching. This is the paper's strongest
  contribution. To my knowledge, no prior work simultaneously
  incorporates (a) per-epoch covariance matrices, (b)
  protection levels as candidate search bounds, and (c)
  Mahalanobis projection into a single HMM pipeline. While
  individual components exist in prior work (Lahrech & Soulhi
  2023 used Mahalanobis distance for map matching; Kim et al.
  2019 used PL for LiDAR localization evaluation), the
  SYSTEMATIC integration across all HMM stages is new.

  Contribution 2 — Calibration analysis of map matching 
  confidence scores. Applying ECE, reliability diagrams, and
  Brier scores to evaluate whether a map matching algorithm's
  confidence scores reflect true correctness probabilities —
  this is genuinely novel. The map matching literature has
  extensively studied accuracy but almost never studied
  calibration. The ablation study decomposing ECE improvements
  by model component is a strong methodological contribution.

  Contribution 3 — Filtering posterior as per-epoch 
  trustworthiness. Using the forward-algorithm marginal
  posterior $P(x_t = i^* \mid z_{1:t})$ as a per-epoch
  confidence score is mathematically clean. The paper
  correctly identifies that the joint path posterior decays
  with trajectory length, and the filtering posterior avoids
  this degeneracy.

  1.2 What Is NOT Novel (and Should Not Be Claimed as Such)

  1. WLS covariance derivation (Section IV-A). This is
  standard GNSS textbook material. The detailed derivation of
  $\mathbf{\Sigma}_x =
  (\mathbf{H}^T\mathbf{W}\mathbf{H})^{-1}$ occupies ~1 page
  but adds no novelty. It should be condensed to 3–4 lines
  with a citation.
  2. RAIM/ARAIM PL computation (Section IV-A.2). The RAIM
  slope method ($\text{HPL} = K \cdot \sigma_{\text{major}}$)
  is standard. The ARAIM MHSS description is textbook material
  and is NOT what the code implements. This misalignment is
  problematic (see §2.2).
  3. Viterbi and forward algorithms. Standard HMM machinery.
  The paper should not claim these as contributions — only the
  way they are APPLIED with covariance inputs is novel.
  4. Using Mahalanobis distance for candidate projection.
  Lahrech & Soulhi (2023) already did this. The paper should
  acknowledge this explicitly and distinguish its contribution
  as the full pipeline integration, not the projection
  operator.

  1.3 Positioning Strategy for T-ITS

  The paper should reposition itself as a SYSTEMS paper — the
  contribution is the systematic integration and experimental
  validation of GNSS uncertainty propagation through HMM map
  matching, not any single algorithmic component. This is a
  legitimate contribution pattern for T-ITS. The current
  framing (claiming novelty in each component) will irritate
  reviewers who know the GNSS literature.

  Recommendation: Restructure the introduction to emphasize
  the GAP (no prior work integrates covariance+PL+integrity
  into HMM map matching) and the SYSTEM contribution (unified
  framework with calibration analysis), rather than claiming
  novelty in individual components.

  ---
  2. Mathematical Rigor — Issues Found
  
  2.1 🚨 Critical: PL Formula Mismatch

  The paper's Eq. (7) describes the ARAIM MHSS iterative PL
  computation with solution separation terms $T\Delta_i$,
  fault hypotheses $P_{H_i}$, and unmonitored probability
  $P_{NM}$. However, the code (compute_raim_pl.py and the C++
  simulation) implements simple RAIM: $\text{HPL} = K \cdot
  \sigma_{\text{major}}$, where $\sigma_{\text{major}}$ is the
  semi-major axis of the error ellipse.

  This is a significant disconnect between the paper and the
  implementation. Reviewers from the GNSS integrity community
  will catch this immediately. You must either: (a) implement
  ARAIM MHSS and verify the paper's claims, or (b) rewrite
  Section IV-A.2 to describe RAIM honestly and note ARAIM as
  future work.

  Currently, the paper claims both RAIM and ARAIM are used
  ("Two major algorithms exist...") but the code only uses
  one. The Stanford plots use RAIM-derived PLs, so all the PL
  coverage claims in Table II are based on RAIM, not ARAIM.

  2.2 🚨 Critical: HPL as Search Radius — Theoretical 
  Justification Gap

  The paper defines the candidate search radius as $r_i =
  \text{HPL}_i$ (Eq. 8). The justification is: "the HPL
  guarantees that the true horizontal position error does not
  exceed $r_i$ with the specified integrity risk."

  This is logically flawed. HPL is a STATISTICAL BOUND on the
  position error under fault-free or single-fault conditions,
  not a GUARANTEE that the true position lies within the HPL
  disk. Specifically:
  - Under nominal conditions (no fault), HPL $\gg$ actual
  error (e.g., at $\sigma=1$m, HPL=14.4m while mean error=1.9m
  — Table II). The HPL is conservative by design.
  - Using HPL as search radius means: in good conditions, you
  search a 14m radius when the error is ~2m. This is
  unnecessarily wide and may include irrelevant road segments.
  - In bad conditions (fault present), HPL can underestimate
  the actual error (the missed detection probability
  $\text{P}_{\text{md}}$ is non-zero).
  
  The practical consequence: The candidate search is
  conservative (wide) under nominal conditions, which is
  actually fine for accuracy (better to include the true edge
  than miss it). But the paper's THEORETICAL justification is
  wrong. The paper should state honestly:

  ▎ "We use HPL as an ADAPTIVE, CONSERVATIVE search radius 
  ▎ that widens under degraded geometry and tightens under 
  ▎ good geometry. While HPL does not guarantee coverage of 
  ▎ the true position, it provides a principled, 
  ▎ parameter-free mechanism for adapting the search region to
  ▎ GNSS quality."

  2.3 Important: Forward Algorithm Degeneracy Claim

  The paper claims (Section IV-C.2):

  ▎ "The joint path posterior $\delta_t / \sum \alpha_t$ 
  ▎ decays exponentially with trajectory length... the forward
  ▎ denominator $\sum_j \alpha_t(j)$ accumulates probability 
  ▎ mass from all competing paths at a rate of approximately 
  ▎ $\log K$ per epoch relative to the Viterbi numerator
  ▎ $\delta_t$."

  This claim is mathematically suspect. In a properly
  row-normalized HMM where $\sum_j a(i,j) = 1$ for all $i$:
  - $\sum_j \alpha_t(j) = \sum_j [\sum_i \alpha_{t-1}(i)
  a(i,j)] e_t(j)$
  - With uniform emission (all $e_t(j) = 1/K$), $\sum_j
  \alpha_t(j) = \sum_i \alpha_{t-1}(i) \sum_j a(i,j) / K =
  \sum_i \alpha_{t-1}(i) / K$
  
  So $\sum \alpha_t$ actually SHRINKS by factor $K$ per epoch,
  not grows. The observed decay of $\delta_t / \sum \alpha_t$
  in the code is caused by:
  1. The emission probabilities are NOT properly normalized
  (they are probability densities, not probabilities)
  2. The background state absorbs mass from the real
  candidates
  3. Numerical underflow in log-space conversions

  The paper should either correct this claim or provide a
  rigorous derivation of the decay rate. RECORDS.md actually
  acknowledges that the TW value approaches zero because
  "$\delta_t \ll \alpha_t$" but this is due to the gap between
  Viterbi max and forward sum, not exponential growth of
  $\sum \alpha_t$.

  2.4 Moderate: Background State Justification

  The background state with fixed $p_{\text{bg}} = 0.1$ (Eq. 
  9) is described as representing the "off-road hypothesis."
  The theoretical justification is thin:
  - Why 0.1? This is an engineering parameter with no
  principled derivation.
  - The background state is a form of Laplace smoothing /
  add-one smoothing. The paper should connect to this
  established technique. 
  - The background state breaks the proper probabilistic
  semantics of the emission model — the Gaussian density
  integrates to 1 over $\mathbb{R}^2$, but the discrete set of
  road candidates + background state creates a partially
  discrete, partially continuous mixture model.

  Recommendation: Acknowledge the background state as a
  regularization technique (analogous to Laplace smoothing in
  Naive Bayes), not as a properly motivated probabilistic
  component. Discuss its effect on calibration in the ablation
  study.

  2.5 Minor: Emission Probability Density vs. Probability Mass

  Equation (13) gives a bivariate Gaussian density, which
  integrates to 1 over $\mathbb{R}^2$. However, map matching
  evaluates this density at discrete candidate points. The
  density values are NOT probabilities — they can exceed 1
  (when $|\Sigma|$ is small) and their sum over candidates
  does NOT equal 1.

  The paper correctly uses log-probabilities throughout and
  applies softmax normalization, so the PRACTICAL
  implementation is sound. But the paper should explicitly
  state that Eq. (13) gives the LIKELIHOOD (density), not the
  EMISSION PROBABILITY, and that proper normalization occurs
  via the forward algorithm and layer-wise softmax. This
  distinction matters for mathematical rigor.

  2.6 Minor: Transition Probability Model

  The transition model $w_{a \to b} = \exp(-|d_{\text{road}} -
  d_{\text{gnss}}| / \beta)$ (Eq. 10) is the standard FMM
  formulation. However, the parameter $\beta$ is never defined
  or given a value in the paper. What is $\beta$? Is it
  tuned? Is it derived from something?

  The code uses $\beta = 1.0$ by default (from the FMM
  inheritance). This should be stated.

  ---
  3. Experimental Validation — Assessment
  
  3.1 Strengths

  1. Dual validation (simulation + real data). Having both
  controlled Monte Carlo experiments AND real-vehicle
  validation with RTK ground truth is a significant strength.
  The synthetic experiments establish internal validity; the
  real experiments establish external validity.
  2. RTK ground truth at 16,155 epochs. Hand-labeled ground
  truth on this scale is costly and rigorous. The per-epoch
  segment-level evaluation is much stronger than the
  trajectory-level evaluation common in map matching papers.
  3. Diverse experimental dimensions: sigma sweep (7 levels),
  sample rate sweep (4 rates), mismatch analysis (5
  conditions), degradation modes (4 conditions at
  $\sigma=30$m). The coverage is comprehensive.
  4. Calibration analysis is novel in map matching. ECE +
  reliability diagrams + Brier score + ROC/AUC — this battery
  of metrics is standard in ML but unprecedented in map
  matching evaluation.
  5. Honest reporting of failure cases. The Traj 22
  wrong-direction false lock (Section VI-G) is discussed
  openly. This builds reviewer trust.

  3.2 Weaknesses

  3.2.1 🚨 Critical: Only 7 Trajectories, One City, One 
  Receiver

  For IEEE T-ITS, 7 trajectories from a single city (Haikou)
  with a single receiver model (Tersus BX50C) is thin. Typical
  T-ITS map matching papers use:
  - Multiple cities (e.g., Newson & Krumm 2009: Seattle)
  - Multiple receiver types (smartphone, automotive, high-end)
  - Larger datasets (Global Positioning Data from Didi,
  OpenStreetMap traces, etc.)
  
  The paper is vulnerable to the criticism: "These results may
  only hold for Haikou's road network geometry and the BX50C 
  receiver's covariance characteristics."

  Mitigation: The paper MUST acknowledge this as a limitation.
  The synthetic experiments partially mitigate this concern
  (they are not specific to Haikou), but the real-vehicle
  validation is narrow. If possible, add a second city or a
  second receiver type before submission.

  3.2.2 🚨 Critical: No Statistical Significance Tests

  None of the reported numbers include confidence intervals or
  hypothesis tests. For example:
  - CMM 91.4% vs FMM 88.1% — is this difference statistically
  significant? A McNemar's test on matched pairs would answer
  this.
  - ECE 0.078 vs 0.107 — is this difference significant?
  Bootstrap confidence intervals are needed.
  - The entire Table III (sigma sweep) reports point estimates
  with no error bars.
  
  This is a major gap. At minimum, add:
  1. Bootstrap 95% CI for all ECE/accuracy/AUC values
  2. McNemar's test or paired bootstrap for CMM vs FMM
  accuracy differences
  3. Error bars on the reliability diagram (Fig. 7)
  4. DeLong test for AUC comparison

  3.2.3 Important: FMM AUC > CMM AUC — Incomplete Defense

  The paper argues that FMM's AUC=0.965 is an "artifact of TW
  compression" (Section VI-G). This is plausible but the
  defense is incomplete:

  1. The paper shows TW separation (CMM 0.329 vs FMM 0.085) as
  evidence of CMM's superiority. But separation =
  $\mu_{\text{correct}} - \mu_{\text{wrong}}$ is
  scale-dependent — FMM could have high AUC AND low separation
  if the TW distributions have different shapes.
  2. The threshold-based analysis (CMM rejects 43% of wrong
  matches at threshold 0.9) is more convincing but needs to be
  presented as a TABLE, not just text.
  3. The paper should compute partial AUC (pAUC) in the
  practically relevant FPR range (e.g., FPR < 0.3), where CMM
  likely outperforms FMM.
  4. The paper should acknowledge that CMM's AUC=0.764 is 
  modest in absolute terms. In ML, AUC < 0.8 is considered
  "fair" discrimination. This is acceptable for a
  first-attempt trustworthiness metric but should be discussed
  honestly.

  3.2.4 Important: Missing Ablation Study in Main Text

  Section VI-E (Ablation Study) in the paper is a single
  paragraph with one row: FMM ECE=0.107 vs CMM ECE=0.078. The
  RECORDS.md shows that the code contains individual fixes (TP
  normalization, background state, uniform prior,
  Viterbi-forward separation) whose individual contributions
  ARE measurable (the accuracy changed from 94.8%→91.4% after
  the math refactor).

  The paper MUST include a proper multi-row ablation table
  showing:

  Configuration: FMM baseline
  Accuracy: 88.1% 
  ECE: 0.107      
  AUC: 0.965
  TW Separation: 0.085
  ────────────────────────────────────────
  Configuration: CMM: isotropic EP + norm TP + uniform prior
  Accuracy: ? 
  ECE: ?
  AUC: ?
  TW Separation: ?
  ────────────────────────────────────────
  Configuration: CMM: Mahalanobis EP + unnorm TP + EP prior
  Accuracy: ?
  ECE: ? 
  AUC: ?
  TW Separation: ?
  ────────────────────────────────────────
  Configuration: CMM: Mahalanobis EP + norm TP + uniform prior

    (lag=0)
  Accuracy: 91.4%
  ECE: 0.072
  AUC: 0.764
  TW Separation: 0.329
  ────────────────────────────────────────
  Configuration: CMM: + lag smoothing
  Accuracy: ?
  ECE: ?
  AUC: ?
  TW Separation: ?

  This would QUANTIFY the per-component contribution — which
  is exactly what the paper claims as a contribution
  ("decompose the calibration improvement via ablation
  studies").

  3.2.5 Minor Issues

  - Lag sweep shows L=0 is optimal on real data ($5.4). This 
  is a NEGATIVE result (smoothing doesn't help) that should be
  discussed: why doesn't future evidence improve calibration?
  Possible answer: RAIM PL already provides sufficient
  geometric constraint. But this contradicts the simulation
  where L=5 was used.
  - Synthetic data uses 8 satellites at 45–80° elevation —
  this is unrealistically clean. No low-elevation satellites,
  no multipath simulation beyond the simple occlusion model.
  The paper should discuss this limitation.
  - FMM configuration may be suboptimal: FMM uses fixed
  $r=0.03^\circ$ (~3km at equator) search radius. This is VERY
  wide and likely includes many irrelevant candidates,
  disadvantaging FMM unfairly. A fairer comparison would tune
  FMM's radius to match CMM's average candidate count.

  ---
  4. Paper Structure and Writing Quality
  
  4.1 Strengths

  - Clear logical flow: problem → classical HMM → limitations
  → proposed method → simulation → real experiment
  - Good use of figures to illustrate concepts (HMM trellis,
  covariance ellipse, Mahalanobis vs Euclidean projection)
  - Algorithm pseudocode (Algorithm 1) is comprehensive and
  matches the code
  - Honest discussion of limitations and failure cases

  4.2 Weaknesses

  4.2.1 Related Work Section Is Unfocused

  Section II (3 pages) lists ~35 papers across HMM map
  matching, GNSS integrity, and calibration. It reads as an
  annotated bibliography, not a critical analysis. A reviewer
  will ask: "What is the GAP that THIS paper fills?"

  Recommendation: Restructure related work around GAPS, not
  topics:
  1. Gap 1: HMM map matching ignores GNSS covariance → all use
  isotropic emission
  2. Gap 2: GNSS integrity concepts (PL, integrity risk) have
  never been integrated into map matching
  3. Gap 3: Map matching confidence scores have never been
  evaluated for probabilistic calibration
  
  Then cite papers as evidence of these gaps, not as a survey
  of the field.

  4.2.2 WLS Derivation Is Too Long

  Section IV-A.1 (WLS covariance) is ~50 lines of standard
  GNSS textbook material. It can be compressed to:

  ▎ "The GNSS position and its $4\times4$ covariance matrix 
  ▎ are obtained via weighted least-squares: $\hat{\mathbf{x}}
  ▎ = (\mathbf{H}^T\mathbf{W}\mathbf{H})^{-1}\mathbf{H}^T\mat
  ▎ hbf{W}\mathbf{y}$, with $\mathbf{\Sigma}_x = 
  ▎ (\mathbf{H}^T\mathbf{W}\mathbf{H})^{-1}$. The $2\times2$
  ▎ horizontal block $\mathbf{\Sigma}_i$ is extracted for the
  ▎ emission model."

  This saves ~30 lines for more important content.

  4.2.3 Missing Formal Definition of Trustworthiness

  The term "trustworthiness" appears in the abstract but is
  not formally defined until Section IV-C.2 (page 8). The
  paper would benefit from an early, crisp definition:
  "Trustworthiness is the per-epoch filtering posterior
  probability $P(x_t = i^* \mid z_{1:t})$ that the
  Viterbi-optimal road edge is correct given all observations
  up to epoch $t$."

  4.2.4 English and Minor Errors

  - "researchen" (line 359) → "researched"? This appears to be
  a typo.
  - "procedures" → "procedure" or "proceeds"
  - Some sentences are very long (50+ words) and hard to parse
  - The abstract mentions "0.078" but the conclusion says
  "0.072" — this inconsistency needs to be resolved (the code
  shows 0.072)

  4.2.5 References Need Updating

  The bibliography has 41 entries, which is adequate for
  T-ITS. However:
  - Some recent 2024–2025 map matching papers are cited (Guo
  2025, Wang 2024) but others may be missing
  - The paper should cite work on uncertainty-aware map 
  matching specifically — what prior work exists on this
  topic?
  - Lahrech & Soulhi (2023) used Mahalanobis distance for map
  matching — this is the closest prior work and needs more
  discussion

  ---
  5. Code Quality Assessment
  
  5.1 Strengths

  - Clean separation of concerns: CMM algorithm
  (cmm_algorithm.cpp), configuration (cmm_config), and types
  (mm_type.hpp) are well-organized
  - Log-space computation throughout: Prevents numerical
  underflow
  - OpenMP parallelization: Batch processing is efficient
  - Comprehensive configuration: 18 XML parameters with
  validation
  - Detailed RECORDS.md: Excellent development diary
  documenting every bug fix and design decision

  5.2 Issues

  - 2200+ line implementation file: cmm_algorithm.cpp at 2782
  lines is very long. Consider splitting: candidate search
  (~400 lines), emission computation (~100 lines), layer
  update (~300 lines), lag smoothing (~150 lines) into
  separate files
  - H0 lambda mechanism is partially commented out with #if 0
  — either complete it or remove it
  - The C++ Mahalanobis projection (line ~950 in
  cmm_algorithm.cpp) uses a brute-force search over edge
  segments — this could be optimized with analytical solutions
  for line segments
  - No unit tests for CMM-specific functions (Mahalanobis
  projection, covariance propagation, softmax normalization).
  The test suite only covers geometry and network operations.

  ---
  6. Specific Suggestions — Priority-Ordered
  
  P0: Must Fix Before Submission

  1. Fix PL formula description: Either implement ARAIM or
  rewrite Section IV-A.2 to describe RAIM only. The current
  text describes ARAIM but the code uses RAIM.
  2. Add statistical significance tests: Bootstrap CIs for all
  key metrics, McNemar's test for accuracy, DeLong test for
  AUC comparison.
  3. Fix the "exponential decay" claim (§IV-C.2): The forward
  denominator $\sum \alpha_t$ does NOT grow exponentially in a
  properly normalized HMM. Revise the explanation of why the
  joint posterior decays.
  4. Add proper ablation table (§VI-E): Show per-component
  contributions to accuracy/ECE/AUC. This is claimed as a
  contribution but not delivered in the current draft.
  5. Resolve ECE inconsistency: Abstract says 0.078,
  conclusion says 0.072. Pick one (the code shows 0.072).
  6. Add honest limitations paragraph: Single city, single
  receiver, 7 trajectories, synthetic data limitations (8
  satellites, 45–80° elevation).

  P1: Should Fix Before Submission

  7. Restructure Related Work around gaps, not topics.
  8. Compress WLS derivation to 5 lines with a citation.
  9. Define "trustworthiness" formally in Section I
  (introduction).
  10. Add partial AUC analysis to defend against FMM's
  inflated AUC.
  11. Discuss FMM configuration fairness: Is FMM's
  $r=0.03^\circ$ giving it an unfair disadvantage? 
  12. Connect background state to Laplace smoothing
  literature.
  13. Expand Traj 22 failure analysis: The RECORDS.md
  deep-dive into the Viterbi dead-end at seq 1676 is excellent
  — include a condensed version in the paper.

  P2: If Time Permits

  14. Add a second validation dataset (another city or
  receiver).
  15. Sweep background_prob values {0.01, 0.05, 0.1, 0.2} and
  report sensitivity.
  16. Add entropy-based analysis: The paper defines $\Delta
  H_t$ but doesn't use it experimentally. Does $\Delta H_t$
  predict mismatches better than trustworthiness?
  17. Profile runtime: What is the computational overhead of
  Mahalanobis vs Euclidean emission?
  18. Refactor cmm_algorithm.cpp into smaller files.

  ---
  7. Answers to Your Three Questions
  
  Q1: Is the essay novel enough for IEEE T-ITS or TVT?

  Yes, borderline. The systematic integration of GNSS
  covariance+PL+integrity into HMM map matching is novel. The
  calibration analysis of map matching confidence scores is
  genuinely new. For T-ITS, the contribution level is adequate
  IF the paper is repositioned as a systems/integration paper
  rather than claiming novelty in each component. For TVT
  (which is more GNSS-focused), the GNSS community may find
  the HMM map matching application less central.

  My recommendation: Target T-ITS and position it as bridging
  GNSS integrity and map matching — a contribution that
  neither community has made.

  Q2: Is the work solid enough and theory valid in math 
  derivation?

  Mostly yes, with three caveats that MUST be fixed:

  1. The PL formula description (ARAIM vs RAIM) is wrong.
  2. The "exponential decay" claim about the forward
  denominator needs correction.
  3. The HPL-as-search-radius theoretical justification is
  flawed (the PRACTICE is fine, but the THEORY section needs 
  rewriting).

  The core HMM derivation (forward algorithm, filtering
  posterior, softmax normalization) is mathematically correct
  and well-implemented in the code.

  Q3: Are the experiments enough to prove the statements and 
  convince reviewers?

  Partially. The strengths are real: dual simulation+real
  validation, RTK ground truth, comprehensive metric suite,
  honest failure case reporting.

  The gaps are: (1) no statistical significance tests — this
  alone could trigger rejection from a statistically-minded
  reviewer, (2) the FMM AUC paradox is not fully resolved —
  you need stronger evidence that CMM's TW is genuinely better
  despite lower AUC, (3) the ablation study in the paper is a
  single sentence — it doesn't deliver on the claimed
  contribution of "decomposing ECE improvement by component."

  With these three gaps fixed, the experimental section would
  be strong enough for T-ITS.

  ---
  8. Suggested Follow-up Prompts
  
  1. "Help me rewrite Section II (Related Work) around the 
  three gaps: (1) isotropic emission in HMM map matching, (2) 
  GNSS integrity never integrated into map matching, (3) no 
  calibration analysis of map matching confidence scores."
  2. "Help me design and run the ablation study: measure 
  per-component contribution of TP normalization, background 
  state, uniform prior, and Mahalanobis emission to ECE, 
  accuracy, and AUC."
  3. "Review my PL Stanford plot code (exp2_stanford_pl.py) 
  and verify that the RAIM HPL computation is correct against 
  the paper's Eq. (7)."