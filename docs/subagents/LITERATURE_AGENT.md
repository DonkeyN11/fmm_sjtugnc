# Literature Search & Validation Agent — Knowledge Transfer File

**Created**: 2026-06-07
**Last updated**: 2026-06-07 (enhanced with full 6-step methodology, safety protocols, code-to-paper mapping, domain notation, CNKI skill inventory, ARS plugin modes, cross-agent collaboration)
**Source sessions**: ARS lit-review + CNKI-search debugging + ars-plan (Socratic chapter planning) + research-agent-design (6-step methodology formalization) + memory-organization (global/project CLAUDE.md split)
**Purpose**: Enable a new chat session to resume literature search, citation validation, research area mapping, and annotated bibliography generation for the CMM paper without re-discovering the search strategy, paper taxonomy, or existing reference state.

## Companion Files (Read These First)

| File | Content | When to Read |
|------|---------|-------------|
| `~/.claude/CLAUDE.md` | Global memory: user identity, safety rules, full 6-step methodology (domain-agnostic) | Every session — loaded automatically |
| `CLAUDE.md` (project root) | Project memory: C++ standards, build system, CMM architecture, experiment inventory | When working with code or experiments |
| `docs/research_agent_prompt.md` | Concise invocable research agent prompt (~120 lines) | Quick activation of any research step |
| `docs/specialized_research_agent.md` | Full research agent reference with CMM-specific templates (543 lines) | Deep-dive research work |
| `docs/subagents/LITERATURE_AGENT.md` | This file — literature-specific knowledge transfer | Literature search, citation validation, taxonomy work |

---

## 1. What This Agent Does (That Others Can't)

This agent specializes in the *scholarly landscape* surrounding the CMM paper. No other subagent in this folder performs literature tasks:

| Capability | Other Subagents | This Agent |
|------------|:---:|:---:|
| Search for papers across Web, Google Scholar, CNKI, IEEE Xplore | No | **Yes** |
| Validate citations: verify DOI, journal, year, author names, page numbers | No | **Yes** |
| Classify papers into the 3-stream taxonomy (HMM-MM / GNSS Integrity / Calibration) | No | **Yes** |
| Identify research camps, contested questions, and gaps per the 6-step methodology | No | **Yes** |
| Produce annotated bibliographies with 1-sentence contribution, method, and gap per paper | No | **Yes** |
| Write and update the LaTeX `\section{Related Work}` with proper IEEE citations | Plot subagent can edit LaTeX | **Yes** |
| Maintain `references.bib` with consistent keys, verified DOIs, and IEEE formatting | No | **Yes** |
| Summarize current development in map matching, GNSS integrity, and probabilistic calibration | No | **Yes** |
| Flag outdated or suspicious references (e.g., broken DOIs, preprints passed off as published) | No | **Yes** |
| Search CNKI (知网) for Chinese-language map matching and GNSS literature | No | **Yes** |
| Recommend which papers to cite vs. which to drop based on venue quality and relevance | No | **Yes** |

### 1.1 Relationship to the 6-Step Research Agent

This literature agent implements **Steps 1–3** of the full 6-step methodology (defined in `~/.claude/CLAUDE.md` and `docs/specialized_research_agent.md`):

| Step | This Agent's Implementation |
|------|---------------------------|
| **Step 1 — Draw the Map** | 3-stream taxonomy (§2), concept mapping, core paper identification |
| **Step 2 — Find Relevant Literature** | Citation tracing, camp identification, contested question mapping (§4) |
| **Step 3 — Collect Neighboring Areas** | Cross-disciplinary search (CV, robotics, NLP, geostatistics, ML, sensor fusion → map matching) |

For **Steps 4–6** (Question & Analyze, Write Fluently, Review & Polish), activate the full research agent via `docs/research_agent_prompt.md` or use the ARS plugin's `/ars-plan`, `/ars-revision`, and `academic-paper-reviewer` skills.

## 2. The Research Taxonomy (3-Stream Framework)

The CMM paper sits at the intersection of three streams. Every paper this agent finds should be classified accordingly:

### Stream A: HMM-Based Map Matching
**Core question**: How to infer vehicle road position from noisy GNSS observations using probabilistic graphical models.

**Key camps**:
- **Classical HMM** — Newson & Krumm 2009, Yang & Gidófalvi 2018 (FMM/UBODT)
- **Online/real-time HMM** — Goh 2012 (VSW), Jagadeesh & Srikanthan 2017 (route choice), Taguchi 2019 (route prediction), Barefoot
- **Higher-order / hybrid models** — Fu 2021 (2nd-order HMM), Guo 2025 (adaptive 2nd-order), Li 2024 (HMM-CRF)
- **Alternative paradigms** — Wöltche 2023 (MDP+RL), Duffield 2020 (particle smoothing), Kempinska 2017 (PST-Matching)
- **Surveys & benchmarks** — Quddus 2007, Hashemi & Karimi 2014, Chao 2020, Singh 2023, Chen 2025

**Unresolved gap (CMM's contribution)**: ALL methods assume isotropic Gaussian emission with fixed scalar σ. No method uses the full 2×2 GNSS covariance matrix.

### Stream B: GNSS Integrity & Protection Levels
**Core question**: How to bound positioning error with a guaranteed risk probability and detect faulty measurements.

**Key camps**:
- **Classical RAIM/ARAIM** — Zhang 2018 (M-estimation RAIM), Zhang 2019 (MHSS thresholds), solution separation methods
- **Automotive integrity** — Maaref & Kassas 2022 (cellular SOP + IMU), Teng 2023 (5G+GNSS), Elsayed 2024 (PPP-RTK)
- **Integrity-constrained optimization** — Xia 2023 (FGO with switchable constraints), Al Hage 2023 (Student's t filter)
- **Protection Level theory** — Lee 2023 (IMU fault PL), Kim 2019 (PL for vehicle localization)
- **Surveys** — Maharmeh 2024 (73.3% GNSS-centric; 26.7% non-GNSS)
- **Stochastic modeling** — Zhang 2022 J.Geodesy (composite elevation-C/N₀ model), Li 2022 MST (C/N₀ template functions)

**Unresolved gap (CMM's contribution)**: PL/PHMI concepts have never been integrated *within* HMM emission probability — PL is used as a binary gate, not as a probabilistic weighting factor.

### Stream C: Probabilistic Calibration & Trustworthiness
**Core question**: Do confidence scores reflect true correctness rates, and can we detect when the system is wrong?

**Key camps**:
- **Calibration metrics** — Guo 2017 (ECE, reliability diagrams), Nixon 2019 (ACE, GCE, adaptive binning), Roelofs 2024 (proper scoring rules)
- **Map matching reliability** — Quddus 2006 (empirical integrity metric 0–100, 98.2% valid warnings)
- **Confidence in localization** — Wang 2023 PMHT (PDA weights + map feature variability), Neamati 2023 (zonotope risk bounds)
- **Sensor fusion integrity** — Bai 2023 (GVIM FGO), Lu 2024 (GNSS/IMU/Camera/HD Map EKF)
- **Bayesian verification** — Phillips 2016 (aggregating weak individual measurements for strong verification)

**Unresolved gap (CMM's contribution)**: No prior work evaluates HMM map matching output through ECE, reliability diagrams, or ablation-decomposed calibration analysis.

## 3. Current Reference State (as of 2026-06-07)

### `references.bib` — 43 entries
**File**: `docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/references.bib`

**Stream A (HMM Map Matching)**: 26 entries — `HMM`, `FMM`, `Quddus2007`, `Quddus2006`, `Hashemi2014`, `Hashemi2016`, `Jagadeesh2017`, `Taguchi2019`, `Goh2012`, `Woltche2023`, `Wang2024`, `Li2023`, `Feng2023`, `Qu2023`, `Kempinska2017`, `Duffield2020`, `Fu2021`, `Guo2025_HMM`, `Li2024_IJDE`, `Huang2024`, `Liu2025_LiMM`, `Chao2020`, `Singh2023`, `Hansson2021`, `Lahrech2023`, `Chen2025_Survey` (if present)

**Stream B (GNSS Integrity)**: 12 entries — `Zhang2018`, `Zhang2019`, `Maaref2022`, `AlHage2023`, `Maharmeh2024`, `Elsayed2024`, `Lee2023`, `Xia2023`, `Neamati2023`, `Zhang2022_Geodesy`, `Kim2019`, `Park2024`

**Stream C (Calibration & Trustworthiness)**: 5 entries — `Guo2017`, `Nixon2019`, `Roelofs2024`, `Wang2023_PMHT`, `Phillips2016`

### LaTeX Related Work Section — ~200 lines, 42+ citations
**File**: `docs/Trustworthiness Evaluation Framework for Map Matching based on Covariance Ellipse/Trustworthiness Evaluation Framework.tex`
**Location**: Lines 147–347 (`\section{Related Work}`)

Each subsection (A/B/C) ends with a **gap paragraph** that:
1. Acknowledges what the literature has achieved
2. Identifies the specific limitation shared by all prior work
3. States how CMM addresses that gap

## 4. Search Methodology

When searching for new papers, use this multi-pass strategy:

### Pass 1 — Broad Web Search (3 parallel queries per stream)
```
Stream A: "HMM map matching [variant] survey 2023 2024 IEEE T-ITS"
Stream B: "GNSS integrity [RAIM|ARAIM|PL] autonomous vehicle 2022 2023 2024"
Stream C: "calibration [ECE|reliability diagram] [localization|map matching] confidence 2023"
```

### Pass 2 — Targeted Deep Dive
For each high-value paper found in Pass 1:
- Forward citation search (who cites this?)
- Backward citation search (what foundational work does this cite?)
- Author follow-up (what else has this group published?)

### Pass 3 — Validation
Before adding any paper to `references.bib`, verify:
- [ ] DOI resolves to the actual paper
- [ ] Journal/conference name is correct (check publisher website)
- [ ] Year, volume, pages match the published version
- [ ] Author names are spelled correctly (check diacritics: Gidófalvi, Wöltche)
- [ ] The paper actually exists (preprints are OK if labeled as such)
- [ ] Venue quality is appropriate for IEEE T-ITS citation

## 5. Key Knowledge About the CMM Paper

### What the paper claims (verify new citations support these):
1. **Anisotropic > isotropic**: Full-covariance Mahalanobis emission dramatically improves calibration (ECE 0.978 → 0.121)
2. **Trustworthiness as diagnostic**: Softmax smoothing posterior is a calibrated mismatch detector (AUC ≈ 0.72)
3. **Integrity integration**: RAIM-derived HPL guides candidate search; PHMI risk bound weights emission

### What the paper does NOT claim (avoid overstating):
- CMM is not the first to use Mahalanobis distance in localization (Lahrech 2023 uses it in PF map matching)
- CMM is not the first to do confidence scoring in map matching (Quddus 2006, Kempinska 2017)
- CMM does not achieve lane-level accuracy (it's road-segment-level, ~3m median error)

### Competitor awareness (papers that a reviewer might ask about):
- **Barefoot** (Java, open-source) — real-time HMM with variable sliding window
- **Valhalla/Meili** — open-source HMM map matching used by Mapbox
- **LeuvenMapMatching** — Python HMM library with customizable distributions
- **GraphMM / DeepMM / L2MM** — deep learning alternatives to HMM
- **OSMnx + NetworkX** — Python ecosystem for road network analysis

## 6. Complete Search Tool Inventory

### 6.1 Built-in Claude Code Tools

| Tool | Status | Use for |
|------|--------|---------|
| `WebSearch` | **Working** | Broad topic search, Google Scholar queries, paper discovery |
| `WebFetch` | **Working** | Fetch specific URLs, resolve DOIs, check publisher pages |

### 6.2 CNKI (知网) MCP Skills

**CRITICAL UPDATE (2026-06-07):** CNKI skills are now registered as MCP tools in the current session. However, they depend on browser automation and may still encounter the Tencent slider captcha on `kns.cnki.net`. Test before relying on them.

| Skill | Status | Use for |
|-------|--------|---------|
| `cnki-search` | Registered (may hit captcha) | Search CNKI by keyword |
| `cnki-advanced-search` | Registered (may hit captcha) | Filtered search by author, title, journal, date, source category (SCI/EI/CSSCI/北大核心) |
| `cnki-parse-results` | Registered | Parse current search results page into structured data |
| `cnki-paper-detail` | Registered | Extract full paper details (title, authors, abstract, keywords, fund, classification) |
| `cnki-journal-search` | Registered | Find journals by name, ISSN, CN, or sponsor |
| `cnki-journal-index` | Registered | Check journal indexing status (北大核心, CSSCI, CSCD, SCI, EI), impact factors |
| `cnki-journal-toc` | Registered | Browse journal issues, table of contents |
| `cnki-download` | Registered | Download paper PDF/CAJ (requires login) |
| `cnki-export` | Registered | Export to Zotero or save as RIS file |
| `cnki-navigate-pages` | Registered | Navigate search result pages, change sort order |

**Captcha workaround**: If CNKI browser skills fail, fall back to `WebSearch` with `site:cnki.net` or Google Scholar queries. For critical CNKI-only papers, recommend the user search manually from their local browser (MobaXterm has a built-in browser).

**Chinese-language search keywords**:
```
地图匹配 + 隐马尔可夫 (map matching + HMM)
GNSS 完好性 + 保护水平 (GNSS integrity + protection level)
定位可靠性 + 可信度 (positioning reliability + trustworthiness)
协方差 + 误差椭圆 + 地图匹配 (covariance + error ellipse + map matching)
```

**Key Chinese journals**:
- 武汉大学学报·信息科学版 (Geomatics and Information Science of Wuhan Univ.)
- 测绘学报 (Acta Geodaetica et Cartographica Sinica)
- 中国惯性技术学报 (Journal of Chinese Inertial Technology)

### 6.3 Academic Research Skills (ARS) Plugin — v3.7.0+

The ARS plugin provides a full academic paper writing pipeline. All modes below are available as slash commands or skills.

**Paper Writing Pipeline:**

| Skill / Command | Mode | Use |
|-----------------|------|-----|
| `/ars-full` | Full pipeline | Research → write → review → revise → finalize |
| `/ars-plan` | Plan only | Socratic chapter-by-chapter planning with evidence map |
| `/ars-outline` | Outline only | Detailed outline + evidence map |
| `/ars-abstract` | Abstract only | Bilingual abstract + keywords |
| `/ars-lit-review` | Lit review | Annotated bibliography in paper format |
| `/ars-revision` | Revision | Revised draft + R&R responses |
| `/ars-revision-coach` | Revision coach | Revision Roadmap + Response Letter Skeleton |
| `/ars-format-convert` | Format convert | Convert to LaTeX / DOCX / PDF / Markdown |
| `/ars-citation-check` | Citation check | Citation error report |
| `/ars-disclosure` | Disclosure | Venue-specific AI-usage statement |

**Research & Review Pipeline:**

| Skill | Use |
|-------|-----|
| `deep-research` | 13-agent literature search + synthesis with APA 7.0 cited reports. 7 modes: full research, quick brief, paper review, lit-review, fact-check, Socratic guided, systematic review with optional meta-analysis |
| `academic-paper-reviewer` | 5-persona simulated peer review (EIC + 3 peer reviewers + Devil's Advocate). Modes: full review, re-review, quick assessment, methodology focus, Socratic guided, calibration |
| `academic-pipeline` | End-to-end orchestration: research → write → integrity check → review → revise → re-review → finalize. 10-stage workflow with quality gates |

**ARS Plugin Agents:**
- `synthesis_agent` — synthesizes findings across sources
- `research_architect_agent` — designs research methodology
- `report_compiler_agent` — compiles final APA 7.0 reports

### 6.4 MCP Browser Tools

| Tool | Status | Notes |
|------|--------|-------|
| `chrome-devtools-mcp` | Working (patched) | Chrome finds and launches via patched `browser.js`. Headless forced to `true`. |
| `playwright-mcp` | May not work | Not patched, may not find Chrome. Prefer chrome-devtools. |
| `microsoft-docs:microsoft-learn` | **Working** | Microsoft Learn documentation search + code samples + fetch |

**Known browser issues:**
- Both MCP servers ignore `mcpServers` manual config args; the Claude Code plugin system uses `npx ...@latest` independently
- `PUPPETEER_EXECUTABLE_PATH` and `CHROME_PATH` env vars are NOT respected by bundled `@puppeteer/browsers`
- `chrome-devtools-mcp` source patch at `/home/ncz/.npm/_npx/15c61037b1978c83/node_modules/chrome-devtools-mcp/build/src/browser.js` may be overwritten on `npx` cache refresh
- X11 forwarding broken on this server (SSH key restrictions despite `X11Forwarding yes` in sshd_config)

### 6.5 Other Plugins

| Plugin | Status | Use |
|--------|--------|-----|
| `microsoft-docs` | **Working** | Microsoft Learn documentation search, code samples, fetch |
| `github` MCP | **Disabled** | Needs `GITHUB_PERSONAL_ACCESS_TOKEN` env var in `~/.bashrc` to enable |

## 6a. CMM Paper Revision Plan (ARS Plan Mode Output, 2026-06-07)

During the `ars-plan` session, three experiment gaps were identified and a full re-run of all quantitative experiments was agreed upon because the PHMI multiplier changed from 5 (old, cherry-picked) to 1 (principled: HPL directly defines the classification boundary).

### Root Cause: PHMI multiplier=5 → multiplier=1

The `phmi_pl_multiplier` config parameter controls how HPL scales the candidate classification boundary for emission probability regularization. Old code used multiplier=5 as an engineering tuning parameter. New code classifies candidates by whether they fall within the HPL ellipse directly (multiplier=1), making the PHMI boundary theoretically grounded in RAIM integrity theory.

**Impact:** ALL quantitative results run with multiplier=5 are invalidated, including ECE, ablation, lag sweep, and ROC numbers.

### Experiment Re-run Plan (Priority Order)

| # | Experiment | Script | Location in paper | Status |
|---|-----------|--------|-------------------|--------|
| A1 | ECE/MCE/Brier recalibration | `exp1_reliability_diagram.py` | §VI-C | To re-run with multiplier=1 |
| A2 | Ablation: FMM→aniso→+lag→+PHMI | `exp4_ablation_ece.py` | §VI-D | To re-run; PHMI row expected to show improvement now |
| A3 | Multi-traj lag sweep | `exp1_multitraj_lag_sweep.py` | §VI-E | To re-run; confirm L optimal |
| A4 | ROC: trustworthiness vs error | `evaluate_match_metrics.py` | §VI-F | To re-run; new AUC |
| A5 | Synthetic validation | `exp2_synthetic_validation.py` | §V | To re-run/confirm internal validity |

### New Experiments (Three Gaps)

| # | Gap | Design | Location | Description |
|---|-----|--------|----------|-------------|
| B1 | **Covariance sensitivity** | $\Sigma_{input} = \alpha \cdot \Sigma_{true}$, sweep $\alpha \in [0.1, 0.2, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0]$. Generate synthetic observations with true covariance, feed perturbed covariance to CMM, measure ECE vs $\alpha$ curve. Add FMM baseline as horizontal dashed line. | §V (Simulation) | Tests robustness to receiver covariance estimation error. Answers: "If the receiver under/over-estimates $\Sigma$, does calibration hold?" |
| B2 | **PL Stanford plot** | Horizontal axis = HPL, vertical axis = actual matching error. Color points by PDOP or visible satellite count. Use all 7 trajectories (16,155 epochs). | §VI (Real-vehicle) | Directly answers: "Does PL bound actual error at the claimed integrity risk?" First known connection of Stanford diagrams to map matching evaluation. |
| B3 | **PHMI multiplier rationale** | Not an experiment — rewrite §IV-D to explain why multiplier=1 is principled (HPL directly defines the RAIM integrity bound) vs multiplier=5 (empirical tuning). | §IV-D (Method) | Replaces the now-obsolete multiplier sweep analysis (Exp 3a). |

### Execution Order

```
1. Update cmm_config_omp.xml → phmi_pl_multiplier=1
2. Re-run A1-A5 (existing experiments with new config)
3. Run B1 (covariance sensitivity, simulation)
4. Run B2 (Stanford plot, real data)
5. Rewrite §IV-D PHMI derivation (B3)
6. Update all figures, tables, numbers in LaTeX
7. Compile and verify
```

### Key INSIGHTs from the Planning Session

1. **Multiplier=1 is principled**: multiplier=5 was engineering tuning; multiplier=1 makes HPL the direct RAIM integrity classification boundary
2. **Covariance sensitivity is a novel contribution**: No prior work has evaluated the impact of receiver covariance estimation error on map matching calibration
3. **Stanford plot bridges two fields**: GNSS integrity uses Stanford diagrams; map matching doesn't. This paper could be the first to connect them.
4. **PHMI classification = binary regularization**: Candidates inside/outside PL get different emission probability regularization coefficients (1-PHMI vs PHMI), which is prior injection into the probabilistic model, not merely search filtering
5. **The re-run makes the story stronger**: Old data had multiplier=5 cherry-picking vulnerability; new data has theoretical grounding for every parameter

## 7. Output Templates

### Annotated Bibliography Entry (per paper)
```markdown
### [Citation Key]
**Full citation**: Authors. "Title." *Journal*, vol, no, pp, year. DOI.
**Core contribution** (1 sentence):
**Method** (problem formulation, key assumption):
**Evidence** (dataset, evaluation, metrics):
**Gap left** (what this paper does NOT address, motivating CMM):
**Relevance to CMM** (Stream A/B/C, which subsection):
```

### Research Area Summary (for a stream)
```markdown
## Stream [A/B/C]: [Name]
### Current State (2024-2026)
[3-5 sentence synthesis of where the field stands]

### Major Camps
| Camp | Core Belief | Representative Papers | Blind Spot |
|------|------------|----------------------|------------|

### Contested Questions
1. [Question] — Camp X says A; Camp Y says B

### Gap → CMM Contribution
[How the present work fills the remaining gap]
```

## 8. Quick-Start for a New Session

To activate this agent in a new chat, tell Claude:

> "Adopt the literature search & validation agent persona defined in `docs/subagents/LITERATURE_AGENT.md`. First read the current state of `references.bib` and the Related Work section in the LaTeX article. Then help me [search for papers on X / validate these new citations / update the Related Work / summarize current developments in Y]."

Or more compactly:

> "Read `docs/subagents/LITERATURE_AGENT.md`. You are the literature agent. Find papers on [topic] and classify them into the 3-stream taxonomy."

## 9. Files This Agent Owns

| File | Content | This Agent's Responsibility |
|------|---------|---------------------------|
| `docs/.../references.bib` | 43 BibTeX entries | Add, validate, format, deduplicate entries |
| `docs/.../Trustworthiness Evaluation Framework.tex` (§II) | Related Work section, lines 147–347 | Write/update narrative, ensure all claims are cited |
| `docs/.../Trustworthiness Evaluation Framework.tex` (§V, §VI) | Simulation + Real-vehicle experiment sections | Update with re-run numbers after multiplier=1 experiment reset |
| `docs/.../ARTICLE_REFINEMENT_NOTES.md` | Comprehensive article status (sections, figures, results, TODOs) | Keep up to date; this is the canonical reference for article state |
| `docs/literatures/*.pdf` | 25 local PDFs | Track which are cited vs. uncited; recommend additions |
| `docs/subagents/LITERATURE_AGENT.md` | This file | Keep up to date as references and experiment status change |
| `input/config/cmm_config_omp.xml` | CMM runtime configuration | phmi_pl_multiplier must be set to 1 for all re-run experiments |
| `tests/python/exp1_*.py` through `exp4_*.py` | Experiment scripts | Know which script produces which result; verify before reporting numbers |
| `src/mm/cmm/cmm_algorithm.cpp` | CMM implementation (~2200+ lines) | Cross-reference article claims with actual code behavior |

## 10. Known Issues & Watchlist

### Experiment Status (multiplier=1 re-run)
- [ ] **ALL quantitative results need re-running** with `phmi_pl_multiplier=1` (see §6a Revision Plan)
- [ ] Config XML still has `<phmi_pl_multiplier>5</phmi_pl_multiplier>` — must change to `1` before any re-run
- [ ] Expected outcome: PHMI ablation row will show improvement (not +0.000 as before)
- [ ] Expected outcome: ECE may improve further with multiplier=1 (more epochs with good/bad candidate discrimination)

### Reference Quality
- [ ] Some DOIs in `references.bib` may be inaccurate (auto-generated during search). Validate against doi.org.
- [ ] `Maharmeh2024` is labeled as a preprint — verify if it has been published in IEEE Access yet.
- [ ] `Li2023` (improved HMM) has incomplete journal info — find correct volume/pages/DOI.
- [ ] `Zhang2018` and `Zhang2019` are in *Mathematical Problems in Engineering* (low-tier journal) — consider replacing with stronger RAIM/ARAIM references from *GPS Solutions*, *NAVIGATION*, or *IEEE T-AES*.
- [ ] 25 local PDFs in `docs/literatures/` — 5 cited, ~20 uncited. Worth a review pass for additional relevant work.
- [ ] The papers `ERNet.pdf`, `quantum map matching.pdf`, `DeepMM...pdf`, and `Transformer-based matching method.pdf` represent deep-learning approaches — could be cited as contrasting paradigms in a "Beyond HMM" paragraph.
- [ ] Chinese-language papers from the 25 PDFs (e.g., lane-level HMM) are untapped — may contain relevant work not in English indices.

### Tool Infrastructure
- [ ] CNKI captcha blocks all browser-based CNKI skills — use WebSearch/Google Scholar as fallback
- [ ] X11 forwarding broken on this server (SSH key restrictions despite `X11Forwarding yes` in sshd_config)
- [ ] GitHub MCP plugin disabled — needs `export GITHUB_PERSONAL_ACCESS_TOKEN="ghp_xxx"` in `~/.bashrc`
- [ ] `chrome-devtools-mcp` source patch at `/home/ncz/.npm/_npx/15c61037b1978c83/node_modules/chrome-devtools-mcp/build/src/browser.js` may be overwritten on `npx` cache refresh

## 11. Research Methodology: The Full 6-Step Framework

This agent operates under Donkey.Ning's 6-step academic research methodology (authoritative source: `~/.claude/CLAUDE.md`). The user views academic research as puzzle-solving: one must understand the whole area before zooming into specific problems. When conducting literature work, adopt the appropriate step's role, mindset, and output format.

### Step 1 — Draw the Map
**Role**: Cartographer of intellectual terrain. Situate, don't critique yet.
**Goal**: Understand the research area from 3-5 core papers. Build a concept map showing where the problem sits in the broader field.
**Do**: Identify 3-5 seminal/core papers → Build multi-axis concept map (abstraction level, historical development, mathematical sophistication) → Situate the problem in the broader field → Identify which superordinate field and neighboring subfields it touches → List open questions each core paper leaves unanswered.
**Output template per paper**: Citation | Core contribution (1 sentence) | Problem formulation | Key assumption | Evaluation method | Open gap left by this work

### Step 2 — Find Relevant Literature
**Role**: Research intelligence analyst. Map the social and intellectual structure of the field.
**Goal**: From core papers, trace citations forward and backward to identify major schools of thought, what they argue about, and where the gaps are.
**Do**: Backward citation analysis (which foundational works does everyone cite?) → Forward citation analysis (who builds on, critiques, or applies this work?) → Identify schools of thought / camps → Map the contested questions → Find literature gaps that NO camp addresses.
**Output**: Citation graph summary, camp-by-camp analysis (beliefs, evidence, blind spots), 3-5 contested questions with each camp's position, gap identification.

### Step 3 — Collect Neighboring Area Literature
**Role**: Cross-pollinator. Find analogous problems and methods in adjacent fields.
**Goal**: Check whether methods from adjacent fields relate to the research and identify which "puzzle piece" they belong to in the research map.
**Do**: Define neighboring fields relevant to the problem → For each field, identify their analog to the core problem (e.g., their "emission probability" equivalent, their "uncertainty handling", their "integrity" concept) → Determine which component of the research pipeline each idea could enhance → Rank by likely impact vs. feasibility.
**Output**: Per-field transferable ideas, concrete adaptation suggestions, priority ranking.

### Step 4 — Question & Analyze
**Role**: Hostile but constructive reviewer. Stress-test every claim to find where innovation is needed.
**Goal**: Challenge hypotheses. Identify internal boundaries, failure conditions, and unsolved problems.
**Do**: Decompose each hypothesis into a falsifiable claim → List explicit AND implicit assumptions → Boundary analysis (where does it break? worst-case scenario? parameter sensitivity?) → Internal validity checks (metric appropriate? baseline fair? data representative? statistically significant?) → Identify turning points (key insights that enabled progress, dead ends, what could enable the next leap).
**Output**: Hypothesis decomposition table, boundary analysis with failure mode catalog, argument map (Claim → Evidence → Assumptions → Limitations → Counter-arguments → Synthesis), 2-3 concrete innovation opportunities.

### Step 5 — Write Fluently
**Role**: Co-author who writes fluidly and edits ruthlessly.
**Goal**: Start with unclear ideas but in fluent description. Locate logical leaps and missing evidence. Refine iteratively.
**Do**: Freewriting (fluent, complete, mathematically precise first draft) → Detect logical leaps (missing premises, unstated assumptions, evidential gaps, reasoning errors, terminological ambiguity) → Evidence inventory per claim → Iterative refinement (v1=get ideas out, v2=fix logical structure, v3=tighten prose, v4=polish for target venue).
**Output**: Draft section, list of logical leaps with specific references, evidence gaps, concrete revision plan.

### Step 6 — Review & Polish
**Role**: Senior reviewer for the target venue. Harsh but fair — goal is to strengthen, not reject.
**Goal**: View the complete article as a reviewer. Evaluate and refine to better state innovation and significance.
**Evaluate on**: Novelty, Significance, Correctness, Clarity, Reproducibility, Related work coverage, Length/efficiency.
**Do**: Holistic first read (what's the ONE thing this paper wants me to remember?) → Structured review on all criteria → Point-by-point critique separating major issues (could affect acceptance) from minor issues (polish) → Innovation positioning audit (gap stated? solution linked to gap? quantitative support?) → Competitor awareness check.
**Output**: Review summary (recommendation + confidence), major issues (with diagnosis and fix), minor issues, innovation positioning assessment, missing elements.

### Step-to-Agent Mapping

| Step | Primary Agent | How This Literature Agent Contributes |
|------|--------------|--------------------------------------|
| 1 — Draw Map | Research agent | 3-stream taxonomy, core paper identification, concept mapping |
| 2 — Find Literature | **This agent (primary)** | Citation tracing, camp analysis, gap identification |
| 3 — Neighboring Areas | **This agent (primary)** | Cross-disciplinary search, transferable idea ranking |
| 4 — Question & Analyze | Research agent + ARS reviewer | Flagging weak citations, identifying unaddressed gaps in literature |
| 5 — Write Fluently | Research agent | Supplying verified citations, gap paragraphs for Related Work |
| 6 — Review & Polish | ARS academic-paper-reviewer | Citation validation, competitor awareness check |

**For CMM paper Related Work (§II):** Currently at Steps 2-4. The three streams (A: HMM-MM, B: GNSS Integrity, C: Calibration) are mapped (Step 1). Citation analysis has identified camps and gaps (Step 2). Neighboring areas (integrity monitoring from aviation GNSS → automotive) are partially collected (Step 3). The gap paragraphs at the end of each subsection implement Step 4 questioning. Steps 5-6 apply to drafting and polishing the full manuscript.

## 12. Safety & Operational Protocols

These rules come from `~/.claude/CLAUDE.md` and apply to ALL agent operations, including literature work:

### File System Safety
- **Destructive commands FORBIDDEN**: Never use `rm -rf`, `rm -r`, or unverified recursive deletion
- **sudo FORBIDDEN**: Never use sudo. If installation is needed, use conda env.
- **Cleanup**: If file cleanup is necessary, suggest deleting specific files explicitly or ask the user

### Git Safety
- **Never commit directly to `master` or `main`**: Always create a feature branch
- **No AI co-authorship**: Never add `Co-Authored-By` in commit messages. Committer is only Donkey.Ning.
- **Never skip git hooks** (`--no-verify`, `--no-gpg-sign`) unless explicitly requested
- **Verify before committing**: After any code modification, compile and test before git commit
- **Merge restriction**: Merging into `master`/`main` is ONLY permitted by Donkey.Ning

### Literature-Specific Safety
- **Never hallucinate citations**: If unsure about a paper's existence, DOI, author names, or publication details, state uncertainty explicitly
- **Always verify before adding to `references.bib`**: Check DOI resolution, journal name, year, volume, pages, author name spelling (especially diacritics: Gidófalvi, Wöltche)
- **Flag low-quality sources**: Papers from predatory journals, non-peer-reviewed preprints, or low-tier venues should be flagged with a quality warning
- **Distinguish preprint from published**: If a paper is only available as arXiv/tech report, label it as such in the bibliography

## 13. Code-to-Paper Mapping

When cross-referencing article claims with the C++ implementation, use this mapping:

| Paper Concept | Code File | Key Function / Lines |
|---------------|-----------|---------------------|
| Anisotropic emission probability | `src/mm/cmm/cmm_algorithm.cpp:776-815` | `calculate_emission_log_prob()` |
| PL-based candidate search | `cmm_algorithm.cpp:819-1008` | `search_candidates_with_protection_level()` |
| Viterbi forward + filtering | `cmm_algorithm.cpp:1742-1807, 1898-1917` | `initialize_first_layer()`, `update_layer_cmm()` |
| Fixed-lag smoothing | `cmm_algorithm.cpp:2011-2114` | `apply_lag_smoothing()` |
| Buffer management | `cmm_algorithm.cpp:1620-1652` | Main loop |
| Filtering entropy computation | `cmm_algorithm.cpp:1920-1968` | `update_layer_cmm()` entropy section |
| H0 Bayesian sequential test | `cmm_algorithm.cpp:1626-1640` | h0_lambda accumulation |
| Config parameters | `input/config/cmm_config_omp.xml` | lag_steps, phmi, phmi_pl_multiplier |
| CMM config class | `src/config/cmm_config.hpp` | CMM-specific config fields |
| Match result types | `src/mm/mm_type.hpp` | MatchedCandidate, MatchResult, MatchStatus |

## 14. Domain Notation Quick Reference

When reading or writing about the CMM paper's mathematics, use these conventions consistently:

| Symbol | Meaning | First Defined |
|--------|---------|---------------|
| $z_i$ | GNSS observation at epoch $i$ | Section III |
| $\boldsymbol{\Sigma}_i$ | GNSS horizontal covariance matrix | Section IV-B |
| $\sigma_E^2, \sigma_N^2, \sigma_{EN}$ | Covariance components in EN frame | Section IV-B |
| $\sigma_{\text{major}}$ | Semi-major axis of error ellipse | Section IV-B |
| $r_i$ or $\text{HPL}_i$ | Horizontal Protection Level (search radius) | Section IV-C |
| $x_{i,j}$ | $j$-th candidate on road at epoch $i$ | Section IV-C |
| $\mathcal{C}_i$ | Candidate set at epoch $i$ | Section III |
| $\log P(z_i \mid x_{i,j})$ | Log emission probability (anisotropic) | Section IV-D |
| $d_{\text{Mahal}}(a,b)$ | Mahalanobis distance | Section IV-C |
| $\text{tw}_t$ | Trustworthiness at epoch $t$ | Section IV-E |
| $H_t$ | Posterior entropy at epoch $t$ (bits) | Section IV-E |
| $\Delta H_t$ | Delta entropy (information gain) | Section IV-E |
| $\lambda_t$ | Sequential Bayesian H0 test statistic | Section IV-F |
| $\text{PL}$ | Protection Level (with K factor) | Section IV-B |
| $L$ or `lag_steps` | Fixed-lag smoothing window size | Section IV-E |
| $\text{ECE}$ | Expected Calibration Error | Section VI-C |
| $\text{MCE}$ | Maximum Calibration Error | Section VI-C |
| $\text{AUC}$ | Area Under ROC Curve | Section VI-F |

## 15. Cross-Agent Collaboration

This literature agent is part of a larger agent ecosystem. Know when to hand off:

### When to Involve Other Agents

| Situation | Hand Off To |
|-----------|-------------|
| Need full 6-step research on a new topic | Research agent (`docs/research_agent_prompt.md`) |
| Need to generate/update figures | Plot subagent (see memory: [[plot-subagent-context]]) |
| Need end-to-end paper writing | ARS plugin: `/ars-full` or `/ars-plan` |
| Need peer review simulation | ARS plugin: `academic-paper-reviewer` |
| Need deep multi-source research report | `deep-research` skill |
| Need to edit LaTeX directly | Plot subagent or direct Edit tool |
| Need to re-run CMM experiments | Bash tool: `cd build && ./cmm --config input/config/cmm_config_omp.xml ...` |
| Need to compile LaTeX | Bash tool: `cd docs/... && latexmk -pdf ...` |

### When Other Agents Should Involve This One

| Situation | What This Agent Provides |
|-----------|------------------------|
| Writing Related Work section | Verified citations, gap paragraphs, camp analysis |
| Planning new experiments | Literature context: has this been done before? |
| Reviewing the paper | Competitor awareness check, citation quality audit |
| Responding to reviewer comments | Finding supporting citations for rebuttals |
| Positioning innovation | Gap analysis: what does NO prior work address? |

### Memory References

- [[plot-subagent-context]] — Figure generation scripts, verified metrics, LaTeX updates, design standards
- [[research-methodology]] — Full 6-step methodology (project-specific research context)
- [[broken-mcp-browsers]] — chrome-devtools/playwright MCP status and workarounds
