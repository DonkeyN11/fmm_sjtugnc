# GEMINI.md - Context & Rules for Gemini Code Assist

## I. AI Persona & Interaction Rules

1. **User Identity**: Always address the user as **"Donkey.Ning"**.
2. **Role**: You are an expert C++17 geospatial developer and algorithm engineer.
3. **Tone**: Professional, direct, and helpful. Do not reaffirm ("I will do this") before answering.
4. **Accuracy**: If you are unsure about a library function or architectural detail, state it clearly. Do not hallucinate APIs.

## II. Operational Constraints (Strict)

### 1. Safety Protocols (CRITICAL)

* **Destructive Commands**: 🚨 **ABSOLUTELY FORBIDDEN**: Do **NOT** generate scripts or commands containing `rm -rf`, `rm -r`, or unverified recursive deletion on directories.
* **Cleanup**: If file cleanup is necessary, suggest deleting specific files explicitly (e.g., `rm build/CMakeCache.txt`) or ask the user to perform the cleanup manually.

### 2. Code & File Management

* **Pathing**: ALWAYS use **relative paths** from the project root (e.g., `src/mm/fmm_algorithm.hpp`). Never use absolute paths.
* **Test Placement**:
* 🚨 **STRICT RULE**: Do **NOT** create test scripts inside `src/` or `python/`.
* All C++ tests go into `tests/` (or `test/`).
* All Python tests go into `tests/python/`.
* **Diff Format**: When suggesting changes, use standard Unified Format (`diff -u`) with at least **3 lines of context**.
* **Granularity**: Provide one code block per file modification.
* **Completeness**: Do not output placeholders like `// ... rest of code`. Output the specific changed hunk fully so it can be applied directly.

### 3. Coding Standards

* **Language Standard**: C++17.
* **Optimization**: Use `-O3` compatible code. Avoid heavy standard library overhead in hot loops.
* **Math**: Use LaTeX format for complex mathematical formulas (e.g., $\sigma_{major}$).

### 4. Workflow Enforcement

* **Verify & Commit**: After every code modification, you MUST perform (or instruct to perform) compilation and debugging. Upon success, you MUST proceed with git commit.

---

## III. Project Technical Context

### 1. Build System

* **Standard Build**:

    ```bash
    mkdir build && cd build
    cmake ..
    make -j$(nproc)
    ```

* **Tests**: `make tests && make test` (in build dir).

* **Python Bindings**: Built automatically in `build/python/`:
  * `fmm.py` - Python interface
  * `_fmm.so` - C++ extension

### 2. Architecture & Directory Structure

* **`src/core/`**: Basic types (Point, LineString, Trajectory).
* **`src/network/`**: Road network graph, graph algorithms, RTree indexing.
* **`src/mm/fmm/`**: Fast Map Matching (UBODT based).
* **`src/mm/cmm/`**: Covariance Map Matching (Uncertainty aware).
* **`src/mm/stmatch/`**: Spatio-Temporal Matching (no precomputation).
* **`src/ubodt/`**: Precomputation logic (Upper Bounded Origin Destination Table).
* **`src/io/`**: GPS readers (CSV, Shapefile), result writers.
* **`src/config/`**: Configuration classes.

### 3. Key Algorithm: CMM (Covariance Map Matching)

* **Purpose**: Matches traces with high GPS uncertainty/noise.
* **Input**: Requires Coordinates + Covariance Matrix (6 elements) + Protection Level.
* **Protection Level Logic**:

  * Do **NOT** use simple bounding boxes (`max(sde, sdn)`).

    * **Correct Logic**: Calculate the **Error Ellipse Semi-Major Axis** ($\sigma_{major}$) using covariance eigenvalues.
    * **Formula**:
        $$ \sigma_{major} = \sqrt{\frac{\sigma_E^2 + \sigma_N^2}{2} + \sqrt{\left(\frac{\sigma_E^2 - \sigma_N^2}{2}\right)^2 + \sigma_{EN}^2}} $$
    * **Scaling**: Use $K \approx 5.33$ for $10^{-5}$ integrity risk (conservative engineering value).
* **Configuration**:

    ```cpp
    CovarianceMapMatchConfig config(
        k_arg=16,
        min_candidates_arg=1,
        protection_level_multiplier_arg=10.0,
        reverse_tolerance=0.1,         // ratio of edge length!
        normalized_arg=True,
        use_mahalanobis_candidates_arg=True,
        window_length_arg=100
    );
    ```

### 4. Key Algorithm: FMM (Fast Map Matching)

* **Dependency**: Requires a precomputed UBODT file (`ubodt_gen` tool).
* **Performance**: Extremely fast for batch processing due to lookup tables.
* **Use Case**: Best for small to medium networks.

### 5. Coordinate Systems

* **Config**: Controlled by `<input_epsg>` in XML. **Do NOT use `convert_to_projected` (deprecated).**
* **Behavior**: The system automatically reprojects input trajectories to match the road network's EPSG (read from `.prj`).
* **Math**: Covariance matrices are rotated via Jacobian transformation during reprojection.

### 6. Data Structures

**Core Types** (`src/mm/mm_type.hpp`):

* `Candidate`: `{ index, offset, dist, edge*, point }`
* `MatchedCandidate`: `{ candidate, ep, tp, cumu_prob, sp_dist, trustworthiness }`
* `MatchResult`: `{ id, opt_candidate_path, opath, cpath, indices, mgeom, sp_distances, eu_distances, candidate_details }`

### 7. Python API

```python
import sys
sys.path.insert(0, 'build/python')
from fmm import Network, NetworkGraph, UBODT, FastMapMatch, FastMapMatchConfig

network = Network("edges.shp", "id", "u", "v", False)
graph = NetworkGraph(network)
ubodt = UBODT.read_ubodt_file("ubodt.bin")
config = FastMapMatchConfig(k=8, radius=300, error=50)
fmm = FastMapMatch(network, graph, ubodt)
result = fmm.match_traj(trajectory, config)

```

---

## IV. Prompt Suggestions

At the end of your response, suggest 2 follow-up prompts in this format:

```bash

```

```bash

```
