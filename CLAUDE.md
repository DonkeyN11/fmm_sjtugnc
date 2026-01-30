# CLAUDE.md

This file provides specific instructions and rules for Claude Code (and other AI assistants) when interacting with this repository, followed by the technical documentation for the FMM framework.

---

# I. AI Interaction Rules

## General Interactions
1. **Addressing the User**: Always address the user as "Donkey.Ning" in every response. Call yourself "Claude" and call me "Donkey" instead of I/me and you. 中文同样适用。
2. **Tone**: Maintain a conversational and direct tone. Do NOT reaffirm ("I will do that") before answering.
3. **Accuracy**: Do not make things up. If you are unsure, state it.

## Code & Diff Formatting
1. **Pathing**: Use **relative paths** from the project root for all file headers in diffs.
2. **Diff Format**: Use standard Unified Format (`diff -u`).
   * **Context**: Include at least **3 lines of unchanged context** around every edit to ensure patches apply correctly.
   * **New Files**: Start new files with `--- /dev/null`.
   * **Scope**: Only modify files present in the provided context.
3. **Granularity**: Provide one code block per file modification.
4. **Completeness**: Do not output placeholders like `// ... rest of code`. Output the specific changed hunk fully so it can be applied directly.

## Prompt Suggestions
At the very end of your response, after all other content, suggest up to two brief prompts using the following format:

```

```

---

# II. Project Documentation

## Build Commands

### Standard Build
```bash
mkdir build
cd build
cmake ..
make -j$(nproc)

```

### Clean Build

```bash
cd build
make clean
cmake ..
make -j$(nproc)

```

### Installation (Optional)

```bash
sudo make install

```

### Python Bindings

Python bindings are built automatically with the main build. They are located in `build/python/`:

* `fmm.py` - Python interface
* `_fmm.so` - C++ extension

## Testing

Tests are currently disabled in the default build. To enable and run tests:

```bash
# In build directory
make tests  # Build test executables
make test   # Run all tests

```

Individual test executables (when built):

* `./build/algorithm_test`
* `./build/fmm_test`
* `./build/network_test`
* `./build/network_graph_test`

## High-Level Architecture

FMM is a Fast Map Matching framework that implements Hidden Markov Model (HMM) based map matching algorithms with precomputation optimization.

### Core Components

**Data Flow Pipeline:**

```
GPS Data → Trajectory → Candidates → HMM → MatchResult → Output

```

**Module Hierarchy:**

```
src/
├── core/           # Basic data types (Point, LineString, Trajectory)
├── network/        # Road network, graph algorithms, RTree indexing
├── algorithm/      # Geometric algorithms, distance calculations
├── io/             # GPS readers (CSV, Shapefile), result writers
├── config/         # Configuration classes for all components
├── util/           # Utilities, debug tools
├── mm/             # Map matching algorithms
│   ├── fmm/        # Fast Map Matching (precomputation-based)
│   ├── cmm/        # Covariance Map Matching (uncertainty-aware)
│   ├── stmatch/    # Spatio-Temporal Matching (no precomputation)
│   └── h3mm/       # Uber H3 hexagonal grid matching
└── app/            # CLI applications (fmm, cmm, stmatch, ubodt_gen)

```

### Key Algorithms

**1. FMM (Fast Map Matching)**

* Uses UBODT (Upper Bounded Origin Destination Table) for precomputed shortest paths
* Fastest for batch processing with precomputation
* Best for small to medium networks

**2. CMM (Covariance Map Matching)**

* Incorporates GNSS uncertainty through covariance matrices
* Uses Mahalanobis distance for candidate selection
* Requires: coordinates + covariance matrices + protection levels per point
* More accurate but computationally expensive

**3. STMatch**

* No precomputation required
* Suitable for large-scale networks
* Slower per trajectory but no UBODT generation overhead

### Data Structures

**Core Types** (`src/mm/mm_type.hpp`):

```cpp
Candidate { index, offset, dist, edge*, point }
MatchedCandidate { candidate, ep, tp, cumu_prob, sp_dist, trustworthiness }
MatchResult { id, opt_candidate_path, opath, cpath, indices, mgeom,
              sp_distances, eu_distances, candidate_details }

```

**Path Types:**

* `O_Path` (Optimal Path): edge IDs matched to each GPS point
* `C_Path` (Complete Path): topologically connected edge sequence
* `indices`: mapping from opath to cpath positions

### Coordinate System Handling

**Important**: Use `input_epsg` parameter (NOT `convert_to_projected` - deprecated).

```xml
<other>
  <input_epsg>4326</input_epsg>  <input_epsg>32649</input_epsg>  </other>

```

The system automatically:

1. Reads network EPSG from shapefile .prj
2. Reads input trajectory EPSG from `input_epsg` config
3. Reprojects trajectory if EPSG codes differ
4. Transforms covariance matrices via Jacobian when reprojecting

### UBODT System

**UBODT Types:**

* **Full UBODT**: Complete precomputation (binary format)
* **PartialUBODT**: Region-based loading (reduces memory)
* **CachedUBODT**: LRU cache for on-demand loading
* **Memory-mapped UBODT**: OS-managed file mapping

**UBODT Management:**

```cpp
// Simple loading
auto ubodt = UBODT::read_ubodt_file("path.bin");

// With manager (caching, recommended)
#include "mm/fmm/ubodt_manager.hpp"
auto ubodt = UBODTHelper::load_ubodt("path.bin", multiplier, keep_in_memory);
UBODTHelper::release_all_ubodts();  // When done

```

**UBODT File Formats:**

* `*_ubodt.txt` - Plain text (large)
* `*_ubodt.bin` - Binary (standard)
* `*_ubodt_indexed.bin` - Binary with index (fastest random access)
* `*_ubodt_mmap.bin` - Memory-mapped format

### Configuration System

**XML Configuration** (preferred for batch processing):

```xml
<config>
  <input>
    <network><file>network.shp</file></network>
    <gps><file>trajectory.csv</file></gps>
  </input>
  <other>
    <input_epsg>4326</input_epsg>
    <k>8</k>
    <radius>300</radius>
    <error>50</error>
  </other>
</config>

```

**Command Line Arguments**:

```bash
fmm --network edges.shp --gps traj.csv --ubodt ubodt.bin --output result.csv

```

### Input Data Formats

**Network**: ESRI Shapefile with fields:

* `id` (or `key`): edge ID
* `source` (or `u`): source node
* `target` (or `v`): target node

**GPS Trajectories**:

1. **CSV Trajectory**: `id;geom;timestamp` (geom is WKT LineString)
2. **CSV Point**: `id;x;y;timestamp` (sorted by id, timestamp)
3. **Shapefile**: One feature per trajectory

### Python API

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

### CMM Specifics

**CMM Input Requirements:**

* Trajectory coordinates (LineString)
* Covariance matrices per point (6 values: sdn, sde, sdu, sdne, sdeu, sdun)
* Protection levels per point (meters)

**CMM Configuration:**

```cpp
CovarianceMapMatchConfig config(
    k_arg=16,                      // candidates
    min_candidates_arg=1,          // minimum to keep
    protection_level_multiplier_arg=10.0,  // search radius multiplier
    reverse_tolerance=0.1,         // ratio of edge length (unitless!)
    normalized_arg=True,           // use normalized probabilities
    use_mahalanobis_candidates_arg=True,  // Mahalanobis-based search
    window_length_arg=100          // sliding window size
);

```

### Performance Optimization

**OpenMP Parallelization:**

```bash
export OMP_NUM_THREADS=128  # Use all cores
fmm --config config.xml

```

**UBODT Caching**:

* Use `UBODTHelper` for batch processing
* Pre-generate UBODT: `ubodt_gen --network edges.shp --output ubodt.bin`
* Use indexed binary format for fastest loading

### Common Patterns

**Adding a New Map Matching Algorithm:**

1. Create `src/mm/your_algorithm/` directory
2. Implement algorithm class inheriting from base interface
3. Create config class in `src/config/`
4. Add CLI app in `src/app/your_algorithm_app.cpp`
5. Update CMakeLists.txt
6. Add Python bindings in `src/python/`

**Working with Candidates:**

```cpp
// Search k nearest candidates within radius
auto candidates = network.search_tr_cs_knn(trajectory.geom, k, radius);

// Calculate shortest path distance between candidates
double sp_dist = ubodt.look_up(ca->edge->source, cb->edge->target);

// Calculate transition probability
double tp = TransitionGraph::calc_tp(sp_dist, eu_dist);

// Calculate emission probability
double ep = TransitionGraph::calc_ep(dist, gps_error);

```

### Important Notes

* **No `is_matched()` method**: Check `len(result.cpath) > 0` instead
* **Distance vectors**: Use `sum(result.sp_distances)` and `sum(result.eu_distances)`
* **reverse_tolerance**: This is a ratio (0.1 = 10%), NOT an absolute distance
* **C++17**: Code uses C++17 standard
* **Boost ABI**: Uses old C++11 ABI (`_GLIBCXX_USE_CXX11_ABI=0`)
* **Release build**: Always use Release mode (`-O3` optimizations)

### Development Workflow

1. **Make changes** to source files
2. **Rebuild**: `cd build && make -j$(nproc)`
3. **Test**: Run with debug logging (`<log_level>1</log_level>` in config)
4. **Python bindings**: Automatically regenerated when headers change

### Key Dependencies

* **GDAL >= 2.2**: Geospatial data I/O
* **Boost >= 1.56.0**: Serialization, geometry, graph
* **OpenMP**: Parallel processing
* **SWIG**: Python bindings (build time only)
* **H3**: Hexagonal indexing (optional, for h3mm)

```


```

# III. Operational Constraints (Strict)

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

### 3. Coding Standards
* **Language Standard**: C++17.
* **Optimization**: Use `-O3` compatible code. Avoid heavy standard library overhead in hot loops.
* **Math**: Use LaTeX format for complex mathematical formulas (e.g., $\sigma_{major}$).

---