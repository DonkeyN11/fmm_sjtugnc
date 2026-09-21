# Covariance-based Map Matching (CMM)

## Overview

CMM (Covariance-based Map Matching) is an enhanced version of the FMM algorithm that incorporates GNSS positioning uncertainty information through covariance matrices and protection levels. This approach provides more accurate map matching by considering the actual positioning error characteristics at each GPS point.

## Key Differences from FMM

### 1. **Emission Probability Calculation**
- **FMM**: Uses simple Gaussian distribution with fixed GPS error parameter
- **CMM**: Uses multivariate Gaussian distribution with full covariance matrix
- **Formula**: CMM emission probability = N(x; mu, Sigma) where Sigma is the 2D covariance matrix

### 2. **Candidate Point Selection**
- **FMM**: Uses fixed search radius for all points
- **CMM**: Uses a dynamic search radius equal to the epoch's protection level
- **Formula**: `search_radius = protection_level`, doubled (up to 8 times, i.e. at most 256×HPL) while the candidate set is below `min_candidates`; the epoch is skipped only if it is still empty at the last doubling. A segment admitted by doubling lies outside the protection level and receives the `PHMI` weight in the emission normalization.

### 3. **Input Data Requirements**
- **FMM**: Trajectory coordinates only
- **CMM**: Trajectory coordinates + covariance matrices + protection levels

## Covariance Matrix Format

Each GPS point requires a 6-element covariance matrix:
- `sdn`: North standard deviation (m)
- `sde`: East standard deviation (m)
- `sdu`: Up standard deviation (m)
- `sdne`: North-East covariance (m²)
- `sdeu`: East-Up covariance (m²)
- `sdun`: Up-North covariance (m²)

Example: For a point (121, 31) with covariance values [2.3751, 2.4249, 6.4817, 0.7646, -1.8492, -0.6391]

## Protection Level

Protection level is the bound on horizontal position error at the integrity risk `PHMI` (default 10⁻⁵), computed by RAIM as `HPL = K · σ_major`. HPL is a *horizontal* containment radius, so `K` is the two-degree-of-freedom factor: `K = 4.7985` at 10⁻⁵ ($\chi^2_2$). (The frequently quoted `K ≈ 5.33` is the 1-D two-sided factor for 10⁻⁷, not for 10⁻⁵ — see the table in `CLAUDE.md` §GNSS Stochastic Model.) It is the search radius for candidate selection, verbatim — there is no multiplier. Note the units follow the network CRS: with an EPSG:4326 network the values are degrees, not metres, and the same number means something ~111,320× smaller.

## Implementation Structure

```
src/mm/cmm/
    cmm_algorithm.hpp     # Header with CMM classes and structures
    cmm_algorithm.cpp     # Implementation of CMM algorithm
    CMakeLists.txt        # Build configuration

example/
    cmm_example.cpp       # Usage example
```

## Core Components

### 1. CovarianceMatrix Struct
```cpp
struct CovarianceMatrix {
    double sde, sdn, sdu;    // Standard deviations
    double sdne, sdeu, sdun;  // Covariances

    Matrix2d to_2d_matrix() const;      // 2x2 horizontal block; Eigen is not used
    double get_2d_uncertainty() const;  // sqrt(det) of that block; currently unused
};
```

Only `sde`, `sdn` and `sdne` reach the emission model — `to_2d_matrix()` reads nothing else. A covariance that is not positive definite, or has a non-positive or non-finite `sde`/`sdn`, is rejected by `is_covariance_usable()` and replaced by `fallback_covariance()`, an isotropic 5 m ellipse.

### 2. CMMTrajectory Struct
```cpp
struct CMMTrajectory {
    int id;
    CORE::LineString geom;
    std::vector<double> timestamps;
    std::vector<CovarianceMatrix> covariances;
    std::vector<double> protection_levels;
};
```

### 3. CovarianceMapMatch Class
Main algorithm class with methods:
- `match_traj()`: Match a trajectory with covariance data
- `calculate_emission_probability()`: Calculate emission prob using covariance
- `search_candidates_with_protection_level()`: Dynamic candidate selection

## Usage Example

```cpp
#include "mm/cmm/cmm_algorithm.hpp"

// Create CMM instance
CovarianceMapMatch cmm(network, graph, ubodt);

// Configure parameters
CovarianceMapMatchConfig config(8, 3, 2.0, 0.1);

// Create trajectory with covariance data
std::vector<CovarianceMatrix> covariances = {...};
std::vector<double> protection_levels = {...};
CMMTrajectory trajectory(id, geom, timestamps, covariances, protection_levels);

// Perform matching
MatchResult result = cmm.match_traj(trajectory, config);
```

## Configuration Parameters

- `k`: Number of candidates per point (default: 8)
- `min_candidates`: Minimum candidates to keep -- the floor the search-radius fallback doubles toward (default: 3)
- `reverse_tolerance`: Reverse movement tolerance (default: 0.0)
- `phmi`: Integrity risk; the inside/outside weight of the grouped emission normalization (default: 1.0e-5)

The full key list is in [CLAUDE.md](CLAUDE.md).

## Mathematical Foundation

### Emission Probability
Log-space form actually computed, with `d = z_i - x_(i,j)` and `S = Sigma_i`:

```
log p = -0.5 * ( log(2*pi) + log|S| + d^T S^-1 d )
```

The emission is then normalized across the layer's candidates, split at `HPL_i` into
an inside group weighted by `(1 - PHMI)` and an outside group weighted by `PHMI`,
each divided by its own sum.

### Search Radius
```
r = HPL                     # m = 0
r = 2 * r                   # retry, while |candidates| < min_candidates
# at most 8 doublings => 256 * HPL; skip the epoch only if still empty
```

## Dependencies

- Boost.Geometry: For spatial calculations
- GDAL/OGR: For geographic data handling

CMM previously used Eigen3; it now uses a small `Matrix2d` struct declared in
`src/mm/cmm/cmm_algorithm.hpp`.

## Build Instructions

Build the whole project with CMake (see the top-level [README.md](README.md)). The
usage example is `example/cmm_example.cpp`, built as part of the default target.

## Performance Considerations

- CMM is computationally more expensive than FMM due to the per-candidate matrix operations
- Matrix inversions are performed for each candidate point
- Protection level-based search can reduce unnecessary computations
- Parallel processing is supported for batch matching

## Future Enhancements

1. Support for different coordinate systems
2. Efficient matrix operation optimizations
3. Dynamic protection level estimation
4. Integration with real-time GNSS receivers