#!/usr/bin/env python3
"""
CMM Python Interface Example
Demonstrates how to use CovarianceMapMatch in Python similar to the C++ example.

This script shows:
1. Loading network and UBODT
2. Creating CMM configuration
3. Preparing trajectory with covariance data
4. Running map matching
5. Accessing results including trustworthiness scores
"""

import sys
sys.path.insert(0, '/home/dell/fmm_sjtugnc/build/python')

from fmm import *

def main():
    print("\n" + "=" * 60)
    print("Covariance-based Map Matching (CMM) Python Example")
    print("=" * 60 + "\n")

    # Step 1: Load network
    print("Step 1: Loading network...")
    network_file = "input/map/haikou/edges.shp"
    network = Network(network_file, "id", "source", "target")
    print(f"  Loaded network with {network.get_edge_count()} edges\n")

    # Step 2: Create graph from network
    print("Step 2: Creating network graph...")
    graph = NetworkGraph(network)
    print(f"  Graph has {graph.get_num_vertices()} vertices\n")

    # Step 3: Load UBODT
    print("Step 3: Loading UBODT...")
    ubodt_file = "input/map/haikou_ubodt_mmap.bin"
    ubodt = UBODT.read_ubodt_mmap_binary(ubodt_file)
    print(f"  Loaded UBODT with {ubodt.get_num_rows()} rows\n")

    # Step 4: Create CMM algorithm instance
    print("Step 4: Creating CMM algorithm instance...")
    cmm = CovarianceMapMatch(network, graph, ubodt)
    print("  CMM algorithm initialized\n")

    # Step 5: Create CMM configuration
    print("Step 5: Creating CMM configuration...")
    config = CovarianceMapMatchConfig(
        k_arg=8,                          # Number of candidates per point
        min_candidates_arg=3,             # Minimum candidates to keep
        protection_level_multiplier_arg=1.0,  # Protection level multiplier
        reverse_tolerance=0.0,            # Reverse movement tolerance (meters)
        normalized_arg=True,              # Normalize emission probabilities
        use_mahalanobis_candidates_arg=True,  # Use Mahalanobis for candidates
        window_length_arg=10,             # Window length for trustworthiness
        margin_used_trustworthiness_arg=True   # Use margin for trustworthiness
    )
    print(f"  Configuration: k={config.k}, window_length={config.window_length}\n")

    # Step 6: Create a trajectory with covariance data
    print("Step 6: Creating test trajectory with covariance data...")

    # Create trajectory geometry
    geom = LineString()
    geom.add_point(110.1975, 20.0145)  # Point 1
    geom.add_point(110.1995, 20.0165)  # Point 2
    geom.add_point(110.2015, 20.0185)  # Point 3
    geom.add_point(110.2035, 20.0205)  # Point 4
    geom.add_point(110.2055, 20.0225)  # Point 5

    # Create timestamps (seconds)
    timestamps = DoubleVector()
    for i in range(5):
        timestamps.append(float(i * 30))  # 30-second intervals

    # Create covariance matrices for each point
    # These represent GNSS error estimates (typically from receiver)
    covariances = CovarianceMatrixVector()
    for i in range(5):
        cov = CovarianceMatrix()
        cov.sde = 2.0   # East standard deviation (meters)
        cov.sdn = 1.5   # North standard deviation (meters)
        cov.sdu = 3.0   # Up standard deviation (meters)
        cov.sdne = 0.1  # North-East covariance
        cov.sdeu = 0.05 # East-Up covariance
        cov.sdun = 0.08 # Up-North covariance
        covariances.append(cov)

    # Create protection levels for each point
    # These represent the integrity protection (e.g., from RAIM or SBAS)
    protection_levels = DoubleVector()
    for i in range(5):
        protection_levels.append(5.0)  # 5 meters protection level

    # Create CMMTrajectory
    traj = CMMTrajectory()
    traj.id = 1
    traj.geom = geom
    traj.timestamps = timestamps
    traj.covariances = covariances
    traj.protection_levels = protection_levels

    print(f"  Created trajectory with {traj.geom.get_num_points()} points")
    print(f"  Valid: {traj.is_valid()}\n")

    # Step 7: Match trajectory
    print("Step 7: Running map matching...")
    result = cmm.match_traj(traj, config)
    print(f"  Match completed\n")

    # Step 8: Print results
    print("Step 8: Results:")
    print(f"  Matched: {result.is_matched()}")
    print(f"  Matched distance (sp_dist): {result.sp_dist:.2f} meters")
    print(f"  Euclidean distance (eu_dist): {result.eu_dist:.2f} meters")
    print(f"  Matching ratio: {result.eu_dist / result.sp_dist if result.sp_dist > 0 else 0:.4f}")
    print(f"  Number of matched edges: {len(result.cpath)}")
    print(f"  Matched path (edge indices): {list(result.cpath)}")

    # Print emission probabilities and trustworthiness
    if hasattr(result, 'ep') and result.ep:
        print(f"\n  Emission probabilities (first 5): {[f'{ep:.4e}' for ep in result.ep[:5]]}")

    if hasattr(result, 'tp') and result.tp:
        print(f"  Transition probabilities (first 5): {[f'{tp:.4e}' for tp in result.tp[:5]]}")

    if hasattr(result, 'trustworthiness') and result.trustworthiness:
        print(f"  Trustworthiness scores (first 5): {[f'{tw:.4f}' for tw in result.trustworthiness[:5]]}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
