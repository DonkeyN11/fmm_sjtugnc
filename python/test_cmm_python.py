#!/usr/bin/env python3
"""
Test script for CMM Python bindings.
Demonstrates basic usage of CovarianceMapMatch in Python.
"""

import sys
sys.path.insert(0, '/home/dell/fmm_sjtugnc/build/python')

from fmm import *

def test_covariance_matrix():
    """Test CovarianceMatrix creation and properties."""
    print("=== Testing CovarianceMatrix ===")
    cov = CovarianceMatrix()
    cov.sde = 2.0
    cov.sdn = 1.5
    cov.sdu = 3.0
    cov.sdne = 0.1
    cov.sdeu = 0.05
    cov.sdun = 0.08

    print(f"  sde (East std): {cov.sde}")
    print(f"  sdn (North std): {cov.sdn}")
    print(f"  sdu (Up std): {cov.sdu}")
    print(f"  sdne (NE covariance): {cov.sdne}")
    print(f"  2D uncertainty: {cov.get_2d_uncertainty():.4f}")
    print("  ✓ CovarianceMatrix works\n")

def test_cmm_config():
    """Test CovarianceMapMatchConfig creation."""
    print("=== Testing CovarianceMapMatchConfig ===")
    config = CovarianceMapMatchConfig(
        k_arg=8,
        min_candidates_arg=3,
        protection_level_multiplier_arg=1.0,
        reverse_tolerance=0.0,
        normalized_arg=True,
        use_mahalanobis_candidates_arg=True,
        window_length_arg=10,
        margin_used_trustworthiness_arg=True
    )

    print(f"  k (candidates): {config.k}")
    print(f"  min_candidates: {config.min_candidates}")
    print(f"  protection_level_multiplier: {config.protection_level_multiplier}")
    print(f"  reverse_tolerance: {config.reverse_tolerance}")
    print(f"  normalized: {config.normalized}")
    print(f"  use_mahalanobis_candidates: {config.use_mahalanobis_candidates}")
    print(f"  window_length: {config.window_length}")
    print(f"  margin_used_trustworthiness: {config.margin_used_trustworthiness}")
    print(f"  validate: {config.validate()}")
    print("  ✓ CovarianceMapMatchConfig works\n")

def test_cmm_trajectory():
    """Test CMMTrajectory creation."""
    print("=== Testing CMMTrajectory ===")

    # Create a simple trajectory
    geom = LineString()
    geom.add_point(110.2, 20.0)
    geom.add_point(110.21, 20.01)
    geom.add_point(110.22, 20.02)

    traj = CMMTrajectory()
    traj.id = 1
    traj.geom = geom

    # Create timestamps vector
    timestamps = DoubleVector()
    timestamps.append(0.0)
    timestamps.append(1.0)
    timestamps.append(2.0)
    traj.timestamps = timestamps

    # Add covariance matrices
    cov1 = CovarianceMatrix()
    cov1.sde = 2.0
    cov1.sdn = 1.5
    cov1.sdu = 3.0
    cov1.sdne = 0.1
    cov1.sdeu = 0.05
    cov1.sdun = 0.08

    cov2 = CovarianceMatrix()
    cov2.sde = 2.0
    cov2.sdn = 1.5
    cov2.sdu = 3.0
    cov2.sdne = 0.1
    cov2.sdeu = 0.05
    cov2.sdun = 0.08

    cov3 = CovarianceMatrix()
    cov3.sde = 2.0
    cov3.sdn = 1.5
    cov3.sdu = 3.0
    cov3.sdne = 0.1
    cov3.sdeu = 0.05
    cov3.sdun = 0.08

    covs = CovarianceMatrixVector()
    covs.append(cov1)
    covs.append(cov2)
    covs.append(cov3)
    traj.covariances = covs

    # Create protection levels vector
    pl = DoubleVector()
    pl.append(5.0)
    pl.append(5.0)
    pl.append(5.0)
    traj.protection_levels = pl

    print(f"  Trajectory ID: {traj.id}")
    print(f"  Number of points: {traj.geom.get_num_points()}")
    print(f"  Number of timestamps: {len(traj.timestamps)}")
    print(f"  Number of covariances: {len(traj.covariances)}")
    print(f"  Number of protection levels: {len(traj.protection_levels)}")
    print(f"  Is valid: {traj.is_valid()}")
    print("  ✓ CMMTrajectory works\n")

def test_matrix2d_vector2d():
    """Test Matrix2d and Vector2d helper classes."""
    print("=== Testing Matrix2d and Vector2d ===")

    # Matrix2d is an internal helper class, exposed for completeness
    # Note: These are primarily used internally by the CMM algorithm
    mat = Matrix2d(4.0, 0.1, 0.1, 2.25)

    det = mat.determinant()
    print(f"  Determinant of matrix: {det:.4f}")

    inv = mat.inverse()
    inv_det = inv.determinant()
    print(f"  Inverse matrix determinant: {inv_det:.4f}")

    # Vector2d is also an internal helper
    v1 = Vector2d(1.0, 2.0)
    v2 = Vector2d(3.0, 4.0)
    dot = v1 * v2
    print(f"  Dot product (1,2)·(3,4): {dot}")
    print("  ✓ Matrix2d and Vector2d work (internal helpers)\n")

def main():
    print("\n" + "=" * 50)
    print("CMM Python Bindings Test")
    print("=" * 50 + "\n")

    try:
        test_covariance_matrix()
        test_cmm_config()
        test_cmm_trajectory()
        test_matrix2d_vector2d()

        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        return 0
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
