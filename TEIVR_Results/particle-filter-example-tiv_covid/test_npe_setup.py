#!/usr/bin/env python
"""
Quick test script to verify NPE setup is correct.

Usage:
    python test_npe_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    errors = []

    # Test standard packages
    try:
        import numpy as np
        print("  ✓ numpy:", np.__version__)
    except ImportError as e:
        errors.append(f"  ✗ numpy: {e}")

    try:
        import scipy
        print("  ✓ scipy:", scipy.__version__)
    except ImportError as e:
        errors.append(f"  ✗ scipy: {e}")

    # Test PyTorch
    try:
        import torch
        print("  ✓ torch:", torch.__version__)
    except ImportError as e:
        errors.append(f"  ✗ torch: {e}")
        errors.append("    Install with: pip install torch")

    # Test SBI
    try:
        import sbi
        print("  ✓ sbi:", sbi.__version__)
    except ImportError as e:
        errors.append(f"  ✗ sbi: {e}")
        errors.append("    Install with: pip install sbi")

    # Test local modules
    try:
        from src import npe_utils
        print("  ✓ src.npe_utils")
    except ImportError as e:
        errors.append(f"  ✗ src.npe_utils: {e}")

    try:
        from src import JSF_Solver_BasePython
        print("  ✓ src.JSF_Solver_BasePython")
    except ImportError as e:
        errors.append(f"  ✗ src.JSF_Solver_BasePython: {e}")

    return errors


def test_config():
    """Test that config file can be loaded."""
    print("\nTesting configuration...")
    errors = []

    try:
        from src import npe_utils
        config = npe_utils.load_config("config/cli-refractory-tiv-jsf.toml")
        print("  ✓ Config loaded")

        lower_bounds, upper_bounds, param_names = npe_utils.get_prior_bounds(config)
        print(f"  ✓ Found {len(param_names)} inference parameters: {param_names}")

        fixed_params = npe_utils.get_fixed_params(config)
        print(f"  ✓ Found {len(fixed_params)} fixed parameters")

    except Exception as e:
        errors.append(f"  ✗ Config loading failed: {e}")

    return errors


def test_data():
    """Test that patient data exists."""
    print("\nTesting data availability...")
    errors = []

    data_dir = Path("data")
    if not data_dir.exists():
        errors.append(f"  ✗ Data directory not found: {data_dir}")
        return errors

    ssv_files = list(data_dir.glob("*.ssv"))
    if not ssv_files:
        errors.append(f"  ✗ No .ssv files found in {data_dir}")
    else:
        print(f"  ✓ Found {len(ssv_files)} patient files")
        for f in ssv_files:
            print(f"    - {f.name}")

    return errors


def test_output_dirs():
    """Test that output directories exist."""
    print("\nTesting output directories...")
    errors = []

    output_dir = Path("npe_outputs")
    subdirs = ["training", "models", "inference"]

    for subdir in subdirs:
        path = output_dir / subdir
        if path.exists():
            print(f"  ✓ {path}")
        else:
            print(f"  ✗ {path} (will be created on first run)")

    return errors


def test_minimal_simulation():
    """Test a single simulation."""
    print("\nTesting minimal simulation...")
    errors = []

    try:
        import numpy as np
        from src import npe_utils

        # Load config
        config = npe_utils.load_config("config/cli-refractory-tiv-jsf.toml")
        lower_bounds, upper_bounds, param_names = npe_utils.get_prior_bounds(config)
        fixed_params = npe_utils.get_fixed_params(config)
        obs_scale = npe_utils.get_observation_scale(config)

        # Sample one parameter set
        theta = npe_utils.sample_from_prior(lower_bounds, upper_bounds, 1, seed=42)[0]
        print(f"  ✓ Sampled parameters: {dict(zip(param_names, theta))}")

        # Try to simulate (this will test JSF integration)
        print("  - Attempting single JSF simulation (may take 1-2 seconds)...")
        from COVID_TEIVR_NPE import simulate_trajectory

        obs = simulate_trajectory(theta, fixed_params, n_timepoints=10, obs_scale=obs_scale, seed=42)
        print(f"  ✓ Simulation successful! Observations shape: {obs.shape}")
        print(f"    Sample observations (first 5): {obs[:5]}")

    except Exception as e:
        errors.append(f"  ✗ Simulation failed: {e}")
        import traceback
        errors.append(traceback.format_exc())

    return errors


def main():
    """Run all tests."""
    print("=" * 70)
    print("NPE Setup Verification")
    print("=" * 70)

    all_errors = []

    # Run tests
    all_errors.extend(test_imports())
    all_errors.extend(test_config())
    all_errors.extend(test_data())
    all_errors.extend(test_output_dirs())

    # Only run simulation test if imports worked
    if not any("torch" in e or "sbi" in e for e in all_errors):
        all_errors.extend(test_minimal_simulation())
    else:
        print("\nSkipping simulation test (missing dependencies)")

    # Summary
    print("\n" + "=" * 70)
    if all_errors:
        print("VERIFICATION FAILED")
        print("=" * 70)
        print("\nErrors found:")
        for error in all_errors:
            print(error)
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements_npe.txt")
        sys.exit(1)
    else:
        print("VERIFICATION SUCCESSFUL")
        print("=" * 70)
        print("\nAll tests passed! You can now run:")
        print("  python COVID_TEIVR_NPE.py simulate --num-trajectories 100")
        print("\nOr submit SLURM job:")
        print("  sbatch run_npe.sh")
        sys.exit(0)


if __name__ == "__main__":
    main()
