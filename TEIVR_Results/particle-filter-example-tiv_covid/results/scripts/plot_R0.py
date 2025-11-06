#!/usr/bin/env python3
"""
Derive within-host R0 posteriors from NPE parameter posteriors.

This script loads NPE posterior samples for the TEIVR model and calculates
the within-host basic reproduction number R0 using the formula:

    R0 = (pi * beta * T0) / (delta * c)

where:
    - pi: virion production rate (from posterior)
    - beta: infection rate (from posterior, scaled by 10^-9 in model)
    - T0: initial target cells (fixed constant = 8E7)
    - delta: infected cell clearance rate (from posterior)
    - c: viral clearance rate (fixed constant = 10.0)

Note: This is an approximation that ignores the refractory cell dynamics
(phi, rho parameters). The complete R0 derivation for the full TEIVR model
is an ongoing TODO.

Usage:
    python plot_R0.py [patient_ids...] [--results-dir PATH]

Examples:
    # Process all patients from latest run
    python plot_R0.py

    # Process specific patients
    python plot_R0.py 432192 444332

    # Use custom results directory
    python plot_R0.py --results-dir /path/to/results/npe/20251103_182129
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from scipy import stats

# Try to use science plots for publication-quality figures
try:
    import scienceplots
    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    print("Warning: scienceplots package not found. Install with: pip install scienceplots")
    print("Continuing without scienceplots styling...")
    SCIENCEPLOTS_AVAILABLE = None

# Model constants (from config)
T0 = 8E7  # Initial target cells
VIRAL_CLEARANCE_RATE = 10.0  # c parameter (fixed)

# Parameter indices in samples array
PARAM_INDICES = {
    'lnV0': 0,
    'beta': 1,
    'phi': 2,
    'rho': 3,
    'delta': 4,
    'pi': 5
}


def calculate_r0(samples):
    """
    Calculate within-host R0 from parameter samples.

    Args:
        samples: NumPy array of shape (n_samples, 6) with parameter samples
                 in order [lnV0, beta, phi, rho, delta, pi]

    Returns:
        NumPy array of R0 values, shape (n_samples,)
    """
    # Extract relevant parameters
    beta = samples[:, PARAM_INDICES['beta']]
    delta = samples[:, PARAM_INDICES['delta']]
    pi = samples[:, PARAM_INDICES['pi']]

    # Apply scaling factor to beta (scaled by 10^-9 in the model equations)
    beta_scaled = beta * 1e-9

    # Calculate R0 = (pi * beta * T0) / (delta * c)
    r0 = (pi * beta_scaled * T0) / (delta * VIRAL_CLEARANCE_RATE)

    return r0


def compute_summary_stats(r0_samples):
    """
    Compute summary statistics for R0 distribution.

    Args:
        r0_samples: Array of R0 values

    Returns:
        Dictionary with summary statistics
    """
    return {
        'mean': np.mean(r0_samples),
        'std': np.std(r0_samples),
        'median': np.median(r0_samples),
        'q025': np.percentile(r0_samples, 2.5),
        'q975': np.percentile(r0_samples, 97.5)
    }


def plot_r0_distribution(r0_samples, patient_id, output_path):
    """
    Create a histogram with KDE for R0 distribution.

    Args:
        r0_samples: Array of R0 values
        patient_id: Patient identifier (string or int)
        output_path: Path to save the figure
    """
    # Apply publication-quality styling
    if SCIENCEPLOTS_AVAILABLE is not None:
        plt.style.use('science')

    # Compute statistics
    stats_dict = compute_summary_stats(r0_samples)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot histogram
    counts, bins, patches = ax.hist(r0_samples, bins=100, density=True,
                                     alpha=0.8, color='steelblue',
                                     edgecolor='black', linewidth=0.5,
                                     label='Posterior samples')

    # Add kernel density estimate
    kde = stats.gaussian_kde(r0_samples)
    x_range = np.linspace(r0_samples.min(), r0_samples.max(), 200)
    ax.plot(x_range, kde(x_range), color='darkblue', linewidth=2.5, label='KDE')

    # Add vertical lines for mean and credible interval
    mean_val = stats_dict['mean']
    ci_lower = stats_dict['q025']
    ci_upper = stats_dict['q975']

    ax.axvline(mean_val, color='navy', linestyle='--',
               linewidth=2, label=f"Mean = {mean_val:.2f}")
    ax.axvline(ci_lower, color='gray', linestyle=':',
               linewidth=1.5, alpha=0.7)
    ci_label = r"95\% CI: [{:.2f}, {:.2f}]".format(ci_lower, ci_upper)
    ax.axvline(ci_upper, color='gray', linestyle=':',
               linewidth=1.5, alpha=0.7, label=ci_label)

    # Labels and title
    ax.set_xlabel(r'$R_0$', fontsize=22)
    ax.set_ylabel('Probability Density', fontsize=22)
    ax.set_title(f'Patient {patient_id}',
                 fontsize=24)
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {output_path}")


def find_latest_results_dir(base_path):
    """
    Find the most recent NPE results directory.

    Args:
        base_path: Base path containing NPE results (parent of timestamped dirs)

    Returns:
        Path to the latest results directory
    """
    base_path = Path(base_path)

    # Look for timestamped directories (YYYYMMDD_HHMMSS format)
    result_dirs = [d for d in base_path.iterdir()
                   if d.is_dir() and d.name.replace('_', '').isdigit()]

    if not result_dirs:
        raise FileNotFoundError(f"No NPE results directories found in {base_path}")

    # Sort by directory name (timestamp) and get the latest
    latest_dir = sorted(result_dirs, key=lambda x: x.name)[-1]
    return latest_dir


def get_available_patients(results_dir):
    """
    Get list of patient IDs with inference results.

    Args:
        results_dir: Path to NPE results directory (timestamped)

    Returns:
        List of patient IDs (as strings)
    """
    inference_dir = results_dir / "inference"

    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    # Get all subdirectories (patient IDs)
    patient_dirs = [d.name for d in inference_dir.iterdir() if d.is_dir()]

    # Filter to only those with samples.npy
    available_patients = []
    for patient_id in patient_dirs:
        samples_file = inference_dir / patient_id / "samples.npy"
        if samples_file.exists():
            available_patients.append(patient_id)

    return sorted(available_patients)


def process_patient(patient_id, results_dir, output_dir):
    """
    Process a single patient: load samples, calculate R0, create plot.

    Args:
        patient_id: Patient identifier
        results_dir: Path to NPE results directory
        output_dir: Directory to save outputs

    Returns:
        Dictionary with R0 statistics
    """
    print(f"\nProcessing patient {patient_id}...")

    # Load posterior samples
    samples_file = results_dir / "inference" / str(patient_id) / "samples.npy"

    if not samples_file.exists():
        print(f"  WARNING: samples.npy not found for patient {patient_id}")
        return None

    try:
        samples = np.load(samples_file)
        print(f"  Loaded {len(samples)} posterior samples")
    except Exception as e:
        print(f"  ERROR loading samples: {e}")
        return None

    # Calculate R0
    r0_samples = calculate_r0(samples)
    print(f"  Calculated R0 for {len(r0_samples)} samples")

    # Compute statistics
    stats_dict = compute_summary_stats(r0_samples)
    stats_dict['patient_id'] = patient_id
    stats_dict['n_samples'] = len(r0_samples)

    # Print statistics
    print(f"  R0 statistics:")
    print(f"    Mean: {stats_dict['mean']:.3f}")
    print(f"    Median: {stats_dict['median']:.3f}")
    print(f"    Std: {stats_dict['std']:.3f}")
    print(f"    95% CI: [{stats_dict['q025']:.3f}, {stats_dict['q975']:.3f}]")

    # Create plot
    output_path = output_dir / f"r0_distribution_{patient_id}.png"
    plot_r0_distribution(r0_samples, patient_id, output_path)

    return stats_dict


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Derive and plot within-host R0 from NPE posteriors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('patient_ids', nargs='*',
                        help='Patient IDs to process (default: all available)')
    parser.add_argument('--results-dir', type=str,
                        help='Path to NPE results directory (default: latest)')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save output plots (default: same as results-dir)')

    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Find latest results directory
        script_dir = Path(__file__).parent
        base_npe_dir = script_dir.parent / "npe"

        try:
            results_dir = find_latest_results_dir(base_npe_dir)
            print(f"Using latest results directory: {results_dir}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    if not results_dir.exists():
        print(f"ERROR: Results directory does not exist: {results_dir}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which patients to process
    if args.patient_ids:
        patient_ids = args.patient_ids
        print(f"\nProcessing specified patients: {', '.join(patient_ids)}")
    else:
        try:
            patient_ids = get_available_patients(results_dir)
            print(f"\nFound {len(patient_ids)} patients: {', '.join(patient_ids)}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    if not patient_ids:
        print("ERROR: No patients to process")
        sys.exit(1)

    # Process each patient
    print("\n" + "="*70)
    print("CALCULATING R0 POSTERIORS")
    print("="*70)

    all_stats = []
    for patient_id in patient_ids:
        stats = process_patient(patient_id, results_dir, output_dir)
        if stats:
            all_stats.append(stats)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_df = summary_df[['patient_id', 'n_samples', 'mean', 'std',
                                  'median', 'q025', 'q975']]
        print(summary_df.to_string(index=False))

        # Save summary to CSV
        summary_file = output_dir / "r0_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
    else:
        print("No patients were successfully processed.")

    print("\nDone!")


if __name__ == "__main__":
    main()
