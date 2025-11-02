#!/usr/bin/env python
"""
Compare NPE and Particle Filter results.

This script loads results from both methods and generates comparison plots
and statistics.

Usage:
    python compare_npe_pf.py --patient 432192
    python compare_npe_pf.py --patient all
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_particle_filter_results(patient_id, pf_output_dir="outputs4"):
    """
    Load particle filter results from fit_result.pkl.

    Args:
        patient_id: Patient identifier
        pf_output_dir: Base directory for particle filter outputs

    Returns:
        Dictionary with parameter samples or summaries
    """
    # Find the particle filter output directory
    patient_dir = Path(pf_output_dir) / patient_id

    if not patient_dir.exists():
        raise FileNotFoundError(f"Particle filter results not found: {patient_dir}")

    # Find subdirectory (e.g., src.tiv.RefractoryCellModel_JSF_1000)
    subdirs = list(patient_dir.glob("*"))
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {patient_dir}")

    # Use the first (or most recent) subdirectory
    result_dir = subdirs[0]
    fit_result_path = result_dir / "fit_result.pkl"

    if not fit_result_path.exists():
        raise FileNotFoundError(f"fit_result.pkl not found in {result_dir}")

    # Load the pickle file
    with open(fit_result_path, 'rb') as f:
        fit_result = pickle.load(f)

    return fit_result


def load_npe_results(patient_id, npe_output_dir="npe_outputs"):
    """
    Load NPE results.

    Args:
        patient_id: Patient identifier
        npe_output_dir: Base directory for NPE outputs

    Returns:
        Dictionary with samples and summary
    """
    inference_dir = Path(npe_output_dir) / "inference" / patient_id

    if not inference_dir.exists():
        raise FileNotFoundError(f"NPE results not found: {inference_dir}")

    # Load samples
    samples_path = inference_dir / "samples.npy"
    if not samples_path.exists():
        raise FileNotFoundError(f"NPE samples not found: {samples_path}")

    samples = np.load(samples_path)

    # Load summary
    summary_path = inference_dir / "summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        summary = None

    return {'samples': samples, 'summary': summary}


def extract_pf_samples(fit_result, n_samples=10000):
    """
    Extract posterior samples from particle filter results.

    The particle filter stores results in fit_result['forecasts'][10.0].
    We need to extract the state vectors for the inference parameters.

    Args:
        fit_result: Loaded particle filter results
        n_samples: Number of samples to extract

    Returns:
        Array of samples (n_samples x n_params)
    """
    # The particle filter stores forecasts at different times
    # We want the forecast at time 10.0
    if 'forecasts' not in fit_result or 10.0 not in fit_result['forecasts']:
        raise ValueError("Particle filter results don't contain forecasts at time 10.0")

    forecast = fit_result['forecasts'][10.0]

    # Extract parameter names and samples
    # The exact structure depends on pypfilt output format
    # Typically: forecast.state_vec contains particle states with weights
    param_names = ['lnV0', 'beta', 'phi', 'rho', 'delta', 'pi']

    # Try to extract samples (structure may vary)
    try:
        state_vec = forecast.state_vec
        weights = forecast.weights

        # Resample according to weights to get representative samples
        n_particles = len(weights)
        indices = np.random.choice(n_particles, size=n_samples, p=weights, replace=True)

        samples = np.zeros((n_samples, len(param_names)))
        for i, param in enumerate(param_names):
            if param in state_vec.dtype.names:
                samples[:, i] = state_vec[param][indices]

        return samples, param_names

    except Exception as e:
        print(f"Warning: Could not extract samples from particle filter: {e}")
        print("Attempting alternative extraction method...")

        # Alternative: use summary statistics if available
        # This is a fallback - exact implementation depends on your output format
        return None, param_names


def compare_posteriors(npe_samples, pf_samples, param_names, patient_id, output_dir="comparison_plots"):
    """
    Generate comparison plots for NPE vs PF posteriors.

    Args:
        npe_samples: NPE posterior samples (n_samples x n_params)
        pf_samples: PF posterior samples (n_samples x n_params) or None
        param_names: List of parameter names
        patient_id: Patient identifier
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_params = len(param_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, param in enumerate(param_names):
        ax = axes[i]

        # NPE posterior
        ax.hist(npe_samples[:, i], bins=50, alpha=0.5, label='NPE', color='blue', density=True)

        # PF posterior (if available)
        if pf_samples is not None:
            ax.hist(pf_samples[:, i], bins=50, alpha=0.5, label='Particle Filter', color='red', density=True)

        # Add statistics
        npe_mean = np.mean(npe_samples[:, i])
        npe_std = np.std(npe_samples[:, i])
        ax.axvline(npe_mean, color='blue', linestyle='--', linewidth=2, label=f'NPE mean: {npe_mean:.2f}')

        if pf_samples is not None:
            pf_mean = np.mean(pf_samples[:, i])
            ax.axvline(pf_mean, color='red', linestyle='--', linewidth=2, label=f'PF mean: {pf_mean:.2f}')

        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_title(f'{param} (std={npe_std:.2f})')

    plt.suptitle(f'NPE vs Particle Filter Comparison - Patient {patient_id}', fontsize=14)
    plt.tight_layout()

    # Save
    save_path = output_path / f"comparison_{patient_id}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")

    plt.close()


def print_comparison_stats(npe_samples, pf_samples, param_names):
    """
    Print comparison statistics.

    Args:
        npe_samples: NPE posterior samples
        pf_samples: PF posterior samples or None
        param_names: List of parameter names
    """
    print("\nPosterior Comparison Statistics:")
    print("=" * 80)

    if pf_samples is None:
        print("Particle filter samples not available - showing NPE statistics only")
        print("-" * 80)
        print(f"{'Parameter':<10s} {'NPE Mean':>12s} {'NPE Std':>12s} {'NPE 95% CI':>25s}")
        print("-" * 80)

        for i, param in enumerate(param_names):
            npe_mean = np.mean(npe_samples[:, i])
            npe_std = np.std(npe_samples[:, i])
            npe_ci = np.percentile(npe_samples[:, i], [2.5, 97.5])

            print(f"{param:<10s} {npe_mean:>12.4f} {npe_std:>12.4f} [{npe_ci[0]:>8.4f}, {npe_ci[1]:>8.4f}]")

    else:
        print(f"{'Parameter':<10s} {'NPE Mean':>12s} {'PF Mean':>12s} {'Difference':>12s} {'NPE Std':>12s} {'PF Std':>12s}")
        print("-" * 80)

        for i, param in enumerate(param_names):
            npe_mean = np.mean(npe_samples[:, i])
            pf_mean = np.mean(pf_samples[:, i])
            diff = npe_mean - pf_mean

            npe_std = np.std(npe_samples[:, i])
            pf_std = np.std(pf_samples[:, i])

            print(f"{param:<10s} {npe_mean:>12.4f} {pf_mean:>12.4f} {diff:>12.4f} {npe_std:>12.4f} {pf_std:>12.4f}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare NPE and Particle Filter results")
    parser.add_argument('--patient', type=str, required=True,
                        help='Patient ID or "all"')
    parser.add_argument('--pf-dir', type=str, default='outputs4',
                        help='Particle filter output directory')
    parser.add_argument('--npe-dir', type=str, default='npe_outputs',
                        help='NPE output directory')
    parser.add_argument('--output-dir', type=str, default='comparison_plots',
                        help='Output directory for comparison plots')

    args = parser.parse_args()

    # Get patient list
    if args.patient == 'all':
        # Get all patients from NPE output
        npe_inference_dir = Path(args.npe_dir) / "inference"
        if not npe_inference_dir.exists():
            print(f"Error: NPE inference directory not found: {npe_inference_dir}")
            return

        patient_ids = [p.name for p in npe_inference_dir.iterdir() if p.is_dir()]
    else:
        patient_ids = [args.patient]

    print(f"Comparing results for {len(patient_ids)} patient(s)")

    # Process each patient
    for patient_id in patient_ids:
        print(f"\n{'=' * 80}")
        print(f"Patient: {patient_id}")
        print('=' * 80)

        try:
            # Load NPE results
            print("Loading NPE results...")
            npe_results = load_npe_results(patient_id, args.npe_dir)
            npe_samples = npe_results['samples']
            print(f"  Loaded {len(npe_samples)} NPE samples")

            # Load PF results
            print("Loading Particle Filter results...")
            try:
                pf_fit_result = load_particle_filter_results(patient_id, args.pf_dir)
                pf_samples, param_names = extract_pf_samples(pf_fit_result)

                if pf_samples is not None:
                    print(f"  Loaded {len(pf_samples)} PF samples")
                else:
                    print("  Warning: Could not extract PF samples")
            except Exception as e:
                print(f"  Warning: Could not load PF results: {e}")
                pf_samples = None
                param_names = npe_results['summary']['parameter'].tolist()

            # Print comparison
            print_comparison_stats(npe_samples, pf_samples, param_names)

            # Generate plots
            compare_posteriors(npe_samples, pf_samples, param_names, patient_id, args.output_dir)

        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}")
    print("Comparison complete!")
    print('=' * 80)


if __name__ == '__main__':
    main()
