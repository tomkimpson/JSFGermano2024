#!/usr/bin/env python3
"""
Generate corner plots for particle filter posterior samples.

This script loads posterior samples from particle filter results and creates
corner plots with parameter transformations and proper visualization.

Usage:
    python plot_corner_particle_filter.py --patient 432192
    python plot_corner_particle_filter.py --all
    python plot_corner_particle_filter.py --patient 432192 --particles 1000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle

try:
    import corner
except ImportError:
    print("Error: corner package not found. Install with: pip install corner")
    sys.exit(1)


def transform_samples(samples: np.ndarray) -> np.ndarray:
    """
    Transform and reorder parameter samples for display in corner plots.

    Converts from internal order [lnV0, beta, phi, rho, delta, pi] to
    display order [beta, rho, pi, phi, delta, log10(V0)] and applies
    necessary transformations.

    Parameters:
    -----------
    samples : np.ndarray
        Parameter samples with shape (n_samples, 6) in order:
        [lnV0, beta, phi, rho, delta, pi]

    Returns:
    --------
    np.ndarray
        Transformed samples with shape (n_samples, 6) in order:
        [beta, rho, pi, phi, delta, log10(V0)]
    """
    # Extract parameters from internal order [lnV0, beta, phi, rho, delta, pi]
    lnV0 = samples[:, 0]
    beta = samples[:, 1]
    phi = samples[:, 2]
    rho = samples[:, 3]
    delta = samples[:, 4]
    pi = samples[:, 5]

    # Transform lnV0 to log10(V0)
    # V0 = exp(lnV0), then log10(V0) = lnV0 / ln(10)
    # Clamp to avoid log(0) = -inf
    V0 = np.exp(lnV0)
    V0_safe = np.clip(V0, 1e-10, None)
    log10_V0 = np.log10(V0_safe)

    # Reorder to display order [beta, rho, pi, phi, delta, log10(V0)]
    display_samples = np.column_stack([
        beta,       # β (position 0)
        rho,        # ρ (position 1)
        pi,         # π (position 2)
        phi,        # φ (position 3)
        delta,      # δ (position 4)
        log10_V0    # log₁₀V₀ (position 5)
    ])

    return display_samples


def load_particle_filter_samples(
    patient_id: str,
    input_dir: str = "results/particle_filter/<timestamp>",
    n_particles: int = 6000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load particle filter samples from pickle file.

    Parameters:
    -----------
    patient_id : str
        Patient ID
    input_dir : str
        Directory containing patient results
    n_particles : int
        Number of particles used in the run (1000 or 6000)

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        samples: Parameter samples with shape (n_samples, 6)
        weights: Sample weights with shape (n_samples,)
    """
    # Construct path to pickle file
    patient_dir = Path(input_dir) / patient_id
    run_dir = patient_dir / f"src.tiv.RefractoryCellModel_JSF_{n_particles}"
    pickle_path = run_dir / "fit_result.pkl"

    # Check if file exists
    if not pickle_path.exists():
        raise FileNotFoundError(f"Results file not found: {pickle_path}")

    print(f"  Loading results from: {pickle_path}")

    # Load pickle file
    with open(pickle_path, 'rb') as f:
        fit_result = pickle.load(f)

    # Extract snapshot table
    snapshot = fit_result.estimation.tables['snapshot']

    # Get final time particles
    times = np.unique(snapshot['time'])
    final_time = times[-1]
    final_particles = snapshot[snapshot['time'] == final_time]

    print(f"  Loaded {len(final_particles)} particles at final time t={final_time}")

    # Extract parameter samples in order: [lnV0, beta, phi, rho, delta, pi]
    param_names = ['lnV0', 'beta', 'phi', 'rho', 'delta', 'pi']
    samples = np.column_stack([final_particles[p] for p in param_names])

    # Extract weights
    weights = final_particles['weight']

    print(f"  Weight statistics: min={weights.min():.6f}, max={weights.max():.6f}, "
          f"sum={weights.sum():.6f}")

    return samples, weights


def resample_if_needed(
    samples: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.01
) -> np.ndarray:
    """
    Resample particles if weights are non-uniform.

    If the coefficient of variation of weights exceeds the threshold,
    resample particles according to their weights. Otherwise, return
    samples as-is.

    Parameters:
    -----------
    samples : np.ndarray
        Parameter samples with shape (n_samples, n_params)
    weights : np.ndarray
        Sample weights with shape (n_samples,)
    threshold : float
        Coefficient of variation threshold for resampling

    Returns:
    --------
    np.ndarray
        Resampled (or original) samples
    """
    # Calculate coefficient of variation
    cv = np.std(weights) / np.mean(weights)

    if cv > threshold:
        print(f"  Weights are non-uniform (CV={cv:.4f}), resampling...")
        # Normalize weights
        weights_norm = weights / weights.sum()
        # Resample
        n_samples = len(samples)
        indices = np.random.choice(n_samples, size=n_samples, replace=True, p=weights_norm)
        resampled = samples[indices]
        return resampled
    else:
        print(f"  Weights are approximately uniform (CV={cv:.4f}), using all samples")
        return samples


def plot_corner_for_patient(
    patient_id: str,
    input_dir: str = "results/particle_filter/<timestamp>",
    n_particles: int = 6000,
    output_filename: str = "corner_plot.png",
    smooth: float = 1.0,
    **corner_kwargs
) -> Path:
    """
    Generate corner plot for a single patient's particle filter posterior samples.

    Parameters:
    -----------
    patient_id : str
        Patient ID
    input_dir : str
        Directory containing patient results
    n_particles : int
        Number of particles used in the run (1000 or 6000)
    output_filename : str
        Name for output corner plot file
    smooth : float
        Smoothing parameter for corner plots
    **corner_kwargs : additional arguments for corner.corner

    Returns:
    --------
    Path
        Path to saved corner plot
    """
    print(f"\nProcessing patient {patient_id} (n_particles={n_particles})...")

    # Load samples and weights
    samples, weights = load_particle_filter_samples(patient_id, input_dir, n_particles)

    # Resample if weights are non-uniform
    samples = resample_if_needed(samples, weights)

    # Transform and reorder parameters
    display_samples = transform_samples(samples)

    # Parameter labels in display order
    param_labels = [
        r'$\beta$',
        r'$\rho$',
        r'$\pi$',
        r'$\phi$',
        r'$\delta$',
        r'$\log_{10}V_0$'
    ]

    # Prior bounds matching exact prior ranges from config
    # Display order: [β, ρ, π, φ, δ, log₁₀V₀]
    # Note: β and φ are scaled (×10⁻⁹ and ×10⁻⁵ respectively) in the model
    # but the priors in the config are defined in the scaled space
    prior_bounds = [
        (0.0, 20.0),      # β: Uniform(0, 20)
        (0.0, 1.0),       # ρ: Uniform(0, 1)
        (200.0, 600.0),   # π: Uniform(200, 600)
        (0.0, 15.0),      # φ: Uniform(0, 15)
        (1.0, 11.0),      # δ: Uniform(1, 11)
        (0.0, 2.17)       # log₁₀V₀: V₀ = exp(Uniform(0,5)) → log₁₀(V₀) ∈ [0, 2.17]
    ]

    # Default corner plot settings
    default_kwargs = {
        'labels': param_labels,
        'color': 'teal',
        'bins': 30,
        'range': prior_bounds,
        'plot_datapoints': True,
        'plot_density': True,
        'plot_contours': True,
        'data_kwargs': {'alpha': 0.2, 'color': 'lightblue'},
        'hist_kwargs': {'alpha': 0.8, 'color': 'teal'},
        'contour_kwargs': {'colors': 'teal'},
        'smooth': smooth,
        'smooth1d': smooth,
        'quantiles': [0.16, 0.5, 0.84],
        'show_titles': True,
        'title_kwargs': {"fontsize": 12},
        'label_kwargs': {"fontsize": 14}
    }

    # Update with user-provided kwargs
    default_kwargs.update(corner_kwargs)

    # Create corner plot
    print(f"  Generating corner plot...")
    fig = corner.corner(display_samples, **default_kwargs)

    # Add patient ID and particle count as suptitle
    fig.suptitle(f'Patient {patient_id} - Particle Filter Posterior (N={n_particles})',
                 fontsize=16, y=0.995)

    # Construct output path
    patient_dir = Path(input_dir) / patient_id
    run_dir = patient_dir / f"src.tiv.RefractoryCellModel_JSF_{n_particles}"
    output_path = run_dir / output_filename

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Corner plot saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate corner plots for particle filter posterior samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate corner plot for a single patient (6000 particles)
  python plot_corner_particle_filter.py --patient 432192 --input-dir results/particle_filter/20250103_143022

  # Generate corner plot for 1000 particle run
  python plot_corner_particle_filter.py --patient 432192 --input-dir results/particle_filter/20250103_143022 --particles 1000

  # Generate corner plots for all patients
  python plot_corner_particle_filter.py --all --input-dir results/particle_filter/20250103_143022
        """
    )

    parser.add_argument(
        '--patient',
        type=str,
        help='Patient ID to process'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all patients in the input directory'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=None,
        required=True,
        help='Input directory containing patient results (e.g., results/particle_filter/20250103_143022)'
    )
    parser.add_argument(
        '--particles',
        type=int,
        default=6000,
        choices=[1000, 6000],
        help='Number of particles used in the run (default: 6000)'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='corner_plot.png',
        help='Output filename for corner plots (default: corner_plot.png)'
    )
    parser.add_argument(
        '--smooth',
        type=float,
        default=1.0,
        help='Smoothing parameter for corner plots (default: 1.0)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.patient:
        parser.error("Must specify either --patient or --all")

    if args.all and args.patient:
        parser.error("Cannot specify both --patient and --all")

    # Get list of patients to process
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if args.all:
        # Get all patient directories
        patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        patient_ids = [d.name for d in patient_dirs]

        if not patient_ids:
            print(f"Error: No patient directories found in {input_dir}")
            sys.exit(1)

        patient_ids.sort()
        print(f"Found {len(patient_ids)} patients: {', '.join(patient_ids)}")
    else:
        patient_ids = [args.patient]

    # Process each patient
    print(f"\n{'=' * 70}")
    print("GENERATING CORNER PLOTS FOR PARTICLE FILTER RESULTS")
    print(f"{'=' * 70}")

    success_count = 0
    failed_patients = []

    for patient_id in patient_ids:
        try:
            output_path = plot_corner_for_patient(
                patient_id,
                input_dir=str(input_dir),
                n_particles=args.particles,
                output_filename=args.output_filename,
                smooth=args.smooth
            )
            success_count += 1
        except Exception as e:
            print(f"  Error processing patient {patient_id}: {e}")
            failed_patients.append(patient_id)
            continue

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Successfully processed: {success_count}/{len(patient_ids)} patients")

    if failed_patients:
        print(f"Failed patients: {', '.join(failed_patients)}")

    print(f"\nDone!")


if __name__ == '__main__':
    main()
