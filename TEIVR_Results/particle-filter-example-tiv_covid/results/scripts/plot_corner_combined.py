#!/usr/bin/env python3
"""
Generate combined corner plots comparing NPE and particle filter posterior samples.

This script loads posterior samples from both NPE inference and particle filter
results, and creates overlaid corner plots for direct visual comparison.

Usage:
    python plot_corner_combined.py --patient 432192
    python plot_corner_combined.py --patient 432192 --particles 1000
    python plot_corner_combined.py --patient 432192 --match-samples
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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


def load_npe_samples(
    patient_id: str,
    input_dir: str = "results/npe/<timestamp>/inference"
) -> np.ndarray:
    """
    Load NPE samples from numpy file.

    Parameters:
    -----------
    patient_id : str
        Patient ID
    input_dir : str
        Directory containing NPE inference results

    Returns:
    --------
    np.ndarray
        Parameter samples with shape (n_samples, 6)
    """
    # Construct path to samples file
    patient_dir = Path(input_dir) / patient_id
    samples_path = patient_dir / "samples.npy"

    # Check if file exists
    if not samples_path.exists():
        raise FileNotFoundError(f"NPE samples file not found: {samples_path}")

    print(f"  Loading NPE samples from: {samples_path}")

    # Load samples
    samples = np.load(samples_path)
    print(f"  Loaded {samples.shape[0]} NPE samples with {samples.shape[1]} parameters")

    return samples


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
        raise FileNotFoundError(f"Particle filter results file not found: {pickle_path}")

    print(f"  Loading particle filter results from: {pickle_path}")

    # Load pickle file
    with open(pickle_path, 'rb') as f:
        fit_result = pickle.load(f)

    # Extract snapshot table
    snapshot = fit_result.estimation.tables['snapshot']

    # Get final time particles
    times = np.unique(snapshot['time'])
    final_time = times[-1]
    final_particles = snapshot[snapshot['time'] == final_time]

    print(f"  Loaded {len(final_particles)} particle filter samples at final time t={final_time}")

    # Extract parameter samples in order: [lnV0, beta, phi, rho, delta, pi]
    param_names = ['lnV0', 'beta', 'phi', 'rho', 'delta', 'pi']
    samples = np.column_stack([final_particles[p] for p in param_names])

    # Extract weights
    weights = final_particles['weight']

    return samples, weights


def resample_particles(
    samples: np.ndarray,
    weights: np.ndarray,
    n_samples: int
) -> np.ndarray:
    """
    Resample particles according to weights.

    Parameters:
    -----------
    samples : np.ndarray
        Parameter samples with shape (n_samples, n_params)
    weights : np.ndarray
        Sample weights with shape (n_samples,)
    n_samples : int
        Number of samples to draw

    Returns:
    --------
    np.ndarray
        Resampled samples with shape (n_samples, n_params)
    """
    # Normalize weights
    weights_norm = weights / weights.sum()

    # Resample
    indices = np.random.choice(len(samples), size=n_samples, replace=True, p=weights_norm)
    resampled = samples[indices]

    return resampled


def plot_combined_corner(
    patient_id: str,
    npe_dir: str = "results/npe/<timestamp>/inference",
    pf_dir: str = "results/particle_filter/<timestamp>",
    n_particles: int = 6000,
    output_dir: str = None,
    output_filename: str = "corner_plot_combined.png",
    match_samples: bool = False,
    smooth: float = 1.0
) -> Path:
    """
    Generate combined corner plot comparing NPE and particle filter posteriors.

    Parameters:
    -----------
    patient_id : str
        Patient ID
    npe_dir : str
        Directory containing NPE inference results
    pf_dir : str
        Directory containing particle filter results
    n_particles : int
        Number of particles used in the PF run (1000 or 6000)
    output_dir : str
        Output directory (if None, uses particle filter run directory)
    output_filename : str
        Name for output corner plot file
    match_samples : bool
        If True, resample to match sample sizes
    smooth : float
        Smoothing parameter for corner plots

    Returns:
    --------
    Path
        Path to saved corner plot
    """
    print(f"\nProcessing patient {patient_id}...")
    print("=" * 70)

    # Load NPE samples
    npe_samples = load_npe_samples(patient_id, npe_dir)

    # Load particle filter samples
    pf_samples, pf_weights = load_particle_filter_samples(patient_id, pf_dir, n_particles)

    # Match sample sizes if requested
    if match_samples:
        n_npe = len(npe_samples)
        n_pf = len(pf_samples)

        if n_npe > n_pf:
            # Resample NPE to match PF
            print(f"\n  Resampling NPE from {n_npe} to {n_pf} samples...")
            npe_samples = npe_samples[np.random.choice(n_npe, size=n_pf, replace=False)]
        elif n_pf > n_npe:
            # Resample PF to match NPE
            print(f"\n  Resampling particle filter from {n_pf} to {n_npe} samples...")
            pf_samples = resample_particles(pf_samples, pf_weights, n_npe)
    else:
        # Just use all PF samples (resampled by weights)
        print(f"\n  Using all {len(npe_samples)} NPE samples and {len(pf_samples)} PF samples")

    # Transform both to display space
    print("\n  Transforming samples to display space...")
    npe_display = transform_samples(npe_samples)
    pf_display = transform_samples(pf_samples)

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
    prior_bounds = [
        (0.0, 20.0),      # β: Uniform(0, 20)
        (0.0, 1.0),       # ρ: Uniform(0, 1)
        (200.0, 600.0),   # π: Uniform(200, 600)
        (0.0, 15.0),      # φ: Uniform(0, 15)
        (1.0, 11.0),      # δ: Uniform(1, 11)
        (0.0, 2.17)       # log₁₀V₀: V₀ = exp(Uniform(0,5)) → log₁₀(V₀) ∈ [0, 2.17]
    ]

    # Create figure with NPE samples first (blue/teal)
    print("\n  Generating combined corner plot...")

    npe_kwargs = {
        'labels': param_labels,
        'color': 'steelblue',
        'bins': 30,
        'range': prior_bounds,
        'plot_datapoints': True,
        'plot_density': True,
        'plot_contours': True,
        'data_kwargs': {'alpha': 0.15, 'color': 'lightblue'},
        'hist_kwargs': {'alpha': 0.6, 'color': 'steelblue', 'linewidth': 2},
        'contour_kwargs': {'colors': 'steelblue', 'alpha': 0.6},
        'smooth': smooth,
        'smooth1d': smooth,
        'quantiles': [0.16, 0.5, 0.84],
        'show_titles': False,  # Disable to avoid clutter with two datasets
        'label_kwargs': {"fontsize": 14}
    }

    # Create base corner plot with NPE
    # Normalize using weights so histograms are comparable
    npe_weights = np.ones(len(npe_display)) / len(npe_display)
    fig = corner.corner(npe_display, weights=npe_weights, **npe_kwargs)

    # Overlay particle filter samples (orange/red)
    pf_kwargs = {
        'color': 'darkorange',
        'bins': 30,
        'range': prior_bounds,
        'plot_datapoints': True,
        'plot_density': True,
        'plot_contours': True,
        'data_kwargs': {'alpha': 0.15, 'color': 'lightsalmon'},
        'hist_kwargs': {'alpha': 0.6, 'color': 'darkorange', 'linewidth': 2},
        'contour_kwargs': {'colors': 'darkorange', 'alpha': 0.6},
        'smooth': smooth,
        'smooth1d': smooth,
        'quantiles': [0.16, 0.5, 0.84],
        'show_titles': False,
    }

    # Overlay on existing figure
    # Normalize using weights so histograms are comparable
    pf_weights = np.ones(len(pf_display)) / len(pf_display)
    corner.corner(pf_display, weights=pf_weights, fig=fig, **pf_kwargs)

    # Add title
    fig.suptitle(f'Patient {patient_id} - NPE vs Particle Filter (N={n_particles})',
                 fontsize=16, y=0.995)

    # Add legend
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.6, label=f'NPE (n={len(npe_display)})'),
        Patch(facecolor='darkorange', alpha=0.6, label=f'Particle Filter (n={len(pf_display)})')
    ]

    # Place legend in the top-right area
    axes = fig.get_axes()
    # Use the top-right subplot for legend placement
    axes[1].legend(handles=legend_elements, loc='upper right',
                   fontsize=12, framealpha=0.9)

    # Determine output path
    if output_dir is None:
        # Default: save in particle filter run directory
        patient_dir = Path(pf_dir) / patient_id
        run_dir = patient_dir / f"src.tiv.RefractoryCellModel_JSF_{n_particles}"
        output_path = run_dir / output_filename
    else:
        output_path = Path(output_dir) / output_filename

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Combined corner plot saved to: {output_path}")
    print("=" * 70)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined corner plots comparing NPE and particle filter posteriors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate combined corner plot for patient 432192 (6000 particles)
  python plot_corner_combined.py --patient 432192 \\
    --npe-dir results/npe/20250103_143022/inference \\
    --pf-dir results/particle_filter/20250103_143022

  # Use 1000 particle run for comparison
  python plot_corner_combined.py --patient 432192 --particles 1000 \\
    --npe-dir results/npe/20250103_143022/inference \\
    --pf-dir results/particle_filter/20250103_143022

  # Match sample sizes (resample to same number)
  python plot_corner_combined.py --patient 432192 --match-samples \\
    --npe-dir results/npe/20250103_143022/inference \\
    --pf-dir results/particle_filter/20250103_143022
        """
    )

    parser.add_argument(
        '--patient',
        type=str,
        required=True,
        help='Patient ID to process'
    )
    parser.add_argument(
        '--npe-dir',
        type=str,
        default=None,
        required=True,
        help='Directory containing NPE inference results (e.g., results/npe/20250103_143022/inference)'
    )
    parser.add_argument(
        '--pf-dir',
        type=str,
        default=None,
        required=True,
        help='Directory containing particle filter results (e.g., results/particle_filter/20250103_143022)'
    )
    parser.add_argument(
        '--particles',
        type=int,
        default=6000,
        choices=[1000, 6000],
        help='Number of particles used in the particle filter run (default: 6000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: particle filter run directory)'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        default='corner_plot_combined.png',
        help='Output filename for corner plot (default: corner_plot_combined.png)'
    )
    parser.add_argument(
        '--match-samples',
        action='store_true',
        help='Resample to match sample sizes between NPE and particle filter'
    )
    parser.add_argument(
        '--smooth',
        type=float,
        default=1.0,
        help='Smoothing parameter for corner plots (default: 1.0)'
    )

    args = parser.parse_args()

    # Validate directories exist
    npe_dir = Path(args.npe_dir)
    pf_dir = Path(args.pf_dir)

    if not npe_dir.exists():
        print(f"Error: NPE directory not found: {npe_dir}")
        sys.exit(1)

    if not pf_dir.exists():
        print(f"Error: Particle filter directory not found: {pf_dir}")
        sys.exit(1)

    # Generate combined corner plot
    print("\n" + "=" * 70)
    print("GENERATING COMBINED CORNER PLOT")
    print("=" * 70)

    try:
        output_path = plot_combined_corner(
            patient_id=args.patient,
            npe_dir=str(npe_dir),
            pf_dir=str(pf_dir),
            n_particles=args.particles,
            output_dir=args.output_dir,
            output_filename=args.output_filename,
            match_samples=args.match_samples,
            smooth=args.smooth
        )
        print(f"\nSuccess! Combined corner plot created.")
        print(f"View at: {output_path}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
