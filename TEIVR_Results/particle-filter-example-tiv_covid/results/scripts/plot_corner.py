#!/usr/bin/env python3
"""
Generate corner plots for NPE posterior samples.

This script loads posterior samples from NPE inference results and creates
corner plots with parameter transformations and proper visualization.

Usage:
    python plot_corner.py --patient 432192
    python plot_corner.py --all
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    import corner
except ImportError:
    print("Error: corner package not found. Install with: pip install corner")
    sys.exit(1)

try:
    import scienceplots
except ImportError:
    print("Warning: scienceplots package not found. Install with: pip install scienceplots")
    print("Continuing without scienceplots styling...")
    scienceplots = None


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


def plot_corner_for_patient(
    patient_id: str,
    input_dir: str = "results/npe/<timestamp>/inference",
    output_filename: str = "corner_plot.png",
    smooth: float = 1.0,
    **corner_kwargs
) -> Path:
    """
    Generate corner plot for a single patient's posterior samples.

    Parameters:
    -----------
    patient_id : str
        Patient ID
    input_dir : str
        Directory containing patient results
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
    # Construct paths
    patient_dir = Path(input_dir) / patient_id
    samples_path = patient_dir / "samples.npy"
    output_path = patient_dir / output_filename

    # Check if samples file exists
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    print(f"\nProcessing patient {patient_id}...")
    print(f"  Loading samples from: {samples_path}")

    # Apply publication-quality styling
    if scienceplots is not None:
        plt.style.use('science')

    # Load samples
    samples = np.load(samples_path)
    print(f"  Loaded {samples.shape[0]} samples with {samples.shape[1]} parameters")

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
        (0.0, 25.0),      # β: Extended range for better visualization
        (-0.1, 1.0),      # ρ: Extended range for better visualization
        (200.0, 600.0),   # π: Uniform(200, 600)
        (-2.0, 20.0),     # φ: Extended range for better visualization
        (-0.2, 12.0),     # δ: Extended range for better visualization
        (-0.1, 3.0)       # log₁₀V₀: Extended range for better visualization
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
        'title_kwargs': {"fontsize": 20},
        'label_kwargs': {"fontsize": 22}
    }

    # Update with user-provided kwargs
    default_kwargs.update(corner_kwargs)

    # Create corner plot
    print(f"  Generating corner plot...")
    fig = corner.corner(display_samples, **default_kwargs)

    # Add patient ID as suptitle
    fig.suptitle(f'Patient {patient_id} - NPE Posterior',
                 fontsize=24, y=0.995)

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Corner plot saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate corner plots for NPE posterior samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate corner plot for a single patient (specify timestamped directory)
  python plot_corner.py --patient 432192 --input-dir results/npe/20250103_143022/inference

  # Generate corner plots for all patients
  python plot_corner.py --all --input-dir results/npe/20250103_143022/inference
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
        help='Input directory containing patient results (e.g., results/npe/20250103_143022/inference)'
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
    print("GENERATING CORNER PLOTS")
    print(f"{'=' * 70}")

    success_count = 0
    failed_patients = []

    for patient_id in patient_ids:
        try:
            output_path = plot_corner_for_patient(
                patient_id,
                input_dir=str(input_dir),
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
