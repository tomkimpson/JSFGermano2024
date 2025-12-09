import marimo

__generated_with = "0.9.34"
app = marimo.App(width="medium")


@app.cell
def __():
    """
    Section 1: Marginalised Posterior Plots

    This section recreates the marginalised posterior plots for multiple patients
    in a grid layout, showing 1D marginalised posteriors for each parameter.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from pathlib import Path
    from scipy.stats import gaussian_kde

    try:
        import scienceplots
    except ImportError:
        print("Warning: scienceplots package not found. Install with: pip install scienceplots")
        print("Continuing without scienceplots styling...")
        scienceplots = None

    return np, plt, mpl, Path, gaussian_kde, scienceplots


@app.cell
def __():
    """Configuration: Set paths and parameters"""

    # Input directory containing patient subdirectories with samples.npy files
    INPUT_DIR = "../results/npe/20250103_existing_primary/inference"

    # Output filename for the plot
    OUTPUT_FILENAME = "marginalised_posteriors.png"

    # Figure size (width, height) in inches
    FIGSIZE = (20, 24)

    return INPUT_DIR, OUTPUT_FILENAME, FIGSIZE


@app.cell
def __(np):
    """Helper function: Transform samples for display"""

    def transform_samples(samples: np.ndarray) -> np.ndarray:
        """
        Transform and reorder parameter samples for display.

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

    return transform_samples,


@app.cell
def __(np):
    """Helper function: Plot KDE with statistics"""

    def plot_kde_with_stats(ax, samples, param_label, patient_id, color='teal', show_title=True):
        """
        Plot a filled KDE with mean and 95% credible interval.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes object to plot on
        samples : np.ndarray
            1D array of parameter samples
        param_label : str
            Parameter label for x-axis
        patient_id : str
            Patient ID for subplot title
        color : str
            Color for the KDE fill
        show_title : bool
            Whether to show the patient ID as subplot title
        """
        from scipy.stats import gaussian_kde

        # Compute statistics
        mean_val = np.mean(samples)
        ci_lower = np.percentile(samples, 2.5)
        ci_upper = np.percentile(samples, 97.5)

        # Compute KDE
        kde = gaussian_kde(samples)

        # Create smooth x-axis for KDE evaluation
        x_min = np.min(samples)
        x_max = np.max(samples)
        x_range = x_max - x_min
        x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 500)

        # Evaluate KDE
        density = kde(x_eval)

        # Plot filled KDE
        ax.fill_between(x_eval, density, alpha=0.6, color=color, edgecolor=color, linewidth=1.5)

        # Add mean line
        ax.axvline(mean_val, color='darkred', linestyle='--', linewidth=1.5,
                   label=f'Mean: {mean_val:.2f}', zorder=10)

        # Add 95% credible interval lines
        ax.axvline(ci_lower, color='navy', linestyle=':', linewidth=1.2,
                   alpha=0.7, zorder=10)
        ax.axvline(ci_upper, color='navy', linestyle=':', linewidth=1.2,
                   alpha=0.7, label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]', zorder=10)

        # Styling
        ax.set_xlabel(param_label, fontsize=16)
        ax.set_ylabel('Density', fontsize=16)
        if show_title:
            ax.set_title(f'Patient {patient_id}', fontsize=18)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=14)

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

    return plot_kde_with_stats,


@app.cell
def __(np, plt, mpl, Path, scienceplots, transform_samples, plot_kde_with_stats, INPUT_DIR, OUTPUT_FILENAME, FIGSIZE):
    """Main function: Generate marginalised posteriors grid"""

    def plot_marginalised_posteriors_grid(
        input_dir: str,
        output_filename: str,
        figsize: tuple
    ) -> Path:
        """
        Generate a grid of marginalised posterior plots for all patients.

        Creates a 6×6 grid where each row represents a different patient and
        each column represents a different parameter.

        Parameters:
        -----------
        input_dir : str
            Directory containing patient subdirectories with samples.npy files
        output_filename : str
            Name for output plot file
        figsize : tuple
            Figure size (width, height) in inches

        Returns:
        --------
        Path
            Path to saved plot
        """
        # Apply publication-quality styling
        if scienceplots is not None:
            plt.style.use('science')

        # Disable LaTeX if it causes issues
        mpl.rcParams['text.usetex'] = False

        # Get all patient directories
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find patient directories
        patient_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
        patient_ids = [d.name for d in patient_dirs]

        if not patient_ids:
            raise ValueError(f"No patient directories found in {input_dir}")

        print(f"Found {len(patient_ids)} patients: {', '.join(patient_ids)}")

        # Parameter labels in display order
        param_labels = [
            r'$\beta$',
            r'$\rho$',
            r'$\pi$',
            r'$\phi$',
            r'$\delta$',
            r'$\log_{10}V_0$'
        ]

        # Number of patients and parameters
        n_patients = len(patient_ids)
        n_params = 6

        # Create figure with subplots
        fig, axes = plt.subplots(n_patients, n_params, figsize=figsize)

        # Ensure axes is 2D even if there's only one patient
        if n_patients == 1:
            axes = axes.reshape(1, -1)

        print("\nGenerating marginalised posterior plots...")

        # Loop through patients (rows)
        for i, patient_id in enumerate(patient_ids):
            patient_dir = patient_dirs[i]
            samples_path = patient_dir / "samples.npy"

            if not samples_path.exists():
                print(f"Warning: Samples file not found for patient {patient_id}, skipping...")
                # Clear the row of subplots
                for j in range(n_params):
                    axes[i, j].text(0.5, 0.5, 'No data', ha='center', va='center',
                                   transform=axes[i, j].transAxes)
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                continue

            # Load and transform samples
            print(f"  Processing patient {patient_id}...")
            samples = np.load(samples_path)
            display_samples = transform_samples(samples)

            # Loop through parameters (columns)
            for j in range(n_params):
                ax = axes[i, j]
                param_samples = display_samples[:, j]

                # Plot KDE with statistics
                # Only show patient ID title for top row
                plot_kde_with_stats(
                    ax,
                    param_samples,
                    param_labels[j],
                    patient_id,
                    color='teal',
                    show_title=(i == 0)
                )

                # Only show x-label and x-tick labels on bottom row
                if i < n_patients - 1:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])

                # Only show y-label on leftmost column
                if j > 0:
                    ax.set_ylabel('')

        # Adjust layout
        plt.tight_layout()

        # Save figure
        output_path = input_path / output_filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        print(f"\nMarginalised posterior plot saved to: {output_path}")
        return fig, output_path

    # Generate the plot
    fig_marginals, output_path_marginals = plot_marginalised_posteriors_grid(
        input_dir=INPUT_DIR,
        output_filename=OUTPUT_FILENAME,
        figsize=FIGSIZE
    )

    return plot_marginalised_posteriors_grid, fig_marginals, output_path_marginals


@app.cell
def __(fig_marginals):
    """Display the plot"""
    fig_marginals
    return


if __name__ == "__main__":
    app.run()
