import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    """Import marimo for markdown functionality"""
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    """Notebook introduction"""
    mo.md(
        """
        # Plots for Paper

        This notebook generates all the publication-ready plots for the TEIVR analysis paper.
        Each section below produces a specific figure or set of figures that will be included
        in the final manuscript.

        ## Overview

        The notebook is organized into sections, each corresponding to a specific analysis
        or comparison:

        1. **Marginalised Posterior Distributions** - Grid of 1D marginalised posteriors for all patients
        2. **Corner Plots** - 2D posterior correlations and 1D marginals for all patients
        3. _(Additional sections to be added)_

        All plots are saved with high resolution (300 DPI) and publication-quality styling
        using the `scienceplots` package.
        """
    )
    return


@app.cell
def _(mo):
    """Section 1 introduction"""
    mo.md(
        """
        ---

        ## Section 1: Marginalised Posterior Distributions

        This section generates a comprehensive grid visualization showing the marginalised
        posterior distributions for all six model parameters across all patients.

        **Model Parameters:**
        - β (beta): Viral production rate
        - ρ (rho): Death rate of infected cells
        - π (pi): Clearance rate of virus
        - φ (phi): Infection rate
        - δ (delta): Death rate of target cells
        - log₁₀V₀: Initial viral load (log-transformed)

        Each subplot displays:
        - Kernel Density Estimate (KDE) of the posterior distribution
        - Mean value (red dashed line)
        - 95% credible interval (blue dotted lines)

        The grid layout shows one row per patient and one column per parameter,
        making it easy to compare posterior distributions across patients and parameters.
        """
    )
    return


@app.cell
def _():
    """Section 1: Import libraries for marginalised posteriors"""
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from pathlib import Path
    from scipy.stats import gaussian_kde

    try:
        import scienceplots
        print("Scienceplots package imported successfully")
        plt.style.use('science') #Set once, globally.
        SCIENCE_RCPARAMS = dict(plt.rcParams)
    except ImportError:
        print("Warning: scienceplots package not found. Install with: pip install scienceplots")
        print("Continuing without scienceplots styling...")
        scienceplots = None
    return Path, SCIENCE_RCPARAMS, mpl, np, plt


@app.cell
def _():
    """Configuration: Set paths and parameters"""

    # Input directory containing patient subdirectories with samples.npy files
    INPUT_DIR = "../results/npe/20250103_existing_primary/inference"

    # Output filename for the plot
    OUTPUT_FILENAME = "marginalised_posteriors_canonical.png"

    # Figure size (width, height) in inches
    FIGSIZE = (20, 24)
    return FIGSIZE, INPUT_DIR, OUTPUT_FILENAME


@app.cell
def _(np):
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
    return (transform_samples,)


@app.cell
def _(np):
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
    return (plot_kde_with_stats,)


@app.cell
def _(
    FIGSIZE,
    INPUT_DIR,
    OUTPUT_FILENAME,
    Path,
    mpl,
    np,
    plot_kde_with_stats,
    plt,
    transform_samples,
):
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
    return (fig_marginals,)


@app.cell
def _(fig_marginals):
    """Display the plot"""
    fig_marginals
    return


@app.cell
def _(mo):
    """Section 2 introduction"""
    mo.md(
        """
        ---

        ## Section 2: Corner Plots

        This section generates corner plots for all patients, showing both 2D posterior
        correlations between parameters and 1D marginal distributions.

        **Model Parameters:**
        - β (beta): Viral production rate
        - ρ (rho): Death rate of infected cells
        - π (pi): Clearance rate of virus
        - φ (phi): Infection rate
        - δ (delta): Death rate of target cells
        - log₁₀V₀: Initial viral load (log-transformed)

        **Visualization Features:**
        - **1D histograms** on the diagonal showing marginal distributions
        - **2D contour plots** in off-diagonal panels showing parameter correlations
        - **Quantiles** displayed: 16th, 50th (median), and 84th percentiles
        - **Mean and credible intervals** shown in subplot titles

        One corner plot is generated for each patient, displaying the full joint
        posterior distribution and revealing correlations between parameters that
        may not be apparent from marginal distributions alone.
        """
    )
    return


@app.cell
def _():
    """Section 2: Import corner library"""
    try:
        import corner
    except ImportError:
        print("Error: corner package not found. Install with: pip install corner")
        corner = None
    return (corner,)


@app.cell
def _():
    """Section 2: Configuration for corner plots"""

    # Output filename for corner plots (saved per patient)
    CORNER_OUTPUT_FILENAME = "corner_plot_npe_canonical.png"

    # Smoothing parameter for KDE in corner plots
    CORNER_SMOOTH = 1.0
    return CORNER_OUTPUT_FILENAME, CORNER_SMOOTH


@app.cell
def _(Path, SCIENCE_RCPARAMS, corner, np, plt, transform_samples):
    """Section 2: Corner plot function"""

    def plot_corner_for_patient(
        patient_id: str,
        input_dir: str,
        output_filename: str,
        smooth: float = 1.0
    ) -> tuple:
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

        Returns:
        --------
        tuple
            (figure, output_path) or (None, None) if error
        """
        # Construct paths
        patient_dir = Path(input_dir) / patient_id
        samples_path = patient_dir / "samples.npy"
        output_path = patient_dir / output_filename

        # Check if samples file exists
        if not samples_path.exists():
            print(f"Warning: Samples file not found for patient {patient_id}, skipping...")
            return None, None

        print(f"  Processing patient {patient_id}...")


        # Load samples
        samples = np.load(samples_path)

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
        prior_bounds = [
            (0.0, 25.0),      # β: Extended range for better visualization
            (-0.1, 1.0),      # ρ: Extended range for better visualization
            (200.0, 600.0),   # π: Uniform(200, 600)
            (-2.0, 20.0),     # φ: Extended range for better visualization
            (-0.2, 12.0),     # δ: Extended range for better visualization
            (-0.1, 3.0)       # log₁₀V₀: Extended range for better visualization
        ]

        # Corner plot settings
        corner_kwargs = {
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

        # Create corner plot
        plt.rcParams.update(SCIENCE_RCPARAMS) #set science plot parameters manually. Corner seems to override things and handiling with marimo reactivity is a bit tricky. This option is a bit hacky but works nicely
        fig = corner.corner(display_samples, **corner_kwargs)

        # Add patient ID as suptitle
        fig.suptitle(f'Patient {patient_id} - NPE Posterior',
                     fontsize=24, y=0.995)

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        print(f"  Corner plot saved to: {output_path}")

        return fig, output_path
    return (plot_corner_for_patient,)


@app.cell
def _(
    CORNER_OUTPUT_FILENAME,
    CORNER_SMOOTH,
    INPUT_DIR,
    Path,
    plot_corner_for_patient,
):
    """Section 2: Generate corner plots for all patients"""

    # Get all patient directories
    input_path = Path(INPUT_DIR)
    patient_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    patient_ids = [d.name for d in patient_dirs]

    print(f"\nGenerating corner plots for {len(patient_ids)} patients...")

    # Generate corner plots for all patients
    corner_figs = []
    corner_patient_ids = []

    for patient_idx in patient_ids:
        corner_fig, output_path = plot_corner_for_patient(
            patient_id=patient_idx,
            input_dir=INPUT_DIR,
            output_filename=CORNER_OUTPUT_FILENAME,
            smooth=CORNER_SMOOTH
        )

        if corner_fig is not None:
            corner_figs.append(corner_fig)
            corner_patient_ids.append(patient_idx)

    print(f"\nSuccessfully generated {len(corner_figs)} corner plots")
    return corner_figs, corner_patient_ids


@app.cell
def _(corner_figs, corner_patient_ids, mo):
    """Section 2: Display corner plots"""

    # Display all corner plots with patient ID headers
    plots = []
    for i, (fig, patient_id) in enumerate(zip(corner_figs, corner_patient_ids)):
        plots.append(mo.md(f"### Patient {patient_id}"))
        plots.append(fig)

    mo.vstack(plots) if plots else mo.md("_No corner plots generated_")
    return


@app.cell
def _(mo):
    """Section 2.1 introduction"""
    mo.md("""
    ---

    ## Section 2.1: Corner Plots - Particle Filter

    This section generates corner plots for particle filter posterior samples,
    analogous to the NPE corner plots above but using weighted particle filter results.

    **Key Differences from NPE:**
    - Samples loaded from pickle files containing particle filter fit results
    - Weighted samples may be resampled if weights are non-uniform (CV > 0.01)
    - User specifies full paths to run directories in `PF_RUN_DIRS`
    - Patient ID and particle count are automatically extracted from the directory path

    **Configuration:**
    Specify the full path to each run directory in the `PF_RUN_DIRS` list:
    ```python
    PF_RUN_DIRS = [
        "../results/particle_filter/<timestamp>/<patient_id>/src.tiv.RefractoryCellModel_JSF_<n_particles>",
        # Add more paths as needed
    ]
    ```

    **Directory Structure:**
    ```
    <run_directory>/
        fit_result.pkl      # input
        corner_plot.png     # output
    ```
    """)
    return


@app.cell
def _():
    """Section 2.1: Configuration for particle filter corner plots"""

    # List of particle filter run directories to process
    # Each path should point to a specific run directory containing fit_result.pkl
    PF_RUN_DIRS = [
        "../results/particle_filter/20250103_existing/432192/src.tiv.RefractoryCellModel_JSF_6000/"
    ]

    # Output filename for corner plots (saved per patient)
    PF_CORNER_OUTPUT_FILENAME = "corner_plot_pf_canonical.png"

    # Smoothing parameter for KDE in corner plots
    PF_CORNER_SMOOTH = 1.0

    # Resampling threshold (coefficient of variation)
    PF_RESAMPLE_THRESHOLD = 0.01
    return (
        PF_CORNER_OUTPUT_FILENAME,
        PF_CORNER_SMOOTH,
        PF_RESAMPLE_THRESHOLD,
        PF_RUN_DIRS,
    )


@app.cell
def _(Path, np):
    """Section 2.1: Helper function to load particle filter samples"""

    import pickle

    def load_particle_filter_samples_pf(run_dir: str) -> tuple[np.ndarray, np.ndarray, str, int]:
        """
        Load particle filter samples from pickle file at specified run directory.

        Parameters:
        -----------
        run_dir : str
            Full path to run directory containing fit_result.pkl

        Returns:
        --------
        tuple[np.ndarray, np.ndarray, str, int]
            samples: Parameter samples with shape (n_samples, 6) in order [lnV0, beta, phi, rho, delta, pi]
            weights: Sample weights with shape (n_samples,)
            patient_id: Patient ID extracted from path
            n_particles: Number of particles extracted from directory name
        """
        run_path = Path(run_dir)
        pickle_path = run_path / "fit_result.pkl"

        # Extract patient ID from path (second-to-last directory)
        # e.g., .../432192/src.tiv.RefractoryCellModel_JSF_6000/
        patient_id = run_path.parent.name

        # Extract particle count from directory name
        # e.g., "src.tiv.RefractoryCellModel_JSF_6000" -> 6000
        n_particles = int(run_path.name.split('_')[-1])

        # Check if file exists
        if not pickle_path.exists():
            raise FileNotFoundError(f"Results file not found: {pickle_path}")

        print(f"  Loading patient {patient_id} (N={n_particles}) from: {pickle_path}")

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

        print(f"  Weight stats: min={weights.min():.6f}, max={weights.max():.6f}, "
              f"sum={weights.sum():.6f}")

        return samples, weights, patient_id, n_particles
    return (load_particle_filter_samples_pf,)


@app.cell
def _(np):
    """Section 2.1: Helper function to resample weighted particles"""

    def resample_if_needed_pf(
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
            print(f"  Weights non-uniform (CV={cv:.4f}), resampling...")
            # Normalize weights
            weights_norm = weights / weights.sum()
            # Resample
            n_samples = len(samples)
            indices = np.random.choice(n_samples, size=n_samples, replace=True, p=weights_norm)
            resampled = samples[indices]
            return resampled
        else:
            print(f"  Weights approximately uniform (CV={cv:.4f}), using all samples")
            return samples
    return (resample_if_needed_pf,)


@app.cell
def _(
    Path,
    SCIENCE_RCPARAMS,
    corner,
    load_particle_filter_samples_pf,
    plt,
    resample_if_needed_pf,
    transform_samples,
):
    """Section 2.1: Corner plot function for particle filter"""

    def plot_corner_for_patient_pf(
        run_dir: str,
        output_filename: str,
        smooth: float = 1.0,
        resample_threshold: float = 0.01
    ) -> tuple:
        """
        Generate corner plot for particle filter posterior samples from specified run directory.

        Parameters:
        -----------
        run_dir : str
            Full path to run directory containing fit_result.pkl
        output_filename : str
            Name for output corner plot file
        smooth : float
            Smoothing parameter for corner plots
        resample_threshold : float
            Coefficient of variation threshold for resampling

        Returns:
        --------
        tuple
            (figure, output_path, patient_id, n_particles) or (None, None, None, None) if error
        """
        try:
            # Load samples, weights, patient ID, and particle count
            samples, weights, patient_id, n_particles = load_particle_filter_samples_pf(run_dir)

            # Resample if weights are non-uniform
            samples = resample_if_needed_pf(samples, weights, resample_threshold)

            # Transform and reorder parameters
            display_samples = transform_samples(samples)

        except FileNotFoundError as e:
            print(f"  Warning: {e}, skipping...")
            return None, None, None, None

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
        prior_bounds = [
            (0.0, 25.0),      # β: Extended range for better visualization
            (-0.1, 1.0),      # ρ: Extended range for better visualization
            (200.0, 600.0),   # π: Uniform(200, 600)
            (-2.0, 20.0),     # φ: Extended range for better visualization
            (-0.2, 12.0),     # δ: Extended range for better visualization
            (-0.1, 3.0)       # log₁₀V₀: Extended range for better visualization
        ]

        # Corner plot settings
        corner_kwargs = {
            'labels': param_labels,
            'color': 'darkorange',  # Different color from NPE (teal)
            'bins': 30,
            'range': prior_bounds,
            'plot_datapoints': True,
            'plot_density': True,
            'plot_contours': True,
            'data_kwargs': {'alpha': 0.2, 'color': 'peachpuff'},
            'hist_kwargs': {'alpha': 0.8, 'color': 'darkorange'},
            'contour_kwargs': {'colors': 'darkorange'},
            'smooth': smooth,
            'smooth1d': smooth,
            'quantiles': [0.16, 0.5, 0.84],
            'show_titles': True,
            'title_kwargs': {"fontsize": 20},
            'label_kwargs': {"fontsize": 22}
        }

        # Create corner plot
        plt.rcParams.update(SCIENCE_RCPARAMS)
        fig = corner.corner(display_samples, **corner_kwargs)

        # Add patient ID and particle count as suptitle
        fig.suptitle(f'Patient {patient_id} - Particle Filter Posterior (N={n_particles})',
                     fontsize=24, y=0.995)

        # Construct output path
        output_path = Path(run_dir) / output_filename

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        print(f"  Corner plot saved to: {output_path}")

        return fig, output_path, patient_id, n_particles
    return (plot_corner_for_patient_pf,)


@app.cell
def _(
    PF_CORNER_OUTPUT_FILENAME,
    PF_CORNER_SMOOTH,
    PF_RESAMPLE_THRESHOLD,
    PF_RUN_DIRS,
    plot_corner_for_patient_pf,
):
    """Section 2.1: Generate corner plots for all specified run directories"""

    print(f"\nGenerating particle filter corner plots for {len(PF_RUN_DIRS)} run(s)...")

    # Generate corner plots for all specified run directories
    pf_corner_figs = []
    pf_corner_patient_ids = []
    pf_corner_particle_counts = []

    for run_dir in PF_RUN_DIRS:
        pf_fig, pf_output_path, patient_id_idx, n_particles_idx = plot_corner_for_patient_pf(
            run_dir=run_dir,
            output_filename=PF_CORNER_OUTPUT_FILENAME,
            smooth=PF_CORNER_SMOOTH,
            resample_threshold=PF_RESAMPLE_THRESHOLD
        )

        if pf_fig is not None:
            pf_corner_figs.append(pf_fig)
            pf_corner_patient_ids.append(patient_id_idx)
            pf_corner_particle_counts.append(n_particles_idx)

    print(f"\nSuccessfully generated {len(pf_corner_figs)} particle filter corner plots")
    return pf_corner_figs, pf_corner_particle_counts, pf_corner_patient_ids


@app.cell
def _(mo, pf_corner_figs, pf_corner_particle_counts, pf_corner_patient_ids):
    """Section 2.1: Display particle filter corner plots"""

    # Display all corner plots with patient ID and particle count headers
    pf_plots = []
    for k, (fig_k, patient_id_k, n_particles_k) in enumerate(
        zip(pf_corner_figs, pf_corner_patient_ids, pf_corner_particle_counts)
    ):
        pf_plots.append(
            mo.md(f"### Patient {patient_id_k} (N={n_particles_k} particles)")
        )
        pf_plots.append(fig_k)

    mo.vstack(pf_plots) if pf_plots else mo.md("_No particle filter corner plots generated_")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
