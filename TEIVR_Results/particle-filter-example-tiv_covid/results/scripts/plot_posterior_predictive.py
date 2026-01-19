#!/usr/bin/env python3
"""
Plot posterior predictive viral load trajectories with observed data.

This script creates plots showing:
- Observed patient viral load data (circles)
- Posterior predictive median and credible intervals from particle filter
- Vertical line separating observation period from forecast period

Figures are saved in the results directory alongside the predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from pathlib import Path
import argparse
import scienceplots

# Apply science plots style
plt.style.use('science')
mpl.rcParams['text.usetex'] = False

# Detection limit for visualization
DETECTION_LIMIT_LOG10 = -0.65  # log10 copies/mL
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data'

# Color palette
ASSIMILATION_COLOR = '#2ca02c'  # green
FORECAST_COLOR = '#1f77b4'      # blue


def load_patient_observations(patient_id, data_dir='data'):
    """Load observed viral load data for a patient."""
    data_file = Path(data_dir) / f"{patient_id}.ssv"
    df = pd.read_csv(data_file, sep=' ')

    times = df['time'].values
    log10_viral_load = df['value'].values

    return times, log10_viral_load


def load_predictions(results_dir, patient_id):
    """Load posterior predictive results for a patient."""
    patient_dir = Path(results_dir) / patient_id

    # Load metadata
    with open(patient_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Load summary statistics for forward predictions
    summary_df = pd.read_csv(patient_dir / 'summary_statistics.csv')

    # Load raw predictions (for additional analysis if needed)
    predicted_obs = np.load(patient_dir / 'predicted_observations.npy')

    # Load filter trajectory if available
    filter_df = None
    filter_traj_file = patient_dir / 'filter_trajectory.csv'
    if filter_traj_file.exists():
        filter_df = pd.read_csv(filter_traj_file)

    return metadata, summary_df, predicted_obs, filter_df


def plot_patient(patient_id, results_dir, data_dir='data', save_dir=None):
    """
    Create a plot for a single patient showing observations and predictions.

    Parameters
    ----------
    patient_id : str
        Patient identifier
    results_dir : str
        Path to posterior predictive results directory
    data_dir : str
        Path to data directory containing observations
    save_dir : str, optional
        Directory to save plots (default: results_dir)
    """
    # Load data
    obs_times, obs_values = load_patient_observations(patient_id, data_dir)
    metadata, summary_df, predicted_obs, filter_df = load_predictions(results_dir, patient_id)

    # Extract prediction times
    # Time 0 in predictions corresponds to T_obs in observations
    T_obs = metadata['T_obs']
    horizon = len(summary_df) - 1  # predictions include t=0
    pred_times = np.arange(T_obs, T_obs + horizon + 1)

    # Determine parameter inference horizon (default to T_obs if unavailable)
    parameter_horizon = T_obs
    npe_timepoints = metadata.get('n_timepoints_npe')
    times = metadata.get('times')
    if isinstance(npe_timepoints, int) and npe_timepoints > 0 and isinstance(times, (list, tuple)):
        if len(times) >= npe_timepoints:
            parameter_horizon = float(times[npe_timepoints - 1])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot particle filter trajectory (assimilation period) if available
    if filter_df is not None:
        filter_times = filter_df['time'].values
        filter_q025 = filter_df['q025'].values
        filter_q975 = filter_df['q975'].values
        filter_q25 = filter_df['q25'].values
        filter_q75 = filter_df['q75'].values
        filter_median = filter_df['median'].values

        pre_mask = filter_times <= parameter_horizon + 1e-9
        post_mask = (filter_times >= parameter_horizon - 1e-9) & (filter_times <= T_obs + 1e-9)

        if np.any(pre_mask):
            ft = filter_times[pre_mask]
            ax.fill_between(ft,
                             filter_q025[pre_mask],
                             filter_q975[pre_mask],
                             alpha=0.15, color=ASSIMILATION_COLOR,
                             label='Assimilation 95% CI')
            ax.fill_between(ft,
                             filter_q25[pre_mask],
                             filter_q75[pre_mask],
                             alpha=0.3, color=ASSIMILATION_COLOR,
                             label='Assimilation 50% CI')
            ax.plot(ft, filter_median[pre_mask],
                    '-', color=ASSIMILATION_COLOR, linewidth=2,
                    label='Assimilation median', zorder=5)

        if np.any(post_mask):
            ft = filter_times[post_mask]
            ax.fill_between(ft,
                             filter_q025[post_mask],
                             filter_q975[post_mask],
                             alpha=0.15, color=FORECAST_COLOR,
                             label='Post-fit 95% CI')
            ax.fill_between(ft,
                             filter_q25[post_mask],
                             filter_q75[post_mask],
                             alpha=0.3, color=FORECAST_COLOR,
                             label='Post-fit 50% CI')
            ax.plot(ft, filter_median[post_mask],
                    '-', color=FORECAST_COLOR, linewidth=2,
                    label='Post-fit median', zorder=5)

    # Plot observed data
    ax.plot(obs_times, obs_values, 'o',
            color='black', markersize=8,
            label='Observed', zorder=10)

    # Plot posterior predictive intervals (forward forecast)
    # 95% credible interval
    ax.fill_between(pred_times,
                     summary_df['q025'],
                     summary_df['q975'],
                     alpha=0.2, color=FORECAST_COLOR,
                     label='Forecast 95% CI')

    # 50% credible interval
    ax.fill_between(pred_times,
                     summary_df['q25'],
                     summary_df['q75'],
                     alpha=0.4, color=FORECAST_COLOR,
                     label='Forecast 50% CI')

    # Median prediction
    ax.plot(pred_times, summary_df['median'],
            '-', color=FORECAST_COLOR, linewidth=2,
            label='Forecast median')

    # Add vertical line at parameter inference horizon
    ax.axvline(parameter_horizon, color='gray', linestyle='--',
               linewidth=1.5, alpha=0.7,
               label='Parameter-fit horizon')

    # Add detection limit line
    ax.axhline(DETECTION_LIMIT_LOG10, color='red',
               linestyle=':', linewidth=1, alpha=0.5,
               label='Detection limit')

    # Labels and formatting
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(r'Viral load (log$_{10}$ copies/mL)')
    ax.set_title(f'Patient {patient_id}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True, fancybox=False)

    # Set y-axis limits with some padding
    y_min = min(obs_values.min(), summary_df['q025'].min()) - 0.5
    y_max = max(obs_values.max(), summary_df['q975'].max())

    # Also consider filter trajectory if available
    if filter_df is not None:
        y_max = max(y_max, filter_df['q975'].max())

    y_max += 1.0  # Increased padding at top
    ax.set_ylim(y_min, y_max)

    # Save figure
    if save_dir is None:
        save_dir = results_dir

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save original version
    output_file = save_dir / f'{patient_id}_posterior_predictive.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")

    # Create and save square version with legend
    fig.set_size_inches(6, 6)
    output_file_square = save_dir / f'{patient_id}_posterior_predictive_square.png'
    plt.savefig(output_file_square, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file_square}")

    plt.close()

    # Print diagnostic info
    print(f"  Observations: {len(obs_times)} timepoints")
    print(f"  Predictions: {len(pred_times)} timepoints (including t=0)")
    print(f"  Obs range: [{obs_values.min():.2f}, {obs_values.max():.2f}]")
    print(f"  Pred range: [{summary_df['median'].min():.2f}, {summary_df['median'].max():.2f}]")

    # Check if predictions show extinction
    if summary_df['median'].max() == DETECTION_LIMIT_LOG10:
        print(f"  WARNING: All predictions at detection limit (extinction state)")


def main():
    parser = argparse.ArgumentParser(
        description='Plot posterior predictive trajectories with observations'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Path to posterior predictive results directory'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Path to data directory (default: repository data directory)'
    )
    parser.add_argument(
        '--patients',
        type=str,
        nargs='+',
        default=None,
        help='Patient IDs to plot (default: all in results directory)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: results_dir/plots)'
    )

    args = parser.parse_args()

    # Determine which patients to process
    results_path = Path(args.results_dir).resolve()
    # Resolve data directory with sensible fallbacks
    data_dir_candidates = []
    if args.data_dir is None:
        data_dir_candidates.append(DEFAULT_DATA_DIR)
    else:
        explicit = Path(args.data_dir).resolve()
        data_dir_candidates.append(explicit)
        # If user supplied relative path that doesn't exist from cwd, try relative to project root
        data_dir_candidates.append((PROJECT_ROOT / args.data_dir).resolve())

    data_dir_path = None
    for candidate in data_dir_candidates:
        if candidate.exists():
            data_dir_path = candidate
            break
    if data_dir_path is None:
        raise FileNotFoundError(
            f"Could not locate data directory. Tried: "
            f"{', '.join(str(p) for p in data_dir_candidates)}"
        )

    if args.patients is None:
        # Find all patient subdirectories
        patient_dirs = [d for d in results_path.iterdir()
                       if d.is_dir() and d.name.isdigit()]
        patient_ids = sorted([d.name for d in patient_dirs])
    else:
        patient_ids = args.patients

    # Set save directory
    if args.save_dir is None:
        save_dir = results_path / 'plots'
    else:
        save_dir = args.save_dir

    print(f"Results directory: {results_path}")
    print(f"Data directory: {data_dir_path}")
    print(f"Save directory: {save_dir}")
    print(f"\nProcessing {len(patient_ids)} patients...\n")

    # Process each patient
    for patient_id in patient_ids:
        print(f"Processing Patient {patient_id}...")
        try:
            plot_patient(
                patient_id,
                args.results_dir,
                data_dir=data_dir_path,
                save_dir=save_dir
            )
        except Exception as e:
            print(f"  ERROR: Failed to plot patient {patient_id}: {e}")
        print()

    print(f"All plots completed! Check {save_dir} for results.")

    # Create a summary plot with all patients in subplots
    print("\nCreating summary figure with all patients...")
    create_summary_figure(patient_ids, args.results_dir, data_dir_path, save_dir)


def create_summary_figure(patient_ids, results_dir, data_dir, save_dir):
    """Create a figure with subplots for all patients."""
    n_patients = len(patient_ids)
    n_cols = 2
    n_rows = (n_patients + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, patient_id in enumerate(patient_ids):
        ax = axes[idx]

        try:
            # Load data
            obs_times, obs_values = load_patient_observations(patient_id, data_dir)
            metadata, summary_df, _, filter_df = load_predictions(results_dir, patient_id)

            T_obs = metadata['T_obs']
            parameter_horizon = T_obs
            npe_timepoints = metadata.get('n_timepoints_npe')
            times = metadata.get('times')
            if isinstance(npe_timepoints, int) and npe_timepoints > 0 and isinstance(times, (list, tuple)):
                if len(times) >= npe_timepoints:
                    parameter_horizon = float(times[npe_timepoints - 1])
            horizon = len(summary_df) - 1
            pred_times = np.arange(T_obs, T_obs + horizon + 1)

            # Plot filter trajectory if available
            if filter_df is not None:
                filter_times = filter_df['time'].values
                filter_q025 = filter_df['q025'].values
                filter_q975 = filter_df['q975'].values
                filter_q25 = filter_df['q25'].values
                filter_q75 = filter_df['q75'].values
                filter_median = filter_df['median'].values

                pre_mask = filter_times <= parameter_horizon + 1e-9
                post_mask = (filter_times >= parameter_horizon - 1e-9) & (filter_times <= T_obs + 1e-9)

                if np.any(pre_mask):
                    ft = filter_times[pre_mask]
                    ax.fill_between(ft, filter_q025[pre_mask], filter_q975[pre_mask],
                                   alpha=0.15, color=ASSIMILATION_COLOR,
                                   label='Assimilation 95%' if idx == 0 else '')
                    ax.fill_between(ft, filter_q25[pre_mask], filter_q75[pre_mask],
                                   alpha=0.3, color=ASSIMILATION_COLOR,
                                   label='Assimilation 50%' if idx == 0 else '')
                    ax.plot(ft, filter_median[pre_mask], '-', color=ASSIMILATION_COLOR,
                           linewidth=2, label='Assimilation median' if idx == 0 else '', zorder=4)

                if np.any(post_mask):
                    ft = filter_times[post_mask]
                    ax.fill_between(ft, filter_q025[post_mask], filter_q975[post_mask],
                                   alpha=0.15, color=FORECAST_COLOR,
                                   label='Post-fit 95%' if idx == 0 else '')
                    ax.fill_between(ft, filter_q25[post_mask], filter_q75[post_mask],
                                   alpha=0.3, color=FORECAST_COLOR,
                                   label='Post-fit 50%' if idx == 0 else '')
                    ax.plot(ft, filter_median[post_mask], '-', color=FORECAST_COLOR,
                           linewidth=2, label='Post-fit median' if idx == 0 else '', zorder=4)

            # Plot observed data
            ax.plot(obs_times, obs_values, 'o', color='black',
                   markersize=6, label='Observed' if idx == 0 else '', zorder=10)

            # Plot forward forecast
            ax.fill_between(pred_times, summary_df['q025'], summary_df['q975'],
                           alpha=0.2, color=FORECAST_COLOR, label='Forecast 95%' if idx == 0 else '')
            ax.fill_between(pred_times, summary_df['q25'], summary_df['q75'],
                           alpha=0.4, color=FORECAST_COLOR, label='Forecast 50%' if idx == 0 else '')
            ax.plot(pred_times, summary_df['median'], '-', color=FORECAST_COLOR,
                   linewidth=2, label='Forecast' if idx == 0 else '')

            ax.axvline(parameter_horizon, color='gray', linestyle='--',
                      linewidth=1, alpha=0.7)
            ax.axhline(DETECTION_LIMIT_LOG10, color='red',
                      linestyle=':', linewidth=1, alpha=0.5)

            ax.set_title(f'Patient {patient_id}')
            ax.grid(True, alpha=0.3)

            if idx % n_cols == 0:
                ax.set_ylabel(r'Viral load (log$_{10}$)')
            if idx >= n_patients - n_cols:
                ax.set_xlabel('Time (days)')

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {patient_id}',
                   ha='center', va='center', transform=ax.transAxes)

    # Hide unused subplots
    for idx in range(n_patients, len(axes)):
        axes[idx].axis('off')

    # Add legend to the figure
    if n_patients > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                  bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=True)

    plt.tight_layout()

    output_file = Path(save_dir) / 'all_patients_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Summary figure saved: {output_file}")


if __name__ == '__main__':
    main()
