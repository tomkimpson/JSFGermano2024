"""
Neural Posterior Estimation for COVID TIV Inference.

This script implements a three-stage pipeline for parameter inference using
Simulation-Based Inference (SBI) with Neural Posterior Estimation:

Stage 1 (simulate): Generate training data by simulating the JSF model
Stage 2 (train): Train a neural density estimator on the simulated data
Stage 3 (infer): Perform inference on patient data using the trained posterior

Usage:
    python COVID_TEIVR_NPE.py simulate --num-trajectories 1000
    python COVID_TEIVR_NPE.py train
    python COVID_TEIVR_NPE.py infer --patients all
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# SBI imports
from sbi import utils as sbi_utils
from sbi.inference import SNPE
from sbi import analysis as sbi_analysis

# Local imports
from src import npe_utils
from src import JSF_Solver_BasePython as JSF
from src.tiv import RefractoryCellModel_JSF


# Default paths
DEFAULT_CONFIG = "config/cli-refractory-tiv-jsf.toml"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "npe_outputs"


def simulate_trajectory(
    theta: np.ndarray,
    fixed_params: Dict[str, float],
    n_timepoints: int = 10,
    obs_scale: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate a single trajectory using the JSF simulator.

    Args:
        theta: Parameter vector [lnV0, beta, phi, rho, delta, pi]
        fixed_params: Dictionary of fixed parameters
        n_timepoints: Number of time points to simulate (default: 10)
        obs_scale: Observation noise scale
        seed: Random seed for simulator

    Returns:
        observations: Array of observations in log10 space (shape: n_timepoints,)
    """
    if seed is not None:
        random.seed(seed)

    # Extract parameters
    lnV0, beta, phi, rho, delta, pi = theta
    c = fixed_params['c']
    k = fixed_params['k']
    T0 = fixed_params['T0']
    E0 = fixed_params['E0']
    I0 = fixed_params['I0']
    R0 = fixed_params['R0']

    # Initial state: [T, E, I, R, V]
    # Matching RefractoryCellModel_JSF.init:116
    V0 = np.round(np.exp(lnV0))
    x0 = [T0, E0, I0, R0, V0]

    # Parameter vector for rate function
    # Matching RefractoryCellModel_JSF.update:133-141
    theta_rates = [beta, phi, rho, k, delta, pi, c]

    # Stoichiometry (from RefractoryCellModel_JSF)
    stoich = RefractoryCellModel_JSF._stoich

    # Rate function (from RefractoryCellModel_JSF._rates)
    def rates(x, time):
        t, r, e, i, v = x
        m_beta = theta_rates[0] * 10**(-9)
        m_phi = theta_rates[1] * 10**(-5)
        m_rho = theta_rates[2]
        m_k = theta_rates[3]
        m_delta = theta_rates[4]
        m_pi = theta_rates[5]
        m_c = theta_rates[6]
        return [
            m_beta * (t * v),
            m_phi * i * t,
            m_rho * r,
            m_k * e,
            m_delta * i,
            m_pi * i,
            m_c * v
        ]

    # Simulation options (from RefractoryCellModel_JSF.update:122-128)
    threshold = 100
    options = {
        'EnforceDo': [0, 0, 0, 0, 0],
        'dt': 0.00005,
        'SwitchingThreshold': [threshold] * 5
    }

    # Simulate day-by-day
    virus_trajectory = []
    x_current = x0

    for day in range(n_timepoints):
        # Simulate one day (dt=1)
        xs, ts = JSF.JumpSwitchFlowSimulator(
            x_current,
            rates,
            stoich,
            1.0,  # Simulate for 1 day
            options
        )

        # Extract final state: [T, R, E, I, V]
        # Note: JSF returns in format matching x0 order
        x_current = [xs[i][-1] for i in range(len(xs))]

        # Extract virus count (index 4)
        V = x_current[4]
        virus_trajectory.append(V)

    # Convert to numpy array
    virus_trajectory = np.array(virus_trajectory)

    # Apply observation model (log10 + detection limit + Gaussian noise)
    # Reshape for observation model: (1, n_timepoints)
    virus_counts = virus_trajectory.reshape(1, -1)
    observations = npe_utils.apply_observation_model(
        virus_counts,
        scale=obs_scale,
        seed=seed
    )

    return observations.flatten()


def _simulate_single(args_tuple):
    """
    Worker wrapper for parallel simulation.

    Args:
        args_tuple: Tuple of (theta, fixed_params, obs_scale, n_timepoints, seed)

    Returns:
        observations: Array of observations in log10 space (shape: n_timepoints,)
    """
    theta, fixed_params, obs_scale, n_timepoints, seed = args_tuple
    return simulate_trajectory(
        theta,
        fixed_params,
        n_timepoints=n_timepoints,
        obs_scale=obs_scale,
        seed=seed
    )


def stage1_simulate(args):
    """
    Stage 1: Generate training data by simulating the JSF model.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: SIMULATE TRAINING DATA")
    print("=" * 70)

    # Load configuration
    print(f"\nLoading configuration from {args.config}")
    config = npe_utils.load_config(args.config)

    # Extract priors and fixed parameters
    lower_bounds, upper_bounds, param_names = npe_utils.get_prior_bounds(config)
    fixed_params = npe_utils.get_fixed_params(config)
    obs_scale = npe_utils.get_observation_scale(config)

    # Print summary
    npe_utils.print_prior_summary(lower_bounds, upper_bounds, param_names)
    print(f"\nFixed parameters:")
    for key, val in fixed_params.items():
        print(f"  {key}: {val}")
    print(f"\nObservation scale: {obs_scale}")
    print(f"\nNumber of trajectories to simulate: {args.num_trajectories}")
    print(f"Number of time points per trajectory: {args.num_timepoints}")
    print(f"Random seed: {args.seed}")
    print(f"Number of parallel workers: {args.num_workers}")

    # Validate CPU count
    cpu_count = os.cpu_count() or 1
    if args.num_workers > cpu_count:
        print(f"\nWARNING: Requested {args.num_workers} workers but only {cpu_count} CPUs available.")
        print(f"         This may lead to suboptimal performance. Consider reducing --num-workers.")

    # Sample parameters from prior
    print("\nSampling parameters from prior...")
    theta_samples = npe_utils.sample_from_prior(
        lower_bounds,
        upper_bounds,
        args.num_trajectories,
        seed=args.seed
    )

    # Simulate trajectories
    print("\nSimulating trajectories...")
    start_time = time.time()

    if args.num_workers == 1:
        # Sequential simulation (original behavior)
        x_obs_list = []
        for i in tqdm(range(args.num_trajectories), desc="Simulations"):
            # Use different seed for each simulation
            sim_seed = args.seed + i if args.seed is not None else None

            obs = simulate_trajectory(
                theta_samples[i],
                fixed_params,
                n_timepoints=args.num_timepoints,
                obs_scale=obs_scale,
                seed=sim_seed
            )
            x_obs_list.append(obs)
    else:
        # Parallel simulation using ProcessPoolExecutor
        print(f"Using {args.num_workers} parallel workers...")

        # Prepare argument tuples for all simulations
        args_list = [
            (
                theta_samples[i],
                fixed_params,
                obs_scale,
                args.num_timepoints,
                args.seed + i if args.seed is not None else None
            )
            for i in range(args.num_trajectories)
        ]

        # Execute simulations in parallel with progress bar
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            x_obs_list = list(
                tqdm(
                    executor.map(_simulate_single, args_list),
                    total=args.num_trajectories,
                    desc="Simulations"
                )
            )

    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
    print(f"Average time per simulation: {elapsed_time / args.num_trajectories:.2f} seconds")
    if args.num_workers > 1:
        print(f"Parallelization: {args.num_workers} workers (theoretical speedup: {args.num_workers}x)")

    # Convert to array
    x_obs = np.array(x_obs_list)

    # Prepare metadata
    metadata = {
        'param_names': param_names,
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'fixed_params': fixed_params,
        'obs_scale': obs_scale,
        'n_timepoints': args.num_timepoints,
        'seed': args.seed,
        'num_workers': args.num_workers,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Save training data
    output_path = Path(args.output_dir) / "training" / f"simulations_{args.num_trajectories}.npz"
    npe_utils.save_training_data(theta_samples, x_obs, str(output_path), metadata)

    # Print summary statistics
    print("\nData summary:")
    print(f"  Theta shape: {theta_samples.shape}")
    print(f"  X_obs shape: {x_obs.shape}")
    print(f"\nSaved to: {output_path}")

    return output_path


def stage2_train(args):
    """
    Stage 2: Train a neural density estimator on the simulated data.
    """
    print("\n" + "=" * 70)
    print("STAGE 2: TRAIN NEURAL POSTERIOR ESTIMATOR")
    print("=" * 70)

    # Load training data
    training_file = Path(args.output_dir) / "training" / f"simulations_{args.num_trajectories}.npz"

    if not training_file.exists():
        print(f"\nError: Training data not found at {training_file}")
        print("Please run 'simulate' stage first.")
        sys.exit(1)

    print(f"\nLoading training data from {training_file}")
    theta, x_obs, metadata = npe_utils.load_training_data(str(training_file))

    print(f"\nTraining data loaded:")
    print(f"  Number of simulations: {len(theta)}")
    print(f"  Parameter dimension: {theta.shape[1]}")
    print(f"  Observation dimension: {x_obs.shape[1]}")

    # Extract prior bounds
    lower_bounds = metadata['lower_bounds']
    upper_bounds = metadata['upper_bounds']
    param_names = metadata['param_names']

    npe_utils.print_prior_summary(lower_bounds, upper_bounds, param_names)

    # Convert to tensors
    print("\nConverting to PyTorch tensors...")
    theta_tensor = npe_utils.to_tensor(theta)
    x_obs_tensor = npe_utils.to_tensor(x_obs)

    # Define prior for SBI
    print("\nSetting up SBI prior...")
    prior = sbi_utils.BoxUniform(
        low=torch.tensor(lower_bounds, dtype=torch.float32),
        high=torch.tensor(upper_bounds, dtype=torch.float32)
    )

    # Initialize SNPE
    print("\nInitializing SNPE...")
    inference = SNPE(prior=prior, density_estimator='nsf')  # Neural Spline Flow

    # Append simulations
    print("Appending simulations to SNPE...")
    inference.append_simulations(theta_tensor, x_obs_tensor)

    # Train
    print(f"\nTraining posterior estimator...")
    print(f"  Training fraction: {args.training_fraction}")
    print(f"  Max epochs: {args.max_epochs}")

    density_estimator = inference.train(
        training_batch_size=args.batch_size,
        max_num_epochs=args.max_epochs,
        validation_fraction=1 - args.training_fraction,
        show_train_summary=True,
    )

    # Build posterior
    print("\nBuilding posterior...")
    posterior = inference.build_posterior(density_estimator)

    # Save posterior
    output_path = Path(args.output_dir) / "models" / f"posterior_N{args.num_trajectories}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving posterior to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(posterior, f)

    # Also save metadata
    metadata_path = output_path.with_suffix('.meta.npz')
    np.savez(
        metadata_path,
        param_names=param_names,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    print(f"\nTraining completed successfully!")
    print(f"Posterior saved to: {output_path}")

    return output_path


def stage3_infer(args):
    """
    Stage 3: Perform inference on patient data using the trained posterior.
    """
    print("\n" + "=" * 70)
    print("STAGE 3: INFERENCE ON PATIENT DATA")
    print("=" * 70)

    # Load posterior
    posterior_path = Path(args.output_dir) / "models" / f"posterior_N{args.num_trajectories}.pkl"

    if not posterior_path.exists():
        print(f"\nError: Posterior not found at {posterior_path}")
        print("Please run 'train' stage first.")
        sys.exit(1)

    print(f"\nLoading posterior from {posterior_path}")
    with open(posterior_path, 'rb') as f:
        posterior = pickle.load(f)

    # Load metadata
    metadata_path = posterior_path.with_suffix('.meta.npz')
    metadata = np.load(metadata_path, allow_pickle=True)
    param_names = metadata['param_names'].tolist()

    print(f"Posterior loaded successfully")

    # Get patient list
    if args.patients == 'all':
        patient_ids = npe_utils.get_all_patient_ids(args.data_dir)
    else:
        patient_ids = args.patients.split(',')

    print(f"\nPatients to process: {patient_ids}")
    print(f"Number of posterior samples: {args.num_samples}")

    # Process each patient
    for patient_id in patient_ids:
        print(f"\n{'-' * 70}")
        print(f"Processing patient: {patient_id}")
        print(f"{'-' * 70}")

        try:
            # Load patient data
            times, observations = npe_utils.load_patient_data(
                patient_id,
                data_dir=args.data_dir,
                n_timepoints=args.num_timepoints
            )

            print(f"Loaded {len(observations)} observations")

            # Convert to tensor
            x_obs = npe_utils.to_tensor(observations.reshape(1, -1))

            # Sample from posterior
            print(f"Sampling {args.num_samples} samples from posterior...")
            posterior_samples = posterior.sample((args.num_samples,), x=x_obs)

            # Convert to numpy
            samples_np = npe_utils.from_tensor(posterior_samples)

            # Compute summary statistics
            mean = np.mean(samples_np, axis=0)
            std = np.std(samples_np, axis=0)
            q025 = np.percentile(samples_np, 2.5, axis=0)
            q975 = np.percentile(samples_np, 97.5, axis=0)
            median = np.median(samples_np, axis=0)

            # Print summary
            print("\nPosterior summary:")
            print(f"{'Parameter':<10s} {'Mean':>10s} {'Std':>10s} {'2.5%':>10s} {'Median':>10s} {'97.5%':>10s}")
            print("-" * 62)
            for i, name in enumerate(param_names):
                print(f"{name:<10s} {mean[i]:>10.4f} {std[i]:>10.4f} {q025[i]:>10.4f} {median[i]:>10.4f} {q975[i]:>10.4f}")

            # Save results
            output_dir = Path(args.output_dir) / "inference" / patient_id
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save samples
            samples_path = output_dir / "samples.npy"
            np.save(samples_path, samples_np)
            print(f"\nSamples saved to: {samples_path}")

            # Save summary statistics
            summary_path = output_dir / "summary.csv"
            import pandas as pd
            summary_df = pd.DataFrame({
                'parameter': param_names,
                'mean': mean,
                'std': std,
                'q025': q025,
                'median': median,
                'q975': q975,
            })
            summary_df.to_csv(summary_path, index=False)
            print(f"Summary saved to: {summary_path}")

            # Generate diagnostic plots
            if args.plot:
                print("Generating diagnostic plots...")
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()

                for i, name in enumerate(param_names):
                    axes[i].hist(samples_np[:, i], bins=50, alpha=0.7, edgecolor='black')
                    axes[i].axvline(mean[i], color='red', linestyle='--', label='Mean')
                    axes[i].axvline(median[i], color='green', linestyle='--', label='Median')
                    axes[i].set_xlabel(name)
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()

                plt.tight_layout()
                plot_path = output_dir / "posterior_histograms.png"
                plt.savefig(plot_path, dpi=150)
                plt.close()
                print(f"Plots saved to: {plot_path}")

        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
            continue

    print(f"\n{'=' * 70}")
    print("Inference completed!")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Neural Posterior Estimation for COVID TIV Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage 1: Simulate 1000 training trajectories
  python COVID_TEIVR_NPE.py simulate --num-trajectories 1000

  # Stage 2: Train the neural posterior estimator
  python COVID_TEIVR_NPE.py train --num-trajectories 1000

  # Stage 3: Infer parameters for all patients
  python COVID_TEIVR_NPE.py infer --patients all

  # Or for specific patients
  python COVID_TEIVR_NPE.py infer --patients 432192,443108
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Stage 1: Simulate
    parser_sim = subparsers.add_parser('simulate', help='Generate training data')
    parser_sim.add_argument('--num-trajectories', type=int, default=1000,
                            help='Number of trajectories to simulate (default: 1000)')
    parser_sim.add_argument('--num-timepoints', type=int, default=10,
                            help='Number of time points per trajectory (default: 10)')
    parser_sim.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                            help=f'Path to TOML config file (default: {DEFAULT_CONFIG})')
    parser_sim.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                            help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser_sim.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
    parser_sim.add_argument('--num-workers', type=int, default=1,
                            help='Number of parallel workers for simulation (default: 1, sequential)')

    # Stage 2: Train
    parser_train = subparsers.add_parser('train', help='Train neural posterior estimator')
    parser_train.add_argument('--num-trajectories', type=int, default=1000,
                              help='Number of trajectories used in simulation (default: 1000)')
    parser_train.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                              help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser_train.add_argument('--batch-size', type=int, default=128,
                              help='Training batch size (default: 128)')
    parser_train.add_argument('--max-epochs', type=int, default=100,
                              help='Maximum number of epochs (default: 100)')
    parser_train.add_argument('--training-fraction', type=float, default=0.8,
                              help='Fraction of data for training (default: 0.8)')

    # Stage 3: Infer
    parser_infer = subparsers.add_parser('infer', help='Perform inference on patient data')
    parser_infer.add_argument('--patients', type=str, default='all',
                              help='Patient IDs (comma-separated) or "all" (default: all)')
    parser_infer.add_argument('--num-trajectories', type=int, default=1000,
                              help='Number of trajectories used in training (default: 1000)')
    parser_infer.add_argument('--num-samples', type=int, default=10000,
                              help='Number of posterior samples (default: 10000)')
    parser_infer.add_argument('--num-timepoints', type=int, default=10,
                              help='Number of time points to use (default: 10)')
    parser_infer.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                              help=f'Data directory (default: {DEFAULT_DATA_DIR})')
    parser_infer.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                              help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser_infer.add_argument('--plot', action='store_true',
                              help='Generate diagnostic plots')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == 'simulate':
        stage1_simulate(args)
    elif args.command == 'train':
        stage2_train(args)
    elif args.command == 'infer':
        stage3_infer(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
