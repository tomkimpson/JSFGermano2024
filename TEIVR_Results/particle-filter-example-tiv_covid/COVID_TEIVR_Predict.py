#!/usr/bin/env python3
"""
COVID TIV Posterior Predictive Simulation

Generate posterior predictive viral load trajectories by combining:
1. Parameter uncertainty from Neural Posterior Estimator (NPE)
2. Latent-state uncertainty from particle filter

Usage:
    python COVID_TEIVR_Predict.py \\
        --config config/npe_config.toml \\
        --posterior-run results/npe/20250103_existing_primary \\
        --patients 432192 443108 \\
        --num-samples 1000 \\
        --particles 6000 \\
        --horizon 14 \\
        --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import pypfilt
from multiprocessing import Pool
from functools import partial

from src import npe_utils
from src import posterior_predictive as pp


def setup_logging(log_file=None):
    """Configure logging to console and optionally to file."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def process_single_sample(
    sample_idx: int,
    theta: np.ndarray,
    config_path: str,
    patient_id: str,
    T_obs: float,
    num_particles: int,
    horizon: int,
    fixed_params: dict,
    base_seed: int
) -> tuple:
    """
    Process a single parameter sample: filter → extract state → simulate forward.

    Parameters
    ----------
    sample_idx : int
        Index of this sample (for seeding)
    theta : np.ndarray
        Parameter vector [lnV0, beta, phi, rho, delta, pi]
    config_path : str
        Path to pypfilt config
    patient_id : str
        Patient identifier
    T_obs : float
        Final observation time
    num_particles : int
        Number of particles for filter
    horizon : int
        Days to simulate forward
    fixed_params : dict
        Fixed parameters (c, k, etc.)
    base_seed : int
        Base random seed

    Returns
    -------
    latent_state : dict
        Sampled latent state at T_obs
    predicted_states : np.ndarray
        State trajectory, shape (H+1, 5)
    predicted_obs : np.ndarray
        Observation trajectory, shape (H+1,)
    """
    logger = logging.getLogger(__name__)

    # Set seeds for reproducibility
    filter_seed = base_seed + sample_idx * 1000
    state_seed = base_seed + sample_idx * 1000 + 1
    sim_seed = base_seed + sample_idx * 1000 + 2

    try:
        # Build pypfilt context with fixed parameters
        context = pp.build_fixed_param_context(
            config_path=config_path,
            theta=theta,
            patient_id=patient_id,
            num_particles=num_particles,
            seed=filter_seed
        )

        # Run particle filter to assimilate observations
        logger.info(f"Sample {sample_idx}: Running particle filter")
        fit_result = pypfilt.forecast(context, [T_obs], filename=None)

        # Extract filter trajectory over assimilation period
        filter_times, filter_stats = pp.extract_filter_trajectory(
            fit_result,
            context=context,
            obs_time=T_obs
        )
        logger.info(f"Sample {sample_idx}: Extracted filter trajectory over {len(filter_times)} timepoints")

        # Extract latent state from particle cloud
        latent_state = pp.extract_latent_state(
            fit_result,
            context=context,
            obs_time=T_obs,
            seed=state_seed
        )
        logger.info(f"Sample {sample_idx}: Sampled latent state - V={latent_state['V']:.2e}")

        # Simulate forward from latent state
        logger.info(f"Sample {sample_idx}: Simulating {horizon} days forward")
        predicted_states, predicted_obs = pp.simulate_forward(
            x0=latent_state,
            theta=theta,
            fixed_params=fixed_params,
            horizon=horizon,
            seed=sim_seed
        )

        logger.info(f"Sample {sample_idx}: Complete")
        return latent_state, predicted_states, predicted_obs, (filter_times, filter_stats)

    except Exception as e:
        logger.exception(f"Sample {sample_idx}: Failed with error:")
        # Return NaN arrays to indicate failure
        latent_state = {'T': np.nan, 'E': np.nan, 'I': np.nan, 'R': np.nan, 'V': np.nan}
        predicted_states = np.full((horizon + 1, 5), np.nan)
        predicted_obs = np.full(horizon + 1, np.nan)
        filter_traj = (np.array([]), {})
        return latent_state, predicted_states, predicted_obs, filter_traj


def process_patient(
    patient_id: str,
    config_path: str,
    pypfilt_config_path: str,
    posterior_run: str,
    num_samples: int,
    num_particles: int,
    horizon: int,
    output_dir: str,
    seed: int,
    num_workers: int
):
    """
    Generate posterior predictive trajectories for a single patient.

    Parameters
    ----------
    patient_id : str
        Patient identifier
    config_path : str
        Path to NPE config
    pypfilt_config_path : str
        Path to pypfilt config
    posterior_run : str
        Path to NPE run directory
    num_samples : int
        Number of parameter samples
    num_particles : int
        Number of particles per filter
    horizon : int
        Days to predict forward
    output_dir : str
        Base output directory
    seed : int
        Random seed
    num_workers : int
        Number of parallel workers (1 = sequential)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing patient: {patient_id}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Load posterior and sample parameters
    logger.info(f"Loading posterior and sampling {num_samples} parameter sets")
    theta_samples, metadata = pp.load_posterior_and_samples(
        run_dir=posterior_run,
        patient_id=patient_id,
        num_samples=num_samples,
        seed=seed
    )

    T_obs = metadata['T_obs']
    logger.info(f"Observation horizon: T={T_obs:.1f} days")
    logger.info(f"Prediction horizon: H={horizon} days beyond observations")

    # Load fixed parameters
    config = npe_utils.load_config(config_path)
    fixed_params = npe_utils.get_fixed_params(config)

    # Process each parameter sample
    logger.info(f"Processing {num_samples} samples with {num_particles} particles each")

    if num_workers > 1:
        logger.info(f"Using parallel processing with {num_workers} workers")
        # Parallel processing
        process_func = partial(
            process_single_sample,
            config_path=pypfilt_config_path,
            patient_id=patient_id,
            T_obs=T_obs,
            num_particles=num_particles,
            horizon=horizon,
            fixed_params=fixed_params,
            base_seed=seed
        )

        with Pool(num_workers) as pool:
            results = pool.starmap(
                process_func,
                [(i, theta_samples[i]) for i in range(num_samples)]
            )
    else:
        logger.info("Using sequential processing")
        # Sequential processing
        results = []
        for i in range(num_samples):
            result = process_single_sample(
                sample_idx=i,
                theta=theta_samples[i],
                config_path=pypfilt_config_path,
                patient_id=patient_id,
                T_obs=T_obs,
                num_particles=num_particles,
                horizon=horizon,
                fixed_params=fixed_params,
                base_seed=seed
            )
            results.append(result)

    # Unpack results
    latent_states = [r[0] for r in results]
    predicted_states = np.array([r[1] for r in results])  # Shape: (M, H+1, 5)
    predicted_observations = np.array([r[2] for r in results])  # Shape: (M, H+1)
    filter_trajectories = [r[3] for r in results]  # List of (times, stats) tuples

    # Check for failures
    num_failed = np.sum(np.isnan(predicted_observations[:, 0]))
    if num_failed > 0:
        logger.warning(f"{num_failed}/{num_samples} samples failed")

    # Save results
    patient_output_dir = Path(output_dir) / patient_id
    logger.info(f"Saving results to {patient_output_dir}")

    pp.save_predictions(
        output_dir=str(patient_output_dir),
        theta_samples=theta_samples,
        latent_states=latent_states,
        predicted_states=predicted_states,
        predicted_observations=predicted_observations,
        metadata=metadata,
        config_path=pypfilt_config_path,
        filter_trajectories=filter_trajectories
    )

    elapsed = time.time() - start_time
    logger.info(f"Patient {patient_id} completed in {elapsed:.1f}s ({elapsed/num_samples:.2f}s per sample)")

    # Save timing information
    with open(patient_output_dir / 'timing.txt', 'w') as f:
        f.write(f"Total time: {elapsed:.2f} seconds\n")
        f.write(f"Time per sample: {elapsed/num_samples:.2f} seconds\n")
        f.write(f"Samples: {num_samples}\n")
        f.write(f"Particles: {num_particles}\n")
        f.write(f"Horizon: {horizon}\n")
        f.write(f"Workers: {num_workers}\n")
        f.write(f"Failed: {num_failed}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate posterior predictive trajectories combining NPE and particle filtering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/cli-refractory-tiv-jsf.toml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--pypfilt-config',
        type=str,
        default='config/cli-refractory-tiv-jsf.toml',
        help='Path to pypfilt configuration file'
    )

    parser.add_argument(
        '--posterior-run',
        type=str,
        required=True,
        help='Path to NPE run directory (e.g., results/npe/20250103_existing_primary)'
    )

    parser.add_argument(
        '--patients',
        type=str,
        nargs='+',
        help='Patient IDs to process (default: all patients in posterior run)'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of parameter samples to draw from posterior'
    )

    parser.add_argument(
        '--particles',
        type=int,
        default=6000,
        help='Number of particles for particle filter'
    )

    parser.add_argument(
        '--horizon',
        type=int,
        default=14,
        help='Number of days to predict forward beyond observations'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of parallel workers (1=sequential)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: results/posterior_predictive/<timestamp>)'
    )

    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'results/posterior_predictive/{timestamp}'

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = output_path / 'predict.log'
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info("="*60)
    logger.info("COVID TIV Posterior Predictive Simulation")
    logger.info("="*60)
    logger.info(f"Posterior run: {args.posterior_run}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"pypfilt config: {args.pypfilt_config}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Particles per filter: {args.particles}")
    logger.info(f"Prediction horizon: {args.horizon} days")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Parallel workers: {args.num_workers}")

    # Determine patients to process
    if args.patients is None:
        # Auto-detect from posterior run
        inference_dir = Path(args.posterior_run) / 'inference'
        if not inference_dir.exists():
            logger.error(f"Inference directory not found: {inference_dir}")
            sys.exit(1)

        patient_dirs = [d.name for d in inference_dir.iterdir() if d.is_dir()]
        if not patient_dirs:
            logger.error(f"No patient results found in {inference_dir}")
            sys.exit(1)

        args.patients = sorted(patient_dirs)
        logger.info(f"Auto-detected {len(args.patients)} patients: {args.patients}")
    else:
        logger.info(f"Processing {len(args.patients)} patients: {args.patients}")

    # Process each patient
    overall_start = time.time()

    for patient_id in args.patients:
        try:
            process_patient(
                patient_id=patient_id,
                config_path=args.config,
                pypfilt_config_path=args.pypfilt_config,
                posterior_run=args.posterior_run,
                num_samples=args.num_samples,
                num_particles=args.particles,
                horizon=args.horizon,
                output_dir=args.output_dir,
                seed=args.seed,
                num_workers=args.num_workers
            )
        except Exception as e:
            logger.error(f"Failed to process patient {patient_id}: {e}", exc_info=True)
            continue

    overall_elapsed = time.time() - overall_start
    logger.info("\n" + "="*60)
    logger.info(f"All patients completed in {overall_elapsed:.1f}s")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
