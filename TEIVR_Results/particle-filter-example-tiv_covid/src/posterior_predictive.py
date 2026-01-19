"""
Posterior Predictive Utilities

This module provides functions for generating posterior predictive trajectories
by combining parameter uncertainty from the Neural Posterior Estimator (NPE)
with latent-state uncertainty from a particle filter.
"""

import numpy as np
import pickle
import pypfilt
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging

from src import npe_utils
from src import JSF_Solver_BasePython as JSF


logger = logging.getLogger(__name__)


def load_posterior_and_samples(
    run_dir: str,
    patient_id: str,
    num_samples: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Load trained NPE posterior and draw parameter samples for a patient.

    Parameters
    ----------
    run_dir : str
        Path to NPE run directory (e.g., 'results/npe/20250103_existing_primary')
    patient_id : str
        Patient identifier
    num_samples : int
        Number of parameter samples to draw
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    theta_samples : np.ndarray
        Parameter samples, shape (num_samples, 6)
        Order: [lnV0, beta, phi, rho, delta, pi]
    metadata : dict
        Dictionary containing posterior path, patient data, etc.
    """
    run_path = Path(run_dir)

    # Find posterior pickle file
    model_dir = run_path / 'models'
    posterior_files = list(model_dir.glob('posterior_N*.pkl'))
    if not posterior_files:
        raise FileNotFoundError(f"No posterior file found in {model_dir}")

    # Use the most recent or largest N
    posterior_path = sorted(posterior_files)[-1]
    logger.info(f"Loading posterior from {posterior_path}")

    with open(posterior_path, 'rb') as f:
        posterior = pickle.load(f)

    # Load patient observations
    patient_data_path = run_path / 'inference' / patient_id
    if not patient_data_path.exists():
        raise FileNotFoundError(f"No inference results for patient {patient_id} in {run_dir}")

    # Load patient observations from original data
    # First, read the file to determine how many timepoints exist
    import pandas as pd
    data_file = Path('data') / f"{patient_id}.ssv"
    if not data_file.exists():
        raise FileNotFoundError(f"Patient data file not found: {data_file}")

    data = pd.read_csv(data_file, sep=' ')
    n_available = len(data)

    # Load all available timepoints for particle filter assimilation
    times_all, observations_all = npe_utils.load_patient_data(
        patient_id,
        data_dir='data',
        n_timepoints=n_available
    )

    # For NPE posterior sampling, use only first 10 timepoints
    # (must match the training data shape)
    times_npe, observations_npe = npe_utils.load_patient_data(
        patient_id,
        data_dir='data',
        n_timepoints=10
    )

    # Convert NPE observations to tensor for posterior sampling
    x_obs = npe_utils.to_tensor(observations_npe)

    # Sample from posterior
    if seed is not None:
        np.random.seed(seed)

    logger.info(f"Sampling {num_samples} parameter sets from posterior")
    theta_tensor = posterior.sample((num_samples,), x=x_obs)
    theta_samples = npe_utils.from_tensor(theta_tensor)

    metadata = {
        'posterior_path': str(posterior_path),
        'patient_id': patient_id,
        'times': times_all,
        'observations': observations_all,
        'T_obs': times_all[-1],  # Final observation time
        'num_samples': num_samples,
        'seed': seed,
        'n_timepoints_npe': 10,  # Timepoints used for NPE sampling
        'n_timepoints_filter': n_available  # Timepoints used for particle filter
    }

    return theta_samples, metadata


def build_fixed_param_context(
    config_path: str,
    theta: np.ndarray,
    patient_id: str,
    num_particles: int,
    seed: int
):
    """
    Build pypfilt context with fixed parameter values from NPE posterior sample.

    Parameters
    ----------
    config_path : str
        Path to pypfilt configuration file
    theta : np.ndarray
        Parameter vector [lnV0, beta, phi, rho, delta, pi]
    patient_id : str
        Patient identifier for loading observations
    num_particles : int
        Number of particles for the filter
    seed : int
        Random seed for particle filter

    Returns
    -------
    context
        pypfilt context ready for forecasting
    """
    # Load pypfilt instance
    inst = list(pypfilt.load_instances(config_path))[0]

    # Set observation file
    inst.settings['observations']['V']['file'] = f'data/{patient_id}.ssv'

    # Override priors with constant values from theta
    # The priors are nested under scenario.inference.prior in the config
    param_names = ['lnV0', 'beta', 'phi', 'rho', 'delta', 'pi']

    # Ensure the scenario.inference structure exists
    if 'scenario' not in inst.settings:
        inst.settings['scenario'] = {}
    if 'inference' not in inst.settings['scenario']:
        inst.settings['scenario']['inference'] = {}
    if 'prior' not in inst.settings['scenario']['inference']:
        inst.settings['scenario']['inference']['prior'] = {}

    # Set constant priors for each parameter
    for i, param_name in enumerate(param_names):
        inst.settings['scenario']['inference']['prior'][param_name] = {
            "name": "constant",
            "args": {"value": float(theta[i])}
        }

    # Set particle count and seed
    inst.settings['filter']['particles'] = num_particles
    inst.settings['filter']['prng_seed'] = seed

    # Build context
    context = inst.build_context()

    return context


def _get_history_snapshot(
    forecast,
    context,
    target_time: float,
    atol: float = 1e-8
):
    """
    Return a snapshot of the particle history at the requested time.

    Parameters
    ----------
    forecast
        The forecast result for a specific observation horizon.
    context
        The pypfilt simulation context used to generate the forecast.
    target_time : float
        The assimilation time to retrieve.
    atol : float
        Absolute tolerance when matching times (default: 1e-8)

    Returns
    -------
    snapshot : pypfilt.state.Snapshot or None
        The snapshot at the requested time, or None if unavailable.
    actual_time : float
        The time stamp associated with the returned snapshot (may differ
        slightly from target_time due to floating-point tolerances).
    """
    history = forecast.history
    times = np.asarray(history.times, dtype=float)

    if times.size == 0:
        logger.warning(
            "No history times recorded when requesting snapshot at %.3f",
            target_time
        )
        return None, target_time

    match_idx = np.where(np.isclose(times, target_time, atol=atol))[0]
    if match_idx.size == 0:
        # Fall back to the nearest earlier (or equal) time within tolerance.
        candidate_idx = np.where(times <= target_time + atol)[0]
        if candidate_idx.size == 0:
            logger.warning(
                "No particle history available at or before time %.3f",
                target_time
            )
            return None, target_time
        nearest_idx = candidate_idx[-1]
        nearest_time = times[nearest_idx]
        if abs(nearest_time - target_time) > atol:
            logger.warning(
                "Closest particle history time %.3f exceeds tolerance for "
                "requested time %.3f",
                nearest_time,
                target_time
            )
            return None, target_time
        match_idx = np.array([nearest_idx])

    hist_ix = int(match_idx[-1])
    actual_time = float(times[hist_ix])
    if hist_ix >= history.matrix.shape[0]:
        logger.warning(
            "History index %d out of bounds (size %d) for time %.3f",
            hist_ix,
            history.matrix.shape[0],
            actual_time
        )
        return None, actual_time
    steps_per_unit = context.settings['time']['steps_per_unit']

    try:
        snapshot = pypfilt.state.Snapshot(
            actual_time,
            steps_per_unit,
            history.matrix,
            hist_ix
        )
        return snapshot, actual_time
    except Exception as exc:
        logger.warning(
            "Failed to construct snapshot at time %.3f: %s",
            actual_time,
            exc
        )
        return None, actual_time


def extract_latent_state(
    fit_result,
    context,
    obs_time: float,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Sample a single latent state from the weighted particle cloud.

    Parameters
    ----------
    fit_result
        Result from pypfilt.forecast
    context
        pypfilt context used for forecasting
    obs_time : float
        Observation time to extract state from
    seed : int, optional
        Random seed for state sampling

    Returns
    -------
    state : dict
        Sampled state with keys: 'T', 'E', 'I', 'R', 'V'
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract particle cloud at observation time
    # Use estimation results (not forecasts) at the final observation time
    # Get the snapshot at the observation time
    forecast = fit_result.forecasts[obs_time]

    # Get the last assimilation time (not the forecast endpoint)
    # forecast.history.times[-1] may be the forecast end (e.g., 21.0)
    # We want the last time <= obs_time
    all_times = np.array(forecast.history.times)
    assimilation_times = all_times[all_times <= obs_time]
    last_time = assimilation_times[-1] if len(assimilation_times) > 0 else obs_time

    # Get snapshot with context
    snap, actual_time = _get_history_snapshot(forecast, context, last_time)
    if snap is None:
        raise ValueError(
            f"No particle history snapshot available at time {last_time}"
        )
    state_vec = snap.state_vec
    weights = snap.weights

    # Normalize weights (should already be normalized, but be safe)
    weights = weights / np.sum(weights)

    # Sample single particle index
    idx = np.random.choice(len(weights), p=weights)

    # Extract state
    state = {
        'T': float(state_vec[idx]['T']),
        'E': float(state_vec[idx]['E']),
        'I': float(state_vec[idx]['I']),
        'R': float(state_vec[idx]['R']),
        'V': float(state_vec[idx]['V'])
    }

    return state


def extract_filter_trajectory(
    fit_result,
    context,
    obs_time: float
) -> Tuple[np.ndarray, Dict]:
    """
    Extract summary statistics of particle filter trajectory over assimilation period.

    Parameters
    ----------
    fit_result
        Result from pypfilt.forecast
    context
        pypfilt context used for forecasting
    obs_time : float
        Final observation time

    Returns
    -------
    times : np.ndarray
        Array of time points
    trajectory_stats : dict
        Dictionary with keys 'median', 'q025', 'q25', 'q75', 'q975'
        Each value is an array of viral load observations (log10 space) at each time
    """
    forecast = fit_result.forecasts[obs_time]

    # Get all available times from the history
    all_times = np.array(forecast.history.times)

    # Filter to only include times up to and including obs_time
    mask = all_times <= obs_time
    times = all_times[mask]

    if len(times) == 0:
        # No history available, return empty
        logger.warning("No filter history available")
        return np.array([]), {}

    # Storage for viral load at each time
    V_median = np.zeros(len(times))
    V_q025 = np.zeros(len(times))
    V_q25 = np.zeros(len(times))
    V_q75 = np.zeros(len(times))
    V_q975 = np.zeros(len(times))

    detection_limit = -0.65

    # Extract viral load from each time point
    for i, t in enumerate(times):
        try:
            snap, actual_time = _get_history_snapshot(forecast, context, float(t))
            if snap is None:
                raise ValueError(
                    f"No particle history snapshot available at time {t}"
                )
            state_vec = snap.state_vec
            weights = snap.weights / np.sum(snap.weights)

            # Extract viral load from all particles
            V_particles = state_vec['V']

            # Convert to log10 space with detection limit
            V_log10 = np.zeros_like(V_particles)
            zero_mask = V_particles == 0
            V_log10[zero_mask] = detection_limit
            V_log10[~zero_mask] = np.log10(V_particles[~zero_mask])
            V_log10[V_log10 < detection_limit] = detection_limit

            # Compute weighted quantiles
            # For weighted quantiles, we need to sort and compute cumulative weights
            sorted_idx = np.argsort(V_log10)
            V_sorted = V_log10[sorted_idx]
            weights_sorted = weights[sorted_idx]
            cumsum_weights = np.cumsum(weights_sorted)

            # Find quantiles with bounds checking
            idx = int(np.searchsorted(cumsum_weights, 0.5))
            V_median[i] = V_sorted[min(idx, len(V_sorted) - 1)]
            idx = int(np.searchsorted(cumsum_weights, 0.025))
            V_q025[i] = V_sorted[min(idx, len(V_sorted) - 1)]
            idx = int(np.searchsorted(cumsum_weights, 0.25))
            V_q25[i] = V_sorted[min(idx, len(V_sorted) - 1)]
            idx = int(np.searchsorted(cumsum_weights, 0.75))
            V_q75[i] = V_sorted[min(idx, len(V_sorted) - 1)]
            idx = int(np.searchsorted(cumsum_weights, 0.975))
            V_q975[i] = V_sorted[min(idx, len(V_sorted) - 1)]

        except Exception as e:
            logger.warning(f"Failed to extract snapshot at time {t}: {e}")
            # Fill with detection limit if extraction fails
            V_median[i] = detection_limit
            V_q025[i] = detection_limit
            V_q25[i] = detection_limit
            V_q75[i] = detection_limit
            V_q975[i] = detection_limit

    trajectory_stats = {
        'median': V_median,
        'q025': V_q025,
        'q25': V_q25,
        'q75': V_q75,
        'q975': V_q975
    }

    return times, trajectory_stats


def _create_rate_function(theta_rates):
    """
    Create rate function for JSF simulator.

    Parameters
    ----------
    theta_rates : list
        Rate parameters [beta, phi, rho, k, delta, pi, c]

    Returns
    -------
    rates : callable
        Function that computes reaction rates from state
    """
    def rates(x, time):
        t, r, e, i, v = x  # State ordering: [T, R, E, I, V] in JSF internal
        # Note: JSF uses [T, R, E, I, V] internally, but we pass [T, E, I, R, V]
        # The tiv.py model handles this conversion

        # Apply scaling factors
        m_beta = theta_rates[0] * 10**(-9)
        m_phi = theta_rates[1] * 10**(-5)
        m_rho = theta_rates[2]
        m_k = theta_rates[3]
        m_delta = theta_rates[4]
        m_pi = theta_rates[5]
        m_c = theta_rates[6]

        return [
            m_beta * (t * v),    # Infection
            m_phi * i * t,       # Target -> Refractory
            m_rho * r,           # Refractory -> Target
            m_k * e,             # Eclipse -> Infected
            m_delta * i,         # Infected death
            m_pi * i,            # Virus production
            m_c * v              # Virus clearance
        ]

    return rates


def _create_stoichiometry():
    """
    Create stoichiometry matrix for JSF simulator.

    Returns
    -------
    stoich : dict
        Stoichiometry specification matching src.tiv._stoich
    """
    # Net change matrix (7 reactions × 5 compartments)
    # Compartments: [T, R, E, I, V]
    # Reactions: [infection, T->R, R->T, E->I, I death, V production, V clearance]
    _nu = [
        [-1, 0, 1, 0, 0],   # Infection: T->E
        [-1, 1, 0, 0, 0],   # T -> R
        [1, -1, 0, 0, 0],   # R -> T
        [0, 0, -1, 1, 0],   # E -> I
        [0, 0, 0, -1, 0],   # I death
        [0, 0, 0, 0, 1],    # V production
        [0, 0, 0, 0, -1]    # V clearance
    ]

    _nu_reactants = [
        [1, 0, 0, 0, 1],    # Infection needs T and V
        [1, 0, 0, 1, 0],    # T->R needs T and I
        [0, 1, 0, 0, 0],    # R->T needs R
        [0, 0, 1, 0, 0],    # E->I needs E
        [0, 0, 0, 1, 0],    # I death needs I
        [0, 0, 0, 1, 0],    # V production needs I
        [0, 0, 0, 0, 1]     # V clearance needs V
    ]

    _nu_products = [
        [0, 0, 1, 0, 1],    # Infection produces E (V unchanged in stoich, consumed in reactant)
        [0, 1, 0, 1, 0],    # T->R produces R (I unchanged, consumed in reactant)
        [1, 0, 0, 0, 0],    # R->T produces T
        [0, 0, 0, 1, 0],    # E->I produces I
        [0, 0, 0, 0, 0],    # I death produces nothing
        [0, 0, 0, 1, 1],    # V production produces V (I unchanged)
        [0, 0, 0, 0, 0]     # V clearance produces nothing
    ]

    stoich = {
        'nu': _nu,
        'DoDisc': [0, 0, 0, 0, 0, 0],
        'nuReactant': _nu_reactants,
        'nuProduct': _nu_products
    }

    return stoich


def simulate_forward(
    x0: Dict[str, float],
    theta: np.ndarray,
    fixed_params: Dict[str, float],
    horizon: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate viral dynamics forward from a latent state.

    Parameters
    ----------
    x0 : dict
        Initial latent state with keys 'T', 'E', 'I', 'R', 'V'
    theta : np.ndarray
        Parameter vector [lnV0, beta, phi, rho, delta, pi]
    fixed_params : dict
        Fixed parameters: 'c', 'k'
    horizon : int
        Number of days to simulate forward
    seed : int, optional
        Random seed for JSF simulation

    Returns
    -------
    states : np.ndarray
        State trajectories, shape (horizon+1, 5) [T, E, I, R, V]
        Includes initial state at index 0
    observations : np.ndarray
        Viral load observations in log10 space, shape (horizon+1,)
    """
    if seed is not None:
        np.random.seed(seed)

    # Prepare initial state vector [T, E, I, R, V]
    x_current = [x0['T'], x0['E'], x0['I'], x0['R'], x0['V']]

    # Prepare rate parameters [beta, phi, rho, k, delta, pi, c]
    theta_rates = [
        theta[1],  # beta
        theta[2],  # phi
        theta[3],  # rho
        fixed_params['k'],
        theta[4],  # delta
        theta[5],  # pi
        fixed_params['c']
    ]

    # Create rate function and stoichiometry
    rate_func = _create_rate_function(theta_rates)
    stoich = _create_stoichiometry()

    # JSF options (matching src.tiv.RefractoryCellModel_JSF)
    options = {
        'EnforceDo': [0, 0, 0, 0, 0],
        'dt': 0.00005,
        'SwitchingThreshold': [100, 100, 100, 100, 100]
    }

    # Storage for trajectories
    states = np.zeros((horizon + 1, 5))
    states[0] = x_current

    # Simulate forward day by day
    for day in range(horizon):
        xs, ts = JSF.JumpSwitchFlowSimulator(
            x_current,
            rate_func,
            stoich,
            1.0,  # Simulate 1 day
            options
        )

        # Extract final state
        # JSF returns xs as list of compartment trajectories
        # xs[i] is trajectory of compartment i
        x_current = [xs[i][-1] for i in range(len(xs))]
        states[day + 1] = x_current

    # Apply observation model to viral load trajectory
    # Extract V (last column)
    V_trajectory = states[:, 4]

    # Convert to log10 space with detection limit
    detection_limit = -0.65
    observations = np.zeros_like(V_trajectory)

    # Zero viral load -> detection limit
    zero_mask = V_trajectory == 0
    observations[zero_mask] = detection_limit

    # Non-zero viral load -> log10, clipped at detection limit
    observations[~zero_mask] = np.log10(V_trajectory[~zero_mask])
    observations[observations < detection_limit] = detection_limit

    return states, observations


def save_predictions(
    output_dir: str,
    theta_samples: np.ndarray,
    latent_states: List[Dict[str, float]],
    predicted_states: np.ndarray,
    predicted_observations: np.ndarray,
    metadata: Dict,
    config_path: str,
    filter_trajectories: Optional[List[Tuple[np.ndarray, Dict]]] = None
) -> None:
    """
    Save posterior predictive results to disk.

    Parameters
    ----------
    output_dir : str
        Directory to save results
    theta_samples : np.ndarray
        Parameter samples, shape (M, 6)
    latent_states : list of dict
        Sampled latent states at T_obs
    predicted_states : np.ndarray
        Predicted state trajectories, shape (M, H+1, 5)
    predicted_observations : np.ndarray
        Predicted observations, shape (M, H+1)
    metadata : dict
        Metadata from the prediction run
    config_path : str
        Path to config file used
    filter_trajectories : list of (times, stats), optional
        Particle filter trajectories for each sample during assimilation
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(output_path / 'theta_samples.npy', theta_samples)
    np.save(output_path / 'predicted_states.npy', predicted_states)
    np.save(output_path / 'predicted_observations.npy', predicted_observations)

    # Convert latent states to array
    M = len(latent_states)
    latent_array = np.zeros((M, 5))
    for i, state in enumerate(latent_states):
        latent_array[i] = [state['T'], state['E'], state['I'], state['R'], state['V']]
    np.save(output_path / 'latent_states_T.npy', latent_array)

    # Compute and save summary statistics
    import pandas as pd

    H = predicted_observations.shape[1] - 1
    summary_data = []

    for t in range(H + 1):
        obs_t = predicted_observations[:, t]
        summary_data.append({
            'time': t,
            'mean': np.mean(obs_t),
            'median': np.median(obs_t),
            'std': np.std(obs_t),
            'q025': np.quantile(obs_t, 0.025),
            'q25': np.quantile(obs_t, 0.25),
            'q75': np.quantile(obs_t, 0.75),
            'q975': np.quantile(obs_t, 0.975)
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path / 'summary_statistics.csv', index=False)

    # Save metadata
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        meta_save = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                meta_save[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                meta_save[key] = value.item()  # Convert numpy scalar to Python scalar
            else:
                meta_save[key] = value
        json.dump(meta_save, f, indent=2)

    # Copy config file
    import shutil
    shutil.copy(config_path, output_path / 'config.toml')

    # Save filter trajectories if provided
    if filter_trajectories is not None:
        # Check if we have valid filter trajectories
        # Require all quantile keys to be present
        required_keys = {'median', 'q025', 'q25', 'q75', 'q975'}
        valid_trajs = [(t, s) for t, s in filter_trajectories
                       if len(t) > 0 and
                          isinstance(s, dict) and
                          required_keys.issubset(s.keys())]

        if len(valid_trajs) > 0:
            # Aggregate filter trajectories across all samples
            # Average the summary statistics from each sample
            M = len(valid_trajs)
            times_ref = valid_trajs[0][0]  # Reference times from first sample

            # Initialize accumulators
            median_acc = np.zeros((M, len(times_ref)))
            q025_acc = np.zeros((M, len(times_ref)))
            q25_acc = np.zeros((M, len(times_ref)))
            q75_acc = np.zeros((M, len(times_ref)))
            q975_acc = np.zeros((M, len(times_ref)))

            # Collect trajectories from each sample
            for i, (times, stats) in enumerate(valid_trajs):
                median_acc[i] = stats['median']
                q025_acc[i] = stats['q025']
                q25_acc[i] = stats['q25']
                q75_acc[i] = stats['q75']
                q975_acc[i] = stats['q975']

            # Average across samples to get marginal over parameter uncertainty
            filter_summary = pd.DataFrame({
                'time': times_ref,
                'median': np.median(median_acc, axis=0),
                'q025': np.quantile(q025_acc, 0.025, axis=0),
                'q25': np.quantile(q25_acc, 0.25, axis=0),
                'q75': np.quantile(q75_acc, 0.75, axis=0),
                'q975': np.quantile(q975_acc, 0.975, axis=0),
            })

            filter_summary.to_csv(output_path / 'filter_trajectory.csv', index=False)
            np.save(output_path / 'filter_times.npy', times_ref)
            logger.info("Filter trajectories saved")
        else:
            logger.warning("No valid filter trajectories to save")

    logger.info(f"Results saved to {output_path}")
