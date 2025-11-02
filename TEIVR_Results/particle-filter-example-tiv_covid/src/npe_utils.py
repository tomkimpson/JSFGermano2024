"""
Utility functions for Neural Posterior Estimation workflow.

This module provides helpers for:
- TOML configuration parsing
- Prior sampling and transformation
- Observation preprocessing (log10 + detection limit)
- Data persistence and loading
"""

import tomli
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import torch


# Detection limit matching src.tiv.Gaussian
DETECTION_LIMIT = -0.65


def load_config(config_path: str = "config/cli-refractory-tiv-jsf.toml") -> Dict:
    """
    Load and parse the TOML configuration file.

    Args:
        config_path: Path to TOML configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
    return config


def get_prior_bounds(config: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract prior bounds for inference parameters from TOML config.

    Returns uniform prior bounds for the 6 inference parameters:
    lnV0, beta, phi, rho, delta, pi

    Args:
        config: Loaded TOML configuration

    Returns:
        lower_bounds: Array of lower bounds (shape: 6,)
        upper_bounds: Array of upper bounds (shape: 6,)
        param_names: List of parameter names
    """
    scenario = config['scenario']['inference']['prior']

    # Define the 6 inference parameters in order
    param_names = ['lnV0', 'beta', 'phi', 'rho', 'delta', 'pi']

    lower_bounds = []
    upper_bounds = []

    for param in param_names:
        if param in scenario:
            prior_spec = scenario[param]
            if prior_spec['name'] == 'uniform':
                loc = prior_spec['args']['loc']
                scale = prior_spec['args']['scale']
                lower_bounds.append(loc)
                upper_bounds.append(loc + scale)
            else:
                raise ValueError(f"Unsupported prior type for {param}: {prior_spec['name']}")
        else:
            raise ValueError(f"Parameter {param} not found in TOML configuration")

    return np.array(lower_bounds), np.array(upper_bounds), param_names


def get_fixed_params(config: Dict) -> Dict[str, float]:
    """
    Extract fixed parameters from TOML config.

    Returns:
        Dictionary of fixed parameter values
    """
    scenario = config['scenario']['inference']['prior']
    prior = config['prior']

    fixed_params = {
        'c': scenario['c']['args']['value'],
        'k': scenario['k']['args']['value'],
        'T0': prior['T0']['args']['value'],
        'E0': prior['E0']['args']['value'],
        'I0': prior['I0']['args']['value'],
        'R0': prior['R0']['args']['value'],
    }

    return fixed_params


def get_observation_scale(config: Dict) -> float:
    """
    Extract observation noise scale from TOML config.

    Returns:
        Observation scale (standard deviation)
    """
    return config['observations']['V']['scale']


def sample_from_prior(
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    n_samples: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample parameters from uniform prior.

    Args:
        lower_bounds: Lower bounds for each parameter
        upper_bounds: Upper bounds for each parameter
        n_samples: Number of samples to draw
        seed: Random seed for reproducibility

    Returns:
        Array of parameter samples (shape: n_samples x n_params)
    """
    if seed is not None:
        np.random.seed(seed)

    n_params = len(lower_bounds)
    samples = np.random.uniform(
        low=lower_bounds,
        high=upper_bounds,
        size=(n_samples, n_params)
    )

    return samples


def apply_observation_model(
    virus_counts: np.ndarray,
    scale: float,
    detection_limit: float = DETECTION_LIMIT,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Apply observation model: log10 transformation with detection limit and Gaussian noise.

    Matches the logic in src.tiv.Gaussian:
    - Zero counts -> detection_limit
    - Non-zero counts -> log10(count), clipped at detection_limit
    - Add Gaussian noise with given scale

    Args:
        virus_counts: Array of virus counts (shape: n_samples x n_timepoints)
        scale: Standard deviation of Gaussian noise
        detection_limit: Detection limit value (default: -0.65)
        seed: Random seed for reproducibility

    Returns:
        Array of noisy observations in log10 space (shape: n_samples x n_timepoints)
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize output array
    observations = np.zeros_like(virus_counts, dtype=float)

    # Handle zero counts
    zero_mask = virus_counts == 0
    observations[zero_mask] = detection_limit

    # Handle non-zero counts
    non_zero_mask = ~zero_mask
    observations[non_zero_mask] = np.log10(virus_counts[non_zero_mask])

    # Clip at detection limit
    observations = np.maximum(observations, detection_limit)

    # Add Gaussian noise
    # For values at detection limit, use very small scale (matching tiv.py:327)
    noise = np.random.normal(0, scale, size=observations.shape)
    at_limit_mask = observations == detection_limit
    noise[at_limit_mask] *= 1e-32  # Very small noise for detection limit

    observations += noise

    return observations


def load_patient_data(
    patient_id: str,
    data_dir: str = "data",
    n_timepoints: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load patient observation data from SSV file.

    Args:
        patient_id: Patient identifier (e.g., '432192')
        data_dir: Directory containing patient data files
        n_timepoints: Number of time points to load (default: 10)

    Returns:
        times: Array of observation times
        observations: Array of observations in log10 space
    """
    file_path = Path(data_dir) / f"{patient_id}.ssv"

    if not file_path.exists():
        raise FileNotFoundError(f"Patient data file not found: {file_path}")

    # Load data
    data = pd.read_csv(file_path, sep=' ')

    # Extract first n_timepoints
    times = data['time'].values[:n_timepoints]
    observations = data['value'].values[:n_timepoints]

    if len(times) < n_timepoints:
        raise ValueError(
            f"Patient {patient_id} has only {len(times)} time points, "
            f"but {n_timepoints} were requested"
        )

    return times, observations


def save_training_data(
    theta: np.ndarray,
    x_obs: np.ndarray,
    output_path: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save training data to disk.

    Args:
        theta: Parameter samples (shape: n_samples x n_params)
        x_obs: Simulated observations (shape: n_samples x n_timepoints)
        output_path: Path to save file (will use .npz format)
        metadata: Optional metadata dictionary
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'theta': theta,
        'x_obs': x_obs,
    }

    if metadata is not None:
        # Convert metadata to saveable format
        for key, value in metadata.items():
            save_dict[f'meta_{key}'] = value

    np.savez(output_file, **save_dict)
    print(f"Training data saved to {output_path}")


def load_training_data(input_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load training data from disk.

    Args:
        input_path: Path to saved .npz file

    Returns:
        theta: Parameter samples
        x_obs: Simulated observations
        metadata: Metadata dictionary
    """
    data = np.load(input_path, allow_pickle=True)

    theta = data['theta']
    x_obs = data['x_obs']

    # Extract metadata
    metadata = {}
    for key in data.files:
        if key.startswith('meta_'):
            metadata[key[5:]] = data[key].item() if data[key].ndim == 0 else data[key]

    return theta, x_obs, metadata


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor with appropriate dtype.

    Args:
        x: Numpy array

    Returns:
        PyTorch tensor (float32)
    """
    return torch.tensor(x, dtype=torch.float32)


def from_tensor(x: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.

    Args:
        x: PyTorch tensor

    Returns:
        Numpy array
    """
    return x.detach().cpu().numpy()


def get_all_patient_ids(data_dir: str = "data") -> List[str]:
    """
    Get list of all patient IDs from data directory.

    Args:
        data_dir: Directory containing patient data files

    Returns:
        List of patient IDs
    """
    data_path = Path(data_dir)
    patient_files = list(data_path.glob("*.ssv"))
    patient_ids = [f.stem for f in patient_files]
    return sorted(patient_ids)


def print_prior_summary(lower_bounds: np.ndarray, upper_bounds: np.ndarray, param_names: List[str]) -> None:
    """
    Print a summary of the prior bounds.

    Args:
        lower_bounds: Lower bounds for each parameter
        upper_bounds: Upper bounds for each parameter
        param_names: Names of parameters
    """
    print("\nPrior bounds:")
    print("-" * 50)
    for name, lb, ub in zip(param_names, lower_bounds, upper_bounds):
        print(f"{name:10s}: [{lb:8.3f}, {ub:8.3f}]")
    print("-" * 50)
