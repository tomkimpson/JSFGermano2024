# Neural Posterior Estimation (NPE) for COVID TIV Inference

This directory contains a complete implementation of Neural Posterior Estimation using Simulation-Based Inference (SBI) as an alternative to the particle filter workflow.

## Overview

The NPE pipeline replaces the particle filter in `COVID_TEIVR_Inf_loop.py` while keeping the JSF simulator (`src/JSF_Solver_BasePython.py`) unchanged. It uses neural density estimation to learn a posterior distribution over the 6 inference parameters: `lnV0`, `beta`, `phi`, `rho`, `delta`, `pi`.

## Files Created

```
├── COVID_TEIVR_NPE.py          # Main entry point with 3-stage CLI
├── run_npe.sh                   # SLURM script for automated execution
├── requirements_npe.txt         # NPE-specific dependencies
├── src/npe_utils.py             # Utility functions
├── npe_outputs/                 # Output directory structure
│   ├── training/                # Simulated training data
│   ├── models/                  # Trained posteriors
│   └── inference/               # Per-patient inference results
└── NPE_README.md                # This file
```

## Installation

### 1. Install NPE Dependencies

The NPE workflow requires PyTorch and SBI. Install them in your existing virtual environment:

```bash
# Activate your existing virtual environment
source venv/bin/activate

# Install NPE-specific requirements
pip install -r requirements_npe.txt
```

**Note:** If you encounter issues with PyTorch installation on your HPC system, you may need to:
- Use a CPU-only version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Or load a pre-installed PyTorch module if available on your cluster

### 2. Verify Installation

```bash
python -c "import torch; import sbi; print('PyTorch:', torch.__version__); print('SBI:', sbi.__version__)"
```

## Usage

The NPE workflow consists of three stages that can be run separately or together:

### Stage 1: Simulate Training Data

Generate training data by running the JSF simulator with parameters sampled from the prior:

```bash
python COVID_TEIVR_NPE.py simulate --num-trajectories 1000
```

**Options:**
- `--num-trajectories N`: Number of simulations (default: 1000)
- `--num-timepoints T`: Time points per trajectory (default: 10)
- `--config PATH`: Path to TOML config (default: config/cli-refractory-tiv-jsf.toml)
- `--seed S`: Random seed (default: 42)

**Output:** `npe_outputs/training/simulations_1000.npz`

**Time estimate:** With default settings, ~1-2 seconds per simulation
- 1,000 simulations: ~20-30 minutes
- 10,000 simulations: ~3-5 hours

### Stage 2: Train Neural Posterior Estimator

Train a neural spline flow (NSF) to learn the posterior distribution:

```bash
python COVID_TEIVR_NPE.py train --num-trajectories 1000
```

**Options:**
- `--num-trajectories N`: Must match Stage 1 (default: 1000)
- `--batch-size B`: Training batch size (default: 128)
- `--max-epochs E`: Maximum training epochs (default: 100)
- `--training-fraction F`: Fraction for training vs validation (default: 0.8)

**Output:** `npe_outputs/models/posterior_N1000.pkl`

**Time estimate:**
- 1,000 simulations: ~5-10 minutes
- 10,000 simulations: ~30-60 minutes

### Stage 3: Inference on Patient Data

Use the trained posterior to infer parameters for each patient:

```bash
# All patients
python COVID_TEIVR_NPE.py infer --patients all --plot

# Specific patients
python COVID_TEIVR_NPE.py infer --patients 432192,443108 --plot
```

**Options:**
- `--patients IDS`: Comma-separated patient IDs or "all" (default: all)
- `--num-trajectories N`: Must match Stage 2 (default: 1000)
- `--num-samples S`: Posterior samples to draw (default: 10000)
- `--plot`: Generate diagnostic plots

**Output (per patient):**
- `npe_outputs/inference/{patient_id}/samples.npy`: Raw posterior samples
- `npe_outputs/inference/{patient_id}/summary.csv`: Posterior statistics
- `npe_outputs/inference/{patient_id}/posterior_histograms.png`: Diagnostic plots

**Time estimate:** ~1-5 seconds per patient

## Running on SLURM

The `run_npe.sh` script automates all three stages:

### Basic Usage

```bash
# Submit job to run all three stages
sbatch run_npe.sh

# With custom settings
sbatch run_npe.sh --num-trajectories 5000 --patients 432192
```

### Skip Stages

If you've already completed some stages and want to resume:

```bash
# Skip simulation, only train and infer
sbatch run_npe.sh --skip-simulate

# Only run inference (simulation and training already done)
sbatch run_npe.sh --skip-simulate --skip-train --patients all
```

### Command-line Options

- `--num-trajectories N`: Number of simulations (default: 1000)
- `--skip-simulate`: Skip Stage 1
- `--skip-train`: Skip Stage 2
- `--skip-infer`: Skip Stage 3
- `--patients IDS`: Patient list (default: all)
- `--num-samples S`: Posterior samples (default: 10000)

### Resource Allocation

The default SLURM settings in `run_npe.sh`:
- **CPUs:** 4
- **Memory:** 32GB
- **Time:** 48 hours
- **Job name:** covid_npe_inference

Adjust these in `run_npe.sh` based on your needs and cluster policies.

## Quick Start Guide

### Minimal Test Run

Test the pipeline with a small number of simulations first:

```bash
# 1. Generate 100 training simulations (fast test)
python COVID_TEIVR_NPE.py simulate --num-trajectories 100

# 2. Train on 100 simulations
python COVID_TEIVR_NPE.py train --num-trajectories 100

# 3. Infer for one patient
python COVID_TEIVR_NPE.py infer --patients 432192 --num-trajectories 100 --plot
```

### Full Production Run

For publication-quality results:

```bash
# Submit SLURM job with 10,000 simulations
sbatch run_npe.sh --num-trajectories 10000
```

Or run stages manually:

```bash
# Stage 1: Simulate (may take several hours)
python COVID_TEIVR_NPE.py simulate --num-trajectories 10000

# Stage 2: Train
python COVID_TEIVR_NPE.py train --num-trajectories 10000 --max-epochs 200

# Stage 3: Infer
python COVID_TEIVR_NPE.py infer --patients all --num-trajectories 10000 --plot
```

## Output Interpretation

### Summary Statistics

For each patient, `summary.csv` contains:
- **mean**: Posterior mean (point estimate)
- **std**: Posterior standard deviation (uncertainty)
- **q025, q975**: 95% credible interval
- **median**: Posterior median

### Posterior Samples

The `samples.npy` file contains raw MCMC-like samples from the posterior:
- Shape: `(num_samples, 6)` where 6 = number of parameters
- Can be used for downstream analysis, correlation analysis, etc.

### Diagnostic Plots

The `posterior_histograms.png` shows marginal posterior distributions for each parameter with mean and median marked.

## Comparison with Particle Filter

To compare NPE results with the existing particle filter:

1. **Particle filter outputs:** `outputs4/{patient_id}/*/fit_result.pkl`
2. **NPE outputs:** `npe_outputs/inference/{patient_id}/summary.csv`

You can create a comparison script or load both and compute:
- Overlap of credible intervals
- KL divergence between posterior samples
- Difference in point estimates

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements_npe.txt
```

### Memory Issues

If training runs out of memory:
- Reduce `--batch-size` (try 64 or 32)
- Reduce `--num-trajectories` for initial tests
- Request more memory in SLURM script

### Slow Simulations

JSF simulations are stochastic and can vary in runtime:
- Start with small `--num-trajectories` to estimate time
- Consider parallelizing Stage 1 if needed (can be done by submitting multiple jobs with different seeds and merging data)

### Training Not Converging

If validation loss doesn't decrease:
- Increase `--max-epochs`
- Try different `--batch-size`
- Ensure training data quality (check `simulations_*.npz` for NaNs)

## Design Decisions

### Parameter Space
- NPE learns in the **raw parameter space** from TOML (e.g., beta ∈ [0, 20])
- Scaling (e.g., `beta × 10^-9`) happens inside the simulator
- This ensures direct comparability with particle filter results

### Observation Model
- Matches `src.tiv.Gaussian` exactly:
  - Zero virus counts → detection limit (-0.65)
  - Non-zero counts → log10(count), clipped at -0.65
  - Gaussian noise with scale from TOML (default: 1.0)

### Fixed vs Inference Parameters
- **Inference parameters (6):** lnV0, beta, phi, rho, delta, pi
- **Fixed parameters:** c=10.0, k=4.0, T0=8E7, E0=1.0, I0=0.0, R0=0.0
- Matches particle filter configuration in `config/cli-refractory-tiv-jsf.toml`

### Neural Architecture
- Uses Neural Spline Flows (NSF) from the `sbi` package
- NSF can capture complex, multimodal posteriors
- No manual feature engineering required

## References

- **SBI Package:** [sbi-dev/sbi](https://github.com/sbi-dev/sbi)
- **SBI Tutorial:** [https://www.mackelab.org/sbi/](https://www.mackelab.org/sbi/)
- **NPE Paper:** Papamakarios et al. (2019), "Sequential Neural Likelihood"

## Contact

For questions or issues with the NPE implementation, check:
1. This README
2. The original plan in `npe_plan.md`
3. Inline documentation in `COVID_TEIVR_NPE.py` and `src/npe_utils.py`
