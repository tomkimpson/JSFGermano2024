# Neural Posterior Estimation Implementation Plan

## Goals & Constraints
- Replace the particle-filter workflow (`COVID_TEIVR_Inf_loop.py`, `run.sh`) with an SBI-based neural posterior estimation (NPE) pipeline while keeping the JSF simulator untouched.
- Condition all training and inference on the first 10 observation times, matching the existing particle-filter outputs (`forecasts[10.0]` in saved `fit_result` pickles).
- Use the same six inference parameters (`lnV0`, `beta`, `phi`, `rho`, `delta`, `pi`) and identical prior ranges as defined in `config/cli-refractory-tiv-jsf.toml`.
- Start with a light training set of 1,000 simulated trajectories; design the code so the sample size is configurable for future runs.
- Deliver a dedicated SLURM entry point `run_npe.sh`; keep the current `run.sh` untouched.

## Stage 0 – Preparation
- Add SBI and Torch dependencies to `requirements.txt` (or a separate NPE requirements file if isolation is preferred).
- Introduce a new Python entry point (e.g., `COVID_TEIVR_NPE.py`) with sub-commands or CLI flags to run the three stages: `simulate`, `train`, and `infer`.
- Create an output directory structure (e.g., `npe_outputs/`) mirroring the current `outputs4/` layout (patient-specific folders, per-model subdirectories).
- Build utility helpers:
  - TOML loader to extract priors, observation noise, and shared settings from `config/cli-refractory-tiv-jsf.toml`.
  - Conversion routines for transforming simulated virus counts into log10 measurements with the detection limit at `-0.65`, matching `src.tiv.Gaussian`.
  - Reusable sampling utilities for drawing parameters from uniform priors and for seeding RNGs to ensure reproducibility.

## Stage 1 – Simulate Training Data
1. **Parameter sampling**
   - Parse the priors for `lnV0`, `beta`, `phi`, `rho`, `delta`, and `pi` from the TOML file.
   - Sample 1,000 parameter vectors (`torch` or `numpy`) from these priors; allow the sample size to be configurable via CLI.
2. **Initial state construction**
   - Use the constant priors (`T0`, `E0`, `I0`, `R0`, `c`, `k`) as fixed initial conditions/parameters, mirroring `src.tiv.RefractoryCellModel_JSF.init`.
   - Compute the initial viral load as `V0 = round(exp(lnV0))`.
3. **Trajectory generation**
   - For each sampled parameter set, call the JSF simulator:
     - Use `JSF.JumpSwitchFlowSimulator` exactly as in `src/tiv.RefractoryCellModel_JSF.update`, with `dt = 5e-5` and the same `SwitchingThreshold`.
     - Simulate day-to-day dynamics for a total time of 10 units (matching the inference window) with 1-day aggregation using the existing Euler/logics.
   - Cache any expensive setup (e.g., stoichiometry matrices) to avoid work duplication.
4. **Observation model**
   - Transform latent virus counts to log10 space with the detection limit logic implemented in `src.tiv.Gaussian`:
     - Zero counts → `-0.65`; non-zero counts → `log10(count)` clipped at `-0.65`.
     - Add Gaussian noise using `scale = 1.0` (from `config/observations.V.scale`).
   - Record summary statistics per trajectory as a 10-step sequence (optionally flatten to a 10-element vector or store as `(10, 1)` tensor).
5. **Data persistence**
   - Save simulations to disk (e.g., `npe_outputs/training/data.npz` or a Torch file) containing:
     - `theta` array with sampled parameters.
     - `x_obs` array with the noisy 10-step observations.
     - Metadata (prior ranges, detection limit, RNG seeds) to ensure reproducibility.
   - Optionally store raw latent trajectories for diagnostics.

## Stage 2 – Train the NPE Model
1. **Dataset loader**
   - Create a dataset module that reads the saved simulation file, converts arrays to Torch tensors, and optionally standardises observations (e.g., z-score across the training set). Record any transforms for later inference.
2. **SBI setup**
   - Instantiate an SBI `NeuralPosteriorEstimator` (e.g., `sbi.inference.SNPE` with an NSF or MAF density estimator).
   - Provide a prior object to SBI that matches the simulation priors (likely a `sbi.utils.BoxUniform` built from the TOML bounds).
   - Feed the 1,000 simulation pairs `(theta, x_obs)` to `SNPE.append_simulations(...)` and run `train()`.
   - Configure training hyperparameters for a small dataset (e.g., batch size 128, patience-based early stopping).
   - Save the trained posterior to `npe_outputs/models/posterior.pkl` (or Torch checkpoint).
3. **Diagnostics**
   - After training, sample diagnostics (e.g., posterior predictive checks on held-out simulations) and store quick plots/logs.
   - Log training metadata (epochs, loss curve, PyTorch seed) for reproducibility.

## Stage 3 – Perform Inference Per Patient
1. **Observation preprocessing**
   - For each patient file in `data/*.ssv`, read the first 10 rows (time 1–10).
   - Apply the same log10 + detection limit transform used during simulation.
   - Apply any normalisation fitted in Stage 2 (e.g., subtract training mean, divide by std).
2. **Posterior evaluation**
   - Load the trained posterior and condition on each patient’s observation vector using `posterior.set_default_x(x_obs)`.
   - Draw posterior samples (e.g., 10,000) for summary statistics and credible intervals.
3. **Outputs**
   - For each patient, store:
     - Posterior samples (e.g., `samples.npy`).
     - Summary statistics (posterior means, 95% intervals) in CSV.
     - Optional diagnostic plots comparing posterior vs. prior.
   - Save results to `npe_outputs/<patient_id>/`.
4. **Optional predictive checks**
   - (Future work) Generate posterior predictive trajectories via the JSF simulator for validation.

## Automation – `run_npe.sh`
- Create a new SLURM script (`run_npe.sh`) that activates the virtual environment, prints environment info, and sequentially invokes:
  1. `python COVID_TEIVR_NPE.py simulate --num-trajectories 1000`
  2. `python COVID_TEIVR_NPE.py train`
  3. `python COVID_TEIVR_NPE.py infer --patients all` (or a comma-separated list)
- Provide CLI flags to skip stages (e.g., `--skip-simulate`) when reusing artifacts.
- Mirror logging behaviour from `run.sh` (job info, start/end timestamps, exit-code check).

## Validation & Next Steps
- Verify the simulated observation distribution visually against actual patient data to ensure the detection-limit handling matches expectations.
- Compare NPE posterior summaries for at least one patient against the particle-filter posterior to sanity-check ranges.
- Plan for scaling up:
  - Increase simulation count and monitor training time/memory.
  - Explore amortising over longer observation windows if needed.
- Document instructions in `readme.org` once the implementation is complete.

