# Posterior Predictive Integration Plan

## Goal
- Produce posterior predictive viral load trajectories that combine parameter uncertainty from the Neural Posterior Estimator (NPE) with latent-state uncertainty from a particle filter that conditions on patient observations.
- Reuse existing infrastructure in `COVID_TEIVR_NPE.py` (NPE pipeline) and `src/tiv.py` (particle filter model) while avoiding the full cost of the original inference loop.

## Current State Recap
- Stage 1–3 of the NPE workflow (`COVID_TEIVR_NPE.py`) already simulate training data, train an SNPE posterior (`sbi`), and export patient-specific posterior samples to `results/npe/<run>/inference/<patient>/samples.npy`.
- The particle filter pipeline (`COVID_TEIVR_Inf_loop.py`, `COVID_TEIVR_Analysis.py`) uses `pypfilt` with `src.tiv.RefractoryCellModel_JSF` to assimilate observations and generate forecasts/diagnostic tables saved under `results/particle_filter/<run>/...`.
- `RefractoryCellModel_JSF.init` expects arrays of parameter values per particle in `ctx.data['prior']`, and `update` advances each particle by running the JSF simulator for a one-day step.
- Observations (viral load in log10 space with detection limit) are read from `data/<patient>.ssv`; time grid typically spans 14 days.

## End-to-End Predictive Workflow
1. **Load Artefacts**
   - Locate the trained posterior pickle (`results/npe/<run>/models/posterior_N*.pkl`) and metadata (`*.meta.npz`) to recover parameter ordering.
   - Read patient observations using `src/npe_utils.load_patient_data` to determine the assimilation horizon `T` (largest observation time).

2. **Sample Parameter Sets**
   - Draw `M` parameter samples from the posterior (`posterior.sample((M,), x=obs_tensor)`), optionally in batches to control GPU/CPU memory.
   - Persist the sampled `theta` (shape `M × 6`) for reuse without resampling during later stages.
   - Provide CLI options for `M`, random seed, and patient list.

3. **Build Fixed-Parameter Particle Filter Context**
   - Start from a fresh `pypfilt` instance (`pypfilt.load_instances(config_path)`).
   - For each sampled `theta`, overwrite the prior specification before `inst.build_context()`:
     - Set `inst.settings['scenario']['inference']['prior'][param]` to `{"name": "constant", "args": {"value": theta[i]}}` for `lnV0`, `beta`, `phi`, `rho`, `delta`, `pi`.
     - Optionally reduce `inst.settings['filter']['particles']` (e.g., 1000 instead of 6000) and tweak `prng_seed` per sample to decorrelate trajectories.
   - Point `inst.settings['observations']['V']['file']` at the patient’s `.ssv` file.
   - Encapsulate this logic in a new helper (e.g., `src/posterior_predictive.py::build_context_for_theta(...)`).

4. **Assimilate Observations with Simplified Filter**
   - Define assimilation horizon `T_obs` as the maximum observation time (e.g., `times[-1]` from data).
   - Run `pypfilt.forecast(context, [T_obs], filename=None)` to propagate particles while conditioning on the patient data.
   - Extract the weighted particle cloud at `T_obs` via `fit_result.forecasts[T_obs].state_vec` and `fit_result.forecasts[T_obs].weights`.
   - Implement a utility to draw a single latent state sample `x_T` from this discrete distribution (`np.random.choice` with weights).
   - Collect `x_T` in state ordering `[T, R, E, I, V]`; map to `(T, E, I, R, V)` if convenient.

5. **Forward Simulation for Posterior Predictive Trajectories**
   - Reuse JSF stepping (`JSF.JumpSwitchFlowSimulator`) to project `(theta, x_T)` forward for `H` days (user-configurable).
   - Derive rate function identical to `simulate_trajectory` but starting from the latent state `x_T` instead of deterministic initial conditions.
   - Emit both latent compartment trajectories and observation model outputs (log10 viral load with detection limit via `npe_utils.apply_observation_model`).
   - Store time series per sample as arrays of shape `(H+1, state_dim)` (including day 0 = current state).

6. **Batching and Parallelism**
   - Because running `pypfilt` `M` times is costly, offer batching options:
     - Sequential mode (default) for reproducibility.
     - Optional multiprocessing where each worker handles a chunk of `theta` samples (mirroring `stage1_simulate` pattern).
   - Monitor runtime and allow particle count, `dt`, and `SwitchingThreshold` overrides through CLI flags to tune the “simplified” filter.

7. **Result Packaging**
   - Create a new results tree, e.g., `results/posterior_predictive/<timestamp>/<patient>/`.
   - Save:
     - `theta_samples.npy`
     - `latent_states_T.npy` (sampled `x_T`)
     - `predicted_states.npy` (shape `M × (H+1) × 5`)
     - `predicted_observations.npy` (modelled viral load trajectories)
     - Summary statistics CSV (mean/median/quantiles per time for V).
   - Optionally generate quicklook plots (fan chart of viral load predictions).

8. **CLI / API Surface**
   - Extend `COVID_TEIVR_NPE.py` with a `predict` subcommand or add a new script (`COVID_TEIVR_Predict.py`):
     - Arguments: `--config`, `--posterior-run`, `--patients`, `--num-parameter-samples`, `--particles-per-filter`, `--horizon`, `--seed`, `--num-workers`.
     - Reuse helper functions for IO, seeding, and logging.

9. **Validation Plan**
   - Smoke test on a single patient with small `M` (e.g., 5 samples) and `H=5`.
   - Compare the marginal parameter distributions of sampled `theta` with NPE summaries to ensure truncation isn’t occurring.
   - Check that the latent state distribution after assimilation is sensible (e.g., V non-negative, extinction flags).
   - Validate predictive summaries against historical particle-filter forecasts for consistency.
   - Document runtime and resource usage in `docs/`.

10. **Future Enhancements / Considerations**
    - Caching: memoize contexts or re-use random streams if multiple `theta` share similar values.
    - Diagnostics: track effective sample size of particles post-assimilation; warn if degeneracy occurs.
    - Support for drawing multiple `x_T` per `theta` to capture state uncertainty conditional on parameters.
    - Allow optional reweighting of `theta` samples if posterior sampling is truncated by filter failures.

## Next Steps
- Implement helpers in `src/posterior_predictive.py` (context builder, state sampler, forward simulator).
- Wire new `predict` subcommand that orchestrates the workflow and writes artefacts.
- Add lightweight regression test (e.g., under `tests/`) to ensure a fixed seed produces stable output dimensions and non-NaN trajectories.
- Update project docs/readme with usage instructions once implementation is complete.
