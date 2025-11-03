# Parallel Simulation Plan

## Objectives
- Accelerate Stage 1 of `COVID_TEIVR_NPE.py` by distributing JSF simulations across multiple CPU cores.
- Preserve existing deterministic behaviour (per-trajectory seeding, prior sampling, saved metadata).
- Expose core-count controls both via CLI (`--num-workers`) and SLURM script (`run_npe.sh`), keeping defaults compatible with single-core runs.

## Proposed Changes
- **Stage 1 orchestration (`COVID_TEIVR_NPE.py`)**
  - Add a `--num-workers` option (default = 1) to the `simulate` sub-command.
  - Introduce a worker wrapper (`_simulate_single`) that packages the arguments needed per trajectory (parameter draw, fixed params, observation scale, seed).
  - Use `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor` to map over the sampled parameter list when `num_workers > 1`; fall back to the current sequential loop when `num_workers == 1`.
  - Ensure each worker receives a unique seed derived from the base seed and trajectory index; maintain the existing metadata (`seed` field) to record the base seed and the worker count used.
  - Guard the parallel block with `if __name__ == "__main__":` semantics (already true because we launch via CLI) to avoid issues on some platforms.

- **Progress reporting**
  - Replace the existing single `tqdm` loop with:
    - `tqdm(executor.map(...), total=num_trajectories)` so progress is preserved; or
    - For `multiprocessing.Pool`, use `imap_unordered` with manual progress updates.
  - Confirm progress bars still appear cleanly in SLURM logs.

- **run_npe.sh**
  - Add an optional `--num-workers` flag passthrough.
  - Update `#SBATCH --cpus-per-task` comment/documentation to remind users to request at least as many CPUs as workers; possibly bump the default from 4 → 8 to reflect parallel usage.

- **Metadata & reproducibility**
  - Extend the saved `.npz` metadata (`save_training_data`) with `num_workers` so downstream stages can report the configuration used.
  - Document the parallel option in `NPE_README.md` (usage example, guidance on resource selection).

## Implementation Steps
1. Modify the CLI parser in `COVID_TEIVR_NPE.py` (`parser_sim`) to accept `--num-workers`.
2. Refactor `stage1_simulate`:
   - Generate the list of `(theta, seed)` tuples once.
   - Invoke `_simulate_single` either sequentially or via a process pool based on `args.num_workers`.
   - Aggregate the returned observation arrays into the existing `x_obs` structure.
3. Update `run_npe.sh` to forward `--num-workers` and document the expected CPU allocation.
4. Persist the new metadata field in `npe_utils.save_training_data` and load/display it when present.
5. Amend `NPE_README.md` to explain parallel simulation usage and provide SLURM examples (e.g., `sbatch run_npe.sh --num-trajectories 20000 --num-workers 16`).

## Validation
- Run `python COVID_TEIVR_NPE.py simulate --num-trajectories 20 --num-workers 1` and confirm outputs match the current sequential behaviour.
- Run with `--num-workers 4` and verify:
  - Runtime decreases roughly in proportion to worker count.
  - Resulting `.npz` files are identical (up to ordering) when seeds are fixed.
  - Progress bars/log output remain readable.
- Optionally add a quick check to `test_npe_setup.py` to ensure the new CLI option parses without raising errors.

