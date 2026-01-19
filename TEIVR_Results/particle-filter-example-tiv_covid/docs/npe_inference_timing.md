# NPE Inference Timing Notes

## Observed behaviour

| Patient ID | Runtime for 10 000 samples | Time per sample | Source |
|------------|---------------------------:|----------------:|--------|
| 432192     | 8.65 s                     | 0.86 ms         | `TEIVR_Results/particle-filter-example-tiv_covid/results/npe/20251103_182129/inference/432192/inference_timing.txt` |
| 443108     | 38.62 s                    | 3.86 ms         | `TEIVR_Results/particle-filter-example-tiv_covid/results/npe/20251103_182129/inference/443108/inference_timing.txt` |
| 444332     | 0.09 s                     | 0.01 ms         | `TEIVR_Results/particle-filter-example-tiv_covid/results/npe/20251103_182129/inference/444332/inference_timing.txt` |
| 444391     | 0.35 s                     | 0.03 ms         | `TEIVR_Results/particle-filter-example-tiv_covid/results/npe/20251103_182129/inference/444391/inference_timing.txt` |
| 445602     | 0.09 s                     | 0.01 ms         | `TEIVR_Results/particle-filter-example-tiv_covid/results/npe/20251103_182129/inference/445602/inference_timing.txt` |
| 451152     | 0.08 s                     | 0.01 ms         | `TEIVR_Results/particle-filter-example-tiv_covid/results/npe/20251103_182129/inference/451152/inference_timing.txt` |

## Why the timings vary

- Stage‑3 inference draws posterior samples via `posterior.sample((args.num_samples,), x=x_obs)` in `TEIVR_Results/particle-filter-example-tiv_covid/COVID_TEIVR_NPE.py:449`. The returned `posterior` is an `sbi.inference.posteriors.direct_posterior.DirectPosterior`.
- `DirectPosterior.sample()` (see `/fred/oz022/tkimpson/miniconda3/lib/python3.10/site-packages/sbi/inference/posteriors/direct_posterior.py:134`) calls `rejection.accept_reject_sample(...)` to filter proposals from the flow so that samples respect the uniform-prior bounds.
- The helper `accept_reject_sample()` (`/fred/oz022/tkimpson/miniconda3/lib/python3.10/site-packages/sbi/samplers/rejection/rejection.py:254-358`) keeps looping until enough proposals have been accepted. Each iteration re-evaluates the neural flow and discards samples that fall outside the prior support.
- Patients whose observations keep the learned posterior well inside the prior box achieve a high acceptance rate, so the loop exits almost immediately (`<< 1 s`).
- Patients whose observations push the flow toward the prior boundaries—or where the flow “leaks” significant mass outside the prior—cause many rejections. The loop needs several additional batches and therefore takes longer (the 8–40 s runs in the log).

## Implications and options

- The variation is intrinsic to the SNPE rejection correction; it is not due to GPU activity (the job ran fully on CPU) nor to data I/O (patient files are tiny).
- For diagnostic visibility you can estimate acceptance directly with `posterior.leakage_correction(x_obs, num_rejection_samples=...)`, or temporarily reduce `--num-samples` during triage.
- If slow cases become common, consider rebuilding the posterior with `build_posterior(..., sample_with='mcmc')` or using fewer samples for routine checks, reserving large draws for the final patients of interest.
