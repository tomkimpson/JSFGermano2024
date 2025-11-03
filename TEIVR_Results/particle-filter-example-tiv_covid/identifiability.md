# Identifiability Notes for TEIVR Inference

## Context
- Particle-filter runs in `config/cli-refractory-tiv-jsf.toml` use kernel regularisation with broad bounds (`beta` allowed up to `1e4`, etc.) even though the stated priors are narrower (e.g. `config/cli-refractory-tiv-jsf.toml:62`, `config/cli-refractory-tiv-jsf.toml:106`).
- SNPE posteriors for patient `432192` (`npe_outputs_saved/inference/432192/`) retain multi-modal structure, most notably in `(φ, δ)`.
- Viral load data (the only observable passed to both inference pipelines) is taken from a single column in `data/432192.ssv` via `src/npe_utils.py:191`.

## Posterior Behaviour
- Clustering the SNPE samples produces three dense groups in the `(φ, δ)` plane: roughly `(2.5, 4.8)`, `(7.7, 7.4)`, and `(11.6, 4.0)` (all in scaled units).
- Each cluster keeps similar values for `β`, `π`, `ρ`, and `ln V₀`, implying that distinct φ–δ trade-offs generate the same viral trajectory.
- The particle filter does not expose this structure because resampling plus kernel regularisation tends to concentrate particles in one of the equivalent regions.

## Mechanistic Explanation
- In the JSF model (`src/tiv.py:56-68`), φ accelerates the removal of target cells into the refractory class through infected cells, while δ removes infectious cells and feeds the viral production term `π · I`. The only data stream is viral load `V`, whose differential equation is driven predominantly by `π · I − c · V`.
- Because `I(t)` itself is governed by φ and δ, the viral trajectory depends on composite quantities such as the build-up and decay rates of `I`, not on φ and δ individually. Multiple φ–δ combinations can yield indistinguishable `I(t)` curves, hence identical `V(t)`.
- Formally, with only `V(t)` observed the mapping from parameters `(β, φ, ρ, δ, π, ln V₀)` to observables lacks injectivity: the likelihood is flat along several directions. This is a structural/practical identifiability issue.

## Implications
- Any posterior method capable of representing multi-modality (e.g. SNPE) will surface these alternative modes; particle filters with aggressive shrinkage may hide them.
- Narrow priors do not eliminate the degeneracy—they simply select one mode arbitrarily. Widening priors or regularisation ranges makes the ambiguity more visible.
- Reporting single-point estimates without acknowledging the multiple modes risks overconfidence and misleading biological conclusions.



