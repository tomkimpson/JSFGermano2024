# TEIVR Compartment Model Mathematics

The TEIVR model describes within-host viral dynamics with immune-mediated refractory cells. The simulator in `src/tiv.py` evolves the five compartments via a hybrid jump–ODE scheme, but the underlying mathematics is defined by a continuous-time Markov jump process with the reactions listed below.

## Compartments
- `T` — target epithelial cells that are susceptible to infection.
- `E` — eclipse-phase cells that are infected but not yet producing virus.
- `I` — productively infected cells.
- `V` — free virions in the sampled compartment (RNA copies per mL).
- `R` — refractory (antiviral) cells temporarily protected by interferon signalling.

The initial state is controlled by the prior entries `T0`, `E0`, `I0`, `R0`, and the log-viral load parameter `lnV0`, which sets `V(0) = exp(lnV0)`.

## Reaction Network and Interactions
Each reaction is parameterised by a rate constant and acts on the compartment vector `(T, R, E, I, V)` using the stoichiometry encoded in `_nu_reactants` and `_nu_products`.

| Rate constant | Reaction | Effect on compartments |
| --- | --- | --- |
| `β` | `T + V → E + V` | Virus infects a target cell, moving it into the eclipse compartment. |
| `ϕ` | `T + I → R + I` | Interferon from infected cells renders nearby targets refractory. |
| `ρ` | `R → T` | Refractory cells lose protection and rejoin the target pool. |
| `k` | `E → I` | Eclipse-phase cells become productively infected. |
| `δ` | `I → ∅` | Infected cells die or are cleared. |
| `π` | `I → I + V` | Infected cells release new virions. |
| `c` | `V → ∅` | Free virus is cleared or degraded. |

These transitions imply the mean-field differential equations (also implemented for the ODE fallback):

```math
\begin{aligned}
\frac{dT}{dt} &= -\beta\,T V - \phi\,I T + \rho\,R, \\
\frac{dE}{dt} &= \beta\,T V - k\,E, \\
\frac{dI}{dt} &= k\,E - \delta\,I, \\
\frac{dV}{dt} &= \pi\,I - c\,V, \\
\frac{dR}{dt} &= \phi\,I T - \rho\,R.
\end{aligned}
```

The hybrid Jump–Switch–Flow integrator transitions between discrete events and deterministic flows when any compartment exceeds the switching threshold `Ω = 100`.

## Free Parameters
The parameter vector `θ = (β, ϕ, ρ, k, δ, π, c)` defines the reaction rates, and the inference pipeline additionally samples `lnV0`. During inference the `context` in `config/cli-refractory-tiv-jsf.toml` places priors on:

- `lnV0` — log initial viral load.
- `β` — infection rate per target cell and virion.
- `ϕ` — interferon-mediated conversion of targets to refractory cells.
- `ρ` — rate at which refractory cells revert to targets.
- `k` — eclipse-to-infected transition rate.
- `δ` — infected-cell clearance rate.
- `π` — virion production rate.
- `c` — viral clearance rate (held constant in some runs but part of the model).

Other quantities such as `T0`, `E0`, `I0`, and `R0` are fixed by constants in the prior but are part of the state vector.

## Observation Model
The observation classes in `src/tiv.py` implement log-scale Gaussian measurements with a detection limit:

- Observations are specified per compartment in the TOML configuration (e.g. `observations.V`, `observations.T`, …).
- For each particle, the simulator records `y = log10(state)` when the compartment is positive.
- Values below the detection limit `log10(y) = -0.65` are truncated to that limit and assigned an almost-zero variance so the likelihood concentrates at the censoring threshold.
- Otherwise, observations follow `Normal(log10(state), σ)` with the compartment-specific `σ` (the `scale` entry in the TOML).

This observation model is used both for viral-load data (`V`) and synthetic observations of cell populations when supplied in the data files.

