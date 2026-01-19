# Manuscript Details

## Title
**Real-Time Inference for Multiscale Hybrid Models of Infectious Disease: Overcoming Algorithmic Non-Identifiability**

## Abstract
Multiscale infectious disease models are essential for capturing the interplay between large-scale deterministic population growth and small-scale stochastic events, such as viral extinction. While hybrid simulation frameworks like Jump-Switch-Flow (JSF) efficiently handle these dynamics, calibrating them to noisy clinical data remains a significant computational bottleneck. Traditional likelihood-free inference methods, such as Sequential Monte Carlo (SMC), are computationally intensive "online" algorithms that scale linearly with cohort size. Furthermore, they are prone to particle degeneracy, often failing to resolve late-stage dynamics where stochastic diversity is lost during resampling.

In this work, we overcome these barriers by coupling hybrid simulation with Neural Posterior Estimation (NPE). This approach amortizes the computational cost by training a neural network to approximate the posterior distribution offline, allowing for near-instantaneous inference on new data. Using the Target-cell, Eclipsed-cell, Infectious-cell, Refractory-cell, Virion (TEIRV) model of within-host viral dynamics as a case study, we demonstrate that the JSF-NPE framework reduces inference latency from hours to less than one second per patient. Crucially, we demonstrate that NPE overcomes the limitations of sequential methods regarding algorithmic identifiability. While particle filters struggle to identify parameters governing immune memory and feedback loops due to path dependency, NPE’s global inference capability accurately resolves these identifiable manifolds and complex multimodal landscapes. By resolving the trade-off between model complexity, parameter identifiability, and inference speed, this framework enables the deployment of sophisticated stochastic models in time-critical medical decision-making and population-level analysis.

***

# Core Contributions & Novelty

### 1. The Speed/Latency Breakthrough (Clinical Utility)
* **Real-time Decision Support:** Reduces inference time from ~5 hours to <1 second. This moves the model from a "retrospective analysis tool" to a "real-time clinical support tool" (e.g., for adjusting antiviral dosage in the ICU).
* **Scalability/Throughput:** Solves the linear scaling problem. Calibrating a cohort of 1,000 patients becomes trivial with NPE (seconds), whereas SMC would require thousands of compute hours.

### 2. The "Algorithmic Identifiability" Discovery (Methodological Novelty)
* **Recovering "Lost" Parameters:** The paper demonstrates that parameters governing late-stage dynamics (specifically $\rho$, the refractory reversion rate) are **structurally identifiable** but **algorithmically non-identifiable** via standard SMC.
* **Global vs. Local Inference:** It shows that because NPE learns a global mapping rather than exploring locally (and greedily) like SMC, it avoids "particle degeneracy" (where the diversity needed to estimate late-acting parameters is resampled away early in the trajectory).

### 3. Resolution of Multimodality
* **Topological Accuracy:** The results show that NPE captures complex, non-Gaussian posterior shapes (e.g., the correlation ridge between $\beta$ and $\delta$) and distinct modes (High $\phi$ vs. High $\delta$) that particle filters often miss due to mode collapse.

### 4. Hybrid Compatibility
* **Bridging Regimes:** It successfully demonstrates a pipeline that handles the non-differentiable, discrete stochastic jumps of the JSF framework within a deep learning inference context, proving feasibility for this specific class of "difficult" biological models.

***

# Anticipated Criticisms & Rebuttals

### Criticism 1: "Incremental Innovation"
* **The Critique:** "You just took an existing simulator (JSF, Germano 2024) and applied an off-the-shelf neural network (NPE). This is just an application study."
* **The Rebuttal:** While the components are existing, the *interaction* reveals a new finding about **algorithmic identifiability**. We do not just show it is faster; we show that the standard method (SMC) is mathematically flawed for late-stage parameters in this specific multiscale context, and that global amortized methods are the necessary fix.

### Criticism 2: "The Black Box Problem"
* **The Critique:** "Neural networks are unreliable black boxes. How do we know these posteriors are real and not hallucinations? Standard SMC is safer."
* **The Rebuttal:** We benchmark directly against the SMC gold standard. Where they agree (early kinetics), the accuracy matches. Where they disagree (late kinetics), we provide a mechanistic explanation (degeneracy) for why the NN is actually *more* correct. (TK note: we should ensure our Posterior Predictive Checks in Appendix A are robust to support this).

### Criticism 3: "Training Overhead"
* **The Critique:** "You claim it's faster, but training the network took $10^5$ simulations. For a single patient, SMC might actually be cheaper."
* **The Rebuttal:** Valid for $N=1$, but biological modeling rarely stops at one patient. The "break-even" point is low (approx. 6 patients based on Table 3). For any scenario involving clinical trials, surveillance, or cohorts ($N>10$), the amortized method is strictly superior in terms of total compute cost.

### Criticism 4: "Generalizability"
* **The Critique:** "This works for TEIRV, but will it work for other hybrid models?"
* **The Rebuttal:** The underlying principle—mapping time-series summaries to parameter space—is model-agnostic. The success with TEIRV, which exhibits extreme scale separation (1 virion to $10^9$ virions), suggests the method is robust enough for less extreme biological models.
