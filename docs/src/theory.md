# Theory

This page gives a concise overview of the ideas behind optimal experimental design. For a thorough treatment, see the references at the end.

## What is experimental design?

An experiment measures a system at chosen conditions — a time point, a dose, a frequency, a temperature. The **design** is the set of conditions and how many times each is measured. A good design extracts as much information as possible from a limited measurement budget.

Optimal experimental design formalises "as much information as possible" using the **Fisher Information Matrix** and a scalar criterion.

## The Fisher Information Matrix

For a model ``y = f(\theta, x) + \varepsilon`` with Gaussian noise ``\varepsilon \sim \mathcal{N}(0, \sigma^2(\theta, x))``, the Fisher Information Matrix (FIM) at a single design point ``x`` and parameter value ``\theta`` is:

```math
M(\theta, x) = \frac{1}{\sigma^2(\theta, x)} J(\theta, x)^\top J(\theta, x)
```

where ``J = \partial f / \partial \theta`` is the Jacobian. For a design with ``n`` measurements at points ``x_1, \ldots, x_n``, the total FIM is the sum:

```math
M(\theta, d) = \sum_{i=1}^{n} M(\theta, x_i)
```

The FIM captures how sensitively the model responds to each parameter at each design point. A larger FIM (in some matrix sense) means more precise parameter estimates.

## Design criteria

A **criterion** maps the ``p \times p`` FIM to a scalar measuring design quality:

| Criterion | Formula | Interpretation |
|-----------|---------|---------------|
| **D-optimal** | ``\log \det M`` | Minimise the volume of the confidence ellipsoid |
| **A-optimal** | ``-\operatorname{tr}(M^{-1})`` | Minimise the average parameter variance |
| **E-optimal** | ``\lambda_{\min}(M)`` | Minimise the worst-case parameter variance |

### Ds-optimality

Often you care about a subset of parameters — say, a rate constant — while others (amplitudes, baselines) are nuisance parameters. **Ds-optimality** targets the subset by applying a transformation ``g(\theta)`` that extracts the parameters of interest, then optimising the D-criterion on the transformed FIM via the delta method.

In OptimalDesign.jl, this is specified with `select`:

```julia
transformation = select(:k)            # interested in k only
transformation = select(:k₁, :k₂)     # interested in two rates
```

## Bayesian (pseudo-Bayesian) design

An experimental design ``\xi`` is a collection of measurement points and weights ``(x_i, w_i)``.

Classical optimal design assumes the parameters are known (or guessed). But the FIM — and therefore the optimal design — depends on the true parameter values, which are unknown before the experiment.

**Bayesian optimal design** (also called pseudo-Bayesian or robust design) averages the criterion over prior parameter uncertainty:

```math
\Phi(\xi) = \int \phi\bigl(M(\theta, \xi)\bigr) \, \pi(\theta) \, \mathrm{\xi}\theta
```

where ``\pi(\theta)`` is the prior density and ``\phi`` is the criterion (e.g., ``\log\det``). This produces designs that are good on average across plausible parameter values, rather than optimal only at a single guess.

OptimalDesign.jl approximates this integral by a weighted sum over particles drawn from the prior (or from the current posterior, in the adaptive case):

```math
\Phi(\xi) \approx \sum_{i=1}^{N} w_i \, \phi\bigl(M(\theta_i, \xi)\bigr)
```

## The exchange algorithm

Finding the optimal design is an optimisation problem over the space of probability measures on the candidate set. The **exchange algorithm** (Fedorov 1972, Wynn 1972) solves this iteratively:

1. Start with uniform weights over all candidates
2. At each step, transfer weight from the worst candidate to the best, using the Gateaux derivative as a guide
3. Repeat until convergence

For batch designs with ``n`` measurements, the continuous weights are discretised into integer counts using an apportionment method.

## The General Equivalence Theorem

The **General Equivalence Theorem** (Kiefer & Wolfowitz 1960) provides a condition for optimality. For a D-optimal design ``\xi`` with transformed dimension ``q``, the Gateaux (directional) derivative ``\phi'(\xi, x)`` satisfies:

- ``\phi'(\xi, x) \leq q`` for all candidates ``x``
- ``\phi'(\xi, x) = q`` at the support points of the design

If the maximum Gateaux derivative exceeds ``q``, the design is not optimal and the maximiser indicates which candidate should receive more weight.

OptimalDesign.jl computes and plots the Gateaux derivative via `verify_optimality` and `plot_gateaux`.

## Expected Information Gain (EIG)

FIM-based criteria assume the posterior is approximately Gaussian near the current parameter estimate. This is a good approximation when the prior is tight or the model is nearly linear, but it breaks down for:

- **broad priors** — the FIM is evaluated at particles spread across a wide region, but the average FIM can overestimate the true information gain when the model is nonlinear
- **multi-modal models** — a periodic model has many parameter values that explain any single observation equally well (aliasing); the FIM cannot see this

**Expected Information Gain (EIG)** scores a candidate ``x`` by the mutual information between the parameters ``\theta`` and the next observation ``y``:

```math
\text{EIG}(x) = \mathbb{E}_{\theta \sim \pi} \left[
    \log \frac{p(y \mid \theta, x)}{\int p(y \mid \theta', x) \pi(\theta') \, d\theta'}
\right]
```

This is estimated by **nested Monte Carlo** using ``M`` outer samples (over ``\theta``) and ``N`` inner samples (over the denominator):

```math
\widehat{\text{EIG}}(x) = \frac{1}{M} \sum_{m=1}^{M} \left[
    \log p(y_m \mid \theta_m, x) - \log \frac{1}{N} \sum_{n=1}^{N} p(y_m \mid \theta_n, x)
\right]
```

where ``y_m \sim p(\cdot \mid \theta_m, x)`` is a synthetic observation from the outer sample.

For a `DeltaMethod` transformation built with `select(...)`, a **marginal-EIG** estimator is used: inner samples permute only the selected parameters while holding the nuisance parameters fixed at the outer value. This isolates the information about the parameters of interest.

### EIG vs FIM-based criteria

| Property | FIM (D/A/E) | EIG |
|----------|:-----------:|:---:|
| Exact for Gaussian linear models | ✓ | ✓ |
| Handles broad priors | ~ | ✓ |
| Handles multi-modal posteriors | ✗ | ✓ |
| Supports batch weight optimisation (exchange algorithm) | ✓ | ✗ |
| Computation cost | low | ``O(M \times N \times K)`` |
| Gateaux / GET certificate | ✓ | ✗ |

Because EIG is non-convex in design weights, the exchange algorithm is not available. Batch EIG designs use **greedy selection**: candidates are added one at a time, each time choosing the point that maximises the EIG score given what has already been selected.

In OptimalDesign.jl, EIG is requested with:

```julia
criterion = EIGCriterion(outer_samples = 50, inner_samples = 50)
```

See the [EIG vs D-optimality](@ref) example for a worked comparison.

## Efficiency

The **efficiency** of a design ``ξ_1`` relative to ``ξ_2`` is the ratio of their criterion values, raised to the power ``1/q``:

```math
\text{eff}(ξ_1, ξ_2) = \left( \frac{\det M(ξ_1)}{\det M(ξ_2)} \right)^{1/q}
```

An efficiency of 0.5 means ``ξ_1`` would need roughly twice as many measurements to match the precision of ``ξ_2``.

## Adaptive design

In an **adaptive** (sequential) experiment, the design is updated after each observation:

1. **Design**: choose the next measurement(s) using the current posterior
2. **Acquire**: measure at the chosen design point
3. **Update**: incorporate the observation into the posterior via Bayes' rule

This creates a feedback loop where the experiment learns from its own data. Adaptive designs can outperform batch designs when the optimal measurement locations depend strongly on the (initially unknown) parameter values.

OptimalDesign.jl implements adaptive design using a particle filter posterior. After each observation, particle weights are updated by the likelihood and, when necessary, particles are resampled and jittered to maintain diversity (see [Posterior Inference](@ref)).
