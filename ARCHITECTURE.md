# OptimalDesign.jl — Architecture Specification

## Overview

OptimalDesign.jl is a general-purpose Julia package for Bayesian optimal experimental design with nonlinear models. It supports batch (static) and adaptive (sequential) design through a unified interface: scoring and selecting measurements from a finite set of candidates. It includes a live interactive dashboard (Makie) for monitoring and controlling adaptive experiments in real time.

The package is domain-agnostic. It provides the mathematical machinery for optimal design; users supply forward models and acquisition functions appropriate to their domain.


## Design Philosophy

1. **The problem defines the physics; the solver defines the search.** A `DesignProblem` encodes the forward model, parameters, noise, costs, and constraints. The candidate list and solving strategy are arguments to the solver.

2. **The design space is always a finite list of candidates.** Continuous design variables are discretised upstream by the user. This makes the interface uniform across all problem types — single-variable, multi-dimensional, multi-experiment, multi-sample — without requiring separate machinery for continuous vs discrete vs mixed optimisation. Batch design optimises weights over this list (convex for FIM-based criteria). Adaptive design scores candidates and selects the best.

3. **Composition over hierarchy.** Different experiments, samples, and conditions are fields on candidate NamedTuples, not separate type hierarchies. The forward model is a closure that dispatches internally on whatever fields it needs.

4. **Interactive by default.** The package includes a live Makie dashboard for adaptive experiments, with real-time visualisation and pause/stop controls. A headless mode is available for testing and batch computation.


## Core Types

### DesignProblem

```julia
struct DesignProblem{F,J,S,T}
    predict::F             # (θ, ξ) -> ŷ (scalar or vector)
    jacobian::J            # (θ, ξ) -> J matrix, or nothing => ForwardDiff
    sigma::S               # (θ, ξ) -> σ (scalar, vector, or matrix)
    parameters::NamedTuple # :name => prior (Distributions.jl)
    transformation::T      # θ -> τ(θ), defaults to identity
    cost::Function         # (ξ_prev, ξ) -> time cost
    constraint::Function   # (ξ, θ_est) -> Bool
end
```

Three callables define the physics:

**predict**: Maps parameters θ (ComponentArray) and design point ξ (NamedTuple) to predicted observations (scalar or vector). A closure that captures the physical model at construction time. Returns the signal only — no noise information.

**jacobian**: Maps (θ, ξ) to the Jacobian matrix ∂ŷ/∂θ. Defaults to `nothing`, meaning ForwardDiff is used automatically. An analytic Jacobian can be substantially faster — for example, exploiting matrix exponential structure in Bloch–McConnell models, or reusing eigendecompositions. Keeping the Jacobian separate from predict avoids AD entanglement with the noise.

**sigma**: Maps (θ, ξ) to the observation noise — a scalar σ (constant or parameter/design-dependent), a vector of per-element σ values (for vector observations), or a full covariance matrix. Defaults to `Returns(1.0)` (unit noise). This function is never differentiated — it is evaluated alongside the Jacobian to construct the weighted FIM, but ForwardDiff only touches predict (or is bypassed entirely when an analytic Jacobian is supplied).

The remaining fields:

**parameters**: The full parameter space with prior distributions (Distributions.jl). All parameters that `predict` depends on must appear here, including nuisance/instrumental parameters.

**transformation**: Maps the full parameter vector to quantities of interest. D-optimality on the transformed information matrix (via the Delta method) gives Ds-optimality as a special case. Defaults to identity.

**cost**: Returns the time cost of measurement ξ given that the previous measurement was ξ_prev (receives `nothing` for the first measurement). Encodes both measurement-dependent costs and switching costs.

**constraint**: Returns `false` for design points where the forward model is untrustworthy given current parameter estimates. Re-evaluated as estimates refine.


### Construction

Keyword arguments with sensible defaults:

```julia
# Simplest: constant unit noise, ForwardDiff Jacobian, identity transformation
prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    parameters = (A=Normal(1, 0.1), R₂=LogUniform(1, 50)),
)

# With analytic Jacobian and known constant noise
prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    jacobian = (θ, ξ) -> [exp(-θ.R₂ * ξ.t)  -θ.A * ξ.t * exp(-θ.R₂ * ξ.t)],
    sigma = (θ, ξ) -> 0.05,
    parameters = (A=Normal(1, 0.1), R₂=LogUniform(1, 50)),
    transformation = DeltaMethod(select(:R₂)),
    cost = (prev, ξ) -> ξ.t + 0.1,
)

# Parameter-dependent noise (e.g. fitted rate with θ-dependent uncertainty)
prob = DesignProblem(
    (θ, ξ) -> forward_model(θ, ξ.Ω, ξ.ω₁),
    sigma = (θ, ξ) -> fitting_uncertainty(θ, ξ),
    parameters = exchange_priors,
    transformation = DeltaMethod(select(:k_ex, :p_B, :Δω)),
)

# Vector-valued observations with per-element noise
prob = DesignProblem(
    (θ, ξ) -> [model(θ, ξ.x, T) for T in time_points],
    sigma = (θ, ξ) -> fill(0.01, length(time_points)),
    parameters = priors,
)
```


### Transformation

```julia
abstract type Transformation end

struct Identity <: Transformation end

struct DeltaMethod{F} <: Transformation
    f::F   # θ -> τ(θ), differentiable
end
```

The transformed information matrix:

$$M_\tau(\xi, \theta) = \left[ (\nabla\tau)\, M^{-1}(\xi, \theta)\, (\nabla\tau)' \right]^{-1}$$

where ∇τ is computed by ForwardDiff.

Convenience constructor for selecting named parameters:

```julia
transformation = DeltaMethod(select(:k_ex, :p_B))
```

This replaces the traditional interest/nuisance parameter partition entirely.


### Design Criteria

```julia
abstract type DesignCriterion end

struct DCriterion <: DesignCriterion end    # log det M
struct ACriterion <: DesignCriterion end    # -tr(M⁻¹)
struct ECriterion <: DesignCriterion end    # λ_min(M)
```

The criterion receives a matrix and returns a scalar. It knows nothing about experiments or parameters.


### Candidates

A candidate is a NamedTuple describing one possible measurement. Candidates are collected in a `Vector` and passed to the solver. Different candidates may have different fields.

The package does not own candidate generation. Users construct candidate lists directly. The package provides optional convenience functions for common patterns (Cartesian products, Latin hypercube sampling) without making them a core abstraction.

```julia
# Simple grid
candidates = [(τ=τ,) for τ in range(0.01, 5.0, length=200)]

# Multi-experiment: concatenate lists with different fields
candidates = vcat(
    [(experiment=:A, x=x, t=t) for x in grid_x for t in grid_t],
    [(experiment=:B, x=x) for x in grid_x],
)

# Physics-aware spacing (e.g. frequency resolution scales with field strength)
candidates = [
    (Ω=Ω, ω₁=ω₁)
    for ω₁ in [100, 200, 400]
    for Ω in range(-3000, 3000, step=ω₁/2)
]

# Stochastic
candidates = [random_candidate() for _ in 1:500]
```

Candidates can be regenerated between adaptive steps — refined near promising regions, filtered by updated constraints, or resampled.


### Posterior

```julia
abstract type Posterior end

sample(posterior, n)              # -> vector of ComponentArrays
update!(posterior, prob, ξ, y)    # incorporate observation
posterior_mean(posterior)          # -> ComponentArray
```

**ParticlePosterior**: Vector of ComponentArray particles with a log-weight vector. Update by reweighting according to the observation likelihood. Systematic resampling when effective sample size drops below threshold, with optional kernel jittering to prevent particle impoverishment.

```julia
struct ParticlePosterior{T}
    particles::Vector{T}
    log_weights::Vector{Float64}
end
```

**LaplacePosterior**: Gaussian approximation at the MAP. Maintained by re-optimising after each observation. Fast, suitable for unimodal posteriors.

For multi-sample/multi-compound problems with independent parameter blocks, the particle posterior works without special decomposition — reweighting only depends on the active block's likelihood contribution, so independence is exploited automatically.


### Parameter Representation

ComponentArray (from ComponentArrays.jl) is used internally for parameter vectors. It supports named access (`θ.k_ex`), mutation, ForwardDiff compatibility, and hierarchical structure (`θ.samples[3].R₂`). Users specify parameters as NamedTuples of priors; the package converts to ComponentArray internally.


## Computation

### Fisher Information Matrix

The FIM at a single (θ, ξ) pair:

```julia
function information(prob, θ, ξ)
    J = if prob.jacobian === nothing
        ForwardDiff.jacobian(θ_ -> prob.predict(θ_, ξ), θ)
    else
        prob.jacobian(θ, ξ)
    end
    σ = prob.sigma(θ, ξ)
    weighted_fim(J, σ)
end
```

The `weighted_fim` function constructs Σ⁻¹ from σ and computes J'Σ⁻¹J:
- Scalar σ: F = J'J / σ²
- Vector σ: F = J' diag(1./σ.²) J
- Matrix Σ: F = J' Σ⁻¹ J

The separation of predict, jacobian, and sigma ensures that ForwardDiff only ever differentiates the signal model (predict), never the noise model (sigma). When an analytic Jacobian is supplied, ForwardDiff is not invoked at all, giving maximum performance.

When discrete design variables create block-sparse Jacobian structure (e.g. only one compound measured at a time in a cocktail screen), efficiency is recovered by differentiating only with respect to the active parameter block. This is a performance optimisation — ForwardDiff gives correct dense results regardless.


### Expected Utility

$$U(\xi) = \mathbb{E}_{\theta \sim \pi}\left[\Phi\!\left(M_\tau(\xi, \theta)\right)\right]$$

Evaluated by Monte Carlo over posterior samples, with mini-batch support:

```julia
function expected_utility(prob, criterion, particles, ξ; posterior_samples=50)
    idx = rand(1:length(particles), posterior_samples)
    mean(criterion(transform(prob, information(prob, particles[i], ξ))) for i in idx)
end
```

Mini-batch evaluation is stochastic but unbiased. It reduces cost per candidate by a factor of `length(particles) / posterior_samples` at the expense of noise in the ranking.


### Expected Information Gain (EIG)

For problems where the FIM-based Gaussian approximation is inadequate:

$$\mathrm{EIG}(\xi) = \mathbb{E}_\theta \mathbb{E}_{y|\theta,\xi}\left[\log p(y|\theta,\xi) - \log \int p(y|\theta',\xi)\,p(\theta')\,d\theta'\right]$$

Computed via nested Monte Carlo or accelerated by Laplace importance sampling (which reuses the Jacobian/Hessian machinery and bridges between FIM and fully Bayesian design). When the Laplace approximation is exact (linear model, Gaussian posterior), it recovers the FIM criterion identically.

EIG is appropriate for adaptive sequential design. For batch design, FIM-based weight optimisation is preferred — it is convex, whereas EIG is not (observations interact through the posterior).


### Observation Diagnostics

Each observation is scored against the current posterior to detect model deviations:

```julia
function observation_diagnostics(post, prob, ξ, y)
    log_ml = logsumexp(
        post.log_weights[i] + loglikelihood(prob, post.particles[i], ξ, y)
        for i in eachindex(post.particles)
    )
    w = exp.(post.log_weights)
    mean_residual = sum(
        w[i] * (y - prob.predict(post.particles[i], ξ))
        for i in eachindex(post.particles)
    )
    (mean_residual=mean_residual, log_marginal=log_ml)
end
```

The `loglikelihood` function uses `prob.sigma(θ, ξ)` to construct the noise model for the likelihood calculation — the same sigma used for design planning. When observations carry their own uncertainty (structured observations from an intermediate fitting step, e.g. `(value=R, σ=σ_R)`), the likelihood uses the realised σ from the observation instead.

A running series of `log_marginal` values constitutes sequential Bayesian model checking. Sharp drops indicate observations surprising under the current model, flagging systematic model failure.


## Solver

### Unified Interface

Both batch and adaptive design reduce to scoring candidates and selecting the best n:

```julia
function select(
    problem::DesignProblem,
    candidates::Vector{<:NamedTuple},
    posterior;
    n = 1,
    criterion = DCriterion(),
    budget = Inf,
    posterior_samples = 50,
    ξ_prev = nothing,
)
    # Returns: Vector{Tuple{NamedTuple, Int}} — (ξ, count) pairs
end
```

**Batch design** (n = N, prior as posterior, no intermediate updates): Optimises weights over the candidate list. Convex for D-optimality. Solved by the exchange algorithm (iteratively shift weight towards the candidate with the highest Gateaux derivative). Returns an allocation of N measurements across candidates. The Gateaux derivative provides an optimality certificate.

**Adaptive sequential design** (n = 1 or small batch, posterior updates between rounds): Scores candidates by expected utility gain per unit cost, selects the top n. For n > 1, greedy sequential selection (pick one, temporarily update the running FIM, pick the next) approximates joint selection without combinatorial explosion.

**Efficiency calculations**: Relative efficiency between two designs quantifies how many more measurements one needs to match the other.

**Apportionment**: Converts continuous weights to integer counts summing to N.


### Stochastic Evaluation

Mini-batch evaluation (random subset of posterior particles) provides unbiased utility estimates at reduced cost. Two strategies:

- **Coarse-then-refine**: Score all candidates with small batch, re-score top candidates with full particle set.
- **Stochastic gradient optimisation (SGO)**: Gradient ascent on expected utility treating ξ as continuous, using mini-batch gradients via ForwardDiff. Complements discrete candidate evaluation for high-dimensional or continuous refinement.


## Acquisition

The acquisition function is a user-provided callable, curried over any external state:

```julia
# Simulated (for development and validation)
acquire = let θ_true = true_params, pred = predict, σ = 0.05
    ξ -> pred(θ_true, ξ) + σ * randn()
end

# Manual entry
acquire = ξ -> begin
    println("Measure at: ", ξ)
    parse(Float64, readline())
end

# Domain-specific instrument (user code, not in package)
acquire = let conn = instrument_connection
    ξ -> run_measurement(conn, ξ)
end
```

The acquire function is called by `run_experiment` on a background task (`Threads.@spawn`) so it does not block the Makie UI.

Observations may be structured — e.g. `(value=R, σ=σ_R)` — when the measurement involves an intermediate fitting step. The sigma function in the DesignProblem predicts noise for design planning; the structured observation carries the realised noise for posterior updating.


## Interactive Dashboard

### Dependencies

Makie is a core dependency. Both GLMakie (interactive windows for live monitoring) and CairoMakie (publication-quality static output) are supported via backend switching. A headless mode (no GUI) is available for testing and batch computation.

### run_experiment

The primary user-facing function for adaptive experiments:

```julia
function run_experiment(
    prob::DesignProblem,
    candidates,
    posterior,
    acquire;                    # callable: ξ -> y
    budget,
    criterion = DCriterion(),
    posterior_samples = 50,
    n_per_step = 1,
    headless = false,           # suppress GUI for testing
    prediction_grid = nothing,  # dense ξ grid for credible band plots
)
    # Returns: (posterior=posterior, log=ExperimentLog)
end
```

The function:
1. Opens a Makie dashboard (unless headless).
2. Runs the adaptive loop: select → acquire (on background task) → update → display.
3. Responds to pause/stop controls from the dashboard.
4. Returns the final posterior and full experiment log.

### Live Dashboard Panels

- **Design points**: Where measurements have been taken (scatter on design space axes).
- **Posterior marginals**: Histograms or KDEs for parameters of interest, updating live.
- **Information gain per step**: Running plot of utility gained at each measurement.
- **Budget remaining**: Time/measurement budget tracker.
- **Controls**: Pause, resume, stop buttons. Possibility for manual override to force measurement at a user-selected candidate.

If a `prediction_grid` is provided, the dashboard also shows posterior credible bands on the prediction, overlaid with actual observations.

### Static Plotting

For post-hoc analysis and publication figures (CairoMakie backend):

- **Credible bands**: Dense predictions from posterior samples with median and quantile envelope.
- **Design trajectory**: Sequential snapshots showing how uncertainty narrowed as measurements accumulated.
- **Residual diagnostics**: Residual series and cumulative log evidence from the ExperimentLog.
- **Design allocation**: For batch designs, weight distribution across candidates.

Building blocks:

```julia
posterior_predictions(prob, posterior, ξ_grid; n_samples=200)
credible_band(predictions; level=0.9)  # -> (lower, median, upper)
```


## Experiment Log

```julia
struct ExperimentLog
    history::Vector{NamedTuple}  # (ξ, y, cost, diagnostics) per step
end
```

Records the full experiment history. Supports post-hoc analysis, trajectory replay, and reproducibility.


## Design Patterns

### Currying the Model

The physical model is absorbed into the predict closure. The package never sees it.

```julia
solver = SomePhysicsSolver(config...)
predict = (θ, ξ) -> solver(θ, ξ.x, ξ.t)
prob = DesignProblem(predict, parameters=...)
```

### Analytic Jacobians for Performance

When the forward model has exploitable structure, an analytic Jacobian avoids ForwardDiff overhead. The Jacobian is a separate callable, keeping predict clean:

```julia
predict = (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t)
jac = (θ, ξ) -> begin
    e = exp(-θ.R₂ * ξ.t)
    [e  -θ.A * ξ.t * e]   # 1×2 for scalar observation, 2 parameters
end

prob = DesignProblem(predict, jacobian=jac, parameters=...)
```

ForwardDiff can be used to validate analytic Jacobians during testing.

### Multi-Experiment Design

Experiment identity is a field on candidates. The predict function branches on it.

```julia
predict = (θ, ξ) -> if ξ.experiment == :A
    model_a(θ, ξ.x, ξ.t)
else
    model_b(θ, ξ.x)
end
candidates = vcat(candidates_A, candidates_B)
```

Cost and constraint functions dispatch on the same field.

### Multi-Sample / Multi-Compound Design

Sample/compound index is a field on candidates. Switching costs are handled by the cost function.

```julia
predict = (θ, ξ) -> measure(θ.samples[ξ.sample], ξ.x)
cost = (prev, ξ) -> 1.5 + (prev !== nothing && prev.sample != ξ.sample ? 300.0 : 0.0)
```

### Heteroscedastic / Parameter-Dependent Noise

When observation uncertainty depends on what you're measuring, sigma encodes this:

```julia
# Noise depends on design point and parameters
sigma = (θ, ξ) -> fitting_uncertainty(θ, ξ)

# For posterior updating, acquire returns structured observation
acquire = ξ -> begin
    data = measure(ξ)
    val, σ = fit_and_estimate_uncertainty(data)
    (value=val, σ=σ)
end
```

The sigma function predicts noise for design planning. The structured observation carries the realised noise for posterior updating. They need not agree exactly — the design is robust to moderate misspecification because the expected utility integrates over the prior.

### Imperfect Forward Models

In order of preference:

1. **Use the correct model.** Include the physical effect in predict. The FIM naturally avoids regions where the effect dominates.
2. **Inflate the noise.** Use sigma to increase σ where model error is expected.
3. **Constrain the design space.** Exclude problematic regions via the constraint function.


## Worked Examples (Doctests)

The examples form a progression that exercises increasingly complex features of the framework. Each serves as a doctest and a validation target.


### Example 1: Exponential Decay — Scalar Observation, Batch Design

Simplest case. One design variable (time t), two parameters (A, R₂), interest in R₂ via transformation.

**Demonstrates**: Basic DesignProblem setup, transformation for Ds-optimality, batch design via `select`, efficiency comparison against uniform spacing, Gateaux derivative check.

```julia
prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    parameters = (A=Normal(1, 0.1), R₂=LogUniform(1, 50)),
    transformation = DeltaMethod(select(:R₂)),
    sigma = (θ, ξ) -> 0.05,
    cost = (prev, ξ) -> ξ.t + 0.1,
)
candidates = [(t=t,) for t in range(0.001, 0.5, length=200)]
prior = ParticlePosterior(prob, 1000)

# Batch design
design = select(prob, candidates, prior; n=20, criterion=DCriterion())

# Efficiency vs uniform
uniform = uniform_allocation(candidates, 20)
eff = efficiency(uniform, design, prob, prior)
```


### Example 2: Inversion Recovery — Analytic Jacobian, Transformation

Three parameters (R₁, A, B), Ds-optimality via transformation onto R₁. Analytic Jacobian supplied for performance.

**Demonstrates**: Analytic Jacobian alongside ForwardDiff validation, Ds-optimality via transformation, validation against known result (optimal delays concentrate at specific τ/T₁ ratios).

```julia
prob = DesignProblem(
    (θ, ξ) -> θ.A - θ.B * exp(-θ.R₁ * ξ.τ),
    jacobian = (θ, ξ) -> begin
        e = exp(-θ.R₁ * ξ.τ)
        [1.0  -e  θ.B * ξ.τ * e]
    end,
    parameters = (A=Normal(1, 0.1), B=Normal(2, 0.1), R₁=LogUniform(0.1, 5)),
    transformation = DeltaMethod(select(:R₁)),
    cost = (prev, ξ) -> 1.0 + ξ.τ,
)
candidates = [(τ=τ,) for τ in range(0.01, 5.0, length=200)]

# Test: analytic Jacobian matches ForwardDiff
θ_test = draw(prob.parameters)
ξ_test = candidates[50]
J_analytic = prob.jacobian(θ_test, ξ_test)
J_ad = ForwardDiff.jacobian(θ_ -> prob.predict(θ_, ξ_test), θ_test)
@test J_analytic ≈ J_ad
```


### Example 3: Two Decays, Vector Observation — Simultaneous Measurement

Two exponential decays observed simultaneously as a vector (y₁, y₂) at a single time t. Four parameters (A₁, R₂₁, A₂, R₂₂), interest in both rates. The Jacobian is a 2×4 matrix. The FIM sums information from both observables.

**Demonstrates**: Vector-valued predict, vector sigma, FIM from multiple simultaneous observables, batch design where a single time point informs both rates.

```julia
prob = DesignProblem(
    (θ, ξ) -> [θ.A₁ * exp(-θ.R₂₁ * ξ.t),
               θ.A₂ * exp(-θ.R₂₂ * ξ.t)],
    parameters = (A₁=Normal(1, 0.1), R₂₁=LogUniform(1, 50),
                  A₂=Normal(1, 0.1), R₂₂=LogUniform(1, 50)),
    transformation = DeltaMethod(select(:R₂₁, :R₂₂)),
    sigma = (θ, ξ) -> [0.05, 0.05],     # per-element noise
    cost = (prev, ξ) -> ξ.t + 0.1,
)
candidates = [(t=t,) for t in range(0.001, 0.5, length=200)]

# The optimal design balances information for both rates.
# If R₂₁ ≫ R₂₂, optimal times will span a wide range to
# capture both fast and slow decays.
```


### Example 4: Two Decays, Discrete Control Variable — Selective Measurement

Same two decays, but now a control variable `i ∈ {1, 2}` selects which one is observed. Each measurement returns a scalar. The parameter space is the same, but only one decay contributes to the FIM per measurement.

**Demonstrates**: Discrete control variable as a candidate field, block-sparse Jacobian (only 2 of 4 parameters have nonzero derivatives per measurement), adaptive design choosing which decay to measure next, block independence in the posterior.

```julia
prob = DesignProblem(
    (θ, ξ) -> if ξ.i == 1
        θ.A₁ * exp(-θ.R₂₁ * ξ.t)
    else
        θ.A₂ * exp(-θ.R₂₂ * ξ.t)
    end,
    parameters = (A₁=Normal(1, 0.1), R₂₁=LogUniform(1, 50),
                  A₂=Normal(1, 0.1), R₂₂=LogUniform(1, 50)),
    transformation = DeltaMethod(select(:R₂₁, :R₂₂)),
    sigma = (θ, ξ) -> 0.05,
    cost = (prev, ξ) -> ξ.t + 0.1,
)
candidates = [
    (i=i, t=t)
    for i in [1, 2]
    for t in range(0.001, 0.5, length=200)
]
```

This is a key test case because it exposes the block structure. The Jacobian for `i=1` has the form `[∂y/∂A₁  ∂y/∂R₂₁  0  0]` and for `i=2` has `[0  0  ∂y/∂A₂  ∂y/∂R₂₂]`. The FIM from a single measurement is rank-deficient (at most rank 1), so the batch design must include measurements of both decays. The adaptive design must decide at each step which decay benefits more from another measurement, given what's already known about each.

The posterior particles have four dimensions but the two blocks `(A₁, R₂₁)` and `(A₂, R₂₂)` are *a posteriori* independent — measuring decay 1 only reweights particles based on parameters (A₁, R₂₁), leaving (A₂, R₂₂) unchanged. This independence is not enforced by the posterior type; it falls out automatically from the likelihood being block-diagonal.

**Comparison with Example 3**: The same physical system measured two different ways. Example 3 (vector observation) gets information about both rates from every measurement. Example 4 (selective observation) must allocate measurements between the two decays. Comparing the batch designs and efficiencies of the two approaches quantifies the value of simultaneous measurement.


### Example 5: Exponential Decay — Adaptive Design with Simulated Acquisition

Uses Example 1's problem setup but runs a full adaptive experiment against a simulated ground truth.

**Demonstrates**: `run_experiment` with simulated acquisition, posterior convergence, live dashboard (headless mode for testing), comparison of adaptive vs batch posterior precision.

```julia
# Ground truth
θ_true = ComponentArray(A=1.0, R₂=25.0)

# Simulated acquisition
acquire = let θ = θ_true, σ = 0.05
    ξ -> θ.A * exp(-θ.R₂ * ξ.t) + σ * randn()
end

# Set up problem and prior
prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    parameters = (A=Normal(1, 0.1), R₂=LogUniform(1, 50)),
    transformation = DeltaMethod(select(:R₂)),
    sigma = (θ, ξ) -> 0.05,
    cost = (prev, ξ) -> ξ.t + 0.1,
)
candidates = [(t=t,) for t in range(0.001, 0.5, length=200)]
prior = ParticlePosterior(prob, 1000)

# Run adaptive experiment
result = run_experiment(
    prob, candidates, prior, acquire;
    budget = 10.0,
    criterion = DCriterion(),
    n_per_step = 1,
    headless = true,           # no GUI for testing
)

# Check posterior has converged near truth
post_mean = posterior_mean(result.posterior)
@test abs(post_mean.R₂ - 25.0) < 5.0

# Compare: batch design with same budget
batch = select(prob, candidates, ParticlePosterior(prob, 1000);
               n=20, criterion=DCriterion())
```


### Example 6: Two Decays, Discrete Control — Adaptive with Switching

Uses Example 4's problem but runs adaptively, demonstrating how the selector balances measurements between the two decays.

**Demonstrates**: Adaptive design with discrete control variable, switching cost, block-sparse posterior updates, the selector preferring to stay on the current decay unless the other is substantially more informative.

```julia
θ_true = ComponentArray(A₁=1.0, R₂₁=10.0, A₂=1.0, R₂₂=40.0)

acquire = let θ = θ_true, σ = 0.05
    ξ -> (ξ.i == 1 ? θ.A₁ * exp(-θ.R₂₁ * ξ.t) : θ.A₂ * exp(-θ.R₂₂ * ξ.t)) + σ * randn()
end

prob = DesignProblem(
    (θ, ξ) -> ξ.i == 1 ? θ.A₁ * exp(-θ.R₂₁ * ξ.t) : θ.A₂ * exp(-θ.R₂₂ * ξ.t),
    parameters = (A₁=Normal(1, 0.1), R₂₁=LogUniform(1, 50),
                  A₂=Normal(1, 0.1), R₂₂=LogUniform(1, 50)),
    transformation = DeltaMethod(select(:R₂₁, :R₂₂)),
    sigma = (θ, ξ) -> 0.05,
    cost = (prev, ξ) -> begin
        t_measure = ξ.t + 0.1
        t_switch = (prev !== nothing && prev.i != ξ.i) ? 1.0 : 0.0
        t_measure + t_switch
    end,
)
candidates = [(i=i, t=t) for i in [1, 2] for t in range(0.001, 0.5, length=200)]
prior = ParticlePosterior(prob, 1000)

result = run_experiment(
    prob, candidates, prior, acquire;
    budget = 20.0,
    n_per_step = 1,
    headless = true,
)

# Because R₂₂ > R₂₁, the fast decay is harder to characterise
# and should receive more short-time measurements.
# The selector should initially explore both, then focus on
# whichever has larger posterior uncertainty relative to cost.
```


### Example 7: Dose-Response (Sigmoid Emax) — Domain-Agnostic Validation

Four parameters, full D-optimality. Validates against Kirstine.jl published results for the same model.

**Demonstrates**: Non-domain-specific use, full D-optimality (no transformation), validation against an independent implementation.

```julia
prob = DesignProblem(
    (θ, ξ) -> θ.E0 + θ.Emax * ξ.dose^θ.h / (θ.ED50^θ.h + ξ.dose^θ.h),
    parameters = (E0=Normal(1, 0.5), Emax=Normal(2, 0.5),
                  ED50=LogNormal(-1, 0.5), h=LogNormal(1, 0.5)),
    cost = (prev, ξ) -> 1.0,
)
candidates = [(dose=d,) for d in range(0, 1, length=50)]
```


### Example Progression Summary

| # | Example | Observation | Design vars | Key feature |
|---|---------|------------|-------------|-------------|
| 1 | Exponential decay | scalar | continuous t | Basic batch design, efficiency |
| 2 | Inversion recovery | scalar | continuous τ | Analytic Jacobian, transformation |
| 3 | Two decays (vector) | vector (y₁,y₂) | continuous t | Simultaneous observables, vector sigma |
| 4 | Two decays (selective) | scalar | discrete i + continuous t | Block-sparse Jacobian, block independence |
| 5 | Decay (adaptive) | scalar | continuous t | Simulated adaptive, run_experiment |
| 6 | Two decays (adaptive) | scalar | discrete i + continuous t | Switching cost, block posterior updates |
| 7 | Dose-response | scalar | continuous dose | Domain-agnostic, Kirstine validation |


## Implementation Guide

### Phase 1: Types and FIM (foundation)

Implement the core type definitions and FIM computation. This is the mathematical foundation everything else builds on.

1. Package scaffolding: `Project.toml`, module structure, CI.
2. `DesignProblem` struct with keyword constructor and defaults (`jacobian=nothing`, `sigma=Returns(1.0)`, `transformation=Identity()`, `cost=(prev,ξ)->1.0`, `constraint=(ξ,θ)->true`).
3. `Transformation`: `Identity`, `DeltaMethod` with `select()` convenience.
4. `DesignCriterion`: `DCriterion`, `ACriterion`, `ECriterion`.
5. ComponentArray integration: constructor that draws from prior NamedTuple.
6. `information(prob, θ, ξ)`: Jacobian dispatch (ForwardDiff or analytic), sigma evaluation, weighted FIM construction (`J'Σ⁻¹J`).
7. Transformed information matrix via Delta method.
8. `expected_utility` with mini-batch support.

**Validate with**: Examples 1 and 2. Check FIM values by hand for simple cases. Verify analytic Jacobian matches ForwardDiff. Confirm transformed FIM gives correct Ds-optimal behaviour.

### Phase 2: Posterior and solver

1. `ParticlePosterior`: particles + log_weights, constructor from priors, `sample`, `posterior_mean`, `update!`, effective sample size, systematic resampling with kernel jitter.
2. `loglikelihood` using sigma from the problem (and handling structured observations with `y.σ`).
3. `select` — unified interface: single-point scoring by utility/cost, greedy multi-point, and weight optimisation for batch.
4. Exchange algorithm for batch weight optimisation.
5. Gateaux derivative computation and optimality verification.
6. `efficiency` and `apportion`.

**Validate with**: Examples 1–4 for batch design. Check Gateaux derivative ≤ 0 at all candidates for optimal designs. Compare efficiency of optimal vs uniform. For Example 4, verify that batch design allocates measurements to both decays.

### Phase 3: Adaptive loop and diagnostics

1. `observation_diagnostics`: log marginal likelihood, mean residual.
2. `ExperimentLog`: history storage, serialisation, accessors.
3. `run_experiment`: the adaptive loop with headless mode.
4. Simulated acquisition helper.

**Validate with**: Examples 5 and 6. Verify posterior convergence to ground truth. Check that diagnostics flag surprising observations when the true model deviates. Verify switching cost behaviour in Example 6.

### Phase 4: Dashboard

1. Static plotting (CairoMakie): credible bands, design allocation, Gateaux derivative, residuals.
2. `posterior_predictions` and `credible_band` building blocks.
3. Live dashboard (GLMakie): Observable-based reactive panels, pause/resume/stop controls.
4. Integration with `run_experiment`.
5. Headless mode: same logic path, no window creation.

**Validate with**: All examples. Visual inspection of plots. Dashboard responsiveness testing.

### Phase 5: Advanced features

1. EIG via nested Monte Carlo.
2. Laplace importance sampling for EIG acceleration.
3. `LaplacePosterior` (MAP + Gaussian approximation via Optim.jl + Hessian).
4. SGO for continuous design point refinement.

**Validate with**: Compare EIG and FIM-based designs on Examples 1 and 5. Verify Laplace posterior agrees with particle posterior for unimodal cases.


## Design Decisions

- **Three callables (predict, jacobian, sigma)** define the physics cleanly. Predict returns signal only. Jacobian defaults to ForwardDiff but can be overridden analytically for performance. Sigma is never differentiated, avoiding AD entanglement. Each does one thing.
- **ComponentArray** for parameter vectors. Named access, mutation, ForwardDiff, hierarchical structure.
- **Makie as core dependency.** GLMakie for live dashboard, CairoMakie for publication, headless for testing.
- **Candidates as plain vectors of NamedTuples.** No design space type.
- **Transformation replaces interest/nuisance partition.**
- **Acquisition is a curried callable**, run on a background task.
- **Posterior decomposition for independent sub-problems** requires no special machinery. Block-diagonal likelihood means reweighting automatically only affects the active parameter block.
- **FIM-based weight optimisation for batch; greedy selection for adaptive.** Both use `select`. Weight optimisation is convex for FIM criteria. EIG is used only in the adaptive case.


## Open Questions

- **Model comparison.** Screening problems may require distinguishing competing models (e.g. effect present vs absent). The utility function for model selection (expected Bayes factor update, EIG on a discrete model indicator) is structurally different from parameter estimation. May require a generalisation of the criterion concept. Deferred for initial implementation.

- **Candidate generation helpers.** Scope of convenience functions for common patterns to be determined during implementation.

- **Non-Gaussian likelihoods.** The current FIM formulation assumes Gaussian observations. For Poisson, binomial, or other observation models, the FIM takes a different form. This can be handled by allowing sigma to return a special type that triggers a different FIM calculation, or by providing an alternative `information` method. To be designed when a concrete use case arises.


## Dependencies

| Package | Role |
|---------|------|
| ComponentArrays.jl | Parameter vector representation |
| Distributions.jl | Prior specification |
| ForwardDiff.jl | Automatic Jacobians and Hessians |
| GLMakie.jl | Interactive live dashboard |
| CairoMakie.jl | Publication-quality static plots |
| LinearAlgebra (stdlib) | FIM operations |
| Random (stdlib) | Sampling, mini-batch selection |
| Statistics (stdlib) | Mean, quantile for credible bands |


## Package Structure

```
OptimalDesign.jl/
├── src/
│   ├── OptimalDesign.jl        # module, exports
│   ├── types.jl                # DesignProblem, Transformation, Criterion
│   ├── information.jl          # FIM, Jacobian dispatch, weighted FIM
│   ├── criteria.jl             # D, A, E criterion implementations
│   ├── eig.jl                  # Expected information gain
│   ├── select.jl               # Unified select interface
│   ├── exchange.jl             # Exchange algorithm for weight optimisation
│   ├── gateaux.jl              # Gateaux derivative, optimality verification
│   ├── efficiency.jl           # Relative efficiency, apportionment
│   ├── diagnostics.jl          # Observation diagnostics, log evidence
│   ├── posteriors/
│   │   ├── particle.jl         # ParticlePosterior
│   │   └── laplace.jl          # LaplacePosterior
│   ├── experiment.jl           # run_experiment, ExperimentLog
│   └── dashboard/
│       ├── live.jl             # GLMakie live dashboard
│       └── static.jl           # CairoMakie publication plots
├── examples/
│   ├── 1_exponential_decay.jl       # Scalar, batch design, efficiency
│   ├── 2_inversion_recovery.jl      # Analytic Jacobian, transformation
│   ├── 3_two_decays_vector.jl       # Vector observation, simultaneous
│   ├── 4_two_decays_selective.jl    # Discrete control, block sparsity
│   ├── 5_adaptive_decay.jl          # Simulated adaptive experiment
│   ├── 6_adaptive_selective.jl      # Adaptive with switching cost
│   └── 7_dose_response.jl           # Domain-agnostic validation
├── test/
└── docs/
```
