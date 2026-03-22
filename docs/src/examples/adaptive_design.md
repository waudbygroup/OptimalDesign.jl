# Adaptive Design

In an adaptive experiment, each measurement is chosen based on what you've learned so far. The algorithm loops: **design → acquire → update posterior → repeat**. This can outperform batch design when the optimal measurement locations depend on the (unknown) parameter values.

## The model

The same exponential decay from the [Batch Design](@ref) example:

```math
y = A \exp(-R_2 \, t) + \varepsilon
```

But now with a cost per measurement — longer times cost more — and a fixed budget.

## Problem setup

```julia
using OptimalDesign
using ComponentArrays, Distributions

prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    parameters = (A = LogUniform(0.1, 10), R₂ = Uniform(1, 50)),
    transformation = select(:R₂),
    sigma = (θ, ξ) -> 0.2,
    cost = ξ -> ξ.t + 1,
)

candidates = [(t = t,) for t in range(0.001, 0.5, length = 200)]
```

The `cost` function means short measurements are cheap (cost ≈ 1) and long measurements are expensive (cost up to 1.5). The budget constraint will force the algorithm to balance information gain against cost.

## Running the adaptive experiment

```julia
θ_true = ComponentArray(A = 2.0, R₂ = 42.0)
acquire = ξ -> θ_true.A * exp(-θ_true.R₂ * ξ.t) + 0.1 * randn()

prior = Particles(prob, 1000)

result = run_adaptive(
    prob, candidates, prior, acquire;
    budget = 100.0,
    n_per_step = 1,
    record_posterior = true,
)
```

`run_adaptive` returns:

- `result.posterior` — the final posterior after all observations
- `result.log` — an `ExperimentLog` containing the full history: design points, observations, costs, and (if `record_posterior = true`) posterior snapshots at each step

### Key options

| Keyword | Default | Meaning |
|---------|---------|---------|
| `budget` | (required) | Total cost budget |
| `n_per_step` | `1` | Measurements per adaptive step (1 = fully sequential) |
| `headless` | `false` | If `false`, opens a live dashboard (requires GLMakie) |
| `record_posterior` | `false` | Save posterior snapshots for animation |

## Inspecting the results

```julia
log = result.log
n = length(log)                           # number of observations
spent = sum(e.cost for e in log)          # total cost spent
μ = mean(result.posterior)       # final parameter estimates
```

## Comparing to a batch design

A fair comparison uses the same number of measurements:

```julia
prior_batch = Particles(prob, 1000)
d = design(prob, candidates, prior_batch; n = n)

posterior_batch = Particles(prob, 1000)
result_batch = run_batch(d, prob, posterior_batch, acquire)
```

The adaptive design has the advantage of updating its strategy as data arrives. Early observations narrow the posterior on ``R_2``, which shifts the optimal measurement time — the adaptive algorithm exploits this. The batch design is frozen at the prior and cannot adapt.

## Design trajectory

The sequence of chosen design points reveals how the algorithm's strategy evolves:

```julia
fig = Figure(size = (700, 500))

ax = Axis(fig[1, 1], xlabel = "Step", ylabel = "Design time t",
    title = "Adaptive Design Trajectory")
scatter!(ax, 1:n, [e.ξ.t for e in log],
    color = 1:n, colormap = :viridis, markersize = 8)
```

Early steps may explore broadly; later steps concentrate on the times most informative for the current best estimate of ``R_2``.

## Posterior convergence and ESS

Tracking the posterior mean and effective sample size over time shows how quickly the posterior converges:

```julia
prior_replay = Particles(prob, 1000)
ess_history = Float64[]
r2_history = Float64[]

for entry in log
    OptimalDesign.update!(prior_replay, prob, entry.ξ, entry.y)
    push!(ess_history, effective_sample_size(prior_replay))
    push!(r2_history, mean(prior_replay).R₂)
end
```

The ESS will dip after informative observations (as likelihood tempering reweights particles) and recover after resampling. If ESS stays consistently low, consider increasing the number of particles.

## Posterior evolution animation

With `record_posterior = true`, you can create a video of the posterior evolving:

```julia
if OptimalDesign.has_posterior_history(log)
    record_corner_animation(log, "posterior_evolution.mp4";
        params = [:A, :R₂],
        truth = (A = 2.0, R₂ = 42.0),
        framerate = 5)
end
```

## Observation diagnostics

```julia
fig = plot_residuals(log)
```

This shows two panels: mean residuals per step (should scatter around zero) and cumulative log marginal likelihood (should increase steadily — a sustained decrease suggests model misspecification).

See [`examples/5_adaptive_decay.jl`](https://github.com/your-org/OptimalDesign.jl/blob/main/examples/5_adaptive_decay.jl) for the full runnable script.
