# Adaptive Design

In an adaptive experiment, each measurement is chosen based on what you've learned so far. The algorithm loops: **design → acquire → update posterior → repeat**. This can outperform batch design when the optimal measurement locations depend on the (unknown) parameter values.

## The model

The same exponential decay from the [Batch Design](@ref) example, but now with a cost per measurement and a fixed budget.

## Setup

```@example adaptive
using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random; Random.seed!(42) # hide

function model(θ, x)
    θ.A * exp(-θ.R₂ * x.t)
end

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A = 2.0, R₂ = 42.0)
σ = 0.1
acquire(x) = model(θ_true, x) + σ * randn()

# Design problem with per-measurement cost
prob = DesignProblem(
    model,
    parameters = (A = LogUniform(0.1, 10), R₂ = Uniform(1, 50)),
    transformation = select(:R₂),
    sigma = Returns(0.2),
    cost = x -> x.t + 1,
)

candidates = candidate_grid(t = range(0.001, 0.5, length = 200))
nothing # hide
```

The `cost` function means short measurements are cheap (cost ≈ 1) and long measurements are expensive (cost up to 1.5).

## Running the adaptive experiment

```@example adaptive
prior = Particles(prob, 1000)

result = run_adaptive(
    prob, candidates, prior, acquire;
    budget = 20.0,
    n_per_step = 1,
    headless = true,
    record_posterior = true,
)

log = result.log
n = length(log)
nothing # hide
```

## Batch comparison

A fair comparison uses the same number of measurements:

```@example adaptive
ξ_batch = design(prob, candidates, prior; n = n)

result_batch = run_batch(ξ_batch, prob, prior, acquire)
nothing # hide
```

## Design trajectory

The sequence of chosen design points reveals how the algorithm's strategy evolves:

```@example adaptive
fig = Figure(size = (700, 400))
ax = CairoMakie.Axis(fig[1, 1], xlabel = "Step", ylabel = "Design time t",
    title = "Adaptive Design Trajectory")
scatter!(ax, 1:n, [e.x.t for e in log],
    color = 1:n, colormap = :viridis, markersize = 8)
lines!(ax, 1:n, [e.x.t for e in log], color = :gray, linewidth = 0.5)
fig
```

## Credible bands

```@example adaptive
plot_credible_bands(prob, result, result_batch;
    labels = ["Adaptive", "Batch"], truth = θ_true)
```

## Corner plot — adaptive vs batch

```@example adaptive
plot_corner(result, result_batch;
    labels = ["Adaptive", "Batch"], truth = θ_true)
```

## Posterior evolution

With `record_posterior = true`, you can create an animation of the posterior evolving as observations arrive:

```@example adaptive
record_corner_animation(log, "posterior_evolution.gif";
    truth = θ_true, framerate = 3)
nothing # hide
```

![Posterior evolution](posterior_evolution.gif)
