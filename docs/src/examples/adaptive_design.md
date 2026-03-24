# Adaptive Design

In an adaptive experiment, each measurement is chosen based on what you've learned so far. The algorithm loops: **design → acquire → update posterior → repeat**. This can outperform batch design when the optimal measurement locations depend on the (unknown) parameter values.

## The model

The same exponential decay from the [Batch Design](@ref) example:

```math
y = A \exp(-k \, t) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
```

```@example adaptive
using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random; Random.seed!(42) # hide

function model(θ, x)
    θ.A * exp(-θ.k * x.t)
end

θ_true = ComponentArray(A = 2.0, k = 42.0)
σ = 0.1
acquire(x) = model(θ_true, x) + σ * randn()
nothing # hide
```

## Defining the design problem

This time we add a per-measurement `cost` — short measurements are cheap (cost ≈ 1) and long measurements are more expensive (cost up to 1.5). The experiment will be controlled by a total `budget` rather than a fixed number of measurements:

```@example adaptive
prob = DesignProblem(
    model,
    parameters = (A = LogUniform(0.1, 10), k = Uniform(1, 50)),
    transformation = select(:k),
    sigma = Returns(σ),
    cost = x -> x.t + 1,
)

candidates = candidate_grid(t = range(0.001, 0.5, length = 200))
nothing # hide
```

## Running the adaptive experiment

`run_adaptive` takes a `budget` and loops until it is exhausted. At each step it calls `design` internally using the current posterior, acquires a measurement, and updates. With `n_per_step = 1` the experiment is fully sequential — each observation informs the next.

```@example adaptive
prior = Particles(prob, 5000)

result = run_adaptive(
    prob, candidates, prior, acquire;
    budget = 50.0,
    headless = true, # hide
)
nothing # hide
```

A live dashboard is displayed of the acquisition process and evolution of parameter estimates. This can also be saved as a gif or mp4 following acquisition:

```@example adaptive
record_dashboard(result, prob; filename="dashboard.gif")
nothing # hide
```

![Acquisition dashboard](dashboard.gif)



## Batch comparison

For a fair comparison, we design a batch experiment with the same budget. Since both use cost-aware design, the difference is purely whether the design adapts to incoming data:

```@example adaptive
result_batch = run_batch(prob, candidates, prior, acquire;
    budget = 50.0)
nothing # hide
```

## Convergence

`plot_convergence` shows how the parameter estimates evolve as observations accumulate. The shaded band is ± 1 posterior standard deviation; the dashed line marks the true value:

```@example adaptive
plot_convergence(result; truth = θ_true)
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
record_corner_animation(result.log, "posterior_evolution.gif";
    truth = θ_true, framerate = 3)
nothing # hide
```

![Posterior evolution](posterior_evolution.gif)
