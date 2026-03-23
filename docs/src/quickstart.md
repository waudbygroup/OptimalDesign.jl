# Quickstart

A complete worked example: design an experiment for an exponential decay, acquire simulated data, estimate the parameters, and visualise the results.

## The model and ground truth

```@example quickstart
using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random; Random.seed!(42) # hide

# The physical model: exponential decay with amplitude A and rate R₂
function model(θ, x)
    θ.A * exp(-θ.R₂ * x.t)
end

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A = 1.0, R₂ = 25.0)
σ = 0.1

# Simulated instrument: call this to "measure" at design point x
acquire(x) = model(θ_true, x) + σ * randn()
nothing # hide
```

## Setting up the design problem

The `DesignProblem` tells the algorithm what it needs to know: the model, prior uncertainty on each parameter, and what we want to estimate. We want to estimate ``R_2`` as precisely as possible, treating ``A`` as a nuisance parameter.

```@example quickstart
prob = DesignProblem(
    model,
    parameters = (A = LogUniform(0.1, 10), R₂ = Uniform(1, 50)),
    transformation = select(:R₂),
    sigma = Returns(σ),
)

candidates = candidate_grid(t = range(0.001, 0.5, length = 200))
prior = Particles(prob, 1000)
nothing # hide
```

## Computing the optimal design

```@example quickstart
ξ = design(prob, candidates, prior; n = 20)
```

## Running the experiment

```@example quickstart
result = run_batch(ξ, prob, prior, acquire)
```

## Visualising results

Credible bands show how the posterior prediction narrows compared to the prior:

```@example quickstart
plot_credible_bands(prob, result; truth = θ_true)
```

A corner plot shows the joint posterior over the parameters:

```@example quickstart
plot_corner(result; truth = θ_true)
```

That's it. The key objects are:

| Object | What it is |
|--------|-----------|
| `DesignProblem` | Your model, noise, prior, and what you want to learn |
| `Particles` | A weighted particle set representing parameter uncertainty |
| `ExperimentalDesign` | Which design points to measure and how many times |

Next steps:

- [Workflows](@ref) — batch vs adaptive vs design-only
- [Defining Problems](@ref) — all the options for `DesignProblem`
- [Batch Design example](@ref "Batch Design") — full walkthrough with optimality checking and plots
