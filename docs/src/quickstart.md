# Quickstart

A complete worked example: design an experiment for an exponential decay, acquire simulated data, estimate the parameters, and visualise the results.

## The model

An exponential decay with amplitude ``A`` and rate ``k``, observed with additive Gaussian noise:

```math
y = A \exp(-k \, t) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
```

```@example quickstart
using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random; Random.seed!(42) # hide

function model(θ, x)
    θ.A * exp(-θ.k * x.t)
end

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A = 1.0, k = 25.0)
σ = 0.1
nothing # hide
```

## The acquisition function

The `acquire` function is what gets called to make a measurement at a design point `x`. In a real experiment this would talk to your instrument; here we simulate by evaluating the model at the true parameters and adding noise:

```@example quickstart
acquire(x) = model(θ_true, x) + σ * randn()
nothing # hide
```

## Setting up the design problem

A `DesignProblem` bundles the model, prior uncertainty on each parameter, and what we want to estimate. Here we want to estimate ``k`` as precisely as possible, treating ``A`` as a nuisance parameter:

```@example quickstart
prob = DesignProblem(
    model,
    parameters = (A = LogUniform(0.1, 10), k = Uniform(1, 50)),
    transformation = select(:k),
    sigma = Returns(σ),
)

candidates = candidate_grid(t = range(0.001, 0.5, length = 200))
prior = Particles(prob, 1000)
nothing # hide
```

## Computing the optimal design

```@example quickstart
# create a design with 20 measurements
ξ = design(prob, candidates, prior; n = 20)
```

## Running the experiment

```@example quickstart
result = run_batch(ξ, prob, prior, acquire)
```

## Extracting estimates

The result carries both the original prior and the updated posterior. Use `mean` and `std` to summarise:

```@example quickstart
using Statistics
mean(result)
```

```@example quickstart
std(result)
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

## Next steps

- [Workflows](@ref) — batch vs adaptive vs design-only
- [Defining Problems](@ref) — all the options for `DesignProblem`
- [Batch Design example](@ref "Batch Design") — full walkthrough with optimality checking and plots
