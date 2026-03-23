# Vector Observations

When each measurement returns multiple values simultaneously — for example, two spectral channels or two decays observed in parallel — the model is **vector-valued**. OptimalDesign.jl handles this naturally: the Jacobian becomes a matrix, and the FIM accumulates information from all outputs.

## The model

Two exponential decays observed simultaneously at time ``t``:

```math
\mathbf{y} = \begin{bmatrix} A_1 \exp(-R_{2,1} \, t) \\ A_2 \exp(-R_{2,2} \, t) \end{bmatrix} + \boldsymbol{\varepsilon}
```

Four parameters ``(A_1, R_{2,1}, A_2, R_{2,2})``, interest in both rates ``(R_{2,1}, R_{2,2})``.

## Setup

The only difference from a scalar model is that `model` and `sigma` return vectors:

```@example vector
using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random; Random.seed!(42) # hide

function model(θ, x)
    [θ.A₁ * exp(-θ.R₂₁ * x.t),
     θ.A₂ * exp(-θ.R₂₂ * x.t)]
end

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A₁ = 7.0, R₂₁ = 8.0, A₂ = 1.0, R₂₂ = 80.0)
σ = 0.05
acquire(x) = model(θ_true, x) .+ σ .* randn(2)

# Design problem
prob = DesignProblem(
    model,
    parameters = (
        A₁  = LogUniform(0.1, 10), R₂₁ = Uniform(0.1, 100),
        A₂  = LogUniform(0.1, 10), R₂₂ = Uniform(0.1, 100)),
    transformation = select(:R₂₁, :R₂₂),
    sigma = Returns([σ, σ]),
    cost = x -> x.t + 1,
)

candidates = candidate_grid(t = range(0.001, 0.5, length = 200))
prior = Particles(prob, 1000)
nothing # hide
```

## Computing the design

```@example vector
ξ = design(prob, candidates, prior; n = 100, exchange_steps = 200)
```

Because both decays contribute to the FIM at every time point, the design must balance information for the fast decay (needs short times) and the slow decay (needs longer times).

## Acquisition and posterior

```@example vector
result = run_batch(ξ, prob, prior, acquire)
nothing # hide
```

## Credible bands

For vector models, `plot_credible_bands` automatically creates one column per output component:

```@example vector
plot_credible_bands(prob, result; truth = θ_true)
```

## Corner plots

With four parameters, the corner plot is a 4×4 grid:

```@example vector
plot_corner(result; truth = θ_true)
```

Because the two decays share no parameters, the scatter plots should show approximate independence between the ``(A_1, R_{2,1})`` and ``(A_2, R_{2,2})`` blocks — the posterior factorises.
