# Vector Observations

When each measurement returns multiple values simultaneously — for example, two spectral channels or two decays observed in parallel — the model is **vector-valued**. OptimalDesign.jl handles this naturally: the Jacobian becomes a matrix, and the FIM accumulates information from all outputs.

## The model

Two exponential decays observed simultaneously at time ``t``:

```math
\mathbf{y} = \begin{bmatrix} A_1 \exp(-k_1 \, t) \\ A_2 \exp(-k_2 \, t) \end{bmatrix} + \boldsymbol{\varepsilon}
```

Four parameters ``(A_1, k_1, A_2, k_2)``, interest in both rates ``(k_1, k_2)``.

```@example vector
using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random; Random.seed!(42) # hide

function model(θ, x)
    [θ.A₁ * exp(-θ.k₁ * x.t),
     θ.A₂ * exp(-θ.k₂ * x.t)]
end

θ_true = ComponentArray(A₁ = 7.0, k₁ = 8.0, A₂ = 1.0, k₂ = 80.0)
σ = 0.05
acquire(x) = model(θ_true, x) .+ σ .* randn(2)
nothing # hide
```

## Defining the design problem

The key difference from a scalar model: `model` returns a vector and `sigma` must return a vector of matching length, giving the noise standard deviation for each output component.

We also add a `cost` function — longer measurements cost more. The `budget` keyword (below) will determine how many measurements to make.

```@example vector
prob = DesignProblem(
    model,
    parameters = (
        A₁ = LogUniform(0.1, 10), k₁ = Uniform(0.1, 100),
        A₂ = LogUniform(0.1, 10), k₂ = Uniform(0.1, 100)),
    transformation = select(:k₁, :k₂),
    sigma = Returns([σ, σ]),
    cost = x -> x.t + 1,
)

candidates = candidate_grid(t = range(0.001, 0.5, length = 200))
nothing # hide
```

## Computing the design

With `budget`, the number of measurements is determined automatically from the per-measurement costs — no need to specify `n` explicitly. The design must balance information for the fast decay (short times) and the slow decay (longer times).

```@example vector
prior = Particles(prob, 1000)
ξ = design(prob, candidates, prior; budget = 100.0)
```

## Running the experiment

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

With four parameters, the corner plot is a 4×4 grid. Because the two decays share no parameters, the off-diagonal blocks should show approximate independence — the posterior factorises:

```@example vector
plot_corner(result; truth = θ_true)
```
