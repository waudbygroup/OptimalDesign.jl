# Vector Observations

When each measurement returns multiple values simultaneously — for example, two spectral channels or two decays observed in parallel — the model is **vector-valued**. OptimalDesign.jl handles this naturally: the Jacobian becomes a matrix, and the FIM accumulates information from all outputs.

## The model

Two exponential decays observed simultaneously at time ``t``:

```math
\mathbf{y} = \begin{bmatrix} A_1 \exp(-R_{2,1} \, t) \\ A_2 \exp(-R_{2,2} \, t) \end{bmatrix} + \boldsymbol{\varepsilon}
```

Four parameters ``(A_1, R_{2,1}, A_2, R_{2,2})``, interest in both rates ``(R_{2,1}, R_{2,2})``.

## Problem setup

The only difference from a scalar model is that `predict` and `sigma` return vectors:

```julia
using OptimalDesign
using ComponentArrays, Distributions

prob = DesignProblem(
    (θ, ξ) -> [θ.A₁ * exp(-θ.R₂₁ * ξ.t),
               θ.A₂ * exp(-θ.R₂₂ * ξ.t)],
    parameters = (
        A₁  = LogUniform(0.1, 10), R₂₁ = Uniform(0.1, 100),
        A₂  = LogUniform(0.1, 10), R₂₂ = Uniform(0.1, 100)),
    transformation = select(:R₂₁, :R₂₂),
    sigma = (θ, ξ) -> [0.05, 0.05],
    cost = ξ -> ξ.t + 1,
)
```

Note:

- `predict` returns a 2-element vector — one per decay
- `sigma` returns a matching 2-element vector of noise standard deviations
- `select(:R₂₁, :R₂₂)` targets both rates for Ds-optimality
- `cost` makes longer measurements more expensive

## Computing the design

```julia
candidates = [(t = t,) for t in range(0.001, 0.5, length = 200)]
prior = Particles(prob, 1000)

d = design(prob, candidates, prior; n = 100, exchange_steps = 200)
```

Because both decays contribute to the FIM at every time point, the design must balance information for the fast decay (needs short times) and the slow decay (needs longer times). The exchange algorithm finds this balance automatically.

## Acquisition and posterior

```julia
θ_true = ComponentArray(A₁ = 7.0, R₂₁ = 8.0, A₂ = 1.0, R₂₂ = 80.0)
acquire = ξ -> [θ_true.A₁ * exp(-θ_true.R₂₁ * ξ.t),
                θ_true.A₂ * exp(-θ_true.R₂₂ * ξ.t)] .+ 0.05 .* randn(2)

posterior = Particles(prob, 1000)
result = run_batch(d, prob, posterior, acquire)

mean(result.posterior)
# ≈ (A₁ = 7.0, R₂₁ = 8.0, A₂ = 1.0, R₂₂ = 80.0)
```

## Credible bands

For vector models, `plot_credible_bands` automatically creates one column per output component:

```julia
grid = [(t = t,) for t in range(0.001, 0.5, length = 100)]

fig = OptimalDesign.plot_credible_bands(prob,
    [prior, result.posterior], grid;
    labels = ["Prior", "Posterior"],
    truth = θ_true,
    observations = [nothing, result.observations])
```

This produces a figure with two columns — one for each decay — showing how the posterior bands tighten around the true signal.

## Corner plots

With four parameters, the corner plot is a 4×4 grid:

```julia
fig = plot_corner(prior, result.posterior;
    params = [:A₁, :A₂, :R₂₁, :R₂₂],
    labels = ["Prior", "Posterior"],
    truth = (A₁ = 7.0, A₂ = 1.0, R₂₁ = 8.0, R₂₂ = 80.0))
```

Because the two decays share no parameters, the scatter plots should show approximate independence between the ``(A_1, R_{2,1})`` and ``(A_2, R_{2,2})`` blocks — the posterior factorises.

## Comparison with selective measurement

In this example, each measurement observes **both** decays simultaneously. An alternative is to choose which decay to observe at each time point (a discrete control variable). See the [Switching Costs](@ref) example for that approach, where you must also account for the cost of changing measurement configurations.

See [`examples/3_two_decays_vector.jl`](https://github.com/your-org/OptimalDesign.jl/blob/main/examples/3_two_decays_vector.jl) for the full runnable script.
