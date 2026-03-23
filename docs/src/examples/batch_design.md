# Batch Design

The simplest use of OptimalDesign.jl: compute an optimal batch design for an exponential decay, verify its optimality, compare it to uniform spacing, then acquire simulated data and examine the posterior.

## The model

An exponential decay with amplitude ``A`` and rate ``R_2``:

```math
y = A \exp(-R_2 \, t) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
```

We want to estimate ``R_2`` as precisely as possible (Ds-optimality), treating ``A`` as a nuisance parameter.

## Setup

```@example batch
using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random; Random.seed!(42) # hide

function model(θ, x)
    θ.A * exp(-θ.R₂ * x.t)
end

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A = 1.0, R₂ = 25.0)
σ = 0.1
acquire(x) = model(θ_true, x) + σ * randn()

# Design problem: model, priors, what to estimate
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

Key choices:

- `select(:R₂)` tells the algorithm we care about ``R_2`` — this switches from D-optimal to Ds-optimal design
- The prior distributions on ``A`` and ``R_2`` define the Bayesian averaging: the design will be good across the full range of plausible parameter values, not just at a single guess
- `sigma` is constant here but can depend on ``\theta`` and ``x``

!!! tip "D-optimal vs Ds-optimal"
    Omit the `transformation` keyword (or set `transformation = Identity()`) to get a full D-optimal design that treats all parameters equally. With `select(:R₂)`, the design focuses information on ``R_2`` and may sacrifice some precision on ``A``.

## Computing the design

```@example batch
ξ = design(prob, candidates, prior; n = 20, exchange_steps = 200)
```

The Ds-optimal design concentrates measurements at two extremes (very short and very long times) with a cluster near ``1/R_2``. This is characteristic of designs for exponential models — short times pin down the amplitude, long times pin down the rate.

## Checking optimality

The General Equivalence Theorem provides a certificate. If the maximum Gateaux derivative equals the transformed dimension ``q``, the design is optimal:

```@example batch
opt = verify_optimality(prob, candidates, prior, ξ;
    posterior_samples = 1000)
opt
```

Visualise the Gateaux derivative — it should touch the ``q`` bound at the support points and lie below it everywhere else:

```@example batch
plot_gateaux(opt)
```

## Efficiency comparison

How much better is the optimal design than uniform spacing?

```@example batch
ξ_unif = uniform_allocation(candidates, 20)

eff = efficiency(ξ_unif, ξ, prob, candidates, prior; posterior_samples = 1000)
nothing # hide
```

## Simulated acquisition

Run the experiment with both designs and compare:

```@example batch
# Optimal design
result_opt = run_batch(ξ, prob, prior, acquire)

# Uniform design
result_unif = run_batch(ξ_unif, prob, prior, acquire)
nothing # hide
```

## Credible bands

```@example batch
plot_credible_bands(prob, result_opt, result_unif;
    labels = ["Optimal (20 obs)", "Uniform (20 obs)"], truth = θ_true)
```

## Corner plots

Prior vs optimal posterior:

```@example batch
plot_corner(result_opt; truth = θ_true)
```

Optimal vs uniform posterior:

```@example batch
plot_corner(result_unif, result_opt;
    labels = ["Uniform", "Optimal"], truth = θ_true)
```

The corner plot shows the prior (broad, diffuse) contracting to a tight posterior around the true values. The optimal design produces a tighter posterior on ``R_2`` than the uniform design — exactly as the efficiency ratio modeled.
