# Batch Design

The simplest use of OptimalDesign.jl: compute an optimal batch design for an exponential decay, verify its optimality, compare it to uniform spacing, then acquire simulated data and examine the posterior.

## The model

An exponential decay with amplitude ``A`` and rate ``R_2``:

```math
y = A \exp(-R_2 \, t) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2)
```

We want to estimate ``R_2`` as precisely as possible (Ds-optimality), treating ``A`` as a nuisance parameter.

## Problem setup

```julia
using OptimalDesign
using ComponentArrays, Distributions

prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    parameters = (A = LogUniform(0.1, 10), R₂ = Uniform(1, 50)),
    transformation = select(:R₂),
    sigma = (θ, ξ) -> 0.1,
)

candidates = [(t = t,) for t in range(0.001, 0.5, length = 200)]
prior = Particles(prob, 1000)
```

Key choices:

- `select(:R₂)` tells the algorithm we care about ``R_2`` — this switches from D-optimal to Ds-optimal design
- The prior distributions on ``A`` and ``R_2`` define the Bayesian averaging: the design will be good across the full range of plausible parameter values, not just at a single guess
- `sigma` is constant here but can depend on ``\theta`` and ``\xi``

!!! tip "D-optimal vs Ds-optimal"
    Omit the `transformation` keyword (or set `transformation = Identity()`) to get a full D-optimal design that treats all parameters equally. With `select(:R₂)`, the design focuses information on ``R_2`` and may sacrifice some precision on ``A``.

## Computing the design

```julia
d = design(prob, candidates, prior; n = 20, exchange_steps = 200)
```

This runs the exchange algorithm to find the optimal allocation of 20 measurements across the candidate time points. The result displays as:

```
ExperimentalDesign: 20 measurements at 3 support points
  t=0.001   ×7  ███████
  t=0.0416  ×6  ██████
  t=0.5     ×7  ███████
```

The Ds-optimal design concentrates measurements at two extremes (very short and very long times) with a cluster near ``1/R_2``. This is characteristic of designs for exponential models — short times pin down the amplitude, long times pin down the rate.

## Checking optimality

The General Equivalence Theorem provides a certificate. If the maximum Gateaux derivative equals the transformed dimension ``q``, the design is optimal:

```julia
opt = OptimalDesign.verify_optimality(prob, candidates, prior, d;
    posterior_samples = 1000)

opt.is_optimal      # true
opt.max_derivative   # ≈ q
opt.dimension        # q = 1 for select(:R₂)
```

Visualise with:

```julia
gd = OptimalDesign.gateaux_derivative(prob, candidates, prior, d;
    posterior_samples = 1000)

fig = OptimalDesign.plot_gateaux(candidates, gd, opt.dimension)
```

The Gateaux derivative should touch the ``q`` bound at the support points and lie below it everywhere else.

## Efficiency comparison

How much better is the optimal design than uniform spacing?

```julia
u = OptimalDesign.uniform_allocation(candidates, 20)

eff = efficiency(u, d, prob, candidates, prior; posterior_samples = 1000)
# eff ≈ 0.3–0.5 typically
```

An efficiency of 0.4 means the uniform design would need roughly ``1/0.4 = 2.5\times`` more measurements to match the optimal design's precision on ``R_2``.

## Simulated acquisition

Define a ground truth and an acquisition function, then run the experiment:

```julia
θ_true = ComponentArray(A = 1.0, R₂ = 25.0)
acquire = ξ -> θ_true.A * exp(-θ_true.R₂ * ξ.t) + 0.1 * randn()

# Optimal design
posterior_opt = Particles(prob, 1000)
result_opt = run_batch(d, prob, posterior_opt, acquire)

# Uniform design
posterior_unif = Particles(prob, 1000)
result_unif = run_batch(u, prob, posterior_unif, acquire)
```

Compare the posteriors:

```julia
mean(result_opt.posterior)   # ≈ (A = 1.0, R₂ = 25.0)
mean(result_unif.posterior)  # less precise on R₂
```

## Visualisation

### Credible bands

```julia
grid = [(t = t,) for t in range(0.001, 0.5, length = 100)]

fig = OptimalDesign.plot_credible_bands(prob,
    [prior, result_opt.posterior, result_unif.posterior], grid;
    labels = ["Prior", "Optimal (20 obs)", "Uniform (20 obs)"],
    truth = θ_true,
    observations = [nothing, result_opt.observations, result_unif.observations])
```

### Corner plots

```julia
# Prior vs optimal posterior
fig = plot_corner(prior, result_opt.posterior;
    params = [:A, :R₂], labels = ["Prior", "Optimal"],
    truth = (A = 1.0, R₂ = 25.0))

# Optimal vs uniform posterior
fig = plot_corner(result_unif.posterior, result_opt.posterior;
    params = [:A, :R₂], labels = ["Uniform", "Optimal"],
    truth = (A = 1.0, R₂ = 25.0))
```

The corner plot shows the prior (broad, diffuse) contracting to a tight posterior around the true values. The optimal design produces a tighter posterior on ``R_2`` than the uniform design — exactly as the efficiency ratio predicted.

See [`examples/1_exponential_decay.jl`](https://github.com/your-org/OptimalDesign.jl/blob/main/examples/1_exponential_decay.jl) for the full runnable script.
