# Quickstart

A complete example in under 30 lines: design an experiment for an exponential decay, acquire simulated data, and estimate the parameters.

```julia
using OptimalDesign
using ComponentArrays
using Distributions

# 1. Define the model and prior uncertainty
prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),          # model: y = A exp(-R₂ t)
    parameters = (A = LogUniform(0.1, 10),         # prior on amplitude
                  R₂ = Uniform(1, 50)),            # prior on decay rate
    transformation = select(:R₂),                  # we care about R₂
    sigma = (θ, ξ) -> 0.1,                         # measurement noise σ
)

# 2. Candidate measurement times and a prior particle set
candidates = [(t = t,) for t in range(0.001, 0.5, length = 200)]
prior = Particles(prob, 1000)

# 3. Compute the optimal design (20 measurements)
ξ = design(prob, candidates, prior; n = 20)
display(ξ)   # displays support points, counts, and a bar chart

# 4. Acquire data (here simulated; replace with your instrument)
θ_true = ComponentArray(A = 1.0, R₂ = 25.0)
function acquire(x)
    A = θ_true.A
    R₂ = θ_true.R₂
    t = x.t
    A * exp(-R₂ * t) + 0.1 * randn()
end

posterior = Particles(prob, 1000)
result = run_batch(ξ, prob, posterior, acquire)

# 5. Inspect the posterior
display(mean(result.posterior))   # ≈ (A = 1.0, R₂ = 25.0)
display(std(result.posterior))
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
