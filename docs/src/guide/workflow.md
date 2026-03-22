# Workflows

OptimalDesign.jl supports three workflows, each built from the same core functions.

## 1. Batch design → acquire

You know how many measurements you can afford. Compute the optimal allocation up front, then acquire all data at once.

```julia
# Compute where to measure
d = design(prob, candidates, prior; n = 20)

# Acquire data (replace `acquire` with your instrument)
posterior = Particles(prob, 1000)
result = run_batch(d, prob, posterior, acquire)

# result.posterior  — updated particle posterior
# result.observations — vector of (ξ=..., y=...) named tuples
```

`design()` returns an `ExperimentalDesign` — a list of (design point, count) pairs. When displayed, it shows a compact summary with bar charts:

```
ExperimentalDesign: 20 measurements at 3 support points
  t=0.001   ×7  ███████
  t=0.0416  ×6  ██████
  t=0.5     ×7  ███████
```

`run_batch()` iterates over the design, calls your `acquire` function at each point, and updates the posterior with all observations using likelihood tempering.

## 2. Adaptive sequential design

You have a budget and want to choose each measurement based on what you've learned so far. The algorithm loops: design one (or a few) measurements → acquire → update posterior → repeat.

```julia
result = run_adaptive(
    prob, candidates, posterior, acquire;
    budget = 100.0,       # total cost budget
    n_per_step = 1,       # measurements per step (1 = fully sequential)
)

# result.posterior — final posterior after all observations
# result.log — ExperimentLog with full history
```

Each step, `run_adaptive` calls `design()` internally to pick the next best measurement given the current posterior. The posterior is updated after each observation, so later measurements are informed by earlier ones.

Adaptive design is most valuable when:

- The optimal design depends strongly on the (unknown) parameter values
- You can afford the overhead of redesigning between measurements
- You want to monitor convergence and stop early if the posterior is precise enough

## 3. Design only (no acquisition)

Sometimes you just want to compute and inspect a design without acquiring any data — for example, to compare criteria, check optimality, or plan a future experiment.

```julia
d = design(prob, candidates, prior; n = 20)

# Check optimality via the General Equivalence Theorem
opt = OptimalDesign.verify_optimality(prob, candidates, prior, d)
opt.is_optimal      # true if the Gateaux derivative is ≤ q everywhere
opt.max_derivative  # should be ≤ q (the transformed dimension)

# Compare to a uniform design
u = OptimalDesign.uniform_allocation(candidates, 20)
eff = efficiency(u, d, prob, candidates, prior)
# eff < 1 means the uniform design is less efficient
```

## Choosing between workflows

| Scenario | Workflow |
|----------|----------|
| Fixed number of measurements, no feedback loop | Batch |
| Budget-limited, can redesign between measurements | Adaptive |
| Planning phase, comparing design strategies | Design only |
| Expensive switching between measurement conditions | Adaptive with `switching_cost` |

## The `acquire` function

All workflows that collect data need an `acquire` function: a callable that takes a design point `ξ` (a `NamedTuple`) and returns an observation `y` (a scalar or vector).

For simulated experiments:
```julia
acquire = ξ -> prob.predict(θ_true, ξ) + σ * randn()
```

For real instruments, `acquire` would send commands to hardware and return the measured value. The design point `ξ` contains all the experimental settings (e.g., `ξ.t` for time, `ξ.dose` for concentration).
