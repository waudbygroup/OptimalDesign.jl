# Workflows

OptimalDesign.jl supports three workflows, each built from the same core functions.

## 1. Batch design → acquire

You know the experiment size up front. Compute the optimal allocation, then acquire all data at once.

Specify the size either as a number of measurements (`n`) or a total cost budget (`budget`):

```julia
# Fixed number of measurements
ξ = design(prob, candidates, prior; n = 20)

# Or: budget-based — number of measurements determined by costs
ξ = design(prob, candidates, prior; budget = 50.0)
```

Then acquire and update:

```julia
result = run_batch(ξ, prob, prior, acquire)

# result.prior        — original particle prior
# result.posterior     — updated particle posterior
# result.observations  — vector of (x=..., y=...) named tuples
# result.design        — the ExperimentalDesign used
```

`design()` returns an `ExperimentalDesign` — a list of (design point, count) pairs. `run_batch()` iterates over the design, calls your `acquire` function at each point, and updates the posterior with all observations using likelihood tempering.

There is also a convenience form that designs and acquires in one call:

```julia
result = run_batch(prob, candidates, prior, acquire; budget = 50.0)
```

## 2. Adaptive sequential design

You have a budget and want to choose each measurement based on what you've learned so far. The algorithm loops: design one (or a few) measurements → acquire → update posterior → repeat.

```julia
result = run_adaptive(
    prob, candidates, prior, acquire;
    budget = 100.0,       # total cost budget
    n_per_step = 1,       # measurements per step (1 = fully sequential)
)

# result.prior        — original prior
# result.posterior     — final posterior after all observations
# result.log           — ExperimentLog with full history
# result.observations  — extracted from log
```

Each step, `run_adaptive` calls `design()` internally to pick the next best measurement given the current posterior. The posterior is updated after each observation, so later measurements are informed by earlier ones.

Use `n_per_step > 1` to design mini-batches — useful when switching costs make fully sequential design wasteful.

Adaptive design is most valuable when:

- The optimal design depends strongly on the (unknown) parameter values
- You can afford the overhead of redesigning between measurements
- You want to monitor convergence and stop early if the posterior is precise enough

## 3. Design only (no acquisition)

Sometimes you just want to compute and inspect a design without acquiring any data — for example, to compare criteria, check optimality, or plan a future experiment.

```julia
ξ = design(prob, candidates, prior; n = 20)

# Check optimality visually
plot_gateaux(prob, candidates, prior, ξ)

# Compare to a uniform design
ξ_unif = uniform_allocation(candidates, 20)
efficiency(ξ_unif, ξ, prob, candidates, prior)
```

## Choosing between workflows

| Scenario | Workflow |
|----------|----------|
| Fixed number of measurements, no feedback loop | Batch |
| Budget-limited, can redesign between measurements | Adaptive |
| Planning phase, comparing design strategies | Design only |
| Expensive switching between measurement conditions | Adaptive with `switching_cost` |

## The `acquire` function

All workflows that collect data need an `acquire` function: a callable that takes a design point `x` (a `NamedTuple`) and returns an observation `y` (a scalar or vector).

For simulated experiments:
```julia
acquire(x) = model(θ_true, x) + σ * randn()
```

For real instruments, `acquire` would send commands to hardware and return the measured value. The design point `x` contains all the experimental settings (e.g., `x.t` for time, `x.dose` for concentration).
