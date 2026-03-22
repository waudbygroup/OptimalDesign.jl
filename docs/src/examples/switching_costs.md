# Switching Costs

In many experiments, changing measurement configuration is expensive — switching an instrument channel, moving a sample, or recalibrating. OptimalDesign.jl models this with a **switching cost**: a fixed penalty incurred whenever a discrete design variable changes value between consecutive measurements.

## The model

Two exponential decays, selectively measured via a discrete control variable ``i \in \{1, 2\}``:

```math
y = A_i \exp(-R_{2,i} \, t) + \varepsilon
```

Each measurement observes **one** decay (chosen by ``i``). Switching between decays costs 50 time units — for instance, physically repositioning a probe.

## Problem setup

```julia
using OptimalDesign
using ComponentArrays, Distributions

prob = DesignProblem(
    (θ, ξ) -> ξ.i == 1 ? θ.A₁ * exp(-θ.R₂₁ * ξ.t) :
                          θ.A₂ * exp(-θ.R₂₂ * ξ.t),
    parameters = (
        A₁  = Normal(1, 0.1),    R₂₁ = LogUniform(1, 50),
        A₂  = Normal(1, 0.1),    R₂₂ = LogUniform(1, 50)),
    transformation = select(:R₂₁, :R₂₂),
    sigma = (θ, ξ) -> 0.05,
    cost = ξ -> ξ.t + 1,
    switching_cost = (:i, 50.0),
)
```

The `switching_cost = (:i, 50.0)` argument does two things:

1. Creates a `SwitchingDesignProblem` (instead of a plain `DesignProblem`)
2. Adds 50 to the cost whenever the value of `ξ.i` changes between consecutive measurements

The per-measurement cost (`ξ.t + 1`) is always paid; the switching cost is additional.

## Candidates

The candidate set includes both decays crossed with a range of times:

```julia
candidates = [(i = i, t = t) for i in [1, 2]
                               for t in range(0.001, 0.5, length = 200)]
```

This gives 400 candidates: 200 times × 2 channels.

## Running the adaptive experiment

```julia
θ_true = ComponentArray(A₁ = 1.0, R₂₁ = 10.0, A₂ = 1.0, R₂₂ = 40.0)
acquire = ξ -> (ξ.i == 1 ?
    θ_true.A₁ * exp(-θ_true.R₂₁ * ξ.t) :
    θ_true.A₂ * exp(-θ_true.R₂₂ * ξ.t)) + 0.05 * randn()

prior = Particles(prob, 1000)

result = run_adaptive(
    prob, candidates, prior, acquire;
    budget = 200.0,
    n_per_step = 1,
    record_posterior = true,
)
```

## Interpreting the results

The switching cost creates a tension: the algorithm wants to measure **both** decays to learn both rates, but each switch consumes budget. The result is characteristic behaviour:

- **Blocks** of measurements on one decay, then a switch, then a block on the other
- Fewer total observations than a zero-switching-cost experiment with the same budget
- The algorithm switches only when the information gain from the other decay exceeds the switching penalty

```julia
log = result.log
n = length(log)

n_decay1 = count(e -> e.ξ.i == 1, log)
n_decay2 = count(e -> e.ξ.i == 2, log)
n_switches = count(i -> log[i].ξ.i != log[i-1].ξ.i, 2:n)

println("$n observations: $n_decay1 on decay 1, $n_decay2 on decay 2")
println("$n_switches switches (costing $(n_switches * 50.0) total)")
```

## Design trajectory

The trajectory plot reveals the blocking structure:

```julia
fig = Figure(size = (800, 700))

# Panel 1: which decay and what time
ax1 = Axis(fig[1, 1], ylabel = "Design time t",
    title = "Adaptive Design Trajectory")

steps_1 = [i for i in 1:n if log[i].ξ.i == 1]
steps_2 = [i for i in 1:n if log[i].ξ.i == 2]

scatter!(ax1, steps_1, [log[i].ξ.t for i in steps_1],
    color = :blue, markersize = 8, label = "Decay 1")
scatter!(ax1, steps_2, [log[i].ξ.t for i in steps_2],
    color = :orange, markersize = 8, label = "Decay 2")

# Mark switches
for i in 2:n
    if log[i].ξ.i != log[i-1].ξ.i
        vlines!(ax1, [i], color = (:red, 0.3), linewidth = 1)
    end
end
axislegend(ax1)

# Panel 2: cumulative cost with switch markers
ax2 = Axis(fig[2, 1], xlabel = "Step", ylabel = "Cumulative cost")
cumcost = cumsum([e.cost for e in log])
lines!(ax2, 1:n, cumcost, color = :black, linewidth = 2)
hlines!(ax2, [200], color = :gray, linestyle = :dash)  # budget line
```

Vertical red lines mark switches. The cumulative cost curve shows jumps at each switch.

## Comparing to batch design

A batch design computed from the prior doesn't know about switching costs — it freely interleaves both decays. The adaptive design learns which decay needs more attention and sequences measurements to minimise switching:

```julia
prior_batch = Particles(prob, 1000)
d = design(prob, candidates, prior_batch; n = n, exchange_steps = 200)

posterior_batch = Particles(prob, 1000)
result_batch = run_batch(d, prob, posterior_batch, acquire)

fig = plot_corner(result.posterior, result_batch.posterior;
    params = [:R₂₁, :R₂₂], labels = ["Adaptive", "Batch"],
    truth = (R₂₁ = 10.0, R₂₂ = 40.0))
```

## Posterior evolution

```julia
if OptimalDesign.has_posterior_history(log)
    record_corner_animation(log, "switching_posterior.mp4";
        params = [:R₂₁, :R₂₂],
        truth = (R₂₁ = 10.0, R₂₂ = 40.0),
        framerate = 5)
end
```

The animation shows the posterior on ``R_{2,1}`` tightening while measuring decay 1, then ``R_{2,2}`` tightening when the algorithm switches.

See [`examples/6_adaptive_selective.jl`](https://github.com/your-org/OptimalDesign.jl/blob/main/examples/6_adaptive_selective.jl) for the full runnable script.
