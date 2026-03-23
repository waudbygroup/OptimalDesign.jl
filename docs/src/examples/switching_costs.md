# Switching Costs

In many experiments, changing measurement configuration is expensive — switching an instrument channel, moving a sample, or recalibrating. OptimalDesign.jl models this with a **switching cost**: a fixed penalty incurred whenever a discrete design variable changes value between consecutive measurements.

## The model

Two exponential decays, selectively measured via a discrete control variable ``i \in \{1, 2\}``:

```math
y = A_i \exp(-R_{2,i} \, t) + \varepsilon
```

Each measurement observes **one** decay (chosen by ``i``). Switching between decays costs 50 time units.

## Setup

```@example switching
using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random; Random.seed!(42) # hide

function model(θ, x)
    if x.i == 1
        θ.A₁ * exp(-θ.R₂₁ * x.t)
    else
        θ.A₂ * exp(-θ.R₂₂ * x.t)
    end
end

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A₁ = 1.0, R₂₁ = 10.0, A₂ = 1.0, R₂₂ = 40.0)
σ = 0.05
acquire(x) = model(θ_true, x) + σ * randn()

# Design problem with switching cost on the discrete variable i
prob = DesignProblem(
    model,
    parameters = (
        A₁  = Normal(1, 0.1),    R₂₁ = LogUniform(1, 50),
        A₂  = Normal(1, 0.1),    R₂₂ = LogUniform(1, 50)),
    transformation = select(:R₂₁, :R₂₂),
    sigma = Returns(σ),
    cost = x -> x.t + 1,
    switching_cost = (:i, 50.0),
)

candidates = candidate_grid(i = [1, 2], t = range(0.001, 0.5, length = 200))
nothing # hide
```

The `switching_cost = (:i, 50.0)` adds 50 to the cost whenever the value of `x.i` changes between consecutive measurements.

## Running the adaptive experiment

```@example switching
prior = Particles(prob, 1000)

result = run_adaptive(
    prob, candidates, prior, acquire;
    budget = 200.0,
    n_per_step = 1,
    headless = true,
    record_posterior = true,
)

log = result.log
n = length(log)

n_decay1 = count(e -> e.x.i == 1, log)
n_decay2 = count(e -> e.x.i == 2, log)
n_switches = count(i -> log[i].x.i != log[i-1].x.i, 2:n)

println("$n observations: $n_decay1 on decay 1, $n_decay2 on decay 2")
println("$n_switches switches (costing $(n_switches * 50.0) total)")
nothing # hide
```

## Design trajectory

The trajectory reveals the blocking structure — the algorithm measures one decay in a block, then switches:

```@example switching
fig = Figure(size = (800, 500))

ax = CairoMakie.Axis(fig[1, 1], xlabel = "Step", ylabel = "Design time t",
    title = "Adaptive Design Trajectory")

steps_1 = [i for i in 1:n if log[i].x.i == 1]
steps_2 = [i for i in 1:n if log[i].x.i == 2]

scatter!(ax, steps_1, [log[i].x.t for i in steps_1],
    color = :blue, markersize = 8, label = "Decay 1")
scatter!(ax, steps_2, [log[i].x.t for i in steps_2],
    color = :orange, markersize = 8, label = "Decay 2")

for i in 2:n
    if log[i].x.i != log[i-1].x.i
        vlines!(ax, [i], color = (:red, 0.3), linewidth = 1)
    end
end
axislegend(ax)
fig
```

## Budget consumption

```@example switching
fig2 = Figure(size = (700, 300))
ax2 = CairoMakie.Axis(fig2[1, 1], xlabel = "Step", ylabel = "Cumulative cost")
cumcost = cumsum([e.cost for e in log])
lines!(ax2, 1:n, cumcost, color = :black, linewidth = 2)
hlines!(ax2, [200], color = :gray, linestyle = :dash)

for i in 2:n
    if log[i].x.i != log[i-1].x.i
        scatter!(ax2, [i], [cumcost[i]], color = :red, markersize = 10, marker = :diamond)
    end
end
fig2
```

Red diamonds mark switch points — each one jumps the cumulative cost by 50.

## Corner plot

```@example switching
plot_corner(result; truth = θ_true)
```

## Posterior evolution

```@example switching
if has_posterior_history(log)
    record_corner_animation(log, "switching_posterior.gif";
        params = [:R₂₁, :R₂₂],
        truth = θ_true, framerate = 3)
end
nothing # hide
```

![Posterior evolution](switching_posterior.gif)
