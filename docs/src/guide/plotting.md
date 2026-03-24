# Plotting

OptimalDesign.jl provides plotting functions built on [Makie.jl](https://docs.makie.org/). All plotting functions are exported.

## Corner plots

The simplest form takes a result object and automatically shows prior vs posterior:

```julia
# From a result object (shows prior vs posterior automatically)
fig = plot_corner(result; truth = θ_true)
```

To compare two results directly (e.g., optimal vs uniform):

```julia
fig = plot_corner(result_opt, result_unif;
    labels = ["Optimal", "Uniform"], truth = θ_true)
```

- `truth` overlays dashed red crosshairs at known parameter values (accepts a `ComponentArray` or `NamedTuple`)
- `labels` names each posterior in the legend
- `params` selects which parameters to show (useful for high-dimensional models; defaults to all)
- `highlight` bolds specific parameter labels (auto-set from `select(...)` when using the result form)
- Pass any number of results or `Particles` arguments to compare them

## Credible bands

Plot model predictions with uncertainty bands. Pass one or more result objects:

```julia
# Single result — shows prior vs posterior with observations
fig = plot_credible_bands(prob, result; truth = θ_true)

# Compare two results — prior panel + one panel per result
fig = plot_credible_bands(prob, result_opt, result_unif;
    labels = ["Optimal", "Uniform"], truth = θ_true)
```

The prediction grid is inferred from observations automatically. Pass `x_grid` to override.

For vector-valued models, components are displayed as separate columns automatically.

## Convergence

Track how parameter estimates evolve over an adaptive experiment. Shows the posterior mean ± 1 standard deviation at each step, with optional true value overlay:

```julia
fig = plot_convergence(result; truth = θ_true)

# Show only specific parameters
fig = plot_convergence(result; truth = θ_true, params = [:k₁, :k₂])

# Plot against cumulative cost instead of observation number
fig = plot_convergence(result; truth = θ_true, x_axis = :cost)
```

Requires `record_posterior = true` in `run_adaptive`.

## Design allocation

Visualise where a batch design places its measurements:

```julia
# 1D design variable — stem plot
fig = plot_design_allocation(ξ, candidates)

# 2D design variables — bubble plot (auto-detected)
fig = plot_design_allocation(ξ, candidates)
```

## Gateaux derivative

Check optimality visually — the derivative should touch the bound at support points and lie below everywhere else:

```julia
# One-call from arguments
fig = plot_gateaux(prob, candidates, prior, ξ)

# Or from an OptimalityResult
opt = verify_optimality(prob, candidates, prior, ξ)
fig = plot_gateaux(opt)
```

## Residual diagnostics

After an adaptive experiment, plot observation residuals and cumulative log evidence:

```julia
fig = plot_residuals(result.log)
```

## Posterior evolution animation

Record an animation of the posterior evolving as observations arrive:

```julia
record_corner_animation(result.log, "posterior_evolution.gif";
    truth = θ_true, framerate = 5)
```

Requires `record_posterior = true` in `run_adaptive`.

## Backend notes

- Static figures use `CairoMakie` (vector graphics, good for papers)
- The live dashboard in `run_adaptive` uses `GLMakie` (interactive window)
- Both must be loaded in your environment (`using CairoMakie` or `using GLMakie`)
