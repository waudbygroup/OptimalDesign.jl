# Plotting

OptimalDesign.jl provides plotting functions built on [Makie.jl](https://docs.makie.org/). Most are not exported — call them as `OptimalDesign.plot_...` or bring them into scope explicitly.

## Corner plots (exported)

Compare one or more posteriors as marginal histograms and pairwise scatter plots:

```julia
fig = plot_corner(prior, posterior;
    params = [:A, :R₂],
    labels = ["Prior", "Posterior"],
    truth = (A = 1.0, R₂ = 25.0))
```

- `params` selects which parameters to show (useful for high-dimensional models)
- `labels` names each posterior in the legend
- `truth` overlays dashed red crosshairs at known parameter values
- Pass any number of `Particles` arguments to compare them

## Credible bands

Plot model predictions with uncertainty bands:

```julia
# Single posterior
fig = OptimalDesign.plot_credible_bands(prob, posterior, grid)

# Compare several posteriors (stacked panels)
fig = OptimalDesign.plot_credible_bands(prob,
    [prior, posterior_opt, posterior_unif], grid;
    labels = ["Prior", "Optimal", "Uniform"],
    truth = θ_true,
    observations = [nothing, result_opt.observations, result_unif.observations])
```

For vector-valued models, components are displayed as separate columns automatically.

## Design allocation

Visualise where a batch design places its measurements:

```julia
# 1D design variable — stem plot
fig = OptimalDesign.plot_design_allocation(d, candidates)

# 2D design variables — bubble plot (auto-detected)
fig = OptimalDesign.plot_design_allocation(d, candidates)
```

## Gateaux derivative

Check optimality via the General Equivalence Theorem:

```julia
gd = OptimalDesign.gateaux_derivative(prob, candidates, prior, d)
opt = OptimalDesign.verify_optimality(prob, candidates, prior, d)

# 1D — line plot
fig = OptimalDesign.plot_gateaux(candidates, gd, opt.dimension)

# 2D — scatter plot with colour (auto-detected)
fig = OptimalDesign.plot_gateaux(candidates, gd, opt.dimension)
```

## Residual diagnostics (exported)

After an adaptive experiment, plot observation residuals and cumulative log evidence:

```julia
fig = plot_residuals(result.log)
```

## Posterior evolution animation (exported)

Record an animation of the posterior evolving as observations arrive:

```julia
record_corner_animation(result.log, "posterior_evolution.mp4";
    params = [:A, :R₂],
    truth = (A = 1.0, R₂ = 25.0),
    framerate = 5)
```

Requires `record_posterior = true` in `run_adaptive`.

## Backend notes

- Static figures use `CairoMakie` (vector graphics, good for papers)
- The live dashboard in `run_adaptive` uses `GLMakie` (interactive window)
- Both must be loaded in your environment (`using CairoMakie` or `using GLMakie`)
