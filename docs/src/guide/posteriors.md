# Posterior Inference

OptimalDesign.jl uses a particle filter to represent parameter uncertainty. A `Particles` is a weighted set of parameter samples — initially drawn from the prior, then reweighted as data arrives.

## Creating a prior

```julia
prior = Particles(prob, 1000)
```

This draws 1000 particles from the prior distributions specified in `prob.parameters`. All particles start with equal weight.

More particles give better coverage of the prior but cost more in design computation.

## Updating with data

After acquiring observations, the posterior is updated via `run_batch` or `run_adaptive`. These functions do not mutate the prior — they make an internal copy and return a result containing both the original `prior` and the updated `posterior`. Under the hood, they call `update!`, which:

1. Computes the likelihood of each observation under each particle
2. Uses **likelihood tempering** to gradually increase the data weight, avoiding particle collapse
3. Resamples and applies a Liu--West kernel when the effective sample size drops too low

You don't need to call `update!` directly — `run_batch` and `run_adaptive` handle it.

## Extracting estimates

### Point estimate

```julia
μ = mean(result.posterior)
# μ.A ≈ 1.0, μ.R₂ ≈ 25.0

# Or directly on the result:
μ = mean(result)
```

Returns a `ComponentArray` with the weighted mean of each parameter.

### Effective sample size

```julia
ess = effective_sample_size(result.posterior)
```

The ESS measures how many "independent" particles the posterior effectively contains. If ESS drops very low (e.g., below 50), the posterior may be poorly represented. The package automatically resamples when this happens during updates.

## Predictions and credible bands

To propagate posterior uncertainty through the model:

```julia
# Predict at a grid of design points
grid = candidate_grid(t = range(0, 0.5, length = 100))
preds = posterior_predictions(prob, result.posterior, grid; n_samples = 200)
```

`preds` is a matrix (rows = grid points, columns = posterior samples) for scalar models, or a vector of matrices for vector models.

To compute quantile-based credible bands:

```julia
band = credible_band(preds; level = 0.9)
# band.lower, band.median, band.upper — vectors over the grid
```

These can be plotted directly or passed to `plot_credible_bands` (see [Plotting](@ref)).

## Resampling strategies

Particle filters lose diversity over time as a few particles accumulate most of the weight. OptimalDesign.jl triggers a **resample** step whenever the effective sample size (ESS) drops below a configurable fraction of the particle count. Three strategies are available:

### `LiuWestResampling` (default)

Systematic resampling followed by a small kernel jitter in unconstrained parameter space, preserving bounded priors:

```julia
prior = Particles(prob, 2000)  # LiuWestResampling(a=0.98, ess_threshold=0.5) by default
prior = Particles(prob, 2000; resampling = LiuWestResampling(a = 0.95, ess_threshold = 0.4))
```

The shrinkage coefficient `a` (0 < a < 1) controls the jitter magnitude: ``h^2 = 1 - a^2``. Larger `a` → smaller perturbation → better moment preservation but less diversity recovery.

### `SystematicResampling`

Plain systematic resampling with no jitter — the fastest option:

```julia
prior = Particles(prob, 2000; resampling = SystematicResampling(ess_threshold = 0.5))
```

### `GMMResampling`

Fits a Gaussian Mixture Model (BIC-guided number of components) to the current particle cloud and draws a fresh sample from it. Handles multi-modal posteriors that the other strategies collapse:

```julia
prior = Particles(prob, 2000; resampling = GMMResampling(ess_threshold = 0.8, k_max = 8))
```

Optional diagnostic figures are written per resample event when `log_dir` is set:

```julia
prior = Particles(prob, 2000;
    resampling = GMMResampling(k_max = 6, log_dir = "gmm_diagnostics/"))
```

See the [Resampling Strategies](@ref) example page for a worked comparison on a challenging multi-modal problem.

## How likelihood tempering works

When many observations arrive at once (as in `run_batch`), a naive likelihood update would concentrate all weight on a handful of particles, effectively collapsing the posterior. OptimalDesign avoids this with **adaptive tempering**:

1. The full likelihood ``L(\theta)`` is raised to a power ``\beta``, starting at ``\beta = 0``
2. ``\beta`` is increased in steps, each time choosing the largest ``\Delta\beta`` that keeps the ESS above a threshold (default: 50% of particle count)
3. Between steps, particles are resampled and jittered using a Liu--West kernel to maintain diversity
4. The process continues until ``\beta = 1``, at which point the full likelihood has been absorbed

This is equivalent to sequential Monte Carlo (SMC) with an adaptive tempering schedule. It works for both batch and single-observation updates — even a single very informative observation can benefit from tempering.
