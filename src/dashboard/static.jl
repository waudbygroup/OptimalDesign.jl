"""
    posterior_predictions(prob, posterior, ξ_grid; n_samples=200)

Generate posterior predictions over a grid of design points.
Returns a matrix of size (n_samples, length(ξ_grid)).
"""
function posterior_predictions(
    prob::DesignProblem,
    posterior::ParticlePosterior,
    ξ_grid::AbstractVector;
    n_samples::Int=200,
)
    particles = sample(posterior, n_samples)
    predictions = Matrix{Float64}(undef, n_samples, length(ξ_grid))

    for (j, ξ) in enumerate(ξ_grid)
        for i in 1:n_samples
            y = prob.predict(particles[i], ξ)
            predictions[i, j] = y isa Real ? y : first(y)
        end
    end

    predictions
end

"""
    credible_band(predictions; level=0.9)

Compute credible band from posterior predictions.
Returns `(lower, median, upper)` vectors.
"""
function credible_band(predictions::AbstractMatrix; level::Real=0.9)
    α = (1 - level) / 2
    n_grid = size(predictions, 2)

    lower = Vector{Float64}(undef, n_grid)
    med = Vector{Float64}(undef, n_grid)
    upper = Vector{Float64}(undef, n_grid)

    for j in 1:n_grid
        col = sort(predictions[:, j])
        lower[j] = quantile(col, α)
        med[j] = quantile(col, 0.5)
        upper[j] = quantile(col, 1 - α)
    end

    (lower=lower, median=med, upper=upper)
end

# --- CairoMakie plotting functions ---
# These use CairoMakie for publication-quality static plots.

"""
    plot_credible_bands(prob, posterior, ξ_grid; level=0.9, observations=nothing)

Plot posterior credible bands with optional observation overlay.
"""
function plot_credible_bands(
    prob::DesignProblem,
    posterior::ParticlePosterior,
    ξ_grid::AbstractVector;
    level::Real=0.9,
    observations=nothing,
    x_field::Symbol=first(keys(first(ξ_grid))),
    n_samples::Int=200,
)
    preds = posterior_predictions(prob, posterior, ξ_grid; n_samples=n_samples)
    band = credible_band(preds; level=level)

    x_vals = [getfield(ξ, x_field) for ξ in ξ_grid]

    fig = CairoMakie.Figure(size=(600, 400))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(x_field),
        ylabel="Prediction",
        title="Posterior Credible Band ($(round(Int, level*100))%)")

    CairoMakie.band!(ax, x_vals, band.lower, band.upper, color=(:blue, 0.2))
    CairoMakie.lines!(ax, x_vals, band.median, color=:blue, linewidth=2)

    if observations !== nothing
        obs_x = [getfield(o.ξ, x_field) for o in observations]
        obs_y = [o.y isa NamedTuple ? o.y.value : o.y for o in observations]
        CairoMakie.scatter!(ax, obs_x, obs_y, color=:red, markersize=8)
    end

    fig
end

"""
    plot_design_allocation(candidates, weights; x_field)

Plot the weight distribution across candidates for a batch design.
"""
function plot_design_allocation(
    candidates::AbstractVector{<:NamedTuple},
    weights::AbstractVector;
    x_field::Symbol=first(keys(first(candidates))),
)
    x_vals = [getfield(ξ, x_field) for ξ in candidates]

    fig = CairoMakie.Figure(size=(600, 300))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(x_field),
        ylabel="Weight",
        title="Design Allocation")

    CairoMakie.stem!(ax, x_vals, weights, color=:blue)

    fig
end

"""
    plot_gateaux(candidates, gd, weights; x_field)

Plot the Gateaux derivative at each candidate, with the optimality bound.
"""
function plot_gateaux(
    candidates::AbstractVector{<:NamedTuple},
    gd::AbstractVector,
    p::Int;
    x_field::Symbol=first(keys(first(candidates))),
)
    x_vals = [getfield(ξ, x_field) for ξ in candidates]

    fig = CairoMakie.Figure(size=(600, 300))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(x_field),
        ylabel="Gateaux derivative",
        title="Optimality Check")

    CairoMakie.lines!(ax, x_vals, gd, color=:blue, linewidth=1.5)
    CairoMakie.hlines!(ax, [p], color=:red, linestyle=:dash, label="p = $p")
    CairoMakie.axislegend(ax)

    fig
end

"""
    plot_residuals(log::ExperimentLog)

Plot residual diagnostics from an experiment log.
"""
function plot_residuals(log::ExperimentLog)
    n = length(log)
    steps = 1:n
    residuals = [h.diagnostics.mean_residual for h in log]
    log_evidence = log_evidence_series(log)

    # Handle scalar vs vector residuals
    resid_scalar = [r isa Real ? r : norm(r) for r in residuals]

    fig = CairoMakie.Figure(size=(600, 500))

    ax1 = CairoMakie.Axis(fig[1, 1],
        ylabel="Mean residual",
        title="Observation Diagnostics")
    CairoMakie.scatter!(ax1, steps, resid_scalar, color=:blue, markersize=6)
    CairoMakie.hlines!(ax1, [0], color=:gray, linestyle=:dash)

    ax2 = CairoMakie.Axis(fig[2, 1],
        xlabel="Step",
        ylabel="Log marginal likelihood")
    CairoMakie.lines!(ax2, steps, log_evidence, color=:blue, linewidth=1.5)
    CairoMakie.scatter!(ax2, steps, log_evidence, color=:blue, markersize=5)

    fig
end

"""
    plot_posterior_marginals(posterior; params=nothing)

Plot histogram of posterior marginals for each parameter.
Kept for backward compatibility — see `plot_corner` for the full corner plot.
"""
function plot_posterior_marginals(
    posterior::ParticlePosterior;
    params::Union{Nothing,Vector{Symbol}}=nothing,
)
    plot_corner(posterior; params=params)
end

"""
    plot_corner(posteriors...; params, truth, labels, colors, bins, level)

Corner plot (pair plot) showing 1D marginal histograms on the diagonal
and 2D weighted scatter plots on the lower triangle.

Accepts one or more `ParticlePosterior` objects for overlay comparison
(e.g. prior vs posterior, or optimal vs uniform).

# Keyword arguments
- `params::Vector{Symbol}`: which parameters to show (default: all)
- `truth::Union{Nothing, NamedTuple, AbstractVector}`: true parameter values to mark
- `labels::Vector{String}`: legend labels for each posterior (default: "1", "2", …)
- `colors`: color per posterior (default: blue, orange, green, …)
- `bins::Int = 30`: histogram bin count
- `level::Real = 0.9`: credible interval level for 1D marginals
"""
function plot_corner(
    posteriors::ParticlePosterior...;
    params::Union{Nothing,Vector{Symbol}}=nothing,
    truth=nothing,
    labels::Union{Nothing,Vector{String}}=nothing,
    colors=nothing,
    bins::Int=30,
    level::Real=0.9,
)
    n_dist = length(posteriors)
    n_dist >= 1 || error("At least one posterior required")

    # Default parameter names from first posterior
    θ1 = first(first(posteriors).particles)
    names = params !== nothing ? params : collect(keys(θ1))
    d = length(names)

    # Default colours and labels
    default_colors = [(:royalblue, 0.6), (:orange, 0.6), (:green, 0.6), (:purple, 0.6)]
    cs = colors !== nothing ? colors : default_colors[1:min(n_dist, 4)]
    ls = labels !== nothing ? labels : [string(i) for i in 1:n_dist]

    # Extract weighted samples for each posterior
    all_vals = Vector{Vector{Vector{Float64}}}(undef, n_dist)   # [dist][param][particle]
    all_w = Vector{Vector{Float64}}(undef, n_dist)
    for (di, post) in enumerate(posteriors)
        w = exp.(post.log_weights .- logsumexp(post.log_weights))
        all_w[di] = w
        vals = [Float64[getproperty(p, name) for p in post.particles] for name in names]
        all_vals[di] = vals
    end

    # Truth values
    truth_vals = if truth !== nothing
        Float64[getproperty(truth, name) for name in names]
    else
        nothing
    end

    fig = CairoMakie.Figure(size=(250 * d, 250 * d))

    # Store axes for linking
    axes_grid = Matrix{Any}(nothing, d, d)

    for i in 1:d
        for j in 1:d
            if j > i
                # Upper triangle: empty
                continue
            end

            # Axis labels: only on edges
            xlabel = i == d ? string(names[j]) : ""
            ylabel = j == 1 && i > 1 ? string(names[i]) : (j == 1 && i == 1 ? "Density" : "")

            if i == j
                # ── Diagonal: 1D marginal histogram ──
                ax = CairoMakie.Axis(fig[i, j]; xlabel, ylabel,
                    title=i == 1 ? string(names[i]) : "")

                for di in 1:n_dist
                    vals = all_vals[di][i]
                    w = all_w[di]
                    CairoMakie.hist!(ax, vals; weights=w, bins, color=cs[di],
                        label=n_dist > 1 ? ls[di] : nothing,
                        normalization=:pdf)
                end

                if truth_vals !== nothing
                    CairoMakie.vlines!(ax, [truth_vals[i]]; color=:red,
                        linewidth=2, linestyle=:dash)
                end

                if i == 1 && n_dist > 1
                    CairoMakie.axislegend(ax; position=:rt, framevisible=false)
                end

                # Hide y-axis ticks on diagonal except first
                if j > 1
                    CairoMakie.hideydecorations!(ax; grid=false)
                end

            else
                # ── Lower triangle: 2D scatter ──
                ax = CairoMakie.Axis(fig[i, j]; xlabel, ylabel)

                for di in 1:n_dist
                    xv = all_vals[di][j]
                    yv = all_vals[di][i]
                    w = all_w[di]

                    # Resample for visual clarity (weighted scatter)
                    n_draw = min(500, length(xv))
                    idx = _weighted_sample_indices(w, n_draw)

                    base_color = cs[di] isa Tuple ? cs[di][1] : cs[di]
                    CairoMakie.scatter!(ax, xv[idx], yv[idx];
                        color=(base_color, 0.2), markersize=5,
                        label=n_dist > 1 ? ls[di] : nothing)
                end

                if truth_vals !== nothing
                    CairoMakie.vlines!(ax, [truth_vals[j]]; color=:red,
                        linewidth=1.5, linestyle=:dash)
                    CairoMakie.hlines!(ax, [truth_vals[i]]; color=:red,
                        linewidth=1.5, linestyle=:dash)
                end

                # Hide interior tick labels
                if i < d
                    CairoMakie.hidexdecorations!(ax; grid=false)
                end
                if j > 1
                    CairoMakie.hideydecorations!(ax; grid=false)
                end
            end

            axes_grid[i, j] = ax
        end
    end

    # Link axes: same column shares x-limits, same row shares y-limits (for off-diag)
    for j in 1:d
        col_axes = [axes_grid[i, j] for i in j:d if axes_grid[i, j] !== nothing]
        length(col_axes) > 1 && CairoMakie.linkxaxes!(col_axes...)
    end
    for i in 2:d
        row_axes = [axes_grid[i, j] for j in 1:i-1 if axes_grid[i, j] !== nothing]
        length(row_axes) > 1 && CairoMakie.linkyaxes!(row_axes...)
    end

    fig
end

"""Systematic resampling of n indices proportional to weights (for plotting)."""
function _weighted_sample_indices(w::AbstractVector, n::Int)
    cumw = cumsum(w)
    u = rand() / n
    indices = Vector{Int}(undef, n)
    j = 1
    for i in 1:n
        target = u + (i - 1) / n
        while j < length(cumw) && cumw[j] < target
            j += 1
        end
        indices[i] = j
    end
    indices
end
