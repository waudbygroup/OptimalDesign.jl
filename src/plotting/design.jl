"""
    plot_design_allocation(candidates, weights; ...)

Plot the weight distribution across candidates for a batch design.
Auto-detects 1D (stem plot) vs 2D (scatter/bubble plot) design variables.
"""
function plot_design_allocation(
    candidates::AbstractVector{<:NamedTuple},
    w::AbstractVector;
    fields::Union{Nothing,NTuple{N,Symbol} where N}=nothing,
)
    ks = keys(first(candidates))
    fs = fields !== nothing ? fields : Tuple(ks)
    ndim = length(fs)

    if ndim == 1
        _plot_design_1d(candidates, w, fs[1])
    elseif ndim == 2
        _plot_design_2d(candidates, w, fs[1], fs[2])
    else
        error("plot_design_allocation supports 1 or 2 design variables, got $ndim")
    end
end

function _plot_design_1d(candidates, w, xf::Symbol)
    x_vals = [getfield(x, xf) for x in candidates]

    fig = CairoMakie.Figure(size=(600, 300))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(xf), ylabel="Weight",
        title="Design Allocation")
    CairoMakie.stem!(ax, x_vals, w, color=:blue)

    fig
end

function _plot_design_2d(candidates, w, xf::Symbol, yf::Symbol)
    x_vals = [getfield(x, xf) for x in candidates]
    y_vals = [getfield(x, yf) for x in candidates]

    fig = CairoMakie.Figure(size=(600, 500))

    # Support points (non-zero weight) as scaled bubbles
    mask = w .> 0
    if any(mask)
        ax = CairoMakie.Axis(fig[1, 1],
            xlabel=string(xf), ylabel=string(yf),
            title="Design Allocation")

        # Bubble area proportional to weight
        w_nz = w[mask]
        ms = 5 .+ 40 .* (w_nz ./ maximum(w_nz))

        CairoMakie.scatter!(ax, x_vals[mask], y_vals[mask];
            markersize=ms, color=w_nz, colormap=:viridis,
            colorrange=(0, maximum(w_nz)))
        CairoMakie.Colorbar(fig[1, 2]; colormap=:viridis,
            colorrange=(0, maximum(w_nz)), label="Weight")

        # Label counts
        for i in findall(mask)
            CairoMakie.text!(ax, x_vals[i], y_vals[i];
                text=string(round(w[i]; digits=3)),
                fontsize=9, align=(:center, :bottom), offset=(0, 5))
        end
    end

    fig
end

"""
    plot_design_allocation(ξ::ExperimentalDesign, candidates; ...)

Plot the weight distribution for an `ExperimentalDesign`.
"""
function plot_design_allocation(
    ξ::ExperimentalDesign,
    candidates::AbstractVector{<:NamedTuple};
    kwargs...,
)
    plot_design_allocation(candidates, weights(ξ, candidates); kwargs...)
end

"""
    plot_gateaux(candidates, gd, p; ...)

Plot the Gateaux derivative at each candidate, with the optimality bound.
Auto-detects 1D (line plot) vs 2D (heatmap/scatter) design variables.
"""
function plot_gateaux(
    candidates::AbstractVector{<:NamedTuple},
    gd::AbstractVector,
    p;
    fields::Union{Nothing,NTuple{N,Symbol} where N}=nothing,
)
    ks = keys(first(candidates))
    fs = fields !== nothing ? fields : Tuple(ks)
    ndim = length(fs)

    if ndim == 1
        _plot_gateaux_1d(candidates, gd, p, fs[1])
    elseif ndim == 2
        _plot_gateaux_2d(candidates, gd, p, fs[1], fs[2])
    else
        error("plot_gateaux supports 1 or 2 design variables, got $ndim")
    end
end

function _plot_gateaux_1d(candidates, gd, p, xf::Symbol)
    x_vals = [getfield(x, xf) for x in candidates]

    fig = CairoMakie.Figure(size=(600, 300))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(xf), ylabel="Gateaux derivative",
        title="Optimality Check")
    CairoMakie.lines!(ax, x_vals, gd, color=:blue, linewidth=1.5)
    CairoMakie.hlines!(ax, [p], color=:red, linestyle=:dash, label="q = $p")
    CairoMakie.axislegend(ax)

    fig
end

function _plot_gateaux_2d(candidates, gd, p, xf::Symbol, yf::Symbol)
    x_vals = [getfield(x, xf) for x in candidates]
    y_vals = [getfield(x, yf) for x in candidates]

    crange = (min(minimum(gd), 0.0), max(maximum(gd), p * 1.1))

    fig = CairoMakie.Figure(size=(700, 500))
    ax = CairoMakie.Axis(fig[1, 1],
        xlabel=string(xf), ylabel=string(yf),
        title="Gateaux Derivative (bound q = $(round(p; digits=1)))")

    CairoMakie.scatter!(ax, x_vals, y_vals;
        color=gd, markersize=8,
        colorrange=crange)
    CairoMakie.Colorbar(fig[1, 2];
        colorrange=crange, label="Gateaux derivative")

    # Mark points exceeding the bound
    above = gd .> p
    if any(above)
        CairoMakie.scatter!(ax, x_vals[above], y_vals[above];
            color=:transparent, strokecolor=:red, strokewidth=2,
            markersize=12, marker=:circle, label="Above bound")
    end

    fig
end

"""
    plot_gateaux(r::OptimalityResult; kwargs...)

Plot the Gateaux derivative from a `verify_optimality` result.
"""
plot_gateaux(r::OptimalityResult; kwargs...) =
    plot_gateaux(r.candidates, r.gateaux, r.dimension; kwargs...)

"""
    plot_gateaux(prob, candidates, posterior, ξ; kwargs...)

Compute the Gateaux derivative and plot it in one call.
Equivalent to `plot_gateaux(verify_optimality(prob, candidates, posterior, ξ; kwargs...))`.
"""
function plot_gateaux(
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    posterior::Particles,
    ξ::ExperimentalDesign;
    kwargs...,
)
    opt = verify_optimality(prob, candidates, posterior, ξ; kwargs...)
    plot_gateaux(opt)
end

"""
    plot_convergence(log::ExperimentLog; truth=nothing, params=nothing, x_axis=:obs)

Plot the convergence of parameter estimates over the course of an adaptive experiment.
Shows the posterior mean ± 1 standard deviation at each step.

Requires `record_posterior = true` in `run_adaptive`.

# Keyword arguments
- `truth`: true parameter values to overlay as dashed horizontal lines
- `params::Vector{Symbol}`: which parameters to show (default: all)
- `x_axis::Symbol`: `:obs` for observation number, `:cost` for cumulative cost
"""
function plot_convergence(
    log::ExperimentLog;
    truth=nothing,
    params::Union{Nothing,Vector{Symbol}}=nothing,
    x_axis::Symbol=:obs,
)
    has_posterior_history(log) || error(
        "ExperimentLog has no posterior snapshots. " *
        "Use record_posterior=true in run_adaptive.")

    θ1 = first(log.prior_snapshot.particles)
    names = params !== nothing ? params : collect(keys(θ1))
    n_params = length(names)
    n_steps = length(log)

    # Compute mean and std at each step from snapshots
    means = [zeros(n_steps) for _ in 1:n_params]
    stds = [zeros(n_steps) for _ in 1:n_params]

    for s in 1:n_steps
        snap = log[s].posterior_snapshot
        w = exp.(snap.log_weights .- logsumexp(snap.log_weights))
        for (pi, name) in enumerate(names)
            vals = [getproperty(p, name) for p in snap.particles]
            μ = sum(w .* vals)
            σ = sqrt(sum(w .* (vals .- μ).^2))
            means[pi][s] = μ
            stds[pi][s] = σ
        end
    end

    # X axis: observation number or cumulative cost
    x_vals = if x_axis == :cost
        cumulative_cost(log)
    else
        collect(1:n_steps)
    end
    x_label = x_axis == :cost ? "Cumulative cost" : "Observation"

    default_colors = [:royalblue, :orange, :green, :purple, :red, :cyan]

    fig = CairoMakie.Figure(size=(700, 250 * n_params))

    for (pi, name) in enumerate(names)
        ax = CairoMakie.Axis(fig[pi, 1];
            xlabel=pi == n_params ? x_label : "",
            ylabel=string(name))

        c = default_colors[mod1(pi, length(default_colors))]

        CairoMakie.band!(ax, x_vals, means[pi] .- stds[pi], means[pi] .+ stds[pi];
            color=(c, 0.2))
        CairoMakie.lines!(ax, x_vals, means[pi]; color=c, linewidth=2)

        if truth !== nothing
            tv = getproperty(truth, name)
            CairoMakie.hlines!(ax, [tv]; color=:red, linestyle=:dash, linewidth=1.5)
        end

        if pi < n_params
            CairoMakie.hidexdecorations!(ax; grid=false)
        end
    end

    fig
end

"""
    plot_convergence(result::AdaptiveResult; kwargs...)

Plot convergence from an `AdaptiveResult`. Convenience wrapper.
"""
plot_convergence(result::AdaptiveResult; kwargs...) =
    plot_convergence(result.log; kwargs...)

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
