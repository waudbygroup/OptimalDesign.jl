# Live dashboard using GLMakie Observables for reactive updates.
# Panels: design trajectory, posterior marginals, log marginal likelihood,
# budget tracker, pause/resume/stop controls.

"""
    _interest_params(prob) → Vector{Symbol}

Extract the parameter names of interest from the design problem's transformation.
Falls back to all parameter names if no selection is available.
"""
function _interest_params(prob)
    t = prob.transformation
    if t isa DeltaMethod && t.selected !== nothing
        collect(t.selected)
    else
        collect(keys(prob.parameters))
    end
end

function _create_dashboard(prob, posterior, prediction_grid, budget)
    try
        interest = _interest_params(prob)
        n_interest = length(interest)

        fig = GLMakie.Figure(size=(1200, 800), figure_padding=20)

        # ── Row 1: Design trajectory + Posterior marginals ──

        # Design panel (left)
        ax_design = GLMakie.Axis(fig[1, 1], title="Design Trajectory")
        obs_design_x = GLMakie.Observable(Float64[])
        obs_design_y = GLMakie.Observable(Float64[])
        obs_design_color = GLMakie.Observable(Float64[])

        GLMakie.scatter!(ax_design, obs_design_x, obs_design_y;
            color=obs_design_color, colormap=:viridis, colorrange=(1, 2),
            markersize=8)

        # Posterior marginal histograms (right, stacked vertically)
        post_layout = fig[1, 2] = GLMakie.GridLayout()
        ax_posterior = Any[]
        for (i, name) in enumerate(interest)
            ax = GLMakie.Axis(post_layout[i, 1],
                # xlabel=i == n_interest ? "Value" : "",
                xlabel=string(name),
                ylabel="Density",
                # title=string(name),
                title=i == 1 ? "Posterior Marginals" : "",
            )
            # if i < n_interest
            #     GLMakie.hidexdecorations!(ax; grid=false)
            # end
            push!(ax_posterior, ax)
        end

        # ── Row 2: Log marginal likelihood + Budget ──

        ax_logml = GLMakie.Axis(fig[2, 1],
            title="Log Marginal Likelihood",
            xlabel="Step", ylabel="Log p(y)")

        obs_logml = GLMakie.Observable(Float64[])
        GLMakie.scatter!(ax_logml,
            GLMakie.@lift(collect(1:length($obs_logml))),
            obs_logml, color=:blue, markersize=5)
        GLMakie.lines!(ax_logml,
            GLMakie.@lift(collect(1:length($obs_logml))),
            obs_logml, color=:blue)

        # Budget bar
        ax_budget = GLMakie.Axis(fig[2, 2],
            title="Budget", xlabel="", ylabel="",
            limits=(nothing, (0, budget * 1.1)))
        GLMakie.hideydecorations!(ax_budget)
        GLMakie.hidexdecorations!(ax_budget)

        obs_budget_spent = GLMakie.Observable(0.0)
        GLMakie.barplot!(ax_budget, [1, 2],
            GLMakie.@lift([$obs_budget_spent, budget - $obs_budget_spent]),
            color=[:orange, :lightgray],
            bar_labels=GLMakie.@lift([
                "Spent: $(round($obs_budget_spent; digits=1))",
                "Left: $(round(budget - $obs_budget_spent; digits=1))"]))

        # ── Row 3: Controls ──

        control_state = Ref(:running)
        btn_layout = fig[3, 1:2] = GLMakie.GridLayout()
        btn_pause = GLMakie.Button(btn_layout[1, 1], label="Pause")
        btn_resume = GLMakie.Button(btn_layout[1, 2], label="Resume")
        btn_stop = GLMakie.Button(btn_layout[1, 3], label="Stop")

        GLMakie.on(btn_pause.clicks) do _
            control_state[] = :paused
        end
        GLMakie.on(btn_resume.clicks) do _
            control_state[] = :running
        end
        GLMakie.on(btn_stop.clicks) do _
            control_state[] = :stopped
        end

        screen = GLMakie.display(fig)

        LiveDashboard(
            fig, screen,
            ax_design, obs_design_x, obs_design_y, obs_design_color,
            (:_,),   # design_fields — detected on first update
            ax_posterior, interest,
            ax_logml, obs_logml,
            obs_budget_spent, Float64(budget),
            control_state,
        )
    catch e
        @warn "Could not create live dashboard" exception = (e, catch_backtrace())
        nothing
    end
end

function _check_controls(dashboard::LiveDashboard)
    dashboard.control_state[]
end

function _check_controls(::Nothing)
    :running
end

function _update_dashboard!(dashboard::LiveDashboard, prob, posterior, log, spent, budget, prediction_grid)
    isempty(log.history) && return

    last_entry = log.history[end]
    x = last_entry.x
    step = length(log.history)

    # ── Design trajectory ──
    fs = keys(x)
    ndim = length(fs)

    if ndim == 1
        # 1D: x-axis = design variable, y-axis = observation
        xval = Float64(x[fs[1]])
        yval = _scalar_obs(last_entry.y)
        if step == 1
            dashboard.ax_design.xlabel = string(fs[1])
            dashboard.ax_design.ylabel = "Observation"
            dashboard.design_fields = (fs[1],)
        end
        push!(dashboard.obs_design_x[], xval)
        push!(dashboard.obs_design_y[], yval)
        push!(dashboard.obs_design_color[], Float64(step))
        GLMakie.notify(dashboard.obs_design_x)
        GLMakie.notify(dashboard.obs_design_y)
        GLMakie.notify(dashboard.obs_design_color)
        plots = GLMakie.plots(dashboard.ax_design)
        if !isempty(plots)
            plots[1].colorrange = (1, max(Float64(step), 2))
        end
    else
        # 2D+: clear and redraw as bubble plot with visit counts
        if step == 1
            dashboard.ax_design.xlabel = string(fs[1])
            dashboard.ax_design.ylabel = string(fs[2])
            dashboard.design_fields = (fs[1], fs[2])
        end
        _update_design_2d!(dashboard, log, fs[1], fs[2])
    end
    GLMakie.autolimits!(dashboard.ax_design)

    # ── Log marginal likelihood ──
    push!(dashboard.obs_logml[], last_entry.diagnostics.log_marginal)
    GLMakie.notify(dashboard.obs_logml)
    GLMakie.autolimits!(dashboard.ax_logml)

    # ── Budget ──
    dashboard.obs_budget_spent[] = spent

    # ── Posterior marginal histograms ──
    # Update every 5 steps (histograms are expensive to redraw every step)
    if step % 5 == 0 || step <= 3
        _update_posterior_panel!(dashboard, posterior)
    end
end

"""Rebuild the 2D design scatter with marker sizes proportional to visit count."""
function _update_design_2d!(dashboard::LiveDashboard, log, f1::Symbol, f2::Symbol)
    # Count visits per unique (x1, x2) pair
    counts = Dict{Tuple{Float64,Float64},Int}()
    for entry in log.history
        key = (Float64(entry.x[f1]), Float64(entry.x[f2]))
        counts[key] = get(counts, key, 0) + 1
    end

    xs = Float64[k[1] for k in keys(counts)]
    ys = Float64[k[2] for k in keys(counts)]
    cs = Float64[v for v in values(counts)]
    max_c = maximum(cs)
    ms = 5.0 .+ 25.0 .* (cs ./ max_c)

    # Clear and redraw — can't mutate markersize from scalar to vector
    GLMakie.empty!(dashboard.ax_design)
    GLMakie.scatter!(dashboard.ax_design, xs, ys;
        color=cs, markersize=ms, colormap=:viridis,
        colorrange=(0, max(max_c, 1)))
end

"""Redraw posterior marginal histograms for interest parameters."""
function _update_posterior_panel!(dashboard::LiveDashboard, posterior)
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))

    for (i, name) in enumerate(dashboard.interest_params)
        ax = dashboard.ax_posterior[i]
        vals = [getproperty(p, name) for p in posterior.particles]

        # Weighted histogram via binning
        lo, hi = extrema(vals)
        if lo ≈ hi
            lo -= 1.0
            hi += 1.0
        end
        nbins = 30
        edges = range(lo, hi; length=nbins + 1)
        bin_counts = zeros(nbins)
        for (v, wt) in zip(vals, w)
            bin = clamp(searchsortedlast(edges, v), 1, nbins)
            bin_counts[bin] += wt
        end
        centres = [(edges[j] + edges[j+1]) / 2 for j in 1:nbins]

        # Clear and redraw
        GLMakie.empty!(ax)
        GLMakie.barplot!(ax, centres, bin_counts;
            width=(hi - lo) / nbins, color=(:royalblue, 0.6))
        ax.title = i == 1 ? "Posterior Marginals" : ""
    end
end

"""Extract a scalar from a possibly structured observation."""
_scalar_obs(y::Real) = Float64(y)
_scalar_obs(y::AbstractVector) = Float64(first(y))
_scalar_obs(y::NamedTuple) = Float64(first(values(y)))

function _update_dashboard!(::Nothing, args...)
    nothing
end

function _finalize_dashboard(dashboard::LiveDashboard)
    sleep(1)  # brief pause so user can see the final state
end

function _finalize_dashboard(::Nothing)
    nothing
end

# ── Dashboard GIF recording ──

"""
    record_dashboard(result, prob; kwargs...) → filename

Replay an adaptive experiment as a dashboard animation and save to a GIF or MP4.
This is useful for documentation — it produces the same dashboard layout as the
live `run_adaptive` display but rendered offline via CairoMakie.

# Keyword arguments
- `filename = "dashboard.gif"`: output path (extension determines format)
- `duration = 5.0`: target duration in seconds
- `size = (1200, 800)`: figure size in pixels
"""
function record_dashboard(
    result::AdaptiveResult,
    prob::AbstractDesignProblem;
    filename::String="dashboard.gif",
    duration::Float64=5.0,
    size::Tuple{Int,Int}=(1200, 800),
)
    log = result.log
    n = length(log)
    n == 0 && error("Empty experiment log")

    interest = _interest_params(prob)
    n_interest = length(interest)
    budget = sum(e.cost for e in log)

    # Determine frame sampling: target duration at a reasonable framerate
    max_fps = 30
    min_fps = 2
    fps = clamp(round(Int, n / duration), min_fps, max_fps)
    step_interval = max(1, round(Int, n / (duration * fps)))
    frame_steps = collect(1:step_interval:n)
    if last(frame_steps) != n
        push!(frame_steps, n)
    end
    actual_fps = clamp(round(Int, length(frame_steps) / duration), min_fps, max_fps)

    # Detect design dimensionality
    fs = keys(first(log).x)
    ndim = length(fs)

    fig = CairoMakie.Figure(; size, figure_padding=20)

    # ── Row 1: Design trajectory + Posterior ──
    ax_design = CairoMakie.Axis(fig[1, 1], title="Design Trajectory")

    post_layout = fig[1, 2] = CairoMakie.GridLayout()
    ax_post = CairoMakie.Axis[]
    for (i, name) in enumerate(interest)
        ax = CairoMakie.Axis(post_layout[i, 1],
            xlabel=string(name),
            ylabel="Density",
            title=i == 1 ? "Posterior Marginals" : "")
        # if i < n_interest
        #     CairoMakie.hidexdecorations!(ax; grid=false)
        # end
        push!(ax_post, ax)
    end

    # ── Row 2: Log ML + Budget ──
    ax_logml = CairoMakie.Axis(fig[2, 1],
        title="Log Marginal Likelihood",
        xlabel="Step", ylabel="Log p(y)")
    ax_budget = CairoMakie.Axis(fig[2, 2], title="Budget",
        limits=(nothing, (0, budget * 1.1)))
    CairoMakie.hideydecorations!(ax_budget)
    CairoMakie.hidexdecorations!(ax_budget)

    # Step counter label
    step_label = CairoMakie.Label(fig[0, 1:2],
        "Step 0 / $n", fontsize=16)

    @info "Recording dashboard animation: $(length(frame_steps)) frames → $filename ($(actual_fps) fps)"

    CairoMakie.record(fig, filename, frame_steps; framerate=actual_fps) do step_idx
        entries = log.history[1:step_idx]
        spent = sum(e.cost for e in entries)

        step_label.text[] = "Step $step_idx / $n"

        # ── Design trajectory ──
        CairoMakie.empty!(ax_design)
        if ndim == 1
            ax_design.xlabel = string(fs[1])
            ax_design.ylabel = "Observation"
            xs = Float64[e.x[fs[1]] for e in entries]
            ys = Float64[_scalar_obs(e.y) for e in entries]
            CairoMakie.scatter!(ax_design, xs, ys;
                color=1:step_idx, colormap=:viridis,
                colorrange=(1, max(n, 2)), markersize=8)
        else
            ax_design.xlabel = string(fs[1])
            ax_design.ylabel = string(fs[2])
            counts = Dict{Tuple{Float64,Float64},Int}()
            for e in entries
                key = (Float64(e.x[fs[1]]), Float64(e.x[fs[2]]))
                counts[key] = get(counts, key, 0) + 1
            end
            px = Float64[k[1] for k in keys(counts)]
            py = Float64[k[2] for k in keys(counts)]
            cv = Float64[v for v in values(counts)]
            max_c = maximum(cv; init=1)
            ms = 5.0 .+ 25.0 .* (cv ./ max_c)
            CairoMakie.scatter!(ax_design, px, py;
                color=cv, markersize=ms, colormap=:viridis,
                colorrange=(0, max(max_c, 1)))
        end

        # ── Posterior histograms ──
        if has_posterior_history(log)
            snap = log[step_idx].posterior_snapshot
            w = exp.(snap.log_weights .- logsumexp(snap.log_weights))
            for (i, name) in enumerate(interest)
                CairoMakie.empty!(ax_post[i])
                vals = [getproperty(p, name) for p in snap.particles]
                lo, hi = extrema(vals)
                if lo ≈ hi
                    lo -= 1.0
                    hi += 1.0
                end
                nbins = 30
                edges = range(lo, hi; length=nbins + 1)
                bin_counts = zeros(nbins)
                for (v, wt) in zip(vals, w)
                    bin = clamp(searchsortedlast(edges, v), 1, nbins)
                    bin_counts[bin] += wt
                end
                centres = [(edges[j] + edges[j+1]) / 2 for j in 1:nbins]
                CairoMakie.barplot!(ax_post[i], centres, bin_counts;
                    width=(hi - lo) / nbins, color=(:royalblue, 0.6))
                ax_post[i].title = i == 1 ? "Posterior Marginals" : ""
            end
        end

        # ── Log ML ──
        CairoMakie.empty!(ax_logml)
        ml = Float64[e.diagnostics.log_marginal for e in entries]
        CairoMakie.lines!(ax_logml, 1:step_idx, ml, color=:blue)
        CairoMakie.scatter!(ax_logml, 1:step_idx, ml,
            color=:blue, markersize=5)

        # ── Budget bar ──
        CairoMakie.empty!(ax_budget)
        CairoMakie.barplot!(ax_budget, [1, 2],
            [spent, budget - spent],
            color=[:orange, :lightgray],
            bar_labels=[
                "Spent: $(round(spent; digits=1))",
                "Left: $(round(budget - spent; digits=1))"])
    end

    @info "Dashboard animation saved: $filename"
    filename
end
