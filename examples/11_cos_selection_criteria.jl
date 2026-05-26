# Example 11: y = A cos(ω t) exp(−R t) — Comparing selection criteria (Ds-optimality)
#
# The damped cosine model encodes three distinct physical features:
#
#   A — amplitude   → dominates the signal at early times (large signal, no decay)
#   ω — frequency   → revealed by oscillations at intermediate times
#   R — decay rate  → visible in the signal envelope across a range of times
#
# Ds-optimality allows the design to focus on a chosen parameter subset while
# treating the others as nuisances. This produces strikingly different designs
# that match the physical intuition about where each parameter is best resolved.
#
# This example computes four batch designs with 20 measurements each:
#   1. D-optimal              — optimise log det(FIM) for all parameters jointly
#   2. Ds-optimal for A       — target amplitude; treat ω, R as nuisance
#   3. Ds-optimal for ω       — target frequency; treat A, R as nuisance
#   4. Ds-optimal for R       — target decay rate; treat A, ω as nuisance
#
# Expected result:
#   select(:A)  — concentrates at very early t (large signal, cos ≈ 1)
#   select(:ω)  — spreads across intermediate t (oscillations distinguishable)
#   select(:R)  — uses a wider time range (needs multiple decay amplitudes)
#   D-optimal   — some compromise across these regions

using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random

Random.seed!(42)

# ═══════════════════════════════════════════════════
# 1. Model and ground truth
# ═══════════════════════════════════════════════════

model(θ, x) = θ.A * cos(θ.ω * x.t) * exp(-θ.R * x.t)

θ_true = ComponentArray(A=50.0, ω=20.0, R=3.0)
σ_true = 5.0
acquire(x) = model(θ_true, x) + σ_true * randn()

println("Model: y = A cos(ω t) exp(−R t) + noise")
println("Truth: A = $(θ_true.A), ω = $(θ_true.ω), R = $(θ_true.R)")
println("Noise: σ = $σ_true\n")

# ═══════════════════════════════════════════════════
# 2. Prior and candidate grid
# ═══════════════════════════════════════════════════

prior_specs = (
    A = LogUniform(5.0, 500.0),
    ω = Uniform(3.0, 50.0),
    R = Uniform(0.2, 15.0),
)

candidates = candidate_grid(t=range(0.001, 0.7, length=300))
n_meas = 20
n_particles = 2000

# ═══════════════════════════════════════════════════
# 3. Four design problems — same model, different targets
# ═══════════════════════════════════════════════════

labels = ["D-optimal (all)", "Ds: A only", "Ds: ω only", "Ds: R only"]

problems = [
    DesignProblem(model; parameters=prior_specs, sigma=Returns(σ_true)),
    DesignProblem(model; parameters=prior_specs, sigma=Returns(σ_true), transformation=select(:A)),
    DesignProblem(model; parameters=prior_specs, sigma=Returns(σ_true), transformation=select(:ω)),
    DesignProblem(model; parameters=prior_specs, sigma=Returns(σ_true), transformation=select(:R)),
]

# ═══════════════════════════════════════════════════
# 4. Compute designs
# ═══════════════════════════════════════════════════

designs = []
for (label, prob) in zip(labels, problems)
    println("Computing design: $label")
    Random.seed!(42)
    prior = Particles(prob, n_particles)
    ξ = design(prob, candidates, prior; n=n_meas)
    push!(designs, ξ)
    display(ξ)
    println()
end

# ═══════════════════════════════════════════════════
# 5. Comparison figure — design allocations overlaid on the signal
# ═══════════════════════════════════════════════════

t_fine = range(0.0, 0.7, length=500)
y_true = [model(θ_true, (t=t,)) for t in t_fine]
t_cands = [c.t for c in candidates]

colors = [:steelblue, :darkorange, :seagreen, :orchid]

fig = Figure(size=(1100, 900))
Label(fig[0, 1:2], "Ds-optimal batch designs: y = A cos(ω t) exp(−R t)";
    fontsize=16, font=:bold)

for (i, (label, ξ, col)) in enumerate(zip(labels, designs, colors))
    row = 1 + (i - 1) ÷ 2
    col_idx = 1 + (i - 1) % 2

    ax = Makie.Axis(fig[row, col_idx];
        xlabel="t (s)", ylabel="measurement weight",
        title=label)

    w = weights(ξ, candidates)
    mask = w .> 0

    if any(mask)
        wmax = maximum(w[mask])

        # Background: scaled signal for context
        lines!(ax, collect(t_fine), y_true .* wmax ./ maximum(abs.(y_true));
            color=(:gray, 0.3), linewidth=1.5)

        # Design allocation — stems
        for (xi, wi) in zip(t_cands[mask], w[mask])
            lines!(ax, [xi, xi], [0.0, wi]; color=col, linewidth=2.5)
        end
        scatter!(ax, t_cands[mask], w[mask]; color=col, markersize=10)
    end

    ylims!(ax, 0, nothing)
end

display(fig)
save("ex11_selection_criteria.png", fig)

# ═══════════════════════════════════════════════════
# 6. Gateaux optimality verification
# ═══════════════════════════════════════════════════

println("\nGateaux checks (max derivative should equal q at support points):")

fig_g = Figure(size=(1100, 900))
Label(fig_g[0, 1:2], "Gateaux derivatives — optimality check";
    fontsize=16, font=:bold)

for (i, (label, prob, ξ, col)) in enumerate(zip(labels, problems, designs, colors))
    row = 1 + (i - 1) ÷ 2
    col_idx = 1 + (i - 1) % 2

    Random.seed!(42)
    prior = Particles(prob, n_particles)
    result = verify_optimality(prob, candidates, prior, ξ)

    status = result.is_optimal ? "✓" : "✗"
    println("  $label: max = $(round(result.max_derivative; digits=3)), " *
            "q = $(result.q)  [$status]")

    ax = Makie.Axis(fig_g[row, col_idx];
        xlabel="t (s)", ylabel="Gateaux derivative",
        title=label)
    lines!(ax, t_cands, result.gateaux; color=col, linewidth=1.5)
    hlines!(ax, [result.q]; color=:red, linestyle=:dash, linewidth=1.5,
        label="bound q = $(result.q)")
    axislegend(ax; position=:rt)
end

display(fig_g)
save("ex11_gateaux.png", fig_g)

# ═══════════════════════════════════════════════════
# 7. Run experiments and compare posteriors
# ═══════════════════════════════════════════════════

println("\nRunning experiments...")
results = []
for (label, prob, ξ) in zip(labels, problems, designs)
    Random.seed!(42)
    prior = Particles(prob, n_particles)
    r = run_batch(ξ, prob, prior, acquire)
    push!(results, r)
    μ = mean(r); s = std(r)
    println("  $label:")
    println("    A = $(round(μ.A; digits=2)) ± $(round(s.A; digits=2))  (truth: $(θ_true.A))")
    println("    ω = $(round(μ.ω; digits=2)) ± $(round(s.ω; digits=2))  (truth: $(θ_true.ω))")
    println("    R = $(round(μ.R; digits=2)) ± $(round(s.R; digits=2))  (truth: $(θ_true.R))")
    println()
end

# Posterior corner plot — all four designs
fig_c = plot_corner(
    [r.posterior for r in results]...;
    labels=labels,
    truth=θ_true,
)
display(fig_c)
save("ex11_posteriors.png", fig_c)

println("Done.")
