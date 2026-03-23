# Example 5: Exponential Decay — Adaptive Design with Simulated Acquisition
#
# Uses Example 1's exponential decay model but runs a full adaptive experiment
# against a simulated ground truth, then compares adaptive vs batch posterior.
#
# Demonstrates:
#   1. Simulated acquisition function
#   2. run_adaptive for adaptive sequential design
#   3. Posterior convergence tracking
#   4. Observation diagnostics (log marginal likelihood, residuals)
#   5. Head-to-head: adaptive vs batch posterior precision
#   6. Design point trajectory (where the adaptive algorithm chooses to measure)


using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using LinearAlgebra
using Random
using GLMakie

# ENV["JULIA_DEBUG"] = OptimalDesign
Random.seed!(42)

# ═══════════════════════════════════════════════
# 1. The model and ground truth
# ═══════════════════════════════════════════════

function model(θ, x)
    θ.A * exp(-θ.R₂ * x.t)
end

# Ground truth (unknown to algorithm)
θ_true = ComponentArray(A=2, R₂=42.0)
σ_true = 0.1
budget = 100.0

# Simulated acquisition function (closure over ground truth)
acquire(x) = model(θ_true, x) + σ_true * randn()

println("Truth: A = $(θ_true.A), R₂ = $(θ_true.R₂)")

# ═══════════════════════════════════════════════
# 2. Design problem and prior
# ═══════════════════════════════════════════════

prob = DesignProblem(
    model,
    parameters=(A=LogUniform(0.1, 10), R₂=Uniform(1, 50)),
    transformation=select(:R₂),
    sigma=Returns(0.2),
    cost=x -> x.t + 1,
)
display(prob)

candidates = candidate_grid(t=range(0.001, 0.5, length=200))

# ═══════════════════════════════════════════════
# 3. Run adaptive experiment
# ═══════════════════════════════════════════════

println("Running adaptive experiment (budget=$budget)...")
prior_adaptive = Particles(prob, 1000)

result = run_adaptive(
    prob, candidates, prior_adaptive, acquire;
    budget=budget,
    n_per_step=1,
    headless=false,
    record_posterior=true,
)

display(result)

posterior_adaptive = result.posterior
log_adaptive = result.log
n_adaptive = length(log_adaptive)

# ═══════════════════════════════════════════════
# 4. Batch design for comparison (same n)
# ═══════════════════════════════════════════════

println("\n--- Batch design comparison (n=$n_adaptive) ---")
prior_batch = Particles(prob, 1000)

ξ_batch = design(prob, candidates, prior_batch; n=n_adaptive)

result_batch = run_batch(ξ_batch, prob, prior_batch, acquire)
display(result_batch)

# ═══════════════════════════════════════════════
# 5. Comparison summary
# ═══════════════════════════════════════════════

μ_adaptive = mean(result.posterior)
μ_batch = mean(result_batch.posterior)
err_adaptive = abs(μ_adaptive.R₂ - θ_true.R₂)
err_batch = abs(μ_batch.R₂ - θ_true.R₂)

println("\n=== Head-to-head comparison ===")
println("  Adaptive |R₂ error|: $(round(err_adaptive; digits=2))")
println("  Batch    |R₂ error|: $(round(err_batch; digits=2))")
println("  (Both use $n_adaptive measurements)")

# ═══════════════════════════════════════════════
# 6. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")

# --- Figure 1: Adaptive design trajectory ---

fig1 = Figure(size=(700, 500))

ax1a = GLMakie.Axis(fig1[1, 1], ylabel="Design time t",
    title="Adaptive Design Trajectory")
scatter!(ax1a, 1:n_adaptive, [e.x.t for e in log_adaptive],
    color=1:n_adaptive, colormap=:viridis, markersize=8)
lines!(ax1a, 1:n_adaptive, [e.x.t for e in log_adaptive],
    color=:gray, linewidth=0.5)

ax1b = GLMakie.Axis(fig1[2, 1], xlabel="Step", ylabel="Log marginal likelihood",
    title="Sequential Model Checking")
log_ml = log_evidence_series(log_adaptive)
lines!(ax1b, 1:n_adaptive, log_ml, color=:blue, linewidth=1.5)
scatter!(ax1b, 1:n_adaptive, log_ml, color=:blue, markersize=5)

fig1

# --- Figure 2: Adaptive vs Batch credible bands ---

fig2 = plot_credible_bands(prob, result, result_batch;
    labels=["Adaptive ($n_adaptive obs)", "Batch ($n_adaptive obs)"], truth=θ_true)

# --- Figure 3: Corner plot — adaptive vs batch posterior ---

fig3 = plot_corner(result_batch, result;
    labels=["Batch", "Adaptive"], truth=θ_true)


# --- Figure 4: Observation diagnostics from adaptive run ---

fig4 = plot_residuals(log_adaptive)

# --- Figure 5: ESS evolution ---

println("\nRerunning adaptive experiment to track ESS evolution...")
prior_ess = Particles(prob, 1000)
ess_history = Float64[]
r2_history = Float64[]

for entry in log_adaptive
    update!(prior_ess, prob, entry.x, entry.y)
    push!(ess_history, effective_sample_size(prior_ess))
    push!(r2_history, mean(prior_ess).R₂)
end

fig5 = Figure(size=(700, 500))

ax5a = GLMakie.Axis(fig5[1, 1], ylabel="R₂ estimate",
    title="Posterior Convergence")
lines!(ax5a, 1:n_adaptive, r2_history, color=:blue, linewidth=2)
hlines!(ax5a, [θ_true.R₂], color=:red, linestyle=:dash, label="Truth")
axislegend(ax5a)

ax5b = GLMakie.Axis(fig5[2, 1], xlabel="Step", ylabel="ESS",
    title="Effective Sample Size")
lines!(ax5b, 1:n_adaptive, ess_history, color=:blue, linewidth=2)
hlines!(ax5b, [100], color=:gray, linestyle=:dash, label="Warning threshold")
axislegend(ax5b)

fig5

# --- Figure 6: Animated corner plot ---

if has_posterior_history(log_adaptive)
    println("Recording posterior evolution animation...")
    record_corner_animation(log_adaptive, "ex5_posterior_evolution.gif";
        truth=θ_true, framerate=5)
end

println("Done. Figures created.")
