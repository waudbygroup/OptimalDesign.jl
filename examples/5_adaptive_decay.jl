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
    θ.A * exp(-θ.k * x.t)
end

# Ground truth (unknown to algorithm)
θ_true = ComponentArray(A=2, k=42.0)
σ_true = 0.1
budget = 100.0

acquire(x) = model(θ_true, x) + σ_true * randn()

# ═══════════════════════════════════════════════
# 2. Design problem and prior
# ═══════════════════════════════════════════════

prob = DesignProblem(
    model,
    parameters=(A=LogUniform(0.1, 10), k=Uniform(1, 50)),
    transformation=select(:k),
    sigma=Returns(σ_true),
    cost=x -> x.t + 1,
)
display(prob)

candidates = candidate_grid(t=range(0.001, 0.5, length=200))

# ═══════════════════════════════════════════════
# 3. Run adaptive experiment
# ═══════════════════════════════════════════════

prior = Particles(prob, 1000)

result = run_adaptive(
    prob, candidates, prior, acquire;
    budget=budget,
)
display(result)

log_adaptive = result.log
n_adaptive = length(log_adaptive)

# ═══════════════════════════════════════════════
# 4. Batch design for comparison (same budget)
# ═══════════════════════════════════════════════

result_batch = run_batch(prob, candidates, prior, acquire;
    budget=budget)
display(result_batch)

# ═══════════════════════════════════════════════
# 5. Comparison summary
# ═══════════════════════════════════════════════

μ_adaptive = mean(result)
μ_batch = mean(result_batch)
err_adaptive = abs(μ_adaptive.k - θ_true.k)
err_batch = abs(μ_batch.k - θ_true.k)

println("\n=== Head-to-head comparison ===")
println("  Adaptive |k error|: $(round(err_adaptive; digits=2))")
println("  Batch    |k error|: $(round(err_batch; digits=2))")

# ═══════════════════════════════════════════════
# 6. Plots
# ═══════════════════════════════════════════════

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
    labels=["Adaptive", "Batch"], truth=θ_true)

# --- Figure 3: Corner plot — adaptive vs batch posterior ---

fig3 = plot_corner(result_batch, result;
    labels=["Batch", "Adaptive"], truth=θ_true)

# --- Figure 4: Observation diagnostics from adaptive run ---

fig4 = plot_residuals(log_adaptive)

# --- Figure 5: Convergence ---

fig5 = plot_convergence(result; truth=θ_true)

# --- Figure 6: Animated corner plot ---

if has_posterior_history(log_adaptive)
    record_corner_animation(log_adaptive, "ex5_posterior_evolution.gif";
        truth=θ_true, framerate=5)
end

# --- Figure 7: Dashboard replay animation ---

record_dashboard(result, prob; filename="ex5_dashboard.gif")

println("Done. Figures created.")
