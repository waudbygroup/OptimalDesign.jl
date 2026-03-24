# Example 8: R1ρ Dispersion — Adaptive Design with Two Continuous Design Variables
#
# Model: I = I₀ exp(-R₁ρ(νSL) · tSL)
# where  R₁ρ(νSL) = R₂₀ + A · K² / (4π² νSL² + K²)
#
# Four parameters (I₀, R₂₀, A, K), two continuous design variables (tSL, νSL).
# Interest in K via Ds-optimality.
#
# Demonstrates:
#   1. Two-dimensional design space (spin-lock time × spin-lock frequency)
#   2. Adaptive experiment with 2D candidates
#   3. Batch design for comparison (same budget)
#   4. 2D design allocation (bubble plot) and Gateaux derivative
#   5. Posterior convergence and evolution animation
#   6. Corner plots for dispersion parameters

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

# R1ρ dispersion: R₁ρ(ν) = R₂₀ + A K² / (4π² ν² + K²)
function R1rho(R₂₀, A, K, ν)
    R₂₀ + A * K^2 / (4π^2 * ν^2 + K^2)
end

function model(θ, x)
    θ.I₀ * exp(-R1rho(θ.R₂₀, θ.A, θ.K, x.νSL) * x.tSL)
end

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(I₀=2.0, R₂₀=5.0, A=20.0, K=15000.0)
σ_true = 0.2
budget = 100.0

acquire(x) = model(θ_true, x) + σ_true * randn()

# ═══════════════════════════════════════════════
# 2. Design problem and prior
# ═══════════════════════════════════════════════

prob = DesignProblem(
    model,
    parameters=(
        I₀=LogUniform(0.001, 1000),
        R₂₀=Uniform(0, 50),
        A=Uniform(0, 100),
        K=LogUniform(100, 100_000),
    ),
    transformation=select(:K),
    sigma=Returns(σ_true),
)
display(prob)

# 2D candidate grid: tSL × νSL
tSL_vals = range(0.001, 0.08, length=20)
νSL_vals = range(300, 15_000, length=25)
candidates = candidate_grid(tSL=tSL_vals, νSL=νSL_vals)

# ═══════════════════════════════════════════════
# 3. Adaptive experiment
# ═══════════════════════════════════════════════

prior = Particles(prob, 10000)

result_adaptive = run_adaptive(
    prob, candidates, prior, acquire;
    budget=budget,
)
display(result_adaptive)

log_adaptive = result_adaptive.log
n_adaptive = length(log_adaptive)

# ═══════════════════════════════════════════════
# 4. Batch design for comparison (same n)
# ═══════════════════════════════════════════════

ξ = design(prob, candidates, prior; n=n_adaptive)
display(ξ)

result_batch = run_batch(ξ, prob, prior, acquire)
display(result_batch)

# ═══════════════════════════════════════════════
# 5. Optimality verification (batch design)
# ═══════════════════════════════════════════════

opt_check = verify_optimality(prob, candidates, prior, ξ)
display(opt_check)

# ═══════════════════════════════════════════════
# 6. Plots
# ═══════════════════════════════════════════════

# --- Figure 1: Adaptive trajectory in 2D design space ---

fig1 = Figure(size=(700, 500))

ax1 = GLMakie.Axis(fig1[1, 1],
    xlabel="tSL (s)", ylabel="νSL (Hz)",
    title="Adaptive Design Trajectory ($n_adaptive steps)")

scatter!(ax1, [e.x.tSL for e in log_adaptive], [e.x.νSL for e in log_adaptive];
    color=1:n_adaptive, colormap=:viridis, markersize=10)
lines!(ax1, [e.x.tSL for e in log_adaptive], [e.x.νSL for e in log_adaptive];
    color=:gray, linewidth=0.5)

CairoMakie.Colorbar(fig1[1, 2]; colormap=:viridis,
    colorrange=(1, n_adaptive), label="Step")

fig1

# --- Figure 2: 2D batch design allocation (bubble plot) ---

fig2 = plot_design_allocation(ξ, candidates)

# --- Figure 3: Gateaux derivative over 2D candidate space ---

fig3 = plot_gateaux(opt_check)

# --- Figure 4: Dispersion curve — credible bands at fixed tSL ---

t_slice = 0.08
slice_grid = candidate_grid(tSL=[t_slice], νSL=range(300, 15_000, length=100))

fig4 = plot_credible_bands(prob, result_adaptive, result_batch;
    labels=["Adaptive", "Batch"],
    truth=θ_true, x_grid=slice_grid)

# --- Figure 5: Corner plot — adaptive vs batch posterior ---

fig5 = plot_corner(result_adaptive, result_batch;
    labels=["Adaptive", "Batch"], truth=θ_true)

# --- Figure 6: Convergence ---

fig6 = plot_convergence(result_adaptive; truth=θ_true, params=[:R₂₀, :A, :K])

# --- Figure 7: Posterior evolution animation ---

if has_posterior_history(log_adaptive)
    record_corner_animation(log_adaptive, "ex8_posterior_evolution.gif";
        params=[:R₂₀, :A, :K],
        truth=θ_true, framerate=5)
end

# --- Figure 8: Dashboard replay animation ---

record_dashboard(result_adaptive, prob; filename="ex8_dashboard.gif")

println("\nDone. Figures created.")
