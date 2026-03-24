# Example 3: Two Decays, Vector Observation — Simultaneous Measurement
#
# Two exponential decays observed simultaneously as a vector (y₁, y₂) at
# a single time t. Four parameters (A₁, k₁, A₂, k₂), interest in both rates.
# The Jacobian is a 2×4 matrix. The FIM sums information from both observables.
#
# Demonstrates: Vector-valued predict, vector sigma, FIM from multiple
# simultaneous observables, batch design where a single time point informs both rates.

using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using LinearAlgebra
using Random

# ENV["JULIA_DEBUG"] = OptimalDesign
Random.seed!(42)

# ═══════════════════════════════════════════════
# 1. The model and ground truth
# ═══════════════════════════════════════════════

function model(θ, x)
    [θ.A₁ * exp(-θ.k₁ * x.t),
        θ.A₂ * exp(-θ.k₂ * x.t)]
end

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A₁=7.0, k₁=8.0, A₂=1.0, k₂=80.0)
σ_true = 0.05
n_obs = 100

acquire(x) = model(θ_true, x) .+ σ_true .* randn(2)

println("Truth: A₁=$(θ_true.A₁), k₁=$(θ_true.k₁), A₂=$(θ_true.A₂), k₂=$(θ_true.k₂)")

# ═══════════════════════════════════════════════
# 2. Design problem and prior
# ═══════════════════════════════════════════════

prob = DesignProblem(
    model,
    parameters=(
        A₁=LogUniform(0.1, 10), k₁=Uniform(0.1, 100),
        A₂=LogUniform(0.1, 10), k₂=Uniform(0.1, 100)),
    transformation=select(:k₁, :k₂),
    sigma=Returns([σ_true, σ_true]),
    cost=x -> x.t + 1,
)
display(prob)

candidates = candidate_grid(t=range(0.001, 0.5, length=200))
prior = Particles(prob, 1000)

# ═══════════════════════════════════════════════
# 3. Batch design via exchange algorithm
# ═══════════════════════════════════════════════

println("Calculating batch design (n=$n_obs)...")
ξ = design(prob, candidates, prior; n=n_obs, exchange_steps=200)
display(ξ)

# ═══════════════════════════════════════════════
# 4. Optimality verification (Gateaux derivative)
# ═══════════════════════════════════════════════

opt = verify_optimality(prob, candidates, prior, ξ;
    posterior_samples=1000)
display(opt)

# ═══════════════════════════════════════════════
# 5. Efficiency comparison against uniform
# ═══════════════════════════════════════════════

ξ_unif = uniform_allocation(candidates, n_obs)

eff = efficiency(ξ_unif, ξ, prob, candidates, prior; posterior_samples=1000)

# ═══════════════════════════════════════════════
# 6. Simulated acquisition — optimal vs uniform
# ═══════════════════════════════════════════════

println("\n--- Simulated experiments ---")

result_opt = run_batch(ξ, prob, prior, acquire)
display(result_opt)

result_unif = run_batch(ξ_unif, prob, prior, acquire)
display(result_unif)

# ═══════════════════════════════════════════════
# 7. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")

# --- Figure 1: Design allocation + Gateaux derivative ---

fig1 = plot_gateaux(opt)

# --- Figure 2: Credible bands — vector model auto-splits by component ---

fig2 = plot_credible_bands(prob, result_opt, result_unif;
    labels=["Optimal ($n_obs obs)", "Uniform ($n_obs obs)"], truth=θ_true)

# --- Figure 3: Corner plot — prior vs optimal posterior ---

fig3 = plot_corner(result_opt; truth=θ_true)

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(result_unif, result_opt;
    labels=["Uniform", "Optimal"], truth=θ_true)

println("Done. Figures created.")
