# Example 1: Exponential Decay — Batch Design, Simulated Experiment, Posterior
#
# Simplest case. One design variable (time t), two parameters (A, R₂),
# interest in R₂ via Ds-optimality (DeltaMethod transformation).
#
# Demonstrates the full workflow:
#   1. Problem setup and prior
#   2. Batch design via exchange algorithm
#   3. Optimality verification (Gateaux derivative)
#   4. Efficiency comparison against uniform spacing
#   5. Simulated acquisition using run_batch
#   6. Posterior credible bands and corner plots


using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random
using GLMakie

# ENV["JULIA_DEBUG"] = OptimalDesign
Random.seed!(42)

# ═══════════════════════════════════════════════════
# 1. The model and ground truth
# ═══════════════════════════════════════════════════

function model(θ, x)
    θ.A * exp(-θ.R₂ * x.t)
end

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A=1.0, R₂=25.0)
σ_true = 0.1

# Simulated acquisition function
acquire(x) = model(θ_true, x) + σ_true * randn()

n = 50

println("Problem: y = A exp(-R₂ t) + noise")
println("Truth:   A = $(θ_true.A), R₂ = $(θ_true.R₂)")
println("Acquire: $n measurements")
println("Goal:    Ds-optimal design for R₂\n")

# ═══════════════════════════════════════════════════
# 2. Design problem and prior
# ═══════════════════════════════════════════════════

prob = DesignProblem(
    model,
    parameters=(A=LogUniform(0.1, 10), R₂=Uniform(1, 50)),
    transformation=select(:R₂),
    sigma=Returns(σ_true),
)

candidates = candidate_grid(t=range(0.001, 0.5, length=200))
prior = Particles(prob, 1000)

# ═══════════════════════════════════════════════════
# 3. Batch design via exchange algorithm
# ═══════════════════════════════════════════════════

println("Calculating batch design (n=$n)...")
ξ = design(prob, candidates, prior; n)
display(ξ)

# ═══════════════════════════════════════════════════
# 4. Optimality verification (Gateaux derivative)
# ═══════════════════════════════════════════════════

opt = verify_optimality(prob, candidates, prior, ξ;
    posterior_samples=1000)
display(opt)

# ═══════════════════════════════════════════════════
# 5. Efficiency comparison against uniform
# ═══════════════════════════════════════════════════

ξ_unif = uniform_allocation(candidates, n)
eff = efficiency(ξ_unif, ξ, prob, candidates, prior)

# ═══════════════════════════════════════════════════
# 6. Simulated acquisition — optimal vs uniform
# ═══════════════════════════════════════════════════

println("\n--- Simulated experiments ---")

result_opt = run_batch(ξ, prob, prior, acquire)
result_unif = run_batch(ξ_unif, prob, prior, acquire)

μ_opt = mean(result_opt.posterior)
μ_unif = mean(result_unif.posterior)
println("Posterior mean (optimal):  A = $(round(μ_opt.A; digits=3)), R₂ = $(round(μ_opt.R₂; digits=2))")
println("Posterior mean (uniform):  A = $(round(μ_unif.A; digits=3)), R₂ = $(round(μ_unif.R₂; digits=2))")

# ═══════════════════════════════════════════════════
# 7. Plots
# ═══════════════════════════════════════════════════

println("\nGenerating plots...")

# --- Figure 1: Design allocation + Gateaux derivative ---

fig1 = plot_gateaux(opt)

# --- Figure 2: Prior → Posterior credible bands ---

fig2 = plot_credible_bands(prob, result_opt, result_unif;
    labels=["Optimal ($n obs)", "Uniform ($n obs)"], truth=θ_true)

# --- Figure 3: Corner plot — prior vs optimal posterior ---

fig3 = plot_corner(result_opt; truth=θ_true)

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(result_unif, result_opt;
    labels=["Uniform", "Optimal"], truth=θ_true)

println("Done. Figures created.")
