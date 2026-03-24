# Example 4: Two Decays, Discrete Control Variable — Selective Measurement
#
# Same two decays as Example 3, but now a control variable i ∈ {1, 2} selects
# which one is observed. Each measurement returns a scalar. The parameter space
# is the same, but only one decay contributes to the FIM per measurement.
#
# Demonstrates: Discrete control variable as a candidate field, block-sparse
# Jacobian, batch design choosing how to allocate between decays,
# block independence in the posterior.

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
    if x.i == 1
        θ.A₁ * exp(-θ.k₁ * x.t)
    else
        θ.A₂ * exp(-θ.k₂ * x.t)
    end
end

# Ground truth (unknown to the design algorithm)
θ_true = ComponentArray(A₁=1.0, k₁=10.0, A₂=1.0, k₂=40.0)
σ_true = 0.05
n_obs = 20

acquire(x) = model(θ_true, x) + σ_true * randn()

println("Truth: A₁=$(θ_true.A₁), k₁=$(θ_true.k₁), A₂=$(θ_true.A₂), k₂=$(θ_true.k₂)")

# ═══════════════════════════════════════════════
# 2. Design problem and prior
# ═══════════════════════════════════════════════

prob = DesignProblem(
    model,
    parameters=(A₁=Normal(1, 0.1), k₁=LogUniform(1, 50),
        A₂=Normal(1, 0.1), k₂=LogUniform(1, 50)),
    transformation=select(:k₁, :k₂),
    sigma=Returns(σ_true),
    cost=x -> x.t + 0.1,
)
display(prob)

candidates = candidate_grid(i=[1, 2], t=range(0.001, 0.5, length=200))

prior = Particles(prob, 1000)

# ═══════════════════════════════════════════════
# 3. Examine block-sparse FIM structure
# ═══════════════════════════════════════════════

M1 = information(prob, θ_true, (i=1, t=0.1))
M2 = information(prob, θ_true, (i=2, t=0.05))

println("FIM measuring decay 1 (i=1, t=0.1):")
display(round.(M1, digits=4))
println("\nFIM measuring decay 2 (i=2, t=0.05):")
display(round.(M2, digits=4))

println("\nBlock sparsity verification:")
println("  M1 rows 3:4 norm: ", round(norm(M1[3:4, :]), digits=10), " (should be ≈ 0)")
println("  M2 rows 1:2 norm: ", round(norm(M2[1:2, :]), digits=10), " (should be ≈ 0)")

M_combined = M1 + M2
println("\nCombined FIM rank: ", rank(M_combined))

# ═══════════════════════════════════════════════
# 4. Batch design via exchange algorithm
# ═══════════════════════════════════════════════

println("\nCalculating batch design (n=$n_obs)...")
ξ = design(prob, candidates, prior; n=n_obs, exchange_steps=200)
display(ξ)

n_decay1 = sum(c for (x, c) in ξ if x.i == 1; init=0)
n_decay2 = sum(c for (x, c) in ξ if x.i == 2; init=0)
println("  Allocation: $n_decay1 on decay 1, $n_decay2 on decay 2")

# ═══════════════════════════════════════════════
# 5. Optimality verification
# ═══════════════════════════════════════════════

opt = verify_optimality(prob, candidates, prior, ξ;
    posterior_samples=1000)
display(opt)

# ═══════════════════════════════════════════════
# 6. Efficiency comparison against uniform
# ═══════════════════════════════════════════════

ξ_unif = uniform_allocation(candidates, n_obs)

eff = efficiency(ξ_unif, ξ, prob, candidates, prior; posterior_samples=1000)

# ═══════════════════════════════════════════════
# 7. Simulated acquisition — optimal vs uniform
# ═══════════════════════════════════════════════

println("\n--- Simulated experiments ---")

result_opt = run_batch(ξ, prob, prior, acquire)
display(result_opt)

result_unif = run_batch(ξ_unif, prob, prior, acquire)
display(result_unif)

# ═══════════════════════════════════════════════
# 8. Plots
# ═══════════════════════════════════════════════

println("\nGenerating plots...")

# --- Figure 1: Design allocation + Gateaux derivative ---

fig1 = plot_gateaux(opt)

# --- Figure 2: Credible bands for decay 1 ---

fig2 = plot_credible_bands(prob, result_opt, result_unif;
    labels=["Optimal", "Uniform"], truth=θ_true,
    x_grid=candidate_grid(i=[1], t=range(0.001, 0.5, length=100)))

# --- Figure 3: Corner plot — prior vs optimal posterior ---

fig3 = plot_corner(result_opt; truth=θ_true)

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(result_unif, result_opt;
    labels=["Uniform", "Optimal"], truth=θ_true)
