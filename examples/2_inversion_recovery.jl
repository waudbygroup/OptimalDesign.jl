# Example 2: Inversion Recovery — Analytic Jacobian, Batch Design, Transformation
#
# Model: y = A - B exp(-R₁ τ)   (inversion recovery in NMR/MRI)
# Three parameters (A, B, R₁), Ds-optimality for R₁ via DeltaMethod.
# Analytic Jacobian supplied for performance and validated against ForwardDiff.
#
# Demonstrates:
#   1. Analytic Jacobian with ForwardDiff validation
#   2. Ds-optimality via DeltaMethod transformation
#   3. Batch design via exchange algorithm
#   4. Optimality verification (Gateaux derivative)
#   5. Efficiency comparison against uniform spacing
#   6. Simulated acquisition and posterior inference
#   7. Validation: optimal delays near τ/T₁ ≈ 1.2


using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using ForwardDiff
using LinearAlgebra
using Random

# ENV["JULIA_DEBUG"] = OptimalDesign
Random.seed!(42)

# ═══════════════════════════════════════════════
# 1. The model and ground truth
# ═══════════════════════════════════════════════

function model(θ, x)
    θ.A - θ.B * exp(-θ.R₁ * x.τ)
end

function jac(θ, x)
    e = exp(-θ.R₁ * x.τ)
    # ∂y/∂A = 1, ∂y/∂B = -e, ∂y/∂R₁ = B τ e
    [1.0 -e θ.B * x.τ * e]
end

# Ground truth (unknown to design algorithm)
θ_true = ComponentArray(A=1.0, B=2.0, R₁=1.3)
σ_true = 0.05
T₁_true = 1.0 / θ_true.R₁   # = 1.0 s

acquire(x) = model(θ_true, x) + σ_true * randn()

println("Truth: A = $(θ_true.A), B = $(θ_true.B), R₁ = $(θ_true.R₁)  (T₁ = $(T₁_true) s)")

# ═══════════════════════════════════════════════
# 2. Validate analytic Jacobian against ForwardDiff
# ═══════════════════════════════════════════════

prob_with_jac = DesignProblem(
    model,
    jacobian=jac,
    parameters=(A=Normal(1, 0.5), B=Normal(2, 0.5), R₁=Uniform(0.1, 5)),
    transformation=select(:R₁),
    sigma=Returns(σ_true),
)

θ_test = draw(prob_with_jac.parameters)
x_test = (τ=1.0,)

J_analytic = jac(θ_test, x_test)
J_ad = ForwardDiff.jacobian(θ_ -> [model(θ_, x_test)], θ_test)

println("Jacobian validation:")
println("  Analytic:    ", round.(J_analytic, digits=6))
println("  ForwardDiff: ", round.(J_ad, digits=6))
println("  Max error:   ", round(maximum(abs.(J_analytic .- J_ad)); sigdigits=3))

# Also compare FIM: create an equivalent problem without analytic Jacobian
prob_ad = DesignProblem(
    model,
    parameters=(A=Normal(1, 0.1), B=Normal(2, 0.1), R₁=LogUniform(0.1, 5)),
    transformation=select(:R₁),
    sigma=Returns(σ_true),
)

θ_eval = ComponentArray(A=1.0, B=2.0, R₁=1.0)
x_eval = (τ=1.0,)
M_analytic = information(prob_with_jac, θ_eval, x_eval)
M_ad = information(prob_ad, θ_eval, x_eval)
println("  FIM agreement: ", isapprox(M_analytic, M_ad, atol=1e-10), "\n")

# ═══════════════════════════════════════════════
# 3. Design problem and prior
# ═══════════════════════════════════════════════

prob = prob_with_jac
display(prob)

candidates = candidate_grid(τ=range(0.01, 5.0, length=200))
prior = Particles(prob, 1000)

# ═══════════════════════════════════════════════
# 4. Batch design via exchange algorithm
# ═══════════════════════════════════════════════

n_obs = 20
println("Running exchange algorithm for batch design (n=$n_obs)...")
ξ = design(prob, candidates, prior; n=n_obs)
display(ξ)

# ═══════════════════════════════════════════════
# 5. Optimality verification (Gateaux derivative)
# ═══════════════════════════════════════════════

opt = verify_optimality(prob, candidates, prior, ξ;
    posterior_samples=1000)
display(opt)

# ═══════════════════════════════════════════════
# 6. Efficiency comparison against uniform
# ═══════════════════════════════════════════════

ξ_unif = uniform_allocation(candidates, n_obs)

eff = efficiency(ξ_unif, ξ, prob, candidates, prior)

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

# --- Figure 2: Prior → Posterior credible bands ---

fig2 = plot_credible_bands(prob, result_opt, result_unif;
    labels=["Optimal ($n_obs obs)", "Uniform ($n_obs obs)"], truth=θ_true)

# --- Figure 3: Corner plot — prior vs optimal posterior ---

fig3 = plot_corner(result_opt; truth=θ_true)

# --- Figure 4: Corner plot — optimal vs uniform posterior ---

fig4 = plot_corner(result_opt, result_unif;
    labels=["Optimal", "Uniform"], truth=θ_true)

println("Done. Figures created.")
