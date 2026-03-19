# Example 2: Inversion Recovery — Analytic Jacobian, Transformation
#
# Three parameters (R₁, A, B), Ds-optimality via transformation onto R₁.
# Analytic Jacobian supplied for performance.
#
# Demonstrates: Analytic Jacobian alongside ForwardDiff validation,
# Ds-optimality via transformation, validation against known result
# (optimal delays concentrate at specific τ/T₁ ratios).

using OptimalDesign
using ComponentArrays
using Distributions
using ForwardDiff
using LinearAlgebra
using Random

Random.seed!(42)

# --- Problem with analytic Jacobian ---

predict = (θ, ξ) -> θ.A - θ.B * exp(-θ.R₁ * ξ.τ)

jac = (θ, ξ) -> begin
    e = exp(-θ.R₁ * ξ.τ)
    [1.0  -e  θ.B * ξ.τ * e]
end

prob = DesignProblem(
    predict,
    jacobian = jac,
    parameters = (A=Normal(1, 0.1), B=Normal(2, 0.1), R₁=LogUniform(0.1, 5)),
    transformation = select(:R₁),
    cost = (prev, ξ) -> 1.0 + ξ.τ,
)

candidates = [(τ=τ,) for τ in range(0.01, 5.0, length=200)]

# --- Validate analytic Jacobian against ForwardDiff ---

θ_test = draw(prob.parameters)
ξ_test = candidates[50]

J_analytic = prob.jacobian(θ_test, ξ_test)
J_ad = ForwardDiff.jacobian(θ_ -> [prob.predict(θ_, ξ_test)], θ_test)

println("Jacobian validation:")
println("  Analytic:    ", round.(J_analytic, digits=6))
println("  ForwardDiff: ", round.(J_ad, digits=6))
println("  Max error:   ", maximum(abs.(J_analytic .- J_ad)))

# --- Compare FIM with and without analytic Jacobian ---

prob_ad = DesignProblem(
    predict,
    parameters = (A=Normal(1, 0.1), B=Normal(2, 0.1), R₁=LogUniform(0.1, 5)),
    transformation = select(:R₁),
    cost = (prev, ξ) -> 1.0 + ξ.τ,
)

θ_eval = ComponentArray(A=1.0, B=2.0, R₁=1.0)
ξ_eval = (τ=1.0,)

M_analytic = information(prob, θ_eval, ξ_eval)
M_ad = information(prob_ad, θ_eval, ξ_eval)

println("\nFIM agreement: ", isapprox(M_analytic, M_ad, atol=1e-10))

# --- Score candidates for Ds-optimality on R₁ ---

prior = ParticlePosterior(prob, 1000)

println("\nScoring candidates for Ds-optimality (R₁)...")
scores = score_candidates(prob, DCriterion(), prior.particles, candidates; batch_size=100)

ranking = sortperm(scores, rev=true)
println("\nTop 10 delay times for R₁ estimation:")
for i in 1:10
    idx = ranking[i]
    println("  τ = $(round(candidates[idx].τ, digits=4)),  utility = $(round(scores[idx], digits=3))")
end

# Known result: for inversion recovery with R₁ ≈ 1, optimal delays
# concentrate near τ/T₁ ≈ 1.2 (i.e., τ ≈ 1.2 for R₁ = 1).
println("\nExpected: optimal τ values near τ/T₁ ≈ 1.2")
