# Example 1: Exponential Decay — Scalar Observation, Batch Design
#
# Simplest case. One design variable (time t), two parameters (A, R₂),
# interest in R₂ via transformation.
#
# Demonstrates: Basic DesignProblem setup, transformation for Ds-optimality,
# FIM computation, candidate scoring, efficiency comparison against uniform spacing.

using OptimalDesign
using ComponentArrays
using Distributions
using LinearAlgebra
using Random

Random.seed!(42)

# --- Problem setup ---

prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    parameters = (A=Normal(1, 0.1), R₂=LogUniform(1, 50)),
    transformation = select(:R₂),
    sigma = (θ, ξ) -> 0.05,
    cost = (prev, ξ) -> ξ.t + 0.1,
)

candidates = [(t=t,) for t in range(0.001, 0.5, length=200)]

# --- Draw prior particles ---

prior = ParticlePosterior(prob, 1000)

# --- Examine FIM at a single (θ, ξ) ---

θ_test = ComponentArray(A=1.0, R₂=25.0)
ξ_test = (t=0.1,)

M = information(prob, θ_test, ξ_test)
println("FIM at θ=(A=1, R₂=25), t=0.1:")
display(M)

# Transformed (Ds-optimal for R₂)
Mt = transform(prob, M + information(prob, θ_test, (t=0.02,)), θ_test)
println("\nTransformed FIM (interest in R₂ only):")
display(Mt)

# --- Score all candidates by expected utility ---

println("\nScoring candidates by D-optimality...")
scores = score_candidates(prob, DCriterion(), prior.particles, candidates; batch_size=100)

# Find the best candidates
ranking = sortperm(scores, rev=true)
println("\nTop 10 design points:")
for i in 1:10
    idx = ranking[i]
    println("  t = $(round(candidates[idx].t, digits=4)),  utility = $(round(scores[idx], digits=3))")
end

# --- Compare with A-optimality and E-optimality ---

scores_A = score_candidates(prob, ACriterion(), prior.particles, candidates; batch_size=100)
scores_E = score_candidates(prob, ECriterion(), prior.particles, candidates; batch_size=100)

ranking_A = sortperm(scores_A, rev=true)
ranking_E = sortperm(scores_E, rev=true)

println("\nBest design point by criterion:")
println("  D-optimal: t = $(round(candidates[ranking[1]].t, digits=4))")
println("  A-optimal: t = $(round(candidates[ranking_A[1]].t, digits=4))")
println("  E-optimal: t = $(round(candidates[ranking_E[1]].t, digits=4))")

# --- Batch design (Phase 2) ---
# design = select(prob, candidates, prior; n=20, criterion=DCriterion())
# uniform = uniform_allocation(candidates, 20)
# eff = efficiency(uniform, design, prob, prior)
