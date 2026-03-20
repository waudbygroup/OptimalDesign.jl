# Example 4: Two Decays, Discrete Control Variable — Selective Measurement
#
# Same two decays as Example 3, but now a control variable i ∈ {1, 2} selects
# which one is observed. Each measurement returns a scalar. The parameter space
# is the same, but only one decay contributes to the FIM per measurement.
#
# Demonstrates: Discrete control variable as a candidate field, block-sparse
# Jacobian, adaptive design choosing which decay to measure next,
# block independence in the posterior.

using OptimalDesign
using ComponentArrays
using Distributions
using LinearAlgebra
using Random

Random.seed!(42)

# --- Problem setup ---

prob = DesignProblem(
    (θ, ξ) -> if ξ.i == 1
        θ.A₁ * exp(-θ.R₂₁ * ξ.t)
    else
        θ.A₂ * exp(-θ.R₂₂ * ξ.t)
    end,
    parameters=(A₁=Normal(1, 0.1), R₂₁=LogUniform(1, 50),
        A₂=Normal(1, 0.1), R₂₂=LogUniform(1, 50)),
    transformation=select(:R₂₁, :R₂₂),
    sigma=(θ, ξ) -> 0.05,
    cost=(prev, ξ) -> ξ.t + 0.1,
)

candidates = [
    (i=i, t=t)
    for i in [1, 2]
    for t in range(0.001, 0.5, length=200)
]

# --- Examine block-sparse FIM structure ---

θ_test = ComponentArray(A₁=1.0, R₂₁=10.0, A₂=1.0, R₂₂=40.0)

M1 = information(prob, θ_test, (i=1, t=0.1))
M2 = information(prob, θ_test, (i=2, t=0.05))

println("FIM measuring decay 1 (i=1, t=0.1):")
display(round.(M1, digits=4))
println("\nFIM measuring decay 2 (i=2, t=0.05):")
display(round.(M2, digits=4))

println("\nBlock sparsity verification:")
println("  M1 rows 3:4 norm: ", round(norm(M1[3:4, :]), digits=10), " (should be ≈ 0)")
println("  M2 rows 1:2 norm: ", round(norm(M2[1:2, :]), digits=10), " (should be ≈ 0)")

# Combined FIM from both measurements
M_combined = M1 + M2
println("\nCombined FIM rank: ", rank(M_combined))

# --- Score candidates ---

prior = ParticlePosterior(prob, 500)

println("\nScoring all candidates (both decays, all times)...")
scores = score_candidates(prob, DCriterion(), prior.particles, candidates; posterior_samples=100)

ranking = sortperm(scores, rev=true)
println("\nTop 10 design points:")
for k in 1:10
    idx = ranking[k]
    c = candidates[idx]
    println("  i=$(c.i), t=$(round(c.t, digits=4)),  utility=$(round(scores[idx], digits=3))")
end

# --- Compare: which decay is more informative per measurement? ---

scores_decay1 = [scores[k] for k in eachindex(candidates) if candidates[k].i == 1]
scores_decay2 = [scores[k] for k in eachindex(candidates) if candidates[k].i == 2]
times_decay1 = [candidates[k].t for k in eachindex(candidates) if candidates[k].i == 1]
times_decay2 = [candidates[k].t for k in eachindex(candidates) if candidates[k].i == 2]

best1 = argmax(scores_decay1)
best2 = argmax(scores_decay2)
println("\nBest for each decay:")
println("  Decay 1: t=$(round(times_decay1[best1], digits=4)), utility=$(round(scores_decay1[best1], digits=3))")
println("  Decay 2: t=$(round(times_decay2[best2], digits=4)), utility=$(round(scores_decay2[best2], digits=3))")

# --- Comparison with Example 3 ---
# Example 3 (vector observation) gets information about both rates from every
# measurement. This example (selective observation) must allocate measurements
# between the two decays. Comparing efficiencies quantifies the value of
# simultaneous measurement.
