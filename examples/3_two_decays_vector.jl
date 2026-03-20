# Example 3: Two Decays, Vector Observation — Simultaneous Measurement
#
# Two exponential decays observed simultaneously as a vector (y₁, y₂) at
# a single time t. Four parameters (A₁, R₂₁, A₂, R₂₂), interest in both rates.
# The Jacobian is a 2×4 matrix. The FIM sums information from both observables.
#
# Demonstrates: Vector-valued predict, vector sigma, FIM from multiple
# simultaneous observables, batch design where a single time point informs both rates.

using OptimalDesign
using ComponentArrays
using Distributions
using LinearAlgebra
using Random

Random.seed!(42)

# --- Problem setup ---

prob = DesignProblem(
    (θ, ξ) -> [θ.A₁ * exp(-θ.R₂₁ * ξ.t),
        θ.A₂ * exp(-θ.R₂₂ * ξ.t)],
    parameters=(A₁=Normal(1, 0.1), R₂₁=LogUniform(1, 50),
        A₂=Normal(1, 0.1), R₂₂=LogUniform(1, 50)),
    transformation=select(:R₂₁, :R₂₂),
    sigma=(θ, ξ) -> [0.05, 0.05],
    cost=(prev, ξ) -> ξ.t + 0.1,
)

candidates = [(t=t,) for t in range(0.001, 0.5, length=200)]

# --- Examine FIM structure ---

θ_test = ComponentArray(A₁=1.0, R₂₁=10.0, A₂=1.0, R₂₂=40.0)
ξ_test = (t=0.05,)

M = information(prob, θ_test, ξ_test)
println("FIM at t=0.05 (vector observation, 4 parameters):")
display(M)
println("\nRank: ", rank(M))
println("Eigenvalues: ", round.(eigvals(Symmetric(M)), digits=4))

# With R₂₁=10, R₂₂=40, the fast decay (R₂₂) needs short times while
# the slow decay (R₂₁) needs longer times. A single time point provides
# information about BOTH rates simultaneously.

# --- Score candidates ---

prior = ParticlePosterior(prob, 500)

println("\nScoring candidates...")
scores = score_candidates(prob, DCriterion(), prior.particles, candidates; posterior_samples=100)

ranking = sortperm(scores, rev=true)
println("\nTop 10 design points (simultaneous observation):")
for i in 1:10
    idx = ranking[i]
    println("  t = $(round(candidates[idx].t, digits=4)),  utility = $(round(scores[idx], digits=3))")
end

# The optimal design should balance information for both rates.
# If R₂₁ ≪ R₂₂, optimal times will span a wide range to
# capture both fast and slow decays.
println("\nNote: optimal times should span a range covering both decay rates")
