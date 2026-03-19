# Example 5: Exponential Decay — Adaptive Design with Simulated Acquisition
#
# Uses Example 1's problem setup but runs a full adaptive experiment against
# a simulated ground truth.
#
# Demonstrates: Simulated acquisition, sequential posterior updating,
# adaptive candidate scoring, posterior convergence.
#
# Note: run_experiment (Phase 3) is not yet implemented. This example
# performs the adaptive loop manually to demonstrate the mechanics.

using OptimalDesign
using ComponentArrays
using Distributions
using LinearAlgebra
using Random

Random.seed!(42)

# --- Ground truth ---

θ_true = ComponentArray(A=1.0, R₂=25.0)

# --- Simulated acquisition ---

acquire = let θ = θ_true, σ = 0.05
    ξ -> θ.A * exp(-θ.R₂ * ξ.t) + σ * randn()
end

# --- Problem and prior ---

prob = DesignProblem(
    (θ, ξ) -> θ.A * exp(-θ.R₂ * ξ.t),
    parameters = (A=Normal(1, 0.1), R₂=LogUniform(1, 50)),
    transformation = select(:R₂),
    sigma = (θ, ξ) -> 0.05,
    cost = (prev, ξ) -> ξ.t + 0.1,
)

candidates = [(t=t,) for t in range(0.001, 0.5, length=200)]
posterior = ParticlePosterior(prob, 1000)

# --- Manual adaptive loop ---

budget = 10.0
spent = 0.0
step = 0
ξ_prev = nothing
history = NamedTuple[]

println("Starting adaptive experiment (budget=$budget)")
println("True parameters: A=$(θ_true.A), R₂=$(θ_true.R₂)")
println()

while spent < budget
    step += 1

    # Score candidates by expected utility per unit cost
    scores = score_candidates(prob, DCriterion(), posterior.particles, candidates; batch_size=100)

    # Adjust for cost
    cost_adjusted = [
        scores[k] / prob.cost(ξ_prev, candidates[k])
        for k in eachindex(candidates)
    ]

    # Select best candidate
    best_idx = argmax(cost_adjusted)
    ξ = candidates[best_idx]
    c = prob.cost(ξ_prev, ξ)

    if spent + c > budget
        break
    end

    # Acquire observation
    y = acquire(ξ)
    spent += c

    # Update posterior
    update!(posterior, prob, ξ, y)

    # Record
    push!(history, (ξ=ξ, y=y, cost=c))

    μ = posterior_mean(posterior)
    ess = effective_sample_size(posterior)

    println("Step $step: t=$(round(ξ.t, digits=4)), y=$(round(y, digits=4)), " *
            "cost=$(round(c, digits=2)), spent=$(round(spent, digits=2)), " *
            "ESS=$(round(ess, digits=0))")
    println("  Posterior mean: A=$(round(μ.A, digits=4)), R₂=$(round(μ.R₂, digits=2))")

    ξ_prev = ξ
end

# --- Results ---

μ_final = posterior_mean(posterior)
println("\n--- Final Results ---")
println("Steps taken: $step")
println("Budget spent: $(round(spent, digits=2)) / $budget")
println("Posterior mean: A=$(round(μ_final.A, digits=4)), R₂=$(round(μ_final.R₂, digits=2))")
println("True values:    A=$(θ_true.A), R₂=$(θ_true.R₂)")
println("Error in R₂:    $(round(abs(μ_final.R₂ - θ_true.R₂), digits=2))")

# --- Phase 3: run_experiment interface ---
# result = run_experiment(
#     prob, candidates, ParticlePosterior(prob, 1000), acquire;
#     budget = 10.0,
#     criterion = DCriterion(),
#     n_per_step = 1,
#     headless = true,
# )
