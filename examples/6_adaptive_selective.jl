# Example 6: Two Decays, Discrete Control — Adaptive with Switching
#
# Uses Example 4's problem but runs adaptively, demonstrating how the selector
# balances measurements between the two decays with switching costs.
#
# Demonstrates: Adaptive design with discrete control variable, switching cost,
# block-sparse posterior updates, the selector preferring to stay on the current
# decay unless the other is substantially more informative.

using OptimalDesign
using ComponentArrays
using Distributions
using LinearAlgebra
using Random

Random.seed!(42)

# --- Ground truth ---

θ_true = ComponentArray(A₁=1.0, R₂₁=10.0, A₂=1.0, R₂₂=40.0)

# --- Simulated acquisition ---

acquire = let θ = θ_true, σ = 0.05
    ξ -> (ξ.i == 1 ? θ.A₁ * exp(-θ.R₂₁ * ξ.t) : θ.A₂ * exp(-θ.R₂₂ * ξ.t)) + σ * randn()
end

# --- Problem with switching cost ---

prob = DesignProblem(
    (θ, ξ) -> ξ.i == 1 ? θ.A₁ * exp(-θ.R₂₁ * ξ.t) : θ.A₂ * exp(-θ.R₂₂ * ξ.t),
    parameters=(A₁=Normal(1, 0.1), R₂₁=LogUniform(1, 50),
        A₂=Normal(1, 0.1), R₂₂=LogUniform(1, 50)),
    transformation=select(:R₂₁, :R₂₂),
    sigma=(θ, ξ) -> 0.05,
    cost=(prev, ξ) -> begin
        t_measure = ξ.t + 0.1
        t_switch = (prev !== nothing && prev.i != ξ.i) ? 1.0 : 0.0
        t_measure + t_switch
    end,
)

candidates = [(i=i, t=t) for i in [1, 2] for t in range(0.001, 0.5, length=200)]
posterior = ParticlePosterior(prob, 1000)

# --- Manual adaptive loop ---

budget = 20.0
spent = 0.0
step = 0
ξ_prev = nothing
history = NamedTuple[]

println("Starting adaptive experiment with switching costs (budget=$budget)")
println("True: A₁=$(θ_true.A₁), R₂₁=$(θ_true.R₂₁), A₂=$(θ_true.A₂), R₂₂=$(θ_true.R₂₂)")
println()

while spent < budget
    step += 1

    # Score candidates
    scores = score_candidates(prob, DCriterion(), posterior.particles, candidates; posterior_samples=100)

    # Adjust for cost (including switching)
    cost_adjusted = [
        scores[k] / prob.cost(ξ_prev, candidates[k])
        for k in eachindex(candidates)
    ]

    best_idx = argmax(cost_adjusted)
    ξ = candidates[best_idx]
    c = prob.cost(ξ_prev, ξ)

    if spent + c > budget
        break
    end

    # Acquire
    y = acquire(ξ)
    spent += c

    # Update posterior
    update!(posterior, prob, ξ, y)

    push!(history, (ξ=ξ, y=y, cost=c))

    switched = ξ_prev !== nothing && ξ_prev.i != ξ.i
    μ = posterior_mean(posterior)

    println("Step $step: decay=$(ξ.i), t=$(round(ξ.t, digits=4))" *
            (switched ? " [SWITCH]" : "") *
            ", cost=$(round(c, digits=2)), spent=$(round(spent, digits=2))")
    println("  Posterior: R₂₁=$(round(μ.R₂₁, digits=2)), R₂₂=$(round(μ.R₂₂, digits=2))")

    ξ_prev = ξ
end

# --- Summary ---

μ_final = posterior_mean(posterior)
n_decay1 = count(h -> h.ξ.i == 1, history)
n_decay2 = count(h -> h.ξ.i == 2, history)
n_switches = count(i -> history[i].ξ.i != history[i-1].ξ.i, 2:length(history))

println("\n--- Final Results ---")
println("Steps: $step ($n_decay1 on decay 1, $n_decay2 on decay 2, $n_switches switches)")
println("Budget spent: $(round(spent, digits=2)) / $budget")
println("Posterior mean: R₂₁=$(round(μ_final.R₂₁, digits=2)), R₂₂=$(round(μ_final.R₂₂, digits=2))")
println("True values:    R₂₁=$(θ_true.R₂₁), R₂₂=$(θ_true.R₂₂)")

# Because R₂₂ > R₂₁, the fast decay is harder to characterise
# and should receive more short-time measurements.
# The selector should initially explore both, then focus on
# whichever has larger posterior uncertainty relative to cost.
