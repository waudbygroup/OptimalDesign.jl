# Example 1b: Exponential Decay — EIG Design
#
# Same model as Example 1 (y = A exp(-k t)), but the design criterion is
# Expected Information Gain (EIG) rather than Ds-optimality.
#
# EIG maximises the mutual information MI(θ; y | x), averaged over the prior,
# without requiring a choice of "parameter of interest".  It is criterion-free
# in that sense, but it does use the prior directly, so it gives a different
# (and often tighter) design than the FIM-based Ds criterion.
#
# The example compares three 50-measurement batch designs:
#   1. Ds-optimal   (exchange algorithm on FIM, interest in k)
#   2. EIG-optimal  (exchange algorithm with nested-MC criterion)
#   3. Uniform      (evenly spaced reference)
#
# and then runs each against the simulator and compares the posteriors.

using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random

Random.seed!(42)

# ═══════════════════════════════════════════════════
# 1. Model and ground truth
# ═══════════════════════════════════════════════════

function model(θ, x)
    θ.A * exp(-θ.k * x.t)
end

θ_true = ComponentArray(A=1.0, k=25.0)
σ_true = 0.1
acquire(x) = model(θ_true, x) + σ_true * randn()

n = 50

println("Problem: y = A exp(-k t) + noise")
println("Truth:   A = $(θ_true.A), k = $(θ_true.k)")
println("Acquire: $n measurements\n")

# ═══════════════════════════════════════════════════
# 2. Design problems — one per criterion
# ═══════════════════════════════════════════════════

prior_specs = (A=LogUniform(0.1, 10), k=Uniform(1, 50))

prob_ds = DesignProblem(
    model,
    parameters=prior_specs,
    transformation=select(:k),
    sigma=Returns(σ_true),
)

prob_eig = DesignProblem(
    model,
    parameters=prior_specs,
    sigma=Returns(σ_true),
    criterion=EIGCriterion(outer_samples=200, inner_samples=200),
)

candidates = candidate_grid(t=range(0.001, 0.5, length=200))

# Use identical prior particles so the comparison is fair
Random.seed!(42)
prior_ds  = Particles(prob_ds,  1000)
Random.seed!(42)
prior_eig = Particles(prob_eig, 1000)

# ═══════════════════════════════════════════════════
# 3. Compute designs
# ═══════════════════════════════════════════════════

println("--- Ds-optimal design ---")
ξ_ds  = design(prob_ds,  candidates, prior_ds;  n)
display(ξ_ds)

println("\n--- EIG-optimal design ---")
ξ_eig = design(prob_eig, candidates, prior_eig; n)
display(ξ_eig)

ξ_unif = uniform_allocation(candidates, n)

# ═══════════════════════════════════════════════════
# 4. Score all candidates under each criterion
# ═══════════════════════════════════════════════════

println("\nScoring all candidates...")
scores_ds  = score_candidates(prob_ds,  prior_ds,  candidates; posterior_samples=500)
scores_eig = score_candidates(prob_eig, prior_eig, candidates; posterior_samples=200)
ts = [c.t for c in candidates]

# ═══════════════════════════════════════════════════
# 5. Run experiments
# ═══════════════════════════════════════════════════

println("\n--- Simulated experiments ---")

Random.seed!(42)
result_ds   = run_batch(ξ_ds,   prob_ds,   prior_ds,   acquire)
Random.seed!(42)
result_eig  = run_batch(ξ_eig,  prob_eig,  prior_eig,  acquire)
Random.seed!(42)
result_unif = run_batch(ξ_unif, prob_eig,  prior_eig,  acquire)

for (label, r) in [("Ds-optimal", result_ds), ("EIG", result_eig), ("Uniform", result_unif)]
    μ = mean(r); s = std(r)
    println("  $label:  k = $(round(μ.k; digits=2)) ± $(round(s.k; digits=2))  " *
            "(A = $(round(μ.A; digits=3)) ± $(round(s.A; digits=3)))")
end

# ═══════════════════════════════════════════════════
# 6. Plots
# ═══════════════════════════════════════════════════

println("\nGenerating plots...")

# --- Figure 1: Criterion scores vs t ---

fig1 = Figure(size=(800, 350))
ax1 = Makie.Axis(fig1[1, 1]; xlabel="t", ylabel="Ds score (log det FIM)",
    title="Ds-optimal score")
lines!(ax1, ts, scores_ds; color=:steelblue)
vlines!(ax1, [x.t for (x, _) in ξ_ds]; color=:steelblue, linestyle=:dash, alpha=0.6)

ax2 = Makie.Axis(fig1[1, 2]; xlabel="t", ylabel="EIG (nats)",
    title="EIG score")
lines!(ax2, ts, scores_eig; color=:darkorange)
vlines!(ax2, [x.t for (x, _) in ξ_eig]; color=:darkorange, linestyle=:dash, alpha=0.6)

display(fig1)

# --- Figure 2: Credible bands, all three designs ---

fig2 = plot_credible_bands(prob_eig, result_ds, result_eig, result_unif;
    labels=["Ds-optimal ($n)", "EIG ($n)", "Uniform ($n)"], truth=θ_true)
display(fig2)

# --- Figure 3: Corner plot — Ds vs EIG posteriors ---

fig3 = plot_corner(result_ds.posterior, result_eig.posterior;
    labels=["Ds-optimal", "EIG"], truth=θ_true)
display(fig3)

println("Done.")
