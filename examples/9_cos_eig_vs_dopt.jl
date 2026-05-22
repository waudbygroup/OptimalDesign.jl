# Example 9: y = A cos(ω t) — EIG vs D-optimality with a broad frequency prior
#
# A periodic model is a textbook case where the local-Gaussian (FIM-based)
# design assumption breaks down. The Fisher information about ω scales as
# (A t sin(ω t))², so the FIM-optimal time keeps growing with t. But for a
# broad prior on ω, large t aliases — many ω values fit the same y — and the
# information per measurement collapses.
#
# Expected Information Gain (EIG) integrates over the prior on ω before scoring
# the design point. It "sees" the aliasing and prefers shorter times where the
# observation is unambiguous.
#
# This example compares:
#   1. D-optimal batch design (exchange algorithm on FIM)
#   2. EIG-optimal greedy design (nested-MC over particles)
# on the same problem, the same prior, and the same number of measurements,
# then repeats the comparison adaptively (posterior updated between picks)
# where EIG is intended to shine.

using OptimalDesign
using CairoMakie
using ComponentArrays
using Distributions
using Random
using Statistics

Random.seed!(2024)

# ═══════════════════════════════════════════════════
# 1. Model and ground truth
# ═══════════════════════════════════════════════════

model(θ, x) = θ.A * cos(θ.ω * x.t)

θ_true = ComponentArray(A=1.0, ω=35)
σ_true = 0.05
acquire(x) = model(θ_true, x) + σ_true * randn()

n = 20

println("Problem: y = A cos(ω t) + noise")
println("Truth:   A = $(θ_true.A), ω = $(θ_true.ω)")
println("Prior:   A ≈ 1 (tight), ω ∈ Uniform(1, 100)\n")

# ═══════════════════════════════════════════════════
# 2. Two problems: same physics, different criterion
# ═══════════════════════════════════════════════════

prior_specs = (A=Normal(1.0, 0.05), ω=Uniform(1.0, 100.0))

prob_dopt = DesignProblem(
    model;
    parameters=prior_specs,
    # transformation=select(:ω),
    sigma=Returns(σ_true),
    criterion=DCriterion(),
)

prob_eig = DesignProblem(
    model;
    parameters=prior_specs,
    # transformation=select(:ω),
    sigma=Returns(σ_true),
    criterion=EIGCriterion(outer_samples=200, inner_samples=200),
)

candidates = candidate_grid(t=range(0.005, 1.0, length=200))

# Identical prior particles for both, to make the comparison fair.
Random.seed!(2024)
prior_dopt = Particles(prob_dopt, 1000)
Random.seed!(2024)
prior_eig = Particles(prob_eig, 1000)

# ═══════════════════════════════════════════════════
# 3. Adaptive comparison — posterior updated between picks
# ═══════════════════════════════════════════════════

println("\n--- Adaptive experiments (same budget = $n measurements) ---")

# Fresh, identical priors for the adaptive runs.
Random.seed!(2024)
prior_dopt_a = Particles(prob_dopt, 5000)
Random.seed!(2024)
prior_eig_a = Particles(prob_eig, 5000)

# Default cost = 1 per measurement, so budget = n gives n measurements.
result_dopt_a = run_adaptive(prob_dopt, candidates, prior_dopt_a, acquire;
    budget=Float64(n), n_per_step=1, headless=true, record_posterior=true)
result_eig_a = run_adaptive(prob_eig, candidates, prior_eig_a, acquire;
    budget=Float64(n), n_per_step=1, headless=true, record_posterior=true)

μ_dopt_a = mean(result_dopt_a.posterior)
μ_eig_a = mean(result_eig_a.posterior)
σω_dopt_a = std(result_dopt_a.posterior).ω
σω_eig_a = std(result_eig_a.posterior).ω

println("Adaptive posterior on ω (truth = $(θ_true.ω)):")
println("  D-optimal:  mean = $(round(μ_dopt_a.ω; digits=3)), std = $(round(σω_dopt_a; digits=3))")
println("  EIG:        mean = $(round(μ_eig_a.ω; digits=3)), std = $(round(σω_eig_a; digits=3))")

# ═══════════════════════════════════════════════════
# 4. Adaptive plot — sequence of picks + posterior on ω
# ═══════════════════════════════════════════════════

println("\nPlotting adaptive comparison...")

dopt_seq_t = [e.x.t for e in result_dopt_a.log]
eig_seq_t = [e.x.t for e in result_eig_a.log]
steps = 1:n

# Sample posterior on ω for histograms.
post_dopt_ω = [θ.ω for θ in OptimalDesign.sample(result_dopt_a.posterior, 2000)]
post_eig_ω = [θ.ω for θ in OptimalDesign.sample(result_eig_a.posterior, 2000)]

fig2 = Figure(size=(900, 600))

ax_a1 = Makie.Axis(fig2[1, 1:2]; xlabel="step", ylabel="t selected",
    title="Adaptive pick sequence — D-optimal picks one region; EIG explores then refines")
scatter!(ax_a1, steps, dopt_seq_t; color=:steelblue, marker=:circle,
    markersize=12, label="D-optimal")
scatter!(ax_a1, steps, eig_seq_t; color=:darkorange, marker=:diamond,
    markersize=12, label="EIG")
axislegend(ax_a1; position=:rt)

ax_a2 = Makie.Axis(fig2[2, 1]; xlabel="ω", ylabel="density",
    title="Adaptive posterior on ω (truth = $(θ_true.ω))")
hist!(ax_a2, post_dopt_ω; normalization=:pdf,
    color=:steelblue, label="D-optimal")
hist!(ax_a2, post_eig_ω; normalization=:pdf,
    color=:darkorange, label="EIG")
vlines!(ax_a2, [θ_true.ω]; color=:black, linewidth=1.5, label="truth")
axislegend(ax_a2; position=:rt)

# Final posterior credible bands from the adaptive runs
grid = candidate_grid(t=range(0.0, 1.0, length=400))
t_grid = [c.t for c in grid]
truth = [model(θ_true, c) for c in grid]

preds_dopt_a = posterior_predictions(prob_dopt, result_dopt_a.posterior, grid; n_samples=200)
preds_eig_a = posterior_predictions(prob_eig, result_eig_a.posterior, grid; n_samples=200)
band_dopt_a = credible_band(preds_dopt_a; level=0.9)
band_eig_a = credible_band(preds_eig_a; level=0.9)
mean_dopt_a = vec(mean(preds_dopt_a; dims=1))
mean_eig_a = vec(mean(preds_eig_a; dims=1))

obs_t_dopt_a = [o.x.t for o in result_dopt_a.observations]
obs_y_dopt_a = [o.y isa NamedTuple ? o.y.value : o.y for o in result_dopt_a.observations]
obs_t_eig_a = [o.x.t for o in result_eig_a.observations]
obs_y_eig_a = [o.y isa NamedTuple ? o.y.value : o.y for o in result_eig_a.observations]

ax_a3 = Makie.Axis(fig2[2, 2]; xlabel="t", ylabel="y",
    title="Adaptive posterior predictions (90% band)")
band!(ax_a3, t_grid, band_dopt_a.lower, band_dopt_a.upper;
    color=(:steelblue, 0.4), label="D-optimal")
band!(ax_a3, t_grid, band_eig_a.lower, band_eig_a.upper;
    color=(:darkorange, 0.4), label="EIG")
lines!(ax_a3, t_grid, mean_dopt_a; color=:steelblue, label="D-optimal mean")
lines!(ax_a3, t_grid, mean_eig_a; color=:darkorange, label="EIG mean")
scatter!(ax_a3, obs_t_dopt_a, obs_y_dopt_a;
    color=:steelblue, marker=:circle, label="D-optimal observations")
scatter!(ax_a3, obs_t_eig_a, obs_y_eig_a;
    color=:darkorange, marker=:diamond, label="EIG observations")
lines!(ax_a3, t_grid, truth; color=:black, label="truth")
# axislegend(ax_a3; position=:rb)

display(fig2)

println("\nDone.")
