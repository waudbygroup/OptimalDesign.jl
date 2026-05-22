# Example 10: y = A cos(ω t) — GMM resampling vs systematic resampling
#
# The cosine model from example 9 is a good stress-test for particle filters:
# the posterior on ω can become multi-modal due to aliasing when only a few
# short-time observations are available. A plain systematic resampler collapses
# to a single mode early, whereas GMM resampling fits the multi-modal density
# explicitly and draws fresh particles from all modes.
#
# This example runs the same adaptive EIG experiment twice with identical priors
# and random seeds, swapping only the resampling strategy. Diagnostic figures
# from the GMM runs are written to examples/gmm_resample_log/.

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
σ_true = 0.5
acquire(x) = model(θ_true, x) + σ_true * randn()

n_particles = 3000
budget = 50

println("Problem: y = A cos(ω t) + noise")
println("Truth:   A = $(θ_true.A), ω = $(θ_true.ω)")
println("Prior:   A ≈ 1 (tight), ω ∈ Uniform(1, 100)")
println("Budget:  $budget measurements, $n_particles particles\n")

# ═══════════════════════════════════════════════════
# 2. Problem (EIG criterion for both runs)
# ═══════════════════════════════════════════════════

prior_specs = (A=Normal(1.0, 0.05), ω=Uniform(1.0, 100.0))

prob = DesignProblem(
    model;
    parameters=prior_specs,
    sigma=Returns(σ_true),
    criterion=EIGCriterion(outer_samples=80, inner_samples=80),
)

candidates = candidate_grid(t=range(0.005, 1.0, length=200))

log_dir = joinpath(@__DIR__, "gmm_resample_log")
mkpath(log_dir)
foreach(f -> rm(joinpath(log_dir, f)), readdir(log_dir))

# ═══════════════════════════════════════════════════
# 3. Adaptive runs
# ═══════════════════════════════════════════════════

println("--- Run 1: SystematicResampling ---")
Random.seed!(2024)
prior_sys = Particles(prob, n_particles; resampling=SystematicResampling(ess_threshold=0.5))
result_sys = run_adaptive(prob, candidates, prior_sys, acquire;
    budget=Float64(budget), n_per_step=1, headless=false)

println("--- Run 2: GMMResampling (log → $log_dir) ---")
Random.seed!(2024)
prior_gmm = Particles(prob, n_particles;
    resampling=GMMResampling(ess_threshold=0.5, k_max=8, log_dir=log_dir))
result_gmm = run_adaptive(prob, candidates, prior_gmm, acquire;
    budget=Float64(budget), n_per_step=1, headless=false)

# ═══════════════════════════════════════════════════
# 4. Summary
# ═══════════════════════════════════════════════════

μ_sys = mean(result_sys.posterior)
μ_gmm = mean(result_gmm.posterior)
σω_sys = std(result_sys.posterior).ω
σω_gmm = std(result_gmm.posterior).ω

println("\nPosterior on ω (truth = $(θ_true.ω)):")
println("  Systematic:  mean = $(round(μ_sys.ω; digits=3)),  std = $(round(σω_sys; digits=3))")
println("  GMM:         mean = $(round(μ_gmm.ω; digits=3)),  std = $(round(σω_gmm; digits=3))")
n_figs = length(readdir(log_dir))
println("\nGMM diagnostic figures saved to $log_dir  ($n_figs files)")

# ═══════════════════════════════════════════════════
# 5. Comparison figure
# ═══════════════════════════════════════════════════

post_sys_ω = [θ.ω for θ in OptimalDesign.sample(result_sys.posterior, 2000)]
post_gmm_ω = [θ.ω for θ in OptimalDesign.sample(result_gmm.posterior, 2000)]

steps = 1:budget
seq_sys = [e.x.t for e in result_sys.log]
seq_gmm = [e.x.t for e in result_gmm.log]

grid = candidate_grid(t=range(0.0, 1.0, length=400))
t_grid = [c.t for c in grid]
truth = [model(θ_true, c) for c in grid]

preds_sys = posterior_predictions(prob, result_sys.posterior, grid; n_samples=200)
preds_gmm = posterior_predictions(prob, result_gmm.posterior, grid; n_samples=200)
band_sys = credible_band(preds_sys; level=0.9)
band_gmm = credible_band(preds_gmm; level=0.9)

fig = Figure(size=(1000, 700))

# Pick sequences
ax1 = Makie.Axis(fig[1, 1:2]; xlabel="step", ylabel="t selected",
    title="Adaptive pick sequence")
scatter!(ax1, steps, seq_sys; color=:steelblue, marker=:circle, markersize=8, label="Systematic")
scatter!(ax1, steps, seq_gmm; color=:darkorange, marker=:diamond, markersize=8, label="GMM")
axislegend(ax1; position=:rt)

# Posterior histograms on ω
ax2 = Makie.Axis(fig[2, 1]; xlabel="ω", ylabel="density",
    title="Posterior on ω (truth = $(θ_true.ω))")
hist!(ax2, post_sys_ω; normalization=:pdf, color=(:steelblue, 0.6), label="Systematic")
hist!(ax2, post_gmm_ω; normalization=:pdf, color=(:darkorange, 0.6), label="GMM")
vlines!(ax2, [θ_true.ω]; color=:black, linewidth=1.5, label="truth")
axislegend(ax2; position=:rt)

# Posterior predictive bands
ax3 = Makie.Axis(fig[2, 2]; xlabel="t", ylabel="y",
    title="Posterior predictions (90% band)")
band!(ax3, t_grid, band_sys.lower, band_sys.upper; color=(:steelblue, 0.3))
band!(ax3, t_grid, band_gmm.lower, band_gmm.upper; color=(:darkorange, 0.3))
lines!(ax3, t_grid, vec(mean(preds_sys; dims=1)); color=:steelblue, label="Systematic")
lines!(ax3, t_grid, vec(mean(preds_gmm; dims=1)); color=:darkorange, label="GMM")
lines!(ax3, t_grid, truth; color=:black, linewidth=1.5, label="truth")
axislegend(ax3; position=:rb)

display(fig)
println("\nDone.")
