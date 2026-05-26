# Example 10: y = A cos(ω t) exp(−R t) — Comparing resampling strategies
#
# A particle filter maintains parameter uncertainty through sequential updating.
# When observations arrive, particle weights are updated via the likelihood,
# but repeated resampling from a finite particle set causes "particle impoverishment":
# diversity is lost and the filter eventually collapses onto a single parameter value.
#
# The damped cosine model is an excellent stress-test because the frequency ω can
# have a multi-modal posterior: if measurements are placed at early times to
# resolve A (as EIG with select(:A) does), ω remains under-constrained and
# multiple aliased frequencies fit the data equally well.
#
# Three resampling strategies are compared:
#
#   LiuWestResampling  (default) — systematic resample + kernel jitter in
#                                  unconstrained space; preserves moments but
#                                  struggles with sharp multi-modality
#   SystematicResampling         — plain resample, no jitter; fastest but
#                                  most prone to particle collapse
#   GMMResampling                — fits a Gaussian mixture model (BIC-guided K)
#                                  to the current particle cloud and draws fresh
#                                  particles from it; best for multi-modal posteriors
#
# All three runs use identical priors and random seeds — only the resampling
# strategy differs.

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

model(θ, x) = θ.A * cos(θ.ω * x.t) * exp(-θ.R * x.t)

θ_true = ComponentArray(A=50.0, ω=66.7, R=8.0)
σ_true = 10.0
acquire(x) = model(θ_true, x) + σ_true * randn()

n_particles = 5000
budget = 20

println("Model: y = A cos(ω t) exp(−R t) + noise")
println("Truth: A = $(θ_true.A), ω = $(θ_true.ω), R = $(θ_true.R), σ = $σ_true")
println("Budget: $budget measurements, $n_particles particles\n")

# ═══════════════════════════════════════════════════
# 2. Design problem — EIG focusing on amplitude A
#
# Using select(:A) deliberately creates a "hard" scenario for ω:
# the algorithm places measurements at early times where amplitude
# dominates, leaving the frequency under-constrained.  The ω posterior
# then becomes multi-modal as aliased frequencies all fit the data.
# ═══════════════════════════════════════════════════

prior_specs = (
    A = LogUniform(1.0, 500.0),
    ω = Uniform(1.0, 100.0),
    R = Uniform(0.1, 10.0),
)

prob = DesignProblem(
    model;
    parameters=prior_specs,
    transformation=select(:A),
    sigma=Returns(σ_true),
    criterion=EIGCriterion(outer_samples=80, inner_samples=80),
)

candidates = candidate_grid(t=range(0.005, 1.0, length=200))

# Diagnostic figures from GMM resampling events are written here:
log_dir = joinpath(@__DIR__, "gmm_resample_log")
mkpath(log_dir)
foreach(f -> rm(joinpath(log_dir, f)), readdir(log_dir))

# ═══════════════════════════════════════════════════
# 3. Adaptive runs — identical priors, different resamplers
# ═══════════════════════════════════════════════════

println("--- Run 1: LiuWestResampling (default) ---")
Random.seed!(2024)
prior_lw = Particles(prob, n_particles; resampling=LiuWestResampling(a=0.98, ess_threshold=0.5))
result_lw = run_adaptive(prob, candidates, prior_lw, acquire;
    budget=Float64(budget), n_per_step=1, headless=true)

println("--- Run 2: SystematicResampling ---")
Random.seed!(2024)
prior_sys = Particles(prob, n_particles; resampling=SystematicResampling(ess_threshold=0.5))
result_sys = run_adaptive(prob, candidates, prior_sys, acquire;
    budget=Float64(budget), n_per_step=1, headless=true)

println("--- Run 3: GMMResampling (log → $log_dir) ---")
Random.seed!(2024)
prior_gmm = Particles(prob, n_particles;
    resampling=GMMResampling(ess_threshold=0.8, k_max=6, log_dir=log_dir))
result_gmm = run_adaptive(prob, candidates, prior_gmm, acquire;
    budget=Float64(budget), n_per_step=1, headless=true)

# ═══════════════════════════════════════════════════
# 4. Summary statistics
# ═══════════════════════════════════════════════════

for (label, r) in [("LiuWest", result_lw), ("Systematic", result_sys), ("GMM", result_gmm)]
    μ = mean(r); s = std(r)
    println("$label posterior:")
    println("  ω: $(round(μ.ω; digits=2)) ± $(round(s.ω; digits=2))  (truth: $(θ_true.ω))")
    println("  A: $(round(μ.A; digits=2)) ± $(round(s.A; digits=2))  (truth: $(θ_true.A))")
    println("  R: $(round(μ.R; digits=2)) ± $(round(s.R; digits=2))  (truth: $(θ_true.R))")
    println()
end

n_figs = length(readdir(log_dir))
println("GMM diagnostic figures saved to $log_dir  ($n_figs files)")

# ═══════════════════════════════════════════════════
# 5. Comparison figure
# ═══════════════════════════════════════════════════

post_lw_ω  = [θ.ω for θ in OptimalDesign.sample(result_lw.posterior,  3000)]
post_sys_ω = [θ.ω for θ in OptimalDesign.sample(result_sys.posterior, 3000)]
post_gmm_ω = [θ.ω for θ in OptimalDesign.sample(result_gmm.posterior, 3000)]

steps    = 1:budget
seq_lw   = [e.x.t for e in result_lw.log]
seq_sys  = [e.x.t for e in result_sys.log]
seq_gmm  = [e.x.t for e in result_gmm.log]

grid   = candidate_grid(t=range(0.0, 1.0, length=400))
t_grid = [c.t for c in grid]
truth  = [model(θ_true, c) for c in grid]

preds_lw  = posterior_predictions(prob, result_lw.posterior,  grid; n_samples=200)
preds_sys = posterior_predictions(prob, result_sys.posterior, grid; n_samples=200)
preds_gmm = posterior_predictions(prob, result_gmm.posterior, grid; n_samples=200)
band_lw   = credible_band(preds_lw;  level=0.9)
band_sys  = credible_band(preds_sys; level=0.9)
band_gmm  = credible_band(preds_gmm; level=0.9)

colors = (:royalblue, :steelblue, :darkorange)
labels = ["LiuWest", "Systematic", "GMM"]

fig = Figure(size=(1100, 800))

# Pick sequences
ax1 = Makie.Axis(fig[1, 1:3]; xlabel="step", ylabel="t selected",
    title="Adaptive pick sequence (same EIG criterion, different resampler)")
scatter!(ax1, steps, seq_lw;  color=colors[1], marker=:circle,   markersize=7, label=labels[1])
scatter!(ax1, steps, seq_sys; color=colors[2], marker=:utriangle, markersize=7, label=labels[2])
scatter!(ax1, steps, seq_gmm; color=colors[3], marker=:diamond,   markersize=7, label=labels[3])
axislegend(ax1; position=:rt)

# Posterior on ω
ax2 = Makie.Axis(fig[2, 1]; xlabel="ω", ylabel="density",
    title="Posterior on ω — does it find the truth?")
hist!(ax2, post_lw_ω;  normalization=:pdf, color=(colors[1], 0.6), label=labels[1])
hist!(ax2, post_sys_ω; normalization=:pdf, color=(colors[2], 0.5), label=labels[2])
hist!(ax2, post_gmm_ω; normalization=:pdf, color=(colors[3], 0.5), label=labels[3])
vlines!(ax2, [θ_true.ω]; color=:black, linewidth=2, label="truth")
axislegend(ax2; position=:rt)

# Posterior predictive bands
ax3 = Makie.Axis(fig[2, 2:3]; xlabel="t", ylabel="y",
    title="Posterior predictions (90% credible band)")
band!(ax3, t_grid, band_lw.lower,  band_lw.upper;  color=(colors[1], 0.25))
band!(ax3, t_grid, band_sys.lower, band_sys.upper; color=(colors[2], 0.25))
band!(ax3, t_grid, band_gmm.lower, band_gmm.upper; color=(colors[3], 0.25))
lines!(ax3, t_grid, vec(mean(preds_lw;  dims=1)); color=colors[1],  linewidth=2, label=labels[1])
lines!(ax3, t_grid, vec(mean(preds_sys; dims=1)); color=colors[2],  linewidth=2, label=labels[2])
lines!(ax3, t_grid, vec(mean(preds_gmm; dims=1)); color=colors[3], linewidth=2, label=labels[3])
lines!(ax3, t_grid, truth; color=:black, linewidth=2, label="truth")
axislegend(ax3; position=:rb)

display(fig)
save("ex10_resampling_comparison.png", fig)

println("\nDone.")
