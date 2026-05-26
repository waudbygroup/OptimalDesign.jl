# Example 9: y = A cos(ω t) — EIG vs D-optimality with a broad frequency prior
#
# A periodic model is a textbook case where the local-Gaussian (FIM-based)
# design assumption breaks down. The Fisher information about ω scales as
# (A t sin(ω t))², so the FIM-based criterion keeps preferring larger t.
# But for a broad prior on ω, large t aliases — many ω values produce the
# same y — so the information per measurement collapses.
#
# Expected Information Gain (EIG) integrates over the full posterior before
# scoring each candidate. It "sees" the aliasing and prefers shorter times
# where each observation is unambiguous.
#
# This example shows:
#   Part A — Batch comparison
#     • Score every candidate under D-opt (FIM) and EIG
#     • D-opt: score peaks at large t;  EIG: score peaks at moderate t
#     • Run both designs and compare posteriors on ω
#   Part B — Adaptive comparison
#     • Both algorithms update the posterior after each measurement
#     • EIG is expected to resolve ω more reliably when the prior is broad

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

θ_true = ComponentArray(A=1.0, ω=35.0)
σ_true = 5.0   # broad noise to make the problem challenging
acquire(x) = model(θ_true, x) + σ_true * randn()

println("Model: y = A cos(ω t) + noise")
println("Truth: A = $(θ_true.A), ω = $(θ_true.ω), σ = $σ_true")
println("Prior: A ≈ 1 (tight), ω ∈ Uniform(1, 100)  [broad — aliasing expected]\n")

# ═══════════════════════════════════════════════════
# 2. Design problems
# ═══════════════════════════════════════════════════

prior_specs = (A=Normal(1.0, 0.05), ω=Uniform(1.0, 100.0))

prob_dopt = DesignProblem(
    model;
    parameters=prior_specs,
    sigma=Returns(σ_true),
    criterion=DCriterion(),
)

prob_eig = DesignProblem(
    model;
    parameters=prior_specs,
    sigma=Returns(σ_true),
    criterion=EIGCriterion(outer_samples=200, inner_samples=200),
)

candidates = candidate_grid(t=range(0.005, 1.0, length=200))
ts = [c.t for c in candidates]
n_batch = 200    # number of batch measurements

# ═══════════════════════════════════════════════════
# Part A: Batch comparison
# ═══════════════════════════════════════════════════

println("=== Part A: Batch comparison ($n_batch measurements) ===\n")

# --- A1. Score every candidate under each criterion ---

println("Scoring candidates...")
Random.seed!(2024)
prior_dopt_b = Particles(prob_dopt, 5000)
Random.seed!(2024)
prior_eig_b  = Particles(prob_eig,  5000)

scores_dopt = score_candidates(prob_dopt, prior_dopt_b, candidates; posterior_samples=500)
scores_eig  = score_candidates(prob_eig,  prior_eig_b,  candidates; posterior_samples=200)

println("D-opt peak at t = $(round(ts[argmax(scores_dopt)]; digits=3))")
println("EIG   peak at t = $(round(ts[argmax(scores_eig)];  digits=3))\n")

# --- A2. Compute designs ---

println("Computing D-optimal batch design...")
Random.seed!(2024)
ξ_dopt_b = design(prob_dopt, candidates, Particles(prob_dopt, 5000); n=n_batch)

println("Computing EIG batch design (greedy)...")
Random.seed!(2024)
ξ_eig_b = design(prob_eig, candidates, Particles(prob_eig, 5000); n=n_batch)

# --- A3. Run experiments ---

println("Running batch experiments...")
Random.seed!(2024)
result_dopt_b = run_batch(ξ_dopt_b, prob_dopt, Particles(prob_dopt, 5000), acquire)
Random.seed!(2024)
result_eig_b  = run_batch(ξ_eig_b,  prob_eig,  Particles(prob_eig,  5000), acquire)

μ_dopt_b = mean(result_dopt_b)
μ_eig_b  = mean(result_eig_b)
println("Batch posterior on ω (truth = $(θ_true.ω)):")
println("  D-optimal: mean = $(round(μ_dopt_b.ω; digits=2)), std = $(round(std(result_dopt_b).ω; digits=2))")
println("  EIG:       mean = $(round(μ_eig_b.ω; digits=2)), std = $(round(std(result_eig_b).ω; digits=2))\n")

# ═══════════════════════════════════════════════════
# Part B: Adaptive comparison
# ═══════════════════════════════════════════════════

n_adaptive = 400

println("=== Part B: Adaptive comparison ($n_adaptive steps) ===\n")

Random.seed!(2024)
prior_dopt_a = Particles(prob_dopt, 10000, resampling=SystematicResampling(0.25))
Random.seed!(2024)
prior_eig_a  = Particles(prob_eig,  10000, resampling=SystematicResampling(0.25))

result_dopt_a = run_adaptive(prob_dopt, candidates, prior_dopt_a, acquire;
    budget=Float64(n_adaptive), n_per_step=1, headless=true, record_posterior=true)
result_eig_a  = run_adaptive(prob_eig,  candidates, prior_eig_a,  acquire;
    budget=Float64(n_adaptive), n_per_step=1, headless=true, record_posterior=true)

println("Adaptive posterior on ω (truth = $(θ_true.ω)):")
println("  D-optimal: mean = $(round(mean(result_dopt_a).ω; digits=2)), " *
        "std = $(round(std(result_dopt_a).ω; digits=2))")
println("  EIG:       mean = $(round(mean(result_eig_a).ω; digits=2)), " *
        "std = $(round(std(result_eig_a).ω; digits=2))\n")

# ═══════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════

println("Generating plots...")

# --- Figure 1: Criterion score curves ---

fig1 = Figure(size=(900, 380))

ax1a = Makie.Axis(fig1[1, 1]; xlabel="t", ylabel="D-opt score (log det FIM)",
    title="D-optimal: score peaks at large t (FIM ∝ t²)")
lines!(ax1a, ts, scores_dopt; color=:steelblue, linewidth=2)
vlines!(ax1a, [ts[argmax(scores_dopt)]]; color=:steelblue, linestyle=:dash)

ax1b = Makie.Axis(fig1[1, 2]; xlabel="t", ylabel="EIG (nats)",
    title="EIG: score peaks at moderate t (avoids aliasing)")
lines!(ax1b, ts, scores_eig; color=:darkorange, linewidth=2)
vlines!(ax1b, [ts[argmax(scores_eig)]]; color=:darkorange, linestyle=:dash)

display(fig1)
save("ex9_score_curves.png", fig1)

# --- Figure 2: Batch design allocations ---

fig2 = Figure(size=(900, 350))

w_dopt = weights(ξ_dopt_b, candidates)
w_eig  = weights(ξ_eig_b,  candidates)

ax2a = Makie.Axis(fig2[1, 1]; xlabel="t", ylabel="weight",
    title="D-optimal batch ($n_batch measurements)")
Makie.stem!(ax2a, ts, w_dopt; color=:steelblue)

ax2b = Makie.Axis(fig2[1, 2]; xlabel="t", ylabel="weight",
    title="EIG batch ($n_batch measurements, greedy)")
Makie.stem!(ax2b, ts, w_eig; color=:darkorange)

display(fig2)
save("ex9_batch_allocations.png", fig2)

# --- Figure 3: Adaptive results — pick sequence + posteriors on ω ---

post_dopt_ω = [θ.ω for θ in OptimalDesign.sample(result_dopt_a.posterior, 3000)]
post_eig_ω  = [θ.ω for θ in OptimalDesign.sample(result_eig_a.posterior, 3000)]

dopt_seq = [e.x.t for e in result_dopt_a.log]
eig_seq  = [e.x.t for e in result_eig_a.log]
steps = 1:n_adaptive

grid = candidate_grid(t=range(0.0, 1.0, length=400))
t_grid = [c.t for c in grid]

preds_dopt = posterior_predictions(prob_dopt, result_dopt_a.posterior, grid; n_samples=200)
preds_eig  = posterior_predictions(prob_eig,  result_eig_a.posterior,  grid; n_samples=200)
band_dopt  = credible_band(preds_dopt; level=0.9)
band_eig   = credible_band(preds_eig;  level=0.9)

fig3 = Figure(size=(1000, 700))

ax3a = Makie.Axis(fig3[1, 1:2]; xlabel="step", ylabel="t selected",
    title="Adaptive pick sequence — D-optimal visits large t; EIG stays at shorter t")
scatter!(ax3a, steps, dopt_seq; color=:steelblue, marker=:circle,  markersize=4, label="D-optimal")
scatter!(ax3a, steps, eig_seq;  color=:darkorange, marker=:diamond, markersize=4, label="EIG")
axislegend(ax3a; position=:rt)

ax3b = Makie.Axis(fig3[2, 1]; xlabel="ω", ylabel="density",
    title="Adaptive posterior on ω (truth = $(θ_true.ω))")
hist!(ax3b, post_dopt_ω; normalization=:pdf, color=(:steelblue, 0.6), label="D-optimal")
hist!(ax3b, post_eig_ω;  normalization=:pdf, color=(:darkorange, 0.6), label="EIG")
vlines!(ax3b, [θ_true.ω]; color=:black, linewidth=2, label="truth")
axislegend(ax3b; position=:rt)

ax3c = Makie.Axis(fig3[2, 2]; xlabel="t", ylabel="y",
    title="Posterior predictions (90% credible band)")
band!(ax3c, t_grid, band_dopt.lower, band_dopt.upper; color=(:steelblue, 0.3))
band!(ax3c, t_grid, band_eig.lower,  band_eig.upper;  color=(:darkorange, 0.3))
lines!(ax3c, t_grid, vec(mean(preds_dopt; dims=1)); color=:steelblue,  linewidth=2, label="D-optimal")
lines!(ax3c, t_grid, vec(mean(preds_eig;  dims=1)); color=:darkorange, linewidth=2, label="EIG")
lines!(ax3c, t_grid, [model(θ_true, (t=t,)) for t in t_grid]; color=:black, linewidth=2, label="truth")
axislegend(ax3c; position=:rb)

display(fig3)
save("ex9_adaptive_comparison.png", fig3)

println("\nDone.")
