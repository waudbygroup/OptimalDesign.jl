module OptimalDesign

using ComponentArrays
using Distributions
using ForwardDiff
using LinearAlgebra
using LogExpFunctions: logsumexp
using Random
import Statistics
using Statistics: mean, median, quantile, std, var

import CairoMakie
import GLMakie

# Problem specification
export DesignProblem, select,
       DCriterion, ACriterion, ECriterion

# Posterior
export AbstractPosterior, Particles

# Design
export ExperimentalDesign, design, run_batch, run_adaptive,
       candidate_grid, uniform_allocation, n_obs, weights

# Results
export AbstractExperimentResult, BatchResult, AdaptiveResult, OptimalityResult

# Logging
export ExperimentLog, has_posterior_history,
       design_points, cumulative_cost, log_evidence_series

# Inference and analysis
export effective_sample_size,
       posterior_predictions, credible_band,
       efficiency, verify_optimality, gateaux_derivative,
       information, observation_diagnostics, update!,
       draw, score_candidates

# Plotting
export plot_corner, plot_residuals, record_corner_animation,
       plot_gateaux, plot_design_allocation, plot_credible_bands,
       plot_convergence, record_dashboard

# Types and problem definition
include("types.jl")
include("design_type.jl")
include("problem.jl")
include("sampling.jl")
include("information.jl")
include("utility.jl")
include("posteriors/particle.jl")
include("show.jl")
include("candidates.jl")
include("predictions.jl")

# Design optimisation
include("select.jl")
include("sequencing.jl")
include("exchange.jl")
include("gateaux.jl")
include("efficiency.jl")

# Experiment loop and logging
include("log.jl")
include("experiment.jl")

# Plotting
include("plotting/predictions.jl")
include("plotting/design.jl")
include("plotting/posterior.jl")
include("plotting/dashboard.jl")

end
