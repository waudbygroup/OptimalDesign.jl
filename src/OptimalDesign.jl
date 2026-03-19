module OptimalDesign

using ComponentArrays
using Distributions
using ForwardDiff
using LinearAlgebra
using LogExpFunctions: logsumexp
using Random
using Statistics

export DesignProblem, Identity, DeltaMethod, select,
       DCriterion, ACriterion, ECriterion,
       information, transform, weighted_fim,
       expected_utility, score_candidates,
       ParticlePosterior, sample, posterior_mean,
       effective_sample_size, update!, draw,
       loglikelihood

include("types.jl")
include("information.jl")
include("utility.jl")
include("posteriors/particle.jl")

end
