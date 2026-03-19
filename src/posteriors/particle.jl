"""
    ParticlePosterior{T}

Weighted particle approximation to a posterior distribution.
Particles are ComponentArrays, weights are in log-space.
"""
struct ParticlePosterior{T}
    particles::Vector{T}
    log_weights::Vector{Float64}
end

"""
    ParticlePosterior(prob::DesignProblem, n::Int)

Construct a ParticlePosterior by drawing n particles from the prior.
All particles have equal weight.
"""
function ParticlePosterior(prob::DesignProblem, n::Int)
    particles = draw(prob.parameters, n)
    log_weights = fill(-log(n), n)
    ParticlePosterior(particles, log_weights)
end

"""
    sample(posterior::ParticlePosterior, n::Int)

Draw n particles from the posterior (with replacement, proportional to weights).
"""
function sample(posterior::ParticlePosterior, n::Int)
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    indices = systematic_resample(w, n)
    posterior.particles[indices]
end

"""
    posterior_mean(posterior::ParticlePosterior)

Compute the weighted mean of the posterior particles.
"""
function posterior_mean(posterior::ParticlePosterior)
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    result = sum(w[i] * posterior.particles[i] for i in eachindex(posterior.particles))
    result
end

"""
    effective_sample_size(posterior::ParticlePosterior)

Compute the effective sample size (ESS) of the weighted particles.
"""
function effective_sample_size(posterior::ParticlePosterior)
    lw = posterior.log_weights .- logsumexp(posterior.log_weights)
    exp(-logsumexp(2 .* lw))
end

"""
    loglikelihood(prob::DesignProblem, θ, ξ, y)

Log-likelihood of observation y at (θ, ξ) under the noise model defined by prob.sigma.

Handles scalar, vector, and structured observations (NamedTuple with :value and :σ).
"""
function loglikelihood(prob::DesignProblem, θ, ξ, y)
    ŷ = prob.predict(θ, ξ)
    # Structured observation: use realised noise
    if y isa NamedTuple && haskey(y, :value) && haskey(y, :σ)
        return _loglikelihood_gaussian(y.value, ŷ, y.σ)
    end
    σ = prob.sigma(θ, ξ)
    _loglikelihood_gaussian(y, ŷ, σ)
end

function _loglikelihood_gaussian(y::Real, ŷ::Real, σ::Real)
    -0.5 * log(2π) - log(σ) - 0.5 * ((y - ŷ) / σ)^2
end

function _loglikelihood_gaussian(y::AbstractVector, ŷ::AbstractVector, σ::AbstractVector)
    n = length(y)
    -0.5 * n * log(2π) - sum(log.(σ)) - 0.5 * sum(((y .- ŷ) ./ σ) .^ 2)
end

function _loglikelihood_gaussian(y::AbstractVector, ŷ::AbstractVector, Σ::AbstractMatrix)
    n = length(y)
    r = y .- ŷ
    -0.5 * n * log(2π) - 0.5 * logdet(Σ) - 0.5 * r' * inv(Σ) * r
end

# Scalar y with vector prediction or vice versa — promote
function _loglikelihood_gaussian(y::Real, ŷ::AbstractVector, σ)
    _loglikelihood_gaussian([y], ŷ, σ)
end

function _loglikelihood_gaussian(y::AbstractVector, ŷ::Real, σ)
    _loglikelihood_gaussian(y, [ŷ], σ isa Real ? [σ] : σ)
end

"""
    update!(posterior::ParticlePosterior, prob::DesignProblem, ξ, y; ess_threshold=0.5)

Incorporate observation y at design point ξ by reweighting particles.
Triggers systematic resampling with kernel jittering when ESS drops below threshold.
"""
function update!(posterior::ParticlePosterior, prob::DesignProblem, ξ, y; ess_threshold::Float64=0.5)
    n = length(posterior.particles)
    for i in 1:n
        ll = loglikelihood(prob, posterior.particles[i], ξ, y)
        posterior.log_weights[i] += ll
    end
    # Normalize
    lse = logsumexp(posterior.log_weights)
    posterior.log_weights .-= lse

    # Check ESS and resample if needed
    ess = effective_sample_size(posterior)
    if ess < ess_threshold * n
        resample!(posterior)
    end
    posterior
end

"""
    systematic_resample(weights, n)

Systematic resampling: returns n indices sampled proportional to weights.
"""
function systematic_resample(weights::AbstractVector, n::Int)
    cumw = cumsum(weights)
    u = rand() / n
    indices = Vector{Int}(undef, n)
    j = 1
    for i in 1:n
        target = u + (i - 1) / n
        while j < length(cumw) && cumw[j] < target
            j += 1
        end
        indices[i] = j
    end
    indices
end

"""
    resample!(posterior::ParticlePosterior; jitter_scale=0.01)

Systematic resampling with optional kernel jittering to prevent particle impoverishment.
"""
function resample!(posterior::ParticlePosterior; jitter_scale::Float64=0.01)
    n = length(posterior.particles)
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    indices = systematic_resample(w, n)

    new_particles = posterior.particles[indices]

    # Kernel jittering: add small Gaussian noise proportional to weighted std
    if jitter_scale > 0
        # Compute weighted standard deviation per component
        μ = sum(w[i] * posterior.particles[i] for i in 1:n)
        var_est = sum(w[i] * (posterior.particles[i] .- μ) .^ 2 for i in 1:n)
        σ_jitter = jitter_scale .* sqrt.(max.(var_est, 1e-20))
        for i in 1:n
            new_particles[i] = new_particles[i] .+ σ_jitter .* randn(length(σ_jitter))
        end
    end

    copyto!(posterior.particles, new_particles)
    fill!(posterior.log_weights, -log(n))
    posterior
end
