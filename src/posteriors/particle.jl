include("liu_west.jl")
include("systematic.jl")
include("gmm.jl")

"""
    Particles(prob::AbstractDesignProblem, n::Int; resampling=LiuWestResampling())

Construct a Particles by drawing n particles from the prior.
All particles have equal weight. The resampling strategy defaults to
`LiuWestResampling()` (a=0.98, ess_threshold=0.5).
"""
function Particles(prob::AbstractDesignProblem, n::Int;
    resampling::ResamplingStrategy=LiuWestResampling())
    particles = draw(prob.parameters, n)
    log_weights = fill(-log(n), n)
    Particles(particles, log_weights, resampling)
end

"""
    sample(posterior::Particles, n::Int)

Draw n particles from the posterior (with replacement, proportional to weights).
"""
function sample(posterior::Particles, n::Int)
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    indices = systematic_resample(w, n)
    posterior.particles[indices]
end

"""
    mean(p::Particles)

Weighted mean of the particles. Returns a `ComponentArray` with one entry per parameter.
"""
function Statistics.mean(p::Particles)
    w = exp.(p.log_weights .- logsumexp(p.log_weights))
    sum(w[i] * p.particles[i] for i in eachindex(p.particles))
end

"""
    var(p::Particles)

Weighted variance of the particles. Returns a `ComponentArray` with one entry per parameter.
"""
function Statistics.var(p::Particles)
    w = exp.(p.log_weights .- logsumexp(p.log_weights))
    μ = Statistics.mean(p)
    sum(w[i] * (p.particles[i] .- μ) .^ 2 for i in eachindex(p.particles))
end

"""
    std(p::Particles)

Weighted standard deviation of the particles. Returns a `ComponentArray` with one entry per parameter.
"""
Statistics.std(p::Particles) = sqrt.(Statistics.var(p))

"""
    effective_sample_size(posterior::Particles)

Compute the effective sample size (ESS) of the weighted particles.
"""
function effective_sample_size(posterior::Particles)
    lw = posterior.log_weights .- logsumexp(posterior.log_weights)
    exp(-logsumexp(2 .* lw))
end

"""
    loglikelihood(prob::AbstractDesignProblem, θ, x, y)

Log-likelihood of observation y at (θ, x) under the noise model defined by prob.sigma.

Handles scalar, vector, and structured observations (NamedTuple with :value and :σ).
"""
function loglikelihood(prob::AbstractDesignProblem, θ, x, y)
    ŷ = prob.predict(θ, x)
    if y isa NamedTuple && haskey(y, :value) && haskey(y, :σ)
        return _loglikelihood_gaussian(y.value, ŷ, y.σ)
    end
    σ = prob.sigma(θ, x)
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

function _loglikelihood_gaussian(y::Real, ŷ::AbstractVector, σ)
    _loglikelihood_gaussian([y], ŷ, σ)
end

function _loglikelihood_gaussian(y::AbstractVector, ŷ::Real, σ)
    _loglikelihood_gaussian(y, [ŷ], σ isa Real ? [σ] : σ)
end

"""
    update!(posterior::Particles, prob::AbstractDesignProblem, x, y)

Incorporate a single observation y at design point x.
Delegates to the batch method for the active resampling strategy.
"""
function update!(posterior::Particles, prob::AbstractDesignProblem, x, y)
    update!(posterior, prob, [(x=x, y=y)])
end

"""
    update!(posterior::Particles, prob::AbstractDesignProblem, data::AbstractVector{<:NamedTuple})

Batch update. Dispatches to the strategy-specific implementation via `posterior.resampling`.
Each element of `data` must have fields `x` (design point) and `y` (observation).
"""
function update!(posterior::Particles, prob::AbstractDesignProblem,
        data::AbstractVector{<:NamedTuple})
    update!(posterior, prob, data, posterior.resampling)
end

"""
    observation_diagnostics(posterior, prob, x, y)

Score an observation against the current posterior to detect model deviations.

Returns `(mean_residual, log_marginal)`:
- `mean_residual`: posterior-weighted mean residual (y - ŷ)
- `log_marginal`: log marginal likelihood p(y | data so far)

A running series of `log_marginal` values constitutes sequential Bayesian
model checking. Sharp drops indicate observations surprising under the current model.
"""
function observation_diagnostics(posterior::Particles, prob::AbstractDesignProblem, x, y)
    n = length(posterior.particles)

    ll_terms = [
        posterior.log_weights[i] + loglikelihood(prob, posterior.particles[i], x, y)
        for i in 1:n
    ]
    log_ml = logsumexp(ll_terms)

    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    y_scalar = y isa NamedTuple ? y.value : y

    mean_pred = sum(
        w[i] * prob.predict(posterior.particles[i], x)
        for i in 1:n
    )
    mean_residual = y_scalar .- mean_pred

    (mean_residual=mean_residual, log_marginal=log_ml)
end
