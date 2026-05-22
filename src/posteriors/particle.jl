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
    # Structured observation: use realised noise
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

# Scalar y with vector prediction or vice versa — promote
function _loglikelihood_gaussian(y::Real, ŷ::AbstractVector, σ)
    _loglikelihood_gaussian([y], ŷ, σ)
end

function _loglikelihood_gaussian(y::AbstractVector, ŷ::Real, σ)
    _loglikelihood_gaussian(y, [ŷ], σ isa Real ? [σ] : σ)
end

"""
    update!(posterior::Particles, prob::AbstractDesignProblem, x, y)

Incorporate observation y at design point x. Delegates to the batch method
with adaptive tempering, so even a single highly informative observation
is tempered in gracefully. Resampling parameters are read from `posterior.resampling`.
"""
function update!(posterior::Particles, prob::AbstractDesignProblem, x, y)
    update!(posterior, prob, [(x=x, y=y)])
end

"""
    update!(posterior::Particles, prob::AbstractDesignProblem, data::AbstractVector{<:NamedTuple})

Batch update using adaptive likelihood tempering (SMC sampler).

Computes each particle's total log-likelihood across all data, then raises
the tempering exponent β from 0 → 1 in adaptive steps. At each step, the
step size Δβ is chosen by bisection so that the ESS stays just above
`posterior.resampling.ess_threshold × n`, then particles are resampled according
to the strategy in `posterior.resampling`.

Each element of `data` must have fields `x` (design point) and `y` (observation).
"""
function update!(posterior::Particles, prob::AbstractDesignProblem,
    data::AbstractVector{<:NamedTuple})
    n = length(posterior.particles)
    target_ess = posterior.resampling.ess_threshold * n

    # Compute total log-likelihood for each particle
    total_ll = _compute_total_ll(posterior, prob, data)

    β = 0.0
    step = 0
    while β < 1.0
        step += 1
        remaining = 1.0 - β

        # Find largest Δβ ∈ (0, remaining] keeping trial ESS ≥ target
        Δβ = _bisect_Δβ(posterior.log_weights, total_ll, remaining, target_ess)

        # Apply the step
        for i in 1:n
            posterior.log_weights[i] += Δβ * total_ll[i]
        end
        lse = logsumexp(posterior.log_weights)
        posterior.log_weights .-= lse
        β += Δβ

        ess = effective_sample_size(posterior)
        # @debug "Tempering step $step: Δβ=$(round(Δβ; digits=4)), β=$(round(β; digits=4)), ESS=$(round(ess; digits=1))"

        if ess < target_ess
            resample!(posterior; prob=prob)
            total_ll = _compute_total_ll(posterior, prob, data)
        end
    end
    # @debug "Tempering complete in $step steps"
    posterior
end

"""Compute total log-likelihood of all data for each particle."""
function _compute_total_ll(posterior::Particles, prob::AbstractDesignProblem,
    data::AbstractVector{<:NamedTuple})
    n = length(posterior.particles)
    total_ll = Vector{Float64}(undef, n)
    for i in 1:n
        θ = posterior.particles[i]
        ll = 0.0
        for d in data
            ll += loglikelihood(prob, θ, d.x, d.y)
        end
        total_ll[i] = ll
    end
    total_ll
end

"""
Bisect for the largest Δβ ∈ (0, remaining] such that trial ESS ≥ target.
If even Δβ = remaining keeps ESS above target, return remaining (finish in one step).
"""
function _bisect_Δβ(log_weights::Vector{Float64}, total_ll::Vector{Float64},
    remaining::Float64, target_ess::Float64;
    max_iter::Int=30, tol::Float64=1e-6)
    # First check: can we take the full remaining step?
    trial_ess = _trial_ess(log_weights, total_ll, remaining)
    trial_ess >= target_ess && return remaining

    # Bisect between lo (safe) and hi (too aggressive)
    lo = 0.0
    hi = remaining
    for _ in 1:max_iter
        mid = (lo + hi) / 2
        (hi - lo) < tol && break
        if _trial_ess(log_weights, total_ll, mid) >= target_ess
            lo = mid
        else
            hi = mid
        end
    end
    # Return lo (the safe side); but ensure we make some progress
    max(lo, remaining * 1e-6)
end

"""Compute ESS that would result from adding Δβ * total_ll to log_weights, without modifying them."""
function _trial_ess(log_weights::Vector{Float64}, total_ll::Vector{Float64}, Δβ::Float64)
    n = length(log_weights)
    trial = Vector{Float64}(undef, n)
    for i in 1:n
        trial[i] = log_weights[i] + Δβ * total_ll[i]
    end
    lse = logsumexp(trial)
    trial .-= lse
    exp(-logsumexp(2 .* trial))
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

# --- Resampling strategy implementations ---

include("particles/liu_west.jl")
include("particles/systematic.jl")

# --- Observation diagnostics ---

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

    # Log marginal likelihood: logsumexp of weighted log-likelihoods
    ll_terms = [
        posterior.log_weights[i] + loglikelihood(prob, posterior.particles[i], x, y)
        for i in 1:n
    ]
    log_ml = logsumexp(ll_terms)

    # Posterior-weighted mean residual
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    y_scalar = y isa NamedTuple ? y.value : y

    mean_pred = sum(
        w[i] * prob.predict(posterior.particles[i], x)
        for i in 1:n
    )
    mean_residual = y_scalar .- mean_pred

    (mean_residual=mean_residual, log_marginal=log_ml)
end
