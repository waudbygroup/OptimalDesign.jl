"""
    SystematicResampling(; ess_threshold=0.5)

Pure systematic resampling with no jitter. Weights are reset to uniform after
resampling. Useful as a baseline or when particle diversity is already sufficient.

- `ess_threshold`: fraction of N below which resampling is triggered.
"""
struct SystematicResampling <: ResamplingStrategy
    ess_threshold::Float64
end
SystematicResampling(; ess_threshold::Float64=0.5) = SystematicResampling(ess_threshold)

export SystematicResampling

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

function resample!(posterior::Particles{T, SystematicResampling};
        prob::Union{AbstractDesignProblem,Nothing}=nothing) where {T}
    n = length(posterior.particles)
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    indices = systematic_resample(w, n)
    copyto!(posterior.particles, posterior.particles[indices])
    fill!(posterior.log_weights, -log(n))
    @debug "Resampled (systematic, no jitter)"
    posterior
end

"""
    update!(posterior, prob, data, ::SystematicResampling)

Direct importance weighting update. Applies likelihoods observation-by-observation,
normalises, and resamples when ESS drops below `ess_threshold × n`.
No likelihood tempering — simpler semantics suitable for baseline comparisons.
"""
function update!(posterior::Particles{T, SystematicResampling}, prob::AbstractDesignProblem,
        data::AbstractVector{<:NamedTuple}, strategy::SystematicResampling) where {T}
    n = length(posterior.particles)
    for d in data
        for i in 1:n
            posterior.log_weights[i] += loglikelihood(prob, posterior.particles[i], d.x, d.y)
        end
        posterior.log_weights .-= logsumexp(posterior.log_weights)
        if effective_sample_size(posterior) < strategy.ess_threshold * n
            resample!(posterior; prob=prob)
        end
    end
    posterior
end
