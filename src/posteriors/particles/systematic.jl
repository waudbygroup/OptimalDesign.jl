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
