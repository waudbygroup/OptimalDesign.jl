"""
    LiuWestResampling(; a=0.98, ess_threshold=0.5)

Systematic resampling followed by Liu-West kernel jittering in (optionally
transformed) unconstrained parameter space.

- `a`: shrinkage coefficient. Controls jitter magnitude: h² = 1 − a².
- `ess_threshold`: fraction of N below which resampling is triggered.
"""
struct LiuWestResampling <: ResamplingStrategy
    a::Float64
    ess_threshold::Float64
end
LiuWestResampling(; a::Float64=0.98, ess_threshold::Float64=0.5) =
    LiuWestResampling(a, ess_threshold)

export LiuWestResampling

"""
    _param_transforms(parameters::NamedTuple)

Derive (forward, inverse) transform pairs from prior distributions.
Forward maps to unconstrained space; inverse maps back.
"""
function _param_transforms(parameters::NamedTuple)
    map(parameters) do dist
        sup = support(dist)
        lo = minimum(sup)
        hi = maximum(sup)
        if lo == -Inf && hi == Inf
            # Unbounded (e.g., Normal)
            (forward=identity, inverse=identity)
        elseif isfinite(lo) && hi == Inf
            # Lower-bounded (e.g., LogUniform, Exponential)
            (forward=x -> log(x - lo + eps()), inverse=z -> lo + exp(z))
        elseif lo == -Inf && isfinite(hi)
            # Upper-bounded (rare)
            (forward=x -> -log(hi - x + eps()), inverse=z -> hi - exp(-z))
        else
            # Bounded [lo, hi] (e.g., Uniform, Beta)
            (forward=x -> log((x - lo + eps()) / (hi - x + eps())),
                inverse=z -> lo + (hi - lo) / (1 + exp(-z)))
        end
    end
end

"""
    resample!(posterior; prob=nothing)

Resample particles according to the strategy stored in `posterior.resampling`.

- `LiuWestResampling`: systematic resampling followed by shrink-and-noise jitter in
  (optionally transformed) unconstrained space, preserving the posterior's first two moments.
- `SystematicResampling`: plain systematic resampling, weights reset to uniform.

`prob` is required by `LiuWestResampling` for bound-aware parameter transforms; it is
optional (and ignored) for `SystematicResampling`.
"""
function resample!(posterior::Particles{T, LiuWestResampling};
        prob::Union{AbstractDesignProblem,Nothing}=nothing) where {T}
    a = posterior.resampling.a
    n = length(posterior.particles)
    d = length(first(posterior.particles))
    w = exp.(posterior.log_weights .- logsumexp(posterior.log_weights))
    indices = systematic_resample(w, n)

    new_particles = posterior.particles[indices]

    # Liu-West kernel: shrink + correlated noise preserving moments
    h² = 1 - a^2
    h = sqrt(h²)

    # Get parameter transforms (identity if no prob)
    transforms = if prob !== nothing
        _param_transforms(prob.parameters)
    else
        nothing
    end

    pnames = keys(first(posterior.particles))

    # Transform particles to unconstrained space
    Z = Matrix{Float64}(undef, d, n)  # columns are particles in transformed space
    for i in 1:n
        θ = posterior.particles[i]
        for (ki, k) in enumerate(pnames)
            val = getproperty(θ, k)
            Z[ki, i] = transforms !== nothing ? transforms[ki].forward(val) : val
        end
    end

    # Weighted mean and covariance in transformed space
    μ_z = Z * w
    Z_centered = Z .- μ_z
    Σ_z = (Z_centered .* w') * Z_centered'

    # Cholesky with regularisation
    C = cholesky(Symmetric(Σ_z + 1e-8 * I); check=false)
    if issuccess(C)
        L = C.L
    else
        # Fallback: diagonal jitter
        @warn "Liu-West: covariance Cholesky failed, falling back to diagonal jitter"
        L = Diagonal(sqrt.(max.(diag(Σ_z), 1e-20)))
    end

    # Apply Liu-West to resampled particles
    for i in 1:n
        # Get resampled particle in transformed space
        θ_old = new_particles[i]
        z_i = Vector{Float64}(undef, d)
        for (ki, k) in enumerate(pnames)
            z_i[ki] = transforms !== nothing ? transforms[ki].forward(getproperty(θ_old, k)) : getproperty(θ_old, k)
        end

        # Shrink toward mean + correlated noise
        m_i = a .* z_i .+ (1 - a) .* μ_z
        z_new = m_i .+ h .* (L * randn(d))

        # Back-transform to original space
        vals = ntuple(d) do ki
            z = z_new[ki]
            transforms !== nothing ? transforms[ki].inverse(z) : z
        end
        new_particles[i] = ComponentArray(NamedTuple{pnames}(vals))
    end

    copyto!(posterior.particles, new_particles)
    fill!(posterior.log_weights, -log(n))

    @debug "Resampled with Liu-West kernel (a=$a, h=$(round(h; digits=4)))"
    posterior
end
