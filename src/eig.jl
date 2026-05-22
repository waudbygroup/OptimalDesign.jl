"""
    simulate_observation(prob, θ, x)

Draw a synthetic observation `y ~ p(y | θ, x)` under the Gaussian noise model
defined by `prob.sigma`. Dispatches on σ scalar/vector/matrix in the same shape
as [`loglikelihood`](@ref).
"""
function simulate_observation(prob::AbstractDesignProblem, θ, x)
    ŷ = prob.predict(θ, x)
    σ = prob.sigma(θ, x)
    _simulate_gaussian(ŷ, σ)
end

_simulate_gaussian(ŷ::Real, σ::Real) = ŷ + σ * randn()

function _simulate_gaussian(ŷ::AbstractVector, σ::Real)
    ŷ .+ σ .* randn(length(ŷ))
end

function _simulate_gaussian(ŷ::AbstractVector, σ::AbstractVector)
    ŷ .+ σ .* randn(length(ŷ))
end

function _simulate_gaussian(ŷ::AbstractVector, Σ::AbstractMatrix)
    C = cholesky(Symmetric(Σ))
    ŷ .+ C.L * randn(length(ŷ))
end

_simulate_gaussian(ŷ::Real, σ::AbstractVector) = _simulate_gaussian([ŷ], σ)

"""
    _splice_selected(θ_m, ν_n, selected)

Build a `ComponentArray` that takes the `selected` fields from `ν_n` (inner sample)
and the remaining fields from `θ_m` (outer sample). Used by the marginal-EIG
contrastive estimator: the nuisance complement ν is held fixed at the outer value
ν_m while the parameters of interest τ are varied across inner samples τ_n.
"""
function _splice_selected(θ_m, ν_n, selected::Tuple{Vararg{Symbol}})
    pnames = keys(θ_m)
    vals = ntuple(length(pnames)) do i
        name = pnames[i]
        name in selected ? getproperty(ν_n, name) : getproperty(θ_m, name)
    end
    ComponentArray(NamedTuple{pnames}(vals))
end

"""
    eig_score(prob, particles, x; outer_samples, inner_samples) -> Float64

Estimate the Expected Information Gain at candidate `x` by nested Monte Carlo
over the supplied particles.

For `Identity` transformation this estimates `MI(θ; y | x)`. For
`DeltaMethod(_, selected)` it estimates the marginal information gain about
`τ(θ)` using the nuisance-permutation contrastive estimator — exact under
posterior independence of τ and the complement, biased when they couple.

`particles` should already be drawn from the current posterior (the solver
passes a weight-corrected resample).
"""
function eig_score(
    prob::AbstractDesignProblem,
    particles::AbstractVector,
    x;
    outer_samples::Int=50,
    inner_samples::Int=50,
)
    eig_score(prob.transformation, prob, particles, x; outer_samples, inner_samples)
end

# Total EIG: MI(θ; y | x)
function eig_score(
    ::Identity,
    prob::AbstractDesignProblem,
    particles::AbstractVector,
    x;
    outer_samples::Int,
    inner_samples::Int,
)
    n = length(particles)
    M = min(outer_samples, n)
    N = min(inner_samples, n)
    outer_idx = randperm(n)[1:M]
    inner_idx = randperm(n)[1:N]

    log_terms = Vector{Float64}(undef, N)
    total = 0.0
    count = 0
    for m in outer_idx
        θ_m = particles[m]
        y_m = simulate_observation(prob, θ_m, x)
        log_p_num = loglikelihood(prob, θ_m, x, y_m)
        for (j, ni) in enumerate(inner_idx)
            log_terms[j] = loglikelihood(prob, particles[ni], x, y_m)
        end
        log_p_den = logsumexp(log_terms) - log(N)
        contrib = log_p_num - log_p_den
        if isfinite(contrib)
            total += contrib
            count += 1
        end
    end
    count == 0 ? -Inf : total / count
end

# Marginal EIG on τ = MI(τ(θ); y | x) via nuisance permutation
function eig_score(
    dm::DeltaMethod,
    prob::AbstractDesignProblem,
    particles::AbstractVector,
    x;
    outer_samples::Int,
    inner_samples::Int,
)
    selected = dm.selected
    selected === nothing && throw(ArgumentError(
        "Marginal EIG requires a coordinate-selection DeltaMethod (built via " *
        "`select(:name, ...)`). Smooth transformations are not supported in v1; " *
        "use `Identity()` for total EIG or a FIM-based criterion (DCriterion etc.) " *
        "for general τ."))

    n = length(particles)
    M = min(outer_samples, n)
    N = min(inner_samples, n)
    outer_idx = randperm(n)[1:M]
    inner_idx = randperm(n)[1:N]

    log_terms = Vector{Float64}(undef, N)
    total = 0.0
    count = 0
    for m in outer_idx
        θ_m = particles[m]
        y_m = simulate_observation(prob, θ_m, x)
        log_p_num = loglikelihood(prob, θ_m, x, y_m)
        for (j, ni) in enumerate(inner_idx)
            θ_prime = _splice_selected(θ_m, particles[ni], selected)
            log_terms[j] = loglikelihood(prob, θ_prime, x, y_m)
        end
        log_p_den = logsumexp(log_terms) - log(N)
        contrib = log_p_num - log_p_den
        if isfinite(contrib)
            total += contrib
            count += 1
        end
    end
    count == 0 ? -Inf : total / count
end
