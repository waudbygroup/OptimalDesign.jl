"""
    efficiency(ξ_a, ξ_b, prob, candidates, posterior; kwargs...)

Relative efficiency of design `ξ_a` vs design `ξ_b`.

For D-optimality: (det M_a / det M_b)^(1/q) where q = dimension of interest.
For A-optimality: Φ_b / Φ_a (both negative).
For E-optimality: Φ_a / Φ_b.

For `EIGCriterion`, returns the difference `EIG_a − EIG_b` in nats rather than
a ratio — EIG is already on an additive (log) scale, and there is no natural
exponent like `1/q` to render it dimensionless. Positive means ξ_a is better.

Efficiency > 1 (or > 0 for EIG) means ξ_a is better; < 1 (< 0) means ξ_b is better.
"""
function efficiency(
    ξ_a::ExperimentalDesign,
    ξ_b::ExperimentalDesign,
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    posterior::Particles;
    posterior_samples::Int=50,
)
    particles = _get_particles(posterior)
    eff = efficiency(weights(ξ_a, candidates), weights(ξ_b, candidates),
        prob, candidates, particles; posterior_samples=posterior_samples)
    if prob.criterion isa EIGCriterion
        @info "EIG advantage of A vs B: $(round(eff; digits=4)) nats"
    else
        extra = eff < 1 ? " (A needs ~$(round(1/eff; digits=1))× more measurements to match B)" : ""
        @info "Efficiency of A vs B: $(round(eff; digits=4))$extra"
    end
    eff
end

function efficiency(
    weights_a::AbstractVector,
    weights_b::AbstractVector,
    prob::AbstractDesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    particles::AbstractVector;
    posterior_samples::Int=50,
)
    criterion = prob.criterion
    # Compute average criterion value for each design
    Φ_a = _average_criterion(prob, candidates, particles, weights_a;
        criterion=criterion, posterior_samples=posterior_samples)
    Φ_b = _average_criterion(prob, candidates, particles, weights_b;
        criterion=criterion, posterior_samples=posterior_samples)

    _efficiency(criterion, Φ_a, Φ_b, _transformed_dimension(prob))
end

"""
Compute the average criterion value for a given weight vector. For FIM-based
criteria this is `E_θ[Φ(M_τ(w,θ))]`. For `EIGCriterion`, EIG has no weighted
FIM analogue, so this returns the weight-averaged per-candidate EIG
`Σ w_k · EIG(x_k)` — i.e., expected single-measurement EIG under the design's
allocation distribution.
"""
function _average_criterion(
    prob, candidates, particles, weights;
    criterion, posterior_samples,
)
    _average_criterion(criterion, prob, candidates, particles, weights;
        posterior_samples=posterior_samples)
end

function _average_criterion(
    criterion::DesignCriterion, prob, candidates, particles, weights;
    posterior_samples,
)
    n_particles = length(particles)
    bs = min(posterior_samples, n_particles)
    idx = randperm(n_particles)[1:bs]

    total = 0.0
    count = 0

    for j in idx
        θ = particles[j]
        M_w = _particle_weighted_fim(prob, θ, candidates, weights)
        Mt = transform(prob, M_w, θ)
        val = safe_criterion(criterion, Mt)
        if isfinite(val)
            total += val
            count += 1
        end
    end

    count == 0 ? -Inf : total / count
end

function _average_criterion(
    criterion::EIGCriterion, prob, candidates, particles, weights;
    posterior_samples,
)
    total = 0.0
    wsum = 0.0
    for k in eachindex(candidates)
        weights[k] > 1e-10 || continue
        s = eig_score(prob, particles, candidates[k];
            outer_samples=criterion.outer_samples,
            inner_samples=criterion.inner_samples)
        if isfinite(s)
            total += weights[k] * s
            wsum += weights[k]
        end
    end
    wsum > 0 ? total / wsum : -Inf
end

function _efficiency(::DCriterion, Φ_a, Φ_b, q)
    # Φ = log det M, so (det_a / det_b)^(1/q) = exp((Φ_a - Φ_b) / q)
    exp((Φ_a - Φ_b) / q)
end

function _efficiency(::ACriterion, Φ_a, Φ_b, q)
    # Φ = -tr(M⁻¹), so efficiency = Φ_b / Φ_a (both negative, ratio > 1 means a is better)
    Φ_b / Φ_a
end

function _efficiency(::ECriterion, Φ_a, Φ_b, q)
    # Φ = λ_min, efficiency = Φ_a / Φ_b
    Φ_a / Φ_b
end

function _efficiency(::EIGCriterion, Φ_a, Φ_b, q)
    # EIG is already on a log/additive scale (nats). No exponent or ratio.
    Φ_a - Φ_b
end
