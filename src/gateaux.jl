"""
    gateaux_derivative(prob, candidates, particles, weights; kwargs...)

Compute the Gateaux derivative of the expected design criterion at each
candidate, for a design with the given weights.

For D-optimality (Identity):  d(ξ) = tr(M⁻¹ M(ξ))
For Ds-optimality (DeltaMethod): d(ξ) = tr(C M(ξ)) where C = M⁻¹ ∇τ' Mτ ∇τ M⁻¹

For A/E criteria, numerical differentiation is used.

Returns a vector of derivatives (one per candidate).
"""
function gateaux_derivative(
    prob::DesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    particles::AbstractVector,
    weights::AbstractVector;
    criterion::DesignCriterion=DCriterion(),
    posterior_samples::Int=50,
)
    K = length(candidates)
    n_particles = length(particles)
    bs = min(posterior_samples, n_particles)
    idx = bs >= n_particles ? (1:n_particles) : randperm(n_particles)[1:bs]
    p = length(first(particles))

    gd = zeros(K)
    count = 0

    for j in idx
        θ = particles[j]

        # Build weighted FIM in full parameter space
        M_w = _particle_weighted_fim(prob, θ, candidates, weights)

        C = cholesky(Symmetric(M_w); check=false)
        if !issuccess(C)
            continue
        end

        count += 1

        # Compute per-candidate Gateaux derivatives for this particle
        gd .+= _gateaux_for_particle(criterion, prob, θ, M_w, candidates)
    end

    count == 0 ? fill(-Inf, K) : gd ./ count
end

"""
Build the weighted FIM for a single particle θ: M_w(θ) = Σ_k w_k M_k(θ).
Returns a p×p matrix in the full parameter space (no transformation).
"""
function _particle_weighted_fim(prob, θ, candidates, weights)
    p = length(θ)
    M_w = zeros(p, p)
    for k in eachindex(candidates)
        if weights[k] > 1e-10
            M_w .+= weights[k] .* information(prob, θ, candidates[k])
        end
    end
    M_w
end

# --- D-criterion: analytical Gateaux derivative ---

function _gateaux_for_particle(::DCriterion, prob, θ, M_w, candidates)
    M_w_inv = inv(Symmetric(M_w))

    # Precompute the "sensitivity" matrix C such that d_k = tr(C M_k)
    C = _d_sensitivity_matrix(prob, M_w_inv, θ)

    map(candidates) do ξ
        M_k = information(prob, θ, ξ)
        tr(C * M_k)
    end
end

"""
For D-optimality with Identity: C = M⁻¹
For Ds-optimality with DeltaMethod: C = M⁻¹ ∇τ' Mτ ∇τ M⁻¹

In both cases, d(ξ) = tr(C M(ξ)).
"""
function _d_sensitivity_matrix(prob, M_w_inv, θ)
    _d_sensitivity_matrix(prob.transformation, M_w_inv, θ)
end

function _d_sensitivity_matrix(::Identity, M_w_inv, θ)
    M_w_inv
end

function _d_sensitivity_matrix(dm::DeltaMethod, M_w_inv, θ)
    ∇τ = ForwardDiff.jacobian(dm.f, θ)
    # Mτ = (∇τ M⁻¹ ∇τ')⁻¹
    Mt = inv(Symmetric(∇τ * M_w_inv * ∇τ'))
    # C = M⁻¹ ∇τ' Mτ ∇τ M⁻¹
    M_w_inv * ∇τ' * Mt * ∇τ * M_w_inv
end

# --- A-criterion and E-criterion: numerical Gateaux derivative ---

function _gateaux_for_particle(criterion::DesignCriterion, prob, θ, M_w, candidates)
    Mt = transform(prob, M_w, θ)
    Φ0 = safe_criterion(criterion, Mt)
    isfinite(Φ0) || return fill(-Inf, length(candidates))

    ε = 1e-6
    map(candidates) do ξ
        M_k = information(prob, θ, ξ)
        Mt_ε = transform(prob, M_w + ε * M_k, θ)
        Φ_ε = safe_criterion(criterion, Mt_ε)
        isfinite(Φ_ε) ? (Φ_ε - Φ0) / ε : -Inf
    end
end

# --- Optimality dimension ---

"""
Dimension q of the parameter space of interest.
For D-optimality, the GEQ bound is d(ξ) ≤ q at all candidates.
"""
function _transformed_dimension(prob)
    if prob.transformation isa Identity
        Float64(length(keys(prob.parameters)))
    else
        θ = draw(prob.parameters)
        ∇τ = ForwardDiff.jacobian(prob.transformation.f, θ)
        Float64(size(∇τ, 1))
    end
end

# --- Optimality verification ---

"""
    verify_optimality(prob, candidates, particles, weights; kwargs...)

Check the General Equivalence Theorem: at an optimal design, the
Gateaux derivative should be ≤ q (dimension of interest) at all candidates,
with equality at support points.

Returns `(is_optimal, max_derivative, dimension)`.
"""
function verify_optimality(
    prob::DesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    particles::AbstractVector,
    weights::AbstractVector;
    criterion::DesignCriterion=DCriterion(),
    posterior_samples::Int=50,
    tol::Float64=0.05,
)
    gd = gateaux_derivative(prob, candidates, particles, weights;
        criterion=criterion, posterior_samples=posterior_samples)

    q = _transformed_dimension(prob)
    max_gd = maximum(gd)

    (is_optimal=max_gd ≤ q + tol,
        max_derivative=max_gd,
        dimension=q)
end
