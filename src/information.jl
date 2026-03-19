"""
    weighted_fim(J, σ)

Compute J'Σ⁻¹J from Jacobian J and noise specification σ.

- Scalar σ: F = J'J / σ²
- Vector σ: F = J' diag(1 ./ σ.²) J
- Matrix Σ: F = J' Σ⁻¹ J
"""
function weighted_fim(J::AbstractMatrix, σ::Real)
    return J' * J / σ^2
end

function weighted_fim(J::AbstractMatrix, σ::AbstractVector)
    W = Diagonal(1 ./ σ .^ 2)
    return J' * W * J
end

function weighted_fim(J::AbstractMatrix, Σ::AbstractMatrix)
    Σ_inv = inv(Σ)
    return J' * Σ_inv * J
end

# Handle scalar predict (1D Jacobian comes back as a vector, reshape to 1×p)
function weighted_fim(J::AbstractVector, σ)
    return weighted_fim(reshape(J, 1, length(J)), σ)
end

"""
    information(prob, θ, ξ)

Compute the Fisher Information Matrix at (θ, ξ) for the given DesignProblem.

Dispatches on whether an analytic Jacobian is provided or ForwardDiff is used.
"""
function information(prob::DesignProblem, θ, ξ)
    J = if prob.jacobian === nothing
        # ForwardDiff: differentiate predict w.r.t. θ
        y = prob.predict(θ, ξ)
        if y isa Real
            # Scalar observation: gradient -> reshape to 1×p
            g = ForwardDiff.gradient(θ_ -> prob.predict(θ_, ξ), θ)
            reshape(g, 1, length(g))
        else
            ForwardDiff.jacobian(θ_ -> prob.predict(θ_, ξ), θ)
        end
    else
        prob.jacobian(θ, ξ)
    end
    σ = prob.sigma(θ, ξ)
    weighted_fim(J, σ)
end

"""
    transform(prob, M)

Apply the transformation to the information matrix.

- Identity: returns M unchanged.
- DeltaMethod: computes [∇τ M⁻¹ ∇τ']⁻¹ via ForwardDiff.
"""
function transform(prob::DesignProblem, M::AbstractMatrix, θ)
    transform(prob.transformation, M, θ)
end

function transform(::Identity, M::AbstractMatrix, θ)
    M
end

function transform(dm::DeltaMethod, M::AbstractMatrix, θ)
    ∇τ = ForwardDiff.jacobian(dm.f, θ)
    # [∇τ M⁻¹ ∇τ']⁻¹
    inv(∇τ * inv(M) * ∇τ')
end
