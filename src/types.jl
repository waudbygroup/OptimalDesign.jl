# --- Transformation ---

abstract type Transformation end

struct Identity <: Transformation end

struct DeltaMethod{F} <: Transformation
    f::F
end

"""
    select(names::Symbol...)

Convenience constructor for DeltaMethod that selects named parameters.
Returns a function that extracts the named components from a ComponentArray.
"""
function select(names::Symbol...)
    DeltaMethod(θ -> ComponentArray(NamedTuple{names}(ntuple(i -> getproperty(θ, names[i]), length(names)))))
end

# --- Design Criteria ---

abstract type DesignCriterion end

struct DCriterion <: DesignCriterion end
struct ACriterion <: DesignCriterion end
struct ECriterion <: DesignCriterion end

(::DCriterion)(M::AbstractMatrix) = logdet(Symmetric(M))
(::ACriterion)(M::AbstractMatrix) = -tr(inv(Symmetric(M)))
(::ECriterion)(M::AbstractMatrix) = eigmin(Symmetric(M))

# --- DesignProblem ---

struct DesignProblem{F,J,S,T,C,K}
    predict::F
    jacobian::J
    sigma::S
    parameters::NamedTuple
    transformation::T
    cost::C
    constraint::K
end

"""
    DesignProblem(predict; kwargs...)

Construct a DesignProblem with keyword arguments and sensible defaults.

- `jacobian`: (θ, ξ) -> J matrix, or `nothing` for ForwardDiff (default: `nothing`)
- `sigma`: (θ, ξ) -> noise (default: `Returns(1.0)`)
- `parameters`: NamedTuple of prior distributions (required)
- `transformation`: defaults to `Identity()`
- `cost`: (prev, ξ) -> time cost (default: `(prev, ξ) -> 1.0`)
- `constraint`: (ξ, θ) -> Bool (default: `(ξ, θ) -> true`)
"""
function DesignProblem(
    predict;
    jacobian = nothing,
    sigma = Returns(1.0),
    parameters::NamedTuple,
    transformation::Transformation = Identity(),
    cost = (prev, ξ) -> 1.0,
    constraint = (ξ, θ) -> true,
)
    DesignProblem(predict, jacobian, sigma, parameters, transformation, cost, constraint)
end

# --- Parameter utilities ---

"""
    draw(parameters::NamedTuple)

Draw a single sample from the prior, returning a ComponentArray.
"""
function draw(parameters::NamedTuple)
    vals = map(rand, parameters)
    ComponentArray(vals)
end

"""
    draw(parameters::NamedTuple, n::Int)

Draw n samples from the prior, returning a Vector of ComponentArrays.
"""
function draw(parameters::NamedTuple, n::Int)
    [draw(parameters) for _ in 1:n]
end
