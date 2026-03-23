# --- Transformation ---

abstract type Transformation end

struct Identity <: Transformation end

struct DeltaMethod{F} <: Transformation
    f::F
    selected::Union{Nothing,Tuple{Vararg{Symbol}}}  # parameter names, if constructed via select()
end
DeltaMethod(f) = DeltaMethod(f, nothing)

# --- Design Criteria ---

abstract type DesignCriterion end

struct DCriterion <: DesignCriterion end
struct ACriterion <: DesignCriterion end
struct ECriterion <: DesignCriterion end

# --- DesignProblem ---

abstract type AbstractDesignProblem end

struct DesignProblem{F,J,S,T,CR<:DesignCriterion,C,K} <: AbstractDesignProblem
    predict::F
    jacobian::J
    sigma::S
    parameters::NamedTuple
    transformation::T
    criterion::CR
    cost::C              # x -> Real (per-measurement cost)
    constraint::K
end

struct SwitchingDesignProblem{F,J,S,T,CR<:DesignCriterion,C,K} <: AbstractDesignProblem
    predict::F
    jacobian::J
    sigma::S
    parameters::NamedTuple
    transformation::T
    criterion::CR
    cost::C              # x -> Real (per-measurement cost)
    switching_param::Symbol
    switching_cost::Float64
    constraint::K
end

# --- ExperimentalDesign ---

struct ExperimentalDesign{T<:NamedTuple}
    allocation::Vector{Tuple{T, Int}}
end

# --- Abstract posterior ---

abstract type AbstractPosterior end

# --- Particles ---

struct Particles{T} <: AbstractPosterior
    particles::Vector{T}
    log_weights::Vector{Float64}
end

# --- ExperimentLog ---

struct ExperimentLog
    history::Vector{NamedTuple}
    prior_snapshot::Union{Nothing,NamedTuple}   # (particles, log_weights) before any data
end

# --- GradientCache ---

struct GradientCache
    g_buf::Vector{Float64}
    cfg::ForwardDiff.GradientConfig
end

# --- LiveDashboard ---

mutable struct LiveDashboard
    fig::Any                    # GLMakie.Figure
    screen::Any                 # GLMakie.Screen
    obs_design_x::Any           # Observable: x coords of design points
    obs_design_y::Any           # Observable: y coords (observations)
    obs_posterior_vals::Any      # Observable: Dict of param name => values
    obs_posterior_weights::Any   # Observable: weights
    obs_info_gain::Any          # Observable: info gain per step
    obs_budget_spent::Any       # Observable: budget spent
    obs_budget_total::Any       # budget total (constant)
    obs_pred_lower::Any         # Observable: credible band lower
    obs_pred_median::Any        # Observable: credible band median
    obs_pred_upper::Any         # Observable: credible band upper
    control_state::Ref{Symbol}  # :running, :paused, :stopped
end

# --- Experiment results ---

abstract type AbstractExperimentResult end

struct BatchResult{P,D<:ExperimentalDesign} <: AbstractExperimentResult
    prior::P
    posterior::P
    design::D
    observations::Vector{NamedTuple}
end

struct AdaptiveResult{P} <: AbstractExperimentResult
    prior::P
    posterior::P
    log::ExperimentLog
    observations::Vector{NamedTuple}
end

# --- Optimality verification ---

struct OptimalityResult{T<:NamedTuple}
    is_optimal::Bool
    max_derivative::Float64
    dimension::Float64
    gateaux::Vector{Float64}
    candidates::Vector{T}
end
