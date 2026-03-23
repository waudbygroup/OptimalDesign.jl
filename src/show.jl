# --- Display methods ---
# Included after posteriors/particle.jl and problem.jl so mean/std/selected_parameters are available.

"""Format a parameter ComponentArray for compact display."""
function _format_params(θ)
    parts = [string(k, "=", round(getproperty(θ, k); digits=3))
             for k in keys(θ)]
    join(parts, ", ")
end

# --- Particles ---

function Base.show(io::IO, p::Particles)
    n = length(p.particles)
    ess = round(effective_sample_size(p); digits=1)
    print(io, "Particles($n, ESS=$ess)")
end

function Base.show(io::IO, ::MIME"text/plain", p::Particles)
    n = length(p.particles)
    ess = round(effective_sample_size(p); digits=1)
    μ = mean(p)
    σ = std(p)
    println(io, "Particles: $n samples, ESS=$ess")
    println(io, "  Mean: $(_format_params(μ))")
    print(io, "  Std:  $(_format_params(σ))")
end

# --- DesignProblem ---

function Base.show(io::IO, prob::DesignProblem)
    np = length(prob.parameters)
    sel = selected_parameters(prob)
    sel_str = sel !== nothing ? ", select($(join([":" * string(s) for s in sel], ", ")))" : ""
    print(io, "DesignProblem($(np) parameters$sel_str)")
end

function Base.show(io::IO, ::MIME"text/plain", prob::DesignProblem)
    np = length(prob.parameters)
    sel = selected_parameters(prob)
    crit = prob.criterion

    println(io, "DesignProblem")
    println(io, "  Parameters ($(np)):")
    for (k, v) in pairs(prob.parameters)
        marker = sel !== nothing && k in sel ? " *" : ""
        println(io, "    $k ~ $v$marker")
    end
    if sel !== nothing
        println(io, "  Estimating: $(join(sel, ", ")) ($(typeof(crit)))")
    else
        println(io, "  Estimating: all parameters ($(typeof(crit)))")
    end
    print(io, "  Jacobian: $(prob.jacobian === nothing ? "ForwardDiff" : "analytic")")
end

function Base.show(io::IO, prob::SwitchingDesignProblem)
    np = length(prob.parameters)
    print(io, "SwitchingDesignProblem($(np) parameters, switch on :$(prob.switching_param))")
end

function Base.show(io::IO, ::MIME"text/plain", prob::SwitchingDesignProblem)
    np = length(prob.parameters)
    sel = selected_parameters(prob)
    crit = prob.criterion

    println(io, "SwitchingDesignProblem")
    println(io, "  Parameters ($(np)):")
    for (k, v) in pairs(prob.parameters)
        marker = sel !== nothing && k in sel ? " *" : ""
        println(io, "    $k ~ $v$marker")
    end
    if sel !== nothing
        println(io, "  Estimating: $(join(sel, ", ")) ($(typeof(crit)))")
    else
        println(io, "  Estimating: all parameters ($(typeof(crit)))")
    end
    println(io, "  Switching: :$(prob.switching_param) costs $(prob.switching_cost)")
    print(io, "  Jacobian: $(prob.jacobian === nothing ? "ForwardDiff" : "analytic")")
end

# --- BatchResult ---

function Base.show(io::IO, r::BatchResult)
    print(io, "BatchResult($(length(r.observations)) obs, $(length(r.design.allocation)) support points)")
end

function Base.show(io::IO, ::MIME"text/plain", r::BatchResult)
    n = length(r.observations)
    ns = length(r.design.allocation)
    ess = round(effective_sample_size(r.posterior); digits=1)
    println(io, "BatchResult: $n observations from $ns support points")
    println(io, "  Mean: $(_format_params(mean(r.posterior)))")
    println(io, "  Std:  $(_format_params(std(r.posterior)))")
    print(io, "  ESS:  $ess")
end

# --- AdaptiveResult ---

function Base.show(io::IO, r::AdaptiveResult)
    print(io, "AdaptiveResult($(length(r.observations)) obs)")
end

function Base.show(io::IO, ::MIME"text/plain", r::AdaptiveResult)
    n = length(r.observations)
    ess = round(effective_sample_size(r.posterior); digits=1)
    cost = isempty(r.log) ? 0.0 : sum(e.cost for e in r.log)
    println(io, "AdaptiveResult: $n observations, total cost $(round(cost; digits=1))")
    println(io, "  Mean: $(_format_params(mean(r.posterior)))")
    println(io, "  Std:  $(_format_params(std(r.posterior)))")
    print(io, "  ESS:  $ess")
end

# --- OptimalityResult ---

function Base.show(io::IO, r::OptimalityResult)
    status = r.is_optimal ? "✓ optimal" : "✗ not optimal"
    print(io, "OptimalityResult($status, max=$(round(r.max_derivative; digits=4)), q=$(round(r.dimension; digits=4)))")
end

function Base.show(io::IO, ::MIME"text/plain", r::OptimalityResult)
    status = r.is_optimal ? "✓ Optimal" : "✗ Not optimal"
    println(io, "OptimalityResult: $status")
    println(io, "  Max derivative: $(round(r.max_derivative; digits=4))")
    print(io, "  Bound (q):      $(round(r.dimension; digits=4))")
end

# --- Convenience accessors on results ---

Statistics.mean(r::AbstractExperimentResult) = mean(r.posterior)
Statistics.std(r::AbstractExperimentResult) = std(r.posterior)
Statistics.var(r::AbstractExperimentResult) = var(r.posterior)
