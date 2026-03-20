"""
    select(problem, candidates, posterior; kwargs...)

Unified interface for both batch and adaptive design.

Returns `Vector{Tuple{NamedTuple, Int}}` — (ξ, count) pairs.

**Batch design** (`n` large, prior as posterior): optimises weights via exchange algorithm.
**Adaptive design** (`n = 1` or small): scores by utility/cost, greedy multi-point selection.

# Keyword arguments
- `n = 1`: number of measurements to allocate
- `criterion = DCriterion()`: design criterion
- `budget = Inf`: total cost budget
- `posterior_samples = 0`: number of posterior samples for utility evaluation
- `ξ_prev = nothing`: previous design point (for cost computation)
- `exchange_algorithm = (n > 5)`: if true, use exchange algorithm for weight optimisation
- `exchange_steps = 100`: iterations for exchange algorithm
"""
function select(
    problem::DesignProblem,
    candidates::AbstractVector{<:NamedTuple},
    posterior;
    n::Int=1,
    criterion::DesignCriterion=DCriterion(),
    budget::Real=Inf,
    posterior_samples::Int=0,
    ξ_prev=nothing,
    exchange_algorithm::Bool=n > 5,
    exchange_steps::Int=100,
)
    particles = _get_particles(posterior)

    if posterior_samples ≤ 0
        posterior_samples = length(particles)
    end
    if posterior_samples > length(particles)
        posterior_samples = length(particles)
    end

    if exchange_algorithm
        _select_batch(problem, candidates, particles, n;
            criterion, posterior_samples, exchange_steps)
    else
        _select_greedy(problem, candidates, particles, n;
            criterion, posterior_samples, budget, ξ_prev)
    end
end

"""
Extract particles from posterior (supports ParticlePosterior or plain Vector).
"""
_get_particles(post::ParticlePosterior) = post.particles
_get_particles(particles::AbstractVector) = particles

"""
Greedy sequential selection: pick the best candidate, add its FIM to a
running total, then pick the next based on marginal gain.

When `n=1` (adaptive case), this is a simple argmax of utility/cost.
When `n>1`, the accumulated FIM ensures each successive pick adds
complementary information rather than repeating the same candidate.

Precomputes all per-particle FIMs to avoid redundant ForwardDiff calls.
"""
function _select_greedy(
    prob, candidates, particles, n;
    criterion, posterior_samples, budget, ξ_prev,
)
    selected = NamedTuple[]
    remaining_budget = budget
    prev = ξ_prev
    n_particles = length(particles)
    bs = min(posterior_samples, n_particles)
    idx = bs >= n_particles ? (1:n_particles) : randperm(n_particles)[1:bs]
    p = length(first(particles))
    K = length(candidates)

    # Precompute FIM for each (particle, candidate) pair — the expensive part
    # M_cache[ji][k] = information(prob, particles[idx[ji]], candidates[k])
    @debug "Precomputing FIM cache: $(length(idx)) particles × $K candidates"
    M_cache = Vector{Vector{Matrix{Float64}}}(undef, length(idx))
    for ji in eachindex(idx)
        θ = particles[idx[ji]]
        M_cache[ji] = [information(prob, θ, candidates[k]) for k in 1:K]
    end

    # Running FIM per particle: accumulates information from already-selected points
    M_running = [zeros(p, p) for _ in idx]

    for step in 1:n
        # Score each candidate by E[Φ(M_running + M_k)] / cost
        scores = fill(-Inf, K)
        for k in 1:K
            c = prob.cost(prev, candidates[k])
            if c > remaining_budget
                continue
            end

            total = 0.0
            count = 0
            for (ji, j) in enumerate(idx)
                M_new = M_running[ji] + M_cache[ji][k]
                Mt = transform(prob, M_new, particles[j])
                val = safe_criterion(criterion, Mt)
                if isfinite(val)
                    total += val
                    count += 1
                end
            end
            score = count == 0 ? -Inf : total / count
            scores[k] = score / max(c, eps())
        end

        best_idx = argmax(scores)
        best_score = scores[best_idx]

        best_score == -Inf && break

        ξ = candidates[best_idx]
        push!(selected, ξ)
        remaining_budget -= prob.cost(prev, ξ)
        prev = ξ

        # Update running FIM from cache (no recomputation)
        for ji in eachindex(idx)
            M_running[ji] .+= M_cache[ji][best_idx]
        end

        @debug "Greedy step $step: ξ=$(ξ), score=$(round(best_score; digits=4)), budget_left=$(round(remaining_budget; digits=2))"
    end

    _compress(selected)
end

"""
Batch selection via exchange algorithm.
"""
function _select_batch(
    prob, candidates, particles, n;
    criterion, posterior_samples, exchange_steps,
)
    @info "Running exchange algorithm for batch design..."
    weights = exchange(prob, candidates, particles;
        criterion=criterion,
        posterior_samples=posterior_samples,
        max_iter=exchange_steps)

    counts = apportion(weights, n)

    result = Tuple{eltype(candidates),Int}[]
    for k in eachindex(candidates)
        if counts[k] > 0
            push!(result, (candidates[k], counts[k]))
        end
    end
    result
end

"""
Compress a list of selected candidates into (ξ, count) pairs.
"""
function _compress(selected::Vector{<:NamedTuple})
    isempty(selected) && return Tuple{NamedTuple,Int}[]

    result = Tuple{eltype(selected),Int}[]
    current = selected[1]
    count = 1

    for i in 2:length(selected)
        if selected[i] == current
            count += 1
        else
            push!(result, (current, count))
            current = selected[i]
            count = 1
        end
    end
    push!(result, (current, count))
    result
end

"""
    uniform_allocation(candidates, n)

Allocate n measurements uniformly spaced across the candidate list.
Selects n evenly-spaced indices from the candidate vector.
Returns Vector{Tuple{NamedTuple, Int}}.
"""
function uniform_allocation(candidates::AbstractVector{<:NamedTuple}, n::Int)
    K = length(candidates)
    if n >= K
        # More measurements than candidates: distribute evenly
        counts = apportion(fill(1.0 / K, K), n)
        result = Tuple{eltype(candidates),Int}[]
        for k in eachindex(candidates)
            if counts[k] > 0
                push!(result, (candidates[k], counts[k]))
            end
        end
        return result
    end

    # Select n evenly-spaced indices
    indices = round.(Int, range(1, K, length=n))

    # Count duplicates (can happen when n is close to K)
    result = Tuple{eltype(candidates),Int}[]
    i = 1
    while i <= length(indices)
        idx = indices[i]
        c = 1
        while i + c <= length(indices) && indices[i+c] == idx
            c += 1
        end
        push!(result, (candidates[idx], c))
        i += c
    end
    result
end
