"""
    expected_utility(prob, criterion, particles, ξ; batch_size=50)

Compute the expected utility of design point ξ by Monte Carlo over posterior particles.

Uses mini-batch evaluation: randomly samples `batch_size` particles for an unbiased
but lower-variance estimate.
"""
function expected_utility(prob::DesignProblem, criterion::DesignCriterion, particles::AbstractVector, ξ; batch_size::Int=50)
    n = length(particles)
    bs = min(batch_size, n)
    idx = randperm(n)[1:bs]
    total = 0.0
    count = 0
    for i in idx
        θ = particles[i]
        M = information(prob, θ, ξ)
        Mt = transform(prob, M, θ)
        val = try
            criterion(Mt)
        catch
            nothing
        end
        if val !== nothing && isfinite(val)
            total += val
            count += 1
        end
    end
    count == 0 ? -Inf : total / count
end

"""
    score_candidates(prob, criterion, particles, candidates; batch_size=50)

Score all candidates by expected utility. Returns a vector of scores.
"""
function score_candidates(prob::DesignProblem, criterion::DesignCriterion, particles::AbstractVector, candidates::AbstractVector; batch_size::Int=50)
    [expected_utility(prob, criterion, particles, ξ; batch_size=batch_size) for ξ in candidates]
end
