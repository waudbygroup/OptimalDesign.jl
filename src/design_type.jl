# --- ExperimentalDesign: iteration, display, and utility methods ---

# Iteration protocol
Base.iterate(ξ::ExperimentalDesign, args...) = iterate(ξ.allocation, args...)
Base.length(ξ::ExperimentalDesign) = length(ξ.allocation)
Base.getindex(ξ::ExperimentalDesign, i) = ξ.allocation[i]
Base.lastindex(ξ::ExperimentalDesign) = lastindex(ξ.allocation)
Base.eltype(::Type{ExperimentalDesign{T}}) where {T} = Tuple{T, Int}
Base.keys(ξ::ExperimentalDesign) = keys(ξ.allocation)
Base.isempty(ξ::ExperimentalDesign) = isempty(ξ.allocation)

"""
    n_obs(ξ::ExperimentalDesign)

Total number of measurements in the design.
"""
n_obs(ξ::ExperimentalDesign) = sum(c for (_, c) in ξ.allocation)

"""
    weights(ξ::ExperimentalDesign, candidates)

Convert design allocation to a weight vector over the candidate set.
Returns a vector of length `length(candidates)` summing to 1.
"""
function weights(ξ::ExperimentalDesign, candidates::AbstractVector)
    n = n_obs(ξ)
    w = zeros(length(candidates))
    for (x, count) in ξ.allocation
        idx = findfirst(c -> c == x, candidates)
        idx !== nothing && (w[idx] = count / n)
    end
    w
end

# --- Display ---

function Base.show(io::IO, ξ::ExperimentalDesign)
    n = n_obs(ξ)
    print(io, "ExperimentalDesign($n measurements, $(length(ξ.allocation)) support points)")
end

function Base.show(io::IO, ::MIME"text/plain", ξ::ExperimentalDesign)
    n = n_obs(ξ)
    println(io, "ExperimentalDesign: $n measurements at $(length(ξ.allocation)) support points")
    isempty(ξ.allocation) && return
    rows = [(join(["$k=$(round(v; digits=4))" for (k, v) in pairs(x)], ", "), count)
            for (x, count) in ξ.allocation]
    max_val = maximum(length(r[1]) for r in rows)
    max_cnt = maximum(ndigits(r[2]) for r in rows)
    for (vals, count) in rows
        bar = repeat("█", count)
        println(io, "  ", rpad(vals, max_val), "  ×", lpad(string(count), max_cnt), "  ", bar)
    end
end

