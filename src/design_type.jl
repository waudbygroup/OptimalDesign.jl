# --- ExperimentalDesign: iteration, display, and utility methods ---

# Iteration protocol
Base.iterate(d::ExperimentalDesign, args...) = iterate(d.allocation, args...)
Base.length(d::ExperimentalDesign) = length(d.allocation)
Base.getindex(d::ExperimentalDesign, i) = d.allocation[i]
Base.lastindex(d::ExperimentalDesign) = lastindex(d.allocation)
Base.eltype(::Type{ExperimentalDesign{T}}) where {T} = Tuple{T, Int}
Base.keys(d::ExperimentalDesign) = keys(d.allocation)
Base.isempty(d::ExperimentalDesign) = isempty(d.allocation)

"""
    n_obs(d::ExperimentalDesign)

Total number of measurements in the design.
"""
n_obs(d::ExperimentalDesign) = sum(c for (_, c) in d.allocation)

"""
    weights(d::ExperimentalDesign, candidates)

Convert design allocation to a weight vector over the candidate set.
Returns a vector of length `length(candidates)` summing to 1.
"""
function weights(d::ExperimentalDesign, candidates::AbstractVector)
    n = n_obs(d)
    w = zeros(length(candidates))
    for (ξ, count) in d.allocation
        idx = findfirst(c -> c == ξ, candidates)
        idx !== nothing && (w[idx] = count / n)
    end
    w
end

# --- Display ---

function Base.show(io::IO, d::ExperimentalDesign)
    n = n_obs(d)
    print(io, "ExperimentalDesign($n measurements, $(length(d.allocation)) support points)")
end

function Base.show(io::IO, ::MIME"text/plain", d::ExperimentalDesign)
    n = n_obs(d)
    println(io, "ExperimentalDesign: $n measurements at $(length(d.allocation)) support points")
    for (ξ, count) in d.allocation
        bar = repeat("█", count)
        vals = join(["$k=$(round(v; digits=4))" for (k, v) in pairs(ξ)], ", ")
        println(io, "  $vals  ×$count  $bar")
    end
end
