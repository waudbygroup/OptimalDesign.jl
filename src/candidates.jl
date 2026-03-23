"""
    candidate_grid(; kwargs...)

Generate candidate design points from keyword arguments. Each keyword maps
a design variable name to a collection of values; the result is the full
outer product as a vector of NamedTuples.

# Examples

```julia
candidate_grid(t = range(0, 0.5, length=200))
# → [(t=0.0,), (t=0.0025,), ..., (t=0.5,)]

candidate_grid(t = range(0, 0.5, 50), dose = [1, 10, 100])
# → [(t=0.0, dose=1), (t=0.0, dose=10), ..., (t=0.5, dose=100)]
```
"""
function candidate_grid(; kwargs...)
    ks = keys(kwargs)
    vals = values(kwargs)
    vec([(; zip(ks, combo)...) for combo in Iterators.product(vals...)])
end
