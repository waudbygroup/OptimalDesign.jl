# Defining Problems

A `DesignProblem` bundles everything the design algorithm needs to know: the model, the noise structure, prior parameter uncertainty, what you want to learn, and any cost constraints.

## Minimal example

```julia
prob = DesignProblem(
    (θ, x) -> θ.A * exp(-θ.R₂ * x.t),
    parameters = (A = LogUniform(0.1, 10), R₂ = Uniform(1, 50)),
    sigma = Returns(0.1),
)
```

The only required arguments are the prediction function and `parameters`. Everything else has sensible defaults.

## Constructor arguments

### `predict` (positional, required)

A function `(θ, x) -> y` mapping parameters `θ` and design point `x` to a predicted observation. Both `θ` and `x` are accessed by named fields.

Scalar output:
```julia
(θ, x) -> θ.A * exp(-θ.R₂ * x.t)
```

Vector output (multiple simultaneous observables):
```julia
(θ, x) -> [θ.A₁ * exp(-θ.R₂₁ * x.t),
            θ.A₂ * exp(-θ.R₂₂ * x.t)]
```

### `parameters` (keyword, required)

A `NamedTuple` of `Distributions.Distribution` objects specifying the prior on each parameter:

```julia
parameters = (A = LogUniform(0.1, 10), R₂ = Uniform(1, 50))
```

The parameter names define the fields available on `θ` inside `predict` and `sigma`. Any distribution from [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) works.

### `sigma` (keyword, default: constant 1)

A callable `(θ, x) -> σ` giving the noise standard deviation. Can depend on both parameters and design point. For constant noise, use `Returns(σ)`:

```julia
sigma = Returns(0.1)                       # constant noise
sigma = (θ, x) -> 0.05 * abs(θ.A)         # signal-dependent noise
sigma = Returns([0.1, 0.2])               # constant vector noise (for vector models)
sigma = (θ, x) -> [0.1, 0.2]              # vector noise that could depend on θ, x
```

For vector models, `sigma` should return a vector of the same length as `predict`.

### `transformation` (keyword, default: `Identity()`)

Controls which parameters you want to estimate precisely.

- `Identity()` — full D-optimality over all parameters (default)
- `select(:R₂)` — Ds-optimality for a single parameter
- `select(:R₂₁, :R₂₂)` — Ds-optimality for a subset

`select` constructs a `DeltaMethod` transformation that projects the Fisher information onto the parameters of interest.

### `criterion` (keyword, default: `DCriterion()`)

The optimality criterion:

| Criterion | Objective | Interpretation |
|-----------|-----------|---------------|
| `DCriterion()` | maximise `log det(M)` | minimise volume of confidence ellipsoid |
| `ACriterion()` | maximise `-tr(M⁻¹)` | minimise average variance |
| `ECriterion()` | maximise `λ_min(M)` | minimise worst-case variance |

### `jacobian` (keyword, default: `nothing`)

An optional analytic Jacobian `(θ, x) -> J` where `J` is a `1×p` matrix (scalar model) or `m×p` matrix (vector model). If omitted, the Jacobian is computed automatically via ForwardDiff.

```julia
jacobian = (θ, x) -> begin
    e = exp(-θ.R₂ * x.t)
    [θ.A * x.t * e   -θ.A * e]   # [∂y/∂A  ∂y/∂R₂]  — but transposed to 1×2
end
```

Providing an analytic Jacobian avoids automatic differentiation overhead and can be significantly faster for models evaluated many times.

### `cost` (keyword, default: constant 1)

A function `x -> Real` giving the cost of a single measurement at design point `x`:

```julia
cost = x -> x.t + 1        # longer measurements cost more
cost = Returns(1.0)         # unit cost (default)
```

Cost is used by `run_adaptive` to track budget consumption and by `design` for budget-aware allocation. When you pass `budget` instead of `n` to `design`, the number of measurements is determined automatically from the costs.

### `switching_cost` (keyword, default: `nothing`)

A tuple `(:field, value)` specifying a fixed cost incurred when a discrete design variable changes value between consecutive measurements:

```julia
switching_cost = (:channel, 50.0)   # costs 50 to switch channels
```

This creates a `SwitchingDesignProblem` instead of a plain `DesignProblem`. The switching cost is added on top of the per-measurement `cost` whenever the named field changes. See the [Switching Costs example](@ref "Switching Costs").

### `constraint` (keyword, default: always true)

A function `(x, θ) -> Bool` that restricts the design space. Only candidates where the constraint returns `true` are considered:

```julia
constraint = (x, θ) -> x.dose ≤ θ.max_dose   # parameter-dependent constraint
```

## Candidates

Design points are represented as `NamedTuple`s. Use `candidate_grid` to generate the full outer product from named ranges:

```jldoctest
julia> using OptimalDesign

julia> candidate_grid(t = [0.1, 0.2, 0.3])
3-element Vector{@NamedTuple{t::Float64}}:
 (t = 0.1,)
 (t = 0.2,)
 (t = 0.3,)

julia> candidate_grid(channel = [1, 2], t = [0.1, 0.2])
4-element Vector{@NamedTuple{channel::Int64, t::Float64}}:
 (channel = 1, t = 0.1)
 (channel = 2, t = 0.1)
 (channel = 1, t = 0.2)
 (channel = 2, t = 0.2)
```

The field names in candidates must match what your `predict` function expects on `x`.
