"""
    factory(a, args...; kwargs...) -> a
    factory(a::AbstractArray{<:Union{Nothing, AbstractEstimator, AbstractAlgorithm,
                                     AbstractResult, StatsBase.CovarianceEstimator}},
            args...; kwargs...) -> AbstractArray

Rebuild an estimator, an algorithm or a result with runtime values. These two methods are the leaf and the array step of the traversal.

`factory` and [`port_opt_view`](@ref) are the two propagation mechanisms of this library.
`factory` passes **runtime values**, such as prior moments, observation weights and
previous portfolio weights, down a tree of composed structs. `port_opt_view` passes an
**index selection**, a subset of the assets, down the same tree.

`factory` is a traversal, so it owns the identity on every leaf, as `fmap` in Functors.jl and `adapt_structure` in Adapt.jl do. A value that has no method of its own comes back with no change: a data field, a scalar, `nothing`, a set of observation weights and a covariance estimator of `StatsBase`. So a [`@propagatable`](@ref) struct sends each `@fprop` field to `factory` directly, whatever the field holds.

The array method is the **one forwarding contract** for every propagation field that holds an array of estimators, algorithms, results or covariance estimators. It applies `factory` to each element, and it forwards `args...` and `kwargs...` with no change. A family that accepts an array of them needs no method of its own. A family that needs more than the forward defines its own, more specific method. A concrete element type, through [`concrete_typed_array_if_abstract`](@ref), is an example. An array of data, such as a vector of numbers, is a leaf, and it comes back with no change and with no copy.

# Algorithm

The leaf method:

 1. Return `a` with no change, and drop `args...` and `kwargs...`. This method is the leaf of the recursion. Because of it, a call of the verb on a type with no tags returns the value and raises no error.

The array method:

 1. For each element `ai` of `a`, call `factory` on `ai`, and forward `args...` and `kwargs...` with no change.
 2. Collect the results into a new array of the shape of `a`, in the order of `a`, and return it.

A [`@propagatable`](@ref) struct always has a generated method that is more specific than the leaf method. If the struct has at least one `@fprop` or `@wprop` field, that method builds the struct again with its keyword constructor. It sends each `@fprop` field through `factory` and each `@wprop` field through [`_wprop`](@ref). If the struct has no such field, that method returns `x`.

The leaf method does **not** fill a `nothing` field. A weights field that must take the incoming weights when it holds `nothing` carries `@wprop`, and it goes to [`_wprop`](@ref).

# Arguments

  - `a`: Any value, which the leaf method returns with no change. For the array method, an array of estimators, algorithms, results, covariance estimators or `nothing`, which it rebuilds one element at a time.
  - `args...`: Positional arguments. The leaf method ignores them, and the array method forwards them.
  - `kwargs...`: Keyword arguments. The leaf method ignores them, and the array method forwards them.

# Returns

  - `a`: The input with no change, for the leaf method.
  - `v::AbstractArray`: The rebuilt elements, in an array of the shape of `a`, for the array method.

# Examples

```jldoctest
julia> factory(nothing, 1, 2; x = 3)

julia> factory([1.0, 2.0], 1, 2; x = 3)
2-element Vector{Float64}:
 1.0
 2.0

julia> factory(MeanValue())
MeanValue
  w ┴ nothing
```

# Related

  - [`port_opt_view`](@ref)
  - [`@propagatable`](@ref)
  - [`_wprop`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
function factory(a, args...; kwargs...)
    return a
end
function factory(a::AbstractArray{<:Union{Nothing, AbstractEstimator, AbstractAlgorithm,
                                          AbstractResult, StatsBase.CovarianceEstimator}},
                 args...; kwargs...)
    return [factory(ai, args...; kwargs...) for ai in a]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

New value of an observation-weights field with a [`@wprop`](@ref) tag, in a call of
[`factory`](@ref).

When `factory` passes an [`ObsWeights`](@ref) argument, those weights **replace** the
field. Otherwise, the field keeps its value. [`factory`](@ref), which
[`@fprop`](@ref) uses, is different. It recurses into sub-estimators, and it returns
`nothing` and every value that is not an estimator with no change. A weights field that
holds `nothing` and an optional sub-estimator that is `nothing` need different rules, so
they use different functions.

# Algorithm

The method that Julia selects is the algorithm, and the selection reads `args...`, never the field.

 1. The first threaded positional argument is an [`ObsWeights`](@ref): return that value, whatever the field held.
 2. The first threaded positional argument is not an [`ObsWeights`](@ref), or no argument is threaded: return `field` with no change.

The value of the field never selects the branch. An incoming [`ObsWeights`](@ref) replaces a field that holds `nothing` and a field that holds weights in the same way. Without an incoming `ObsWeights`, both fields keep their values.

# Related

  - [`@wprop`](@ref)
  - [`@propagatable`](@ref)
  - [`factory`](@ref)
"""
_wprop(field, args...; kwargs...) = field
_wprop(::Any, w::ObsWeights, args...; kwargs...) = w
"""
    resolve_deferred_quantities(x, pr::AbstractPriorResult, slv = nothing)

Resolve every **Deferred Quantity** that `x` holds against a prior result. The result is
a struct of the same type whose deferred slots hold plain values.

The function resolves the deferred state and **nothing else**. A slot that the caller did
not state stays `nothing`, so the fallback of each consumer works as before. On the
factory path, that fallback is `sel`. On the `JuMP` path, it is
[`chol_sigma_selector`](@ref) and its siblings. The two paths are separate. A `JuMP` model
builder reads the slots of the risk measure directly and never calls [`factory`](@ref), so
both entry points resolve.

With a prior result, the rule has two parts. **The recursion into a container comes from
[`deferred_slots`](@ref)**, so a type that only holds children needs no method. **A type
that resolves a quantity of its own defines a method**, which overrides the derived
method. That method is per type and not per field, so it resolves together the slots that
go together. For example, a deferred `sigma` gives `chol` from the same fit, so the pair
never comes from two different sources.

The third argument, `slv`, is the effective solver, and a **Calibration Rule** in the same struct reads it. It holds the solver that the optimisation selected, so a rule resolves against one solver on both routes. On the [`factory`](@ref) route, the [`@cprop`](@ref) selection already put that solver on the struct, so the argument keeps its default. On the `JuMP` route, no selection runs, so [`set_risk_constraints!`](@ref) reads the solver from the estimator and passes it here. A type that holds a solver of its own selects it locally as `sel(x.slv, slv)`, as it does for its observation weights. A type that holds no solver gives its rules no solver on either route.

# Algorithm

The method that Julia selects is the algorithm. It is the method that [`deferred_slots`](@ref) derives for a container, or the hand-written method of a type that resolves a quantity of its own. Each method takes a prior result as its second argument. A second argument that is not a prior result raises a `MethodError`, because without a prior the function cannot fit anything, and no caller asks it to.

# Related

  - [`@propagatable`](@ref)
  - [`resolve_slot`](@ref)
  - [`deferred_slots`](@ref)
  - [`factory`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function resolve_deferred_quantities end

export factory
