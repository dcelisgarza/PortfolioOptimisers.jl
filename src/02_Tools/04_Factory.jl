"""
    factory(a::Union{Nothing, AbstractEstimator, AbstractAlgorithm,
                     AbstractResult}, args...; kwargs...) -> a
    factory(a::AbstractVector{<:Union{Nothing, AbstractEstimator, AbstractAlgorithm,
                                      AbstractResult}}, args...; kwargs...) -> Vector

Rebuild an estimator, an algorithm or a result with runtime values. These two methods are the fallbacks of the verb.

`factory` and [`port_opt_view`](@ref) are the two propagation mechanisms of this library.
`factory` passes **runtime values**, such as prior moments, observation weights and
previous portfolio weights, down a tree of composed structs. `port_opt_view` passes an
**index selection**, a subset of the assets, down the same tree.

The vector method is the **one forwarding contract** for every propagation field that holds a vector. It applies `factory` to each element, and it forwards `args...` and `kwargs...` with no change. A family that accepts a vector of estimators, algorithms or results needs no method of its own. A family that needs more than the forward defines its own, more specific method. A concrete element type, through [`concrete_typed_array_if_abstract`](@ref), is an example.

# Algorithm

The scalar method:

 1. Return `a` with no change, and drop `args...` and `kwargs...`. This method is the leaf of the recursion. Because of it, a call of the verb on a type with no tags returns the value and raises no error.

The vector method:

 1. For each element `ai` of `a`, call `factory` on `ai`, and forward `args...` and `kwargs...` with no change.
 2. Collect the results into a new vector, in the order of `a`, and return it.

A [`@propagatable`](@ref) struct always has a generated method that is more specific than the scalar method. If the struct has at least one `@fprop` or `@wprop` field, that method builds the struct again with its keyword constructor. It sends each `@fprop` field through [`factory_child`](@ref) and each `@wprop` field through [`_wprop`](@ref). If the struct has no such field, that method returns `x`.

# Arguments

  - `a`: An estimator, an algorithm, a result or `nothing`, which the scalar method returns with no change. For the vector method, a vector of them, which it rebuilds one element at a time.
  - `args...`: Positional arguments. The scalar method ignores them, and the vector method forwards them.
  - `kwargs...`: Keyword arguments. The scalar method ignores them, and the vector method forwards them.

# Returns

  - `a`: The input with no change, for the scalar method.
  - `v::Vector`: The rebuilt elements, for the vector method.

# Examples

```jldoctest
julia> factory(nothing, 1, 2; x = 3)

julia> factory(MeanValue())
MeanValue
  w ┴ nothing
```

# Related

  - [`port_opt_view`](@ref)
  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
function factory(a::Union{Nothing, AbstractEstimator, AbstractAlgorithm, AbstractResult},
                 args...; kwargs...)
    return a
end
function factory(a::AbstractVector{<:Union{Nothing, AbstractEstimator, AbstractAlgorithm,
                                           AbstractResult}}, args...; kwargs...)
    return [factory(ai, args...; kwargs...) for ai in a]
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Recursion into one field. The [`factory`](@ref) methods that [`@propagatable`](@ref) makes call it for each `@fprop` field.

It dispatches on the type of the field value. An estimator, an algorithm or a result recurses through [`factory`](@ref). Every other value comes back with no change, and this includes `nothing` and a set of observation weights. Only [`_wprop`](@ref) replaces observation weights with an incoming [`ObsWeights`](@ref).

# Algorithm

The method that Julia selects is the algorithm.

 1. `v` is an estimator, an algorithm or a result: return [`factory`](@ref) of `v`, and forward `args...` and `kwargs...`. The recursion goes down one level of the struct tree.
 2. `v` is an array of them: apply step 1 to each element, and collect the results into a new array of the same shape.
 3. `v` is any other value: return `v` with no change. A data field, a scalar and a `nothing` all take this branch.

Because of step 3, this verb does **not** fill a `nothing` field. A weights field that must take the incoming weights when it holds `nothing` carries `@wprop`, and it goes to [`_wprop`](@ref).

# Related

  - [`@propagatable`](@ref)
  - [`factory`](@ref)
"""
factory_child(v, args...; kwargs...) = v
function factory_child(v::Union{AbstractEstimator, AbstractAlgorithm, AbstractResult},
                       args...; kwargs...)
    return factory(v, args...; kwargs...)
end
function factory_child(v::AbstractArray{<:Union{AbstractEstimator, AbstractAlgorithm,
                                                AbstractResult}}, args...; kwargs...)
    return [factory_child(vi, args...; kwargs...) for vi in v]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

New value of an observation-weights field with a [`@wprop`](@ref) tag, in a call of
[`factory`](@ref).

When `factory` passes an [`ObsWeights`](@ref) argument, those weights **replace** the
field. Otherwise, the field keeps its value. [`factory_child`](@ref), which
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
  - [`factory_child`](@ref)
"""
_wprop(field, args...; kwargs...) = field
_wprop(::Any, w::ObsWeights, args...; kwargs...) = w
"""
$(DocStringExtensions.TYPEDSIGNATURES)

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

 1. Return `x` with no change. This method takes a second argument that is **not** a prior result. Without a prior, the function cannot fit anything, so the deferred state stays as it is.

On a prior result, a more specific method takes the call. It is the method that [`deferred_slots`](@ref) derives for a container, or the hand-written method of a type that resolves a quantity of its own.

# Related

  - [`@propagatable`](@ref)
  - [`resolve_slot`](@ref)
  - [`deferred_slots`](@ref)
  - [`factory`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
resolve_deferred_quantities(x, ::Any, ::Any = nothing) = x

export factory
public factory_child
