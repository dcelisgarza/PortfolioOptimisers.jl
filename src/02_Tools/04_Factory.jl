"""
    factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm,
                     <:AbstractResult}, args...; kwargs...) -> a
    factory(a::AbstractVector{<:Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm,
                                      <:AbstractResult}}, args...; kwargs...) -> Vector

No-op factory function for constructing objects with a uniform interface.

Defining methods which dispatch on the first argument allows for a consistent factory interface across different types.

`factory` and [`port_opt_view`](@ref) are the two propagation mechanisms in this library.
They are duals: `factory` threads **runtime values** (prior moments, observation weights,
previous portfolio weights) down through a composed struct tree; `port_opt_view` threads an
**index selection** (a subset of assets or observations) down through the same tree.

The vector method is the **one forwarding contract** for every vector-valued propagation field: it applies `factory` to each element and forwards `args...` and `kwargs...` unchanged, so a family that admits a vector of estimators, algorithms, or results needs no method of its own. A family that needs more than the forward, such as a concrete element type ([`concrete_typed_array_if_abstract`](@ref)), defines its own more specific method.

# Algorithm

The scalar method:

 1. Return `a` unchanged, and drop `args...` and `kwargs...`. This method is the leaf of the recursion, and it is what makes an untagged type safe to call the verb on.

The vector method:

 1. For each element `ai` of `a`, call `factory` on `ai`, and forward `args...` and `kwargs...` unchanged.
 2. Collect the results into a new vector, in the order of `a`, and return it.

A [`@propagatable`](@ref) struct with at least one `@fprop`- or `@wprop`-tagged field carries a generated method that dominates the scalar method. That method rebuilds the struct with its keyword constructor, sending each `@fprop` field through [`factory_child`](@ref) and each `@wprop` field through [`_wprop`](@ref).

# Arguments

  - `a`: Indicates no object should be constructed, or a vector whose elements are rebuilt one by one.
  - `args...`: Arbitrary positional arguments (ignored by the scalar method, forwarded by the vector method).
  - `kwargs...`: Arbitrary keyword arguments (ignored by the scalar method, forwarded by the vector method).

# Returns

  - `a`: The input unchanged.
  - `v::Vector`: The element-wise rebuilds, for the vector method.

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
function factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm,
                          <:AbstractResult}, args...; kwargs...)
    return a
end
function factory(a::AbstractVector{<:Union{Nothing, <:AbstractEstimator,
                                           <:AbstractAlgorithm, <:AbstractResult}}, args...;
                 kwargs...)
    return [factory(ai, args...; kwargs...) for ai in a]
end

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Per-field recursion helper called by [`@propagatable`](@ref)-generated [`factory`](@ref) methods.

Dispatches on the field value type: estimators, algorithms, and results recurse via [`factory`](@ref); observation-weight fields (`::Nothing` or `::StatsBase.AbstractWeights`) are replaced by the incoming [`ObsWeights`](@ref) argument; everything else passes through unchanged.

# Algorithm

The method that Julia selects is the algorithm.

 1. `v` is an estimator, an algorithm or a result: return [`factory`](@ref) of `v`, forwarding `args...` and `kwargs...`. The recursion descends one level of the struct tree.
 2. `v` is an array of them: apply step 1 to each element, and collect the results into a new vector.
 3. `v` is anything else: return `v` unchanged. A data field, a scalar and a `nothing` all take this branch.

Step 3 is why a `nothing` field is **not** filled in by this verb. A weights field that must be replaced when it holds `nothing` carries `@wprop` and reaches [`_wprop`](@ref) instead.

# Related

  - [`@propagatable`](@ref)
  - [`factory`](@ref)
"""
factory_child(v, args...; kwargs...) = v
function factory_child(v::Union{<:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult},
                       args...; kwargs...)
    return factory(v, args...; kwargs...)
end
function factory_child(v::AbstractArray{<:Union{<:AbstractEstimator, <:AbstractAlgorithm,
                                                <:AbstractResult}}, args...; kwargs...)
    return [factory_child(vi, args...; kwargs...) for vi in v]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve the new value of a [`@wprop`](@ref)-tagged observation-weights field during
[`factory`](@ref) propagation.

When an [`ObsWeights`](@ref) argument is threaded through `factory`, the field is
**replaced** by those weights; otherwise the existing field value is kept. This is
distinct from [`factory_child`](@ref) (used by [`@fprop`](@ref)), which recurses into
sub-estimators and leaves `nothing`/non-estimator values unchanged — a weights slot
must not be confused with an optional sub-estimator that happens to be `nothing`.

# Algorithm

The method that Julia selects is the algorithm, and the selection reads `args...`, never the field.

 1. The first threaded positional argument is an [`ObsWeights`](@ref): return that value, whatever the field held.
 2. No such argument is threaded: return `field` unchanged.

The field's own value never selects the branch, so a field holding `nothing` and a field holding weights are both replaced by an incoming [`ObsWeights`](@ref), and both are kept when none is threaded.

# Related

  - [`@wprop`](@ref)
  - [`@propagatable`](@ref)
  - [`factory_child`](@ref)
"""
_wprop(field, args...; kwargs...) = field
_wprop(::Any, w::ObsWeights, args...; kwargs...) = w
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Resolve every **Deferred Quantity** held by `x` against prior result `pr`, returning a
struct of the same type whose deferred slots hold plain values.

This resolves the deferred state and **nothing else**. A slot left unstated stays `nothing`,
so whichever fallback the consumer already applies — `sel` on the factory path,
[`chol_sigma_selector`](@ref) and its siblings on the `JuMP` path — keeps working unchanged.
The two paths are separate: a `JuMP` model builder reads the risk measure's slots directly
and never calls [`factory`](@ref), so both entry points resolve.

Given a prior result the rule has two halves. **Container recursion is derived** from
[`deferred_slots`](@ref), so a type that only holds children needs no method at all. **A
type that resolves a quantity of its own defines a method**, which overrides the derived one.
Writing that half per type — rather than per field — is what lets slots that travel together
be resolved together: a deferred `sigma` supplies `chol` from the same fit, so the pair is
never mixed across two sources.

`slv` is the effective solver, and it is what a **Calibration Rule** in the same struct reads. It carries the value the optimisation settled on, so a rule resolves against one solver on both routes. On the [`factory`](@ref) route the [`@cprop`](@ref) selection has already put that solver on the struct, so the argument stays at its default. On the `JuMP` route no selection runs, so [`set_risk_constraints!`](@ref) reads the solver off the estimator and threads it here. A type that carries a solver of its own settles it locally as `sel(x.slv, slv)`, beside the observation weights it already settles that way, and a type that carries none gives its rules none on either route.

# Algorithm

 1. Return `x` unchanged. This method is the arm for a second argument that is **not** a prior result: with no prior in hand nothing can be fitted, so the deferred state travels on.

A more specific method dominates this one on a prior result: the one that [`deferred_slots`](@ref) derives for a container, and the hand-written one of a type that resolves a quantity of its own.

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
