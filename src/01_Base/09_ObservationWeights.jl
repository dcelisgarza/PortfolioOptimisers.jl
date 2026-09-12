"""
    const ObsWeights = Union{<:DynamicAbstractWeights, <:StatsBase.AbstractWeights}

Union type for observation weights accepted by estimators.

Accepts either a [`DynamicAbstractWeights`](@ref) subtype (weights computed from data at fit time) or a `StatsBase.AbstractWeights` instance (pre-computed numeric weights).

# Related

  - [`DynamicAbstractWeights`](@ref)
  - [`get_observation_weights`](@ref)
"""
const ObsWeights = Union{<:DynamicAbstractWeights, <:StatsBase.AbstractWeights}
"""
    get_observation_weights(
        w::Option{<:ObsWeights},
        args...;
        kwargs...
    ) -> Option{<:VecNum}

Get the observation weights for statistical estimation.

`nothing` is returned only when `w === nothing`, and means *no weights were requested* — every `isnothing` branch downstream reads it that way and computes an unweighted result. It never means *weights were unavailable*: a [`DynamicAbstractWeights`](@ref) with no method for the given input shape throws [`ObservationWeightsError`](@ref) rather than resolving to `nothing`, because returning `nothing` there would silently yield an unweighted answer that looks plausible.

This is why call sites need no strictness check of their own. A `DynamicAbstractWeights` is resolved *before* dispatch (see [`average_drawdown`](@ref) for the pattern), so the estimator downstream only ever sees a concrete weight vector or a deliberate `nothing`.

# The returned vector is borrowed, not owned

For a `StatsBase.AbstractWeights` this returns **the stored object itself**, not a copy — an estimator's `w` field is handed straight back. So the caller may **read** it but must never **mutate** it: writing through it permutes the estimator's own configuration, and every later evaluation of that estimator is then wrong.

This is the same obligation the rest of `src/` already meets: a `reverse!` or a `sort!` is applied only to a vector the surrounding expression has just allocated. Beware the indirect route in particular — `view(w, order)` is a **view**, so `reverse!` on the view writes through into `w` just as surely as `reverse!(w)` would. Reverse the permutation instead, or sort into a fresh vector.

A defensive copy here was considered and rejected: it would cost an allocation on every evaluation of every weighted estimator, and the obligation is cheap to keep.

# Algorithm

The method Julia selects on the type of `w` is the algorithm. Three methods share the name, and a fourth is written by the caller.

 1. `w === nothing` selects the method that returns `nothing`, which is the deliberate request for an unweighted result.
 2. A `StatsBase.AbstractWeights`, which is a `VecNum`, selects the method that returns the stored object itself. It allocates nothing, which is why the returned vector is borrowed.
 3. A [`DynamicAbstractWeights`](@ref) for which the caller wrote no method selects the fallback, which raises. The fallback builds `shape` from the first positional argument, naming its dimension count and its size when that argument is an array and naming it as "the given input" otherwise, then raises [`ObservationWeightsError`](@ref) carrying `shape` and the two method signatures to write.
 4. A [`DynamicAbstractWeights`](@ref) for which the caller wrote a method selects that method instead, because it is more specific than the fallback of step 3.

# Arguments

  - $(arg_dict[:oow])
  - $(arg_dict[:ignargs])
  - $(arg_dict[:ignkwargs])

# Validation

  - `w` is `nothing`, a `StatsBase.AbstractWeights`, or a [`DynamicAbstractWeights`](@ref) that has a method for the shape of the given input. A [`DynamicAbstractWeights`](@ref) with no such method raises [`ObservationWeightsError`](@ref), which names the two signatures to write. It never resolves to `nothing`, because that would silently give an unweighted answer that looks plausible.

# Returns

  - `w::Option{<:VecNum}`: The observation weights, or `nothing` when `w` is `nothing`.

# Related

  - [`ObsWeights`](@ref)
  - [`DynamicAbstractWeights`](@ref)
  - [`ObservationWeightsError`](@ref)
"""
function get_observation_weights(::Nothing, args...; kwargs...)
    return nothing
end
function get_observation_weights(w::DynamicAbstractWeights, args...; kwargs...)
    name = nameof(typeof(w))
    X = isempty(args) ? nothing : first(args)
    shape = if isa(X, AbstractArray)
        "a $(ndims(X))-dimensional input of size $(size(X))"
    else
        "the given input"
    end
    return throw(ObservationWeightsError("$name is a DynamicAbstractWeights with no `get_observation_weights` method for $shape. Implement `get_observation_weights(w::$name, X::VecNum; kwargs...)` and/or `get_observation_weights(w::$name, X::MatNum; dims::Int = 1, kwargs...)`, or pass a `StatsBase.AbstractWeights` instead (or `nothing` to compute unweighted). See the `DynamicAbstractWeights` docstring for a worked example."))
end
function get_observation_weights(w::VecNum, args...; kwargs...)
    return w
end
