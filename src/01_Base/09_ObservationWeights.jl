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
    const ScenarioProbabilities = Union{<:StatsBase.AnalyticWeights,
                                        <:StatsBase.ProbabilityWeights}

Union type for the probabilities of the scenarios of a prior that reweights its scenarios.

The type of the weights fixes the bias correction of a weighted variance, and `corrected = true` is the default of the library. Probabilities of scenarios sum to one. Under that sum, `StatsBase.AnalyticWeights` divides by ``1 - \\sum_t p_t^2``, the reliability correction ``n_{eff} / (n_{eff} - 1)`` of the Kish count ``n_{eff} = 1 / \\sum_t p_t^2``. `StatsBase.ProbabilityWeights` gives ``T / (T - 1)`` whatever the weights are. The other kinds give no usable correction: `StatsBase.FrequencyWeights` divides by ``\\sum_t p_t - 1 = 0``, and `StatsBase.Weights` refuses `corrected = true`.

# Related

  - [`scenario_probabilities`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`OpinionPoolingPrior`](@ref)
"""
const ScenarioProbabilities = Union{<:StatsBase.AnalyticWeights,
                                    <:StatsBase.ProbabilityWeights}
"""
    scenario_probabilities(w::VecNum)
    scenario_probabilities(w, x::VecNum)

Wrap the probabilities `x` of the scenarios in the kind of the weights `w`. The one-argument method reads `w` as the probabilities, and returns it unchanged when it already has a kind of [`ScenarioProbabilities`](@ref).

A prior that reweights its scenarios answers posterior probabilities of the same kind as its prior probabilities, so a caller who gives `StatsBase.ProbabilityWeights` keeps their correction. Without such weights, the probabilities are `StatsBase.AnalyticWeights`, whose correction is the one of reliability weights. [`ScenarioProbabilities`](@ref) states the two corrections.

# Algorithm

The method that Julia selects is the algorithm.

 1. `w` is a [`ScenarioProbabilities`](@ref): return weights of the kind of `w` over `x`. With no `x`, return `w`.
 2. `w` is anything else, `nothing` and a plain vector included: return `StatsBase.aweights(x)`. With no `x`, return `StatsBase.aweights(w)`.

# Arguments

  - `w`: The weights whose kind the result takes, or `nothing`.
  - `x`: The probabilities of the scenarios. It is `w` when it is not given.

# Returns

  - `p::ScenarioProbabilities`: The weights over `x`.

# Examples

```jldoctest
julia> PortfolioOptimisers.scenario_probabilities(nothing, [0.25, 0.75])
2-element AnalyticWeights{Float64, Float64, Vector{Float64}}:
 0.25
 0.75

julia> PortfolioOptimisers.scenario_probabilities(pweights([0.5, 0.5]), [0.25, 0.75])
2-element ProbabilityWeights{Float64, Float64, Vector{Float64}}:
 0.25
 0.75
```

# Related

  - [`ScenarioProbabilities`](@ref)
  - [`ep_prior_probabilities`](@ref)
  - [`entropy_pooling`](@ref)
"""
function scenario_probabilities(w::ScenarioProbabilities)
    return w
end
function scenario_probabilities(w::VecNum)
    return StatsBase.aweights(w)
end
function scenario_probabilities(w::ScenarioProbabilities, x::VecNum)
    return Base.typename(typeof(w)).wrapper(x)
end
function scenario_probabilities(::Any, x::VecNum)
    return StatsBase.aweights(x)
end
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
public get_observation_weights
