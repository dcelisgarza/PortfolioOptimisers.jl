"""
    const ObsWeights = Union{DynamicAbstractWeights, StatsBase.AbstractWeights}

Union type for the observation weights that an estimator accepts.

It holds a subtype of [`DynamicAbstractWeights`](@ref), which computes the weights from the data when the estimator fits. It also holds a `StatsBase.AbstractWeights`, which holds weights that the caller computed before.

# Related

  - [`DynamicAbstractWeights`](@ref)
  - [`get_observation_weights`](@ref)
"""
const ObsWeights = Union{DynamicAbstractWeights, StatsBase.AbstractWeights}
"""
    const ScenarioProbabilities = Union{StatsBase.AnalyticWeights,
                                        StatsBase.ProbabilityWeights}

Union type for the probabilities of the scenarios of a prior that reweights its scenarios.

The type of the weights fixes the bias correction of a weighted variance, and `corrected = true` is the default of the library. Probabilities of scenarios sum to one. Under that sum, `StatsBase.AnalyticWeights` divides by ``1 - \\sum_t p_t^2``, the reliability correction ``n_{eff} / (n_{eff} - 1)`` of the Kish count ``n_{eff} = 1 / \\sum_t p_t^2``. `StatsBase.ProbabilityWeights` gives ``T / (T - 1)`` whatever the weights are. The other kinds give no usable correction: `StatsBase.FrequencyWeights` divides by ``\\sum_t p_t - 1 = 0``, and `StatsBase.Weights` refuses `corrected = true`.

# Related

  - [`scenario_probabilities`](@ref)
  - [`EntropyPoolingPrior`](@ref)
  - [`MeucciEntropyPoolingPrior`](@ref)
  - [`OpinionPoolingPrior`](@ref)
"""
const ScenarioProbabilities = Union{StatsBase.AnalyticWeights, StatsBase.ProbabilityWeights}
"""
    scenario_probabilities(w::VecNum)
    scenario_probabilities(w, x::VecNum)

Wrap the probabilities `x` of the scenarios in the kind of the weights `w`. The one-argument method reads `w` as the probabilities, and returns it unchanged when it already has a kind of [`ScenarioProbabilities`](@ref).

A prior that reweights its scenarios returns posterior probabilities of the same kind as its prior probabilities. A caller who gives `StatsBase.ProbabilityWeights` thus keeps that correction. With no such weights, the probabilities are `StatsBase.AnalyticWeights`, whose correction is the correction of reliability weights. [`ScenarioProbabilities`](@ref) states the two corrections.

# Algorithm

The method that Julia selects is the algorithm.

 1. When `w` is a [`ScenarioProbabilities`](@ref), return weights of the kind of `w` over `x`. With no `x`, return `w`.
 2. When `w` is a different value, which includes `nothing` and a plain vector, return `StatsBase.aweights(x)`. With no `x`, return `StatsBase.aweights(w)`.

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

Get the observation weights for a statistical estimate.

The function returns `nothing` only when `w === nothing`, and `nothing` means that the caller asked for no weights. Each `isnothing` branch after it reads `nothing` in that way, and computes an unweighted result. `nothing` never means that the weights are not available. A [`DynamicAbstractWeights`](@ref) with no method for the shape of the input throws [`ObservationWeightsError`](@ref), and does not resolve to `nothing`. A `nothing` there gives an unweighted answer that looks correct, with no message.

A call site thus needs no strictness check of its own. The library resolves a `DynamicAbstractWeights` before dispatch, and [`average_drawdown`](@ref) shows the pattern. The estimator after it thus sees only a concrete weight vector, or a `nothing` that the caller asked for.

# The returned vector is borrowed

For a `StatsBase.AbstractWeights`, the function returns the stored object itself, and not a copy. It gives back the `w` field of the estimator, which still owns the vector. The caller can read the vector, but must never mutate it. A write to it changes the configuration of the estimator, and every later evaluation of that estimator is then wrong.

The rest of `src/` already obeys this rule. A `reverse!` or a `sort!` applies only to a vector that the expression around it allocated. Look for the indirect case. `view(w, order)` is a view, so `reverse!` on the view writes into `w` in the same way as `reverse!(w)`. Reverse the permutation instead, or sort into a fresh vector.

A defensive copy in this function costs an allocation on each evaluation of each weighted estimator. The rule costs almost nothing to obey, so the function makes no copy.

# Algorithm

The method that Julia selects on the type of `w` is the algorithm. Three methods share the name, and the caller writes a fourth.

 1. `w === nothing` selects the method that returns `nothing`, which is the request for an unweighted result.
 2. A `StatsBase.AbstractWeights`, which is a `VecNum`, selects the method that returns the stored object itself. It allocates nothing, so the returned vector is borrowed.
 3. A [`DynamicAbstractWeights`](@ref) for which the caller wrote no method selects the fallback, which raises. The fallback builds `shape` from the first positional argument. When that argument is an array, `shape` names its number of dimensions and its size, and in the other case it is "the given input". The fallback then raises [`ObservationWeightsError`](@ref) with `shape` and the two method signatures to write.
 4. A [`DynamicAbstractWeights`](@ref) for which the caller wrote a method selects that method, because it is more specific than the fallback of step 3.

# Arguments

  - $(arg_dict[:oow])
  - $(arg_dict[:ignargs])
  - $(arg_dict[:ignkwargs])

# Validation

  - `w` is `nothing`, a `StatsBase.AbstractWeights`, or a [`DynamicAbstractWeights`](@ref) that has a method for the shape of the given input. A [`DynamicAbstractWeights`](@ref) with no such method raises [`ObservationWeightsError`](@ref), which names the two signatures to write. It never resolves to `nothing`, because a `nothing` gives an unweighted answer that looks correct, with no message.

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
"""
    observation_weight_count(w::Nothing, T::Integer) -> Integer
    observation_weight_count(w::StatsBase.FrequencyWeights, T::Integer) -> Number
    observation_weight_count(w::AbstractVector, T::Integer) -> Number

Return the number of observations that a sample of `T` rows under the weights `w` is worth.

The function dispatches on the weight type, as [`simple_variance_count`](@ref) and [`glm_weights`](@ref) do. Frequency weights count replicated observations, so their count is their sum. Every other weight vector states the relative weight of each observation and no count, so its count is the Kish count. With no weights the count is `T`.

# Mathematical definition

```math
\\begin{align}
T_{e} &= \\dfrac{\\left(\\sum_{t} w_{t}\\right)^{2}}{\\sum_{t} w_{t}^{2}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_T_e])
  - $(math_dict[:w_t_obs])

# Arguments

  - `w`: The resolved observation weights, or `nothing`. [`get_observation_weights`](@ref) resolves them.
  - `T`: The number of rows of the sample.

# Returns

  - `T_e::Number`: The effective number of observations.

# Related

  - [`effective_observation_count`](@ref)
  - [`get_observation_weights`](@ref)

# References

  - $(ref_dict[:kish1965]) Whole book. $(ref_dict[:no_original_source])
"""
function observation_weight_count(::Nothing, T::Integer)
    return T
end
function observation_weight_count(w::StatsBase.FrequencyWeights, ::Integer)
    return sum(w)
end
function observation_weight_count(w::AbstractVector, ::Integer)
    return weight_observation_count(KishObservationCount(), w)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all rules that count the observations that a vector of weights is worth.

Most estimators weight their observations with observation weights, and [`observation_weight_count`](@ref) counts those weights. Some estimators weight their observations with a rule of their own, such as the temporal decay of [`GerberIQCovariance`](@ref). Their statistic is not a weighted moment, so no count follows from it. Such an estimator holds a field of this type, and the rule states the count of the weights that the estimator builds. `nothing` in that field reads every row as one observation.

[`effective_observation_count`](@ref) reads the count, and the denoising step of a [`MatrixProcessing`](@ref) reads it through that function.

# Interfaces

To add a rule, subtype `AbstractObservationCountAlgorithm` and implement this method:

## `weight_observation_count`

  - `weight_observation_count(alg::MyObservationCount, w::AbstractVector) -> Real`: The number of observations that the weights `w` are worth. A subtype that implements no method raises an `ArgumentError` that names the subtype.

### Arguments

  - `alg`: The instance of the subtype.
  - `w`: The weights of the observations, one entry for each row.

### Returns

  - `T_e::Real`: The effective number of observations.

# Related

  - [`KishObservationCount`](@ref)
  - [`weight_observation_count`](@ref)
  - [`effective_observation_count`](@ref)
  - [`GerberIQCovariance`](@ref)
"""
abstract type AbstractObservationCountAlgorithm <: AbstractAlgorithm end
"""
    KishObservationCount()

Counts the observations that a vector of weights is worth with the Kish count.

The Kish count is the number of equally weighted observations that give the precision of the weighted sample. It is `T` under equal weights on `T` rows, and `k` under equal weights on `k` of the rows and zero on the others. [`weight_observation_count`](@ref) states the formula.

# Examples

```jldoctest
julia> PortfolioOptimisers.weight_observation_count(KishObservationCount(), [1.0, 1.0, 0.0])
2.0
```

# Related

  - [`AbstractObservationCountAlgorithm`](@ref)
  - [`weight_observation_count`](@ref)
  - [`observation_weight_count`](@ref)

# References

  - $(ref_dict[:kish1965]) Whole book. $(ref_dict[:no_original_source])
"""
struct KishObservationCount <: AbstractObservationCountAlgorithm end
"""
    weight_observation_count(alg::KishObservationCount, w::AbstractVector) -> Real
    weight_observation_count(alg::AbstractObservationCountAlgorithm, w::AbstractVector)

Return the number of observations that the weights `w` are worth, under the rule `alg`.

# Mathematical definition

```math
\\begin{align}
T_{e} &= \\dfrac{\\left(\\sum_{t} w_{t}\\right)^{2}}{\\sum_{t} w_{t}^{2}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_T_e])
  - $(math_dict[:w_t_obs])

# Arguments

  - `alg`: The rule of the count.
      + `::KishObservationCount`: The Kish count.
      + `::AbstractObservationCountAlgorithm`: The fallback of the family, for a subtype that implements no method of its own. It always throws.
  - `w`: The weights of the observations, one entry for each row.

# Validation

  - A subtype of [`AbstractObservationCountAlgorithm`](@ref) that implements no method raises an `ArgumentError` that names the subtype.

# Returns

  - `T_e::Real`: The effective number of observations.

# Related

  - [`AbstractObservationCountAlgorithm`](@ref)
  - [`KishObservationCount`](@ref)
  - [`observation_weight_count`](@ref)

# References

  - $(ref_dict[:kish1965]) Whole book. $(ref_dict[:no_original_source])
"""
function weight_observation_count(::KishObservationCount, w::AbstractVector)
    return sum(w)^2 / sum(abs2, w)
end
function weight_observation_count(alg::AbstractObservationCountAlgorithm, ::AbstractVector)
    return throw(ArgumentError("$(typeof(alg)) subtypes AbstractObservationCountAlgorithm but does not implement weight_observation_count. Extension authors: a count rule must implement weight_observation_count(alg, w) -> Real, the number of observations that the weights w are worth."))
end
export KishObservationCount
public AbstractObservationCountAlgorithm, weight_observation_count
public get_observation_weights
