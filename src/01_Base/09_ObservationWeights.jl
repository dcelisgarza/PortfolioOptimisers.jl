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
    weighted_moment_guard(val::Number, w::StatsBase.AbstractWeights, corrected::Bool) -> Number
    weighted_moment_guard(val::AbstractArray, w::StatsBase.AbstractWeights,
                          corrected::Bool) -> AbstractArray

Return a weighted variance or covariance that `StatsBase` computed, or `NaN` in each entry when the divisor of the weights is not positive.

`StatsBase` divides a weighted sum of squares by the divisor ``D`` of the weights. A corrected fit on one observation of positive weight has ``D = 0``, as an unweighted corrected fit on one observation does. But the weighted mean of one observation can differ from it by one unit in the last place. So the sum of squares is about ``10^{-36}`` and not zero, and the fit is `Inf` in place of the `NaN` of the unweighted fit. Frequency weights that sum to less than one give ``D < 0``, and a variance below zero. A divisor that is not positive gives no variance, so the function returns `NaN`, as the unweighted fit on one observation does.

# Mathematical definition

```math
\\begin{align}
D &= \\begin{cases}
\\sum_{t} w_{t} & \\textrm{uncorrected}\\\\
\\sum_{t} w_{t} - \\dfrac{\\sum_{t} w_{t}^{2}}{\\sum_{t} w_{t}} & \\textrm{corrected analytic weights}\\\\
\\sum_{t} w_{t} - 1 & \\textrm{corrected frequency weights}\\\\
\\dfrac{\\left(\\sum_{t} w_{t}\\right) \\left(n_{+} - 1\\right)}{n_{+}} & \\textrm{corrected probability weights}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``D``: The divisor of the weighted sum of squares. It is the inverse of `StatsBase.varcorrection(w, corrected)`.
  - $(math_dict[:w_t_obs])
  - ``n_{+}``: The number of observations of positive weight.

# Arguments

  - `val`: The variance or the covariance matrix that `StatsBase` computed with `w`. The array method writes into it. A caller that needs a standard deviation takes the square root after the guard, because a variance below zero has no square root.
  - `w`: The resolved observation weights of the fit. [`get_observation_weights`](@ref) resolves them.
  - `corrected`: The bias correction of the fit.

# Returns

  - `val`: `val` without change when ``D > 0``. Otherwise `NaN` of the element type of `val`, in each entry.

# Related

  - [`get_observation_weights`](@ref)
  - [`SimpleVariance`](@ref)
  - [`GeneralCovariance`](@ref)
  - [`StdValue`](@ref)
  - [`VarValue`](@ref)
"""
function weighted_moment_guard(val::Number, w::StatsBase.AbstractWeights, corrected::Bool)
    return if inv(StatsBase.varcorrection(w, corrected)) > zero(val)
        val
    else
        convert(typeof(val), NaN)
    end
end
function weighted_moment_guard(val::AbstractArray, w::StatsBase.AbstractWeights,
                               corrected::Bool)
    return if inv(StatsBase.varcorrection(w, corrected)) > zero(eltype(val))
        val
    else
        fill!(val, convert(eltype(val), NaN))
    end
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
"""
    observed_sample(X::VecNum, w::Option{<:Union{ObsWeights, VecNum}},
                    obs::AbstractVector{Bool}; kwargs...) -> (VecNum, Option{<:VecNum})
    observed_sample(X::MatNum, w::Option{<:Union{ObsWeights, VecNum}},
                    obs::AbstractVector{Bool}; dims::Int = 1, kwargs...)
        -> (MatNum, Option{<:VecNum})
    observed_sample(X::VecNum, w::Option{<:Union{ObsWeights, VecNum}}; kwargs...)
        -> (VecNum, Option{<:VecNum})
    observed_sample(X::MatNum, w::Option{<:Union{ObsWeights, VecNum}}; dims::Int = 1,
                    kwargs...) -> (MatNum, Option{<:VecNum})

Drop the observations of `X` that `obs` marks `false`, and drop the observation weight of each of them with it.

A fit that cannot read an observation, such as an observation with a `NaN` return, drops it and reads the others. This function is the one rule for the weights of the observations that remain. [`observed_weights`](@ref) states the rule for each type of weights. The weights of the observed sample are the weights of the same observations in the full sample. They keep their type, and they are not scaled.

# The mask of the observed sample

With no `obs`, the function reads the mask from `X`: an observation is observed when every entry of it is finite. A return series and the columns of one pair of assets use this form.

Give `obs` when the reason for the drop is outside `X`. The rows of a portfolio drop where the portfolio return `X * w` is not finite, and a `NaN` in an asset of zero weight does not always give one. A gap in a Shared Series drops the row, but a gap of one asset in `X` only removes that asset from the Coverage Universe. An inactive cell of the active mask can hold a finite return.

# The returned values are borrowed when `obs` keeps every observation

When `obs` is `true` at every observation, the function returns `X` itself, and the weights of [`get_observation_weights`](@ref), with no copy. The caller must not mutate them, as [`get_observation_weights`](@ref) states. When `obs` drops an observation, `X_obs` and `w_obs` are new arrays.

# Algorithm

 1. With no `obs`, check `dims` for a matrix. Take `obs` as `true` at each observation whose entries are all finite.
 2. Check that `obs` has one entry for each observation of `X`. For a matrix, check `dims` first, and read the observations along `dims`.
 3. Keep `X` when `obs` keeps every observation. Otherwise index `X` with `obs` along the observations.
 4. Get the weights of the observed sample with [`observed_weights`](@ref).

# Arguments

  - `X`: The sample. A vector holds one observation in each entry. A matrix holds the observations along `dims`.
  - $(arg_dict[:oow])
  - `obs`: `true` at each observation that the fit reads, and `false` at each observation that it drops. With no `obs`, the observations whose entries are all finite.
  - `dims`: The dimension of a matrix `X` that holds the observations.
  - `kwargs...`: More keyword arguments for [`get_observation_weights`](@ref), where a [`DynamicAbstractWeights`](@ref) can read them.

# Validation

  - `dims in (1, 2)`, for a matrix `X`, through [`assert_dims`](@ref).
  - `length(obs)` is the number of observations of `X`. A `DimensionMismatch` is thrown otherwise.
  - The rules of [`observed_weights`](@ref).

# Returns

  - `X_obs`: The observations of `X` at which `obs` is `true`.
  - `w_obs::Option{<:VecNum}`: The observation weights of `X_obs`, or `nothing` when `w` is `nothing`.

# Examples

```jldoctest
julia> X, w = PortfolioOptimisers.observed_sample([0.01, NaN, -0.02, 0.03], fweights([2, 1, 1, 3]),
                                                  [true, false, true, true]);

julia> X
3-element Vector{Float64}:
  0.01
 -0.02
  0.03

julia> w
3-element FrequencyWeights{Int64, Int64, Vector{Int64}}:
 2
 1
 3
```

# Related

  - [`observed_weights`](@ref)
  - [`has_positive_weight`](@ref)
  - [`get_observation_weights`](@ref)
  - [`ObsWeights`](@ref)
"""
function observed_sample(X::VecNum, w::Option{<:Union{ObsWeights, VecNum}},
                         obs::AbstractVector{Bool}; kwargs...)
    assert_observation_mask(obs, length(X))
    X_obs = all(obs) ? X : X[obs]
    return X_obs, observed_weights(w, X_obs, obs; kwargs...)
end
function observed_sample(X::MatNum, w::Option{<:Union{ObsWeights, VecNum}},
                         obs::AbstractVector{Bool}; dims::Int = 1, kwargs...)
    assert_dims(dims)
    assert_observation_mask(obs, size(X, dims))
    X_obs = if all(obs)
        X
    elseif dims == 1
        X[obs, :]
    else
        X[:, obs]
    end
    return X_obs, observed_weights(w, X_obs, obs; dims = dims, kwargs...)
end
function observed_sample(X::VecNum, w::Option{<:Union{ObsWeights, VecNum}}; kwargs...)
    return observed_sample(X, w, isfinite.(X); kwargs...)
end
function observed_sample(X::MatNum, w::Option{<:Union{ObsWeights, VecNum}}; dims::Int = 1,
                         kwargs...)
    assert_dims(dims)
    obs = vec(all(isfinite, X; dims = 3 - dims))
    return observed_sample(X, w, obs; dims = dims, kwargs...)
end
"""
    assert_observation_mask(obs::AbstractVector{Bool}, T::Integer) -> Nothing

Check that the mask `obs` of the observed sample has one entry for each of the `T` observations of the sample.

# Arguments

  - `obs`: `true` at each observation that the fit reads.
  - `T`: The number of observations of the sample.

# Validation

  - `length(obs) == T`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`observed_sample`](@ref)
"""
function assert_observation_mask(obs::AbstractVector{Bool}, T::Integer)::Nothing
    @argcheck(length(obs) == T,
              DimensionMismatch("length(obs) == $T must hold: the mask of the observed sample has one entry for each observation. Got\nlength(obs) => $(length(obs))"))
    return nothing
end
"""
    observed_weights(w::Nothing, X_obs::VecNum_MatNum, obs::AbstractVector{Bool};
                     kwargs...) -> Nothing
    observed_weights(w::DynamicAbstractWeights, X_obs::VecNum_MatNum,
                     obs::AbstractVector{Bool}; kwargs...) -> VecNum
    observed_weights(w::VecNum, X_obs::VecNum_MatNum, obs::AbstractVector{Bool};
                     kwargs...) -> VecNum

Return the observation weights of the observed sample `X_obs`, the observations of a sample at which `obs` is `true`.

The rule is one for every type of weights. An observation that the fit drops takes its weight with it, and the other observations keep their weights.

  - **Every type drops the observation.** A `StatsBase.FrequencyWeights` counts the copies of one observation, so the drop removes every copy.
  - **The weights keep their type.** The function indexes `w` with `obs`, and `StatsBase` returns weights of the type of `w`. `StatsBase.varcorrection` then gives the bias correction of that type on the observed sample: ``\\sum_t w_t - 1`` for frequency weights, ``\\sum_t w_t - \\sum_t w_t^2 / \\sum_t w_t`` for analytic weights, and ``\\left(\\sum_t w_t\\right) (n_{+} - 1) / n_{+}`` for probability weights, where ``n_{+}`` counts the observed weights that are not zero. A `view` of `w` is a `SubArray`, which loses the type, so the function never takes one.
  - **The weights are not scaled.** The analytic, probability and plain corrections do not change when the weights scale, and every weighted kernel divides by ``\\sum_t w_t``. A frequency weight is a count, so a scale makes it wrong.
  - **A [`DynamicAbstractWeights`](@ref) resolves on the observed sample.** It is a rule that computes the weights from the sample, so it reads the sample that the fit reads. A decay then ages each observation by the observations that the fit reads, as [`ExpWeightedCovariance`](@ref) ages each asset on the clock of its own valid observations.

A sample with no observation of positive weight left has no value. The consumer answers `NaN`, and documents it. [`has_positive_weight`](@ref) answers the test.

# Algorithm

The method that Julia selects on the type of `w` is the algorithm.

 1. `nothing`: return `nothing`. The fit stays unweighted.
 2. A [`DynamicAbstractWeights`](@ref): resolve it on `X_obs` with [`get_observation_weights`](@ref), and pass `kwargs...` to it.
 3. A weight vector: return the weights of the two-argument method at `obs`. It checks that `w` has one entry for each entry of `obs`, and returns `w` itself when `obs` keeps every observation.

# Arguments

  - $(arg_dict[:oow])
  - `X_obs`: The observed sample. A dynamic weight reads it. The other methods ignore it.
  - `obs`: `true` at each observation of the full sample that the fit reads.
  - `kwargs...`: More keyword arguments for [`get_observation_weights`](@ref). A matrix sample passes `dims` here.

# Validation

  - A weight vector has `length(w) == length(obs)`. A `DimensionMismatch` that names `w` is thrown otherwise.

# Returns

  - `w_obs::Option{<:VecNum}`: The weights of the observed sample, of the type of `w`, or `nothing`.

# Related

  - [`observed_sample`](@ref)
  - [`has_positive_weight`](@ref)
  - [`get_observation_weights`](@ref)
  - [`DynamicAbstractWeights`](@ref)
"""
function observed_weights(::Nothing, ::VecNum_MatNum, ::AbstractVector{Bool}; kwargs...)
    return nothing
end
function observed_weights(w::DynamicAbstractWeights, X_obs::VecNum_MatNum,
                          ::AbstractVector{Bool}; kwargs...)
    return get_observation_weights(w, X_obs; kwargs...)
end
function observed_weights(w::VecNum, ::VecNum_MatNum, obs::AbstractVector{Bool}; kwargs...)
    return observed_weights(w, obs)
end
"""
    observed_weights(w::Nothing, obs::AbstractVector{Bool}) -> Nothing
    observed_weights(w::DynamicAbstractWeights, obs::AbstractVector{Bool})
        -> DynamicAbstractWeights
    observed_weights(w::VecNum, obs::AbstractVector{Bool}) -> VecNum

Return the observation weights that an estimator holds, cut to the observations at which `obs` is `true`, before the fit.

This is the rule of the three-argument method for weights that an estimator holds in a field, before it reads a sample. [`shared_series_estimator`](@ref) calls it on each weights field of an estimator tree, through [`obs_weights_index`](@ref), when a Shared Series gap drops observations from the returns data.

# Algorithm

The method that Julia selects on the type of `w` is the algorithm.

 1. `nothing`: return `nothing`. The fit stays unweighted.
 2. A [`DynamicAbstractWeights`](@ref): return `w`. It resolves on the sample that the fit reads, which is the observed sample, as the three-argument method states.
 3. A weight vector: check that it has one entry for each entry of `obs`. Return `w` itself when `obs` keeps every observation, and the weights of the same type at the kept observations otherwise, through [`nothing_scalar_array_getindex`](@ref). That function also keeps the type of weights that store a range.

# Arguments

  - $(arg_dict[:oow])
  - `obs`: `true` at each observation of the full sample that the fit reads.

# Validation

  - A weight vector has `length(w) == length(obs)`. A `DimensionMismatch` that names `w` is thrown otherwise.

# Returns

  - `w_obs::Option{<:ObsWeights}`: The weights of the observed sample, of the type of `w`, or `nothing`.

# Examples

```jldoctest
julia> PortfolioOptimisers.observed_weights(aweights(1.0:4.0), [true, false, true, true])
3-element AnalyticWeights{Float64, Float64, Vector{Float64}}:
 1.0
 3.0
 4.0
```

# Related

  - [`observed_sample`](@ref)
  - [`obs_weights_index`](@ref)
  - [`shared_series_estimator`](@ref)
"""
function observed_weights(::Nothing, ::AbstractVector{Bool})
    return nothing
end
function observed_weights(w::DynamicAbstractWeights, ::AbstractVector{Bool})
    return w
end
function observed_weights(w::VecNum, obs::AbstractVector{Bool})
    @argcheck(length(w) == length(obs),
              DimensionMismatch("length(w) == length(obs) must hold: an observation that the fit drops takes its own observation weight with it. Got\nlength(w) => $(length(w))\nlength(obs) => $(length(obs))"))
    return all(obs) ? w : nothing_scalar_array_getindex(w, obs)
end
"""
    has_positive_weight(w::Nothing, T::Integer) -> Bool
    has_positive_weight(w::AbstractVector, T::Integer) -> Bool

Answer whether a sample of `T` observations under the weights `w` holds an observation of positive weight.

A sample with no such observation has no weighted value: a weighted mean divides zero by zero. A fit on an observed sample, see [`observed_sample`](@ref), answers `NaN` when this function answers `false`, and documents it.

# Algorithm

The method that Julia selects on the type of `w` is the algorithm.

 1. `nothing`: every observation weighs one, so answer `T > 0`.
 2. A weight vector: answer whether an entry of `w` is above zero.

# Arguments

  - `w`: The resolved observation weights of the sample, or `nothing`. [`observed_weights`](@ref) resolves them.
  - `T`: The number of observations of the sample.

# Returns

  - `flag::Bool`: `true` when an observation of the sample has a positive weight.

# Related

  - [`observed_sample`](@ref)
  - [`observed_weights`](@ref)
  - [`observation_weight_count`](@ref)
"""
function has_positive_weight(::Nothing, T::Integer)
    return T > zero(T)
end
function has_positive_weight(w::AbstractVector, ::Integer)
    return any(x -> x > zero(x), w)
end
export KishObservationCount
public AbstractObservationCountAlgorithm, weight_observation_count
public get_observation_weights
