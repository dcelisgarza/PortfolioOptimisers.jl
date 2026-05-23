"""
$(DocStringExtensions.TYPEDEF)

A flexible variance estimator for `PortfolioOptimisers.jl` supporting optional expected returns estimators, observation weights, and bias correction.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SimpleVariance(;
        me::Option{<:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
        w::Option{<:ObsWeights} = nothing,
        corrected::Bool = true
    ) -> SimpleVariance

Keywords correspond to the struct's fields.

## Validation

$(val_dict[:oow])

# Examples

```jldoctest
julia> SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> SimpleVariance(; w = StatsBase.Weights([0.2, 0.3, 0.5]), corrected = false)
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected ┴ Bool: false
```

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
"""
@concrete struct SimpleVariance <: AbstractVarianceEstimator
    "$(field_dict[:ome])"
    me
    "$(field_dict[:ow])"
    w
    "$(field_dict[:corrected])"
    corrected
    function SimpleVariance(me::Option{<:AbstractExpectedReturnsEstimator},
                            w::Option{<:ObsWeights}, corrected::Bool)
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(me), typeof(w), typeof(corrected)}(me, w, corrected)
    end
end
function SimpleVariance(;
                        me::Option{<:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
                        w::Option{<:ObsWeights} = nothing,
                        corrected::Bool = true)::SimpleVariance
    return SimpleVariance(me, w, corrected)
end
"""
    Statistics.std(
        ve::SimpleVariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...,
    ) -> ArrNum

Compute the standard deviation using a [`SimpleVariance`](@ref) estimator for a matrix.

This method computes the standard deviation of the input matrix `X` using the configuration specified in `ve`.

# Summary Statistics

Unweighted:

```math
\\hat{\\sigma}_j = \\sqrt{\\hat{\\sigma}^2_j}
```

where, for `corrected = true`:

```math
\\hat{\\sigma}^2_j = \\frac{1}{T-1} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2
```

and for `corrected = false`:

```math
\\hat{\\sigma}^2_j = \\frac{1}{T} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2
```

Weighted:

```math
\\hat{\\sigma}^2_j = \\frac{\\sum_{t=1}^{T} w_t (r_{tj} - \\hat{\\mu}_j)^2}{\\sum_{t=1}^{T} w_t - c}
```

where ``c = 1`` if `corrected = true`, else ``c = 0``.

Where:

  - ``\\hat{\\sigma}^2_j``: Estimated variance of asset ``j``.
  - ``r_{tj}``: Return of asset ``j`` at time ``t``.
  - ``\\hat{\\mu}_j``: Estimated mean of asset ``j``.
  - ``T``: Number of observations.
  - ``w_t``: Observation weight at time ``t``.
  - ``c``: Bias correction factor.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - $(ret_dict[:stdarr])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> Xmat = [1.0 2.0; 3.0 4.0];

julia> std(sv, Xmat; dims = 1)
1×2 Matrix{Float64}:
 1.41421  1.41421
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.std`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.std)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
"""
function Statistics.std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ve.me, X; dims = dims, kwargs...) : mean
    w = get_observation_weights(ve.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        Statistics.std(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        Statistics.std(X, w, dims; corrected = ve.corrected, mean = mu)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`SimpleVariance{Nothing}` overload of [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Uses [`SimpleExpectedReturns`](@ref) to compute the mean when none is provided, ignoring the `me` field.
"""
function Statistics.std(ve::SimpleVariance{Nothing}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    mu = if isnothing(mean)
        Statistics.mean(SimpleExpectedReturns(), X; dims = dims, kwargs...)
    else
        mean
    end
    w = get_observation_weights(ve.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        Statistics.std(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        Statistics.std(X, w, dims; corrected = ve.corrected, mean = mu)
    end
end
"""
    Statistics.std(
        ve::SimpleVariance,
        X::VecNum;
        mean = nothing
    ) -> Number

Compute the standard deviation using a [`SimpleVariance`](@ref) estimator for a vector.

This method computes the standard deviation of the input vector `X` using the configuration specified in `ve`.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:Xv])
  - $(arg_dict[:omean])

# Returns

  - $(ret_dict[:stdnum])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> X = [1.0, 2.0, 3.0];

julia> std(sv, X)
1.0

julia> svw = SimpleVariance(; w = StatsBase.Weights([0.2, 0.3, 0.5]), corrected = false)
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected ┴ Bool: false

julia> std(svw, X)
0.7810249675906654
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.std`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.std)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
"""
function Statistics.std(ve::SimpleVariance, X::VecNum; mean = nothing)
    w = get_observation_weights(ve.w, X)
    return if isnothing(w)
        Statistics.std(X; corrected = ve.corrected, mean = mean)
    else
        Statistics.std(X, w; corrected = ve.corrected, mean = mean)
    end
end
"""
    Statistics.var(
        ve::SimpleVariance,
        X::MatNum;
        dims::Int = 1,
        mean = nothing,
        kwargs...
    ) -> ArrNum

Compute the variance using a [`SimpleVariance`](@ref) estimator for a matrix.

This method computes the variance of the input matrix `X` using the configuration specified in `ve`.

# Summary Statistics

Unweighted, for `corrected = true`:

```math
\\hat{\\sigma}^2_j = \\frac{1}{T-1} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2
```

Unweighted, for `corrected = false`:

```math
\\hat{\\sigma}^2_j = \\frac{1}{T} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2
```

Weighted:

```math
\\hat{\\sigma}^2_j = \\frac{\\sum_{t=1}^{T} w_t (r_{tj} - \\hat{\\mu}_j)^2}{\\sum_{t=1}^{T} w_t - c}
```

where ``c = 1`` if `corrected = true`, else ``c = 0``.

Where:

  - ``\\hat{\\sigma}^2_j``: Estimated variance of asset ``j``.
  - ``r_{tj}``: Return of asset ``j`` at time ``t``.
  - ``\\hat{\\mu}_j``: Estimated mean of asset ``j``.
  - ``T``: Number of observations.
  - ``w_t``: Observation weight at time ``t``.
  - ``c``: Bias correction factor.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:omean])
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - $(ret_dict[:vararr])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> Xmat = [1.0 2.0; 3.0 4.0];

julia> var(sv, Xmat; dims = 1)
1×2 Matrix{Float64}:
 2.0  2.0
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.var`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`var(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing,
                        kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ve.me, X; dims = dims, kwargs...) : mean
    w = get_observation_weights(ve.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        Statistics.var(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        Statistics.var(X, w, dims; corrected = ve.corrected, mean = mu)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`SimpleVariance{Nothing}` overload of [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Uses [`SimpleExpectedReturns`](@ref) to compute the mean when none is provided, ignoring the `me` field.
"""
function Statistics.var(ve::SimpleVariance{Nothing}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    me = SimpleExpectedReturns()
    mu = isnothing(mean) ? Statistics.mean(me, X; dims = dims, kwargs...) : mean
    w = get_observation_weights(me.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        Statistics.var(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        Statistics.var(X, w, dims; corrected = ve.corrected, mean = mu)
    end
end
"""
    Statistics.var(
        ve::SimpleVariance,
        X::VecNum;
        mean = nothing
    ) -> Number

Compute the variance using a [`SimpleVariance`](@ref) estimator for a vector.

This method computes the variance of the input vector `X` using the configuration specified in `ve`.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:Xv])
  - $(arg_dict[:omean])

# Returns

  - $(ret_dict[:varnum])

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> X = [1.0, 2.0, 3.0];

julia> var(sv, X)
1.0

julia> svw = SimpleVariance(; w = StatsBase.Weights([0.2, 0.3, 0.5]), corrected = false)
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected ┴ Bool: false

julia> var(svw, X)
0.61
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.var`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var)
  - [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::VecNum; mean = nothing)`](@ref)
  - [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, X::VecNum; mean = nothing)
    w = get_observation_weights(ve.w, X)
    return if isnothing(w)
        Statistics.var(X; corrected = ve.corrected, mean = mean)
    else
        Statistics.var(X, w; corrected = ve.corrected, mean = mean)
    end
end
"""
    factory(
        ve::SimpleVariance,
        w::ObsWeights
    ) -> SimpleVariance

Return a new `SimpleVariance` estimator with the specified observation weights.

# Arguments

  - $(arg_dict[:ve])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ve])

# Details

  - The mean estimator is updated using `factory(ve.me, w)` for consistency.
  - Sets `w` to the new weights.
  - The bias correction flag is preserved from the original estimator.

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ nothing
          w ┼ nothing
  corrected ┴ Bool: true

julia> factory(sv, StatsBase.Weights([0.2, 0.3, 0.5]))
SimpleVariance
         me ┼ SimpleExpectedReturns
            │   w ┴ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
          w ┼ StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected ┴ Bool: true
```

# Related

  - [`SimpleVariance`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`factory`](@ref)
"""
function factory(ve::SimpleVariance, w::ObsWeights)::SimpleVariance
    return SimpleVariance(; me = factory(ve.me, w), w = w, corrected = ve.corrected)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the simple variance for the `i`-th element(s).

# Arguments

  - $(arg_dict[:ve])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:vev])

# Related

  - [`SimpleVariance`](@ref)
"""
function moment_view(ve::SimpleVariance, i)::SimpleVariance
    return SimpleVariance(; me = moment_view(ve.me, i), w = ve.w, corrected = ve.corrected)
end

export SimpleVariance, var, std
