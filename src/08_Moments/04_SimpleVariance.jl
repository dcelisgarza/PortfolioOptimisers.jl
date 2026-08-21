"""
$(DocStringExtensions.TYPEDEF)

Computes the marginal variance and standard deviation, optionally weighted and optionally bias-corrected.

`me` centres the data when no `mean` is supplied, `w` weights the observations, and `corrected` selects the bias correction.

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

  - $(val_dict[:oow])
  - `corrected = true` needs a weight type that supports bias correction. See the note on the weighted formula in [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref).

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
@propagatable @concrete struct SimpleVariance <: AbstractVarianceEstimator
    """
    $(field_dict[:ome])
    """
    @fprop @vprop me
    """
    $(field_dict[:ow])
    """
    @wprop w
    """
    $(field_dict[:corrected])
    """
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
$(DocStringExtensions.TYPEDSIGNATURES)

Dispersion kernel shared by the [`SimpleVariance`](@ref) methods of `Statistics.std` and `Statistics.var`.

# Arguments

  - `f`: Dispersion function to apply, either `Statistics.std` or `Statistics.var`.
  - `ve::SimpleVariance`: Variance estimator. Supplies the observation weights and the `corrected` flag.
  - `me::AbstractExpectedReturnsEstimator`: Expected returns estimator used when no `mean` is provided. Matrix methods only.
  - `X::VecNum_MatNum`: Data matrix or vector.
  - `dims::Int = 1`: Dimension along which to operate. Matrix methods only.
  - `mean = nothing`: Precomputed mean.
  - `kwargs...`: Forwarded to the mean and weight resolution. Matrix methods only.

# Validation

  - $(val_dict[:dims]) Matrix methods only.

# Returns

  - `sigma::Union{<:Number, <:ArrNum}`: Dispersion of `X` computed by `f`.

# Details

  - For a matrix, resolves the mean from `mean` when given, else from `me`. A vector defers the mean to `f`.
  - Resolves the observation weights from `ve.w` with [`get_observation_weights`](@ref).
  - Applies `f` to the weighted branch when the resolved weights are not `nothing`, else to the unweighted branch.

# Related

  - [`SimpleVariance`](@ref)
  - [`get_observation_weights`](@ref)
"""
function simple_variance_kernel(f::F, ve::SimpleVariance,
                                me::AbstractExpectedReturnsEstimator, X::MatNum;
                                dims::Int = 1, mean = nothing, kwargs...) where {F}
    assert_dims(dims)
    mu = isnothing(mean) ? Statistics.mean(me, X; dims = dims, kwargs...) : mean
    w = get_observation_weights(ve.w, X; dims = dims, kwargs...)
    return if isnothing(w)
        f(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        f(X, w, dims; corrected = ve.corrected, mean = mu)
    end
end
function simple_variance_kernel(f::F, ve::SimpleVariance, X::VecNum;
                                mean = nothing) where {F}
    w = get_observation_weights(ve.w, X)
    return if isnothing(w)
        f(X; corrected = ve.corrected, mean = mean)
    else
        f(X, w; corrected = ve.corrected, mean = mean)
    end
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

# Mathematical definition

Unweighted:

```math
\\begin{align}
\\hat{\\sigma}_j &= \\sqrt{\\hat{\\sigma}^2_j}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}_j``: Standard deviation of asset ``j``.
  - ``\\hat{\\sigma}^2_j``: Variance of asset ``j``.

For `corrected = true`:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{1}{T-1} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2\\,.
\\end{align}
```

For `corrected = false`:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{1}{T} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2\\,.
\\end{align}
```

Weighted:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{\\sum_{t=1}^{T} w_t (r_{tj} - \\hat{\\mu}_j)^2}{\\sum_{t=1}^{T} w_t - c}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}^2_j``: Estimated variance of asset ``j``.
  - ``r_{tj}``: Return of asset ``j`` at time ``t``.
  - ``\\hat{\\mu}_j``: Estimated mean of asset ``j``.
  - ``T``: Number of observations.
  - ``w_t``: Observation weight at time ``t``.
  - ``c``: Bias correction factor, fixed by the **type** of `w`, not by the estimator.

`corrected = false` sets ``c = 0`` for every weight type. `corrected = true` reads ``c`` from the weight type: `StatsBase.FrequencyWeights` gives ``c = 1``, `StatsBase.AnalyticWeights` gives ``c = \\sum_t w_t^2 / \\sum_t w_t``, and `StatsBase.ProbabilityWeights` gives ``c = \\sum_t w_t / T``. **A plain `StatsBase.Weights` supports no correction and throws an `ArgumentError`**, so `SimpleVariance(; w = StatsBase.Weights(...))` must also pass `corrected = false`.

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
    return simple_variance_kernel(Statistics.std, ve, ve.me, X; dims = dims, mean = mean,
                                  kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`SimpleVariance{Nothing}` overload of [`std(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Uses [`SimpleExpectedReturns`](@ref) to compute the mean when none is provided, ignoring the `me` field.
"""
function Statistics.std(ve::SimpleVariance{Nothing}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    return simple_variance_kernel(Statistics.std, ve, SimpleExpectedReturns(), X;
                                  dims = dims, mean = mean, kwargs...)
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
    return simple_variance_kernel(Statistics.std, ve, X; mean = mean)
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

# Mathematical definition

Unweighted, `corrected = true`:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{1}{T-1} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2\\,.
\\end{align}
```

Unweighted, `corrected = false`:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{1}{T} \\sum_{t=1}^{T} (r_{tj} - \\hat{\\mu}_j)^2\\,.
\\end{align}
```

Weighted:

```math
\\begin{align}
\\hat{\\sigma}^2_j &= \\frac{\\sum_{t=1}^{T} w_t (r_{tj} - \\hat{\\mu}_j)^2}{\\sum_{t=1}^{T} w_t - c}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}^2_j``: Estimated variance of asset ``j``.
  - ``r_{tj}``: Return of asset ``j`` at time ``t``.
  - ``\\hat{\\mu}_j``: Estimated mean of asset ``j``.
  - ``T``: Number of observations.
  - ``w_t``: Observation weight at time ``t``.
  - ``c``: Bias correction factor, fixed by the **type** of `w`, not by the estimator.

`corrected = false` sets ``c = 0`` for every weight type. `corrected = true` reads ``c`` from the weight type: `StatsBase.FrequencyWeights` gives ``c = 1``, `StatsBase.AnalyticWeights` gives ``c = \\sum_t w_t^2 / \\sum_t w_t``, and `StatsBase.ProbabilityWeights` gives ``c = \\sum_t w_t / T``. **A plain `StatsBase.Weights` supports no correction and throws an `ArgumentError`**, so `SimpleVariance(; w = StatsBase.Weights(...))` must also pass `corrected = false`.

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
    return simple_variance_kernel(Statistics.var, ve, ve.me, X; dims = dims, mean = mean,
                                  kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

`SimpleVariance{Nothing}` overload of [`var(ve::SimpleVariance, X::MatNum; dims::Int = 1, mean = nothing, kwargs...)`](@ref). Uses [`SimpleExpectedReturns`](@ref) to compute the mean when none is provided, ignoring the `me` field.
"""
function Statistics.var(ve::SimpleVariance{Nothing}, X::MatNum; dims::Int = 1,
                        mean = nothing, kwargs...)
    return simple_variance_kernel(Statistics.var, ve, SimpleExpectedReturns(), X;
                                  dims = dims, mean = mean, kwargs...)
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
    return simple_variance_kernel(Statistics.var, ve, X; mean = mean)
end
export SimpleVariance, var, std
