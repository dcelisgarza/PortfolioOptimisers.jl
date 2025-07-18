"""
    SimpleVariance{T1 <: Union{Nothing, <:AbstractExpectedReturnsEstimator},
                   T2 <: Union{Nothing, <:AbstractWeights}, T3 <: Bool}

A flexible variance estimator for PortfolioOptimisers.jl supporting optional expected returns estimators, observation weights, and bias correction.

`SimpleVariance` enables users to specify an expected returns estimator (for mean-centering), optional observation weights, and whether to apply bias correction (Bessel's correction). This type is suitable for both unweighted and weighted variance estimation workflows.

# Fields

  - `me::Union{Nothing, <:AbstractExpectedReturnsEstimator}`: Optional expected returns estimator. If `nothing`, the mean is not estimated.
  - `w::Union{Nothing, <:AbstractWeights}`: Optional observation weights. If `nothing`, the estimator is unweighted.
  - `corrected::Bool`: Whether to apply Bessel's correction (unbiased variance).

# Constructor

    SimpleVariance(; me::Union{Nothing, <:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
                    w::Union{Nothing, <:AbstractWeights} = nothing,
                    corrected::Bool = true)

Construct a `SimpleVariance` estimator with the specified expected returns estimator, optional weights, and bias correction flag.

# Related

  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::AbstractVector; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractVector; mean = nothing)`](@ref)
"""
struct SimpleVariance{T1 <: Union{Nothing, <:AbstractExpectedReturnsEstimator},
                      T2 <: Union{Nothing, <:AbstractWeights}, T3 <: Bool} <:
       AbstractVarianceEstimator
    me::T1
    w::T2
    corrected::T3
end
"""
    SimpleVariance(; me::Union{Nothing, <:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
                    w::Union{Nothing, <:AbstractWeights} = nothing,
                    corrected::Bool = true)

Construct a [`SimpleVariance`](@ref) estimator for flexible variance estimation with optional mean-centering, observation weights, and bias correction.

This constructor creates a `SimpleVariance` object using the specified expected returns estimator for mean-centering, optional observation weights, and a flag for Bessel's correction (bias correction). If no weights are provided, the estimator defaults to unweighted variance estimation. If weights are provided, they must not be empty.

# Arguments

  - `me::Union{Nothing, <:AbstractExpectedReturnsEstimator}`: Expected returns estimator. Necessary when estimating the variance or standard deviation along the dimension of a matrix. Not used when computed on a vector.
  - `w::Union{Nothing, <:AbstractWeights}`: Optional observation weights. If `nothing`, the estimator is unweighted.
  - `corrected::Bool`: Whether to apply Bessel's correction.

# Returns

  - `SimpleVariance`: A variance estimator configured with the specified mean estimator, weights, and bias correction flag.

# Validation

  - If `w` is provided, it must not be empty.

# Examples

```jldoctest
julia> using StatsBase

julia> sv = SimpleVariance()
SimpleVariance
         me | SimpleExpectedReturns
            |   w | nothing
          w | nothing
  corrected | Bool: true

julia> w = Weights([0.2, 0.3, 0.5]);

julia> svw = SimpleVariance(; w = w, corrected = false)
SimpleVariance
         me | SimpleExpectedReturns
            |   w | nothing
          w | StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected | Bool: false
```

# Related

  - [`SimpleVariance`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::AbstractVector; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractVector; mean = nothing)`](@ref)
"""
function SimpleVariance(;
                        me::Union{Nothing, <:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
                        w::Union{Nothing, <:AbstractWeights} = nothing,
                        corrected::Bool = true)
    if isa(me, AbstractWeights)
        @smart_assert(!isempty(w))
    end
    return SimpleVariance{typeof(me), typeof(w), typeof(corrected)}(me, w, corrected)
end

"""
    std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)

Compute the standard deviation using a [`SimpleVariance`](@ref) estimator for an array.

This method computes the standard deviation of the input array `X` using the configuration specified in `ve`, including optional mean-centering (via `ve.me`), observation weights (`ve.w`), and bias correction (`ve.corrected`). If a mean is not provided, it is estimated using the expected returns estimator in `ve.me`.

# Arguments

  - `ve::SimpleVariance`: Variance estimator specifying the mean estimator, weights, and bias correction.
  - `X::AbstractArray`: Data array (vector or matrix) for which to compute the standard deviation.
  - `dims::Int`: Dimension along which to compute the standard deviation (for matrices).
  - `mean`: Optional mean value or vector for centering. If not provided, estimated using `ve.me`.
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - Standard deviation of `X`, computed according to the estimator configuration.

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me | SimpleExpectedReturns
            |   w | nothing
          w | nothing
  corrected | Bool: true

julia> Xmat = [1.0 2.0; 3.0 4.0];

julia> std(sv, Xmat; dims = 1)
1×2 Matrix{Float64}:
 1.41421  1.41421
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.std`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.std)
  - [`std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::AbstractVector; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractVector; mean = nothing)`](@ref)
"""
function Statistics.std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing,
                        kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ve.me, X; dims = dims, kwargs...) : mean
    return if isnothing(ve.w)
        std(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        std(X, ve.w, dims; corrected = ve.corrected, mean = mu)
    end
end

"""
    std(ve::SimpleVariance, X::AbstractVector; mean = nothing)

Compute the standard deviation using a [`SimpleVariance`](@ref) estimator for a vector.

This method computes the standard deviation of the input vector `X` using the configuration specified in `ve`, including optional observation weights (`ve.w`) and bias correction (`ve.corrected`). If a mean is provided, it is used for centering; otherwise, the default mean is used.

# Arguments

  - `ve::SimpleVariance`: Variance estimator specifying weights and bias correction.
  - `X::AbstractVector`: Data vector for which to compute the standard deviation.
  - `mean`: Optional mean value for centering. If not provided, the default mean is used.

# Returns

  - Standard deviation of `X`, computed according to the estimator configuration.

# Examples

```jldoctest
julia> using StatsBase

julia> sv = SimpleVariance()
SimpleVariance
         me | SimpleExpectedReturns
            |   w | nothing
          w | nothing
  corrected | Bool: true

julia> X = [1.0, 2.0, 3.0];

julia> std(sv, X)
1.0

julia> w = Weights([0.2, 0.3, 0.5]);

julia> svw = SimpleVariance(; w = w, corrected = false)
SimpleVariance
         me | SimpleExpectedReturns
            |   w | nothing
          w | StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected | Bool: false

julia> std(svw, X)
0.7810249675906654
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.std`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.std)
  - [`std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractVector; mean = nothing)`](@ref)
"""
function Statistics.std(ve::SimpleVariance, X::AbstractVector; mean = nothing)
    return if isnothing(ve.w)
        std(X; corrected = ve.corrected, mean = mean)
    else
        std(X, ve.w; corrected = ve.corrected, mean = mean)
    end
end

"""
    var(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)

Compute the variance using a [`SimpleVariance`](@ref) estimator for an array.

This method computes the variance of the input array `X` using the configuration specified in `ve`, including optional mean-centering (via `ve.me`), observation weights (`ve.w`), and bias correction (`ve.corrected`). If a mean is not provided, it is estimated using the expected returns estimator in `ve.me`.

# Arguments

  - `ve::SimpleVariance`: Variance estimator specifying the mean estimator, weights, and bias correction.
  - `X::AbstractArray`: Data array (vector or matrix) for which to compute the variance.
  - `dims::Int`: Dimension along which to compute the variance (for matrices).
  - `mean`: Optional mean value or vector for centering. If not provided, estimated using `ve.me`.
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - Variance of `X`, computed according to the estimator configuration.

# Examples

```jldoctest
julia> sv = SimpleVariance()
SimpleVariance
         me | SimpleExpectedReturns
            |   w | nothing
          w | nothing
  corrected | Bool: true

julia> Xmat = [1.0 2.0; 3.0 4.0];

julia> var(sv, Xmat; dims = 1)
1×2 Matrix{Float64}:
 2.0  2.0
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.var`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var)
  - [`std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::AbstractVector; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractVector; mean = nothing)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing,
                        kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ve.me, X; dims = dims, kwargs...) : mean
    return if isnothing(ve.w)
        var(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        var(X, ve.w, dims; corrected = ve.corrected, mean = mu)
    end
end
"""
    var(ve::SimpleVariance, X::AbstractVector; mean = nothing)

Compute the variance using a [`SimpleVariance`](@ref) estimator for a vector.

This method computes the variance of the input vector `X` using the configuration specified in `ve`, including optional observation weights (`ve.w`) and bias correction (`ve.corrected`). If a mean is provided, it is used for centering; otherwise, the default mean is used.

# Arguments

  - `ve::SimpleVariance`: Variance estimator specifying weights and bias correction.
  - `X::AbstractVector`: Data vector for which to compute the variance.
  - `mean`: Optional mean value for centering. If not provided, the default mean is used.

# Returns

  - Variance of `X`, computed according to the estimator configuration.

# Examples

```jldoctest
julia> using StatsBase

julia> sv = SimpleVariance()
SimpleVariance
         me | SimpleExpectedReturns
            |   w | nothing
          w | nothing
  corrected | Bool: true

julia> X = [1.0, 2.0, 3.0];

julia> var(sv, X)
1.0

julia> w = Weights([0.2, 0.3, 0.5]);

julia> svw = SimpleVariance(; w = w, corrected = false)
SimpleVariance
         me | SimpleExpectedReturns
            |   w | nothing
          w | StatsBase.Weights{Float64, Float64, Vector{Float64}}: [0.2, 0.3, 0.5]
  corrected | Bool: false

julia> var(svw, X)
0.61
```

# Related

  - [`SimpleVariance`](@ref)
  - [`Statistics.var`](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var)
  - [`std(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::AbstractVector; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractArray; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, X::AbstractVector; mean = nothing)
    return if isnothing(ve.w)
        var(X; corrected = ve.corrected, mean = mean)
    else
        var(X, ve.w; corrected = ve.corrected, mean = mean)
    end
end
function factory(ve::SimpleVariance, w::Union{Nothing, <:AbstractWeights} = nothing)
    return SimpleVariance(; me = factory(ve.me, w), w = isnothing(w) ? ve.w : w,
                          corrected = ve.corrected)
end

export SimpleVariance
