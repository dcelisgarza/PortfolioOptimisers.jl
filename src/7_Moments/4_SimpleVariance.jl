"""
```julia
struct SimpleVariance{T1, T2, T3} <: AbstractVarianceEstimator
    me::T1
    w::T2
    corrected::T3
end
```

A flexible variance estimator for PortfolioOptimisers.jl supporting optional expected returns estimators, observation weights, and bias correction.

`SimpleVariance` enables users to specify an expected returns estimator (for mean-centering), optional observation weights, and whether to apply bias correction (Bessel's correction). This type is suitable for both unweighted and weighted variance estimation workflows.

# Fields

  - `me`: Optional expected returns estimator. If `nothing`, the mean is not estimated.
  - `w`: Optional observation weights. If `nothing`, the estimator is unweighted.
  - `corrected`: Whether to apply Bessel's correction (unbiased variance).

# Constructor

```julia
SimpleVariance(;
               me::Union{Nothing, <:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
               w::Union{Nothing, <:AbstractWeights} = nothing, corrected::Bool = true)
```

Keyword arguments correspond to the fields above.

## Validation

  - If `w` is provided, `!isempty(w)`.

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

  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
  - [`std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::AbstractVector; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractVector; mean = nothing)`](@ref)
"""
struct SimpleVariance{T1, T2, T3} <: AbstractVarianceEstimator
    me::T1
    w::T2
    corrected::T3
end
function SimpleVariance(;
                        me::Union{Nothing, <:AbstractExpectedReturnsEstimator} = SimpleExpectedReturns(),
                        w::Union{Nothing, <:AbstractWeights} = nothing,
                        corrected::Bool = true)
    if isa(me, AbstractWeights)
        @argcheck(!isempty(w))
    end
    return SimpleVariance(me, w, corrected)
end

"""
```julia
std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
```

Compute the standard deviation using a [`SimpleVariance`](@ref) estimator for an array.

This method computes the standard deviation of the input array `X` using the configuration specified in `ve`, including optional mean-centering (via `ve.me`), observation weights (`ve.w`), and bias correction (`ve.corrected`). If a mean is not provided, it is estimated using the expected returns estimator in `ve.me`.

# Arguments

  - `ve`: Variance estimator specifying the mean estimator, weights, and bias correction.
  - `X`: Data array (vector or matrix) for which to compute the standard deviation.
  - `dims`: Dimension along which to compute the standard deviation (for matrices).
  - `mean`: Optional mean value or vector for centering. If not provided, estimated using `ve.me`.
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - `sd::Vector{<:Real}`: Standard deviation vector of `X`.

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
  - [`std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::AbstractVector; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractVector; mean = nothing)`](@ref)
"""
function Statistics.std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ve.me, X; dims = dims, kwargs...) : mean
    return if isnothing(ve.w)
        std(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        std(X, ve.w, dims; corrected = ve.corrected, mean = mu)
    end
end

"""
```julia
std(ve::SimpleVariance, X::AbstractVector; mean = nothing)
```

Compute the standard deviation using a [`SimpleVariance`](@ref) estimator for a vector.

This method computes the standard deviation of the input vector `X` using the configuration specified in `ve`, including optional observation weights (`ve.w`) and bias correction (`ve.corrected`). If a mean is provided, it is used for centering; otherwise, the default mean is used.

# Arguments

  - `ve`: Variance estimator specifying weights and bias correction.
  - `X`: Data vector for which to compute the standard deviation.
  - `mean`: Optional Mean value for centering. If not provided, the default mean is used.

# Returns

  - `sd::Real`: Standard deviation of `X`.

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
  - [`std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
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
```julia
var(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)
```

Compute the variance using a [`SimpleVariance`](@ref) estimator for an array.

This method computes the variance of the input array `X` using the configuration specified in `ve`, including optional mean-centering (via `ve.me`), observation weights (`ve.w`), and bias correction (`ve.corrected`). If a mean is not provided, it is estimated using the expected returns estimator in `ve.me`.

# Arguments

  - `ve`: Variance estimator specifying the mean estimator, weights, and bias correction.
  - `X`: Data array (vector or matrix) for which to compute the variance.
  - `dims`: Dimension along which to compute the variance (for matrices).
  - `mean`: Optional mean value or vector for centering. If not provided, estimated using `ve.me`.
  - `kwargs...`: Additional keyword arguments passed to the mean estimator.

# Returns

  - `v::Vector{<:Real}`: Variance vector of `X`.

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
  - [`std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::AbstractVector; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractVector; mean = nothing)`](@ref)
"""
function Statistics.var(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1,
                        mean = nothing, kwargs...)
    mu = isnothing(mean) ? Statistics.mean(ve.me, X; dims = dims, kwargs...) : mean
    return if isnothing(ve.w)
        var(X; dims = dims, corrected = ve.corrected, mean = mu)
    else
        var(X, ve.w, dims; corrected = ve.corrected, mean = mu)
    end
end

"""
```julia
var(ve::SimpleVariance, X::AbstractVector; mean = nothing)
```

Compute the variance using a [`SimpleVariance`](@ref) estimator for a vector.

This method computes the variance of the input vector `X` using the configuration specified in `ve`, including optional observation weights (`ve.w`) and bias correction (`ve.corrected`). If a mean is provided, it is used for centering; otherwise, the default mean is used.

# Arguments

  - `ve`: Variance estimator specifying weights and bias correction.
  - `X`: Data vector for which to compute the variance.
  - `mean`: Optional mean value for centering. If not provided, the default mean is used.

# Returns

  - `v::Real`: Variance of `X`.

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
  - [`std(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`std(ve::SimpleVariance, X::AbstractVector; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
  - [`var(ve::SimpleVariance, X::AbstractMatrix; dims::Int = 1, mean = nothing, kwargs...)`](@ref)
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
