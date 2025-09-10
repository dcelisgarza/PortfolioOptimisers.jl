"""
```julia
abstract type AbstractDistanceEstimator <: AbstractEstimator end
```

Abstract supertype for all distance estimator types in PortfolioOptimisers.jl.

All concrete types implementing distance-based estimation algorithms should subtype `AbstractDistanceEstimator`. This enables a consistent interface for distance-based measures (such as correlation distance, absolute distance, or information-theoretic distances) throughout the package.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractDistanceAlgorithm`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
abstract type AbstractDistanceEstimator <: AbstractEstimator end

"""
```julia
abstract type AbstractDistanceAlgorithm <: AbstractAlgorithm end
```

Abstract supertype for all distance algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific distance-based algorithms (such as correlation distance, absolute distance, log distance, or information-theoretic distances) should subtype `AbstractDistanceAlgorithm`. This enables flexible extension and dispatch of distance routines for use in portfolio optimization and risk analysis.

# Related

  - [`AbstractDistanceEstimator`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
abstract type AbstractDistanceAlgorithm <: AbstractAlgorithm end

"""
```julia
struct SimpleDistance <: AbstractDistanceAlgorithm end
```

Simple distance algorithm for portfolio optimization.

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{\\dfrac{1 - \\rho_{i,\\,j}}{2}}\\,,
\\end{align}
```

where ``d`` is the distance, ``\\rho`` is the correlation coefficient, and each subscript denotes an asset.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct SimpleDistance <: AbstractDistanceAlgorithm end

"""
```julia
struct SimpleAbsoluteDistance <: AbstractDistanceAlgorithm end
```

Simple absolute distance algorithm for portfolio optimization.

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{1 - \\lvert\\rho_{i,\\,j}\\rvert}\\,,
\\end{align}
```

where ``d`` is the distance, ``\\rho`` is the correlation coefficient, and each subscript denotes an asset.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct SimpleAbsoluteDistance <: AbstractDistanceAlgorithm end

"""
```julia
struct LogDistance <: AbstractDistanceAlgorithm end
```

Logarithmic distance algorithm for portfolio optimization.

```math
\\begin{align}
    d_{i,\\,j} &= -\\log{\\lvert\\rho_{i,\\,j}\\rvert}\\,,
\\end{align}
```

where ``d`` is the distance, ``\\rho`` is the correlation coefficient, and each subscript denotes an asset.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct LogDistance <: AbstractDistanceAlgorithm end

"""
```julia
struct CorrelationDistance <: AbstractDistanceAlgorithm end
```

Correlation distance algorithm for portfolio optimization.

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{1 - \\rho_{i,\\,j}}\\,,
\\end{align}
```

where ``d`` is the distance, ``\\rho`` is the correlation coefficient, and each subscript denotes an asset.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct CorrelationDistance <: AbstractDistanceAlgorithm end

"""
```julia
struct VariationInfoDistance{T1, T2} <: AbstractDistanceAlgorithm
    bins::T1
    normalise::T2
end
```

Variation of Information (VI) distance algorithm for portfolio optimization.

`VariationInfoDistance` specifies the use of the Variation of Information (VI) metric, an information-theoretic distance based on entropy and mutual information.

# Fields

  - `bins`: Binning strategy or number of bins. If an integer, must be strictly positive.
  - `normalise`: Whether to normalise the VI distance to the range [0, 1].

# Constructor

    VariationInfoDistance(; bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                            normalise::Bool = true)

Keyword arguments correspond to the fields above.

## Validation

  - If `bins` is an integer, `bins > 0`.

# Examples

```jldoctest
julia> VariationInfoDistance()
VariationInfoDistance
       bins | HacineGharbiRavier()
  normalise | Bool: true
```

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct VariationInfoDistance{T1, T2} <: AbstractDistanceAlgorithm
    bins::T1
    normalise::T2
end
function VariationInfoDistance(;
                               bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                               normalise::Bool = true)
    if isa(bins, Integer)
        @argcheck(bins > zero(bins))
    end
    return VariationInfoDistance(bins, normalise)
end

"""
```julia
struct CanonicalDistance <: AbstractDistanceAlgorithm end
```

Canonical distance algorithm for portfolio optimization.

Defines the canonical distance metric for a given covariance estimator. The resulting distance metric is consistent with the properties of the covariance estimator (relevant when the covariance estimator is [`MutualInfoCovariance`](@ref)).

| Covariance Estimator                                                               | Distance Metric                 |
| ----------------------------------------------------------------------------------:|:------------------------------- |
| [`MutualInfoCovariance`](@ref)                                                     | [`VariationInfoDistance`](@ref) |
| [`LTDCovariance`](@ref)                                                            | [`LogDistance`](@ref)           |
| [`DistanceCovariance`](@ref)                                                       | [`CorrelationDistance`](@ref)   |
| [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/) | [`SimpleDistance`](@ref)        |

The table also applies to [`PortfolioOptimisersCovariance`](@ref) where `ce` is one of the aforementioned estimators.

When used with a covariance matrix directly, uses [`SimpleDistance`](@ref).

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`MutualInfoCovariance`](@ref)
  - [`LTDCovariance`](@ref)
  - [`DistanceCovariance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct CanonicalDistance <: AbstractDistanceAlgorithm end
function distance end
function cor_and_dist end

export SimpleDistance, SimpleAbsoluteDistance, LogDistance, CorrelationDistance,
       VariationInfoDistance, CanonicalDistance, distance, cor_and_dist
