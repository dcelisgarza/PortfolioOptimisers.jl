"""
    abstract type AbstractDistanceEstimator <: AbstractEstimator end

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
    abstract type AbstractDistanceAlgorithm <: AbstractAlgorithm end

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
    struct SimpleDistance <: AbstractDistanceAlgorithm end

Simple distance algorithm for portfolio optimization.

`SimpleDistance` specifies the use of a basic distance metric (typically Euclidean or squared Euclidean distance) for distance-based estimation in PortfolioOptimisers.jl. It is used as an algorithm type for distance estimators, enabling straightforward computation of pairwise distances between assets or features.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct SimpleDistance <: AbstractDistanceAlgorithm end

"""
    struct SimpleAbsoluteDistance <: AbstractDistanceAlgorithm end

Simple absolute distance algorithm for portfolio optimization.

`SimpleAbsoluteDistance` specifies the use of the absolute distance metric (L1 norm) for distance-based estimation in PortfolioOptimisers.jl. It is used as an algorithm type for distance estimators, enabling computation of pairwise absolute differences between assets or features.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct SimpleAbsoluteDistance <: AbstractDistanceAlgorithm end

"""
    struct LogDistance <: AbstractDistanceAlgorithm end

Log distance algorithm for portfolio optimization.

`LogDistance` specifies the use of a logarithmic distance metric for distance-based estimation in PortfolioOptimisers.jl. This algorithm is useful for measuring relative differences or ratios between asset returns or features, and can be more robust to scale differences than standard Euclidean or absolute distances.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct LogDistance <: AbstractDistanceAlgorithm end

"""
    struct CorrelationDistance <: AbstractDistanceAlgorithm end

Correlation distance algorithm for portfolio optimization.

`CorrelationDistance` specifies the use of a correlation-based distance metric for distance-based estimation in PortfolioOptimisers.jl. This algorithm measures the dissimilarity between assets or features based on their correlation, typically using the formula `distance = 1 - correlation`. It is useful for clustering, risk analysis, and constructing distance matrices that reflect the co-movement structure of asset returns.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct CorrelationDistance <: AbstractDistanceAlgorithm end

"""
    struct CanonicalDistance <: AbstractDistanceAlgorithm end

Canonical distance algorithm for portfolio optimization.

`CanonicalDistance` specifies the use of a canonical (or Mahalanobis-like) distance metric for distance-based estimation in PortfolioOptimisers.jl. This algorithm measures the dissimilarity between assets or features by accounting for the covariance structure of the data, making it sensitive to correlations and scale differences. It is useful for clustering, risk analysis, and constructing distance matrices that reflect both variance and correlation among asset returns.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct CanonicalDistance <: AbstractDistanceAlgorithm end

"""
    struct VariationInfoDistance{T1, T2} <: AbstractDistanceAlgorithm
        bins::T1
        normalise::T2
    end

Variation of Information (VI) distance algorithm for portfolio optimization.

`VariationInfoDistance` specifies the use of the Variation of Information (VI) metric, an information-theoretic distance based on entropy and mutual information, for distance-based estimation in PortfolioOptimisers.jl. This algorithm is useful for quantifying the dissimilarity between distributions of asset returns or features, and can be applied to both continuous and discrete data via binning.

# Fields

  - `bins`: Binning strategy or number of bins. If an integer, must be strictly positive.
  - `normalise`: Whether to normalise the VI distance to the range [0, 1].

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct VariationInfoDistance{T1, T2} <: AbstractDistanceAlgorithm
    bins::T1
    normalise::T2
end
"""
    VariationInfoDistance(; bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                           normalise::Bool = true)

Construct a [`VariationInfoDistance`](@ref) algorithm for information-theoretic distance estimation.

This constructor creates a `VariationInfoDistance` object with the specified binning strategy and normalisation option. The VI distance quantifies the dissimilarity between distributions using entropy and mutual information, and can be applied to both continuous and discrete data via binning.

# Arguments

  - `bins`: Binning strategy or number of bins. If an integer, must be strictly positive.
  - `normalise`: Whether to normalise the VI distance to the range [0, 1].

# Returns

  - `VariationInfoDistance`: A configured VI distance algorithm.

# Validation

  - If `bins` is an integer, it must be strictly positive.

# Examples

```jldoctest
julia> VariationInfoDistance()
VariationInfoDistance
       bins | HacineGharbiRavier()
  normalise | Bool: true
```

# Related

  - [`VariationInfoDistance`](@ref)
  - [`AbstractDistanceAlgorithm`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
function VariationInfoDistance(;
                               bins::Union{<:AbstractBins, <:Integer} = HacineGharbiRavier(),
                               normalise::Bool = true)
    if isa(bins, Integer)
        @argcheck(bins > zero(bins))
    end
    return VariationInfoDistance(bins, normalise)
end
function distance end
function cor_and_dist end

export SimpleDistance, SimpleAbsoluteDistance, LogDistance, CorrelationDistance,
       CanonicalDistance, VariationInfoDistance, distance, cor_and_dist
