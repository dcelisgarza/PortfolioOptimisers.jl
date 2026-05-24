"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all distance estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract  types implementing distance-based estimation algorithms should be subtypes of `AbstractDistanceEstimator`.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractDistanceAlgorithm`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
abstract type AbstractDistanceEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all distance algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing specific distance-based algorithms (such as correlation distance, absolute distance, log distance, or information-theoretic distances) should be subtypes of `AbstractDistanceAlgorithm`.

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
$(DocStringExtensions.TYPEDEF)

Simple distance algorithm for portfolio optimization.

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{\\dfrac{1 - \\rho_{i,\\,j}}{2}}\\,,
\\end{align}
```

Where:

  - ``d_{i,\\,j}``: Pairwise distance between assets ``i`` and ``j``.
  - ``\\rho_{i,\\,j}``: Pairwise correlation coefficient between assets ``i`` and ``j``.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct SimpleDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Simple absolute distance algorithm for portfolio optimization.

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{1 - \\lvert\\rho_{i,\\,j}\\rvert}\\,,
\\end{align}
```

Where:

  - ``d_{i,\\,j}``: Pairwise distance between assets ``i`` and ``j``.
  - ``\\rho_{i,\\,j}``: Pairwise correlation coefficient between assets ``i`` and ``j``.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct SimpleAbsoluteDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Logarithmic distance algorithm for portfolio optimization.

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= -\\log{\\lvert\\rho_{i,\\,j}\\rvert}\\,,
\\end{align}
```

Where:

  - ``d_{i,\\,j}``: Pairwise distance between assets ``i`` and ``j``.
  - ``\\rho_{i,\\,j}``: Pairwise correlation coefficient between assets ``i`` and ``j``.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct LogDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Correlation distance algorithm for portfolio optimization.

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{1 - \\rho_{i,\\,j}}\\,,
\\end{align}
```

Where:

  - ``d_{i,\\,j}``: Pairwise distance between assets ``i`` and ``j``.
  - ``\\rho_{i,\\,j}``: Pairwise correlation coefficient between assets ``i`` and ``j``.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct CorrelationDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Variation of Information (VI) distance algorithm for portfolio optimization.

`VariationInfoDistance` specifies the use of the Variation of Information (VI) metric, an information-theoretic distance based on entropy and mutual information.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VariationInfoDistance(;
        bins::Int_Bin = HacineGharbiRavier(),
        normalise::Bool = true
    ) -> VariationInfoDistance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:bins])

# Examples

```jldoctest
julia> VariationInfoDistance()
VariationInfoDistance
       bins ┼ HacineGharbiRavier()
  normalise ┴ Bool: true
```

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
@concrete struct VariationInfoDistance <: AbstractDistanceAlgorithm
    "$(field_dict[:bins])"
    bins
    "$(field_dict[:normalise])"
    normalise
    function VariationInfoDistance(bins::Int_Bin, normalise::Bool)
        if isa(bins, Integer)
            @argcheck(zero(bins) < bins, DomainError)
        end
        return new{typeof(bins), typeof(normalise)}(bins, normalise)
    end
end
function VariationInfoDistance(; bins::Int_Bin = HacineGharbiRavier(),
                               normalise::Bool = true)::VariationInfoDistance
    return VariationInfoDistance(bins, normalise)
end
"""
$(DocStringExtensions.TYPEDEF)

Canonical distance algorithm for portfolio optimization.

Defines the canonical distance metric for a given covariance estimator. The resulting distance metric is consistent with the properties of the covariance estimator (relevant when the covariance estimator is [`MutualInfoCovariance`](@ref)).

| Covariance Estimator                                                               | Distance Metric                 |
| ----------------------------------------------------------------------------------:|:------------------------------- |
| [`MutualInfoCovariance`](@ref)                                                     | [`VariationInfoDistance`](@ref) |
| [`LowerTailDependenceCovariance`](@ref)                                            | [`LogDistance`](@ref)           |
| [`DistanceCovariance`](@ref)                                                       | [`CorrelationDistance`](@ref)   |
| [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/) | [`SimpleDistance`](@ref)        |

The table also applies to [`PortfolioOptimisersCovariance`](@ref) where `ce` is one of the aforementioned estimators.

When used with a covariance matrix directly, uses [`SimpleDistance`](@ref).

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`MutualInfoCovariance`](@ref)
  - [`LowerTailDependenceCovariance`](@ref)
  - [`DistanceCovariance`](@ref)
  - [`VariationInfoDistance`](@ref)
  - [`LogDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`SimpleDistance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
struct CanonicalDistance <: AbstractDistanceAlgorithm end

export SimpleDistance, SimpleAbsoluteDistance, LogDistance, CorrelationDistance,
       VariationInfoDistance, CanonicalDistance
