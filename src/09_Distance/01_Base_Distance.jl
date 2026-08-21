"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all distance estimator types.

All concrete and/or abstract types implementing distance-based estimation algorithms should be subtypes of `AbstractDistanceEstimator`.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractDistanceAlgorithm`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
"""
abstract type AbstractDistanceEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all distance algorithm types.

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

Turns a signed correlation into a distance by ``\\sqrt{(1 - \\rho) / 2}``.

The halving keeps the result in ``[0,\\,1]`` over the whole signed range of the correlation, which is what makes this the algorithm for a codependence measure on ``[-1,\\,1]``: Pearson, Spearman, Kendall and the Gerber statistic.

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{\\mathrm{clamp}\\left(\\dfrac{1 - \\rho_{i,\\,j}}{2},\\, 0,\\, 1\\right)}\\,,
\\end{align}
```

Where:

  - ``d_{i,\\,j}``: Pairwise distance between assets ``i`` and ``j``.
  - ``\\rho_{i,\\,j}``: Pairwise correlation coefficient between assets ``i`` and ``j``.

The clamp is a numerical guard and nothing more. ``(1 - \\rho_{i,\\,j}) / 2`` already lies in ``[0,\\,1]`` for every ``\\lvert\\rho_{i,\\,j}\\rvert \\leq 1``, so it binds only on a correlation that a shrinking, denoising or repairing estimator pushed a hair outside ``[-1,\\,1]``. Contrast [`CorrelationDistance`](@ref), where the same clamp binds on ordinary data.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 6.2.1, Equation 6.22.
  - $(ref_dict[:mlp1]) Chapter 3.
"""
struct SimpleDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Turns the magnitude of a correlation into a distance by ``\\sqrt{1 - \\lvert\\rho\\rvert}``.

Reading the magnitude discards the sign, so two assets that move together and two that move oppositely are equally close. This is the algorithm for a codependence measure that is already non-negative, and for the case where only the strength of the relationship matters.

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{\\mathrm{clamp}\\left(1 - \\lvert\\rho_{i,\\,j}\\rvert,\\, 0,\\, 1\\right)}\\,,
\\end{align}
```

Where:

  - ``d_{i,\\,j}``: Pairwise distance between assets ``i`` and ``j``.
  - ``\\rho_{i,\\,j}``: Pairwise correlation coefficient between assets ``i`` and ``j``.

The clamp is a numerical guard and nothing more, as in [`SimpleDistance`](@ref): ``1 - \\lvert\\rho_{i,\\,j}\\rvert`` already lies in ``[0,\\,1]`` for every ``\\lvert\\rho_{i,\\,j}\\rvert \\leq 1``.

The absolute value is taken over the **whole** matrix or not at all. A matrix with no negative entry is passed through untouched, which is why a non-negative codependence measure reaches this algorithm unchanged.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`SimpleDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 6.2.1, Equation 6.23.
  - $(ref_dict[:mlp1]) Chapter 3.
"""
struct SimpleAbsoluteDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Turns the magnitude of a correlation into an unbounded distance by ``-\\log\\lvert\\rho\\rvert``.

The range is ``[0,\\,\\infty)`` rather than ``[0,\\,1]``, so the distance grows without limit as the relationship weakens. This is the algorithm for a tail dependence coefficient, and the one [`CanonicalDistance`](@ref) selects for [`LowerTailDependenceCovariance`](@ref).

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\max\\left(-\\log{\\lvert\\rho_{i,\\,j}\\rvert},\\, 0\\right)\\,,
\\end{align}
```

Where:

  - ``d_{i,\\,j}``: Pairwise distance between assets ``i`` and ``j``.
  - ``\\rho_{i,\\,j}``: Pairwise correlation coefficient between assets ``i`` and ``j``.

The floor at zero is not cosmetic. A covariance estimator that shrinks, denoises or repairs a matrix can return ``\\lvert\\rho_{i,\\,j}\\rvert`` a hair above one, and ``-\\log`` of that is *negative* — unlike the square-root algorithms, which already clamp before taking the root. A negative distance inverts the ordering it is meant to express and is unsound under the shortest-path routines that consume it.

Perfectly uncorrelated assets remain infinitely far apart: ``\\rho_{i,\\,j} = 0`` gives ``d_{i,\\,j} = \\infty``, which is a meaningful value here and is left alone. It is also the entry that the two bounded similarity members cannot take, so [`assert_similarity_domain`](@ref) refuses this algorithm under [`MaximumDistanceSimilarity`](@ref) on the PMFG path.

The absolute value is taken over the **whole** matrix or not at all, as in [`SimpleAbsoluteDistance`](@ref).

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`LowerTailDependenceCovariance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`assert_similarity_domain`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 6.2.3, Equation 6.26.
  - $(ref_dict[:luca2011])
"""
struct LogDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Turns a non-negative codependence into a distance by ``\\sqrt{1 - \\rho}``, without halving.

This is [`SimpleAbsoluteDistance`](@ref) with the absolute value dropped, and it is the algorithm for a codependence measure whose own range is ``[0,\\,1]``. It is what [`CanonicalDistance`](@ref) selects for [`DistanceCovariance`](@ref).

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{\\mathrm{clamp}\\left(1 - \\rho_{i,\\,j},\\, 0,\\, 1\\right)}\\,,
\\end{align}
```

Where:

  - ``d_{i,\\,j}``: Pairwise distance between assets ``i`` and ``j``.
  - ``\\rho_{i,\\,j}``: Pairwise correlation coefficient between assets ``i`` and ``j``.

!!! warning "The clamp binds on a negative correlation"

    This is the one algorithm in the family whose clamp is not a numerical guard. Without the halving, ``1 - \\rho_{i,\\,j}`` runs over ``[0,\\,2]`` on a signed correlation, so the clamp truncates every negative entry to a distance of exactly `1`.

    At ``\\rho_{i,\\,j} = -0.9311319132604445`` the algorithm returns `1.0` where ``\\sqrt{1 - \\rho_{i,\\,j}}`` is `1.3896517237280874`. The truncation is not monotone: ``\\rho_{i,\\,j} = -0.1`` and ``\\rho_{i,\\,j} = -1`` are both reported as `1`, so the ordering the distance is meant to express is lost across the whole negative half.

    The intended domain has no negative entry. Give a signed correlation to [`SimpleDistance`](@ref), which halves and therefore never saturates, or to [`SimpleAbsoluteDistance`](@ref), which reads the magnitude.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`SimpleDistance`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`DistanceCovariance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 6.2.1, Equation 6.23.
"""
struct CorrelationDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Measures the information one asset loses about another, from the entropies of a joint histogram.

This is the only algorithm in the family that reads the data matrix rather than a correlation matrix, so it captures a non-linear relationship that no correlation coefficient sees. It is what [`CanonicalDistance`](@ref) selects for [`MutualInfoCovariance`](@ref).

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= H(X_{i}) + H(X_{j}) - 2\\,I(X_{i};X_{j})\\,,
\\end{align}
```

When `normalise` is `true`, the result is divided by the joint entropy:

```math
\\begin{align}
    \\tilde{d}_{i,\\,j} &= \\dfrac{H(X_{i}) + H(X_{j}) - 2\\,I(X_{i};X_{j})}{H(X_{i}) + H(X_{j}) - I(X_{i};X_{j})}\\,.
\\end{align}
```

Where:

  - ``d_{i,\\,j}``: Pairwise variation of information between assets ``i`` and ``j``.
  - ``\\tilde{d}_{i,\\,j}``: Normalised pairwise variation of information.
  - ``H(X_{i})``: Marginal Shannon entropy of asset ``i``, estimated from a histogram whose bin count comes from `bins`.
  - ``I(X_{i};X_{j})``: Mutual information between assets ``i`` and ``j``.

Equation 6.25 of the source normalises by ``\\max(H(X_{i}),\\, H(X_{j}))`` instead. This algorithm divides by the joint entropy, which keeps the result a metric on ``[0,\\,1]``. See [`variation_info`](@ref), which computes it.

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
  - [`MutualInfoCovariance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`variation_info`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 6.2.2, Equations 6.24 and 6.25.
  - $(ref_dict[:mlp1]) Chapter 3.
"""
@concrete struct VariationInfoDistance <: AbstractDistanceAlgorithm
    """
    $(field_dict[:bins])
    """
    bins
    """
    $(field_dict[:normalise])
    """
    normalise
    function VariationInfoDistance(bins::Int_Bin, normalise::Bool)
        if isa(bins, Integer)
            @argcheck(zero(bins) < bins, DomainError(bins, "bins must be positive"))
            assert_resource_cap(bins, RESOURCE_LIMITS[].max_bins, :bins, :max_bins)
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

Selects the distance algorithm that matches the covariance estimator it is given.

It owns no formula of its own. It is a redirect, and it exists so that a codependence measure reaches the distance transform its own range calls for: a signed correlation must be halved, a mutual information has no correlation to transform at all, and a tail dependence coefficient wants an unbounded distance.

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
