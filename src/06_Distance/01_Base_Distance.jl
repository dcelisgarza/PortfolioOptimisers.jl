"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all distance estimator types.

A subtype turns a correlation matrix or a data matrix into a distance matrix. [`distance`](@ref) and [`cor_and_dist`](@ref) dispatch on it.

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

A subtype names the transform that a distance estimator applies. Four subtypes transform a correlation matrix, [`VariationInfoDistance`](@ref) reads the data matrix, and [`CanonicalDistance`](@ref) selects one of the others from the covariance estimator.

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

The factor ``1/2`` keeps the result in ``[0,\\,1]`` over the whole signed range of the correlation. So this is the algorithm for a codependence measure on ``[-1,\\,1]``, such as the Pearson, Spearman and Kendall correlations and the Gerber statistic.

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{\\mathrm{clamp}\\left(\\dfrac{1 - \\rho_{i,\\,j}}{2},\\, 0,\\, 1\\right)}\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_ij_dist])
  - $(math_dict[:rho_ij])

The clamp is only a numerical guard. ``(1 - \\rho_{i,\\,j}) / 2`` already lies in ``[0,\\,1]`` for every ``\\lvert\\rho_{i,\\,j}\\rvert \\leq 1``. So the clamp binds only when an estimator that shrinks, denoises or repairs the matrix moves a correlation slightly outside ``[-1,\\,1]``. In [`CorrelationDistance`](@ref), the same clamp binds on every negative correlation.

Lemma 3 of [vandongen2012](@cite) states this distance as ``\\sin(\\theta / 2)`` of the angle ``\\theta = \\arccos(\\rho)``. The lemma proves that it is a metric for a Pearson or a Spearman correlation and for a cosine. [lopezdeprado2016](@cite) also states it for the correlation of asset returns. That work gives a separate proof that it is a metric, and a worked example.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`SimpleAbsoluteDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)

# References

  - $(ref_dict[:vandongen2012]) Section 4, Lemma 3.
  - $(ref_dict[:lopezdeprado2016]) Online supplementary materials, Stage 1, Example 1, and Appendix A.1.
"""
struct SimpleDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Turns the magnitude of a correlation into a distance by ``\\sqrt{1 - \\lvert\\rho\\rvert}``.

The magnitude discards the sign. So two assets that move together and two assets that move in opposite directions with the same strength are equally close. This is the algorithm for a codependence measure that is already non-negative, and for the case where only the strength of the relationship matters.

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{\\mathrm{clamp}\\left(1 - \\lvert\\rho_{i,\\,j}\\rvert,\\, 0,\\, 1\\right)}\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_ij_dist])
  - $(math_dict[:rho_ij])

As in [`SimpleDistance`](@ref), the clamp is only a numerical guard. ``1 - \\lvert\\rho_{i,\\,j}\\rvert`` already lies in ``[0,\\,1]`` for every ``\\lvert\\rho_{i,\\,j}\\rvert \\leq 1``.

Appendix A.1 of the online supplementary materials of [lopezdeprado2016](@cite) states this distance and proves that it is a metric for a Pearson correlation.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`SimpleDistance`](@ref)
  - [`CorrelationDistance`](@ref)
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`_absguard`](@ref): supplies the magnitude. It decides only whether to allocate a copy, and it does not change the mathematics. On a non-negative codependence measure it returns the same matrix and allocates nothing.

# References

  - $(ref_dict[:lopezdeprado2016]) Online supplementary materials, Appendix A.1.
"""
struct SimpleAbsoluteDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Turns the magnitude of a correlation into an unbounded distance by ``-\\log\\lvert\\rho\\rvert``.

The range is ``[0,\\,\\infty)`` and not ``[0,\\,1]``, so the distance grows without limit as the relationship weakens. This is the algorithm for a tail dependence coefficient, and [`CanonicalDistance`](@ref) selects it for [`LowerTailDependenceCovariance`](@ref).

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\max\\left(-\\log{\\lvert\\rho_{i,\\,j}\\rvert},\\, 0\\right)\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_ij_dist])
  - $(math_dict[:rho_ij])

The floor at zero is necessary. A covariance estimator that shrinks, denoises or repairs a matrix can return ``\\lvert\\rho_{i,\\,j}\\rvert`` slightly above one, and ``-\\log`` of that value is negative. The square-root algorithms do not have this problem, because they clamp before the root. A negative distance inverts the order that the distance expresses, and a shortest-path routine that reads it gives a wrong result.

Two uncorrelated assets are infinitely far apart. ``\\rho_{i,\\,j} = 0`` gives ``d_{i,\\,j} = \\infty``, and the algorithm keeps that value.

Equation 2 of Section 3 of [luca2011](@cite) proposes ``-\\log \\hat{\\lambda}_L`` for an estimated lower tail dependence coefficient ``\\hat{\\lambda}_L``. That section calls it a dissimilarity and not a distance, because it does not satisfy the triangle inequality. A tail dependence coefficient lies in ``[0,\\,1]``, so the absolute value and the floor at zero are this library's guards.

# Related

  - [`AbstractDistanceAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`LowerTailDependenceCovariance`](@ref)
  - [`CanonicalDistance`](@ref)
  - [`assert_similarity_domain`](@ref): on the PMFG path, refuses an entry above `1` under [`ComplementSimilarity`](@ref) and an infinite entry under [`MaximumDistanceSimilarity`](@ref). This algorithm can give both.
  - [`distance`](@ref)
  - [`cor_and_dist`](@ref)
  - [`_absguard`](@ref): supplies the magnitude, as it does for [`SimpleAbsoluteDistance`](@ref). It decides only whether to allocate a copy, and it does not change the mathematics.

# References

  - $(ref_dict[:luca2011]) Section 3, Equation 2.
"""
struct LogDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Turns a non-negative codependence into a distance by ``\\sqrt{1 - \\rho}``, with no factor ``1/2``.

It is [`SimpleAbsoluteDistance`](@ref) without the absolute value. It is the algorithm for a codependence measure whose range is ``[0,\\,1]``, and [`CanonicalDistance`](@ref) selects it for [`DistanceCovariance`](@ref).

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= \\sqrt{\\mathrm{clamp}\\left(1 - \\rho_{i,\\,j},\\, 0,\\, 1\\right)}\\,,
\\end{align}
```

Where:

  - $(math_dict[:d_ij_dist])
  - $(math_dict[:rho_ij])

!!! warning "The clamp binds on a negative correlation"

    This is the one algorithm in the family whose clamp is not a numerical guard. Without the factor ``1/2``, ``1 - \\rho_{i,\\,j}`` runs over ``[0,\\,2]`` on a signed correlation, so the clamp truncates every negative entry to a distance of exactly `1`.

    The truncation removes the order of the negative entries. ``\\rho_{i,\\,j} = -0.1`` and ``\\rho_{i,\\,j} = -1`` both give ``1``, but ``\\sqrt{1 - \\rho_{i,\\,j}}`` gives ``\\sqrt{1.1}`` and ``\\sqrt{2}``. So the distance does not separate two negative correlations.

    The intended domain has no negative entry. Give a signed correlation to [`SimpleDistance`](@ref), which halves the radicand and so never saturates. Or give it to [`SimpleAbsoluteDistance`](@ref), which reads the magnitude.

On a non-negative ``\\rho_{i,\\,j}`` the formula is the distance ``\\sqrt{1 - \\lvert\\rho_{i,\\,j}\\rvert}`` of Appendix A.1 of the online supplementary materials of [lopezdeprado2016](@cite). That appendix proves the metric property for a Pearson correlation alone. Section 3.2 of [mlp1](@cite) proves it again for a correlation estimate with ``\\sigma[X, Y] = \\rho[X, Y] \\sigma[X] \\sigma[Y]``, and it does not name a distance correlation. Section 6.2.1 of [cajas2025](@cite) gives the formula for a distance correlation. A search found no earlier work that states that use.

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

  - $(ref_dict[:lopezdeprado2016]) Online supplementary materials, Appendix A.1.
  - $(ref_dict[:mlp1]) Section 3.2.
  - $(ref_dict[:cajas2025]) Section 6.2.1, Equation 6.23. $(ref_dict[:no_original_source])
"""
struct CorrelationDistance <: AbstractDistanceAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Measures the information one asset loses about another, from the entropies of a joint histogram.

It is the only algorithm in the family that reads the data matrix and not a correlation matrix. So it can measure a non-linear dependence that a correlation coefficient does not show. [`CanonicalDistance`](@ref) selects it for [`MutualInfoCovariance`](@ref).

# Mathematical definition

```math
\\begin{align}
    d_{i,\\,j} &= H(X_{i}) + H(X_{j}) - 2\\,I(X_{i};X_{j})\\,,
\\end{align}
```

When `normalise` is `true`, the algorithm divides the result by the joint entropy:

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

The first formula is the variation of information of Equation 19 of [meila2007](@cite), which proves that it is a metric. The second is ``d = 1 - I(X_{i};X_{j}) / H(X_{i},\\,X_{j})`` of Equation 1 of [rajski1961](@cite), which proves that it is a distance on discrete distributions. Theorem 1 of [kraskov2005](@cite) proves again that it is a metric on ``[0,\\,1]``. Their Theorem 2 gives a second metric on ``[0,\\,1]``, which divides by ``\\max(H(X_{i}),\\, H(X_{j}))`` instead. This algorithm does not compute that second metric.

# Algorithm

It is the only algorithm of the family that runs a procedure. The other algorithms apply a closed form to each correlation entry. [`variation_info`](@ref) does these steps for each pair of asset columns, and `bins` and `normalise` are the two fields below.

 1. Take the number of histogram bins from `bins`. An `Integer` is that count. [`HacineGharbiRavier`](@ref) computes the count from the sample size and the correlation of the pair. A bin-width rule divides the range of each column by its bin width, and takes the larger of the two counts.
 2. Fit a histogram to each of the two columns over that bin count, and take its Shannon entropy. The results are ``H(X_{i})`` and ``H(X_{j})``.
 3. Fit the joint histogram of the two columns over the same bin edges.
 4. Compute the mutual information ``I(X_{i};X_{j})`` from the joint histogram and its two marginal sums.
 5. Compute the variation of information of the first formula above.
 6. When `normalise` is `true`, divide by the joint entropy ``H(X_{i}) + H(X_{j}) - I(X_{i};X_{j})`` to get the second formula.
 7. Floor the result at zero. The diagonal is exactly zero, and the procedure does not estimate it.

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

  - $(ref_dict[:meila2007]) Section 3, Equation 19.
  - $(ref_dict[:rajski1961]) Theorem, Equation 1.
  - $(ref_dict[:kraskov2005]) Section 2.3, Theorems 1 and 2.
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

It has no formula. It is a redirect that sends each codependence measure to the distance transform that fits its range. A signed correlation needs the factor ``1/2``. A mutual information has no correlation to transform. A tail dependence coefficient needs an unbounded distance.

| Covariance estimator                                                               | Distance algorithm              |
| ----------------------------------------------------------------------------------:|:------------------------------- |
| [`MutualInfoCovariance`](@ref)                                                     | [`VariationInfoDistance`](@ref) |
| [`LowerTailDependenceCovariance`](@ref)                                            | [`LogDistance`](@ref)           |
| [`DistanceCovariance`](@ref)                                                       | [`CorrelationDistance`](@ref)   |
| [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/) | [`SimpleDistance`](@ref)        |

The table also applies to a [`PortfolioOptimisersCovariance`](@ref) whose `ce` is one of these estimators.

With a correlation or a covariance matrix and no estimator, it uses [`SimpleDistance`](@ref).

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
