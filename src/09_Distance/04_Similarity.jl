"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all similarity matrix algorithms.

Every subtype defines a pure transformation of a distance matrix into a similarity matrix, applied by [`distance_to_similarity`](@ref). The family is consumed in two places: the Planar Maximally Filtered Graph (PMFG) construction, which the similarity branch of [`NetworkEstimator`](@ref) and both [`DBHT`](@ref) and [`LoGo`](@ref) all reach, and the similarity slot returned by [`cor_and_dist`](@ref).

The family is **open**: an extension defines a member and a [`distance_to_similarity`](@ref) method for it.

The first consumer needs a similarity that cannot go below zero, so **all three** of the fields feeding it — [`NetworkEstimator`](@ref)'s `alg`, [`DBHT`](@ref)'s `sim` and [`LoGo`](@ref)'s `sim` — take the narrower [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref) rather than this type. The second takes any member: [`FeatureDistance`](@ref)'s `sim` keeps all five, and never reaches a PMFG, because every PMFG entry point recomputes the similarity from its own field and discards the one [`cor_and_dist`](@ref) returned.

# Related

  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`default_similarity`](@ref)
  - [`DBHT`](@ref)
  - [`LoGo`](@ref)
"""
abstract type AbstractSimilarityMatrixAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the similarity matrix algorithms admitted on the PMFG path.

A member of this family promises that [`distance_to_similarity`](@ref) returns **no negative entry for any finite distance matrix in the member's own domain**. [`Tree_SimMat`](@ref), [`DBHT`](@ref)'s `sim` and [`LoGo`](@ref)'s `sim` are bounded by this type, so a similarity that can go negative fails at **construction** rather than inside [`PMFG_T2s`](@ref).

# Where the requirement comes from

Not from [`PMFG_T2s`](@ref), which runs on signed input: its gain argmax compares sums of exactly three weights, and its seed mask is shift-invariant. The requirement comes from **downstream DBHT**. `DirectHb` compares unnormalised weight sums over different-sized bubbles and writes the winner into `Hc`, where a separating bubble is detected by an exact zero row sum — so cancelling signs manufacture one. `BubbleMember` divides by a signed total that can reach zero, `Inf` or a negative, in which case `argmax` picks the worst bubble. Both failures are **silent**: the caller gets wrong clusters and no error.

The type bound is deliberately **wider** than that provenance. [`calc_adjacency`](@ref) binarises the PMFG, [`clusterise`](@ref) takes matrix powers of it, and `logo!` reads structure only, so none of them reaches the bubble machinery. They are bound anyway, because relaxing the one guard that presently makes them safe is a change with its own justification to make.

# Non-negative, not positive

`exp(-Inf)` is `0` exactly, and that route is live — [`LogDistance`](@ref) maps an exactly zero correlation to an infinite distance. A zero similarity is admissible; a negative one is not.

# The guarantee is per member, and it carries a domain

The type says *which algorithm*. It cannot say *which data*, so a member whose guarantee needs a precondition on `D` declares it through [`assert_similarity_domain`](@ref), which the PMFG entry points call before they transform.

| Member                                 | Holds when     |
|:-------------------------------------- |:-------------- |
| [`ExponentialSimilarity`](@ref)        | always         |
| [`GeneralExponentialSimilarity`](@ref) | always         |
| [`MaximumDistanceSimilarity`](@ref)    | `D` is finite  |
| [`ComplementSimilarity`](@ref)         | `all(D .<= 1)` |

[`AngularSimilarity`](@ref) is excluded permanently, and not for want of a precondition: it is negative wherever the correlation is negative, which is ordinary data.

# Open by declaration, not by proof

The family is open, and membership is a **claim a subtype makes** rather than one the library verifies. A probe cannot check it: the contract quantifies over every admissible distance matrix, so a probe that passes [`ComplementSimilarity`](@ref) at `D = 0.5` still misses its failure at `D = 7`. [`PMFG_T2s`](@ref)'s own non-negativity check is therefore kept as the backstop against an extension that claims membership and does not keep it.

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`assert_similarity_domain`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`Tree_SimMat`](@ref)
  - [`PMFG_T2s`](@ref)
  - [`DBHT`](@ref)
  - [`LoGo`](@ref)
"""
abstract type AbstractNonNegativeSimilarityMatrixAlgorithm <:
              AbstractSimilarityMatrixAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Subtracts the squared distance from a ceiling placed above the largest squared distance.

The ceiling is what makes the result non-negative for a distance matrix of any scale. The source states the transformation as ``\\rho_{i,\\,j} = 1 - d_{i,\\,j}^{2}``, which this reproduces exactly whenever ``\\max(\\mathbf{D}) \\leq 1`` — the case for [`SimpleDistance`](@ref), [`SimpleAbsoluteDistance`](@ref) and [`CorrelationDistance`](@ref). Above that the source's form goes negative and this one does not: on a [`LogDistance`](@ref) matrix whose largest entry was `6.759438300750648`, ``1 - d_{i,\\,j}^{2}`` reached `-44.690006141654806` while this transformation stayed at `0.30999385834519444`.

# Mathematical definition

```math
\\begin{align}
S_{i,\\,j} &= \\left\\lceil\\max(\\mathbf{D})^2\\right\\rceil - D_{i,\\,j}^2\\,,
\\end{align}
```

Where:

  - ``S_{i,\\,j}``: Similarity between assets ``i`` and ``j``.
  - ``\\mathbf{D}``: Distance matrix.
  - ``D_{i,\\,j}``: Distance between assets ``i`` and ``j``.

!!! warning

    The transformation is defined only for a **finite** distance matrix. An infinite entry makes `ceil(Inf^2) - Inf^2`, which is `NaN`, and every other entry `Inf`. This is not a corner case: [`LogDistance`](@ref) maps an exactly zero correlation to an infinite distance, and this member is the default of both [`DBHT`](@ref) and [`LoGo`](@ref). [`assert_similarity_domain`](@ref) refuses it on the PMFG path.

# Related

  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`assert_similarity_domain`](@ref)
  - [`distance_to_similarity`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 13.1.
"""
struct MaximumDistanceSimilarity <: AbstractNonNegativeSimilarityMatrixAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Maps a distance of any magnitude into ``(0,\\,1]`` by ``e^{-D}``.

The one member of the family with no domain at all: it is non-negative for every finite distance, and it takes an infinite one too, where `exp(-Inf)` is `0` exactly. This is the member to reach for when the distance is unbounded, as under [`LogDistance`](@ref), [`DistanceDistance`](@ref) or [`VariationInfoDistance`](@ref) with `normalise = false`.

# Mathematical definition

```math
\\begin{align}
S_{i,\\,j} &= e^{-D_{i,\\,j}}\\,,
\\end{align}
```

Where:

  - ``S_{i,\\,j}``: Similarity between assets ``i`` and ``j``.
  - ``\\mathbf{D}``: Distance matrix.
  - ``D_{i,\\,j}``: Distance between assets ``i`` and ``j``.

# Related

  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
"""
struct ExponentialSimilarity <: AbstractNonNegativeSimilarityMatrixAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Applies ``e^{-c D^{p}}``, adding a scale and an exponent to the exponential transformation.

`coef` sets how fast the similarity decays with distance and `power` sets the shape of that decay, so a single distance matrix can be sharpened towards its nearest neighbours or flattened across the universe. It inherits [`ExponentialSimilarity`](@ref)'s freedom from a domain, and reduces to it at `coef = 1`, `power = 1`.

# Mathematical definition

```math
\\begin{align}
S_{i,\\,j} &= e^{-c \\cdot D_{i,\\,j}^p}\\,,
\\end{align}
```

Where:

  - ``S_{i,\\,j}``: Similarity between assets ``i`` and ``j``.
  - ``\\mathbf{D}``: Distance matrix.
  - ``D_{i,\\,j}``: Distance between assets ``i`` and ``j``.
  - ``c``: Scale factor.
  - ``p``: Exponent.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GeneralExponentialSimilarity(;
        coef::Number = 1,
        power::Number = 1
    ) -> GeneralExponentialSimilarity

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:dbhtcoef])
  - $(val_dict[:dbhtpower])

# Examples

```jldoctest
julia> GeneralExponentialSimilarity()
GeneralExponentialSimilarity
   coef ┼ Int64: 1
  power ┴ Int64: 1
```

# Related

  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
"""
@concrete struct GeneralExponentialSimilarity <:
                 AbstractNonNegativeSimilarityMatrixAlgorithm
    """
    $(field_dict[:dbhtcoef])
    """
    coef
    """
    $(field_dict[:dbhtpower])
    """
    power
    function GeneralExponentialSimilarity(coef::Number, power::Number)
        assert_nonempty_gt0_finite_val(coef, :coef)
        @argcheck(zero(power) < power, DomainError)
        return new{typeof(coef), typeof(power)}(coef, power)
    end
end
function GeneralExponentialSimilarity(; coef::Number = 1,
                                      power::Number = 1)::GeneralExponentialSimilarity
    return GeneralExponentialSimilarity(coef, power)
end
"""
$(DocStringExtensions.TYPEDEF)

Takes the linear complement ``1 - D``, the exact counterpart of a metric that is itself one minus a similarity.

It is the only member that recovers a *named* similarity rather than a monotone stand-in for one, which is why [`default_similarity`](@ref) uses it as the fallback. It is also the only member with both a domain and a pairing to get right, and both warnings below are about that.

# Mathematical definition

```math
\\begin{align}
S_{i,\\,j} &= 1 - D_{i,\\,j}\\,,
\\end{align}
```

Where:

  - ``S_{i,\\,j}``: Similarity between assets ``i`` and ``j``.
  - ``\\mathbf{D}``: Distance matrix.
  - ``D_{i,\\,j}``: Distance between assets ``i`` and ``j``.

This recovers the named similarity counterpart of every distance that is itself one minus a similarity. `Distances.CosineDist` returns the cosine similarity, `Distances.Jaccard` the Ruzicka similarity, `Distances.BrayCurtis` the Sørensen–Dice similarity, and `Distances.CorrDist` the Pearson correlation.

!!! warning "The domain is `D <= 1`"

    The result is only correlation-like when ``\\mathbf{D} \\in [0,\\,1]``. Above `1` the similarity is negative: ``D_{i,\\,j} = 7`` gives ``S_{i,\\,j} = -6``.

    What happens next depends on the path. On the [`FeatureDistance`](@ref) path the value is **kept**, lands outside the ``[-1,\\,1]`` range that [`plot_clusters`](@ref) assumes, and is silently clipped there rather than flagged. On the PMFG path the same input is **refused** by [`assert_similarity_domain`](@ref), because [`PMFG_T2s`](@ref)'s consumers cannot take a negative weight. Symmetry and the unit diagonal survive either way.

    The rule is `D <= 1`, **not** "the metric is unbounded". `Distances.CosineDist` and `Distances.CorrDist` are bounded — by `2`, not by `1` — and are refused whenever they exceed `1`, which `CorrDist` does at every negative correlation. In-library sources that exceed `1`: [`LogDistance`](@ref), [`DistanceDistance`](@ref) — whose `Distances.Euclidean` **default** puts most of its entries above `1` — and [`VariationInfoDistance`](@ref) with `normalise = false`. Use [`ExponentialSimilarity`](@ref) or [`GeneralExponentialSimilarity`](@ref) for a member with no domain at all.

!!! warning "The pairing is not checked"

    This member is the honest inverse of a **specific** set of metrics, listed above. Paired with any other distance it returns a number that is in domain, non-negative, and wrong — and nothing catches it, on any path.

    [`SimpleDistance`](@ref) is ``\\sqrt{(1 - \\rho) / 2}``, so a correlation of `0.003` gives `D = 0.706` and this member reports a similarity of `0.29`. No check placed anywhere can detect that, because `0.706` is a perfectly legal bounded distance. [`default_similarity`](@ref) pairs a metric with its inverse on the [`FeatureDistance`](@ref) path; [`NetworkEstimator`](@ref)'s `alg`, [`DBHT`](@ref)'s `sim` and [`LoGo`](@ref)'s `sim` take a member with no reference to the distance estimator that produced ``\\mathbf{D}``, so on those the pairing is the caller's to get right.

# Related

  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`assert_similarity_domain`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`default_similarity`](@ref)
"""
struct ComplementSimilarity <: AbstractNonNegativeSimilarityMatrixAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Recovers a correlation from a normalised angular distance by ``\\cos(\\pi D)``.

It is the only member that returns the codependence itself rather than a monotone transformation of it, and the only one that can go negative. Both facts follow from the same thing: it is the exact algebraic inverse of [`AngularDist`](@ref), and a correlation is signed.

# Mathematical definition

```math
\\begin{align}
S_{i,\\,j} &= \\cos\\left(\\pi D_{i,\\,j}\\right)\\,,
\\end{align}
```

Where:

  - ``S_{i,\\,j}``: Similarity between assets ``i`` and ``j``.
  - ``\\mathbf{D}``: Distance matrix.
  - ``D_{i,\\,j}``: Distance between assets ``i`` and ``j``.

For an angular distance ``D_{i,\\,j} = \\arccos(\\rho_{i,\\,j}) / \\pi`` this recovers ``\\rho_{i,\\,j}`` exactly, without reference to the data the distance was computed from. Against [`AngularDist`](@ref) on an 8-asset feature matrix the recovered cosine matched the one computed from the features to `3.608224830031759e-16`. It maps ``[0,\\,1] \\to [1,\\,-1]``, so the similarity is bounded and the diagonal is unity whenever the distance matrix has a zero diagonal.

# Where this member is correct, and where it is refused

It is correct on the [`FeatureDistance`](@ref) path, which is the one path that pairs a metric with its inverse: [`default_similarity`](@ref) selects this member for [`AngularDist`](@ref), and the recovered ``\\rho`` is exact.

It is **not** a member of [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref), so it cannot be given to [`NetworkEstimator`](@ref)'s `alg`, [`DBHT`](@ref)'s `sim` or [`LoGo`](@ref)'s `sim` — those refuse it at construction. The exclusion is permanent rather than pending a domain precondition. Even paired correctly this member returns a negative wherever ``\\rho_{i,\\,j} < 0``, which is ordinary data, and [`PMFG_T2s`](@ref)'s consumers cannot take a negative weight.

!!! warning "The pairing is not checked"

    Paired with a distance that is not an angular one, this member returns a number that is not a correlation. [`SimpleDistance`](@ref) is ``\\sqrt{(1 - \\rho) / 2}`` and shares this member's ``[0,\\,1]`` range exactly, so it type-checks: a correlation of `0.003` gives `D = 0.706` and ``\\cos(\\pi D)`` reports `-0.618`, which is not a weak negative correlation but nonsense.

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AngularDist`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`default_similarity`](@ref)

# References

  - $(ref_dict[:vandongen2012])
"""
struct AngularSimilarity <: AbstractSimilarityMatrixAlgorithm end
"""
    distance_to_similarity(se::AbstractSimilarityMatrixAlgorithm; D::MatNum, kwargs...)

Compute a similarity matrix from a distance matrix using the specified similarity algorithm.

This function dispatches on the type of `se` to apply the appropriate similarity transformation to the distance matrix `D`.

# Arguments

  - `se`: Similarity matrix algorithm.

      + `se::MaximumDistanceSimilarity`: Uses the maximum distance transformation.
      + `se::ExponentialSimilarity`: Uses the exponential transformation.
      + `se::GeneralExponentialSimilarity`: Uses a generalised exponential transformation.
      + `se::ComplementSimilarity`: Uses the linear complement of the distance.
      + `se::AngularSimilarity`: Inverts a normalised angular distance.

  - `D`: Distance matrix.

  - `kwargs...`: Additional keyword arguments (not used).

# Returns

  - `S::Matrix{<:Number}`: Similarity matrix of the same size as `D`.

# Examples

```jldoctest
julia> D = [0.0 0.25 0.5; 0.25 0.0 1.0; 0.5 1.0 0.0];

julia> PortfolioOptimisers.distance_to_similarity(ComplementSimilarity(); D = D)
3×3 Matrix{Float64}:
 1.0   0.75  0.5
 0.75  1.0   0.0
 0.5   0.0   1.0

julia> D = [0.0 1/3 2/3; 1/3 0.0 1.0; 2/3 1.0 0.0];

julia> PortfolioOptimisers.distance_to_similarity(AngularSimilarity(); D = D)
3×3 Matrix{Float64}:
  1.0   0.5  -0.5
  0.5   1.0  -1.0
 -0.5  -1.0   1.0
```

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`default_similarity`](@ref)
"""
function distance_to_similarity(::MaximumDistanceSimilarity; D::MatNum, kwargs...)
    return ceil(maximum(D)^2) .- D .^ 2
end
function distance_to_similarity(::ExponentialSimilarity; D::MatNum, kwargs...)
    return exp.(-D)
end
function distance_to_similarity(se::GeneralExponentialSimilarity; D::MatNum, kwargs...)
    power = se.power
    coef = se.coef
    return exp.(-coef * D .^ power)
end
function distance_to_similarity(::ComplementSimilarity; D::MatNum, kwargs...)
    return one(eltype(D)) .- D
end
function distance_to_similarity(::AngularSimilarity; D::MatNum, kwargs...)
    return cos.(pi .* D)
end
"""
    assert_similarity_domain(sim::AbstractSimilarityMatrixAlgorithm,
                             de::AbstractDistanceEstimator, D::MatNum)
    assert_similarity_domain(sim::ComplementSimilarity, de::AbstractDistanceEstimator,
                             D::MatNum)
    assert_similarity_domain(sim::MaximumDistanceSimilarity,
                             de::AbstractDistanceEstimator, D::MatNum)

Assert that `D` lies in `sim`'s domain, so that `sim`'s non-negativity guarantee holds on it.

The other half of [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)'s enforcement. The type bound says *which algorithm* may reach the PMFG path; it cannot say *which data*, and two of the four admitted members are non-negative only over part of the distances a caller can produce.

| Member                              | Precondition                   |
|:----------------------------------- |:------------------------------ |
| [`ComplementSimilarity`](@ref)      | `all(D .<= 1)`                 |
| [`MaximumDistanceSimilarity`](@ref) | `all(isfinite, D)`             |
| every other member                  | none — the fallback is a no-op |

# Interface-scoped, not member-wide

The five call sites are the PMFG entry points, and nothing else: [`calc_weighted_adjacency_graph`](@ref), [`calc_distance_weighted_graph`](@ref), both [`clusterise`](@ref) methods that build a PMFG, and `logo!`. It is deliberately **not** called inside [`distance_to_similarity`](@ref), which stays a pure transformation with no domain of its own.

The scope follows the failure, which is path-dependent rather than member-wide. [`ComplementSimilarity`](@ref) against an unbounded distance is documented, tested behaviour on the [`FeatureDistance`](@ref) path — every `Distances.SemiMetric` yields a similarity there and nothing throws — and it is only on the PMFG path that the same value is unusable.

# What it costs a caller

Nothing that works today. Every pairing this refuses already throws, at [`PMFG_T2s`](@ref)'s own check and one transformation later, with a message that names `W` rather than the configuration that produced it. The gain is the message: it names **both halves**, which is why `de` is passed in — the distance estimator that produced the offending value, and the similarity that refused it.

# Arguments

  - `sim`: Similarity matrix algorithm.
  - `de`: Distance estimator that produced `D`. Read only to name it in the error.
  - `D`: Distance matrix.

# Validation

  - Under [`ComplementSimilarity`](@ref): `all(x -> x <= 1, D)`.
  - Under [`MaximumDistanceSimilarity`](@ref): `all(isfinite, D)`.

# Returns

  - `nothing`.

# Related

  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`PMFG_T2s`](@ref)
  - [`assert_metric_domain`](@ref)
"""
function assert_similarity_domain(::AbstractSimilarityMatrixAlgorithm,
                                  ::AbstractDistanceEstimator, ::MatNum)::Nothing
    return nothing
end
function assert_similarity_domain(sim::ComplementSimilarity, de::AbstractDistanceEstimator,
                                  D::MatNum)::Nothing
    @argcheck(all(<=(one(eltype(D))), D),
              DomainError(maximum(D),
                          "all(x -> x <= 1, D) must hold for $(nameof(typeof(sim))), whose similarity is 1 - D. Got\nmaximum(D) => $(maximum(D)), from $(typeof(de))."))
    return nothing
end
function assert_similarity_domain(sim::MaximumDistanceSimilarity,
                                  de::AbstractDistanceEstimator, D::MatNum)::Nothing
    @argcheck(all(isfinite, D),
              DomainError(maximum(D),
                          "all(isfinite, D) must hold for $(nameof(typeof(sim))), whose similarity is ceil(maximum(D)^2) - D^2 and is NaN at a non-finite entry. Got\nmaximum(D) => $(maximum(D)), from $(typeof(de))."))
    return nothing
end
"""
    default_similarity(metric::Distances.SemiMetric)

Select the similarity matrix algorithm that is the natural counterpart of a distance metric.

Used to default the similarity field of a distance algorithm from its metric, so that the resolved value is visible on the printed object rather than hidden inside the distance kernel. The fallback is [`ComplementSimilarity`](@ref), which is the named counterpart of every metric expressible as one minus a similarity; metrics whose inversion is not linear add their own method.

# Arguments

  - `metric`: Distance metric.

# Returns

  - `sim::AbstractSimilarityMatrixAlgorithm`: Similarity matrix algorithm.

# Examples

```jldoctest
julia> PortfolioOptimisers.default_similarity(PortfolioOptimisers.Distances.CosineDist())
ComplementSimilarity()
```

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
"""
function default_similarity(::Distances.SemiMetric)::ComplementSimilarity
    return ComplementSimilarity()
end

export MaximumDistanceSimilarity, ExponentialSimilarity, GeneralExponentialSimilarity,
       ComplementSimilarity, AngularSimilarity
