"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all similarity matrix algorithms.

Every subtype defines a pure transformation of a distance matrix into a similarity matrix, and [`distance_to_similarity`](@ref) applies it. Two places read the family. The first is the construction of the Planar Maximally Filtered Graph (PMFG), which the similarity branch of [`NetworkEstimator`](@ref), [`DBHT`](@ref) and [`LoGo`](@ref) reach. The second is the similarity that [`cor_and_dist`](@ref) returns.

The family is open. An extension defines a member and a [`distance_to_similarity`](@ref) method for it.

The PMFG path needs a similarity that is not negative, so the three fields that give it a similarity take the narrower [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref), not this type. These fields are `alg` of [`NetworkEstimator`](@ref), `sim` of [`DBHT`](@ref) and `sim` of [`LoGo`](@ref). The field `sim` of [`FeatureDistance`](@ref) takes all five members. Its similarity never reaches a PMFG, because each PMFG entry point computes the similarity again from its own field and discards the similarity that [`cor_and_dist`](@ref) returned.

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

Abstract supertype for the similarity matrix algorithms that the PMFG path admits.

A member of this family promises that [`distance_to_similarity`](@ref) returns no negative entry for a finite distance matrix in the domain of the member. This type bounds [`Tree_SimMat`](@ref), `sim` of [`DBHT`](@ref) and `sim` of [`LoGo`](@ref), so a similarity that can go negative fails at construction, not inside [`PMFG_T2s`](@ref).

# Where the requirement comes from

The requirement does not come from [`PMFG_T2s`](@ref), which runs on a signed input. Its argmax of the gain compares sums of exactly three weights, and a constant shift of the weights does not change its seed mask.

The requirement comes from DBHT, which runs after the PMFG. `DirectHb` compares the weight sums of bubbles of different sizes without a normalisation, and writes the winner into `Hc`. In `Hc`, a row sum of exactly zero marks a separating bubble, so weights of opposite sign can cancel and make a false one. `BubbleMember` divides by a signed total that can be zero, `Inf` or negative, and then `argmax` selects the worst bubble. Neither failure raises an error, so the caller gets wrong clusters without a message.

The type bound covers more consumers than DBHT. [`calc_adjacency`](@ref) makes the PMFG binary, [`clusterise`](@ref) takes matrix powers of it, and `logo!` reads only its structure. None of them reaches the bubble code of DBHT. The bound holds on them too, because it is the one guard that makes them safe now, and its removal needs a reason of its own.

# Non-negative, not positive

`exp(-Inf)` is exactly `0`, and real data reaches it, because [`LogDistance`](@ref) maps a correlation of exactly zero to an infinite distance. The type admits a similarity of zero. It does not admit a negative similarity.

# The guarantee is per member, and it carries a domain

The type selects the algorithm. It cannot select the data. A member whose guarantee needs a precondition on `D` states the precondition in a method of [`assert_similarity_domain`](@ref). The PMFG entry points call that function before the transformation.

| Member                                 | Holds when     |
|:-------------------------------------- |:-------------- |
| [`ExponentialSimilarity`](@ref)        | always         |
| [`GeneralExponentialSimilarity`](@ref) | always         |
| [`MaximumDistanceSimilarity`](@ref)    | `D` is finite  |
| [`ComplementSimilarity`](@ref)         | `all(D .<= 1)` |

The family never admits [`AngularSimilarity`](@ref), and no precondition can change that. It is negative at each negative correlation, and a negative correlation is ordinary data.

# Open by declaration, not by proof

The family is open. A subtype claims its membership, and the library does not check the claim. A probe cannot check it, because the contract holds over every admissible distance matrix. A probe that passes [`ComplementSimilarity`](@ref) at `D = 0.5` does not find its failure at `D = 7`. [`PMFG_T2s`](@ref) keeps its own check of non-negativity, to stop an extension that claims membership and breaks the promise.

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

Subtracts each squared distance from the largest squared distance, rounded up to an integer.

The ceiling makes the result non-negative for a distance matrix of any scale. The source states the transformation as ``\\rho_{i,\\,j} = 1 - d_{i,\\,j}^{2}``. This member gives the same result when ``\\max(\\mathbf{D}) \\leq 1``, which holds for [`SimpleDistance`](@ref), [`SimpleAbsoluteDistance`](@ref) and [`CorrelationDistance`](@ref). Above that bound the form of the source goes negative, and this member does not.

# Mathematical definition

```math
\\begin{align}
S_{i,\\,j} &= \\left\\lceil\\max(\\mathbf{D})^2\\right\\rceil - D_{i,\\,j}^2\\,,
\\end{align}
```

Where:

  - $(math_dict[:S_ij_sim])
  - $(math_dict[:D_mat_dist])
  - $(math_dict[:D_ij_dist])

Equation 4 of Section 3 of [gower1966](@cite) relates a similarity matrix with a unit diagonal to the distance of its points by ``\\Delta_{i,\\,j}^{2} = 2(1 - S_{i,\\,j})``, so ``S_{i,\\,j} = 1 - \\Delta_{i,\\,j}^{2} / 2``. That relation halves the squared distance, and this transformation does not. The audit of the sources found no work that states ``1 - d_{i,\\,j}^{2}``.

!!! warning

    The transformation needs a finite distance matrix. At an infinite entry it computes `ceil(Inf^2) - Inf^2`, which is `NaN`, and each finite entry becomes `Inf`. Real data reaches this case. [`LogDistance`](@ref) maps a correlation of exactly zero to an infinite distance, and this member is the default of [`DBHT`](@ref) and [`LoGo`](@ref). [`assert_similarity_domain`](@ref) refuses an infinite entry on the PMFG path.

# Algorithm

The branch of [`distance_to_similarity`](@ref) that this tag selects runs these steps.

 1. Take the largest entry of `D`, square that scalar, and round it up with `ceil`. The result is the ceiling. The code squares the maximum, and does not take the maximum of the squares. The two are equal, because a distance is never negative.
 2. Square every entry of `D`.
 3. Subtract the squared entries of step 2 from the scalar of step 1. The result is the similarity matrix. It keeps the element type of `D`, so a `Float32` distance matrix gives a `Float32` similarity matrix.

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

  - $(ref_dict[:cajas2025]) Section 13.1.4.2, footnote 7. $(ref_dict[:no_original_source])
  - $(ref_dict[:gower1966]) Section 3, Equation 4.
"""
struct MaximumDistanceSimilarity <: AbstractNonNegativeSimilarityMatrixAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Maps a distance of any magnitude into ``(0,\\,1]`` by ``e^{-D}``.

This member has no domain. It is non-negative at every finite distance, and it also takes an infinite distance, where `exp(-Inf)` is exactly `0`. Use it when the distance has no upper bound, as under [`LogDistance`](@ref), [`DistanceDistance`](@ref) or [`VariationInfoDistance`](@ref) with `normalise = false`. [`GeneralExponentialSimilarity`](@ref) is the other member with no domain.

# Mathematical definition

```math
\\begin{align}
S_{i,\\,j} &= e^{-D_{i,\\,j}}\\,,
\\end{align}
```

Where:

  - $(math_dict[:S_ij_sim])
  - $(math_dict[:D_mat_dist])
  - $(math_dict[:D_ij_dist])

# Algorithm

The branch of [`distance_to_similarity`](@ref) that this tag selects runs these steps.

 1. Negate every entry of `D`.
 2. Take the exponential of each negated entry. The result is the similarity matrix. It keeps the element type of `D`, so a `Float32` distance matrix gives a `Float32` similarity matrix.

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

Applies ``e^{-c D^{p}}``, the exponential transformation with a scale and an exponent.

`coef` sets the rate at which the similarity decreases with the distance, and `power` sets the shape of that decrease. A larger `coef` lowers the similarity at each positive distance. A larger `power` moves the similarity towards `1` below `D = 1`, and towards `0` above `D = 1`. Like [`ExponentialSimilarity`](@ref), this member has no domain, and at `coef = 1` and `power = 1` the two members give the same result.

# Mathematical definition

```math
\\begin{align}
S_{i,\\,j} &= e^{-c \\cdot D_{i,\\,j}^p}\\,,
\\end{align}
```

Where:

  - $(math_dict[:S_ij_sim])
  - $(math_dict[:D_mat_dist])
  - $(math_dict[:D_ij_dist])
  - ``c``: The scale factor, the field `coef`.
  - ``p``: The exponent, the field `power`.

# Algorithm

The branch of [`distance_to_similarity`](@ref) that this tag selects runs these steps.

 1. Read `power` and `coef` from the tag.
 2. Raise every entry of `D` to `power`.
 3. Multiply the result of step 2 by `-coef`. `coef` is a scalar, so one product scales the whole matrix.
 4. Take the exponential of each entry. The result is the similarity matrix. It keeps the element type of `D` when neither field has a wider type. The `Integer` defaults are not wider, so a `Float32` distance matrix gives a `Float32` similarity matrix.

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
        assert_gt0(power, :power)
        return new{typeof(coef), typeof(power)}(coef, power)
    end
end
function GeneralExponentialSimilarity(; coef::Number = 1,
                                      power::Number = 1)::GeneralExponentialSimilarity
    return GeneralExponentialSimilarity(coef, power)
end
"""
$(DocStringExtensions.TYPEDEF)

Takes the linear complement ``1 - D``, the exact inverse of a metric that is one minus a similarity.

For many metrics it recovers a similarity with a name, not a monotone function of one, so [`default_similarity`](@ref) uses it as the fallback. It is the one member that has a domain and also needs the correct metric. The two warnings below are about these two conditions.

# Mathematical definition

```math
\\begin{align}
S_{i,\\,j} &= 1 - D_{i,\\,j}\\,,
\\end{align}
```

Where:

  - $(math_dict[:S_ij_sim])
  - $(math_dict[:D_mat_dist])
  - $(math_dict[:D_ij_dist])

It recovers the named similarity of each distance that is one minus a similarity. Against `Distances.CosineDist` it returns the cosine similarity, against `Distances.Jaccard` the Ruzicka similarity, against `Distances.BrayCurtis` the Sørensen-Dice similarity, and against `Distances.CorrDist` the Pearson correlation.

!!! warning "The domain is `D <= 1`"

    The result is a non-negative similarity only when ``\\mathbf{D} \\in [0,\\,1]``. Above `1` the similarity is negative.

    The result of a negative value depends on the path. The [`FeatureDistance`](@ref) path keeps the value, and [`plot_clusters`](@ref) colours it on the signed scale ``[-1,\\,1]``. A distance above `2` gives a similarity below `-1`, and that scale clips it without a warning. On the PMFG path, [`assert_similarity_domain`](@ref) refuses the same input, because the consumers of [`PMFG_T2s`](@ref) cannot take a negative weight. On each path the similarity stays symmetric, with a unit diagonal.

    The rule is `D <= 1`. It is not a rule against a metric without an upper bound. `Distances.CosineDist` and `Distances.CorrDist` have the upper bound `2`, not `1`, and the check refuses them at each entry above `1`. `Distances.CorrDist` goes above `1` at each negative correlation.

    In the library, [`LogDistance`](@ref), [`DistanceDistance`](@ref) and [`VariationInfoDistance`](@ref) with `normalise = false` can go above `1`. Under its default `Distances.Euclidean`, [`DistanceDistance`](@ref) can go above `1` also when the base distance is in `[0, 1]`. For a member with no domain, use [`ExponentialSimilarity`](@ref) or [`GeneralExponentialSimilarity`](@ref).

!!! warning "The library does not check the pair"

    This member is the inverse of the metrics listed above, and of no other metric. With a different distance it returns a number that is in the domain, non-negative and wrong, and no path detects the error.

    [`SimpleDistance`](@ref) is ``\\sqrt{(1 - \\rho) / 2}``. A correlation of `0.003` gives `D = 0.706`, which is a legal bounded distance, and this member reports a similarity of `0.29`. No check can detect that error. On the [`FeatureDistance`](@ref) path, [`default_similarity`](@ref) pairs a metric with its inverse. The fields `alg` of [`NetworkEstimator`](@ref), `sim` of [`DBHT`](@ref) and `sim` of [`LoGo`](@ref) take a member without a reference to the distance estimator of ``\\mathbf{D}``. On those paths, the caller must pair the two correctly.

# Algorithm

The branch of [`distance_to_similarity`](@ref) that this tag selects runs these steps.

 1. Make the unit in the element type of `D` with `one(eltype(D))`. That unit keeps the element type of `D`, so a `Float32` distance matrix gives a `Float32` similarity matrix.
 2. Subtract every entry of `D` from that unit. The result is the similarity matrix.

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

It returns the correlation itself, not a monotone function of it. It is the one member that goes negative at a distance in `[0, 1]`, in this case above `D = 0.5`. Both facts have one cause. It is the exact algebraic inverse of [`AngularDist`](@ref), and a correlation can be negative.

# Mathematical definition

```math
\\begin{align}
S_{i,\\,j} &= \\cos\\left(\\pi D_{i,\\,j}\\right)\\,,
\\end{align}
```

Where:

  - $(math_dict[:S_ij_sim])
  - $(math_dict[:D_mat_dist])
  - $(math_dict[:D_ij_dist])

For an angular distance ``D_{i,\\,j} = \\arccos(\\rho_{i,\\,j}) / \\pi`` this member recovers ``\\rho_{i,\\,j}`` exactly, without the data of the distance. No source states the inverse, because it is algebra. The references below are the sources of the angular distance, which [`AngularDist`](@ref) also names. Against [`AngularDist`](@ref), the recovered cosine and the cosine of the features agree to floating-point precision. The member maps ``[0,\\,1] \\to [1,\\,-1]``, so the similarity is bounded, and its diagonal is `1` when the diagonal of the distance matrix is zero.

# Where this member is correct, and where it is refused

It is correct on the [`FeatureDistance`](@ref) path, which is the one path that pairs a metric with its inverse. On that path, [`default_similarity`](@ref) selects this member for [`AngularDist`](@ref), and the recovered ``\\rho`` is exact.

It is not a member of [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref), so `alg` of [`NetworkEstimator`](@ref), `sim` of [`DBHT`](@ref) and `sim` of [`LoGo`](@ref) refuse it at construction. No domain precondition can admit it. With the correct metric, this member is negative at each ``\\rho_{i,\\,j} < 0``, which is ordinary data, and the consumers of [`PMFG_T2s`](@ref) cannot take a negative weight.

!!! warning "The library does not check the pair"

    With a distance that is not angular, this member returns a number that is not a correlation. [`SimpleDistance`](@ref) is ``\\sqrt{(1 - \\rho) / 2}``, and its range is ``[0,\\,1]``, the same range as the domain of this member. No check can find the error. A correlation of `0.003` gives `D = 0.706`, and this member reports `-0.603`, which is not a weak negative correlation but a wrong number.

# Algorithm

The branch of [`distance_to_similarity`](@ref) that this tag selects runs these steps.

 1. Multiply every entry of `D` by `pi`. `pi` is an `Irrational`, which takes the element type of the number that it multiplies, so a `Float32` distance matrix stays `Float32`.
 2. Take the cosine of each scaled entry. The result is the similarity matrix.

The sign comes from step 2 alone. ``\\cos(\\pi D)`` is positive below ``D = 0.5``, zero at ``D = 0.5`` and negative above it. Thus [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref) does not admit this member.

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

  - $(ref_dict[:vandongen2012]) Section 4, Lemma 3.
  - $(ref_dict[:charikar2002]) Section 1.
"""
struct AngularSimilarity <: AbstractSimilarityMatrixAlgorithm end
"""
    distance_to_similarity(se::AbstractSimilarityMatrixAlgorithm; D::MatNum, kwargs...)

Compute a similarity matrix from a distance matrix with a similarity matrix algorithm.

# Algorithm

 1. Select the method by the type of `se`. Each member of [`AbstractSimilarityMatrixAlgorithm`](@ref) has one method, and the docstring of the member states the closed form and the steps of its branch.
 2. Apply that transformation to each entry of `D`. The result is a new matrix of the same size. The function does not write to `D`, and no method reads a keyword.

The function is a pure transformation with no domain of its own. It does not call [`assert_similarity_domain`](@ref). The caller checks the precondition of a member on `D`, at the five PMFG entry points.

# Arguments

  - `se`: The similarity matrix algorithm.

      + `se::MaximumDistanceSimilarity`: Uses the maximum distance transformation.
      + `se::ExponentialSimilarity`: Uses the exponential transformation.
      + `se::GeneralExponentialSimilarity`: Uses a generalised exponential transformation.
      + `se::ComplementSimilarity`: Uses the linear complement of the distance.
      + `se::AngularSimilarity`: Inverts a normalised angular distance.

  - `D`: The distance matrix.

  - `kwargs...`: More keyword arguments. No method reads them.

# Returns

  - `S::Matrix{<:Number}`: The similarity matrix, with the size of `D`.

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

Assert that `D` is in the domain of `sim`, so that the non-negativity guarantee of `sim` holds on it.

It is the second half of the enforcement of [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref). The type bound selects the algorithms that can reach the PMFG path, but it cannot select the data. Two of the four admitted members are non-negative only over a part of the distances that a caller can give.

| Member                              | Precondition                                |
|:----------------------------------- |:------------------------------------------- |
| [`ComplementSimilarity`](@ref)      | `all(D .<= 1)`                              |
| [`MaximumDistanceSimilarity`](@ref) | `all(isfinite, D)`                          |
| every other member                  | none, the fallback method does no operation |

# Scoped to the interface, not to the member

The five call sites are the PMFG entry points, and there is no other call site. They are [`calc_weighted_adjacency_graph`](@ref), [`calc_distance_weighted_graph`](@ref), the two [`clusterise`](@ref) methods that build a PMFG, and `logo!`. [`distance_to_similarity`](@ref) does not call this function, so it stays a pure transformation with no domain of its own.

The scope follows the failure, and the failure depends on the path, not on the member. On the [`FeatureDistance`](@ref) path, [`ComplementSimilarity`](@ref) against a distance without an upper bound is documented and tested behaviour. There, each `Distances.SemiMetric` gives a similarity, and no call raises an error. The same value is unusable only on the PMFG path.

# What it costs a caller

It costs nothing that works today. Each pair that this function refuses already raises an error one transformation later, at the check of [`PMFG_T2s`](@ref). That message names `W`, and not the configuration that made it. This function gives a better message, because it names the two halves of the pair. These are the distance estimator `de` that made the bad value, and the similarity that refused it. This is the reason that the caller gives `de`.

# Algorithm

 1. Select the method by the type of `sim`. Each of the two members of the table above has a method. Each other member reaches the method of [`AbstractSimilarityMatrixAlgorithm`](@ref), which does no operation and returns at once.
 2. Test the precondition of that member over all of `D`.
 3. Raise a `DomainError` when the test fails. The message holds `maximum(D)`, the name of the similarity that refused it, the closed form of that similarity, and the concrete type of `de`. Return `nothing` when the precondition holds.

# Arguments

  - `sim`: The similarity matrix algorithm.
  - `de`: The distance estimator that made `D`. The function reads it only to name it in the error.
  - `D`: The distance matrix.

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
    default_similarity(metric::AngularDist)

Select the similarity matrix algorithm that is the inverse of a distance metric.

A distance algorithm calls it to get the default of its similarity field from its metric. The printed object then shows the selected similarity, and the distance kernel does not select it out of view. The fallback is [`ComplementSimilarity`](@ref), the named inverse of each metric that is one minus a similarity. A metric whose inverse is not linear adds its own method.

# Algorithm

 1. Select the method by the type of `metric`. Dispatch does the selection, so a metric whose inverse is not linear adds a method of its own, and this function needs no branch for it.

 2. Return the similarity that the method names. There are two methods now.

      + `metric::Distances.SemiMetric` is the fallback, and returns [`ComplementSimilarity`](@ref). Each metric that is one minus a similarity reaches this method, and the complement recovers that named similarity.
      + `metric::AngularDist` returns [`AngularSimilarity`](@ref), the exact algebraic inverse of the angular distance. `src/06_Distance/05_FeatureDistance.jl` declares that method, next to [`AngularDist`](@ref), and this docstring documents it.

The pair that this function makes is correct, because it selects the inverse of the metric that it gets. A similarity that the caller selects has no such guarantee. [`ComplementSimilarity`](@ref) and [`AngularSimilarity`](@ref) state what a wrong pair returns.

# Arguments

  - `metric`: The distance metric.

# Returns

  - `sim::AbstractSimilarityMatrixAlgorithm`: The similarity matrix algorithm.

# Examples

```jldoctest
julia> PortfolioOptimisers.default_similarity(PortfolioOptimisers.Distances.CosineDist())
ComplementSimilarity()

julia> PortfolioOptimisers.default_similarity(PortfolioOptimisers.AngularDist())
AngularSimilarity()
```

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`AngularDist`](@ref)
  - [`FeatureDistance`](@ref)
  - [`distance_to_similarity`](@ref)
"""
function default_similarity(::Distances.SemiMetric)::ComplementSimilarity
    return ComplementSimilarity()
end

export MaximumDistanceSimilarity, ExponentialSimilarity, GeneralExponentialSimilarity,
       ComplementSimilarity, AngularSimilarity
