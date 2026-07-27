"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all similarity matrix algorithms.

Every subtype defines a pure transformation of a distance matrix into a similarity matrix, applied by [`distance_to_similarity`](@ref). The family is consumed in three places: the Planar Maximally Filtered Graph (PMFG) construction used by [`DBHT`](@ref) and [`LoGo`](@ref), the similarity-based [`NetworkEstimator`](@ref) adjacency and clustering routines, and the similarity slot returned by [`cor_and_dist`](@ref).

# Related

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

Similarity matrix algorithm using the maximum distance transformation.

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

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
"""
struct MaximumDistanceSimilarity <: AbstractSimilarityMatrixAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Similarity matrix algorithm using the exponential transformation.

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

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
"""
struct ExponentialSimilarity <: AbstractSimilarityMatrixAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Similarity matrix algorithm using a generalised exponential transformation.

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
        coef::Number = 1.0,
        power::Number = 1.0
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

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
"""
@concrete struct GeneralExponentialSimilarity <: AbstractSimilarityMatrixAlgorithm
    """
    $(field_dict[:dbhtcoef])
    """
    coef
    """
    $(field_dict[:dbhtpower])
    """
    power
    function GeneralExponentialSimilarity(coef::Number, power::Number)
        @argcheck(zero(coef) < coef, DomainError)
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

Similarity matrix algorithm using the linear complement of the distance.

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

!!! warning

    The result is only correlation-like when ``\\mathbf{D} \\in [0,\\,1]``. An unbounded metric produces arbitrarily negative similarities: `Distances.Euclidean` with ``D_{i,\\,j} = 7`` gives ``S_{i,\\,j} = -6``, outside the ``[-1,\\,1]`` range that [`plot_clusters`](@ref) assumes, where it is silently clipped rather than flagged. Symmetry and the unit diagonal always survive. Use [`ExponentialSimilarity`](@ref) or [`GeneralExponentialSimilarity`](@ref) for a bounded alternative.

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`AngularSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`default_similarity`](@ref)
"""
struct ComplementSimilarity <: AbstractSimilarityMatrixAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Similarity matrix algorithm inverting a normalised angular distance.

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

For an angular distance ``D_{i,\\,j} = \\arccos(\\rho_{i,\\,j}) / \\pi`` this recovers ``\\rho_{i,\\,j}`` exactly, without reference to the data the distance was computed from. It maps ``[0,\\,1] \\to [1,\\,-1]``, so the similarity is bounded and the diagonal is unity whenever the distance matrix has a zero diagonal.

# Related

  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)
  - [`ComplementSimilarity`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`default_similarity`](@ref)
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
