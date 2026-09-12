"""
    jlogo!(jlogo::MatNum, sigma::MatNum, source::MatNum, sign::Integer)

Efficiently accumulate contributions to the sparse inverse covariance matrix for LoGo/DBHT.

This internal function updates the `jlogo` matrix in-place by iterating over a list of cliques or separators (`source`), extracting the corresponding submatrix from the covariance matrix `sigma`, inverting it, and adding (or subtracting) the result to the appropriate block in `jlogo`, scaled by `sign`.

Every row of `source` names the same number of vertices, because `tmp` is allocated once at `size(source, 2)` and reused. [`PMFG_T2s`](@ref) satisfies that: its 3-cliques all carry three vertices and its 4-cliques all carry four.

# Algorithm

 1. Allocate `tmp`, one square block of the width of a row of `source`.
 2. For each row `i` of `source`, read the index set `v` and gather the submatrix `sigma[v, v]` into `tmp`.
 3. Invert `tmp`.
 4. Add `sign` times each entry of the inverse into `jlogo`, at the pair of `v` that entry belongs to.

# Arguments

  - `jlogo`: `N × N` matrix to be updated in-place. It is added to, never cleared, so the caller sets what it starts from.
  - `sigma`: `N × N` covariance matrix. Only the blocks the rows of `source` name are read.
  - `source`: `Ns×k` index matrix. Each row holds the `k` vertices of one clique or separator, and `k` is `4` for the cliques and `3` for the separators of a PMFG.
  - `sign`: `+1` for cliques, `-1` for separators.

# Returns

  - `nothing`. Updates `jlogo` in-place.

# Related

  - [`J_LoGo`](@ref)
  - [`LoGo`](@ref)
"""
function jlogo!(jlogo::MatNum, sigma::MatNum, source::MatNum, sign::Integer)
    tmp = Matrix{eltype(sigma)}(undef, size(source, 2), size(source, 2))

    # Pre-compute indices for better cache locality
    for i in axes(source, 1)
        v = view(source, i, :)

        # Fill temp matrix directly
        idx = 1
        for b in axes(source, 2)
            for a in axes(source, 2)
                tmp[idx] = sigma[v[a], v[b]]
                idx += 1
            end
        end

        # Compute inverse once
        tmp_inv = inv(tmp)

        # Update jlogo matrix directly
        idx = 1
        for b in axes(source, 2)
            for a in axes(source, 2)
                jlogo[v[a], v[b]] += sign * tmp_inv[idx]
                idx += 1
            end
        end
    end
    return nothing
end
"""
    J_LoGo(sigma::MatNum, separators::MatNum, cliques::MatNum)

Compute the sparse inverse covariance matrix using the LoGo (Local-Global) algorithm [J_LoGo](@cite).

This function implements the LoGo sparse inverse covariance estimation by combining clique and separator contributions from a Planar Maximally Filtered Graph (PMFG) or similar clique tree structure. It efficiently accumulates the inverses of covariance submatrices corresponding to cliques and separators, producing a sparse precision (inverse covariance) matrix suitable for robust portfolio optimization and risk management.

# Mathematical definition

```math
J_{i,\\,j} = \\sum_{c \\in \\mathcal{C}} \\mathbf{1}\\left[i \\in c \\land j \\in c\\right] \\left(\\mathbf{\\Sigma}_{c,\\,c}\\right)^{-1}_{i,\\,j}
           - \\sum_{s \\in \\mathcal{S}} \\mathbf{1}\\left[i \\in s \\land j \\in s\\right] \\left(\\mathbf{\\Sigma}_{s,\\,s}\\right)^{-1}_{i,\\,j}\\,.
```

Where:

  - ``\\mathbf{J}``: LoGo precision matrix, ``N \\times N``.
  - ``\\mathbf{\\Sigma}``: Covariance matrix, ``N \\times N``.
  - ``\\mathbf{\\Sigma}_{c,\\,c}``: Its submatrix on the index set ``c``.
  - ``\\mathcal{C}``: Set of the cliques of the network.
  - ``\\mathcal{S}``: Set of its separators.
  - $(math_dict[:N])

``J_{i,\\,j}`` is exactly zero for a pair that no clique holds together, so the sparsity pattern of ``\\mathbf{J}`` is the edge set of the network. That is the conditional independence the filtering states, and it survives in the precision alone.

# Algorithm

 1. Set `jlogo` to a zero matrix of the size of `sigma`.
 2. Add the inverse of every clique block with [`jlogo!`](@ref) at `sign = 1`.
 3. Subtract the inverse of every separator block with [`jlogo!`](@ref) at `sign = -1`.

# Arguments

  - `sigma`: `N × N` covariance matrix.
  - `separators`: `Ns×3` index matrix. Each row holds the vertices of one separator, which are the 3-cliques of a PMFG.
  - `cliques`: `Nq×4` index matrix. Each row holds the vertices of one clique, which are the 4-cliques of a PMFG.

# Returns

  - `jlogo::Matrix{<:Number}`: `N × N` LoGo sparse precision matrix. The covariance it stands for is its inverse, and that inverse is dense.

# Related

  - [`jlogo!`](@ref)
  - [`LoGo`](@ref)
"""
function J_LoGo(sigma::MatNum, separators::MatNum, cliques::MatNum)
    jlogo = zeros(eltype(sigma), size(sigma))
    jlogo!(jlogo, sigma, cliques, 1)
    jlogo!(jlogo, sigma, separators, -1)
    return jlogo
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

No-op fallback: return `nothing` when no LoGo algorithm is configured.

This is the branch a matrix processing pipeline takes when its sparsification field is `nothing`, so a caller composes the step in and out without a branch of its own.

# Algorithm

 1. Return `nothing`. No matrix is read and no matrix is written, and `args` and `kwargs` are discarded.

# Arguments

  - `::Nothing`: No LoGo algorithm configured.
  - `args...`: Optional arguments (ignored).
  - `kwargs...`: Optional keyword arguments (ignored).

# Returns

  - `nothing`. The caller's `sigma` is left as it stands.

# Related

  - [`LoGo`](@ref)
  - [`logo!`](@ref)
"""
function logo!(::Nothing, args...; kwargs...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all inverse matrix sparsification algorithms.

A member of this family imposes a sparsity pattern on the **inverse** of a covariance matrix rather than on the matrix itself. The covariance that comes back is dense; what is sparse is its precision, and the zeros there are the conditional independences the information filtering network selected.

The family declares no seam of its own, and no method dispatches on this supertype. A concrete subtype is reached through the [`matrix_processing_algorithm!`](@ref) of [`AbstractMatrixProcessingAlgorithm`](@ref), which is the interface it inherits and which `src/07_MatrixProcessing.jl` owns. [`LoGo`](@ref) is the shipped member, and [`matrix_processing_algorithm!`](@ref) states the contract that method satisfies.

# Related

  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`LoGo`](@ref)

# References

  - $(ref_dict[:J_LoGo])
"""
abstract type InverseMatrixSparsificationAlgorithm <: AbstractMatrixProcessingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Sparsifies the inverse covariance matrix on the cliques of an information filtering network.

`LoGo` is a composable algorithm type for estimating sparse inverse covariance matrices using the Planar Maximally Filtered Graph (PMFG) and clique-based decomposition, as described in [J_LoGo](@cite). It combines a distance estimator and a similarity matrix algorithm, both validated and extensible, to produce a robust, interpretable sparse precision matrix for use in portfolio optimization and risk management.

# What is sparse is the precision, not the covariance

[`J_LoGo`](@ref) sums the inverse of each clique block and subtracts the inverse of each separator block, and the matrix that comes out is **exactly zero** wherever the network carries no edge. Measured over a 20-asset sample, the triangulated maximally filtered graph holds `54` edges — the `3n - 6` of a maximal planar graph — and the largest absolute entry of the precision matrix away from those edges is `0.0`.

`sigma` is then replaced by the inverse of that precision matrix, so what the caller receives is dense. The filtering is a statement about which pairs are conditionally independent given the rest, and it survives only in the precision.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LoGo(;
        de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
        sim::AbstractNonNegativeSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
        pdm::Option{<:AbstractPosdefEstimator} = Posdef()
    ) -> LoGo

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> LoGo()
LoGo
   de ┼ Distance
      │   power ┼ nothing
      │     alg ┴ CanonicalDistance()
  sim ┼ MaximumDistanceSimilarity()
  pdm ┼ Posdef
      │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
      │   kwargs ┴ @NamedTuple{}: NamedTuple()
```

# Related

  - [`InverseMatrixSparsificationAlgorithm`](@ref)
  - [`AbstractDistanceEstimator`](@ref)
  - [`AbstractNonNegativeSimilarityMatrixAlgorithm`](@ref)
  - [`AbstractSimilarityMatrixAlgorithm`](@ref)
  - [`MaximumDistanceSimilarity`](@ref)
  - [`ExponentialSimilarity`](@ref)
  - [`GeneralExponentialSimilarity`](@ref)

# References

  - $(ref_dict[:J_LoGo])
"""
@concrete struct LoGo <: InverseMatrixSparsificationAlgorithm
    """
    $(field_dict[:de])
    """
    de
    """
    $(field_dict[:sim])
    """
    sim <: AbstractNonNegativeSimilarityMatrixAlgorithm
    """
    $(field_dict[:pdm])
    """
    pdm
    function LoGo(de::AbstractDistanceEstimator,
                  sim::AbstractNonNegativeSimilarityMatrixAlgorithm,
                  pdm::Option{<:AbstractPosdefEstimator} = Posdef())
        return new{typeof(de), typeof(sim), typeof(pdm)}(de, sim, pdm)
    end
end
function LoGo(; de::AbstractDistanceEstimator = Distance(; alg = CanonicalDistance()),
              sim::AbstractNonNegativeSimilarityMatrixAlgorithm = MaximumDistanceSimilarity(),
              pdm::Option{<:AbstractPosdefEstimator} = Posdef())
    return LoGo(de, sim, pdm)
end
"""
    const DVarInfo_DDVarInfo = Union{<:Distance{<:Any, <:VariationInfoDistance},
                                     <:DistanceDistance{<:Any, <:Any, <:Any, <:Any,
                                                        <:VariationInfoDistance}}

Alias for distance types using variation of information metrics.

Matches either a [`VariationInfoDistance`](@ref)-based [`Distance`](@ref) or a [`VariationInfoDistance`](@ref)-based [`DistanceDistance`](@ref). Used for dispatch in DBHT-based phylogeny computation.

# Related

  - [`VariationInfoDistance`](@ref)
  - [`Distance`](@ref)
  - [`DistanceDistance`](@ref)
"""
const DVarInfo_DDVarInfo = Union{<:Distance{<:Any, <:VariationInfoDistance},
                                 <:DistanceDistance{<:Any, <:Any, <:Any, <:Any,
                                                    <:VariationInfoDistance}}
"""
    LoGo_dist_assert(de::DVarInfo_DDVarInfo, sigma::MatNum, X::MatNum)

Validate compatibility of the distance estimator and covariance matrix for LoGo sparse inverse covariance estimation by checking `size(sigma, 1) == size(X, 2)`.

The check runs for a [`VariationInfoDistance`](@ref) estimator alone, which is the only family that reads `X` rather than the correlation matrix. Every other estimator takes the no-op fallback, so a mismatched `X` passes. The narrow signature is what makes that so: it is bounded by [`DVarInfo_DDVarInfo`](@ref), and the configurations that reach it are a [`Distance`](@ref) or a [`DistanceDistance`](@ref) whose algorithm is a [`VariationInfoDistance`](@ref).

# Arguments

  - `de`: Distance estimator whose algorithm is a [`VariationInfoDistance`](@ref).
  - `sigma`: `N × N` covariance matrix.
  - `X`: `T × N` data matrix. `size(X, 2)` is the asset axis, which is the axis the check reads.

# Validation

  - `size(sigma, 1) == size(X, 2)`.

# Returns

  - `nothing`.

# Related

  - [`LoGo`](@ref)
  - [`logo!`](@ref)
"""
function LoGo_dist_assert(::DVarInfo_DDVarInfo, sigma::MatNum, X::MatNum)
    @argcheck(size(sigma, 1) == size(X, 2), DimensionMismatch)
    return nothing
end
"""
    LoGo_dist_assert(args...)

No-op fallback for other distance estimators.

Every distance estimator outside [`DVarInfo_DDVarInfo`](@ref) derives its distance from the correlation matrix and never reads `X`, so there is no shape of `X` for it to disagree with. This method makes that the default and leaves the check to the one family that owns it.

# Algorithm

 1. Return `nothing`. No shape is read, and `args` is discarded.

# Arguments

  - `args...`: The distance estimator, the covariance matrix and the data matrix (all ignored).

# Returns

  - `nothing`.

# Related

  - [`DVarInfo_DDVarInfo`](@ref)
  - [`LoGo`](@ref)
  - [`logo!`](@ref)
"""
function LoGo_dist_assert(args...)
    return nothing
end
"""
    logo!(je::LoGo, sigma::MatNum, X::MatNum;
          dims::Int = 1, kwargs...)

Compute the LoGo (Local-Global) covariance matrix and update `sigma` in-place.

This method implements the LoGo algorithm for sparse inverse covariance estimation using the Planar Maximally Filtered Graph (PMFG) and clique-based decomposition. It validates inputs, computes the similarity and distance matrices, constructs the PMFG, identifies cliques and separators, and updates the input covariance matrix `sigma` in-place by inverting the LoGo sparse inverse covariance estimate. The result is projected to the nearest positive definite matrix if a `Posdef` estimator is not `nothing`.

# Algorithm

 1. Check that `sigma` is square, and check its asset axis against `X` through [`LoGo_dist_assert`](@ref).
 2. Read the diagonal of `sigma` into `s`. When any entry of `s` is not one, `sigma` is a covariance matrix: replace `s` with its square roots and derive the correlation matrix `S` with `StatsBase.cov2cor`. `sigma` itself stays a covariance, and it is what step 6 decomposes.
 3. Take the distance matrix `D` from `S` and `X` with `je.de`, and check that `D` lies in the domain `je.sim` needs.
 4. Map `D` to the non-negative similarity `S` with `je.sim`, through [`distance_to_similarity`](@ref).
 5. Build the TMFG on `S` with [`PMFG_T2s`](@ref) at `nargout = 4`, and take its 3-cliques as the separators and its 4-cliques as the cliques.
 6. Build the LoGo precision matrix from `sigma` with [`J_LoGo`](@ref), invert it, and write the result into `sigma`.
 7. Repair `sigma` with [`posdef!`](@ref) through `je.pdm`, which does nothing when `je.pdm` is `nothing`.

# Arguments

  - `je`: LoGo algorithm instance.
  - `sigma`: Covariance matrix (`N × N`), updated in-place with the LoGo sparse inverse covariance.
  - `X`: Data matrix (`T × N`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to distance and similarity estimators.

# Validation

  - `size(sigma, 1) == size(sigma, 2)`.
  - `size(sigma, 1) == size(X, 2)`, **only when `je.de` reads `X`**. [`LoGo_dist_assert`](@ref) carries the check, and it has a method for the variation-of-information estimators alone; every other estimator takes the no-op fallback. A default `LoGo()` therefore accepts a `20 × 20` `sigma` beside a `400 × 10` `X` and returns without raising, because [`CanonicalDistance`](@ref) derives the distance from the correlation matrix and never touches `X`.

# Returns

  - `nothing`. The input `sigma` is updated in-place.

# Related

  - [`LoGo`](@ref)
  - [`J_LoGo`](@ref)
  - [`LoGo_dist_assert`](@ref)
  - [`PMFG_T2s`](@ref)
  - [`distance_to_similarity`](@ref)
  - [`Posdef`](@ref)
"""
function logo!(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1, kwargs...)
    assert_matrix_issquare(sigma, :sigma)
    LoGo_dist_assert(je.de, sigma, X)
    s = LinearAlgebra.diag(sigma)
    iscov = any(!isone, s)
    S = if iscov
        s .= sqrt.(s)
        StatsBase.cov2cor(sigma, s)
    else
        sigma
    end
    D = distance(je.de, S, X; dims = dims, kwargs...)
    assert_similarity_domain(je.sim, je.de, D)
    S = distance_to_similarity(je.sim; S = S, D = D)
    separators, cliques = PMFG_T2s(S, 4)[3:4]
    sigma .= J_LoGo(sigma, separators, cliques) \ LinearAlgebra.I
    posdef!(je.pdm, sigma)
    return nothing
end
"""
    logo(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1, kwargs...) -> MatNum

Apply the LoGo (Local-Global) transformation to the covariance matrix and return the result as a new matrix.

This is the non-mutating variant of [`logo!`](@ref). It copies `sigma` before applying the transformation.

# Algorithm

 1. Copy `sigma`.
 2. Run [`logo!`](@ref) on the copy, which carries every step and every check of this transformation.
 3. Return the copy.

# Arguments

  - `je::LoGo`: LoGo algorithm configuration.
  - `sigma::MatNum`: `N × N` covariance matrix to transform (not mutated).
  - `X::MatNum`: `T × N` returns data matrix.
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to distance and similarity estimators.

# Validation

  - Every check of [`logo!`](@ref) applies, and it raises from step 2.

# Returns

  - `sigma::MatNum`: `N × N` copy of the input with the LoGo transformation applied.

# Related

  - [`logo!`](@ref)
  - [`LoGo`](@ref)
  - [`J_LoGo`](@ref)
"""
function logo(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1, kwargs...)
    sigma = copy(sigma)
    logo!(je, sigma, X; dims = dims, kwargs...)
    return sigma
end
"""
    matrix_processing_algorithm!(je::LoGo, sigma::MatNum,
                                 X::MatNum; dims::Int = 1, kwargs...)

Apply the LoGo (Local-Global) transformation in-place to the covariance matrix, as a step of the matrix processing pipeline.

This method provides a standard interface for applying the LoGo algorithm to a covariance matrix within the matrix processing pipeline of `PortfolioOptimisers.jl`. It validates inputs, computes the LoGo sparse inverse covariance matrix, and updates `sigma` in-place. If a positive definite matrix estimator (`pdm`) is not `nothing`, the result is projected to the nearest positive definite matrix.

This is the contract [`LoGo`](@ref) satisfies as a member of [`AbstractMatrixProcessingAlgorithm`](@ref): the pipeline calls `matrix_processing_algorithm!(alg, sigma, X; dims, kwargs...)` on each of its algorithms in turn, each one writes into the same `sigma`, and each returns `nothing`. The family lives in `src/07_MatrixProcessing.jl`, and this method is the only one of it that this file declares.

# Algorithm

 1. Forward every argument to [`logo!`](@ref), which carries the steps and the checks of the transformation.

# Arguments

  - `je`: LoGo algorithm instance (`LoGo`). Its own `pdm` field carries the positive definite repair, so there is no `pdm` argument here.
  - `sigma`: Covariance matrix (`N × N`), updated in-place.
  - `X`: Data matrix (`T × N` or `N × T`).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to distance and similarity estimators.

# Validation

  - Every check of [`logo!`](@ref) applies, and it raises from step 1.

# Returns

  - `nothing`. The input `sigma` is updated in-place.

# Related

  - [`LoGo`](@ref)
  - [`logo!`](@ref)
  - [`Posdef`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
"""
function matrix_processing_algorithm!(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1,
                                      kwargs...)
    return logo!(je, sigma, X; dims = dims, kwargs...)
end

export LoGo
