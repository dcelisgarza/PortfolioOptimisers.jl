"""
    jlogo!(jlogo::MatNum, sigma::MatNum, source::MatNum, sign::Integer)

Add `sign` times the inverse of each block of `sigma` that a row of `source` names into the same block of `jlogo`.

[`J_LoGo`](@ref) calls this function two times, once on the cliques and once on the separators. For each row of `source`, the function takes the submatrix of `sigma` on the vertices of the row and inverts it. It then adds `sign` times the inverse into `jlogo` in place.

Every row of `source` must name the same number of vertices, because the function allocates `tmp` once at `size(source, 2)` and uses it again for each row. [`PMFG_T2s`](@ref) satisfies that. Each of its 3-cliques has three vertices, and each of its 4-cliques has four.

# Algorithm

 1. Allocate `tmp`, one square block of the width of a row of `source`.
 2. For each row `i` of `source`, read the index set `v` and gather the submatrix `sigma[v, v]` into `tmp`.
 3. Invert `tmp`.
 4. Add `sign` times each entry of the inverse into `jlogo`, at the pair of `v` that entry belongs to.

# Arguments

  - `jlogo`: `N × N` matrix that the function changes in place. The function adds to it and never clears it, so the caller sets its first value.
  - `sigma`: `N × N` covariance matrix. The function reads only the blocks that the rows of `source` name.
  - `source`: `Ns×k` index matrix. Each row holds the `k` vertices of one clique or separator. `k` is `4` for the cliques and `3` for the separators of a TMFG.
  - `sign`: `+1` for the cliques, and `-1` for the separators.

# Returns

  - `nothing`. The function changes `jlogo` in place.

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

Compute the sparse precision matrix of the LoGo (Local-Global) method of [J_LoGo](@cite).

The function adds the inverse of the covariance block of each clique of a Triangulated Maximally Filtered Graph (TMFG), and it subtracts the inverse of the covariance block of each separator. The precision matrix is the inverse of a covariance matrix.

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

``J_{i,\\,j}`` is exactly zero for a pair that no clique holds together, so the sparsity pattern of ``\\mathbf{J}`` is the edge set of the network. Each zero is a conditional independence that the filtering states, and only the precision matrix holds it.

Equation 4 of [J_LoGo](@cite) weights the inverse of each separator block by ``k(S) - 1``, where ``k(S)`` is the number of components that the removal of the separator leaves. In a TMFG every separator joins exactly two cliques, so the weight is one and the formula above is Equation 4. On another clique tree the formula needs that weight.

# Algorithm

 1. Set `jlogo` to a zero matrix of the size of `sigma`.
 2. Add the inverse of every clique block with [`jlogo!`](@ref) at `sign = 1`.
 3. Subtract the inverse of every separator block with [`jlogo!`](@ref) at `sign = -1`.

# Arguments

  - `sigma`: `N × N` covariance matrix.
  - `separators`: `Ns×3` index matrix. Each row holds the vertices of one separator, which are the 3-cliques of a TMFG.
  - `cliques`: `Nq×4` index matrix. Each row holds the vertices of one clique, which are the 4-cliques of a TMFG.

# Returns

  - `jlogo::Matrix{<:Number}`: `N × N` LoGo sparse precision matrix. Its inverse is the covariance matrix, and that inverse is dense.

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

Return `nothing` when no LoGo algorithm is set.

No method of the library calls this method. When the algorithm of the matrix processing pipeline is `nothing`, the pipeline calls the `nothing` method of [`matrix_processing_algorithm!`](@ref) instead.

# Algorithm

 1. Return `nothing`. The method reads no matrix and writes no matrix, and it ignores `args` and `kwargs`.

# Arguments

  - `::Nothing`: No LoGo algorithm is set.
  - `args...`: Positional arguments, which the method ignores.
  - `kwargs...`: Keyword arguments, which the method ignores.

# Returns

  - `nothing`. The method does not change the `sigma` of the caller.

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

A member of this family makes the inverse of a covariance matrix sparse, and it does not make the matrix itself sparse. The covariance matrix that it writes is dense. What is sparse is its precision matrix, and each zero of the precision matrix is a conditional independence that the information filtering network selects.

The family declares no interface of its own, and no method dispatches on this supertype. The pipeline reaches a concrete subtype through the [`matrix_processing_algorithm!`](@ref) method of [`AbstractMatrixProcessingAlgorithm`](@ref). The family gets that interface from its supertype, and `src/04_MatrixProcessing/04_MatrixProcessing.jl` owns it. [`LoGo`](@ref) is the one member that the library ships, and its [`matrix_processing_algorithm!`](@ref) method states the contract that it satisfies.

# Related

  - [`AbstractMatrixProcessingAlgorithm`](@ref)
  - [`LoGo`](@ref)

# References

  - $(ref_dict[:J_LoGo]) Section II.C, Equation 4.
"""
abstract type InverseMatrixSparsificationAlgorithm <: AbstractMatrixProcessingAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Sparsifies the inverse covariance matrix on the cliques of an information filtering network.

`LoGo` selects the LoGo method of [J_LoGo](@cite) as the algorithm of a [`MatrixProcessing`](@ref) estimator. The method builds a Triangulated Maximally Filtered Graph (TMFG) on a similarity matrix of the assets, and it computes the precision matrix from the covariance blocks of the cliques of the graph. `de` gives the distance matrix, and `sim` maps it to the similarity matrix.

# What is sparse is the precision, not the covariance

[`J_LoGo`](@ref) sums the inverse of each clique block and subtracts the inverse of each separator block. The result is exactly zero at each pair that no edge of the network joins.

[`logo!`](@ref) then writes the inverse of that precision matrix into `sigma`, so the caller gets a dense matrix. The filtering states which pairs are conditionally independent given the other assets, and only the precision matrix holds that statement.

# The graph comes from `sim`, not from the likelihood

Algorithm 1 of [J_LoGo](@cite) builds the TMFG by the log-likelihood. At each step it inserts the vertex that maximises ``\\log\\det\\hat{\\mathbf{R}}_{S} - \\log\\det\\hat{\\mathbf{R}}_{C}``. The same work states that a TMFG on the squared correlations approximates that choice to second order in the correlations. This type builds the TMFG with [`PMFG_T2s`](@ref) on the similarity matrix of `sim`, and its default similarity is neither of the two. [J_LoGo](@cite) returns the precision matrix. The inversion and the repair by `pdm` come from this library alone.

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

  - $(ref_dict[:J_LoGo]) Section II.C, Equation 4 and Algorithm 1.
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

Alias for the distance estimators whose algorithm is a [`VariationInfoDistance`](@ref).

The alias matches a [`Distance`](@ref) or a [`DistanceDistance`](@ref) whose algorithm is a [`VariationInfoDistance`](@ref). [`LoGo_dist_assert`](@ref) dispatches on it, because the distance of these estimators reads the asset axis of `X`.

# Related

  - [`VariationInfoDistance`](@ref)
  - [`Distance`](@ref)
  - [`DistanceDistance`](@ref)
  - [`LoGo_dist_assert`](@ref)
"""
const DVarInfo_DDVarInfo = Union{<:Distance{<:Any, <:VariationInfoDistance},
                                 <:DistanceDistance{<:Any, <:Any, <:Any, <:Any,
                                                    <:VariationInfoDistance}}
"""
    LoGo_dist_assert(de::DVarInfo_DDVarInfo, sigma::MatNum, X::MatNum; dims::Int = 1)

Check that the asset axis of `X` has the size of `sigma`, for a distance estimator that reads `X`.

The check runs for a [`VariationInfoDistance`](@ref) estimator alone, because the distance of that family reads the asset axis of `X`. Every other estimator of the library takes the fallback, which does nothing, so a mismatched `X` passes. The narrow signature selects the check. Its bound is [`DVarInfo_DDVarInfo`](@ref), so a [`Distance`](@ref) or a [`DistanceDistance`](@ref) whose algorithm is a [`VariationInfoDistance`](@ref) reaches it.

# Arguments

  - `de`: Distance estimator whose algorithm is a [`VariationInfoDistance`](@ref).
  - `sigma`: `N × N` covariance matrix.
  - `X`: `T × N` data matrix when `dims` is `1`, and `N × T` when `dims` is `2`. The check reads the asset axis, `size(X, 2)` when `dims` is `1` and `size(X, 1)` when `dims` is `2`.
  - $(arg_dict[:dims])

# Validation

  - `size(sigma, 1) == size(X, isone(dims) ? 2 : 1)`, raising `DimensionMismatch`.

# Returns

  - `nothing`.

# Related

  - [`LoGo`](@ref)
  - [`logo!`](@ref)
"""
function LoGo_dist_assert(::DVarInfo_DDVarInfo, sigma::MatNum, X::MatNum; dims::Int = 1)
    @argcheck(size(sigma, 1) == size(X, isone(dims) ? 2 : 1), DimensionMismatch)
    return nothing
end
"""
    LoGo_dist_assert(args...; kwargs...)

Do nothing for a distance estimator that does not read the asset axis of `X`.

A distance estimator of the library outside [`DVarInfo_DDVarInfo`](@ref) derives its distance from the correlation matrix, so [`logo!`](@ref) reads no axis of `X` for it. A [`FeatureDistance`](@ref) is the exception, and it raises an `IsNothingError` in [`logo!`](@ref), because matrix processing gives it no panel. This method is the default, and the narrow method holds the check for the one family that needs it.

# Algorithm

 1. Return `nothing`. The method reads no size, and it ignores `args` and `kwargs`.

# Arguments

  - `args...`: The distance estimator, the covariance matrix and the data matrix, which the method ignores.
  - `kwargs...`: The keyword `dims`, which the method ignores.

# Returns

  - `nothing`.

# Related

  - [`DVarInfo_DDVarInfo`](@ref)
  - [`LoGo`](@ref)
  - [`logo!`](@ref)
"""
function LoGo_dist_assert(args...; kwargs...)
    return nothing
end
"""
    logo!(je::LoGo, sigma::MatNum, X::MatNum;
          dims::Int = 1, kwargs...)

Write the LoGo (Local-Global) covariance matrix into `sigma`.

The method builds a Triangulated Maximally Filtered Graph (TMFG) on a similarity matrix of the assets. It computes the LoGo precision matrix from the clique and separator blocks of `sigma`, and it writes the inverse of that precision matrix into `sigma`. When `je.pdm` is not `nothing`, [`posdef!`](@ref) then repairs `sigma` to the nearest positive definite matrix.

# Algorithm

 1. Check that `sigma` is square. Check with [`LoGo_dist_assert`](@ref) that the asset axis of `X` has the size of `sigma`.
 2. Read the diagonal of `sigma` into `s`. When an entry of `s` is not one, `sigma` is a covariance matrix. Then replace `s` with its square roots, and get the correlation matrix `S` with `StatsBase.cov2cor`. `sigma` stays a covariance matrix, and step 6 decomposes it.
 3. Take the distance matrix `D` from `S` and `X` with `je.de`. Check with [`assert_similarity_domain`](@ref) that `D` lies in the domain that `je.sim` needs.
 4. Map `D` to the non-negative similarity `S` with `je.sim`, through [`distance_to_similarity`](@ref).
 5. Build the TMFG on `S` with [`PMFG_T2s`](@ref) at `nargout = 4`, and take its 3-cliques as the separators and its 4-cliques as the cliques.
 6. Build the LoGo precision matrix from `sigma` with [`J_LoGo`](@ref), invert it, and write the result into `sigma`.
 7. Repair `sigma` with [`posdef!`](@ref) through `je.pdm`, which does nothing when `je.pdm` is `nothing`.

# Arguments

  - `je`: The LoGo algorithm.
  - `sigma`: `N × N` covariance or correlation matrix. The method writes the LoGo covariance matrix into it.
  - `X`: `T × N` data matrix when `dims` is `1`, and `N × T` when `dims` is `2`.
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments that the method passes to [`distance`](@ref).

# Validation

  - `size(sigma, 1) == size(sigma, 2)`.
  - The asset axis of `X` has `size(sigma, 1)` entries, when `je.de` reads `X`. [`LoGo_dist_assert`](@ref) holds this check, and it has a method for the variation-of-information estimators alone. Every other estimator takes the fallback, which does nothing. Thus a default `LoGo()` accepts a `20 × 20` `sigma` with a `400 × 10` `X` and raises no error, because [`CanonicalDistance`](@ref) derives the distance from the correlation matrix and does not read `X`.

# Returns

  - `nothing`. The method changes `sigma` in place.

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
    LoGo_dist_assert(je.de, sigma, X; dims = dims)
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

Return a copy of `sigma` with the LoGo (Local-Global) transformation applied.

This is the out-of-place form of [`logo!`](@ref). It copies `sigma`, and it runs [`logo!`](@ref) on the copy.

# Algorithm

 1. Copy `sigma`.
 2. Run [`logo!`](@ref) on the copy. It does every step and every check of the transformation.
 3. Return the copy.

# Arguments

  - `je::LoGo`: The LoGo algorithm.
  - `sigma::MatNum`: `N × N` covariance or correlation matrix. The function does not change it.
  - `X::MatNum`: `T × N` returns data matrix when `dims` is `1`, and `N × T` when `dims` is `2`.
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments that [`logo!`](@ref) passes to [`distance`](@ref).

# Validation

  - Every check of [`logo!`](@ref) applies. The error comes from step 2.

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

Apply the LoGo (Local-Global) transformation to `sigma` in place, as the `:alg` step of [`matrix_processing!`](@ref).

The method runs [`logo!`](@ref) on `sigma` and returns `sigma`. When `je.pdm` is not `nothing`, [`logo!`](@ref) repairs the result to the nearest positive definite matrix.

[`LoGo`](@ref) satisfies the contract of the [`AbstractMatrixProcessingAlgorithm`](@ref) family with this method. The pipeline calls `matrix_processing_algorithm!(alg, sigma, X; kwargs...)` on its algorithm, the algorithm writes into `sigma`, and the method returns `sigma`. `src/04_MatrixProcessing/04_MatrixProcessing.jl` holds the family, and this file declares one method of it.

# Algorithm

 1. Pass every argument to [`logo!`](@ref), which does the steps and the checks of the transformation.
 2. Return `sigma`.

# Arguments

  - `je`: The LoGo algorithm. Its field `pdm` holds the positive definite repair, so the method takes no `pdm` argument.
  - `sigma`: `N × N` covariance or correlation matrix. The method writes the result into it.
  - `X`: `T × N` data matrix when `dims` is `1`, and `N × T` when `dims` is `2`.
  - $(arg_dict[:dims])
  - `kwargs...`: Keyword arguments that [`logo!`](@ref) passes to [`distance`](@ref).

# Validation

  - Every check of [`logo!`](@ref) applies. The error comes from step 1.

# Returns

  - `sigma::MatNum`: The input matrix `sigma`, which now holds the LoGo covariance matrix.

# Related

  - [`LoGo`](@ref)
  - [`logo!`](@ref)
  - [`Posdef`](@ref)
  - [`AbstractMatrixProcessingAlgorithm`](@ref)
"""
function matrix_processing_algorithm!(je::LoGo, sigma::MatNum, X::MatNum; dims::Int = 1,
                                      kwargs...)
    logo!(je, sigma, X; dims = dims, kwargs...)
    return sigma
end

export LoGo
