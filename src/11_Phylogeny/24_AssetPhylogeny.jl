"""
    asset_phylogeny(w::VecNum, X::MatNum)

Compute the asset phylogeny score for a set of weights and a phylogeny matrix.

This function computes the weighted sum of the phylogeny matrix, normalised by the sum of absolute weights. The asset phylogeny score quantifies the degree of phylogenetic (network or cluster-based) structure present in the portfolio allocation. It is the percentage invested in connected assets of [`NetworkEstimator`](@ref)'s Equation 13.7,

```math
\\begin{align}
    \\mathrm{CA}(\\boldsymbol{x}) &= \\dfrac{\\boldsymbol{1}_n^{\\intercal} \\left(\\mathbf{B}_{1,\\,l} \\odot \\lvert \\boldsymbol{x}\\boldsymbol{x}^{\\intercal} \\rvert\\right) \\boldsymbol{1}_n}{\\boldsymbol{1}_n^{\\intercal} \\lvert \\boldsymbol{x}\\boldsymbol{x}^{\\intercal} \\rvert \\boldsymbol{1}_n}\\,,
\\end{align}
```

Where:

  - ``\\mathbf{B}_{1,\\,l}``: Phylogeny matrix from [`phylogeny_matrix`](@ref).
  - ``\\odot``: Hadamard, element-wise product.
  - ``\\boldsymbol{x}``: Portfolio weight vector.
  - ``\\boldsymbol{1}_n``: Column vector of ones of length ``n``.

Two assets that are not related contribute nothing, and a pair contributes nothing when either weight is zero. Measured over the minimum spanning tree of the last 253 observations of the 20-asset sample in `test/assets/SP500.csv.gz` at a two-hop budget and equal weights, the code and the formula agree exactly.

# Algorithm

 1. Take the outer product of `w` with itself and its absolute value, giving `aw`, the gross weight of every pair.
 2. Take the dot product of `X` and `aw`, which adds up the gross weight of the related pairs alone.
 3. Divide by the sum of `aw`, the gross weight of every pair, giving the share invested in related assets.

# Arguments

  - `w`: Weights vector.
  - `X`: Phylogeny matrix.

# Returns

  - `p::Number`: Asset phylogeny score.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`asset_phylogeny`](@ref)
"""
function asset_phylogeny(w::VecNum, X::MatNum)
    aw = abs.(w * transpose(w))
    c = LinearAlgebra.dot(X, aw)
    c /= sum(aw)
    return c
end
"""
    asset_phylogeny(pl::PhylogenyResult{<:MatNum}, w::VecNum, args...;
                    kwargs...)

Compute the asset phylogeny score for a set of portfolio weights and a phylogeny matrix result, forwarding additional arguments.

This method provides compatibility with workflows that pass extra positional or keyword arguments. It extracts the phylogeny matrix from the `PhylogenyResult` and delegates to `asset_phylogeny(w, pl)`, ignoring any additional arguments.

# Arguments

  - `pl::PhylogenyResult{<:MatNum}`: Phylogeny matrix result object.
  - `w::VecNum`: Portfolio weights vector.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

# Returns

  - `score::Number`: Asset phylogeny score.

# Related

  - [`PhylogenyResult`](@ref)
  - [`asset_phylogeny`](@ref)
"""
function asset_phylogeny(pl::PhylogenyResult{<:MatNum}, w::VecNum, args...; kwargs...)
    return asset_phylogeny(w, pl.X)
end
"""
    asset_phylogeny(cle::NwE_ClE_Cl,
                    w::VecNum, X::MatNum; dims::Int = 1, kwargs...)

Compute the asset phylogeny score for a set of weights and a network or clustering estimator.

This function computes the phylogeny matrix using the estimator and data, then computes the asset phylogeny score using the weights.

# Algorithm

 1. Build the phylogeny matrix from `X` with [`phylogeny_matrix`](@ref).
 2. Score `w` against that matrix with [`asset_phylogeny`](@ref).

# Arguments

  - `cle`: NetworkEstimator or clustering estimator.
  - `w`: Weights vector.
  - `X`: Data matrix (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments.

# Returns

  - `p::Number`: Asset phylogeny score.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`asset_phylogeny`](@ref)
"""
function asset_phylogeny(cle::NwE_ClE_Cl, w::VecNum, X::MatNum; dims::Int = 1, kwargs...)
    return asset_phylogeny(phylogeny_matrix(cle, X; dims = dims, kwargs...), w)
end

export asset_phylogeny
