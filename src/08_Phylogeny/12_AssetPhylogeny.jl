"""
    asset_phylogeny(w::VecNum, X::MatNum)

Compute the asset phylogeny score of the weights `w` on the phylogeny matrix `X`.

The score is the share of the gross weight of all pairs that the related pairs hold. It is the percentage invested in connected assets of Equation 13.7 of [cajas2025](@cite),

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

A pair of assets that are not related adds nothing, and a pair adds nothing when either weight is zero. A `w` of zeros gives `NaN`, because the denominator is zero.

# Algorithm

 1. Take the absolute value of the outer product of `w` with itself. The result `aw` is the gross weight of every pair.
 2. Take the dot product of `X` and `aw`, which adds up the gross weight of the related pairs.
 3. Divide by the sum of `aw`, and return the share invested in related assets.

# Arguments

  - `w`: Portfolio weight vector.
  - `X`: Phylogeny matrix.

# Returns

  - `p::Number`: Asset phylogeny score.

# Related

  - [`phylogeny_matrix`](@ref)
  - [`asset_phylogeny`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 13.1.5.2, Equation 13.7.
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

Compute the asset phylogeny score of the weights `w` on a precomputed phylogeny matrix.

The method calls `asset_phylogeny(w, pl.X)`. It accepts `args` and `kwargs` and does not read them, so a caller can pass a precomputed result where it passes an estimator and its data.

# Arguments

  - `pl::PhylogenyResult{<:MatNum}`: Precomputed phylogeny matrix.
  - `w::VecNum`: Portfolio weight vector.
  - `args...`: Positional arguments. The method does not read them.
  - `kwargs...`: Keyword arguments. The method does not read them.

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

Compute the asset phylogeny score of the weights `w` on the phylogeny matrix that `cle` builds from `X`.

# Algorithm

 1. Build the phylogeny matrix from `X` with [`phylogeny_matrix`](@ref).
 2. Score `w` against that matrix with [`asset_phylogeny`](@ref).

# Arguments

  - `cle`: Network estimator, clustering estimator, or clustering result.
  - `w`: Portfolio weight vector.
  - $(arg_dict[:X])
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
