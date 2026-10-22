"""
$(DocStringExtensions.TYPEDEF)

Carries a validated phylogeny matrix or a centrality vector.

`PhylogenyResult` holds one of two arrays. [`phylogeny_matrix`](@ref) returns a phylogeny matrix, with `1` for a pair of related assets and `0` for every other pair. [`centrality_vector`](@ref) returns a vector with one centrality score for each asset.

[`SemiDefinitePhylogeny`](@ref), [`IntegerPhylogeny`](@ref), [`centrality_vector`](@ref) and [`asset_phylogeny`](@ref) can take a `PhylogenyResult` in place of the estimator that makes it. The result holds the assets of the call that made it. A call on a different set of assets reads the same array.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PhylogenyResult(;
        X::ArrNum
    ) -> PhylogenyResult

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:Xe])
  - $(val_dict[:phX_Xv])

# Examples

```jldoctest
julia> PhylogenyResult(; X = [0 1 0; 1 0 1; 0 1 0])
PhylogenyResult
  X ┴ 3×3 Matrix{Int64}

julia> PhylogenyResult(; X = [0.2, 0.5, 0.3])
PhylogenyResult
  X ┴ Vector{Float64}: [0.2, 0.5, 0.3]
```

# Related

  - [`AbstractPhylogenyResult`](@ref)
  - [`phylogeny_matrix`](@ref)
  - [`centrality_vector`](@ref)
"""
@concrete struct PhylogenyResult <: AbstractPhylogenyResult
    """
    $(field_dict[:phX_Xv])
    """
    X
    function PhylogenyResult(X::ArrNum)
        @argcheck(!isempty(X), IsEmptyError)
        if isa(X, MatNum)
            @argcheck(LinearAlgebra.issymmetric(X) && all(iszero, LinearAlgebra.diag(X)),
                      ArgumentError("a phylogeny matrix must be symmetric with a zero diagonal. Got a $(ifelse(LinearAlgebra.issymmetric(X), "symmetric", "non-symmetric")) $(ifelse(all(iszero, LinearAlgebra.diag(X)), "zero diagonal", "non-zero diagonal")) matrix."))
        end
        return new{typeof(X)}(X)
    end
end
function PhylogenyResult(; X::ArrNum)::PhylogenyResult
    return PhylogenyResult(X)
end
"""
    phylogeny_matrix(plr::PhylogenyResult{<:MatNum}, args...; kwargs...)

Return a precomputed phylogeny matrix unchanged.

A [`PhylogenyResult`](@ref) that holds a matrix already holds the answer, so this method builds nothing and returns `plr`. The constructor of `plr` checked the matrix, and this method does not check it again. The method reads no data, so the matrix keeps the size of the set of assets that made it.

# Arguments

  - `plr::PhylogenyResult{<:MatNum}`: Precomputed phylogeny matrix.
  - `args...`: Positional arguments. The method does not read them.
  - `kwargs...`: Keyword arguments. The method does not read them.

# Returns

  - The input `plr` object.

# Examples

```jldoctest
julia> plr = PhylogenyResult(; X = [0 1 0; 1 0 1; 0 1 0]);

julia> phylogeny_matrix(plr)
PhylogenyResult
  X ┴ 3×3 Matrix{Int64}
```

# Related

  - [`PhylogenyResult`](@ref)
  - [`phylogeny_matrix`](@ref)
"""
function phylogeny_matrix(plr::PhylogenyResult{<:MatNum}, args...; kwargs...)
    return plr
end
"""
    centrality_vector(plr::PhylogenyResult{<:VecNum}, args...; kwargs...)

Return a precomputed centrality vector unchanged.

A [`PhylogenyResult`](@ref) that holds a vector already holds the answer, so this method builds nothing and returns `plr`. It does not read a centrality algorithm, so an algorithm in `args` does not change the scores. The vector keeps the length of the set of assets that made it.

# Arguments

  - `plr::PhylogenyResult{<:VecNum}`: Precomputed centrality vector.
  - `args...`: Positional arguments. The method does not read them.
  - `kwargs...`: Keyword arguments. The method does not read them.

# Returns

  - The input `plr` object.

# Examples

```jldoctest
julia> plr = PhylogenyResult(; X = [0.2, 0.5, 0.3]);

julia> centrality_vector(plr)
PhylogenyResult
  X ┴ Vector{Float64}: [0.2, 0.5, 0.3]
```

# Related

  - [`PhylogenyResult`](@ref)
  - [`centrality_vector`](@ref)
"""
function centrality_vector(plr::PhylogenyResult{<:VecNum}, args...; kwargs...)
    return plr
end

export PhylogenyResult
