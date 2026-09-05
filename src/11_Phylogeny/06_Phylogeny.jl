"""
$(DocStringExtensions.TYPEDEF)

Carries a validated phylogeny matrix or a centrality vector.

`PhylogenyResult` stores the output of phylogeny-based estimation routines, such as network or clustering-based phylogeny matrices, or centrality vectors. It is used throughout the package to represent validated phylogeny structures for constraint generation, centrality analysis, and related workflows.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PhylogenyResult(;
        X::ArrNum
    ) -> PhylogenyResult

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:Xe]).
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
                      ArgumentError("phylogeny needs a distance matrix (symmetric, zero diagonal). Got a $(ifelse(LinearAlgebra.issymmetric(X), "symmetric", "non-symmetric")) $(ifelse(all(iszero, LinearAlgebra.diag(X)), "zero diagonal", "non-zero diagonal")) matrix."))
        end
        return new{typeof(X)}(X)
    end
end
function PhylogenyResult(; X::ArrNum)::PhylogenyResult
    return PhylogenyResult(X)
end
"""
    phylogeny_matrix(plr::PhylogenyResult{<:MatNum}, args...; kwargs...)

Fallback no-op for returning a validated phylogeny matrix result as-is.

This method provides a generic interface for handling precomputed phylogeny matrices wrapped in a [`PhylogenyResult`](@ref). It simply returns the input object unchanged, enabling consistent downstream workflows for constraint generation and analysis.

# Arguments

  - `plr::PhylogenyResult{<:MatNum}`: Phylogeny matrix result object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

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

Fallback no-op for returning a validated centrality vector result as-is.

This method provides a generic interface for handling precomputed centrality vectors wrapped in a [`PhylogenyResult`](@ref). It simply returns the input object unchanged, enabling consistent downstream workflows for centrality-based analysis and constraint generation.

# Arguments

  - `plr::PhylogenyResult{<:VecNum}`: Centrality vector result object.
  - `args...`: Additional positional arguments (ignored).
  - `kwargs...`: Additional keyword arguments (ignored).

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
