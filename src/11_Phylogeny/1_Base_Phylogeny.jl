"""
```julia
abstract type AbstractPhylogenyEstimator <: AbstractEstimator end
```

Abstract supertype for all phylogeny estimator types in PortfolioOptimisers.jl.

All concrete types implementing phylogeny-based estimation algorithms should subtype `AbstractPhylogenyEstimator`. This enables a consistent interface for phylogeny estimators throughout the package.

# Related

  - [`AbstractPhylogenyAlgorithm`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyEstimator <: AbstractEstimator end

"""
```julia
abstract type AbstractPhylogenyAlgorithm <: AbstractAlgorithm end
```

Abstract supertype for all phylogeny algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific phylogeny algorithms should subtype `AbstractPhylogenyAlgorithm`. This enables flexible extension and dispatch of phylogeny routines.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyAlgorithm <: AbstractAlgorithm end

"""
```julia
abstract type AbstractPhylogenyResult <: AbstractResult end
```

Abstract supertype for all phylogeny result types in PortfolioOptimisers.jl.

All concrete types representing the result of a phylogeny estimation should subtype `AbstractPhylogenyResult`. This enables a consistent interface for phylogeny results throughout the package.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyAlgorithm`](@ref)
"""
abstract type AbstractPhylogenyResult <: AbstractResult end

struct PhylogenyResult{T} <: AbstractPhylogenyResult
    X::T
end
function PhylogenyResult(; X::Union{<:AbstractMatrix, <:AbstractVector})
    @argcheck(!isempty(X))
    if isa(X, AbstractMatrix)
        @argcheck(issymmetric(X))
        @argcheck(all(x -> iszero(x), diag(X)))
    end
    return PhylogenyResult(X)
end
function phylogeny_matrix(ph::PhylogenyResult{<:AbstractMatrix}, args...; kwargs...)
    return ph
end
function centrality_vector(ph::PhylogenyResult{<:AbstractVector}, args...; kwargs...)
    return ph
end

export PhylogenyResult
