"""
    abstract type AbstractPhylogenyEstimator <: AbstractEstimator end

Abstract supertype for all phylogeny estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing phylogeny-based estimation algorithms should be subtypes of `AbstractPhylogenyEstimator`.

# Related

  - [`AbstractPhylogenyAlgorithm`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyEstimator <: AbstractEstimator end
"""
    abstract type AbstractPhylogenyAlgorithm <: AbstractAlgorithm end

Abstract supertype for all phylogeny algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing specific phylogeny algorithms should be subtypes of `AbstractPhylogenyAlgorithm`.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyAlgorithm <: AbstractAlgorithm end
"""
    abstract type AbstractPhylogenyResult <: AbstractResult end

Abstract supertype for all phylogeny result types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing the result of a phylogeny estimation should be subtypes of `AbstractPhylogenyResult`.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyAlgorithm`](@ref)
"""
abstract type AbstractPhylogenyResult <: AbstractResult end
const PlE_Pl = Union{<:AbstractPhylogenyEstimator, <:AbstractPhylogenyResult}
function factory(pl::PlE_Pl, args...)
    return pl
end
function factory(alg::AbstractPhylogenyAlgorithm, args...)
    return alg
end
