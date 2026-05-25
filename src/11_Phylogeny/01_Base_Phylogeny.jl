"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny estimator types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing phylogeny-based estimation algorithms should be subtypes of `AbstractPhylogenyEstimator`.

# Related

  - [`AbstractPhylogenyAlgorithm`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny algorithm types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types implementing specific phylogeny algorithms should be subtypes of `AbstractPhylogenyAlgorithm`.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all phylogeny result types in `PortfolioOptimisers.jl`.

All concrete and/or abstract types representing the result of a phylogeny estimation should be subtypes of `AbstractPhylogenyResult`.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyAlgorithm`](@ref)
"""
abstract type AbstractPhylogenyResult <: AbstractResult end
"""
    const PlE_Pl = Union{<:AbstractPhylogenyEstimator, <:AbstractPhylogenyResult}

Alias for a phylogeny estimator or result.

Matches either an [`AbstractPhylogenyEstimator`](@ref) or an [`AbstractPhylogenyResult`](@ref). Used internally for dispatch when either a phylogeny estimation configuration or pre-computed result is accepted.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
const PlE_Pl = Union{<:AbstractPhylogenyEstimator, <:AbstractPhylogenyResult}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the phylogeny estimator or result `pl` unchanged.

Identity pass-through used when a phylogeny estimator or pre-computed result is provided in a context that calls [`factory`](@ref).

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
  - [`factory`](@ref)
"""
function factory(pl::PlE_Pl, args...; kwargs...)::PlE_Pl
    return pl
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the phylogeny algorithm `alg` unchanged.

Identity pass-through used when a phylogeny algorithm is provided in a context that calls [`factory`](@ref).

# Related

  - [`AbstractPhylogenyAlgorithm`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::AbstractPhylogenyAlgorithm, args...;
                 kwargs...)::AbstractPhylogenyAlgorithm
    return alg
end
