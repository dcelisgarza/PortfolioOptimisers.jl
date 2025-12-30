"""
    abstract type AbstractPhylogenyEstimator <: AbstractEstimator end

Abstract supertype for all phylogeny estimator types in PortfolioOptimisers.jl.

All concrete types implementing phylogeny-based estimation algorithms should subtype `AbstractPhylogenyEstimator`. This enables a consistent interface for phylogeny estimators throughout the package.

# Related

  - [`AbstractPhylogenyAlgorithm`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyEstimator <: AbstractEstimator end
"""
    abstract type AbstractPhylogenyAlgorithm <: AbstractAlgorithm end

Abstract supertype for all phylogeny algorithm types in PortfolioOptimisers.jl.

All concrete types implementing specific phylogeny algorithms should subtype `AbstractPhylogenyAlgorithm`. This enables flexible extension and dispatch of phylogeny routines.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyResult`](@ref)
"""
abstract type AbstractPhylogenyAlgorithm <: AbstractAlgorithm end
"""
    abstract type AbstractPhylogenyResult <: AbstractResult end

Abstract supertype for all phylogeny result types in PortfolioOptimisers.jl.

All concrete types representing the result of a phylogeny estimation should subtype `AbstractPhylogenyResult`. This enables a consistent interface for phylogeny results throughout the package.

# Related

  - [`AbstractPhylogenyEstimator`](@ref)
  - [`AbstractPhylogenyAlgorithm`](@ref)
"""
abstract type AbstractPhylogenyResult <: AbstractResult end
