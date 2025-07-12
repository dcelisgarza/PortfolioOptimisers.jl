"""
    AbstractEstimator

Abstract supertype for all estimator types in PortfolioOptimisers.jl.

All custom estimators (e.g., for moments, risk, or priors) should subtype `AbstractEstimator`.
This enables a consistent interface for estimation routines throughout the package.

# Related

  - [`AbstractAlgorithm`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractEstimator end

"""
    AbstractAlgorithm

Abstract supertype for all optimization algorithm types in PortfolioOptimisers.jl.

All optimization algorithms (e.g., solvers, metaheuristics) should subtype `AbstractAlgorithm`.
This allows for flexible extension and dispatch of optimization routines.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractResult`](@ref)
"""
abstract type AbstractAlgorithm end

"""
    AbstractResult

Abstract supertype for all result types returned by optimizers in PortfolioOptimisers.jl.

All result objects (e.g., optimization outputs, solution summaries) should subtype `AbstractResult`.
This ensures a unified interface for accessing results across different algorithms.

# Related

  - [`AbstractEstimator`](@ref)
  - [`AbstractAlgorithm`](@ref)
"""
abstract type AbstractResult end
