"""
    struct EquilibriumExpectedReturns{T1 <: StatsBase.CovarianceEstimator,
                                      T2 <: Union{Nothing, <:AbstractWeights}, T3 <: Real} <: AbstractShrunkExpectedReturnsEstimator
        ce::T1
        w::T2
        l::T3
    end

Container type for equilibrium expected returns estimators.

`EquilibriumExpectedReturns` encapsulates the covariance estimator, equilibrium weights, and risk aversion parameter for computing equilibrium expected returns (e.g., as in Black-Litterman). This enables modular workflows for reverse optimization and equilibrium-based return estimation.

# Fields

  - `ce::StatsBase.CovarianceEstimator`: Covariance estimator.
  - `w::Union{Nothing, <:AbstractWeights}`: Equilibrium portfolio weights. If `nothing`, uses equal weights.
  - `l::Real`: Risk aversion parameter.

# Constructor

    EquilibriumExpectedReturns(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               w::Union{Nothing, <:AbstractWeights} = nothing,
                               l::Real = 1.0)

Construct an `EquilibriumExpectedReturns` estimator with the specified covariance estimator, weights, and risk aversion.

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
"""
struct EquilibriumExpectedReturns{T1 <: StatsBase.CovarianceEstimator,
                                  T2 <: Union{Nothing, <:AbstractWeights}, T3 <: Real} <:
       AbstractShrunkExpectedReturnsEstimator
    ce::T1
    w::T2
    l::T3
end
"""
    EquilibriumExpectedReturns(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               w::Union{Nothing, <:AbstractWeights} = nothing,
                               l::Real = 1.0)

Construct an [`EquilibriumExpectedReturns`](@ref) estimator for equilibrium-based expected returns.

# Arguments

  - `ce::StatsBase.CovarianceEstimator`: Covariance estimator.

  - `w::Union{Nothing, <:AbstractWeights}`: Equilibrium portfolio weights. If `nothing`, uses equal weights.
  - `l::Real`: Risk aversion parameter.

# Returns

  - `EquilibriumExpectedReturns`: Configured equilibrium expected returns estimator.

# Examples

```jldoctest
julia> EquilibriumExpectedReturns()
EquilibriumExpectedReturns
  ce | PortfolioOptimisersCovariance
     |   ce | Covariance
     |      |    me | SimpleExpectedReturns
     |      |       |   w | nothing
     |      |    ce | GeneralWeightedCovariance
     |      |       |   ce | StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     |      |       |    w | nothing
     |      |   alg | Full()
     |   mp | DefaultMatrixProcessing
     |      |       pdm | PosdefEstimator
     |      |           |   alg | UnionAll: NearestCorrelationMatrix.Newton
     |      |   denoise | nothing
     |      |    detone | nothing
     |      |       alg | nothing
   w | nothing
   l | Float64: 1.0
```

# Related

  - [`EquilibriumExpectedReturns`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
"""
function EquilibriumExpectedReturns(;
                                    ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                    w::Union{Nothing, <:AbstractWeights} = nothing,
                                    l::Real = 1.0)
    if isa(w, AbstractWeights)
        @smart_assert(!isempty(w))
    end
    return EquilibriumExpectedReturns{typeof(ce), typeof(w), typeof(l)}(ce, w, l)
end

"""
    mean(me::EquilibriumExpectedReturns, X::AbstractArray; dims::Int = 1, kwargs...)

Compute equilibrium expected returns from a covariance estimator, weights, and risk aversion.

This method computes equilibrium expected returns as `λ * Σ * w`, where `λ` is the risk aversion parameter, `Σ` is the covariance matrix, and `w` are the equilibrium weights. If `w` is not provided, equal weights are used.

# Arguments

  - `me::EquilibriumExpectedReturns`: Equilibrium expected returns estimator.
  - `X::AbstractArray`: Data matrix (observations × assets).
  - `dims::Int`: Dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `mu::AbstractArray`: Equilibrium expected returns vector.

# Related

  - [`EquilibriumExpectedReturns`](@ref)
"""
function Statistics.mean(me::EquilibriumExpectedReturns, X::AbstractArray; dims::Int = 1,
                         kwargs...)
    sigma = cov(me.ce, X; dims = dims, kwargs...)
    w = !isnothing(me.w) ? me.w : fill(inv(size(sigma, 1)), size(sigma, 1))
    return me.l * sigma * w
end
function factory(ce::EquilibriumExpectedReturns, args...)
    return ce
end

export EquilibriumExpectedReturns
