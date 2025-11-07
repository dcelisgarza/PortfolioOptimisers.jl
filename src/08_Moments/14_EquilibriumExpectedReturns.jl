"""
    struct EquilibriumExpectedReturns{T1, T2, T3} <: AbstractShrunkExpectedReturnsEstimator
        ce::T1
        w::T2
        l::T3
    end

Container type for equilibrium expected returns estimators.

`EquilibriumExpectedReturns` encapsulates the covariance estimator, equilibrium weights, and risk aversion parameter for computing equilibrium expected returns (e.g., as in Black-Litterman). This enables modular workflows for reverse optimization and equilibrium-based return estimation.

# Fields

  - `ce`: Covariance estimator.
  - `w`: Equilibrium portfolio weights. If `nothing`, uses equal weights.
  - `l`: Risk aversion parameter.

# Constructor

    EquilibriumExpectedReturns(;
                               ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                               w::Union{Nothing, <:NumVec} = nothing, l::Number = 1)

Keyword arguments correspond to the fields above.

## Validation

  - If `w` is provided, `!isempty(w)`.

# Examples

```jldoctest
julia> EquilibriumExpectedReturns()
EquilibriumExpectedReturns
  ce ┼ PortfolioOptimisersCovariance
     │   ce ┼ Covariance
     │      │    me ┼ SimpleExpectedReturns
     │      │       │   w ┴ nothing
     │      │    ce ┼ GeneralCovariance
     │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │      │       │    w ┴ nothing
     │      │   alg ┴ Full()
     │   mp ┼ DefaultMatrixProcessing
     │      │       pdm ┼ Posdef
     │      │           │   alg ┴ UnionAll: NearestCorrelationMatrix.Newton
     │      │   denoise ┼ nothing
     │      │    detone ┼ nothing
     │      │       alg ┴ nothing
   w ┼ nothing
   l ┴ Int64: 1
```

# Related

  - [`AbstractShrunkExpectedReturnsEstimator`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
  - [`StatsBase.AbstractWeights`](https://juliastats.org/StatsBase.jl/stable/weights/)
"""
struct EquilibriumExpectedReturns{T1, T2, T3} <: AbstractShrunkExpectedReturnsEstimator
    ce::T1
    w::T2
    l::T3
    function EquilibriumExpectedReturns(ce::StatsBase.CovarianceEstimator,
                                        w::Union{Nothing, <:NumVec}, l::Number)
        assert_nonempty_finite_val(w, :w)
        return new{typeof(ce), typeof(w), typeof(l)}(ce, w, l)
    end
end
function EquilibriumExpectedReturns(;
                                    ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                                    w::Union{Nothing, <:NumVec} = nothing, l::Number = 1)
    return EquilibriumExpectedReturns(ce, w, l)
end
function factory(ce::EquilibriumExpectedReturns, w::WeightsType = nothing)
    return EquilibriumExpectedReturns(; ce = factory(ce.ce, w), w = ce.w, l = ce.l)
end
"""
    mean(me::EquilibriumExpectedReturns, X::NumMat; dims::Int = 1, kwargs...)

Compute equilibrium expected returns from a covariance estimator, weights, and risk aversion.

This method computes equilibrium expected returns as `λ * Σ * w`, where `λ` is the risk aversion parameter, `Σ` is the covariance matrix, and `w` are the equilibrium weights. If `w` is not provided in the estimator, equal weights are used.

# Arguments

  - `me`: Equilibrium expected returns estimator.
  - `X`: Data matrix (observations × assets).
  - `dims`: Dimension along which to compute the covariance.
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `mu::NumArr`: Equilibrium expected returns vector.

# Related

  - [`EquilibriumExpectedReturns`](@ref)
"""
function Statistics.mean(me::EquilibriumExpectedReturns, X::NumMat; dims::Int = 1,
                         kwargs...)
    sigma = cov(me.ce, X; dims = dims, kwargs...)
    w = !isnothing(me.w) ? me.w : fill(inv(size(sigma, 1)), size(sigma, 1))
    return me.l * sigma * w
end

export EquilibriumExpectedReturns
