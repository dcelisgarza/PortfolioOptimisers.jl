"""
    struct EmpiricalPrior{T1, T2, T3} <: AbstractLowOrderPriorEstimator_A
        ce::T1
        me::T2
        horizon::T3
    end

Empirical prior estimator for asset returns.

`EmpiricalPrior` is a low order prior estimator that computes the mean and covariance of asset returns using empirical (sample-based) statistics. It supports custom expected returns and covariance estimators, as well as an optional investment horizon for log-normalisation and scaling.

# Fields

  - `ce`: Covariance estimator.
  - `me`: Expected returns estimator.
  - `horizon`: Optional investment horizon.

# Constructor

    EmpiricalPrior(; ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                   me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                   horizon::Union{Nothing, <:Number} = nothing)

Keyword arguments correspond to the fields above.

## Validation

  - If `horizon` is not `nothing`, `horizon > 0`.

# Examples

```jldoctest
julia> EmpiricalPrior()
EmpiricalPrior
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
       me ┼ SimpleExpectedReturns
          │   w ┴ nothing
  horizon ┴ nothing
```

# Related

  - [`AbstractLowOrderPriorEstimator_A`](@ref)
  - [`StatsBase.CovarianceEstimator`](https://juliastats.org/StatsBase.jl/stable/cov/)
  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`SimpleExpectedReturns`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`prior`](@ref)
"""
struct EmpiricalPrior{T1, T2, T3} <: AbstractLowOrderPriorEstimator_A
    ce::T1
    me::T2
    horizon::T3
    function EmpiricalPrior(ce::StatsBase.CovarianceEstimator,
                            me::AbstractExpectedReturnsEstimator,
                            horizon::Union{Nothing, <:Number})
        if !isnothing(horizon)
            @argcheck(horizon > 0)
        end
        return new{typeof(ce), typeof(me), typeof(horizon)}(ce, me, horizon)
    end
end
function EmpiricalPrior(;
                        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        horizon::Union{Nothing, <:Number} = nothing)
    return EmpiricalPrior(ce, me, horizon)
end
function factory(pe::EmpiricalPrior, w::WeightsType = nothing)
    return EmpiricalPrior(; me = factory(pe.me, w), ce = factory(pe.ce, w),
                          horizon = pe.horizon)
end
"""
    prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing}, X::NumMat, args...; dims::Int = 1,
          kwargs...)

Compute empirical prior moments for asset returns (no horizon adjustment).

`prior` estimates the mean and covariance of asset returns using the specified empirical prior estimator, without log-normalisation or scaling for investment horizon. The mean and covariance are computed using the estimators stored in `pe`, and returned in a [`LowOrderPrior`](@ref) result.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `args...`: Additional positional arguments (ignored).
  - `dims`: Dimension along which to compute moments.
  - `kwargs...`: Additional keyword arguments passed to mean and covariance estimators.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, mean vector, and covariance matrix.

# Validation

  - `dims in (1, 2)`.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing}, X::NumMat, args...; dims::Int = 1,
               kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    mu = vec(mean(pe.me, X; kwargs...))
    sigma = cov(pe.ce, X; kwargs...)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma)
end
"""
    prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number}, X::NumMat, args...; dims::Int = 1,
          kwargs...)

Compute empirical prior moments for asset returns with investment horizon adjustment.

`prior` estimates the mean and covariance of asset returns using the specified empirical prior estimator, applying log-normalisation and scaling for the investment horizon. The asset returns are log-transformed, moments are computed using the estimators stored in `pe`, and then rescaled according to the investment horizon. The final mean and covariance are transformed back to arithmetic returns and returned in a [`LowOrderPrior`](@ref) result.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `args...`: Additional positional arguments (ignored).
  - `dims`: Dimension along which to compute moments.
  - `kwargs...`: Additional keyword arguments passed to mean and covariance estimators.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, mean vector, and covariance matrix.

# Validation

  - `dims in (1, 2)`.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number}, X::NumMat, args...;
               dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2))
    if dims == 2
        X = transpose(X)
    end
    X_log = log1p.(X)
    mu = vec(mean(pe.me, X_log; kwargs...))
    sigma = cov(pe.ce, X_log; kwargs...)
    mu .*= pe.horizon
    sigma .*= pe.horizon
    mu .= exp.(mu + 0.5 * diag(sigma))
    sigma .= (mu ⊗ mu) ⊙ (exp.(sigma) .- one(eltype(sigma)))
    mu .-= one(eltype(mu))
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma)
end

export EmpiricalPrior
