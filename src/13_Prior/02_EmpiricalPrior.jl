"""
$(DocStringExtensions.TYPEDEF)

Empirical prior estimator for asset returns.

`EmpiricalPrior` is a low order prior estimator that computes the mean and covariance of asset returns using empirical (sample-based) statistics. It supports custom expected returns and covariance estimators, as well as an optional investment horizon for log-normalisation and scaling.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EmpiricalPrior(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
        horizon::Option{<:Number} = nothing
    ) -> EmpiricalPrior

Keywords correspond to the struct's fields.

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
          │      │   alg ┴ FullMoment()
          │   mp ┼ MatrixProcessing
          │      │     pdm ┼ Posdef
          │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
          │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
          │      │      dn ┼ nothing
          │      │      dt ┼ nothing
          │      │     alg ┼ nothing
          │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
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
@propagatable @concrete struct EmpiricalPrior <: AbstractLowOrderPriorEstimator_A
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    """
    $(field_dict[:me])
    """
    @fprop @vprop me
    """
    $(field_dict[:horizon])
    """
    horizon
    function EmpiricalPrior(ce::StatsBase.CovarianceEstimator,
                            me::AbstractExpectedReturnsEstimator, horizon::Option{<:Number})
        if !isnothing(horizon)
            @argcheck(horizon > 0, DomainError(horizon, "horizon must be > 0"))
        end
        return new{typeof(ce), typeof(me), typeof(horizon)}(ce, me, horizon)
    end
end
function EmpiricalPrior(;
                        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance(),
                        me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns(),
                        horizon::Option{<:Number} = nothing)::EmpiricalPrior
    return EmpiricalPrior(ce, me, horizon)
end
"""
    prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing}, X::MatNum, args...; dims::Int = 1,
          kwargs...)

Compute empirical prior moments for asset returns (no horizon adjustment).

`prior` estimates the mean and covariance of asset returns using the specified empirical prior estimator, without log-normalisation or scaling for investment horizon. The mean and covariance are computed using the estimators stored in `pe`, and returned in a [`LowOrderPrior`](@ref) result.

# Mathematical definition

The empirical prior directly estimates first and second moments from the sample:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\frac{1}{T} \\sum_{t=1}^{T} \\boldsymbol{x}_t\\,, \\\\
\\hat{\\mathbf{\\Sigma}} &= \\frac{1}{T-1} \\sum_{t=1}^{T} (\\boldsymbol{x}_t - \\hat{\\boldsymbol{\\mu}})(\\boldsymbol{x}_t - \\hat{\\boldsymbol{\\mu}})^\\intercal\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` sample mean vector.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` sample covariance matrix.
  - ``\\boldsymbol{x}_t``: ``N \\times 1`` vector of asset returns at time ``t``.
  - $(math_dict[:T])

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `args...`: Additional positional arguments (ignored).
  - $(arg_dict[:dims])
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
function prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing}, X::MatNum, args...; dims::Int = 1,
               kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
    if dims == 2
        X = transpose(X)
    end
    mu = vec(Statistics.mean(pe.me, X; kwargs...))
    sigma = Statistics.cov(pe.ce, X; kwargs...)
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma)
end
"""
    prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number}, X::MatNum, args...; dims::Int = 1,
          kwargs...)

Compute empirical prior moments for asset returns with investment horizon adjustment.

`prior` estimates the mean and covariance of asset returns using the specified empirical prior estimator, applying log-normalisation and scaling for the investment horizon. The asset returns are log-transformed, moments are computed using the estimators stored in `pe`, and then rescaled according to the investment horizon. The final mean and covariance are transformed back to arithmetic returns and returned in a [`LowOrderPrior`](@ref) result.

# Mathematical definition

Log-returns are computed and scaled by the investment horizon ``h``, then converted back to arithmetic returns:

```math
\\begin{align}
\\tilde{\\boldsymbol{\\mu}} &= h \\cdot \\hat{\\boldsymbol{\\mu}}_{\\log}\\,, \\\\
\\tilde{\\mathbf{\\Sigma}} &= h \\cdot \\hat{\\mathbf{\\Sigma}}_{\\log}\\,.
\\end{align}
```

```math
\\begin{align}
\\hat{\\mu}_i &= \\exp\\!\\left(\\tilde{\\mu}_i + \\tfrac{1}{2}\\tilde{\\sigma}_{ii}\\right) - 1\\,, \\\\
\\hat{\\sigma}_{ij} &= (\\hat{\\mu}_i + 1)(\\hat{\\mu}_j + 1)\\left(\\exp(\\tilde{\\sigma}_{ij}) - 1\\right)\\,.
\\end{align}
```

Where:

  - ``\\tilde{\\boldsymbol{\\mu}}``, ``\\tilde{\\mathbf{\\Sigma}}``: Horizon-scaled log-return mean and covariance.
  - ``h``: Investment horizon.
  - ``\\hat{\\boldsymbol{\\mu}}_{\\log}``, ``\\hat{\\mathbf{\\Sigma}}_{\\log}``: Sample mean and covariance of log-returns ``\\log(1 + x_t)``.
  - ``\\hat{\\mu}_i``: Arithmetic mean return for asset ``i``.
  - ``\\hat{\\sigma}_{ij}``: Arithmetic covariance between assets ``i`` and ``j``.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `args...`: Additional positional arguments (ignored).
  - $(arg_dict[:dims])
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
function prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number}, X::MatNum, args...;
               dims::Int = 1, kwargs...)
    @argcheck(dims in (1, 2), DomainError(dims, "dims must be 1 or 2"))
    if dims == 2
        X = transpose(X)
    end
    X_log = log1p.(X)
    mu = vec(Statistics.mean(pe.me, X_log; kwargs...))
    sigma = Statistics.cov(pe.ce, X_log; kwargs...)
    mu .*= pe.horizon
    sigma .*= pe.horizon
    mu .= exp.(mu + 0.5 * LinearAlgebra.diag(sigma))
    sigma .= (mu ⊗ mu) ⊙ (exp.(sigma) .- one(eltype(sigma)))
    mu .-= one(eltype(mu))
    return LowOrderPrior(; X = X, mu = mu, sigma = sigma)
end

export EmpiricalPrior
