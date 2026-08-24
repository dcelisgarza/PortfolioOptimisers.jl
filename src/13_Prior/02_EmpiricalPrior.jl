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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).
  - `me`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).
  - `me`: Recursively viewed via [`port_opt_view`](@ref).

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
          │      │   alg ┼ FullMoment()
          │      │     w ┴ nothing
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
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 3.1.
  - $(ref_dict[:meucci2005]) Chapter 3.
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

`pe.me` computes the mean and `pe.ce` computes the covariance, so both moments are whatever those estimators return. Under the default pair — [`SimpleExpectedReturns`](@ref) and [`PortfolioOptimisersCovariance`](@ref) with no observation weights — they reduce to the sample moments:

```math
\\begin{align}
\\hat{\\boldsymbol{\\mu}} &= \\frac{1}{T} \\sum_{t=1}^{T} \\boldsymbol{x}_t\\,, \\\\
\\hat{\\mathbf{\\Sigma}} &= \\frac{1}{T-1} \\sum_{t=1}^{T} (\\boldsymbol{x}_t - \\hat{\\boldsymbol{\\mu}})(\\boldsymbol{x}_t - \\hat{\\boldsymbol{\\mu}})^\\intercal\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{\\mu}}``: ``N \\times 1`` mean vector.
  - ``\\hat{\\mathbf{\\Sigma}}``: ``N \\times N`` covariance matrix.
  - ``\\boldsymbol{x}_t``: ``N \\times 1`` vector of asset returns at time ``t``.
  - $(math_dict[:T])

Every choice inside `pe.me` and `pe.ce` reaches the result. A shrunk mean and a denoised covariance move both away from the display above rather than refining it.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `args...`: Additional positional arguments (ignored).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to mean and covariance estimators.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, mean vector, and covariance matrix.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::EmpiricalPrior{<:Any, <:Any, Nothing}, X::MatNum, args...; dims::Int = 1,
               kwargs...)
    X = dims_oriented(dims, X)
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

`pe.me` and `pe.ce` are applied to the **log-returns** ``\\log(1 + x_t)`` rather than to `X` itself. The two log-moments are scaled by the investment horizon ``h``, then converted back to arithmetic returns:

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
  - ``\\hat{\\boldsymbol{\\mu}}_{\\log}``, ``\\hat{\\mathbf{\\Sigma}}_{\\log}``: Mean and covariance of the log-returns ``\\log(1 + x_t)``, computed by `pe.me` and `pe.ce`.
  - ``\\hat{\\mu}_i``: Arithmetic mean return for asset ``i``.
  - ``\\hat{\\sigma}_{ij}``: Arithmetic covariance between assets ``i`` and ``j``.

`X` in the returned [`LowOrderPrior`](@ref) is the arithmetic returns matrix the caller supplied. Only the moments are computed in log space.

# Arguments

  - `pe`: Empirical prior estimator.
  - `X`: Asset returns matrix (observations × assets).
  - `args...`: Additional positional arguments (ignored).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to mean and covariance estimators.

# Validation

  - `dims in (1, 2)`.

# Returns

  - `pr::LowOrderPrior`: Result object containing asset returns, mean vector, and covariance matrix.

# Related

  - [`EmpiricalPrior`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`prior`](@ref)
"""
function prior(pe::EmpiricalPrior{<:Any, <:Any, <:Number}, X::MatNum, args...;
               dims::Int = 1, kwargs...)
    X = dims_oriented(dims, X)
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

function factor_residual_config(::EmpiricalPrior)
    # An empirical prior estimates the asset covariance directly. There is no factor lift,
    # so there is no residual block to remove (see [`factor_residual_config`](@ref)).
    return nothing
end

export EmpiricalPrior
