"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns the asset standard deviations.

`StandardDeviationExpectedReturns` computes "expected returns" as the standard deviation of each asset, as estimated by the underlying covariance estimator. This can be useful in certain risk-based portfolio construction approaches where the expected return proxy is the asset's volatility.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StandardDeviationExpectedReturns(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance()
    ) -> StandardDeviationExpectedReturns

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> StandardDeviationExpectedReturns()
StandardDeviationExpectedReturns
  ce ┼ PortfolioOptimisersCovariance
     │   ce ┼ Covariance
     │      │    me ┼ SimpleExpectedReturns
     │      │       │   w ┴ nothing
     │      │    ce ┼ GeneralCovariance
     │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │      │       │    w ┴ nothing
     │      │   alg ┴ Full()
     │   mp ┼ MatrixProcessing
     │      │     pdm ┼ Posdef
     │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      │      dn ┼ nothing
     │      │      dt ┼ nothing
     │      │     alg ┼ nothing
     │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct StandardDeviationExpectedReturns <:
                               AbstractExpectedReturnsEstimator
    """
    $(field_dict[:ce])
    """
    @fprop ce
    function StandardDeviationExpectedReturns(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
#= Old factory function:
function factory(ce::StandardDeviationExpectedReturns,
                 w::ObsWeights)::StandardDeviationExpectedReturns
    return StandardDeviationExpectedReturns(; ce = factory(ce.ce, w))
end
=#
function StandardDeviationExpectedReturns(;
                                          ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance())::StandardDeviationExpectedReturns
    return StandardDeviationExpectedReturns(ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the expected returns estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:me])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:mev])

# Related

  - [`StandardDeviationExpectedReturns`](@ref)
"""
function port_opt_view(me::StandardDeviationExpectedReturns,
                       i)::StandardDeviationExpectedReturns
    return StandardDeviationExpectedReturns(; me = port_opt_view(me.ce, i))
end
"""
    Statistics.mean(me::StandardDeviationExpectedReturns, X::MatNum;
                    dims::Int = 1, kwargs...)

Compute expected returns as the standard deviation of each asset.

This method returns the standard deviation vector of `X` as estimated by the covariance estimator `me.ce`.

# Mathematical definition

```math
\\begin{align}
\\hat{\\mu}_j &= \\hat{\\sigma}_j = \\sqrt{\\frac{1}{T-c} \\sum_{t=1}^{T} w_t (r_{tj} - \\hat{\\mu}_j^{(0)})^2}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}_j``: Standard deviation of asset ``j``.
  - ``c``: Bias correction (``c = 1`` if `corrected = true`, else ``c = 0``).
  - ``T``: Number of observations.
  - ``w_t``: Observation weights.

# Arguments

  - `me`: Standard deviation expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `mu::Matrix{<:Number}`: Standard deviation vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`StandardDeviationExpectedReturns`](@ref)
"""
function Statistics.mean(me::StandardDeviationExpectedReturns, X::MatNum; dims::Int = 1,
                         kwargs...)
    return Statistics.std(me.ce, X; dims = dims, kwargs...)
end

"""
$(DocStringExtensions.TYPEDEF)

Expected returns estimator that returns the asset variances.

`VarianceExpectedReturns` computes "expected returns" as the variance of each asset, as estimated by the underlying covariance estimator. This can be useful in certain risk-based portfolio construction approaches where the expected return proxy is the asset's variance. Variance is the square of volatility (standard deviation).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    VarianceExpectedReturns(;
        ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance()
    ) -> VarianceExpectedReturns

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).

# Examples

```jldoctest
julia> VarianceExpectedReturns()
VarianceExpectedReturns
  ce ┼ PortfolioOptimisersCovariance
     │   ce ┼ Covariance
     │      │    me ┼ SimpleExpectedReturns
     │      │       │   w ┴ nothing
     │      │    ce ┼ GeneralCovariance
     │      │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │      │       │    w ┴ nothing
     │      │   alg ┴ Full()
     │   mp ┼ MatrixProcessing
     │      │     pdm ┼ Posdef
     │      │         │      alg ┼ UnionAll: NearestCorrelationMatrix.Newton
     │      │         │   kwargs ┴ @NamedTuple{}: NamedTuple()
     │      │      dn ┼ nothing
     │      │      dt ┼ nothing
     │      │     alg ┼ nothing
     │      │   order ┴ NTuple{4, Symbol}: (:pdm, :dn, :dt, :alg)
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`PortfolioOptimisersCovariance`](@ref)
  - [`factory`](@ref)
"""
@propagatable @concrete struct VarianceExpectedReturns <: AbstractExpectedReturnsEstimator
    """
    $(field_dict[:ce])
    """
    @fprop ce
    function VarianceExpectedReturns(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
#= Old factory function:
function factory(ce::VarianceExpectedReturns, w::ObsWeights)::VarianceExpectedReturns
    return VarianceExpectedReturns(; ce = factory(ce.ce, w))
end
=#
function VarianceExpectedReturns(;
                                 ce::StatsBase.CovarianceEstimator = PortfolioOptimisersCovariance())::VarianceExpectedReturns
    return VarianceExpectedReturns(ce)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Gets the view of the expected returns estimator for the `i`-th element(s).

# Arguments

  - $(arg_dict[:me])
  - `i`: Index or indices to view.

# Returns

  - $(ret_dict[:mev])

# Related

  - [`VarianceExpectedReturns`](@ref)
"""
function port_opt_view(me::VarianceExpectedReturns, i)::VarianceExpectedReturns
    return VarianceExpectedReturns(; me = port_opt_view(me.ce, i))
end
"""
    Statistics.mean(me::VarianceExpectedReturns, X::MatNum;
                    dims::Int = 1, kwargs...)

Compute expected returns as the variance of each asset.

This method returns the variance vector of `X` as estimated by the covariance estimator `me.ce`.

# Arguments

  - `me`: Variance expected returns estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the covariance estimator.

# Returns

  - `mu::Matrix{<:Number}`: Variance vector, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`VarianceExpectedReturns`](@ref)
"""
function Statistics.mean(me::VarianceExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    return Statistics.var(me.ce, X; dims = dims, kwargs...)
end

export StandardDeviationExpectedReturns, VarianceExpectedReturns
