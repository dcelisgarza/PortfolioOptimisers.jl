"""
$(DocStringExtensions.TYPEDEF)

A covariance estimator that returns the correlation matrix as both the covariance and correlation.

`CorrelationCovariance` wraps another covariance estimator and delegates both `cov` and `cor` calls to the underlying estimator's `cor` method. This is useful when a correlation matrix is needed in contexts that accept a covariance estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CorrelationCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance()
    ) -> CorrelationCovariance

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> CorrelationCovariance()
CorrelationCovariance
  ce ┼ Covariance
     │    me ┼ SimpleExpectedReturns
     │       │   w ┴ nothing
     │    ce ┼ GeneralCovariance
     │       │   ce ┼ StatsBase.SimpleCovariance: StatsBase.SimpleCovariance(true)
     │       │    w ┴ nothing
     │   alg ┴ Full()
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`Covariance`](@ref)
"""
@concrete struct CorrelationCovariance <: AbstractCovarianceEstimator
    "$(field_dict[:ce])"
    ce
    function CorrelationCovariance(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
function CorrelationCovariance(; ce::StatsBase.CovarianceEstimator = Covariance())
    return CorrelationCovariance(ce)
end
"""
    factory(ce::CorrelationCovariance, w::ObsWeights) -> CorrelationCovariance

Return a new [`CorrelationCovariance`](@ref) estimator with observation weights `w` applied to the underlying covariance estimator.

# Arguments

  - $(arg_dict[:ce])
  - $(arg_dict[:ow])

# Returns

  - $(ret_dict[:ce])

# Related

  - [`CorrelationCovariance`](@ref)
  - [`factory`](@ref)
"""
function factory(ce::CorrelationCovariance, w::ObsWeights)
    return CorrelationCovariance(; ce = factory(ce.ce, w))
end
"""
    Statistics.cov(ce::CorrelationCovariance, X::MatNum; dims::Int = 1,
                   kwargs...)

Compute the correlation matrix using the underlying estimator.

This method delegates to `Statistics.cor(ce.ce, X; dims = dims, kwargs...)`, returning the correlation matrix as the "covariance". This is useful when a correlation matrix is required in a context that accepts a covariance estimator.

# Arguments

  - `ce`: Correlation covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`CorrelationCovariance`](@ref)
  - [`cor(ce::CorrelationCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::CorrelationCovariance, X::MatNum; dims::Int = 1, kwargs...)
    return Statistics.cor(ce.ce, X; dims = dims, kwargs...)
end
"""
    Statistics.cor(ce::CorrelationCovariance, X::MatNum; dims::Int = 1,
                   kwargs...)

Compute the correlation matrix using the underlying estimator.

This method delegates to `Statistics.cor(ce.ce, X; dims = dims, kwargs...)`.

# Arguments

  - `ce`: Correlation covariance estimator.
  - `X`: Data matrix of asset returns (observations × assets).
  - $(arg_dict[:dims])
  - `kwargs...`: Additional keyword arguments passed to the underlying estimator.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`CorrelationCovariance`](@ref)
  - [`cov(ce::CorrelationCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::CorrelationCovariance, X::MatNum; dims::Int = 1, kwargs...)
    return Statistics.cor(ce.ce, X; dims = dims, kwargs...)
end

export CorrelationCovariance
