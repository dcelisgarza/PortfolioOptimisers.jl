"""
$(DocStringExtensions.TYPEDEF)

Answers both `cov` and `cor` with the wrapped estimator's correlation matrix.

Use it where a caller demands a covariance estimator but the computation wants the correlation — a clustering distance, for instance, which reads a scale-free matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CorrelationCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance()
    ) -> CorrelationCovariance

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).

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
     │   alg ┼ FullMoment()
     │     w ┴ nothing
```

# Related

  - [`AbstractCovarianceEstimator`](@ref)
  - [`Covariance`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct CorrelationCovariance <: AbstractCovarianceEstimator
    """
    $(field_dict[:ce])
    """
    @fprop @vprop ce
    function CorrelationCovariance(ce::StatsBase.CovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
function CorrelationCovariance(;
                               ce::StatsBase.CovarianceEstimator = Covariance())::CorrelationCovariance
    return CorrelationCovariance(ce)
end
"""
    Statistics.cov(ce::CorrelationCovariance, X::MatNum; dims::Int = 1,
                   kwargs...)

Compute the correlation matrix using the underlying estimator.

This method delegates to `Statistics.cor(ce.ce, X; dims = dims, kwargs...)`, returning the correlation matrix as the "covariance". This is useful when a correlation matrix is required in a context that accepts a covariance estimator.

# Algorithm

 1. Call `Statistics.cor(ce.ce, X; dims = dims, kwargs...)` and return its result.

The returned matrix carries a unit diagonal, so a caller that reads the diagonal for a variance
reads ones, not variances.

# Arguments

  - `ce`: Correlation covariance estimator.
  - $(arg_dict[:X])
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

# Algorithm

 1. Call `Statistics.cor(ce.ce, X; dims = dims, kwargs...)` and return its result.

`cov` and `cor` on a [`CorrelationCovariance`](@ref) return the same matrix.

# Arguments

  - `ce`: Correlation covariance estimator.
  - $(arg_dict[:X])
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
