"""
$(DocStringExtensions.TYPEDEF)

Answers both `cov` and `cor` with the wrapped estimator's correlation matrix.

Use it where a caller demands a covariance estimator but the computation wants the correlation — a clustering distance, for instance, which reads a scale-free matrix.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CorrelationCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> CorrelationCovariance

Keywords correspond to the struct's fields.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `ce`: Recursively updated via [`factory`](@ref).
  - `cache`: Carried unchanged via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `ce`: Recursively viewed via [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets via [`port_opt_view`](@ref).

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
    """
    $(field_dict[:pfcache])
    """
    @fprop @vprop cache
    function CorrelationCovariance(ce::StatsBase.CovarianceEstimator,
                                   cache::Option{<:AbstractPartialFitState})
        return new{typeof(ce), typeof(cache)}(ce, cache)
    end
end
function CorrelationCovariance(; ce::StatsBase.CovarianceEstimator = Covariance(),
                               cache::Option{<:AbstractPartialFitState} = nothing)::CorrelationCovariance
    return CorrelationCovariance(ce, cache)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`CorrelationCovariance`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one. Set `set_show_nothing_fields!(:CorrelationCovariance, true)` to render it.

# Arguments

  - `::CorrelationCovariance`: Correlation covariance estimator, read for its type alone.

# Returns

  - `fields::Tuple`: The field names to render, which is `(:ce,)`.

# Related

  - [`CorrelationCovariance`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
show_fields(::CorrelationCovariance) = (:ce,)
"""
    Statistics.cov(ce::CorrelationCovariance, X::MatNum; dims::Int = 1,
                   kwargs...)

Compute the correlation matrix using the underlying estimator.

This method delegates to `Statistics.cor(library_covariance_estimator(ce.ce), X; dims = dims, kwargs...)`, returning the correlation matrix as the "covariance". This is useful when a correlation matrix is required in a context that accepts a covariance estimator.

# Algorithm

 1. Call `Statistics.cor(library_covariance_estimator(ce.ce), X; dims = dims, kwargs...)` and return its result.

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
    return Statistics.cor(library_covariance_estimator(ce.ce), X; dims = dims, kwargs...)
end
"""
    gap_fill_value(ce::CorrelationCovariance) -> Number

Answer what `ce.ce` answers, because the estimator forwards the sample and its keywords untouched.

This estimator reads no cell of the sample itself: it calls the correlation verb of `ce.ce` and answers what comes back. A gap is therefore the inner estimator's to keep or to lose.

# Arguments

  - `ce`: Correlation covariance estimator.

# Returns

  - `fv::Number`: [`gap_fill_value`](@ref) of `ce.ce`.

# Related

  - [`CorrelationCovariance`](@ref)
  - [`gap_fill_value`](@ref)
"""
function gap_fill_value(ce::CorrelationCovariance)
    return gap_fill_value(ce.ce)
end
"""
    Statistics.cor(ce::CorrelationCovariance, X::MatNum; dims::Int = 1,
                   kwargs...)

Compute the correlation matrix using the underlying estimator.

This method delegates to `Statistics.cor(library_covariance_estimator(ce.ce), X; dims = dims, kwargs...)`.

# Algorithm

 1. Call `Statistics.cor(library_covariance_estimator(ce.ce), X; dims = dims, kwargs...)` and return its result.

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
    return Statistics.cor(library_covariance_estimator(ce.ce), X; dims = dims, kwargs...)
end

export CorrelationCovariance
