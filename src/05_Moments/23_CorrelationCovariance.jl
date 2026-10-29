"""
$(DocStringExtensions.TYPEDEF)

Returns the correlation matrix of the wrapped estimator from both `cov` and `cor`.

Use it where a caller needs a covariance estimator but the computation needs the correlation matrix. A clustering distance is an example, because it reads a matrix that has no scale.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CorrelationCovariance(;
        ce::StatsBase.CovarianceEstimator = Covariance(),
        cache::Option{<:AbstractPartialFitState} = nothing
    ) -> CorrelationCovariance

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:ce_cov])

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated through [`factory`](@ref).
  - `cache`: Carried unchanged through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ce`: Recursively viewed through [`port_opt_view`](@ref).
  - `cache`: Sliced to the selected assets through [`port_opt_view`](@ref).

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
        assert_covariance_estimator(ce, :ce)
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

A `cache` holds the state of an incremental fit. That state is not part of the configuration, and it prints under the estimator at each site that renders the estimator. Set `set_show_nothing_fields!(:CorrelationCovariance, true)` to render it.

# Arguments

  - `::CorrelationCovariance`: The correlation covariance estimator. The method reads only its type.

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

Compute the correlation matrix with the wrapped estimator, and return it as the covariance matrix.

Use this method where a context accepts a covariance estimator but needs a correlation matrix.

# Algorithm

 1. Call `Statistics.cor(library_covariance_estimator(ce.ce), X; dims = dims, kwargs...)` and return its result.

The returned matrix has a unit diagonal. A caller that reads the diagonal as variances reads ones.

# Arguments

  - `ce`: The correlation covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for the wrapped estimator.

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

Return the gap fill value of `ce.ce`, because the estimator passes the sample and its keywords to `ce.ce` without change.

The estimator reads no cell of the sample. It calls `Statistics.cor` on `ce.ce` and returns the result, so `ce.ce` alone decides what a gap does to the result.

# Arguments

  - `ce`: The correlation covariance estimator.

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

Compute the correlation matrix with the wrapped estimator.

# Algorithm

 1. Call `Statistics.cor(library_covariance_estimator(ce.ce), X; dims = dims, kwargs...)` and return its result.

`cov` and `cor` on a [`CorrelationCovariance`](@ref) return the same matrix.

# Arguments

  - `ce`: The correlation covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for the wrapped estimator.

# Returns

  - $(ret_dict[:rho])

# Related

  - [`CorrelationCovariance`](@ref)
  - [`cov(ce::CorrelationCovariance, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::CorrelationCovariance, X::MatNum; dims::Int = 1, kwargs...)
    return Statistics.cor(library_covariance_estimator(ce.ce), X; dims = dims, kwargs...)
end
# The estimator passes `X` to `ce.ce` without change, so the count is the count of `ce.ce`.
function effective_observation_count(ce::CorrelationCovariance, X::MatNum)
    return effective_observation_count(ce.ce, X)
end

export CorrelationCovariance
