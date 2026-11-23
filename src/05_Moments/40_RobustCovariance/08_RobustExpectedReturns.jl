"""
$(DocStringExtensions.TYPEDEF)

Reads the robust location of a robust covariance fit as an expected-returns vector, so that any `me` slot can hold it.

The adapter fits `ce` on the returns that it receives, and keeps the `mu` of the [`RobustCovarianceResult`](@ref). An outlier moves the sample mean in proportion to its size, and it moves the location of a robust fit little. A prior can take the adapter as its `me` and a robust covariance estimator as its `ce`. With one estimator in both slots, the prior fits it twice.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RobustExpectedReturns(;
        ce::AbstractRobustCovarianceEstimator = MCDCovariance()
    ) -> RobustExpectedReturns

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `ce`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `ce`: Recursively viewed through [`port_opt_view`](@ref).

# Examples

```jldoctest
julia> RobustExpectedReturns(; ce = TylerCovariance())
RobustExpectedReturns
  ce ┼ TylerCovariance
     │   me ┼ nothing
     │   ve ┼ nothing
     │    w ┼ nothing
     │    q ┴ Float64: 0.975
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractRobustCovarianceEstimator`](@ref)
  - [`MCDCovariance`](@ref)
  - [`TylerCovariance`](@ref)
  - [`PriorExpectedReturns`](@ref)
"""
@propagatable @concrete struct RobustExpectedReturns <: AbstractExpectedReturnsEstimator
    """
    $(field_dict[:rob_ce])
    """
    @fprop @vprop ce
    function RobustExpectedReturns(ce::AbstractRobustCovarianceEstimator)
        return new{typeof(ce)}(ce)
    end
end
function RobustExpectedReturns(;
                               ce::AbstractRobustCovarianceEstimator = MCDCovariance())::RobustExpectedReturns
    return RobustExpectedReturns(ce)
end
"""
    Statistics.mean(me::RobustExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Fits the robust covariance estimator of the adapter on `X`, and returns the robust location as a matrix.

# Algorithm

 1. Check `dims`.
 2. Fit `me.ce` on `X` with `StatsAPI.fit`.
 3. Reshape the `mu` of the fit to `(1, N)` when `dims == 1`, and to `(N, 1)` when `dims == 2`.

# Arguments

  - `me`: The adapter.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `kwargs...`: More keyword arguments for the fit.

# Validation

  - $(val_dict[:dims])
  - Everything the fit of `me.ce` refuses.

# Returns

  - `mu::Matrix{<:Number}`: The robust location, shaped `(1, N)` when `dims == 1` and `(N, 1)` when `dims == 2`.

# Related

  - [`RobustExpectedReturns`](@ref)
  - [`RobustCovarianceResult`](@ref)
"""
function Statistics.mean(me::RobustExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    mu = StatsAPI.fit(me.ce, X; dims = dims, kwargs...).mu
    return isone(dims) ? reshape(mu, 1, :) : reshape(mu, :, 1)
end

export RobustExpectedReturns
