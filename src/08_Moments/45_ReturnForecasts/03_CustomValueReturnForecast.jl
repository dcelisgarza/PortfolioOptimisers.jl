"""
$(DocStringExtensions.TYPEDEF)

A Return Forecast the caller states outright.

The member fits nothing. It carries the forecast the caller wrote, one entry per asset of the coverage universe and in return units, and [`return_forecast`](@ref) checks it against the block and hands it back. It is the stated member of its family, as [`CustomValueExpectedReturns`](@ref) is of the expected returns family and [`ConstantExposure`](@ref) is of the Factor Exposure family, and a caller uses it to carry a forecast another tool produced through the prior's split.

An asset the caller forecasts nothing for carries a `NaN`, which is how every member of the family says the same thing.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Related

  - [`AbstractReturnForecastEstimator`](@ref)
  - [`CustomValueReturnForecastResult`](@ref)
  - [`return_forecast`](@ref)
  - [`FixedWeightedReturnForecast`](@ref)
  - [`CustomValueExpectedReturns`](@ref)
"""
@concrete struct CustomValueReturnForecast <: AbstractReturnForecastEstimator
    """
    The stated Return Forecast, one entry per asset of the coverage universe, in return units, `NaN` where the caller forecasts nothing.
    """
    mu
    function CustomValueReturnForecast(mu::VecNum)
        @argcheck(!isempty(mu),
                  IsEmptyError("mu carries the stated Return Forecast, so it cannot be empty"))
        return new{typeof(mu)}(mu)
    end
end
function CustomValueReturnForecast(; mu::VecNum)::CustomValueReturnForecast
    return CustomValueReturnForecast(mu)
end
"""
$(DocStringExtensions.TYPEDEF)

Result type produced by [`CustomValueReturnForecast`](@ref).

The member states its forecast and computes none, so `hist` is `nothing`: there is no history behind a stated vector. The two fields are the two reads [`AbstractReturnForecastResult`](@ref) states.

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Related

  - [`AbstractReturnForecastResult`](@ref)
  - [`CustomValueReturnForecast`](@ref)
  - [`return_forecast`](@ref)
"""
@concrete struct CustomValueReturnForecastResult <: AbstractReturnForecastResult
    """
    $(field_dict[:rf_mu])
    """
    mu
    """
    $(field_dict[:rf_hist])
    """
    hist
    function CustomValueReturnForecastResult(mu::VecNum)
        @argcheck(!isempty(mu), IsEmptyError("mu cannot be empty"))
        return new{typeof(mu), Nothing}(mu, nothing)
    end
end
function CustomValueReturnForecastResult(; mu::VecNum)::CustomValueReturnForecastResult
    return CustomValueReturnForecastResult(mu)
end
"""
    return_forecast(rfe::CustomValueReturnForecast, rd::ReturnsResult,
                    csfm::CrossSectionalFactorModel) -> CustomValueReturnForecastResult

Return the Return Forecast a caller stated.

# Algorithm

 1. Check the stated forecast against the asset count of the block.
 2. Carry it onto the member's own Result, with no history.

The carrier is not read: the member reads no Panel Field and fits nothing.

# Arguments

  - `rfe`: Stated Return Forecast Estimator.
  - $(arg_dict[:rd])
  - `csfm`: The fitted factor-model block, read for its asset count alone.

# Validation

  - `length(rfe.mu)` matches the asset count of `csfm`. Raises a `DimensionMismatch`.

# Returns

  - `rf::CustomValueReturnForecastResult`: The stated forecast.

# Examples

```jldoctest
julia> csfm = CrossSectionalFactorModel(; M = reshape([1.0, 1.0], 2, 1), b = [0.0, 0.0]);

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(2, 2));

julia> rf = return_forecast(CustomValueReturnForecast(; mu = [0.01, NaN]), rd, csfm);

julia> rf.mu
2-element Vector{Float64}:
   0.01
 NaN
```

# Related

  - [`CustomValueReturnForecast`](@ref)
  - [`CustomValueReturnForecastResult`](@ref)
  - [`AbstractReturnForecastEstimator`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function return_forecast(rfe::CustomValueReturnForecast, ::ReturnsResult,
                         csfm::CrossSectionalFactorModel)::CustomValueReturnForecastResult
    N = size(csfm.M, 1)
    @argcheck(length(rfe.mu) == N,
              DimensionMismatch("mu ($(length(rfe.mu))) states one Return Forecast per asset of the coverage universe, so it must match the asset count of the factor model block ($N)"))
    return CustomValueReturnForecastResult(; mu = rfe.mu)
end

export CustomValueReturnForecast, CustomValueReturnForecastResult
