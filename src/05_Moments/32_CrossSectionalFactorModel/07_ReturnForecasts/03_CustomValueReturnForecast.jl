"""
$(DocStringExtensions.TYPEDEF)

A Return Forecast that the caller states.

The member fits nothing. It carries the forecast the caller wrote, one entry per asset of the coverage universe and in return units. [`return_forecast`](@ref) checks it against the block and returns it. It is the stated member of its family, as [`CustomValueExpectedReturns`](@ref) is of the expected returns family and [`ConstantExposure`](@ref) is of the Factor Exposure family. A caller uses it to give a Prior a forecast that another tool made, and the Prior splits it against its latest Factor Exposures as it splits a fitted one.

An asset the caller forecasts nothing for carries a `NaN`. Every member of the family marks such an asset the same way.

# Mathematical definition

```math
\\begin{align}
\\alpha_{Ti} &= m_{i}\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_ti_fc]) The member publishes the row of the latest observation ``T`` and no other row.
  - ``m_{i}``: Entry ``i`` of `mu`, or `NaN` when the caller forecasts nothing for asset ``i``.
  - $(math_dict[:T])

# Fields

$(DocStringExtensions.TYPEDFIELDS)

# Constructors

    CustomValueReturnForecast(; mu::VecNum) -> CustomValueReturnForecast

Keywords correspond to the struct's fields. The constructor keeps `mu` as the caller wrote it, with its number type. The asset count is not known until the block arrives, so [`return_forecast`](@ref) checks the length of `mu`.

## Validation

  - `!isempty(mu)`. Raises an [`IsEmptyError`](@ref).

# Examples

```jldoctest
julia> CustomValueReturnForecast(; mu = [0.01, NaN])
CustomValueReturnForecast
  mu ┴ Vector{Float64}: [0.01, NaN]
```

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

The member states its forecast and computes none, so `hist` is `nothing`. A stated vector has no history. The two fields are the two reads [`AbstractReturnForecastResult`](@ref) states.

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

 1. Check the length of `rfe.mu` against the asset count of the block, the number of rows of `csfm.M`.
 2. Put `rfe.mu` on the member's own Result, with no history. The Result holds the vector of the estimator and not a copy, as the vector branch of [`CustomValueExpectedReturns`](@ref) does.

The member reads no Panel Field and fits nothing, so it does not read `rd`.

# Arguments

  - `rfe`: Stated Return Forecast Estimator.
  - $(arg_dict[:rd])
  - `csfm`: The fitted factor-model block. The member reads its asset count and nothing else.

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

"""
    port_opt_view(rf::CustomValueReturnForecastResult, i, args...)

Return a view of a [`CustomValueReturnForecastResult`](@ref), selecting only the assets indexed by `i`.

The member computes no history, so `mu` is the one field the view cuts.

# Arguments

  - `rf`: A stated Return Forecast result.
  - `i`: Indices of the assets to select.
  - `args...`: More positional arguments. The method ignores them.

# Returns

  - `rf::CustomValueReturnForecastResult`: A new result whose forecast is restricted to the selected assets.

# Related

  - [`CustomValueReturnForecastResult`](@ref)
  - [`port_opt_view`](@ref)
  - [`CrossSectionalFactorModel`](@ref)
"""
function port_opt_view(rf::CustomValueReturnForecastResult, i,
                       args...)::CustomValueReturnForecastResult
    return CustomValueReturnForecastResult(; mu = view(rf.mu, i))
end

export CustomValueReturnForecast, CustomValueReturnForecastResult
