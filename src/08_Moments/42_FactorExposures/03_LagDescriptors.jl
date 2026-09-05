"""
    assert_descriptor_lag(lag::Integer) -> nothing

Check that a lag is a positive integer.

The three lag Descriptors read a Panel Field `lag` observations back, and a lag of zero would compare a value with itself. The check runs once, in each constructor.

# Arguments

  - `lag`: The number of observations to look back.

# Validation

  - `lag >= 1`. Raises a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`GrowthRate`](@ref)
  - [`ChangeToScale`](@ref)
  - [`ChangeInIntensity`](@ref)
"""
function assert_descriptor_lag(lag::Integer)::Nothing
    @argcheck(lag >= one(lag),
              DomainError(lag,
                          "lag is the number of observations a Descriptor looks back, so it must be a positive integer, got $lag"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Growth of a non-negative Panel Field over a fixed lag, at every observation.

This is the archetype of every growth Descriptor: the growth of total assets, of sales, of the share count. The first `lag` observations are `NaN`, because no lagged value exists there, and so is every cell whose lagged value is zero. The Panel Field must be non-negative wherever it is observed and active: a growth rate over a negative base flips its sign, so the estimator raises on a negative value rather than rank a shrinking loss as growth. A field that can turn negative, a net income or an earnings per share, takes [`ChangeToScale`](@ref) instead.

# Mathematical definition

```math
\\begin{align}
d_{t,i} &= \\begin{cases} \\dfrac{z_{t,i}}{z_{t-\\ell,i}} - 1 & \\text{if } t > \\ell \\text{ and } z_{t-\\ell,i} > 0 \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - ``z_{t,i}``: The Panel Field's value for asset ``i`` at observation ``t``.
  - ``\\ell``: The lag, in observations.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GrowthRate(; field::AbstractString, lag::Integer) -> GrowthRate

Keywords correspond to the struct's fields. `lag` takes no default, because it depends on the data frequency: `252` is one year of daily observations, `12` one year of monthly ones, `4` one year of quarterly ones. The named growth Descriptors fix it at `252`.

## Validation

  - `!isempty(field)`.
  - `lag >= 1`.

# Examples

```jldoctest
julia> GrowthRate(; field = \"sales_ttm\", lag = 252)
GrowthRate
  field ┼ String: \"sales_ttm\"
    lag ┴ Int64: 252
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`ChangeToScale`](@ref)
  - [`ChangeInIntensity`](@ref)
  - [`AssetsGrowthRate`](@ref)
  - [`SalesGrowthRate`](@ref)
  - [`IssuanceGrowthRate`](@ref)
"""
@concrete struct GrowthRate <: AbstractDescriptorEstimator
    """
    Name of the Panel Field whose growth is measured. It must be non-negative wherever it is observed and active.
    """
    field
    """
    Number of observations to look back.
    """
    lag
    function GrowthRate(field::AbstractString, lag::Integer)
        assert_panel_terms(field, :field)
        assert_descriptor_lag(lag)
        return new{typeof(field), typeof(lag)}(field, lag)
    end
end
function GrowthRate(; field::AbstractString, lag::Integer)::GrowthRate
    return GrowthRate(field, lag)
end
"""
$(DocStringExtensions.TYPEDEF)

Change of a Panel Field over a fixed lag, scaled by the current value of a second Panel Field.

This is the growth archetype for a field that can be negative. A growth rate over a negative base flips its sign, so an earnings change is instead divided by the current market capitalisation, and the sign of the Descriptor is the direction of the change. The first `lag` observations are `NaN`, and so is every cell where the scale is not strictly positive.

# Mathematical definition

```math
\\begin{align}
d_{t,i} &= \\begin{cases} \\dfrac{z_{t,i} - z_{t-\\ell,i}}{s_{t,i}} & \\text{if } t > \\ell \\text{ and } s_{t,i} > 0 \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - ``z_{t,i}``: The Panel Field's value for asset ``i`` at observation ``t``.
  - ``s_{t,i}``: The scale Panel Field's value for asset ``i`` at observation ``t``.
  - ``\\ell``: The lag, in observations.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ChangeToScale(; field::AbstractString, scale::AbstractString, lag::Integer) -> ChangeToScale

Keywords correspond to the struct's fields. `lag` takes no default, because it depends on the data frequency.

## Validation

  - `!isempty(field)` and `!isempty(scale)`.
  - `lag >= 1`.

# Examples

```jldoctest
julia> ChangeToScale(; field = \"net_income_ttm\", scale = \"market_cap\", lag = 252)
ChangeToScale
  field ┼ String: \"net_income_ttm\"
  scale ┼ String: \"market_cap\"
    lag ┴ Int64: 252
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`GrowthRate`](@ref)
  - [`ChangeInIntensity`](@ref)
  - [`EarningsChangeToPrice`](@ref)
"""
@concrete struct ChangeToScale <: AbstractDescriptorEstimator
    """
    Name of the Panel Field whose change is measured.
    """
    field
    """
    Name of the Panel Field the change is divided by, read at the current observation. The Descriptor is `NaN` wherever it is not strictly positive.
    """
    scale
    """
    Number of observations to look back.
    """
    lag
    function ChangeToScale(field::AbstractString, scale::AbstractString, lag::Integer)
        assert_panel_terms(field, :field)
        assert_panel_terms(scale, :scale)
        assert_descriptor_lag(lag)
        return new{typeof(field), typeof(scale), typeof(lag)}(field, scale, lag)
    end
end
function ChangeToScale(; field::AbstractString, scale::AbstractString,
                       lag::Integer)::ChangeToScale
    return ChangeToScale(field, scale, lag)
end
"""
$(DocStringExtensions.TYPEDEF)

Change of the ratio of two Panel Fields over a fixed lag.

Where [`ChangeToScale`](@ref) divides the change of a level by the current scale, this archetype forms the ratio at both ends and takes the difference, so it measures a change in intensity: a capital expenditure that grew with the assets it serves reads zero. The first `lag` observations are `NaN`, and so is every cell where either ratio is undefined because its scale is not strictly positive.

# Mathematical definition

```math
\\begin{align}
d_{t,i} &= \\begin{cases} \\dfrac{z_{t,i}}{s_{t,i}} - \\dfrac{z_{t-\\ell,i}}{s_{t-\\ell,i}} & \\text{if } t > \\ell \\text{, } s_{t,i} > 0 \\text{ and } s_{t-\\ell,i} > 0 \\\\ \\mathrm{NaN} & \\text{otherwise} \\end{cases}\\,.
\\end{align}
```

Where:

  - ``d_{t,i}``: Descriptor of asset ``i`` at observation ``t``.
  - ``z_{t,i}``: The Panel Field's value for asset ``i`` at observation ``t``.
  - ``s_{t,i}``: The scale Panel Field's value for asset ``i`` at observation ``t``.
  - ``\\ell``: The lag, in observations.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ChangeInIntensity(; field::AbstractString, scale::AbstractString, lag::Integer) -> ChangeInIntensity

Keywords correspond to the struct's fields. `lag` takes no default, because it depends on the data frequency.

## Validation

  - `!isempty(field)` and `!isempty(scale)`.
  - `lag >= 1`.

# Examples

```jldoctest
julia> ChangeInIntensity(; field = \"capex_ttm\", scale = \"total_assets\", lag = 252)
ChangeInIntensity
  field ┼ String: \"capex_ttm\"
  scale ┼ String: \"total_assets\"
    lag ┴ Int64: 252
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`descriptor`](@ref)
  - [`GrowthRate`](@ref)
  - [`ChangeToScale`](@ref)
  - [`CapexToAssetsChangeInIntensity`](@ref)
"""
@concrete struct ChangeInIntensity <: AbstractDescriptorEstimator
    """
    Name of the Panel Field whose intensity is measured.
    """
    field
    """
    Name of the Panel Field the intensity is measured against, read at both ends of the lag. The Descriptor is `NaN` wherever either value is not strictly positive.
    """
    scale
    """
    Number of observations to look back.
    """
    lag
    function ChangeInIntensity(field::AbstractString, scale::AbstractString, lag::Integer)
        assert_panel_terms(field, :field)
        assert_panel_terms(scale, :scale)
        assert_descriptor_lag(lag)
        return new{typeof(field), typeof(scale), typeof(lag)}(field, scale, lag)
    end
end
function ChangeInIntensity(; field::AbstractString, scale::AbstractString,
                           lag::Integer)::ChangeInIntensity
    return ChangeInIntensity(field, scale, lag)
end
"""
    descriptor(de::GrowthRate, rd::ReturnsResult) -> Matrix{<:Real}
    descriptor(de::ChangeToScale, rd::ReturnsResult) -> Matrix{<:Real}
    descriptor(de::ChangeInIntensity, rd::ReturnsResult) -> Matrix{<:Real}

Compute a lag Descriptor from the Panel Fields of a carrier.

The three archetypes read through [`panel_field_values`](@ref), walk the observations from `lag + 1` to the end, and end through [`descriptor_active_fill!`](@ref). The first `lag` rows stay `NaN`, and a `NaN` at either end of the lag is a `NaN` in the Descriptor. A carrier with no more observations than the lag returns an all-`NaN` Descriptor rather than an error, because a fold of a cross-validation can be that short.

# Algorithm

The method that Julia selects is the algorithm.

 1. [`GrowthRate`](@ref): check that the field is non-negative through [`assert_nonneg_panel_fields`](@ref), then write `z[t] / z[t - lag] - 1` through [`positive_divide`](@ref).
 2. [`ChangeToScale`](@ref): write `(z[t] - z[t - lag]) / s[t]` through [`positive_divide`](@ref).
 3. [`ChangeInIntensity`](@ref): write `z[t] / s[t] - z[t - lag] / s[t - lag]`, each ratio through [`positive_divide`](@ref).

Every method then writes `NaN` into the inactive cells.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`panel_field_values`](@ref) for every Panel Field the estimator names.
  - The rule of [`assert_nonneg_panel_fields`](@ref) for a [`GrowthRate`](@ref).

# Returns

  - `D::Matrix{<:Real}`: The Descriptor, `observations × assets`.

# Examples

```jldoctest
julia> pnl = asset_panel([NumericPanelInput(; name = \"sales_ttm\",
                                            vals = [100.0 50.0; 110.0 40.0; 121.0 0.0]),
                          NumericPanelInput(; name = \"market_cap\",
                                            vals = [1000.0 500.0; 1000.0 500.0; 1000.0 0.0])];
                         amsk = trues(3, 2), emsk = trues(3, 2));

julia> rd = ReturnsResult(; nx = [\"A\", \"B\"], X = zeros(3, 2), pnl = pnl);

julia> descriptor(GrowthRate(; field = \"sales_ttm\", lag = 1), rd)
3×2 Matrix{Float64}:
 NaN    NaN
   0.1   -0.2
   0.1   -1.0

julia> descriptor(ChangeToScale(; field = \"sales_ttm\", scale = \"market_cap\", lag = 1), rd)
3×2 Matrix{Float64}:
 NaN      NaN
   0.01    -0.02
   0.011  NaN

julia> descriptor(ChangeInIntensity(; field = \"sales_ttm\", scale = \"market_cap\", lag = 2), rd)
3×2 Matrix{Float64}:
 NaN      NaN
 NaN      NaN
   0.021  NaN
```

# Related

  - [`AbstractDescriptorEstimator`](@ref)
  - [`GrowthRate`](@ref)
  - [`ChangeToScale`](@ref)
  - [`ChangeInIntensity`](@ref)
  - [`panel_field_values`](@ref)
  - [`positive_divide`](@ref)
  - [`descriptor_active_fill!`](@ref)
"""
function descriptor(de::GrowthRate, rd::ReturnsResult)::Matrix{<:Real}
    assert_nonneg_panel_fields(rd, [String(de.field)])
    V = panel_field_values(rd, de.field)
    Tf = eltype(V)
    D = fill(Tf(NaN), size(V))
    lag = de.lag
    for i in axes(V, 2), t in (lag + 1):size(V, 1)
        D[t, i] = positive_divide(V[t, i], V[t - lag, i]) - one(Tf)
    end
    descriptor_active_fill!(D, rd.pnl)
    return D
end
function descriptor(de::ChangeToScale, rd::ReturnsResult)::Matrix{<:Real}
    V = panel_field_values(rd, de.field)
    S = panel_field_values(rd, de.scale)
    Tf = eltype(V)
    D = fill(Tf(NaN), size(V))
    lag = de.lag
    for i in axes(V, 2), t in (lag + 1):size(V, 1)
        D[t, i] = positive_divide(V[t, i] - V[t - lag, i], S[t, i])
    end
    descriptor_active_fill!(D, rd.pnl)
    return D
end
function descriptor(de::ChangeInIntensity, rd::ReturnsResult)::Matrix{<:Real}
    V = panel_field_values(rd, de.field)
    S = panel_field_values(rd, de.scale)
    Tf = eltype(V)
    D = fill(Tf(NaN), size(V))
    lag = de.lag
    for i in axes(V, 2), t in (lag + 1):size(V, 1)
        D[t, i] = positive_divide(V[t, i], S[t, i]) -
                  positive_divide(V[t - lag, i], S[t - lag, i])
    end
    descriptor_active_fill!(D, rd.pnl)
    return D
end
"""
    AssetsGrowthRate(; field::AbstractString = "total_assets", lag::Integer = 252) -> GrowthRate

Growth of total assets over one year, the investment Descriptor.

The value is `total_assets(t) / total_assets(t - lag) - 1`, `NaN` on the first `lag` observations and where the lagged value is zero. A firm whose balance sheet expands quickly tends to earn a lower return afterwards, which is what the investment factor prices.

# Arguments

  - `field`: Name of the total assets Panel Field.
  - `lag`: Number of observations to look back. `252` is one year of daily observations.

# Returns

  - `de::GrowthRate`: The estimator, with the Panel Field and the lag fixed.

# Examples

```jldoctest
julia> AssetsGrowthRate()
GrowthRate
  field ┼ String: \"total_assets\"
    lag ┴ Int64: 252
```

# Related

  - [`GrowthRate`](@ref)
  - [`descriptor`](@ref)
  - [`IssuanceGrowthRate`](@ref)
  - [`CapexToAssetsChangeInIntensity`](@ref)
"""
function AssetsGrowthRate(; field::AbstractString = "total_assets",
                          lag::Integer = 252)::GrowthRate
    return GrowthRate(; field = field, lag = lag)
end
"""
    SalesGrowthRate(; field::AbstractString = "sales_ttm", lag::Integer = 252) -> GrowthRate

Growth of trailing sales over one year, the growth Descriptor.

The value is `sales_ttm(t) / sales_ttm(t - lag) - 1`, `NaN` on the first `lag` observations and where the lagged value is zero. With a trailing twelve-month field and a one-year lag, the two ends of the comparison cover disjoint fiscal content.

# Arguments

  - `field`: Name of the sales Panel Field.
  - `lag`: Number of observations to look back. `252` is one year of daily observations.

# Returns

  - `de::GrowthRate`: The estimator, with the Panel Field and the lag fixed.

# Examples

```jldoctest
julia> SalesGrowthRate()
GrowthRate
  field ┼ String: \"sales_ttm\"
    lag ┴ Int64: 252
```

# Related

  - [`GrowthRate`](@ref)
  - [`descriptor`](@ref)
  - [`AssetsGrowthRate`](@ref)
  - [`EarningsChangeToPrice`](@ref)
"""
function SalesGrowthRate(; field::AbstractString = "sales_ttm",
                         lag::Integer = 252)::GrowthRate
    return GrowthRate(; field = field, lag = lag)
end
"""
    IssuanceGrowthRate(; field::AbstractString = "adj_shares_outstanding",
                       lag::Integer = 252) -> GrowthRate

Growth of the split-adjusted share count over one year, the net issuance Descriptor.

The value is `adj_shares_outstanding(t) / adj_shares_outstanding(t - lag) - 1`, `NaN` on the first `lag` observations and where the lagged value is zero. A positive value is a net issuance, and a negative one a net buyback.

# Arguments

  - `field`: Name of the shares outstanding Panel Field.
  - `lag`: Number of observations to look back. `252` is one year of daily observations.

# Returns

  - `de::GrowthRate`: The estimator, with the Panel Field and the lag fixed.

# Examples

```jldoctest
julia> IssuanceGrowthRate()
GrowthRate
  field ┼ String: \"adj_shares_outstanding\"
    lag ┴ Int64: 252
```

# Related

  - [`GrowthRate`](@ref)
  - [`descriptor`](@ref)
  - [`AssetsGrowthRate`](@ref)
  - [`ShareholderYield`](@ref)
"""
function IssuanceGrowthRate(; field::AbstractString = "adj_shares_outstanding",
                            lag::Integer = 252)::GrowthRate
    return GrowthRate(; field = field, lag = lag)
end
"""
    EarningsChangeToPrice(; field::AbstractString = "net_income_ttm",
                          scale::AbstractString = "market_cap",
                          lag::Integer = 252) -> ChangeToScale

Change of trailing net income over one year, divided by the current market capitalisation.

The value is `(net_income_ttm(t) - net_income_ttm(t - lag)) / market_cap(t)`, `NaN` on the first `lag` observations and where the market capitalisation is not strictly positive. It is the earnings momentum Descriptor, and it stays well defined through a loss, where a growth rate of the earnings would not.

# Arguments

  - `field`: Name of the net income Panel Field.
  - `scale`: Name of the market capitalisation Panel Field.
  - `lag`: Number of observations to look back. `252` is one year of daily observations.

# Returns

  - `de::ChangeToScale`: The estimator, with the two Panel Fields and the lag fixed.

# Examples

```jldoctest
julia> EarningsChangeToPrice()
ChangeToScale
  field ┼ String: \"net_income_ttm\"
  scale ┼ String: \"market_cap\"
    lag ┴ Int64: 252
```

# Related

  - [`ChangeToScale`](@ref)
  - [`descriptor`](@ref)
  - [`EarningsToPrice`](@ref)
  - [`SalesGrowthRate`](@ref)
"""
function EarningsChangeToPrice(; field::AbstractString = "net_income_ttm",
                               scale::AbstractString = "market_cap",
                               lag::Integer = 252)::ChangeToScale
    return ChangeToScale(; field = field, scale = scale, lag = lag)
end
"""
    CapexToAssetsChangeInIntensity(; field::AbstractString = "capex_ttm",
                                   scale::AbstractString = "total_assets",
                                   lag::Integer = 252) -> ChangeInIntensity

Change of the capital expenditure to total assets ratio over one year.

The value is `capex_ttm(t) / total_assets(t) - capex_ttm(t - lag) / total_assets(t - lag)`, `NaN` on the first `lag` observations and where either total assets value is not strictly positive. A positive value says that the firm invests a larger share of its assets than a year ago.

# Arguments

  - `field`: Name of the capital expenditure Panel Field.
  - `scale`: Name of the total assets Panel Field.
  - `lag`: Number of observations to look back. `252` is one year of daily observations.

# Returns

  - `de::ChangeInIntensity`: The estimator, with the two Panel Fields and the lag fixed.

# Examples

```jldoctest
julia> CapexToAssetsChangeInIntensity()
ChangeInIntensity
  field ┼ String: \"capex_ttm\"
  scale ┼ String: \"total_assets\"
    lag ┴ Int64: 252
```

# Related

  - [`ChangeInIntensity`](@ref)
  - [`descriptor`](@ref)
  - [`AssetsGrowthRate`](@ref)
"""
function CapexToAssetsChangeInIntensity(; field::AbstractString = "capex_ttm",
                                        scale::AbstractString = "total_assets",
                                        lag::Integer = 252)::ChangeInIntensity
    return ChangeInIntensity(; field = field, scale = scale, lag = lag)
end

export GrowthRate, ChangeToScale, ChangeInIntensity, AssetsGrowthRate, SalesGrowthRate,
       IssuanceGrowthRate, EarningsChangeToPrice, CapexToAssetsChangeInIntensity
