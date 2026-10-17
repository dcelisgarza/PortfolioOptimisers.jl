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

    GrowthRate(; field::AbstractString, lag::Integer,
               cache::Option{<:AbstractPartialFitState} = nothing) -> GrowthRate

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
    """
    $(field_dict[:lag_desc_cache])
    """
    cache
    function GrowthRate(field::AbstractString, lag::Integer,
                        cache::Option{<:AbstractPartialFitState})
        assert_panel_terms(field, :field)
        assert_descriptor_lag(lag)
        return new{typeof(field), typeof(lag), typeof(cache)}(field, lag, cache)
    end
end
function GrowthRate(; field::AbstractString, lag::Integer,
                    cache::Option{<:AbstractPartialFitState} = nothing)::GrowthRate
    return GrowthRate(field, lag, cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Change of a Panel Field over a fixed lag, scaled by the current value of a second Panel Field.

This is the growth archetype for a field that can be negative. A growth rate over a negative base flips its sign, so an earnings change is instead divided by the current market capitalisation, and the sign of the Descriptor is the direction of the change. The first `lag` observations are `NaN`, and so is every cell where the scale is not strictly positive. Under `gt0 = true` a scale at or below zero is a data error instead, and the Descriptor refuses it: [`EarningsChangeToPrice`](@ref) sets it, because a market capitalisation is positive by construction.

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

    ChangeToScale(; field::AbstractString, scale::AbstractString, lag::Integer,
                  gt0::Bool = false,
                  cache::Option{<:AbstractPartialFitState} = nothing) -> ChangeToScale

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
    lag ┼ Int64: 252
    gt0 ┴ Bool: false
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
    """
    Whether the scale must be strictly positive wherever it is observed and active. A value at or below zero then raises a `DomainError`; otherwise its cell is `NaN`.
    """
    gt0
    """
    $(field_dict[:lag_desc_cache])
    """
    cache
    function ChangeToScale(field::AbstractString, scale::AbstractString, lag::Integer,
                           gt0::Bool, cache::Option{<:AbstractPartialFitState})
        assert_panel_terms(field, :field)
        assert_panel_terms(scale, :scale)
        assert_descriptor_lag(lag)
        fs = (field, scale, lag, gt0, cache)
        return new{map(typeof, fs)...}(fs...)
    end
end
function ChangeToScale(; field::AbstractString, scale::AbstractString, lag::Integer,
                       gt0::Bool = false,
                       cache::Option{<:AbstractPartialFitState} = nothing)::ChangeToScale
    return ChangeToScale(field, scale, lag, gt0, cache)
end
"""
$(DocStringExtensions.TYPEDEF)

Change of the ratio of two Panel Fields over a fixed lag.

Where [`ChangeToScale`](@ref) divides the change of a level by the current scale, this archetype forms the ratio at both ends and takes the difference, so it measures a change in intensity: a capital expenditure that grew with the assets it serves reads zero. The first `lag` observations are `NaN`, and so is every cell where either ratio is undefined because its scale is not strictly positive. Under `gt0 = true` a scale at or below zero is a data error instead, and the Descriptor refuses it: [`CapexToAssetsChangeInIntensity`](@ref) sets it, because a total of assets is positive by construction.

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

    ChangeInIntensity(; field::AbstractString, scale::AbstractString, lag::Integer,
                      gt0::Bool = false,
                      cache::Option{<:AbstractPartialFitState} = nothing) -> ChangeInIntensity

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
    lag ┼ Int64: 252
    gt0 ┴ Bool: false
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
    """
    Whether the scale must be strictly positive wherever it is observed and active. A value at or below zero then raises a `DomainError`; otherwise every ratio over it is `NaN`.
    """
    gt0
    """
    $(field_dict[:lag_desc_cache])
    """
    cache
    function ChangeInIntensity(field::AbstractString, scale::AbstractString, lag::Integer,
                               gt0::Bool, cache::Option{<:AbstractPartialFitState})
        assert_panel_terms(field, :field)
        assert_panel_terms(scale, :scale)
        assert_descriptor_lag(lag)
        fs = (field, scale, lag, gt0, cache)
        return new{map(typeof, fs)...}(fs...)
    end
end
function ChangeInIntensity(; field::AbstractString, scale::AbstractString, lag::Integer,
                           gt0::Bool = false,
                           cache::Option{<:AbstractPartialFitState} = nothing)::ChangeInIntensity
    return ChangeInIntensity(field, scale, lag, gt0, cache)
end
"""
    descriptor(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity},
               rd::ReturnsResult) -> Matrix{<:Real}

Compute a lag Descriptor from the Panel Fields of a [`ReturnsResult`](@ref).

The three archetypes read through [`descriptor_field_values`](@ref), walk the observations from `lag + 1` to the end, and end through [`descriptor_active_fill!`](@ref). The first `lag` rows stay `NaN`, and a `NaN` at either end of the lag is a `NaN` in the Descriptor. A `ReturnsResult` with no more observations than the lag returns an all-`NaN` Descriptor rather than an error, because a fold of a cross-validation can be that short.

The batch call is the fold [`lag_descriptor_fold`](@ref) from no state, so the step [`descriptor_step`](@ref) and the batch call run one kernel, and a folded observation equals the batch call to the last bit.

# Algorithm

 1. Check the Panel Fields and read them through [`lag_descriptor_inputs`](@ref): for a [`GrowthRate`](@ref), check that the field is non-negative through [`assert_panel_field_sign`](@ref); for a [`ChangeToScale`](@ref) or a [`ChangeInIntensity`](@ref), check the scale under `gt0`.
 2. Write the Descriptor of each cell from its current values and from the lagged quantity `lag` observations back through [`lag_descriptor_value`](@ref):
     1. [`GrowthRate`](@ref): `z[t] / z[t - lag] - 1` through [`positive_divide`](@ref).
     2. [`ChangeToScale`](@ref): `(z[t] - z[t - lag]) / s[t]` through [`positive_divide`](@ref).
     3. [`ChangeInIntensity`](@ref): `z[t] / s[t] - z[t - lag] / s[t - lag]`, each ratio through [`positive_divide`](@ref).
 3. Write `NaN` into the inactive cells.

# Arguments

  - `de`: Descriptor Estimator.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rules of [`descriptor_field_values`](@ref) for every Panel Field the estimator names.
  - The rule of [`assert_panel_field_sign`](@ref) for a [`GrowthRate`](@ref), and for the scale of a [`ChangeToScale`](@ref) or a [`ChangeInIntensity`](@ref) under `gt0 = true`.

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
  - [`descriptor_field_values`](@ref)
  - [`positive_divide`](@ref)
  - [`descriptor_active_fill!`](@ref)
  - [`lag_descriptor_fold`](@ref)
"""
function descriptor(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity},
                    rd::ReturnsResult)::Matrix{<:Real}
    return lag_descriptor_fold(de, rd, nothing).D
end
function lookback(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity})::Integer
    return de.lag + 1
end
"""
$(DocStringExtensions.TYPEDEF)

Carried state of a lag Descriptor: the lagged quantity of the last `lag` observations.

A [`GrowthRate`](@ref), a [`ChangeToScale`](@ref) and a [`ChangeInIntensity`](@ref) read two observations of each cell: the current one, and the one `lag` observations back. The value that the observation `lag` back gives is the lagged quantity: the field `z` for a [`GrowthRate`](@ref) and a [`ChangeToScale`](@ref), and the ratio `z / s` for a [`ChangeInIntensity`](@ref), as [`lag_descriptor_lagged`](@ref) gives it. So the last `lag` rows of that quantity give the Descriptor of the next observation, and the state holds no other value. The state keeps each row with the bits of the batch call, so the Descriptor of a new observation equals the one of the batch call to the last bit.

Each row is a vector of its own, and no verb changes a row after the state carries it. A step copies the buffer, which copies the references to the rows and no row, so the state before the step stays as it was.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LagDescriptorState(; q::DataStructures.CircularBuffer{<:AbstractVector{<:Real}}) -> LagDescriptorState

Keywords correspond to the struct's fields. [`descriptor_step`](@ref) seeds an empty buffer of capacity `lag`.

## Validation

  - Every row of `q` holds the same number of assets. A `DimensionMismatch` is thrown otherwise.

# Related

  - [`GrowthRate`](@ref)
  - [`ChangeToScale`](@ref)
  - [`ChangeInIntensity`](@ref)
  - [`descriptor_step`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
@concrete struct LagDescriptorState <: AbstractPartialFitState
    """
    The lagged quantity of each asset, one row per observation, oldest first. The capacity of the buffer is the lag.
    """
    q
    function LagDescriptorState(q::DataStructures.CircularBuffer{<:AbstractVector{<:Real}})
        @argcheck(isempty(q) || all(r -> length(r) == length(q[1]), q),
                  DimensionMismatch("every row of a LagDescriptorState holds the same assets, got rows of $(unique(length.(q))) assets"))
        return new{typeof(q)}(q)
    end
end
function LagDescriptorState(;
                            q::DataStructures.CircularBuffer{<:AbstractVector{<:Real}})::LagDescriptorState
    return LagDescriptorState(q)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Merges two [`LagDescriptorState`](@ref) fitted on consecutive blocks into the state of the concatenated block.

The state of a block is the lagged quantity of its last `lag` observations. So the state of the concatenated block is the last `lag` rows of the rows of `a` followed by the rows of `b`, and the merge is exact.

# Arguments

  - `a`: The state of the first block.
  - `b`: The state of the second block.

# Validation

  - `a` and `b` have the same capacity, the lag. A `DimensionMismatch` is thrown otherwise.
  - The rule of [`LagDescriptorState`](@ref) on the merged rows.

# Returns

  - `state::LagDescriptorState`: The state of the concatenated block.

# Related

  - [`LagDescriptorState`](@ref)
  - [`merge_states`](@ref)
"""
function merge_states(a::LagDescriptorState, b::LagDescriptorState)::LagDescriptorState
    @argcheck(DataStructures.capacity(a.q) == DataStructures.capacity(b.q),
              DimensionMismatch("two LagDescriptorState merge when they carry the rows of one lag, got the lags $(DataStructures.capacity(a.q)) and $(DataStructures.capacity(b.q))"))
    q = rolling_state_buffer(identity, a.q)
    append!(q, b.q)
    return LagDescriptorState(q)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`LagDescriptorState`](@ref), so that the copy shares no row with the original.

# Arguments

  - `x`: The state to copy.

# Returns

  - `state::LagDescriptorState`: A new state, equal to `x`.

# Related

  - [`LagDescriptorState`](@ref)
"""
function Base.copy(x::LagDescriptorState)::LagDescriptorState
    return LagDescriptorState(rolling_state_buffer(copy, x.q))
end
"""
    lag_descriptor_inputs(de::GrowthRate, rd::ReturnsResult)
    lag_descriptor_inputs(de::Union{ChangeToScale, ChangeInIntensity}, rd::ReturnsResult)

Checks the Panel Fields of a lag Descriptor, and reads the field and the scale through [`descriptor_field_values`](@ref).

A [`GrowthRate`](@ref) reads no scale, so its scale is the field.

# Arguments

  - `de`: The lag Descriptor.
  - $(arg_dict[:rd]) It must carry an Asset Panel in `rd.pnl`.

# Validation

  - The rule of [`assert_panel_field_sign`](@ref) for the field of a [`GrowthRate`](@ref), and for the scale under `gt0 = true`.

# Returns

  - `inputs::NamedTuple`: `V`, the values of the field, and `S`, the values of the scale, `observations × assets`.

# Related

  - [`lag_descriptor_fold`](@ref)
"""
function lag_descriptor_inputs(de::GrowthRate, rd::ReturnsResult)
    assert_panel_field_sign(rd, [String(de.field)], false)
    V = descriptor_field_values(rd, de.field)
    return (; V = V, S = V)
end
function lag_descriptor_inputs(de::Union{ChangeToScale, ChangeInIntensity},
                               rd::ReturnsResult)
    if de.gt0
        assert_panel_field_sign(rd, [String(de.scale)], true)
    end
    return (; V = descriptor_field_values(rd, de.field),
            S = descriptor_field_values(rd, de.scale))
end
"""
    lag_descriptor_lagged(::Union{GrowthRate, ChangeToScale}, V::AbstractMatrix, S::AbstractMatrix)
    lag_descriptor_lagged(::ChangeInIntensity, V::AbstractMatrix, S::AbstractMatrix)

Returns the lagged quantity of each cell: the value that a lag Descriptor reads `lag` observations back, and that a [`LagDescriptorState`](@ref) carries.

It is the field for a [`GrowthRate`](@ref) and a [`ChangeToScale`](@ref), and the ratio of the field to the scale through [`positive_divide`](@ref) for a [`ChangeInIntensity`](@ref).

# Arguments

  - `V`: The values of the field, `observations × assets`.
  - `S`: The values of the scale, `observations × assets`.

# Returns

  - `Q::AbstractMatrix{<:Real}`: The lagged quantity, `observations × assets`.

# Related

  - [`lag_descriptor_fold`](@ref)
  - [`LagDescriptorState`](@ref)
"""
function lag_descriptor_lagged(::Union{GrowthRate, ChangeToScale}, V::AbstractMatrix,
                               ::AbstractMatrix)::AbstractMatrix{<:Real}
    return V
end
function lag_descriptor_lagged(::ChangeInIntensity, V::AbstractMatrix,
                               S::AbstractMatrix)::AbstractMatrix{<:Real}
    return positive_divide.(V, S)
end
"""
    lag_descriptor_value(::GrowthRate, v::Real, s::Real, q::Real)
    lag_descriptor_value(::ChangeToScale, v::Real, s::Real, q::Real)
    lag_descriptor_value(::ChangeInIntensity, v::Real, s::Real, q::Real)

Returns the Descriptor of one cell from the current value of the field, the current value of the scale, and the lagged quantity, with the arithmetic of the mathematical definition of each lag Descriptor.

# Arguments

  - `v`: The current value of the field.
  - `s`: The current value of the scale. A [`GrowthRate`](@ref) ignores it.
  - `q`: The lagged quantity, as [`lag_descriptor_lagged`](@ref) gives it.

# Returns

  - `d::Real`: The Descriptor of the cell.

# Related

  - [`lag_descriptor_fold`](@ref)
  - [`positive_divide`](@ref)
"""
function lag_descriptor_value(::GrowthRate, v::Real, ::Real, q::Real)::Real
    return positive_divide(v, q) - one(v)
end
function lag_descriptor_value(::ChangeToScale, v::Real, s::Real, q::Real)::Real
    return positive_divide(v - q, s)
end
function lag_descriptor_value(::ChangeInIntensity, v::Real, s::Real, q::Real)::Real
    return positive_divide(v, s) - q
end
"""
    lag_state_seed(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity}, ::Nothing,
                   Q::AbstractMatrix)
    lag_state_seed(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity},
                   st::LagDescriptorState, Q::AbstractMatrix)

Returns the buffer that a fold of a lag Descriptor reads its lagged rows from and pushes its new rows onto: an empty buffer of capacity `lag` when the estimator carries no state, or a copy of the buffer of its state that shares its rows.

# Arguments

  - `de`: The lag Descriptor.
  - `st`: The carried state, or `nothing`.
  - `Q`: The lagged quantity of the new observations, `observations × assets`.

# Validation

  - A carried state has the capacity `de.lag`, and holds the assets of `Q`. A `DimensionMismatch` is thrown otherwise.

# Returns

  - `q::DataStructures.CircularBuffer`: A buffer that no other estimator holds.

# Related

  - [`lag_descriptor_fold`](@ref)
  - [`LagDescriptorState`](@ref)
"""
function lag_state_seed(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity}, ::Nothing,
                        Q::AbstractMatrix)
    return DataStructures.CircularBuffer{Vector{eltype(Q)}}(de.lag)
end
function lag_state_seed(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity},
                        st::LagDescriptorState, Q::AbstractMatrix)
    q = st.q
    @argcheck(DataStructures.capacity(q) == de.lag,
              DimensionMismatch("the state of this $(nameof(typeof(de))) carries the rows of a lag of $(DataStructures.capacity(q)), and the lag of the Descriptor is $(de.lag)"))
    @argcheck(isempty(q) || length(q[end]) == size(Q, 2),
              DimensionMismatch("the state of this $(nameof(typeof(de))) carries $(length(q[end])) assets, and the step brings $(size(Q, 2))"))
    return rolling_state_buffer(identity, q)
end
"""
    lag_descriptor_fold(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity},
                        rd::ReturnsResult, st::Option{<:LagDescriptorState})

Folds the observations of a [`ReturnsResult`](@ref) into the state of a lag Descriptor, and returns the Descriptor of each one. The batch call [`descriptor`](@ref) is this fold from no state, and the step [`descriptor_step`](@ref) is this fold from the carried state.

The lagged row of observation `t` is the row `t - lag` of the block when `t > lag`, and otherwise a row of the state. An observation with no lagged row, before the first `lag` observations from the start, stays `NaN`. Each cell runs the arithmetic of [`lag_descriptor_value`](@ref) on the same bits wherever its lagged row comes from, so a folded observation equals the batch call over every observation from the first one to the last bit.

# Algorithm

 1. Read the Panel Fields through [`lag_descriptor_inputs`](@ref), and take their lagged quantity through [`lag_descriptor_lagged`](@ref).
 2. Take the buffer of the state through [`lag_state_seed`](@ref).
 3. Write the Descriptor of each cell that has a lagged row through [`lag_descriptor_value`](@ref).
 4. Push the lagged quantity of the last `lag` observations of the block onto the buffer.
 5. Write `NaN` into the inactive cells through [`descriptor_active_fill!`](@ref).

# Arguments

  - `de`: The lag Descriptor.
  - $(arg_dict[:rd]) It holds the new observations alone.
  - `st`: The carried state, or `nothing` for the batch call.

# Validation

  - The rules of [`lag_descriptor_inputs`](@ref) and [`lag_state_seed`](@ref).

# Returns

  - `fold::NamedTuple`: `st`, the [`LagDescriptorState`](@ref) after the observations, and `D`, the Descriptor of each observation, `observations × assets`.

# Related

  - [`descriptor`](@ref)
  - [`descriptor_step`](@ref)
  - [`LagDescriptorState`](@ref)
"""
function lag_descriptor_fold(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity},
                             rd::ReturnsResult, st::Option{<:LagDescriptorState})
    (; V, S) = lag_descriptor_inputs(de, rd)
    Q = lag_descriptor_lagged(de, V, S)
    q = lag_state_seed(de, st, Q)
    k, lag, T = length(q), de.lag, size(V, 1)
    D = fill(promote_type(eltype(V), eltype(S))(NaN), size(V))
    for i in axes(V, 2), t in max(1, lag - k + 1):T
        j = t - lag
        D[t, i] = lag_descriptor_value(de, V[t, i], S[t, i], j >= 1 ? Q[j, i] : q[k + j][i])
    end
    for t in max(1, T - lag + 1):T
        push!(q, Q[t, :])
    end
    descriptor_active_fill!(D, rd.pnl)
    return (; st = LagDescriptorState(q), D = D)
end
"""
    descriptor_step(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity},
                    rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into the carried state of a lag Descriptor, and returns the Descriptor of each one.

The Descriptor of an observation equals the one of the batch call [`descriptor`](@ref) over every observation that the state folded and the observations before it, to the last bit, as [`lag_descriptor_fold`](@ref) states. The step copies the buffer of the state and no row, so the estimator it gets keeps its state.

# Arguments

  - `de`: The estimator, with or without a state.
  - $(arg_dict[:rd]) It holds the new observations alone.

# Validation

  - The rules of [`lag_descriptor_fold`](@ref).

# Returns

  - `step::NamedTuple`: `de`, the estimator with the state after the observations in `cache`, and `D`, the Descriptor of each observation, `observations × assets`.

# Related

  - [`LagDescriptorState`](@ref)
  - [`partial_fit!`](@ref)
  - [`descriptor_carry`](@ref)
"""
function descriptor_step(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity},
                         rd::ReturnsResult)
    (; st, D) = lag_descriptor_fold(de, rd, de.cache)
    return (; de = Accessors.@set(de.cache = st), D = D)
end
"""
    partial_fit!(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity}, rd::ReturnsResult)

Folds the observations of a [`ReturnsResult`](@ref) into the carried state of a lag Descriptor, and returns the estimator with the state after them in `cache`. [`descriptor_step`](@ref) states the fold, and also returns the Descriptor of each observation.

# Arguments

  - `de`: The estimator, with no state or with its state.
  - $(arg_dict[:rd]) It holds the new observations alone.

# Validation

  - The rules of [`descriptor_step`](@ref).

# Returns

  - `de::Union{GrowthRate, ChangeToScale, ChangeInIntensity}`: The estimator, with its `cache` field set to the state after the observations.

# Related

  - [`descriptor_step`](@ref)
  - [`LagDescriptorState`](@ref)
"""
function partial_fit!(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity},
                      rd::ReturnsResult)
    return descriptor_step(de, rd).de
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Renders every field of a [`GrowthRate`](@ref), a [`ChangeToScale`](@ref) or a [`ChangeInIntensity`](@ref) except `cache`.

The state a `cache` holds is the running detail of an incremental fit, not the configuration a reader looks the type up for, and it prints under the estimator at every site that renders one, such as a [`CompositeExposure`](@ref). Set `set_show_nothing_fields!(:GrowthRate, true)` to render it.

# Arguments

  - `de`: The estimator.

# Returns

  - `fields::Tuple`: The field names to render, every field name but `:cache`.

# Related

  - [`GrowthRate`](@ref)
  - [`show_fields`](@ref)
  - [`set_show_nothing_fields!`](@ref)
"""
function show_fields(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity})
    return filter(!=(:cache), fieldnames(typeof(de)))
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

The value is `(net_income_ttm(t) - net_income_ttm(t - lag)) / market_cap(t)`, `NaN` on the first `lag` observations. A market capitalisation at or below zero is a data error, and the estimator raises on one. It is the earnings momentum Descriptor, and it stays well defined through a loss, where a growth rate of the earnings would not.

# Arguments

  - `field`: Name of the net income Panel Field.
  - `scale`: Name of the market capitalisation Panel Field.
  - `lag`: Number of observations to look back. `252` is one year of daily observations.

# Returns

  - `de::ChangeToScale`: The estimator, with the two Panel Fields and the lag fixed, and with `gt0 = true`.

# Examples

```jldoctest
julia> EarningsChangeToPrice()
ChangeToScale
  field ┼ String: \"net_income_ttm\"
  scale ┼ String: \"market_cap\"
    lag ┼ Int64: 252
    gt0 ┴ Bool: true
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
    return ChangeToScale(; field = field, scale = scale, lag = lag, gt0 = true)
end
"""
    CapexToAssetsChangeInIntensity(; field::AbstractString = "capex_ttm",
                                   scale::AbstractString = "total_assets",
                                   lag::Integer = 252) -> ChangeInIntensity

Change of the capital expenditure to total assets ratio over one year.

The value is `capex_ttm(t) / total_assets(t) - capex_ttm(t - lag) / total_assets(t - lag)`, `NaN` on the first `lag` observations. Total assets at or below zero are a data error, and the estimator raises on them. A positive value says that the firm invests a larger share of its assets than a year ago.

# Arguments

  - `field`: Name of the capital expenditure Panel Field.
  - `scale`: Name of the total assets Panel Field.
  - `lag`: Number of observations to look back. `252` is one year of daily observations.

# Returns

  - `de::ChangeInIntensity`: The estimator, with the two Panel Fields and the lag fixed, and with `gt0 = true`.

# Examples

```jldoctest
julia> CapexToAssetsChangeInIntensity()
ChangeInIntensity
  field ┼ String: \"capex_ttm\"
  scale ┼ String: \"total_assets\"
    lag ┼ Int64: 252
    gt0 ┴ Bool: true
```

# Related

  - [`ChangeInIntensity`](@ref)
  - [`descriptor`](@ref)
  - [`AssetsGrowthRate`](@ref)
"""
function CapexToAssetsChangeInIntensity(; field::AbstractString = "capex_ttm",
                                        scale::AbstractString = "total_assets",
                                        lag::Integer = 252)::ChangeInIntensity
    return ChangeInIntensity(; field = field, scale = scale, lag = lag, gt0 = true)
end

export GrowthRate, ChangeToScale, ChangeInIntensity, AssetsGrowthRate, SalesGrowthRate,
       IssuanceGrowthRate, EarningsChangeToPrice, CapexToAssetsChangeInIntensity
