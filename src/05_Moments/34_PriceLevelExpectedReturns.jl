"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the statistics of a window of price levels that [`PriceLevelExpectedReturns`](@ref) turns into an expected return.

Every statistic is read on the reconstructed price path: the level of the last observation is set to one in every asset, the earlier levels are reconstructed from the returns of the window, the statistic is then a vector over the assets, and the expected return is `stat / p_t .- 1` with `p_t = 1`. A statistic that acts on each asset's levels alone — the moving averages, the peak, the lagged price, the reweighted relative and the trend switches — is homogeneous of degree one in that asset's levels, so it reads the same off the path as off the prices. A statistic that couples the assets — the spatial median through its Euclidean distances, the kernel trend pattern through its regression — is not invariant to scaling each asset by its own last price, so on the path it is the paper's statistic taken over prices normalised to one at the current period, not over the prices themselves; each such statistic's docstring states it. A statistic is **windowed** when it reads a fixed number of levels ([`MovingAverage`](@ref), [`SpatialMedian`](@ref), [`WindowPeak`](@ref), [`LaggedPrice`](@ref)) and **folding** when it is an exact recursion over the price relatives ([`ExponentialMovingAverage`](@ref), [`ReweightedPriceRelative`](@ref)): a folding statistic carries one vector on a Partial Fit State and reads no rows, and its batch form over a matrix of returns is the same recursion seeded at the first level, so a fold and a batch over the same rows agree exactly. A folding statistic may also carry a **memory**, the last [`memory_rows`](@ref) relatives on the same state ([`KernelTrendPattern`](@ref)): its recursion reads the carried vector and the memory together, so it is self-contained like any fold — a host holds no rows for it — and its batch form is the recursion from the first row with the memory filling as it goes.

Over the first rows a windowed statistic **truncates its window** to the levels available, as the reversion papers' reference implementations do: with `window = 5` and two returns folded, the statistic reads three levels. A folding statistic starts from its seed at the first row. A composite statistic truncates every window it holds the same way.

A folding statistic is **mask-aware**: [`partial_fit!`](@ref) reads the active mask of the row, folds the active assets alone, and resets an inactive asset to [`cold_statistic`](@ref), so a relisting starts cold rather than warm. An asset that has folded no level since its last reset is `NaN` at the read-out. The reduction is the caller's, so a statistic that couples the assets reads the active assets alone and needs no mask of its own.

# Interfaces

In order to implement a new price-level statistic, subtype `AbstractPriceLevelStatistic` with all its parameters as part of the struct, and implement the following methods:

  - `price_level_statistic(alg::AbstractPriceLevelStatistic, P::AbstractMatrix) -> AbstractVector`: The statistic of the levels `P`, `levels × assets`, whose last row is the current level, one per asset. A folding statistic runs its recursion over every row of `P` from the first.
  - `window_rows(alg::AbstractPriceLevelStatistic) -> Union{Nothing, Integer}`: The number of return rows the batch form reads to reconstruct its levels, `window - 1` for a statistic over `window` levels, and `nothing` for one that reads every row it is handed. The default reads `alg.window - 1`.
  - `folds(alg::AbstractPriceLevelStatistic) -> Bool`: `true` when the statistic is an exact recursion over price relatives, carried by [`fold_statistic`](@ref). The default is `false`.
  - `fold_statistic(alg::AbstractPriceLevelStatistic, stat, x::AbstractVector) -> AbstractVector`: For a folding statistic, the recursion: from the carried statistic `stat` in relative terms (`nothing` before the first row) and the price relative `x` of the row, the statistic after the row, as a new vector.
  - `cold_statistic(alg::AbstractPriceLevelStatistic, x::AbstractVector) -> AbstractVector`: For a folding statistic, the carried statistic before the first row, in the units the recursion reads it: the value that makes the recursion answer its own seed. The default is one in every asset.
  - `memory_rows(alg::AbstractPriceLevelStatistic) -> Integer`: The number of relatives a folding statistic carries as its memory, `0` for one whose recursion reads the carried vector alone, which is the default. A statistic with a memory implements the four-argument `fold_statistic(alg, stat, hist, x)` instead, `hist` being the last `memory_rows(alg)` relatives, the row's own last, or fewer over the first rows.

## Arguments

  - `alg`: The concrete statistic.
  - `P`: The reconstructed price levels of the window, `levels × assets`, the last row being ones.
  - `stat`: The carried statistic, in units of the last level, or `nothing`.
  - `hist`: The carried memory of relatives, `rows × assets`, the last row being `x`.
  - `x`: The price relative of one row, `1 .+ r`.

## Returns

  - `stat::AbstractVector`: The statistic per asset, `assets × 1`.
  - `rows::Union{Nothing, Integer}`: The number of return rows the batch form reads.
  - `folds::Bool`: Whether the statistic folds.
  - `memory::Integer`: The number of relatives the memory holds.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`MovingAverage`](@ref)
  - [`ExponentialMovingAverage`](@ref)
  - [`SpatialMedian`](@ref)
  - [`WindowPeak`](@ref)
  - [`LaggedPrice`](@ref)
  - [`ReweightedPriceRelative`](@ref)
  - [`KernelTrendPattern`](@ref)
  - [`price_level_statistic`](@ref)
  - [`fold_statistic`](@ref)
  - [`memory_rows`](@ref)
  - [`rows_needed`](@ref)
"""
abstract type AbstractPriceLevelStatistic <: AbstractExpectedReturnsAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

The simple moving average of the last `window` price levels, the forecast of the moving-average reversion of Li and Hoi (2012).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    MovingAverage(; window::Integer = 5) -> MovingAverage

Keywords correspond to the struct's fields.

## Validation

  - `window >= 2`. A `DomainError` is thrown otherwise: one level is the current price, and its ratio to itself forecasts nothing.

# Examples

```jldoctest
julia> MovingAverage()
MovingAverage
  window ┴ Int64: 5
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`MovingAverageReversion`](@ref)
"""
struct MovingAverage{T1 <: Integer} <: AbstractPriceLevelStatistic
    """
    $(field_dict[:price_window])
    """
    window::T1
    function MovingAverage(window::Integer)
        assert_price_window(window)
        return new{typeof(window)}(window)
    end
end
function MovingAverage(; window::Integer = 5)::MovingAverage
    return MovingAverage(window)
end
"""
$(DocStringExtensions.TYPEDEF)

The exponential moving average of the price levels, the forecast of the second form of the moving-average reversion of Li, Hoi, Sahoo and Liu (2015): a folding statistic.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MA}_{t} &= \\alpha \\boldsymbol{p}_{t} + (1 - \\alpha) \\mathrm{MA}_{t-1}\\,,\\quad \\mathrm{MA}_{1} = \\boldsymbol{p}_{1}\\,,\\\\
\\hat{\\boldsymbol{x}}_{t+1} &= \\frac{\\mathrm{MA}_{t}}{\\boldsymbol{p}_{t}} = \\alpha + (1 - \\alpha)\\, \\hat{\\boldsymbol{x}}_{t} \\oslash \\boldsymbol{x}_{t}\\,,\\quad \\hat{\\boldsymbol{x}}_{1} = \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{p}_{t}``: Price levels of observation ``t``, per asset.
  - ``\\boldsymbol{x}_{t}``: Price relative of observation ``t``, ``\\boldsymbol{p}_{t} \\oslash \\boldsymbol{p}_{t-1}``.
  - ``\\alpha``: The smoothing weight on the current level.
  - ``\\hat{\\boldsymbol{x}}_{t+1}``: The Price Relative Forecast, per asset.

The recursion is exact in relative terms, so the statistic reads no rows and carries one vector. The paper seeds the average at the first price, so the forecast after one level is one — the same value the closed expansion ``\\hat{\\boldsymbol{x}}_{2} = \\boldsymbol{1}`` gives — and the two readings of the one-step seed coincide from the second level on. The paper states no value of ``\\alpha``; `0.5` is the middle of the plateau its sensitivity study reports and is the library's choice.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ExponentialMovingAverage(; alpha::Real = 0.5) -> ExponentialMovingAverage

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha <= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ExponentialMovingAverage()
ExponentialMovingAverage
  alpha ┴ Float64: 0.5
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`ExponentialMovingAverageReversion`](@ref)
  - [`fold_statistic`](@ref)

# References

  - $(ref_dict[:li2015olmar])
"""
struct ExponentialMovingAverage{T1 <: Real} <: AbstractPriceLevelStatistic
    """
    The smoothing weight on the current level, `0 < alpha <= 1`; smaller is a longer memory.
    """
    alpha::T1
    function ExponentialMovingAverage(alpha::Real)
        @argcheck(zero(alpha) < alpha <= one(alpha),
                  DomainError(alpha, "alpha must be in (0, 1]"))
        return new{typeof(alpha)}(alpha)
    end
end
function ExponentialMovingAverage(; alpha::Real = 0.5)::ExponentialMovingAverage
    return ExponentialMovingAverage(alpha)
end
"""
$(DocStringExtensions.TYPEDEF)

The spatial (``L_1``) median of the last `window` price levels, the forecast of the robust median reversion of Huang, Zhou, Li, Hoi and Zhou (2016).

The median is the point minimising the sum of Euclidean distances to the `window` level vectors, found by the modified Weiszfeld iteration of Vardi and Zhang (2000), which is seeded at the coordinatewise median and stays defined when an iterate lands on one of the level vectors.

The distances couple the assets, so the median is not invariant to scaling each asset by its own price: it is read on the reconstructed path, every asset's last level at one, where it is the paper's median taken over prices normalised to the current period rather than over the prices themselves. The two differ — by a few hundredths in the Price Relative Forecast when one asset trades at a hundred times the price of the others — and the normalised one is scale-free, so no asset dominates the distances by its quotation. The paper stops its iteration at a relative change of the iterate's ``L_1`` norm, within two hundred iterations; the iteration here runs to an absolute Euclidean change of `tol`, which at the default is the minimiser to machine precision where the paper's rule leaves the forecast a few thousandths away.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SpatialMedian(; window::Integer = 5, iters::Integer = 100, tol::Real = 1e-8) -> SpatialMedian

Keywords correspond to the struct's fields.

## Validation

  - `window >= 2`. A `DomainError` is thrown otherwise.
  - `iters >= 1`. A `DomainError` is thrown otherwise.
  - `tol > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> SpatialMedian()
SpatialMedian
  window ┼ Int64: 5
   iters ┼ Int64: 100
     tol ┴ Float64: 1.0e-8
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`RobustMedianReversion`](@ref)
  - [`spatial_median`](@ref)
"""
struct SpatialMedian{T1 <: Integer, T2 <: Integer, T3 <: Real} <:
       AbstractPriceLevelStatistic
    """
    $(field_dict[:price_window])
    """
    window::T1
    """
    Maximum number of Weiszfeld iterations.
    """
    iters::T2
    """
    Convergence tolerance on the Euclidean distance between two successive iterates.
    """
    tol::T3
    function SpatialMedian(window::Integer, iters::Integer, tol::Real)
        assert_price_window(window)
        @argcheck(iters >= 1, DomainError(iters, "iters must be at least 1"))
        @argcheck(tol > zero(tol), DomainError(tol, "tol must be positive"))
        return new{typeof(window), typeof(iters), typeof(tol)}(window, iters, tol)
    end
end
function SpatialMedian(; window::Integer = 5, iters::Integer = 100,
                       tol::Real = 1e-8)::SpatialMedian
    return SpatialMedian(window, iters, tol)
end
"""
$(DocStringExtensions.TYPEDEF)

The peak of the last `window` price levels, the forecast of the peak price tracking of Lai, Dai, Ren and Huang (2018) and of the short-term sparse portfolio of Lai, Yang, Fang and Wu (2018).

The peak is never below the last price, so the Price Relative Forecast is at least one in every asset, ``\\hat{x}_{t+1, i} = \\max_{0 \\leq k < w} p_{t-k, i} / p_{t, i} \\geq 1``, and it is exactly one for an asset sitting at its own window peak.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WindowPeak(; window::Integer = 5) -> WindowPeak

Keywords correspond to the struct's fields.

## Validation

  - `window >= 2`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> WindowPeak()
WindowPeak
  window ┴ Int64: 5
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`PeakPriceTracking`](@ref)
  - [`ShortTermSparsePortfolio`](@ref)

# References

  - $(ref_dict[:lai2018ppt])
  - $(ref_dict[:lai2018sspo])
"""
struct WindowPeak{T1 <: Integer} <: AbstractPriceLevelStatistic
    """
    $(field_dict[:price_window])
    """
    window::T1
    function WindowPeak(window::Integer)
        assert_price_window(window)
        return new{typeof(window)}(window)
    end
end
function WindowPeak(; window::Integer = 5)::WindowPeak
    return WindowPeak(window)
end
"""
$(DocStringExtensions.TYPEDEF)

The price level `lag` observations ago, the forecast of the first form of the transaction-cost optimisation of Li, Wang, Huang and Hoi (2018): a bet that the last move reverts, ``\\hat{\\boldsymbol{x}}_{t+1} = \\boldsymbol{p}_{t - \\ell} \\oslash \\boldsymbol{p}_{t}``, which at `lag = 1` is `1 ./ x_t`.

At `lag = 0` the forecast is the current price, so the Price Relative Forecast is one in every asset: the *hold* branch of a switched statistic.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LaggedPrice(; lag::Integer = 1) -> LaggedPrice

Keywords correspond to the struct's fields.

## Validation

  - `lag >= 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> LaggedPrice()
LaggedPrice
  lag ┴ Int64: 1
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`TransactionCostOptimisation`](@ref)

# References

  - $(ref_dict[:li2018tco])
"""
struct LaggedPrice{T1 <: Integer} <: AbstractPriceLevelStatistic
    """
    The number of observations between the forecast level and the current one, `0` for the current level.
    """
    lag::T1
    function LaggedPrice(lag::Integer)
        @argcheck(lag >= 0, DomainError(lag, "lag must be non-negative"))
        return new{typeof(lag)}(lag)
    end
end
function LaggedPrice(; lag::Integer = 1)::LaggedPrice
    return LaggedPrice(lag)
end
"""
$(DocStringExtensions.TYPEDEF)

The reweighted price relative of Lai, Yang, Fang and Wu (2018), as restated with equations by Li, Luo and Xu (2023): an exponential moving average of the price levels whose smoothing weight is per asset and data dependent, a folding statistic.

# Mathematical definition

```math
\\begin{align}
\\gamma_{t+1, i} &= \\frac{\\theta x_{t, i}}{\\theta x_{t, i} + \\hat{\\varphi}_{t, i}}\\,,\\quad
\\hat{\\boldsymbol{\\varphi}}_{t+1} = \\boldsymbol{\\gamma}_{t+1} + (\\boldsymbol{1} - \\boldsymbol{\\gamma}_{t+1}) \\odot \\hat{\\boldsymbol{\\varphi}}_{t} \\oslash \\boldsymbol{x}_{t}\\,,\\quad
\\hat{\\boldsymbol{\\varphi}}_{1} = \\boldsymbol{x}_{1}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{x}_{t}``: Price relative of observation ``t``, per asset.
  - ``\\theta``: The reweighting strength; larger puts more weight on the current relative.
  - ``\\hat{\\boldsymbol{\\varphi}}_{t+1}``: The Price Relative Forecast, per asset.

[`ExponentialMovingAverage`](@ref) is the case of a constant weight ``\\boldsymbol{\\gamma} = \\alpha \\boldsymbol{1}``. The paper's own defaults for ``\\theta`` are not read; the restating paper's `theta = 0.7` is the library's default on [`ReweightedPriceRelativeTracking`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReweightedPriceRelative(; theta::Real = 0.7) -> ReweightedPriceRelative

Keywords correspond to the struct's fields.

## Validation

  - `theta > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ReweightedPriceRelative()
ReweightedPriceRelative
  theta ┴ Float64: 0.7
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`ExponentialMovingAverage`](@ref)
  - [`ReweightedPriceRelativeTracking`](@ref)
  - [`fold_statistic`](@ref)

# References

  - $(ref_dict[:lai2018rprt])
  - $(ref_dict[:liluoxu2023])
"""
struct ReweightedPriceRelative{T1 <: Real} <: AbstractPriceLevelStatistic
    """
    The reweighting strength `theta > 0` on the current price relative.
    """
    theta::T1
    function ReweightedPriceRelative(theta::Real)
        @argcheck(theta > zero(theta), DomainError(theta, "theta must be positive"))
        return new{typeof(theta)}(theta)
    end
end
function ReweightedPriceRelative(; theta::Real = 0.7)::ReweightedPriceRelative
    return ReweightedPriceRelative(theta)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a price-level window below two levels.

# Related

  - [`MovingAverage`](@ref)
  - [`SpatialMedian`](@ref)
  - [`WindowPeak`](@ref)
"""
function assert_price_window(window::Integer)::Nothing
    @argcheck(window >= 2,
              DomainError(window,
                          "window must be at least 2: one level is the current price, and its ratio to itself forecasts nothing"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

The Partial Fit State of a [`PriceLevelExpectedReturns`](@ref) whose statistic folds: the statistic in units of the last level, one entry per asset, the number of rows folded, the number of levels each asset has folded since its last reset, and the memory of a statistic that carries one.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`AbstractPartialFitState`](@ref)
  - [`fold_statistic`](@ref)
  - [`memory_rows`](@ref)
"""
@concrete struct PriceLevelForecastState <: AbstractPartialFitState
    """
    $(field_dict[:pf_n])
    """
    n
    """
    The number of levels each asset has folded since its last reset, `assets × 1`. An asset at zero carries the cold seed, and the read-out answers `NaN` for it.
    """
    nu
    """
    The folded statistic divided by the current level, `assets × 1`: the Price Relative Forecast.
    """
    stat
    """
    The last [`memory_rows`](@ref) relatives folded, `rows × assets`, the newest last, or `nothing` for a statistic that carries no memory.
    """
    hist
end
function merge_states(::PriceLevelForecastState, ::PriceLevelForecastState)
    return throw(ArgumentError("a `PriceLevelForecastState` is not merged: the recursion it carries is order-dependent, so two states folded on different rows have no common continuation. Fold the rows of one into the other."))
end
function Base.copy(x::PriceLevelForecastState)
    return PriceLevelForecastState(x.n, copy(x.nu), copy(x.stat),
                                   isnothing(x.hist) ? nothing : copy(x.hist))
end
function port_opt_view(x::PriceLevelForecastState, i, args...)
    return PriceLevelForecastState(x.n, x.nu[i], x.stat[i],
                                   isnothing(x.hist) ? nothing : x.hist[:, i])
end
"""
$(DocStringExtensions.TYPEDEF)

The expected return that takes the last price to a statistic of a window of price levels: the Price Relative Forecast of the online portfolio selection family, as an expected-returns estimator any `me` slot may hold.

The statistics of the literature — a moving average, an exponential moving average, a spatial median, a window peak, a lagged price, a reweighted relative and the composites over them — are homogeneous of degree one in the levels, so the forecast is a function of the returns alone: the level of the last observation is set to one, the earlier levels are reconstructed from the returns, and the expected return is the statistic divided by the last level, less one. The mean is therefore a forecast of the **next** period's return under the reversion or trend hypothesis of the paper that defines the statistic, not an average of past returns.

A windowed statistic reads its last `window - 1` returns and truncates the window over the first rows, as the reversion papers' reference implementations do. A folding statistic ([`ExponentialMovingAverage`](@ref), [`ReweightedPriceRelative`](@ref)) is an exact recursion over the price relatives: [`partial_fit!`](@ref) folds one row into the state the `cache` field carries and `mean(me)` reads it, so an online host carries one vector for it and no rows; its batch form over a matrix is the same recursion seeded at the first level, and the two agree exactly over the same rows.

# Mathematical definition

```math
\\begin{align}
p_{t} &= \\boldsymbol{1}\\,,\\quad p_{t-k} = p_{t-k+1} \\oslash (\\boldsymbol{1} + \\boldsymbol{r}_{t-k+1})\\,,\\quad k = 1, \\ldots, w - 1\\,,\\\\
\\hat{\\boldsymbol{\\mu}} &= \\mathrm{stat}(p_{t-w+1}, \\ldots, p_{t}) \\oslash p_{t} - \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - ``p_{t-k}``: Reconstructed price levels ``k`` periods before the last observation, per asset.
  - ``\\boldsymbol{r}_{t}``: Returns of observation ``t``, per asset.
  - ``w``: The window of the statistic, in levels.
  - ``\\mathrm{stat}``: The statistic on `alg`, a vector over the assets.
  - ``\\hat{\\boldsymbol{\\mu}}``: The expected return, per asset.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriceLevelExpectedReturns(;
        alg::AbstractPriceLevelStatistic = MovingAverage(),
        cache::Option{<:PriceLevelForecastState} = nothing
    ) -> PriceLevelExpectedReturns

Keywords correspond to the struct's fields.

## View parameters

When [`port_opt_view`](@ref) is called on this type, `cache` is sliced to the selected assets through the state's own view.

# Examples

```jldoctest
julia> PriceLevelExpectedReturns()
PriceLevelExpectedReturns
  alg ┼ MovingAverage
      │   window ┴ Int64: 5
```

# Related

  - [`AbstractExpectedReturnsEstimator`](@ref)
  - [`AbstractPriceLevelStatistic`](@ref)
  - [`PriceLevelForecastState`](@ref)
  - [`MovingAverage`](@ref)
  - [`SpatialMedian`](@ref)
  - [`ForecastReversion`](@ref)
  - [`rows_needed`](@ref)
  - [`partial_fit!`](@ref)
"""
struct PriceLevelExpectedReturns{T1 <: AbstractPriceLevelStatistic,
                                 T2 <: Option{<:PriceLevelForecastState}} <:
       AbstractExpectedReturnsEstimator
    """
    The price-level statistic the forecast reads.
    """
    alg::T1
    """
    $(field_dict[:pfcache])
    """
    cache::T2
    function PriceLevelExpectedReturns(alg::AbstractPriceLevelStatistic,
                                       cache::Option{<:PriceLevelForecastState})
        return new{typeof(alg), typeof(cache)}(alg, cache)
    end
end
function PriceLevelExpectedReturns(; alg::AbstractPriceLevelStatistic = MovingAverage(),
                                   cache::Option{<:PriceLevelForecastState} = nothing)::PriceLevelExpectedReturns
    return PriceLevelExpectedReturns(alg, cache)
end
# The state a `cache` holds is the running detail of a fold, not the configuration a reader
# looks the type up for (see [`show_fields`](@ref)).
show_fields(::PriceLevelExpectedReturns) = (:alg,)
function port_opt_view(me::PriceLevelExpectedReturns, i, args...)
    return PriceLevelExpectedReturns(; alg = me.alg,
                                     cache = if isnothing(me.cache)
                                         nothing
                                     else
                                         port_opt_view(me.cache, i, args...)
                                     end)
end
"""
    window_rows(alg::AbstractPriceLevelStatistic)
    window_rows(alg::ExponentialMovingAverage)
    window_rows(alg::LaggedPrice)
    window_rows(alg::ReweightedPriceRelative)

The number of return rows the batch form of a statistic reads, `nothing` for one that reads every row it is handed.

A statistic over `window` levels reads `window - 1` rows, which is the default; the lagged price reads `lag`; a folding statistic reads every row, because its batch form is the recursion from the first level.

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`rows_needed`](@ref)
"""
function window_rows(alg::AbstractPriceLevelStatistic)
    return alg.window - 1
end
function window_rows(::ExponentialMovingAverage)
    return nothing
end
function window_rows(alg::LaggedPrice)
    return alg.lag
end
function window_rows(::ReweightedPriceRelative)
    return nothing
end
"""
    folds(alg::AbstractPriceLevelStatistic)
    folds(alg::ExponentialMovingAverage)
    folds(alg::ReweightedPriceRelative)

Whether a statistic is an exact recursion over price relatives. `false` by default.

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`fold_statistic`](@ref)
"""
function folds(::AbstractPriceLevelStatistic)
    return false
end
function folds(::ExponentialMovingAverage)
    return true
end
function folds(::ReweightedPriceRelative)
    return true
end
"""
    memory_rows(alg::AbstractPriceLevelStatistic)

The number of relatives a folding statistic carries as its memory on the state, `0` for one whose recursion reads the carried vector alone, which is the default.

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`fold_statistic`](@ref)
  - [`PriceLevelForecastState`](@ref)
"""
function memory_rows(::AbstractPriceLevelStatistic)
    return 0
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The memory after a row: the last [`memory_rows`](@ref) relatives with the row appended, `nothing` for a statistic that carries none.

# Arguments

  - `alg`: The statistic.
  - `hist`: The carried memory, or `nothing`.
  - `x`: The price relative of the row.

# Returns

  - `hist'::Union{Nothing, Matrix}`: The memory after the row, a new matrix.

# Related

  - [`memory_rows`](@ref)
  - [`fold_statistic`](@ref)
"""
function push_memory(alg::AbstractPriceLevelStatistic, hist::Option{<:AbstractMatrix},
                     x::AbstractVector)
    m = memory_rows(alg)
    if iszero(m)
        return nothing
    end
    row = reshape(collect(x), 1, :)
    if isnothing(hist)
        return row
    end
    H = vcat(hist, row)
    return H[max(1, size(H, 1) - m + 1):end, :]
end
"""
    rows_needed(me::PriceLevelExpectedReturns)
    rows_needed(me::AbstractExpectedReturnsEstimator)
    rows_needed(me::WindowedExpectedReturns)
    rows_needed(alg::AbstractPriceLevelStatistic)

The number of return rows an expected-returns estimator reads at a step of the online portfolio selection family, `1` for one that folds, or `nothing` when it reads every row folded so far.

A windowed price-level statistic over `window` levels reads `window - 1` returns, and a folding one reads the current row alone, because its state is one vector and the row it folds is the head's row verbatim — its gaps and its active mask included — which the head hands it as a one-row carrier rather than the finite price relative of the step. An estimator with an exact fold of its own — one for which [`supports_partial_fit`](@ref) answers `true` — reads the current row on the same terms, and a [`WindowedExpectedReturns`](@ref) reads its window. Any other expected-returns estimator answers `nothing`, which is unbounded: the head keeps every row for it and it refits on the whole prefix at every step, at `O(tN)` a step after `t` rows. [`ForecastReversion`](@ref) forwards to its slots, and the head takes the maximum over its rule tree.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`AbstractPriceLevelStatistic`](@ref)
  - [`ForecastReversion`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
"""
function rows_needed(me::PriceLevelExpectedReturns)
    return rows_needed(me.alg)
end
function rows_needed(me::AbstractExpectedReturnsEstimator)
    return supports_partial_fit(me) ? 1 : nothing
end
function rows_needed(me::WindowedExpectedReturns)
    return me.window
end
function rows_needed(alg::AbstractPriceLevelStatistic)
    return folds(alg) ? 1 : window_rows(alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Reconstructs the price levels of a window of returns, the last level set to one.

Row `k` of the answer is the level `k` observations before the end of `X` — the last row is ones, the row above it `1 ./ (1 .+ r_T)`, and so on back to the first row of `X` — so the answer holds `size(X, 1) + 1` rows and the statistic reads the last `window` of them, or all of them when the window truncates.

# Arguments

  - `X`: Returns, `observations × assets`.

# Returns

  - `P::Matrix`: The levels, `(observations + 1) × assets`, the last row ones.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`price_level_statistic`](@ref)
"""
function price_levels(X::MatNum)
    T, N = size(X)
    P = Matrix{typeof(one(eltype(X)) / one(eltype(X)))}(undef, T + 1, N)
    P[T + 1, :] .= one(eltype(P))
    for t in T:-1:1
        @views P[t, :] .= P[t + 1, :] ./ (one(eltype(P)) .+ X[t, :])
    end
    return P
end
"""
    price_level_statistic(alg::MovingAverage, P::AbstractMatrix)
    price_level_statistic(alg::ExponentialMovingAverage, P::AbstractMatrix)
    price_level_statistic(alg::SpatialMedian, P::AbstractMatrix)
    price_level_statistic(alg::WindowPeak, P::AbstractMatrix)
    price_level_statistic(alg::LaggedPrice, P::AbstractMatrix)
    price_level_statistic(alg::ReweightedPriceRelative, P::AbstractMatrix)

The statistic of a window of price levels, one value per asset, homogeneous of degree one in the levels.

The moving average is the column mean of the levels. The exponential moving average is the recursion over every row from the first. The spatial median is the point minimising the sum of Euclidean distances to the level vectors, through [`spatial_median`](@ref). The window peak is the column maximum. The lagged price is the row `lag` above the last, or the first row when the window truncates. The reweighted relative is its recursion over the relatives of successive rows through [`fold_levels`](@ref), times the last level.

# Arguments

  - `alg`: The statistic.
  - `P`: The levels, `levels × assets`, the last row the current level.

# Returns

  - `stat::Vector`: The statistic per asset.

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`fold_statistic`](@ref)
  - [`fold_levels`](@ref)
  - [`spatial_median`](@ref)
"""
function price_level_statistic(::MovingAverage, P::AbstractMatrix)
    return vec(Statistics.mean(P; dims = 1))
end
function price_level_statistic(alg::ExponentialMovingAverage, P::AbstractMatrix)
    ma = P[1, :] .* one(alg.alpha)
    for t in 2:size(P, 1)
        @views ma .= alg.alpha .* P[t, :] .+ (one(alg.alpha) - alg.alpha) .* ma
    end
    return ma
end
function price_level_statistic(alg::SpatialMedian, P::AbstractMatrix)
    return spatial_median(P, alg.iters, alg.tol)
end
function price_level_statistic(::WindowPeak, P::AbstractMatrix)
    return vec(maximum(P; dims = 1))
end
function price_level_statistic(alg::LaggedPrice, P::AbstractMatrix)
    return P[max(1, size(P, 1) - alg.lag), :]
end
function price_level_statistic(alg::ReweightedPriceRelative, P::AbstractMatrix)
    return fold_levels(alg, P)
end
"""
    cold_statistic(alg::AbstractPriceLevelStatistic, x::AbstractVector)
    cold_statistic(alg::ReweightedPriceRelative, x::AbstractVector)
    cold_statistic(alg::KernelTrendPattern, x::AbstractVector)

The carried statistic of a folding statistic before its first row, in the units its own recursion reads: the value that makes [`fold_statistic`](@ref) answer the seed the statistic states.

The exponential moving average seeds its forecast at one and reads the carried vector as it stands, so its cold seed is one in every asset. The reweighted relative seeds at the row's own relative, and the kernel trend pattern divides the carried prediction by that relative to bring it into the current level's units, so both seed at `x`.

This is the value [`partial_fit!`](@ref) writes at an asset the active mask turns off, so that a relisting re-enters the recursion cold.

# Arguments

  - `alg`: The folding statistic.
  - `x`: The price relative of the row, `1 .+ r`.

# Returns

  - `stat::Vector`: The cold seed per asset, a new vector.

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`fold_statistic`](@ref)
  - [`partial_fit!`](@ref)
"""
function cold_statistic(::AbstractPriceLevelStatistic, x::AbstractVector)
    return fill(one(eltype(x)), length(x))
end
function cold_statistic(::ReweightedPriceRelative, x::AbstractVector)
    return collect(x)
end
"""
    fold_statistic(alg::ExponentialMovingAverage, stat, x::AbstractVector)
    fold_statistic(alg::ReweightedPriceRelative, stat, x::AbstractVector)
    fold_statistic(alg::KernelTrendPattern, stat, hist::AbstractMatrix, x::AbstractVector)

One row of a folding statistic's recursion, in units of the current level.

The exponential moving average steps ``\\hat{\\boldsymbol{x}}' = \\alpha + (1 - \\alpha)\\, \\hat{\\boldsymbol{x}} \\oslash \\boldsymbol{x}`` from ``\\boldsymbol{1}``. The reweighted relative is seeded at the first row's relative and steps ``\\hat{\\boldsymbol{\\varphi}}' = \\boldsymbol{\\gamma} + (\\boldsymbol{1} - \\boldsymbol{\\gamma}) \\odot \\hat{\\boldsymbol{\\varphi}} \\oslash \\boldsymbol{x}`` with ``\\boldsymbol{\\gamma} = \\theta \\boldsymbol{x} \\oslash (\\theta \\boldsymbol{x} + \\hat{\\boldsymbol{\\varphi}})``, so its forecast after the first row is one. The four-argument form is the recursion of a statistic with a memory, handed the last [`memory_rows`](@ref) relatives with the row appended; the fold and the batch choose the arity by the memory. The kernel trend pattern's recursion is stated on [`KernelTrendPattern`](@ref).

# Arguments

  - `alg`: The statistic.
  - `stat`: The carried statistic, or `nothing` before the first row.
  - `hist`: The carried memory with the row appended, or `nothing`.
  - `x`: The price relative of the row, `1 .+ r`.

# Returns

  - `stat'::Vector`: The statistic after the row, a new vector.

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`ExponentialMovingAverage`](@ref)
  - [`ReweightedPriceRelative`](@ref)
  - [`KernelTrendPattern`](@ref)
  - [`PriceLevelForecastState`](@ref)
  - [`memory_rows`](@ref)
"""
function fold_statistic(alg::ExponentialMovingAverage, stat::Option{<:AbstractVector},
                        x::AbstractVector)
    prev = isnothing(stat) ? cold_statistic(alg, x) : stat
    return alg.alpha .+ (one(alg.alpha) - alg.alpha) .* prev ./ x
end
function fold_statistic(alg::ReweightedPriceRelative, stat::Option{<:AbstractVector},
                        x::AbstractVector)
    prev = isnothing(stat) ? cold_statistic(alg, x) : stat
    gamma = alg.theta .* x ./ (alg.theta .* x .+ prev)
    return gamma .+ (one(eltype(gamma)) .- gamma) .* prev ./ x
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

One row of a folding statistic's recursion, at the arity its memory asks for: the three-argument [`fold_statistic`](@ref) for a statistic that carries no memory, and the four-argument one for a statistic that carries one.

# Arguments

  - `alg`: The folding statistic.
  - `stat`: The carried statistic, or `nothing` before the first row.
  - `hist`: The carried memory with the row appended, or `nothing`.
  - `x`: The price relative of the row, `1 .+ r`.

# Returns

  - `stat'::AbstractVector`: The statistic after the row, a new vector.

# Related

  - [`fold_statistic`](@ref)
  - [`fold_active`](@ref)
  - [`fold_levels`](@ref)
"""
function fold_statistic_row(alg::AbstractPriceLevelStatistic,
                            stat::Option{<:AbstractVector}, hist::Option{<:AbstractMatrix},
                            x::AbstractVector)
    return if isnothing(hist)
        fold_statistic(alg, stat, x)
    else
        fold_statistic(alg, stat, hist, x)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

One row of a folding statistic's recursion over the active assets alone, written back into the carried statistic.

`stat` already carries the cold seed at every asset the mask turns off, so an inactive asset keeps that seed and re-enters the recursion cold when it relists. The active assets are the row's Coverage Universe, and the recursion reads them alone: a statistic that couples the assets — the kernel trend pattern's regression — therefore pools the live assets and needs no mask of its own. A row on which no asset is active folds nothing.

# Arguments

  - `alg`: The folding statistic.
  - `stat`: The carried statistic, the cold seed at every inactive asset.
  - `hist`: The carried memory with the row appended, or `nothing`.
  - `x`: The price relative of the row, one at every asset the fold holds.
  - `active`: The active assets of the row.

# Returns

  - `stat'::AbstractVector`: The statistic after the row, a new vector.

# Related

  - [`fold_statistic_row`](@ref)
  - [`cold_statistic`](@ref)
  - [`partial_fit!`](@ref)
"""
function fold_active(alg::AbstractPriceLevelStatistic, stat::AbstractVector,
                     hist::Option{<:AbstractMatrix}, x::AbstractVector,
                     active::AbstractVector{<:Bool})
    if all(active)
        return fold_statistic_row(alg, stat, hist, x)
    end
    i = findall(active)
    if isempty(i)
        return copy(stat)
    end
    y = fold_statistic_row(alg, stat[i], isnothing(hist) ? nothing : hist[:, i], x[i])
    out = similar(stat, promote_type(eltype(stat), eltype(y)))
    copyto!(out, stat)
    out[i] = y
    return out
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The batch form of a folding statistic over a matrix of levels: the recursion from the first row, the memory filling as it goes, times the last level.

# Arguments

  - `alg`: The folding statistic.
  - `P`: The levels, `levels × assets`, the last row the current level.

# Returns

  - `stat::Vector`: The statistic per asset, in units of the levels.

# Related

  - [`price_level_statistic`](@ref)
  - [`fold_statistic`](@ref)
"""
function fold_levels(alg::AbstractPriceLevelStatistic, P::AbstractMatrix)
    stat = nothing
    hist = nothing
    for t in 2:size(P, 1)
        x = view(P, t, :) ./ view(P, t - 1, :)
        hist = push_memory(alg, hist, x)
        stat = fold_statistic_row(alg, stat, hist, x)
    end
    return isnothing(stat) ? P[end, :] : stat .* P[end, :]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The spatial median of the rows of `P`, by the modified Weiszfeld iteration of Vardi and Zhang (2000).

# Mathematical definition

Seeded at the coordinatewise median ``\\boldsymbol{y}_0``, each iteration forms

```math
\\begin{align}
\\tilde{T}(\\boldsymbol{y}) &= \\frac{\\sum_{i : \\boldsymbol{p}_i \\neq \\boldsymbol{y}} \\boldsymbol{p}_i / \\lVert \\boldsymbol{p}_i - \\boldsymbol{y} \\rVert}{\\sum_{i : \\boldsymbol{p}_i \\neq \\boldsymbol{y}} 1 / \\lVert \\boldsymbol{p}_i - \\boldsymbol{y} \\rVert}\\,,\\quad
R(\\boldsymbol{y}) = \\left\\lVert \\sum_{i : \\boldsymbol{p}_i \\neq \\boldsymbol{y}} \\frac{\\boldsymbol{p}_i - \\boldsymbol{y}}{\\lVert \\boldsymbol{p}_i - \\boldsymbol{y} \\rVert} \\right\\rVert\\,,\\\\
\\boldsymbol{y}' &= \\max\\left(0, 1 - \\frac{\\eta}{R}\\right) \\tilde{T}(\\boldsymbol{y}) + \\min\\left(1, \\frac{\\eta}{R}\\right) \\boldsymbol{y}\\,,
\\end{align}
```

where ``\\eta`` is the number of rows equal to ``\\boldsymbol{y}``, so the plain Weiszfeld step is taken away from every data point and the iterate stays defined on one. The iteration stops when ``\\lVert \\boldsymbol{y}' - \\boldsymbol{y} \\rVert < \\texttt{tol}`` or after `iters` steps.

# Arguments

  - `P`: The points, one per row.
  - `iters`: Maximum number of iterations.
  - `tol`: Convergence tolerance.

# Returns

  - `y::Vector`: The spatial median.

# Related

  - [`SpatialMedian`](@ref)
  - [`price_level_statistic`](@ref)

# References

  - $(ref_dict[:vardizhang2000])
"""
function spatial_median(P::AbstractMatrix, iters::Integer, tol::Real)
    y = vec(Statistics.median(P; dims = 1))
    num = similar(y)
    dir = similar(y)
    for _ in 1:iters
        fill!(num, zero(eltype(num)))
        fill!(dir, zero(eltype(dir)))
        den = zero(eltype(num))
        eta = 0
        for i in axes(P, 1)
            p = view(P, i, :)
            d = LinearAlgebra.norm(p .- y)
            if iszero(d)
                eta += 1
                continue
            end
            num .+= p ./ d
            dir .+= (p .- y) ./ d
            den += inv(d)
        end
        if iszero(den)
            # Every row equals the iterate: it is the median.
            return y
        end
        T = num ./ den
        R = LinearAlgebra.norm(dir)
        ynew = if iszero(eta)
            T
        elseif iszero(R)
            y
        else
            max(zero(R), one(R) - eta / R) .* T .+ min(one(R), eta / R) .* y
        end
        if LinearAlgebra.norm(ynew .- y) < tol
            return ynew
        end
        y = ynew
    end
    return y
end
"""
    Statistics.mean(me::PriceLevelExpectedReturns, X::MatNum; dims::Int = 1,
                    active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)

Compute the expected return as the ratio of a price-level statistic to the last price, less one.

A **windowed** statistic reconstructs the levels with [`price_levels`](@ref) from the last [`window_rows`](@ref) rows of `X`, or from every row when `X` holds fewer, reads the statistic on `me.alg` through [`price_level_statistic`](@ref), and answers the statistic less one, because the last level is one. It reads no active mask: the Asset Panel seam reduces it to the Coverage Universe of its window.

A **folding** statistic runs its own recursion over the rows through [`partial_fit!`](@ref), from a cold state, under `active_mask`. The batch and the fold are therefore the same code over the same rows, so they agree exactly, and the batch reads a gap and a relisting as the fold does: an asset the mask turns off starts cold when it relists, and an asset that has folded no level is `NaN`.

# Arguments

  - `me`: The estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the Asset Panel, `observations × assets`, or `nothing`. A windowed statistic refuses one.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - `X` holds at least one observation. An `IsEmptyError` is thrown otherwise.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`. A `DimensionMismatch` is thrown otherwise.
  - `me.alg` folds when `active_mask` is given. An `ArgumentError` is thrown otherwise.

# Returns

  - `mu::Matrix{<:Number}`: The expected return, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`price_levels`](@ref)
  - [`price_level_statistic`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.mean(me::PriceLevelExpectedReturns, X::MatNum; dims::Int = 1,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    X, amsk = dims_oriented(dims, X, active_mask)
    @argcheck(size(X, 1) >= 1,
              IsEmptyError("X must hold at least one observation to reconstruct a price level"))
    @argcheck(isnothing(amsk) || size(X) == size(amsk),
              DimensionMismatch("size(X) ($(size(X))) must match size(active_mask) ($(size(amsk)))"))
    mu = if folds(me.alg)
        vec(Statistics.mean(partial_fit!(PriceLevelExpectedReturns(; alg = me.alg), X;
                                         dims = 1, active_mask = amsk)))
    else
        @argcheck(isnothing(amsk),
                  ArgumentError("`$(typeof(me.alg).name.name)` is a windowed statistic with no recursion to reset, so it reads no active mask: the Asset Panel seam reduces it to the Coverage Universe of its window."))
        need = window_rows(me.alg)
        k = isnothing(need) ? size(X, 1) : min(size(X, 1), need)
        P = price_levels(view(X, (size(X, 1) - k + 1):size(X, 1), :))
        price_level_statistic(me.alg, P) .- one(eltype(P))
    end
    return dims == 1 ? reshape(mu, 1, :) : reshape(mu, :, 1)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The Asset Panel method of the expected return of a price-level statistic.

A **folding** statistic is mask-aware, so it overrides the reduce-and-expand root of the verb and reads the panel's active mask itself. The answer therefore lives on the whole universe rather than on the Coverage Universe: an asset that lists inside the window is answered from the levels it has, and it is `NaN` only while it has folded none. A **windowed** statistic takes the root: it is fitted on the Coverage Universe of the window and expanded with `NaN` outside it.

# Arguments

  - `me`: The estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `mu::Matrix{<:Number}`: The expected return, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`Statistics.mean(me::AbstractExpectedReturnsEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.mean(me::PriceLevelExpectedReturns, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    if !folds(me.alg)
        cmsk, Xc = coverage_reduction(X, pnl; dims = dims)
        return expand_moment(Statistics.mean(me, Xc; dims = dims, kwargs...), cmsk, dims)
    end
    amsk, _ = panel_moment_masks(pnl)
    return Statistics.mean(me, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    Statistics.mean(me::PriceLevelExpectedReturns; kwargs...)

Reads the expected return out of the state a folding statistic carries: the folded statistic less one, shaped `(1, N)`.

An asset that has folded no level since its last reset carries the cold seed rather than a forecast, so the read-out answers `NaN` for it, as [`ExpWeightedExpectedReturns`](@ref) answers below its `min_obs`. Every folding statistic truncates to the levels it holds, so one folded level is enough, and the `NaN` names an asset the mask has turned off or one that has quoted nothing yet.

# Validation

  - `me.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`PriceLevelForecastState`](@ref)
  - [`partial_fit!`](@ref)
"""
function Statistics.mean(me::PriceLevelExpectedReturns; kwargs...)
    state = partial_fit_cache(me)
    mu = state.stat .- one(eltype(state.stat))
    mu[iszero.(state.nu)] .= NaN
    return reshape(mu, 1, :)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a fold of a statistic that has no exact recursion, by name.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`partial_fit!`](@ref)
"""
function assert_folding_statistic(alg::AbstractPriceLevelStatistic)::Nothing
    @argcheck(folds(alg),
              ArgumentError("`$(typeof(alg).name.name)` is a windowed statistic with no exact recursion over price relatives, so `partial_fit!` cannot fold it: a host that carries the rows refits it with `mean(me, X)`, and the online portfolio selection head holds `window - 1` rows for it."))
    return nothing
end
"""
    partial_fit!(me::PriceLevelExpectedReturns, x::VecNum;
                 active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
    partial_fit!(me::PriceLevelExpectedReturns, X::MatNum; dims::Int = 1,
                 active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)

Folds one row of returns, or a block of them in order, into the state a folding statistic carries, seeding it on the first row and resetting an asset the active mask turns off.

An asset is **valid** at a row when its return is finite and the mask admits it, which is the condition the exponentially weighted family reads. With no mask the finite assets are the active ones, so a gap is a delisting rather than a holiday, because nothing else states which it is.

# Algorithm

 1. Read the valid assets and the active ones, and refuse a mask with no entry per asset.
 2. Turn the row into the price relative `1 .+ x`, at one wherever the asset is not valid: the level did not move, which is the reading a holiday inside a listing takes.
 3. Append the relative to the memory of a statistic that carries one through [`push_memory`](@ref), and flatten the memory of every inactive asset to one, so that its levels carry nothing from before the delisting.
 4. Write [`cold_statistic`](@ref) at every inactive asset, and at every asset that has folded no level, so that a relisting re-enters the recursion cold.
 5. Fold the active assets alone with [`fold_active`](@ref).
 6. Add one to the count of every valid asset, zero the count of every inactive one, and add one to the number of rows folded.

# Arguments

  - `me`: The estimator.
  - `x`: One row of returns, one entry per asset.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `active_mask`: The active mask of the Asset Panel at this row, or over the block, or `nothing`.
  - $(arg_dict[:ignkwargs])

# Validation

  - `me.alg` folds. An `ArgumentError` is thrown otherwise, through [`assert_folding_statistic`](@ref).
  - If `active_mask` is not `nothing`, it holds one entry per asset. A `DimensionMismatch` is thrown otherwise.
  - $(val_dict[:dims])

# Returns

  - `me::PriceLevelExpectedReturns`: The estimator carrying the state after the last row.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`PriceLevelForecastState`](@ref)
  - [`fold_statistic`](@ref)
  - [`fold_active`](@ref)
  - [`cold_statistic`](@ref)
  - [`supports_partial_fit`](@ref)
"""
function partial_fit!(me::PriceLevelExpectedReturns{<:Any,
                                                    <:Option{<:PriceLevelForecastState}},
                      x::VecNum; active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      kwargs...)
    assert_folding_statistic(me.alg)
    @argcheck(isnothing(active_mask) || length(active_mask) == length(x),
              DimensionMismatch("the active mask must have one entry per asset, but the row has $(length(x)) entries and the mask has $(length(active_mask))."))
    state = me.cache
    finite = isfinite.(x)
    active = isnothing(active_mask) ? finite : collect(active_mask)
    valid = finite .& active
    xr = ifelse.(valid, one(eltype(x)) .+ x, one(eltype(x)))
    hist = push_memory(me.alg, isnothing(state) ? nothing : state.hist, xr)
    if !isnothing(hist) && !all(active)
        hist[:, .!active] .= one(eltype(hist))
    end
    nu = isnothing(state) ? zeros(Int, length(x)) : state.nu
    cold = cold_statistic(me.alg, xr)
    prev = isnothing(state) ? cold : ifelse.(active .& .!iszero.(nu), state.stat, cold)
    stat = fold_active(me.alg, prev, hist, xr, active)
    n = isnothing(state) ? 0 : state.n
    return Accessors.@reset me.cache = PriceLevelForecastState(n + 1,
                                                               ifelse.(active, nu .+ valid,
                                                                       zero(eltype(nu))),
                                                               stat, hist)
end
function partial_fit!(me::PriceLevelExpectedReturns{<:Any,
                                                    <:Option{<:PriceLevelForecastState}},
                      X::MatNum; dims::Int = 1,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    X, amsk = dims_oriented(dims, X, active_mask)
    for i in axes(X, 1)
        me = partial_fit!(me, view(X, i, :);
                          active_mask = isnothing(amsk) ? nothing : view(amsk, i, :),
                          kwargs...)
    end
    return me
end
# A folding statistic is an exact recursion over price relatives, so the estimator folds
# exactly when its statistic does (see [`supports_partial_fit`](@ref)).
function supports_partial_fit(me::PriceLevelExpectedReturns)
    return folds(me.alg)
end
export PriceLevelExpectedReturns, MovingAverage, ExponentialMovingAverage, SpatialMedian,
       WindowPeak, LaggedPrice, ReweightedPriceRelative
public AbstractPriceLevelStatistic, PriceLevelForecastState, price_level_statistic,
       fold_statistic, cold_statistic, rows_needed, window_rows, folds, memory_rows
