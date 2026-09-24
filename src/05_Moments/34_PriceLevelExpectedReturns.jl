"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the statistics of a window of price levels that [`PriceLevelExpectedReturns`](@ref) turns into an expected return.

Every statistic reads the reconstructed price path. The level of the last observation is one in every asset, and the returns of the window give the earlier levels. The statistic is a vector over the assets, and the expected return is the statistic divided by the last level, minus one.

A statistic that reads each asset's levels alone is homogeneous of degree one in them, so it has the same value on the path as on the prices. The moving averages, the peak, the lagged price, the reweighted relative and the trend switches are of this kind. A statistic that couples the assets changes when each asset is scaled by its own last price. The spatial median couples them through its Euclidean distances, and the kernel trend pattern through its regression. On the path, each of them is the paper's statistic of prices normalised to one at the current period, and its docstring states the difference.

A statistic is **windowed** when it reads a fixed number of levels. [`MovingAverage`](@ref), [`SpatialMedian`](@ref), [`WindowPeak`](@ref) and [`LaggedPrice`](@ref) are windowed. A statistic is **folding** when it is an exact recursion over the price relatives, as [`ExponentialMovingAverage`](@ref) and [`ReweightedPriceRelative`](@ref) are. A folding statistic carries one vector on a Partial Fit State and reads no rows. Its batch form over a matrix of returns runs the same recursion from the first level, so a fold and a batch over the same rows agree exactly.

A folding statistic can also carry a memory on the same state, the last [`memory_rows`](@ref) relatives, as [`KernelTrendPattern`](@ref) does. Its recursion reads the carried vector and the memory together, so a host holds no rows for it. Its batch form fills the memory as it runs from the first row.

Over the first rows a windowed statistic reads the levels it has. With `window = 5` and two returns folded, it reads three levels. This truncation is the library's choice. Over their first `window` rows, the authors' code for the moving-average, median and peak price papers forecasts the last price relative instead. A folding statistic starts from its seed at the first row, and a composite statistic truncates every window it holds.

A folding statistic reads the active mask. [`partial_fit!`](@ref) folds the active assets of the row alone, and it resets an inactive asset to [`cold_statistic`](@ref), so a relisted asset starts cold. The read-out is `NaN` at an asset that has folded no level since its last reset. The caller reduces the assets, so a statistic that couples them reads the active assets alone and needs no mask of its own.

# Interfaces

To implement a new price-level statistic, subtype `AbstractPriceLevelStatistic` with all its parameters as part of the struct, and implement the following methods:

  - `price_level_statistic(alg::AbstractPriceLevelStatistic, P::AbstractMatrix) -> AbstractVector`: The statistic of the levels `P`, `levels × assets`, whose last row is the current level. A folding statistic runs its recursion over every row of `P` from the first.
  - `window_rows(alg::AbstractPriceLevelStatistic) -> Union{Nothing, Integer}`: The number of return rows the batch form reads to reconstruct its levels. It is `window - 1` for a statistic over `window` levels, and `nothing` for one that reads every row it gets. The default reads `alg.window - 1`.
  - `folds(alg::AbstractPriceLevelStatistic) -> Bool`: `true` when the statistic is an exact recursion over price relatives, which [`fold_statistic`](@ref) carries. The default is `false`.
  - `fold_statistic(alg::AbstractPriceLevelStatistic, stat, x::AbstractVector) -> AbstractVector`: The recursion of a folding statistic. It takes the carried statistic `stat` in relative terms, `nothing` before the first row, and the price relative `x` of the row. It returns the statistic after the row as a new vector.
  - `cold_statistic(alg::AbstractPriceLevelStatistic, x::AbstractVector) -> AbstractVector`: The carried statistic of a folding statistic before its first row, in the units the recursion reads. It is the value for which the recursion returns its own seed. The default is one in every asset.
  - `memory_rows(alg::AbstractPriceLevelStatistic) -> Integer`: The number of relatives a folding statistic carries as its memory. The default is `0`, for a recursion that reads the carried vector alone. A statistic with a memory implements the four-argument `fold_statistic(alg, stat, hist, x)` instead. There `hist` holds the last `memory_rows(alg)` relatives with the row's own last, or fewer over the first rows.

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

Averages the last `window` price levels, which is the forecast of the moving-average reversion of Li and Hoi (2012).

The journal version of the paper, Li, Hoi, Sahoo and Liu (2015), allows `window >= 2` and runs its experiments at `window = 5`, the default.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{x}}_{t+1} &= \\frac{1}{w} \\sum_{k = 0}^{w - 1} \\boldsymbol{p}_{t-k} \\oslash \\boldsymbol{p}_{t}\\,.
\\end{align}
```

Where:

  - $(math_dict[:xhat_fc])
  - $(math_dict[:p_t_level])
  - $(math_dict[:w_levels])

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

# References

  - $(ref_dict[:lihoi2012])
  - $(ref_dict[:li2015olmar]) Equation (1).
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

Averages the price levels with weights that fall exponentially with age, which is the forecast of the second form of the moving-average reversion of Li, Hoi, Sahoo and Liu (2015).

It is a folding statistic. The recursion is exact in relative terms, so the statistic reads no rows and carries one vector. The paper admits ``\\alpha \\in (0, 1)`` and states no default. Its sensitivity study finds high wealth over a wide range of ``\\alpha`` and poor wealth at the two endpoints. The default `alpha = 0.5` is the value of the authors' toolbox example. The library also admits `alpha = 1`, where the forecast is one in every asset.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MA}_{t} &= \\alpha \\boldsymbol{p}_{t} + (1 - \\alpha) \\mathrm{MA}_{t-1}\\,,\\quad \\mathrm{MA}_{0} = \\boldsymbol{p}_{0}\\,,\\\\
\\hat{\\boldsymbol{x}}_{t+1} &= \\mathrm{MA}_{t} \\oslash \\boldsymbol{p}_{t} = \\alpha \\boldsymbol{1} + (1 - \\alpha)\\, \\hat{\\boldsymbol{x}}_{t} \\oslash \\boldsymbol{x}_{t}\\,,\\quad \\hat{\\boldsymbol{x}}_{1} = \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{MA}_{t}``: The exponential moving average of the levels up to period ``t``.
  - $(math_dict[:p_t_level]) The first level is ``\\boldsymbol{p}_{0}``.
  - $(math_dict[:x_t_rel])
  - $(math_dict[:alpha_ema])
  - $(math_dict[:xhat_fc])

The paper writes the first price as ``\\boldsymbol{p}_{1}`` and seeds the average there. The seed above is the same seed with the first level indexed zero, so the forecast after one level is one.

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

  - $(ref_dict[:li2015olmar]) Equation (2).
"""
struct ExponentialMovingAverage{T1 <: Real} <: AbstractPriceLevelStatistic
    """
    The smoothing weight on the current level, `0 < alpha <= 1`. A smaller weight gives a longer memory.
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

Takes the spatial (``L_1``) median of the last `window` price levels, which is the forecast of the robust median reversion of Huang, Zhou, Li, Hoi and Zhou (2016).

[`spatial_median`](@ref) finds the median with the modified Weiszfeld iteration of Vardi and Zhang (2000), after it tests each level as the median.

The distances couple the assets, so the median changes when each asset is scaled by its own price. The library reads it on the reconstructed path, with every asset's last level at one. The paper takes the median of the raw prices, and the authors' code takes it of prices rebased to one on the first day of the series. On the test fixture, the library's forecast differs from the rebased one by up to 0.003. It differs from the raw one by up to 0.08 when one asset trades at a hundred times the price of the others. The normalised median is scale-free, so no asset dominates the distances through its quotation.

The paper stops its iteration when ``\\lVert \\boldsymbol{\\mu}_{i-1} - \\boldsymbol{\\mu}_{i} \\rVert_1 \\leq \\tau \\lVert \\boldsymbol{\\mu}_{i} \\rVert_1``, and it gives no value for its maximum number of iterations. The authors' code stops at ``\\tau = 10^{-9}`` or after 200 iterations. The library stops when the Euclidean change of the iterate is below `tol`, or after `iters` iterations. On the test fixture the defaults leave the forecast within ``3 \\times 10^{-8}`` of the minimiser, as close as the authors' code.

When the median lies very near a level but not on it, the Weiszfeld iteration converges slowly. The cap of `iters` can then stop it with the forecast about ``10^{-4}`` from the minimiser, while the sum of distances is within ``10^{-6}`` of its minimum. The steps are then small, so `tol` stops it early too. Raise `iters` and lower `tol` together where that difference matters.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{m}_{t} &= \\underset{\\boldsymbol{y}}{\\arg\\min} \\sum_{k = 0}^{w - 1} \\lVert \\boldsymbol{p}_{t-k} - \\boldsymbol{y} \\rVert\\,,\\\\
\\hat{\\boldsymbol{x}}_{t+1} &= \\boldsymbol{m}_{t} \\oslash \\boldsymbol{p}_{t}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{m}_{t}``: The spatial median of the window that ends at period ``t``.
  - $(math_dict[:p_t_level])
  - $(math_dict[:w_levels])
  - $(math_dict[:xhat_fc])

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

# References

  - $(ref_dict[:huang2016]) Equations (2) and (5), Algorithm 1.
  - $(ref_dict[:vardizhang2000])
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

Takes the highest of the last `window` price levels, which is the forecast of the peak price tracking of Lai, Dai, Ren and Huang (2018).

The peak price tracking paper is not open access. Li, Luo and Xu (2023, eq. 5) restate its peak, and the short-term sparse portfolio of Lai, Yang, Fang and Wu (2018, eq. 10) uses the same peak. That portfolio does not read the ratio itself. It reads ``1.1 \\log \\hat{x}_{t+1, i} + 1`` (eq. 11), a transform that [`ShortTermSparsePortfolio`](@ref) applies.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{x}}_{t+1} &= \\max_{0 \\leq k < w} \\boldsymbol{p}_{t-k} \\oslash \\boldsymbol{p}_{t} \\geq \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - $(math_dict[:xhat_fc])
  - $(math_dict[:p_t_level])
  - $(math_dict[:w_levels])

The maximum is taken per asset. The peak is never below the last level, so the forecast is at least one in every asset, and it is exactly one for an asset at its own window peak.

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
  - $(ref_dict[:lai2018sspo]) Equations (10) and (11).
  - $(ref_dict[:liluoxu2023]) Equation (5).
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

Takes the price level `lag` observations ago, which is the forecast of the first form of the transaction-cost optimisation of Li, Wang, Huang and Hoi (2018).

The forecast bets that the moves of the last `lag` periods revert. The paper's first form reads `lag = 1` (Algorithm 2). At `lag = 0` the forecast is one in every asset, which is the hold branch of a switched statistic. Over fewer than `lag` rows the statistic reads the first level it has.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{x}}_{t+1} &= \\boldsymbol{p}_{t-\\ell} \\oslash \\boldsymbol{p}_{t}\\,.
\\end{align}
```

Where:

  - $(math_dict[:xhat_fc])
  - $(math_dict[:p_t_level])
  - ``\\ell``: The lag, the number of periods between the forecast level and the current one.

At ``\\ell = 1`` the forecast is ``\\boldsymbol{1} \\oslash \\boldsymbol{x}_{t}``, the reciprocal of the last price relative ``\\boldsymbol{x}_{t}``.

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

  - $(ref_dict[:li2018tco]) Algorithm 2.
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

Averages the price levels with a weight per asset that depends on the data, which is the reweighted price relative of Lai, Yang, Fang and Wu (2020).

It is a folding statistic. The paper is not open access. Li, Luo and Xu (2023, eqs. 8 and 11) restate the recursion, and the MATLAB code that the authors publish runs the same recursion. The seed ``\\hat{\\boldsymbol{\\varphi}}_{1} = \\boldsymbol{x}_{1}`` is that of the restatement, so the forecast after the first row is one. The authors' code seeds the forecast at one instead. On the test fixture at `theta = 0.8`, the two forecasts differ by 0.024 after the first row and by less than ``10^{-4}`` after ten rows. The default `theta = 0.8` is the value of the authors' code. The restating paper sets 0.7 for its own rule.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{\\gamma}_{t+1} &= \\theta \\boldsymbol{x}_{t} \\oslash (\\theta \\boldsymbol{x}_{t} + \\hat{\\boldsymbol{\\varphi}}_{t})\\,,\\\\
\\hat{\\boldsymbol{\\varphi}}_{t+1} &= \\boldsymbol{\\gamma}_{t+1} + (\\boldsymbol{1} - \\boldsymbol{\\gamma}_{t+1}) \\odot \\hat{\\boldsymbol{\\varphi}}_{t} \\oslash \\boldsymbol{x}_{t}\\,,\\quad
\\hat{\\boldsymbol{\\varphi}}_{1} = \\boldsymbol{x}_{1}\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_t_rel])
  - ``\\boldsymbol{\\gamma}_{t+1}``: The weight of each asset on its current relative.
  - ``\\theta``: The reweighting strength. A larger value puts more weight on the current relative.
  - ``\\hat{\\boldsymbol{\\varphi}}_{t+1}``: The Price Relative Forecast for period ``t + 1``, in the symbol of the restating paper.

[`ExponentialMovingAverage`](@ref) is the case of a constant weight ``\\boldsymbol{\\gamma}_{t+1} = \\alpha \\boldsymbol{1}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ReweightedPriceRelative(; theta::Real = 0.8) -> ReweightedPriceRelative

Keywords correspond to the struct's fields.

## Validation

  - `theta > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ReweightedPriceRelative()
ReweightedPriceRelative
  theta ┴ Float64: 0.8
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`ExponentialMovingAverage`](@ref)
  - [`ReweightedPriceRelativeTracking`](@ref)
  - [`fold_statistic`](@ref)

# References

  - $(ref_dict[:lai2018rprt])
  - $(ref_dict[:liluoxu2023]) Equations (8) and (11).
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
function ReweightedPriceRelative(; theta::Real = 0.8)::ReweightedPriceRelative
    return ReweightedPriceRelative(theta)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a price-level window below two levels.

One level is the current price, and its ratio to itself forecasts nothing.

# Validation

  - `window >= 2`. A `DomainError` is thrown otherwise.

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

The Partial Fit State of a [`PriceLevelExpectedReturns`](@ref) whose statistic folds.

It holds the number of rows folded, the number of levels each asset has folded since its last reset, the statistic in units of the last level, and the memory of a statistic that carries one. The state is not merged, because its recursion depends on the order of the rows. Two states folded on different rows have no common continuation.

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

Forecasts the next return of each asset as the move from the last price level to a statistic of a window of price levels.

The forecast is the Price Relative Forecast of the online portfolio selection family, as an expected-returns estimator that any `me` field can hold. It is a forecast of the next period's return under the reversion or trend hypothesis of the paper that defines the statistic, not an average of past returns. The estimator reads every statistic on the reconstructed path, so the forecast is a function of the returns alone. [`AbstractPriceLevelStatistic`](@ref) states where this path changes the paper's statistic.

A windowed statistic reads its last `window - 1` returns, and over the first rows it reads the levels it has. A folding statistic, such as [`ExponentialMovingAverage`](@ref) or [`ReweightedPriceRelative`](@ref), is an exact recursion over the price relatives. [`partial_fit!`](@ref) folds one row into the state that the `cache` field carries, and `mean(me)` reads it, so an online host carries one vector for it and no rows. Its batch form over a matrix runs the same recursion from the first level, and the two agree exactly over the same rows.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{p}_{t} &= \\boldsymbol{1}\\,,\\quad \\boldsymbol{p}_{t-k} = \\boldsymbol{p}_{t-k+1} \\oslash (\\boldsymbol{1} + \\boldsymbol{r}_{t-k+1})\\,,\\quad k = 1, \\ldots, w - 1\\,,\\\\
\\hat{\\boldsymbol{\\mu}} &= \\mathrm{stat}(\\boldsymbol{p}_{t-w+1}, \\ldots, \\boldsymbol{p}_{t}) \\oslash \\boldsymbol{p}_{t} - \\boldsymbol{1}\\,.
\\end{align}
```

Where:

  - $(math_dict[:p_t_level]) Here the levels are reconstructed from the returns, with the last level at one.
  - ``\\boldsymbol{r}_{t}``: Returns vector of period ``t``, one entry per asset.
  - $(math_dict[:w_levels])
  - ``\\mathrm{stat}``: The statistic of `alg`, a vector over the assets.
  - ``\\hat{\\boldsymbol{\\mu}}``: The expected return, one entry per asset.
  - $(math_dict[:xhat_fc])

The expected return is the Price Relative Forecast minus one, ``\\hat{\\boldsymbol{\\mu}} = \\hat{\\boldsymbol{x}}_{t+1} - \\boldsymbol{1}``.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PriceLevelExpectedReturns(;
        alg::AbstractPriceLevelStatistic = MovingAverage(),
        cache::Option{<:PriceLevelForecastState} = nothing
    ) -> PriceLevelExpectedReturns

Keywords correspond to the struct's fields.

## View parameters

`PriceLevelExpectedReturns` defines its own [`port_opt_view`](@ref) method rather than deriving one from field tags.

  - `alg` passes through unchanged, because a statistic holds no per-asset data.
  - `cache` recurses through [`port_opt_view`](@ref) of [`PriceLevelForecastState`](@ref), which slices the count, the statistic and the columns of the memory to the selected assets. The number of rows folded stays.

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
    window_rows(alg::GaussianWeightedDoubleEstimate)
    window_rows(test::AbstractTrendTest)
    window_rows(alg::TrendSwitch)
    window_rows(alg::CompositeTrend)
    window_rows(alg::KernelTrendPattern)

Returns the number of return rows the batch form of a statistic reads, or `nothing` for one that reads every row it gets.

A statistic over `window` levels reads `window - 1` rows, which is the default, and so does a trend test. The lagged price reads `lag` rows. A folding statistic reads every row, because its batch form is the recursion from the first level. The Gaussian double estimate reads the number of levels in its Gaussian window. A trend switch reads the largest number of its test and its branches, and a composite trend reads `window - 1` rows more than its widest trend. Either of them reads every row when one of its members does.

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
    folds(alg::KernelTrendPattern)

Returns `true` when a statistic is an exact recursion over price relatives, and `false` by default.

A composite statistic does not fold, even when a member does, because it reads each member over a window of levels.

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
    memory_rows(alg::KernelTrendPattern)

Returns the number of relatives a folding statistic carries as its memory on the state.

The default is `0`, for a statistic whose recursion reads the carried vector alone. The kernel trend pattern carries `2 window` relatives.

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

Appends the price relative of a row to the memory of a statistic and keeps the last [`memory_rows`](@ref) relatives.

# Algorithm

 1. Read `m`, the number of relatives the statistic carries. Return `nothing` when `m` is zero.
 2. Make `row`, the relative `x` as a one-row matrix. Return it when `hist` is `nothing`.
 3. Stack `row` under `hist`, giving `H`, and return the last `m` rows of `H`.

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

Returns the number of return rows an expected-returns estimator reads at a step of the online portfolio selection family, `1` for one that folds, or `nothing` when it reads every row folded so far.

A windowed price-level statistic over `window` levels reads `window - 1` returns. A folding statistic reads the current row alone, because its state is one vector. The head gives it that row as it stands, with its gaps and its active mask, as a one-row carrier, and not the finite price relative of the step. An estimator for which [`supports_partial_fit`](@ref) returns `true` has an exact fold of its own, and it reads the current row on the same terms. A [`WindowedExpectedReturns`](@ref) reads its window.

Every other expected-returns estimator returns `nothing`, which is unbounded. The head keeps every row for it, and it refits on the whole prefix at every step, at a cost of `O(tN)` a step after `t` rows. [`ForecastReversion`](@ref) forwards to its fields, and the head takes the maximum over its rule tree.

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

Reconstructs the price levels of a window of returns, with the last level at one.

The last row of the result is ones, and the row above it is `1 ./ (1 .+ r_T)`, where `r_T` is the last row of `X`. Each earlier row divides the row below it by one plus the return between them, back to the level before the first row of `X`. So the result holds `size(X, 1) + 1` rows. The statistic reads the last `window` of them, or all of them when the window truncates.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{p}_{T} &= \\boldsymbol{1}\\,,\\quad \\boldsymbol{p}_{t-1} = \\boldsymbol{p}_{t} \\oslash (\\boldsymbol{1} + \\boldsymbol{r}_{t})\\,,\\quad t = T, \\ldots, 1\\,.
\\end{align}
```

Where:

  - $(math_dict[:p_t_level])
  - ``\\boldsymbol{r}_{t}``: Returns vector of period ``t``, one entry per asset.
  - $(math_dict[:T])

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
    price_level_statistic(alg::TruncatedExponentialMovingAverage, P::AbstractMatrix)
    price_level_statistic(alg::GaussianWeightedDoubleEstimate, P::AbstractMatrix)
    price_level_statistic(alg::TrendSwitch, P::AbstractMatrix)
    price_level_statistic(alg::CompositeTrend, P::AbstractMatrix)
    price_level_statistic(alg::KernelTrendPattern, P::AbstractMatrix)

Returns the statistic of a window of price levels, one value per asset.

Each composite statistic states its own statistic in its `# Mathematical definition`, and its method returns that statistic times the last level. A trend switch and a composite trend read each member over the member's own window with [`member_statistic`](@ref). The kernel trend pattern runs its recursion through [`fold_levels`](@ref).

Every statistic is homogeneous of degree one when all the levels are scaled by one number. The moving average is the column mean of the levels. The exponential moving average runs its recursion over every row from the first. The spatial median is the point that minimises the sum of Euclidean distances to the rows, which [`spatial_median`](@ref) finds. The window peak is the column maximum. The lagged price is the row `lag` above the last, or the first row when the window truncates. The reweighted relative runs its recursion over the relatives of successive rows through [`fold_levels`](@ref), and multiplies the result by the last level.

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

Returns the carried statistic of a folding statistic before its first row, in the units that its recursion reads.

It is the value for which [`fold_statistic`](@ref) returns the seed that the statistic states. The exponential moving average seeds its forecast at one and reads the carried vector as it stands, so its cold seed is one in every asset. The reweighted relative seeds at the row's own relative. The kernel trend pattern divides the carried prediction by that relative to bring it into the units of the current level. So both of them seed at `x`.

[`partial_fit!`](@ref) writes this value at an asset that the active mask turns off, so a relisted asset starts the recursion again from its seed.

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

Runs one row of the recursion of a folding statistic, in units of the current level.

The four-argument form is the recursion of a statistic with a memory. It gets the last [`memory_rows`](@ref) relatives with the row appended, and the fold and the batch choose the arity by the memory. [`KernelTrendPattern`](@ref) states its own recursion.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{x}}' &= \\alpha \\boldsymbol{1} + (1 - \\alpha)\\, \\hat{\\boldsymbol{x}} \\oslash \\boldsymbol{x}\\,,\\quad \\hat{\\boldsymbol{x}}_{\\mathrm{cold}} = \\boldsymbol{1}\\,,\\\\
\\boldsymbol{\\gamma} &= \\theta \\boldsymbol{x} \\oslash (\\theta \\boldsymbol{x} + \\hat{\\boldsymbol{\\varphi}})\\,,\\quad
\\hat{\\boldsymbol{\\varphi}}' = \\boldsymbol{\\gamma} + (\\boldsymbol{1} - \\boldsymbol{\\gamma}) \\odot \\hat{\\boldsymbol{\\varphi}} \\oslash \\boldsymbol{x}\\,,\\quad \\hat{\\boldsymbol{\\varphi}}_{\\mathrm{cold}} = \\boldsymbol{x}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{x}}``, ``\\hat{\\boldsymbol{x}}'``: The forecast of [`ExponentialMovingAverage`](@ref) before and after the row.
  - ``\\hat{\\boldsymbol{\\varphi}}``, ``\\hat{\\boldsymbol{\\varphi}}'``: The forecast of [`ReweightedPriceRelative`](@ref) before and after the row.
  - ``\\boldsymbol{x}``: The price relative of the row.
  - $(math_dict[:alpha_ema]) It is the weight of the exponential moving average.
  - ``\\theta``, ``\\boldsymbol{\\gamma}``: The reweighting strength and the per-asset weight of the reweighted relative.
  - ``\\hat{\\boldsymbol{x}}_{\\mathrm{cold}}``, ``\\hat{\\boldsymbol{\\varphi}}_{\\mathrm{cold}}``: The carried value before the first row, which [`cold_statistic`](@ref) returns.

The reweighted relative after the first row is one in every asset, because ``\\hat{\\boldsymbol{\\varphi}}_{\\mathrm{cold}} = \\boldsymbol{x}`` gives ``\\hat{\\boldsymbol{\\varphi}}' = \\boldsymbol{1}``.

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

Runs one row of the recursion of a folding statistic, at the arity that its memory needs.

A statistic that carries no memory takes the three-argument [`fold_statistic`](@ref), and a statistic that carries one takes the four-argument form.

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

Runs one row of the recursion of a folding statistic over the active assets alone, and writes the result back into the carried statistic.

The caller has already written the cold seed into `stat` at every asset that the mask turns off. So an inactive asset keeps that seed, and it starts the recursion from its seed when it relists. The active assets are the Coverage Universe of the row, and the recursion reads them alone. So a statistic that couples the assets, such as the regression of the kernel trend pattern, pools the listed assets and needs no mask of its own.

# Algorithm

 1. When every asset is active, return the row of the recursion from [`fold_statistic_row`](@ref).
 2. Find `i`, the indices of the active assets. When `i` is empty, return a copy of `stat`, which holds the cold seed at every asset.
 3. Run the row of the recursion on the active entries of `stat`, of the columns `i` of `hist` and of `x`, giving `y`.
 4. Make `out`, a copy of `stat` in the promoted element type, write `y` at `i`, and return it.

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

Runs the batch form of a folding statistic over a matrix of levels, from the first row.

# Algorithm

 1. Start with no carried statistic `stat` and no memory `hist`.
 2. For each row `t` from the second, form the price relative `x` of row `t` over row `t - 1`.
 3. Append `x` to `hist` with [`push_memory`](@ref).
 4. Step `stat` with [`fold_statistic_row`](@ref).
 5. Multiply `stat` by the last level and return it. For a single level there is no relative, and the result is the last level.

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

The three sums of one step of the modified Weiszfeld iteration at a point `y`, over the rows of `P` that differ from `y`, and the number of rows equal to it.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{s}(\\boldsymbol{y}) &= \\sum_{i : \\boldsymbol{p}_i \\neq \\boldsymbol{y}} \\frac{\\boldsymbol{p}_i}{\\lVert \\boldsymbol{p}_i - \\boldsymbol{y} \\rVert}\\,,\\quad
d(\\boldsymbol{y}) = \\sum_{i : \\boldsymbol{p}_i \\neq \\boldsymbol{y}} \\frac{1}{\\lVert \\boldsymbol{p}_i - \\boldsymbol{y} \\rVert}\\,,\\\\
r(\\boldsymbol{y}) &= \\left\\lVert \\sum_{i : \\boldsymbol{p}_i \\neq \\boldsymbol{y}} \\frac{\\boldsymbol{p}_i - \\boldsymbol{y}}{\\lVert \\boldsymbol{p}_i - \\boldsymbol{y} \\rVert} \\right\\rVert\\,,\\quad
\\eta(\\boldsymbol{y}) = \\left\\lvert \\{ i : \\boldsymbol{p}_i = \\boldsymbol{y} \\} \\right\\rvert\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{p}_i``: Row ``i`` of `P`, one point.
  - ``\\boldsymbol{y}``: The point the sums are taken at.
  - ``r(\\boldsymbol{y})``: The norm of the sum of the unit vectors from ``\\boldsymbol{y}`` to the other points, which is the norm of the gradient of the sum of distances where no point equals ``\\boldsymbol{y}``.
  - ``\\eta(\\boldsymbol{y})``: The number of points equal to ``\\boldsymbol{y}``.

# Arguments

  - `P`: The points, one per row.
  - `y`: The point.

# Returns

  - `num::Vector`: The sum ``\\boldsymbol{s}(\\boldsymbol{y})``.
  - `den::Number`: The sum ``d(\\boldsymbol{y})``, zero when every row equals `y`.
  - `R::Number`: The norm ``r(\\boldsymbol{y})``.
  - `eta::Int`: The count ``\\eta(\\boldsymbol{y})``.

# Related

  - [`spatial_median`](@ref)
"""
function weiszfeld_sums(P::AbstractMatrix, y::AbstractVector)
    num = zero(y)
    dir = zero(y)
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
    return num, den, LinearAlgebra.norm(dir), eta
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The spatial median of the rows of `P`, by the modified Weiszfeld iteration of Vardi and Zhang (2000).

The plain Weiszfeld iteration converges slowly when the median is one of the points, and it cannot step from an iterate that equals a point. The function first tests every point with the optimality condition of Vardi and Zhang, so a median that is one of the points is found exactly. The iteration then uses the modified map, which stays defined on a point.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{m} &= \\underset{\\boldsymbol{y}}{\\arg\\min} \\sum_{i} \\lVert \\boldsymbol{p}_i - \\boldsymbol{y} \\rVert\\,,\\\\
\\boldsymbol{p}_k = \\boldsymbol{m} &\\iff r(\\boldsymbol{p}_k) \\leq \\eta(\\boldsymbol{p}_k)\\,,\\\\
T(\\boldsymbol{y}) &= \\max\\left(0, 1 - \\frac{\\eta(\\boldsymbol{y})}{r(\\boldsymbol{y})}\\right) \\frac{\\boldsymbol{s}(\\boldsymbol{y})}{d(\\boldsymbol{y})} + \\min\\left(1, \\frac{\\eta(\\boldsymbol{y})}{r(\\boldsymbol{y})}\\right) \\boldsymbol{y}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{m}``: The spatial median.
  - ``\\boldsymbol{p}_i``: Row ``i`` of `P`, one point.
  - ``\\boldsymbol{s}``, ``d``, ``r``, ``\\eta``: The sums and the count of [`weiszfeld_sums`](@ref).
  - ``T``: The modified Weiszfeld map. At a point that equals no row it is the plain Weiszfeld step ``\\boldsymbol{s} / d``, and the median is its fixed point.

# Algorithm

 1. For each row `k` of `P`, take the sums at the row with [`weiszfeld_sums`](@ref). When `R <= eta`, the row is the median, and the function returns it.
 2. Otherwise seed `y` at the coordinatewise median of the rows.
 3. Take the sums at `y`, and form the next iterate `ynew` with the map ``T``.
 4. Return `ynew` when its Euclidean distance to `y` is below `tol`. Otherwise set `y = ynew` and repeat step 3, at most `iters` times, and then return `y`.

# Arguments

  - `P`: The points, one per row.
  - `iters`: Maximum number of iterations.
  - `tol`: Convergence tolerance on the Euclidean distance between two successive iterates.

# Returns

  - `y::Vector`: The spatial median.

# Related

  - [`SpatialMedian`](@ref)
  - [`price_level_statistic`](@ref)
  - [`weiszfeld_sums`](@ref)

# References

  - $(ref_dict[:vardizhang2000])
"""
function spatial_median(P::AbstractMatrix, iters::Integer, tol::Real)
    for k in axes(P, 1)
        y = P[k, :]
        _, _, R, eta = weiszfeld_sums(P, y)
        if R <= eta
            return y
        end
    end
    y = vec(Statistics.median(P; dims = 1))
    for _ in 1:iters
        num, den, R, eta = weiszfeld_sums(P, y)
        T = num ./ den
        ynew = if iszero(eta)
            T
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

Computes the expected return as the ratio of a price-level statistic to the last price level, minus one.

A windowed statistic reads no active mask. The Asset Panel method reduces it to the Coverage Universe of its window. A folding statistic runs its recursion through [`partial_fit!`](@ref), so the batch and the fold are the same code over the same rows, and they agree exactly. The batch reads a gap and a relisting as the fold does. An asset that the mask turns off starts from its seed when it relists, and an asset that has folded no level is `NaN`.

# Algorithm

 1. Orient `X` and `active_mask` so that the observations are rows, and check them.
 2. For a folding statistic, fold every row of `X` into a new estimator with [`partial_fit!`](@ref) under the mask, from no state, and read `mu` from its state.
 3. For a windowed statistic, refuse a mask. Take the last [`window_rows`](@ref) rows of `X`, or every row when `X` holds fewer, and reconstruct their levels `P` with [`price_levels`](@ref).
 4. Read the statistic of `P` with [`price_level_statistic`](@ref), and subtract one, which is the last level, giving `mu`.
 5. Shape `mu` as `dims` asks.

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

Computes the expected return of a price-level statistic on an Asset Panel.

A folding statistic reads the active mask, so this method replaces the reduce-and-expand root of the verb for it, and passes the mask of the panel. The result then covers the whole universe, not only the Coverage Universe. An asset that lists inside the window gets a forecast from the levels it has, and it is `NaN` only while it has folded no level. A windowed statistic takes the root. The method fits it on the Coverage Universe of the window and fills `NaN` outside it. A composite statistic is windowed even when it holds a folding member. So it is `NaN` at an asset with a gap anywhere in its window, while the bare folding member admits that asset again one row after the gap.

# Algorithm

 1. For a windowed statistic, reduce `X` to the Coverage Universe of the panel, compute the expected return there, and expand it with `NaN` to the whole universe.
 2. For a folding statistic, read the active mask of the panel with [`panel_moment_masks`](@ref), and orient it as `X` is oriented.
 3. Compute the expected return of `X` under that mask.

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
    # A panel's masks are `observations × assets` whatever `dims` says, and the mask-aware
    # verb reads a mask shaped as `X` is, so a transposed sample takes a transposed mask.
    amsk, _ = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.mean(me, X; dims = dims, active_mask = amsk, kwargs...)
end
"""
    Statistics.mean(me::PriceLevelExpectedReturns; kwargs...)

Reads the expected return from the state that a folding statistic carries, as the folded statistic minus one.

An asset that has folded no level since its last reset carries its cold seed, not a forecast. So the read-out is `NaN` for it, as [`ExpWeightedExpectedReturns`](@ref) is below its `min_obs`. One folded level is enough for a forecast, so a `NaN` marks an asset that the mask turned off, or one that has no return yet.

# Validation

  - `me.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - `mu::Matrix{<:Number}`: The expected return, shaped `(1, N)`.

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

# Validation

  - `folds(alg)`. An `ArgumentError` that names the statistic is thrown otherwise.

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

Folds one row of returns, or a block of rows in order, into the state that a folding statistic carries.

The first row seeds the state, and the fold resets an asset that the active mask turns off. An asset is **valid** at a row when its return is finite and the mask admits it, which is the condition that the exponentially weighted family reads. With no mask the finite assets are the active ones. So a gap is a delisting and not a holiday, because nothing else states which it is. The block method folds its rows one at a time through the row method.

# Algorithm

 1. Refuse a statistic that does not fold, and a mask without one entry per asset. Read the valid assets and the active ones.
 2. Turn the row into the price relative `1 .+ x`, with one wherever the asset is not valid. The level did not move, which is how a holiday inside a listing reads.
 3. Append the relative to the memory of a statistic that carries one, through [`push_memory`](@ref). Set the memory of every inactive asset to one, so its levels carry nothing from before the delisting.
 4. Write [`cold_statistic`](@ref) at every inactive asset, and at every asset that has folded no level, so a relisted asset starts the recursion from its seed.
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
