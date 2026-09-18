"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the statistics of a window of price levels that [`PriceLevelExpectedReturns`](@ref) turns into an expected return.

Every statistic is homogeneous of degree one in the levels, so the level of the last observation is set to one and the earlier levels are reconstructed from the returns of the window; the statistic is then a vector over the assets, and the expected return is `stat / p_t .- 1` with `p_t = 1`.

# Interfaces

In order to implement a new price-level statistic, subtype `AbstractPriceLevelStatistic` with all its parameters as part of the struct, and implement the following methods:

  - `price_level_statistic(alg::AbstractPriceLevelStatistic, P::AbstractMatrix) -> AbstractVector`: The statistic of the levels `P`, `levels × assets`, whose last row is the current level, one per asset.
  - `rows_needed(alg::AbstractPriceLevelStatistic) -> Integer`: The number of return rows the statistic reads to reconstruct its levels, `window - 1` for a statistic over `window` levels.

## Arguments

  - `alg`: The concrete statistic.
  - `P`: The reconstructed price levels of the window, `levels × assets`, the last row being ones.

## Returns

  - `stat::AbstractVector`: The statistic per asset, `assets × 1`.
  - `rows::Integer`: The number of return rows the statistic reads.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`MovingAverage`](@ref)
  - [`SpatialMedian`](@ref)
  - [`price_level_statistic`](@ref)
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

The spatial (``L_1``) median of the last `window` price levels, the forecast of the robust median reversion of Huang, Zhou, Li, Hoi and Zhou (2016).

The median is the point minimising the sum of Euclidean distances to the `window` level vectors, found by the modified Weiszfeld iteration of Vardi and Zhang (2000), which is seeded at the coordinatewise median and stays defined when an iterate lands on one of the level vectors.

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
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a price-level window below two levels.

# Related

  - [`MovingAverage`](@ref)
  - [`SpatialMedian`](@ref)
"""
function assert_price_window(window::Integer)::Nothing
    @argcheck(window >= 2,
              DomainError(window,
                          "window must be at least 2: one level is the current price, and its ratio to itself forecasts nothing"))
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

The expected return that takes the last price to a statistic of a window of price levels: the Price Relative Forecast of the online portfolio selection family, as an expected-returns estimator any `me` slot may hold.

The five statistics of the literature — a moving average, an exponential moving average, a spatial median, a window peak and a lagged price — are homogeneous of degree one in the levels, so the forecast is a function of the last `window - 1` returns alone: the level of the last observation is set to one, the earlier levels are reconstructed from the returns, and the expected return is the statistic divided by the last level, less one. The mean is therefore a forecast of the **next** period's return under the reversion hypothesis of the paper that defines the statistic, not an average of past returns. This build ships [`MovingAverage`](@ref) and [`SpatialMedian`](@ref).

Fewer than `window - 1` returns truncate the window, as the reversion papers' reference implementations do: over the first rows the statistic reads the levels available.

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

    PriceLevelExpectedReturns(; alg::AbstractPriceLevelStatistic = MovingAverage()) -> PriceLevelExpectedReturns

Keywords correspond to the struct's fields.

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
  - [`MovingAverage`](@ref)
  - [`SpatialMedian`](@ref)
  - [`ForecastReversion`](@ref)
  - [`rows_needed`](@ref)
"""
struct PriceLevelExpectedReturns{T1 <: AbstractPriceLevelStatistic} <:
       AbstractExpectedReturnsEstimator
    """
    The price-level statistic the forecast reads.
    """
    alg::T1
    function PriceLevelExpectedReturns(alg::AbstractPriceLevelStatistic)
        return new{typeof(alg)}(alg)
    end
end
function PriceLevelExpectedReturns(;
                                   alg::AbstractPriceLevelStatistic = MovingAverage())::PriceLevelExpectedReturns
    return PriceLevelExpectedReturns(alg)
end
"""
    rows_needed(me::PriceLevelExpectedReturns)
    rows_needed(me::AbstractExpectedReturnsEstimator)
    rows_needed(alg::AbstractPriceLevelStatistic)

The number of return rows an expected-returns estimator reads at a step of the online portfolio selection family, or `nothing` when it reads every row folded so far.

A price-level statistic over `window` levels reads `window - 1` returns. Any other expected-returns estimator answers `nothing`, which is unbounded: the head keeps every row for it and it refits on the whole prefix at every step. [`ForecastReversion`](@ref) forwards to its `me` slot, and the head takes the maximum over its rule tree.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`AbstractPriceLevelStatistic`](@ref)
  - [`ForecastReversion`](@ref)
  - [`OnlinePortfolioSelection`](@ref)
"""
function rows_needed(me::PriceLevelExpectedReturns)
    return rows_needed(me.alg)
end
function rows_needed(::AbstractExpectedReturnsEstimator)
    return nothing
end
function rows_needed(alg::AbstractPriceLevelStatistic)
    return alg.window - 1
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
    price_level_statistic(alg::SpatialMedian, P::AbstractMatrix)

The statistic of a window of price levels, one value per asset.

The moving average is the column mean of the levels. The spatial median is the point minimising the sum of Euclidean distances to the level vectors, through [`spatial_median`](@ref).

# Arguments

  - `alg`: The statistic.
  - `P`: The levels, `levels × assets`, the last row the current level.

# Returns

  - `stat::Vector`: The statistic per asset.

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`PriceLevelExpectedReturns`](@ref)
  - [`spatial_median`](@ref)
"""
function price_level_statistic(::MovingAverage, P::AbstractMatrix)
    return vec(Statistics.mean(P; dims = 1))
end
function price_level_statistic(alg::SpatialMedian, P::AbstractMatrix)
    return spatial_median(P, alg.iters, alg.tol)
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
    Statistics.mean(me::PriceLevelExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)

Compute the expected return as the ratio of a price-level statistic to the last price, less one.

The levels are reconstructed with [`price_levels`](@ref) from the last `window - 1` rows of `X`, or from every row when `X` holds fewer, the statistic on `me.alg` is read through [`price_level_statistic`](@ref), and the answer is the statistic less one, because the last level is one.

# Arguments

  - `me`: The estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - `X` holds at least one observation. An `IsEmptyError` is thrown otherwise.

# Returns

  - `mu::Matrix{<:Number}`: The expected return, shaped as `(1, N)` if `dims == 1` or `(N, 1)` if `dims == 2`.

# Related

  - [`PriceLevelExpectedReturns`](@ref)
  - [`price_levels`](@ref)
  - [`price_level_statistic`](@ref)
"""
function Statistics.mean(me::PriceLevelExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
    assert_dims(dims)
    if dims == 2
        X = transpose(X)
    end
    @argcheck(size(X, 1) >= 1,
              IsEmptyError("X must hold at least one observation to reconstruct a price level"))
    k = min(size(X, 1), rows_needed(me.alg))
    P = price_levels(view(X, (size(X, 1) - k + 1):size(X, 1), :))
    mu = price_level_statistic(me.alg, P) .- one(eltype(P))
    return dims == 1 ? reshape(mu, 1, :) : reshape(mu, :, 1)
end
export PriceLevelExpectedReturns, MovingAverage, SpatialMedian
public AbstractPriceLevelStatistic, price_level_statistic, rows_needed
