"""
$(DocStringExtensions.TYPEDEF)

The exponential average of the last `window` price levels, truncated and unnormalised: the rising-trend forecast of the trend-promote price tracking of Dai, Liang, Dai, Huang and Adnan (2022).

# Mathematical definition

```math
\\begin{align}
\\mathrm{stat} &= \\alpha \\sum_{k = 0}^{w - 1} (1 - \\alpha)^{k}\\, \\boldsymbol{p}_{t - k}\\,.
\\end{align}
```

The weights fall with age and sum to ``1 - (1 - \\alpha)^{w}``, not to one, so the forecast sits a little below the smoothed level at every `alpha` below one. The paper's own expression for this branch references the price one period ahead and gives the oldest price the largest weight, which is not computable; the reading taken here is the five-term exponential moving average its text names, with the weights decreasing with age.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TruncatedExponentialMovingAverage(; alpha::Real = 0.5, window::Integer = 5) -> TruncatedExponentialMovingAverage

Keywords correspond to the struct's fields.

## Validation

  - `0 < alpha <= 1`. A `DomainError` is thrown otherwise.
  - `window >= 2`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> TruncatedExponentialMovingAverage()
TruncatedExponentialMovingAverage
   alpha ┼ Float64: 0.5
  window ┴ Int64: 5
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`ExponentialMovingAverage`](@ref)
  - [`TrendSwitch`](@ref)
  - [`TrendPromotePriceTracking`](@ref)

# References

  - $(ref_dict[:dai2022tppt])
"""
struct TruncatedExponentialMovingAverage{T1 <: Real, T2 <: Integer} <:
       AbstractPriceLevelStatistic
    """
    The smoothing weight on the current level, `0 < alpha <= 1`.
    """
    alpha::T1
    """
    $(field_dict[:price_window])
    """
    window::T2
    function TruncatedExponentialMovingAverage(alpha::Real, window::Integer)
        @argcheck(zero(alpha) < alpha <= one(alpha),
                  DomainError(alpha, "alpha must be in (0, 1]"))
        assert_price_window(window)
        return new{typeof(alpha), typeof(window)}(alpha, window)
    end
end
function TruncatedExponentialMovingAverage(; alpha::Real = 0.5,
                                           window::Integer = 5)::TruncatedExponentialMovingAverage
    return TruncatedExponentialMovingAverage(alpha, window)
end
function price_level_statistic(alg::TruncatedExponentialMovingAverage, P::AbstractMatrix)
    K = size(P, 1)
    stat = zeros(typeof(alg.alpha * one(eltype(P))), size(P, 2))
    for k in 0:(min(K, alg.window) - 1)
        @views stat .+= alg.alpha * (one(alg.alpha) - alg.alpha)^k .* P[K - k, :]
    end
    return stat
end
"""
$(DocStringExtensions.TYPEDEF)

The Gaussian-weighted double estimate of the next price of Cai and Ye (2019): the left half of a Gaussian over the recent levels, averaged with the same estimate taken with the current level replaced by the previous period's estimate.

# Mathematical definition

With ``g_k = \\exp(-k^2 / (2 \\tau^2))`` for ``k = 1, \\ldots, l`` and ``l = \\lfloor \\sqrt{-2 \\tau^2 \\ln \\epsilon_w} \\rfloor`` the length at which the weight falls below the cutoff,

```math
\\begin{align}
\\hat{\\boldsymbol{p}}^{(1)}_{t} &= \\frac{\\sum_{k = 1}^{l} g_k\\, \\boldsymbol{p}_{t - k + 1}}{\\sum_{k = 1}^{l} g_k}\\,,\\quad
\\hat{\\boldsymbol{p}}^{(2)}_{t} = \\frac{g_1\\, \\hat{\\boldsymbol{p}}^{(1)}_{t - 1} + \\sum_{k = 2}^{l} g_k\\, \\boldsymbol{p}_{t - k + 1}}{\\sum_{k = 1}^{l} g_k}\\,,\\quad
\\mathrm{stat} = \\tfrac{1}{2}\\left(\\hat{\\boldsymbol{p}}^{(1)}_{t} + \\hat{\\boldsymbol{p}}^{(2)}_{t}\\right)\\,.
\\end{align}
```

The previous estimate ``\\hat{\\boldsymbol{p}}^{(1)}_{t - 1}`` is recomputed from the levels ``\\boldsymbol{p}_{t - l}, \\ldots, \\boldsymbol{p}_{t - 1}``, so the statistic is stateless and reads `l` returns. At the paper's `tau = 2.8` and `cutoff = 0.005`, `l = 9`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GaussianWeightedDoubleEstimate(; tau::Real = 2.8, cutoff::Real = 0.005) -> GaussianWeightedDoubleEstimate

Keywords correspond to the struct's fields.

## Validation

  - `tau > 0`. A `DomainError` is thrown otherwise.
  - `0 < cutoff < 1`. A `DomainError` is thrown otherwise.
  - The window `l` is at least one, so `-2 tau^2 log(cutoff) >= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> GaussianWeightedDoubleEstimate()
GaussianWeightedDoubleEstimate
     tau ┼ Float64: 2.8
  cutoff ┴ Float64: 0.005
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`GaussianWeightingReversion`](@ref)

# References

  - $(ref_dict[:caiye2019])
"""
struct GaussianWeightedDoubleEstimate{T1 <: Real, T2 <: Real} <: AbstractPriceLevelStatistic
    """
    The width of the Gaussian, in periods.
    """
    tau::T1
    """
    The weight below which a level leaves the window.
    """
    cutoff::T2
    function GaussianWeightedDoubleEstimate(tau::Real, cutoff::Real)
        @argcheck(tau > zero(tau), DomainError(tau, "tau must be positive"))
        @argcheck(zero(cutoff) < cutoff < one(cutoff),
                  DomainError(cutoff, "cutoff must be in (0, 1)"))
        @argcheck(-2 * tau^2 * log(cutoff) >= 1,
                  DomainError(cutoff,
                              "the Gaussian window is empty: -2 tau^2 log(cutoff) must be at least 1"))
        return new{typeof(tau), typeof(cutoff)}(tau, cutoff)
    end
end
function GaussianWeightedDoubleEstimate(; tau::Real = 2.8,
                                        cutoff::Real = 0.005)::GaussianWeightedDoubleEstimate
    return GaussianWeightedDoubleEstimate(tau, cutoff)
end
function window_rows(alg::GaussianWeightedDoubleEstimate)
    return floor(Int, sqrt(-2 * alg.tau^2 * log(alg.cutoff)))
end
function price_level_statistic(alg::GaussianWeightedDoubleEstimate, P::AbstractMatrix)
    l = window_rows(alg)
    K = size(P, 1)
    g = [exp(-k^2 / (2 * alg.tau^2)) for k in 1:l]
    # The weighted mean of the `l` levels ending at row `u`, truncated at the first row.
    function estimate(u)
        num = zeros(typeof(g[1] * one(eltype(P))), size(P, 2))
        den = zero(g[1])
        for k in 1:min(l, u)
            @views num .+= g[k] .* P[u - k + 1, :]
            den += g[k]
        end
        return num ./ den
    end
    p1 = estimate(K)
    if K == 1
        return p1
    end
    prev = estimate(K - 1)
    num = g[1] .* prev
    den = g[1]
    for k in 2:min(l, K)
        @views num .+= g[k] .* P[K - k + 1, :]
        den += g[k]
    end
    return (p1 .+ num ./ den) ./ 2
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the per-asset trend tests a [`TrendSwitch`](@ref) statistic switches on.

# Interfaces

In order to implement a new trend test, subtype `AbstractTrendTest` with its parameters as part of the struct, and implement:

  - `trend_sign(test::AbstractTrendTest, P::AbstractMatrix) -> AbstractVector`: The sign of the trend per asset over the levels `P`, `+1` rising, `0` flat, `-1` falling.
  - `window_rows(test::AbstractTrendTest) -> Integer`: The number of return rows the test reads, `window - 1` for a test over `window` levels, which is the default.

# Related

  - [`TrendSwitch`](@ref)
  - [`PairwiseSlopeSum`](@ref)
  - [`RegressionSlope`](@ref)
"""
abstract type AbstractTrendTest <: AbstractAlgorithm end
function window_rows(test::AbstractTrendTest)
    return test.window - 1
end
"""
$(DocStringExtensions.TYPEDEF)

The sign of the sum of the pairwise two-point slopes among the last `window` levels, per asset: the trend test of the trend-promote price tracking of Dai, Liang, Dai, Huang and Adnan (2022).

# Mathematical definition

```math
\\begin{align}
s_{i} &= \\operatorname{sign} \\sum_{a < b} \\frac{p_{b, i} - p_{a, i}}{b - a}\\,,\\quad a, b \\in \\{t - w + 1, \\ldots, t\\}\\,.
\\end{align}
```

At `window = 5` the sum runs over the ten pairs the paper's text and its sum name; the paper's displayed slope formula writes only the four slopes anchored at the current level, and the sign of those four can differ from the sign of the ten.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    PairwiseSlopeSum(; window::Integer = 5) -> PairwiseSlopeSum

Keywords correspond to the struct's fields.

## Validation

  - `window >= 2`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> PairwiseSlopeSum()
PairwiseSlopeSum
  window ┴ Int64: 5
```

# Related

  - [`AbstractTrendTest`](@ref)
  - [`TrendSwitch`](@ref)
  - [`TrendPromotePriceTracking`](@ref)

# References

  - $(ref_dict[:dai2022tppt])
"""
struct PairwiseSlopeSum{T1 <: Integer} <: AbstractTrendTest
    """
    $(field_dict[:price_window])
    """
    window::T1
    function PairwiseSlopeSum(window::Integer)
        assert_price_window(window)
        return new{typeof(window)}(window)
    end
end
function PairwiseSlopeSum(; window::Integer = 5)::PairwiseSlopeSum
    return PairwiseSlopeSum(window)
end
"""
    trend_sign(test::PairwiseSlopeSum, P::AbstractMatrix)
    trend_sign(test::RegressionSlope, P::AbstractMatrix)

The sign of the trend of each asset over the levels `P`, `+1` rising, `0` flat and `-1` falling: the sign of the sum of the pairwise slopes, or of the regression slope less its threshold.

# Arguments

  - `test`: The trend test.
  - `P`: The levels of the test's window, `levels × assets`.

# Returns

  - `s::Vector`: The sign per asset.

# Related

  - [`AbstractTrendTest`](@ref)
  - [`TrendSwitch`](@ref)
"""
function trend_sign(::PairwiseSlopeSum, P::AbstractMatrix)
    K = size(P, 1)
    s = zeros(typeof(one(eltype(P)) / 1), size(P, 2))
    for a in 1:(K - 1), b in (a + 1):K
        @views s .+= (P[b, :] .- P[a, :]) ./ (b - a)
    end
    return sign.(s)
end
"""
$(DocStringExtensions.TYPEDEF)

The sign of the ridge-regularised slope of a straight line through the last `window` levels against a threshold, per asset: the trend test of the local adaptive learning of Guan and An (2019).

# Mathematical definition

With ``\\tau = 1, \\ldots, w`` the positions of the window's levels and bars their means,

```math
\\begin{align}
a_{i} &= \\frac{\\sum_{\\tau} (\\tau - \\bar{\\tau})(p_{\\tau, i} - \\bar{p}_{i})}{\\sum_{\\tau} (\\tau - \\bar{\\tau})^2 + \\lambda}\\,,\\quad
s_{i} = \\operatorname{sign}(a_{i} - \\eta)\\,.
\\end{align}
```

The slope is on the reconstructed price path, whose last level is one, so the threshold `eta` is in units of the current price per period, which is what makes the paper's `0.1` comparable across assets; the paper itself regresses on raw prices, where the same threshold scales with each asset's level. The paper states no ridge weight, and `lambda = 0` is plain least squares.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RegressionSlope(; window::Integer = 5, threshold::Real = 0.1, lambda::Real = 0) -> RegressionSlope

Keywords correspond to the struct's fields.

## Validation

  - `window >= 2`. A `DomainError` is thrown otherwise.
  - `lambda >= 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> RegressionSlope()
RegressionSlope
     window ┼ Int64: 5
  threshold ┼ Float64: 0.1
     lambda ┴ Int64: 0
```

# Related

  - [`AbstractTrendTest`](@ref)
  - [`TrendSwitch`](@ref)
  - [`LocalAdaptiveLearning`](@ref)

# References

  - $(ref_dict[:guanan2019])
"""
struct RegressionSlope{T1 <: Integer, T2 <: Real, T3 <: Real} <: AbstractTrendTest
    """
    $(field_dict[:price_window])
    """
    window::T1
    """
    The slope above which an asset is rising.
    """
    threshold::T2
    """
    The ridge weight on the slope, `0` for plain least squares.
    """
    lambda::T3
    function RegressionSlope(window::Integer, threshold::Real, lambda::Real)
        assert_price_window(window)
        @argcheck(lambda >= zero(lambda),
                  DomainError(lambda, "lambda must be non-negative"))
        return new{typeof(window), typeof(threshold), typeof(lambda)}(window, threshold,
                                                                      lambda)
    end
end
function RegressionSlope(; window::Integer = 5, threshold::Real = 0.1,
                         lambda::Real = 0)::RegressionSlope
    return RegressionSlope(window, threshold, lambda)
end
function trend_sign(test::RegressionSlope, P::AbstractMatrix)
    K = size(P, 1)
    tau = 1:K
    tbar = Statistics.mean(tau)
    pbar = vec(Statistics.mean(P; dims = 1))
    num = zeros(typeof(one(eltype(P)) * one(tbar)), size(P, 2))
    den = sum(abs2, tau .- tbar) + test.lambda
    for t in tau
        @views num .+= (t - tbar) .* (P[t, :] .- pbar)
    end
    return iszero(den) ? zero(num) : sign.(num ./ den .- test.threshold)
end
"""
$(DocStringExtensions.TYPEDEF)

A statistic that switches per asset between three others on the sign of a trend test: rising, flat or falling.

The trend-promote price tracking of Dai, Liang, Dai, Huang and Adnan (2022) switches on the [`PairwiseSlopeSum`](@ref) between a [`TruncatedExponentialMovingAverage`](@ref) on a rising asset, the current price on a flat one and the [`WindowPeak`](@ref) on a falling one — reversion on winners, tracking on losers. The local adaptive learning of Guan and An (2019) switches on the [`RegressionSlope`](@ref) between the [`WindowPeak`](@ref) above the threshold and the [`ExponentialMovingAverage`](@ref) otherwise, so its `flat` and `falling` branches are the same statistic. Each branch is evaluated over its own window of the levels the composite holds, which is the largest of the test's and the branches'; a folding branch reads every row, so the composite then holds every row and the branch is the paper's full-history recursion.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TrendSwitch(;
        test::AbstractTrendTest = PairwiseSlopeSum(),
        rising::AbstractPriceLevelStatistic = TruncatedExponentialMovingAverage(),
        flat::AbstractPriceLevelStatistic = LaggedPrice(; lag = 0),
        falling::AbstractPriceLevelStatistic = WindowPeak()
    ) -> TrendSwitch

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> TrendSwitch()
TrendSwitch
     test ┼ PairwiseSlopeSum
          │   window ┴ Int64: 5
   rising ┼ TruncatedExponentialMovingAverage
          │    alpha ┼ Float64: 0.5
          │   window ┴ Int64: 5
     flat ┼ LaggedPrice
          │   lag ┴ Int64: 0
  falling ┼ WindowPeak
          │   window ┴ Int64: 5
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`AbstractTrendTest`](@ref)
  - [`TrendPromotePriceTracking`](@ref)
  - [`LocalAdaptiveLearning`](@ref)

# References

  - $(ref_dict[:dai2022tppt])
  - $(ref_dict[:guanan2019])
"""
struct TrendSwitch{T1 <: AbstractTrendTest, T2 <: AbstractPriceLevelStatistic,
                   T3 <: AbstractPriceLevelStatistic, T4 <: AbstractPriceLevelStatistic} <:
       AbstractPriceLevelStatistic
    """
    The per-asset trend test.
    """
    test::T1
    """
    The statistic of a rising asset.
    """
    rising::T2
    """
    The statistic of a flat asset.
    """
    flat::T3
    """
    The statistic of a falling asset.
    """
    falling::T4
    function TrendSwitch(test::AbstractTrendTest, rising::AbstractPriceLevelStatistic,
                         flat::AbstractPriceLevelStatistic,
                         falling::AbstractPriceLevelStatistic)
        return new{typeof(test), typeof(rising), typeof(flat), typeof(falling)}(test,
                                                                                rising,
                                                                                flat,
                                                                                falling)
    end
end
function TrendSwitch(; test::AbstractTrendTest = PairwiseSlopeSum(),
                     rising::AbstractPriceLevelStatistic = TruncatedExponentialMovingAverage(),
                     flat::AbstractPriceLevelStatistic = LaggedPrice(; lag = 0),
                     falling::AbstractPriceLevelStatistic = WindowPeak())::TrendSwitch
    return TrendSwitch(test, rising, flat, falling)
end
function window_rows(alg::TrendSwitch)
    return rows_needed_max(rows_needed_max(window_rows(alg.test), window_rows(alg.rising)),
                           rows_needed_max(window_rows(alg.flat), window_rows(alg.falling)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The statistic of a member of a composite over the last `window_rows(alg) + 1` levels of `P` ending at row `u`, truncated at the first row.

# Arguments

  - `alg`: The member statistic.
  - `P`: The levels the composite holds.
  - `u`: The row of `P` the member's window ends at.

# Returns

  - `stat::Vector`: The member's statistic per asset, in units of the levels.

# Related

  - [`TrendSwitch`](@ref)
  - [`CompositeTrend`](@ref)
"""
function member_statistic(alg::AbstractPriceLevelStatistic, P::AbstractMatrix, u::Integer)
    need = window_rows(alg)
    lo = isnothing(need) ? 1 : max(1, u - need)
    return price_level_statistic(alg, view(P, lo:u, :))
end
function price_level_statistic(alg::TrendSwitch, P::AbstractMatrix)
    K = size(P, 1)
    s = trend_sign(alg.test, view(P, max(1, K - window_rows(alg.test)):K, :))
    up = member_statistic(alg.rising, P, K)
    flat = member_statistic(alg.flat, P, K)
    down = member_statistic(alg.falling, P, K)
    stat = similar(up)
    for i in eachindex(s)
        stat[i] = s[i] > zero(s[i]) ? up[i] : (s[i] < zero(s[i]) ? down[i] : flat[i])
    end
    return stat
end
"""
$(DocStringExtensions.TYPEDEF)

The adaptive input and composite trend representation of Lai, Dai, Ren and Huang (2018): a radial-basis mix of several trend forecasts, centred on the trend whose simplex-projected forecast had the best worst return over the last `window` periods.

# Mathematical definition

With ``\\hat{\\boldsymbol{x}}_{l}`` the Price Relative Forecast of trend ``l`` over the current window and ``\\tilde{\\boldsymbol{x}}_{l} = \\mathrm{Proj}_{\\Delta}(\\hat{\\boldsymbol{x}}_{l})`` its trend portfolio,

```math
\\begin{align}
R_{l, t - k} &= \\langle \\tilde{\\boldsymbol{x}}_{l, t - k - 1}, \\boldsymbol{x}_{t - k} \\rangle\\,,\\quad k = 0, \\ldots, w - 1\\,,\\quad
\\ast = \\arg\\max_{l} \\min_{k} R_{l, t - k}\\,,\\\\
\\varphi_{l} &= \\exp\\left(-\\frac{\\lVert \\tilde{\\boldsymbol{x}}_{\\ast} - \\tilde{\\boldsymbol{x}}_{l} \\rVert^2}{2 \\sigma^2}\\right)\\,,\\quad
\\mathrm{stat} = \\sum_{l} \\varphi_{l}\\, \\hat{\\boldsymbol{x}}_{l}\\,,
\\end{align}
```

where the trend portfolio formed at ``t - k - 1`` is scored on the relative of the period that followed. The paper leaves the weights unnormalised, ``\\varphi_{\\ast} = 1`` and the others below it, because the tracking step that reads the composite normalises its centred direction and only the ratios matter; the statistic divides by ``\\sum_l \\varphi_l`` so that it is a level forecast on its own, which changes nothing in the step. The back-test needs the trend forecasts of the last `window` periods, so the composite reads `window - 1` rows more than its widest trend; over the first rows it back-tests the periods available, and with none it centres on the first trend. The paper's trends are the simple moving average, the exponential moving average and the window peak, and it states no smoothing weight for the exponential average; the library's default is taken and the docstring of [`AdaptiveInputCompositeTrend`](@ref) says so.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    CompositeTrend(;
        trends::AbstractVector{<:AbstractPriceLevelStatistic} = [MovingAverage(), ExponentialMovingAverage(), WindowPeak()],
        window::Integer = 5,
        sigma2::Real = 0.0025
    ) -> CompositeTrend

Keywords correspond to the struct's fields.

## Validation

  - `trends` is non-empty. An `IsEmptyError` is thrown otherwise.
  - `window >= 2`. A `DomainError` is thrown otherwise.
  - `sigma2 > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> CompositeTrend()
CompositeTrend
  trends ┼ 3-element Vector{PortfolioOptimisers.AbstractPriceLevelStatistic}
         │ MovingAverage ⋯
         │ ExponentialMovingAverage ⋯
         │ WindowPeak ⋯
  window ┼ Int64: 5
  sigma2 ┴ Float64: 0.0025
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`AdaptiveInputCompositeTrend`](@ref)
  - [`ForecastTracking`](@ref)
  - [`project_simplex`](@ref)

# References

  - $(ref_dict[:lai2018aictr])
"""
struct CompositeTrend{T1 <: AbstractVector{<:AbstractPriceLevelStatistic}, T2 <: Integer,
                      T3 <: Real} <: AbstractPriceLevelStatistic
    """
    The trend statistics mixed, one forecast each.
    """
    trends::T1
    """
    The number of periods the trend portfolios are back-tested over to choose the centre.
    """
    window::T2
    """
    The squared width of the radial basis function over the trend portfolios.
    """
    sigma2::T3
    function CompositeTrend(trends::AbstractVector{<:AbstractPriceLevelStatistic},
                            window::Integer, sigma2::Real)
        @argcheck(!isempty(trends), IsEmptyError("trends must hold at least one statistic"))
        assert_price_window(window)
        @argcheck(sigma2 > zero(sigma2), DomainError(sigma2, "sigma2 must be positive"))
        return new{typeof(trends), typeof(window), typeof(sigma2)}(trends, window, sigma2)
    end
end
function CompositeTrend(;
                        trends::AbstractVector{<:AbstractPriceLevelStatistic} = [MovingAverage(),
                                                                                 ExponentialMovingAverage(),
                                                                                 WindowPeak()],
                        window::Integer = 5, sigma2::Real = 0.0025)::CompositeTrend
    return CompositeTrend(trends, window, sigma2)
end
function window_rows(alg::CompositeTrend)
    widest = window_rows(alg.trends[1])
    for trend in alg.trends
        widest = rows_needed_max(widest, window_rows(trend))
    end
    return isnothing(widest) ? nothing : widest + alg.window - 1
end
function price_level_statistic(alg::CompositeTrend, P::AbstractMatrix)
    K = size(P, 1)
    L = length(alg.trends)
    xhat = [member_statistic(trend, P, K) ./ view(P, K, :) for trend in alg.trends]
    xt = project_simplex.(xhat)
    # The back-test: the trend portfolio formed at row `u` scored on the relative of row
    # `u + 1`, over the last `window` periods the levels reach.
    us = [K - k - 1 for k in 0:(alg.window - 1) if K - k - 1 >= 1]
    star = if isempty(us)
        1
    else
        argmax([minimum(LinearAlgebra.dot(project_simplex(member_statistic(alg.trends[l], P,
                                                                           u) ./
                                                          view(P, u, :)),
                                          view(P, u + 1, :) ./ view(P, u, :)) for u in us)
                for l in 1:L])
    end
    stat = zero(xhat[1])
    total = zero(alg.sigma2 * one(eltype(stat)))
    for l in 1:L
        phi = exp(-sum(abs2, xt[star] .- xt[l]) / (2 * alg.sigma2))
        stat .+= phi .* xhat[l]
        total += phi
    end
    return stat .* view(P, K, :) ./ total
end
"""
$(DocStringExtensions.TYPEDEF)

The elastic-net regularisation path of Friedman, Hastie and Tibshirani (2010), read at its middle point: the regression of the kernel trend pattern's initial state on the window's price columns.

# Mathematical definition

For a response ``\\boldsymbol{y}`` over the assets and the columns ``\\boldsymbol{P}`` of one level each, the coefficients at a strength ``\\lambda`` solve the elastic net of Zou and Hastie (2005),

```math
\\begin{align}
\\hat{\\boldsymbol{z}}(\\lambda) &= \\arg\\min_{\\boldsymbol{z}} \\lVert \\boldsymbol{y} - \\boldsymbol{P} \\boldsymbol{z} \\rVert^2 + \\lambda \\left( 2 \\vartheta \\lVert \\boldsymbol{z} \\rVert_1 + (1 - \\vartheta) \\lVert \\boldsymbol{z} \\rVert^2 \\right)\\,,
\\end{align}
```

by cyclic coordinate descent, each coordinate in turn being ``z_k = S(\\boldsymbol{P}_k^\\intercal \\boldsymbol{r}_k, \\lambda \\vartheta) / (\\lVert \\boldsymbol{P}_k \\rVert^2 + \\lambda (1 - \\vartheta))`` with ``\\boldsymbol{r}_k`` the residual without column ``k`` and ``S`` the soft threshold. Every coefficient is zero from ``\\lambda_{\\max} = \\max_k \\lvert \\boldsymbol{P}_k^\\intercal \\boldsymbol{y} \\rvert / \\vartheta`` up. The path runs from ``\\lambda_{\\max}`` down to ``\\lambda_{\\max} \\cdot \\texttt{ratio}`` on a log scale, so its middle point is ``\\lambda_{\\max} \\sqrt{\\texttt{ratio}}``, and the coefficients are solved there alone, seeded at zero: the problem has as many columns as the window has levels, so the rest of the path costs nothing to skip. The columns are the levels of consecutive periods, nearly collinear, where the sweeps alone converge slowly; each sweep therefore ends with the exact solve of [`elastic_net_polish`](@ref) on the sign pattern it left, accepted as the optimum when the optimality conditions hold, which the strict convexity makes sufficient, so the sweeps only have to find the active set. They find it slowly: on random collinear panels shaped like the kernel trend pattern's, a cap of 1000 sweeps left about three in five unpolished, and the default of 10 000 polished every one; an exit at `tol` or at `iters` returns the last sweep, which is not the optimum. `ratio` is the floor the paper that defines the path uses; `theta = 0.99` is the kernel trend pattern's, nearly the lasso and strictly convex.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ElasticNetPath(; theta::Real = 0.99, ratio::Real = 1e-3, iters::Integer = 10_000, tol::Real = 1e-10) -> ElasticNetPath

Keywords correspond to the struct's fields.

## Validation

  - `0 < theta <= 1`. A `DomainError` is thrown otherwise: at `theta = 0` the penalty is a ridge alone, which has no strength above which every coefficient is zero.
  - `0 < ratio < 1`. A `DomainError` is thrown otherwise.
  - `iters >= 1`. A `DomainError` is thrown otherwise.
  - `tol > 0`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> ElasticNetPath()
ElasticNetPath
  theta ┼ Float64: 0.99
  ratio ┼ Float64: 0.001
  iters ┼ Int64: 10000
    tol ┴ Float64: 1.0e-10
```

# Related

  - [`KernelTrendPattern`](@ref)
  - [`elastic_net_path`](@ref)

# References

  - $(ref_dict[:friedman2010])
  - $(ref_dict[:zouhastie2005])
"""
struct ElasticNetPath{T1 <: Real, T2 <: Real, T3 <: Integer, T4 <: Real} <:
       AbstractAlgorithm
    """
    The share of the ``L_1`` penalty, `1` for the lasso.
    """
    theta::T1
    """
    The floor of the path as a fraction of the strength above which every coefficient is zero.
    """
    ratio::T2
    """
    Maximum number of coordinate-descent sweeps, each ending with the exact solve on its sign pattern.
    """
    iters::T3
    """
    Tolerance on the largest coefficient change over a sweep, below which the sweeps stop should no sign pattern have been accepted.
    """
    tol::T4
    function ElasticNetPath(theta::Real, ratio::Real, iters::Integer, tol::Real)
        @argcheck(zero(theta) < theta <= one(theta),
                  DomainError(theta, "theta must be in (0, 1]"))
        @argcheck(zero(ratio) < ratio < one(ratio),
                  DomainError(ratio, "ratio must be in (0, 1)"))
        @argcheck(iters >= 1, DomainError(iters, "iters must be at least 1"))
        @argcheck(tol > zero(tol), DomainError(tol, "tol must be positive"))
        return new{typeof(theta), typeof(ratio), typeof(iters), typeof(tol)}(theta, ratio,
                                                                             iters, tol)
    end
end
function ElasticNetPath(; theta::Real = 0.99, ratio::Real = 1e-3, iters::Integer = 10_000,
                        tol::Real = 1e-10)::ElasticNetPath
    return ElasticNetPath(theta, ratio, iters, tol)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The coefficients of the elastic net of `y` on the columns of `P` at the middle point of the path, by cyclic coordinate descent from zero.

# Arguments

  - `path`: The path.
  - `P`: The columns, `observations × columns`.
  - `y`: The response, `observations × 1`.

# Returns

  - `z::Vector`: The coefficients, `columns × 1`.

# Related

  - [`ElasticNetPath`](@ref)
  - [`elastic_net_polish`](@ref)
  - [`KernelTrendPattern`](@ref)
"""
function elastic_net_path(path::ElasticNetPath, P::AbstractMatrix, y::AbstractVector)
    w = size(P, 2)
    g = map(k -> LinearAlgebra.dot(view(P, :, k), y), 1:w)
    lmax = maximum(abs, g) / path.theta
    lam = lmax * sqrt(path.ratio)
    z = zeros(typeof(lam), w)
    if iszero(lmax)
        return z
    end
    a = lam * path.theta
    nrm2 = vec(sum(abs2, P; dims = 1))
    den = nrm2 .+ lam * (one(path.theta) - path.theta)
    r = collect(y)
    for _ in 1:(path.iters)
        delta = zero(eltype(z))
        for k in 1:w
            Pk = view(P, :, k)
            rho = LinearAlgebra.dot(Pk, r) + nrm2[k] * z[k]
            znew = sign(rho) * max(abs(rho) - a, zero(a)) / den[k]
            if znew != z[k]
                r .-= (znew - z[k]) .* Pk
                delta = max(delta, abs(znew - z[k]))
                z[k] = znew
            end
        end
        zp = elastic_net_polish(P, y, z, lam, path.theta)
        if !isnothing(zp)
            return zp
        end
        delta < path.tol && break
    end
    return z
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The exact elastic-net optimum on the sign pattern of `z`, when that pattern is the optimum's: the ridge normal equations over the active columns with the ``L_1`` subgradient fixed at the pattern's signs, accepted when the solved coefficients keep those signs and every inactive column's residual correlation is inside the threshold ``\\lambda \\vartheta``. The problem is strictly convex, so a point meeting both is its one optimum; `nothing` says the pattern is not it.

# Arguments

  - `P`: The columns, `observations × columns`.
  - `y`: The response.
  - `z`: The coefficients whose sign pattern is tried.
  - `lam`: The regularisation strength.
  - `theta`: The share of the ``L_1`` penalty.

# Returns

  - `z'::Union{Nothing, Vector}`: The optimum, or `nothing`.

# Related

  - [`elastic_net_path`](@ref)
"""
function elastic_net_polish(P::AbstractMatrix, y::AbstractVector, z::AbstractVector,
                            lam::Real, theta::Real)
    A = findall(!iszero, z)
    if isempty(A)
        return nothing
    end
    PA = view(P, :, A)
    s = sign.(view(z, A))
    zA = (PA' * PA + lam * (one(theta) - theta) * LinearAlgebra.I) \
         (PA' * y .- lam * theta .* s)
    if any(v -> v <= zero(v), zA .* s)
        return nothing
    end
    zp = zero(z)
    zp[A] .= zA
    corr = P' * (y .- P * zp)
    for k in eachindex(zp)
        if iszero(zp[k]) && abs(corr[k]) > lam * theta
            return nothing
        end
    end
    return zp
end
"""
$(DocStringExtensions.TYPEDEF)

The three-state price prediction of the kernel-based trend pattern tracking of Lai, Yang, Wu and Fang (2018): a folding statistic with a memory, which carries its previous prediction and the last `2 window` relatives.

# Mathematical definition

With ``L \\leq 2w + 1`` the levels the memory reaches, ``\\boldsymbol{p}_{t}`` the current one and ``\\hat{\\boldsymbol{p}}_{t}`` the prediction the previous row made for it, the three states are

```math
\\begin{align}
\\tilde{\\boldsymbol{p}}_{t+1} &= \\max_{0 \\leq k < w} \\boldsymbol{p}_{t-k}\\,,\\quad
\\boldsymbol{y}_{t+1} = \\nu \\tilde{\\boldsymbol{p}}_{t+1} + (1 - \\nu)\\, \\hat{\\boldsymbol{p}}_{t}\\,,\\\\
\\hat{\\boldsymbol{y}}_{t+1} &= \\max\\left(\\boldsymbol{P}_{t} \\hat{\\boldsymbol{z}}_{t}, 0\\right)\\,,\\quad
\\boldsymbol{P}_{t} = [\\boldsymbol{p}_{t-w+1}, \\ldots, \\boldsymbol{p}_{t}]\\,,\\\\
\\lambda_{t+1} &= \\frac{1}{(L - 2)\\, d} \\sum_{i, k} \\mathbb{1}\\left[ (p_{k, i} - p_{k-1, i})(p_{k-2, i} - p_{k-1, i}) > 0 \\right]\\,,\\\\
\\hat{\\boldsymbol{p}}_{t+1} &= \\boldsymbol{c} \\odot \\tilde{\\boldsymbol{p}}_{t+1} + (\\boldsymbol{1} - \\boldsymbol{c}) \\odot \\hat{\\boldsymbol{y}}_{t+1}\\,,\\quad
\\boldsymbol{c} = \\min\\left( \\frac{\\lambda_{t+1}}{2 \\boldsymbol{x}_{t}}, 1 \\right)\\,,
\\end{align}
```

where ``\\hat{\\boldsymbol{z}}_{t}`` is the [`ElasticNetPath`](@ref) regression of ``\\boldsymbol{y}_{t+1}`` on the columns of ``\\boldsymbol{P}_{t}``, the assets being the observations, and the sum in ``\\lambda_{t+1}`` runs over the assets and the levels ``k = 3, \\ldots, L``, so it counts the turning points — a rise followed by a fall, or a fall by a rise — in the memory. The initial state ``\\boldsymbol{y}`` mixes the window peak with the previous prediction; the intermediate state ``\\hat{\\boldsymbol{y}}`` is that mix re-expressed by the recent levels, negatives clipped; the final state moves from ``\\hat{\\boldsymbol{y}}`` toward the peak by the reverting strength, long-term in ``\\lambda`` and short-term in ``1 / \\boldsymbol{x}_{t}``, and the Price Relative Forecast is ``\\hat{\\boldsymbol{p}}_{t+1} \\oslash \\boldsymbol{p}_{t}``. Over the first rows the window and the memory hold the levels available, as the paper's cold start states, and ``\\lambda`` is zero until three levels exist; the first prediction is seeded at the current price.

The regression pools the assets as observations, so it is not homogeneous in each asset's level alone: it is read on the reconstructed price path, whose last level is one for every asset, so that the assets enter in comparable units. The paper regresses on raw prices, where an asset quoted at a hundred times another's dominates the fit.

Under an active mask the fit takes the live assets alone, because [`partial_fit!`](@ref) folds the row's Coverage Universe and holds the rest. An asset the mask turns off leaves the memory flat over its dead span, so the levels it brings back when it relists are its relisting level repeated: the window peak is then the peak of the levels the asset has, which is the paper's own truncation, and the fit reads one flat column for it until the memory fills again.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KernelTrendPattern(; window::Integer = 5, nu::Real = 0.5, path::ElasticNetPath = ElasticNetPath()) -> KernelTrendPattern

Keywords correspond to the struct's fields, and the defaults are the paper's. The statistic folds with a memory of `2 window` relatives, so the head holds no rows for it.

## Validation

  - `window >= 2`. A `DomainError` is thrown otherwise.
  - `0 <= nu <= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> KernelTrendPattern()
KernelTrendPattern
  window ┼ Int64: 5
      nu ┼ Float64: 0.5
    path ┼ ElasticNetPath
         │   theta ┼ Float64: 0.99
         │   ratio ┼ Float64: 0.001
         │   iters ┼ Int64: 10000
         │     tol ┴ Float64: 1.0e-10
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`ElasticNetPath`](@ref)
  - [`WindowPeak`](@ref)
  - [`KernelTrendTracking`](@ref)
  - [`KernelTrendPatternTracking`](@ref)
  - [`fold_statistic`](@ref)
  - [`memory_rows`](@ref)

# References

  - $(ref_dict[:lai2018ktpt])
"""
struct KernelTrendPattern{T1 <: Integer, T2 <: Real, T3 <: ElasticNetPath} <:
       AbstractPriceLevelStatistic
    """
    $(field_dict[:price_window])
    """
    window::T1
    """
    The weight of the window peak in the initial state, the previous prediction taking the rest.
    """
    nu::T2
    """
    The elastic-net path of the intermediate state.
    """
    path::T3
    function KernelTrendPattern(window::Integer, nu::Real, path::ElasticNetPath)
        assert_price_window(window)
        @argcheck(zero(nu) <= nu <= one(nu), DomainError(nu, "nu must be in [0, 1]"))
        return new{typeof(window), typeof(nu), typeof(path)}(window, nu, path)
    end
end
function KernelTrendPattern(; window::Integer = 5, nu::Real = 0.5,
                            path::ElasticNetPath = ElasticNetPath())::KernelTrendPattern
    return KernelTrendPattern(window, nu, path)
end
function folds(::KernelTrendPattern)
    return true
end
function window_rows(::KernelTrendPattern)
    return nothing
end
function memory_rows(alg::KernelTrendPattern)
    return 2 * alg.window
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

The trend-reverting fraction of a matrix of levels: the share of turning points among the cells of the levels from the third on, a turning point being a level whose two neighbouring differences have opposite signs. Zero below three levels.

# Arguments

  - `P`: The levels, `levels × assets`.

# Returns

  - `lambda::Real`: The fraction, in `[0, 1]`.

# Related

  - [`KernelTrendPattern`](@ref)
"""
function trend_reverting_fraction(P::AbstractMatrix)
    L, N = size(P)
    m = L - 2
    if m <= 0
        return zero(inv(N))
    end
    count = 0
    for j in 1:N, i in 3:L
        if (P[i, j] - P[i - 1, j]) * (P[i - 2, j] - P[i - 1, j]) > zero(eltype(P))
            count += 1
        end
    end
    return count / (m * N)
end
function cold_statistic(::KernelTrendPattern, x::AbstractVector)
    return collect(x)
end
function fold_statistic(alg::KernelTrendPattern, stat::Option{<:AbstractVector},
                        hist::AbstractMatrix, x::AbstractVector)
    P = price_levels(hist .- one(eltype(hist)))
    L = size(P, 1)
    Pw = view(P, max(1, L - alg.window + 1):L, :)
    ptilde = vec(maximum(Pw; dims = 1))
    # The previous prediction was made in units of the previous level; the current level is
    # one, so it is divided by the row's relative. The cold seed is the row's relative, so
    # that an asset the fold has just reset enters at a prediction of one.
    prev = (isnothing(stat) ? cold_statistic(alg, x) : stat) ./ x
    y = alg.nu .* ptilde .+ (one(alg.nu) - alg.nu) .* prev
    Pt = permutedims(Pw)
    yhat = max.(Pt * elastic_net_path(alg.path, Pt, y), zero(eltype(y)))
    c = min.(trend_reverting_fraction(P) ./ (2 .* x), one(eltype(x)))
    return c .* ptilde .+ (one(eltype(c)) .- c) .* yhat
end
function price_level_statistic(alg::KernelTrendPattern, P::AbstractMatrix)
    return fold_levels(alg, P)
end
export TruncatedExponentialMovingAverage, GaussianWeightedDoubleEstimate, PairwiseSlopeSum,
       RegressionSlope, TrendSwitch, CompositeTrend, ElasticNetPath, KernelTrendPattern
public AbstractTrendTest, trend_sign, member_statistic
