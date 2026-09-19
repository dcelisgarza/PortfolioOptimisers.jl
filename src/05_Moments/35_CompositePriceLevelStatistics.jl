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

At `window = 5` the sum runs over the ten pairs the paper names.

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
export TruncatedExponentialMovingAverage, GaussianWeightedDoubleEstimate, PairwiseSlopeSum,
       RegressionSlope, TrendSwitch, CompositeTrend
public AbstractTrendTest, trend_sign, member_statistic
