"""
$(DocStringExtensions.TYPEDEF)

Takes an exponential average of the last `window` price levels, truncated and not normalised, which is the forecast of a rising asset in the trend promote price tracing, TPPT.

Equation (7) and Algorithm 1 of [dai2022tppt](@cite) print this forecast as a sum of five terms. Its first term reads the level of the next period, ``\\boldsymbol{p}_{t+1}``, which is not known at period ``t``. If that term means ``\\boldsymbol{p}_{t-4}``, the oldest level gets the largest weight. The library keeps the five coefficients of [dai2022tppt](@cite) and gives the largest one to the current level, as an exponential moving average does. Algorithm 1 of [dai2022tppt](@cite) sets `alpha = 0.5`, and the experiments of [dai2022tppt](@cite) use `window = 5`. These are the defaults.

# Mathematical definition

```math
\\begin{align}
\\hat{\\boldsymbol{x}}_{t+1} &= \\alpha \\sum_{k = 0}^{w - 1} (1 - \\alpha)^{k}\\, \\boldsymbol{p}_{t - k} \\oslash \\boldsymbol{p}_{t}\\,.
\\end{align}
```

Where:

  - $(math_dict[:xhat_fc])
  - $(math_dict[:alpha_ema])
  - $(math_dict[:p_t_level])
  - $(math_dict[:w_levels])

The weights fall with age and sum to ``1 - (1 - \\alpha)^{w}``, not to one. On a flat path the forecast is therefore below one at every ``\\alpha`` below one, and at the defaults the weights sum to 0.96875. At ``\\alpha = 1`` the forecast is one in every asset.

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

  - $(ref_dict[:dai2022tppt]) Equation (7) and Algorithm 1.
"""
@concrete struct TruncatedExponentialMovingAverage <: AbstractPriceLevelStatistic
    """
    The smoothing weight on the current level, `0 < alpha <= 1`.
    """
    alpha
    """
    $(field_dict[:price_window])
    """
    window
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

Averages two Gaussian-weighted estimates of the next price level, which is the forecast of the Gaussian weighting reversion, GWR.

It is the forecast of [caiye2019](@citet). The first estimate weights the last ``l`` levels with the left half of a Gaussian. The second estimate is the same weighted mean, with the current level replaced by the first estimate of the previous period. The weights fall with age, and the window ``l`` is the number of levels whose weight is at least the cutoff. At `tau = 2.8` and `cutoff = 0.005`, the values of [caiye2019](@cite) and the defaults, ``l = 9``. The previous estimate is a function of the levels ``\\boldsymbol{p}_{t-l}, \\ldots, \\boldsymbol{p}_{t-1}``, so the statistic carries no state and reads ``l`` returns. Over fewer levels, each estimate reads the levels it has.

# Mathematical definition

```math
\\begin{align}
g_k &= \\exp\\left(-\\frac{k^2}{2 \\tau^2}\\right)\\,,\\quad k = 1, \\ldots, l\\,,\\quad l = \\left\\lfloor \\sqrt{-2 \\tau^2 \\ln \\epsilon_w} \\right\\rfloor\\,,\\\\
\\hat{\\boldsymbol{p}}^{(1)}_{t} &= \\frac{\\sum_{k = 1}^{l} g_k\\, \\boldsymbol{p}_{t - k + 1}}{\\sum_{k = 1}^{l} g_k}\\,,\\\\
\\hat{\\boldsymbol{p}}^{(2)}_{t} &= \\frac{g_1\\, \\hat{\\boldsymbol{p}}^{(1)}_{t - 1} + \\sum_{k = 2}^{l} g_k\\, \\boldsymbol{p}_{t - k + 1}}{\\sum_{k = 1}^{l} g_k}\\,,\\\\
\\hat{\\boldsymbol{x}}_{t+1} &= \\tfrac{1}{2}\\left(\\hat{\\boldsymbol{p}}^{(1)}_{t} + \\hat{\\boldsymbol{p}}^{(2)}_{t}\\right) \\oslash \\boldsymbol{p}_{t}\\,.
\\end{align}
```

Where:

  - ``g_k``: Gaussian weight of the level ``k - 1`` periods before the current one.
  - ``\\tau``: Width of the Gaussian, in periods.
  - ``\\epsilon_w``: Cutoff, the smallest weight that a level in the window takes.
  - ``l``: Gaussian window, the number of levels that each estimate reads.
  - ``\\hat{\\boldsymbol{p}}^{(1)}_{t}``, ``\\hat{\\boldsymbol{p}}^{(2)}_{t}``: First and second estimates of the next level, made at period ``t``.
  - $(math_dict[:p_t_level])
  - $(math_dict[:xhat_fc])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GaussianWeightedDoubleEstimate(; tau::Real = 2.8, cutoff::Real = 0.005) -> GaussianWeightedDoubleEstimate

Keywords correspond to the struct's fields.

## Validation

  - `tau > 0`. A `DomainError` is thrown otherwise.
  - `0 < cutoff < 1`. A `DomainError` is thrown otherwise.
  - `-2 tau^2 log(cutoff) >= 1`, so that the window holds at least one level. A `DomainError` is thrown otherwise.

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

  - $(ref_dict[:caiye2019]) Equations (2) to (6).
"""
@concrete struct GaussianWeightedDoubleEstimate <: AbstractPriceLevelStatistic
    """
    The width of the Gaussian, in periods.
    """
    tau
    """
    The weight below which a level leaves the window.
    """
    cutoff
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

Abstract supertype for the per-asset trend tests that a [`TrendSwitch`](@ref) switches on.

A test returns one sign per asset: `+1` for a rising asset, `0` for a flat one and `-1` for a falling one. Over fewer than two levels no trend exists, and the tests of the library return zero in every asset.

# Interfaces

To implement a new trend test, subtype `AbstractTrendTest` with its parameters as part of the struct, and implement the following methods:

  - `trend_sign(test::AbstractTrendTest, P::AbstractMatrix) -> AbstractVector`: The trend sign of each asset over the levels `P`, `levels × assets`, whose last row is the current level.
  - `window_rows(test::AbstractTrendTest) -> Integer`: The number of return rows that the test reads. The default reads `test.window - 1`, for a test over `window` levels.

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

Takes the sign of the sum of the slopes between every pair of the last `window` levels, per asset, which is the trend test of the trend promote price tracing, TPPT.

It is the trend test of [dai2022tppt](@citet). At `window = 5` the sum runs over ten pairs. The text of [dai2022tppt](@cite) counts these ten pairs, and its Equation (7) sums them. The slope formula of [dai2022tppt](@cite), Equation (6), writes only the four slopes from the current level, and the sign of those four can differ from the sign of the ten. The test has no threshold, so an asset is flat only when the sum is exactly zero, as it is on a constant window.

# Mathematical definition

```math
\\begin{align}
s_{i} &= \\operatorname{sign} \\sum_{t - w < a < b \\leq t} \\frac{p_{b, i} - p_{a, i}}{b - a}\\,.
\\end{align}
```

Where:

  - $(math_dict[:s_i_trend])
  - $(math_dict[:p_ti_level])
  - $(math_dict[:t_period])
  - $(math_dict[:w_levels])

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

  - $(ref_dict[:dai2022tppt]) Equations (6) and (7).
"""
@concrete struct PairwiseSlopeSum <: AbstractTrendTest
    """
    $(field_dict[:price_window])
    """
    window
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

Returns the trend sign of each asset over the levels `P`: `+1` for a rising asset, `0` for a flat one and `-1` for a falling one.

The pairwise test takes the sign of the sum of its slopes, and the regression test takes the sign of its slope minus its threshold. Each test states its sign in its own `# Mathematical definition`. Over fewer than two levels both tests return zero in every asset.

# Arguments

  - `test`: The trend test.
  - `P`: The levels of the test's window, `levels × assets`, whose last row is the current level.

# Returns

  - `s::Vector`: The sign per asset, in the element type of the levels or a wider one.

# Related

  - [`AbstractTrendTest`](@ref)
  - [`TrendSwitch`](@ref)
"""
function trend_sign(::PairwiseSlopeSum, P::AbstractMatrix)
    K = size(P, 1)
    s = zeros(float_if_integer(eltype(P)), size(P, 2))
    for a in 1:(K - 1), b in (a + 1):K
        @views s .+= (P[b, :] .- P[a, :]) ./ (b - a)
    end
    return sign.(s)
end
"""
$(DocStringExtensions.TYPEDEF)

Compares the ridge-regularised slope of a straight line through the last `window` levels with a threshold, per asset, which is the trend test of the local adaptive learning, LOAD.

It is the trend test of [guanan2019](@citet). The line has a free intercept, and the ridge weight penalises the slope alone, as in Equation (4) of [guanan2019](@cite). [guanan2019](@citet) state no ridge weight, and `lambda = 0` is plain least squares. The experiments of [guanan2019](@cite) use `window = 5` and `threshold = 0.1`, the defaults. An asset is rising when its slope is above the threshold, flat when the slope equals it, and falling when the slope is below it. So an asset with a small positive slope is falling. Local adaptive learning reads the same statistic on the flat and the falling branch, so this does not change the rule of [guanan2019](@cite).

The test takes the slope on the reconstructed path, whose last level is one. The threshold is therefore in units of the current price per period, and the value `0.1` of [guanan2019](@cite) means the same for every asset. [guanan2019](@citet) regress on their own price series and state no normalisation, so there the threshold scales with the level of each asset.

# Mathematical definition

```math
\\begin{align}
a_{i} &= \\frac{\\sum_{\\tau = 1}^{w} (\\tau - \\bar{\\tau})(p_{t - w + \\tau, i} - \\bar{p}_{i})}{\\sum_{\\tau = 1}^{w} (\\tau - \\bar{\\tau})^2 + \\lambda}\\,,\\\\
s_{i} &= \\operatorname{sign}(a_{i} - \\eta)\\,.
\\end{align}
```

Where:

  - ``a_i``: Ridge slope of asset ``i``, in levels per period.
  - ``\\tau``: Position of a level in the window, one for the oldest.
  - ``\\bar{\\tau} = (w + 1) / 2``: Mean position.
  - ``\\bar{p}_{i}``: Mean level of asset ``i`` over the window.
  - ``\\lambda``: Ridge weight on the slope.
  - ``\\eta``: Threshold on the slope.
  - $(math_dict[:s_i_trend])
  - $(math_dict[:p_ti_level])
  - $(math_dict[:t_period])
  - $(math_dict[:w_levels])

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

  - $(ref_dict[:guanan2019]) Equations (3) and (4).
"""
@concrete struct RegressionSlope <: AbstractTrendTest
    """
    $(field_dict[:price_window])
    """
    window
    """
    The slope above which an asset is rising.
    """
    threshold
    """
    The ridge weight on the slope, `0` for plain least squares.
    """
    lambda
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
    # The centre of the positions is exact, so it takes the element type of the levels.
    tbar = (K + 1) // 2
    pbar = vec(Statistics.mean(P; dims = 1))
    num = zeros(typeof(one(eltype(P)) * one(tbar)), size(P, 2))
    if K < 2
        return num
    end
    den = sum(t -> abs2(t - tbar), 1:K) + test.lambda
    for t in 1:K
        @views num .+= (t - tbar) .* (P[t, :] .- pbar)
    end
    return sign.(num ./ den .- test.threshold)
end
"""
$(DocStringExtensions.TYPEDEF)

Switches per asset between three statistics on the sign of a trend test, one statistic for a rising asset, one for a flat asset and one for a falling asset.

The trend promote price tracing of [dai2022tppt](@citet) switches on the [`PairwiseSlopeSum`](@ref). It takes the [`TruncatedExponentialMovingAverage`](@ref) of a rising asset, the current price of a flat one and the [`WindowPeak`](@ref) of a falling one. The local adaptive learning of [guanan2019](@citet) switches on the [`RegressionSlope`](@ref). It takes the [`WindowPeak`](@ref) above the threshold and the [`ExponentialMovingAverage`](@ref) otherwise, so its `flat` and `falling` branches hold the same statistic.

The composite holds the largest window of its test and its branches, and it reads each of them over its own window. A folding branch reads every level. The composite then holds every row, and the branch runs its published recursion over the full history.

# Mathematical definition

```math
\\begin{align}
\\hat{x}_{t+1, i} &= \\begin{cases}
u_{i} / p_{t, i} & s_{i} = +1\\,,\\\\
f_{i} / p_{t, i} & s_{i} = 0\\,,\\\\
d_{i} / p_{t, i} & s_{i} = -1\\,.
\\end{cases}
\\end{align}
```

Where:

  - $(math_dict[:xhat_fc]) Its entry ``i`` is ``\\hat{x}_{t+1, i}``.
  - $(math_dict[:s_i_trend]) The test of the switch gives it.
  - ``u_i``, ``f_i``, ``d_i``: Statistics of asset ``i`` under the rising, the flat and the falling branch, each over its own window.
  - $(math_dict[:p_ti_level])

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

  - $(ref_dict[:dai2022tppt]) Equation (7).
  - $(ref_dict[:guanan2019]) Equations (5) and (6).
"""
@concrete struct TrendSwitch <: AbstractPriceLevelStatistic
    """
    The per-asset trend test.
    """
    test
    """
    The statistic of a rising asset.
    """
    rising
    """
    The statistic of a flat asset.
    """
    flat
    """
    The statistic of a falling asset.
    """
    falling
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

Returns the statistic of a member of a composite over the member's own window of the levels that end at row `u`.

# Algorithm

 1. Read `need`, the number of return rows that the member reads, from [`window_rows`](@ref).
 2. Take `lo`, the first row of the member's window. It is the later of `u - need` and the first row of `P`, or the first row when `need` is `nothing`.
 3. Return the member's statistic over the rows `lo` to `u` of `P`.

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

Mixes several trend forecasts with radial basis weights, centred on the trend whose simplex-projected forecast had the best worst return over the last `window` periods, which is the forecast of the adaptive input and composite trend representation, AICTR.

It is the forecast of [lai2018aictr](@citet). [lai2018aictr](@citet) do not normalise the weights, so the centre takes weight one and every other trend takes less. The composite divides by the sum of the weights, so that it is a forecast on its own. The tracking step of [`AdaptiveInputCompositeTrend`](@ref) scales its centred direction to a fixed length, as Equation (15) and Algorithm 1 of [lai2018aictr](@cite) do, so the division does not change the step. Equation (16) of [lai2018aictr](@cite) calls that scaling a projection onto a ball, which would keep the length of a shorter vector. The library follows eq. (15).

The back-test reads the trend forecasts of the last `window` periods, so the composite reads `window - 1` rows more than its widest trend. Over the first rows it back-tests the periods that it has, and without a period to back-test it centres on the first trend. [lai2018aictr](@citet) state no rule for these rows. The trends of [lai2018aictr](@cite) are the simple moving average, the exponential moving average and the window peak, over `window = 5` levels, with `sigma2 = 0.0025`. These are the defaults. [lai2018aictr](@citet) state no smoothing weight for the exponential moving average, and the default of [`ExponentialMovingAverage`](@ref) stands.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\boldsymbol{x}}_{l, t+1} &= \\mathrm{Proj}_{\\Delta_N}(\\hat{\\boldsymbol{x}}_{l, t+1})\\,,\\\\
R_{l, t - k} &= \\langle \\tilde{\\boldsymbol{x}}_{l, t - k}, \\boldsymbol{x}_{t - k} \\rangle\\,,\\quad k = 0, \\ldots, w - 1\\,,\\\\
\\ast &= \\underset{l}{\\arg\\max} \\min_{k} R_{l, t - k}\\,,\\\\
\\varphi_{l} &= \\exp\\left(-\\frac{\\lVert \\tilde{\\boldsymbol{x}}_{\\ast, t+1} - \\tilde{\\boldsymbol{x}}_{l, t+1} \\rVert^2}{2 \\sigma^2}\\right)\\,,\\\\
\\hat{\\boldsymbol{x}}_{t+1} &= \\frac{\\sum_{l} \\varphi_{l}\\, \\hat{\\boldsymbol{x}}_{l, t+1}}{\\sum_{l} \\varphi_{l}}\\,.
\\end{align}
```

Where:

  - ``\\hat{\\boldsymbol{x}}_{l, t+1}``: Price Relative Forecast of trend ``l`` for period ``t + 1``, made after the row of period ``t``.
  - ``\\tilde{\\boldsymbol{x}}_{l, t+1}``: Trend portfolio of trend ``l`` for period ``t + 1``.
  - ``\\mathrm{Proj}_{\\Delta_N}``: Euclidean projection onto ``\\Delta_N``, which [`project_simplex`](@ref) computes.
  - $(math_dict[:Delta_N_simplex])
  - ``R_{l, t - k}``: Return of the trend portfolio of trend ``l`` over period ``t - k``.
  - ``w``: Back-test window, the number of periods over which the back-test scores the trend portfolios.
  - ``\\ast``: Centre, the trend with the largest worst return over the back-test window.
  - ``\\varphi_{l}``: Radial basis weight of trend ``l``, one at the centre.
  - ``\\sigma^2``: Squared width of the radial basis function.
  - $(math_dict[:x_t_rel])
  - $(math_dict[:xhat_fc])

The composite is a weighted mean of the trend forecasts, so it lies between the smallest and the largest of them in every asset.

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

  - $(ref_dict[:lai2018aictr]) Equations (5) to (7) and (9) to (12), and (15).
"""
@concrete struct CompositeTrend <: AbstractPriceLevelStatistic
    """
    The trend statistics that the composite mixes, one forecast each.
    """
    trends
    """
    The number of periods over which the back-test scores the trend portfolios to choose the centre.
    """
    window
    """
    The squared width of the radial basis function over the trend portfolios.
    """
    sigma2
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

Predicts the next price level in three states, which is the forecast of the kernel-based trend pattern tracking, KTPT.

It is the forecast of [lai2018ktpt](@citet). It is a folding statistic with a memory. It carries its previous prediction and the last `2 window` price relatives. The initial state mixes the window peak with the previous prediction. The intermediate state is the regression of that mix on the recent levels, with negative values clipped to zero as the text of [lai2018ktpt](@cite) states. The final state moves from the intermediate state toward the peak by a reverting strength, which is long-term through the trend-reverting fraction and short-term through the reciprocal of the last price relative. The defaults `window = 5` and `nu = 0.5` are the values of [lai2018ktpt](@cite).

Over the first rows the window and the memory hold the levels that exist, as Algorithm 1 of [lai2018ktpt](@cite) states. [lai2018ktpt](@citet) state no rule for fewer than three levels or for the first prediction. The library sets the trend-reverting fraction to zero below three levels, and it seeds the first prediction at the current price.

`tgt` is the regression target of the intermediate state. The default [`ElasticNet`](@ref) is the elastic net of Equation (15) of [lai2018ktpt](@cite), at the middle point of its path, with no intercept and no standardisation. The last level of the window is one for every asset, so the last column of the design is a column of ones. Equation (15) penalises its coefficient, so the fold does not name it with [`InterceptColumn`](@ref). Any other [`AbstractRegressionTarget`](@ref) with `StatsAPI.fit` and `StatsAPI.predict` can replace the elastic net, for example [`LinearModel`](@ref).

The regression pools the assets as its observations, so the fit changes when the levels of one asset are scaled. The library reads it on the reconstructed path, whose last level is one for every asset, so the assets enter in the same units. The data sets of [lai2018ktpt](@cite) hold price relatives, and [lai2018ktpt](@citet) do not state the anchor of the prices that they rebuild from them. Under any other anchor, an asset with higher levels than the others weighs more in the fit.

Under an active mask the fit takes the active assets alone, because [`partial_fit!`](@ref) folds the Coverage Universe of the row and holds the other assets. The memory of an asset that the mask turns off stays flat over the inactive rows. When the asset relists, the memory holds its relisting level repeated. The window peak is then the peak of the levels that the asset has, which is the truncation of [lai2018ktpt](@cite), and the fit reads a flat column for the asset until the memory fills again.

# Mathematical definition

```math
\\begin{align}
\\tilde{\\boldsymbol{p}}_{t+1} &= \\max_{0 \\leq k < w} \\boldsymbol{p}_{t-k}\\,,\\\\
\\boldsymbol{y}_{t+1} &= \\nu \\tilde{\\boldsymbol{p}}_{t+1} + (1 - \\nu)\\, \\hat{\\boldsymbol{p}}_{t}\\,,\\\\
\\hat{\\boldsymbol{y}}_{t+1} &= \\max\\left(\\mathbf{P}_{t} \\hat{\\boldsymbol{z}}_{t}, 0\\right)\\,,\\quad
\\mathbf{P}_{t} = [\\boldsymbol{p}_{t-w+1}, \\ldots, \\boldsymbol{p}_{t}]\\,,\\\\
\\lambda_{t+1} &= \\frac{1}{(L - 2) N} \\sum_{i = 1}^{N} \\sum_{k = t - L + 3}^{t} \\mathbb{1}\\left[ (p_{k, i} - p_{k-1, i})(p_{k-2, i} - p_{k-1, i}) > 0 \\right]\\,,\\\\
\\boldsymbol{c} &= \\min\\left( \\frac{\\lambda_{t+1}}{2} \\boldsymbol{1} \\oslash \\boldsymbol{x}_{t}, \\boldsymbol{1} \\right)\\,,\\\\
\\hat{\\boldsymbol{p}}_{t+1} &= \\boldsymbol{c} \\odot \\tilde{\\boldsymbol{p}}_{t+1} + (\\boldsymbol{1} - \\boldsymbol{c}) \\odot \\hat{\\boldsymbol{y}}_{t+1}\\,,\\\\
\\hat{\\boldsymbol{x}}_{t+1} &= \\hat{\\boldsymbol{p}}_{t+1} \\oslash \\boldsymbol{p}_{t}\\,.
\\end{align}
```

Where:

  - ``\\tilde{\\boldsymbol{p}}_{t+1}``: Window peak, the highest level of each asset in the window.
  - ``\\hat{\\boldsymbol{p}}_{t}``: Prediction of the level of period ``t``, made at period ``t - 1``.
  - ``\\nu``: Weight of the window peak in the initial state.
  - ``\\boldsymbol{y}_{t+1}``, ``\\hat{\\boldsymbol{y}}_{t+1}``: Initial and intermediate states.
  - ``\\mathbf{P}_{t}``: Window of levels, one row per asset and one column per level.
  - ``\\hat{\\boldsymbol{z}}_{t}``: Coefficients of the fit of `tgt` of ``\\boldsymbol{y}_{t+1}`` on the columns of ``\\mathbf{P}_{t}``. Under the default, the elastic net at the middle point of its path, [`MiddleOfPath`](@ref). The assets are the observations.
  - ``\\lambda_{t+1}``: Trend-reverting fraction, the share of the interior levels of the memory at which the path of an asset turns.
  - ``L``: Number of levels that the memory reaches, at most ``2w + 1``.
  - ``\\boldsymbol{c}``: Reverting strength, one entry per asset.
  - $(math_dict[:p_t_level])
  - $(math_dict[:p_ti_level])
  - $(math_dict[:x_t_rel])
  - $(math_dict[:xhat_fc])
  - $(math_dict[:w_levels])
  - $(math_dict[:N])

A level turns when its two neighbours are both above it or both below it. The fraction is in ``[0, 1]``, so ``\\boldsymbol{c}`` is at most one half when no asset fell in the last period.

# Algorithm

For each row, the fold runs these steps.

 1. Rebuild `P`, the levels of the memory, from the relatives that the memory holds, with the current level at one.
 2. Take `Pw`, the last `window` levels of `P`, and `ptilde`, their peak per asset.
 3. Bring the carried prediction into the units of the current level, which gives `prev`. Before the first row, the carried value is the cold seed, so `prev` is one.
 4. Mix `ptilde` and `prev` with the weight `nu`, which gives the initial state `y`.
 5. Fit `tgt` to `y` on the columns of the transpose of `Pw` through `StatsAPI.fit`, predict the same design through `StatsAPI.predict`, and clip the prediction at zero, which gives `yhat`.
 6. Compute the reverting strength `c` from [`trend_reverting_fraction`](@ref) over `P` and the relative `x` of the row.
 7. Return the final state, the mix of `ptilde` and `yhat` with the weights `c`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    KernelTrendPattern(; window::Integer = 5, nu::Real = 0.5, tgt::AbstractRegressionTarget = ElasticNet()) -> KernelTrendPattern

Keywords correspond to the struct's fields. The statistic folds with a memory of `2 window` relatives, so the head holds no rows for it.

## Validation

  - `window >= 2`. A `DomainError` is thrown otherwise.
  - `0 <= nu <= 1`. A `DomainError` is thrown otherwise.

# Examples

```jldoctest
julia> KernelTrendPattern()
KernelTrendPattern
  window ┼ Int64: 5
      nu ┼ Float64: 0.5
     tgt ┼ ElasticNet
         │      theta ┼ Float64: 0.99
         │   strength ┼ MiddleOfPath
         │            │   ratio ┴ Float64: 0.001
         │         ve ┼ nothing
         │      iters ┼ Int64: 10000
         │        tol ┼ Float64: 1.0e-10
         │          w ┼ nothing
         │         ic ┴ nothing
```

# Related

  - [`AbstractPriceLevelStatistic`](@ref)
  - [`ElasticNet`](@ref)
  - [`AbstractRegressionTarget`](@ref)
  - [`WindowPeak`](@ref)
  - [`KernelTrendTracking`](@ref)
  - [`KernelTrendPatternTracking`](@ref)
  - [`fold_statistic`](@ref)
  - [`memory_rows`](@ref)

# References

  - $(ref_dict[:lai2018ktpt]) Equations (11) to (16) and (26) to (31), Algorithm 1.
"""
@concrete struct KernelTrendPattern <: AbstractPriceLevelStatistic
    """
    $(field_dict[:price_window])
    """
    window
    """
    The weight of the window peak in the initial state. The previous prediction takes the rest.
    """
    nu
    """
    The regression target of the intermediate state.
    """
    tgt
    function KernelTrendPattern(window::Integer, nu::Real, tgt::AbstractRegressionTarget)
        assert_price_window(window)
        @argcheck(zero(nu) <= nu <= one(nu), DomainError(nu, "nu must be in [0, 1]"))
        return new{typeof(window), typeof(nu), typeof(tgt)}(window, nu, tgt)
    end
end
function KernelTrendPattern(; window::Integer = 5, nu::Real = 0.5,
                            tgt::AbstractRegressionTarget = ElasticNet())::KernelTrendPattern
    return KernelTrendPattern(window, nu, tgt)
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

Returns the trend-reverting fraction of a matrix of levels, the share of the interior levels, over every asset, at which the path turns.

A level turns when its two neighbours are both above it or both below it, so the path rises and then falls there, or falls and then rises. A matrix of ``L`` levels and ``N`` assets has ``(L - 2) N`` interior levels. The fraction is zero below three levels. [`KernelTrendPattern`](@ref) states it as ``\\lambda_{t+1}``, after Equation (27) of [lai2018ktpt](@cite).

# Arguments

  - `P`: The levels, `levels × assets`.

# Returns

  - `lambda::Real`: The fraction, in `[0, 1]`, in the element type of the levels or a wider one.

# Related

  - [`KernelTrendPattern`](@ref)

# References

  - $(ref_dict[:lai2018ktpt]) Equation (27).
"""
function trend_reverting_fraction(P::AbstractMatrix)
    L, N = size(P)
    m = L - 2
    if m <= 0
        return zero(one(eltype(P)) / N)
    end
    count = 0
    for j in 1:N, i in 3:L
        if (P[i, j] - P[i - 1, j]) * (P[i - 2, j] - P[i - 1, j]) > zero(eltype(P))
            count += 1
        end
    end
    return count * one(eltype(P)) / (m * N)
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
    yhat = max.(StatsAPI.predict(StatsAPI.fit(alg.tgt, Pt, y), Pt), zero(eltype(y)))
    c = min.(trend_reverting_fraction(P) ./ (2 .* x), one(eltype(x)))
    return c .* ptilde .+ (one(eltype(c)) .- c) .* yhat
end
function price_level_statistic(alg::KernelTrendPattern, P::AbstractMatrix)
    return fold_levels(alg, P)
end
export TruncatedExponentialMovingAverage, GaussianWeightedDoubleEstimate, PairwiseSlopeSum,
       RegressionSlope, TrendSwitch, CompositeTrend, KernelTrendPattern
public AbstractTrendTest, trend_sign, member_statistic
