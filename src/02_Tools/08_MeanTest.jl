"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the tests of the null hypothesis that a serially dependent series has a mean of zero.

A Mean Test reads one series, such as the information coefficient series of a factor or of a Return Forecast. It answers a t-statistic, a two-sided p-value and a confidence interval of the mean. The consecutive entries of such a series can be correlated, so a test that treats them as independent overstates the evidence. Each member of the family states how it reads that dependence.

# Interfaces

In order to implement a new Mean Test that works seamlessly with the library, subtype `AbstractMeanTestEstimator` and implement the following method:

## Test method

  - `mean_test(mt::AbstractMeanTestEstimator, c::VecNum, window::Real, rng) -> NamedTuple`: Tests the mean of `c` against zero.

### Arguments

  - `mt`: Concrete subtype instance.
  - `c`: The series. A `NaN` entry is an observation with no value.
  - `window`: Number of entries that the data of one entry spans, from the grid of the series, `horizon / step` for a forward window. A window above one means that consecutive entries share data. A test can ignore it.
  - `rng`: The random number generator that [`mean_test_rng`](@ref) resolves, or `nothing`.

### Returns

  - `res::NamedTuple`: `(; t_stat, p_value, mean_ci_lower, mean_ci_upper)`.

## Generator method

  - `mean_test_rng(mt::AbstractMeanTestEstimator) -> Option{<:Random.AbstractRNG}`: The generator that the test draws from. The default method answers `nothing`. A test that draws random numbers implements it, and resolves its generator with [`resolve_rng`](@ref).

# Related

  - [`OverlapMeanTest`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
  - [`mean_test`](@ref)
  - [`mean_test_rng`](@ref)
  - [`exposure_ic_summary`](@ref)
  - [`forecast_ic_summary`](@ref)
"""
abstract type AbstractMeanTestEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the kernels that weight the autocovariances of a long-run variance.

The long-run variance of a series is its variance plus twice the weighted sum of its autocovariances. The kernel gives the weight of the autocovariance at lag ``j`` as ``k(j / B)``, where ``B`` is the bandwidth. A kernel has ``k(0) = 1``.

# Interfaces

In order to implement a new kernel that works seamlessly with the library, subtype `AbstractLongRunKernel` and implement the following method:

## Weight method

  - `long_run_kernel_weight(kernel::AbstractLongRunKernel, x::Real) -> Real`: The weight ``k(x)`` at the lag ``x`` in units of the bandwidth.

### Arguments

  - `kernel`: Concrete subtype instance.
  - `x`: The lag over the bandwidth, `x > 0`.

### Returns

  - `w::Real`: The weight. A weight of zero removes the lag from the sum.

A kernel that implements the weight alone takes the generic path. Its long-run variance reads every pair of the series, and its wild bootstrap does one matrix product for each block of draws, so the cost is quadratic in the length of the series. [`BartlettKernel`](@ref) has a form that is linear in the length.

# Related

  - [`BartlettKernel`](@ref)
  - [`QuadraticSpectralKernel`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
  - [`long_run_kernel_weight`](@ref)
"""
abstract type AbstractLongRunKernel <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Weights the autocovariance at lag ``j`` by ``1 - j / B`` up to the bandwidth ``B``, and by zero after it.

The Bartlett kernel gives a long-run variance that is never negative. Its fixed-bandwidth form is the one that the source recommends with the wild bootstrap. The library reads it off sums of windows of the series, so its cost is linear in the length of the series at every bandwidth.

# Mathematical definition

```math
k(x) = \\max\\left(1 - |x|,\\, 0\\right)\\,.
```

Where:

  - ``x``: The lag over the bandwidth.

# Constructors

    BartlettKernel() -> BartlettKernel

# Examples

```jldoctest
julia> PortfolioOptimisers.long_run_kernel_weight(BartlettKernel(), 0.25)
0.75
```

# Related

  - [`AbstractLongRunKernel`](@ref)
  - [`QuadraticSpectralKernel`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
  - [`long_run_kernel_weight`](@ref)

# References

  - $(ref_dict[:neweywest1987])
  - $(ref_dict[:demetrescu2026robust])
"""
struct BartlettKernel <: AbstractLongRunKernel end
"""
$(DocStringExtensions.TYPEDEF)

Weights the autocovariance at every lag by the quadratic spectral kernel, which is positive near zero and decays with an oscillation.

The kernel has no finite support, so the long-run variance reads every lag of the series. Its estimate is never negative. The cost of its wild bootstrap is quadratic in the length of the series.

# Mathematical definition

```math
k(x) = \\frac{25}{12 \\pi^{2} x^{2}} \\left( \\frac{\\sin(6 \\pi x / 5)}{6 \\pi x / 5} - \\cos(6 \\pi x / 5) \\right)\\,.
```

Where:

  - ``x``: The lag over the bandwidth.

# Constructors

    QuadraticSpectralKernel() -> QuadraticSpectralKernel

# Examples

```jldoctest
julia> PortfolioOptimisers.long_run_kernel_weight(QuadraticSpectralKernel(), 0.5)
0.6869307300640595
```

# Related

  - [`AbstractLongRunKernel`](@ref)
  - [`BartlettKernel`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
  - [`long_run_kernel_weight`](@ref)

# References

  - $(ref_dict[:andrews1991]) Equation 2.7.
  - $(ref_dict[:demetrescu2026robust])
"""
struct QuadraticSpectralKernel <: AbstractLongRunKernel end
"""
    long_run_kernel_weight(kernel::BartlettKernel, x::Real) -> Real
    long_run_kernel_weight(kernel::QuadraticSpectralKernel, x::Real) -> Real

Return the weight of a kernel at the lag `x`, in units of the bandwidth.

# Arguments

  - `kernel`: The kernel.
  - `x`: The lag over the bandwidth.

# Returns

  - `w::Real`: ``k(x)``. The quadratic spectral kernel is `1` at `x = 0`.

# Related

  - [`AbstractLongRunKernel`](@ref)
  - [`BartlettKernel`](@ref)
  - [`QuadraticSpectralKernel`](@ref)
"""
function long_run_kernel_weight(::BartlettKernel, x::Real)
    return max(one(x) - abs(x), zero(x))
end
function long_run_kernel_weight(::QuadraticSpectralKernel, x::Real)
    if iszero(x)
        return one(x) / one(x)
    end
    z = 6 * oftype(x / one(x), pi) * x / 5
    return 25 / (12 * oftype(z, pi)^2 * x^2) * (sin(z) / z - cos(z))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the rules that set the span of a series, which a fixed-fraction bandwidth and a history check read.

# Interfaces

In order to implement a new span rule that works seamlessly with the library, subtype `AbstractMeanTestSpan` and implement the following method:

## Length method

  - `mean_test_span_length(span::AbstractMeanTestSpan, P::Integer, i1::Integer, i2::Integer) -> Integer`: The span of the series.

### Arguments

  - `span`: Concrete subtype instance.
  - `P`: The number of entries of the series.
  - `i1`, `i2`: The positions of its first and its last finite entry.

### Returns

  - `T::Integer`: The span, at least `i2 - i1 + 1`.

# Related

  - [`FiniteEntrySpan`](@ref)
  - [`FullGridSpan`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
  - [`mean_test_span_length`](@ref)
"""
abstract type AbstractMeanTestSpan <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Sets the span of a series to the entries from its first finite entry to its last.

A run of missing entries at an end of the series, such as the dates whose forward window has not matured, does not widen the bandwidth. A missing entry inside the span keeps its position.

# Constructors

    FiniteEntrySpan() -> FiniteEntrySpan

# Examples

```jldoctest
julia> PortfolioOptimisers.mean_test_span_length(FiniteEntrySpan(), 10, 3, 8)
6
```

# Related

  - [`AbstractMeanTestSpan`](@ref)
  - [`FullGridSpan`](@ref)
  - [`mean_test_span_length`](@ref)
"""
struct FiniteEntrySpan <: AbstractMeanTestSpan end
"""
$(DocStringExtensions.TYPEDEF)

Sets the span of a series to every entry of its grid, the missing entries at its ends included.

# Constructors

    FullGridSpan() -> FullGridSpan

# Examples

```jldoctest
julia> PortfolioOptimisers.mean_test_span_length(FullGridSpan(), 10, 3, 8)
10
```

# Related

  - [`AbstractMeanTestSpan`](@ref)
  - [`FiniteEntrySpan`](@ref)
  - [`mean_test_span_length`](@ref)
"""
struct FullGridSpan <: AbstractMeanTestSpan end
"""
    mean_test_span_length(span::FiniteEntrySpan, P::Integer, i1::Integer, i2::Integer) -> Integer
    mean_test_span_length(span::FullGridSpan, P::Integer, i1::Integer, i2::Integer) -> Integer

Return the span of a series under a span rule.

# Arguments

  - `span`: The span rule.
  - `P`: The number of entries of the series.
  - `i1`, `i2`: The positions of its first and its last finite entry.

# Returns

  - `T::Integer`: `i2 - i1 + 1` under [`FiniteEntrySpan`](@ref), and `P` under [`FullGridSpan`](@ref).

# Related

  - [`AbstractMeanTestSpan`](@ref)
"""
function mean_test_span_length(::FiniteEntrySpan, ::Integer, i1::Integer, i2::Integer)
    return i2 - i1 + 1
end
function mean_test_span_length(::FullGridSpan, P::Integer, ::Integer, ::Integer)
    return P
end
"""
    long_run_quadratic_form(kernel::AbstractLongRunKernel, d::VecNum, B::Real, buf::VecNum)
    long_run_quadratic_form(kernel::BartlettKernel, d::VecNum, B::Real, buf::VecNum)

Return the sum of the products of every pair of entries of a centred series, each weighted by the kernel at the distance of the pair.

The value is the long-run variance of the series times its count less one. The generic method sums the products at every lag that has a weight that is not zero. The Bartlett method reads the same value off sums of windows of the series, so its cost is linear in the length of the series. Write ``B = m + f``, with ``m`` the integer part. The squared sums of the windows of ``m`` entries give the weight ``m - |j|`` at every lag ``|j| < m``. The products of each entry with the sum of the ``2m + 1`` entries around it give the weight ``f`` at every lag ``|j| \\leq m``. The two together give ``B - |j|`` at every lag below ``B``. An entry of `d` that is zero adds nothing, so a missing observation is zero in `d`, and the distance of a pair is its distance in the series.

# Mathematical definition

```math
\\begin{align}
Q &= \\sum_{i, j} k\\left(\\frac{|i - j|}{B}\\right) d_{i} d_{j}\\,, \\\\
Q_{\\mathrm{Bartlett}} &= \\frac{1}{B} \\left( \\sum_{s = 2 - m}^{P} \\left( \\sum_{t = \\max(s, 1)}^{\\min(s + m - 1, P)} d_{t} \\right)^{2} + f \\sum_{i = 1}^{P} d_{i} \\sum_{t = \\max(i - m, 1)}^{\\min(i + m, P)} d_{t} \\right)\\,.
\\end{align}
```

Where:

  - ``d_{i}``: Entry ``i`` of the centred series.
  - ``k``: The kernel.
  - ``B``: The bandwidth, ``m`` its integer part and ``f`` its fractional part.
  - ``P``: The length of the series.

# Arguments

  - `kernel`: The kernel.
  - `d`: The centred series, with a zero at each missing observation.
  - `B`: The bandwidth, in entries, `B > 0`.
  - `buf`: Work buffer of length `length(d) + 1`. The generic method does not read it.

# Returns

  - `Q::Real`: The weighted sum.

# Related

  - [`AbstractLongRunKernel`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
"""
function long_run_quadratic_form(kernel::AbstractLongRunKernel, d::VecNum, B::Real,
                                 ::VecNum)
    P = length(d)
    Q = zero(eltype(d))
    for t in 1:P
        Q += d[t] * d[t]
    end
    for j in 1:(P - 1)
        w = long_run_kernel_weight(kernel, j / B)
        if !iszero(w)
            Q += 2 * w * lag_cross_sum(d, j)
        end
    end
    return Q
end
function long_run_quadratic_form(::BartlettKernel, d::VecNum, B::Real, buf::VecNum)
    P = length(d)
    buf[1] = zero(eltype(buf))
    for t in 1:P
        buf[t + 1] = buf[t] + d[t]
    end
    m = floor(Int, B)
    f = B - m
    Q = zero(eltype(buf))
    for s in (2 - m):P
        u = buf[min(s + m - 1, P) + 1] - buf[max(s, 1)]
        Q += u * u
    end
    if f > zero(f)
        R = zero(eltype(buf))
        for i in 1:P
            R += d[i] * (buf[min(i + m, P) + 1] - buf[max(i - m, 1)])
        end
        Q += f * R
    end
    return Q / B
end
"""
    lag_cross_sum(d::VecNum, j::Integer) -> Real

Return the sum of the products of the entries of a centred series that are `j` entries apart.

# Arguments

  - `d`: The centred series, with a zero at each missing observation.
  - `j`: The lag.

# Returns

  - `a::Real`: ``\\sum_{t} d_{t} d_{t + j}``.

# Related

  - [`long_run_quadratic_form`](@ref)
  - [`OverlapMeanTest`](@ref)
"""
function lag_cross_sum(d::VecNum, j::Integer)
    a = zero(eltype(d))
    for t in 1:(length(d) - j)
        a += d[t] * d[t + j]
    end
    return a
end
"""
$(DocStringExtensions.TYPEDEF)

Tests the mean of a series against zero over the untapered long-run variance at the order that the overlap of its windows sets.

The test reads the overlap off the `window` that the caller derives from the grid, such as [`forecast_ic_window`](@ref). Its order is ``L = \\lceil w \\rceil - 1``, the number of entries on each side of a given entry that share data with it. It reads no dependence past the overlap. A series that is persistent past the overlap gives a t-statistic that is too large, and the untapered sum can be negative on a short series. [`WildBootstrapMeanTest`](@ref) is the default because of this.

# The divisor removes the bias of the centring

The sum ``Q`` of the centred products reads the deviations from the sample mean, not from the true mean, so its expectation is less than ``n`` times the long-run variance. Under the null hypothesis the series is a moving average of order ``L``. The weights of the untapered sum are one on the band of width ``2L + 1``, and to first order in ``L / n`` the expectation is ``(n - 2L - 1)`` times the long-run variance. The test divides by that count. At ``L = 0`` the count is ``n - 1``, and the test is the classical t-test, whose law under independent Gaussian entries is the Student law with ``n - 1`` degrees of freedom. The test reads its p-value and its interval off the Student law with ``n - 2L - 1`` degrees of freedom at every order. At ``L > 0`` that law is an approximation. A divisor of ``n`` or ``n - 1`` with the normal law rejects a no-skill moving average of order four in 60 entries about 10.6% of the time at the 5% level, and this rule about 7.9%.

# Mathematical definition

```math
\\begin{align}
\\gamma_{j} &= \\frac{1}{n - 2L - 1} \\sum_{t,\\, t + j \\in \\mathcal{T}} \\left( c_{t} - \\bar{c} \\right) \\left( c_{t + j} - \\bar{c} \\right)\\,, \\\\
\\sigma^{2} &= \\gamma_{0} + 2 \\sum_{j = 1}^{L} \\gamma_{j}\\,, \\\\
t &= \\frac{\\bar{c}}{\\sigma} \\sqrt{n}\\,, \\\\
p &= 2 \\left( 1 - F_{\\nu}(|t|) \\right)\\,, \\\\
\\bar{c} &\\pm F_{\\nu}^{-1}(1 - q / 2) \\frac{\\sigma}{\\sqrt{n}}\\,, \\quad \\nu = n - 2L - 1\\,.
\\end{align}
```

Where:

  - ``c_{t}``: Entry ``t`` of the series.
  - ``\\mathcal{T}``: The entries at which it is finite.
  - ``n``: The number of entries in ``\\mathcal{T}``.
  - ``\\bar{c}``: The mean over ``\\mathcal{T}``.
  - ``\\gamma_{j}``: The autocovariance at lag ``j``.
  - ``L``: The order, ``\\lceil w \\rceil - 1``, with ``w`` the `window`.
  - ``\\sigma``: The long-run standard deviation.
  - ``F_{\\nu}``: The distribution function of the Student law with ``\\nu`` degrees of freedom.
  - ``t``: The t-statistic.
  - ``p``: The two-sided p-value.
  - ``q``: `q`, so the last line is the confidence interval of the mean at the level ``1 - q``.

A long-run variance that is not positive, or a series with ``n \\leq 2L + 1``, gives a `NaN` t-statistic, p-value and interval. At a `window` of at most one the order is zero, and the t-statistic is ``\\mathrm{IR} \\sqrt{n}``. The rules of `min_obs` and `min_periods` are those of [`WildBootstrapMeanTest`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OverlapMeanTest(;
        q::Real = 0.05,
        min_obs::Option{<:Integer} = nothing,
        min_periods::Option{<:Real} = nothing,
    ) -> OverlapMeanTest

Keywords correspond to the struct's fields.

## Validation

  - `0 < q < 1`. Raises a `DomainError`.
  - `min_obs >= 2` and `min_periods > 0`, when given. Raise a `DomainError`.

# Examples

```jldoctest
julia> PortfolioOptimisers.mean_test(OverlapMeanTest(), [0.1, -0.2, 0.3, 0.2, NaN, 0.1], 1,
                                     nothing)
(t_stat = 1.1952286093343938, p_value = 0.29801481173121036, mean_ci_lower = -0.13229406353851897, mean_ci_upper = 0.332294063538519)
```

# Related

  - [`AbstractMeanTestEstimator`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
  - [`forecast_ic_lags`](@ref)
  - [`mean_test`](@ref)
"""
@concrete struct OverlapMeanTest <: AbstractMeanTestEstimator
    """
    Significance level of the test, and one less the confidence level of the interval of the mean, in `(0, 1)`.
    """
    q
    """
    Least number of finite entries a series needs before the test answers, or `nothing` for no rule.
    """
    min_obs
    """
    Least number of windows that the span of a series needs before the test answers, the span over `window`, or `nothing` for no rule. For a forward window it is the number of holding periods that the history covers.
    """
    min_periods
    function OverlapMeanTest(q::Real, min_obs::Option{<:Integer},
                             min_periods::Option{<:Real})
        assert_mean_test_rules(q, min_obs, min_periods)
        return new{typeof(q), typeof(min_obs), typeof(min_periods)}(q, min_obs, min_periods)
    end
end
function OverlapMeanTest(; q::Real = 0.05, min_obs::Option{<:Integer} = nothing,
                         min_periods::Option{<:Real} = nothing)::OverlapMeanTest
    return OverlapMeanTest(q, min_obs, min_periods)
end
"""
$(DocStringExtensions.TYPEDEF)

Tests the mean of a series against zero with a fixed-bandwidth long-run variance and a Gaussian wild bootstrap of its null law.

The bandwidth is the fraction `b` of the span of the series, so the long-run variance reads the dependence at every lag up to it, past the overlap of the windows as well. The null law of such a t-statistic is not the normal law, and a change of the volatility over time moves it. The test therefore reads the p-value off `n_sim` draws of the statistic under the null hypothesis, as the source states. Each draw multiplies the centred series by independent standard normal draws, and computes the t-statistic of the product with the same kernel and the same bandwidth.

The t-statistic is not on the scale of the normal law. Read the p-value or the interval to judge the evidence. The p-value is ``(1 + K) / (M + 1)``, so it is never zero, and its resolution is ``1 / (M + 1)``. The interval is the symmetric percentile-t interval of the bootstrap. It excludes zero exactly when the p-value is at most `q`.

# Mathematical definition

```math
\\begin{align}
Q &= \\sum_{i, j \\in \\mathcal{T}} k\\left(\\frac{|i - j|}{B}\\right) \\left( c_{i} - \\bar{c} \\right) \\left( c_{j} - \\bar{c} \\right)\\,, \\\\
s &= \\sqrt{\\frac{Q}{n (n - \\delta)}}\\,, \\quad t = \\frac{\\bar{c}}{s}\\,, \\\\
c_{t}^{*} &= r_{t} \\left( c_{t} - \\bar{c} \\right)\\,, \\quad r_{t} \\sim \\mathcal{N}(0, 1)\\,, \\\\
p &= \\frac{1 + \\#\\left\\{ m : |t_{m}^{*}| \\geq |t| \\right\\}}{M + 1}\\,, \\\\
\\bar{c} &\\pm |t^{*}|_{(k)}\\, s\\,, \\quad k = M + 1 - \\max\\left\\{ c : \\frac{c}{M + 1} \\leq q \\right\\}\\,.
\\end{align}
```

Where:

  - ``c_{t}``: Entry ``t`` of the series.
  - ``\\mathcal{T}``: The entries at which it is finite.
  - ``n``: The number of entries in ``\\mathcal{T}``.
  - ``\\bar{c}``: The mean over ``\\mathcal{T}``.
  - ``k``: The kernel, `kernel`, in the first line.
  - ``B``: The bandwidth, ``b T``, where ``T`` is the span that `span` sets.
  - ``Q``: The weighted sum of [`long_run_quadratic_form`](@ref).
  - ``s``: The long-run standard error of the mean. Every autocovariance divides by ``n - \\delta``, so the long-run variance is ``Q / (n - \\delta)``.
  - ``\\delta``: `ddof`. The default `0` is the divisor ``n`` of the source. The divisor scales the observed statistic and every draw by one factor, so it moves the t-statistic alone, and no p-value and no interval. A series with ``n \\leq \\delta`` has no statistic.
  - ``t``: The t-statistic.
  - ``r_{t}``: The multiplier of entry ``t``.
  - ``t_{m}^{*}``: The t-statistic of draw ``m``, with ``c^{*}`` in place of ``c``.
  - ``M``: `n_sim`.
  - ``p``: The two-sided p-value.
  - ``|t^{*}|_{(k)}``: The ``k``-th smallest of the ``|t_{m}^{*}|``, with ``k`` the order in the last line. [`bootstrap_exceedance_cap`](@ref) gives the count ``c``.
  - ``q``: `q`, so the last line is the confidence interval of the mean at the level ``1 - q``.

A t-statistic of a draw that is not finite counts as one at least as large as ``t``. A series with fewer than two finite entries, or with a long-run variance that is not positive, gives a `NaN` t-statistic, p-value and interval. So does a series with fewer than `min_obs` finite entries, and a series whose span holds fewer than `min_periods` windows, the span over `window`. That is the only use of `window`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    WildBootstrapMeanTest(;
        kernel::AbstractLongRunKernel = BartlettKernel(),
        b::Real = 0.4,
        span::AbstractMeanTestSpan = FiniteEntrySpan(),
        ddof::Integer = 0,
        n_sim::Integer = 1_999,
        q::Real = 0.05,
        min_obs::Option{<:Integer} = nothing,
        min_periods::Option{<:Real} = nothing,
        rng::Random.AbstractRNG = Random.default_rng(),
        seed::Option{<:Integer} = nothing,
    ) -> WildBootstrapMeanTest

Keywords correspond to the struct's fields.

The defaults are those of the source. Its simulations find that a bandwidth of a mid size, such as `b = 0.4`, holds the size of the test, and that the Bartlett kernel has more power than the quadratic spectral kernel. A smaller `b` gives more power and a larger size distortion under strong dependence.

## Validation

  - `0 < b <= 1`. Raises a `DomainError`.
  - `ddof >= 0`. Raises a `DomainError`.
  - `n_sim > 0`, and `n_sim` at most the `max_n_sim` of [`RESOURCE_LIMITS`](@ref). Raises a `DomainError`.
  - `0 < q < 1`, and `1 / (n_sim + 1) <= q`, so the interval has an order statistic. Raises a `DomainError`.
  - `min_obs >= 2` and `min_periods > 0`, when given. Raise a `DomainError`.

# Examples

```jldoctest
julia> mt = WildBootstrapMeanTest(; seed = 42);

julia> res = PortfolioOptimisers.mean_test(mt, [0.1, -0.2, 0.3, 0.2, NaN, 0.1], 1,
                                           PortfolioOptimisers.mean_test_rng(mt));

julia> res.t_stat
1.7320508075688772

julia> 0 < res.p_value <= 1
true

julia> res.mean_ci_lower < 0.1 < res.mean_ci_upper
true
```

# Related

  - [`AbstractMeanTestEstimator`](@ref)
  - [`OverlapMeanTest`](@ref)
  - [`BartlettKernel`](@ref)
  - [`QuadraticSpectralKernel`](@ref)
  - [`FiniteEntrySpan`](@ref)
  - [`FullGridSpan`](@ref)
  - [`mean_test`](@ref)
  - [`mean_test_rng`](@ref)

# References

  - $(ref_dict[:demetrescu2026robust]) Section 3.1 and Section 4.
"""
@concrete struct WildBootstrapMeanTest <: AbstractMeanTestEstimator
    """
    Kernel that weights the autocovariances.
    """
    kernel
    """
    Bandwidth as a fraction of the span of the series, in `(0, 1]`.
    """
    b
    """
    Rule that sets the span of the series, which the bandwidth and `min_periods` read.
    """
    span
    """
    Number that the divisor of the long-run variance subtracts from the count of finite entries, `0` for the divisor ``n`` of the source.
    """
    ddof
    """
    $(field_dict[:n_sim])
    """
    n_sim
    """
    Significance level of the test, and one less the confidence level of the interval of the mean, in `(0, 1)`.
    """
    q
    """
    Least number of finite entries a series needs before the test answers, or `nothing` for no rule.
    """
    min_obs
    """
    Least number of windows that the span of a series needs before the test answers, the span over `window`, or `nothing` for no rule. For a forward window it is the number of holding periods that the history covers.
    """
    min_periods
    """
    $(field_dict[:rng])
    """
    rng
    """
    $(field_dict[:seed])
    """
    seed
    function WildBootstrapMeanTest(kernel::AbstractLongRunKernel, b::Real,
                                   span::AbstractMeanTestSpan, ddof::Integer,
                                   n_sim::Integer, q::Real, min_obs::Option{<:Integer},
                                   min_periods::Option{<:Real}, rng::Random.AbstractRNG,
                                   seed::Option{<:Integer})
        @argcheck(zero(b) < b <= one(b),
                  DomainError(b, "0 < b <= 1 must hold. Got\nb => $b"))
        @argcheck(ddof >= zero(ddof), DomainError(ddof, "ddof must be >= 0"))
        @argcheck(zero(n_sim) < n_sim, DomainError(n_sim, "n_sim must be > 0"))
        assert_resource_cap(n_sim, RESOURCE_LIMITS[].max_n_sim, :n_sim, :max_n_sim)
        assert_mean_test_rules(q, min_obs, min_periods)
        @argcheck(bootstrap_exceedance_cap(q, n_sim) >= 1,
                  DomainError(q,
                              "1 / (n_sim + 1) <= q must hold, so the interval has an order statistic of the draws. Got\nq => $q\nn_sim => $n_sim"))
        return new{typeof(kernel), typeof(b), typeof(span), typeof(ddof), typeof(n_sim),
                   typeof(q), typeof(min_obs), typeof(min_periods), typeof(rng),
                   typeof(seed)}(kernel, b, span, ddof, n_sim, q, min_obs, min_periods, rng,
                                 seed)
    end
end
function WildBootstrapMeanTest(; kernel::AbstractLongRunKernel = BartlettKernel(),
                               b::Real = 0.4,
                               span::AbstractMeanTestSpan = FiniteEntrySpan(),
                               ddof::Integer = 0, n_sim::Integer = 1_999, q::Real = 0.05,
                               min_obs::Option{<:Integer} = nothing,
                               min_periods::Option{<:Real} = nothing,
                               rng::Random.AbstractRNG = Random.default_rng(),
                               seed::Option{<:Integer} = nothing)::WildBootstrapMeanTest
    return WildBootstrapMeanTest(kernel, b, span, ddof, n_sim, q, min_obs, min_periods, rng,
                                 seed)
end
"""
    assert_mean_test_rules(q::Real, min_obs::Option{<:Integer}, min_periods::Option{<:Real})

Refuse a significance level or a minimum-data rule that no series can meet.

# Arguments

  - `q`: Significance level.
  - `min_obs`: Least number of finite entries, or `nothing`.
  - `min_periods`: Least number of windows in the span, or `nothing`.

# Validation

  - `0 < q < 1`. Raises a `DomainError`.
  - `min_obs >= 2` and `min_periods > 0`, when given. Raise a `DomainError`.

# Returns

  - `nothing`.

# Related

  - [`OverlapMeanTest`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
"""
function assert_mean_test_rules(q::Real, min_obs::Option{<:Integer},
                                min_periods::Option{<:Real})::Nothing
    assert_unit_interval(q, :q)
    if !isnothing(min_obs)
        @argcheck(min_obs >= 2, DomainError(min_obs, "min_obs must be >= 2"))
    end
    if !isnothing(min_periods)
        @argcheck(min_periods > zero(min_periods),
                  DomainError(min_periods, "min_periods must be > 0"))
    end
    return nothing
end
"""
    mean_test_rng(mt::AbstractMeanTestEstimator) -> Nothing
    mean_test_rng(mt::WildBootstrapMeanTest) -> Random.AbstractRNG

Return the random number generator that a Mean Test draws from.

A summary of many series resolves the generator once and passes it to each call of [`mean_test`](@ref), so a seeded test of many series is reproducible, and the series take different draws. A test that draws no random number answers `nothing`.

# Arguments

  - `mt`: A Mean Test.

# Returns

  - `rng::Option{<:Random.AbstractRNG}`: The generator that [`resolve_rng`](@ref) resolves from the `rng` and the `seed` of the test, or `nothing`.

# Related

  - [`AbstractMeanTestEstimator`](@ref)
  - [`mean_test`](@ref)
  - [`resolve_rng`](@ref)
"""
function mean_test_rng(::AbstractMeanTestEstimator)
    return nothing
end
function mean_test_rng(mt::WildBootstrapMeanTest)
    return resolve_rng(mt.rng, mt.seed)
end
"""
    mean_test(mt::OverlapMeanTest, c::VecNum, window::Real, rng) -> NamedTuple
    mean_test(mt::WildBootstrapMeanTest, c::VecNum, window::Real,
              rng::Random.AbstractRNG) -> NamedTuple

Return the t-statistic, the two-sided p-value and the confidence interval of the mean of a series, under the null hypothesis that the mean is zero.

The test reads the entries of `c` at which it is finite. A `NaN` entry is in no pair, and the lag between two entries is their distance in the series, not in its finite subsequence. The docstring of each test states its rule.

# Arguments

  - `mt`: A Mean Test.
  - `c`: The series.
  - `window`: Number of entries that the data of one entry spans, `horizon / step` for a forward window, from [`forecast_ic_window`](@ref). [`OverlapMeanTest`](@ref) reads the order of its long-run variance off it. Both tests read it in the rule of `min_periods`.
  - `rng`: The generator of [`mean_test_rng`](@ref). [`OverlapMeanTest`](@ref) does not read it.

# Validation

  - `window > 0`. Raises a `DomainError`.

# Returns

  - `res::NamedTuple`: `(; t_stat, p_value, mean_ci_lower, mean_ci_upper)`.

# Related

  - [`AbstractMeanTestEstimator`](@ref)
  - [`mean_test_rng`](@ref)
  - [`exposure_ic_factor_summary`](@ref)
"""
function mean_test(mt::OverlapMeanTest, c::VecNum, window::Real, ::Any)
    @argcheck(window > zero(window), DomainError(window, "window must be > 0"))
    m, n, d, _, T = mean_test_centred(c, FiniteEntrySpan())
    Ts = typeof(sqrt(one(m)))
    nan = Ts(NaN)
    if !mean_test_answers(mt, m, n, T, window)
        return (; t_stat = nan, p_value = nan, mean_ci_lower = nan, mean_ci_upper = nan)
    end
    P = length(d)
    lags = ceil(Int, window) - 1
    Q = zero(eltype(d))
    for t in 1:P
        Q += d[t] * d[t]
    end
    for j in 1:lags, t in 1:(P - j)
        Q += 2 * d[t] * d[t + j]
    end
    nu = n - 2 * lags - 1
    t = nu > 0 ? studentised_mean(m, Q, n, nu) : nan
    if !isfinite(t)
        return (; t_stat = t, p_value = nan, mean_ci_lower = nan, mean_ci_upper = nan)
    end
    D = Distributions.TDist(nu)
    h = Distributions.quantile(D, 1 - mt.q / 2) * m / t
    return (; t_stat = t, p_value = 2 * Distributions.ccdf(D, abs(t)),
            mean_ci_lower = m - h, mean_ci_upper = m + h)
end
function mean_test(mt::WildBootstrapMeanTest, c::VecNum, window::Real,
                   rng::Random.AbstractRNG)
    @argcheck(window > zero(window), DomainError(window, "window must be > 0"))
    m, n, d, f, T = mean_test_centred(c, mt.span)
    Ts = typeof(sqrt(one(m)))
    nan = Ts(NaN)
    if !mean_test_answers(mt, m, n, T, window)
        return (; t_stat = nan, p_value = nan, mean_ci_lower = nan, mean_ci_upper = nan)
    end
    B = mt.b * T
    buf = similar(d, length(d) + 1)
    nd = n - mt.ddof
    t = if nd > 0
        studentised_mean(m, long_run_quadratic_form(mt.kernel, d, B, buf), n, nd)
    else
        nan
    end
    if !isfinite(t)
        return (; t_stat = t, p_value = nan, mean_ci_lower = nan, mean_ci_upper = nan)
    end
    ats = wild_bootstrap_statistics(mt.kernel, mt.n_sim, d, f, n, nd, B, buf, rng)
    at = abs(t)
    K = count(x -> !(x < at), ats)
    k = mt.n_sim + 1 - bootstrap_exceedance_cap(mt.q, mt.n_sim)
    h = partialsort!(ats, k) * m / t
    return (; t_stat = t, p_value = (1 + K) / (mt.n_sim + 1), mean_ci_lower = m - h,
            mean_ci_upper = m + h)
end
"""
    bootstrap_exceedance_cap(q::Real, M::Integer) -> Int

Return the largest count ``c`` with ``c / (M + 1) \\leq q``, by the same division that gives the p-value of a wild bootstrap.

The p-value is ``(1 + K) / (M + 1)``, so the test rejects at the level `q` exactly when ``1 + K`` is at most this count. The confidence interval takes its order from the count, so the interval excludes zero exactly when the p-value is at most `q`. The function reads the count off the division and not off the product ``q (M + 1)``, because the product can round below an integer, as `0.29 * 100` does. It starts one below the rounded product, which is never above the count, and counts up.

# Arguments

  - `q`: Significance level.
  - `M`: Number of draws.

# Returns

  - `c::Int`: The count, in `1:M` when `q * (M + 1) >= 1` and `q < 1`.

# Related

  - [`WildBootstrapMeanTest`](@ref)
"""
function bootstrap_exceedance_cap(q::Real, M::Integer)
    c = max(floor(Int, q * (M + 1)) - 1, 0)
    while (c + 1) / (M + 1) <= q
        c += 1
    end
    return c
end
"""
    resolve_mean_test_window(lags::Integer, window::Real) -> Real

Return the window that a summary gives its Mean Test, from the two keywords that can state it.

A summary takes `lags`, the number of entries on each side of a given entry that share data with it, and `window`, the number of entries that the data of one entry spans. `window` defaults to `lags + 1`, so a caller who states `lags` alone states the window of a stride that divides the horizon. A caller who states the window of a forward window, `horizon / step`, states the exact count of holding periods that the history rule of a Mean Test reads.

# Arguments

  - `lags`: The order of the overlap.
  - `window`: The window.

# Validation

  - `lags >= 0` and `window > 0`. Raise a `DomainError`.
  - `lags` is `0`, or `lags == ceil(Int, window) - 1`. Raises a [`ConflictingArgumentError`](@ref).

# Returns

  - `w::Real`: `window`.

# Related

  - [`mean_test`](@ref)
  - [`forecast_ic_window`](@ref)
  - [`forecast_ic_lags`](@ref)
"""
function resolve_mean_test_window(lags::Integer, window::Real)
    @argcheck(lags >= zero(lags), DomainError(lags, "lags must be >= 0"))
    @argcheck(window > zero(window), DomainError(window, "window must be > 0"))
    @argcheck(iszero(lags) || ceil(Int, window) - 1 == lags,
              ConflictingArgumentError("lags = $lags and window = $window state two different overlaps. A window of $window entries overlaps $(ceil(Int, window) - 1) entries on each side. Give one of the two, or two that agree."))
    return window
end
"""
    mean_test_answers(mt::AbstractMeanTestEstimator, m::Real, n::Integer, T::Integer,
                      window::Real) -> Bool

Return whether a series carries enough evidence for a Mean Test to answer.

# Arguments

  - `mt`: The test. The function reads its `min_obs` and its `min_periods`.
  - `m`: The mean of the finite entries.
  - `n`: The number of finite entries.
  - `T`: The span of the series.
  - `window`: Number of entries that the data of one entry spans.

# Returns

  - `ok::Bool`: `false` when the mean is not finite, when `n < 2`, when `n < mt.min_obs`, or when `T / window < mt.min_periods`.

# Related

  - [`mean_test`](@ref)
"""
function mean_test_answers(mt::AbstractMeanTestEstimator, m::Real, n::Integer, T::Integer,
                           window::Real)
    if n < 2 || !isfinite(m)
        return false
    end
    if !isnothing(mt.min_obs) && n < mt.min_obs
        return false
    end
    if !isnothing(mt.min_periods) && T / window < mt.min_periods
        return false
    end
    return true
end
"""
    wild_bootstrap_statistics(kernel::BartlettKernel, M::Integer, d::VecNum,
                              f::AbstractVector{Bool}, n::Integer, nd::Integer, B::Real,
                              buf::VecNum,
                              rng::Random.AbstractRNG) -> Vector{<:Real}
    wild_bootstrap_statistics(kernel::AbstractLongRunKernel, M::Integer, d::VecNum,
                              f::AbstractVector{Bool}, n::Integer, nd::Integer, B::Real,
                              buf::VecNum,
                              rng::Random.AbstractRNG) -> Vector{<:Real}

Return the absolute t-statistics of `M` wild bootstrap draws of a centred series.

Each draw multiplies each finite entry of the centred series by an independent standard normal draw, centres the product over the finite entries again, and computes its t-statistic with the kernel and the bandwidth of the observed one. A missing entry stays zero, and it takes no draw. Both methods take the draws in the same order, one per finite entry and draw by draw, so they answer the same statistics from the same generator.

The Bartlett method reads each draw with [`long_run_quadratic_form`](@ref). The generic method writes the weights of every pair into the matrix ``\\mathbf{A} = \\mathbf{D} \\mathbf{W} \\mathbf{D}`` once, and reads a block of draws with one matrix product. For a draw ``\\boldsymbol{r}`` the centred product is ``\\boldsymbol{u} = \\mathbf{D} \\boldsymbol{r} - \\bar{u} \\boldsymbol{f}``, so

```math
\\boldsymbol{u}^{\\intercal} \\mathbf{W} \\boldsymbol{u} = \\boldsymbol{r}^{\\intercal} \\mathbf{A} \\boldsymbol{r} - 2 \\bar{u} \\left( \\mathbf{D} \\mathbf{W} \\boldsymbol{f} \\right)^{\\intercal} \\boldsymbol{r} + \\bar{u}^{2} \\boldsymbol{f}^{\\intercal} \\mathbf{W} \\boldsymbol{f}\\,.
```

Where:

  - ``\\mathbf{D}``: The diagonal matrix of the centred series.
  - ``\\mathbf{W}``: The matrix of the kernel weights of every pair.
  - ``\\boldsymbol{f}``: The mask of the finite entries.
  - ``\\bar{u}``: The mean of the product, ``\\boldsymbol{d}^{\\intercal} \\boldsymbol{r} / n``.

# Arguments

  - `kernel`: The kernel.
  - `M`: The number of draws.
  - `d`: The centred series, with a zero at each missing observation. It is not changed.
  - `f`: Whether each entry of `d` is a finite observation.
  - `n`: The number of finite entries.
  - `nd`: The divisor of the long-run variance, `n - ddof`.
  - `B`: The bandwidth, in entries.
  - `buf`: Work buffer of length `length(d) + 1`.
  - `rng`: The generator.

# Returns

  - `ats::Vector{<:Real}`: The `M` absolute t-statistics. A statistic that is not finite is `Inf`.

# Related

  - [`WildBootstrapMeanTest`](@ref)
  - [`long_run_quadratic_form`](@ref)
"""
function wild_bootstrap_statistics(kernel::BartlettKernel, M::Integer, d::VecNum,
                                   f::AbstractVector{Bool}, n::Integer, nd::Integer,
                                   B::Real, buf::VecNum, rng::Random.AbstractRNG)
    P = length(d)
    u = similar(d)
    ats = Vector{typeof(sqrt(one(eltype(d))))}(undef, M)
    for k in 1:M
        s = zero(eltype(d))
        for i in 1:P
            u[i] = f[i] ? randn(rng) * d[i] : d[i]
            s += u[i]
        end
        ms = s / n
        for i in 1:P
            if f[i]
                u[i] -= ms
            end
        end
        ats[k] = finite_abs(studentised_mean(ms, long_run_quadratic_form(kernel, u, B, buf),
                                             n, nd))
    end
    return ats
end
function wild_bootstrap_statistics(kernel::AbstractLongRunKernel, M::Integer, d::VecNum,
                                   f::AbstractVector{Bool}, n::Integer, nd::Integer,
                                   B::Real, ::VecNum, rng::Random.AbstractRNG)
    P = length(d)
    w = [iszero(j) ? one(B / B) : long_run_kernel_weight(kernel, j / B) for j in 0:(P - 1)]
    # The weights are written in a loop rather than by a two-dimensional comprehension, whose
    # iterator type JET cannot pin.
    W = Matrix{eltype(w)}(undef, P, P)
    for j in 1:P, i in 1:P
        W[i, j] = w[abs(i - j) + 1]
    end
    Wf = W * f
    A = d .* W .* transpose(d)
    g = d .* Wf
    cf = LinearAlgebra.dot(f, Wf)
    nb = min(M, P)
    R = zeros(eltype(A), P, nb)
    Y = similar(R)
    ats = Vector{typeof(sqrt(one(eltype(A))))}(undef, M)
    k = 0
    while k < M
        b = min(nb, M - k)
        for j in 1:b, i in 1:P
            R[i, j] = f[i] ? randn(rng) : zero(eltype(R))
        end
        LinearAlgebra.mul!(Y, A, R)
        for j in 1:b
            r = view(R, :, j)
            ms = LinearAlgebra.dot(d, r) / n
            Q = LinearAlgebra.dot(r, view(Y, :, j)) - 2 * ms * LinearAlgebra.dot(g, r) +
                ms^2 * cf
            ats[k + j] = finite_abs(studentised_mean(ms, Q, n, nd))
        end
        k += b
    end
    return ats
end
"""
    finite_abs(x::Real) -> Real

Return the absolute value of a statistic, with a statistic that is not finite as `Inf`.

# Arguments

  - `x`: The statistic.

# Returns

  - `a::Real`: `abs(x)`, or `Inf` when `x` is not finite.

# Related

  - [`wild_bootstrap_statistics`](@ref)
"""
function finite_abs(x::Real)
    return isfinite(x) ? abs(x) : oftype(abs(x), Inf)
end
"""
    mean_test_centred(c::VecNum, span::AbstractMeanTestSpan) -> Tuple

Return the mean of the finite entries of a series, their count, the series centred on the mean from its first finite entry to its last, the mask of the finite entries of that span, and the span that `span` sets.

A missing entry is zero in the centred series, so it adds nothing to a sum of products. The centred series starts and stops at a finite entry. The span rule sets the span that the bandwidth and the history rule read.

# Arguments

  - `c`: The series.
  - `span`: The span rule.

# Returns

  - `m::Real`: The mean, `NaN` when no entry is finite. When every finite entry has one value, the mean is that value, so every entry of `d` is zero and the long-run variance is not positive.
  - `n::Int`: The number of finite entries.
  - `d::Vector{<:Real}`: The centred span, empty when no entry is finite.
  - `f::BitVector`: Whether each entry of `d` is a finite observation.
  - `T::Int`: The span of [`mean_test_span_length`](@ref), `0` when no entry is finite.

# Related

  - [`mean_test`](@ref)
"""
function mean_test_centred(c::VecNum, span::AbstractMeanTestSpan)
    Tm = float_if_integer(real(eltype(c)))
    n, s, i1, i2 = finite_sum_bounds(c, Tm)
    m = n > 0 ? s / n : Tm(NaN)
    # `s / n` misses the value of a constant series by round-off, which would centre it to a
    # long-run variance that is positive. The value is its mean.
    if n > 0 && all(v -> !isfinite(v) || v == c[i1], view(c, i1:i2))
        m = oftype(m, c[i1])
    end
    P = iszero(n) ? 0 : i2 - i1 + 1
    d = Vector{typeof(m)}(undef, P)
    f = falses(P)
    for i in 1:P
        v = c[i1 + i - 1]
        f[i] = isfinite(v)
        d[i] = f[i] ? Tm(v) - m : zero(m)
    end
    T = iszero(n) ? 0 : mean_test_span_length(span, length(c), i1, i2)
    return m, n, d, f, T
end
"""
    finite_sum_bounds(c::VecNum, Tm::Type) -> Tuple

Return the number of finite entries of a series, their sum in the type `Tm`, and the positions of the first and the last of them.

# Arguments

  - `c`: The series.
  - `Tm`: The type the sum is read in, which [`mean_test_centred`](@ref) derives from `c`.

# Returns

  - `n::Int`: The number of finite entries.
  - `s`: Their sum.
  - `i1::Int`, `i2::Int`: The positions of the first and the last finite entry, `0` when no entry is finite.

# Related

  - [`mean_test_centred`](@ref)
"""
function finite_sum_bounds(c::VecNum, Tm::Type)
    n = 0
    s = zero(Tm)
    i1 = 0
    i2 = 0
    for t in eachindex(c)
        v = c[t]
        if isfinite(v)
            n += 1
            s += Tm(v)
            i1 = iszero(i1) ? t : i1
            i2 = t
        end
    end
    return n, s, i1, i2
end
"""
    studentised_mean(m::Real, Q::Real, n::Integer, nd::Integer) -> Real

Return the mean over its long-run standard error.

The long-run variance is ``Q / n_{d}``. [`WildBootstrapMeanTest`](@ref) divides by ``n``, as its source does. Its divisor scales the t-statistic of the observed series and of every bootstrap draw by one factor, so it moves no p-value and no interval. [`OverlapMeanTest`](@ref) divides by ``n - 2L - 1``, which removes the bias of the centring under its null hypothesis, and its p-value reads the divisor.

# Arguments

  - `m`: The mean.
  - `Q`: The weighted sum of [`long_run_quadratic_form`](@ref).
  - `n`: The number of finite entries.
  - `nd`: The divisor of the long-run variance.

# Returns

  - `t::Real`: ``m / \\sqrt{Q / n_{d}} \\sqrt{n}``, or `NaN` when `Q` is not positive.

# Related

  - [`mean_test`](@ref)
"""
function studentised_mean(m::Real, Q::Real, n::Integer, nd::Integer)
    Ts = typeof(sqrt(one(Q)))
    lrsd = Q > zero(Q) ? sqrt(Q / nd) : Ts(NaN)
    return isfinite(m) && isfinite(lrsd) ? m / lrsd * sqrt(Ts(n)) : Ts(NaN)
end

export OverlapMeanTest, WildBootstrapMeanTest, BartlettKernel, QuadraticSpectralKernel,
       FiniteEntrySpan, FullGridSpan
public AbstractMeanTestEstimator, AbstractLongRunKernel, AbstractMeanTestSpan, mean_test,
       mean_test_rng, long_run_kernel_weight, mean_test_span_length
