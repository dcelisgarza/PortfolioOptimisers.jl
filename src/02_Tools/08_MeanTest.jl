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

A kernel that implements the weight alone takes the generic path. Its long-run variance reads every pair of the series. Under [`ObservationDraws`](@ref) its wild bootstrap does one matrix product for each block of draws, so the cost of a draw is quadratic in the length of the series. Under [`EigenSumDraws`](@ref) it factors the matrix of the weights once, and the cost of a draw is the rank of the factor. [`BartlettKernel`](@ref) has a form that is linear in the length, and [`QuadraticSpectralKernel`](@ref) one whose draws do not grow with it.

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

The kernel has no finite support, so the long-run variance reads every lag of the series. Its estimate is never negative. The kernel is the cosine transform of a weight on a bounded band of frequencies, the second line below. So [`QuadratureSumDraws`](@ref), its default draws, reads it through a few features whose number does not grow with the length of the series.

# Mathematical definition

```math
\\begin{align}
k(x) &= \\frac{25}{12 \\pi^{2} x^{2}} \\left( \\frac{\\sin(z)}{z} - \\cos(z) \\right) = \\frac{3 \\left( \\sin(z) - z \\cos(z) \\right)}{z^{3}}\\,, \\quad z = \\frac{6 \\pi x}{5}\\,, \\\\
k(x) &= \\frac{3}{4} \\int_{-1}^{1} \\left( 1 - u^{2} \\right) \\cos(z u)\\, \\mathrm{d}u\\,.
\\end{align}
```

Where:

  - ``x``: The lag over the bandwidth.
  - ``z``: The angle of the lag.
  - ``u``: The frequency over the edge of the band.

The closed form cancels near zero, so the weight reads its Taylor series in ``z^{2}`` for ``|z| < 1``, which is exact to round-off there.

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
    z = 6 * oftype(x / one(x), pi) * x / 5
    if abs(z) < one(z)
        # The closed form cancels near zero: at `z = 2e-3` it loses ten digits. Its Taylor
        # series in `z^2`, `3 (sin z - z cos z) / z^3`, is exact to round-off for `|z| < 1`.
        z2 = z * z
        w = zero(z)
        for c in (2_252_687_044_608_000, -7_410_154_752_000, 31_135_104_000, -172_972_800,
                  1_330_560, -15_120, 280, -10, 1)
            w = w * z2 + one(z) / c
        end
        return w
    end
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
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the algorithms that draw the null law of the statistic of a [`WildBootstrapMeanTest`](@ref).

A draw multiplies each finite entry of the centred series by an independent standard normal draw, centres the product again, and computes its t-statistic with the kernel and the bandwidth of the observed one. The statistic of a draw reads the multipliers through two Gaussian quantities alone: the sum of the product, and its quadratic form under the matrix of the kernel weights. Every member draws the same law. The members differ in the cost of a draw, and in how they read the generator, so two members answer the same p-value in law and not draw for draw.

# Interfaces

In order to implement a new draw algorithm that works seamlessly with the library, subtype `AbstractBootstrapDraws` and implement the following method:

## Draw method

  - `wild_bootstrap_statistics(kernel::AbstractLongRunKernel, alg::AbstractBootstrapDraws, M::Integer, d::VecNum, f::AbstractVector{Bool}, n::Integer, nd::Integer, B::Real, buf::VecNum, rng::Random.AbstractRNG) -> Tuple`: The weighted sum of the observed series and the absolute t-statistics of `M` draws. The generic method reads the exact sum, and it draws through `draw_statistics`, which an algorithm that keeps the exact sum implements in place of this method.

### Arguments

  - `kernel`: The kernel of the test.
  - `alg`: Concrete subtype instance.
  - `M`: The number of draws.
  - `d`: The centred series, with a zero at each missing observation. It must not be changed.
  - `f`: Whether each entry of `d` is a finite observation.
  - `n`: The number of finite entries.
  - `nd`: The divisor of the long-run variance, `n - ddof`.
  - `B`: The bandwidth, in entries.
  - `buf`: Work buffer of length `length(d) + 1`.
  - `rng`: The generator.

### Returns

  - `Q::Real`: The weighted sum of the observed series, which the observed t-statistic reads.
  - `ats::Vector{<:Real}`: The `M` absolute t-statistics, empty when `Q` is not positive. A statistic that is not finite is `Inf`.

## Validation method

  - `assert_bootstrap_draws(kernel::AbstractLongRunKernel, alg::AbstractBootstrapDraws) -> Nothing`: Refuses a kernel that the algorithm cannot read, in the constructor of [`WildBootstrapMeanTest`](@ref). The default method accepts every kernel, so an algorithm that reads every kernel does not implement it.

# Related

  - [`ObservationDraws`](@ref)
  - [`QuadratureSumDraws`](@ref)
  - [`EigenSumDraws`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
  - [`wild_bootstrap_statistics`](@ref)
  - [`assert_bootstrap_draws`](@ref)
"""
abstract type AbstractBootstrapDraws <: AbstractAlgorithm end
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

Abstract supertype for the rules that set the divisor of the long-run variance of an [`OverlapMeanTest`](@ref), and the degrees of freedom of the Student law that its p-value and its interval read.

The untapered sum ``Q`` of the centred products reads the deviations from the sample mean, so its expectation is not ``n`` times the long-run variance. The law sets the divisor ``n_{d}`` of ``Q`` and the degrees of freedom ``\\nu``. At a `window` of at most one every law gives ``n_{d} = \\nu = n - 1``, the classical t-test.

# Interfaces

In order to implement a new law that works seamlessly with the library, subtype `AbstractOverlapStudentLaw` and implement the following method:

## Law method

  - `overlap_student_law(law::AbstractOverlapStudentLaw, f::AbstractVector{Bool}, window::Real, n::Integer, T::Type) -> Tuple`: The divisor and the degrees of freedom.

### Arguments

  - `law`: Concrete subtype instance.
  - `f`: Whether each entry of the span of the series, from its first finite entry to its last, is a finite observation.
  - `window`: Number of entries that the data of one entry spans.
  - `n`: The number of finite entries, `count(f)`.
  - `T`: The type of the statistic, which a law that computes its values reads them in.

### Returns

  - `nd::Real`: The divisor of ``Q``. A divisor that is not positive gives a `NaN` test.
  - `nu::Real`: The degrees of freedom. A value that is not positive gives a `NaN` test.

# Related

  - [`BandCountLaw`](@ref)
  - [`NullMomentLaw`](@ref)
  - [`OverlapMeanTest`](@ref)
  - [`overlap_student_law`](@ref)
"""
abstract type AbstractOverlapStudentLaw <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Divides the sum of the centred products by the number of entries less the width of the band, and reads the Student law with as many degrees of freedom.

The rule is ``n_{d} = \\nu = n - 2L - 1``. The divisor is almost unbiased for ``n`` times the variance of the mean under the null hypothesis. The degrees of freedom are not: the untapered sum varies much more than a sum of ``n - 2L - 1`` independent squares, so the test rejects too often at ``L > 0``. At the 5% level it rejects a no-skill moving average of order four in 60 entries 7.8% of the time, and one of order nineteen in 120 entries 12.4% of the time. [`NullMomentLaw`](@ref) holds the level. This law costs nothing past the sum itself.

# Constructors

    BandCountLaw() -> BandCountLaw

# Examples

```jldoctest
julia> PortfolioOptimisers.overlap_student_law(BandCountLaw(), trues(60), 5, 60, Float64)
(51, 51)
```

# Related

  - [`AbstractOverlapStudentLaw`](@ref)
  - [`NullMomentLaw`](@ref)
  - [`OverlapMeanTest`](@ref)
"""
struct BandCountLaw <: AbstractOverlapStudentLaw end
"""
$(DocStringExtensions.TYPEDEF)

Reads the divisor and the degrees of freedom off the first two moments of the sum of the centred products under the null hypothesis of the test.

Under the null hypothesis each entry is the sum of `window` independent periods, sampled one period apart, so the correlation of two entries ``j`` entries apart is ``\\max(1 - j / w, 0)``. That is the correlation of a no-skill score of overlapping windows, and of an equal-weight moving average of order ``L`` when ``w = L + 1``. The divisor makes the variance of the mean unbiased under that law, at every length, with every gap of the series. The degrees of freedom match the variance of ``Q`` to the variance of a scaled chi-squared law, the rule of the source. Both are exact for a Gaussian series, and their only approximation is the Student law itself.

At the 5% level the test rejects a no-skill moving average of order four in 60 entries 4.9% of the time, and one of order nineteen in 120 entries 4.5% of the time. With gaps it holds the level the same way. The degrees of freedom are much smaller than ``n - 2L - 1``, because the untapered sum varies much more than a sum of independent squares. The computation reads each entry of the band once through two prefix sums of the mask of the finite entries, so its cost is linear in the length of the series and in ``L``, the order of the cost of ``Q``.

# Mathematical definition

```math
\\begin{align}
\\left( \\Sigma_{0} \\right)_{i j} &= \\max\\left( 1 - \\frac{|t_{i} - t_{j}|}{w},\\, 0 \\right)\\,, \\quad A = M W M\\,, \\quad M = I - \\frac{1}{n} \\mathbf{1} \\mathbf{1}^{\\intercal}\\,, \\\\
n_{d} &= \\frac{n \\operatorname{tr}\\left( A \\Sigma_{0} \\right)}{\\mathbf{1}^{\\intercal} \\Sigma_{0} \\mathbf{1}}\\,, \\\\
\\nu &= \\frac{\\operatorname{tr}\\left( A \\Sigma_{0} \\right)^{2}}{\\operatorname{tr}\\left( \\left( A \\Sigma_{0} \\right)^{2} \\right)}\\,.
\\end{align}
```

Where:

  - ``t_{i}``: The position in the series of finite entry ``i``.
  - ``w``: `window`.
  - ``\\Sigma_{0}``: The correlation of the finite entries under the null hypothesis.
  - ``W``: The weights of the untapered sum, one when ``|t_{i} - t_{j}| \\leq L`` and zero after it.
  - ``M``: The centring matrix of the ``n`` finite entries.
  - ``n_{d}``: The divisor. ``\\mathbb{E}[Q] = \\operatorname{tr}(A \\Sigma_{0})`` and ``n^{2} \\operatorname{Var}(\\bar{c}) = \\mathbf{1}^{\\intercal} \\Sigma_{0} \\mathbf{1}``, so ``Q / (n\\, n_{d})`` is unbiased for the variance of the mean.
  - ``\\nu``: The degrees of freedom, ``2 \\mathbb{E}[Q]^{2} / \\operatorname{Var}(Q)`` for a Gaussian series.

A span of at most ``L + 1`` entries puts every pair in the band, so ``Q`` is zero and the law gives ``n_{d} = \\nu = 0``.

# Constructors

    NullMomentLaw() -> NullMomentLaw

# Examples

```jldoctest
julia> nd, nu = PortfolioOptimisers.overlap_student_law(NullMomentLaw(), trues(60), 5, 60, Float64);

julia> round(nd; digits = 3), round(nu; digits = 3)
(51.174, 8.111)
```

# Related

  - [`AbstractOverlapStudentLaw`](@ref)
  - [`BandCountLaw`](@ref)
  - [`OverlapMeanTest`](@ref)

# References

  - $(ref_dict[:satterthwaite1946]) Section 2, Equation 6, page 111.
"""
struct NullMomentLaw <: AbstractOverlapStudentLaw end
"""
    overlap_student_law(law::BandCountLaw, f::AbstractVector{Bool}, window::Real,
                        n::Integer, T::Type) -> Tuple
    overlap_student_law(law::NullMomentLaw, f::AbstractVector{Bool}, window::Real,
                        n::Integer, T::Type) -> Tuple

Return the divisor of the long-run variance of an [`OverlapMeanTest`](@ref), and the degrees of freedom of its Student law.

# Arguments

  - `law`: The law.
  - `f`: Whether each entry of the span of the series is a finite observation.
  - `window`: Number of entries that the data of one entry spans.
  - `n`: The number of finite entries.
  - `T`: The type of the statistic. [`NullMomentLaw`](@ref) computes in the promotion of `T` and the type of `window`.

# Returns

  - `nd::Real`: The divisor.
  - `nu::Real`: The degrees of freedom.

# Related

  - [`AbstractOverlapStudentLaw`](@ref)
"""
function overlap_student_law(::BandCountLaw, ::AbstractVector{Bool}, window::Real,
                             n::Integer, ::Type)
    nu = n - 2 * (ceil(Int, window) - 1) - 1
    return nu, nu
end
function overlap_student_law(::NullMomentLaw, f::AbstractVector{Bool}, window::Real,
                             n::Integer, ::Type{T}) where {T}
    z = zero(promote_type(T, typeof(window)))
    L = ceil(Int, window) - 1
    if iszero(L)
        # The band is the diagonal and the null law is independence: the classical t-test.
        return z + (n - 1), z + (n - 1)
    end
    if length(f) <= L + 1
        return z, z
    end
    F0, F1 = band_prefix_sums(f)
    u, v, a, b, uv = overlap_null_rows(f, F0, F1, window, L, z)
    EQ = b - 2 * uv / n + a * b / n^2
    tr2 = overlap_null_second_moment(f, F0, F1, window, L, u, v, (a, b, uv), n, z)
    return n * EQ / b, EQ^2 / tr2
end
"""
    band_prefix_sums(f::AbstractVector{Bool}) -> Tuple

Return the prefix sums of the mask of the finite entries of a series, and of the mask times the position.

# Arguments

  - `f`: Whether each entry is a finite observation.

# Returns

  - `F0::Vector{Int}`: `F0[l + 1]` is the number of finite entries at positions up to `l`, `F0[1] = 0`.
  - `F1::Vector{Int}`: `F1[l + 1]` is the sum of their positions.

# Related

  - [`band_kernel_sum`](@ref)
  - [`NullMomentLaw`](@ref)
"""
function band_prefix_sums(f::AbstractVector{Bool})
    P = length(f)
    F0 = zeros(Int, P + 1)
    F1 = zeros(Int, P + 1)
    for l in 1:P
        F0[l + 1] = F0[l] + f[l]
        F1[l + 1] = F1[l] + l * f[l]
    end
    return F0, F1
end
"""
    band_kernel_sum(F0::AbstractVector{<:Integer}, F1::AbstractVector{<:Integer}, w::Real,
                    lo::Integer, hi::Integer, k::Integer, z::Number) -> Number

Return the sum of ``w - |l - k|`` over the finite entries ``l`` in `lo:hi`, which is an entry of ``W \\Sigma_{0}`` of [`NullMomentLaw`](@ref).

The sum splits at `k`. On each side the term is linear in ``l``, so the two prefix sums of [`band_prefix_sums`](@ref) give it in a fixed number of operations. The integer part is exact, and only the product with `w` rounds.

# Arguments

  - `F0`, `F1`: The prefix sums of [`band_prefix_sums`](@ref).
  - `w`: The window.
  - `lo`, `hi`: The range of ``l``, clipped to the series. The clipped range must not be empty.
  - `k`: The position that the distance reads.
  - `z`: The zero of the type the sum is read in.

# Returns

  - `s::Number`: The sum.

# Related

  - [`band_prefix_sums`](@ref)
  - [`NullMomentLaw`](@ref)
"""
function band_kernel_sum(F0::AbstractVector{<:Integer}, F1::AbstractVector{<:Integer},
                         w::Real, lo::Integer, hi::Integer, k::Integer, z::Number)
    lo = max(lo, 1)
    hi = min(hi, length(F0) - 1)
    m = clamp(k, lo - 1, hi)
    s0l = F0[m + 1] - F0[lo]
    s1l = F1[m + 1] - F1[lo]
    s0r = F0[hi + 1] - F0[m + 1]
    s1r = F1[hi + 1] - F1[m + 1]
    return z + w * (s0l + s0r) - ((k * s0l - s1l) + (s1r - k * s0r))
end
"""
    overlap_null_rows(f::AbstractVector{Bool}, F0::AbstractVector{<:Integer},
                      F1::AbstractVector{<:Integer}, w::Real, L::Integer, z::Number) -> Tuple

Return the row sums of the band of weights and of the null correlation of [`NullMomentLaw`](@ref), their totals, and the inner product of the two.

The correlation is scaled by `w`, ``w - j`` at the lag ``j``. The divisor and the degrees of freedom do not read the scale.

# Arguments

  - `f`: Whether each entry is a finite observation.
  - `F0`, `F1`: The prefix sums of [`band_prefix_sums`](@ref).
  - `w`: The window.
  - `L`: The order of the band.
  - `z`: The zero of the type the sums are read in.

# Returns

  - `u::Vector{Int}`: ``W \\mathbf{1}``, zero at a missing entry.
  - `v::Vector`: ``w \\Sigma_{0} \\mathbf{1}``, zero at a missing entry.
  - `a::Int`: ``\\mathbf{1}^{\\intercal} W \\mathbf{1}``.
  - `b`: ``w \\mathbf{1}^{\\intercal} \\Sigma_{0} \\mathbf{1}``.
  - `uv`: ``u^{\\intercal} v``.

# Related

  - [`NullMomentLaw`](@ref)
"""
function overlap_null_rows(f::AbstractVector{Bool}, F0::AbstractVector{<:Integer},
                           F1::AbstractVector{<:Integer}, w::Real, L::Integer, z::Number)
    P = length(f)
    u = zeros(Int, P)
    v = fill(z, P)
    a = 0
    b = z
    uv = z
    for i in 1:P
        if f[i]
            u[i] = F0[min(i + L, P) + 1] - F0[max(i - L, 1)]
            v[i] = band_kernel_sum(F0, F1, w, i - L, i + L, i, z)
            a += u[i]
            b += v[i]
            uv += u[i] * v[i]
        end
    end
    return u, v, a, b, uv
end
"""
    overlap_null_second_moment(f::AbstractVector{Bool}, F0::AbstractVector{<:Integer},
                               F1::AbstractVector{<:Integer}, w::Real, L::Integer,
                               u::AbstractVector{<:Integer}, v::AbstractVector,
                               abuv::Tuple, n::Integer, z::Number) -> Number

Return ``\\operatorname{tr}((A \\Sigma_{0})^{2})`` of [`NullMomentLaw`](@ref), with the correlation scaled by `w`.

The centring is a correction of rank two, ``M \\Sigma_{0} M = \\Sigma_{0} + U C U^{\\intercal}`` with ``U = [\\mathbf{1}, \\Sigma_{0} \\mathbf{1}]``. So the trace is ``\\operatorname{tr}(W \\Sigma_{0} W \\Sigma_{0})`` of [`overlap_null_square_trace`](@ref), plus terms that read the vectors ``\\Sigma_{0} u`` and ``W v`` and a product of ``2 \\times 2`` matrices.

# Arguments

  - `f`: Whether each entry is a finite observation.
  - `F0`, `F1`: The prefix sums of [`band_prefix_sums`](@ref).
  - `w`: The window.
  - `L`: The order of the band.
  - `u`, `v`: The row sums of [`overlap_null_rows`](@ref).
  - `abuv`: Its totals `(a, b, uv)`.
  - `n`: The number of finite entries.
  - `z`: The zero of the type the sums are read in.

# Returns

  - `tr2::Number`: The trace.

# Related

  - [`NullMomentLaw`](@ref)
"""
function overlap_null_second_moment(f::AbstractVector{Bool}, F0::AbstractVector{<:Integer},
                                    F1::AbstractVector{<:Integer}, w::Real, L::Integer,
                                    u::AbstractVector{<:Integer}, v::AbstractVector,
                                    abuv::Tuple, n::Integer, z::Number)
    a, b, uv = abuv
    P = length(f)
    susu = z
    suwv = z
    vwv = z
    for i in 1:P
        if f[i]
            su = z
            wv = z
            for k in max(1, i - L):min(P, i + L)
                su += (w - abs(i - k)) * u[k]
                wv += v[k]
            end
            susu += u[i] * su
            suwv += su * wv
            vwv += v[i] * wv
        end
    end
    c11 = b / n^2
    c12 = -one(z) / n
    g11 = a * c11 + uv * c12
    g12 = a * c12
    g21 = uv * c11 + vwv * c12
    g22 = uv * c12
    return overlap_null_square_trace(f, F0, F1, w, L, v, z) +
           2 * (b * susu / n^2 - 2 * suwv / n) +
           g11^2 +
           2 * g12 * g21 +
           g22^2
end
"""
    overlap_null_square_trace(f::AbstractVector{Bool}, F0::AbstractVector{<:Integer},
                              F1::AbstractVector{<:Integer}, w::Real, L::Integer,
                              v::AbstractVector, z::Number) -> Number

Return ``\\operatorname{tr}(W \\Sigma_{0} W \\Sigma_{0})`` of [`NullMomentLaw`](@ref), with the correlation scaled by `w`.

The trace is ``\\sum_{i, k} K_{i k} K_{k i}`` with ``K = W \\Sigma_{0}``. An entry ``K_{i k}`` sums the correlation with ``k`` over the finite entries that are at most ``L`` from both ``i`` and ``k``, and ``K_{k i}`` sums the correlation with ``i`` over the same entries. So each pair is two calls of [`band_kernel_sum`](@ref), and only the pairs at most ``2L`` apart are not zero. The diagonal entry ``K_{i i}`` is `v[i]`.

# Arguments

  - `f`: Whether each entry is a finite observation.
  - `F0`, `F1`: The prefix sums of [`band_prefix_sums`](@ref).
  - `w`: The window.
  - `L`: The order of the band.
  - `v`: The row sums of the correlation, from [`overlap_null_rows`](@ref).
  - `z`: The zero of the type the sum is read in.

# Returns

  - `t::Number`: The trace.

# Related

  - [`overlap_null_second_moment`](@ref)
"""
function overlap_null_square_trace(f::AbstractVector{Bool}, F0::AbstractVector{<:Integer},
                                   F1::AbstractVector{<:Integer}, w::Real, L::Integer,
                                   v::AbstractVector, z::Number)
    t = z
    for i in eachindex(f)
        if f[i]
            t += v[i]^2 + 2 * band_pair_products(f, F0, F1, w, L, i, z)
        end
    end
    return t
end
"""
    band_pair_products(f::AbstractVector{Bool}, F0::AbstractVector{<:Integer},
                       F1::AbstractVector{<:Integer}, w::Real, L::Integer, i::Integer,
                       z::Number) -> Number

Return ``\\sum_{k > i} K_{i k} K_{k i}`` of [`overlap_null_square_trace`](@ref), over the finite entries ``k`` at most ``2L`` after ``i``.

# Arguments

  - `f`: Whether each entry is a finite observation.
  - `F0`, `F1`: The prefix sums of [`band_prefix_sums`](@ref).
  - `w`: The window.
  - `L`: The order of the band.
  - `i`: The row.
  - `z`: The zero of the type the sum is read in.

# Returns

  - `s::Number`: The sum.

# Related

  - [`overlap_null_square_trace`](@ref)
"""
function band_pair_products(f::AbstractVector{Bool}, F0::AbstractVector{<:Integer},
                            F1::AbstractVector{<:Integer}, w::Real, L::Integer, i::Integer,
                            z::Number)
    s = z
    for k in (i + 1):min(length(f), i + 2 * L)
        if f[k]
            s += band_kernel_sum(F0, F1, w, k - L, i + L, k, z) *
                 band_kernel_sum(F0, F1, w, k - L, i + L, i, z)
        end
    end
    return s
end
"""
$(DocStringExtensions.TYPEDEF)

Tests the mean of a series against zero over the untapered long-run variance at the order that the overlap of its windows sets.

The test reads the overlap off the `window` that the caller derives from the grid, such as [`forecast_ic_window`](@ref). Its order is ``L = \\lceil w \\rceil - 1``, the number of entries on each side of a given entry that share data with it. It reads no dependence past the overlap. A series that is persistent past the overlap gives a t-statistic that is too large, and the untapered sum can be negative on a short series. [`WildBootstrapMeanTest`](@ref) is the default because of this.

# The law sets the divisor and the degrees of freedom

The sum ``Q`` of the centred products reads the deviations from the sample mean, not from the true mean, so its expectation is not ``n`` times the long-run variance. Under the null hypothesis each entry is a sum of `window` independent periods, the score of a no-skill forecast over overlapping windows. The field `law` reads the divisor ``n_{d}`` of ``Q`` and the degrees of freedom ``\\nu`` of the Student law off that null law. The default [`NullMomentLaw`](@ref) makes the variance of the mean unbiased, and matches the variance of ``Q``. It rejects a no-skill moving average of order four in 60 entries 4.9% of the time at the 5% level. [`BandCountLaw`](@ref), the rule ``n_{d} = \\nu = n - 2L - 1``, rejects it 7.8% of the time, and a divisor of ``n`` with the normal law about 10.6%. At ``L = 0`` both laws give ``n - 1``, and the test is the classical t-test, whose law under independent Gaussian entries is the Student law with ``n - 1`` degrees of freedom.

# Mathematical definition

```math
\\begin{align}
\\gamma_{j} &= \\frac{1}{n_{d}} \\sum_{t,\\, t + j \\in \\mathcal{T}} \\left( c_{t} - \\bar{c} \\right) \\left( c_{t + j} - \\bar{c} \\right)\\,, \\\\
\\sigma^{2} &= \\gamma_{0} + 2 \\sum_{j = 1}^{L} \\gamma_{j}\\,, \\\\
t &= \\frac{\\bar{c}}{\\sigma} \\sqrt{n}\\,, \\\\
p &= 2 \\left( 1 - F_{\\nu}(|t|) \\right)\\,, \\\\
\\bar{c} &\\pm F_{\\nu}^{-1}(1 - q / 2) \\frac{\\sigma}{\\sqrt{n}}\\,.
\\end{align}
```

Where:

  - ``c_{t}``: Entry ``t`` of the series.
  - ``\\mathcal{T}``: The entries at which it is finite.
  - ``n``: The number of entries in ``\\mathcal{T}``.
  - ``\\bar{c}``: The mean over ``\\mathcal{T}``.
  - ``\\gamma_{j}``: The autocovariance at lag ``j``.
  - ``L``: The order, ``\\lceil w \\rceil - 1``, with ``w`` the `window`.
  - ``n_{d}``: The divisor that `law` sets, [`overlap_student_law`](@ref).
  - ``\\sigma``: The long-run standard deviation, ``\\sqrt{n}`` times the standard error of the mean.
  - ``F_{\\nu}``: The distribution function of the Student law with the ``\\nu`` degrees of freedom that `law` sets.
  - ``t``: The t-statistic.
  - ``p``: The two-sided p-value.
  - ``q``: `q`, so the last line is the confidence interval of the mean at the level ``1 - q``.

A long-run variance that is not positive, or a divisor or degrees of freedom that are not positive, give a `NaN` t-statistic, p-value and interval. [`BandCountLaw`](@ref) has no positive count when ``n \\leq 2L + 1``, and [`NullMomentLaw`](@ref) none when the span of the series is at most ``L + 1`` entries. At a `window` of at most one the order is zero, and the t-statistic is ``\\mathrm{IR} \\sqrt{n}``. The rules of `min_obs` and `min_periods` are those of [`WildBootstrapMeanTest`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    OverlapMeanTest(;
        law::AbstractOverlapStudentLaw = NullMomentLaw(),
        q::Real = 0.05,
        min_obs::Option{<:Integer} = nothing,
        min_periods::Option{<:Real} = nothing,
    ) -> OverlapMeanTest

Keywords correspond to the struct's fields.

The law [`NullMomentLaw`](@ref) is the default because it holds the level of the test. Its cost is linear in the length of the series and in ``L``. A test under it takes 2 to 15 times a test under [`BandCountLaw`](@ref): 0.35 ms against 0.03 ms for 1,000 entries at a window of 20, and 1.8 ms against 0.12 ms for 5,000. The default [`WildBootstrapMeanTest`](@ref) takes 16 ms and 97 ms there. [`BandCountLaw`](@ref) is the rule of the library before 2026-10-16.

## Validation

  - `0 < q < 1`. Raises a `DomainError`.
  - `min_obs >= 2` and `min_periods > 0`, when given. Raise a `DomainError`.

# Examples

```jldoctest
julia> PortfolioOptimisers.mean_test(OverlapMeanTest(), [0.1, -0.2, 0.3, 0.2, NaN, 0.1], 1,
                                     nothing)
(t_stat = 1.1952286093343938, p_value = 0.29801481173121036, mean_ci_lower = -0.132294063538519, mean_ci_upper = 0.332294063538519)
```

# Related

  - [`AbstractMeanTestEstimator`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
  - [`AbstractOverlapStudentLaw`](@ref)
  - [`NullMomentLaw`](@ref)
  - [`BandCountLaw`](@ref)
  - [`forecast_ic_lags`](@ref)
  - [`mean_test`](@ref)

# References

  - $(ref_dict[:satterthwaite1946]) Section 2, Equation 6, page 111.
"""
@concrete struct OverlapMeanTest <: AbstractMeanTestEstimator
    """
    Law that sets the divisor of the long-run variance and the degrees of freedom of the Student law.
    """
    law
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
    function OverlapMeanTest(law::AbstractOverlapStudentLaw, q::Real,
                             min_obs::Option{<:Integer}, min_periods::Option{<:Real})
        assert_mean_test_rules(q, min_obs, min_periods)
        return new{typeof(law), typeof(q), typeof(min_obs), typeof(min_periods)}(law, q,
                                                                                 min_obs,
                                                                                 min_periods)
    end
end
function OverlapMeanTest(; law::AbstractOverlapStudentLaw = NullMomentLaw(), q::Real = 0.05,
                         min_obs::Option{<:Integer} = nothing,
                         min_periods::Option{<:Real} = nothing)::OverlapMeanTest
    return OverlapMeanTest(law, q, min_obs, min_periods)
end
"""
$(DocStringExtensions.TYPEDEF)

Tests the mean of a series against zero with a fixed-bandwidth long-run variance and a Gaussian wild bootstrap of its null law.

The bandwidth is the fraction `b` of the span of the series, so the long-run variance reads the dependence at every lag up to it, past the overlap of the windows as well. The null law of such a t-statistic is not the normal law, and a change of the volatility over time moves it. The test therefore reads the p-value off `n_sim` draws of the statistic under the null hypothesis, as the source states. Each draw multiplies the centred series by independent standard normal draws, and computes the t-statistic of the product with the same kernel and the same bandwidth. The field `alg` sets how the test draws that law, an [`AbstractBootstrapDraws`](@ref). Every algorithm draws the same law at a different cost.

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
        alg::AbstractBootstrapDraws = default_bootstrap_draws(kernel),
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

The defaults of `kernel`, `b` and `n_sim` are those of the source. Its simulations find that a bandwidth of a mid size, such as `b = 0.4`, holds the size of the test, and that the Bartlett kernel has more power than the quadratic spectral kernel. The source states an option that the test does not take: it can multiply the residuals of an AR(1) fit in place of the deviations from the mean, to reduce the effect of short-run dynamics. The level `q = 0.05` is the one of the library. The source uses `0.1` in its simulations and `0.01` in its application. A smaller `b` gives more power and a larger size distortion under strong dependence. The default `alg` is [`QuadratureSumDraws`](@ref) for the [`QuadraticSpectralKernel`](@ref) and [`ObservationDraws`](@ref) for every other kernel, from [`default_bootstrap_draws`](@ref).

## Validation

  - `alg` reads `kernel`, by [`assert_bootstrap_draws`](@ref). Raises a [`ConflictingArgumentError`](@ref).
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
  - [`AbstractBootstrapDraws`](@ref)
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
    Algorithm that draws the null law of the statistic. Every algorithm draws the same law, at a different cost.
    """
    alg
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
    function WildBootstrapMeanTest(kernel::AbstractLongRunKernel,
                                   alg::AbstractBootstrapDraws, b::Real,
                                   span::AbstractMeanTestSpan, ddof::Integer,
                                   n_sim::Integer, q::Real, min_obs::Option{<:Integer},
                                   min_periods::Option{<:Real}, rng::Random.AbstractRNG,
                                   seed::Option{<:Integer})
        assert_bootstrap_draws(kernel, alg)
        @argcheck(zero(b) < b <= one(b),
                  DomainError(b, "0 < b <= 1 must hold. Got\nb => $b"))
        @argcheck(ddof >= zero(ddof), DomainError(ddof, "ddof must be >= 0"))
        @argcheck(zero(n_sim) < n_sim, DomainError(n_sim, "n_sim must be > 0"))
        assert_resource_cap(n_sim, RESOURCE_LIMITS[].max_n_sim, :n_sim, :max_n_sim)
        assert_mean_test_rules(q, min_obs, min_periods)
        @argcheck(bootstrap_exceedance_cap(q, n_sim) >= 1,
                  DomainError(q,
                              "1 / (n_sim + 1) <= q must hold, so the interval has an order statistic of the draws. Got\nq => $q\nn_sim => $n_sim"))
        return new{typeof(kernel), typeof(alg), typeof(b), typeof(span), typeof(ddof),
                   typeof(n_sim), typeof(q), typeof(min_obs), typeof(min_periods),
                   typeof(rng), typeof(seed)}(kernel, alg, b, span, ddof, n_sim, q, min_obs,
                                              min_periods, rng, seed)
    end
end
function WildBootstrapMeanTest(; kernel::AbstractLongRunKernel = BartlettKernel(),
                               alg::AbstractBootstrapDraws = default_bootstrap_draws(kernel),
                               b::Real = 0.4,
                               span::AbstractMeanTestSpan = FiniteEntrySpan(),
                               ddof::Integer = 0, n_sim::Integer = 1_999, q::Real = 0.05,
                               min_obs::Option{<:Integer} = nothing,
                               min_periods::Option{<:Real} = nothing,
                               rng::Random.AbstractRNG = Random.default_rng(),
                               seed::Option{<:Integer} = nothing)::WildBootstrapMeanTest
    return WildBootstrapMeanTest(kernel, alg, b, span, ddof, n_sim, q, min_obs, min_periods,
                                 rng, seed)
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
    m, n, d, f, T = mean_test_centred(c, FiniteEntrySpan())
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
    nd, nu = overlap_student_law(mt.law, f, window, n, Ts)
    t = min(nd, nu) > zero(nd) ? studentised_mean(m, Q, n, nd) : nan
    if !isfinite(t)
        return (; t_stat = t, p_value = nan, mean_ci_lower = nan, mean_ci_upper = nan)
    end
    D = Distributions.TDist(nu)
    h = Distributions.quantile(D, 1 - mt.q / 2) * sqrt(Q / nd) / sqrt(Ts(n))
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
    if nd <= 0
        return (; t_stat = nan, p_value = nan, mean_ci_lower = nan, mean_ci_upper = nan)
    end
    Q, ats = wild_bootstrap_statistics(mt.kernel, mt.alg, mt.n_sim, d, f, n, nd, B, buf,
                                       rng)
    t = studentised_mean(m, Q, n, nd)
    if !isfinite(t)
        return (; t_stat = t, p_value = nan, mean_ci_lower = nan, mean_ci_upper = nan)
    end
    at = abs(t)
    K = count(x -> !(x < at), ats)
    k = mt.n_sim + 1 - bootstrap_exceedance_cap(mt.q, mt.n_sim)
    h = partialsort!(ats, k) * sqrt(Q / nd) / sqrt(Ts(n))
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

The history rule reads ``T / w`` as the number of holding periods that the history covers. For a forward window of `horizon` base periods scored every `step` of them, ``w`` is `horizon / step`. Each entry stands for its stride of `step` base periods, so ``T`` entries cover ``T \\cdot`` `step` base periods, which is ``T / w`` holding periods. That is also the number of independent windows whose mean has the variance of the mean of the ``T`` overlapping ones. For an integer ``w`` that is exact up to an edge term of order one. So ``T`` windows that do not overlap count as ``T`` periods.

The oracle counts the base periods from the first date to the last, both included, ``(T - 1) \\cdot`` `step` ``+ 1``. That counts the last entry as one base period and every other entry as `step`. So ``T`` windows that do not overlap count as ``T - 1 + 1 /`` `horizon` periods, and at a `step` above one the oracle needs one more entry than this rule. At `step = 1` the two rules are equal. The library keeps its rule.

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
    if n < 2 ||
       !isfinite(m) ||
       (!isnothing(mt.min_obs) && n < mt.min_obs) ||
       (!isnothing(mt.min_periods) && T / window < mt.min_periods)
        return false
    end
    return true
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

A series whose spread is round-off of its mean is constant: every entry of `d` is zero, so the long-run variance is not positive and the test answers `NaN`. The sum of a constant series over its count can miss its value, as `sum(fill(0.03, 100)) / 100` does, and two entries one unit in the last place apart, such as `0.3` and `0.1 + 0.2`, have a spread of the same size. The computed mean of ``n`` entries misses the exact mean by at most about ``n \\epsilon / 2`` times the largest absolute entry, the bound of a recursive sum, with ``\\epsilon`` the [`round_off_unit`](@ref) of the type of the mean. A largest absolute deviation of at most ``n \\epsilon`` times the largest absolute entry cannot be told apart from that error, and its t-statistic is of the order of ``1 / \\epsilon``. An exact number type has ``\\epsilon = 0``, so only an exactly constant series is constant.

# Arguments

  - `c`: The series.
  - `span`: The span rule.

# Returns

  - `m::Real`: The mean, `NaN` when no entry is finite.
  - `n::Int`: The number of finite entries.
  - `d::Vector{<:Real}`: The centred span, empty when no entry is finite, and all zero when the series is constant to round-off.
  - `f::BitVector`: Whether each entry of `d` is a finite observation.
  - `T::Int`: The span of [`mean_test_span_length`](@ref), `0` when no entry is finite.

# Related

  - [`mean_test`](@ref)
  - [`round_off_unit`](@ref)
"""
function mean_test_centred(c::VecNum, span::AbstractMeanTestSpan)
    Tm = float_if_integer(real(eltype(c)))
    n, s, i1, i2 = finite_sum_bounds(c, Tm)
    m = n > 0 ? s / n : Tm(NaN)
    P = iszero(n) ? 0 : i2 - i1 + 1
    d = Vector{typeof(m)}(undef, P)
    f = falses(P)
    for i in 1:P
        v = c[i1 + i - 1]
        f[i] = isfinite(v)
        d[i] = f[i] ? Tm(v) - m : zero(m)
    end
    zero_round_off_spread!(d, f, m, n)
    T = iszero(n) ? 0 : mean_test_span_length(span, length(c), i1, i2)
    return m, n, d, f, T
end
"""
    zero_round_off_spread!(d::VecNum, f::AbstractVector{Bool}, m::Number, n::Integer) -> VecNum

Set every entry of a centred series to zero when its spread is round-off of its mean, the rule that [`mean_test_centred`](@ref) states.

# Arguments

  - `d`: The centred series, with a zero at each missing observation. It is changed in place.
  - `f`: Whether each entry of `d` is a finite observation.
  - `m`: The mean.
  - `n`: The number of finite entries.

# Returns

  - `d::VecNum`: The series, all zero when its largest absolute entry is at most `n` times the [`round_off_unit`](@ref) of the type of `m` times the largest absolute entry of the series before the centring.

# Related

  - [`mean_test_centred`](@ref)
"""
function zero_round_off_spread!(d::VecNum, f::AbstractVector{Bool}, m::Number, n::Integer)
    dmax = zero(m)
    cmax = zero(m)
    for i in eachindex(d, f)
        dmax = max(dmax, abs(d[i]))
        cmax = f[i] ? max(cmax, abs(d[i] + m)) : cmax
    end
    if dmax <= n * round_off_unit(typeof(m)) * cmax
        fill!(d, zero(m))
    end
    return d
end
"""
    round_off_unit(::Type{T}) where {T <: Real} -> Real
    round_off_unit(::Type{T}) where {T <: Rational} -> Rational

Return the relative round-off of one operation in a number type: `eps(T)`, and zero for an exact `Rational`.

# Arguments

  - `T`: The number type.

# Returns

  - `e::Real`: The unit.

# Related

  - [`mean_test_centred`](@ref)
"""
function round_off_unit(::Type{T}) where {T <: Real}
    return eps(T)
end
function round_off_unit(::Type{T}) where {T <: Rational}
    return zero(T)
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
    studentised_mean(m::Real, Q::Real, n::Integer, nd::Real) -> Real

Return the mean over its long-run standard error.

The long-run variance is ``Q / n_{d}``. [`WildBootstrapMeanTest`](@ref) divides by ``n``, as its source does. Its divisor scales the t-statistic of the observed series and of every bootstrap draw by one factor, so it moves no p-value and no interval. [`OverlapMeanTest`](@ref) divides by the count of its [`AbstractOverlapStudentLaw`](@ref), which removes the bias of the centring under its null hypothesis, and its p-value reads the divisor.

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
function studentised_mean(m::Real, Q::Real, n::Integer, nd::Real)
    Ts = typeof(sqrt(one(Q)))
    lrsd = Q > zero(Q) ? sqrt(Q / nd) : Ts(NaN)
    return isfinite(m) && isfinite(lrsd) ? m / lrsd * sqrt(Ts(n)) : Ts(NaN)
end

export OverlapMeanTest, WildBootstrapMeanTest, BartlettKernel, QuadraticSpectralKernel,
       FiniteEntrySpan, FullGridSpan, BandCountLaw, NullMomentLaw
public AbstractMeanTestEstimator, AbstractLongRunKernel, AbstractMeanTestSpan,
       AbstractBootstrapDraws, AbstractOverlapStudentLaw, mean_test, mean_test_rng,
       long_run_kernel_weight, mean_test_span_length, overlap_student_law
