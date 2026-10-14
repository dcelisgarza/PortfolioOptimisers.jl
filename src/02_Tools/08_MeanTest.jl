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

  - `wild_bootstrap_statistics(kernel::AbstractLongRunKernel, alg::AbstractBootstrapDraws, M::Integer, d::VecNum, f::AbstractVector{Bool}, n::Integer, nd::Integer, B::Real, buf::VecNum, rng::Random.AbstractRNG) -> Vector{<:Real}`: The absolute t-statistics of `M` draws.

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

  - `ats::Vector{<:Real}`: The `M` absolute t-statistics. A statistic that is not finite is `Inf`.

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
$(DocStringExtensions.TYPEDEF)

Draws one multiplier for each finite entry of the series, and computes the statistic of each draw from the product.

It reads every kernel at its exact weights. Every kernel reads the multipliers in the same order, one per finite entry and draw by draw, so two kernels answer their p-values from the same draws of one generator. Under [`BartlettKernel`](@ref) a draw reads sums of windows, so its cost is linear in the length of the series. Under any other kernel a block of draws is one product with the matrix of the weights, so the cost of a draw is quadratic in the length. It is the default of every kernel except [`QuadraticSpectralKernel`](@ref).

# Constructors

    ObservationDraws() -> ObservationDraws

# Examples

```jldoctest
julia> WildBootstrapMeanTest().alg
ObservationDraws()
```

# Related

  - [`AbstractBootstrapDraws`](@ref)
  - [`QuadratureSumDraws`](@ref)
  - [`EigenSumDraws`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
"""
struct ObservationDraws <: AbstractBootstrapDraws end
"""
$(DocStringExtensions.TYPEDEF)

Draws the statistic of each draw from a few Gaussian sums, through quadrature features of the [`QuadraticSpectralKernel`](@ref).

The quadratic spectral kernel is the cosine transform of a weight on a bounded band, so a Gauss-Legendre rule of ``2N`` nodes on the band writes the matrix of its weights as ``\\mathbf{\\Phi} \\mathbf{\\Phi}^{\\intercal}``, with ``2N`` features per entry. The statistic of a draw reads the multipliers only through the sum of the product and its sums against the centred features. These ``2N + 1`` sums are jointly Gaussian, with the covariance ``\\mathbf{H}^{\\intercal} \\mathbf{H}``, so the test draws them directly. A draw then costs ``O(N^{2})``, whatever the length of the series.

The largest angle that the test reads, ``c``, sets the number of nodes. Under [`FiniteEntrySpan`](@ref), ``c < 6 \\pi / (5 b)``, so the count depends on the bandwidth fraction `b` and not on the length of the series. The count is the smallest ``N`` whose bound on the error of the quadrature, at every lag, is at most `tol`. The bound is that of Gauss quadrature for a function analytic in the Bernstein ellipse ``E_{\\rho}``, minimised over ``\\rho``. Measured at a tolerance of `1e-14`, it agrees with the smallest sufficient count to one node: 10 nodes at `b = 0.4`, 20 at `b = 0.1` and 114 at `b = 0.01`. A fixed count is wrong at a small `b`: 32 nodes give an error of 0.57 at `b = 0.02`.

# Mathematical definition

```math
\\begin{align}
k(x) &\\approx \\sum_{l = 1}^{N} a_{l}^{2} \\cos\\left( \\frac{6 \\pi}{5} x u_{l} \\right)\\,, \\quad a_{l} = \\sqrt{\\frac{3}{2} w_{l} \\left( 1 - u_{l}^{2} \\right)}\\,, \\\\
\\boldsymbol{\\phi}_{i} &= \\left( a_{l} \\cos(\\theta_{i} u_{l}),\\, a_{l} \\sin(\\theta_{i} u_{l}) \\right)_{l = 1}^{N}\\,, \\quad \\theta_{i} = \\frac{6 \\pi (i - 1)}{5 B}\\,, \\\\
\\boldsymbol{h}_{i} &= d_{i} \\left( 1,\\, \\boldsymbol{\\phi}_{i} - \\bar{\\boldsymbol{\\phi}} \\right)\\,, \\quad \\mathbf{H}^{\\intercal} \\mathbf{H} = \\mathbf{R}^{\\intercal} \\mathbf{R}\\,, \\\\
\\boldsymbol{v} &= \\mathbf{R}^{\\intercal} \\boldsymbol{z}\\,, \\quad \\boldsymbol{z} \\sim \\mathcal{N}(\\boldsymbol{0}, \\mathbf{I})\\,, \\quad \\bar{u}^{*} = \\frac{v_{1}}{n}\\,, \\quad Q^{*} = \\sum_{k \\geq 2} v_{k}^{2}\\,, \\\\
\\epsilon_{N} &\\leq \\min_{\\rho > 1} \\frac{16}{5} \\left( 1 + \\alpha_{\\rho}^{2} \\right) \\cosh\\left( c \\beta_{\\rho} \\right) \\frac{\\rho^{-4N}}{\\rho^{2} - 1}\\,, \\quad c = \\frac{6 \\pi (P - 1)}{5 B}\\,.
\\end{align}
```

Where:

  - ``u_{l}``, ``w_{l}``: The positive nodes and their weights of the Gauss-Legendre rule of ``2N`` nodes on ``[-1, 1]``, from the eigenvalues of its Jacobi matrix.
  - ``a_{l}``: The amplitude of node ``l``.
  - ``\\boldsymbol{\\phi}_{i}``: The features of entry ``i`` of the span, and ``\\bar{\\boldsymbol{\\phi}}`` their mean over the finite entries.
  - ``B``: The bandwidth, in entries.
  - ``d_{i}``: Entry ``i`` of the centred series. A missing entry has no row.
  - ``\\mathbf{R}``: The triangular factor of the QR factorisation of ``\\mathbf{H}``.
  - ``\\bar{u}^{*}``, ``Q^{*}``: The mean and the weighted sum of the centred product of a draw, which [`studentised_mean`](@ref) reads.
  - ``\\epsilon_{N}``: The largest error of the quadrature over the lags of the span.
  - ``\\alpha_{\\rho} = (\\rho + \\rho^{-1}) / 2``, ``\\beta_{\\rho} = (\\rho - \\rho^{-1}) / 2``: The semi-axes of ``E_{\\rho}``.
  - ``P``: The length of the span, from the first finite entry to the last.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    QuadratureSumDraws(;
        tol::Option{<:Real} = nothing,
        nodes::Option{<:Integer} = nothing,
    ) -> QuadratureSumDraws

Keywords correspond to the struct's fields.

## Validation

  - `0 < tol < 1`, when given. Raises a `DomainError`.
  - `nodes > 0`, when given. Raises a `DomainError`.
  - A [`WildBootstrapMeanTest`](@ref) whose kernel is not the [`QuadraticSpectralKernel`](@ref) refuses it. Raises a [`ConflictingArgumentError`](@ref).

# Examples

```jldoctest
julia> WildBootstrapMeanTest(; kernel = QuadraticSpectralKernel()).alg
QuadratureSumDraws
    tol ┼ nothing
  nodes ┴ nothing

julia> PortfolioOptimisers.quadrature_node_count(6 * pi / (5 * 0.1), 1e-14)
19
```

# Related

  - [`AbstractBootstrapDraws`](@ref)
  - [`ObservationDraws`](@ref)
  - [`EigenSumDraws`](@ref)
  - [`QuadraticSpectralKernel`](@ref)
  - [`WildBootstrapMeanTest`](@ref)

# References

  - $(ref_dict[:golubwelsch1969])
  - $(ref_dict[:trefethen2019])
"""
@concrete struct QuadratureSumDraws <: AbstractBootstrapDraws
    """
    Bound on the error of the quadrature at every lag that the test reads, in `(0, 1)`, or `nothing` for the machine epsilon of the kernel weights. The weight at lag zero is one, so the bound is relative to the largest weight.
    """
    tol
    """
    Number of positive nodes of the rule, or `nothing` to derive it from `tol`. A count overrides the derivation, and a count below the derived one gives a kernel with an error above `tol`.
    """
    nodes
    function QuadratureSumDraws(tol::Option{<:Real}, nodes::Option{<:Integer})
        if !isnothing(tol)
            assert_unit_interval(tol, :tol)
        end
        if !isnothing(nodes)
            @argcheck(nodes > zero(nodes), DomainError(nodes, "nodes must be > 0"))
        end
        return new{typeof(tol), typeof(nodes)}(tol, nodes)
    end
end
function QuadratureSumDraws(; tol::Option{<:Real} = nothing,
                            nodes::Option{<:Integer} = nothing)::QuadratureSumDraws
    return QuadratureSumDraws(tol, nodes)
end
"""
$(DocStringExtensions.TYPEDEF)

Draws the statistic of each draw from a few Gaussian sums, through an eigen factor of the matrix of the kernel weights.

The algorithm factors the symmetric matrix ``\\mathbf{W}`` of the weights of every pair once, as ``\\mathbf{W} = \\mathbf{\\Phi}_{+} \\mathbf{\\Phi}_{+}^{\\intercal} - \\mathbf{\\Phi}_{-} \\mathbf{\\Phi}_{-}^{\\intercal}``, from its eigenvalues above round-off in absolute value. It then draws the Gaussian sums of [`QuadratureSumDraws`](@ref) with these columns as the features, and subtracts the squares of the sums that read ``\\mathbf{\\Phi}_{-}``. The factor is exact, so it reads every kernel, one that a user writes included, and a kernel whose weights are not positive semidefinite. The factorisation costs ``O(P^{3})`` once, and a draw costs ``O(r^{2})``, with ``r`` the rank of the factor. An eigenvalue at most ``P \\epsilon \\max_{k} |\\lambda_{k}|`` in absolute value is round-off of zero, and the factor drops it.

# Constructors

    EigenSumDraws() -> EigenSumDraws

# Examples

```jldoctest
julia> WildBootstrapMeanTest(; alg = EigenSumDraws()).alg
EigenSumDraws()
```

# Related

  - [`AbstractBootstrapDraws`](@ref)
  - [`ObservationDraws`](@ref)
  - [`QuadratureSumDraws`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
"""
struct EigenSumDraws <: AbstractBootstrapDraws end
"""
    default_bootstrap_draws(kernel::AbstractLongRunKernel) -> AbstractBootstrapDraws

Return the draw algorithm that a [`WildBootstrapMeanTest`](@ref) takes by default for a kernel.

# Arguments

  - `kernel`: The kernel.

# Returns

  - `alg::AbstractBootstrapDraws`: [`QuadratureSumDraws`](@ref) for the [`QuadraticSpectralKernel`](@ref), whose draws then do not grow with the length of the series, and [`ObservationDraws`](@ref) for every other kernel.

# Related

  - [`AbstractBootstrapDraws`](@ref)
"""
function default_bootstrap_draws(::AbstractLongRunKernel)
    return ObservationDraws()
end
function default_bootstrap_draws(::QuadraticSpectralKernel)
    return QuadratureSumDraws()
end
"""
    assert_bootstrap_draws(kernel::AbstractLongRunKernel, alg::AbstractBootstrapDraws)

Refuse a kernel that a draw algorithm cannot read.

# Arguments

  - `kernel`: The kernel.
  - `alg`: The draw algorithm.

# Validation

  - [`QuadratureSumDraws`](@ref) reads the [`QuadraticSpectralKernel`](@ref) alone. Raises a [`ConflictingArgumentError`](@ref) for any other kernel.

# Returns

  - `nothing`.

# Related

  - [`AbstractBootstrapDraws`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
"""
function assert_bootstrap_draws(::AbstractLongRunKernel, ::AbstractBootstrapDraws)::Nothing
    return nothing
end
function assert_bootstrap_draws(kernel::AbstractLongRunKernel,
                                ::QuadratureSumDraws)::Nothing
    return throw(ConflictingArgumentError("QuadratureSumDraws reads the quadrature features of the QuadraticSpectralKernel, and $(nameof(typeof(kernel))) has none. Use ObservationDraws() or EigenSumDraws() with it."))
end
function assert_bootstrap_draws(::QuadraticSpectralKernel, ::QuadratureSumDraws)::Nothing
    return nothing
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

The defaults are those of the source. Its simulations find that a bandwidth of a mid size, such as `b = 0.4`, holds the size of the test, and that the Bartlett kernel has more power than the quadratic spectral kernel. A smaller `b` gives more power and a larger size distortion under strong dependence. The default `alg` is [`QuadratureSumDraws`](@ref) for the [`QuadraticSpectralKernel`](@ref) and [`ObservationDraws`](@ref) for every other kernel, from [`default_bootstrap_draws`](@ref).

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
    ats = wild_bootstrap_statistics(mt.kernel, mt.alg, mt.n_sim, d, f, n, nd, B, buf, rng)
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
    wild_bootstrap_statistics(kernel::BartlettKernel, alg::ObservationDraws, M::Integer,
                              d::VecNum, f::AbstractVector{Bool}, n::Integer, nd::Integer,
                              B::Real, buf::VecNum,
                              rng::Random.AbstractRNG) -> Vector{<:Real}
    wild_bootstrap_statistics(kernel::AbstractLongRunKernel, alg::ObservationDraws,
                              M::Integer, d::VecNum, f::AbstractVector{Bool}, n::Integer,
                              nd::Integer, B::Real, buf::VecNum,
                              rng::Random.AbstractRNG) -> Vector{<:Real}
    wild_bootstrap_statistics(kernel::QuadraticSpectralKernel, alg::QuadratureSumDraws,
                              M::Integer, d::VecNum, f::AbstractVector{Bool}, n::Integer,
                              nd::Integer, B::Real, buf::VecNum,
                              rng::Random.AbstractRNG) -> Vector{<:Real}
    wild_bootstrap_statistics(kernel::AbstractLongRunKernel, alg::EigenSumDraws, M::Integer,
                              d::VecNum, f::AbstractVector{Bool}, n::Integer, nd::Integer,
                              B::Real, buf::VecNum,
                              rng::Random.AbstractRNG) -> Vector{<:Real}

Return the absolute t-statistics of `M` wild bootstrap draws of a centred series.

Each draw multiplies each finite entry of the centred series by an independent standard normal draw, centres the product over the finite entries again, and computes its t-statistic with the kernel and the bandwidth of the observed one. A missing entry stays zero, and it takes no draw.

Under [`ObservationDraws`](@ref) both methods take the draws in the same order, one per finite entry and draw by draw, so they answer the same statistics from the same generator. The Bartlett method reads each draw with [`long_run_quadratic_form`](@ref). The generic method writes the weights of every pair into the matrix ``\\mathbf{A} = \\mathbf{D} \\mathbf{W} \\mathbf{D}`` once, and reads a block of draws with one matrix product. For a draw ``\\boldsymbol{r}`` the centred product is ``\\boldsymbol{u} = \\mathbf{D} \\boldsymbol{r} - \\bar{u} \\boldsymbol{f}``, so

```math
\\boldsymbol{u}^{\\intercal} \\mathbf{W} \\boldsymbol{u} = \\boldsymbol{r}^{\\intercal} \\mathbf{A} \\boldsymbol{r} - 2 \\bar{u} \\left( \\mathbf{D} \\mathbf{W} \\boldsymbol{f} \\right)^{\\intercal} \\boldsymbol{r} + \\bar{u}^{2} \\boldsymbol{f}^{\\intercal} \\mathbf{W} \\boldsymbol{f}\\,.
```

Where:

  - ``\\mathbf{D}``: The diagonal matrix of the centred series.
  - ``\\mathbf{W}``: The matrix of the kernel weights of every pair.
  - ``\\boldsymbol{f}``: The mask of the finite entries.
  - ``\\bar{u}``: The mean of the product, ``\\boldsymbol{d}^{\\intercal} \\boldsymbol{r} / n``.

Under [`QuadratureSumDraws`](@ref) and [`EigenSumDraws`](@ref) the method writes ``\\mathbf{W}`` as a product of features, by [`quadrature_kernel_features`](@ref) or [`eigen_kernel_features`](@ref), and draws the Gaussian sums of a draw directly, by [`sum_draw_statistics`](@ref).

# Arguments

  - `kernel`: The kernel.
  - `alg`: The draw algorithm.
  - `M`: The number of draws.
  - `d`: The centred series, with a zero at each missing observation. It is not changed.
  - `f`: Whether each entry of `d` is a finite observation.
  - `n`: The number of finite entries.
  - `nd`: The divisor of the long-run variance, `n - ddof`.
  - `B`: The bandwidth, in entries.
  - `buf`: Work buffer of length `length(d) + 1`. Only the Bartlett method reads it.
  - `rng`: The generator.

# Returns

  - `ats::Vector{<:Real}`: The `M` absolute t-statistics. A statistic that is not finite is `Inf`.

# Related

  - [`AbstractBootstrapDraws`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
  - [`long_run_quadratic_form`](@ref)
"""
function wild_bootstrap_statistics(kernel::BartlettKernel, ::ObservationDraws, M::Integer,
                                   d::VecNum, f::AbstractVector{Bool}, n::Integer,
                                   nd::Integer, B::Real, buf::VecNum,
                                   rng::Random.AbstractRNG)
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
function wild_bootstrap_statistics(kernel::AbstractLongRunKernel, ::ObservationDraws,
                                   M::Integer, d::VecNum, f::AbstractVector{Bool},
                                   n::Integer, nd::Integer, B::Real, ::VecNum,
                                   rng::Random.AbstractRNG)
    P = length(d)
    W = kernel_weight_matrix(kernel, P, B)
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
function wild_bootstrap_statistics(kernel::QuadraticSpectralKernel, alg::QuadratureSumDraws,
                                   M::Integer, d::VecNum, f::AbstractVector{Bool},
                                   n::Integer, nd::Integer, B::Real, ::VecNum,
                                   rng::Random.AbstractRNG)
    Phi = quadrature_kernel_features(kernel, alg, length(d), B)
    return sum_draw_statistics(M, d, f, n, nd, Phi, size(Phi, 2), rng)
end
function wild_bootstrap_statistics(kernel::AbstractLongRunKernel, ::EigenSumDraws,
                                   M::Integer, d::VecNum, f::AbstractVector{Bool},
                                   n::Integer, nd::Integer, B::Real, ::VecNum,
                                   rng::Random.AbstractRNG)
    Phi, kp = eigen_kernel_features(kernel, length(d), B)
    return sum_draw_statistics(M, d, f, n, nd, Phi, kp, rng)
end
"""
    kernel_weight_matrix(kernel::AbstractLongRunKernel, P::Integer, B::Real) -> Matrix{<:Real}

Return the matrix of the kernel weights of every pair of entries of a span.

# Arguments

  - `kernel`: The kernel.
  - `P`: The length of the span.
  - `B`: The bandwidth, in entries.

# Returns

  - `W::Matrix{<:Real}`: The `P × P` matrix with ``W_{ij} = k(|i - j| / B)``.

# Related

  - [`wild_bootstrap_statistics`](@ref)
  - [`eigen_kernel_features`](@ref)
"""
function kernel_weight_matrix(kernel::AbstractLongRunKernel, P::Integer, B::Real)
    w = [iszero(j) ? one(B / B) : long_run_kernel_weight(kernel, j / B) for j in 0:(P - 1)]
    # The weights are written in a loop rather than by a two-dimensional comprehension, whose
    # iterator type JET cannot pin.
    W = Matrix{eltype(w)}(undef, P, P)
    for j in 1:P, i in 1:P
        W[i, j] = w[abs(i - j) + 1]
    end
    return W
end
"""
    quadrature_kernel_features(kernel::QuadraticSpectralKernel, alg::QuadratureSumDraws,
                               P::Integer, B::Real) -> Matrix{<:Real}

Return the quadrature features of the quadratic spectral kernel at every entry of a span.

The number of positive nodes is `alg.nodes`, or [`quadrature_node_count`](@ref) at the largest angle of the span and the tolerance of `alg`. The nodes and the weights come from [`gauss_legendre_positive_nodes`](@ref). [`QuadratureSumDraws`](@ref) states the features.

# Arguments

  - `kernel`: The kernel.
  - `alg`: The draw algorithm, which holds the tolerance and the count.
  - `P`: The length of the span.
  - `B`: The bandwidth, in entries.

# Returns

  - `Phi::Matrix{<:Real}`: The `P × 2N` features, the cosine features of the ``N`` nodes and then their sine features, so ``\\mathbf{\\Phi} \\mathbf{\\Phi}^{\\intercal}`` is the matrix of the weights to the tolerance.

# Related

  - [`QuadratureSumDraws`](@ref)
  - [`wild_bootstrap_statistics`](@ref)
"""
function quadrature_kernel_features(::QuadraticSpectralKernel, alg::QuadratureSumDraws,
                                    P::Integer, B::Real)
    a = 6 * oftype(one(B) / B, pi) / 5
    c = a * (P - 1) / B
    tol = isnothing(alg.tol) ? eps(typeof(c)) : alg.tol
    N = isnothing(alg.nodes) ? quadrature_node_count(c, tol) : alg.nodes
    u, w = gauss_legendre_positive_nodes(N, typeof(c))
    Phi = Matrix{typeof(c)}(undef, P, 2 * N)
    for l in 1:N
        h = sqrt(3 * w[l] * (1 - u[l]^2) / 2)
        for i in 1:P
            s, co = sincos(a * (i - 1) / B * u[l])
            Phi[i, l] = h * co
            Phi[i, N + l] = h * s
        end
    end
    return Phi
end
"""
    quadrature_node_count(c::Real, tol::Real) -> Int

Return the least number of positive Gauss-Legendre nodes whose bound on the error of the quadratic spectral kernel, at every angle up to `c`, is at most `tol`.

[`quadrature_error_bound`](@ref) gives the bound. It falls as the count grows, so the function counts up from one.

# Arguments

  - `c`: The largest angle, ``6 \\pi x / 5`` at the largest lag ``x`` over the bandwidth.
  - `tol`: The bound.

# Returns

  - `N::Int`: The number of positive nodes, half the nodes of the rule.

# Related

  - [`QuadratureSumDraws`](@ref)
  - [`quadrature_kernel_features`](@ref)
"""
function quadrature_node_count(c::Real, tol::Real)
    lt = log(tol)
    N = 1
    while quadrature_error_bound(N, c) > lt
        N += 1
    end
    return N
end
"""
    quadrature_error_bound(N::Integer, c::Real) -> Real

Return the logarithm of the bound on the error of a Gauss-Legendre rule of `2N` nodes for ``\\frac{3}{4} (1 - u^{2}) \\cos(\\theta u)`` on ``[-1, 1]``, at every angle ``\\theta \\leq c``.

The integrand is entire. On the Bernstein ellipse ``E_{\\rho}`` its modulus is at most ``\\frac{3}{4} (1 + \\alpha_{\\rho}^{2}) \\cosh(c \\beta_{\\rho})``, and Gauss quadrature of ``m`` nodes errs by at most ``\\frac{64}{15} M_{\\rho} \\rho^{-2m} / (\\rho^{2} - 1)`` for a function bounded by ``M_{\\rho}`` there. The function minimises the bound over ``\\log \\rho`` in steps of `1 / 50` up to `8`, and works in logarithms so that no factor overflows. [`QuadratureSumDraws`](@ref) states the bound.

# Arguments

  - `N`: The number of positive nodes.
  - `c`: The largest angle.

# Returns

  - `lb::Real`: The logarithm of the bound.

# Related

  - [`quadrature_node_count`](@ref)
"""
function quadrature_error_bound(N::Integer, c::Real)
    lb = oftype(c, Inf)
    for k in 1:400
        t = oftype(c, k) / 50
        # log cosh(y) = y + log1p(exp(-2y)) - log(2), with no overflow at a large angle.
        y = c * sinh(t)
        lc = y + log1p(exp(-2 * y)) - log(oftype(c, 2))
        lb = min(lb,
                 log(oftype(c, 16) / 5) + log1p(cosh(t)^2) + lc - 4 * N * t -
                 log(expm1(2 * t)))
    end
    return lb
end
"""
    gauss_legendre_positive_nodes(N::Integer, T::Type) -> Tuple

Return the positive nodes and their weights of the Gauss-Legendre rule of `2N` nodes on ``[-1, 1]``.

The nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix of the Legendre polynomials, with the off-diagonal entries ``k / \\sqrt{4 k^{2} - 1}``. The weight of a node is twice the square of the first entry of its unit eigenvector. The rule is symmetric, so the positive half carries every node.

# Arguments

  - `N`: The number of positive nodes.
  - `T`: The type the rule is computed in.

# Returns

  - `u::Vector`: The `N` positive nodes, in ascending order.
  - `w::Vector`: Their weights.

# Related

  - [`quadrature_kernel_features`](@ref)
"""
function gauss_legendre_positive_nodes(N::Integer, T::Type)
    m = 2 * N
    # A dense symmetric matrix, not a `SymTridiagonal`: the tridiagonal solver of LAPACK takes
    # a leaf element type that a generic `T` does not state.
    J = zeros(T, m, m)
    for k in 1:(m - 1)
        J[k, k + 1] = k / sqrt(oftype(one(T), 4 * k^2 - 1))
    end
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(J))
    u = E.values[(N + 1):m]
    w = [2 * E.vectors[1, j]^2 for j in (N + 1):m]
    return u, w
end
"""
    eigen_kernel_features(kernel::AbstractLongRunKernel, P::Integer, B::Real) -> Tuple

Return the features of an exact eigen factor of the matrix of the kernel weights of a span, the columns of its positive eigenvalues first.

The matrix is that of [`kernel_weight_matrix`](@ref). [`EigenSumDraws`](@ref) states the factor and the rule that drops an eigenvalue as round-off.

# Arguments

  - `kernel`: The kernel.
  - `P`: The length of the span.
  - `B`: The bandwidth, in entries.

# Returns

  - `Phi::Matrix{<:Real}`: The features, ``\\mathbf{\\Phi} = (\\mathbf{\\Phi}_{+}, \\mathbf{\\Phi}_{-})``, with ``\\mathbf{W} = \\mathbf{\\Phi}_{+} \\mathbf{\\Phi}_{+}^{\\intercal} - \\mathbf{\\Phi}_{-} \\mathbf{\\Phi}_{-}^{\\intercal}`` to round-off.
  - `kp::Int`: The number of columns of ``\\mathbf{\\Phi}_{+}``.

# Related

  - [`EigenSumDraws`](@ref)
  - [`wild_bootstrap_statistics`](@ref)
"""
function eigen_kernel_features(kernel::AbstractLongRunKernel, P::Integer, B::Real)
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(kernel_weight_matrix(kernel, P, B)))
    lam = E.values
    tol = P * eps(eltype(lam)) * maximum(abs, lam)
    pos = findall(>(tol), lam)
    neg = findall(<(-tol), lam)
    Phi = Matrix{eltype(lam)}(undef, P, length(pos) + length(neg))
    for (k, j) in enumerate(vcat(pos, neg))
        h = sqrt(abs(lam[j]))
        for i in 1:P
            Phi[i, k] = h * E.vectors[i, j]
        end
    end
    return Phi, length(pos)
end
"""
    sum_draw_statistics(M::Integer, d::VecNum, f::AbstractVector{Bool}, n::Integer,
                        nd::Integer, Phi::MatNum, kp::Integer,
                        rng::Random.AbstractRNG) -> Vector{<:Real}

Return the absolute t-statistics of `M` wild bootstrap draws, drawn as the Gaussian sums that the statistic of a draw reads.

The rows of ``\\mathbf{H}`` are ``d_{i} (1, \\boldsymbol{\\phi}_{i} - \\bar{\\boldsymbol{\\phi}})`` over the finite entries, so the sums of a draw ``\\boldsymbol{r}`` are ``\\boldsymbol{v} = \\mathbf{H}^{\\intercal} \\boldsymbol{r}``, with the covariance ``\\mathbf{H}^{\\intercal} \\mathbf{H} = \\mathbf{R}^{\\intercal} \\mathbf{R}``. The function draws ``\\boldsymbol{v} = \\mathbf{R}^{\\intercal} \\boldsymbol{z}`` with ``\\boldsymbol{z}`` standard normal, one draw at a time, and reads the statistic off ``v_{1} / n`` and the signed sum of squares of the other sums. [`QuadratureSumDraws`](@ref) states the rule.

# Arguments

  - `M`: The number of draws.
  - `d`: The centred series, with a zero at each missing observation.
  - `f`: Whether each entry of `d` is a finite observation.
  - `n`: The number of finite entries.
  - `nd`: The divisor of the long-run variance, `n - ddof`.
  - `Phi`: The features of every entry of the span.
  - `kp`: The number of leading columns of `Phi` whose squares add. The squares of the other columns subtract.
  - `rng`: The generator.

# Returns

  - `ats::Vector{<:Real}`: The `M` absolute t-statistics. A statistic that is not finite is `Inf`.

# Related

  - [`wild_bootstrap_statistics`](@ref)
  - [`sum_draw_factor`](@ref)
"""
function sum_draw_statistics(M::Integer, d::VecNum, f::AbstractVector{Bool}, n::Integer,
                             nd::Integer, Phi::MatNum, kp::Integer, rng::Random.AbstractRNG)
    R = sum_draw_factor(d, f, n, Phi)
    m, K = size(R)
    z = Vector{eltype(R)}(undef, m)
    v = Vector{eltype(R)}(undef, K)
    ats = Vector{typeof(sqrt(one(eltype(R))))}(undef, M)
    for k in 1:M
        LinearAlgebra.mul!(v, transpose(R), Random.randn!(rng, z))
        Q = sum(j -> j <= kp + 1 ? v[j]^2 : -v[j]^2, 2:K; init = zero(eltype(v)))
        ats[k] = finite_abs(studentised_mean(v[1] / n, Q, n, nd))
    end
    return ats
end
"""
    sum_draw_factor(d::VecNum, f::AbstractVector{Bool}, n::Integer,
                    Phi::MatNum) -> Matrix{<:Real}

Return the triangular factor of the covariance of the Gaussian sums of a wild bootstrap draw.

# Arguments

  - `d`: The centred series, with a zero at each missing observation.
  - `f`: Whether each entry of `d` is a finite observation.
  - `n`: The number of finite entries.
  - `Phi`: The features of every entry of the span.

# Returns

  - `R::Matrix{<:Real}`: The factor ``\\mathbf{R}`` of the QR factorisation of ``\\mathbf{H}``, of `min(n, K + 1)` rows, with ``K`` the number of features. [`sum_draw_statistics`](@ref) states ``\\mathbf{H}``.

# Related

  - [`sum_draw_statistics`](@ref)
"""
function sum_draw_factor(d::VecNum, f::AbstractVector{Bool}, n::Integer, Phi::MatNum)
    P, K = size(Phi)
    Th = promote_type(eltype(d), eltype(Phi))
    mphi = zeros(Th, K)
    for k in 1:K, i in 1:P
        if f[i]
            mphi[k] += Phi[i, k]
        end
    end
    mphi ./= n
    H = Matrix{Th}(undef, n, K + 1)
    r = 0
    for i in 1:P
        if f[i]
            r += 1
            H[r, 1] = d[i]
            for k in 1:K
                H[r, k + 1] = d[i] * (Phi[i, k] - mphi[k])
            end
        end
    end
    return Matrix(LinearAlgebra.qr!(H).R)
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
    m = constant_series_mean(c, i1, i2, n > 0 ? s / n : Tm(NaN))
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
    constant_series_mean(c::VecNum, i1::Integer, i2::Integer, m::Number) -> Number

Return the mean of a series, the common value when every finite entry has one value.

The sum of a constant series over its count can miss its value by round-off, as `sum(fill(0.03, 100)) / 100` does. The centred series would then hold a residue, and the long-run variance of the residue is positive. The check is an equality, so it holds for every number type.

# Arguments

  - `c`: The series.
  - `i1`, `i2`: The positions of its first and its last finite entry, `0` when no entry is finite.
  - `m`: The mean of the finite entries, read as their sum over their count.

# Returns

  - `m::Number`: The value `c[i1]` in the type of `m` when every finite entry equals it, and `m` otherwise.

# Related

  - [`mean_test_centred`](@ref)
"""
function constant_series_mean(c::VecNum, i1::Integer, i2::Integer, m::Number)
    if iszero(i1) || any(v -> isfinite(v) && v != c[i1], view(c, i1:i2))
        return m
    end
    return oftype(m, c[i1])
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
       FiniteEntrySpan, FullGridSpan, ObservationDraws, QuadratureSumDraws, EigenSumDraws
public AbstractMeanTestEstimator, AbstractLongRunKernel, AbstractMeanTestSpan,
       AbstractBootstrapDraws, mean_test, mean_test_rng, long_run_kernel_weight,
       mean_test_span_length, wild_bootstrap_statistics, assert_bootstrap_draws
