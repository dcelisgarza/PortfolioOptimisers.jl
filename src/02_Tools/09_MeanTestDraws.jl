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

Abstract supertype for the rules that compute the weighted sum of the observed series under [`QuadratureSumDraws`](@ref).

The observed t-statistic reads ``Q = \\boldsymbol{d}^{\\intercal} \\mathbf{W} \\boldsymbol{d}``, the sum of the products of every pair of entries of the centred series weighted by the kernel. The quadratic spectral kernel weights every lag, so the exact sum costs ``O(P^{2})``: 1.2 s at 50,000 entries. Through the features of the draws, ``\\mathbf{W} \\approx \\mathbf{\\Phi} \\mathbf{\\Phi}^{\\intercal}``, the sum is ``\\lVert \\mathbf{\\Phi}^{\\intercal} \\boldsymbol{d} \\rVert^{2}`` and costs ``O(P K)``: 0.8 ms at 50,000 entries.

# Interfaces

In order to implement a new rule that works seamlessly with the library, subtype `AbstractLongRunForm` and implement the following method:

## Form method

  - `observed_long_run_form(form::AbstractLongRunForm, kernel::AbstractLongRunKernel, d::VecNum, B::Real, buf::VecNum, Phi::MatNum) -> Real`: The weighted sum ``Q`` of the observed series.

### Arguments

  - `form`: Concrete subtype instance.
  - `kernel`: The kernel of the test.
  - `d`: The centred series, with a zero at each missing observation.
  - `B`: The bandwidth, in entries.
  - `buf`: Work buffer of length `length(d) + 1`.
  - `Phi`: The quadrature features of every entry of the span, from `quadrature_kernel_features`.

### Returns

  - `Q::Real`: The weighted sum.

# Related

  - [`ExactLongRunForm`](@ref)
  - [`FeatureLongRunForm`](@ref)
  - [`QuadratureSumDraws`](@ref)
  - [`observed_long_run_form`](@ref)
"""
abstract type AbstractLongRunForm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes the weighted sum of the observed series at the exact kernel weights, by [`long_run_quadratic_form`](@ref).

It sums every lag of the series, so under the quadratic spectral kernel its cost is quadratic in the length of the series. The observed statistic then reads the exact weights, and the draws read the weights of the quadrature to its tolerance.

# Constructors

    ExactLongRunForm() -> ExactLongRunForm

# Examples

```jldoctest
julia> QuadratureSumDraws(; form = ExactLongRunForm()).form
ExactLongRunForm()
```

# Related

  - [`AbstractLongRunForm`](@ref)
  - [`FeatureLongRunForm`](@ref)
  - [`QuadratureSumDraws`](@ref)
"""
struct ExactLongRunForm <: AbstractLongRunForm end
"""
$(DocStringExtensions.TYPEDEF)

Computes the weighted sum of the observed series through the quadrature features of the draws, as ``\\lVert \\mathbf{\\Phi}^{\\intercal} \\boldsymbol{d} \\rVert^{2}``.

Its cost is linear in the length of the series, and the observed statistic reads the same kernel as the draws. It differs from the exact sum by the error of the quadrature: `1.3e-14` relative at 50,000 entries at the default tolerance. It is the default of [`QuadratureSumDraws`](@ref).

# Constructors

    FeatureLongRunForm() -> FeatureLongRunForm

# Examples

```jldoctest
julia> QuadratureSumDraws().form
FeatureLongRunForm()
```

# Related

  - [`AbstractLongRunForm`](@ref)
  - [`ExactLongRunForm`](@ref)
  - [`QuadratureSumDraws`](@ref)
"""
struct FeatureLongRunForm <: AbstractLongRunForm end
"""
$(DocStringExtensions.TYPEDEF)

Draws the statistic of each draw from a few Gaussian sums, through quadrature features of the [`QuadraticSpectralKernel`](@ref).

The quadratic spectral kernel is the cosine transform of a weight on a bounded band, so a Gauss-Legendre rule of ``2N`` nodes on the band writes the matrix of its weights as ``\\mathbf{\\Phi} \\mathbf{\\Phi}^{\\intercal}``, with ``2N`` features per entry. The statistic of a draw reads the multipliers only through the sum of the product and its sums against the centred features. These ``2N + 1`` sums are jointly Gaussian, with the covariance ``\\mathbf{H}^{\\intercal} \\mathbf{H}``, so the test draws them directly. A draw then costs ``O(N^{2})``, whatever the length of the series.

The observed statistic reads the weighted sum of the observed series by the rule `form`. The default [`FeatureLongRunForm`](@ref) reads it through the same features, so the whole test costs ``O(P K)`` once and ``O(K^{2})`` per draw, with ``K = 2N``. [`ExactLongRunForm`](@ref) reads the exact sum over every lag, at ``O(P^{2})``.

The largest angle that the test reads, ``c``, sets the number of nodes. Under [`FiniteEntrySpan`](@ref), ``c < 6 \\pi / (5 b)``, so the count depends on the bandwidth fraction `b` and not on the length of the series. The count is the smallest ``N`` whose bound on the error of the quadrature, at every lag, is at most `tol`. The bound is that of Gauss quadrature for a function analytic in the Bernstein ellipse ``E_{\\rho}``, minimised over ``\\rho``. Measured at a tolerance of `1e-14`, it agrees with the smallest sufficient count to one node: 9 nodes at `b = 0.4`, 19 at `b = 0.1` and 114 at `b = 0.01`. A fixed count is wrong at a small `b`: 32 nodes give an error of 0.57 at `b = 0.02`.

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
  - ``\\mathbf{R}``: A square-root factor of ``\\mathbf{H}^{\\intercal} \\mathbf{H}``, from its eigenvalues above round-off and their eigenvectors.
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
        form::AbstractLongRunForm = FeatureLongRunForm(),
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
  nodes ┼ nothing
   form ┴ FeatureLongRunForm()

julia> PortfolioOptimisers.quadrature_node_count(6 * pi / (5 * 0.1), 1e-14)
19
```

# Related

  - [`AbstractBootstrapDraws`](@ref)
  - [`ObservationDraws`](@ref)
  - [`EigenSumDraws`](@ref)
  - [`AbstractLongRunForm`](@ref)
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
    """
    Rule that computes the weighted sum of the observed series, an [`AbstractLongRunForm`](@ref).
    """
    form
    function QuadratureSumDraws(tol::Option{<:Real}, nodes::Option{<:Integer},
                                form::AbstractLongRunForm)
        if !isnothing(tol)
            assert_unit_interval(tol, :tol)
        end
        if !isnothing(nodes)
            @argcheck(nodes > zero(nodes), DomainError(nodes, "nodes must be > 0"))
        end
        return new{typeof(tol), typeof(nodes), typeof(form)}(tol, nodes, form)
    end
end
function QuadratureSumDraws(; tol::Option{<:Real} = nothing,
                            nodes::Option{<:Integer} = nothing,
                            form::AbstractLongRunForm = FeatureLongRunForm())::QuadratureSumDraws
    return QuadratureSumDraws(tol, nodes, form)
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
    wild_bootstrap_statistics(kernel::AbstractLongRunKernel, alg::AbstractBootstrapDraws,
                              M::Integer, d::VecNum, f::AbstractVector{Bool}, n::Integer,
                              nd::Integer, B::Real, buf::VecNum,
                              rng::Random.AbstractRNG) -> Tuple
    wild_bootstrap_statistics(kernel::QuadraticSpectralKernel, alg::QuadratureSumDraws,
                              M::Integer, d::VecNum, f::AbstractVector{Bool}, n::Integer,
                              nd::Integer, B::Real, buf::VecNum,
                              rng::Random.AbstractRNG) -> Tuple

Return the weighted sum of the observed series and the absolute t-statistics of `M` wild bootstrap draws of it.

The generic method reads the exact sum by [`long_run_quadratic_form`](@ref), and the draws by [`draw_statistics`](@ref). The quadrature method computes the features of [`quadrature_kernel_features`](@ref) once, reads the sum by [`observed_long_run_form`](@ref) under `alg.form`, and the draws by [`sum_draw_statistics`](@ref). A sum that is not positive gives no statistic, so neither method draws for it.

# Arguments

  - `kernel`: The kernel.
  - `alg`: The draw algorithm.
  - `M`: The number of draws.
  - `d`: The centred series, with a zero at each missing observation. It is not changed.
  - `f`: Whether each entry of `d` is a finite observation.
  - `n`: The number of finite entries.
  - `nd`: The divisor of the long-run variance, `n - ddof`.
  - `B`: The bandwidth, in entries.
  - `buf`: Work buffer of length `length(d) + 1`.
  - `rng`: The generator.

# Returns

  - `Q::Real`: The weighted sum of the observed series.
  - `ats::Vector{<:Real}`: The `M` absolute t-statistics, empty when `Q` is not positive. A statistic that is not finite is `Inf`.

# Related

  - [`AbstractBootstrapDraws`](@ref)
  - [`WildBootstrapMeanTest`](@ref)
  - [`long_run_quadratic_form`](@ref)
"""
function wild_bootstrap_statistics(kernel::AbstractLongRunKernel,
                                   alg::AbstractBootstrapDraws, M::Integer, d::VecNum,
                                   f::AbstractVector{Bool}, n::Integer, nd::Integer,
                                   B::Real, buf::VecNum, rng::Random.AbstractRNG)
    Q = long_run_quadratic_form(kernel, d, B, buf)
    if !(Q > zero(Q))
        return Q, Vector{typeof(sqrt(one(Q)))}(undef, 0)
    end
    return Q, draw_statistics(kernel, alg, M, d, f, n, nd, B, buf, rng)
end
function wild_bootstrap_statistics(kernel::QuadraticSpectralKernel, alg::QuadratureSumDraws,
                                   M::Integer, d::VecNum, f::AbstractVector{Bool},
                                   n::Integer, nd::Integer, B::Real, buf::VecNum,
                                   rng::Random.AbstractRNG)
    Phi = quadrature_kernel_features(kernel, alg, length(d), B)
    Q = observed_long_run_form(alg.form, kernel, d, B, buf, Phi)
    if !(Q > zero(Q))
        return Q, Vector{typeof(sqrt(one(Q)))}(undef, 0)
    end
    return Q, sum_draw_statistics(M, d, f, n, nd, Phi, size(Phi, 2), rng)
end
"""
    observed_long_run_form(form::ExactLongRunForm, kernel::AbstractLongRunKernel,
                           d::VecNum, B::Real, buf::VecNum, Phi::MatNum) -> Real
    observed_long_run_form(form::FeatureLongRunForm, kernel::AbstractLongRunKernel,
                           d::VecNum, B::Real, buf::VecNum, Phi::MatNum) -> Real

Return the weighted sum of the observed series under a rule of [`AbstractLongRunForm`](@ref).

# Arguments

  - `form`: The rule.
  - `kernel`: The kernel.
  - `d`: The centred series, with a zero at each missing observation.
  - `B`: The bandwidth, in entries.
  - `buf`: Work buffer of length `length(d) + 1`.
  - `Phi`: The quadrature features of every entry of the span.

# Returns

  - `Q::Real`: [`long_run_quadratic_form`](@ref) under [`ExactLongRunForm`](@ref), and ``\\lVert \\mathbf{\\Phi}^{\\intercal} \\boldsymbol{d} \\rVert^{2}`` under [`FeatureLongRunForm`](@ref). A missing entry is zero in `d`, so its row of `Phi` adds nothing.

# Related

  - [`AbstractLongRunForm`](@ref)
  - [`wild_bootstrap_statistics`](@ref)
"""
function observed_long_run_form(::ExactLongRunForm, kernel::AbstractLongRunKernel,
                                d::VecNum, B::Real, buf::VecNum, ::MatNum)
    return long_run_quadratic_form(kernel, d, B, buf)
end
function observed_long_run_form(::FeatureLongRunForm, ::AbstractLongRunKernel, d::VecNum,
                                ::Real, ::VecNum, Phi::MatNum)
    return sum(abs2, transpose(Phi) * d)
end
"""
    draw_statistics(kernel::BartlettKernel, alg::ObservationDraws, M::Integer, d::VecNum,
                    f::AbstractVector{Bool}, n::Integer, nd::Integer, B::Real, buf::VecNum,
                    rng::Random.AbstractRNG) -> Vector{<:Real}
    draw_statistics(kernel::AbstractLongRunKernel, alg::ObservationDraws, M::Integer,
                    d::VecNum, f::AbstractVector{Bool}, n::Integer, nd::Integer, B::Real,
                    buf::VecNum, rng::Random.AbstractRNG) -> Vector{<:Real}
    draw_statistics(kernel::AbstractLongRunKernel, alg::EigenSumDraws, M::Integer,
                    d::VecNum, f::AbstractVector{Bool}, n::Integer, nd::Integer, B::Real,
                    buf::VecNum, rng::Random.AbstractRNG) -> Vector{<:Real}

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

Under [`EigenSumDraws`](@ref) the method writes ``\\mathbf{W}`` as a product of features by [`eigen_kernel_features`](@ref), and draws the Gaussian sums of a draw directly, by [`sum_draw_statistics`](@ref).

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

  - [`wild_bootstrap_statistics`](@ref)
"""
function draw_statistics(kernel::BartlettKernel, ::ObservationDraws, M::Integer, d::VecNum,
                         f::AbstractVector{Bool}, n::Integer, nd::Integer, B::Real,
                         buf::VecNum, rng::Random.AbstractRNG)
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
function draw_statistics(kernel::AbstractLongRunKernel, ::ObservationDraws, M::Integer,
                         d::VecNum, f::AbstractVector{Bool}, n::Integer, nd::Integer,
                         B::Real, ::VecNum, rng::Random.AbstractRNG)
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
function draw_statistics(kernel::AbstractLongRunKernel, ::EigenSumDraws, M::Integer,
                         d::VecNum, f::AbstractVector{Bool}, n::Integer, nd::Integer,
                         B::Real, ::VecNum, rng::Random.AbstractRNG)
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

Return a square-root factor of the covariance of the Gaussian sums of a wild bootstrap draw.

# Arguments

  - `d`: The centred series, with a zero at each missing observation.
  - `f`: Whether each entry of `d` is a finite observation.
  - `n`: The number of finite entries.
  - `Phi`: The features of every entry of the span.

# Returns

  - `R::Matrix{<:Real}`: The factor ``\\mathbf{R}`` with ``\\mathbf{R}^{\\intercal} \\mathbf{R} = \\mathbf{H}^{\\intercal} \\mathbf{H}``, one row for each eigenvalue of ``\\mathbf{H}^{\\intercal} \\mathbf{H}`` above ``(K + 1) \\epsilon`` times the largest, with ``K`` the number of features. The row of an eigenvalue ``\\lambda`` with the unit eigenvector ``\\boldsymbol{e}`` is ``\\sqrt{\\lambda} \\boldsymbol{e}^{\\intercal}``. [`sum_draw_statistics`](@ref) states ``\\mathbf{H}``.

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
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(transpose(H) * H))
    lam = E.values
    keep = findall(>((K + 1) * eps(eltype(lam)) * maximum(abs, lam)), lam)
    R = Matrix{eltype(lam)}(undef, length(keep), K + 1)
    for (j, l) in enumerate(keep), k in 1:(K + 1)
        R[j, k] = sqrt(lam[l]) * E.vectors[k, l]
    end
    return R
end

export ObservationDraws, QuadratureSumDraws, EigenSumDraws, ExactLongRunForm,
       FeatureLongRunForm
public AbstractLongRunForm, wild_bootstrap_statistics, assert_bootstrap_draws,
       observed_long_run_form
