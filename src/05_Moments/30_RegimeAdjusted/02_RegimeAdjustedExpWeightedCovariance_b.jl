"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the regime statistic of one observation for a target, from the covariance block that
stands before it.

A target that has no method of its own reads its [`regime_statistic`](@ref) of the block that
[`regime_covariance_block`](@ref) returns, and takes no bias correction.

# Arguments

  - `target::RegimeAdjustedTarget`: Regime-adjustment target.
  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Current centred returns vector of every asset.
  - `idx::AbstractVector{<:Integer}`: Index of the assets that contribute to the statistic.

# Returns

  - `stats::Option{<:VecNum}`: One statistic per calibration direction, or `nothing` where the
    observation takes no regime update.

# Related

  - [`update_regime!`](@ref)
  - [`regime_statistic`](@ref)
  - [`RegimeAdjustedTarget`](@ref)
"""
function regime_target_statistic(target::RegimeAdjustedTarget,
                                 cache::RegimeAdjustedCovarianceState,
                                 ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                                 idx::AbstractVector{<:Integer})
    return regime_statistic(target, X[idx], regime_covariance_block(cache, ce, idx), idx,
                            ce.min_val)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the regime statistic of the diagonal target, with each term divided by the bias of its
own estimated variance, and the sum divided by the factor of its law.

Each term ``u_{i}^{2} / \\hat{C}_{ii}`` reads one estimated variance of ``K_{i}`` observations,
so its mean is ``\\mathbb{E}[Q^{-1}]`` at ``K_{i}``, from [`regime_bias_table`](@ref), and not
one. Each term is divided by that factor whatever the method, so the sum has the mean ``n`` at
every correlation of the assets.

The mean of the sum is ``n`` at every correlation, but its root and its log are not: they read
the law of the sum, which the correlation of the assets and the noise of each estimate set. So the
sum is then divided by [`diagonal_law_factor`](@ref) of the method, which reads the moments of each
term from the table of [`RegimeTermMoments`](@ref).

# Arguments

  - `::DiagonalTarget`: Diagonal regime-adjustment target.
  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Current centred returns vector of every asset.
  - `idx::AbstractVector{<:Integer}`: Index of the assets that contribute to the statistic.

# Returns

  - `stats::VecNum`: One statistic.

# Related

  - [`DiagonalTarget`](@ref)
  - [`regime_bias!`](@ref)
  - [`RegimeTermMoments`](@ref)
  - [`diagonal_law_factor`](@ref)
  - [`update_regime!`](@ref)
"""
function regime_target_statistic(::DiagonalTarget, cache::RegimeAdjustedCovarianceState,
                                 ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                                 idx::AbstractVector{<:Integer})
    m = regime_bias!.(Ref(cache.bias), Ref(RegimeTermMoments(ce.regime_method)), ce.decay,
                      view(cache.obs_count, idx), ce.hac_lags)
    C = regime_covariance_block(cache, ce, idx)
    return regime_statistic(DiagonalTarget(), X[idx] ./ sqrt.(first.(m)), C, idx,
                            ce.min_val) ./
           diagonal_law_factor(ce.regime_method, cache, ce, C, idx, m)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns one, the law factor of a regime method that reads the mean of the diagonal statistic.

The per-term factor of [`regime_target_statistic`](@ref) makes the mean of the sum ``n`` at every
correlation, which is the constant that [`RootMeanSquaredAdjusted`](@ref) divides by. A regime
method that the library does not define takes no law factor either.

# Arguments

  - `::RegimeAdjustedMethod`: Regime adjustment method.
  - `::RegimeAdjustedCovarianceState`: Online covariance computation cache (unused).
  - `::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration (unused).
  - `C::MatNum`: Bias-corrected covariance block of the contributing assets.
  - `::AbstractVector{<:Integer}`: Index of those assets (unused).
  - `::AbstractVector`: Moments of each term (unused).

# Returns

  - `factor::Number`: `one(eltype(C))`.

# Related

  - [`regime_target_statistic`](@ref)
  - [`regime_law_factor`](@ref)
"""
function diagonal_law_factor(::RegimeAdjustedMethod, ::RegimeAdjustedCovarianceState,
                             ::RegimeAdjustedExpWeightedCovariance, C::MatNum,
                             ::AbstractVector{<:Integer}, ::AbstractVector)
    return one(eltype(C))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the factor by which the diagonal statistic is divided, so that its root or its log has
the expectation that the method's constant assumes.

Each term of the diagonal statistic ``S = \\sum_{i} a_{i} z_{i}^{2}`` is the square of a return
with the correlation ``R``, times the noise ``a_{i}`` of its estimated variance. Without the noise,
the law of the sum is ``\\sum_{k} \\mu_{k} \\chi^{2}_{k}(1)`` on the eigenvalues ``\\mu_{k}`` of
``R``, which the constants of [`regime_denom`](@ref) and [`regime_kappa`](@ref) do not read. At 12
assets and a half-life of 10 the squared multiplier on iid Normal returns is 0.93 for the first
moment and 0.94 for the log at a correlation of 0.3, and 0.72 and 0.54 at 0.9.

The noise has the mean one, but the root and the log see it through their own moment, and the
correlation of the assets correlates the noise of their estimates, so the mean's factor of each
term leaves 0.986 and 0.971 at a correlation of 0.9. The method reads the noise in its own
variable ``g_{i}``, ``a_{i}^{1/2}`` for the first moment and ``\\ln a_{i}`` for the log, in which
the statistic is linear along the direction where every term carries the same noise. The factor is
the expectation at the mean of each ``g_{i}``, which is the exact law of
``D \\tilde{R} D``, with ``D = \\mathrm{diag}(b_{i}^{1/2})``, plus the second-order term in the
deviations of ``g``, whose covariance is ``(v_{i} v_{j})^{1/2} \\tilde{r}_{ij}^{2}`` (the
covariance of two estimated variances is ``r_{ij}^{2}`` times their variance). The second-order
term vanishes along that direction, so the factor is exact at one asset and at a correlation of
one, and its error elsewhere is of the third order in the noise: on simulated iid Normal returns it
is within 0.05 % at a half-life of 5, where the mean's factor leaves 3.4 % and 6.6 %.

The true ``R`` is not known, and the eigenvalues of its estimate ``\\hat{R}`` are too dispersed:
the law of ``\\hat{R}`` over-corrects by up to 1.3 % and 2.7 %. So
[`diagonal_law_correlation`](@ref) shrinks ``\\hat{R}`` towards the identity until its dispersion
is the unbiased one. The factor reads the correlation alone, so a scaled return leaves it
unchanged. Where `ce.debias` is `false`, the factor is one, which is the raw statistic.

# Arguments

  - `method::Union{<:FirstMomentRegimeAdjusted, <:LogRegimeAdjusted}`: Regime adjustment method.
  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `C::MatNum`: Bias-corrected covariance block of the contributing assets.
  - `idx::AbstractVector{<:Integer}`: Index of those assets.
  - `m::AbstractVector`: Moments ``(f_{i}, b_{i}, v_{i})`` of each term, from
    [`RegimeTermMoments`](@ref).

# Returns

  - `factor::Number`: The factor of the method's law, or one where `ce.debias` is `false`.

# Related

  - [`regime_target_statistic`](@ref)
  - [`diagonal_law_correlation`](@ref)
  - [`regime_law_factor`](@ref)
  - [`RegimeTermMoments`](@ref)
  - [`DiagonalTarget`](@ref)
"""
function diagonal_law_factor(method::Union{<:FirstMomentRegimeAdjusted,
                                           <:LogRegimeAdjusted},
                             cache::RegimeAdjustedCovarianceState,
                             ce::RegimeAdjustedExpWeightedCovariance, C::MatNum,
                             idx::AbstractVector{<:Integer}, m::AbstractVector)
    if !ce.debias
        return one(eltype(C))
    end

    R = diagonal_law_correlation(cache, ce, C, idx)
    sb = sqrt.(getindex.(m, 2))
    sv = sqrt.(getindex.(m, 3))
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(sb .* R .* transpose(sb)))
    return regime_law_factor(method, max.(E.values, zero(eltype(E.values))), E.vectors,
                             sv .* R .^ 2 .* transpose(sv))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the correlation of the contributing assets, shrunk towards the identity until the
dispersion of its eigenvalues is unbiased.

A sample correlation ``\\hat{r}`` of ``K`` effective observations has the variance
``(1 - r^{2})^{2} / K``, so ``\\hat{r}^{2}`` is too large by that amount on average, and the
eigenvalues of ``\\hat{R}`` are too dispersed. The dispersion of the eigenvalues is
``\\sum_{k} (\\mu_{k} - 1)^{2} = \\sum_{i \\neq j} r_{ij}^{2}``. The shrunk correlation
``\\tilde{R} = I + \\alpha (\\hat{R} - I)`` is a linear shrinkage towards the identity, with the
intensity at which the dispersion of ``\\tilde{R}`` is the unbiased estimate of the dispersion of
``R``. It is exact at the identity and at a correlation of one, where ``\\hat{r}^{2}`` has no
variance.

# Mathematical definition

```math
\\begin{align}
\\frac{1}{K_{ij}} &= \\frac{1 - \\lambda}{1 + \\lambda} \\cdot \\frac{2 - W_{ij}}{W_{ij}}\\,, \\\\
\\alpha^{2} &= \\frac{\\sum_{i \\neq j} \\left(\\hat{r}_{ij}^{2} - (1 - \\hat{r}_{ij}^{2})^{2} / K_{ij}\\right)}
{\\sum_{i \\neq j} \\hat{r}_{ij}^{2}}\\,, \\\\
\\tilde{R} &= I + \\alpha (\\hat{R} - I)\\,.
\\end{align}
```

Where:

  - ``K_{ij}``: Effective count of the observations of the pair, the inverse of the sum of its
    squared normalised weights, from [`pair_weight_square_sum`](@ref). A HAC estimate has the
    law of a plain one on the eigenvalues of its weight matrix, so its count is the inverse of
    ``\\operatorname{tr}(A^{2})``, about half as large at two lags.
  - ``\\lambda``: Decay of the correlation, `cor_decay` on the separate path and `decay` else.
  - ``W_{ij}``: Weight that the pair holds, ``1 - \\lambda^{k}`` after ``k`` joint observations.
  - ``\\hat{r}_{ij}``: Entry of the estimated correlation.
  - ``\\alpha``: Shrinkage intensity, clamped to ``[0, 1]``.

A pair that holds no weight has no estimate, and is left out of both sums.

# Arguments

  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `C::MatNum`: Bias-corrected covariance block of the contributing assets.
  - `idx::AbstractVector{<:Integer}`: Index of those assets.

# Returns

  - `R::MatNum`: The shrunk correlation ``\\tilde{R}``.

# Related

  - [`diagonal_law_factor`](@ref)
  - [`regime_law_factor`](@ref)
  - [`pair_weighted_block`](@ref)
"""
function diagonal_law_correlation(cache::RegimeAdjustedCovarianceState,
                                  ce::RegimeAdjustedExpWeightedCovariance, C::MatNum,
                                  idx::AbstractVector{<:Integer})
    T = eltype(C)
    d = sqrt.(max.(LinearAlgebra.diag(C), ce.min_val))
    rho = clamp.(C ./ (d .* transpose(d)), -one(T), one(T))
    lambda, W = if has_separate_cor_decay(ce)
        ce.cor_decay, view(cache.cor_weight, idx, idx)
    else
        ce.decay, view(cache.weight, idx, idx)
    end
    pairs = Iterators.filter(p -> p[1] != p[2] && W[p] > 0, CartesianIndices(rho))
    q = sum(p -> rho[p]^2, pairs; init = zero(T))
    q_unbiased = sum(p -> rho[p]^2 -
                          (one(T) - rho[p]^2)^2 *
                          pair_weight_square_sum(lambda, W[p], ce.hac_lags), pairs;
                     init = zero(T))
    alpha = sqrt(clamp(q_unbiased / max(q, eps(T)), zero(T), one(T)))
    R = alpha .* rho
    view(R, LinearAlgebra.diagind(R)) .+= one(T) - alpha

    return R
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the sum of the squared normalised weights of a pair's estimate from the weight the pair
holds, the variance of a sample correlation of the pair relative to ``(1 - r^{2})^{2}``.

A pair of ``k`` joint observations holds ``W = 1 - \\lambda^{k}``, so the sum is a function of
``W`` alone. With HAC lags the weights are the eigenvalues of the banded weight matrix ``A`` of
[`regime_bias_table`](@ref), and the sum is ``\\operatorname{tr}(A^{2})``, which adds the squared
Bartlett weight of each lag that reaches a joint observation.

# Mathematical definition

```math
\\begin{align}
\\operatorname{tr}(A^{2}) &= \\frac{v}{W^{2}} \\left(W (2 - W) + 2 \\sum_{i=1}^{L} k_{i}^{2} \\max\\left(1 - \\frac{(1 - W)^{2}}{\\lambda^{2 i}}, 0\\right)\\right)\\,, \\quad v = \\frac{1 - \\lambda}{1 + \\lambda}\\,.
\\end{align}
```

Where:

  - ``k_{i}``: Bartlett weight ``1 - i / (L + 1)`` of lag ``i``.
  - ``L``: `hac_lags`, and the sum is empty where it is `nothing`.

# Arguments

  - `lambda::Number`: Decay of the pair's estimate.
  - `W::Number`: Weight that the pair holds.
  - `hac_lags::Option{<:Integer}`: Count of HAC lags, or `nothing`.

# Returns

  - `s::Number`: The sum of the squared normalised weights.

# Related

  - [`diagonal_law_correlation`](@ref)
  - [`exp_weight_cross_sum`](@ref)
"""
function pair_weight_square_sum(lambda::Number, W::Number, hac_lags::Option{<:Integer})
    v = (one(lambda) - lambda) / (one(lambda) + lambda)
    t = v * (2 - W) / W
    for i in 1:something(hac_lags, 0)
        t += 2 *
             (one(lambda) - i / (hac_lags + 1))^2 *
             v *
             max(one(W) - (one(W) - W)^2 / lambda^(2 * i), zero(W)) / W^2
    end
    return t
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the log of the Laplace transform of a weighted sum of gamma variates, divided by its
count.

The law of ``S / n = \\sum_{k} \\mu_{k} G_{k} / n`` with ``G_{k}`` independent
``\\mathrm{Gamma}(a, y)`` variates has the Laplace transform
``\\prod_{k} (1 + y \\mu_{k} t / n)^{-a}``. At ``a = 1/2`` and ``y = 2`` each ``G_{k}`` is a
``\\chi^{2}(1)`` variate.

# Arguments

  - `mu::VecNum`: Weights of the sum, the eigenvalues of the law.
  - `a::Number`: Shape of each gamma variate.
  - `y::Number`: Scale of each gamma variate.
  - `t::Number`: Argument of the transform.

# Returns

  - `lL::Number`: ``-a \\sum_{k} \\ln(1 + y \\mu_{k} t / n)``.

# Related

  - [`regime_law_factor`](@ref)
  - [`regime_law_correction`](@ref)
"""
function regime_law_log_laplace(mu::VecNum, a::Number, y::Number, t::Number)
    n = length(mu)
    return -a * sum(m -> log1p(y * m * t / n), mu)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the grid of the integration variable ``x = \\ln t`` on which [`regime_law_factor`](@ref)
and [`regime_law_correction`](@ref) take the trapezoid rule.

Each integrand is smooth and analytic in a strip about the real line, so the trapezoid rule
converges faster than any power of the step. The transform of a wide block grows fast inside the
strip, so the step must be small: at a step of ``1/4`` the factors agree with their closed forms at
``\\mu_{k} = 1`` and at one eigenvalue ``n`` to ``10^{-14}``, from ``n = 1`` to ``20\\,000``,
whereas a step of ``1/2`` is ``3 \\times 10^{-9}`` wrong at ``n = 200``. The slowest integrand
decays as ``e^{-\\lvert x \\rvert / 2}`` at both ends, so the grid stops at
``\\lvert x \\rvert = 75``.

# Arguments

  - `T::Type`: Number type of the grid.

# Returns

  - `x::AbstractRange`: The grid, of step ``1/4`` on ``[-75, 75]``.

# Related

  - [`regime_law_factor`](@ref)
  - [`regime_law_correction`](@ref)
"""
function regime_law_grid(T::Type)
    return range(-75 * one(T), 75 * one(T); step = one(T) / 4)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the factor of the first-moment method, the squared ratio of the expected root of a sum of
``\\chi^{2}(1)`` variates to the constant [`regime_denom`](@ref).

# Mathematical definition

```math
\\mathbb{E}[\\sqrt{S}] = \\frac{\\sqrt{n}}{2 \\sqrt{\\pi}} \\int_{0}^{\\infty}
\\left(1 - L(t)\\right) t^{-3/2}\\, \\mathrm{d}t\\,, \\quad
L(t) = \\prod_{k} \\left(1 + 2 \\mu_{k} t / n\\right)^{-1/2}\\,.
```

The integral is taken on ``t = e^{x}`` over [`regime_law_grid`](@ref). At ``\\mu_{k} = 1`` it
is the root of a ``\\chi^{2}(n)`` variate, and at one eigenvalue ``n`` it is
``\\sqrt{2 n / \\pi}``.

# Arguments

  - `method::FirstMomentRegimeAdjusted`: First-moment regime adjustment method.
  - `mu::VecNum`: Eigenvalues of the law, the weight of each ``\\chi^{2}(1)`` variate.

# Returns

  - `factor::Number`: ``(\\mathbb{E}[\\sqrt{S}] / d_{n})^{2}``, with ``d_{n}`` from
    [`regime_denom`](@ref).

# Related

  - [`diagonal_law_factor`](@ref)
  - [`regime_law_log_laplace`](@ref)
  - [`FirstMomentRegimeAdjusted`](@ref)
"""
function regime_law_factor(method::FirstMomentRegimeAdjusted, mu::VecNum)
    T = eltype(mu)
    x = regime_law_grid(T)
    n = length(mu)
    half = one(T) / 2
    integral = sum(x) do xi
        t = exp(xi)
        return -expm1(regime_law_log_laplace(mu, half, 2 * one(T), t)) / sqrt(t)
    end
    root = sqrt(n * one(T)) * step(x) * integral / (2 * sqrt(pi * one(T)))

    return (root / regime_denom(method, DiagonalTarget(), n))^2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the factor of the log method, the exponential of the difference between the expected log
of a sum of gamma variates and the constant [`regime_kappa`](@ref).

The method takes each square as a ``\\mathrm{Gamma}(x, y)`` variate, which is the law that its
constant assumes, so the factor is one at ``\\mu_{k} = 1`` whatever the parameters.

# Mathematical definition

```math
\\mathbb{E}[\\ln S] = \\ln n + \\int_{0}^{\\infty} \\left(\\frac{1}{1 + t} - L(t)\\right)
t^{-1}\\, \\mathrm{d}t - \\gamma\\,, \\quad
L(t) = \\prod_{k} \\left(1 + y \\mu_{k} t / n\\right)^{-x}\\,.
```

Where ``\\gamma`` is the Euler-Mascheroni constant. The integral is taken on ``t = e^{x}`` over
[`regime_law_grid`](@ref).

# Arguments

  - `method::LogRegimeAdjusted`: Log regime adjustment method.
  - `mu::VecNum`: Eigenvalues of the law, the weight of each gamma variate.

# Returns

  - `factor::Number`: ``\\exp(\\mathbb{E}[\\ln S] - \\kappa_{n})``, with ``\\kappa_{n}`` from
    [`regime_kappa`](@ref).

# Related

  - [`diagonal_law_factor`](@ref)
  - [`regime_law_log_laplace`](@ref)
  - [`LogRegimeAdjusted`](@ref)
"""
function regime_law_factor(method::LogRegimeAdjusted, mu::VecNum)
    T = eltype(mu)
    x = regime_law_grid(T)
    n = length(mu)
    integral = sum(x) do xi
        t = exp(xi)
        return inv(one(t) + t) - exp(regime_law_log_laplace(mu, method.x, method.y, t))
    end
    elog = log(n * one(T)) + step(x) * integral - Base.MathConstants.eulergamma

    return exp(elog - regime_kappa(method, DiagonalTarget(), n))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the factor of the first-moment method for a sum whose terms carry the noise of their
estimated variances.

The expected root is that of the law of ``\\mu``, from the two-argument method, plus the
second-order term of [`regime_law_correction`](@ref).

# Arguments

  - `method::FirstMomentRegimeAdjusted`: First-moment regime adjustment method.
  - `mu::VecNum`: Eigenvalues of ``D \\tilde{R} D``.
  - `V::MatNum`: Their eigenvectors.
  - `Cm::MatNum`: Covariance of the deviations of the variable of the method.

# Returns

  - `factor::Number`: ``((\\mathbb{E}[\\sqrt{S_{b}}] + \\Delta) / d_{n})^{2}``.

# Related

  - [`diagonal_law_factor`](@ref)
  - [`regime_law_correction`](@ref)
"""
function regime_law_factor(method::FirstMomentRegimeAdjusted, mu::VecNum, V::MatNum,
                           Cm::MatNum)
    d = regime_denom(method, DiagonalTarget(), length(mu))
    root = sqrt(regime_law_factor(method, mu)) * d
    return ((root + regime_law_correction(method, mu, V, Cm)) / d)^2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the factor of the log method for a sum whose terms carry the noise of their estimated
variances.

The expected log is that of the law of ``\\mu``, from the two-argument method, plus the
second-order term of [`regime_law_correction`](@ref).

# Arguments

  - `method::LogRegimeAdjusted`: Log regime adjustment method.
  - `mu::VecNum`: Eigenvalues of ``D \\tilde{R} D``.
  - `V::MatNum`: Their eigenvectors.
  - `Cm::MatNum`: Covariance of the deviations of the variable of the method.

# Returns

  - `factor::Number`: ``\\exp(\\mathbb{E}[\\ln S_{b}] + \\Delta - \\kappa_{n})``.

# Related

  - [`diagonal_law_factor`](@ref)
  - [`regime_law_correction`](@ref)
"""
function regime_law_factor(method::LogRegimeAdjusted, mu::VecNum, V::MatNum, Cm::MatNum)
    return regime_law_factor(method, mu) * exp(regime_law_correction(method, mu, V, Cm))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the shape of each square and the degrees of freedom of its Wishart form, for the
first-moment method: a ``\\chi^{2}(1)`` variate.

# Arguments

  - `::FirstMomentRegimeAdjusted`: First-moment regime adjustment method.

# Returns

  - `(a, nu)::Tuple`: ``(1/2, 1)``.

# Related

  - [`regime_law_correction`](@ref)
"""
function regime_law_shape(::FirstMomentRegimeAdjusted)
    return (1 // 2, 1)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the shape of each square and the degrees of freedom of its Wishart form, for the log
method, which takes each square as a ``\\mathrm{Gamma}(x, y)`` variate: the diagonal of a Wishart
matrix of ``2 x`` degrees of freedom.

# Arguments

  - `method::LogRegimeAdjusted`: Log regime adjustment method.

# Returns

  - `(a, nu)::Tuple`: ``(x, 2 x)``.

# Related

  - [`regime_law_correction`](@ref)
"""
function regime_law_shape(method::LogRegimeAdjusted)
    return (method.x, 2 * method.x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the weights of the linear and the quadratic integrands of the first-moment correction at
one node, the measure ``\\mathrm{d}\\tau = \\tau\\, \\mathrm{d}x`` included.

``\\mathbb{E}[y_{i}^{2} S^{-1/2}]`` and ``\\mathbb{E}[y_{i}^{2} y_{j}^{2} S^{-3/2}]`` are
``\\Gamma(p)^{-1} \\int \\tau^{p - 1} \\mathbb{E}[\\cdot\\, e^{-\\tau S}]\\, \\mathrm{d}\\tau`` at
``p = 1/2`` and ``p = 3/2``.

# Arguments

  - `::FirstMomentRegimeAdjusted`: First-moment regime adjustment method.
  - `tau::Number`: Node of the integration variable.

# Returns

  - `(wl, wq)::Tuple`: ``(\\tau^{1/2} / \\sqrt{\\pi}, 2 \\tau^{3/2} / \\sqrt{\\pi})``.

# Related

  - [`regime_law_correction`](@ref)
"""
function regime_law_weights(::FirstMomentRegimeAdjusted, tau::Number)
    r = sqrt(tau / pi)
    return (r, 2 * tau * r)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Returns the weights of the linear and the quadratic integrands of the log correction at one node,
the measure ``\\mathrm{d}\\tau = \\tau\\, \\mathrm{d}x`` included.

``\\mathbb{E}[W_{ii} / S]`` and ``\\mathbb{E}[W_{ii} W_{jj} / S^{2}]`` are
``\\int \\tau^{p - 1} \\mathbb{E}[\\cdot\\, e^{-\\tau S}]\\, \\mathrm{d}\\tau`` at ``p = 1`` and
``p = 2``, and the linear one carries the degrees of freedom ``\\nu = 2 x`` of the tilted mean.

# Arguments

  - `method::LogRegimeAdjusted`: Log regime adjustment method.
  - `tau::Number`: Node of the integration variable.

# Returns

  - `(wl, wq)::Tuple`: ``(2 x \\tau, \\tau^{2})``.

# Related

  - [`regime_law_correction`](@ref)
"""
function regime_law_weights(method::LogRegimeAdjusted, tau::Number)
    return (2 * method.x * tau, tau^2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the second-order term of the expected root or log of a diagonal sum whose terms carry
the noise of their estimated variances.

Write each term as ``e^{g_{i}} y_{i}^{2}`` (the log method) or ``(1 + g_{i})^{2} y_{i}^{2}`` (the
first moment), with ``y \\sim \\mathcal{N}(0, D \\tilde{R} D)``, the deviation ``g`` of mean zero and
covariance ``C_{m}``, and ``g`` independent of ``y``. A Taylor expansion of the root or the log of
the sum in ``g`` gives, to the second order,

```math
\\Delta = \\frac{1}{2} \\left(\\sum_{i} C_{m, ii}\\, \\mathbb{E}[y_{i}^{2} h_{1}(S)] -
\\sum_{i, j} C_{m, ij}\\, \\mathbb{E}[y_{i}^{2} y_{j}^{2} h_{2}(S)]\\right)\\,,
```

with ``h_{1} = S^{-1/2}`` and ``h_{2} = S^{-3/2}`` for the root, and ``S^{-1}`` and ``S^{-2}``
for the log. Along the direction where every ``g_{i}`` is equal, the root and the log are linear in
``g``, so the two sums cancel there. Each expectation is an integral over ``\\tau`` of the tilted
law ``e^{-\\tau S}``, under which ``y`` has the covariance
``M(\\tau) = D \\tilde{R} D (I + 2 \\tau D \\tilde{R} D)^{-1} = V \\mathrm{diag}(d(\\tau)) V'``,
with ``d_{k} = \\mu_{k} / (1 + 2 \\tau \\mu_{k})``, and the Wishart moments
``\\mathbb{E}[W_{ii}] = \\nu M_{ii}`` and
``\\mathbb{E}[W_{ii} W_{jj}] = \\nu^{2} M_{ii} M_{jj} + 2 \\nu M_{ij}^{2}``.

# Algorithm

 1. On each node of [`regime_law_grid`](@ref), with ``\\tau = e^{x} / n``, accumulate the vector
    ``j = \\sum w_{l} L d`` and the matrix ``J = \\sum w_{q} L d d'``, with
    ``L = \\prod_{k} (1 + 2 \\tau \\mu_{k})^{-a}`` and the weights of
    [`regime_law_weights`](@ref).
 2. The linear sum is ``(W' \\mathrm{diag}(C_{m}))' j``, with ``W = V \\circ V``, and the part of
    the quadratic sum in ``M_{ii} M_{jj}`` is ``\\langle W' C_{m} W, J \\rangle``.
 3. The part in ``M_{ij}^{2}`` is ``\\sum_{kl} J_{kl} \\sum_{ij} C_{m, ij} V_{ik} V_{jk} V_{il} V_{jl}``.
    ``J`` is positive semi-definite and its eigenvalues fall fast, so with ``J = \\sum_{q} \\sigma_{q} g_{q} g_{q}'`` it is ``\\sum_{q} \\sigma_{q} \\sum_{ij} C_{m, ij} (V \\mathrm{diag}(g_{q}) V')_{ij}^{2}``
    over the ``\\sigma_{q}`` above the machine epsilon of the largest, a few products of order
    ``n^{3}`` in place of one of order ``n^{4}``. At 12 and 40 assets it keeps 2 to 8 terms and agrees
    with the sum over every node to ``10^{-15}``.

# Arguments

  - `method::Union{<:FirstMomentRegimeAdjusted, <:LogRegimeAdjusted}`: Regime adjustment method.
  - `mu::VecNum`: Eigenvalues of ``D \\tilde{R} D``.
  - `V::MatNum`: Their eigenvectors.
  - `Cm::MatNum`: Covariance of the deviations of the variable of the method.

# Returns

  - `Delta::Number`: The second-order term of ``\\mathbb{E}[\\sqrt{S}]`` or of
    ``\\mathbb{E}[\\ln S]``.

# Related

  - [`regime_law_factor`](@ref)
  - [`regime_law_shape`](@ref)
  - [`regime_law_weights`](@ref)
  - [`diagonal_law_factor`](@ref)
"""
function regime_law_correction(method::Union{<:FirstMomentRegimeAdjusted,
                                             <:LogRegimeAdjusted}, mu::VecNum, V::MatNum,
                               Cm::MatNum)
    T = eltype(mu)
    x = regime_law_grid(T)
    n = length(mu)
    a, nu = regime_law_shape(method)
    jl = zeros(T, n)
    J = zeros(T, n, n)
    d = similar(mu)
    for xi in x
        tau = exp(xi) / n
        L = exp(regime_law_log_laplace(mu, a, 2 * one(T), exp(xi)))
        d .= mu ./ (one(T) .+ 2 .* tau .* mu)
        wl, wq = regime_law_weights(method, tau)
        jl .+= (wl * L) .* d
        J .+= (wq * L) .* d .* transpose(d)
    end
    J .*= step(x)
    W = V .^ 2
    lin = step(x) * LinearAlgebra.dot(transpose(W) * LinearAlgebra.diag(Cm), jl)
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(J))
    top = maximum(E.values)
    pairs = sum(findall(>(eps(T) * top), E.values); init = zero(T)) do q
        Mq = V * LinearAlgebra.Diagonal(view(E.vectors, :, q)) * transpose(V)
        return E.values[q] * sum(i -> Cm[i] * Mq[i]^2, eachindex(Cm, Mq))
    end
    quad = nu^2 * LinearAlgebra.dot(transpose(W) * Cm * W, J) + 2 * nu * pairs

    return (lin - quad) / 2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the regime statistic of the portfolio target, divided by the bias of the estimated
variance of each direction.

A direction reads one estimated variance ``p^{\\top} \\hat{C} p``, so its statistic is biased by
the factor of [`regime_bias_table`](@ref) that the method reads, at ``K``, the smallest count of
observations among the contributing assets. That factor is exact for a fixed direction on one
shared history. On the separate correlation path the factor reads `cor_decay`, because the
correlations carry most of the variance of a direction: at 12 assets, a half-life of 10 and a
correlation half-life of 20, the squared multiplier of an equal-weight direction is 1.004 with it
and 0.971 with `decay`. A HAC estimate reads the table of its banded weight matrix, which has
about twice the excess of the plain weights: at two lags the squared multiplier of an
equal-weight direction is 1.015 over 8 seeds, within its standard error of 0.013, from 1.165
raw.

The default inverse-volatility direction (`w = nothing`) is built from the same estimate, so the
direction and the error of the estimate are correlated. Its statistic is also divided by
``1 + \\Delta``, the second-order excess of [`inverse_volatility_bias`](@ref), on the estimated
correlation of the block. At 12 assets and a half-life of 10 the squared multiplier on iid
Normal returns is 1.134 raw, 1.058 after the factor alone and 1.001 after both. The excess
is the same for the three methods to first order. What remains is of the next order, about
``9 s^{2}`` with ``s`` the sum of the squared weights, so it is largest in the warm-up rows: at
12 assets, a half-life of 10 and ``R = I``, 1.34 at ``K = 5``, 1.08 at ``K = 10``, 1.02 at
``K = 20`` and 1.004 in the steady state. At the defaults the first row enters at ``K`` equal to
the half-life, where the rest is 1.006 at a half-life of 40.

# Arguments

  - `target::PortfolioTarget`: Portfolio regime-adjustment target.
  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Current centred returns vector of every asset.
  - `idx::AbstractVector{<:Integer}`: Index of the assets that contribute to the statistic.

# Returns

  - `stats::Option{<:VecNum}`: One statistic per direction, or `nothing` where no direction keeps
    a positive weight.

# Related

  - [`PortfolioTarget`](@ref)
  - [`regime_bias!`](@ref)
  - [`inverse_volatility_bias`](@ref)
  - [`update_regime!`](@ref)
"""
function regime_target_statistic(target::PortfolioTarget,
                                 cache::RegimeAdjustedCovarianceState,
                                 ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                                 idx::AbstractVector{<:Integer})
    C = regime_covariance_block(cache, ce, idx)
    stats = regime_statistic(target, X[idx], C, idx, ce.min_val)
    if isnothing(stats)
        return nothing
    end

    K = minimum(view(cache.obs_count, idx))
    separate = has_separate_cor_decay(ce)
    f = regime_bias!(cache.bias, ce.regime_method, separate ? ce.cor_decay : ce.decay, K,
                     ce.hac_lags)
    if isnothing(target.w) && !isnothing(cache.bias)
        sv = exp_weight_cross_sum(ce.decay, ce.decay, K, ce.hac_lags)
        svc = separate ? exp_weight_cross_sum(ce.decay, ce.cor_decay, K, ce.hac_lags) : sv
        f *= one(f) + inverse_volatility_bias(C, sv, svc, ce.min_val)
    end
    return stats ./ f
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the sum of the products of two sets of normalised exponential weights over ``K``
observations.

# Mathematical definition

```math
\\sum_{j=0}^{K-1} w_{j}(a)\\, w_{j}(b) = \\frac{(1 - a)(1 - b)\\left(1 - (ab)^{K}\\right)}{(1 - ab)\\left(1 - a^{K}\\right)\\left(1 - b^{K}\\right)}\\,, \\quad w_{j}(a) = \\frac{(1 - a)\\, a^{j}}{1 - a^{K}}\\,.
```

At ``a = b`` it is the sum of the squared weights, the variance of an estimate of ``K``
observations relative to the variance of one. With HAC lags the sum is
``\\operatorname{tr}(A_{a} A_{b})`` over the banded weight matrices of
[`regime_bias_table`](@ref), whose law has the eigenvalues of ``A`` as its weights, so each lag
``i`` that reaches an observation adds ``2 k_{i}^{2} (1 - (ab)^{K - i})`` to the factor
``1 - (ab)^{K}`` of the numerator. At ``a = b`` its inverse is the effective count of
observations that [`regime_bias_open`](@ref) reads.

# Arguments

  - `a::Number`: Decay of the first set of weights.
  - `b::Number`: Decay of the second set of weights.
  - `K::Integer`: Count of observations that the weights span.
  - `hac_lags::Option{<:Integer}`: Count of HAC lags, or `nothing`.

# Returns

  - `s::Number`: The sum of the products of the weights.

# Related

  - [`inverse_volatility_bias`](@ref)
"""
function exp_weight_cross_sum(a::Number, b::Number, K::Integer,
                              hac_lags::Option{<:Integer} = nothing)
    ab = a * b
    t = one(ab) - ab^K
    for i in 1:min(something(hac_lags, 0), K - 1)
        t += 2 * (one(ab) - i / (hac_lags + 1))^2 * (one(ab) - ab^(K - i))
    end
    return (one(a) - a) * (one(b) - b) * t /
           ((one(ab) - ab) * (one(a) - a^K) * (one(b) - b^K))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the bias factor of a regime statistic that reads one HAC-adjusted variance estimate, for
every count of observations from one to `K`.

A HAC estimate of ``K`` observations is ``\\hat{v} = \\sigma^{2} Q`` with ``Q = z^{\\top} A z``, a
quadratic form in the correctly calibrated returns. ``A = c B`` is banded: the weight of each
square on its diagonal, and the Bartlett weight ``k_{i}`` times the weight of the newer return
at lag ``i``. Diagonalised, ``Q`` has the law of a plain estimate whose weights are the
eigenvalues ``\\mu`` of ``A``, so its Laplace transform is ``G(s) = \\det(I + s B)^{-1/2}`` on the
scale ``s = 2 c t``, and [`regime_bias_factor`](@ref) reads it as it reads the transform of
[`regime_bias_table`](@ref). The exponential weights break the identity that makes the Bartlett
estimate a sum of squares, so ``A`` has negative eigenvalues from a count of 5 to 50, by the
half-life, and ``Q`` can fall below zero. There the moment of ``Q^{-1}`` is not finite, and the
table reads the moment of the positive part: past the minimum of ``G`` the negative eigenvalues
rule the transform, so the integral stops there.

# Mathematical definition

```math
\\begin{align}
B_{jj} &= \\lambda^{j}\\,, \\quad B_{j, j+i} = B_{j+i, j} = \\lambda^{j} k_{i}\\,, \\quad k_{i} = 1 - \\frac{i}{L + 1}\\,, \\quad i = 1, \\ldots, L\\,.
\\end{align}
```

Where:

  - ``j``: Index of an observation, from zero for the newest.
  - ``\\lambda``: `decay`.
  - ``L``: `hac_lags`.

# Algorithm

 1. Lay the grid of [`regime_bias_table`](@ref).
 2. For each ``K``, add one row of the banded LDLᵀ factorisation of ``I + s B`` at every point of
    the grid with [`hac_ldl_row!`](@ref), so ``\\ln G`` takes the log of the new pivot and the
    whole table costs one pass. A pivot that is not positive marks the point: past it, the
    determinant has crossed zero.
 3. Cut the transform at its minimum with [`hac_laplace!`](@ref), and evaluate the factor of the
    method with [`regime_bias_factor`](@ref).

Where the estimate is positive definite, the table agrees with the eigenvalues of ``B`` on a grid
of step ``1/200`` on ``[-75, 60]`` to ``10^{-10}``, and with a Monte Carlo of 400 000 draws to its
noise. Where it is not, the grid agrees with the finer one to ``5 \\times 10^{-7}``, and to
``4 \\times 10^{-4}`` at a half-life of 5 and five lags, where the estimate is negative in
``6 \\times 10^{-5}`` of draws.

# Arguments

  - `method::Union{<:RegimeAdjustedMethod, <:RegimeTermMoments}`: Regime adjustment method,
    which names the moment, or the table of the moments of one term.
  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Largest count of observations in the table.
  - `hac_lags::Integer`: Count of HAC lags.

# Returns

  - `table::AbstractVector`: The factor for each count from one to `K`, or the moments of
    [`RegimeTermMoments`](@ref).

# Related

  - [`hac_ldl_row!`](@ref)
  - [`hac_laplace!`](@ref)
  - [`regime_bias_table`](@ref)
  - [`regime_bias!`](@ref)
"""
function regime_bias_table(method::Union{<:RegimeAdjustedMethod, <:RegimeTermMoments},
                           decay::Number, K::Integer, hac_lags::Integer)
    h = one(decay) / 10
    s = exp.(range(-60 * one(h), 50 * one(h); step = h))
    T = eltype(s)
    buf = (; l = zeros(T, length(s), hac_lags, hac_lags), d = ones(T, length(s), hac_lags),
           row = zeros(T, length(s), hac_lags), num = zeros(T, length(s)),
           lG = zeros(T, length(s)))
    G = similar(s)
    lam = one(decay)
    return map(1:K) do k
        hac_ldl_row!(buf, s, decay, lam, k - 1)
        lam *= decay
        return regime_bias_factor(method, s, hac_laplace!(G, buf.lG),
                                  (one(decay) - decay) / (one(decay) - decay^k), h)
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds the row of observation `m` to the banded LDLᵀ factorisation of ``I + s B`` at every point
of the grid, and adds the log of its pivot to ``\\ln G``.

The matrix is banded with `L` lags, so a new row reads only the last `L` rows of the factor and
their pivots. The buffers hold them, the last row first.

# Arguments

  - `buf::NamedTuple`: Buffers of the factorisation (mutated): `l[g, r, i]`, the entry of row
    ``m - r`` at column ``m - r - i``; `d[g, r]`, the pivot of row ``m - r``; `row` and `num`,
    scratch; and `lG`, ``\\ln G`` at each point, ``-\\infty`` at a point whose pivot was not
    positive.
  - `s::VecNum`: Grid of ``s``.
  - `decay::Number`: Decay of the weights.
  - `lam::Number`: ``\\lambda^{m}``.
  - `m::Integer`: Index of the new row, from zero for the newest observation.

# Returns

  - `buf::NamedTuple`: The buffers after the row.

# Related

  - [`regime_bias_table`](@ref)
"""
function hac_ldl_row!(buf::NamedTuple, s::VecNum, decay::Number, lam::Number, m::Integer)
    (; l, d, row, num, lG) = buf
    L = size(d, 2)
    p = min(L, m)
    for i in p:-1:1
        num .= s .* (lam / decay^i * (one(decay) - i / (L + 1)))
        for j in (i + 1):p
            num .-= view(row, :, j) .* view(l, :, i, j - i) .* view(d, :, j)
        end
        row[:, i] .= num ./ view(d, :, i)
    end
    num .= one(lam) .+ s .* lam
    for i in 1:p
        num .-= view(row, :, i) .^ 2 .* view(d, :, i)
    end
    # A dead point keeps -Inf whatever its pivots hold, and `max` keeps the log defined there.
    lG .= ifelse.(num .> 0, lG .- log.(max.(num, floatmin(eltype(num)))) ./ 2,
                  -convert(eltype(lG), Inf))
    for r in L:-1:2
        d[:, r] .= view(d, :, r - 1)
        l[:, r, :] .= view(l, :, r - 1, :)
    end
    d[:, 1] .= num
    l[:, 1, :] .= row

    return buf
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Forms the Laplace transform of a HAC estimate on the grid, cut at its minimum.

Where the weight matrix is positive definite, ``G`` falls on the whole grid and nothing is cut.
Where it is not, ``G`` falls to a minimum and then rises to the first zero of the determinant,
where the negative part of the estimate rules the transform. The transform is cut at the minimum,
and the last point takes the weight one half of a closed trapezoid, so the error of the cut is of
the second order in the step.

# Arguments

  - `G::VecNum`: Transform on the grid (mutated).
  - `lG::VecNum`: ``\\ln G`` on the grid, ``-\\infty`` past the first zero of the determinant.

# Returns

  - `G::VecNum`: The cut transform.

# Related

  - [`regime_bias_table`](@ref)
"""
function hac_laplace!(G::VecNum, lG::VecNum)
    i = argmin(j -> isinf(lG[j]) ? typemax(eltype(lG)) : lG[j], eachindex(lG))
    G .= exp.(lG)
    if i < lastindex(G)
        G[i] /= 2
        G[(i + 1):end] .= zero(eltype(G))
    end

    return G
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the second-order excess bias of the regime statistic of the inverse-volatility direction,
which is built from the estimate it divides by.

The direction ``p_{i} \\propto 1 / \\hat{\\sigma}_{i}`` reads the estimated volatilities, so its
statistic is ``q^{\\top} R q / 1^{\\top} \\hat{R} 1`` in units of the true volatilities, with
``q_{i} = \\sigma_{i} / \\hat{\\sigma}_{i}``. An asset whose volatility is under-estimated takes
more weight, and its error enters the numerator but not the denominator. A second-order expansion
in the error of the estimate gives the mean of the statistic as the factor of a fixed direction
times ``1 + \\Delta``. The excess is the same for the three regime methods to first order. On one
decay ``s_{vc} = s_{v}`` and ``\\Delta = 2 s_{v} \\kappa``, with
``\\kappa = 1 - \\sum_{i} c_{i}^{3} / A^{2}``: ``1 - 1/n`` at ``R = I`` and zero at a correlation
of one, where the estimated direction is proportional to the true one.

# Mathematical definition

```math
\\begin{align}
\\Delta &= s_{v} \\left(1 + \\frac{\\Sigma_{3}}{A} - \\frac{2 B}{A^{2}}\\right) + s_{vc} \\left(1 - \\frac{\\Sigma_{3}}{A} - \\frac{2 T}{A^{2}} + \\frac{2 B}{A^{2}}\\right)\\,, \\\\
c &= R 1\\,, \\quad A = 1^{\\top} c\\,, \\quad T = \\sum_{i} c_{i}^{3}\\,, \\quad B = \\sum_{ij} c_{i} R_{ij}^{2} c_{j}\\,, \\quad \\Sigma_{3} = \\sum_{ij} R_{ij}^{3}\\,.
\\end{align}
```

Where:

  - ``R``: Correlation of the block, the plug-in for the true correlation.
  - ``s_{v}``: Sum of the squared weights of the variance estimate.
  - ``s_{vc}``: Sum of the products of the weights of the variance and the correlation estimates.

The plug-in and the next order leave about ``9 s_{v}^{2}``. At 12 assets and a half-life of 10
the formula at the true ``R`` matches the simulated excess to ``5 \\times 10^{-4}``, on both paths.
A block whose ``A`` is not positive has no inverse-volatility portfolio with a variance, and takes
no excess.

# Arguments

  - `C::MatNum`: Bias-corrected covariance block of the contributing assets.
  - `sv::Number`: Sum of the squared weights of the variance estimate.
  - `svc::Number`: Sum of the products of the weights of the variance and the correlation
    estimates.
  - `min_val::Number`: Floor applied to each variance, as in [`regime_statistic`](@ref).

# Returns

  - `delta::Number`: The excess ``\\Delta``.

# Related

  - [`regime_target_statistic`](@ref)
  - [`exp_weight_cross_sum`](@ref)
  - [`PortfolioTarget`](@ref)
"""
function inverse_volatility_bias(C::MatNum, sv::Number, svc::Number, min_val::Number)
    d = inv.(sqrt.(max.(LinearAlgebra.diag(C), min_val)))
    c = (C * d) .* d
    A = sum(c)
    if !(A > zero(A))
        return zero(A) * sv
    end
    B = zero(A)
    S3 = zero(A)
    for j in axes(C, 2), i in axes(C, 1)
        r = d[i] * C[i, j] * d[j]
        B += c[i] * r^2 * c[j]
        S3 += r^3
    end
    T = sum(x -> x^3, c)
    return sv * (one(A) + S3 / A - 2 * B / A^2) +
           svc * (one(A) - S3 / A - 2 * T / A^2 + 2 * B / A^2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes the empty table of bias factors for a new state of a regime-adjusted variance estimator.

# Arguments

  - `ce::RegimeAdjustedExpWeightedVariance`: Estimator configuration.
  - `::Type{T}`: Element type of the state.

# Returns

  - `bias::Option{<:VecNum}`: An empty vector where `ce.debias` is `true` and a regime method is
    set, else `nothing`.

# Related

  - [`regime_bias!`](@ref)
  - [`RegimeAdjustedVarianceState`](@ref)
"""
function regime_bias_state(ce::RegimeAdjustedExpWeightedVariance, ::Type{T}) where {T}
    return ce.debias && !isnothing(ce.regime_method) ? T[] : nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Makes the empty store of bias factors for a new state of a regime-adjusted covariance estimator.

The regime target names the store with [`regime_bias_store`](@ref): the table of a
[`DiagonalTarget`](@ref) holds the moments of one term of [`RegimeTermMoments`](@ref), a triple
for each count; the store of a [`MahalanobisTarget`](@ref) holds the interpolation nodes of its
factor for each count of assets; the table of any other target holds one factor for each count.

# Arguments

  - `ce::RegimeAdjustedExpWeightedCovariance`: Estimator configuration.
  - `::Type{T}`: Element type of the state.

# Returns

  - `bias::Option{<:Union{<:AbstractVector, <:AbstractDict}}`: The empty store where `ce.debias`
    is `true` and a regime method is set, else `nothing`.

# Related

  - [`regime_bias_store`](@ref)
  - [`regime_bias!`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
"""
function regime_bias_state(ce::RegimeAdjustedExpWeightedCovariance, ::Type{T}) where {T}
    if !ce.debias || isnothing(ce.regime_method)
        return nothing
    end

    return regime_bias_store(ce.regime_target,
                             has_separate_cor_decay(ce) ? ce.cor_decay : ce.decay, T)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Factorises a covariance block for the Mahalanobis regime statistic, and refuses rather than
throws when no ridge makes it factorise.

A block with fewer observations than assets is singular. Where the estimator has
`debias = true`, [`regime_target_statistic`](@ref) skips such a block before it reaches this
function, so the ridge serves a block that is singular in the data, and the raw statistic of
`debias = false`.
The regime statistic is one observation of a smoother rather than a result a caller reads, so a
refusal skips that observation's regime update, and the fit continues.

# Algorithm

 1. Try the plain factorisation of the lower triangle of `C`. Return it where it succeeds.
 2. Symmetrise `C`, and take the mean absolute diagonal as the scale. Where that is not a finite
    positive number, take the largest absolute entry, and at least one.
 3. Add a ridge of `max(min_val * scale, eps * scale)` to the diagonal, and try again. Multiply
    the ridge by ten after each failure, for three tries in all.
 4. Return `nothing` where every try fails.

# Arguments

  - `C::MatNum`: Covariance block of the assets that contribute to the statistic.
  - `min_val::Number`: Scale of the first ridge.

# Returns

  - `chol::Option{<:LinearAlgebra.Cholesky}`: The factorisation, or `nothing` where no ridge
    makes the block factorise.

# Related

  - [`regime_statistic`](@ref)
  - [`MahalanobisTarget`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
"""
function safe_regime_cholesky(C::MatNum, min_val::Number)
    chol = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(C, :L); check = false)
    if LinearAlgebra.issuccess(chol)
        return chol
    end
    S = (C + transpose(C)) / 2
    base = Statistics.mean(abs, LinearAlgebra.diag(S))
    scale = if base > zero(base) && isfinite(base)
        base
    else
        max(maximum(abs, S), one(base))
    end
    ridge = max(min_val * scale, eps(scale) * scale)
    for _ in 1:3
        chol = LinearAlgebra.cholesky(LinearAlgebra.Hermitian(S + ridge * LinearAlgebra.I,
                                                              :L); check = false)
        if LinearAlgebra.issuccess(chol)
            return chol
        end
        ridge *= 10
    end

    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the squared Mahalanobis distance of one observation, divided by the bias of the
estimated block that the regime method reads, where `ce.debias` is `true`.

# Algorithm

 1. Where `ce.debias` is `false`, the factor is one. Else take ``K``, the smallest count of
    observations among the contributing assets, and return `nothing` where
    [`regime_bias_open`](@ref) refuses it. Else read the decay of the correlation structure:
    `cor_decay` on the separate path, else `decay`. Without a HAC adjustment, find the factor of
    `ce.regime_method` with [`mahalanobis_regime_bias!`](@ref). With one, find the factor of the
    mean with [`mahalanobis_bias`](@ref) on the banded weight matrix, for every method.
 2. Compute the squared distance with [`regime_statistic`](@ref), and return `nothing` where the
    block does not factorise.
 3. Divide the statistic by the factor.

# Arguments

  - `target::MahalanobisTarget`: Mahalanobis regime-adjustment target.
  - `cache::RegimeAdjustedCovarianceState`: Online covariance computation cache.
  - `ce::RegimeAdjustedExpWeightedCovariance`: Covariance estimator configuration.
  - `X::VecNum`: Current centred returns vector of every asset.
  - `idx::AbstractVector{<:Integer}`: Index of the assets that contribute to the statistic.

# Returns

  - `stats::Option{<:VecNum}`: One statistic, or `nothing` where the observation takes no
    regime update.

# Related

  - [`MahalanobisTarget`](@ref)
  - [`mahalanobis_regime_bias!`](@ref)
  - [`update_regime!`](@ref)
"""
function regime_target_statistic(target::MahalanobisTarget,
                                 cache::RegimeAdjustedCovarianceState,
                                 ce::RegimeAdjustedExpWeightedCovariance, X::VecNum,
                                 idx::AbstractVector{<:Integer})
    K = minimum(view(cache.obs_count, idx))
    n = length(idx)
    decay = has_separate_cor_decay(ce) ? ce.cor_decay : ce.decay
    b = if !ce.debias
        one(ce.decay)
    elseif !regime_bias_open(true, n, decay, K, ce.hac_lags)
        nothing
    elseif isnothing(ce.hac_lags)
        mahalanobis_regime_bias!(cache.bias, ce.regime_method, decay, K, n)
    else
        mahalanobis_bias(decay, K, n, ce.hac_lags)
    end
    if isnothing(b)
        return nothing
    end
    stats = regime_statistic(target, X[idx], regime_covariance_block(cache, ce, idx), idx,
                             ce.min_val)

    return isnothing(stats) ? nothing : stats ./ b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the deterministic equivalent of the mean bias factor of the squared Mahalanobis distance
against an exponentially weighted covariance estimate.

The factor is ``b = \\mathbb{E}[\\operatorname{tr}(W^{-1})] / n`` of
``W = \\sum_{j} w_{j} z_{j} z_{j}^{\\top}``, ``z_{j} \\sim N(0, I_{n})``, which is the mean of
``u^{\\top} \\hat{C}^{-1} u / n`` for a correctly calibrated return ``u`` that is independent of the
estimate ``\\hat{C}``. It depends on the weights and on `n` alone. It solves
``1 / b = \\sum_{j} w_{j} / (1 + (n + 1) w_{j} b)``, which exists where `K > n + 1` and is exact at
equal weights. On exponential weights it is 0.55 % above the mean at 12 assets and a half-life of
10, so [`inverse_wishart_bias`](@ref) takes it only to name the pole of the factor at
`K = n + 1`, and [`mahalanobis_regime_bias!`](@ref) corrects the result.

# Algorithm

 1. Return `nothing` where `K <= n + 1`.
 2. Start at ``b = 1``. The map ``h(b) = b\\, g(b) - 1``, with ``g`` the right-hand side of the
    fixed point, is concave and increasing, and ``h(1) < 0``, so Newton's method from ``b = 1``
    rises to the root without overshoot.
 3. Evaluate ``g`` and ``h'`` with [`mahalanobis_bias_sums`](@ref), and stop when the step is
    below four units in the last place of ``b``.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations in the estimate.
  - `n::Integer`: Count of assets that contribute to the statistic.

# Returns

  - `b::Option{<:Number}`: The bias factor, or `nothing` where `K <= n + 1`.

# Related

  - [`MahalanobisTarget`](@ref)
  - [`inverse_wishart_bias`](@ref)
  - [`mahalanobis_bias_sums`](@ref)
"""
function mahalanobis_bias(decay::Number, K::Integer, n::Integer)
    if K <= n + 1
        return nothing
    end
    w0 = (one(decay) - decay) / (one(decay) - decay^K)
    b = one(w0)
    for _ in 1:100
        g, dh = mahalanobis_bias_sums(decay, K, n + 1, w0, b)
        step = (b * g - one(b)) / dh
        b -= step
        if abs(step) <= 4 * eps(b)
            break
        end
    end

    return b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the right-hand side of the fixed point of [`mahalanobis_bias`](@ref) and the
derivative of ``b\\, g(b)``.

The weights fall geometrically, and a weight ``w_{j}`` with ``c\\, w_{j}\\, b`` below the
machine epsilon of `decay` enters both sums as itself, so the loop stops there and adds the
remaining mass ``1 - \\sum_{i < j} w_{i}`` to each sum. That keeps the cost bounded on a long
history.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations in the estimate.
  - `c::Integer`: `n + 1`.
  - `w0::Number`: Normalised weight of the newest observation, `(1 - decay) / (1 - decay^K)`.
  - `b::Number`: Current value of the bias factor.

# Returns

  - `(g, dh)::Tuple{<:Number, <:Number}`: ``\\sum_{j} w_{j} / (1 + c\\, w_{j}\\, b)`` and
    ``\\sum_{j} w_{j} / (1 + c\\, w_{j}\\, b)^{2}``.

# Related

  - [`mahalanobis_bias`](@ref)
"""
function mahalanobis_bias_sums(decay::Number, K::Integer, c::Integer, w0::Number, b::Number)
    g = zero(w0)
    dh = zero(w0)
    used = zero(w0)
    w = w0
    for _ in 1:K
        t = c * w * b
        if t <= eps(decay)
            break
        end
        g += w / (one(t) + t)
        dh += w / (one(t) + t)^2
        used += w
        w *= decay
    end
    tail = one(w0) - used

    return g + tail, dh + tail
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the bias factor of the squared Mahalanobis distance against a HAC-adjusted covariance
estimate.

A HAC estimate is ``\\hat{C} = Z^{\\top} A Z`` with the banded weight matrix ``A`` of
[`regime_bias_table`](@ref), so it has the law of a plain estimate whose weights are the
eigenvalues ``\\mu`` of ``A``, and the fixed point of [`mahalanobis_bias`](@ref) holds with ``\\mu``
in place of the weights. Its sums are the derivatives of ``\\ell(t) = \\ln \\det(I + t A)``,
so no eigenvalue is computed:

```math
\\begin{align}
\\frac{1}{b} &= \\sum_{k} \\frac{\\mu_{k}}{1 + t \\mu_{k}} = \\ell'(t)\\,, \\quad t = (n + 1) b\\,.
\\end{align}
```

Each term ``\\mu b / (1 + (n + 1) \\mu b)`` is concave in ``b`` where ``1 + t \\mu > 0``, also for a
negative ``\\mu``, so Newton's method from ``b = 1`` rises to the root as in the plain case. A
negative ``\\mu`` bounds ``t``: where a pivot of ``I + t A`` is not positive, or where the slope of
the map is not positive, the map has passed its maximum below zero and no root exists. At 12
assets, a half-life of 10 and two lags, the factor is 2.523 against a Monte Carlo of 2.479, and
1.2654 against 1.2633 at a half-life of 40: the fixed point is a deterministic equivalent, and a
HAC estimate has about half the degrees of freedom of a plain one.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations in the estimate.
  - `n::Integer`: Count of assets that contribute to the statistic.
  - `hac_lags::Integer`: Count of HAC lags.

# Returns

  - `b::Option{<:Number}`: The bias factor, or `nothing` where `K <= n + 1` or no root exists.

# Related

  - [`hac_log_det_slopes`](@ref)
  - [`mahalanobis_bias`](@ref)
  - [`MahalanobisTarget`](@ref)
"""
function mahalanobis_bias(decay::Number, K::Integer, n::Integer, hac_lags::Integer)
    b = one(decay)
    for _ in 1:(K > n + 1 ? 100 : 0)
        slopes = hac_log_det_slopes(decay, K, (n + 1) * b, hac_lags)
        if isnothing(slopes) || !(slopes[2] > zero(b))
            return nothing
        end
        step = (b * slopes[1] - one(b)) / slopes[2]
        b -= step
        if abs(step) <= 4 * eps(b)
            return b
        end
    end

    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Computes the two sums of the HAC fixed point of [`mahalanobis_bias`](@ref) from the banded LDLᵀ
factorisation of ``I + t A``, with the first and second derivatives in ``t`` carried through it.

``\\ell(t) = \\sum_{m} \\ln d_{m}(t)`` over the pivots, so ``\\ell' = \\sum_{m} d_{m}' / d_{m}`` and
``\\ell'' = \\sum_{m} (d_{m}'' / d_{m} - (d_{m}' / d_{m})^{2})``. A row whose diagonal
``t\\, c\\, \\lambda^{m}`` falls below the machine epsilon of `decay` times ``1 - \\lambda`` enters
both sums as its weight, so the loop stops there and adds the remaining mass, as
[`mahalanobis_bias_sums`](@ref) does.

# Arguments

  - `decay::Number`: Decay of the weights.
  - `K::Integer`: Count of observations in the estimate.
  - `t::Number`: Argument, ``(n + 1) b``.
  - `hac_lags::Integer`: Count of HAC lags.

# Returns

  - `slopes::Option{<:Tuple{<:Number, <:Number}}`: ``(\\ell'(t), \\ell'(t) + t \\ell''(t))``, that is
    ``\\sum_{k} \\mu_{k} / (1 + t \\mu_{k})`` and ``\\sum_{k} \\mu_{k} / (1 + t \\mu_{k})^{2}``, or
    `nothing` where ``I + t A`` is not positive definite.

# Related

  - [`hac_ldl_taylor_row!`](@ref)
  - [`mahalanobis_bias`](@ref)
"""
function hac_log_det_slopes(decay::Number, K::Integer, t::Number, hac_lags::Integer)
    c = (one(decay) - decay) / (one(decay) - decay^K)
    z = ntuple(_ -> zero(c * t), 3)
    buf = (; l = fill(z, hac_lags, hac_lags), d = fill(z, hac_lags),
           row = fill(z, hac_lags))
    g1, g2 = zero(c * t), zero(c * t)
    lam = one(decay)
    for m in 0:(K - 1)
        if t * c * lam <= eps(decay) * (one(decay) - decay)
            tail = c * lam * (one(decay) - decay^(K - m)) / (one(decay) - decay)
            return g1 + tail, g1 + t * g2 + tail
        end
        dn = hac_ldl_taylor_row!(buf, t * c * lam, c * lam, decay, min(hac_lags, m))
        if !(dn[1] > zero(dn[1]))
            return nothing
        end
        g1 += dn[2] / dn[1]
        g2 += dn[3] / dn[1] - (dn[2] / dn[1])^2
        lam *= decay
    end

    return g1, g1 + t * g2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds one row to the banded LDLᵀ factorisation of ``I + t A``, with each entry carried as its value
and its first two derivatives in ``t``.

The entries of ``I + t A`` are linear in ``t``, so the diagonal of the row is ``(1 + t a, a, 0)``
and its entry at lag ``i`` is ``(t a_{i}, a_{i}, 0)``, with ``a_{i} = k_{i}\\, a / \\lambda^{i}``.
The recursion is that of [`hac_ldl_row!`](@ref), in [`taylor_mul`](@ref) and
[`taylor_div`](@ref).

# Arguments

  - `buf::NamedTuple`: Buffers of the factorisation (mutated): `l[r, i]`, the entry of row
    ``m - r`` at column ``m - r - i``; `d[r]`, the pivot of row ``m - r``; and `row`, scratch.
  - `ta::Number`: ``t\\, c\\, \\lambda^{m}``, the diagonal of ``t A`` in the row.
  - `a::Number`: ``c\\, \\lambda^{m}``, the diagonal of ``A`` in the row.
  - `decay::Number`: Decay of the weights.
  - `p::Integer`: Count of lags that reach an observation of the estimate, ``\\min(L, m)``.

# Returns

  - `pivot::NTuple{3, <:Number}`: The new pivot and its two derivatives.

# Related

  - [`hac_log_det_slopes`](@ref)
"""
function hac_ldl_taylor_row!(buf::NamedTuple, ta::Number, a::Number, decay::Number,
                             p::Integer)
    (; l, d, row) = buf
    L = length(d)
    for i in p:-1:1
        k = (one(decay) - i / (L + 1)) / decay^i
        num = (ta * k, a * k, zero(a))
        for j in (i + 1):p
            num = num .- taylor_mul(taylor_mul(row[j], l[i, j - i]), d[j])
        end
        row[i] = taylor_div(num, d[i])
    end
    pivot = (one(ta) + ta, a, zero(a))
    for i in 1:p
        pivot = pivot .- taylor_mul(taylor_mul(row[i], row[i]), d[i])
    end
    for r in L:-1:2
        d[r] = d[r - 1]
        l[r, :] .= view(l, r - 1, :)
    end
    d[1] = pivot
    l[1, :] .= row

    return pivot
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Multiplies two values carried with their first two derivatives.

# Arguments

  - `x::NTuple{3, <:Number}`: ``(x, x', x'')``.
  - `y::NTuple{3, <:Number}`: ``(y, y', y'')``.

# Returns

  - `xy::NTuple{3, <:Number}`: ``(x y, x' y + x y', x'' y + 2 x' y' + x y'')``.

# Related

  - [`taylor_div`](@ref)
  - [`hac_ldl_taylor_row!`](@ref)
"""
function taylor_mul(x::NTuple{3, <:Number}, y::NTuple{3, <:Number})
    return (x[1] * y[1], x[2] * y[1] + x[1] * y[2],
            x[3] * y[1] + 2 * x[2] * y[2] + x[1] * y[3])
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Divides two values carried with their first two derivatives.

# Arguments

  - `x::NTuple{3, <:Number}`: ``(x, x', x'')``.
  - `y::NTuple{3, <:Number}`: ``(y, y', y'')``, with ``y \\neq 0``.

# Returns

  - `q::NTuple{3, <:Number}`: ``(q, q', q'')`` of ``q = x / y``: ``q' = (x' - q y') / y`` and
    ``q'' = (x'' - 2 q' y' - q y'') / y``.

# Related

  - [`taylor_mul`](@ref)
  - [`hac_ldl_taylor_row!`](@ref)
"""
function taylor_div(x::NTuple{3, <:Number}, y::NTuple{3, <:Number})
    q = x[1] / y[1]
    q1 = (x[2] - q * y[2]) / y[1]
    return (q, q1, (x[3] - 2 * q1 * y[2] - q * y[3]) / y[1])
end
"""
    Statistics.cov(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> MatNum

Compute the regime-adjusted exponentially weighted covariance matrix.

Iterates over the observation dimension of `X`, updating an online covariance cache at each
step. After the last observation, removes the damping of the zero seed and scales the result by
the square of the regime multiplier.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix with the same size as `X`. When provided, only
    assets where `estimation_mask[i, :]` (or `[:, i]`) is `true` contribute to the regime state
    update for observation `i`.
  - `active_mask`: Optional boolean matrix with the same size as `X`. When provided, assets that
    become inactive have their covariance entries and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - $(ret_dict[:sigma])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> size(cov(ce, X))
(2, 2)
```

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_adjusted_covariance`](@ref)
  - [`Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = regime_adjusted_covariance_pass!(ce, X, dims, estimation_mask, active_mask)
    if !ce.centred
        unseen = cache.obs_count .< one(eltype(cache.obs_count))
        if any(unseen)
            cache.location[unseen] .= NaN
        end
    end

    return regime_adjusted_covariance(cache, ce)
end
"""
    gap_fill_value(ce::RegimeAdjustedExpWeightedCovariance) -> Float64

Answer `NaN`, so a gapped sample reaches the recursion with its gaps intact.

The recursion updates only the sub-block of the assets that are valid at each observation, and the regime weight is taken from that sub-block alone, so a fill would both decay a frozen block and move the regime it is weighted by. The consumer therefore hands the sample as it stands, together with the active mask that explains the gap.

# Arguments

  - $(arg_dict[:ce])

# Returns

  - `fv::Float64`: `NaN`.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`gap_fill_value`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function gap_fill_value(::RegimeAdjustedExpWeightedCovariance)
    return NaN
end
"""
    Statistics.cor(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> MatNum

Compute the regime-adjusted exponentially weighted correlation matrix.

This is the covariance of the same call, rescaled to a unit diagonal. The regime multiplier
scales the whole matrix, so it cancels in the rescale and the correlation does not read it.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix with the same size as `X`. When provided, only
    assets where `estimation_mask[i, :]` (or `[:, i]`) is `true` contribute to the regime state
    update for observation `i`.
  - `active_mask`: Optional boolean matrix with the same size as `X`. When provided, assets that
    become inactive have their covariance entries and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - $(ret_dict[:rho])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> LinearAlgebra.diag(cor(ce, X))
2-element Vector{Float64}:
 1.0
 1.0
```

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`regime_adjusted_correlation`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    return regime_adjusted_correlation(Statistics.cov(ce, X; dims = dims,
                                                      estimation_mask = estimation_mask,
                                                      active_mask = active_mask, kwargs...))
end
"""
    partial_fit!(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> RegimeAdjustedExpWeightedCovariance

Fold a block of observations into the estimator's own online covariance state.

The recursion reads one observation at a time, so a block folded on top of an existing state is
what the same observations give when they are read in one pass. That is what this verb gives and
what [`merge_states`](@ref) refuses: two blocks each fitted from a cold start do not add.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix with the same size as `X`, restricting which
    assets contribute to the regime state update.
  - `active_mask`: Optional boolean matrix with the same size as `X`. An asset that becomes
    inactive has its covariance entries and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `X`.

# Returns

  - `ce::RegimeAdjustedExpWeightedCovariance`: The estimator, with `cache` holding the state
    after the block.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012; 0.008 -0.02; -0.02 0.03];

julia> ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> isequal(cov(partial_fit!(ce, X)), cov(ce, X))
true

julia> halves = partial_fit!(partial_fit!(ce, X[1:3, :]), X[4:6, :]);

julia> isequal(cov(halves), cov(ce, X))
true
```

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_adjusted_covariance_pass!`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance)`](@ref)
"""
function partial_fit!(ce::RegimeAdjustedExpWeightedCovariance{<:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any, <:Any, <:Any,
                                                              <:Any,
                                                              <:Option{<:RegimeAdjustedCovarianceState}},
                      X::MatNum; dims::Int = 1,
                      estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                      active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    cache = regime_adjusted_covariance_pass!(ce, X, dims, estimation_mask, active_mask,
                                             ce.cache)
    Accessors.@reset ce.cache = cache

    return ce
end
"""
    partial_fit!(
        ce::RegimeAdjustedExpWeightedCovariance,
        x::VecNum;
        estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing,
        active_mask::Option{<:AbstractVector{<:Bool}} = nothing,
        kwargs...
    ) -> RegimeAdjustedExpWeightedCovariance

Fold one observation into the estimator's own online covariance state.

The entries of `x` are the assets of a single observation, which is the row the matrix method
folds one at a time. This is the shape a caller has when the observations arrive one by one.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - `x::VecNum`: One observation, with one entry per asset.
  - `estimation_mask`: Optional boolean vector with the same length as `x`, restricting which
    assets contribute to the regime state update.
  - `active_mask`: Optional boolean vector with the same length as `x`. An asset that becomes
    inactive has its covariance entries and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - If `estimation_mask` is not `nothing`, `length(x) == length(estimation_mask)`.
  - If `active_mask` is not `nothing`, `length(x) == length(active_mask)`.
  - If `ce.cache` is not `nothing`, it holds as many assets as `x`.

# Returns

  - `ce::RegimeAdjustedExpWeightedCovariance`: The estimator, with `cache` holding the state
    after the observation.

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012; 0.008 -0.02; -0.02 0.03];

julia> ce = RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2, regime_min_obs = 2);

julia> one_at_a_time = foldl((c, i) -> partial_fit!(c, view(X, i, :)), axes(X, 1); init = ce);

julia> isequal(cov(one_at_a_time), cov(ce, X))
true
```

# Related

  - [`partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance)`](@ref)
"""
function partial_fit!(ce::RegimeAdjustedExpWeightedCovariance{<:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any, <:Any, <:Any,
                                                              <:Any, <:Any, <:Any, <:Any,
                                                              <:Any,
                                                              <:Option{<:RegimeAdjustedCovarianceState}},
                      x::VecNum;
                      estimation_mask::Option{<:AbstractVector{<:Bool}} = nothing,
                      active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
    return partial_fit!(ce, permutedims(x); dims = 1,
                        estimation_mask = if isnothing(estimation_mask)
                            nothing
                        else
                            permutedims(estimation_mask)
                        end, active_mask = if isnothing(active_mask)
                            nothing
                        else
                            permutedims(active_mask)
                        end)
end
"""
    Statistics.cov(
        ce::RegimeAdjustedExpWeightedCovariance,
        state::RegimeAdjustedCovarianceState;
        kwargs...
    ) -> MatNum

Read the regime-adjusted covariance out of a state held by hand.

This is [`regime_adjusted_covariance`](@ref) under the family's public verb, so a state a caller
keeps outside an estimator answers the same call as one the estimator holds. The state is read
and never written.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - `state`: Running state of an incremental fit.
  - $(arg_dict[:ignkwargs])

# Returns

  - $(ret_dict[:sigma])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                             regime_min_obs = 2), X);

julia> isequal(cov(ce, ce.cache), cov(ce))
true
```

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_adjusted_covariance`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance)`](@ref)
"""
function Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance,
                        state::RegimeAdjustedCovarianceState; kwargs...)
    return regime_adjusted_covariance(state, ce)
end
"""
    Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance; kwargs...) -> MatNum

Read the regime-adjusted covariance out of the estimator's own state.

The one-argument form is what an incremental fit answers: [`partial_fit!`](@ref) leaves the
state in the `cache` field, and this verb turns it into the ordinary answer. An estimator that
has been given no observation carries no state, so the call is refused rather than answered with
a zero.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:sigma])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                             regime_min_obs = 2), X);

julia> size(cov(ce))
(2, 2)

julia> cov(RegimeAdjustedExpWeightedCovariance())
ERROR: ArgumentError: `ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `cov(ce, X)` for a fit over a whole sample.
[...]
```

# Related

  - [`partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, state::RegimeAdjustedCovarianceState; kwargs...)`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
"""
function Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance; kwargs...)
    state = ce.cache
    @argcheck(!isnothing(state),
              ArgumentError("`ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `cov(ce, X)` for a fit over a whole sample."))
    return cov(ce, state)
end
"""
    Statistics.cor(
        ce::RegimeAdjustedExpWeightedCovariance,
        state::RegimeAdjustedCovarianceState;
        kwargs...
    ) -> MatNum

Read the regime-adjusted correlation out of a state held by hand.

The rescale of [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, state::RegimeAdjustedCovarianceState; kwargs...)`](@ref) to a unit diagonal. The state is read and never written.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - `state`: Running state of an incremental fit.
  - $(arg_dict[:ignkwargs])

# Returns

  - $(ret_dict[:rho])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                             regime_min_obs = 2), X);

julia> isequal(cor(ce, ce.cache), cor(ce))
true
```

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`regime_adjusted_correlation`](@ref)
  - [`Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance)`](@ref)
"""
function Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance,
                        state::RegimeAdjustedCovarianceState; kwargs...)
    return regime_adjusted_correlation(regime_adjusted_covariance(state, ce))
end
"""
    Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance; kwargs...) -> MatNum

Read the regime-adjusted correlation out of the estimator's own state.

The one-argument form is what an incremental fit answers: [`partial_fit!`](@ref) leaves the
state in the `cache` field, and this verb turns it into the ordinary answer. An estimator that
has been given no observation carries no state, so the call is refused rather than answered with
a zero.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator carrying a state.
  - $(arg_dict[:ignkwargs])

# Validation

  - `ce.cache` is not `nothing`. An `ArgumentError` is thrown otherwise.

# Returns

  - $(ret_dict[:rho])

# Examples

```jldoctest
julia> X = [0.01 -0.02; -0.015 0.03; 0.02 -0.01; -0.005 0.012];

julia> ce = partial_fit!(RegimeAdjustedExpWeightedCovariance(; decay = 0.9, min_obs = 2,
                                                             regime_min_obs = 2), X);

julia> size(cor(ce))
(2, 2)

julia> cor(RegimeAdjustedExpWeightedCovariance())
ERROR: ArgumentError: `ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `cor(ce, X)` for a fit over a whole sample.
[...]
```

# Related

  - [`partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance, state::RegimeAdjustedCovarianceState; kwargs...)`](@ref)
  - [`RegimeAdjustedCovarianceState`](@ref)
"""
function Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance; kwargs...)
    state = ce.cache
    @argcheck(!isnothing(state),
              ArgumentError("`ce` holds no partial-fit state, so there is nothing to read. Call `partial_fit!(ce, X)` first, or `cor(ce, X)` for a fit over a whole sample."))
    return regime_adjusted_correlation(regime_adjusted_covariance(state, ce))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Refuses a pair of regime-adjusted covariance states, because this family does not merge.

A block fitted from a cold start is not what the same block contributes after another one. The
regime statistic scores each observation against the state that stands before it, so a cold
block loses every comparison its first `min_obs` observations would have made, and a correlation
state that is normalised by a running variance carries that variance with it.

Fold the second block into the first with [`partial_fit!`](@ref) instead. A sequential fit is
exact, and it is the route this family gives.

# Algorithm

 1. Refuse the pair with [`assert_mergeable_states`](@ref), which names a type mismatch and an asset-count mismatch first, as the [`AbstractPartialFitState`](@ref) interface asks of every family.
 2. Throw an `ArgumentError` naming the reason this family does not merge.

# Arguments

  - `a`: The first state.
  - `b`: The second state.

# Validation

  - `a` and `b` pass [`assert_mergeable_states`](@ref).

# Returns

  - Never returns. An `ArgumentError` is thrown.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`merge_states`](@ref)
  - [`assert_mergeable_states`](@ref)
  - [`partial_fit!(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
"""
function merge_states(a::RegimeAdjustedCovarianceState, b::RegimeAdjustedCovarianceState)
    assert_mergeable_states(a, b)
    return throw(ArgumentError("a `RegimeAdjustedCovarianceState` pair does not merge, because a block fitted from a cold start is not what the same block contributes after another one. The regime statistic scores each observation against the state that stands before it, and it is gated by the running observation count. Fold the second block into the first with `partial_fit!` instead."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Copies a [`RegimeAdjustedCovarianceState`](@ref), so the copy shares no array with the original.

The `copy` method of the [`AbstractPartialFitState`](@ref) interface, which [`partial_fit`](@ref)
calls before it folds. Every array field is copied, and the two scalar fields pass through. The
circular buffer of recent centred returns is rebuilt at the same capacity, and each observation
it holds is copied into it, so a fold on the copy pushes into a buffer of its own. The two
fields of the separate correlation recursion pass through as `nothing` where they are `nothing`.
The store of bias factors is `nothing`, an array or a dictionary, and `deepcopy` copies an array
or a dictionary and returns `nothing` unchanged.

# Arguments

  - `x`: The cache to copy.

# Returns

  - `state::RegimeAdjustedCovarianceState`: A fresh cache, equal to `x`, whose arrays are fresh.

# Related

  - [`RegimeAdjustedCovarianceState`](@ref)
  - [`partial_fit`](@ref)
  - [`AbstractPartialFitState`](@ref)
"""
function Base.copy(x::RegimeAdjustedCovarianceState)
    ret_buffer = if isnothing(x.ret_buffer)
        nothing
    else
        buffer = DataStructures.CircularBuffer{eltype(x.ret_buffer)}(DataStructures.capacity(x.ret_buffer))
        for X_old in x.ret_buffer
            push!(buffer, copy(X_old))
        end
        buffer
    end

    weight = isnothing(x.weight) ? nothing : copy(x.weight)
    variance = isnothing(x.variance) ? nothing : copy(x.variance)
    cor_state = isnothing(x.cor_state) ? nothing : copy(x.cor_state)
    cor_weight = isnothing(x.cor_weight) ? nothing : copy(x.cor_weight)

    return RegimeAdjustedCovarianceState(ret_buffer, copy(x.covariance), weight, variance,
                                         cor_state, cor_weight, copy(x.XXt), copy(x.Xi),
                                         copy(x.X_old_i), copy(x.location),
                                         copy(x.obs_count), copy(x.active), x.regime_state,
                                         x.n_regime_obs, deepcopy(x.bias))
end
"""
    Statistics.cov(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the regime-adjusted exponentially weighted covariance from a window of an Asset Panel.

This estimator is mask-aware, so it overrides the reduce-and-expand root of the verb and reads the panel's two masks itself: the active mask drives the freeze and the reset, and the estimation mask restricts which assets feed the regime statistic. The answer therefore lives on the whole universe rather than on the Coverage Universe, and a young asset that lists inside the window is answered from the observations it has.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `sigma::MatNum`: Covariance matrix of size `assets × assets`.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`panel_moment_masks`](@ref)
  - [`Statistics.cov(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.cov(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                          kwargs...)
end
"""
    Statistics.cor(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the regime-adjusted exponentially weighted correlation from a window of an Asset Panel.

This is the covariance of the same call, rescaled to a unit diagonal, and it reads the panel's two masks through the same override.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `rho::MatNum`: Correlation matrix of size `assets × assets`.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.cor(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.cor(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                          kwargs...)
end
"""
    Statistics.var(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the marginal variance of the regime-adjusted exponentially weighted covariance from a window of an Asset Panel.

This is the diagonal of the covariance of the same call, and it reads the panel's two masks through the same override.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `var::MatNum`: Marginal variance, as a row where `dims` is `1` and as a column otherwise.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`Statistics.cov(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.var(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.var(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                          kwargs...)
end
"""
    Statistics.std(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> MatNum

Compute the marginal volatility of the regime-adjusted exponentially weighted covariance from a window of an Asset Panel.

This is the square root of the diagonal of the covariance of the same call, and it reads the panel's two masks through the same override.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `std::MatNum`: Marginal volatility, as a row where `dims` is `1` and as a column otherwise.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`Statistics.var(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function Statistics.std(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                        pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return Statistics.std(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                          kwargs...)
end

"""
    variance_series(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum;
        dims::Int = 1,
        estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
        kwargs...
    ) -> Matrix{<:Number}

Compute the point-in-time regime-adjusted exponentially weighted variance series.

Row `t` holds the diagonal of what `cov` returns for the first `t` observations of `X`, so no row reads an observation after its own. The update is a recursion over one observation, so this method overrides the expanding-window fallback with a **single forward pass**: it reads the cache after each observation instead of refitting.

The fallback cannot answer this estimator. It slices `X` once per row and passes every keyword unsliced, so a mask of the whole window meets a window of `t` observations and the size check refuses the call.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:dims])
  - `estimation_mask`: Optional boolean matrix with the same size as `X`. When provided,
    only assets where `estimation_mask[i, :]` (or `[:, i]`) is `true` contribute to the
    regime state update for observation `i`.
  - `active_mask`: Optional boolean matrix with the same size as `X`. When provided,
    assets that become inactive have their covariance and observation count reset.
  - $(arg_dict[:ignkwargs])

# Validation

  - $(val_dict[:dims])
  - If `estimation_mask` is not `nothing`, `size(X) == size(estimation_mask)`.
  - If `active_mask` is not `nothing`, `size(X) == size(active_mask)`.

# Returns

  - `val::Matrix{<:Number}`: Variance series, shaped as `(T, N)` if `dims == 1` or `(N, T)` if
    `dims == 2`. An asset with fewer than `ce.min_obs` observations at row `t` is `NaN` there.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`regime_adjusted_covariance`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1,
                         estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing,
                         active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
    assert_dims(dims)
    val = Matrix{float_if_integer(eltype(X))}(undef, size(X, dims),
                                              size(X, setdiff((1, 2), (dims,))[1]))
    regime_adjusted_covariance_pass!(ce, X, dims, estimation_mask, active_mask) do i, cache
        val[i, :] = LinearAlgebra.diag(regime_adjusted_covariance(cache, ce;
                                                                  repair = false))
        return nothing
    end

    return isone(dims) ? val : permutedims(val)
end
"""
    variance_series(
        ce::RegimeAdjustedExpWeightedCovariance,
        X::MatNum,
        pnl::Option{<:AssetPanel};
        dims::Int = 1,
        kwargs...
    ) -> Matrix{<:Number}

Compute the point-in-time regime-adjusted exponentially weighted variance series from a window of an Asset Panel.

This is the diagonal of the covariance of the same call, read after each observation, and it reads the panel's two masks through the same override.

# Arguments

  - `ce`: Regime-adjusted exponentially weighted covariance estimator.
  - $(arg_dict[:X])
  - $(arg_dict[:pnl_moment])
  - $(arg_dict[:dims])
  - $(arg_dict[:ignkwargs])

# Returns

  - `val::Matrix{<:Number}`: Variance series on the full asset universe, shaped as `(T, N)` if
    `dims == 1` or `(N, T)` if `dims == 2`.

# Related

  - [`RegimeAdjustedExpWeightedCovariance`](@ref)
  - [`variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum; dims::Int = 1, estimation_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)`](@ref)
  - [`variance_series(ce::AbstractCovarianceEstimator, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)`](@ref)
"""
function variance_series(ce::RegimeAdjustedExpWeightedCovariance, X::MatNum,
                         pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
    amsk, emsk = dims_oriented(dims, panel_moment_masks(pnl)...)
    return variance_series(ce, X; dims = dims, estimation_mask = emsk, active_mask = amsk,
                           kwargs...)
end

# Folds in every configuration; only its merge refuses (see [`supports_partial_fit`](@ref)).
function supports_partial_fit(::RegimeAdjustedExpWeightedCovariance)
    return true
end
