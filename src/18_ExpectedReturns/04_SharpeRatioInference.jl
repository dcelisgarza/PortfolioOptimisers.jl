"""
    sharpe_ratio_variance_factor(sr::Number, g1::Number, g2::Number)

Return the factor by which the skewness and the kurtosis of a return series scale the asymptotic variance of its Sharpe ratio.

The factor is the asymptotic variance of the sample Sharpe ratio for independent returns of any distribution, Equation 2 of [mertens2002](@cite). Equation 8 of [sharpe_stderr](@cite) restates it, and Section 2.5 of [sharpe_stderr](@cite) divides it by ``T - 1`` for the variance of the sample Sharpe ratio. [`performance_summary`](@ref) reads it for the standard error of the Sharpe ratio, and [`sharpe_ratio_moments`](@ref) reads it for the Probabilistic Sharpe Ratio.

# Mathematical definition

```math
\\begin{align}
v &= 1 - \\hat{\\gamma}_{3} \\widehat{\\mathrm{SR}} + \\dfrac{\\hat{\\gamma}_{4} - 1}{4} \\widehat{\\mathrm{SR}}^{2}\\,.
\\end{align}
```

Where:

  - ``v``: Variance factor of the Sharpe ratio.
  - $(math_dict[:sr_hat])
  - $(math_dict[:gamma3_hat])
  - $(math_dict[:gamma4_hat])

The factor is ``1 + \\widehat{\\mathrm{SR}}^{2} / 2`` for a normal distribution. The kurtosis of a distribution is at least the square of its skewness plus one, so ``v \\geq (1 - \\hat{\\gamma}_{3} \\widehat{\\mathrm{SR}} / 2)^{2} \\geq 0``. The factor is zero only for a distribution on two points.

# Arguments

  - `sr`: Sample Sharpe ratio per period.
  - `g1`: Sample skewness.
  - `g2`: Sample excess kurtosis ``\\hat{\\gamma}_{4} - 3``, the moment that `StatsBase.kurtosis` returns. So the coefficient of the square is ``(g_{2} + 2) / 4``.

# Returns

  - `v::Number`: The variance factor.

# Related

  - [`sharpe_ratio_moments`](@ref)
  - [`performance_summary`](@ref)
  - [`probabilistic_sharpe_ratio`](@ref)

# References

  - $(ref_dict[:mertens2002]) Section 1, Equation 2.
  - $(ref_dict[:sharpe_stderr]) Equation 8 and Section 2.5.
"""
function sharpe_ratio_variance_factor(sr::Number, g1::Number, g2::Number)
    # `g2` is the EXCESS kurtosis, and the coefficient is `(gamma_4 - 1) / 4` on the raw
    # one, so the `+ 2` converts between them. Without it a normal series loses the whole
    # `SR^2 / 2` term of the naive expression.
    return one(sr) - g1 * sr + (g2 + 2) / 4 * sr^2
end
"""
    sharpe_ratio_moments(x::VecNum, rf::Number)

Return the number of observations, the sample Sharpe ratio and its variance factor for a return series without its `NaN` entries.

[`probabilistic_sharpe_ratio`](@ref) and [`minimum_track_record_length`](@ref) read the three numbers. The closed forms are those of the `# Mathematical definition` of [`probabilistic_sharpe_ratio`](@ref) and of [`sharpe_ratio_variance_factor`](@ref).

# Algorithm

 1. Drop the `NaN` entries of `x`.
 2. Take `T`, the number of entries that remain. When `T < 3`, the Sharpe ratio and the factor are `NaN`. Two observations always have a skewness of zero and a kurtosis of one, so they give no data about the shape of the distribution.
 3. Take `m`, the sample mean of `x`, clamped to the least and the greatest entry, and `s`, the corrected sample standard deviation about `m`. The clamp moves nothing in exact arithmetic. On a constant series it makes `s` exactly zero. When `s` is not positive, the Sharpe ratio and the factor are `NaN`.
 4. Divide the excess `m - rf` by `s`, to get `sr`.
 5. Read the sample skewness and the sample excess kurtosis of `x` about `m`. Give them with `sr` to [`sharpe_ratio_variance_factor`](@ref), to get `v`. When `v` is not positive, it is `NaN`.

# Arguments

  - `x`: Periodic return series.
  - $(arg_dict[:rf])

# Returns

  - `(T, sr, v)`: The number of entries of `x` that are not `NaN`, the sample Sharpe ratio per period, and its variance factor.

# Related

  - [`sharpe_ratio_variance_factor`](@ref)
  - [`probabilistic_sharpe_ratio`](@ref)
  - [`minimum_track_record_length`](@ref)
"""
function sharpe_ratio_moments(x::VecNum, rf::Number)
    if any(isnan, x)
        x = filter(!isnan, x)
    end
    T = length(x)
    # The `NaN` takes the number type of the Sharpe ratio, which the square root of the
    # standard deviation widens.
    o = one(eltype(x))
    nan = oftype((o - rf) / sqrt(o), NaN)
    if T < 3
        return T, nan, nan
    end
    # The clamp moves nothing in exact arithmetic. On a constant series it makes the mean the
    # common value, so `s` is exactly zero.
    m = clamp(mean(x), extrema(x)...)
    s = std(x; mean = m)
    if !(s > zero(s))
        return T, nan, nan
    end
    sr = (m - rf) / s
    v = sharpe_ratio_variance_factor(sr, StatsBase.skewness(x, m), StatsBase.kurtosis(x, m))
    return T, sr, v > zero(v) ? v : oftype(v, NaN)
end
"""
    probabilistic_sharpe_ratio(x::VecNum; bsr::Number = 0, rf::Number = 0)
    probabilistic_sharpe_ratio(X::MatNum; bsr::Number = 0, rf::Number = 0)

Compute the Probabilistic Sharpe Ratio, the probability that the true Sharpe ratio of a return series is greater than a benchmark.

The Probabilistic Sharpe Ratio of [sharpe_stderr](@cite) corrects the sample Sharpe ratio for the length of the series and for the skewness and the kurtosis of its returns. Fat tails make a Sharpe ratio that is not zero less precise, and negative skewness makes a positive Sharpe ratio less precise. When the Sharpe ratio is above the benchmark, a less precise Sharpe ratio has a smaller probability. Every quantity is per period, so `x`, `bsr` and `rf` must have the same frequency, and the function annualises none of them. The matrix method computes one probability for each column.

# Mathematical definition

```math
\\begin{align}
\\mathrm{PSR}\\left(\\mathrm{SR}^{*}\\right) &= \\Phi\\left(\\dfrac{\\left(\\widehat{\\mathrm{SR}} - \\mathrm{SR}^{*}\\right) \\sqrt{T - 1}}{\\sqrt{1 - \\hat{\\gamma}_{3} \\widehat{\\mathrm{SR}} + \\dfrac{\\hat{\\gamma}_{4} - 1}{4} \\widehat{\\mathrm{SR}}^{2}}}\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{PSR}``: Probabilistic Sharpe Ratio.
  - $(math_dict[:sr_star])
  - $(math_dict[:Phi_normal])
  - $(math_dict[:sr_hat])
  - $(math_dict[:T])
  - $(math_dict[:gamma3_hat])
  - $(math_dict[:gamma4_hat])
  - $(math_dict[:xret])

The denominator is the square root of the variance factor of [`sharpe_ratio_variance_factor`](@ref). A Sharpe ratio equal to the benchmark gives a probability of ``1/2``.

# Algorithm

 1. Check `bsr` and `rf`, as `# Validation` states.
 2. Take `T`, `sr` and `v`, the number of observations, the sample Sharpe ratio and its variance factor, through [`sharpe_ratio_moments`](@ref). It drops the `NaN` entries of `x`.
 3. Multiply the excess `sr - bsr` by the square root of `T - 1`, and divide it by the square root of `v`, to get `z`.
 4. Evaluate the standard normal distribution function at `z`. A `NaN` from step 2 gives a `NaN`.

# Arguments

  - $(arg_dict[:sr_x])
  - $(arg_dict[:bsr]) It defaults to `0`.
  - $(arg_dict[:rf]) It defaults to `0`, in the frequency of `x`.

# Validation

  - $(val_dict[:bsr])
  - $(val_dict[:rf])

# Returns

  - `psr::Number`: The probability, for a vector `x`. It is `NaN` when `x` has fewer than three entries that are not `NaN`, when its standard deviation is zero, or when its variance factor is not positive.
  - `psr::Vector`: One probability for each column, for a matrix `X`.

# Examples

```jldoctest
julia> x = [0.012, -0.004, 0.007, 0.015, -0.011, 0.009, 0.003, -0.002, 0.011, 0.006];

julia> round(probabilistic_sharpe_ratio(x); digits = 6)
0.920149

julia> round(probabilistic_sharpe_ratio(x; bsr = 0.2); digits = 6)
0.818392
```

# Related

  - [`deflated_sharpe_ratio`](@ref)
  - [`minimum_track_record_length`](@ref)
  - [`sharpe_ratio_moments`](@ref)
  - [`performance_summary`](@ref): The standard error of its Sharpe ratio reads the same variance factor.

# References

  - $(ref_dict[:sharpe_stderr]) Equation 11.
"""
function probabilistic_sharpe_ratio(x::VecNum; bsr::Number = 0, rf::Number = 0)
    @argcheck(isfinite(bsr), IsNonFiniteError("bsr must be finite, got $bsr"))
    @argcheck(isfinite(rf), IsNonFiniteError("rf must be finite, got $rf"))
    T, sr, v = sharpe_ratio_moments(x, rf)
    # The count enters in the type of the Sharpe ratio, so a `Float32` series stays `Float32`.
    z = (sr - bsr) * sqrt(max(T - 1, 0) * one(sr)) / sqrt(v)
    return Distributions.cdf(Distributions.Normal(zero(z), one(z)), z)
end
function probabilistic_sharpe_ratio(X::MatNum; kwargs...)
    # The `NaN` of an empty column has the type of every column's value, so the vector is
    # typed even when inference cannot see that type.
    T = typeof(probabilistic_sharpe_ratio(similar(X, 0); kwargs...))
    return T[probabilistic_sharpe_ratio(x; kwargs...) for x in eachcol(X)]
end
"""
    minimum_track_record_length(x::VecNum; bsr::Number = 0, rf::Number = 0,
                                alpha::Number = 0.05)
    minimum_track_record_length(X::MatNum; bsr::Number = 0, rf::Number = 0,
                                alpha::Number = 0.05)

Compute the Minimum Track Record Length, the number of observations that a return series needs for its Sharpe ratio to exceed a benchmark at a confidence level.

The Minimum Track Record Length of [sharpe_stderr](@cite) is the least number of observations at which the [`probabilistic_sharpe_ratio`](@ref) of the series reaches `1 - alpha`, for the sample moments of the series. It is a number of observations, not a number of years. The matrix method computes one length for each column.

# Mathematical definition

```math
\\begin{align}
\\mathrm{MinTRL} &= 1 + \\left(1 - \\hat{\\gamma}_{3} \\widehat{\\mathrm{SR}} + \\dfrac{\\hat{\\gamma}_{4} - 1}{4} \\widehat{\\mathrm{SR}}^{2}\\right) \\left(\\dfrac{\\Phi^{-1}(1 - \\alpha)}{\\widehat{\\mathrm{SR}} - \\mathrm{SR}^{*}}\\right)^{2}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{MinTRL}``: Minimum Track Record Length, a number of observations.
  - ``\\alpha``: Significance level, so ``1 - \\alpha`` is the confidence level.
  - $(math_dict[:gamma3_hat])
  - $(math_dict[:sr_hat])
  - $(math_dict[:gamma4_hat])
  - $(math_dict[:Phi_normal])
  - $(math_dict[:sr_star])
  - $(math_dict[:xret])

The length solves ``\\mathrm{PSR}(\\mathrm{SR}^{*}) = 1 - \\alpha`` for ``T``. When ``\\widehat{\\mathrm{SR}} \\leq \\mathrm{SR}^{*}``, the Probabilistic Sharpe Ratio is at most ``1/2`` for every ``T``, so no length is enough and the length is ``+\\infty``.

# Algorithm

 1. Check `alpha`, `bsr` and `rf`, as `# Validation` states.
 2. Take `sr` and `v`, the sample Sharpe ratio and its variance factor, through [`sharpe_ratio_moments`](@ref). It drops the `NaN` entries of `x`.
 3. Take the excess `d = sr - bsr`, and `za`, the quantile of the standard normal distribution at `1 - alpha`.
 4. Add one to `v` times the square of `za / d`, to get `n`.
 5. When `d` is not positive and `n` is not `NaN`, replace `n` with `Inf`.

# Arguments

  - $(arg_dict[:sr_x])
  - $(arg_dict[:bsr]) It defaults to `0`.
  - $(arg_dict[:rf]) It defaults to `0`, in the frequency of `x`.
  - `alpha`: Significance level. It defaults to `0.05`, a confidence level of 95 %.

# Validation

  - `0 < alpha < 1/2`. The quantile ``\\Phi^{-1}(1 - \\alpha)`` is then positive, which the closed form needs.
  - $(val_dict[:bsr])
  - $(val_dict[:rf])

# Returns

  - `n::Number`: The length, for a vector `x`. It is `Inf` when the sample Sharpe ratio does not exceed `bsr`. It is `NaN` in the cases where [`probabilistic_sharpe_ratio`](@ref) is `NaN`.
  - `n::Vector`: One length for each column, for a matrix `X`.

# Examples

```jldoctest
julia> x = [0.012, -0.004, 0.007, 0.015, -0.011, 0.009, 0.003, -0.002, 0.011, 0.006];

julia> round(minimum_track_record_length(x); digits = 6)
13.316333

julia> minimum_track_record_length(x; bsr = 1.0)
Inf
```

# Related

  - [`probabilistic_sharpe_ratio`](@ref)
  - [`sharpe_ratio_moments`](@ref)

# References

  - $(ref_dict[:sharpe_stderr]) Equation 13.
"""
function minimum_track_record_length(x::VecNum; bsr::Number = 0, rf::Number = 0,
                                     alpha::Number = 0.05)
    @argcheck(zero(alpha) < alpha < one(alpha) / 2,
              DomainError(alpha, "alpha must satisfy 0 < alpha < 1/2"))
    @argcheck(isfinite(bsr), IsNonFiniteError("bsr must be finite, got $bsr"))
    @argcheck(isfinite(rf), IsNonFiniteError("rf must be finite, got $rf"))
    _, sr, v = sharpe_ratio_moments(x, rf)
    d = sr - bsr
    za = Distributions.quantile(Distributions.Normal(zero(d), one(d)), one(alpha) - alpha)
    n = one(d) + v * (za / d)^2
    # A Sharpe ratio at or below the benchmark never clears it, however long the record.
    return isnan(n) || d > zero(d) ? n : oftype(n, Inf)
end
function minimum_track_record_length(X::MatNum; kwargs...)
    # As the matrix method of `probabilistic_sharpe_ratio`.
    T = typeof(minimum_track_record_length(similar(X, 0); kwargs...))
    return T[minimum_track_record_length(x; kwargs...) for x in eachcol(X)]
end
"""
    expected_max_sharpe_ratio(N::Number, sigma::Number, mu::Number = 0)

Approximate the expected maximum of the Sharpe ratios of `N` independent trials.

The best of several trials has a Sharpe ratio that the selection itself inflates. The approximation of [bailey2014deflated](@cite) reads the trial Sharpe ratios as independent draws of one normal distribution. [`deflated_sharpe_ratio`](@ref) uses it at `mu = 0` as the benchmark that the selected trial must exceed. Trials with similar parameters have correlated Sharpe ratios, and then [`implied_independent_trials`](@ref) gives a smaller `N`.

# Mathematical definition

```math
\\begin{align}
\\mathbb{E}\\left[\\max_{n} \\widehat{\\mathrm{SR}}_{n}\\right] &\\approx \\mu + \\sigma \\max\\left(0,\\, (1 - \\gamma) \\Phi^{-1}\\left(1 - \\dfrac{1}{N}\\right) + \\gamma \\Phi^{-1}\\left(1 - \\dfrac{1}{N e}\\right)\\right)\\,.
\\end{align}
```

Where:

  - ``\\widehat{\\mathrm{SR}}_{n}``: Sharpe ratio of trial ``n``.
  - ``\\mu``: Mean of the Sharpe ratios of the trials.
  - $(math_dict[:sigma_sr_trials])
  - $(math_dict[:gamma_em])
  - $(math_dict[:Phi_normal])
  - $(math_dict[:n_trials]) It can be fractional, and it is at least one.
  - ``e``: Euler's number.

The maximum of ``N \\geq 1`` draws is at least one draw, so its expectation is at least ``\\mu``. The floor at zero keeps that bound. The source states the approximation for ``N \\gg 1`` and has no floor. Without the floor, the approximation falls below ``\\mu`` for ``N < 1.2836``, and it is ``-\\infty`` at ``N = 1``. For one trial, or for trials with no dispersion, the expected maximum is ``\\mu``.

# Arguments

  - `N`: Number of independent trials. A fractional value is an effective number of trials.
  - `sigma`: Standard deviation of the Sharpe ratios of the trials.
  - `mu`: Mean of the Sharpe ratios of the trials.

# Validation

  - `isfinite(N)` and `N >= 1`.
  - `isfinite(sigma)` and `sigma >= 0`.
  - `isfinite(mu)`.

# Returns

  - `sr::Number`: The approximate expected maximum Sharpe ratio, in the frequency of `sigma` and `mu`.

# Examples

```jldoctest
julia> round(expected_max_sharpe_ratio(100, 1.0); digits = 6)
2.530603

julia> expected_max_sharpe_ratio(1, 0.5, 0.2)
0.2
```

# Related

  - [`deflated_sharpe_ratio`](@ref)
  - [`implied_independent_trials`](@ref)

# References

  - $(ref_dict[:bailey2014deflated]) Equation 1.
"""
function expected_max_sharpe_ratio(N::Number, sigma::Number, mu::Number = 0)
    @argcheck(isfinite(N) && N >= one(N),
              DomainError(N, "N must be finite and at least one"))
    @argcheck(isfinite(sigma) && sigma >= zero(sigma),
              DomainError(sigma, "sigma must be finite and non-negative"))
    @argcheck(isfinite(mu), IsNonFiniteError("mu must be finite, got $mu"))
    # The probabilities take the type of `N` and `sigma` together, so an integer count does
    # not widen a `Float32` dispersion.
    n = N * one(sigma)
    p1 = one(n) - inv(n)
    p2 = one(n) - inv(n * MathConstants.e)
    d = Distributions.Normal(zero(p1), one(p1))
    eg = MathConstants.eulergamma
    q1 = Distributions.quantile(d, p1)
    z = (one(q1) - eg) * q1 + eg * Distributions.quantile(d, p2)
    # The maximum of `N >= 1` draws is never below the mean of one draw, but the
    # approximation is, for `N < 1.2836`, and it is `-Inf` at `N = 1`.
    return mu + sigma * max(z, zero(z))
end
"""
    deflated_sharpe_ratio(x::VecNum_MatNum, srs::VecNum; rf::Number = 0,
                          ntrials::Option{<:Number} = nothing)

Compute the Deflated Sharpe Ratio, the probability that the true Sharpe ratio of the best of several trials is greater than zero.

The Deflated Sharpe Ratio of [bailey2014deflated](@cite) is the [`probabilistic_sharpe_ratio`](@ref) of the selected trial against the [`expected_max_sharpe_ratio`](@ref) of the search. So it corrects for the number and the dispersion of the trials, and for the skewness and the kurtosis of the selected returns. The selected trial meets a confidence level of 95 % when the ratio is at least `0.95`. Every quantity is per period, so `srs` must have the frequency of `x` and the same risk-free rate.

# Mathematical definition

```math
\\begin{align}
\\mathrm{DSR} &= \\mathrm{PSR}\\left(\\mathrm{SR}_{0}\\right)\\,,\\\\
\\mathrm{SR}_{0} &= \\sigma \\max\\left(0,\\, (1 - \\gamma) \\Phi^{-1}\\left(1 - \\dfrac{1}{N}\\right) + \\gamma \\Phi^{-1}\\left(1 - \\dfrac{1}{N e}\\right)\\right)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{DSR}``: Deflated Sharpe Ratio.
  - ``\\mathrm{PSR}``: Probabilistic Sharpe Ratio of the selected trial, as [`probabilistic_sharpe_ratio`](@ref) states it.
  - ``\\mathrm{SR}_{0}``: Expected maximum Sharpe ratio of ``N`` trials whose true Sharpe ratios are zero, as [`expected_max_sharpe_ratio`](@ref) states it.
  - $(math_dict[:sigma_sr_trials]) It is the corrected sample standard deviation of `srs`.
  - $(math_dict[:gamma_em])
  - $(math_dict[:Phi_normal])
  - $(math_dict[:n_trials])
  - ``e``: Euler's number.

# Algorithm

 1. Check `srs`, as `# Validation` states, and drop its `NaN` entries.
 2. Take `M`, the number of entries that remain, and `sigma`, their corrected sample standard deviation. When `M = 1`, `sigma` is zero.
 3. Take the number of trials `ntrials`, or `M` when it is `nothing`, and compute the expected maximum Sharpe ratio at a zero mean with [`expected_max_sharpe_ratio`](@ref), to get `bsr`.
 4. Compute the [`probabilistic_sharpe_ratio`](@ref) of `x` against `bsr`.

# Arguments

  - $(arg_dict[:sr_x]) It holds the selected trial. Every column of a matrix meets the same benchmark.
  - `srs`: Sharpe ratios per period of every trial of the search, the selected trial included. Their number and their dispersion set the benchmark.
  - $(arg_dict[:rf]) It defaults to `0`, in the frequency of `x`.
  - `ntrials`: Number of independent trials. It defaults to `nothing`, which counts every entry of `srs` that is not `NaN` as an independent trial. Correlated trials are fewer independent trials, and [`implied_independent_trials`](@ref) estimates their number.

# Validation

  - No entry of `srs` is infinite.
  - At least one entry of `srs` is not `NaN`.
  - The checks of [`expected_max_sharpe_ratio`](@ref) on `ntrials`, and the checks of [`probabilistic_sharpe_ratio`](@ref) on `rf`.

# Returns

  - `dsr::Number`: The probability, for a vector `x`. It is `NaN` in the cases where [`probabilistic_sharpe_ratio`](@ref) is `NaN`.
  - `dsr::Vector`: One probability for each column, for a matrix `x`.

# Examples

```jldoctest
julia> x = [0.012, -0.004, 0.007, 0.015, -0.011, 0.009, 0.003, -0.002, 0.011, 0.006];

julia> round(deflated_sharpe_ratio(x, [0.1, 0.3, 0.5, 0.2, 0.4]); digits = 6)
0.82579
```

# Related

  - [`probabilistic_sharpe_ratio`](@ref)
  - [`expected_max_sharpe_ratio`](@ref)
  - [`implied_independent_trials`](@ref)

# References

  - $(ref_dict[:bailey2014deflated]) Equation 2.
"""
function deflated_sharpe_ratio(x::VecNum_MatNum, srs::VecNum; rf::Number = 0,
                               ntrials::Option{<:Number} = nothing)
    @argcheck(!any(isinf, srs), IsNonFiniteError("srs must not hold an infinite entry"))
    srs = filter(!isnan, srs)
    @argcheck(!isempty(srs),
              IsEmptyError("srs must hold at least one entry that is not NaN"))
    M = length(srs)
    s = std(srs)
    sigma = M > 1 ? s : zero(s)
    bsr = expected_max_sharpe_ratio(something(ntrials, M), sigma)
    return probabilistic_sharpe_ratio(x; bsr = bsr, rf = rf)
end
"""
    implied_independent_trials(M::Integer, rho::Number)
    implied_independent_trials(C::MatNum)

Estimate the number of independent trials among `M` trials whose returns have the mean correlation `rho`.

The estimate of [bailey2014deflated](@cite) interpolates between one trial, when every pair of trials is perfectly correlated, and `M` trials, when no pair is correlated. Give the result as `ntrials` to [`deflated_sharpe_ratio`](@ref). The matrix method reads `M` and `rho` from the correlation matrix `C` of the returns of the trials.

# Mathematical definition

```math
\\begin{align}
\\hat{\\rho} &= \\dfrac{2}{M (M - 1)} \\sum_{i=1}^{M} \\sum_{j=i+1}^{M} \\rho_{ij}\\,,\\\\
\\hat{N} &= \\hat{\\rho} + (1 - \\hat{\\rho}) M\\,.
\\end{align}
```

Where:

  - ``\\hat{N}``: Implied number of independent trials.
  - ``\\hat{\\rho}``: Mean correlation between the returns of two different trials.
  - ``M``: Number of trials.
  - ``\\rho_{ij}``: Correlation between the returns of trials ``i`` and ``j``, entry ``(i, j)`` of ``\\mathbf{C}``.
  - ``\\mathbf{C}``: Correlation matrix of the returns of the trials, ``M \\times M``.

The source takes a correlation matrix that is positive definite, so ``\\boldsymbol{1}^{\\intercal} \\mathbf{C} \\boldsymbol{1} = M + M (M - 1) \\hat{\\rho} > 0`` and ``\\hat{\\rho} > -1/(M - 1)``. A singular correlation matrix can reach the bound, and the function refuses it. The estimate is ``1`` at ``\\hat{\\rho} = 1`` and ``M`` at ``\\hat{\\rho} = 0``.

# Arguments

  - `M`: Number of trials.
  - `rho`: Mean correlation between the returns of two different trials.
  - `C`: Correlation matrix of the returns of the trials. The diagonal does not enter the mean.

# Validation

  - `M >= 1`.
  - `rho <= 1`, and `rho > -1/(M - 1)` when `M > 1`.
  - `C` is square.

# Returns

  - `n::Number`: The implied number of independent trials.

# Examples

```jldoctest
julia> implied_independent_trials(10, 0.25)
7.75
```

# Related

  - [`deflated_sharpe_ratio`](@ref)
  - [`expected_max_sharpe_ratio`](@ref)

# References

  - $(ref_dict[:bailey2014deflated]) Equations 8 and 9.
"""
function implied_independent_trials(M::Integer, rho::Number)
    @argcheck(M >= 1, DomainError(M, "M must be at least one"))
    @argcheck(rho <= one(rho) && (isone(M) || rho > -inv(M - 1)),
              DomainError(rho, "rho must satisfy -1/(M - 1) < rho <= 1"))
    return rho + (one(rho) - rho) * M
end
function implied_independent_trials(C::MatNum)
    @argcheck(size(C, 1) == size(C, 2),
              DimensionMismatch("C must be square, got size $(size(C))"))
    M = size(C, 1)
    # One trial has no pair, and its mean correlation is zero.
    rho = (sum(C) - sum(LinearAlgebra.diag(C))) / max(M * (M - 1), 1)
    return implied_independent_trials(M, rho)
end

export probabilistic_sharpe_ratio, minimum_track_record_length, expected_max_sharpe_ratio,
       deflated_sharpe_ratio, implied_independent_trials
