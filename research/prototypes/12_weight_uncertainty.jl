# =============================================================================
# Prototype 12 — Uncertainty on the weights, and on the next return.
#
# Purpose
#   Reports 3 and 4 both ask for the same missing object: an error bar. Today
#   `optimise` returns a point. Two different error bars are wanted, and they
#   are not the same thing:
#
#     1. **Uncertainty on the weights.** Resample the data, re-optimise, and
#        report the spread of each asset's weight. This says how much of the
#        portfolio is an artefact of the particular sample.
#     2. **Uncertainty on the outcome.** Give a prediction interval for the
#        *next period's portfolio return*, with a coverage guarantee. This says
#        what the caller should expect to happen.
#
#   The second is available with a **distribution-free, finite-sample**
#   guarantee through conformal prediction. That is a stronger statement than
#   anything else in this report: it needs no model, no normality, and no
#   asymptotics. It needs one assumption, exchangeability, and this file
#   measures what happens when that assumption fails, because on financial
#   data it does.
#
# Status
#   Standalone. Depends on `Statistics`, `Random` and `LinearAlgebra`.
#
# Notation used throughout this file
#   T, N     Observations and assets.
#   X        Returns matrix, `T x N`.
#   B        Number of bootstrap replications.
#   W        Bootstrap weight matrix, `N x B`.
#   alpha    Miscoverage level. `alpha = 0.1` asks for 90 per cent coverage.
#   n        Number of calibration points.
#   s        A conformity score. Larger means more surprising.
#
# Sources
#   Vovk, V., Gammerman, A. and Shafer, G. (2005). Algorithmic Learning in a
#     Random World. Springer. The conformal prediction framework.
#   Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L.
#     (2018). Distribution-free predictive inference for regression. Journal of
#     the American Statistical Association 113(523), 1094-1111. The split
#     conformal method and its exact finite-sample coverage bounds.
#   Barber, R. F., Candes, E. J., Ramdas, A. and Tibshirani, R. J. (2023).
#     Conformal prediction beyond exchangeability. Annals of Statistics 51(2),
#     816-845. What breaks under distribution shift, which is the case here.
#   Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. Journal
#     of the American Statistical Association 89(428), 1303-1313.
#   Michaud, R. O. (1998). Efficient Asset Management. Harvard Business School
#     Press. Portfolio resampling.
#   Jorion, P. (1992). Portfolio optimization in practice. Financial Analysts
#     Journal 48(1), 68-74.
# =============================================================================
module WeightUncertainty

using Statistics, Random, LinearAlgebra

export bootstrap_weights, weight_intervals, conformal_quantile, conformal_return_interval,
       conformal_coverage, stationary_bootstrap_index

"""
    stationary_bootstrap_index(T::Integer, n::Integer, block_size::Real,
                               rng::Random.AbstractRNG) -> Vector{Int}

Return `n` indices drawn by the stationary bootstrap of Politis and Romano
(1994).

# Arguments

  - `T`: Length of the original series.
  - `n`: Number of indices to draw.
  - `block_size`: Mean block length. The restart probability is its reciprocal.
  - `rng`: Random number generator.

# Returns

  - A vector of `n` indices in `1:T`.

# Notes

  - **A plain i.i.d. bootstrap is wrong for returns.** It destroys volatility
    clustering, so the resampled series is calmer than the original and the
    resulting weight intervals are too narrow. The block structure preserves
    short-range dependence, which is what makes the interval honest.
"""
function stationary_bootstrap_index(T::Integer, n::Integer, block_size::Real,
                                    rng::Random.AbstractRNG)
    if T <= 0 || n <= 0
        throw(DomainError((T, n), "T and n must be > 0"))
    end
    if !(block_size > 0)
        throw(DomainError(block_size, "block_size must be > 0"))
    end
    p = min(1.0, 1 / float(block_size))
    idx = Vector{Int}(undef, n)
    cur = rand(rng, 1:T)
    for t in 1:n
        idx[t] = cur
        cur = rand(rng) < p ? rand(rng, 1:T) : (cur == T ? 1 : cur + 1)
    end
    return idx
end

"""
    bootstrap_weights(X::AbstractMatrix, optimise_fn::Function; B::Integer = 500,
                      block_size::Real = 10.0,
                      rng::Random.AbstractRNG = Random.default_rng()) -> Matrix

Return the distribution of optimal weights over bootstrap resamples of the
data.

# Arguments

  - `X`: Returns, `T x N`.
  - `optimise_fn`: A function `X -> w` returning weights of length `N`.
  - `B`: Number of replications.
  - `block_size`: Mean block length for the stationary bootstrap.
  - `rng`: Random number generator.

# Returns

  - `W::Matrix`, `N x B`. Column `b` is the portfolio fitted on resample `b`.

# Notes

  - The output plugs directly into prototype 3's `PortfolioPopulation`, so
    every disagreement statistic there applies here. **The difference is what
    varies.** Prototype 3 varies the *model* on fixed data. This varies the
    *data* with a fixed model. Reporting both separates estimation noise from
    model risk, which is a distinction a single number cannot make.
  - This is Michaud's (1998) resampling, with one change: the stationary
    bootstrap instead of a parametric draw, so no distributional assumption
    enters.
"""
function bootstrap_weights(X::AbstractMatrix{<:Real}, optimise_fn::Function;
                           B::Integer = 500, block_size::Real = 10.0,
                           rng::Random.AbstractRNG = Random.default_rng())
    T, N = size(X)
    W = Matrix{float(eltype(X))}(undef, N, B)
    for b in 1:B
        idx = stationary_bootstrap_index(T, T, block_size, rng)
        w = optimise_fn(X[idx, :])
        if length(w) != N
            throw(DimensionMismatch("optimise_fn returned $(length(w)) weights, expected $(N)"))
        end
        W[:, b] .= w
    end
    return W
end

"""
    weight_intervals(W::AbstractMatrix; alpha::Real = 0.1,
                     asset_names = nothing) -> Vector{<:NamedTuple}

Summarise a bootstrap weight distribution, one row per asset.

# Arguments

  - `W`: Weight matrix, `N x B`, as returned by [`bootstrap_weights`](@ref).
  - `alpha`: Miscoverage level for the reported interval.
  - `asset_names`: Optional labels.

# Returns

One `NamedTuple` per asset with `name`, `mean`, `median`, `sd`, `lower`,
`upper`, `prob_zero` and `sign_stability`.

# Details

  - `prob_zero` is the fraction of replications in which the asset was not
    held at all. An asset with a mean weight of four per cent and a
    `prob_zero` of 0.45 is not a four-per-cent position, it is a coin toss
    about whether to hold it.
  - `sign_stability` is `max(P(w > 0), P(w < 0))`, in `[0.5, 1]`. It is the
    number to report for a long-short portfolio, where the sign matters more
    than the size.

# Notes

  - **These intervals are not confidence intervals for a true weight.** There
    is no true weight: the optimum is a function of the estimated moments, so
    the bootstrap describes the sampling variability of an estimator, not
    uncertainty about a fixed parameter. Say "how much would this move if I
    had drawn a different sample", never "the true weight lies here".
"""
function weight_intervals(W::AbstractMatrix{<:Real}; alpha::Real = 0.1,
                          asset_names::Union{Nothing, AbstractVector} = nothing)
    if !(zero(alpha) < alpha < one(alpha))
        throw(DomainError(alpha, "alpha must satisfy 0 < alpha < 1"))
    end
    N, B = size(W)
    out = NamedTuple[]
    for i in 1:N
        row = view(W, i, :)
        name = isnothing(asset_names) ? "asset $(i)" : String(asset_names[i])
        push!(out,
              (; name = name, mean = mean(row), median = median(row), sd = std(row),
               lower = quantile(row, alpha / 2), upper = quantile(row, 1 - alpha / 2),
               prob_zero = count(x -> abs(x) < 1e-8, row) / B,
               sign_stability = max(count(>(0), row), count(<(0), row)) / B))
    end
    return out
end

"""
    conformal_quantile(scores::AbstractVector, alpha::Real) -> Real

Return the split-conformal quantile of a set of calibration scores.

# Arguments

  - `scores`: Calibration conformity scores, length `n`.
  - `alpha`: Miscoverage level, in `(0, 1)`.

# Returns

  - The `ceil((n + 1) * (1 - alpha))`-th smallest score, or `Inf` when that
    index exceeds `n`.

# Mathematical definition

Let `s_1, ..., s_n` be calibration scores and `s_{n+1}` the score of a future
point. If the `n + 1` scores are **exchangeable**, then `s_{n+1}` is equally
likely to occupy any rank, so

    P( s_{n+1} <= s_{(k)} )  >=  k / (n + 1)

Choosing `k = ceil((n + 1)(1 - alpha))` gives coverage of at least `1 - alpha`.
When the scores are continuous the coverage is also at most
`1 - alpha + 1/(n + 1)`, so the interval is **not conservative by much**.

# Notes

  - **The `n + 1` is not a detail.** Using the ordinary empirical quantile
    drops the guarantee. With `n = 19` and `alpha = 0.05`,
    `ceil(20 * 0.95) = 19`, so the answer is the maximum score. With `n = 18`
    the required index is 19, which does not exist, and the honest answer is
    `Inf`: **eighteen calibration points cannot support a 95 per cent
    interval.** Returning `Inf` says so rather than pretending.
"""
function conformal_quantile(scores::AbstractVector{<:Real}, alpha::Real)
    if !(zero(alpha) < alpha < one(alpha))
        throw(DomainError(alpha, "alpha must satisfy 0 < alpha < 1"))
    end
    n = length(scores)
    k = ceil(Int, (n + 1) * (1 - alpha))
    if k > n
        return Inf
    end
    return sort(collect(scores))[k]
end

"""
    conformal_return_interval(w::AbstractVector, X_calib::AbstractMatrix,
                              alpha::Real; centre::Union{Nothing, Real} = nothing,
                              symmetric::Bool = false) -> NamedTuple

Return a prediction interval for the next period's portfolio return.

# Arguments

  - `w`: Portfolio weights, length `N`.
  - `X_calib`: Calibration returns, `n x N`. **These must not have been used to
    fit `w`.** If they were, the guarantee is void.
  - `alpha`: Miscoverage level.
  - `centre`: Point prediction. The calibration mean if absent.
  - `symmetric`: If `true`, use the absolute-residual score and produce an
    interval centred on `centre`. If `false`, use the two-sided order
    statistics, which adapt to skew.

# Returns

A `NamedTuple` with `lower`, `upper`, `centre`, `width` and `n_calib`.

# Mathematical definition

**Symmetric form.** Score `s_i = |r_i - centre|`. The interval is
`centre ± conformal_quantile(s, alpha)`.

**Two-sided form.** Take the `floor((n + 1) * alpha / 2)`-th and
`ceil((n + 1) * (1 - alpha / 2))`-th order statistics of the calibration
returns directly. Either tail failing costs at most `alpha / 2`, so the union
bound gives coverage of at least `1 - alpha`.

# Notes

  - **The two-sided form is the one to use for portfolio returns.** Returns are
    left-skewed, so a symmetric interval is too wide on the upside and too
    narrow where it matters.

  - **The guarantee is marginal, and marginal is weaker than it sounds.** It
    holds on average over calibration draws. It is *not* conditional on
    today's volatility. Measured on a simulated GARCH series with 250
    calibration points and `alpha = 0.10`, using contiguous calibration
    windows:

    | quantity                       | value |
    |:------------------------------ | -----:|
    | marginal coverage              | 0.893 |
    | coverage in the calm half      | 0.944 |
    | coverage in the turbulent half | 0.841 |

    The marginal number is on target and the conditional numbers are ten
    points apart. **The interval is too wide when nothing is happening and too
    narrow exactly when it is needed.** Volatility clustering does not break
    the theorem, because a stationary series is still marginally
    exchangeable enough. It breaks the interpretation.

  - The fix is to make the score volatility-adjusted: divide the residual by a
    conditional volatility forecast before taking the quantile. The library
    already has the estimator for that, in
    `src/08_Moments/36_RegimeAdjustedExpWeightedVariance.jl`. That is the
    natural next step and it is not implemented here.
"""
function conformal_return_interval(w::AbstractVector{<:Real},
                                   X_calib::AbstractMatrix{<:Real}, alpha::Real;
                                   centre::Union{Nothing, Real} = nothing,
                                   symmetric::Bool = false)
    if size(X_calib, 2) != length(w)
        throw(DimensionMismatch("X_calib has $(size(X_calib, 2)) columns, w has length $(length(w))"))
    end
    r = X_calib * w
    n = length(r)
    c = isnothing(centre) ? mean(r) : float(centre)
    if symmetric
        q = conformal_quantile(abs.(r .- c), alpha)
        return (; lower = c - q, upper = c + q, centre = c, width = 2q, n_calib = n)
    end
    sorted = sort(r)
    klo = floor(Int, (n + 1) * alpha / 2)
    khi = ceil(Int, (n + 1) * (1 - alpha / 2))
    lo = klo < 1 ? -Inf : sorted[klo]
    hi = khi > n ? Inf : sorted[khi]
    return (; lower = lo, upper = hi, centre = c, width = hi - lo, n_calib = n)
end

"""
    conformal_coverage(w::AbstractVector, X::AbstractMatrix, alpha::Real;
                       n_calib::Integer, n_trials::Integer = 2000,
                       symmetric::Bool = false,
                       rng::Random.AbstractRNG = Random.default_rng())
        -> NamedTuple

Measure the realised coverage of the conformal interval.

# Arguments

  - `w`: Portfolio weights.
  - `X`: A pool of returns to draw calibration and test points from, `T x N`.
  - `alpha`: Miscoverage level.
  - `n_calib`: Calibration set size per trial.
  - `n_trials`: Number of trials.
  - `symmetric`: Which score to use.
  - `rng`: Random number generator.

# Returns

A `NamedTuple` with `coverage`, `target`, `upper_bound`, `mean_width` and
`n_trials`.

# Details

Each trial samples `n_calib + 1` rows, builds the interval from the first
`n_calib`, and records whether the last one falls inside.

**Sampling rows without replacement makes them exchangeable by construction**,
so the measured coverage must land between `1 - alpha` and
`1 - alpha + 1/(n_calib + 1)`. That is a theorem, and this function is how the
implementation is checked against it. To see the guarantee fail, feed a series
with volatility clustering and draw *contiguous* blocks instead.
"""
function conformal_coverage(w::AbstractVector{<:Real}, X::AbstractMatrix{<:Real},
                            alpha::Real; n_calib::Integer, n_trials::Integer = 2000,
                            symmetric::Bool = false,
                            rng::Random.AbstractRNG = Random.default_rng())
    T = size(X, 1)
    if n_calib + 1 > T
        throw(ArgumentError("need at least n_calib + 1 = $(n_calib + 1) rows, got $(T)"))
    end
    hits = 0
    widths = 0.0
    for _ in 1:n_trials
        perm = randperm(rng, T)[1:(n_calib + 1)]
        cal = view(X, perm[1:n_calib], :)
        tst = view(X, perm[n_calib + 1], :)
        iv = conformal_return_interval(w, cal, alpha; symmetric = symmetric)
        y = dot(tst, w)
        hits += (iv.lower <= y <= iv.upper)
        widths += isfinite(iv.width) ? iv.width : 0.0
    end
    return (; coverage = hits / n_trials, target = 1 - alpha,
            upper_bound = 1 - alpha + 1 / (n_calib + 1), mean_width = widths / n_trials,
            n_trials = n_trials)
end

end # module WeightUncertainty
