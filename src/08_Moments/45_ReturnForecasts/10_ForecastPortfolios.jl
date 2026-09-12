"""
    forecast_hit_rate(x::AbstractVector{<:Real})

Return the fraction of the finite entries of a series that are positive.

The denominator is the number of finite entries rather than the length of the series, so the hit rate is read against the same observations as the mean and the volatility that sit beside it. That is the convention this file holds, and it differs from [`exposure_ic_summary`](@ref), which counts a `NaN` as a miss: there the four figures are computed on the whole series, and here the summary of a portfolio is computed on the series with its gaps dropped, so a hit rate over the full length would disagree with the ratio printed next to it.

# Arguments

  - `x`: The series.

# Returns

  - `hit::Real`: The fraction of the finite entries that are positive, or `NaN` when none is finite.

# Related

  - [`forecast_portfolio`](@ref)
  - [`forecast_series_summary`](@ref)
  - [`exposure_ic_summary`](@ref)
"""
function forecast_hit_rate(x::AbstractVector{<:Real})
    Tf = real(eltype(x))
    n = 0
    h = 0
    for xi in x
        if isfinite(xi)
            n += 1
            if xi > zero(xi)
                h += 1
            end
        end
    end
    return iszero(n) ? Tf(NaN) : Tf(h) / Tf(n)
end
"""
    forecast_series_summary(x::AbstractVector{<:Real}, ppy::Real)

Return the annualised mean, volatility and information ratio of a series, and its hit rate.

This is the summary a quantity that is not a portfolio earns. A series of weights has a path, so it earns the whole of [`performance_summary`](@ref); a spread of two cross-sectional means has none, so a drawdown of it would state nothing and these four figures are what remain.

Each figure is read over the entries at which the series is finite, so a date the evaluation could not score enters none of them.

# Mathematical definition

```math
\\begin{align}
\\mathrm{ann\\_mean} &= p \\, \\overline{x}\\,, \\qquad \\mathrm{ann\\_vol} = \\sqrt{p} \\, \\sigma_{x}\\,, \\qquad \\mathrm{ann\\_ir} = \\dfrac{\\mathrm{ann\\_mean}}{\\mathrm{ann\\_vol}}\\,.
\\end{align}
```

Where:

  - ``\\overline{x}`` and ``\\sigma_{x}``: Mean and standard deviation of the finite entries of ``x``, the second with one degree of freedom removed.
  - ``p``: Periods per year.

# Algorithm

 1. Drop the non-finite entries of `x`, giving `s`.
 2. Take the mean of `s`, giving `m`. An empty `s` gives a `NaN`.
 3. Take its corrected standard deviation, giving `v`. Fewer than two entries give a `NaN`.
 4. Annualise both by `ppy`, and divide. A non-positive volatility gives a `NaN` ratio.
 5. Read the hit rate of `x` with [`forecast_hit_rate`](@ref).

# Arguments

  - `x`: The series.
  - `ppy`: Periods per year. `1` reports the figures per period.

# Returns

  - `summary::NamedTuple`: `(; ann_mean, ann_vol, ann_ir, hit_rate)`.

# Related

  - [`forecast_quantile_spread`](@ref)
  - [`forecast_hit_rate`](@ref)
  - [`performance_summary`](@ref)
"""
function forecast_series_summary(x::AbstractVector{<:Real}, ppy::Real)
    Tf = real(eltype(x))
    s = x[isfinite.(x)]
    m = isempty(s) ? Tf(NaN) : sum(s) / length(s)
    v = length(s) > 1 ? Statistics.std(s; mean = m, corrected = true) : Tf(NaN)
    ann_mean = m * ppy
    ann_vol = v * sqrt(Tf(ppy))
    ann_ir = ann_vol > zero(Tf) ? ann_mean / ann_vol : Tf(NaN)
    return (; ann_mean = ann_mean, ann_vol = ann_vol, ann_ir = ann_ir,
            hit_rate = forecast_hit_rate(x))
end
"""
    forecast_cross_section!(valid::AbstractVector{Bool}, key::AbstractVector{<:Real},
                            alpha::AbstractMatrix{<:Real}, y::AbstractMatrix{<:Real},
                            t::Integer)

Mark the assets one evaluation date can be scored on, and build their sort key.

An asset enters the cross-section of a date only where both the forecast and the target are finite there, so an asset the panel does not carry, and one whose forward window does not close, take no part in the statistic. The key is the forecast inside the mask and `Inf` outside it, which is the form [`cs_ordinal_ranks`](@ref) takes: the mask then sorts to the end and takes no rank.

# Arguments

  - `valid`: Mask to write, one entry per asset.
  - `key`: Sort key to write, one entry per asset.
  - `alpha`: Return Forecast history `observations × assets`, in return units.
  - `y`: Forward target `observations × assets`, on the same axis as `alpha`.
  - `t`: Row index of the observation.

# Returns

  - `n::Int`: Number of assets in the cross-section.

# Related

  - [`forecast_portfolio_weights`](@ref)
  - [`forecast_quantile_spread`](@ref)
  - [`cs_ordinal_ranks`](@ref)
"""
function forecast_cross_section!(valid::AbstractVector{Bool}, key::AbstractVector{<:Real},
                                 alpha::AbstractMatrix{<:Real}, y::AbstractMatrix{<:Real},
                                 t::Integer)
    Tf = eltype(key)
    n = 0
    for i in eachindex(valid)
        v = isfinite(alpha[t, i]) && isfinite(y[t, i])
        valid[i] = v
        key[i] = v ? Tf(alpha[t, i]) : Tf(Inf)
        n += v
    end
    return n
end
"""
    forecast_centred_weights!(w::AbstractMatrix{<:Real}, k::Integer,
                              s::AbstractVector{<:Real}, valid::AbstractVector{Bool},
                              n::Integer)

Centre one cross-sectional signal and write it into a row of the weight matrix at 200 % gross.

Centring makes the book dollar neutral, and the rescaling makes it carry one unit long and one unit short whatever the spread of the signal is. That fixes the scale across dates and across members, which is what makes two forecasts comparable on the statistic built from it.

A signal that is identically zero once centred — one asset, or a flat cross-section — leaves the row at whatever it held, which is zero: there is no long side to fund a short one.

# Mathematical definition

```math
\\begin{align}
c_{i} &= s_{i} - \\dfrac{1}{n} \\sum_{j \\in \\mathcal{V}} s_{j}\\,, \\qquad i \\in \\mathcal{V}\\,,\\\\
w_{i} &= \\dfrac{2 c_{i}}{\\sum_{j \\in \\mathcal{V}} \\left\\lvert c_{j} \\right\\rvert}\\,.
\\end{align}
```

Where:

  - ``\\mathcal{V}``: The assets of the cross-section, the `true` entries of `valid`.
  - ``s_{i}``: The signal of asset ``i``.
  - ``c_{i}``: The centred signal, zero outside ``\\mathcal{V}``.
  - ``w_{i}``: The weight, zero outside ``\\mathcal{V}``.

# Algorithm

 1. Average the marked entries of `s`, giving `m`.
 2. Subtract `m` from each marked entry, giving `c`, which is zero outside the mask, and accumulate the absolute values into `g`.
 3. Write `2c / g` into row `k` of `w`. A zero `g` writes nothing.

# Arguments

  - `w`: Weight matrix to write, `evaluation dates × assets`.
  - `k`: Row of `w` to write.
  - `s`: The signal, one entry per asset.
  - `valid`: Mask, one entry per asset.
  - `n`: Number of marked entries.

# Returns

  - `nothing`.

# Related

  - [`forecast_portfolio_weights`](@ref)
  - [`forecast_cross_section!`](@ref)
"""
function forecast_centred_weights!(w::AbstractMatrix{<:Real}, k::Integer,
                                   s::AbstractVector{<:Real}, valid::AbstractVector{Bool},
                                   n::Integer)
    Tf = eltype(w)
    m = zero(Tf)
    for i in eachindex(valid)
        m += valid[i] ? s[i] : zero(Tf)
    end
    m /= n
    g = zero(Tf)
    for i in eachindex(valid)
        g += valid[i] ? abs(s[i] - m) : zero(Tf)
    end
    if g > zero(Tf)
        for i in eachindex(valid)
            w[k, i] = valid[i] ? 2 * (s[i] - m) / g : zero(Tf)
        end
    end
    return nothing
end
"""
    forecast_portfolio_weights(alpha::AbstractMatrix{<:Real}, y::AbstractMatrix{<:Real},
                               dates::AbstractVector{<:Integer}, kind::Symbol)

Build the weights of the long-short portfolio a Return Forecast states on its own.

The portfolio holds the forecast and nothing else: it takes no covariance, no constraint and no solver, so what it earns is what the ordering of the forecast is worth. `kind` chooses what that ordering is read off. `:rank` reads the ordinal rank of the cross-section, which measures the order alone and lets one extreme forecast move the book no more than one ordinary forecast does. `:zscore` reads the forecast value, which lets a conviction that is twice as large take twice the weight.

Both are centred and rescaled by [`forecast_centred_weights!`](@ref), so both are dollar neutral at 200 % gross.

# Algorithm

 1. Allocate `w`, one row per evaluation date and one column per asset, and fill it with zeros.
 2. For each date, mark its cross-section with [`forecast_cross_section!`](@ref), giving `valid`, `key` and the count `n`. An empty cross-section leaves the row at zero.
 3. Under `:rank`, rank `key` with [`cs_ordinal_ranks`](@ref), giving the signal `s`. Under `:zscore`, take `key` itself as `s`.
 4. Centre and rescale `s` into row `k` with [`forecast_centred_weights!`](@ref).

# Arguments

  - `alpha`: Return Forecast history `observations × assets`, in return units.
  - `y`: Forward target `observations × assets`, on the same axis as `alpha`. It is read for its finiteness alone, which is what defines the cross-section of a date.
  - `dates`: Row indices of the observations to build a portfolio at.
  - `kind`: `:rank` or `:zscore`.

# Validation

  - `kind` is `:rank` or `:zscore`. Raises a [`ConflictingArgumentError`](@ref).

# Returns

  - `w::MatNum`: Portfolio weights, `evaluation dates × assets`.

# Examples

```jldoctest
julia> alpha = [1.0 2.0 3.0; 3.0 2.0 1.0];

julia> PortfolioOptimisers.forecast_portfolio_weights(alpha, alpha, [1, 2], :zscore)
2×3 Matrix{Float64}:
 -1.0  0.0   1.0
  1.0  0.0  -1.0
```

# Related

  - [`forecast_portfolio`](@ref)
  - [`forecast_cross_section!`](@ref)
  - [`forecast_centred_weights!`](@ref)
  - [`cs_ordinal_ranks`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function forecast_portfolio_weights(alpha::AbstractMatrix{<:Real},
                                    y::AbstractMatrix{<:Real},
                                    dates::AbstractVector{<:Integer}, kind::Symbol)
    @argcheck(kind in (:rank, :zscore),
              ConflictingArgumentError("kind must be :rank or :zscore, got :$(kind)"))
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)))
    N = size(alpha, 2)
    w = zeros(Tf, length(dates), N)
    key = Vector{Tf}(undef, N)
    valid = Vector{Bool}(undef, N)
    for (k, t) in enumerate(dates)
        n = forecast_cross_section!(valid, key, alpha, y, t)
        if !iszero(n)
            s = kind === :rank ? cs_ordinal_ranks(key, valid) : key
            forecast_centred_weights!(w, k, s, valid, n)
        end
    end
    return w
end
"""
    forecast_portfolio(fe::ForecastEvaluationResult; kind::Symbol = :rank) -> NamedTuple

Score the long-short portfolio a Return Forecast states on its own.

This is the second of the two readings of a forecast. The information coefficient states how well the forecast orders the cross-section; this verb states what that ordering is worth once it is held as a book, which is the reading a caller acts on. The portfolio takes no covariance, no constraint and no solver, so nothing but the forecast itself is being measured.

The return series **is** a portfolio, so it earns the whole of [`performance_summary`](@ref) rather than a mean and a volatility: a Sharpe ratio and its standard error, a Sortino ratio, a Calmar ratio, a maximum drawdown and a conditional value at risk. `fe.ppy` is what annualises it, mapped onto that verb's `periods_per_year`.

# The drawdown is of the compressed path

A date whose cross-section carries fewer than `fe.min_count` assets has no portfolio return, and [`performance_summary`](@ref) states the Precomputed-returns contract: its series must be finite, because a `NaN` orders last under `partialsort` and the tail figure then answers a finite wrong number rather than a `NaN`. The gaps are therefore dropped with `ret[isfinite.(ret)]` before the call, which is the remedy that docstring prescribes.

The mean, the volatility and the two ratios built on them are unaffected by that, because each is a sum over the dates that carry a number. `max_drawdown` and `calmar` are **not**: they read a path, and the compressed path joins the date before a gap to the date after it, so a drawdown that opened inside the gap is invisible and one that spans it is understated. No guard is applied and no threshold is imposed — the figure is reported as it stands, and this paragraph is the caveat that rides beside it. A caller who needs the drawdown of the real path lowers `min_count`, or reads `ret` and builds it.

# Algorithm

 1. Build the weights of every evaluation date with [`forecast_portfolio_weights`](@ref), giving `w`.
 2. For each date, contract its weights with its target over the assets that carry both a finite forecast and a finite target, giving `ret`. A date with fewer than `fe.min_count` such assets gets a `NaN` instead.
 3. Take the turnover of `w` with [`calc_turnover`](@ref), and write a `NaN` into every date whose return is one, so the two series read against the same dates.
 4. Drop the gaps of `ret` and summarise the remainder with [`performance_summary`](@ref) at `periods_per_year = fe.ppy`.
 5. Read the hit rate of `ret` with [`forecast_hit_rate`](@ref), and the mean of the finite entries of the turnover.

# Arguments

  - `fe`: The [`ForecastEvaluationResult`](@ref) to score.
  - `kind`: `:rank` or `:zscore`, the two readings [`forecast_portfolio_weights`](@ref) states.

# Validation

  - The rules of [`forecast_portfolio_weights`](@ref).
  - [`performance_summary`](@ref) needs at least one finite return, so a `fe` no date of which reaches `fe.min_count` raises from that verb rather than from this one.

# Returns

  - `portfolio::NamedTuple`: `(; w, ret, turnover, summary, hit_rate, mean_turnover)`.

      + `w::MatNum`: Portfolio weights, `evaluation dates × assets`.
      + `ret::VecNum`: Portfolio target return, one entry per evaluation date, `NaN` below `fe.min_count`.
      + `turnover::VecNum`: Turnover, one entry per evaluation date, `NaN` at the first and wherever `ret` is.
      + `summary::PerformanceSummaryResult`: The summary of `ret` with its gaps dropped.
      + `hit_rate::Real`: The fraction of the finite entries of `ret` that are positive.
      + `mean_turnover::Real`: The mean of the finite entries of `turnover`.

# Related

  - [`forecast_portfolio_weights`](@ref)
  - [`forecast_quantile_spread`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`ForecastEvaluationResult`](@ref)
  - [`performance_summary`](@ref)
  - [`calc_turnover`](@ref)
"""
function forecast_portfolio(fe::ForecastEvaluationResult; kind::Symbol = :rank)
    alpha::AbstractMatrix{<:Real} = fe.alpha
    y::AbstractMatrix{<:Real} = fe.y
    dates::AbstractVector{<:Integer} = fe.dates
    min_count::Integer = fe.min_count
    ppy::Real = fe.ppy
    w = forecast_portfolio_weights(alpha, y, dates, kind)
    Tf = eltype(w)
    N = size(alpha, 2)
    ret = fill(Tf(NaN), length(dates))
    for (k, t) in enumerate(dates)
        n = 0
        s = zero(Tf)
        for i in 1:N
            if isfinite(alpha[t, i]) && isfinite(y[t, i])
                n += 1
                s += w[k, i] * y[t, i]
            end
        end
        ret[k] = n >= min_count ? s : Tf(NaN)
    end
    turnover = calc_turnover(w)
    turnover[.!isfinite.(ret)] .= Tf(NaN)
    ftn = turnover[isfinite.(turnover)]
    return (; w = w, ret = ret, turnover = turnover,
            summary = performance_summary(ret[isfinite.(ret)]; periods_per_year = ppy),
            hit_rate = forecast_hit_rate(ret),
            mean_turnover = isempty(ftn) ? Tf(NaN) : sum(ftn) / length(ftn))
end
"""
    forecast_tail_spread(av::AbstractVector{<:Real}, yv::AbstractVector{<:Real}, q::Real)

Return the top-minus-bottom target of one cross-section at one quantile.

The spread is the mean target of the assets the forecast puts in its top tail, less the mean target of those it puts in its bottom tail. It answers a narrower question than a weighted portfolio does, and a more robust one: only the two tails are read, so the middle of the cross-section — where a forecast carries the least information and the most noise — cannot move the answer.

The quantile is read symmetrically, `q` against `1 - q`, and the cut is inclusive on both sides, so an asset that sits exactly on a threshold enters that tail and a cross-section too small to separate the two tails puts the same asset in both.

# Arguments

  - `av`: Forecasts of the cross-section, one entry per asset in it.
  - `yv`: Targets of the same assets, in the same order.
  - `q`: Tail fraction, in `(0, 0.5]`.

# Returns

  - `s::Real`: The spread.

# Related

  - [`forecast_quantile_spread`](@ref)
  - [`forecast_cross_section!`](@ref)
"""
function forecast_tail_spread(av::AbstractVector{<:Real}, yv::AbstractVector{<:Real},
                              q::Real)
    Tf = promote_type(real(eltype(av)), real(eltype(yv)))
    lo = Statistics.quantile(av, q)
    hi = Statistics.quantile(av, one(q) - q)
    sl = zero(Tf)
    sh = zero(Tf)
    nl = 0
    nh = 0
    for i in eachindex(av)
        sl += av[i] <= lo ? yv[i] : zero(Tf)
        nl += av[i] <= lo
        sh += av[i] >= hi ? yv[i] : zero(Tf)
        nh += av[i] >= hi
    end
    return sh / nh - sl / nl
end
"""
    forecast_quantile_spread(fe::ForecastEvaluationResult;
                             quantiles = (0.1,)) -> NamedTuple

Score the top-minus-bottom spread of a Return Forecast, one entry per quantile.

[`forecast_tail_spread`](@ref) states what the spread of one date is; this verb runs it over the evaluation dates and summarises each column. A spread is a difference of two means and not a book, so it carries no weights and no turnover, and a drawdown of it means nothing: it gets the three figures of [`forecast_series_summary`](@ref) rather than the whole of [`performance_summary`](@ref).

# Algorithm

 1. For each evaluation date, mark its cross-section with [`forecast_cross_section!`](@ref) and collect the forecasts and targets of its assets into `av` and `yv`. A date with fewer than `fe.min_count` of them is left at `NaN`.
 2. For each quantile, take the spread of that cross-section with [`forecast_tail_spread`](@ref), giving a column of `spread`.
 3. Summarise each column with [`forecast_series_summary`](@ref) at `fe.ppy`.

# Arguments

  - `fe`: The [`ForecastEvaluationResult`](@ref) to score.
  - `quantiles`: The tail fractions to cut. Each is read against `1 - q` on the other side.

# Validation

  - `!isempty(quantiles)`. Raises an [`IsEmptyError`](@ref).
  - Every entry of `quantiles` is finite and in `(0, 0.5]`. Raises a `DomainError`.

# Returns

  - `spreads::NamedTuple`: `(; spread, ann_mean, ann_vol, ann_ir, hit_rate)`.

      + `spread::MatNum`: The spread, `evaluation dates × quantiles`, `NaN` below `fe.min_count`.
      + `ann_mean::VecNum`: Annualised mean of each column.
      + `ann_vol::VecNum`: Annualised standard deviation of each column.
      + `ann_ir::VecNum`: Their ratio.
      + `hit_rate::VecNum`: The fraction of the finite dates of each column that are positive.

# Related

  - [`forecast_tail_spread`](@ref)
  - [`forecast_series_summary`](@ref)
  - [`forecast_portfolio`](@ref)
  - [`forecast_evaluation`](@ref)
  - [`ForecastEvaluationResult`](@ref)
"""
function forecast_quantile_spread(fe::ForecastEvaluationResult; quantiles = (0.1,))
    @argcheck(!isempty(quantiles), IsEmptyError("quantiles cannot be empty"))
    @argcheck(all(q -> isfinite(q) && zero(q) < q <= 0.5, quantiles),
              DomainError(quantiles, "every quantile must be finite and in (0, 0.5]"))
    alpha::AbstractMatrix{<:Real} = fe.alpha
    y::AbstractMatrix{<:Real} = fe.y
    dates::AbstractVector{<:Integer} = fe.dates
    min_count::Integer = fe.min_count
    ppy::Real = fe.ppy
    Tf = promote_type(real(eltype(alpha)), real(eltype(y)))
    N = size(alpha, 2)
    Q = length(quantiles)
    spread = fill(Tf(NaN), length(dates), Q)
    key = Vector{Tf}(undef, N)
    valid = Vector{Bool}(undef, N)
    for (k, t) in enumerate(dates)
        n = forecast_cross_section!(valid, key, alpha, y, t)
        if n >= min_count
            av = key[valid]
            yv = Tf[y[t, i] for i in 1:N if valid[i]]
            for (j, q) in enumerate(quantiles)
                spread[k, j] = forecast_tail_spread(av, yv, q)
            end
        end
    end
    summaries = [forecast_series_summary(view(spread, :, j), ppy) for j in 1:Q]
    return (; spread = spread, ann_mean = [s.ann_mean for s in summaries],
            ann_vol = [s.ann_vol for s in summaries],
            ann_ir = [s.ann_ir for s in summaries],
            hit_rate = [s.hit_rate for s in summaries])
end

export forecast_portfolio, forecast_quantile_spread
