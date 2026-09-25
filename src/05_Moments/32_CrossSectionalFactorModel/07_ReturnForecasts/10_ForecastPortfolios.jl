"""
    forecast_hit_rate(x::AbstractVector{<:Real})

Return the fraction of the finite entries of a series that are positive.

The denominator counts the finite entries and not the length of the series, so the hit rate reads the same observations as the mean and the volatility beside it. [`exposure_ic_summary`](@ref) reads its hit rate the same way. A `NaN` marks an observation at which nothing was measured, so it is not a miss.

# Mathematical definition

```math
\\begin{align}
\\mathrm{hit} &= \\dfrac{\\left| \\left\\{ t \\in \\mathcal{F} : x_{t} > 0 \\right\\} \\right|}{\\left| \\mathcal{F} \\right|}\\,, \\qquad \\mathcal{F} = \\left\\{ t : x_{t} \\text{ is finite} \\right\\}\\,.
\\end{align}
```

Where:

  - ``x_{t}``: Entry ``t`` of the series.
  - ``\\mathcal{F}``: The finite entries. An empty ``\\mathcal{F}`` gives a `NaN`.

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
    Tf = eltype(x)
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

A series of weights has a path, so it gets the whole of [`performance_summary`](@ref). A spread of two cross-sectional means has no path, and a drawdown of it states nothing, so it gets these four figures.

Each figure reads the finite entries of the series, so a date that the evaluation could not score enters none of them.

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
 2. Take the mean of `s` and clamp it to the least and the greatest entry of `s`, giving `m`. An empty `s` gives a `NaN`. The clamp moves nothing in exact arithmetic. The rounded mean of a constant `s` can differ from its common value, and the clamp makes `m` equal that value.
 3. Take the corrected standard deviation of `s` about `m`, giving `v`. Fewer than two entries give a `NaN`, and a constant `s` gives exactly zero.
 4. Annualise `m` and `v` by `ppy`, and divide. A non-positive volatility gives a `NaN` ratio.
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
    Tf = eltype(x)
    s = x[isfinite.(x)]
    m = isempty(s) ? Tf(NaN) : clamp(sum(s) / length(s), extrema(s)...)
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

Mark the assets that one evaluation date can score, and build their sort key.

An asset enters the cross-section of a date only where its forecast and its target are both finite there. So an asset that the panel does not carry, and an asset whose forward window does not close, take no part in the statistic. The key is the forecast inside the mask and `Inf` outside it, which is the form [`cs_ordinal_ranks`](@ref) takes. The masked assets then sort to the end and take no rank.

# Mathematical definition

```math
\\begin{align}
k_{i} &= \\begin{cases} \\alpha_{ti} & i \\in \\mathcal{V}_{t}\\,,\\\\ \\infty & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``k_{i}``: Sort key of asset ``i``.
  - $(math_dict[:V_t_cs])
  - $(math_dict[:alpha_ti_fc])
  - $(math_dict[:y_ti_fwd])

# Arguments

  - `valid`: Mask to write, one entry per asset.
  - `key`: Sort key to write, one entry per asset.
  - `alpha`: Return Forecast history `observations × assets`, in return units.
  - `y`: Forward target `observations × assets`, on the same axis as `alpha`.
  - `t`: Row index of the observation.

# Returns

  - `n::Int`: Number of assets in the cross-section, ``\\left| \\mathcal{V}_{t} \\right|``.

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

Centring makes the book dollar neutral. The rescaling makes the book hold one unit long and one unit short whatever the spread of the signal is. That fixes the scale across dates and across members, so two forecasts are comparable on a statistic built from their books.

A signal that is identically zero once centred leaves the row as it was, which is zero. One asset and a flat cross-section give such a signal, and neither has a long side to fund a short side.

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

 1. Average the marked entries of `s` and clamp the average to the least and the greatest marked entry, giving `m`. The clamp moves nothing in exact arithmetic. The rounded average of a flat cross-section can differ from its common value, and without the clamp every entry would centre to the same sign and fill a one-sided book.
 2. Subtract `m` from each marked entry, giving `c`, which is zero outside the mask, and sum the absolute values into `g`.
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
    lo = Tf(Inf)
    hi = Tf(-Inf)
    for i in eachindex(valid)
        if !(valid[i])
            continue
        end
        m += s[i]
        lo = min(lo, s[i])
        hi = max(hi, s[i])
    end
    m = clamp(m / n, lo, hi)
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

Build the weights of the long-short portfolio that a Return Forecast states by itself.

The portfolio holds the forecast and nothing else. It reads no covariance, no constraint and no solver, so its return measures what the ordering of the forecast is worth. `kind` chooses what the ordering reads. `:rank` reads the ordinal rank in the cross-section, which measures the order alone, so one extreme forecast moves the book no more than one ordinary forecast does. `:zscore` reads the forecast value, so a conviction twice as large takes twice the weight.

[`forecast_centred_weights!`](@ref) centres and rescales both kinds, so both are dollar neutral at 200 % gross. A date with an empty cross-section, one asset or a flat signal holds no book.

This builder reads no threshold. A date whose cross-section is smaller than the `min_count` of an evaluation still gets its book, and [`forecast_portfolio`](@ref) decides which dates it scores.

# Algorithm

 1. Allocate `w`, one row per evaluation date and one column per asset, and fill it with zeros.
 2. For each date, mark its cross-section with [`forecast_cross_section!`](@ref), giving `valid`, `key` and the count `n`. An empty cross-section leaves the row at zero.
 3. Under `:rank`, rank `key` with [`cs_ordinal_ranks`](@ref), giving the signal `s`. Under `:zscore`, take `key` itself as `s`.
 4. Centre and rescale `s` into row `k` with [`forecast_centred_weights!`](@ref).

# Arguments

  - `alpha`: Return Forecast history `observations × assets`, in return units.
  - `y`: Forward target `observations × assets`, on the same axis as `alpha`. The builder reads only whether each entry is finite, which defines the cross-section of a date.
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
    Tf = promote_type(eltype(alpha), eltype(y))
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

Score the long-short portfolio that a Return Forecast states by itself.

The information coefficient states how well the forecast orders the cross-section. This verb states what that ordering earns as a book, which is the reading a caller acts on. The book reads no covariance, no constraint and no solver, so the forecast is the only thing it measures.

The return series is the return of a portfolio, so it gets the whole of [`performance_summary`](@ref) and not only a mean and a volatility. That is a Sharpe ratio and its standard error, a Sortino ratio, a Calmar ratio, a maximum drawdown and a conditional value at risk. `fe.ppy` annualises it, as the `periods_per_year` of that verb.

# Mathematical definition

```math
\\begin{align}
r_{j} &= \\begin{cases} \\sum_{i \\in \\mathcal{V}_{t_{j}}} w_{ji} \\, y_{t_{j} i} & \\left| \\mathcal{V}_{t_{j}} \\right| \\geq n_{\\min}\\,,\\\\ \\mathrm{NaN} & \\text{otherwise}\\,, \\end{cases}\\\\
\\tau_{j} &= \\begin{cases} \\sum_{i} \\left\\lvert w_{ji} - w_{j-1,i} \\right\\rvert & j > 1 \\text{ and } r_{j} \\text{ is finite}\\,,\\\\ \\mathrm{NaN} & \\text{otherwise}\\,. \\end{cases}
\\end{align}
```

Where:

  - ``r_{j}``: Portfolio return at evaluation date ``t_{j}``.
  - ``\\tau_{j}``: Turnover at evaluation date ``t_{j}``.
  - ``w_{ji}``: Weight of asset ``i`` at evaluation date ``t_{j}``, from [`forecast_portfolio_weights`](@ref).
  - ``n_{\\min}``: The threshold `fe.min_count`.
  - $(math_dict[:V_t_cs])
  - $(math_dict[:y_ti_fwd])
  - $(math_dict[:t_j_eval])

# The gaps of the return series

A date whose cross-section holds fewer than `fe.min_count` assets has no portfolio return. [`performance_summary`](@ref) states the Precomputed-returns contract, which needs a finite series. A `NaN` sorts last under `partialsort`, so the tail figure would be a finite wrong number and not a `NaN`. This verb therefore drops the gaps with `ret[isfinite.(ret)]` before the call, which is the remedy that docstring gives.

The gaps do not change the mean, the volatility or the two ratios built on them, because each is a sum over the dates that carry a number. They change `max_drawdown` and `calmar`, which read a path. The compressed path joins the date before a gap to the date after it, and the return that the book earned inside the gap is not in it. A loss inside the gap makes the compressed drawdown shallower than the drawdown of the held book, and a gain inside the gap can make it deeper. This verb applies no guard and no threshold, and reports the figure as it is. A caller who needs the drawdown of the held book lowers `min_count`, or contracts `w` with the target at the gap dates.

The book of a gap date stays in `w`, because [`forecast_portfolio_weights`](@ref) reads no threshold. Its turnover is a `NaN`, as its return is. The turnover of the next date reads the trade out of that book.

# Algorithm

 1. Build the weights of every evaluation date with [`forecast_portfolio_weights`](@ref), giving `w`.
 2. For each date, contract its weights with its target over the assets that carry both a finite forecast and a finite target, giving `ret`. A date with fewer than `fe.min_count` such assets gets a `NaN` instead.
 3. Take the turnover of `w` with [`calc_turnover`](@ref), and write a `NaN` into every date whose return is a `NaN`, so the two series read the same dates.
 4. Drop the gaps of `ret` and summarise the remainder with [`performance_summary`](@ref) at `periods_per_year = fe.ppy`.
 5. Read the hit rate of `ret` with [`forecast_hit_rate`](@ref), and the mean of the finite entries of the turnover.

# Arguments

  - `fe`: The [`ForecastEvaluationResult`](@ref) to score.
  - `kind`: `:rank` or `:zscore`, the two readings [`forecast_portfolio_weights`](@ref) states.

# Validation

  - The rules of [`forecast_portfolio_weights`](@ref).
  - [`performance_summary`](@ref) needs at least one finite return. If no date of `fe` reaches `fe.min_count`, that verb raises, and this one does not check first.

# Returns

  - `portfolio::NamedTuple`: `(; w, ret, turnover, summary, hit_rate, mean_turnover)`.

      + `w::MatNum`: Portfolio weights, `evaluation dates × assets`. A date below `fe.min_count` keeps its book.
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

The spread is the mean target of the assets in the top tail of the forecast, less the mean target of the assets in its bottom tail. It reads only the two tails, so the middle of the cross-section cannot move it. The middle is where a forecast carries the least information and the most noise.

The two cuts are symmetric, `q` and `1 - q`, and inclusive. An asset exactly on a threshold enters that tail, and a cross-section too small to separate the two tails puts one asset in both.

# Mathematical definition

```math
\\begin{align}
S &= \\dfrac{1}{\\left| \\mathcal{H} \\right|} \\sum_{i \\in \\mathcal{H}} y_{i} - \\dfrac{1}{\\left| \\mathcal{L} \\right|} \\sum_{i \\in \\mathcal{L}} y_{i}\\,,\\\\
\\mathcal{L} &= \\left\\{ i : a_{i} \\leq Q_{a}(q) \\right\\}\\,, \\qquad \\mathcal{H} = \\left\\{ i : a_{i} \\geq Q_{a}(1 - q) \\right\\}\\,.
\\end{align}
```

Where:

  - ``S``: The spread.
  - ``a_{i}``, ``y_{i}``: Forecast and target of asset ``i`` of the cross-section.
  - ``Q_{a}(p)``: The ``p``-quantile of the forecasts, from `Statistics.quantile`.
  - ``\\mathcal{L}``, ``\\mathcal{H}``: The bottom and the top tail. Neither is empty, because the least forecast is in ``\\mathcal{L}`` and the greatest is in ``\\mathcal{H}``.

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
    Tf = promote_type(eltype(av), eltype(yv))
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

Score the top-minus-bottom spread of a Return Forecast, one column per quantile.

[`forecast_tail_spread`](@ref) states the spread of one date. This verb runs it over the evaluation dates and summarises each column. A spread is a difference of two means and not a book. It has no weights and no turnover, and a drawdown of it states nothing, so it gets the four figures of [`forecast_series_summary`](@ref) and not the whole of [`performance_summary`](@ref).

# Algorithm

 1. For each evaluation date, mark its cross-section with [`forecast_cross_section!`](@ref) and take the forecasts and the targets of its assets as `av` and `yv`. A date with fewer than `fe.min_count` of them stays `NaN`.
 2. For each quantile, take the spread of that cross-section with [`forecast_tail_spread`](@ref), giving a column of `spread`.
 3. Summarise each column with [`forecast_series_summary`](@ref) at `fe.ppy`.

# Arguments

  - `fe`: The [`ForecastEvaluationResult`](@ref) to score.
  - `quantiles`: The tail fractions to cut. The other side of each cut is `1 - q`.

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
    Tf = promote_type(eltype(alpha), eltype(y))
    N = size(alpha, 2)
    Q = length(quantiles)
    spread = fill(Tf(NaN), length(dates), Q)
    key = Vector{Tf}(undef, N)
    valid = Vector{Bool}(undef, N)
    for (k, t) in enumerate(dates)
        n = forecast_cross_section!(valid, key, alpha, y, t)
        if n >= min_count
            av = key[valid]
            yv = view(y, t, valid)
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
