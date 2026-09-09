#=
# Reading a Return Forecast before an optimiser sees it

A [`CrossSectionalFactorPrior`](@ref) takes a **Return Forecast** — an opinion about which
assets are about to do better than the factor model says they should — and folds it into the
expected returns an optimiser then maximises. The optimiser takes that opinion whatever it is
worth. It has no way to tell a forecast that ranks the cross-section from one that ranks it no
better than a coin.

This page is the reading that happens **before** that. It scores a Return Forecast out of
sample, on its own, against what actually happened next, and it puts two forecasts side by side
so the reader can see the difference between one that works and one that does not.

The reading is an **order**, not a list. A caller does not read thirty numbers; they ask three
questions and stop the moment one answers no:

 1. **Does it rank?** Is the forecast's ordering of the cross-section related to the ordering
    that realised? That is the information coefficient.
 2. **Does the ranking pay?** Turn the forecast into a book and see what the book earned, and
    what it cost in turnover.
 3. **Is the magnitude right?** A forecast can order the cross-section perfectly and still be
    ten times too large. That is the calibration slope.

Two further questions are worth asking only after all three answer yes — how long the edge
lasts, and whether it is a factor exposure in disguise — and the page reaches them last.

Everything below runs on the synthetic Asset Panel of the
[deep dive](05_Cross_Sectional_Factor_Model.md), at the same seed, so the alpha the panel was
drawn from is known before the estimator runs and every claim is checkable against it.

!!! warning "These are synthetic numbers"
    The panel plants a constant per-asset alpha inside a field no Factor Exposure reads, and the
    Sharpe ratios below are what a forecast of a planted constant earns on a frictionless
    long-short book. They are a property of the fixture, not a claim about markets. What
    transfers is the **shape** of the reading, and the gap between the two forecasts.
=#

using PortfolioOptimisers, StableRNGs, Statistics, LinearAlgebra, Dates, PrettyTables,
      DataFrames, StatsPlots, GraphRecipes

numfmt = (v, i, j) -> begin
    return isa(v, AbstractFloat) ? round(v; sigdigits = 4) : v
end;

#=
## 1. The panel, and two convictions

The generator is the deep dive's, unchanged and at the same seed. Read
[§1 there](05_Cross_Sectional_Factor_Model.md) for what each Panel Field is; here it is just the
input. The one field to keep in mind is `signal`: it is the assets' true alpha plus a little
noise, and **no Factor Exposure reads it**, so a forecast built on it carries a part the factor
model does not span.
=#

function synthetic_panel(; T = 500, N = 80, seed = 661_001)
    rng = StableRNG(seed)
    industries = ["Energy", "Financials", "Health Care", "Technology"]
    Kind = length(industries)
    ind = rand(rng, 1:Kind, N)
    beta = 1.0 .+ 0.35 .* randn(rng, N)
    Ltrue = randn(rng, N, 4)
    alpha = 0.0004 .* randn(rng, N)
    onehot = Float64[ind[i] == k for i in 1:N, k in 1:Kind]
    Btrue = hcat(beta, onehot, Ltrue)
    ftrue = hcat(0.009 .* randn(rng, T), 0.005 .* randn(rng, T, Kind),
                 0.004 .* randn(rng, T, 4))
    ivol = 0.008 .+ 0.012 .* rand(rng, N)
    X = ftrue * transpose(Btrue) + randn(rng, T, N) .* transpose(ivol) .+ transpose(alpha)
    logcap = 20.0 .+ 1.4 .* transpose(view(Ltrue, :, 1)) .+
             0.3 .* cumsum(randn(rng, T, N); dims = 1) ./ sqrt(T)
    mcap = exp.(logcap)
    shares = mcap ./ 50.0
    fields = ["market_cap" => mcap,
              "book_equity" =>
                  mcap .* exp.(-0.7 .+ 0.5 .* transpose(view(Ltrue, :, 2)) .+
                               0.05 .* randn(rng, T, N)),
              "net_income_ttm" =>
                  mcap .* (0.05 .+ 0.02 .* transpose(view(Ltrue, :, 3)) .+
                           0.004 .* randn(rng, T, N)), "adj_shares_outstanding" => shares,
              "adj_volume" =>
                  shares .* exp.(-4.5 .+ 0.6 .* transpose(view(Ltrue, :, 4)) .+
                                 0.15 .* randn(rng, T, N)),
              ## The assets' own alpha, plus noise. No Factor Exposure reads it.
              "signal" => transpose(alpha) .+ 0.0002 .* randn(rng, T, N)]
    listed = [i <= N ÷ 5 ? rand(rng, 20:120) : 1 for i in 1:N]
    amsk = [t >= listed[i] for t in 1:T, i in 1:N]
    for (_, a) in fields
        a[.!amsk] .= NaN
    end
    inputs = [[NumericPanelInput(; name = n, vals = a, alg = ForwardPanelFill(; val = 0.0))
               for (n, a) in fields]
              CategoricalPanelInput(; name = "industry",
                                    vals = repeat(reshape(industries[ind], 1, N), T, 1),
                                    levels = industries)]
    days = filter(d -> Dates.dayofweek(d) <= 5,
                  Date(2015, 1, 1):Day(1):(Date(2015, 1, 1) + Day(2 * T + 10)))[1:T]
    return ReturnsResult(; nx = ["A" * lpad(i, 3, '0') for i in 1:N],
                         X = ifelse.(amsk, X, NaN), ts = days,
                         pnl = asset_panel(inputs; amsk = amsk, emsk = amsk))
end

rd = synthetic_panel()

#=
The evaluation scores a forecast against the **idiosyncratic** return — the part of an asset's
return the factor model does not explain — because that is the component a Return Forecast
actually forecasts inside a [`CrossSectionalFactorPrior`](@ref). So the factor model has to be
fitted first, and the block it leaves on the prior is what every verb below reads.
=#

style(d) = CompositeExposure(; descriptors = [d], family = "style")
factors = ["market" =>
               CompositeExposure(; descriptors = [EWMarketBeta()], outlier = nothing,
                                 scoring = nothing, family = "market"),
           "industry" => OneHotExposure(; field = "industry", family = "industry"),
           "size" => style(LogMarketCap()), "value" => style(BookToPrice()),
           "earnings_yield" => style(EarningsToPrice()),
           "liquidity" => style(EWShareTurnover())]

signal_scores = DescriptorScores(; descriptors = [Passthrough(; field = "signal")],
                                 outlier = nothing, scoring = nothing)

pe = CrossSectionalFactorPrior(; factors = factors, families = ["industry" => nothing],
                               neutralise = ["style" => "industry"],
                               wa = BlendedInverseVarianceWeights(; lambda = 0.5),
                               rfe = FixedWeightedReturnForecast(; scores = signal_scores,
                                                                 scale = 1.0), lambda = 1.0,
                               c = 1.0)

pr = prior(pe, rd)
csfm = pr.rr

#=
Now the two convictions. Both are Return Forecast Estimators, and they are two of the four the
library ships, so they differ in **how** they turn traits into a forecast as well as in **which**
traits they read:

  - The **signal composite** is a [`FixedWeightedReturnForecast`](@ref) over the `signal` field.
    It says "this field is the alpha, at scale one", and it composes its Descriptor scores
    observation by observation, so it publishes a history of its own.
  - The **trait regression** is a [`TargetReturnForecast`](@ref) that regresses the forward
    idiosyncratic return on four style traits the factor model already spans — size, value,
    earnings yield and liquidity. It is the honest bear case: a forecast built out of the very
    exposures the model already charges for.

One of these is a forecast of something real and the other is not, and the whole point of the
page is that the reading says which without anybody being told.
=#

signal_forecast = FixedWeightedReturnForecast(; scores = signal_scores, scale = 1.0)

trait_forecast = TargetReturnForecast(;
                                      scores = DescriptorScores(;
                                                                descriptors = [LogMarketCap(),
                                                                               BookToPrice(),
                                                                               EarningsToPrice(),
                                                                               EWShareTurnover()]),
                                      tgt = LinearModel(), horizon = 5, lag = 1)

#=
## 2. The pairing

[`forecast_evaluation`](@ref) is the one call that turns an estimator into something scorable. It
fits the member, builds its history, builds the **forward target** the history is scored against,
and hands back a [`ForecastEvaluationResult`](@ref) carrying the two matrices, the observations it
can be scored at, and the parameters that produced all three.

Five parameters shape the pairing, and they are the vocabulary of the rest of the page:

  - `horizon` — how many observations forward the target accumulates over.
  - `lag` — how many observations after the forecast the target window opens, so `lag = 1` means
    the forecast is acted on at the next observation and never sees its own target.
  - `step` — how many observations between two evaluation dates.
  - `min_count` — the fewest assets a cross-section must carry before a statistic is taken of it.
  - `ppy` — periods per year, which is what annualises the book statistics further down.

The Result is deliberately **lean**: it holds the forecast, the target, the dates and the
parameters, and every statistic is a verb over it rather than a field on it. That is what lets a
caller re-parameterise one statistic without re-running the fit.
=#

horizon, lag, step, min_count, ppy = 5, 1, 5, 10, 252

raw_signal = forecast_evaluation(signal_forecast, rd, csfm; horizon = horizon, lag = lag,
                                 step = step, min_count = min_count, ppy = ppy)

raw_trait = forecast_evaluation(trait_forecast, rd, csfm; horizon = horizon, lag = lag,
                                step = step, min_count = min_count, ppy = ppy)

pretty_table(DataFrame("Forecast" => ["signal composite", "trait regression"],
                       "Block rows" =>
                           [size(raw_signal.alpha, 1), size(raw_trait.alpha, 1)],
                       "First scorable row" =>
                           [first(raw_signal.dates), first(raw_trait.dates)],
                       "Evaluation dates" =>
                           [length(raw_signal.dates), length(raw_trait.dates)]);
             title = "The same block, two different warm-ups")

#=
### Two forecasts have to be aligned before they can be compared

The table shows the one thing that bites when a caller compares members rather than
re-parameterising one. The signal composite composes a score at every observation, so it is
scorable from the first row of the block. The trait regression **publishes no history** — it fits
one cross-section over the whole sample and states one row — so
[`forecast_history`](@ref) builds its path by refitting it along the evaluation grid, and a
regression that calibrates itself needs a warm-up before its first fit means anything. The two
forecasts therefore start at different observations, and a comparison of statistics taken over
different samples is not a comparison.

The fix is to cut both forecasts to the sample they share. Because the Result carries the
forecast and the target as plain matrices, that is a read and a blank rather than a re-fit: the
**bare** method of [`forecast_evaluation`](@ref) takes the two matrices directly, which is the
bottom of the `bare arrays → Result → Estimator` hierarchy the whole family is built on.
=#

first_scorable(h) = findfirst(t -> any(isfinite, view(h, t, :)), axes(h, 1))
start = max(first_scorable(raw_signal.alpha), first_scorable(raw_trait.alpha))

## The forward target is the same for both: same block, same `horizon`, same `lag`.
y = raw_signal.y
function align(h)
    return [t >= start ? h[t, i] : convert(eltype(h), NaN)
            for t in axes(h, 1), i in axes(h, 2)]
end

fe_signal = forecast_evaluation(align(raw_signal.alpha), y; horizon = horizon, lag = lag,
                                step = step, min_count = min_count, ppy = ppy)
fe_trait = forecast_evaluation(align(raw_trait.alpha), y; horizon = horizon, lag = lag,
                               step = step, min_count = min_count, ppy = ppy)

pretty_table(DataFrame("Common first row" => start,
                       "Evaluation dates" => length(fe_signal.dates),
                       "Dates agree" => fe_signal.dates == fe_trait.dates);
             title = "Aligned")

#=
## 3. Does it rank?

[`forecast_ic`](@ref) answers the **information coefficient** at every evaluation date: the
cross-sectional correlation between what the forecast said and what realised. It answers both
coefficients at once, in a `dates × 2` matrix — Spearman on the ranks and a weighted Pearson on
the values — because the two say different things and a caller who reads one usually wants the
other beside it. Passing the factor-model block resolves the cross-sectional weights the
correlation is taken under.

[`forecast_ic_summary`](@ref) reduces that matrix to five figures per coefficient. `ic_ir` is the
mean over the standard deviation — the coefficient's own information ratio — and `t_stat` is what
says whether the mean is distinguishable from zero at all.
=#

ic_signal = forecast_ic(fe_signal, csfm)
ic_trait = forecast_ic(fe_trait, csfm)

s_signal = forecast_ic_summary(ic_signal)
s_trait = forecast_ic_summary(ic_trait)

pretty_table(DataFrame("Forecast" => ["signal composite", "trait regression"],
                       "Mean IC" => [s_signal.spearman.mean_ic, s_trait.spearman.mean_ic],
                       "Std IC" => [s_signal.spearman.std_ic, s_trait.spearman.std_ic],
                       "IC IR" => [s_signal.spearman.ic_ir, s_trait.spearman.ic_ir],
                       "t-stat" => [s_signal.spearman.t_stat, s_trait.spearman.t_stat],
                       "Hit rate" =>
                           [s_signal.spearman.hit_rate, s_trait.spearman.hit_rate]);
             formatters = [numfmt], title = "Spearman information coefficient")

#=
That is the whole page in one table. The signal composite's mean coefficient is around eight
hundredths with a t-statistic near seven, and it is positive at better than three dates in four.
The trait regression's is four thousandths with a t-statistic below one and a hit rate a coin
would produce. **A reader who only wanted to know whether the trait regression is worth carrying
can stop here**, and the sections below are what the signal composite has earned the right to.

A coefficient is only as believable as the cross-section it was taken over, so
[`forecast_coverage`](@ref) says what share of the investable universe carried both a finite
forecast and a finite target at each date. It deliberately applies **no** `min_count`: the
coverage is what explains why a date carries no coefficient, so it must answer where the
coefficient does not.
=#

cov_signal = forecast_coverage(fe_signal, csfm)

pretty_table(DataFrame("Mean coverage" => mean(cov_signal),
                       "Least coverage" => minimum(cov_signal),
                       "Dates below 100%" => count(<(1), cov_signal));
             formatters = [numfmt], title = "How much of the universe was scored")

#=
The running sum of the coefficient is the picture worth drawing, because a mean hides whether the
edge was continuous or came from three dates.
=#

plot_forecast_cumulative_ic(fe_signal, csfm; title = "Signal composite: cumulative IC")

#=
## 4. Does the ranking pay?

A coefficient says the ordering is related to the outcome. It does not say a book built on that
ordering earns anything, because the coefficient weighs every asset alike and a book does not.

[`forecast_portfolio`](@ref) builds the book from the forecast alone. Two constructions ship:
`:rank` turns each cross-section into ranks, `:zscore` into standardised values. Both are then
**centred and rescaled to 200 % gross** — one unit long, one unit short, netting to zero — so a
date is dollar neutral whatever the spread of the forecast happens to be that day. That
normalisation is what makes two forecasts comparable at all: without it a book's return would
partly report how large the forecast's numbers are rather than how good its ordering is.

The Result carries the weights, the realised return path, the turnover path, and a
[`performance_summary`](@ref) of the return path.
=#

books = [(nm, kind, forecast_portfolio(fe; kind = kind))
         for (nm, fe) in (("signal composite", fe_signal), ("trait regression", fe_trait))
         for kind in (:rank, :zscore)]

pretty_table(DataFrame("Forecast" => [b[1] for b in books],
                       "Book" => [string(b[2]) for b in books],
                       "Ann. return" => [b[3].summary.ann_return for b in books],
                       "Ann. vol" => [b[3].summary.ann_volatility for b in books],
                       "Sharpe" => [b[3].summary.sharpe for b in books],
                       "Hit rate" => [b[3].hit_rate for b in books],
                       "Mean turnover" => [b[3].mean_turnover for b in books]);
             formatters = [numfmt], title = "The two books of each forecast")

#=
The second question answers the same way the first did. The signal composite's two books earn a
Sharpe ratio near twelve; the trait regression's rank book earns half a unit and its z-score book
is indistinguishable from zero. Note also that both forecasts turn over close to their whole
gross every date — a rank book rebuilt from scratch at each cross-section is a high-turnover
object by construction, and `mean_turnover` is there so a caller can see the cost before an
optimiser with a turnover constraint sees it.

!!! note "Two hit rates, two denominators"
    The hit rate in this table counts against the dates the book **scored**, because a date the
    book could not trade is not a loss. The hit rate in §3's table counts against **every**
    evaluation date, because a date the forecast could not rank is a miss. The summary Result of
    §7 names the two columns apart and states each denominator rather than reconciling them.

[`forecast_quantile_spread`](@ref) asks the same question with a blunter instrument: buy the top
fraction of the cross-section, sell the bottom, and see what the difference earned. It is the
reading to reach for when the concern is that a book's return comes from the middle of the
distribution rather than from the tails the forecast is actually confident about.
=#

qs_signal = forecast_quantile_spread(fe_signal; quantiles = (0.1, 0.2))
qs_trait = forecast_quantile_spread(fe_trait; quantiles = (0.1, 0.2))

pretty_table(DataFrame("Forecast" =>
                           ["signal composite", "signal composite", "trait regression",
                            "trait regression"], "Tail" => ["10%", "20%", "10%", "20%"],
                       "Ann. mean" => [qs_signal.ann_mean; qs_trait.ann_mean],
                       "Ann. IR" => [qs_signal.ann_ir; qs_trait.ann_ir],
                       "Hit rate" => [qs_signal.hit_rate; qs_trait.hit_rate]);
             formatters = [numfmt], title = "Top minus bottom")

#=
The spread is the **noisiest** reading on the page, and this table is where that shows. A tenth
of eighty assets is eight names a side, so the trait regression — whose coefficient is four
thousandths and whose books earn nothing — still reads a tail spread with an annualised
information ratio near three. Read the spread as a check on where a book's return came from, not
as evidence a forecast works; the coefficient and the books are what answer that.
=#

plot_forecast_cumulative_returns(fe_signal;
                                 title = "Signal composite: book cumulative returns")

#=
## 5. Is the magnitude right?

The first two questions are about **ordering**. A forecast can order the cross-section perfectly
and still be a hundred times too large, and an optimiser that maximises expected return will
happily act on the exaggeration.

[`forecast_calibration`](@ref) answers the scale. Its `slope` is a weighted, **zero-intercept**
regression of the realised target on the forecast, pooled over every scorable pair of the whole
evaluation: a slope of one says the forecast's units are already the target's units, a slope of a
tenth says the forecast is ten times too large, and a slope near zero says the magnitude carries
no information whatever the ordering does.

The intercept is refused rather than fitted, because the cross-sectional mean of the target is
what the factor model is for; the forecast is only ever asked about the deviation from it. And
unlike every other statistic on this page, the calibration reads **no** `min_count`: it pools the
pairs of every date into one sample, so a thin cross-section contributes few pairs rather than an
unreliable number.
=#

cal_signal = forecast_calibration(fe_signal, csfm)
cal_trait = forecast_calibration(fe_trait, csfm)

pretty_table(DataFrame("Forecast" => ["signal composite", "trait regression"],
                       "Slope" => [cal_signal.slope, cal_trait.slope],
                       "Mean forecast" => [cal_signal.mean_alpha, cal_trait.mean_alpha],
                       "Std forecast" => [cal_signal.std_alpha, cal_trait.std_alpha],
                       "Mean target" => [cal_signal.mean_y, cal_trait.mean_y],
                       "Std target" => [cal_signal.std_y, cal_trait.std_y]);
             formatters = [numfmt], title = "Calibration")

#=
The signal composite's slope sits just under one, which is the fixture telling the truth: the
`signal` field *is* the alpha plus noise, stated in return units, so a scale multiplier of about
one is the right answer and the reading recovers it. The trait regression's slope is negative and
small, which is what a magnitude with no information looks like.

The **curve** is the same reading without the straight-line assumption: the pooled pairs are cut
into bins of the forecast, and each bin's mean forecast is plotted against its mean realised
target. A forecast whose ordering is right but whose scale bends will show it here and nowhere
else. The weights reach the slope alone — the curve and the pooled moments read every pair alike
— which is what lets one call answer a weighted scale and an unweighted shape.
=#

plot_forecast_calibration(fe_signal, csfm; title = "Signal composite: calibration")

#=
## 6. Two questions you only reach on a yes

### How long does the edge last?

[`forecast_holding_period`](@ref) scores the same forecast against **cumulative** forward windows
— `horizon`, then twice it, then three times — and [`forecast_decay`](@ref) scores it against
**disjoint** ones, each starting where the last ended. The first says how long a position is
worth holding; the second says how quickly the information goes stale.

!!! warning "A table is internally comparable, and not comparable at another `n`"
    A deeper window matures later, so every row of a table is read on the dates **every** window
    of the grid can be scored at. Changing `n` therefore changes the sample as well as the depth,
    and a row from an `n = 4` table must never be set beside a row from an `n = 2` table.
=#

hp = forecast_holding_period(fe_signal, rd, csfm; n = 4)
dc = forecast_decay(fe_signal, rd, csfm; n = 4)

pretty_table(DataFrame("Period" => hp.period, "Horizon" => hp.horizon, "Lag" => hp.lag,
                       "Mean IC" => hp.spearman_mean_ic, "t-stat" => hp.spearman_t_stat,
                       "Rank ann. return" => hp.rank_ann_return,
                       "Rank Sharpe" => hp.rank_sharpe); formatters = [numfmt],
             title = "Holding period: cumulative forward windows")

pretty_table(DataFrame("Period" => dc.period, "Horizon" => dc.horizon, "Lag" => dc.lag,
                       "Mean IC" => dc.spearman_mean_ic, "t-stat" => dc.spearman_t_stat,
                       "Rank Sharpe" => dc.rank_sharpe); formatters = [numfmt],
             title = "Decay: disjoint forward windows")

#=
The two tables answer differently, and the difference is the fixture being honest. The holding
period's coefficient **rises** with depth, because a longer window averages more of the
idiosyncratic noise away while the planted alpha accumulates. The decay's coefficient is
**flat**, because the planted alpha is a constant per asset: a signal that never changes never
goes stale. A real alpha decays, and a flat decay table is the shape that says the fixture's
signal is not one.

### Is it an exposure in disguise?

The last question is whether the forecast is telling the optimiser something the factor model
already knows. [`forecast_factor_correlation`](@ref) answers the contemporaneous correlation
between the forecast and each factor exposure, one column per factor, so a forecast that is
really a size bet reads a large number in the size column.
=#

fc_signal = forecast_factor_correlation(fe_signal, csfm)
fc_trait = forecast_factor_correlation(fe_trait, csfm)

fc_mean(fc, k) = mean(filter(isfinite, view(fc, :, k)))

pretty_table(DataFrame("Factor" => csfm.nf,
                       "signal composite" =>
                           [fc_mean(fc_signal, k) for k in axes(fc_signal, 2)],
                       "trait regression" =>
                           [fc_mean(fc_trait, k) for k in axes(fc_trait, 2)]);
             formatters = [numfmt], title = "Mean correlation with each factor exposure")

#=
Both forecasts read small numbers, and that is worth saying plainly: **a low factor correlation
is necessary, not sufficient.** The trait regression is built out of the exposures themselves and
still reads near zero here, because it was fitted against the idiosyncratic return, which is by
construction what the exposures do not explain. It passes this check and fails every other one.
The reading order matters for exactly this reason — this question is last because a yes here
means nothing on its own.

!!! note "A Neutralisation does not decorrelate"
    A forecast whose Descriptor scores are neutralised against a factor family does **not** read
    zero in that family's columns. Both Neutralisation sites build a cross-sectional regression
    whose `intercept` is `false`, so the residual is orthogonal to its target in the *uncentred*
    sense and keeps a real correlation with it. Set `cre` to a regression with `intercept = true`
    when an uncorrelated residual is what is wanted.
=#

#=
## 7. The comparison is the same Result

[`forecast_evaluation_summary`](@ref) is the top of the family. It answers a **columnar**
[`ForecastSummaryResult`](@ref) whose axis is the forecast: thirty columns, one entry per
forecast. A single evaluation is its length-1 case, and the length-2 case **is** the comparison —
there is no separate comparison type, because a comparison of two forecasts is a summary of two
forecasts.

It computes nothing of its own. Every column is one of the verbs above, read on the same dates
under the same parameters, which is why §2's alignment had to happen first: the summary refuses a
set of evaluations that disagree on `target`, `horizon`, `lag`, `step`, `min_count`, `ppy` or
their dates, rather than quietly reporting statistics taken over different samples.
=#

fs = forecast_evaluation_summary([fe_signal, fe_trait], csfm;
                                 names = ["signal composite", "trait regression"],
                                 quantiles = (0.1, 0.2))

pretty_table(DataFrame("Statistic" =>
                           ["Spearman mean IC", "Spearman IC IR", "Spearman t-stat",
                            "Spearman hit rate", "Rank ann. return", "Rank Sharpe",
                            "Rank mean turnover", "Z-score Sharpe", "Calibration slope",
                            "Mean coverage", "Mean assets scored"],
                       "signal composite" => [fs.spearman_mean_ic[1], fs.spearman_ic_ir[1],
                                              fs.spearman_t_stat[1], fs.spearman_hit_rate[1],
                                              fs.rank_ann_return[1], fs.rank_sharpe[1],
                                              fs.rank_mean_turnover[1], fs.zscore_sharpe[1],
                                              fs.calibration_slope[1], fs.mean_coverage[1],
                                              fs.mean_n_scored[1]],
                       "trait regression" => [fs.spearman_mean_ic[2], fs.spearman_ic_ir[2],
                                              fs.spearman_t_stat[2], fs.spearman_hit_rate[2],
                                              fs.rank_ann_return[2], fs.rank_sharpe[2],
                                              fs.rank_mean_turnover[2], fs.zscore_sharpe[2],
                                              fs.calibration_slope[2], fs.mean_coverage[2],
                                              fs.mean_n_scored[2]]); formatters = [numfmt],
             title = "Eleven of the thirty columns, side by side")

#=
The quantile spreads are the one **second** axis the Result carries, and only on request: pass no
`quantiles` and the four spread fields read back as `nothing`. They are `forecasts × quantiles`,
so a caller compares two forecasts at one tail or one forecast at two.
=#

pretty_table(DataFrame("Tail" => ["10%", "20%"],
                       "signal composite Sharpe" => fs.spread_sharpe[1, :],
                       "trait regression Sharpe" => fs.spread_sharpe[2, :]);
             formatters = [numfmt], title = "Quantile spread, the second axis")

#=
The figure draws ten of the thirty columns — the ones that share a scale. `n_bins` and
`mean_n_scored` are counts in the tens and would dwarf a Sharpe ratio on one axis, so they are
left to the table above.
=#

plot_forecast_evaluation_summary(fs; size = (900, 500))

#=
Three parts of an evaluation are deliberately **not** columns, each because its axis is not the
forecast: the drawdown family, which is of a compressed return path and would set two forecasts
beside each other over two different paths; the holding-period and decay tables, whose axis is the
forward window; and the factor correlations, whose axis is the factor. Read those with the verbs
of §6.
=#

#=
## 8. The other figures

Four figures appear above. Seven more ship, each answering a question this page raised, and all
eleven take the [`ForecastEvaluationResult`](@ref) — the caller pairs once and every figure reads
that pairing, because building `alpha` can cost a rolling refit and no figure should pay for it
twice.

| Figure | What it answers |
|:---|:---|
| [`plot_forecast_rolling_ic`](@ref) | Was the edge continuous, or did a window carry it? |
| [`plot_forecast_quantile_returns`](@ref) | What did the top-minus-bottom spread accumulate? |
| [`plot_forecast_ic_by_holding_period`](@ref) | How does the coefficient move with depth? |
| [`plot_forecast_portfolio_by_holding_period`](@ref) | How does the book move with depth? |
| [`plot_forecast_ic_decay`](@ref) | How fast does the coefficient go stale? |
| [`plot_forecast_portfolio_decay`](@ref) | How fast does the book go stale? |
| [`plot_forecast_factor_correlation`](@ref) | Which exposure is the forecast leaning on? |

=#

plot_forecast_rolling_ic(fe_signal, csfm; title = "Signal composite: rolling IC")

#=
## Where to go next

  - [Cross-sectional factor model, end to end](05_Cross_Sectional_Factor_Model.md) builds the
    panel, the prior and the block this page reads, and hands the same forecast to an optimiser.
  - [Cross-sectional factor model through a Pipeline](06_Cross_Sectional_Factor_Pipeline.md)
    reaches the same weights with the panel entering as a Pipeline Data Slot.
  - [Walk-forward and cross-validation](../5_validation_tuning/01_Cross_Validation.md) is the
    out-of-sample reading of a whole **strategy**, where this page reads a single forecast.
  - [Performance attribution](../6_post_processing/03_Performance_Attribution.md) decomposes what
    a book actually earned, once one has been solved.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src
#src 1. TWO MEMBERS RARELY SHARE AN EVALUATION GRID, AND THE SUMMARY REFUSES THEM WHEN THEY DO
#src    NOT. Measured on this page's fixture: the signal composite is scorable from block row 1
#src    and carries 87 evaluation dates, while the trait regression's refit warms up and starts
#src    at row 66 with 74. `forecast_summary_assert_comparable` reds on `dates`, which is
#src    correct and is the reason §2 aligns through the bare method. This is the first place in
#src    the library where a caller MUST drop from the Estimator layer to the bare one to do an
#src    ordinary thing, and it is worth a look at whether a future ticket should offer an
#src    alignment verb.
#src 2. `forecast_target_history` IS NOT EXPORTED, so an example cannot build `y` directly. It
#src    is read off `raw_signal.y` instead, which is better anyway: the Result carries the
#src    target already cut to the block, so no second pairing is built.
#src 3. THE DECAY TABLE IS FLAT ON THIS FIXTURE AND THAT IS CORRECT. The planted alpha is a
#src    constant per asset, so it never goes stale; the holding-period coefficient rises with
#src    depth (0.083 → 0.140 over n = 4) while the decay coefficient stays near 0.07-0.08. The
#src    prose says so rather than hiding it, because a reader who expects decay and sees none
#src    should learn the fixture's shape, not doubt the verb.
#src 4. THE FACTOR CORRELATION INVERTED THE EXPECTED STORY. The trait regression, built out of
#src    the exposures, reads SMALLER correlations (about -0.07 to -0.11) than the signal
#src    composite (up to 0.22), because it is fitted against the idiosyncratic return, which is
#src    by construction what the exposures do not explain. The prose was rewritten around the
#src    measurement: a low factor correlation is necessary, not sufficient.
#src 5. RUNTIME, measured 2026-09-09: the prior fit is about 44 s and the trait regression's
#src    refit path about 12 s at `step = 5`; everything else is under a second. The refit cost
#src    scales with the number of grid points, so a smaller `step` makes this page much slower.
