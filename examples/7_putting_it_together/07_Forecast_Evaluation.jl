#=
```@meta
Description = "Read a return forecast before an optimiser sees it: forecast evaluation for CrossSectionalFactorPrior in PortfolioOptimisers.jl."
```

# Reading a return forecast before an optimiser sees it

A [`CrossSectionalFactorPrior`](@ref) takes a return forecast, which is an opinion about which
assets are about to do better than the factor model says they should, and adds it to the expected
returns an optimiser maximises. The optimiser acts on that opinion whether it is good or bad. It
cannot tell a forecast that orders the cross-section from one that orders it no better than a coin
toss would.

This page scores a return forecast before an optimiser sees it. It scores the forecast out of
sample, on its own, against what happened next. It also puts two forecasts side by side, so you can
see the difference between one that works and one that does not.

Ask three questions in order, and stop at the first no.

 1. Does it rank? Is the order the forecast puts the cross-section in related to the order that
    realised? That is the information coefficient.
 2. Does the ranking pay? Turn the forecast into a book, and look at what the book earned and what
    it cost in turnover.
 3. Is the magnitude right? A forecast can order the cross-section perfectly and still be ten
    times too large. That is the calibration slope.

Two more questions come after all three answer yes. They ask how long the forecast keeps its
information and whether it is a factor exposure under another name. Section 6 asks them.

Everything below runs on the synthetic panel of the [deep dive](05_Cross_Sectional_Factor_Model.md).
The alpha that the panel was drawn from is known before the estimator runs, and you can
check every claim against it.

!!! warning "These are synthetic numbers"
    The panel plants a constant alpha per asset inside a field that the factor exposures ignore. The
    Sharpe ratios below are what a forecast of a planted constant earns on a long-short book with no
    trading cost. They tell you about the synthetic panel, not about markets. What transfers to real
    data is the order of the questions and the gap between the two forecasts.
=#

using PortfolioOptimisers, StableRNGs, Statistics, LinearAlgebra, Dates, PrettyTables,
      DataFrames, StatsPlots, GraphRecipes

numfmt = (v, i, j) -> begin
    return isa(v, AbstractFloat) ? round(v; sigdigits = 4) : v
end;

#=
## 1. The panel, and two convictions

We copy the panel generator from the deep dive without a change. Section 1 of
[the deep dive](05_Cross_Sectional_Factor_Model.md) says what each field of the panel is. One field
matters here, `signal`. It is the assets' true alpha plus a little noise, and no factor exposure is
built from it. A forecast built on it has a part that the factor model does not span.
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
The evaluation scores a forecast against the idiosyncratic return, the part of an asset's return the
factor model does not explain. That is the part a return forecast forecasts inside a
[`CrossSectionalFactorPrior`](@ref). We fit the factor model first. Every function below
takes the factor-model block that the fit leaves on the prior.
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
Now the two convictions. Both are return forecast estimators. They differ in how they turn traits
into a forecast and in which traits they take.

  - The signal composite is a [`FixedWeightedReturnForecast`](@ref) over the `signal` field. It
    says that this field is the alpha, at scale one, and it composes its descriptor scores one
    observation at a time, so it publishes a history of its own.
  - The trait regression is a [`TargetReturnForecast`](@ref) that regresses the forward
    idiosyncratic return on four style traits the factor model already spans: size, value, earnings
    yield and liquidity. It is the control, a forecast built only from those traits.

One of these forecasts something real and the other does not. The sections below tell them apart.
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

[`forecast_evaluation`](@ref) is the one call that turns an estimator into something you can score.
It fits the forecast, builds its history, builds the forward target the history is scored against,
and returns a [`ForecastEvaluationResult`](@ref). The result has three matrices, the forecast, the
forward target and the universe mask, together with the observations it can be scored at and the
parameters that produced them.

Five parameters shape the pairing, and the rest of the page uses them by name.

  - `horizon` is how many observations forward the target accumulates over.
  - `lag` is how many observations after the forecast the target window opens, so `lag = 1` means
    the forecast is acted on at the next observation and never sees its own target.
  - `step` is how many observations lie between two evaluation dates.
  - `min_count` is the fewest assets a cross-section must carry before a statistic is taken of it.
  - `ppy` is the number of periods in a year. The book statistics further down are annualised with
    it.

Every statistic is a function that takes the result, so you can change the parameters of one
statistic without running the fit again.
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

The table shows the one problem you meet when you compare two forecasts rather than change the
parameters of one. The signal composite composes a score at every observation, so it can be scored
from the first row of the block. The trait regression publishes no history. It fits one
cross-section over the whole sample and states one row, so [`forecast_history`](@ref) builds its
path by refitting it along the evaluation grid, and a regression that calibrates itself needs a
warm-up before its first fit means anything. The two forecasts therefore start at different
observations, and statistics taken over different samples are not a comparison.

The fix is to cut both forecasts to the sample they share, and [`forecast_evaluation_align`](@ref)
does it. Every evaluation's dates run `lo:step:hi` on the block's rows, so the grid two of them
share runs from the later of the two first dates, in steps of `step`, up to the earlier of the two
last dates. The call
rebuilds each result on that grid and leaves its forecast, target and universe alone, with no
re-pairing, no refit and no blank. It refuses a pair that does not answer one question, which is a
pair differing in target, horizon, lag, step, `min_count`, `ppy` or universe, for the same reason
the summary does.
=#

fe_signal, fe_trait = forecast_evaluation_align([raw_signal, raw_trait])

pretty_table(DataFrame("Common first row" => first(fe_signal.dates),
                       "Evaluation dates" => length(fe_signal.dates),
                       "Dates agree" => fe_signal.dates == fe_trait.dates,
                       "Forecast untouched" => fe_signal.alpha === raw_signal.alpha);
             title = "Aligned")

#=
## 3. Does it rank?

[`forecast_ic`](@ref) gives the information coefficient at every evaluation date, which is the
cross-sectional correlation between what the forecast said and what realised. It gives both
coefficients at once in a `dates × 2` matrix, Spearman on the ranks and a weighted Pearson on the
values, because the two say different things and you usually want both. Passing the factor-model
block lets the `weighting` keyword take the Pearson weights from the block. The default, used here,
gives every asset the same weight, and the Spearman column is unweighted.

[`forecast_ic_summary`](@ref) reduces that matrix to five figures per coefficient. `ic_ir` is the
mean over the standard deviation, the coefficient's own information ratio, and `t_stat` says
whether the mean is distinguishable from zero at all.
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
The two rows of this table answer the first question. The signal composite's mean coefficient is an
order of magnitude larger than the trait regression's, its t-statistic is far from zero where the
trait regression's is not, and its hit rate sits well above one half where the trait regression's
sits at it. If you only want to know whether to keep the trait regression, you can stop here. Most
tables below keep both forecasts, so you can compare them on each question. The coverage, the
holding-period tables and the single-forecast figures show the signal composite alone.

A coefficient is only as good as the cross-section it was taken over. [`forecast_coverage`](@ref)
gives the share of the estimation universe that had both a finite forecast and a finite target at
each date. The universe is the panel's estimation mask at that date, which the evaluation stores in
`umsk`. An asset that has not listed yet is outside the universe, not missing from it.
Coverage applies no `min_count`. It explains why a date has no coefficient, and it must give an
answer on exactly those dates.
=#

cov_signal = forecast_coverage(fe_signal, csfm)

pretty_table(DataFrame("Mean coverage" => mean(cov_signal),
                       "Least coverage" => minimum(cov_signal),
                       "Dates below 100%" => count(<(1), cov_signal));
             formatters = [numfmt], title = "How much of the universe was scored")

#=
We draw the running sum of the coefficient, because a mean hides whether the information
arrived at every date or at three of them.
=#

plot_forecast_cumulative_ic(fe_signal, csfm; title = "Signal composite: cumulative IC")

#=
## 4. Does the ranking pay?

A coefficient says the order is related to the outcome. It does not say a book built on that order
earns anything, because the coefficient weighs every asset alike and a book does not.

[`forecast_portfolio`](@ref) builds the book from the forecast alone. Two constructions ship.
`:rank` turns each cross-section into ranks, and `:zscore` turns it into standardised values. Both
are then centred and rescaled to 200% gross, one unit long and one unit short, with a net of zero.
Every date is dollar neutral whatever the spread of the forecast is that day. Two forecasts
are comparable only after that rescaling. Without it a book's return would partly report how large
the forecast's numbers are rather than how good its order is.

The result holds the weights, the realised return path, the turnover path, and a
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
The second question has the same answer as the first. The signal composite's two books earn a high
Sharpe ratio. The trait regression's rank book earns a fraction of that, and its z-score book earns
nothing you could tell from zero. Look at the turnover column as well. Both forecasts turn over
close to their whole gross at every date, because a rank book rebuilt from each cross-section trades
whatever the ranks moved, and `mean_turnover` is there so you can see that cost before an optimiser
with a turnover constraint does.

!!! note "Every hit rate counts against the dates that scored"
    The hit rate in this table counts against the dates the book scored, because a date the book
    could not trade is not a loss. The hit rate in the table of section 3 counts against the dates
    that carried a coefficient, for the same reason. A date the forecast could not rank is a date
    at which nothing was measured, and the coverage of section 3 reports how often that happened.
    Every hit rate therefore sits over the same sample as the mean printed beside it.

[`forecast_quantile_spread`](@ref) asks the same question more coarsely. Buy the top fraction of
the cross-section, sell the bottom, and measure what the difference earned. Reach for it to check
whether the tails the forecast is most confident about earn a return of their own.
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
The spread is the noisiest statistic on the page, and this table shows it. A tenth of eighty assets
is eight names a side. With so few names, the trait regression gets a tail spread with a large
annualised information ratio, although its coefficient is near zero and its books earn little. Use
the spread as a check on where a book's return came from. The coefficient and the books say whether
a forecast works.
=#

plot_forecast_cumulative_returns(fe_signal;
                                 title = "Signal composite: book cumulative returns")

#=
## 5. Is the magnitude right?

The first two questions are about order. A forecast can order the cross-section perfectly and
still be ten times too large, and an optimiser that maximises expected return will act on
the exaggeration.

[`forecast_calibration`](@ref) gives the scale. Its `slope` is a weighted regression of the
realised target on the forecast with the intercept fixed at zero, pooled over every scorable pair
of the whole evaluation. A slope of one says the forecast's units are already the target's units.
A slope of a tenth says the forecast is ten times too large. A slope near zero says the magnitude
carries no information, whatever the order does.

The intercept is fixed at zero, because the cross-sectional mean of the target is what the factor
model is for, and the forecast is only ever asked about the deviation from it. Like the coverage,
the calibration applies no `min_count`. It pools the pairs of every date into one sample, so a thin
cross-section adds only its few pairs to that sample and never a statistic of its own.
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
The signal composite's slope sits just under one, which is what the synthetic panel should give. The
`signal` field is the alpha plus noise in return units, and a scale multiplier near one is right for
it. The trait regression's slope is small and negative, as expected when the magnitude has no
information.

The curve makes the same check without the straight-line assumption. The pooled pairs are cut into
bins of the forecast, and each bin's mean forecast is plotted against its mean realised target. A
forecast whose order is right but whose scale bends will show it here and nowhere else. The weights
enter the slope only. The curve and the pooled moments give every pair the same weight, and one call
thus gives a weighted scale and an unweighted shape.
=#

plot_forecast_calibration(fe_signal, csfm; title = "Signal composite: calibration")

#=
## 6. Holding period and factor overlap

### How long does the forecast keep its information?

[`forecast_holding_period`](@ref) scores the same forecast against cumulative forward windows,
`horizon`, then twice it, then three times, and [`forecast_decay`](@ref) scores it against disjoint
ones, each starting where the last ended. The first says how long to hold a position. The second
says how quickly the information goes stale.

!!! warning "A table is internally comparable, and not comparable at another `n`"
    A deeper window matures later, so every row of a table is scored on the dates at which every
    window of the grid can be scored. Changing `n` therefore changes the sample as well as the
    depth, and a row from a table at `n = 4` must never be set beside a row from a table at `n = 2`.

The stride between two dates stays at the base evaluation's, so from the second row of the
holding-period table on, consecutive windows use the same returns, and their coefficients are not
independent. The `t_stat` column corrects for that overlap. Its standard error is the long-run one,
at the order that the window and the stride imply. A t-statistic that stays large down the column
therefore reflects the forecast, not the overlap. Without it, a forecast with no skill would look
more significant at every row.
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
The two tables give different answers, and the difference comes from the synthetic panel. The
holding period's coefficient rises with depth, because a longer window averages more of the
idiosyncratic noise away while the planted alpha accumulates. The decay's coefficient is flat,
because the planted alpha is a constant per asset, and a signal that never changes never goes stale.
A real alpha decays. A flat decay table here shows that the planted signal is not a real alpha.

### Is the forecast a factor exposure under another name?

The last question is whether the forecast tells the optimiser something the factor model already
knows. [`forecast_factor_correlation`](@ref) gives the same-date correlation between the forecast
and each factor exposure, one column per factor. A forecast that is really a size bet shows a large
number in the size column.

The statistic does not look forward. It is computed on every observation of the forecast, not on the
evaluation grid. The grid keeps the forward windows of the coefficient from overlapping, and a
same-date correlation has no window. The signal composite therefore scores on every row of the
block, the rows before the common grid of section 2 included, because the alignment moved its dates
and not its forecast. The trait regression has a forecast on its grid only. Its column is `NaN`
between two refits, and the mean below is taken over the rows it was fitted on. Pass
`dates = fe.dates` for the correlations on the same dates as the coefficients.
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
Both forecasts give small numbers. The trait regression is built out of the exposures themselves and
still gives numbers near zero here, because it was fitted against the idiosyncratic return, which is
exactly what the exposures do not explain. It passes this check and fails the first three questions.
That is why the questions have an order, and why this question is last. A low correlation alone does
not make a forecast useful.

!!! note "Neutralising a score does not decorrelate it"
    A forecast whose descriptor scores are neutralised against a factor family does not give zero in
    that family's columns. Both neutralisation sites build a cross-sectional regression whose
    `intercept` is `false`, so the residual is orthogonal to its target in the uncentred sense and
    keeps a real correlation with it. Set `cre` to a regression with `intercept = true` when you
    want an uncorrelated residual.
=#

#=
## 7. Comparing forecasts in one summary

[`forecast_evaluation_summary`](@ref) collects the statistics above in one call. It gives a columnar
[`ForecastSummaryResult`](@ref) whose axis is the forecast, thirty columns with one entry per
forecast. A single evaluation gives a result of length one, and two evaluations give the comparison.

It computes nothing of its own. Every column is one of the functions above, evaluated on the same
dates under the same parameters. That is why the alignment of section 2 comes first. The summary
refuses a set of evaluations that disagree on `target`, `horizon`, `lag`, `step`, `min_count`,
`ppy`, their universe or their dates, rather than reporting statistics taken over different samples.
Pass `align = true` with the raw evaluations instead, and the summary runs the alignment of section
2 before it computes anything.
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
The quantile spreads are the only fields of the result with a second axis, and the result fills them
only on request. If you pass no `quantiles`, the four spread fields are `nothing`. They are
`forecasts × quantiles`, so you compare two forecasts at one tail or one forecast at two.
=#

pretty_table(DataFrame("Tail" => ["10%", "20%"],
                       "signal composite Sharpe" => fs.spread_sharpe[1, :],
                       "trait regression Sharpe" => fs.spread_sharpe[2, :]);
             formatters = [numfmt], title = "Quantile spread, the second axis")

#=
The figure draws ten headline columns of the thirty, and the spread Sharpe ratio at each of the
two tails, because this summary carries the quantile block. The columns have different units, so
compare the two forecasts inside each group of bars.
=#

plot_forecast_evaluation_summary(fs; size = (900, 500))

#=
A bar is a mean, and as section 3 noted, a mean hides whether the information arrived at every date
or at three of them. The comparison has two figures for that, the same running sums sections 3 and 4
drew for one forecast, overlaid one series per forecast on the dates the evaluations share. The plot
called with a vector refuses the pairs that the summary refuses. The two lines always cover one
sample.
=#

plot_forecast_cumulative_ic([fe_signal, fe_trait], csfm;
                            names = ["signal composite", "trait regression"],
                            title = "Cumulative Spearman IC, both forecasts")

#=
The book overlay works the same way. It draws one book per forecast, the rank book by default and
the z-score book under `kind = :zscore`, so a figure of four forecasts has four lines and not eight.
=#

plot_forecast_cumulative_returns([fe_signal, fe_trait];
                                 names = ["signal composite", "trait regression"],
                                 title = "Rank book cumulative returns, both forecasts")

#=
Three parts of an evaluation are not columns. The drawdown and the other path statistics of a book
are taken over its return path with the gap dates removed. Two forecasts whose gaps fall on
different dates would then be compared over two different paths. The `summary` of
[`forecast_portfolio`](@ref) in section 4 gives them for one forecast. The holding-period and decay
tables have the forward window for an axis, and the factor correlations have the factor. Section 6
gives both.
=#

#=
## 8. The other figures

Four plotting functions drew the six figures above, and seven more ship, each for a question this
page raised. All eleven take a [`ForecastEvaluationResult`](@ref), because you pair once and every
figure starts from that pairing. The two overlays and the summary plot also take a vector of them,
and the summary plot also takes the [`ForecastSummaryResult`](@ref). Building `alpha` can cost a
rolling refit, and no figure should pay for it twice.

| Figure | What it answers |
|:---|:---|
| [`plot_forecast_rolling_ic`](@ref) | Did the information arrive at every date, or in one window? |
| [`plot_forecast_quantile_returns`](@ref) | What did the top-minus-bottom spread accumulate? |
| [`plot_forecast_ic_by_holding_period`](@ref) | How does the coefficient move with depth? |
| [`plot_forecast_portfolio_by_holding_period`](@ref) | How does the book move with depth? |
| [`plot_forecast_ic_decay`](@ref) | How fast does the coefficient go stale? |
| [`plot_forecast_portfolio_decay`](@ref) | How fast does the book go stale? |
| [`plot_forecast_factor_correlation`](@ref) | Which exposure does the forecast track? |
=#

plot_forecast_rolling_ic(fe_signal, csfm; title = "Signal composite: rolling IC")

#=
## Where to go next

  - [Cross-sectional factor model, end to end](05_Cross_Sectional_Factor_Model.md) builds the panel,
    the prior and the block this page starts from, and hands the same forecast to an optimiser.
  - [Cross-sectional factor model through a Pipeline](06_Cross_Sectional_Factor_Pipeline.md)
    reaches the same weights, with the panel arriving on the returns result rather than in a slot
    of its own.
  - [Walk-forward and cross-validation](../5_validation_tuning/01_Cross_Validation.md) tests a whole
    strategy out of sample, where this page scores a single forecast.
  - [Performance attribution](../6_post_processing/03_Performance_Attribution.md) decomposes what
    a book earned, once one has been solved.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src
#src 1. TWO MEMBERS RARELY SHARE AN EVALUATION GRID, AND THE SUMMARY REFUSES THEM WHEN THEY DO
#src    NOT. Measured on this page's fixture: the signal composite is scorable from block row 1
#src    and carries 87 evaluation dates, while the trait regression's refit warms up and starts
#src    at row 66 with 74. `forecast_summary_assert_comparable` reds on `dates`, which is
#src    correct and is the reason §2 aligns. This was the first place in the library where a
#src    caller had to drop from the Estimator layer to the bare one to do an ordinary thing;
#src    #1073 shipped `forecast_evaluation_align` and the summary's `align = true`, and §2 now
#src    calls the verb.
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
