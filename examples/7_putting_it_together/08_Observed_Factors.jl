#=
```@meta
Description = "Observed factors of a cross-sectional factor model in PortfolioOptimisers.jl: currency and macro factors from raw prices to a cross-validated portfolio."
```

# [Observed factors, from raw prices to a cross-validated portfolio](@id example-observed-factors-from-raw-prices)

A cross-sectional factor model estimates the return of each factor from the cross-section of asset
returns. Some factor returns need no estimate, because you can read them off the market. The return
of a currency is one: an asset priced in euros earns the euro's move against your base currency, and
the exchange rate states that move. The price of oil is another. This page puts both kinds of factor
into one [`CrossSectionalFactorPrior`](@ref): factors the fit estimates, and observed factors whose
returns come in as data.

We draw the data from a model we wrote, so every number below has a known answer.

 1. Build asset prices in three currencies, the exchange rates, the cash indices and an oil price,
    and convert them to returns.
 2. Fit a prior with a market factor, a size factor, two currency factors and an oil factor.
 3. Measure what the observed factors change in `mu`, in `sigma`, in the summary and in the
    attribution.
 4. Run the prior inside a [`NestedClustered`](@ref), a [`SubsetResampling`](@ref) and a
    walk-forward, where every sub-problem fits its own factors.
 5. Compare the log and the simple return rules.

!!! tip "When to reach for this"
    Reach for an observed factor when the return of the factor is a price series you hold and the
    exposure of each asset to it is either known, such as the currency an asset is priced in, or
    measurable, such as its sensitivity to oil. When you hold every factor return as a series and no
    asset trait drives the exposures, [`FactorPrior`](@ref) is the simpler tool.
=#

using PortfolioOptimisers, StableRNGs, Statistics, LinearAlgebra, Dates, PrettyTables,
      DataFrames, Clarabel, TimeSeries

numfmt = (v, i, j) -> begin
    return isa(v, AbstractFloat) ? round(v; sigdigits = 4) : v
end;

#=
## 1. Prices in three currencies

The universe holds 60 assets. A third of them are priced in US dollars, the base currency, a third
in euros and a third in yen. The local log return of each asset is the sum of four parts:

  - a market return times the asset's beta,
  - a size return times the asset's standardised log market capitalisation of the day before,
  - the log return of oil times the asset's oil sensitivity,
  - an idiosyncratic return with a daily volatility of 1.2 %.

The base-currency price of an asset is its local price times the exchange rate of its currency, in
dollars per unit. Each currency also has a cash total-return index, which accumulates the cash rate
of each day: 4 % a year for the dollar, 2 % for the euro and 0 % for the yen.

The function returns the four price tables as `TimeArray`s, a panel with the market
capitalisation and the currency of each asset, and the true parameters.
=#

function synthetic_world(; T = 750, N = 60, seed = 1_370_001)
    rng = StableRNG(seed)
    days = filter(d -> Dates.dayofweek(d) <= 5,
                  Date(2018, 1, 1):Day(1):(Date(2018, 1, 1) + Day(2 * T + 10)))[1:(T + 1)]
    currencies = ["USD", "EUR", "JPY"]
    code = [mod(i - 1, 3) + 1 for i in 1:N]
    beta = 1.0 .+ 0.3 .* randn(rng, N)
    b_oil = 0.5 .* randn(rng, N)
    logcap = 22.0 .+ 1.2 .* randn(rng, 1, N) .+
             0.02 .* cumsum(randn(rng, T + 1, N); dims = 1)
    zsize = (logcap .- mean(logcap; dims = 2)) ./ std(logcap; dims = 2)
    market = 0.0003 .+ 0.009 .* randn(rng, T)
    size_ret = -0.0002 .+ 0.003 .* randn(rng, T)
    oil = 0.02 .* randn(rng, T)
    local_lr = transpose(beta) .* market .+ view(zsize, 1:T, :) .* size_ret .+
               transpose(b_oil) .* oil .+ 0.012 .* randn(rng, T, N)
    local_px = 100 .* exp.(vcat(zeros(1, N), cumsum(local_lr; dims = 1)))
    fx_lr = hcat(0.006 .* randn(rng, T), 0.007 .* randn(rng, T))
    fx = vcat([1.1 0.009], [1.1 0.009] .* exp.(cumsum(fx_lr; dims = 1)))
    rates = [0.04 0.02 0.0] ./ 252
    cash = cumprod(vcat(ones(1, 3), 1 .+ rates .+ 0.00002 .* randn(rng, T, 3)); dims = 1)
    base_px = local_px .* hcat(ones(T + 1), fx)[:, code]
    oil_px = 70 .* exp.(vcat(0.0, cumsum(oil)))
    pnl = asset_panel([NumericPanelInput(; name = "market_cap", vals = exp.(logcap)),
                       CategoricalPanelInput(; name = "currency",
                                             vals = repeat(permutedims(currencies[code]),
                                                           T + 1), levels = currencies)];
                      amsk = trues(T + 1, N), emsk = trues(T + 1, N))
    return (; X = TimeArray(days, base_px, ["A" * lpad(i, 2, '0') for i in 1:N]),
            fx = TimeArray(days, fx, ["EUR", "JPY"]),
            cash = TimeArray(days, cash[:, 2:3], ["EUR", "JPY"]),
            cash_usd = TimeArray(days, cash[:, 1:1], ["USD"]),
            oil = TimeArray(days, oil_px, ["OIL"]), pnl,
            truth = (; code, beta, b_oil, zsize, local_lr, fx_lr))
end

world = synthetic_world()
truth = world.truth;

#=
A currency factor reads the currency excess return: what a dollar investor earns by holding a
deposit in the currency, funded in dollars. [`currency_excess_index`](@ref) builds a price level for
it from the exchange rate and the two cash indices, so that its return is the exchange-rate return
plus the euro or yen cash return, less the dollar cash return. The dollar takes no index, because
its excess return over itself is zero.

The currency indices and the oil price belong to no asset, so they are not panel fields. They go to
[`price_ingestion`](@ref) as its `E` argument, the exogenous series, which the conversion turns into
returns beside the asset returns.
=#

E = merge(currency_excess_index(world.fx, world.cash, world.cash_usd), world.oil)
pr = price_ingestion(PriceIngestion(), world.X; E = E, pnl = world.pnl)
rd = prices_to_returns(pr; ret_method = :log)

#=
We convert with the log rule. Section 5 shows why. The table compares `rd.E` with the log returns of
the three price levels we passed in. Its largest difference is zero, so `rd.E` holds the returns of
the indices and of oil, under the same rule as the asset returns.
=#

pretty_table(DataFrame("Series" => rd.ne,
                       "Largest abs(rd.E - log return of the level)" =>
                           vec(maximum(abs, rd.E - diff(log.(values(E)); dims = 1);
                                       dims = 1))); formatters = [numfmt],
             title = "The exogenous series after the conversion")

#=
## 2. The prior

The factor list holds four members.

  - `"market"` is a [`ConstantExposure`](@ref), an intercept: every asset has an exposure of one, and
    the fit estimates its return.
  - `"size"` is a style factor on [`LogMarketCap`](@ref). The fit estimates its return too.
  - `"currency"` is a [`CurrencyExposure`](@ref). It reads the `currency` field of the panel and
    gives one factor per currency, with an exposure of one on the asset's own currency. `base = "USD"`
    gives the dollar no factor. Each factor return is the column of `rd.E` that the currency names.
  - `"oil"` is an [`ObservedExposure`](@ref). The exposure of each asset is its
    [`EWMacroSensitivity`](@ref) to the `"OIL"` column of `rd.E`, a beta on oil after the market is
    removed. The factor return is the same `"OIL"` column.

The fit estimates the market and size returns from the returns that are left after it subtracts the
currency and oil factors.
=#

style(d) = CompositeExposure(; descriptors = [d], family = "style")
oil_sensitivity = CompositeExposure(; descriptors = [EWMacroSensitivity(; series = "OIL")],
                                    outlier = nothing, scoring = nothing, family = "macro")
factors = ["market" => ConstantExposure(), "size" => style(LogMarketCap()),
           "currency" => CurrencyExposure(; base = "USD"),
           "oil" => ObservedExposure(; xe = oil_sensitivity, series = "OIL")]
pe = CrossSectionalFactorPrior(; factors = factors)

#=
[`cross_sectional_factor_axis`](@ref) states the factor axis before any fit. You can write a
constraint against these names, or check that a currency you expect has a factor.
=#

axis = cross_sectional_factor_axis(pe, rd)
pretty_table(DataFrame("Factor" => axis.nf, "Family" => axis.fam);
             title = "The factor axis before the fit")

#=
Now the fit. `rr` is the fitted factor model. `rr.nf` and `rr.fam` repeat the axis, and `rr.fx`
holds the returns of the observed factors over the observations of the fit. The estimated factors
come first on the axis, and the observed ones after them.
=#

prior_full = prior(pe, rd)
rr = prior_full.rr
n_obs = size(rr.fx, 2)
kind = [k > length(rr.nf) - n_obs ? "observed" : "estimated" for k in eachindex(rr.nf)]
fit_rows = (size(rd.X, 1) - size(rr.fx, 1) + 1):size(rd.X, 1)

#=
The prior fits the last 690 of the 750 observations. The oil sensitivity needs a warm-up before its
first value, and the fit starts after it. The table prints, for each factor, how closely the last
row of fitted exposures follows the true exposure, and the largest difference between `rr.fx` and
the rows of `rd.E` the fit covers. The currency exposures are the one-hot currency of each asset,
so we compare them with that.
=#

true_exposure = hcat(ones(60), truth.zsize[end, :], truth.code .== 2, truth.code .== 3,
                     truth.b_oil)
pretty_table(DataFrame("Factor" => rr.nf, "Family" => rr.fam, "Kind" => kind,
                       "corr(fitted, true) exposure" =>
                           [k == 1 ? NaN : cor(rr.M[:, k], true_exposure[:, k])
                            for k in eachindex(rr.nf)],
                       "Largest abs(rr.fx - rd.E)" => [if k > length(rr.nf) - n_obs
                                                           maximum(abs,
                                                                   rr.fx[:, k - length(rr.nf) + n_obs] -
                                                                   rd.E[fit_rows, k - length(rr.nf) + n_obs])
                                                       else
                                                           NaN
                                                       end
                                                       for k in eachindex(rr.nf)]);
             formatters = [numfmt], title = "What the fit recovered")

#=
The market exposure is one for every asset, so it has no correlation to print. The currency
exposures are exactly the currency of each asset, the size and oil exposures follow the truth with
a correlation above 0.99, and the observed factor returns are the columns of `rd.E` with no change.

## 3. What the observed factors change

The covariance of the prior is `M * fpr.sigma * M'` plus the idiosyncratic variance of each asset,
where `M` holds the exposures of the last row and `fpr.sigma` is the factor covariance. Split the
factor covariance into its estimated block, its observed block and the cross terms between them, and
the variance of each asset splits into four parts. The first table checks that the four parts add
up to `diag(sigma)`. The second averages the share of each part over the assets of each currency.
=#

est = findall(==("estimated"), kind)
obs = findall(==("observed"), kind)
M = rr.M
S = prior_full.fpr.sigma
part(a, b) = vec(sum((M[:, a] * S[a, b]) .* M[:, b]; dims = 2))
var_est = part(est, est)
var_obs = part(obs, obs)
var_cross = 2 .* part(est, obs)
var_idio = rr.vs[end, :]
var_total = diag(prior_full.sigma)

pretty_table(DataFrame("Largest abs(sum of the parts - diag(sigma))" => maximum(abs,
                                                                                var_est + var_obs + var_cross + var_idio - var_total));
             formatters = [numfmt], title = "The four parts of the variance of each asset")

currency_names = ["USD", "EUR", "JPY"]
share(v) = [mean((v ./ var_total)[truth.code .== c]) for c in 1:3]
pretty_table(DataFrame("Currency" => currency_names, "Estimated factors" => share(var_est),
                       "Observed factors" => share(var_obs),
                       "Cross terms" => share(var_cross),
                       "Idiosyncratic" => share(var_idio)); formatters = [numfmt],
             title = "Mean share of the variance of an asset")

#=
A dollar asset has no currency exposure, so its observed share is the oil part alone. A yen asset
carries the most volatile exchange rate, and its observed share is the largest.

`mu` splits the same way, with no cross term: it is `M * fpr.mu`. The table prints the mean of each
part over the assets of each currency, annualised.
=#

mu_est = M[:, est] * prior_full.fpr.mu[est]
mu_obs = M[:, obs] * prior_full.fpr.mu[obs]
group_mean(v) = [252 * mean(v[truth.code .== c]) for c in 1:3]
pretty_table(DataFrame("Currency" => currency_names,
                       "From estimated factors" => group_mean(mu_est),
                       "From observed factors" => group_mean(mu_obs),
                       "mu" => group_mean(prior_full.mu),
                       "Largest abs(sum - mu)" =>
                           maximum(abs, mu_est + mu_obs - prior_full.mu));
             formatters = [numfmt], title = "Annualised mean of mu by currency")

#=
### The same prior without the observed factors

We fit the prior again with the market and size factors alone. The currency and oil moves then have
nowhere to go but the idiosyncratic return of each asset. The data were drawn with an idiosyncratic
volatility of 1.2 % a day, 19.05 % a year, for every asset. The table prints the mean annualised
idiosyncratic volatility of the last row under each prior.
=#

pe_estimated = CrossSectionalFactorPrior(; factors = factors[1:2])
prior_estimated = prior(pe_estimated, rd)
idio_vol(p) = [mean(sqrt.(252 .* p.rr.vs[end, truth.code .== c])) for c in 1:3]
pretty_table(DataFrame("Currency" => currency_names,
                       "With observed factors" => idio_vol(prior_full),
                       "Without observed factors" => idio_vol(prior_estimated),
                       "True" => fill(0.012 * sqrt(252), 3)); formatters = [numfmt],
             title = "Mean annualised idiosyncratic volatility")

#=
With the observed factors the idiosyncratic volatility is close to the true value for every
currency. Without them it is too high by the oil move for dollar assets, and by the oil move and
the exchange rate for euro and yen assets.

### The summary

[`factor_model_summary`](@ref) reports the return statistics of each factor and the regression
statistics of the fit. An observed factor has return statistics, read off its observed series.
The fit did not estimate its return, so it has no t-statistic, and its regression columns are
`NaN`.
=#

summary_full = factor_model_summary(rr; ppy = 252)
pretty_table(DataFrame("Factor" => rr.nf, "Kind" => kind,
                       "Annual return" => summary_full.ann_return,
                       "Annual volatility" => summary_full.ann_volatility,
                       "Mean abs(t)" => summary_full.mean_abs_t,
                       "Share of abs(t) > 2" => summary_full.t_rate); formatters = [numfmt],
             title = "Factor model summary")

#=
### The attribution

[`factor_attribution`](@ref) splits the risk and the return of a book over the factors. We take an
equal-weighted book. The predicted attribution reads the moments of the prior. The realised one
reads the returns, and under `se = true` it also states the standard error of each factor's return
contribution. That error comes from the estimate of the factor return, so an observed factor, which
the fit reports as its observed return, has none, and its error is `NaN`.

The standard error reads the idiosyncratic variance of every observation, and the first rows of the
fit are the warm-up of that variance, where it is unknown. A realised call over every row of the fit
therefore gives `NaN` for every factor. The rolling form takes a window and a step, and its windows
end at `window`, `window + step` and so on. The attribution pairs the exposures of one observation
with the returns of the next, so it reads `rr.lag` rows fewer than the fit. We count the rows of the
warm-up, and choose the window and the step so that the last window starts after the warm-up and
ends at the last row.
=#

w_eq = fill(1 / 60, 60)
predicted = factor_attribution(w_eq, prior_full; ppy = 252)
warmup = count(r -> any(isnan, r), eachrow(rr.vs))
aligned = size(rr.fx, 1) - rr.lag
realised = last(factor_attribution(w_eq, prior_full, rd.X, aligned - warmup; step = warmup,
                                   se = true, ppy = 252))
window_rows = fit_rows[(warmup + rr.lag + 1):end]

#=
The table prints, for each factor, the predicted share of the variance of the book, the realised
mean return of the factor and its standard error. The last column is the annualised mean of the
observed series over the rows of the window, the return that the attribution reports for an
observed factor.
=#

pretty_table(DataFrame("Factor" => rr.nf, "Kind" => kind,
                       "Predicted % variance" => predicted.fbd.pct_var,
                       "Realised mu contribution" => realised.fbd.mu_contrib,
                       "Standard error" => realised.fbd.mu_se,
                       "Realised factor mean" => realised.fbd.mu,
                       "252 * mean of the series" => [if k in obs
                                                          252 * mean(rd.E[window_rows, k - first(obs) + 1])
                                                      else
                                                          NaN
                                                      end
                                                      for k in eachindex(rr.nf)]);
             formatters = [numfmt], title = "Attribution of the equal-weighted book")

#=
## 4. Where a single fit cannot reach

A meta-optimiser splits the universe and solves a sub-problem on each part. Each sub-problem fits the
prior again on its own assets, so it estimates its own market and size returns. The observed factor
returns are the same columns of `rd.E` in every sub-problem.

A cluster holds fewer assets than the universe, and the prior refuses an observation with fewer
than `minra` assets, 30 by default. We lower it, and ask for three clusters, so that the smallest
cluster still has enough assets to estimate two factors.
=#

#! A cluster smaller than `minra` stops the whole optimisation with an `ArgumentError`.
pe_small = CrossSectionalFactorPrior(; factors = factors, minra = 10)
solver = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                check_sol = (; allow_local = true, allow_almost = true),
                settings = Dict("verbose" => false))
inner = MeanRisk(; obj = MinimumRisk(),
                 opt = JuMPOptimiser(; pe = pe_small, slv = solver,
                                     wb = WeightBounds(; lb = 0.0, ub = 0.2)))

clustered = optimise(NestedClustered(; pe = pe_small,
                                     cle = ClustersEstimator(;
                                                             onc = OptimalNumberClusters(;
                                                                                         alg = 3)),
                                     opti = inner, opto = EqualWeighted()), rd)
clusters = [findall(==(k), assignments(clustered.clr)) for k in 1:(clustered.clr.k)]

#=
The table prints, for each cluster, how many assets of each currency it holds, the largest
difference between its observed factor returns and `rd.E`, and the largest difference between its
market factor return and the one of the full fit. A cluster with no yen asset still carries the yen
factor, with an exposure of zero for every asset.
=#

pretty_table(DataFrame("Cluster" => eachindex(clusters),
                       "USD" => [count(==(1), truth.code[c]) for c in clusters],
                       "EUR" => [count(==(2), truth.code[c]) for c in clusters],
                       "JPY" => [count(==(3), truth.code[c]) for c in clusters],
                       "Largest abs(rr.fx - rd.E)" =>
                           [maximum(abs, r.pr.rr.fx - rd.E[fit_rows, :])
                            for r in clustered.resi],
                       "Largest abs(market return - full fit)" =>
                           [maximum(abs, r.pr.fpr.X[:, 1] - prior_full.fpr.X[:, 1])
                            for r in clustered.resi]); formatters = [numfmt],
             title = "Each cluster fits its own market return")

#=
A [`SubsetResampling`](@ref) solves the same problem on random subsets of the assets and averages
the weights. Each subset fits its own factors too.
=#

resampled = optimise(SubsetResampling(; subset_size = 0.9, n_subsets = 8,
                                      rng = StableRNG(7), pe = pe_small, opt = inner), rd)

pretty_table(DataFrame("Subset" => eachindex(resampled.ress),
                       "Assets" => [length(r.w) for r in resampled.ress],
                       "Largest abs(rr.fx - rd.E)" =>
                           [maximum(abs, r.pr.rr.fx - rd.E[fit_rows, :])
                            for r in resampled.ress],
                       "Largest abs(market return - full fit)" =>
                           [maximum(abs, r.pr.fpr.X[:, 1] - prior_full.fpr.X[:, 1])
                            for r in resampled.ress]); formatters = [numfmt],
             title = "Each subset fits its own market return")

#=
A walk-forward fits the prior on each training window and holds the book over the next one. A fold
takes its rows of `rd`, and `rd.E` with them, so the observed factors of a fold are the series of
its own training rows. The table prints, for each fold, the rows it trained on, the rows its prior
fitted after the warm-up, and the largest difference between its observed factor returns and the
rows of `rd.E` it fitted.
=#

walk = IndexWalkForward(250, 125)
folds = cross_val_predict(inner, rd, walk)
train_start = [125 * (k - 1) + 1 for k in eachindex(folds.pred)]
fold_fx = [p.res.pr.rr.fx for p in folds.pred]

pretty_table(DataFrame("Fold" => eachindex(folds.pred),
                       "Training rows" => ["$(s) to $(s + 249)" for s in train_start],
                       "Fitted rows" => [size(f, 1) for f in fold_fx],
                       "Largest abs(rr.fx - rd.E)" =>
                           [maximum(abs, f - rd.E[(s + 250 - size(f, 1)):(s + 249), :])
                            for (f, s) in zip(fold_fx, train_start)]);
             formatters = [numfmt],
             title = "Each fold reads the exogenous series of its own rows")

#=
The predicted returns of the walk-forward carry the exogenous series of the test rows, so a step after
the walk-forward, such as an attribution of the held book, reads the series of the observations it
describes. The cell prints whether `folds.mrd.E` equals the rows of `rd.E` that the folds tested.
=#

folds.mrd.E == rd.E[251:end, :]

#=
## 5. The two return rules

An asset priced in euros has, in dollars, the return `(1 + local) * (1 + fx) - 1`. The prior
subtracts the currency excess return of the asset's currency from that, and the regression then
reads what is left as the local return. Under the log rule the base-currency return is the sum of
the local log return and the log return of the exchange rate, so the subtraction leaves the local
log return plus a term that is the same for every asset of one currency: the difference of the two
cash rates. The currency factor absorbs that term. Under the simple rule, the default of
[`prices_to_returns`](@ref), the subtraction also leaves the cross term `local * fx`, which differs
from asset to asset and no factor can absorb.

We repeat the subtraction by hand on both conversions, and compare what is left with the true local
return of each asset.
=#

rd_simple = prices_to_returns(pr; ret_method = :simple)
foreign = findall(>(1), truth.code)
function local_returns(r)
    L = copy(r.X)
    for i in foreign
        L[:, i] .-= r.E[:, truth.code[i] - 1]
    end
    return L
end
local_simple = expm1.(truth.local_lr)
left_log = local_returns(rd) - truth.local_lr
left_simple = local_returns(rd_simple) - local_simple
function spread(left)
    return maximum(maximum(std(left[t, truth.code .== c]) for c in 2:3)
                   for t in axes(left, 1))
end
cross_term = hcat([local_simple[:, i] .* expm1.(truth.fx_lr[:, truth.code[i] - 1])
                   for i in foreign]...)

#=
The first column prints, for each rule, the largest standard deviation across the assets of one
currency at one observation of what is left after the subtraction, less the true local return. A
term that the currency factor can absorb has a spread of zero. The other columns print the size of
the cross term against the size of the local returns.
=#

pretty_table(DataFrame("Rule" => [":log", ":simple"],
                       "Largest spread within a currency" =>
                           [spread(left_log), spread(left_simple)],
                       "Mean abs(cross term)" => [0.0, mean(abs, cross_term)],
                       "Largest abs(cross term)" => [0.0, maximum(abs, cross_term)],
                       "Mean abs(local return)" => [mean(abs, truth.local_lr[:, foreign]),
                                                    mean(abs, local_simple[:, foreign])]);
             formatters = [numfmt],
             title = "What the subtraction leaves in the local returns")

#=
Under the log rule the spread is at the level of rounding. Under the simple rule the mean cross term
is about half a percent of the mean local return, the largest is about a tenth of it, and the
regression reads the cross term as idiosyncratic return. Convert with `:log` when the universe holds more than
one currency.

## What to take away

  - `rd.E` carries the currency excess indices and the oil price as returns, under the rule of the
    asset returns, and the prior reads its observed factor returns from it with no change.
  - With two currency factors and an oil factor, the idiosyncratic volatility of every currency group
    is within 0.01 of the true 0.19 a year. Without them it is 0.24 to 0.28.
  - An observed factor has return statistics but no t-statistic and no standard error, because the
    fit did not estimate its return.
  - Each cluster, each subset and each fold fits its own market and size returns, and reads the same
    observed returns for the rows it covers.
  - Under the simple rule the subtraction of the currency leaves a cross term in each local return,
    and under the log rule it does not.

## Where to go next

  - [Cross-sectional factor model, end to end](@ref example-cross-sectional-factor-model-end-to-end)
    covers the estimated factors, the uncertainty sets and the constrained book of a cross-sectional
    prior.
  - [Cross-sectional factor model through a Pipeline](@ref example-cross-sectional-factor-model-through-a-pipeline)
    reaches the same book with each estimator as a named step.
  - [Meta-optimisers](@ref example-meta-optimisers) covers the nested clustered optimisation in
    more depth.
=#
