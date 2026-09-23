#=
```@meta
Description = "Performance attribution in PortfolioOptimisers.jl: cumulative returns, drawdowns, risk contributions and fees as diagnostics after optimise."
```

# Performance attribution and post-optimisation diagnostics

An optimiser gives you weights. It does not tell you how the portfolio behaves. Before you trust an
allocation, you measure what it would have done: how wealth would have grown, how deep and how long
the drawdowns were, which assets the risk comes from, and how much the fees take from the return.
None of these is an objective of the optimisation. You compute them after `optimise`, and they
take any weights: the output of an optimiser, a benchmark, a portfolio you already hold, or equal
weights.

The [plotting and reporting](02_Plotting_and_Reporting.md) page draws plots. This page uses the
functions that return the numbers and the series behind those plots, so you can put them in a
table and compare them.

  - [`cumulative_returns`](@ref) returns the equity curve, simple (a sum) or compounded (a
    product).
  - [`drawdowns`](@ref) returns the loss from the running peak, from which the maximum drawdown,
    the average drawdown and the Ulcer index follow.
  - [`calc_net_returns`](@ref) and [`calc_fees`](@ref) return the returns net of fees and the
    fees themselves, so you can see how much of the performance the fees take.
  - [`risk_contribution`](@ref) returns the share of the risk of the portfolio that each asset
    contributes.

!!! tip "When to reach for this"
    Reach for these functions after you choose a portfolio, to compare candidates on the returns
    they would have made, on the concentration of their risk and on their fees, without a new
    optimisation. They take plain weights, so they also report on a benchmark or on a portfolio
    from outside the library.
=#

using PortfolioOptimisers, PrettyTables, DataFrames, Statistics

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Candidate portfolios

We compare three portfolios: a minimum-variance and a maximum-ratio portfolio from
[`MeanRisk`](@ref), and an equal-weight portfolio. The equal-weight portfolio is a plain vector,
and every function below takes it as it takes the output of an optimiser.
=#

using CSV, TimeSeries, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
rf = 4.2 / 100 / 252

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

w_min = optimise(MeanRisk(; obj = MinimumRisk(), opt = JuMPOptimiser(; pe = pr, slv = slv))).w
w_ratio = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                            opt = JuMPOptimiser(; pe = pr, slv = slv))).w
w_ew = fill(inv(length(rd.nx)), length(rd.nx))

books = ["Min variance" => w_min, "Max ratio" => w_ratio, "Equal weight" => w_ew]
## Realised in-sample portfolio return series for each book.
port_ret = [name => rd.X * w for (name, w) in books]

#=
## 2. Cumulative returns: the equity curve

[`cumulative_returns`](@ref) turns a series of returns into a path of wealth. Its `compound` flag
chooses how:

  - `compound = false`, the default, sums the returns, `cumsum(X)`. This is the absolute
    cumulative return. It adds up period by period and is easy to read over a short horizon.
  - `compound = true` multiplies them, `cumprod(1 .+ X)`. This is the relative, or geometric,
    multiple of wealth, which is what an investor who reinvests gets.

[`absolute_cumulative_returns`](@ref) and [`relative_cumulative_returns`](@ref) compute each of the
two directly. The table prints the last compounded multiple of wealth of each portfolio.
=#

final_wealth = [name => cumulative_returns(r, true)[end] for (name, r) in port_ret]
pretty_table(DataFrame(; book = first.(final_wealth),
                       Symbol("compounded wealth (×)") =>
                           [round(last(p); digits = 4) for p in final_wealth]);
             title = "Final compounded wealth multiple over the sample")

#=
## 3. Drawdowns

[`drawdowns`](@ref) returns the drawdown at each point, the loss from the highest value reached
before it. From that series we compute three statistics: the maximum drawdown, which is the worst
loss; the average drawdown, which is the mean depth over time; and the Ulcer index, the root mean
square of the depth, which weighs a long drawdown more than a short spike. We use compounded
drawdowns, to match the compounded equity curve.
=#

dd_stats = map(port_ret) do (name, r)
    dd = drawdowns(r, true)          # compounded drawdown series, ≤ 0
    return (name = name, max_dd = -minimum(dd), avg_dd = -mean(dd),
            ulcer = sqrt(mean(dd .^ 2)))
end
pretty_table(DataFrame(dd_stats); formatters = [resfmt], title = "Drawdown analytics")

#=
## 4. A scorecard of performance

We compute the statistics you would report for a portfolio: the annualised return and volatility,
the annualised Sharpe ratio net of the risk-free rate, the maximum drawdown, and the Calmar ratio,
which is the annualised return divided by the maximum drawdown. All of them come from the return
series of each portfolio, with no new optimisation.
=#

scorecard = map(port_ret) do (name, r)
    ann_ret = mean(r) * 252
    ann_vol = std(r) * sqrt(252)
    sharpe = (mean(r) - rf) / std(r) * sqrt(252)
    max_dd = -minimum(drawdowns(r, true))
    return (book = name, ann_return = ann_ret, ann_vol = ann_vol, sharpe = sharpe,
            max_drawdown = max_dd, calmar = ann_ret / max_dd)
end
pretty_table(DataFrame(scorecard);
             formatters = [(v, i, j) -> (if j in (2, 3, 5)
                                             "$(round(v*100, digits=2)) %"
                                         elseif isa(v, Number)
                                             round(v; digits = 3)
                                         else
                                             v
                                         end)], title = "Realised-performance scorecard")

#=
## 5. Fees: gross and net returns

A portfolio with a good gross return can have a poor one after fees. [`calc_net_returns`](@ref)
deducts a [`Fees`](@ref) schedule from the returns. [`calc_fees`](@ref) returns the fees as a
pair: the charge on every observation, and the one-off charge on the first observation alone.
[`calc_total_fees`](@ref) adds the pair up over a whole holding period.

The fees have two time bases. `l`, `s` and `tn` are rates per period, so
`calc_net_returns(w, X, fees)` deducts them on every row of `X`. `l` is proportional to the long
positions and `s` to the short ones, and `tn` to the turnover. Over 252 daily observations, a rate
`l` per period adds up to about `252 * l` over the year, before compounding. `fl` and `fs` are
amounts of currency charged once for the whole holding period, and `fees.fa` decides where on the
series that charge falls. We charge `l = 0.0005`, five basis points per period on the long
positions, and compare the compounded wealth gross and net of it. The table also prints the fee of
one period and the total over 252 periods.
=#

fees = Fees(; l = 0.0005)
gross_ret = rd.X * w_ratio
net_ret = calc_net_returns(w_ratio, rd.X, fees)
single_period_fee, _ = calc_fees(w_ratio, size(rd.X, 1), fees)

pretty_table(DataFrame(;
                       quantity = ["Gross compounded wealth (×)",
                                   "Net compounded wealth (×)",
                                   "Single-period fee (fraction)",
                                   "Approx annualised fee (252 periods)"],
                       value = [round(cumulative_returns(gross_ret, true)[end]; digits = 4),
                                round(cumulative_returns(net_ret, true)[end]; digits = 4),
                                round(single_period_fee; digits = 5),
                                round(calc_total_fees(w_ratio, 252, fees); digits = 4)]);
             title = "Fee drag on the maximum-ratio book (5 bps per rebalance)")

#=
## 6. Risk attribution

An asset's share of the weight and its share of the risk can differ.
[`risk_contribution`](@ref) splits the total risk into the share of each asset. As with the plots,
a quadratic risk measure needs the covariance of the data, so pass `factory(Variance(), pr)`, not
a bare `Variance()`. We scale the contributions to sum to one, and the table prints the eight
largest of the minimum-variance portfolio beside their weights.
=#

rc = risk_contribution(factory(Variance(), pr), w_min, rd.X)
rc ./= sum(rc)
rc_df = sort(DataFrame(; asset = rd.nx, weight = w_min, risk_share = rc), :risk_share;
             rev = true)
pretty_table(first(rc_df, 8); formatters = [resfmt],
             title = "Top risk contributors — minimum-variance book")

#=
A share of the weight and a share of the risk are different numbers. An asset with a small weight
and a high volatility, or a high correlation with the others, can contribute more risk than its
weight. Risk attribution shows where the two differ.

## 7. The equity curves

The last plot draws the compounded wealth of the three portfolios over the sample.
=#

using StatsPlots, GraphRecipes
curves = [name => cumulative_returns(r, true) for (name, r) in port_ret]
plot(cumulative_returns(port_ret[1][2], true); label = first(port_ret[1]), xlabel = "Day",
     ylabel = "Compounded wealth (×)", title = "Realised equity curves", legend = :topleft)
for (name, r) in port_ret[2:end]
    plot!(cumulative_returns(r, true); label = name)
end
current()

#=
## Summary

After the optimisation, these functions measure how a portfolio would have behaved, with no new
optimisation:

  - [`cumulative_returns`](@ref), simple or compounded, is the equity curve.
    [`drawdowns`](@ref) is the loss from the running peak, from which the maximum drawdown and
    the Ulcer index follow.
  - [`calc_net_returns`](@ref) and [`calc_fees`](@ref) separate the gross performance from the
    net performance.
  - [`risk_contribution`](@ref) shows which assets the risk comes from, which the weights alone
    do not show.

Each of these takes plain weights, so they report on the output of an optimiser, on a benchmark,
or on any portfolio from outside the library.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New page; closes the performance-attribution backlog item for 6_post_processing. Runs
#src   end-to-end under Kaimon (docs env): cumulative_returns, drawdowns, the scorecard,
#src   calc_net_returns/calc_fees, and risk_contribution all populate. Numbers sanity-check:
#src   Max ratio Sharpe 2.39 / Calmar 5.28 / maxDD 9.5%, equal-weight Sharpe ≈ 0; min-variance
#src   risk shares track its weights almost exactly (JNJ 37%, MRK 17%) since it is built on the
#src   same covariance.
#src - FINDING (semantic gotcha → post-processing rollup): `calc_net_returns(w, X, fees)` deducts
#src   the per period terms on EVERY row of `X` (per-period rebalance cost), whereas
#src   `calc_fees(w, T, fees)` returns the pair `(amortised, one_time)` for ONE observation. A
#src   naïve "20 bps" (`l = 0.002`) on a 252-row daily series therefore compounds into a ~40%
#src   wealth drag (net 0.98 vs gross 1.62) — surprising and easy to misread. The page uses
#src   `l = 0.0005` and spells out the per-period time base.
#src   RESOLVED by #898: the rule is now stated in the type. `l`, `s` and `tn` are rates per
#src   period; `fl` and `fs` are charged one time for the whole holding period, and `fees.fa`
#src   names the clock they land on. `calc_total_fees(w, T, fees)` reports the whole horizon, so
#src   the annualised row of the table below is a library call rather than a hand multiplication.
#src - CONFIRMED known gotcha (shared with 02_Plotting_and_Reporting): `risk_contribution` needs
#src   `factory(Variance(), pr)`, not a bare `Variance()`; documented inline in §6.
