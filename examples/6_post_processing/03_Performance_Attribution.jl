#=
# Performance attribution and post-optimisation diagnostics

An optimiser hands you weights; it does not tell you whether the resulting book is any good.
Before trusting an allocation you analyse its *realised* behaviour: how wealth would have grown,
how deep and how long the drawdowns were, where the risk actually sits, and how much trading
costs eat into the result. None of these are objectives — they are **diagnostics you run after
`optimise`**, and they work on *any* weight vector, not just optimiser output (a benchmark, a
live book, an equal-weight sleeve).

Where the [plotting and reporting](02_Plotting_and_Reporting.md) page is a visual tour, this one
is the *quantitative* companion: the raw functions that return the numbers and series behind the
plots, so you can tabulate, compare, and attribute.

  - [`cumulative_returns`](@ref) — the equity curve, simple (sum) or compounded (product).
  - [`drawdowns`](@ref) — the peak-to-trough path, from which max drawdown, average drawdown,
    and the Ulcer index follow.
  - [`calc_net_returns`](@ref) and [`calc_fees`](@ref) — realised returns and cost drag after
    fees, so you can attribute how much performance the trading costs consumed.
  - [`risk_contribution`](@ref) — where the portfolio's risk comes from, by asset.

!!! tip "When to reach for this"
    Reach for these after you have chosen a portfolio, to understand and compare candidates on
    realised path behaviour, risk concentration, and cost drag — rather than re-optimising.
    Because they take a plain weight vector, they are equally the tools for reporting on a
    benchmark or an externally-supplied book.
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
## 1. Candidate books

We build three portfolios to compare: a minimum-variance and a maximum-ratio book from
[`MeanRisk`](@ref), plus a naive equal-weight sleeve. The equal-weight book is just a vector —
it shows that every diagnostic below works on any weights, not only optimiser results.
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

[`cumulative_returns`](@ref) turns a return series into a wealth path. Its `compound` flag picks
the convention:

  - `compound = false` (the default) sums the returns: `cumsum(X)` — the *absolute* cumulative
    return, additive and easy to reason about for short horizons.
  - `compound = true` multiplies them: `cumprod(1 .+ X)` — the *relative* (geometric) wealth
    multiple, which is what an investor actually realises through reinvestment.

The dedicated [`absolute_cumulative_returns`](@ref) and [`relative_cumulative_returns`](@ref)
expose the two conventions directly. We report each book's final compounded wealth multiple.
=#

final_wealth = [name => cumulative_returns(r, true)[end] for (name, r) in port_ret]
pretty_table(DataFrame(; book = first.(final_wealth),
                       Symbol("compounded wealth (×)") =>
                           [round(last(p); digits = 4) for p in final_wealth]);
             title = "Final compounded wealth multiple over the sample")

#=
## 3. Drawdown analytics

[`drawdowns`](@ref) returns the full peak-to-trough series (each point is the loss from the
running high). From it we derive the three headline drawdown statistics: the **maximum
drawdown** (worst single decline), the **average drawdown** (time-average depth), and the
**Ulcer index** (root-mean-square depth, which punishes long drawdowns more than shallow
spikes). We compute compounded drawdowns to match the geometric equity curve.
=#

dd_stats = map(port_ret) do (name, r)
    dd = drawdowns(r, true)          # compounded drawdown series, ≤ 0
    return (name = name, max_dd = -minimum(dd), avg_dd = -mean(dd),
            ulcer = sqrt(mean(dd .^ 2)))
end
pretty_table(DataFrame(dd_stats); formatters = [resfmt], title = "Drawdown analytics")

#=
## 4. A realised-performance scorecard

Putting the path statistics together gives the kind of scorecard you would report for a book:
annualised return and volatility, the annualised Sharpe ratio (net of the risk-free rate), the
maximum drawdown, and the Calmar ratio (annualised return over maximum drawdown). All are derived
from the realised return series — no re-optimisation.
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
## 5. Cost attribution: gross vs net returns

A book that looks good gross can be mediocre net of trading costs. [`calc_net_returns`](@ref)
applies a [`Fees`](@ref) schedule to the realised returns; [`calc_fees`](@ref) reports the cost
of holding the weights for a *single* period.

The important subtlety is the time base: `calc_net_returns(w, X, fees)` deducts the fee on
**every row** of `X` — it models paying the rebalancing cost *each period*. So with 252 daily
observations a per-period fee of `l` accumulates to roughly `252 · l` over the year before
compounding. We therefore use a modest per-rebalance fee of 5 bps (`l = 0.0005`), which is about
a 12–13% annualised cost, and compare gross and net compounded wealth.
=#

fees = Fees(; l = 0.0005)
gross_ret = rd.X * w_ratio
net_ret = calc_net_returns(w_ratio, rd.X, fees)
single_period_fee = calc_fees(w_ratio, fees)

pretty_table(DataFrame(;
                       quantity = ["Gross compounded wealth (×)",
                                   "Net compounded wealth (×)",
                                   "Single-period fee (fraction)",
                                   "Approx annualised fee (252 periods)"],
                       value = [round(cumulative_returns(gross_ret, true)[end]; digits = 4),
                                round(cumulative_returns(net_ret, true)[end]; digits = 4),
                                round(single_period_fee; digits = 5),
                                round(252 * single_period_fee; digits = 4)]);
             title = "Fee drag on the maximum-ratio book (5 bps per rebalance)")

#=
## 6. Risk attribution

A book can look diversified by *weight* yet be concentrated in *risk*. [`risk_contribution`](@ref)
decomposes the total risk into per-asset shares. As with the plotting layer, a quadratic risk
measure must be configured for the data — pass `factory(Variance(), pr)`, not a bare `Variance()`.
We normalise the contributions so they sum to one and report the largest for the minimum-variance
book.
=#

rc = risk_contribution(factory(Variance(), pr), w_min, rd.X)
rc ./= sum(rc)
rc_df = sort(DataFrame(; asset = rd.nx, weight = w_min, risk_share = rc), :risk_share;
             rev = true)
pretty_table(first(rc_df, 8); formatters = [resfmt],
             title = "Top risk contributors — minimum-variance book")

#=
The weight and risk shares are not the same: a low-weight, high-volatility or
highly-correlated name can carry a disproportionate share of the risk. That gap is exactly what
risk attribution surfaces.

## 7. The equity curves

Finally, the compounded wealth paths of the three books side by side — the visual summary of
everything the scorecard quantified.
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

After the optimiser runs, the post-processing toolkit answers "how did this book actually
behave?" without any re-optimisation:

  - [`cumulative_returns`](@ref) (simple/compounded) is the equity curve;
    [`drawdowns`](@ref) gives the loss path and the max-drawdown / Ulcer statistics.
  - [`calc_net_returns`](@ref) and [`calc_fees`](@ref) attribute the cost drag, separating gross
    from net performance.
  - [`risk_contribution`](@ref) shows where the risk lives, which weight alone hides.

Every one of these takes a plain weight vector, so the same diagnostics report on optimiser
output, a benchmark, or any externally-supplied portfolio.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New page; closes the performance-attribution backlog item for 6_post_processing. Runs
#src   end-to-end under Kaimon (docs env): cumulative_returns, drawdowns, the scorecard,
#src   calc_net_returns/calc_fees, and risk_contribution all populate. Numbers sanity-check:
#src   Max ratio Sharpe 2.39 / Calmar 5.28 / maxDD 9.5%, equal-weight Sharpe ≈ 0; min-variance
#src   risk shares track its weights almost exactly (JNJ 37%, MRK 17%) since it is built on the
#src   same covariance.
#src - FINDING (semantic gotcha → post-processing rollup): `calc_net_returns(w, X, fees)` deducts
#src   the fee on EVERY row of `X` (per-period rebalance cost), whereas `calc_fees(w, fees)`
#src   returns the SINGLE-period fee. A naïve "20 bps" (`l = 0.002`) on a 252-row daily series
#src   therefore compounds into a ~40% wealth drag (net 0.98 vs gross 1.62) — surprising and easy
#src   to misread. The page uses `l = 0.0005` and spells out the per-period time base. Worth a
#src   docstring note that `calc_net_returns` models a per-period rebalancing cost (annual ≈ 252·l).
#src - CONFIRMED known gotcha (shared with 02_Plotting_and_Reporting): `risk_contribution` needs
#src   `factory(Variance(), pr)`, not a bare `Variance()`; documented inline in §6.
