#=
```@meta
Description = "Drawdown risk measures in PortfolioOptimisers.jl: average, maximum and conditional drawdown at risk on the path of cumulative returns."
```

# Drawdown risk measures

A drawdown is how far the running sum of the portfolio returns is below its previous high at each
point in time. The variance and the conditional value at risk (CVaR) use the distribution of the
returns of single periods, and the order of those returns does not count. A drawdown measure uses
the *path* of the cumulative returns, which gives the sum of the returns from any peak to any
later trough.

This page uses four of the library's drawdown measures.

| Measure | What it penalises |
| ------- | ----------------- |
| `MaximumDrawdown` | The worst decline from a peak to a trough over the whole period |
| `AverageDrawdown` | The mean depth of the drawdown curve over time |
| `UlcerIndex` | The root mean square of the drawdown curve, which penalises deep drawdowns more than `AverageDrawdown` does |
| `ConditionalDrawdownatRisk` | The conditional drawdown at risk (CDaR), the drawdown form of CVaR. It is the mean drawdown over the worst fraction `α` of days |

!!! tip "When to reach for this"
    Reach for a drawdown measure when the *path to recovery* matters. Examples are a
    trend-following strategy, a strategy for retail investors who can sell at the worst
    moment, and any portfolio whose report states the depth and the length of its drawdowns.
    Minimising the variance ignores the order of the returns, and these measures depend on it.
=#

using PortfolioOptimisers, PrettyTables, DataFrames

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Data and shared setup
=#

using CSV, TimeSeries, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true))]

opt = JuMPOptimiser(; pe = pr, slv = slv)

#=
## 2. Minimising each drawdown measure

[`MeanRisk`](@ref) accepts these measures with their defaults. Each constructor takes an optional
`settings::RiskMeasureSettings`. `AverageDrawdown` and `ConditionalDrawdownatRisk` also take
observation weights `w`, and `ConditionalDrawdownatRisk` takes `alpha`, the tail probability, which
is 0.05 by default. We minimise each measure and print the weights.
=#

r_mdd = MaximumDrawdown()
r_add = AverageDrawdown()
r_uci = UlcerIndex()
r_cdar = ConditionalDrawdownatRisk()

results = map([r_mdd, r_add, r_uci, r_cdar]) do r
    return optimise(MeanRisk(; r = r, opt = opt))
end
labels = ["MDD", "ADD", "Ulcer", "CDaR 5%"]

pretty_table(DataFrame(hcat(rd.nx, [r.w for r in results]...),
                       [:assets; Symbol.(labels)...]); formatters = [resfmt])

#=
`MaximumDrawdown` depends only on the worst decline, so it puts its weight in the assets that
reduce that one loss. `AverageDrawdown` and `UlcerIndex` use the whole drawdown curve, and `CDaR`
uses the worst 5 % of it.
=#

using StatsPlots, GraphRecipes
plot_stacked_bar_composition(results, rd)

#=
Four drawdown curves on one line chart are hard to tell apart, so we compare them with a
heatmap. Each row is one portfolio and each column one day. The colour of a cell shows the
depth of the drawdown, and the colour bar gives the scale.
=#

drawdown_grid = hcat([(-drawdowns(rd.X * res.w)) for res in results]...)
heatmap(eachindex(rd.ts), labels, drawdown_grid'; xlabel = "Day",
        ylabel = "Minimised measure", colorbar_title = "Drawdown",
        title = "Daily drawdown depth of each portfolio")

#=
## 3. The tail level `alpha` of CDaR

`alpha` sets the fraction of the worst days that `ConditionalDrawdownatRisk` averages. As
`alpha → 0` it uses only the deepest drawdowns. With a larger `alpha` the measure averages more
days, and it moves toward the average drawdown. We minimise CDaR for four values of `alpha` and
print the weights.
=#

alphas = [0.01, 0.05, 0.1, 0.25]
cdar_results = [optimise(MeanRisk(; r = ConditionalDrawdownatRisk(; alpha = a), opt = opt))
                for a in alphas]

pretty_table(DataFrame(hcat(rd.nx, [r.w for r in cdar_results]...),
                       [:assets; Symbol.("CDaR_" .* string.(alphas))...]);
             formatters = [resfmt])

#=
As `alpha` rises, the weight of JNJ falls and the weights of MRK and UNH grow. All three move
toward their weights in the `ADD` column of section 2.

## 4. Bounding the drawdown instead of minimising it

A drawdown measure need not be the objective. You can set an upper bound on it and optimise
the return instead. We maximise the risk-adjusted return with CDaR at most 0.08, and set that
bound as the `ub` of [`RiskMeasureSettings`](@ref).
=#

rf = 4.2 / 100 / 252
r_cdar_ub = ConditionalDrawdownatRisk(; settings = RiskMeasureSettings(; ub = 0.08))
res_cdar_max_ratio = optimise(MeanRisk(; r = r_cdar_ub, obj = MaximumRatio(; rf = rf),
                                       opt = opt))
println("CDaR-constrained max-ratio retcode: $(res_cdar_max_ratio.retcode)")

#=
## 5. Drawdown statistics after the optimisation

When you have a portfolio, `drawdowns()` and `cumulative_returns()` return its drawdown series
and its cumulative return series over the sample. By default both add the returns up without
compounding. They are not objectives, so use them to study a portfolio after the optimiser has
run. `expected_risk` with the measures of section 2 gives the statistics. We compare the
minimum-variance portfolio with the minimum-CDaR portfolio of section 2.
=#

w_var = optimise(MeanRisk(; r = Variance(), opt = opt)).w
w_cdar = results[4].w

ret_var = rd.X * w_var
ret_cdar = rd.X * w_cdar

cr_var = cumulative_returns(ret_var)
cr_cdar = cumulative_returns(ret_cdar)

dd_var = drawdowns(ret_var)
dd_cdar = drawdowns(ret_cdar)

rs_dd = [r_mdd, r_add, r_uci, r_cdar]
pretty_table(DataFrame(;
                       :Metric => ["Maximum drawdown", "Average drawdown", "Ulcer index",
                                   "CDaR 5%"],
                       :MinVariance => [expected_risk(r, w_var, rd.X) for r in rs_dd],
                       :MinCDaR => [expected_risk(r, w_cdar, rd.X) for r in rs_dd]);
             formatters = [resfmt])

#=
Compare the two columns row by row. The minimum-variance portfolio has the lower variance, but
the variance ignores the order of the returns, and the four rows compare the depth of the
drawdowns of the two portfolios. We plot the cumulative returns of both portfolios.
=#

plot(cr_var; label = "Minimum variance", xlabel = "Day", ylabel = "Cumulative return",
     title = "Cumulative return paths")
plot!(cr_cdar; label = "Minimum CDaR")

# We plot the drawdowns of the same two portfolios.

plot(dd_var; label = "Minimum variance", xlabel = "Day", ylabel = "Drawdown",
     title = "Drawdown paths")
plot!(dd_cdar; label = "Minimum CDaR")

#=
## Summary

Drawdown measures use the *path* of the cumulative returns.

  - [`MaximumDrawdown`](@ref) depends on the worst decline alone, and its portfolio can put most
    of its weight in a few assets.
  - [`AverageDrawdown`](@ref) and [`UlcerIndex`](@ref) penalise the whole drawdown curve.
  - [`ConditionalDrawdownatRisk`](@ref) is the drawdown form of CVaR, and `alpha` sets its
    tail as it does for CVaR.
  - `drawdowns()` and `cumulative_returns()` return the drawdown series and the cumulative
    return series of any portfolio, `rd.X * w`, and `expected_risk` computes the statistics,
    without a new optimisation.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env): the four drawdown measures, the CDaR
#src   alpha-sweep (0.01–0.25), the CDaR upper-bound-constrained `MaximumRatio` solve
#src   (`OptimisationSuccess`), and the post-optimisation analytics all run with Clarabel.
#src - The narrative holds against the numbers: `MaximumDrawdown` concentrates hardest
#src   (~67% JNJ), the CDaR alpha-sweep shifts weight JNJ→MRK monotonically as alpha rises,
#src   and in the diagnostics table the MinCDaR portfolio beats MinVariance on all four
#src   drawdown metrics — the intended "variance is path-blind" point lands.
#src - The drawdown heatmap (vs. overlaid spaghetti lines) is the right call for comparing many
#src   path-optimised portfolios; this is already captured in the summary as guidance.
#src - No solver warnings or plotting deprecations observed.
