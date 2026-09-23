#=
```@meta
Description = "Plotting and reporting in PortfolioOptimisers.jl: inputs, allocation, risk contribution, performance and one-call dashboards from StatsPlots."
```

# Plotting and reporting

`PortfolioOptimisers.jl` has plotting functions for each stage of an optimisation. They load when
you load `StatsPlots` and `GraphRecipes` with the package. They plot the inputs, the allocation,
the risk of each asset, the performance over the sample, and the portfolios on axes of risk and
return. Two dashboards put several of these plots into one figure. This page draws them for one
pair of portfolios.

!!! tip "When to reach for this"
    Reach for these plots to check the inputs before you optimise, to show an allocation to
    someone else, and to compare candidate portfolios on the same axes. Each plot shows one view.
    The dashboards, [`plot_portfolio_dashboard`](@ref) and [`plot_performance_summary`](@ref), put
    several views in one call.
=#

using PortfolioOptimisers, CSV, TimeSeries, Clarabel, StatsPlots, GraphRecipes

#=
## 1. The portfolios

We compute an empirical prior, a minimum-risk portfolio and a maximum-ratio portfolio to compare,
and an efficient frontier for the plots of risk and return.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
rf = 4.2 / 100 / 252

res_min = optimise(MeanRisk(; obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv)))
res_ratio = optimise(MeanRisk(; obj = MaximumRatio(; rf = rf),
                              opt = JuMPOptimiser(; pe = pr, slv = slv)))
frontier = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                 ret = ArithmeticReturn(;
                                                                        settings = JuMPReturnsSettings(;
                                                                                                       lb = Frontier(;
                                                                                                                     N = 15))))))

#=
## 2. The inputs

Look at the inputs before you trust an optimisation. [`plot_prior`](@ref) plots the prior in one
figure. [`plot_correlation`](@ref) plots the correlation matrix alone, and [`plot_mu`](@ref) plots
the expected returns alone.
=#

plot_prior(pr, rd)

#=
The correlation matrix alone.
=#

plot_correlation(pr)

#=
## 3. The allocation

[`plot_stacked_bar_composition`](@ref) puts the weights of several portfolios side by side. Here
it compares the minimum-risk and the maximum-ratio portfolios, and you can see which of the two
puts more weight on fewer assets.
=#

plot_stacked_bar_composition([res_min, res_ratio], rd;
                             xticks = (1:2, ["Min risk", "Max ratio"]))

#=
## 4. The risk of each asset

A portfolio can spread its weights over many assets and still take most of its risk from a few.
[`plot_risk_contribution`](@ref) splits the risk of the portfolio by asset. It takes the risk
measure as its first argument. A quadratic risk measure needs the covariance of the data, so pass
`factory(Variance(), pr)`, not a bare `Variance()`, which holds no covariance. A measure computed
from returns, such as [`ConditionalValueatRisk`](@ref), needs no `factory`.
=#

plot_risk_contribution(factory(Variance(), pr), res_min, rd)

#=
## 5. Performance over the sample

[`plot_portfolio_cumulative_returns`](@ref) and [`plot_drawdowns`](@ref) plot the returns the
maximum-ratio portfolio would have made over the sample.
=#

plot_portfolio_cumulative_returns(res_ratio.w, rd)

#=
The drawdowns.
=#

plot_drawdowns(res_ratio.w, rd)

#=
[`plot_performance_summary`](@ref) puts the main performance plots into one figure. It plots a
[`PerformanceSummaryResult`](@ref), which [`performance_summary`](@ref) computes. Call
`performance_summary` when you want the numbers and not the plot, for example to put them in a
table, to compare two portfolios, or to test them. It needs no plotting package.
=#

performance_summary(res_ratio, rd)

#=
The same numbers as a plot.
=#

plot_performance_summary(res_ratio, rd)

#=
## 6. Risk and return

[`plot_measures`](@ref) plots portfolios on any pair of risk and return axes.
[`plot_efficient_frontier`](@ref) plots the efficient frontier, the portfolios with the least risk
for each level of return.
=#

plot_efficient_frontier(frontier.w, pr; rt = frontier.ret)

#=
## 7. The dashboard

[`plot_portfolio_dashboard`](@ref) puts the composition, the risk and the performance of one
portfolio into one figure.
=#

plot_portfolio_dashboard(res_ratio, rd; r = factory(Variance(), pr))

#=
This page shows some of the plots, not all. The library also plots networks and clusters
([`plot_network`](@ref), [`plot_dendrogram`](@ref), [`plot_clusters`](@ref),
[`plot_centrality`](@ref)), cross-validation ([`plot_cv_scores`](@ref),
[`plot_cv_dashboard`](@ref)), turnover ([`plot_turnover`](@ref)), and the higher moments
([`plot_coskewness`](@ref), [`plot_cokurtosis`](@ref)).
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New deep dive; closes the 6_post_processing group. All plot calls verified to execute on
#src   kaimon (f102cae9): plot_prior, plot_correlation, plot_mu, plot_stacked_bar_composition,
#src   plot_portfolio_cumulative_returns, plot_drawdowns, plot_performance_summary, plot_measures,
#src   plot_efficient_frontier all ok directly.
#src - CONFIRMED known gotcha (arch-review item): plot_risk_contribution and
#src   plot_portfolio_dashboard with a bare quadratic risk measure (Variance()) throw a cryptic
#src   MethodError. Fix: factory(Variance(), pr) (attaches sigma), or use a return-based measure
#src   (ConditionalValueatRisk()) which needs no factory. Documented inline in §4. A
#src   factory-pointer guard with a clear error message is still wanted.
#src - SIGNATURE: plot_risk_contribution(r, res, rd) — risk measure FIRST; plot_portfolio_dashboard(res, rd; r=...).
