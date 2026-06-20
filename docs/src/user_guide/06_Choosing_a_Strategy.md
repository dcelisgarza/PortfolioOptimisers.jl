The source files can be found in [user_guide/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/user_guide/).

```@meta
EditURL = "../../../user_guide/06_Choosing_a_Strategy.jl"
```

# Choosing a strategy

The previous pages showed *how* to call each tool. This one is about *which* to reach for. There
is no single best optimiser — the right choice falls out of four questions about your situation.
The worked, end-to-end investor profiles live in
[putting it together](../examples/7_putting_it_together/01_Profile_Retail_Daily.md); this page is
the decision framework behind them.

## The four questions

1. **How much compute can you spend per rebalance?** Naive optimisers ([`InverseVolatility`](@ref),
   [`EqualWeighted`](@ref)) are instant and solver-free. JuMP optimisers ([`MeanRisk`](@ref),
   [`RiskBudgeting`](@ref)) solve a convex program — fast, but not free. Meta-optimisers
   ([`NestedClustered`](@ref), [`Stacking`](@ref)) and cross-validated tuning stack many solves
   and cost the most.

2. **How often do you rebalance?** High frequency rewards cheap, stable rules and tight
   [`Turnover`](@ref)/[`Fees`](@ref) control so costs do not eat the edge. Infrequent rebalancing
   can afford a heavier, more bespoke optimisation each time.

3. **How much do you trust your estimates?** If you have genuine forecasts, fold them in with a
   view prior ([`BlackLittermanPrior`](@ref), [`EntropyPoolingPrior`](@ref)). If you mostly
   distrust the noise in the moments, make the optimisation robust ([`UncertaintySetVariance`](@ref),
   a worst-case mean) or lean on the correlation hierarchy ([`HierarchicalRiskParity`](@ref))
   rather than point estimates.

4. **How large and constrained is the capital?** Small accounts need
   [`GreedyAllocation`](@ref) to round into whole shares without wasting cash. Institutional
   mandates pile on constraints — weight bounds, group limits, tracking — which is exactly what
   the JuMP optimisers are built for.

## A rough map

| Situation | Reach for |
| :-- | :-- |
| Minimal compute, just want diversification | [`InverseVolatility`](@ref) / [`EqualWeighted`](@ref) |
| Classic risk/return trade-off | [`MeanRisk`](@ref) with an objective + efficient frontier |
| Want each holding to carry equal risk | [`RiskBudgeting`](@ref) |
| Care about tail losses or drawdowns, not just variance | swap `MeanRisk`'s `r` to a tail ([`ConditionalValueatRisk`](@ref)) or drawdown ([`MaximumDrawdown`](@ref)) measure |
| Many assets, unstable covariance | [`HierarchicalRiskParity`](@ref) and clustering optimisers |
| Distrust a single fit, want robustness | [`NestedClustered`](@ref) / [`Stacking`](@ref), or uncertainty sets |
| Have real views | [`BlackLittermanPrior`](@ref) / [`EntropyPoolingPrior`](@ref) |
| Tight rebalancing budget | add [`Turnover`](@ref) / [`Fees`](@ref) to a [`JuMPOptimiser`](@ref) |
| Trading real money | finish with [`GreedyAllocation`](@ref) |

These are starting points, not rules — most real strategies combine several (a view prior *and*
constraints *and* finite allocation). The
[putting-it-together profiles](../examples/7_putting_it_together/01_Profile_Retail_Daily.md) walk
three complete examples end to end.

## The choice is real

To make the point concrete: three archetypes from the map — pure diversification, the
risk/return workhorse, and the hierarchy-based rule — produce visibly different portfolios on the
same data. Picking a strategy is picking one of these shapes.

````@example 06_Choosing_a_Strategy
using PortfolioOptimisers, CSV, TimeSeries, Clarabel, StatsPlots, GraphRecipes

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res_ew = optimise(EqualWeighted(), rd)
res_mr = optimise(MeanRisk(; obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv)))
res_hrp = optimise(HierarchicalRiskParity(; r = Variance(),
                                          opt = HierarchicalOptimiser(; pe = pr,
                                                                      cle = clusterise(ClustersEstimator(),
                                                                                       pr.X))))

plot_stacked_bar_composition([res_ew, res_mr, res_hrp], rd;
                             xticks = (1:3, ["EqualWeighted", "MinRisk", "HRP"]))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
