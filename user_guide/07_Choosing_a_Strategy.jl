#=
```@meta
Description = "A decision framework for choosing an optimiser, risk measure and prior in PortfolioOptimisers.jl from four questions about your mandate."
```

# Choosing a strategy

The previous pages showed how to call each tool. This page helps you choose one. Four questions
about your situation lead to a choice. The
[investor profiles](../examples/7_putting_it_together/01_Profile_Retail_Daily.md) work through four
complete cases, and this page gives the questions behind them.

## The four questions

### 1. How much compute can you spend on each rebalance?

Naive optimisers ([`InverseVolatility`](@ref), [`EqualWeighted`](@ref)) need no solver and return
at once. JuMP optimisers ([`MeanRisk`](@ref), [`RiskBudgeting`](@ref)) solve a convex program,
which takes longer than a naive rule. Meta-optimisers ([`NestedClustered`](@ref),
[`Stacking`](@ref)) and cross-validated tuning solve many programs and cost the most.

### 2. How often do you rebalance?

If you rebalance often, use cheap and stable rules, and bound the [`Turnover`](@ref) or charge the
[`Fees`](@ref), so that the trading costs do not cancel the excess return. If you rebalance
rarely, you can spend more compute on each optimisation.

### 3. How much do you trust your estimates?

If you have forecasts, add them to the prior with a view prior ([`BlackLittermanPrior`](@ref),
[`EntropyPoolingPrior`](@ref)). If you do not trust the moments, make the optimisation robust
with an uncertainty set on the covariance ([`UncertaintySetVariance`](@ref)) or on the expected
returns. Or use the hierarchy of the correlations ([`HierarchicalRiskParity`](@ref)) in place of
point estimates.

### 4. How large is your capital, and how many constraints does your mandate set?

A small account needs a finite allocation, such as [`GreedyAllocation`](@ref), to buy whole
shares and leave little cash. An institutional mandate has many constraints, such as weight
bounds, group limits and tracking, and the JuMP optimisers take all of them.

## A table of starting points

| Situation | Use |
| :-- | :-- |
| Little compute, and you want diversification | [`InverseVolatility`](@ref) or [`EqualWeighted`](@ref) |
| The trade-off between risk and return | [`MeanRisk`](@ref) with an objective, or its efficient frontier |
| Each asset carries the same share of the risk | [`RiskBudgeting`](@ref) |
| Tail losses or drawdowns matter more than the variance | set the `r` of `MeanRisk` to a tail measure ([`ConditionalValueatRisk`](@ref)) or a drawdown measure ([`MaximumDrawdown`](@ref)) |
| Many assets, and a covariance that changes | [`HierarchicalRiskParity`](@ref) or another clustering optimiser |
| You do not trust one fit | [`NestedClustered`](@ref) or [`Stacking`](@ref), or an uncertainty set |
| You have views | [`BlackLittermanPrior`](@ref) or [`EntropyPoolingPrior`](@ref) |
| Trading costs matter | add [`Turnover`](@ref) or [`Fees`](@ref) to a [`JuMPOptimiser`](@ref) |
| You trade the portfolio | end with [`GreedyAllocation`](@ref) |

Most strategies combine several rows, such as a view prior, constraints and a finite allocation.

## Three strategies on the same data

We run three strategies from the table on the same data: equal weights, the minimum-risk
`MeanRisk` and HRP. We plot their weights. In the plot, the minimum-risk bar has few colours, and
the HRP bar has one colour for every asset.
=#

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

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Final guide page: the distributed-capstone decision framework (4 questions + a map table),
#src   per ADR 0014. Worked profiles deferred to examples/7_putting_it_together (not yet authored
#src   — cross-links resolve once that group lands). Mostly prose; closes with a 3-archetype
#src   comparison plot reusing calls verified in guide 02 (EqualWeighted / MinRisk / HRP).
#src - Completes the user_guide group (00–06). Remaining: delete monolith 01_Basic_Optimisation.jl
#src   once confirmed its content is fully covered by 00–06 + the 3_optimisers examples.
