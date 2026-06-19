The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/3_optimisers/12_Clustering_Mixed_Risks_And_Constraints.jl"
```

# Clustering optimisers with mixed risks and constraints

This example turns the clustering optimiser chapter into a deeper playground for two things
that are easy to miss in the overview:

- clustering optimisers can mix risk measures and scalarisers across the hierarchy;
- the hierarchical optimiser can still carry constraints and fees while the cluster logic
    does the diversification work.

We use the same S&P 500 slice as the rest of the examples, then compare a plain HRP solve,
several mixed-risk HERC variants, and a constrained HERC solve.

!!! tip "When to reach for this"
    Reach for mixed-risk clustering optimisers when no single risk measure captures what you
    care about and you still want the hierarchy to do the diversification — for example
    combining a tail measure with variance, or treating intra-cluster and inter-cluster risk
    differently. The clustering structure keeps the allocation robust while the scalariser
    controls how the mixed risk terms combine.

````@example 12_Clustering_Mixed_Risks_And_Constraints
using PortfolioOptimisers, PrettyTables

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;
nothing #hide
````

## 1. Data and clustering

We compute the prior and cluster hierarchy once, then reuse them across all the examples.

````@example 12_Clustering_Mixed_Risks_And_Constraints
using CSV, TimeSeries, DataFrames, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
clr = clusterise(ClustersEstimator(; alg = DBHT()), pr.X)

# Shared solver and hierarchical optimiser.
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv)
````

## 2. HRP with mixed risk measures

[`HierarchicalRiskParity`](@ref) accepts either a single risk measure or a vector of them.
When we pass a vector, the scalariser chooses how to combine the inner risk terms.

Here we mix tail risk and variance, then sweep a few scalarisers to show that the hierarchy
really is responding to the combination rule rather than to a single hidden default.

````@example 12_Clustering_Mixed_Risks_And_Constraints
r_mix = [ConditionalValueatRisk(),
         Variance(; settings = RiskMeasureSettings(; scale = 2e2))]
hrp_sum = optimise(HierarchicalRiskParity(; r = r_mix, opt = opt, sca = SumScalariser()))
hrp_max = optimise(HierarchicalRiskParity(; r = r_mix, opt = opt, sca = MaxScalariser()))
hrp_min = optimise(HierarchicalRiskParity(; r = r_mix, opt = opt, sca = MinScalariser()))
hrp_lse = optimise(HierarchicalRiskParity(; r = r_mix, opt = opt,
                                          sca = LogSumExpScalariser(; gamma = 1e2)))

pretty_table(DataFrame(; :assets => rd.nx, :Sum => hrp_sum.w, :Max => hrp_max.w,
                       :Min => hrp_min.w, :LogSumExp => hrp_lse.w); formatters = [resfmt])
````

## 3. HERC with mixed inner and outer risks

[`HierarchicalEqualRiskContribution`](@ref) lets the inner and outer levels use different
risk measures and scalarisers. That makes it the cleanest place to show the "mixed risk"
idea: the hierarchy can treat intra-cluster and inter-cluster risk differently.

We pair the same mixed risk vector at both levels, then vary the scalariser to show the
contrast between additive, max, min, and log-sum-exp aggregation.

````@example 12_Clustering_Mixed_Risks_And_Constraints
herc_sum = optimise(HierarchicalEqualRiskContribution(; opt = opt, ri = r_mix, ro = r_mix,
                                                      scai = SumScalariser(),
                                                      scao = SumScalariser()))
herc_max = optimise(HierarchicalEqualRiskContribution(; opt = opt, ri = r_mix, ro = r_mix,
                                                      scai = MaxScalariser(),
                                                      scao = MaxScalariser()))
herc_min = optimise(HierarchicalEqualRiskContribution(; opt = opt, ri = r_mix, ro = r_mix,
                                                      scai = MinScalariser(),
                                                      scao = MinScalariser()))
herc_lse = optimise(HierarchicalEqualRiskContribution(; opt = opt, ri = r_mix, ro = r_mix,
                                                      scai = LogSumExpScalariser(;
                                                                                 gamma = 1e2),
                                                      scao = LogSumExpScalariser(;
                                                                                 gamma = 1e2)))

pretty_table(DataFrame(; :assets => rd.nx, :Sum => herc_sum.w, :Max => herc_max.w,
                       :Min => herc_min.w, :LogSumExp => herc_lse.w); formatters = [resfmt])
````

The scalariser choice can dominate the solution when one risk measure is consistently larger
than the other. That is why the max and min solutions can collapse toward the portfolio that
is effectively minimising the dominating term.

````@example 12_Clustering_Mixed_Risks_And_Constraints
using StatsPlots, GraphRecipes
plot_stacked_bar_composition([hrp_sum, hrp_max, hrp_min, hrp_lse, herc_sum, herc_max,
                              herc_min, herc_lse], rd)
````

## 4. Constrained HERC

The hierarchical optimiser itself can still carry weight bounds and fees. This version keeps
the same mixed risk structure, but asks the optimiser to stay inside a bounded, fee-aware
universe. We set a 10% cap (`ub = 0.1`) deliberately tight enough to bind: the unconstrained
HERC already concentrates around 13% in the largest names, so the cap pulls them down — watch
the largest holdings in the comparison below.

````@example 12_Clustering_Mixed_Risks_And_Constraints
opt_constrained = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv,
                                        wb = WeightBounds(; lb = 0.0, ub = 0.1),
                                        fees = Fees(; l = 0.001))
herc_constrained = optimise(HierarchicalEqualRiskContribution(; opt = opt_constrained,
                                                              ri = r_mix, ro = r_mix,
                                                              scai = SumScalariser(),
                                                              scao = SumScalariser()))

pretty_table(DataFrame(; :assets => rd.nx, :Unconstrained => herc_sum.w,
                       :Constrained => herc_constrained.w); formatters = [resfmt])
````

The risk-contribution view shows how the hierarchy spreads risk rather than capital.
Populate the covariance via [`factory`](@ref) before calling [`plot_risk_contribution`](@ref).

````@example 12_Clustering_Mixed_Risks_And_Constraints
rv = factory(Variance(), pr)
plot_risk_contribution(rv, herc_constrained, rd)
````

## Summary

Clustering optimisers give you a hierarchical lever on diversification.

- [`HierarchicalRiskParity`](@ref) responds to the scalariser when you mix risk measures.
- [`HierarchicalEqualRiskContribution`](@ref) lets inner and outer levels use different
    risk terms and different scalarisers.
- [`HierarchicalOptimiser`](@ref) can still carry practical constraints like weight bounds
    and fees while the hierarchy does the allocation.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
