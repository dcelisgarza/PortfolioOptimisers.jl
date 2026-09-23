#=
```@meta
Description = "Clustering optimisers with mixed risk measures, scalarisers, constraints and fees across the hierarchy in PortfolioOptimisers.jl."
```

# Clustering optimisers with mixed risks and constraints

The clustering optimiser page covers one risk measure at a time and no constraints. This
page covers two things it leaves out:

  - a clustering optimiser takes several risk measures at once, and a different one at each
    level of the hierarchy;
  - the hierarchical optimiser still takes weight bounds and fees while the clustering does
    the spreading.

The data is one year of daily prices for twenty S&P 500 stocks. We solve four HRP runs and
four HERC runs on one mixed pair of risk measures, one run per scalariser. One more HERC run
takes the same pair under a 10% weight cap and a fee.

!!! tip "When to reach for this"
    Reach for a mixed-risk clustering optimiser when one risk measure does not cover what you
    want to control and you still want the hierarchy to spread the money. You might combine a
    tail measure with variance, or measure risk inside a cluster one way and between clusters
    another. The scalariser sets how the risk terms combine.
=#

using PortfolioOptimisers, PrettyTables

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Data and clustering

We compute the prior and the tree once, and every run below reads the same two.
=#

using CSV, TimeSeries, DataFrames, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)
clr = clusterise(ClustersEstimator(; alg = DBHT()), pr.X)

## Shared solver and hierarchical optimiser.
slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
opt = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv)

#=
## 2. HRP with mixed risk measures

[`HierarchicalRiskParity`](@ref) takes one risk measure or a vector of them. When you pass a
vector, the scalariser sets how the risks of the measures combine at each split.
[`SumScalariser`](@ref) adds them. [`MaxScalariser`](@ref) and [`MinScalariser`](@ref) use
the measure whose risk is the largest or the smallest. [`LogSumExpScalariser`](@ref) takes a
smooth maximum, and its `gamma` sets how close it comes to the largest.

The cell below pairs a tail measure, [`ConditionalValueatRisk`](@ref), with a variance, and runs
the four scalarisers over that pair. The `scale` of the variance multiplies its risk by 200
before the scalariser combines the two. That brings a daily variance to the same order of
magnitude as a daily conditional value at risk. Read the four weight columns against each
other.
=#

r_mix = [ConditionalValueatRisk(),
         Variance(; settings = RiskMeasureSettings(; scale = 2e2))]
hrp_sum = optimise(HierarchicalRiskParity(; r = r_mix, opt = opt, sca = SumScalariser()))
hrp_max = optimise(HierarchicalRiskParity(; r = r_mix, opt = opt, sca = MaxScalariser()))
hrp_min = optimise(HierarchicalRiskParity(; r = r_mix, opt = opt, sca = MinScalariser()))
hrp_lse = optimise(HierarchicalRiskParity(; r = r_mix, opt = opt,
                                          sca = LogSumExpScalariser(; gamma = 1e2)))

pretty_table(DataFrame(; :assets => rd.nx, :Sum => hrp_sum.w, :Max => hrp_max.w,
                       :Min => hrp_min.w, :LogSumExp => hrp_lse.w); formatters = [resfmt])

#=
## 3. HERC with mixed inner and outer risks

[`HierarchicalEqualRiskContribution`](@ref) takes a risk measure and a scalariser at each of
its two levels, so it can measure risk inside a cluster one way and between clusters another.

We pass the same pair of risk measures at both levels and run the same four scalarisers at
both levels.
=#

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

#=
At each split the max scalariser uses the measure whose scaled risk is the larger, and the
min scalariser uses the smaller one. When one measure is the larger at most splits, the `Max`
column follows that measure and the `Min` column follows the other. The next plot stacks the
eight allocations of sections 2 and 3.
=#

using StatsPlots, GraphRecipes
plot_stacked_bar_composition([hrp_sum, hrp_max, hrp_min, hrp_lse, herc_sum, herc_max,
                              herc_min, herc_lse], rd)

#=
## 4. Constrained HERC

The hierarchical optimiser takes weight bounds and fees. This run keeps the same pair of risk
measures and adds both. We set the cap at 10%, `ub = 0.1`, which is tighter than the largest
weight the unconstrained HERC gives, near 13%. Read the largest holdings of the two columns
below against each other.
=#

opt_constrained = HierarchicalOptimiser(; pe = pr, cle = clr, slv = slv,
                                        wb = WeightBounds(; lb = 0.0, ub = 0.1),
                                        fees = Fees(; l = 0.001))
herc_constrained = optimise(HierarchicalEqualRiskContribution(; opt = opt_constrained,
                                                              ri = r_mix, ro = r_mix,
                                                              scai = SumScalariser(),
                                                              scao = SumScalariser()))

pretty_table(DataFrame(; :assets => rd.nx, :Unconstrained => herc_sum.w,
                       :Constrained => herc_constrained.w); formatters = [resfmt])

#=
We plot the share of the variance of the constrained portfolio that each asset carries. It measures the variance alone, not the pair of measures the portfolio was built on.
As on the clustering optimiser page, [`factory`](@ref) gives the variance the covariance of the
prior before [`plot_risk_contribution`](@ref) uses it.
=#

rv = factory(Variance(), pr)
plot_risk_contribution(rv, herc_constrained, rd)

#=
## Summary

A clustering optimiser spreads the money through the tree, and you set how it measures risk.

  - [`HierarchicalRiskParity`](@ref) gives different weights under each scalariser once you
    mix risk measures.
  - [`HierarchicalEqualRiskContribution`](@ref) takes a different risk measure and a different
    scalariser inside a cluster and between clusters.
  - [`HierarchicalOptimiser`](@ref) takes weight bounds and fees for either optimiser.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env): HRP and HERC mixed-risk
#src   (CVaR + scaled Variance) scalariser sweeps (Sum/Max/Min/LogSumExp) and the constrained
#src   HERC all solve with Clarabel. The Max/Min collapse behaviour matches the narrative.
#src - FIXED (this session): the "constrained HERC" section originally used `ub = 0.2`, but
#src   unconstrained HERC already maxes around 13%, so the bound never bound and the
#src   Constrained vs Unconstrained columns were identical to three decimals (only the fee
#src   moved them). Tightened to `ub = 0.1`, which now binds (max weight pinned at 0.1, three
#src   names at the cap) so the section actually demonstrates the constraint biting.
#src - No solver warnings or plotting deprecations observed (this page uses
#src   `plot_stacked_bar_composition` / `plot_risk_contribution`, not `plot_clusters`, so it
#src   avoids the `orientation`-deprecation seen in the clustering-overview page).
