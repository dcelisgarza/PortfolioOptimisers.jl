#=
```@meta
Description = "The optimiser families of PortfolioOptimisers.jl with one minimal call each: mean-risk, risk budgeting, hierarchical, near-optimal centering, naive and meta."
```

# Optimisers

This page makes one call from each family of optimisers. Every optimiser takes the same call,
`optimise(estimator)`, or `optimise(estimator, rd)` for the naive optimisers and the
meta-optimisers. The result holds the asset weights in its field `w`. For the objectives, the risk
measures and the variants of each family, follow the links into the
[optimiser examples](../examples/3_optimisers/01_MeanRisk_Objectives.md).

We compute one empirical prior and give it to every JuMP, clustering and meta-optimiser. The naive
optimisers take the returns directly.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, StatsPlots,
      GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

X = TimeArray(CSV.File(joinpath(@__DIR__, "../examples/SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

#=
Every JuMP optimiser below uses this solver. After each solve, the optimiser passes the solver's
`check_sol` field to JuMP's `assert_is_solved_and_feasible`, which decides which solver statuses
count as a solution. The default, `(;)`, accepts `OPTIMAL` or `LOCALLY_SOLVED` at a
`FEASIBLE_POINT`. If the solver marks its solution as approximate, the check fails, and the
optimiser tries the next solver in its list.

`allow_almost = true` also accepts the `ALMOST_*` statuses. We set it, because a conic solver that
stops at its tolerance on a well-posed portfolio problem still returns usable weights.
`allow_local = true` is JuMP's default, so it changes nothing here. To accept only `OPTIMAL`, set
`allow_local = false`.
=#

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

#=
## 1. Naive optimisers

A naive optimiser uses a fixed rule and no solver. [`InverseVolatility`](@ref) weights each asset
by the reciprocal of its volatility. [`EqualWeighted`](@ref) gives each asset the same weight.
[`RandomWeighted`](@ref) draws the weights from a Dirichlet distribution. They take the
[`ReturnsResult`](@ref) directly. [`OnlinePortfolioSelection`](@ref) is also a naive optimiser. It
moves its allocation after each period by a rule of its own, and it has its own page,
[Online portfolio selection](10_Online_Portfolio_Selection.md).
=#

res_iv = optimise(InverseVolatility(), rd)
res_ew = optimise(EqualWeighted(), rd)

#=
## 2. JuMP optimisers and `MeanRisk`

A JuMP optimiser solves a mathematical program, and it accepts more constraints and objectives
than the other families. It needs a [`JuMPOptimiser`](@ref) that holds the prior and a
[`Solver`](@ref). [Clarabel](https://github.com/oxfordcontrol/Clarabel.jl) suits a problem with no
integer variables. The most used JuMP optimiser is [`MeanRisk`](@ref), and its
default objective is [`MinimumRisk`](@ref).
=#

res_mr = optimise(MeanRisk(; obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
`MeanRisk` also takes the objectives [`MaximumUtility`](@ref), [`MaximumRatio`](@ref) and
[`MaximumReturn`](@ref), and it computes efficient frontiers. See
[MeanRisk Objectives](../examples/3_optimisers/01_MeanRisk_Objectives.md) and
[Efficient Frontier](../examples/3_optimisers/02_Efficient_Frontier.md).

The risk measure is the `r` field of `MeanRisk` and of the clustering optimisers below, and its
default is [`Variance`](@ref). The measure sets the kind of risk that the optimiser penalises.
[`Variance`](@ref) penalises the spread of the returns, [`ConditionalValueatRisk`](@ref) the left
tail, [`MaximumDrawdown`](@ref) the largest fall from a peak, and [`OrderedWeightsArray`](@ref) a
weighted sum of the sorted returns. [Risk measures](03_Risk_Measures.md) lists every measure with
its alias, what it penalises and the optimisers that accept it. You can also put several measures
in one objective. See
[Multiple Risk Measures](../examples/3_optimisers/04_Multiple_Risk_Measures.md).

The return is the `ret` field of [`JuMPOptimiser`](@ref), an [`ArithmeticReturn`](@ref) by
default. Like `r`, it takes one term or a vector of terms. The optimiser multiplies each term by
its scale and adds the terms into one return expression. Each term has its own
[`JuMPReturnsSettings`](@ref), which set its scale, its lower bound, and whether it enters the
sum. So a term can bound the portfolio's return and add nothing to the objective. With this you
can measure how much of one return you give up to keep another above a floor
([ℓ1 uncertainty sets](../examples/2_moments_priors/11_L1_Uncertainty_Quintile_Portfolios.md)).

To measure the drawdowns of a portfolio without optimising them, call [`drawdowns`](@ref) on its
returns after the optimisation. See
[Performance Attribution](../examples/6_post_processing/03_Performance_Attribution.md).

The other JuMP optimisers take the same `opt = JuMPOptimiser(...)` keyword:

  - [`RiskBudgeting`](@ref) and [`RelaxedRiskBudgeting`](@ref) give each asset, or each factor, a
    target share of the risk. See [Risk Budgeting](../examples/3_optimisers/09_Risk_Budgeting.md).
  - [`NearOptimalCentering`](@ref) returns the centre of the set of portfolios whose return and
    risk are close to those of the optimal portfolio. See
    [Near Optimal Centering](../examples/3_optimisers/15_Near_Optimal_Centering.md).

We run risk budgeting with its default budget, which gives every asset the same share of the risk.
=#

res_rb = optimise(RiskBudgeting(; opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
### Which risk measures each optimiser family accepts

The family of an optimiser sets the risk measures that it accepts. Every JuMP optimiser with an
`r` field accepts the same [`RiskMeasure`](@ref)s. [`RelaxedRiskBudgeting`](@ref) has no `r`
field. The clustering optimisers accept those measures and also the measures that only a
clustering optimiser can use. You can ask with [`supports_risk_measure`](@ref) and
[`supported_risk_measures`](@ref):

```julia
supports_risk_measure(MeanRisk, ConditionalValueatRisk)   # true
supported_risk_measures(HierarchicalRiskParity)           # OptimisationRiskMeasure
```

For a meta-optimiser, `NestedClustered`, `Stacking` or `SubsetResampling`, the answer depends on
the optimisers that it holds. The [risk measures](03_Risk_Measures.md) page shows every measure
against these classes.

## 3. Clustering optimisers

A clustering optimiser builds the weights from a hierarchy of the assets and not from one
program. It takes a [`HierarchicalOptimiser`](@ref) that holds the prior and a clustering of the
assets. [`HierarchicalRiskParity`](@ref) (HRP) is the best known.
[`HierarchicalEqualRiskContribution`](@ref) and
[`SchurComplementHierarchicalRiskParity`](@ref) are in the same family. See
[Clustering Optimisers](../examples/3_optimisers/11_Clustering_Optimisers.md).
=#

clr = clusterise(ClustersEstimator(), pr.X)
hopt = HierarchicalOptimiser(; pe = pr, cle = clr)
res_hrp = optimise(HierarchicalRiskParity(; opt = hopt, r = Variance()))

#=
### 3.1 Clustering on something other than the returns

`clusterise(ClustersEstimator(), pr.X)` computes its distance from the correlation of the returns,
so the hierarchy shows only the structure that the returns contain. A [`FeatureDistance`](@ref) in
the estimator's `de` field clusters the assets on a matrix of features instead. A feature can be a
sector or a country, a profile of factor loadings, or any other quantity you have for each asset.
That matrix comes from an [`AssetPanel`](@ref). [`panel_input`](@ref) turns keys of a
[`UniverseSets`](@ref) into the fields of the panel, and [`asset_panel`](@ref) builds the panel.
The panel goes on the returns, next to `X`, and not on the estimator.

We label each asset with its sector and with where most of its revenue comes from, and we cluster
the assets on those two labels.
=#

sector = Dict("AAPL" => "Tech", "AMD" => "Tech", "MSFT" => "Tech", "BAC" => "Financials",
              "JPM" => "Financials", "CVX" => "Energy", "XOM" => "Energy",
              "RRC" => "Energy", "GE" => "Industrials", "BBY" => "Discretionary",
              "HD" => "Discretionary", "KO" => "Staples", "PEP" => "Staples",
              "PG" => "Staples", "WMT" => "Staples", "JNJ" => "Health", "LLY" => "Health",
              "MRK" => "Health", "PFE" => "Health", "UNH" => "Health")
revenue = Dict("AAPL" => "Global", "AMD" => "Global", "MSFT" => "Global",
               "BAC" => "Domestic", "JPM" => "Global", "CVX" => "Global", "XOM" => "Global",
               "RRC" => "Domestic", "GE" => "Global", "BBY" => "Domestic",
               "HD" => "Domestic", "KO" => "Global", "PEP" => "Global", "PG" => "Global",
               "WMT" => "Domestic", "JNJ" => "Global", "LLY" => "Global", "MRK" => "Global",
               "PFE" => "Global", "UNH" => "Domestic")

sets_z = UniverseSets(; xkey = "nx",
                      dict = Dict("nx" => rd.nx, "nx_sector" => [sector[a] for a in rd.nx],
                                  "nx_revenue" => [revenue[a] for a in rd.nx]))
vals_z = ["nx_sector", "nx_revenue"]
rd_z = ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts,
                     pnl = asset_panel(panel_input(sets_z, vals_z)))

res_hrp_z = optimise(HierarchicalRiskParity(;
                                            opt = HierarchicalOptimiser(; pe = pr,
                                                                        cle = ClustersEstimator(;
                                                                                                de = FeatureDistance())),
                                            r = Variance()), rd_z)

#=
The `ape` field of [`FeatureDistance`](@ref) sets where the panel comes from. With `nothing`, it
reads the panel on the [`ReturnsResult`](@ref). With [`RegressionPanel`](@ref) or
[`PhylogenyPanel`](@ref), it builds a panel when it runs, from the prior and the returns that it
receives.

The two choices give different panels only when the optimiser runs on part of the data, inside a
cross-validation fold or inside a subproblem of a meta-optimiser. There the optimiser takes the part
of a stored panel that the fold uses. A panel estimator instead fits a new panel on the returns of
the fold. For a fixed label, such as a sector, the two give the same panel. For a panel estimated
from the returns, they give different panels. A panel estimator is also the only way to use a
feature that changes with time on every fold, because it fits the feature on the rows of the fold.

See
[Feature Matrices as a Distance Source](../examples/3_optimisers/16_Feature_Distance_Clustering.md)
for the panel estimators, the features that change with time, and a comparison over a walk-forward.

## 4. Meta-optimisers

A meta-optimiser combines other optimisers. [`NestedClustered`](@ref) (NCO) runs an inner
optimiser inside each cluster, and an outer optimiser across the portfolios of the clusters.
[`Stacking`](@ref) runs several inner optimisers and combines their weights with an outer
optimiser. [`SubsetResampling`](@ref) runs one optimiser on random subsets of the assets and
combines the weights. See [Meta Optimisers](../examples/3_optimisers/13_Meta_Optimisers.md).

In the call below, the inner optimiser holds the computed prior `pr`, and the outer one must not.
The outer problem has one asset for each cluster, so the outer optimiser computes its own prior.
=#

res_nco = optimise(NestedClustered(; pe = pr, cle = clr,
                                   opti = MeanRisk(; obj = MinimumRisk(),
                                                   opt = JuMPOptimiser(; pe = pr,
                                                                       slv = slv)),
                                   opto = MeanRisk(; obj = MinimumRisk(),
                                                   opt = JuMPOptimiser(; slv = slv))), rd)

#=
## 5. Comparing the families

We print the weights of the six optimisers in one table and plot them. The naive rules, risk
budgeting and HRP give some weight to every asset. `MeanRisk` with `MinimumRisk` and NCO put most
of the weight in a few assets with a low variance. The six optimisers use the same returns, so
the differences come from the optimisers alone.
=#

results = [res_iv, res_ew, res_mr, res_rb, res_hrp, res_nco]
labels = ["InvVol", "EqualW", "MinRisk", "RiskBudget", "HRP", "NCO"]

pretty_table(DataFrame(["Asset" => rd.nx,
                        [labels[i] => results[i].w for i in eachindex(results)]...]);
             formatters = [resfmt], title = "Weights by optimiser family")

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Shallow breadth-tour guide page split from monolith §2. One minimal blessed call per
#src   family: naive (InverseVolatility/EqualWeighted), JuMP (MeanRisk MinimumRisk + RiskBudgeting),
#src   clustering (HierarchicalRiskParity), meta (NestedClustered/NCO). Variants/objectives
#src   deferred to the 3_optimisers examples via cross-links.
#src - Clustering needs clusterise(ClustersEstimator(), pr.X) → HierarchicalOptimiser(; pe, cle).
#src   NCO needs inner opti (pe=pr) + outer opto (no pe), matching examples/3_optimisers/07.
#src - VERIFIED end-to-end on kaimon (session f102cae9): all 6 optimisers OptimisationSuccess,
#src   weights sum≈1, len 20. Default ClustersEstimator() clusters cleanly (no explicit DBHT
#src   needed, unlike ex06). Max weights IV 8.2 / EW 5.0 / MinRisk 37 / RB 8.2 / HRP 13.1 /
#src   NCO 31.8 % — naive+RB+HRP spread, MinRisk+NCO concentrate (§5 prose matched to this).
