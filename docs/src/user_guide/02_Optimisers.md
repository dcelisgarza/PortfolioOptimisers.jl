The source files can be found in [user_guide/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/user_guide/).

```@meta
EditURL = "../../../user_guide/02_Optimisers.jl"
```

# Optimisers

This is the breadth tour of the optimiser families. Every optimiser shares the same call —
`optimise(estimator)` (or `optimise(estimator, rd)` for the naive and meta ones) — and returns
a result whose `w` field holds the asset weights. The point of this page is to show the *shape*
of each family with one minimal call; for objectives, risk measures, variants, and trade-offs,
follow the cross-links into the [optimiser examples](../examples/3_optimisers/01_MeanRisk_Objectives.md).

We fix one empirical prior and reuse it everywhere so the families are comparable.

````@example 02_Optimisers
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

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
````

## 1. Naive optimisers

Naive optimisers use simple, solver-free rules that buy robustness through unsophistication.
[`InverseVolatility`](@ref) weights by the reciprocal of each asset's volatility;
[`EqualWeighted`](@ref) splits capital evenly; [`RandomWeighted`](@ref) samples a Dirichlet
allocation. They take the [`ReturnsResult`](@ref) directly.

````@example 02_Optimisers
res_iv = optimise(InverseVolatility(), rd)
res_ew = optimise(EqualWeighted(), rd)
````

## 2. JuMP optimisers — `MeanRisk`

JuMP optimisers solve a mathematical program and are the most flexible on constraints,
objectives, and risk measures. They need a [`JuMPOptimiser`](@ref) carrying the prior and a
[`Solver`](@ref) (we recommend [Clarabel](https://github.com/oxfordcontrol/Clarabel.jl) for
non-MIP problems). The workhorse is [`MeanRisk`](@ref); its default objective is
[`MinimumRisk`](@ref).

````@example 02_Optimisers
res_mr = optimise(MeanRisk(; obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv)))
````

`MeanRisk` also offers [`MaximumUtility`](@ref), [`MaximumRatio`](@ref) and
[`MaximumReturn`](@ref) objectives, swappable risk measures, and efficient frontiers — see
[MeanRisk Objectives](../examples/3_optimisers/01_MeanRisk_Objectives.md) and
[Efficient Frontier](../examples/3_optimisers/02_Efficient_Frontier.md). The other JuMP families
follow the same `opt = JuMPOptimiser(...)` pattern:

- [`RiskBudgeting`](@ref) / [`RelaxedRiskBudgeting`](@ref) — target a risk contribution per
    asset or factor ([Risk Budgeting](../examples/3_optimisers/05_Risk_Budgeting.md)).
- [`NearOptimalCentering`](@ref) — a robust point near the efficient frontier
    ([Near Optimal Centering](../examples/3_optimisers/08_Near_Optimal_Centering.md)).

Here is the minimal risk-budgeting call (equal risk contribution by default):

````@example 02_Optimisers
res_rb = optimise(RiskBudgeting(; opt = JuMPOptimiser(; pe = pr, slv = slv)))
````

## 3. Clustering optimisers

Clustering optimisers build the allocation from the asset correlation hierarchy instead of a
single program. They take a [`HierarchicalOptimiser`](@ref) carrying the prior and a clustering
estimate. [`HierarchicalRiskParity`](@ref) (HRP) is the canonical one;
[`HierarchicalEqualRiskContribution`](@ref) and
[`SchurComplementHierarchicalRiskParity`](@ref) are its siblings — see
[Clustering Optimisers](../examples/3_optimisers/06_Clustering_Optimisers.md).

````@example 02_Optimisers
clr = clusterise(ClustersEstimator(), pr.X)
hopt = HierarchicalOptimiser(; pe = pr, cle = clr)
res_hrp = optimise(HierarchicalRiskParity(; opt = hopt, r = Variance()))
````

## 4. Meta-optimisers

Meta-optimisers compose other optimisers. [`NestedClustered`](@ref) (NCO) runs an **inner**
optimiser within each cluster and an **outer** optimiser across the cluster representatives;
[`Stacking`](@ref) and [`SubsetResampling`](@ref) blend several fits — see
[Meta Optimisers](../examples/3_optimisers/07_Meta_Optimisers.md). The inner optimiser carries
the prior; the outer one does not.

````@example 02_Optimisers
res_nco = optimise(NestedClustered(; pe = pr, cle = clr,
                                   opti = MeanRisk(; obj = MinimumRisk(),
                                                   opt = JuMPOptimiser(; pe = pr,
                                                                       slv = slv)),
                                   opto = MeanRisk(; obj = MinimumRisk(),
                                                   opt = JuMPOptimiser(; slv = slv))), rd)
````

## 5. Comparing the families

One prior, six optimisers, six allocations. The naive rules, risk budgeting, and the clustering
hierarchy spread weight broadly (max weight in single digits to low teens); `MeanRisk(MinimumRisk)`
and NCO concentrate into a few low-variance names (max weight ≈ a third). Same data, very
different portfolios — which is the point of having a menu.

````@example 02_Optimisers
results = [res_iv, res_ew, res_mr, res_rb, res_hrp, res_nco]
labels = ["InvVol", "EqualW", "MinRisk", "RiskBudget", "HRP", "NCO"]

pretty_table(DataFrame(["Asset" => rd.nx,
                        [labels[i] => results[i].w for i in eachindex(results)]...]);
             formatters = [resfmt], title = "Weights by optimiser family")

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
