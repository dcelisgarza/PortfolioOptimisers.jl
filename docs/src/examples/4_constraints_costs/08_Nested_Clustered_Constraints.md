The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/4_constraints_costs/08_Nested_Clustered_Constraints.jl"
```

# Nested clustered optimisation with layered constraints and fees

This example shows how to apply constraints at three distinct layers of
[`NestedClustered`](@ref):

- the inner optimisation within each cluster,
- the outer optimisation across synthetic cluster portfolios,
- and a final overall optimisation pass on the full universe.

It also shows how fees can be attached to the nested stage and to the final overall stage.

!!! tip "When to reach for this"
    Reach for this when you want cluster structure for robustness, but still need practical
    controls (fees and weight limits) at more than one stage of the nested workflow.

````@example 08_Nested_Clustered_Constraints
using PortfolioOptimisers, PrettyTables, StableRNGs

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;
nothing #hide
````

## 1. Data and shared setup

We use one year of S&P 500 data and compute clusters once. Then we compare three nested setups:

- inner-only weight bounds,
- inner + outer bounds plus fees in the nested stage,
- direct overall asset bounds on [`NestedClustered`](@ref).

````@example 08_Nested_Clustered_Constraints
using CSV, TimeSeries, DataFrames, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel2, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.95),
              check_sol = (; allow_local = true, allow_almost = true)),
       Solver(; name = :clarabel3, solver = Clarabel.Optimizer,
              settings = Dict("verbose" => false, "max_step_fraction" => 0.9),
              check_sol = (; allow_local = true, allow_almost = true))]

pr = prior(EmpiricalPrior(), rd)
clr = clusterise(ClustersEstimator(; alg = DBHT()), pr.X)
````

## 2. Layering constraints in NCO

The nested workflow solves:

- an inner [`MeanRisk`](@ref) within each cluster, then
- an outer [`MeanRisk`](@ref) on synthetic cluster returns.

The key wiring rule remains the same: the **outer optimiser must not** consume the
asset-level prior directly; it works on synthetic cluster returns.

````@example 08_Nested_Clustered_Constraints
jopti_inner = JuMPOptimiser(; pe = pr, slv = slv, wb = WeightBounds(; lb = 0.0, ub = 0.35))
jopto_base = JuMPOptimiser(; slv = slv)
jopto_outer = JuMPOptimiser(; slv = slv, wb = WeightBounds(; lb = 0.0, ub = 0.62))

res_inner = optimise(NestedClustered(; pe = pr, cle = clr,
                                     opti = MeanRisk(; obj = MinimumRisk(),
                                                     opt = jopti_inner),
                                     opto = MeanRisk(; obj = MinimumRisk(),
                                                     opt = jopto_base)), rd)

res_inner_outer = optimise(NestedClustered(; pe = pr, cle = clr, fees = Fees(; l = 0.001),
                                           opti = MeanRisk(; obj = MinimumRisk(),
                                                           opt = jopti_inner),
                                           opto = MeanRisk(; obj = MinimumRisk(),
                                                           opt = jopto_outer)), rd)

wb_overall_assets = WeightBounds(; lb = fill(0.0, length(rd.nx)),
                                 ub = fill(0.20, length(rd.nx)))
````

The direct `NestedClustered(wb = ...)` bound always applies to the final aggregated asset
weights. Per-asset vectors are used here to vary the cap by asset; a scalar bound is
equivalent to the vector spelling of the same number repeated across every asset.

````@example 08_Nested_Clustered_Constraints
res_nested_overall = optimise(NestedClustered(; pe = pr, cle = clr, wb = wb_overall_assets,
                                              fees = Fees(; l = 0.001),
                                              opti = MeanRisk(; obj = MinimumRisk(),
                                                              opt = jopti_inner),
                                              opto = MeanRisk(; obj = MinimumRisk(),
                                                              opt = jopto_outer)), rd)

pretty_table(DataFrame(; :assets => rd.nx, :InnerOnlyWB => res_inner.w,
                       :InnerOuterWBFees => res_inner_outer.w,
                       :NestedDirectOverallWB => res_nested_overall.w);
             formatters = [resfmt])
````

The following audit confirms where the constraints are active.

- Inner bound (`0.35`) applies to each cluster-level solve.
- Outer bound (`0.62`) applies to cluster allocation weights.
- Overall bound (`0.20`) can be applied directly in [`NestedClustered`](@ref), applied here using per-asset vector bounds.

````@example 08_Nested_Clustered_Constraints
inner_local_max(res) = maximum(maximum(ri.w) for ri in res.resi)
outer_cluster_max(res) = maximum(res.reso.w)

audit = DataFrame(:Metric => ["Max inner local weight", "Max outer cluster weight",
                              "Max final asset weight"], :Limit => [0.35, 0.62, 0.20],
                  :NestedInnerOnly =>
                      [inner_local_max(res_inner), maximum(res_inner.reso.w),
                       maximum(res_inner.w)],
                  :NestedInnerOuterFees =>
                      [inner_local_max(res_inner_outer), outer_cluster_max(res_inner_outer),
                       maximum(res_inner_outer.w)],
                  :NestedDirectOverallWB => [inner_local_max(res_nested_overall),
                                             outer_cluster_max(res_nested_overall),
                                             maximum(res_nested_overall.w)])

pretty_table(audit; formatters = [resfmt])
````

These runs isolate layer placement:

- inner bounds shape per-cluster compositions,
- outer bounds shape allocation across clusters,
- direct `NestedClustered(wb = ...)` bounds constrain the final aggregated asset weights.

````@example 08_Nested_Clustered_Constraints
using StatsPlots, GraphRecipes
plot_stacked_bar_composition([res_inner, res_inner_outer, res_nested_overall], rd;
                             xticks = ([1, 2, 3],
                                       ["Inner WB", "Inner+Outer WB+Fees", "Overall WB"]))
````

## Summary

Layered controls can be applied around [`NestedClustered`](@ref) without giving up the
cluster-based decomposition.

- Inner `wb` controls weights inside each cluster.
- Outer `wb` controls allocation across synthetic cluster portfolios.
- Direct `NestedClustered(wb = ...)` constrains the final aggregated asset weights, whether
    given as a scalar, a per-asset vector, or an estimator.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
