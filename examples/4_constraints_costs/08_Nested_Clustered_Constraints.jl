#=
```@meta
Description = "Nested clustered optimisation in PortfolioOptimisers.jl: weight bounds on the inner, outer and final layers of NestedClustered, and a fee on the clusters."
```

# Nested clustered optimisation with layered constraints and fees

This example puts weight bounds on three layers of [`NestedClustered`](@ref).

  - The inner optimisation inside each cluster.
  - The outer optimisation over the cluster portfolios, the portfolio that each inner
    optimisation makes of its cluster.
  - The final weights of the assets over the whole universe.

It also puts a fee on `NestedClustered`. The fee reduces the returns of each cluster portfolio
before the outer optimisation sees them.

!!! tip "When to reach for this"
    Reach for this when a mandate caps an asset inside its cluster, a whole cluster, or an asset
    in the final portfolio.
=#

using PortfolioOptimisers, PrettyTables, StableRNGs

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v * 100, digits = 3)) %" : v
    end
end;

#=
## 1. Data and shared setup

We use one year of S&P 500 prices and compute the clusters once. Then we compare three
portfolios.

  - Weight bounds on the inner optimisations only.
  - Weight bounds on the inner and the outer optimisations, and a fee.
  - The same, and bounds on the final asset weights, given to [`NestedClustered`](@ref).
=#

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

#=
## 2. Constraints at each layer

`NestedClustered` solves two kinds of problem.

  - An inner [`MeanRisk`](@ref) in each cluster.
  - An outer [`MeanRisk`](@ref) over the returns of the cluster portfolios, one series per
    cluster.

The outer `JuMPOptimiser` takes no `pe`, because it works on the returns of the cluster
portfolios and not on the assets. The inner bound caps each weight inside a cluster at 35%. The
outer bound caps the weight of each cluster at 62%. The final bound caps each asset at 20% of
the whole portfolio.
=#

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
# The `wb` of `NestedClustered` bounds the final asset weights. We give it one bound per asset,
# which is how you give each asset its own cap. Here every cap is `0.20`, so a scalar bound of
# `0.20` gives the same constraint.

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

#=
For each portfolio, we print three numbers next to the bound of their layer.

  - The largest weight inside any cluster, against the inner bound, `0.35`.
  - The largest weight of a cluster, against the outer bound, `0.62`.
  - The largest final asset weight, against the final bound, `0.20`.
=#

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

#=
Compare each column with the `Limit` column. The first portfolio puts no bound on the outer
optimisation, and its largest cluster weight is above `0.62`. In the first two portfolios, the
final weight of an asset is its weight inside its cluster times the weight of the cluster. The
inner and outer bounds of the second portfolio therefore let an asset reach
`0.35 * 0.62 = 0.217`, which is above the final bound. We plot the three portfolios.
=#

using StatsPlots, GraphRecipes
plot_stacked_bar_composition([res_inner, res_inner_outer, res_nested_overall], rd;
                             xticks = ([1, 2, 3],
                                       ["Inner WB", "Inner+Outer WB+Fees", "Overall WB"]))

#=
## What to take away

  - With no final bound, an asset can reach the product of the inner and the outer bounds. On
    this page that product, `0.217`, is above the final bound of `0.20`.
  - To cap the final weights below that product, give the cap to the `wb` of `NestedClustered`,
    as the third portfolio does. It takes a [`WeightBounds`](@ref) of numbers or of vectors, or
    a [`WeightBoundsEstimator`](@ref).
  - The page does not isolate the fee. The second portfolio adds the fee and the outer bound in
    one step.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end under Kaimon (docs env): all three NCO configurations (inner-only
#src   WB, inner+outer WB+fees, direct overall WB) solve with Clarabel. The audit table confirms
#src   each bound binds where it is applied: the inner 0.35 cap holds on every inner solve; the
#src   outer 0.62 cap binds only on the runs that set it on the outer optimiser (the inner-only
#src   run leaves the outer unbounded, so its cluster max is 63.9% — correctly above 0.62); and
#src   the direct overall 0.20 bound pins JNJ and MRK at exactly 20% in the final weights.
#src - FIXED (this session): the opening admonition body and several bullet lists were indented
#src   4/8 spaces inside the `#=` blocks, which Markdown renders as code blocks rather than
#src   admonition text and lists. Dedented to the 4-space admonition / 2-space list convention
#src   the other examples use.
#src - No solver warnings or plotting deprecations observed.
