#=
```@meta
Description = "Nested clustered optimisation in PortfolioOptimisers.jl: weight bounds on the inner, outer and final layers of NestedClustered, and a fee on one cluster."
```

# Nested clustered optimisation with layered constraints and fees

This example puts weight bounds on three layers of [`NestedClustered`](@ref).

  - The inner optimisation inside each cluster.
  - The outer optimisation over the cluster portfolios, the portfolio that each inner
    optimisation makes of its cluster.
  - The final weights of the assets over the whole universe.

It also charges a fee on the assets of one cluster, and shows which outer objective the fee
changes.

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
  - Weight bounds on the inner and the outer optimisations.
  - The same, and bounds on the final asset weights, given to [`NestedClustered`](@ref).

Section 3 adds a fee to the second portfolio.
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

res_inner_outer = optimise(NestedClustered(; pe = pr, cle = clr,
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
                                              opti = MeanRisk(; obj = MinimumRisk(),
                                                              opt = jopti_inner),
                                              opto = MeanRisk(; obj = MinimumRisk(),
                                                              opt = jopto_outer)), rd)

pretty_table(DataFrame(; :assets => rd.nx, :InnerOnlyWB => res_inner.w,
                       :InnerOuterWB => res_inner_outer.w,
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
                  :NestedInnerOuter =>
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
                                       ["Inner WB", "Inner+Outer WB", "Overall WB"]))

#=
## 3. A fee on one cluster

The `fees` of `NestedClustered` reach the outer optimisation through the returns of the cluster
portfolios. The library deducts the fee of each cluster portfolio from its returns, and the outer
optimisation uses the net returns. The inner optimisations do not see this fee.

We charge a long fee of `0.001` on the three energy stocks, CVX, RRC and XOM. The clustering puts
the three of them in one cluster. A fee by asset name needs a [`UniverseSets`](@ref) that names
the group, and we give it to `NestedClustered` as `sets`. We solve the second portfolio with and
without the fee, under two outer objectives, `MinimumRisk` and [`MaximumRatio`](@ref).
=#

sets = UniverseSets(; dict = Dict("nx" => rd.nx, "energy" => ["CVX", "RRC", "XOM"]))
fees_energy = FeesEstimator(; l = ["energy" => 0.001])

res_minrisk_fee = optimise(NestedClustered(; pe = pr, cle = clr, sets = sets,
                                           fees = fees_energy,
                                           opti = MeanRisk(; obj = MinimumRisk(),
                                                           opt = jopti_inner),
                                           opto = MeanRisk(; obj = MinimumRisk(),
                                                           opt = jopto_outer)), rd)

res_ratio = optimise(NestedClustered(; pe = pr, cle = clr,
                                     opti = MeanRisk(; obj = MinimumRisk(),
                                                     opt = jopti_inner),
                                     opto = MeanRisk(; obj = MaximumRatio(),
                                                     opt = jopto_outer)), rd)

res_ratio_fee = optimise(NestedClustered(; pe = pr, cle = clr, sets = sets,
                                         fees = fees_energy,
                                         opti = MeanRisk(; obj = MinimumRisk(),
                                                         opt = jopti_inner),
                                         opto = MeanRisk(; obj = MaximumRatio(),
                                                         opt = jopto_outer)), rd)

cluster_assets = [join(rd.nx[assignments(clr) .== i], ", ") for i in 1:(clr.k)]
pretty_table(DataFrame("Cluster" => cluster_assets, "MinimumRisk" => res_inner_outer.reso.w,
                       "MinimumRisk, fee" => res_minrisk_fee.reso.w,
                       "MaximumRatio" => res_ratio.reso.w,
                       "MaximumRatio, fee" => res_ratio_fee.reso.w); formatters = [resfmt],
             title = "Weight of each cluster in the outer optimisation")

#=
Compare the columns in pairs. Under `MinimumRisk`, the fee changes no weight. Every asset of the
energy cluster pays the fee, and the inner weights of the cluster sum to one. The fee then
deducts `0.001` from the return of that cluster in every period. A constant deduction lowers the
mean of the returns and leaves their covariance unchanged. `MinimumRisk` with the default
variance uses the covariance alone.

`MaximumRatio` uses the mean, and it moves weight away from the energy cluster. The cluster of
JNJ, LLY, MRK, PFE, PG and UNH takes weight until it reaches the outer bound of `0.62`, and the
cluster of KO, PEP and WMT takes the rest.

## What to take away

  - With no final bound, an asset can reach the product of the inner and the outer bounds. On
    this page that product, `0.217`, is above the final bound of `0.20`.
  - To cap the final weights below that product, give the cap to the `wb` of `NestedClustered`,
    as the third portfolio does. It takes a [`WeightBounds`](@ref) of numbers or of vectors, or
    a [`WeightBoundsEstimator`](@ref).
  - The `fees` of `NestedClustered` lower the return of each cluster portfolio by a fixed
    amount in every period. Under `MinimumRisk` with the default variance, the fee changes no
    weight. Under `MaximumRatio`, which uses the mean return, it moves weight away from the
    cluster that pays it.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - #1292 (2026-09-23): portfolios 2 and 3 used to carry Fees(l = 0.001) on a MinimumRisk
#src   outer. The fee changed no weight (max|dw| 5.1e-15): a uniform long fee on a long-only,
#src   budget-1 inner book is a constant shift of every cluster series, and the variance ignores a
#src   shift. The fee now has its own section: a long fee of 0.001 on CVX/RRC/XOM (DBHT cluster 2).
#src   MinimumRisk outer: fee max|dw| 0.0. MaximumRatio outer, outer weights (clusters 1-4): no
#src   fee [0, 0.407, 0.593, 0], fee [0, 0.304, 0.62, 0.076]. The outer 0.62 bound binds only
#src   with the fee. A uniform Fees(l = 0.001) also moves MaximumRatio (0.075), because it acts as a
#src   higher risk-free rate; the page does not use it. Removing the fee from portfolios 2 and 3
#src   leaves every number of the weights and audit tables unchanged.
#src - Page runs end-to-end under Kaimon (docs env): all three NCO configurations (inner-only
#src   WB, inner+outer WB, direct overall WB) solve with Clarabel. The audit table confirms
#src   each bound binds where it is applied: the inner 0.35 cap holds on every inner solve; the
#src   outer 0.62 cap binds only on the runs that set it on the outer optimiser (the inner-only
#src   run leaves the outer unbounded, so its cluster max is 63.9% — correctly above 0.62); and
#src   the direct overall 0.20 bound pins JNJ and MRK at exactly 20% in the final weights.
#src - FIXED (this session): the opening admonition body and several bullet lists were indented
#src   4/8 spaces inside the `#=` blocks, which Markdown renders as code blocks rather than
#src   admonition text and lists. Dedented to the 4-space admonition / 2-space list convention
#src   the other examples use.
#src - No solver warnings or plotting deprecations observed.
