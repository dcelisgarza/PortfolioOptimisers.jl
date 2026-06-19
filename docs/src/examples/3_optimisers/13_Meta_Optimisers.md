The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/3_optimisers/13_Meta_Optimisers.jl"
```

# Meta-optimisers

Every optimiser so far produces weights by solving *one* problem. **Meta-optimisers** instead
orchestrate *other* optimisers: they split the problem up, solve the pieces with whatever
estimator you like, and recombine the results. They are the package's answer to two practical
worries — estimation error (a single fit on all assets is fragile) and modularity (you may
want different rules for different parts of the universe).

`PortfolioOptimisers` provides three, all sharing the same inner/outer composition idea:

- [`NestedClustered`](@ref) (NCO) — cluster the assets, run an **inner** optimiser inside
    each cluster, then an **outer** optimiser across the cluster aggregates.
- [`Stacking`](@ref) — run several inner optimisers on the **full** universe, then stack
    their portfolios together with an outer optimiser (an ensemble).
- [`SubsetResampling`](@ref) — repeatedly optimise on random **subsets** of the assets and
    average the resampled weights, à la bagging.

Because the inner and outer slots accept *any* optimisation estimator (including other
meta-optimisers), these compose arbitrarily.

!!! tip "When to reach for this"
    Reach for a meta-optimiser when a single global fit feels too fragile or too monolithic:
    NCO when you trust the cluster structure and want a different rule within vs across groups,
    Stacking when you want to hedge model risk by ensembling several optimisers, and
    SubsetResampling when you want bagging-style robustness against the specific asset set and
    estimation noise. If a single optimiser already does what you need, prefer it — these add
    compute and configuration surface in exchange for robustness.

````@example 13_Meta_Optimisers
using PortfolioOptimisers, PrettyTables, StableRNGs

resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. ReturnsResult data and shared ingredients

We use the same S&P 500 slice as the other optimiser examples, and precompute a prior, a
clustering, and a solver to share across the meta-optimisers.

````@example 13_Meta_Optimisers
using CSV, TimeSeries, DataFrames, Clarabel

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)

slv = Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))
pr = prior(EmpiricalPrior(), rd)
clr = clusterise(ClustersEstimator(; alg = DBHT()), pr.X)
````

A recurring pattern below illustrates the **precomputed-result vs estimator** distinction (see
the [`MeanRisk` objectives](01_MeanRisk_Objectives.md) note). The **inner** optimiser is given
the precomputed prior through its [`JuMPOptimiser`](@ref) (`pe = pr`) — fine, because the inner
solves run on the real asset returns. The **outer** optimiser is deliberately *not* given a
prior: it operates on the synthetic per-cluster (or stacked) returns the meta-optimiser builds
internally, where a precomputed asset-level prior would be meaningless. The outer slot is an
estimator-driven slot — it recomputes whatever statistics it needs from those synthetic returns
at solve time, so here it only needs a solver. (Passing `pe = pr` to the *outer* optimiser would
silently feed it the wrong, asset-level prior.)

````@example 13_Meta_Optimisers
jopti = JuMPOptimiser(; pe = pr, slv = slv)
jopto = JuMPOptimiser(; slv = slv)
````

For a reference point we also compute a plain minimum-variance [`MeanRisk`](@ref) over the
whole universe.

````@example 13_Meta_Optimisers
res_bench = optimise(MeanRisk(; obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv)))
````

## 2. Nested clustered optimisation (NCO)

NCO solves a minimum-variance problem *inside* each cluster, collapses each cluster to a
single synthetic asset, then solves a second minimum-variance problem *across* the clusters.
The inner and outer optimisers are independent — here both are [`MeanRisk`](@ref), but either
could be a risk-budgeting, hierarchical, or naive estimator.

````@example 13_Meta_Optimisers
res_nco = optimise(NestedClustered(; pe = pr, cle = clr,
                                   opti = MeanRisk(; obj = MinimumRisk(), opt = jopti),
                                   opto = MeanRisk(; obj = MinimumRisk(), opt = jopto)), rd)
````

## 3. Stacking

Stacking runs a *list* of inner optimisers on the full universe — here a min-variance
[`MeanRisk`](@ref), a [`HierarchicalRiskParity`](@ref), and a naive [`InverseVolatility`](@ref)
— then combines their portfolios with an outer optimiser. The result is an ensemble that
hedges the model risk of any single rule.

````@example 13_Meta_Optimisers
res_stk = optimise(Stacking(; pe = pr,
                            opti = [MeanRisk(; opt = jopti),
                                    HierarchicalRiskParity(;
                                                           opt = HierarchicalOptimiser(;
                                                                                       pe = pr)),
                                    InverseVolatility(; pe = pr)],
                            opto = MeanRisk(; obj = MinimumRisk(), opt = jopto)), rd)
````

## 4. Subset resampling

SubsetResampling draws repeated random subsets of the assets, optimises each one, and averages
the resampled weights — bagging for portfolios. We draw 10 subsets of 70% of the assets with a
fixed RNG/seed so the result is reproducible.

````@example 13_Meta_Optimisers
res_ssr = optimise(SubsetResampling(; pe = pr,
                                    opt = MeanRisk(; obj = MinimumRisk(),
                                                   opt = JuMPOptimiser(; slv = slv)),
                                    subset_size = 0.7, n_subsets = 10, rng = StableRNG(123),
                                    seed = 42), rd)
````

## 5. Comparing the allocations

All four portfolios target minimum variance, but reach it through very different machinery.
NCO and Stacking tend to spread weight more than the plain fit, and SubsetResampling smooths
it further by averaging over universes.

````@example 13_Meta_Optimisers
pretty_table(DataFrame(; :assets => rd.nx, :MinVar => res_bench.w, :NCO => res_nco.w,
                       :Stacking => res_stk.w, :SubsetResampling => res_ssr.w);
             formatters = [resfmt])
````

## 6. Visualising the compositions

The stacked-bar composition makes the diversifying effect of the meta-optimisers visible
against the plain minimum-variance benchmark.

Composition of the benchmark and the three meta-optimisers.

````@example 13_Meta_Optimisers
using StatsPlots, GraphRecipes
plot_stacked_bar_composition([res_bench, res_nco, res_stk, res_ssr], rd)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
