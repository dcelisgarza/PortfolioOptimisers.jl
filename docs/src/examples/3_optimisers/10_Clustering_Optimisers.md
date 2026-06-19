The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/3_optimisers/06_Clustering_Optimisers.jl"
```

# Clustering optimisers

The optimisers we have met so far ([`MeanRisk`](@ref), [`RiskBudgeting`](@ref),
[`NearOptimalCentering`](@ref)) all solve a single global problem over every asset at once.
*Clustering optimisers* take a different route: they first group the assets into a hierarchy
from their dependency structure (a dendrogram), then allocate **within and across** those
groups. Because they never invert the full covariance matrix and need no expected returns,
they are robust to estimation error and require no numerical solver for variance-based risk.

`PortfolioOptimisers` ships three members of this family:

- [`HierarchicalRiskParity`](@ref) (HRP) — recursive bisection of the dendrogram, splitting
    risk between each pair of sub-clusters.
- [`HierarchicalEqualRiskContribution`](@ref) (HERC) — equalises risk contributions both
    *within* each cluster (inner) and *across* clusters (outer), with independent risk
    measures and scalarisers for each level.
- [`SchurComplementHierarchicalRiskParity`](@ref) (SCHRP) — augments each sub-cluster's
    covariance with a Schur-complement correction, interpolating between HRP (`gamma = 0`)
    and a Markowitz-like allocation as `gamma → 1`.

!!! tip "When to reach for this"
    Reach for a clustering optimiser when you want the allocation driven by the *correlation
    structure* of the assets rather than by a return forecast — to diversify across genuine
    groupings, to stay robust when the covariance matrix is noisy or near-singular, or simply
    to avoid running a solver. Use HRP for the classic robust baseline, HERC when you want
    explicit control of the risk split within vs across clusters, and SCHRP when you want to
    dial in some of mean-variance's efficiency without giving up the hierarchy's stability.
    If you want an explicit return/risk trade-off instead, use [`MeanRisk`](@ref).

````@example 06_Clustering_Optimisers
using PortfolioOptimisers, PrettyTables
# Format for pretty tables.
tsfmt = (v, i, j) -> begin
    if j == 1
        return Date(v)
    else
        return v
    end
end;
resfmt = (v, i, j) -> begin
    if j == 1
        return v
    else
        return isa(v, Number) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. ReturnsResult data

We use the same S&P 500 slice as the other optimiser examples.

````@example 06_Clustering_Optimisers
using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

# Compute the returns
rd = prices_to_returns(X)
````

## 2. Prior and clustering

Clustering optimisers need two precomputable ingredients: a prior (for the covariance) and a
clustering of the assets. We compute both once and reuse them across every optimiser so the
comparison is apples-to-apples — only the allocation algorithm changes.

We cluster with the Direct Bubble Hierarchy Tree ([`DBHT`](@ref)) algorithm, which builds the
hierarchy from the correlation-derived distance matrix.

````@example 06_Clustering_Optimisers
pr = prior(EmpiricalPrior(), rd)
clr = clusterise(ClustersEstimator(; alg = DBHT()), pr.X)
````

We can inspect the structure the optimisers will act on: the dendrogram and the reordered
correlation heatmap with the detected cluster boundaries.

````@example 06_Clustering_Optimisers
using StatsPlots, GraphRecipes #= Hierarchical clustering dendrogram. =#

plot_dendrogram(clr, rd.nx) #= Reordered correlation heatmap with cluster boundary boxes. =#

plot_clusters(clr, rd.nx)
````

## 3. Hierarchical risk parity (HRP)

HRP recursively bisects the dendrogram and splits the budget between each pair of
sub-clusters in inverse proportion to their risk. We pass the shared prior and clustering
through a [`HierarchicalOptimiser`](@ref). Variance needs no solver, so none is supplied.

````@example 06_Clustering_Optimisers
opt = HierarchicalOptimiser(; pe = pr, cle = clr)
res_hrp = optimise(HierarchicalRiskParity(; opt = opt, r = Variance()))
````

## 4. Hierarchical equal risk contribution (HERC)

HERC equalises risk contributions within each cluster (the inner problem) and across clusters
(the outer problem). It accepts separate inner/outer risk measures (`ri`, `ro`) and
scalarisers (`scai`, `scao`); here we use [`Variance`](@ref) for both levels.

````@example 06_Clustering_Optimisers
res_herc = optimise(HierarchicalEqualRiskContribution(; opt = opt, ri = Variance(),
                                                      ro = Variance()))
````

## 5. Schur-complement HRP (SCHRP)

SCHRP corrects each sub-cluster's covariance with a Schur complement of the off-diagonal
(inter-cluster) block, controlled by `gamma`. At `gamma = 0` it reduces to HRP; as `gamma`
grows it absorbs more of the cross-cluster information, moving toward a Markowitz-like
allocation while keeping the hierarchical structure. We sweep three values to make the
interpolation visible.

````@example 06_Clustering_Optimisers
res_schur0 = optimise(SchurComplementHierarchicalRiskParity(; opt = opt,
                                                            params = SchurComplementParams(;
                                                                                           r = Variance(),
                                                                                           gamma = 0.0)))
res_schur5 = optimise(SchurComplementHierarchicalRiskParity(; opt = opt,
                                                            params = SchurComplementParams(;
                                                                                           r = Variance(),
                                                                                           gamma = 0.5)))
res_schur9 = optimise(SchurComplementHierarchicalRiskParity(; opt = opt,
                                                            params = SchurComplementParams(;
                                                                                           r = Variance(),
                                                                                           gamma = 0.9)))
````

## 6. Comparing the allocations

With everything sharing one prior and one clustering, the weight differences come purely from
the allocation rule. Note how SCHRP at `gamma = 0` matches HRP, and drifts away from it as
`gamma` increases.

````@example 06_Clustering_Optimisers
pretty_table(DataFrame(; :assets => rd.nx, :HRP => res_hrp.w, :HERC => res_herc.w,
                       Symbol("SCHRP γ=0") => res_schur0.w,
                       Symbol("SCHRP γ=0.5") => res_schur5.w,
                       Symbol("SCHRP γ=0.9") => res_schur9.w); formatters = [resfmt])
````

The composition plot shows the same story visually across the five allocations.

````@example 06_Clustering_Optimisers
plot_stacked_bar_composition([res_hrp, res_herc, res_schur0, res_schur5, res_schur9], rd)
````

Finally, the per-asset variance risk contributions for the HRP portfolio confirm that risk —
not capital — is what the hierarchy spreads out. The risk measure needs its covariance
populated from the prior first, which is what [`factory`](@ref) does.

````@example 06_Clustering_Optimisers
rv = factory(Variance(), pr)
plot_risk_contribution(rv, res_hrp, rd)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
