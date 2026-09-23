#=
```@meta
Description = "Clustering optimisers in PortfolioOptimisers.jl: hierarchical risk parity, HERC and Schur complement allocation from a dendrogram of the assets."
```

# Clustering optimisers

The optimisers we have met so far, [`MeanRisk`](@ref), [`RiskBudgeting`](@ref) and
[`NearOptimalCentering`](@ref), solve one problem over every asset at once. A clustering
optimiser works in two steps. It first groups the assets into a tree, the dendrogram, built
from how they move together. It then splits the budget inside each group and between the
groups. It never inverts the full covariance matrix and it reads no expected returns, so an
error in those estimates moves the weights less, and variance-based risk needs no solver.

`PortfolioOptimisers` has three members of this family.

  - [`HierarchicalRiskParity`](@ref), or HRP, orders the assets by the dendrogram, splits that
    ordered list in half again and again, and splits the budget between the two halves in
    inverse proportion to their risk.
  - [`HierarchicalEqualRiskContribution`](@ref), or HERC, cuts the dendrogram into clusters,
    splits the budget between the clusters down the tree's branches in inverse proportion to
    their risk, and splits each cluster's share between its assets in inverse proportion to
    their own risk. Despite the name, the clusters do not contribute equal risk. Each of the two
    levels takes its own risk measure and its own scalariser.
  - [`SchurComplementHierarchicalRiskParity`](@ref), or SCHRP, corrects the covariance of each
    sub-cluster with a Schur complement. At `gamma = 0` it gives HRP, and as `gamma` rises
    toward 1 it moves toward the minimum-variance portfolio.

!!! tip "When to reach for this"
    Reach for a clustering optimiser when you want the correlation structure of the assets to
    drive the allocation rather than a return forecast. It spreads the money over groups that
    move differently, it holds up when the covariance matrix is noisy or near-singular, and it
    needs no solver. Use HRP for the plain robust case, HERC when you want to set the split
    inside a cluster apart from the split between clusters, and SCHRP when you want some of
    the efficiency of mean-variance without losing the stability of the hierarchy. If you want
    a stated trade-off between return and risk instead, use [`MeanRisk`](@ref).
=#

using PortfolioOptimisers, PrettyTables
## Format for pretty tables.
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

#=
## 1. Data

We use one year of daily prices for twenty S&P 500 stocks.
=#

using CSV, TimeSeries, DataFrames

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
pretty_table(X[(end - 5):end]; formatters = [tsfmt])

## Compute the returns
rd = prices_to_returns(X)

#=
## 2. Prior and clustering

A clustering optimiser needs two things you can compute once, a prior for the covariance and
a clustering of the assets. We compute both here and hand them to every optimiser below, so
the only difference between the results is the allocation rule.

We cluster with the Direct Bubble Hierarchical Tree algorithm, [`DBHT`](@ref), which builds the
tree from the distance matrix the correlations give.
=#

pr = prior(EmpiricalPrior(), rd)
clr = clusterise(ClustersEstimator(; alg = DBHT()), pr.X)

#=
The two plots below show the structure every optimiser on this page acts on. The first is
the dendrogram.
=#

using StatsPlots, GraphRecipes
plot_dendrogram(clr, rd.nx)
# The second is the correlation heatmap, reordered by the tree, with a box around each cluster.
plot_clusters(clr, rd.nx)

#=
## 3. Hierarchical risk parity (HRP)

HRP orders the assets by the dendrogram, splits that ordered list in half again and again, and
splits the budget between the two halves of each split in inverse proportion to their risk. We
pass the shared prior and clustering through a [`HierarchicalOptimiser`](@ref). Variance needs
no solver, so we pass none.
=#

opt = HierarchicalOptimiser(; pe = pr, cle = clr)
res_hrp = optimise(HierarchicalRiskParity(; opt = opt, r = Variance()))

#=
## 4. Hierarchical equal risk contribution (HERC)

HERC splits the budget between the clusters down the tree's branches in inverse proportion to
their risk, which is the outer level, and splits each cluster's share between its assets in
inverse proportion to their own risk, which is the inner level. It takes one risk measure per
level, `ri` and `ro`, and one scalariser per level, `scai` and `scao`. We use
[`Variance`](@ref) at both levels.
=#

res_herc = optimise(HierarchicalEqualRiskContribution(; opt = opt, ri = Variance(),
                                                      ro = Variance()))

#=
## 5. Schur-complement HRP (SCHRP)

SCHRP corrects the covariance of each half of a split with a Schur complement. The correction
subtracts a term that the covariances between the two halves build, and `gamma` scales it. At
`gamma = 0` the result is HRP. As `gamma` rises, more of the covariance between the halves
enters, and the allocation moves toward the minimum-variance portfolio. Under the default
[`MonotonicSchurComplement`](@ref), `gamma` is the upper end of a search. The optimiser uses
the largest value from 0 to `gamma` up to which the variance of the portfolio keeps falling. We
run three values so you can compare them.
=#

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

#=
## 6. Comparing the allocations

Read the SCHRP column at `gamma = 0` against the HRP column, then read the two columns at the
higher values of `gamma`. Under the search, two values of `gamma` can give almost the same
weights, as 0.5 and 0.9 do here.
=#

pretty_table(DataFrame(; :assets => rd.nx, :HRP => res_hrp.w, :HERC => res_herc.w,
                       Symbol("SCHRP γ=0") => res_schur0.w,
                       Symbol("SCHRP γ=0.5") => res_schur5.w,
                       Symbol("SCHRP γ=0.9") => res_schur9.w); formatters = [resfmt])

#=
The next plot stacks the same five allocations.
=#

plot_stacked_bar_composition([res_hrp, res_herc, res_schur0, res_schur5, res_schur9], rd)

#=
The last plot draws the share of the variance of the HRP portfolio that each asset carries.
Read it against the HRP weights in the table above. The plot needs a variance that holds a
covariance, and [`factory`](@ref) gives it the covariance of the prior.
=#

rv = factory(Variance(), pr)
plot_risk_contribution(rv, res_hrp, rd)

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - Page runs end-to-end once the risk-contribution measure is factory-populated. HRP,
#src   HERC and the three-point SCHRP gamma sweep all solve; SCHRP γ=0 reproduces HRP and
#src   drifts as γ grows, as documented.
#src - ERGO (record-only per hybrid policy → issue #125): calling
#src   `plot_risk_contribution(Variance(), res, rd)` with a *bare* Variance (sigma = nothing)
#src   throws a cryptic `MethodError: no method matching *(::Nothing, ::SubArray...)` from
#src   `dot(w, sigma, w)` deep in `expected_risk`, rather than a contextual message telling
#src   the user to populate sigma via `factory(Variance(), prior)`. A `@argcheck`/`isnothing`
#src   guard in the Variance functor (src/16_RiskMeasures/02_Variance.jl:297) would mirror the
#src   friendly error already added for factor risk contribution. Worked around here with
#src   `factory`.
