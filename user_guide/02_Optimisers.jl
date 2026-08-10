#=
# Optimisers

This is the breadth tour of the optimiser families. Every optimiser shares the same call —
`optimise(estimator)` (or `optimise(estimator, rd)` for the naive and meta ones) — and returns
a result whose `w` field holds the asset weights. The point of this page is to show the *shape*
of each family with one minimal call; for objectives, risk measures, variants, and trade-offs,
follow the cross-links into the [optimiser examples](../examples/3_optimisers/01_MeanRisk_Objectives.md).

We fix one empirical prior and reuse it everywhere so the families are comparable.
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

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

#=
## 1. Naive optimisers

Naive optimisers use simple, solver-free rules that buy robustness through unsophistication.
[`InverseVolatility`](@ref) weights by the reciprocal of each asset's volatility;
[`EqualWeighted`](@ref) splits capital evenly; [`RandomWeighted`](@ref) samples a Dirichlet
allocation. They take the [`ReturnsResult`](@ref) directly.
=#

res_iv = optimise(InverseVolatility(), rd)
res_ew = optimise(EqualWeighted(), rd)

#=
## 2. JuMP optimisers — `MeanRisk`

JuMP optimisers solve a mathematical program and are the most flexible on constraints,
objectives, and risk measures. They need a [`JuMPOptimiser`](@ref) carrying the prior and a
[`Solver`](@ref) (we recommend [Clarabel](https://github.com/oxfordcontrol/Clarabel.jl) for
non-MIP problems). The workhorse is [`MeanRisk`](@ref); its default objective is
[`MinimumRisk`](@ref).
=#

res_mr = optimise(MeanRisk(; obj = MinimumRisk(),
                           opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
`MeanRisk` also offers [`MaximumUtility`](@ref), [`MaximumRatio`](@ref) and
[`MaximumReturn`](@ref) objectives and efficient frontiers — see
[MeanRisk Objectives](../examples/3_optimisers/01_MeanRisk_Objectives.md) and
[Efficient Frontier](../examples/3_optimisers/02_Efficient_Frontier.md).

The **risk measure** is the `r` field (of `MeanRisk` and of the clustering optimisers below); the
default is [`Variance`](@ref). Which one you pick encodes *what kind* of risk you penalise —
overall dispersion ([`Variance`](@ref)), the left tail ([`ConditionalValueatRisk`](@ref)),
peak-to-trough paths ([`MaximumDrawdown`](@ref)), or the whole ordered loss curve
([`OrderedWeightsArray`](@ref)). The full menu — every measure with its alias, its meaning, and
which optimisers accept it — is the [risk measures](03_Risk_Measures.md) page; you can also mix
several in one objective ([Multiple Risk Measures](../examples/3_optimisers/04_Multiple_Risk_Measures.md)).

The drawdown notion is also useful purely as a *post-optimisation diagnostic* — via
[`drawdowns`](@ref) on a realised book — when you want to measure rather than optimise it
([Performance Attribution](../examples/6_post_processing/03_Performance_Attribution.md)).

The other JuMP families follow the same `opt = JuMPOptimiser(...)` pattern:

  - [`RiskBudgeting`](@ref) / [`RelaxedRiskBudgeting`](@ref) — target a risk contribution per
    asset or factor ([Risk Budgeting](../examples/3_optimisers/09_Risk_Budgeting.md)).
  - [`NearOptimalCentering`](@ref) — a robust point near the efficient frontier
    ([Near Optimal Centering](../examples/3_optimisers/15_Near_Optimal_Centering.md)).

Here is the minimal risk-budgeting call (equal risk contribution by default):
=#

res_rb = optimise(RiskBudgeting(; opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
### Which risk measures each optimiser family accepts

Compatibility is a property of the optimiser *family*, not the individual optimiser: every
JuMP optimiser accepts the same [`RiskMeasure`](@ref)s, and clustering optimisers additionally
accept the hierarchical-only measures. You can ask programmatically with
[`supports_risk_measure`](@ref) / [`supported_risk_measures`](@ref):

```julia
supports_risk_measure(MeanRisk, ConditionalValueatRisk)   # true
supported_risk_measures(HierarchicalRiskParity)           # OptimisationRiskMeasure
```

Meta-optimisers (`NestedClustered`, `Stacking`, `SubsetResampling`) are the exception: their
acceptance is instance-specific because they *delegate*, accepting a measure only when every
constituent optimiser does (the intersection of their children's categories).

The [risk measures](03_Risk_Measures.md) page tabulates every measure against these classes —
the tables there are generated from the same predicate, so they cannot drift from what the
optimisers actually dispatch on.

## 3. Clustering optimisers

Clustering optimisers build the allocation from the asset correlation hierarchy instead of a
single program. They take a [`HierarchicalOptimiser`](@ref) carrying the prior and a clustering
estimate. [`HierarchicalRiskParity`](@ref) (HRP) is the canonical one;
[`HierarchicalEqualRiskContribution`](@ref) and
[`SchurComplementHierarchicalRiskParity`](@ref) are its siblings — see
[Clustering Optimisers](../examples/3_optimisers/11_Clustering_Optimisers.md).
=#

clr = clusterise(ClustersEstimator(), pr.X)
hopt = HierarchicalOptimiser(; pe = pr, cle = clr)
res_hrp = optimise(HierarchicalRiskParity(; opt = hopt, r = Variance()))

#=
### 3.1 Clustering on something other than the returns

`clusterise(ClustersEstimator(), pr.X)` derives its distance from the correlation, so the
hierarchy can only ever see structure the price history contains. Swapping the estimator's
distance slot for a [`FeatureDistance`](@ref) clusters an **assets × features** matrix instead —
a sector or country classification, a factor loading profile, any per-asset quantity you can
name. [`asset_sets_features`](@ref) builds one from a [`UniverseSets`](@ref) taxonomy, and it
travels beside the returns as data rather than on the estimator.
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
                     nz = asset_sets_feature_names(vals_z, sets_z),
                     Z = asset_sets_features(vals_z, sets_z))

res_hrp_z = optimise(HierarchicalRiskParity(;
                                            opt = HierarchicalOptimiser(; pe = pr,
                                                                        cle = ClustersEstimator(;
                                                                                                de = FeatureDistance()),
                                                                        z_src = :data),
                                            r = Variance()), rd_z)

#=
`z_src` picks which of the two carriers supplies the matrix: `:data` reads the one you supplied
on the [`ReturnsResult`](@ref), `:prior` reads one a producer derived onto the prior result. It
defaults to `:data` — the opposite of `x_src` — because an explicitly supplied matrix outranks a
derived one.

Two consequences only appear once cross-validation is switched on, and neither can be inferred
from the API:

  - **`:data` slices, `:prior` refits.** Inside a fold or a meta-optimiser's subproblem, a
    carried matrix is *subselected* while a derived one is *recomputed on the subproblem's own
    returns*. For a fixed classification the two coincide; for a returns-derived producer they
    are two different questions, so the selector chooses between two semantics rather than two
    copies.
  - **A time-varying literal matrix cannot survive an observation fold.** A three-dimensional
    `observations × assets × features` matrix handed straight to [`FeaturePrior`](@ref) has no
    way to be resliced down its observation axis, so the fit throws a `DimensionMismatch` as
    soon as the observation count changes. Features that must vary with time *and* survive folds
    have to come from a producer, which refits on whatever rows the fold hands it.

See [Feature Matrices as a Distance Source](../examples/3_optimisers/16_Feature_Distance_Clustering.md)
for the four producers, the time-varying shapes, and a walk-forward comparison.

## 4. Meta-optimisers

Meta-optimisers compose other optimisers. [`NestedClustered`](@ref) (NCO) runs an **inner**
optimiser within each cluster and an **outer** optimiser across the cluster representatives;
[`Stacking`](@ref) and [`SubsetResampling`](@ref) blend several fits — see
[Meta Optimisers](../examples/3_optimisers/13_Meta_Optimisers.md). The inner optimiser carries
the prior; the outer one does not.
=#

res_nco = optimise(NestedClustered(; pe = pr, cle = clr,
                                   opti = MeanRisk(; obj = MinimumRisk(),
                                                   opt = JuMPOptimiser(; pe = pr,
                                                                       slv = slv)),
                                   opto = MeanRisk(; obj = MinimumRisk(),
                                                   opt = JuMPOptimiser(; slv = slv))), rd)

#=
## 5. Comparing the families

One prior, six optimisers, six allocations. The naive rules, risk budgeting, and the clustering
hierarchy spread weight broadly (max weight in single digits to low teens); `MeanRisk(MinimumRisk)`
and NCO concentrate into a few low-variance names (max weight ≈ a third). Same data, very
different portfolios — which is the point of having a menu.
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
