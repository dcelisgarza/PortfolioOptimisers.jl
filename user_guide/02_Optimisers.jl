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

#=
Every JuMP optimiser below shares this one solver. Its `check_sol` field is splatted into JuMP's
`assert_is_solved_and_feasible` after each solve, and decides which solver statuses count as a
solved model. The default `(;)` is strict on purpose — it accepts only `OPTIMAL` or
`LOCALLY_SOLVED` at a `FEASIBLE_POINT`, so a solution the solver itself flags as approximate is
rejected rather than silently used. Passing `allow_almost = true` widens it to the `ALMOST_*`
statuses, which is what we want here: a first-order conic solver that reaches its tolerance on a
well-posed portfolio problem gives a usable answer, and rejecting it would only make the optimiser
fall through to the next solver. Tighten it the other way with `allow_local = false` when nothing
short of a certified global optimum will do.
=#

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

The **return** side is the `ret` field of [`JuMPOptimiser`](@ref), an
[`ArithmeticReturn`](@ref) by default. Like `r`, it takes one term or a vector of them, summed
with weights into the model's single return expression. Each term carries its own
[`JuMPReturnsSettings`](@ref) — its weight in that sum, its own lower bound, and whether it
enters the sum at all — so a term can bound the portfolio without being rewarded, which is how
you price what a floor on one quantity costs in another
([ℓ1 uncertainty sets](../examples/2_moments_priors/11_L1_Uncertainty_Quintile_Portfolios.md)).

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
name. That matrix is derived from an [`AssetPanel`](@ref): [`panel_input`](@ref) turns a
[`UniverseSets`](@ref) taxonomy key into one Panel Field, [`asset_panel`](@ref) builds the panel,
and it travels beside the returns as data rather than on the estimator.
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
[`FeatureDistance`](@ref)'s `ape` slot says where the panel comes from. `nothing` reads the one
you put on the [`ReturnsResult`](@ref), and a **producer** — [`RegressionPanel`](@ref) or
[`PhylogenyPanel`](@ref) — builds one at the point of use from the prior result and the returns of
the subproblem that runs it.

The difference only shows once cross-validation is switched on, and it cannot be inferred from the
API: **a carried panel slices, a producer refits.** Inside a fold or a meta-optimiser's subproblem
a carried panel is *subselected*, while a producer is *recomputed on the subproblem's own returns*.
For a fixed classification the two coincide; for a returns-derived producer they are two different
questions, so the slot chooses between two semantics rather than two copies. A producer is also the
only route for features that must vary with time *and* survive folds, because it refits on
whatever rows the fold hands it.

See [Feature Matrices as a Distance Source](../examples/3_optimisers/16_Feature_Distance_Clustering.md)
for the producers, the time-varying shapes, and a walk-forward comparison.

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
