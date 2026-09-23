#=
```@meta
Description = "Cluster on a feature matrix instead of returns with FeatureDistance in PortfolioOptimisers.jl: sectors, loadings or any per-asset quantity."
```

# Feature matrices as a distance source

Every clustering optimiser so far builds its hierarchy from the returns. A covariance estimate
becomes a correlation, a correlation becomes a distance, and the distance becomes a dendrogram.
That route sees only the structure the price history contains.

A [`FeatureDistance`](@ref) replaces the returns with a matrix of assets by features. Feature `k`
is any per-asset quantity you can name, such as a sector membership, a factor loading, a position
in the asset network or a trailing characteristic. Two assets are close when their feature rows
point the same way. The output is an ordinary distance matrix, so you can pass a
`FeatureDistance` wherever the library takes a distance estimator: [`ClustersEstimator`](@ref),
[`NetworkEstimator`](@ref), the clustering optimisers, and the phylogeny and centrality
constraints.

Nothing stores the matrix. An [`AssetPanel`](@ref) holds the values, one field per named quantity,
and [`feature_matrix`](@ref) stacks the fields a selector names into the matrix a distance
measures. The stacking happens where the distance is measured, so one panel serves a taxonomy, a
fundamentals table and a factor model at once.

Use a feature matrix for structure the returns do not contain. A classification, a mandate, a
supply chain or a factor model brings in relationships the price history has no record of. A
feature matrix built from the returns graph is a different tool. Section 3.2 introduces it, and
section 4 covers its settings.

!!! tip "When to reach for this"
    Reach for a [`FeatureDistance`](@ref) when you can name the structure you want the allocation
    to respect and it is not in the price history: a sector or country taxonomy, a regulatory
    bucketing, an ESG classification, a factor exposure profile. Reach for it too when you want
    the hierarchy to hold still between rebalances. A classification does not move when the
    covariance moves, and section 9 measures what that saves in turnover. Stay with the ordinary
    correlation distance when the structure you care about is co-movement itself.
=#

using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, StatsPlots,
      GraphRecipes, Statistics, LinearAlgebra, Clustering

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;

#=
## 1. The returns and a classification

We take one year of daily returns of twenty S&P 500 stocks, and add a two-level classification
of the twenty. The levels nest. Every industry belongs to exactly one sector, so two assets in
the same industry are in the same sector.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd0 = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd0)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

sector = Dict("AAPL" => "Technology", "AMD" => "Technology", "MSFT" => "Technology",
              "BAC" => "Financials", "JPM" => "Financials", "CVX" => "Energy",
              "XOM" => "Energy", "RRC" => "Energy", "GE" => "Industrials",
              "BBY" => "ConsumerDiscretionary", "HD" => "ConsumerDiscretionary",
              "KO" => "ConsumerStaples", "PEP" => "ConsumerStaples",
              "PG" => "ConsumerStaples", "WMT" => "ConsumerStaples", "JNJ" => "HealthCare",
              "LLY" => "HealthCare", "MRK" => "HealthCare", "PFE" => "HealthCare",
              "UNH" => "HealthCare")

industry = Dict("AAPL" => "ConsumerHardware", "AMD" => "Semiconductors",
                "MSFT" => "Software", "BAC" => "Banks", "JPM" => "Banks",
                "CVX" => "IntegratedOil", "XOM" => "IntegratedOil",
                "RRC" => "ExplorationProduction", "GE" => "Conglomerates",
                "BBY" => "SpecialtyRetail", "HD" => "SpecialtyRetail", "KO" => "Beverages",
                "PEP" => "Beverages", "PG" => "HouseholdProducts", "WMT" => "MassMerchants",
                "JNJ" => "Pharmaceuticals", "LLY" => "Pharmaceuticals",
                "MRK" => "Pharmaceuticals", "PFE" => "Pharmaceuticals",
                "UNH" => "ManagedCare")

#=
We store the classification in a [`UniverseSets`](@ref). A key whose values must follow the
assets when the library keeps only some of them, as a fold or a subset does, carries the `xkey`
prefix. Write `"nx_sector"` rather than `"sector"`, so that [`port_opt_view`](@ref), the function
that cuts the data down to a subset of the assets, slices the labels with the asset names.
=#

sets = UniverseSets(; xkey = "nx",
                    dict = Dict("nx" => rd0.nx, "nx_sector" => [sector[a] for a in rd0.nx],
                                "nx_industry" => [industry[a] for a in rd0.nx]))
taxonomy = ["nx_sector", "nx_industry"]

pretty_table(DataFrame("Asset" => rd0.nx, "Sector" => [sector[a] for a in rd0.nx],
                       "Industry" => [industry[a] for a in rd0.nx]);
             title = "The classification the asset panel will hold")

#=
## 2. From a classification to an asset panel

An asset panel is a stack of named fields over the assets, and one field holds one named quantity.
[`panel_input`](@ref) turns one `UniverseSets` key into one raw field. What it returns depends on
the values of the key: a vector of strings becomes a [`CategoricalPanelInput`](@ref), and a vector
of numbers becomes a [`NumericPanelInput`](@ref). The vector form maps over several keys, which is
how a nested taxonomy enters. The field takes its name from the key with the `xkey` prefix stripped,
so `"nx_sector"` becomes the field `"sector"`.

[`asset_panel`](@ref) turns the raw inputs into the [`AssetPanel`](@ref) that the
[`ReturnsResult`](@ref) holds. A categorical field stores one integer code per asset over its own
levels rather than an indicator block, so the panel holds the classification once however many
levels it has.
=#

inputs = panel_input(sets, taxonomy)
pnl = asset_panel(inputs)

pretty_table(DataFrame("Field" => [f.name for f in pnl.pf],
                       "Kind" => [string(nameof(typeof(f))) for f in pnl.pf],
                       "Levels" => [length(PortfolioOptimisers.panel_field_labels(f))
                                    for f in pnl.pf]);
             title = "One panel field per level of the classification")

#=
[`feature_matrix`](@ref) stacks the panel into the matrix a distance measures, and
[`feature_labels`](@ref) names its columns. A categorical field gives one `0`/`1` column per
level, so `Z[i, k] == 1` when asset `i` belongs to group `k`.

Take the column names from [`feature_labels`](@ref) rather than rebuilding the order by hand. Each
label is the selector entry that picks out its own column, so the label vector is itself a
selector. We stack the panel against that vector below and compare the result with `Z`.
=#

Z = feature_matrix(pnl)
nz = feature_labels(pnl)

pretty_table(DataFrame(["Asset" => rd0.nx;
                        [string(first(nz[k]), "=", last(nz[k])) => Z[:, k] for k in 1:6]...]);
             title = "The first six feature columns (of $(length(nz)))")

println("The labels rebuild the matrix: ", feature_matrix(pnl, nz) == Z)

#=
### 2.1 Why this panel needs no standardisation

A categorical field partitions the assets. Each asset lands in exactly one level, so every row of
its indicator block carries exactly one `1`, and a panel of `L` categorical fields gives every
asset a row of norm `sqrt(L)`. The cosine between two assets is then

```
cos(i, j) = shared(i, j) / L
```

the count of levels the two assets share, divided by the number of levels. The default metric,
[`AngularDist`](@ref), therefore takes `L + 1` distinct values here, and it is bounded by `0.5`
rather than `1.0`, because a non-negative row admits no negative cosine.

A numeric or a tensor field carries values on its own scale, so neither the fixed row norm nor
the bound holds for it.
=#

D = distance(FeatureDistance(), Z; dims = 1)
levels = sort(unique(round.(D; digits = 6)))
pretty_table(DataFrame("Quantity" => ["Feature columns", "Levels agreed on (L)",
                                      "Row norm (every row, = sqrt(L))", "Distinct distances",
                                      "The distances themselves"],
                       "Value" => [string(size(Z, 2)), string(length(taxonomy)),
                                   string(round(norm(Z[1, :]); digits = 4)),
                                   string(length(levels)), string(round.(levels; digits = 4))]);
             title = "The feature distance this classification produces")

#=
The table holds three distances, one per level of shared membership, up to the floating-point
noise of `acos`. Two assets in the same industry are at `0.0`, two in the same sector but
different industries at `1/3`, and two that share nothing at `0.5`.

### 2.2 The clustering it produces

The panel is data rather than configuration, so it goes on the [`ReturnsResult`](@ref) with the
returns. The distance estimator goes into an ordinary [`ClustersEstimator`](@ref) through its `de`
keyword, and [`clusterise`](@ref) takes the panel from the [`ReturnsResult`](@ref) you pass it.
=#

rd = ReturnsResult(; nx = rd0.nx, X = rd0.X, ts = rd0.ts, pnl = pnl)

onc = OptimalNumberClusters(;)
cle_cor = ClustersEstimator(; onc = onc)
cle_fea = ClustersEstimator(; de = FeatureDistance(), onc = onc)

clr_cor = clusterise(cle_cor, rd)
clr_fea = clusterise(cle_fea, rd)

pretty_table(DataFrame("Asset" => rd.nx, "Sector" => [sector[a] for a in rd.nx],
                       "Correlation cut" => assignments(clr_cor),
                       "Feature cut" => assignments(clr_fea));
             title = "Cluster assignments according to correlation and feature distance")

#=
We score the two hierarchies against each other with the adjusted Rand index, cut by cut. On
this universe the two cuts are identical at four clusters, where the index is `1.0`, and they
differ at every other cut from two to ten. The feature route also differs in the merge order,
and in what happens when the sample moves, which section 9 measures.
=#

agreement = DataFrame("k" => 2:10,
                      "Adjusted Rand index" => [round(randindex(cutree(clr_cor.res; k = k),
                                                                cutree(clr_fea.res; k = k))[1]; digits = 3)
                                                for k in 2:10])
pretty_table(agreement; title = "How far the two hierarchies agree, cut by cut")

# The feature distance gives these clusters.
plot_clusters(clr_fea, rd.nx)

# The correlation distance gives these clusters.
plot_clusters(clr_cor, rd.nx)

#=
We hand both hierarchies to [`HierarchicalRiskParity`](@ref) and compare the weights. Nothing on
the optimiser names the feature source. The panel comes with the returns, and the distance
estimator takes it from there.
=#

hrp_cor = optimise(HierarchicalRiskParity(;
                                          opt = HierarchicalOptimiser(; pe = pr,
                                                                      cle = cle_cor,
                                                                      slv = slv),
                                          r = Variance()), rd)
hrp_fea = optimise(HierarchicalRiskParity(;
                                          opt = HierarchicalOptimiser(; pe = pr,
                                                                      cle = cle_fea,
                                                                      slv = slv),
                                          r = Variance()), rd)

pretty_table(DataFrame("Asset" => rd.nx, "HRP correlation" => hrp_cor.w,
                       "HRP features" => hrp_fea.w, "Difference" => hrp_fea.w - hrp_cor.w);
             formatters = [resfmt],
             title = "Same risk measure, same prior, two hierarchies")

plot_stacked_bar_composition([hrp_cor, hrp_fea], rd;
                             xticks = (1:2, ["Correlation", "Features"]))

#=
## 3. Where the panel comes from

The `ape` field of [`FeatureDistance`](@ref) decides which panel the distance measures, and it
takes two kinds of value.

  - `ape = nothing`, the default, measures the panel of the [`ReturnsResult`](@ref). That is what
    section 2 does. You built the panel, so you named its fields and you know what is in it.
  - `ape = <producer>` builds a panel at the point of use, from the prior and the returns the
    subproblem was handed. A producer is an [`AbstractAssetPanelEstimator`](@ref), and the library
    ships two.

A producer ignores any panel that came with the returns, and `ape = nothing` builds no panel of
its own. Each value of `ape` therefore gives exactly one panel.

The two routes differ under a fold. A panel that comes with the returns is data, so a fold
slices it. A produced panel is recomputed on the subproblem's own returns, so a fold refits it.
Section 8 measures the difference.

### 3.1 `RegressionPanel`, factor loadings

[`RegressionPanel`](@ref) takes the loadings a factor prior has already fitted, so an asset's
feature row is its position in the factor coordinate system. The loadings come from the same
returns, so this route brings in no outside structure. It is still a different reading of those
returns, because two assets can load alike and co-move weakly.

The panel it returns holds one [`TensorPanelField`](@ref) named `"loadings"`. Its trailing axis
takes the factor names from the returns it was handed. Loadings carry a sign, which decides the
metric in section 5.
=#

F = TimeArray(CSV.File(joinpath(@__DIR__, "..", "Factors.csv.gz")); timestamp = :Date)[(end - 252):end]
rdf = prices_to_returns(price_ingestion(PriceIngestion(), X; F = F))
pr_loadings = prior(FactorPrior(), rdf)
pnl_loadings = asset_panel(RegressionPanel(), pr_loadings, rdf, rdf.X)
Z_loadings = feature_matrix(pnl_loadings)

pretty_table(DataFrame(["Asset" => rd.nx;
                        [last(l) => Z_loadings[:, k]
                         for (k, l) in pairs(feature_labels(pnl_loadings))]...]);
             formatters = [(v, i, j) -> j == 1 ? v : round(v; digits = 4)],
             title = "Factor loadings as a panel field")

#=
A producer that uses a prior needs one, so we hand [`clusterise`](@ref) the prior result and pass
the returns beside it as a keyword.
=#

de_loadings = FeatureDistance(; ape = RegressionPanel())
clr_loadings = clusterise(ClustersEstimator(; de = de_loadings, onc = onc), pr_loadings;
                          rd = rdf)
println("The producer and the panel agree exactly: ",
        clr_loadings.D == distance(FeatureDistance(), Z_loadings; dims = 1))

#=
### 3.2 `PhylogenyPanel`, the returns graph

[`PhylogenyPanel`](@ref) turns the asset network into a square field of assets by assets, where
column `k` says how close each asset is to asset `k`. The graph is filtered out of the
correlation, so this route brings in no outside structure either. What it adds is a graded reading
of the network. [`phylogeny_matrix`](@ref) accumulates a walk count and then clamps it to `0` or
`1`, which drops the step count. This one keeps the step count.

It always takes an estimator, never a precomputed result, so it rebuilds the graph on the
assets of each subproblem. It needs no prior, so it can also run where no prior is fitted yet,
such as a step that preselects assets.
=#

ape_graph = PhylogenyPanel(; pl = NetworkEstimator(; sep = HopCount(; n = 2)),
                           alg = Proximity(; decay = LinearDecay()))
pnl_graph = asset_panel(ape_graph, nothing, rd, rd.X)
Z_graph = feature_matrix(pnl_graph)

pretty_table(DataFrame(["Asset" => rd.nx;
                        [rd.nx[k] => Z_graph[:, k] for k in 1:6]...]);
             formatters = [(v, i, j) -> j == 1 ? v : round(v; digits = 4)],
             title = "The first six columns of the graph panel field")

#=
In the table, `3` is the asset itself, `2` a direct neighbour, `1` a two-hop neighbour, and `0`
an asset the budget does not reach. Section 4 says where those numbers come from, and why only
[`LinearDecay`](@ref) uses the budget as a scale.

### 3.3 The three routes side by side
=#

routes = DataFrame("Route" =>
                       ["The panel of the returns", "RegressionPanel", "PhylogenyPanel"],
                   "`ape`" => ["nothing", "RegressionPanel()", "PhylogenyPanel(; …)"],
                   "Feature axis" => ["whatever you named", "factors", "the assets"],
                   "Shape here" =>
                       [string(size(Z)), string(size(Z_loadings)), string(size(Z_graph))],
                   "Exogenous" => ["depends on the source", "no", "no"],
                   "Signed" => ["depends on the source", "yes", "no"],
                   "Under a fold" => ["sliced", "refitted", "refitted"])
pretty_table(routes; title = "The three routes to a feature matrix")

#=
## 4. Two settings on the graph producer, and neither implies the other

[`PhylogenyPanel`](@ref) takes two settings, and they are fields of two different objects.

  - `sep` on the [`NetworkEstimator`](@ref) decides which pairs are related, which is how far
    apart two assets can be and still score above zero. [`HopCount`](@ref) counts edges with a
    budget of `n` of them. [`PathLength`](@ref) sums distances along the shortest path with a
    budget `dmax` in those units.
  - `decay` on [`Proximity`](@ref) decides how strongly a related pair scores as the separation
    grows.

Setting one does not set the other, and mixing them up raises no error. The default pairing hides
the difference. Under [`LinearDecay`](@ref) with [`HopCount`](@ref) the budget is also the top of
the scale, so changing `sep` looks like it changes the fall-off. Under any other decay the two are
separate. The cell below crosses four decays against two separations.
=#

separations = ["HopCount(; n = 2)" => HopCount(; n = 2),
               "PathLength(; dmax = 1.0)" => PathLength(; dmax = 1.0)]
decays = ["LinearDecay()" => LinearDecay(), "ExponentialDecay()" => ExponentialDecay(),
          "ReciprocalDecay()" => ReciprocalDecay(), "NoDecay()" => NoDecay()]

function graph_panel(sep, decay)
    ape = PhylogenyPanel(; pl = NetworkEstimator(; sep = sep),
                         alg = Proximity(; decay = decay))
    return asset_panel(ape, nothing, rd, rd.X)
end

sweep = DataFrame()
for (sname, sep) in separations, (dname, decay) in decays
    png = graph_panel(sep, decay)
    Zg = feature_matrix(png)
    off = [Zg[i, j] for i in axes(Zg, 1) for j in axes(Zg, 2) if i != j]
    clg = clusterise(cle_fea, ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts, pnl = png))
    append!(sweep,
            DataFrame("Separation" => sname, "Decay" => dname, "Self score" => Zg[1, 1],
                      "Largest off-diagonal" => maximum(off),
                      "Related pairs" => count(!iszero, off),
                      "ARI vs correlation" =>
                          randindex(cutree(clr_cor.res; k = 4), cutree(clg.res; k = 4))[1]))
end

pretty_table(sweep;
             formatters = [(v, i, j) -> isa(v, AbstractFloat) ? round(v; digits = 4) : v],
             title = "Four decays crossed against two separations")

#=
Compare the table column by column rather than row by row.

  - **`Related pairs` moves with the separation and never with the decay.** Which pairs are
    related is `sep`'s question alone.
  - **`Self score` is `1.0` for every decay except [`LinearDecay`](@ref).** Under that decay it is
    the budget plus one, so `3` under `HopCount(; n = 2)` and `2` under `PathLength(; dmax = 1.0)`.
    The other three pin `f(0) = 1` and take the fall-off from their own parameter, whatever the
    budget is. Under the default pairing the budget doubles as the scale, and under every other
    pairing it does not.
  - **`ARI vs correlation` moves with both settings.** The decay changes the distance, and it
    changes the clusters that come out of it.
  - **[`NoDecay`](@ref) does not mean no truncation.** The budget still cuts, so the field is `1`
    inside the budget and `0` outside it, which is an indicator of the neighbourhood rather than a
    matrix of ones. It is also the only decay under which the two separations score the same
    against the correlation cut. An indicator keeps only the support, and the `Related pairs`
    column shows that the two supports differ by two entries.
=#

plot(1:size(sweep, 1), sweep[!, "ARI vs correlation"]; marker = :circle, legend = false,
     xticks = (1:size(sweep, 1),
               [string(first(split(r.Separation, "(")), " / ", first(split(r.Decay, "(")))
                for r in eachrow(sweep)]), xrotation = 45,
     ylabel = "Adjusted Rand index against the correlation cut",
     title = "Both knobs move the clustering")

#=
!!! warning "The same setting does something else on the constraint path"

    `PathLength()` with no `dmax` means the whole connected component. Here that is a reasonable
    choice, because the budget only sets where the scores stop, and the decay still grades every
    pair inside it. On the constraint path the same setting selects pairs rather than grading
    them, and it relates every reachable pair. A [`SemiDefinitePhylogenyEstimator`](@ref) then
    removes the diversification benefit of every pair from the variance the optimiser minimises.
    It forbids no joint holding, but the minimum-risk portfolio puts all its weight in one asset,
    and the solve reports no failure. Section 2.2 of
    [Phylogeny and centrality constraints](../4_constraints_costs/04_Phylogeny_Centrality.md)
    warns about that setting, and its section 3.1 shows the one-asset portfolio. The bare default
    also ties the scale of the field to the sample, which section 8.3 measures.

## 5. Metrics, and the similarity field

The `metric` field of [`FeatureDistance`](@ref) takes any `Distances.SemiMetric`, including one
you write yourself. The default is [`AngularDist`](@ref), the arc-cosine of the cosine similarity
scaled to `[0, 1]`.

The choice changes the answer. Metrics differ in what they are defined on, and one of them returns
a number on input outside its domain rather than raising, so the library checks the domain.
=#

metrics = ["AngularDist()" => AngularDist(),
           "Distances.CosineDist()" => PortfolioOptimisers.Distances.CosineDist(),
           "Distances.Jaccard()" => PortfolioOptimisers.Distances.Jaccard()]

metric_rows = DataFrame()
for (mname, metric) in metrics
    de = FeatureDistance(; metric = metric)
    Dm = distance(de, Z; dims = 1)
    append!(metric_rows,
            DataFrame("Metric" => mname, "Default similarity" => strip(string(de.sim)),
                      "Maximum distance" => round(maximum(Dm); digits = 4),
                      "Distinct values" => length(unique(round.(Dm; digits = 6)))))
end
pretty_table(metric_rows; title = "Three metrics on the same classification panel")

#=
On a matrix of indicator columns each of the three metrics falls as the number of levels two
assets share rises, so all three put the pairs in the same order and give three distinct
distances. The values differ. [`AngularDist`](@ref) stops at `0.5`, because a non-negative matrix
admits no negative cosine. A signed matrix is a different case, and the next cell measures one.
=#

signed_metric = try
    distance(FeatureDistance(; metric = PortfolioOptimisers.Distances.Jaccard()),
             Z_loadings; dims = 1)
    "no error"
catch err
    sprint(showerror, err)
end
println(signed_metric)

#=
`Distances.Jaccard` is the Ruzicka form, and it is defined only on non-negative reals. On signed
input it returns values up to `2` and raises nothing, and a clustering routine then takes those
values as distances. The domain check raises the error printed above instead, and it covers
`Distances.BrayCurtis` and `Distances.ChiSqDist` as well. A signed field, and a block of factor
loadings above all, needs [`AngularDist`](@ref) or `Distances.CosineDist`.

### The `sim` field

A clustering estimator calls [`cor_and_dist`](@ref) rather than [`distance`](@ref), so a feature
distance must also give a similarity matrix. The `sim` field says how to compute it, and
[`default_similarity`](@ref) sets it from the metric. [`AngularDist`](@ref) takes
[`AngularSimilarity`](@ref), which recovers the cosine as `cos(πD)`, and every other metric takes
[`ComplementSimilarity`](@ref), which is `1 - D`. Set `sim` yourself to override that.
=#

S_fea, D_fea = cor_and_dist(FeatureDistance(), nothing, rd.X; rd = rd)
println("S is cos(πD): ", S_fea ≈ cos.(π .* D_fea))

#=
## 6. Both shapes: static and time-varying

A panel is static or time-varying, and the whole panel is one shape. A static panel carries no
mask, and every field is `assets` or `assets × labels`. A time-varying panel leads every field
with an observation axis, and carries an active mask and an estimation mask beside the fields.
[`feature_matrix`](@ref) follows the shape. A static panel stacks to `assets × features`, and a
time-varying one to `observations × assets × features`.

A time-varying matrix must give one distance matrix, and the `alg` field says how. The library
ships four collapse rules, and you can write your own.

The two features here move with the sample: annualised realised volatility over twenty-one days,
and cumulative return over sixty-three. Both are ordinary numeric fields, so
[`asset_panel`](@ref) builds them as it built the taxonomy.
=#

T, N = size(rd.X)
Zvol = zeros(T, N)
Zmom = zeros(T, N)
for t in 1:T, i in 1:N
    Zvol[t, i] = std(view(rd.X, max(1, t - 20):t, i)) * sqrt(252)
    Zmom[t, i] = sum(view(rd.X, max(1, t - 62):t, i))
end
Zvol[1, :] .= Zvol[2, :]

pnl_tv = asset_panel([NumericPanelInput(; name = "volatility", vals = Zvol),
                      NumericPanelInput(; name = "momentum", vals = Zmom)])
Ztv = feature_matrix(pnl_tv)

collapses = ["LastObservation()" => LastObservation(),
             "AggregateFeatures()" => AggregateFeatures(),
             "AggregateDistances()" => AggregateDistances(),
             "StackObservations()" => StackObservations()]

collapse_rows = DataFrame()
for (cname, alg) in collapses
    Dc = distance(FeatureDistance(; alg = alg), Ztv; dims = 1)
    D1 = distance(FeatureDistance(; alg = alg), reshape(Ztv[end, :, :], 1, N, 2); dims = 1)
    append!(collapse_rows,
            DataFrame("Collapse" => cname, "Mean distance" => round(mean(Dc); digits = 4),
                      "Maximum distance" => round(maximum(Dc); digits = 4),
                      "Mean at T = 1" => round(mean(D1); digits = 6)))
end
pretty_table(collapse_rows;
             title = "Four collapse rules on one $(size(Ztv, 1))×$(size(Ztv, 2))×$(size(Ztv, 3)) feature matrix")

#=
The four rules give four different answers.

  - [`LastObservation`](@ref), the default, uses the most recent slice and ignores the rest.
  - [`AggregateFeatures`](@ref) averages the features first, then measures the distance once.
  - [`AggregateDistances`](@ref) measures each period, then averages the distance matrices. Its
    constructor refuses [`MedianCollapse`](@ref), because a convex combination of metrics is a
    metric and a median of distance matrices is not.
  - [`StackObservations`](@ref) joins every period into one long coordinate vector. It is the most
    exposed of the four to scale, because a period with large magnitudes dominates the rest.

The last column of the table is the degenerate case. At one observation the four rules return the
same distances. A static panel ignores `alg`, and setting one there raises no error, because one
estimator serves both shapes.

Both averaging rules take observation weights on a `w` field, so an exponential decay or an
entropy-pooling posterior reaches the collapse.
=#

plot([distance(FeatureDistance(; alg = alg), Ztv; dims = 1)[1, :] for (_, alg) in collapses];
     label = reshape([c for (c, _) in collapses], 1, :), marker = :circle,
     xticks = (1:N, rd.nx), xrotation = 90, ylabel = "Distance from AAPL",
     title = "The collapse rule is a modelling choice, not a detail")

#=
### 6.1 A static field on a time-varying panel is stored once

A panel is one shape throughout, so a static input that joins a time-varying one gains an
observation axis.
[`asset_panel`](@ref) wraps its values in a [`RepeatedLeading`](@ref), which stores them once and
indexes a leading observation axis. A classification that never moves therefore costs its own
memory beside characteristics that move every day, rather than its own memory times the
observation count.
=#

pnl_mixed = asset_panel([panel_input(sets, taxonomy);
                         NumericPanelInput(; name = "volatility", vals = Zvol);
                         NumericPanelInput(; name = "momentum", vals = Zmom)])

held(f) = isa(f, CategoricalPanelField) ? f.codes : f.vals
stored(v) = isa(v, PortfolioOptimisers.RepeatedLeading) ? length(v.parent) : length(v)

pretty_table(DataFrame("Field" => [f.name for f in pnl_mixed.pf],
                       "Values type" =>
                           [string(nameof(typeof(held(f)))) for f in pnl_mixed.pf],
                       "Shape it presents" => [string(size(held(f))) for f in pnl_mixed.pf],
                       "Entries stored" => [stored(held(f)) for f in pnl_mixed.pf]);
             title = "A lifted field presents an observation axis it does not store")

#=
## 7. The selector names fields of the panel

The `sel` field of [`FeatureDistance`](@ref) names what to stack, by the names of the panel's own
fields, levels and labels. You never count integer columns. An entry takes one of four forms.

| Entry                          | Columns it contributes                        |
|:------------------------------ |:--------------------------------------------- |
| `"name"`                       | every column of that field                    |
| `"name" => ["a", "b"]`         | the levels or labels listed, in that order     |
| `"name" => "a"`                | one level or label                            |
| `"name" => :observed`          | that field's observed mask, one `0`/`1` column |

You can mix the forms. The order of the vector is the order of the columns. The default,
`nothing`, stacks every field's values in field order. A bare name gives the values alone and
never the mask.
=#

selectors = ["nothing (every field)" => nothing, "[\"sector\"]" => ["sector"],
             "[\"industry\"]" => ["industry"],
             "[\"sector\" => \"Energy\", \"industry\"]" =>
                 ["sector" => "Energy", "industry"],
             "[\"sector\" => [\"Energy\", \"HealthCare\"]]" =>
                 ["sector" => ["Energy", "HealthCare"]]]

selector_rows = DataFrame()
for (sname, sel) in selectors
    Zs = feature_matrix(pnl, sel)
    Ds = distance(FeatureDistance(), Zs; dims = 1)
    append!(selector_rows,
            DataFrame("Selector" => sname, "Columns" => size(Zs, 2),
                      "Distinct distances" => length(unique(round.(Ds; digits = 6))),
                      "Maximum distance" => round(maximum(Ds); digits = 4)))
end
pretty_table(selector_rows; title = "One namespace, five cuts of the same panel")

#=
Cutting to `"sector"` alone gives two distances rather than three, because the industry level is
what separated pairs inside one sector. Cutting to two levels of one field moves the bound as
well. An asset in neither kept level has a feature row of zeros, so the largest distance reaches
`1.0` rather than the `0.5` a full partition gives.

The selector lives on the estimator, so the cut is configuration and the panel stays data.
=#

de_coarse = FeatureDistance(; sel = ["sector"])
clr_coarse = clusterise(ClustersEstimator(; de = de_coarse, onc = onc), rd)
println("The coarse cut and the full panel differ: ", clr_coarse.D != clr_fea.D)
println("The estimator's own labels: ", feature_labels(de_coarse, pr, rd, rd.X))

#=
### 7.1 `strict`, and an entry the panel does not hold

`strict` is the setting the whole library uses for an input it cannot match. Here it decides what
happens to a name, a level or a label that is not in the panel. Under the default, `strict = false`,
the entry is dropped with a warning. Under `strict = true` the call raises. Dropping the entry is
what lets one estimator serve a universe whose panel has lost a field, which happens inside a fold.
=#

dropped = feature_matrix(pnl, ["not_a_field", "sector"])
println("The unknown entry was dropped: ", size(dropped, 2) == 7)

strict_error = try
    feature_matrix(pnl, ["not_a_field", "sector"]; strict = true)
    "no error"
catch err
    sprint(showerror, err)
end
println(strict_error)

#=
### 7.2 The observed mask is a column you can select

A field built from a table with gaps records which cells were observed, and the mask survives the
fill. `"name" => :observed` puts the mask in the matrix as one `0`/`1` column, so whether an asset
reported at all becomes a feature of its own. A field with no mask gives a column of ones,
because no cell was missing.
=#

Zgap = copy(Zvol)
Zgap[1:40, 1:3] .= NaN
pnl_gap = asset_panel([NumericPanelInput(; name = "volatility", vals = Zgap,
                                         alg = BackwardPanelFill())])
Zobs = feature_matrix(pnl_gap, ["volatility" => :observed])
Zboth = feature_matrix(pnl_gap, ["volatility", "volatility" => :observed])

pretty_table(DataFrame("Selector entry" =>
                           ["[\"volatility\"]", "[\"volatility\" => :observed]",
                            "both, in that order"],
                       "Shape" => [string(size(feature_matrix(pnl_gap, ["volatility"]))),
                                   string(size(Zobs)), string(size(Zboth))],
                       "Mean of the last column" =>
                           [round(mean(Zgap[.!isnan.(Zgap)]); digits = 4),
                            round(mean(Zobs); digits = 4),
                            round(mean(selectdim(Zboth, 3, 2)); digits = 4)]);
             title = "The mask a fill policy left behind")

#=
## 8. Under a fold, and under a meta-optimiser

### 8.1 The square case: slicing and measuring do not commute

[`features_are_assets`](@ref) decides whether the labels of a field are the assets, and it
compares names rather than axis lengths. When the labels of a tensor field are the asset names,
the field's label axis is the asset axis, so the library slices it too when it keeps a subset of
the assets. A square field keyed by asset is therefore the only shape where measuring a
subproblem differs from reading the subproblem out of the distance matrix of the whole universe.
The library stores no flag for this, and it compares the names each time it slices.
=#

rd_square = ReturnsResult(; nx = rd.nx, X = rd.X, ts = rd.ts, pnl = pnl_graph)
subset = [1, 2, 3, 5, 8, 10, 13, 17]

view_square = PortfolioOptimisers.port_opt_view(rd_square, subset)
view_rect = PortfolioOptimisers.port_opt_view(rd, subset)

commute = DataFrame("Feature axis" =>
                        ["The assets (square)", "Taxonomy levels (rectangular)"],
                    "Shape of the view" => [string(size(feature_matrix(view_square.pnl))),
                                            string(size(feature_matrix(view_rect.pnl)))],
                    "Largest disagreement" => [maximum(abs,
                                                       distance(FeatureDistance(), Z_graph; dims = 1)[subset,
                                                                                                      subset] -
                                                       distance(FeatureDistance(),
                                                                feature_matrix(view_square.pnl); dims = 1)),
                                               maximum(abs,
                                                       distance(FeatureDistance(), Z; dims = 1)[subset, subset] -
                                                       distance(FeatureDistance(), feature_matrix(view_rect.pnl);
                                                                dims = 1))])
pretty_table(commute;
             formatters = [(v, i, j) -> isa(v, AbstractFloat) ? round(v; digits = 4) : v],
             title = "Measuring the subproblem against measuring the universe")

#=
The rectangular case shows a difference of zero. Its columns are the same twenty groups whichever
assets you keep, so slicing the rows and measuring commute. The square case shows a difference
well above the floating-point noise. It is the gap between asking how close two assets are inside
this cluster and asking how close they are in the whole universe. Both questions are reasonable,
and the shape of the field decides which one you asked.

A produced panel is refitted on the subproblem, so the question does not arise.
[`PhylogenyPanel`](@ref) rebuilds the graph on the cluster rather than cutting the graph of the
universe down to it.

### 8.2 A meta-optimiser's outer problem

A meta-optimiser solves its outer problem over synthetic assets, such as sub-portfolios, clusters
or predictions, and none of them has a row in any panel. The outer problem still takes features.
The inner weights collapse the panel of the inner universe onto the synthetic assets, so an
outer [`FeatureDistance`](@ref) measures the features of the sub-portfolios rather than raising
an error.

The collapse works field by field on the panel, not on the stacked matrix. A numeric field stays
numeric, and a tensor field stays tensor. A categorical field becomes a tensor field of membership
fractions over its own levels, so a cluster's row holds its weighted average membership of each
group. A selector that named `"sector"` resolves unchanged on the outer panel.
=#

nco = NestedClustered(; pe = pr, cle = cle_fea,
                      opti = HierarchicalRiskParity(;
                                                    opt = HierarchicalOptimiser(; pe = pr,
                                                                                cle = cle_cor,
                                                                                slv = slv),
                                                    r = Variance()),
                      opto = HierarchicalRiskParity(;
                                                    opt = HierarchicalOptimiser(;
                                                                                cle = cle_fea,
                                                                                slv = slv),
                                                    r = Variance()))
res_nco = optimise(nco, rd)
println("Outer problem solved on the collapsed panel: ",
        isa(res_nco.retcode, OptimisationSuccess))

#=
A square field keyed by asset is contracted on both axes at once, so it stays square on the
synthetic universe. A mask collapses to the support of its convex combination, `(Wᵀ · m) .> 0`, so
a synthetic asset counts as observed when any asset that funds it was observed.

### 8.3 A bare `PathLength()` ties the scale to the sample

[`PathLength`](@ref) with no `dmax` resolves its budget to the observed diameter of the graph.
Under [`LinearDecay`](@ref) the top of the scale is the budget plus one, so a diameter that moves
between folds moves the whole field with it. We roll a one-year window forward a quarter at a
time over five years.
=#

Xb = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 1260):end]
rdb = prices_to_returns(Xb)
windows = [(i, i + 251) for i in 1:63:(size(rdb.X, 1) - 251)]

function window_row(lo, hi)
    Xw = prior(EmpiricalPrior(), rdb.X[lo:hi, :]).X
    seps = separation_matrix(PathLength(), NetworkEstimator(), Xw)
    Zbare = phylogeny_features(Proximity(; decay = LinearDecay()),
                               NetworkEstimator(; sep = PathLength()), Xw)
    Zfixed = phylogeny_features(Proximity(; decay = LinearDecay()),
                                NetworkEstimator(; sep = PathLength(; dmax = 1.5)), Xw)
    return (; Window = "$(lo)–$(hi)",
            var"Observed diameter" = maximum(filter(isfinite, seps)),
            var"Self score, bare" = Zbare[1, 1], var"Self score, dmax = 1.5" = Zfixed[1, 1])
end

diameters = DataFrame([window_row(lo, hi) for (lo, hi) in windows])
pretty_table(diameters;
             formatters = [(v, i, j) -> isa(v, AbstractFloat) ? round(v; digits = 4) : v],
             title = "The bare budget follows the sample; a stated one does not")

plot(1:length(windows), diameters[!, "Self score, bare"]; marker = :circle,
     label = "PathLength()", xlabel = "Rolling window", ylabel = "Top of the scale",
     title = "A data-dependent budget moves the whole panel field")
plot!(1:length(windows), diameters[!, "Self score, dmax = 1.5"]; marker = :square,
      label = "PathLength(; dmax = 1.5)")

#=
The diameter about doubles across these windows, and the scale follows it. Going from a bare
budget to a stated one adds a constant to every entry inside the budget rather than multiplying
them by a factor. We measure both below.
=#

Xw1 = prior(EmpiricalPrior(), rdb.X[1:252, :]).X
Z_bare = phylogeny_features(Proximity(; decay = LinearDecay()),
                            NetworkEstimator(; sep = PathLength()), Xw1)
Z_fixed = phylogeny_features(Proximity(; decay = LinearDecay()),
                             NetworkEstimator(; sep = PathLength(; dmax = 3.0)), Xw1)
shared = (Z_bare .!= 0) .&& (Z_fixed .!= 0)

D_bare = distance(FeatureDistance(), Z_bare; dims = 1)
D_scaled = distance(FeatureDistance(), 7.3 .* Z_bare; dims = 1)
Z_shifted = copy(Z_bare)
Z_shifted[Z_bare .!= 0] .+= 1.0
D_shifted = distance(FeatureDistance(), Z_shifted; dims = 1)

pretty_table(DataFrame("Quantity" => ["Distinct differences on the shared support",
                                      "The difference itself",
                                      "Distance change from rescaling by 7.3",
                                      "Distance change from adding 1.0 on the support"],
                       "Value" =>
                           [string(length(unique(round.(Z_bare[shared] - Z_fixed[shared];
                                                        digits = 8)))),
                            string(round(first(unique(round.(Z_bare[shared] -
                                                             Z_fixed[shared]; digits = 8)));
                                         digits = 6)),
                            string(round(maximum(abs, D_scaled - D_bare); digits = 12)),
                            string(round(maximum(abs, D_shifted - D_bare); digits = 4))]);
             title = "A shift is not a rescale, and only one of them is invisible")

#=
[`AngularDist`](@ref) does not change when an asset's whole row is rescaled, which is why the
third row of the table is zero to machine precision. A shift changes the direction the row points,
and the fourth row is not zero. The invariance of the metric therefore does not absorb a moving
diameter, and the distance changes fold by fold.

Two settings close the gap, and each is one keyword.

  - State a numeric `dmax`, which fixes the budget and the scale across every fold.
  - Use a decay that pins `f(0) = 1`, which is [`ExponentialDecay`](@ref),
    [`ReciprocalDecay`](@ref) or [`NoDecay`](@ref). None of the three reads the budget as a scale.

## 9. A walk-forward backtest

The reason to reach for a classification panel is what it does to the turnover. A correlation
hierarchy refits from scratch every fold and moves whenever the covariance moves. A classification
does not move at all. The cells below walk forward over five years with one-year training windows
and quarterly rebalances.
=#

sets_bt = UniverseSets(; xkey = "nx",
                       dict = Dict("nx" => rdb.nx,
                                   "nx_sector" => [sector[a] for a in rdb.nx],
                                   "nx_industry" => [industry[a] for a in rdb.nx]))
rdbz = ReturnsResult(; nx = rdb.nx, X = rdb.X, ts = rdb.ts,
                     pnl = asset_panel(panel_input(sets_bt, taxonomy)))
walk = IndexWalkForward(252, 63)

bt_cor = cross_val_predict(HierarchicalRiskParity(;
                                                  opt = HierarchicalOptimiser(;
                                                                              pe = EmpiricalPrior(),
                                                                              cle = ClustersEstimator(),
                                                                              slv = slv),
                                                  r = Variance()), rdb, walk)
bt_fea = cross_val_predict(HierarchicalRiskParity(;
                                                  opt = HierarchicalOptimiser(;
                                                                              pe = EmpiricalPrior(),
                                                                              cle = ClustersEstimator(;
                                                                                                      de = FeatureDistance()),
                                                                              slv = slv),
                                                  r = Variance()), rdbz, walk)

function backtest_row(name, p)
    r = p.mrd.X
    turn = mean(abs,
                reduce(vcat,
                       [p.pred[i + 1].res.w - p.pred[i].res.w
                        for i in 1:(length(p.pred) - 1)]))
    return (; Hierarchy = name, var"Annual return" = mean(r) * 252,
            var"Annual volatility" = std(r) * sqrt(252),
            var"Sharpe ratio" = mean(r) / std(r) * sqrt(252),
            var"Maximum drawdown" = expected_risk(MaximumDrawdown(), p),
            var"Mean weight change" = turn)
end

pretty_table(DataFrame([backtest_row("Correlation", bt_cor),
                        backtest_row("Classification", bt_fea)]);
             formatters = [(v, i, j) -> isa(v, AbstractFloat) ? round(v; digits = 4) : v],
             title = "Out-of-sample, $(length(bt_cor.pred)) quarterly rebalances")

#=
Return and volatility are close between the two rows, and the drawdown of the classification run
is smaller. The last column shows that the classification hierarchy changes its weights between
rebalances by less than half of what the correlation hierarchy changes. The reason is the source
of the structure. A sector does not move when a correlation moves, so the dendrogram, the merge
order and the recursive bisection hold still, and the risk estimate inside each cluster is the
only thing left moving.

A classification panel cannot react to a structural break that the returns show.
=#

plot_portfolio_cumulative_returns(bt_fea)

#=
## 10. Summary

  - An [`AssetPanel`](@ref) is the data, one field per named quantity, and it goes on the
    [`ReturnsResult`](@ref) with the returns. [`feature_matrix`](@ref) stacks it where a distance
    measures it, and nothing stores the result.
  - [`FeatureDistance`](@ref) turns that matrix into a distance, so every clustering estimator,
    network estimator and constraint that takes a distance estimator takes it unchanged.
  - `ape` picks the panel. `nothing` uses the panel that came with the returns, and a producer
    builds one at the point of use. [`RegressionPanel`](@ref) and [`PhylogenyPanel`](@ref) are the
    two the library ships, and both derive their panel from the returns. The panel that came with
    the returns is the route that brings in structure from outside, and [`panel_input`](@ref) fills
    it from a taxonomy.
  - `sel` names what to stack by the names of the panel's own fields. An entry is a name, a name
    with the levels or labels it keeps, or a name with its observed mask. `strict` decides whether
    an absent entry is dropped or raises.
  - [`PhylogenyPanel`](@ref) takes two settings on two different objects. `sep` chooses which pairs
    are related, and `decay` how strongly. Neither sets the other, and the default pairing hides
    the difference.
  - Under a fold a panel that came with the returns is sliced, and a produced one is refitted. A
    square field keyed by asset is sliced on its label axis too, so measuring a subproblem and
    reading it out of the universe's distance matrix answer different questions.
  - A bare `PathLength()` ties the scale of a graph field to the sample, and the scale invariance
    of [`AngularDist`](@ref) does not absorb a shift.
=#
