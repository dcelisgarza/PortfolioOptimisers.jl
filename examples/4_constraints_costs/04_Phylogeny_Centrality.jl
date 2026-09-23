#=
```@meta
Description = "Phylogeny and centrality constraints in PortfolioOptimisers.jl: limit joint holdings of linked assets, and tilt toward or away from the network's hubs."
```

# Phylogeny and centrality constraints

The constraints in [Linear and group constraints](02_Linear_Group_Constraints.md) act on asset
names and on groups that you write by hand. Phylogeny and centrality constraints act on the asset
network instead. The asset network is a graph whose nodes are the assets, and whose edges join
assets whose returns move together. In place of a cap such as "tech ≤ 30%", you tell the optimiser
to hold few assets that are close in the network, or to tilt toward or away from the assets with
many edges.

`PortfolioOptimisers.jl` builds the network with a [`NetworkEstimator`](@ref) or with a clustering
estimator. We call that estimator the source of the network. Two families of constraints use it.

  - Phylogeny constraints, [`SemiDefinitePhylogenyEstimator`](@ref) and
    [`IntegerPhylogenyEstimator`](@ref), go through the `ple` keyword. They limit how you hold
    assets that are close in the network.
  - Centrality constraints, a [`CentralityConstraint`](@ref) built from a
    [`CentralityEstimator`](@ref), go through the `cte` keyword. They bound the weighted average
    centrality of the portfolio.

!!! tip "When to reach for this"
    Reach for these constraints when you want to diversify by the structure of the returns rather
    than by labels. A portfolio can spread over many sectors and still be one large bet on assets
    whose returns move together. You can also tilt the portfolio toward the hubs of the network,
    or toward its periphery. You write no groups by hand, because the network comes from the
    covariance. The semidefinite phylogeny constraint and the centrality constraint are convex.
    The integer phylogeny constraint needs a mixed-integer solver.

One parameter decides which assets count as related, and you must choose its value with care. It
is the `sep` field of a [`NetworkEstimator`](@ref), and section 2 covers it.
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

#=
## 1. Data

We load a year of prices, fit an empirical prior, and solve the minimum-risk portfolio with no
network constraint. It is the baseline for every comparison below.
=#

X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res_base = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv)))

#=
## 2. The asset network

A [`NetworkEstimator`](@ref) turns the covariance into a graph. Each asset is a node. The
estimator keeps only the strongest connections between the assets, by default as a minimum
spanning tree, and each connection it keeps is an edge. Both families of constraints below use
this graph. You do not build it yourself. The constraint estimators take a `NetworkEstimator()` and
build the graph from the prior.

### 2.1 How far apart two assets can be and still count as related

The graph shows only which assets share an edge. Each constraint below also needs to know how far
apart two assets can be and still count as related. The `sep` field of the
[`NetworkEstimator`](@ref) sets that distance. It decides how much of the universe each
constraint treats as one bet.

A separation measures how far apart two assets are in the graph. The library has two, and they
measure the same graph in different units.

  - [`HopCount`](@ref) counts edges and ignores their lengths. Its budget `n` is a number of
    edges. `HopCount(; n = 1)` is the default, and it relates only the assets that share an edge.
  - [`PathLength`](@ref) adds up the distances along the shortest path between two assets. Its
    budget `dmax` is in those distance units.

Every constraint accepts either separation. The budgets are in different units, so a value of `n`
and a value of `dmax` do not compare. Pick the unit that you find easier to reason about. Then look
at the number of pairs that the budget relates, because the value of the budget alone does not
tell you how tight the constraint is.

We count the related pairs for eight hop budgets and ten `dmax` budgets, and sort the rows by that
count. The last column divides the count by the number of ordered pairs of assets.
=#

n_assets = size(pr.X, 2)
n_pairs = n_assets * (n_assets - 1)
related_pairs(sep) = count(!iszero, phylogeny_matrix(NetworkEstimator(; sep = sep), pr).X)

hop_budgets = 1:8
dmax_budgets = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5]
hop_pairs = [related_pairs(HopCount(; n = n)) for n in hop_budgets]
dmax_pairs = [related_pairs(PathLength(; dmax = d)) for d in dmax_budgets]

ladder = DataFrame("Separation" => [["HopCount" for _ in hop_budgets];
                                    ["PathLength" for _ in dmax_budgets]],
                   "Budget" => [string.(hop_budgets); string.(dmax_budgets)],
                   "Related pairs" => [hop_pairs; dmax_pairs],
                   "Share of pairs" => [hop_pairs; dmax_pairs] ./ n_pairs)
sort!(ladder, "Related pairs")
pretty_table(ladder; formatters = [(v, i, j) -> j == 4 ? "$(round(v*100, digits=1)) %" : v],
             title = "Pairs the network calls related, by separation and budget")

#=
### 2.2 A radius fills the gaps between the hop budgets

The hop budgets give eight values of the count and nothing between them, because one more hop
relates every pair at that distance at once. We call the pairs that one more hop adds a hop shell.
The `PathLength` rows fall in the gaps between the hop shells, and the smallest `dmax` relates
fewer pairs than `HopCount(; n = 1)`.

A radius measures the same neighbourhood in smaller steps. Use `PathLength` when one hop shell
relates more pairs than you want the constraint to cover.

We plot the count of related pairs against `dmax`, with a dashed line at the count of each hop
budget.
=#

plot(dmax_budgets, dmax_pairs; label = "PathLength (radius ball)", marker = :circle,
     xlabel = "Budget: dmax, in distance units", ylabel = "Pairs called related",
     title = "A continuous radius against eight discrete hop shells", legend = :bottomright)
hline!(hop_pairs; label = "HopCount shells (n = 1…8)", linestyle = :dash, color = :grey,
       linealpha = 0.7)

#=
The largest useful budget for either separation is the diameter of the graph, the longest of the
shortest paths between two assets. [`separation_matrix`](@ref) returns the separation of every
pair, and [`separation_budget`](@ref) returns the budget that a separation resolves to. The
constraints call both functions to find their budget. We print the diameter in both units, the
shortest edge, and the budgets that two `PathLength` settings resolve to.
=#

sep_matrix = separation_matrix(PathLength(), NetworkEstimator(), pr.X)
finite_seps = filter(isfinite, sep_matrix)
pretty_table(DataFrame("Quantity" => ["Observed diameter (distance units)",
                                      "Closest linked pair (distance units)", "Diameter in hops",
                                      "Budget resolved from PathLength()",
                                      "Budget resolved from PathLength(; dmax = 100)"],
                       "Value" => [string(round(maximum(finite_seps); digits = 4)),
                                   string(round(minimum(filter(>(0), finite_seps)); digits = 4)),
                                   string(maximum(separation_matrix(HopCount(), NetworkEstimator(),
                                                                    pr.X))),
                                   string(round(separation_budget(PathLength(), NetworkEstimator(),
                                                                  sep_matrix); digits = 4)),
                                   string(round(separation_budget(PathLength(; dmax = 100),
                                                                  NetworkEstimator(), sep_matrix);
                                                digits = 4))]);
             title = "The budgets this graph admits")

#=
The library cuts a `dmax` above the diameter down to the diameter. A large budget therefore
relates no more than the whole connected component.

!!! warning "`PathLength()` with no `dmax` relates everything"

    `dmax = nothing` is the default of [`PathLength`](@ref), and it means the whole connected
    component. The library uses the observed diameter above for it. A constraint then treats every
    reachable pair as related. The phylogeny matrix marks each related pair with a nonzero entry.
    For `NetworkEstimator(; sep = PathLength())` it is ones off the diagonal, and it relates 100 %
    of pairs, as the last rows of the table in section 2.1 do. It is the opposite extreme
    from the default `n = 1` of `HopCount()`, and you reach it when you change the type of the
    separation and nothing else. The library does not guard against it. The optimisation succeeds
    and returns a portfolio of one asset, as section 3.1 shows. Give a numeric `dmax` to relate
    fewer pairs.

A [`PhylogenyPanel`](@ref) also uses a `sep`, through the `NetworkEstimator` in its `pl` field. The
panel does not build a constraint. It builds a proximity matrix, which scores how close each asset
is to each other asset. The `decay` field of [`Proximity`](@ref) sets how that score falls with
distance, and a [`LinearDecay`](@ref) reaches zero at the budget of `sep`. The default
`PathLength()` suits a panel, and it relates every pair in a constraint. `sep` and `decay` are
fields of different types, and neither sets the other.

### 2.3 When you cannot state the budget in advance

A cross-validation fold fits the graph again on different rows. A meta optimiser such as [`NestedClustered`](@ref) or
[`SubsetResampling`](@ref) fits it again on a different set of assets. A `dmax` that you tuned on
one graph then applies to graphs that you did not tune it on.

So each budget field also accepts a rule. A rule is a callable that gets the estimator, the data
and the graph built from them, and returns the budget. `n` accepts a [`HopCountAlgorithm`](@ref),
`dmax` accepts a [`PathLengthAlgorithm`](@ref), and either accepts a plain function with the same
arguments. [`resolve_separation`](@ref) calls the rule when a constraint needs its budget. It passes
the graph that the constraint already built, and a rule never needs a second graph. The library
gives one rule for each field, [`HopCountQuantile`](@ref) and [`PathLengthQuantile`](@ref). Each
puts the budget at a quantile of the observed separations.

A rule changes which quantity stays fixed. A fixed `dmax` keeps the radius fixed, and the number of
related pairs changes with the graph. A quantile rule keeps the number of related pairs fixed, and
the radius changes. The number of related pairs sets the strength of a constraint, and the
quantile rule keeps that strength from fold to fold.

We split the year into four folds and fit the graph on each. For each fold we count the related
pairs under a fixed `dmax` and under the rule `PathLengthQuantile(; q = 0.25)`, and we print the
`dmax` that the rule resolves to.
=#

folds = [1:63, 64:126, 127:189, 190:252]
function fold_sep(sep, f)
    return count(!iszero, phylogeny_matrix(NetworkEstimator(; sep = sep), pr.X[f, :]).X)
end
q_rule = PathLengthQuantile(; q = 0.25)
function resolved_dmax(f)
    return resolve_separation(PathLength(; dmax = q_rule), NetworkEstimator(), pr.X[f, :]).dmax
end

pretty_table(DataFrame("Fold" => ["$(first(f))–$(last(f))" for f in folds],
                       "Fixed dmax = 1.0107" =>
                           [fold_sep(PathLength(; dmax = 1.0107), f) for f in folds],
                       "Rule: resolved dmax" =>
                           [round(resolved_dmax(f); digits = 4) for f in folds],
                       "Rule: related pairs" =>
                           [fold_sep(PathLength(; dmax = q_rule), f) for f in folds]);
             title = "A fixed radius against a quantile rule, over four folds of the same year")

#=
The fixed `dmax = 1.0107` is the 0.25 quantile of the separations over the whole year. The second
column shows that one radius relates a different number of pairs in each fold. The constraint
then has a different strength in each fold, and no message tells you. The fourth column shows that
the rule relates the same number of pairs in every fold. The third column shows the radius that the
rule moves to keep that number.

The two quantile rules do not do this equally well, and the cause is the unit. `q` is continuous,
but a hop count is an integer, so [`HopCountQuantile`](@ref) must round. The hop shells of section
2.2 are large, and the rounding moves the share of related pairs far from `q`. We ask each rule for
six values of `q` and print the share of pairs that it relates.
=#

q_grid = [0.1, 0.2, 0.25, 0.3, 0.5, 0.75]
pretty_table(DataFrame("q" => q_grid,
                       "HopCountQuantile: n" =>
                           [resolve_separation(HopCount(; n = HopCountQuantile(; q = q)),
                                               NetworkEstimator(), pr.X).n for q in q_grid],
                       "HopCountQuantile: share" =>
                           [related_pairs(HopCount(; n = HopCountQuantile(; q = q))) /
                            n_pairs for q in q_grid],
                       "PathLengthQuantile: share" =>
                           [related_pairs(PathLength(; dmax = PathLengthQuantile(; q = q))) /
                            n_pairs for q in q_grid]);
             formatters = [(v, i, j) -> j in (3, 4) ? "$(round(v*100, digits=1)) %" : v],
             title = "Asking for a share of the pairs, in two units")

#=
[`PathLengthQuantile`](@ref) relates a share of the pairs that differs from `q` by at most one
pair. [`HopCountQuantile`](@ref) can miss `q` by a whole hop shell. Three values of `q` resolve to
`n = 2`, because the second hop shell is large. Use `PathLengthQuantile` when you want a set share
of the pairs.

!!! tip "The library checks a rule when it runs, not when you store it"
    A [`HopCountAlgorithm`](@ref) must return an `Integer`, because the library uses `0:n` as a
    range of matrix powers. A [`PathLengthAlgorithm`](@ref) must return a `Number`. The return
    type of a callable is not part of its signature, so [`resolve_separation`](@ref) checks the
    value each time it calls the rule. Your own rule needs a type and one method. The third
    argument is the graph that the constraint built. Use it, and do not build another:

    ```julia
    struct AssetScaledHops <: PortfolioOptimisers.HopCountAlgorithm
        frac::Float64
    end
    function (r::AssetScaledHops)(nte, X, g; dims::Int = 1, kwargs...)
        return max(1, round(Int, r.frac * PortfolioOptimisers.Graphs.nv(g)))
    end
    ```

## 3. Phylogeny constraints

A [`SemiDefinitePhylogenyEstimator`](@ref) forbids the joint holding of two related assets. It
writes the constraint as a semidefinite relaxation, and the problem stays convex. We pass it
through `ple` and compare the minimum-risk weights with the baseline weights.
=#

res_phylo = optimise(MeanRisk(; obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                  ple = SemiDefinitePhylogenyEstimator(;
                                                                                       pl = NetworkEstimator()))))

pretty_table(DataFrame("Asset" => rd.nx, "Baseline" => res_base.w,
                       "Phylogeny" => res_phylo.w); formatters = [resfmt],
             title = "Minimum risk: baseline vs network-phylogeny constrained")

#=
The constraint moves weight between many assets, and section 3.1 prints the size of that move as
a turnover. [`IntegerPhylogenyEstimator`](@ref) sets a hard limit on how many assets you hold from
each neighbourhood of the network. A neighbourhood is an asset and the assets related to it, or a
cluster when the source is a clustering estimator. The constraint needs a mixed-integer solver,
because it uses binary variables. [Cardinality and threshold](03_Cardinality_and_Threshold.md) shows
how to set one up.

### 3.1 The separation sets the strength of the constraint

The estimator above uses the default `sep = HopCount(; n = 1)`. A wider separation makes the
constraint treat more assets as one bet. We solve the minimum-risk portfolio under six
separations. For each one we print the related pairs, the largest weight, the number of assets
held, and the turnover from the baseline.
=#

sep_sweep = ["HopCount(; n = 1)" => HopCount(; n = 1),
             "HopCount(; n = 3)" => HopCount(; n = 3),
             "PathLength(; dmax = 0.5)" => PathLength(; dmax = 0.5),
             "PathLength(; dmax = 1.5)" => PathLength(; dmax = 1.5),
             "PathLengthQuantile(; q = 0.25)" => PathLength(; dmax = q_rule),
             "PathLength()" => PathLength()]
res_sweep = [optimise(MeanRisk(; obj = MinimumRisk(),
                               opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                   ple = SemiDefinitePhylogenyEstimator(;
                                                                                        pl = NetworkEstimator(;
                                                                                                              sep = sep)))))
             for (_, sep) in sep_sweep]

pretty_table(DataFrame("Separation" => ["none (baseline)"; first.(sep_sweep)],
                       "Related pairs" => ["—"; string.(related_pairs.(last.(sep_sweep)))],
                       "Largest weight" =>
                           [maximum(res_base.w); [maximum(r.w) for r in res_sweep]],
                       "Names held" => [count(>(1e-4), res_base.w);
                                        [count(>(1e-4), r.w) for r in res_sweep]],
                       "Turnover vs baseline" =>
                           [0.0; [sum(abs, r.w .- res_base.w) for r in res_sweep]]);
             formatters = [(v, i, j) -> begin
                               return if j in (1, 2, 4)
                                   v
                               else
                                   "$(round(v*100, digits=2)) %"
                               end
                           end],
             title = "Minimum risk under a widening phylogeny separation")

#=
The table shows two facts to know before you tune `sep`.

First, a tighter phylogeny constraint concentrates the weights. A wider separation forbids more
pairs of holdings, and the optimiser holds fewer assets. The largest weight rises as the
constraint becomes tighter. To spread the weights as well, use `ple` together with an upper bound
on the weights or a [regularisation](07_Regularisation.md) term.

Second, the bare `PathLength()` row shows the cost of the setting in the warning of section 2.2.
The constraint forbids every reachable pair, and the only feasible portfolio holds one asset at
100 % weight. The result reports `OptimisationSuccess`. When a portfolio under a phylogeny
constraint holds a single asset, check `sep` first.

## 4. Centrality constraints

A centrality score measures how strongly an asset connects to the rest of the graph. A hub moves
with many other assets. An asset on the periphery of the network moves with few, and it helps to
diversify the portfolio. A [`CentralityEstimator`](@ref) gives every asset a score, and a
[`CentralityConstraint`](@ref) in `cte` bounds the weighted average score of the portfolio. Use
`comp = >=` and a floor to push the portfolio toward the hubs, or `comp = <=` and a ceiling to push
it toward the periphery. We solve one portfolio of each kind and print its average centrality next
to the baseline.
=#

res_hub = optimise(MeanRisk(; obj = MinimumRisk(),
                            opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                cte = CentralityConstraint(;
                                                                           A = CentralityEstimator(),
                                                                           B = 0.20,
                                                                           comp = >=))))
res_periph = optimise(MeanRisk(; obj = MinimumRisk(),
                               opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                   cte = CentralityConstraint(;
                                                                              A = CentralityEstimator(),
                                                                              B = 0.08,
                                                                              comp = <=))))

centrality = centrality_vector(CentralityEstimator(), pr).X
avg_centrality(w) = sum(w .* centrality)
pretty_table(DataFrame("Portfolio" =>
                           ["Baseline", "Hub-tilted (≥ 0.20)", "Periphery (≤ 0.08)"],
                       "Avg centrality" =>
                           [avg_centrality(res_base.w), avg_centrality(res_hub.w),
                            avg_centrality(res_periph.w)]);
             title = "Average network centrality of the portfolio")

#=
Compare the average of each tilted portfolio with its bound, the floor of the hub tilt and the
ceiling of the periphery tilt. A [`CentralityEstimator`](@ref) accepts different algorithms, for
example degree, eigenvector, closeness and betweenness centrality, and each one scores a different
kind of connection.

The algorithm also decides whether the score uses the edge weights of the network. Through
[`centrality_polarity`](@ref), each algorithm declares the polarity that its edge weights must
have, which is a distance or a similarity. The shortest-path measures need distances, and
[`EigenvectorCentrality`](@ref) needs similarities. The library builds the graph to match.

A clustering source always gives the unweighted graph. So do [`DegreeCentrality`](@ref), which is
the default and the algorithm used above, [`Pagerank`](@ref), [`KatzCentrality`](@ref), and
[`EigenvectorCentrality`](@ref) on a tree. None of these cases gives a warning, and the warning on
[`CentralityEstimator`](@ref) lists them. Section 4.2 shows [`TopologyOnly`](@ref), which removes a
declaration and asks for the unweighted graph.

### 4.1 Where `sep` changes the scores, and where it does not

An algorithm that uses the weights works on the graph itself, not on the related pairs that the
phylogeny constraints build from `sep`. For such an algorithm the `sep` of the network estimator
has no effect, and a wider `sep` leaves the scores where they were. An algorithm on the unweighted
graph responds to `sep`. For each of the eight algorithms we print its polarity, and whether its
scores change when `n` goes from 1 to 3.
=#

cts = ["BetweennessCentrality" => BetweennessCentrality(),
       "ClosenessCentrality" => ClosenessCentrality(),
       "DegreeCentrality" => DegreeCentrality(),
       "EigenvectorCentrality" => EigenvectorCentrality(),
       "KatzCentrality" => KatzCentrality(), "Pagerank" => Pagerank(),
       "RadialityCentrality" => RadialityCentrality(),
       "StressCentrality" => StressCentrality()]
function polarity_name(ct)
    p = centrality_polarity(ct)
    return isnothing(p) ? "none (unweighted)" : string(nameof(typeof(p)))
end
function sep_moves(ct)
    c1 = centrality_vector(CentralityEstimator(;
                                               pl = NetworkEstimator(;
                                                                     sep = HopCount(;
                                                                                    n = 1)),
                                               ct = ct), pr).X
    c3 = centrality_vector(CentralityEstimator(;
                                               pl = NetworkEstimator(;
                                                                     sep = HopCount(;
                                                                                    n = 3)),
                                               ct = ct), pr).X
    return maximum(abs, c3 .- c1) > 1e-8
end

pretty_table(DataFrame("Algorithm" => first.(cts), "Polarity" => polarity_name.(last.(cts)),
                       "n = 1 → n = 3 moves the score" =>
                           [sep_moves(ct) ? "yes" : "no" for (_, ct) in cts]);
             title = "Which centralities read the weights, and which read sep")

#=
On this minimum spanning tree the four algorithms with a distance polarity get a weighted graph
and ignore `sep`. The other four get the unweighted graph and respond to it.

The last column, not the polarity, tells you which graph an algorithm gets.
[`EigenvectorCentrality`](@ref) declares a similarity polarity, and it still gets the unweighted
graph here, because a tree has no similarity weights for it to use. The source and the algorithm
decide the graph together.

[`BetweennessCentrality`](@ref) and [`StressCentrality`](@ref) use the weights, but on a tree the
weights do not change their scores. A tree has exactly one path between any two nodes, so no
weighting can change the set of shortest paths. That is a property of a tree, not a limit of the
two algorithms, and it does not hold on a graph with cycles. The table in section 4.2 shows it.

### 4.2 Asking for the unweighted graph

A [`TopologyOnly`](@ref) in the `ov` field of an algorithm removes its declaration.
[`centrality_polarity`](@ref) then returns `nothing`, and the library builds the unweighted graph,
the same graph that [`DegreeCentrality`](@ref), [`Pagerank`](@ref) and [`KatzCentrality`](@ref)
use. We print the polarity of [`ClosenessCentrality`](@ref) without the override and with it.
=#

(centrality_polarity(ClosenessCentrality()),
 centrality_polarity(ClosenessCentrality(; ov = TopologyOnly())))

#=
Only the five algorithms that declare a polarity have the `ov` field. The other three use the
unweighted graph and have no declaration to remove, so `DegreeCentrality(; ov = TopologyOnly())`
throws a `MethodError`. The effect of the override depends on the source. We compare the scores
with and without the override on two sources. One is the default tree. The other is the
triangulated maximally filtered graph that
`NetworkEstimator(; alg = MaximumDistanceSimilarity())` builds, with the similarity that
[`MaximumDistanceSimilarity`](@ref) gives.
=#

ovs = Dict("BetweennessCentrality" => BetweennessCentrality(; ov = TopologyOnly()),
           "ClosenessCentrality" => ClosenessCentrality(; ov = TopologyOnly()),
           "EigenvectorCentrality" => EigenvectorCentrality(; ov = TopologyOnly()),
           "RadialityCentrality" => RadialityCentrality(; ov = TopologyOnly()),
           "StressCentrality" => StressCentrality(; ov = TopologyOnly()))
function ov_moves(nte, name, ct)
    if !(haskey(ovs, name))
        return "no `ov` field"
    end
    declared = centrality_vector(CentralityEstimator(; pl = nte, ct = ct), pr).X
    topology = centrality_vector(CentralityEstimator(; pl = nte, ct = ovs[name]), pr).X
    return maximum(abs, topology .- declared) > 1e-8 ? "yes" : "no"
end
tree_src = NetworkEstimator()
graph_src = NetworkEstimator(; alg = MaximumDistanceSimilarity())

pretty_table(DataFrame("Algorithm" => first.(cts),
                       "Tree source" => [ov_moves(tree_src, n, ct) for (n, ct) in cts],
                       "Graph source" => [ov_moves(graph_src, n, ct) for (n, ct) in cts]);
             title = "Does asking for the topology alone move the score?")

#=
A "yes" marks an algorithm whose score on that source depends on the edge weights. Compare the
count of "yes" on the tree with the four algorithms of section 4.1 that get a weighted graph. On a
tree, two of those four give the same score with and without the weights.

The override removes the weights and never adds them. No value of `ov` forces a polarity onto an
algorithm.

With `ov = TopologyOnly()`, the scores of these five algorithms depend on `sep`. That includes the
four that ignored `sep` in the table of section 4.1, because the library builds the unweighted
graph from the related pairs of `sep`.

A score from the unweighted graph does not change when the estimated weights change, so the
override can look like a way to make the scores more stable from fold to fold. But the override
replaces the weights with a second parameter, `sep`, and the scores depend on that choice. Under a
bare [`PathLength`](@ref) that parameter is the observed diameter, which also changes with the
data. No default uses the override. The default `ct` of [`CentralityEstimator`](@ref) is a
[`DegreeCentrality`](@ref), which uses the unweighted graph. The five algorithms that declare a
polarity keep using the weights of their source unless you ask otherwise.

## 5. Comparing the structural constraints

We plot the weights of six portfolios side by side. They are the baseline, the phylogeny
constraint at three settings of `sep`, and the two centrality tilts.
=#

results = [res_base, res_phylo, res_sweep[2], res_sweep[6], res_hub, res_periph]
labels = ["Baseline", "Phylo n=1", "Phylo n=3", "Phylo bare\nPathLength", "Hub",
          "Periphery"]

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))

#=
The three phylogeny bars show one constraint at three settings of `sep`. The bar of the bare
`PathLength()` is a single block, because a constraint that relates every pair leaves one asset in
the portfolio.
=#

#src ## Findings (authoring dogfooding — stripped from rendered docs)
#src - New deep dive (4_constraints_costs). Verified on kaimon (f102cae9), MinimumRisk base:
#src   - SemiDefinitePhylogenyEstimator(NetworkEstimator()) via `ple`: base maxw 37%→53%,
#src     Δw=0.658 — large structural reshaping. Convex (Clarabel ok).
#src   - CentralityConstraint via `cte`: binds both ways — base avg centrality 0.143; `>=0.20`
#src     lifts it to 0.20, `<=0.08` drops it to 0.08. centrality_vector(CentralityEstimator(),pr).X
#src     range 0.053..0.263 on this slice.
#src - §2.1/2.2/3.1/4.1 added 2026-08-06 for the separation dial (issue #246, map #195). Measured
#src   on this slice (20 assets, last 253 rows), session b0a0d44c:
#src   - phylogeny_matrix related-pair counts out of 380 off-diagonal entries. HopCount n=1..8:
#src     38, 96, 168, 230, 288, 340, 370, 380. PathLength dmax=0.25..3.5: 4, 34, 50, 94, 132,
#src     178, 248, 314, 362, 380. The radius genuinely interleaves (50 between 38 and 96; 132
#src     and 178 straddle 168) and reaches BELOW the tightest hop shell (4 < 38) — that is the
#src     concrete answer to "what does the radius ball buy".
#src   - Observed diameter 3.4743 distance units / 8 hops; closest linked pair 0.2253.
#src     separation_budget(PathLength(; dmax = 100), ...) clamps to 3.4743.
#src   - THE TRAP, PRICED: ple with sep=PathLength() bare → maxw 100 %, ONE name held, Δw=1.2604,
#src     and retcode is OptimisationSuccess. Compare HopCount(n=1) 53 %/10 names,
#src     HopCount(n=3) 74 %/5, PathLength(dmax=1.5) 75 %/6.
#src   - COUNTER-INTUITIVE and now documented: tightening `ple` CONCENTRATES the book (maxw
#src     37 %→53 %→74 %→100 %). Structural diversification ≠ weight diversification.
#src   - sep-inertness table: 4 distance-polarity algs (Betweenness, Closeness, Radiality,
#src     Stress) do NOT move n=1→n=3; the other 4 (Degree, Eigenvector, Katz, Pagerank) DO.
#src     Eigenvector declares SimilarityPolarity yet sits on the unweighted side on a tree — so
#src     the polarity column does NOT predict the inertness column. Said so explicitly.
#src   - NOT reproducible from the public API: the "weighted moves 4 of 8 defaults" claim needs
#src     the pre-weighting answer, and there is no user switch for an unweighted graph. Stated in
#src     prose as a re-measure warning instead of faked with an internal call.
#src - §4.2 added 2026-08-10 for the polarity override (issue #259, map #252). THE 4-OF-8 CLAIM
#src   ABOVE WAS WRONG and is now deleted. #257 re-measured over 7 windows x 9 universes x 7
#src   distance estimators x 7 network algorithms (2608 cells): the count of centralities whose
#src   answer changes between the weighted and unweighted graph is 2 on EVERY tree cell and 5 on
#src   EVERY graph cell. Four NEVER occurs. The four is the sep split in §4.1's table — algorithms
#src   taking a weighted ROUTE, of which only two ANSWER differently on a tree, because betweenness
#src   and stress are invariant there by theorem. Taking a weighted route != the answer moving.
#src   §4.2's table is the public-API demonstration that #246 could not write: TopologyOnly shipped
#src   in #258, so `ov` makes the unweighted answer reachable without an internal calc_centrality
#src   call. Measured on this slice: tree yes = Closeness, Radiality; graph yes = all five.
#src   ALSO MEASURED, and stated in prose: all five become sep-LIVE under ov = TopologyOnly(),
#src   because the override routes them through phylogeny_matrix.
#src   The ADR 0048 Consequences bullet carried the same wrong four and its enumeration also
#src   missed EigenvectorCentrality on the similarity branch — corrected in an appended amendment,
#src   not rewritten.
#src - SCOPE: the two knobs (sep vs Proximity.decay) and the data-dependent dmax hazard on the
#src   proximity field are named here in one paragraph but NOT worked — they are worked in
#src   `3_optimisers/16_Feature_Distance_Clustering.jl` §4, the producers' own page (map #802).
#src - FINDING (→ group issue): SemiDefinitePhylogenyEstimator `p` is INERT for MinimumRisk here —
#src   p=0.0 and p=5.0 give byte-identical weights; the SDP coupling drives the result, not the
#src   penalty p. Did NOT author a p-sweep (would be a flat, misleading table). Worth checking
#src   whether p matters for penalty-style objectives / documenting its actual role.
#src - GOTCHA: result.retcode does NOT == OptimisationSuccess via `==` (retcode is a struct that
#src   only displays "OptimisationSuccess"); verify success behaviourally (sum(w)≈1, constraint
#src   binds) instead. Reused across this group's pages.
#src - IntegerPhylogenyEstimator is MIP (test_18k uses mip_slv) — mentioned, not run here.
