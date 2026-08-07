The source files can be found in [examples/](https://github.com/dcelisgarza/PortfolioOptimisers.jl/tree/main/examples/).

```@meta
EditURL = "../../../../examples/4_constraints_costs/04_Phylogeny_Centrality.jl"
```

# Phylogeny and centrality constraints

The constraints in [Linear and group constraints](02_Linear_Group_Constraints.md) act on names
and hand-drawn groups. **Phylogeny** and **centrality** constraints act on the *structure* of
the asset network instead — the graph of how assets co-move. Rather than telling the optimiser
"tech ≤ 30%", you tell it "don't pile into a tightly-knit cluster" or "tilt toward (away from)
the hubs of the correlation network". The groups are discovered from the data, not declared.

`PortfolioOptimisers.jl` builds the network with a [`NetworkEstimator`](@ref) (or a clustering
estimator) and then exposes two families:

- **Phylogeny constraints** ([`SemiDefinitePhylogenyEstimator`](@ref),
    [`IntegerPhylogenyEstimator`](@ref)) via the `ple` keyword — limit joint exposure to
    network-linked assets.
- **Centrality constraints** ([`CentralityConstraint`](@ref) built from a
    [`CentralityEstimator`](@ref)) via the `cte` keyword — bound the portfolio's average
    network centrality.

!!! tip "When to reach for this"
    Reach for these when your diversification concern is *structural* rather than by label: you
    do not want a book that looks diversified by sector but is actually one big correlated bet,
    or you want to deliberately tilt toward stable hubs or peripheral diversifiers. They need no
    hand-built groups — the structure comes from the covariance. The semidefinite phylogeny and
    centrality forms are convex; the integer phylogeny form needs a MIP solver.

Both families are driven by one dial you have to set deliberately: how far apart two assets may
sit in the network and still count as related. That is the `sep` field of a
[`NetworkEstimator`](@ref), covered in §2.1, and one of its settings quietly collapses the
portfolio onto a single name — see the warning there.

````@example 04_Phylogeny_Centrality
using PortfolioOptimisers, CSV, TimeSeries, DataFrames, PrettyTables, Clarabel, StatsPlots,
      GraphRecipes

resfmt = (v, i, j) -> begin
    return if j == 1
        v
    else
        isa(v, AbstractFloat) ? "$(round(v*100, digits=3)) %" : v
    end
end;
nothing #hide
````

## 1. ReturnsResult data

````@example 04_Phylogeny_Centrality
X = TimeArray(CSV.File(joinpath(@__DIR__, "..", "SP500.csv.gz")); timestamp = :Date)[(end - 252):end]
rd = prices_to_returns(X)
pr = prior(EmpiricalPrior(), rd)

slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
             settings = Dict("verbose" => false),
             check_sol = (; allow_local = true, allow_almost = true))

res_base = optimise(MeanRisk(; obj = MinimumRisk(),
                             opt = JuMPOptimiser(; pe = pr, slv = slv)))
````

## 2. The asset network

A [`NetworkEstimator`](@ref) turns the covariance into a graph: assets are nodes, and edges link
assets whose returns are connected after filtering out the noisy links (a minimum-spanning-tree
or similar backbone). Both constraint families below read this graph. You do not have to build it
by hand — the estimators take a `NetworkEstimator()` and construct it from the prior internally.

### 2.1 How far apart still counts as related

The graph only says which assets are *directly* linked. Every constraint below needs a second
answer: how far apart two assets may sit and still count as related. That is the `sep` field of
the [`NetworkEstimator`](@ref), and it is the single dial controlling how much of the universe
each constraint sees as one bet.

Two separations ship, and they measure the same structure in different units:

- [`HopCount`](@ref) counts **edges**, ignoring their lengths, with a budget of `n` of them.
    `HopCount(; n = 1)` is the default and means "directly linked only".
- [`PathLength`](@ref) adds up the **distances** along the shortest path, with a budget `dmax`
    in those same units.

The two are interchangeable — every consumer takes either — but their numbers are *not*
comparable, because the budgets are in different units. Choose whichever unit you can reason
about, then read the cardinality it produces rather than trusting the budget to feel tight.

````@example 04_Phylogeny_Centrality
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
````

### 2.2 Hop shells are coarse; a radius ball fills in between them

Read the table by its ordering rather than row by row. The hop budgets give a ladder of eight
rungs and nothing between them — on this universe `38`, `96`, `168` pairs and so on, because a
whole shell of neighbours joins at once. The `PathLength` rows slot into the gaps: `50` pairs sits
between the first and second hop shells, `132` and `178` straddle the third, and `4` pairs is
tighter than the tightest hop budget can express.

That is the whole of what the radius buys. It is the **same** notion of neighbourhood at a finer
granularity, not a different one — the two ladders agree closely on which pairs are related, they
just cannot stop at the same places. Reach for `PathLength` when a hop shell overshoots the
concentration you are willing to allow.

````@example 04_Phylogeny_Centrality
plot(dmax_budgets, dmax_pairs; label = "PathLength (radius ball)", marker = :circle,
     xlabel = "Budget: dmax, in distance units", ylabel = "Pairs called related",
     title = "A continuous radius against eight discrete hop shells", legend = :bottomright)
hline!(hop_pairs; label = "HopCount shells (n = 1…8)", linestyle = :dash, color = :grey,
       linealpha = 0.7)
````

The largest budget either family can usefully take is the **diameter** of the graph — the longest
shortest path in it. [`separation_matrix`](@ref) and [`separation_budget`](@ref) expose both
halves of that, and are what a consumer calls internally:

````@example 04_Phylogeny_Centrality
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
````

A `dmax` above the diameter is clamped to it, so an over-large budget cannot select more than the
whole component — which is exactly the hazard below.

!!! warning "`PathLength()` with no `dmax` relates everything"

    `dmax = nothing` is [`PathLength`](@ref)'s default and means *the whole connected component*,
    implemented as the observed diameter above. Read by a constraint, "the whole component" means
    **every reachable pair is related**, so `NetworkEstimator(; sep = PathLength())` yields a
    phylogeny matrix of ones off the diagonal — the last row of the ladder table, at 100 % of
    pairs. It is the opposite end of the dial from `HopCount()`'s default `n = 1`, reached by
    swapping the separation and changing nothing else, and it is deliberately unguarded: it
    optimises successfully and returns a single-asset portfolio (§3.1). State a numeric `dmax` to
    select anything narrower.

The same `sep` is also read by [`PhylogenyFeatures`](@ref), which builds a feature matrix rather
than a constraint. There the budget *shapes* a fall-off instead of selecting pairs — a second knob,
[`Proximity`](@ref)'s `decay`, says how strongly — so the bare default is the natural choice on
that path and the trap above on this one. The two knobs live on two different objects and neither
follows the other: setting `sep` does not imply a `decay`, and setting `decay` does not imply a
`sep`.

## 3. Phylogeny constraints

A [`SemiDefinitePhylogenyEstimator`](@ref) adds a semidefinite constraint that discourages
holding assets which are neighbours in the network — concentrated, mutually-correlated bets.
Passing it through `ple` reshapes the minimum-risk portfolio toward combinations that are
diversified in *network* terms, not just in count.

````@example 04_Phylogeny_Centrality
res_phylo = optimise(MeanRisk(; obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                  ple = SemiDefinitePhylogenyEstimator(;
                                                                                       pl = NetworkEstimator()))))

pretty_table(DataFrame("Asset" => rd.nx, "Baseline" => res_base.w,
                       "Phylogeny" => res_phylo.w); formatters = [resfmt],
             title = "Minimum risk: baseline vs network-phylogeny constrained")
````

The constraint moves a large fraction of the book — it is enforcing genuine structural
diversification, not a cosmetic tweak. For a *hard* limit on the number of names drawn from each
network cluster, [`IntegerPhylogenyEstimator`](@ref) imposes an integer (cardinality-style)
version; being combinatorial it needs a MIP solver (see
[Budget Constraints](01_Budget_Constraints.md) for the Pajarito/HiGHS setup).

### 3.1 The separation is the strength dial

`ple = SemiDefinitePhylogenyEstimator(; pl = NetworkEstimator())` above took the default
`sep = HopCount(; n = 1)`. Widening the separation widens what the constraint treats as one bet,
and the ladder of §2.1 becomes a ladder of portfolios:

````@example 04_Phylogeny_Centrality
sep_sweep = ["HopCount(; n = 1)" => HopCount(; n = 1),
             "HopCount(; n = 3)" => HopCount(; n = 3),
             "PathLength(; dmax = 0.5)" => PathLength(; dmax = 0.5),
             "PathLength(; dmax = 1.5)" => PathLength(; dmax = 1.5),
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
````

Two things are worth reading off that table before you tune `sep`.

**Structural diversification is not weight diversification.** A wider separation forbids more
joint holdings, so the optimiser is pushed out of clusters and into fewer, unrelated names — the
largest weight *rises* as the constraint tightens. If you want both, pair `ple` with a weight
upper bound or a [regularisation](07_Regularisation.md) term.

**The bare `PathLength()` row is the trap of §2.2, priced.** Every reachable pair is forbidden
from being held jointly, so the only feasible book is a single asset, at 100 % weight. It reports
`OptimisationSuccess` — nothing raises, and nothing warns. When a phylogeny-constrained portfolio
collapses onto one name, check `sep` first.

## 4. Centrality constraints

Centrality measures how *central* each asset is in the network — a hub that co-moves with many
others, versus a periphery name that diversifies. A [`CentralityEstimator`](@ref) scores every
asset, and a [`CentralityConstraint`](@ref) bounds the portfolio's weighted-average centrality
through `cte`. You can push the book toward hubs (`comp = >=`, a higher floor) or toward the
periphery (`comp = <=`, a lower ceiling).

````@example 04_Phylogeny_Centrality
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
````

The constraint binds in both directions — the hub tilt lifts the average centrality to its floor,
the periphery tilt drops it to its ceiling. Centrality is not one number: a
[`CentralityEstimator`](@ref) accepts different algorithms (degree, eigenvector, closeness,
betweenness, …), each emphasising a different notion of "central", so the right one depends on
what kind of connectedness you care about.

The algorithm also decides whether the network's *edge weights* are read. Each one declares the
polarity its weights must have through [`centrality_polarity`](@ref) — distances for the
shortest-path measures, similarities for [`EigenvectorCentrality`](@ref) — and the graph is built
to match. Five cases run on the plain unweighted graph and none of them raises: a clustering
source, [`DegreeCentrality`](@ref) (the default used above), [`Pagerank`](@ref),
[`KatzCentrality`](@ref), and [`EigenvectorCentrality`](@ref) on a tree branch. The warning on
[`CentralityEstimator`](@ref) has the details.

### 4.1 Where `sep` bites, and where it is inert

Reading the weights has a consequence that is easy to trip over. A weighted route reads the
*structure* of the graph, not the separation closure the phylogeny constraints build, so on a
weighted route the network estimator's `sep` is **inert** — you can widen it and the scores will
not move at all. On an unweighted route `sep` is live. Which of the two you are on is decided by
the algorithm, not by anything you write next to it:

````@example 04_Phylogeny_Centrality
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

pretty_table(DataFrame("Algorithm" => first.(cts),
                       "Declared polarity" => polarity_name.(last.(cts)),
                       "n = 1 → n = 3 moves the score" =>
                           [sep_moves(ct) ? "yes" : "no" for (_, ct) in cts]);
             title = "Which centralities read the weights, and which read sep")
````

On this minimum-spanning-tree source the split is four and four: the four distance-polarity
algorithms get a weighted graph and ignore `sep`, and the other four run unweighted and respond to
it. So raising `n` moves a degree centrality and leaves a closeness one exactly where it was.

The column to read is the last one, not the polarity. [`EigenvectorCentrality`](@ref) *declares* a
similarity polarity and still lands on the unweighted side here, because a tree carries no
similarity for it to read — so the declaration alone does not tell you which side you are on. The
source decides that jointly with the algorithm.

Two further subtleties are worth stating rather than demonstrating.

- [`BetweennessCentrality`](@ref) and [`StressCentrality`](@ref) do read the weights, and are
    nonetheless unchanged by them *on a tree*: a tree has exactly one path between any two
    vertices, so no weighting can change the shortest-path set. That is a theorem about the graph,
    not a limitation of the algorithm, and it does not hold on a similarity branch.
- Reading the weights at all is recent, and it moved the default answer for **four of the
    eight** algorithms. A centrality number carried over from an older run will not always
    reproduce, so re-measure rather than reusing a bound you calibrated earlier.

## 5. Comparing the structural constraints

````@example 04_Phylogeny_Centrality
results = [res_base, res_phylo, res_sweep[2], res_sweep[5], res_hub, res_periph]
labels = ["Baseline", "Phylo n=1", "Phylo n=3", "Phylo bare\nPathLength", "Hub",
          "Periphery"]

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))
````

The two phylogeny bars and the bare-`PathLength` bar are the same constraint at three settings of
one dial. The last one is a single block: that is what "relate everything" looks like as a book.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*
