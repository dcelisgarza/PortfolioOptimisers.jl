#=
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
## 1. ReturnsResult data
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
=#

plot(dmax_budgets, dmax_pairs; label = "PathLength (radius ball)", marker = :circle,
     xlabel = "Budget: dmax, in distance units", ylabel = "Pairs called related",
     title = "A continuous radius against eight discrete hop shells", legend = :bottomright)
hline!(hop_pairs; label = "HopCount shells (n = 1…8)", linestyle = :dash, color = :grey,
       linealpha = 0.7)

#=
The largest budget either family can usefully take is the **diameter** of the graph — the longest
shortest path in it. [`separation_matrix`](@ref) and [`separation_budget`](@ref) expose both
halves of that, and are what a consumer calls internally:
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
=#

res_phylo = optimise(MeanRisk(; obj = MinimumRisk(),
                              opt = JuMPOptimiser(; pe = pr, slv = slv,
                                                  ple = SemiDefinitePhylogenyEstimator(;
                                                                                       pl = NetworkEstimator()))))

pretty_table(DataFrame("Asset" => rd.nx, "Baseline" => res_base.w,
                       "Phylogeny" => res_phylo.w); formatters = [resfmt],
             title = "Minimum risk: baseline vs network-phylogeny constrained")

#=
The constraint moves a large fraction of the book — it is enforcing genuine structural
diversification, not a cosmetic tweak. For a *hard* limit on the number of names drawn from each
network cluster, [`IntegerPhylogenyEstimator`](@ref) imposes an integer (cardinality-style)
version; being combinatorial it needs a MIP solver (see
[Budget Constraints](01_Budget_Constraints.md) for the Pajarito/HiGHS setup).

### 3.1 The separation is the strength dial

`ple = SemiDefinitePhylogenyEstimator(; pl = NetworkEstimator())` above took the default
`sep = HopCount(; n = 1)`. Widening the separation widens what the constraint treats as one bet,
and the ladder of §2.1 becomes a ladder of portfolios:
=#

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

#=
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
[`CentralityEstimator`](@ref) has the details. §4.2 covers the one thing you can say back to it:
[`TopologyOnly`](@ref), which withdraws a declaration and asks for the topology alone.

### 4.1 Where `sep` bites, and where it is inert

Reading the weights has a consequence that is easy to trip over. A weighted route reads the
*structure* of the graph, not the separation closure the phylogeny constraints build, so on a
weighted route the network estimator's `sep` is **inert** — you can widen it and the scores will
not move at all. On an unweighted route `sep` is live. Which of the two you are on is decided by
the algorithm, not by anything you write next to it:
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
On this minimum-spanning-tree source the split is four and four: the four distance-polarity
algorithms get a weighted graph and ignore `sep`, and the other four run unweighted and respond to
it. So raising `n` moves a degree centrality and leaves a closeness one exactly where it was.

The column to read is the last one, not the polarity. [`EigenvectorCentrality`](@ref) declares a
similarity polarity and still lands on the unweighted side here, because a tree carries no
similarity for it to read — so the declaration alone does not tell you which side you are on. The
source decides that jointly with the algorithm.

[`BetweennessCentrality`](@ref) and [`StressCentrality`](@ref) are a second reason not to read the
polarity column as the answer. They do read the weights, and are nonetheless unchanged by them *on
a tree*: a tree has exactly one path between any two vertices, so no weighting can change the
shortest-path set. That is a theorem about the graph, not a limitation of the algorithm, and it
does not hold on a similarity branch.

### 4.2 Asking for the topology alone

Everything above is decided *for* you, by the algorithm's mathematics and by the branch the source
builds. There is one thing you can say back. A [`TopologyOnly`](@ref) in an algorithm's `ov` field
withdraws its declaration, so [`centrality_polarity`](@ref) answers `nothing` and the graph is
built plain — the same computation the three unweighted algorithms already run:
=#

(centrality_polarity(ClosenessCentrality()),
 centrality_polarity(ClosenessCentrality(; ov = TopologyOnly())))

#=
Only the five algorithms that declare a polarity carry the field. [`DegreeCentrality`](@ref),
[`Pagerank`](@ref) and [`KatzCentrality`](@ref) already read the topology alone, so they have
nothing to withdraw and `DegreeCentrality(; ov = TopologyOnly())` is a `MethodError`. How much the
request changes depends on the source it is made against:
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
Two of the eight move on the tree, five of the eight on the triangulated maximally filtered graph.
That gap is the whole difference between the weighted and the unweighted answer, and it is
remarkably stable — measured across seven windows, nine universes, seven distance estimators and
seven network algorithms, it is two on every tree and five on every graph. **Four** is a different
quantity: it is the split in the first table, and it counts the algorithms that take a weighted
*route*, not the ones whose answer *moves*. The two coincide only on a graph.

The request runs one way only. It removes the weights and never supplies them, and there is no
value that forces a polarity onto an algorithm. Forcing one would succeed rather than raise — the
distance-weighted graph is available on both branches — and the algorithm would read a distance
where it needs a similarity, reversing its own ordering in silence.

The override also puts `sep` back in play. All five of these algorithms respond to `n = 1` versus
`n = 3` once they carry `ov = TopologyOnly()`, including the four that were `sep`-inert in the
first table, because the unweighted route is the one that reads the separation closure.

That is worth knowing before you reach for it as a stabiliser. A topology-only centrality is
sometimes argued to be the more fold-stable of the two, since it does not move when the estimated
weights do. It trades them for a second knob rather than removing one, and under a bare
[`PathLength`](@ref) that knob is the observed diameter — the data-dependent quantity the argument
set out to avoid. Nothing here defaults to it: [`CentralityEstimator`](@ref)'s `ct` is a
[`DegreeCentrality`](@ref), which reads the topology already, and the five that declare a polarity
keep reading the weights their source carries unless you say otherwise.

## 5. Comparing the structural constraints
=#

results = [res_base, res_phylo, res_sweep[2], res_sweep[5], res_hub, res_periph]
labels = ["Baseline", "Phylo n=1", "Phylo n=3", "Phylo bare\nPathLength", "Hub",
          "Periphery"]

plot_stacked_bar_composition(results, rd; xticks = (1:length(labels), labels))

#=
The two phylogeny bars and the bare-`PathLength` bar are the same constraint at three settings of
one dial. The last one is a single block: that is what "relate everything" looks like as a book.
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
#src - SCOPE: the two knobs (sep vs Proximity.decay) and the data-dependent dmax hazard on `Z` are
#src   named here in one paragraph but NOT worked — PhylogenyFeatures has no page, and its page is
#src   #185's (map #160, all four feature-matrix producers). Requirement recorded on #185.
#src - FINDING (→ group issue): SemiDefinitePhylogenyEstimator `p` is INERT for MinimumRisk here —
#src   p=0.0 and p=5.0 give byte-identical weights; the SDP coupling drives the result, not the
#src   penalty p. Did NOT author a p-sweep (would be a flat, misleading table). Worth checking
#src   whether p matters for penalty-style objectives / documenting its actual role.
#src - GOTCHA: result.retcode does NOT == OptimisationSuccess via `==` (retcode is a struct that
#src   only displays "OptimisationSuccess"); verify success behaviourally (sum(w)≈1, constraint
#src   binds) instead. Reused across this group's pages.
#src - IntegerPhylogenyEstimator is MIP (test_18k uses mip_slv) — mentioned, not run here.
