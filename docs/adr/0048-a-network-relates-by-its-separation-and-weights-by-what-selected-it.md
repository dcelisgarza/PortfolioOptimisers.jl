---
status: accepted
---

# A network relates by its separation, and weights by what selected it

## Context

A `NetworkEstimator` builds a graph over the asset universe — a minimum spanning tree, or a
triangulated maximally filtered graph — and every downstream consumer reads that graph. Before this
change all of them read it through two hardcoded choices.

**The first was a bare hop budget.** `NetworkEstimator` carried an `n::Integer`, and each consumer
spelled the same neighbourhood out of it: `phylogeny_matrix` clamped `sum(A^i for i in 0:n)`, both
`clusterise` methods accumulated `∑(Dⁱ - Aⁱ)`, and the feature producer `GradedNeighbourhood` scored a
pair `n + 1 - hops`. The fall-off was not a knob: the linear expression was written into the
producer, and its flat twin needed a *type* of its own, `BinaryNeighbourhood`. Nothing between two
hop shells was expressible — on a twenty-asset triangulated graph the knob steps 54, then 121, then
165 of the 190 pairs, and a caller wanting about 100 could not ask.

**The second was binarisation.** `calc_adjacency` built the graph, then threw away the weights that
had selected its edges — `sparse(Int.(· .!= 0))` — so a centrality algorithm defined over shortest
paths ran on a graph where every edge had length one. `Graphs.jl` weights implicitly (`distmx`
defaults to `weights(g)`), so the capability was one construction away, and half of it already
existed as an unguarded escape hatch: `ClosenessCentrality(; args = (D,))` reached
`Graphs.closeness_centrality`'s second positional slot and worked, while the same call on
`BetweennessCentrality` bound a matrix to `vs` and overflowed the stack.

Issue [#189](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/189) asked for three
things: a decay family, a weighted adjacency beside the binary one, and a weighted graded feature
variant. Chartered as map
[#195](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/195), it grew a fourth
obligation — a resolved answer, including a "no", from *every* consumer that reads a graph.

## Decision

### 1. Truncation and decay are two knobs, and only one of them cuts

`AbstractSeparationDecayAlgorithm` in
[`src/11_Phylogeny/01_Base_Phylogeny.jl`](../../src/11_Phylogeny/01_Base_Phylogeny.jl) turns a
separation into a score, applied by `separation_decay(dk, d, dmax)`. Members: `LinearDecay` (the
default, `dmax + 1 - d`), `ExponentialDecay(; rate)`, `ReciprocalDecay(; power)`, `NoDecay`.

**No member carries a top-of-scale.** The budget arrives as the kernel's third argument and members
may ignore it — only `LinearDecay` reads it. A free `f(0)` below `dmax + 1` goes negative and needs
back the `max(0, ⋅)` floor this design exists to avoid, which is a *second* truncation hiding inside
the decay; above it, rows flatten toward all-ones, where `AngularDist` degenerates.
`ExponentialDecay`'s `rate` buys the same self-versus-neighbour contrast without either hazard.

This is why an exponential is admissible at all: it never reaches zero, so a reach expressed as a
fall-off would have no reach. Truncation stays with the separation's budget; the decay only shapes
the values inside it. `NoDecay` is the sharpest case and the one most easily misread — the budget
still cuts, so `Proximity(; decay = NoDecay())` is the neighbourhood *indicator* the retired
`BinaryNeighbourhood` produced, not a matrix of ones.

**The argument is a real separation, not an integer hop count.** One family therefore serves an
unweighted graph, a weighted path length, and anything else whose separation is continuous — which
is what let the graded producer take both separations without a second decay family.

**The contract is enforced, not merely documented.** `f` is defined for all `d >= 0`, `f(0) > 0` and
maximal, monotone non-increasing, never assumed to reach zero, and `f(d) >= 0` for `0 <= d <= dmax`.
A probing `assert_separation_decay` fallback on the abstract type checks it once before the
`assets²` loop; the four shipped members satisfy it by construction and override the fallback to a
no-op. The probe is therefore **opt-out**: an extension that says nothing about itself gets probed.

Two clauses are less obvious than they look:

- **Maximality at `f(0)` is load-bearing.** The producer's diagonal is the decay at `d = 0`, and a
  decay that does not put an asset at the top of its own scale silently yields a *structural
  equivalence* matrix instead of a proximity one — the two non-adjacent endpoints of a three-node
  path come out identical.
- **Non-negativity is producer-local and budget-scoped.** `0` is the unreachable sentinel, so a
  negative score *inside* the budget puts a reachable pair below an unreachable one. It is not a
  claim that signed scores are wrong in general — the feature matrix is signed-tolerant by decision
  (`AngularDist` admits signed features). And it cannot be widened past the budget, because
  `LinearDecay` itself crosses zero at `d = dmax + 1`. The probe evaluates `f(dmax)` whether or not
  `dmax` appears in the probed range: monotonicity is already promised, so that single endpoint
  closes non-negativity over a *continuum*, which a unit-spaced sample never could.

### 2. A separation is one type answering two questions

`AbstractSeparationAlgorithm` carries both *how far apart* two assets sit and *how far is too far*,
through two kernels: `separation_matrix(sep, nte, X)` and `separation_budget(sep, nte, d)`. Members
are `HopCount(; n = 1)` — the shortest path counted in edges — and `PathLength(; dmax = nothing)` —
the distance summed along it.

**The two questions are one type because they share a unit.** A budget stated apart from the rule
that measures it has no interpretation, and becomes a dead field the moment a member measures
something else. That is why the estimator's `n` moved onto `HopCount` rather than staying beside a
new `sep` field.

**The family sits on `NetworkEstimator`, not on any consumer.** Every consumer of a graph needs to
know which pairs it relates, and the phylogeny constraint path receives nothing but the estimator —
so a rule living on the feature producer would be structurally invisible to it.

**`HopCount.n` stays an `Integer`.** Three readers use `0:(nte.sep.n)` as a *matrix-power count*,
where `0:1.5` silently drops a power instead of failing. Those three admit `HopCount` alone and
refuse a `PathLength` loudly, at dispatch: a radius has no analogue of a matrix power.

**`separation_budget`'s third argument is the separation matrix, not a scalar diameter.** Finding the
largest finite entry *is* an `assets²` reduction, so passing a diameter would charge `HopCount` for
one it ignores. Handing over the matrix pushes the reduction into the member that wants it.

**`dmax = nothing` means the observed diameter, and a chosen `dmax` is clamped to it.** The default
is data-dependent by design: nobody has an intuition for a summed path in an
`AbstractDistanceEstimator`'s units, whereas "look at the whole component and let the decay do the
falling off" is statable. Choosing a number is how a caller buys fold-stability. The clamp truncates
nothing — no pair sits beyond the diameter — so it is a scale-top correction visible only through
`LinearDecay`.

**The unreachable sentinel is passed through unrepaired**, and the comparison against the budget must
short-circuit. Both sentinels punish an unguarded read, for opposite reasons: `ReciprocalDecay`
overflows `1 + d` at `typemax(Int)`, which a fractional `power` turns into a `DomainError`, while
`LinearDecay` at `Inf` returns `-Inf`, which sorts every unreachable pair *below* every reachable one
and raises nothing. The second is why `separation_budget` excludes the sentinel from the diameter
rather than tolerating it.

### 3. Each branch keeps the quantity that selected it

The network is a `SimpleWeightedGraphs.SimpleWeightedGraph`, built at exactly one site, and
**neither branch is re-weighted**: `calc_mst` minimises the distance, so a tree's weights are
distances; `PMFG_T2s` maximises gain over the similarity, so a triangulated graph's weights are
similarities. Re-weighting either with the other quantity would weight a structure by the quantity
that did not select it. The result carries **no polarity tag and no result type** — the polarity is
recoverable by dispatch on `nte.alg`.

The construction in [`src/11_Phylogeny/06_Phylogeny.jl`](../../src/11_Phylogeny/06_Phylogeny.jl) is a
strict chain:

| name                            | what it is                                          |
|:------------------------------- |:--------------------------------------------------- |
| `calc_weighted_adjacency_graph` | the one construction site; per-branch polarity      |
| `calc_weighted_adjacency`       | `Graphs.adjacency_matrix` of it                     |
| `calc_adjacency`                | the round trip through `SimpleGraph` that binarises |
| `calc_distance_weighted_graph`  | the same structure, distances on **both** branches  |

`calc_adjacency` keeps its signature, its return type and its extension-point role, and **loses its
two-branch body**: the branch is decided one tier down. The refactor is subtractive.

**The first three take two entry points at one arity each.** The two-argument form takes *the
selecting quantity itself* — `D` under an `AbstractTreeType`, `S` under an
`AbstractSimilarityMatrixAlgorithm` — and the `(nte, X)` form derives it and forwards. This exists
because a caller may already hold that matrix: `clusterise` does, and re-deriving it there costs
**98% of `clusterise`'s runtime** under `VariationInfoDistance`. The split also makes the tier's
contract literal rather than prose — "the quantity that selected its edges" is now the argument.

**`calc_distance_weighted_graph` is a fourth name, not a fifth tier.** A shortest path over
similarities *minimises total similarity* and prefers the route through the weakest links, so
`PathLength` and every distance-polarity centrality need the structure re-weighted by `D`. That is
legitimate where re-weighting in general is not: every similarity algorithm is a strictly decreasing
function of `D`, so `D` is the selecting quantity's **monotone preimage**, not a foreign quantity —
measured, the three usable transforms give the *identical* triangulated graph. The backwards answer
correlates `0.95` to `0.97` with the right one, so nothing about it looks wrong.

### 4. A zero distance is repaired, not rejected

A distance matrix and a weighted graph disagree about `0`: in the distance codomain it is the floor,
while `SimpleWeightedGraph` reserves it for *absent* and refuses a zero-weight edge. So
`graph_weight_matrix` moves each off-diagonal zero to `nextfloat(zero(·))` before the graph is built.

**Rejection would have been wrong, because the guard cannot tell the routes apart.**
`SimpleAbsoluteDistance` and `LogDistance` are defined on `abs(rho)`, so an exactly anti-correlated
pair — a long/short leg, an inverse ETF — sits at distance zero and is *genuinely maximally related*;
the square-root algorithms reach zero from the other side when `clamp!` maps `rho >= 1` down. Left
unrepaired, the constructor deletes precisely the edge the tree most wants: measured over sixty
universes with one duplicated column, 36 produced an exact `0.0` and all 36 declared that pair
unrelated. The cost was a **spec violation**, not an aesthetic one — `clusterise` (reading `D`) put
the pair in the same cluster at the joint minimum, while `phylogeny_matrix` (reading only the graph)
declared them unrelated, so the phylogeny constraints left unbounded the one pair they exist to
separate.

`eps()` was measured and rejected: a legitimate `LogDistance` between a near-duplicate pair is
`4.44e-15`, only twenty times `eps()`, so an `eps` nudge would outrank a real distance.
`nextfloat(0.0)` has an infinite margin and is absorbed exactly by any sum it enters. Negative and
`NaN` entries are **rejected** instead — they have no nearest representable value, and both are
unsound rather than merely wrong downstream. `Inf` is left alone: it is the honest `LogDistance`
between uncorrelated assets, and a spanning tree takes those edges last.

No `is_connected` check was added. A dense `D` with no zeros is complete, so the repair closes the
disconnection route it would have guarded — a check that provably cannot fire is dead code.

### 5. Every consumer answered, and the noes are the load-bearing half

| consumer                           | reads the weights? | resolution                                                             |
|:---------------------------------- |:------------------ |:---------------------------------------------------------------------- |
| `phylogeny_matrix` — **values**    | no                 | `PhylogenyResult` stays `Int`                                          |
| `phylogeny_matrix` — **selection** | yes                | new radius ball, `weighted_dist <= dmax`, still binary                 |
| `SemiDefinitePhylogeny`            | no                 | weight-inert: `A ⊙ W == 0` is the same constraint at any magnitude     |
| `IntegerPhylogeny`                 | no                 | *broken* by a number — `B` is an integer cardinality                   |
| both `clusterise` methods          | no                 | matrix-power consumer; folded onto the shared routine, `HopCount` only |
| `Proximity`                        | yes                | takes either separation through the same contract                      |
| centrality                         | yes                | by **declared polarity**, per algorithm                                |

Three of these deserve their reasoning recorded.

**No weighted `PhylogenyResult`.** Every surviving consumer of one refuses a number, so a weighted
result would be a value nothing could read. The graded reading of a separation lives on `Proximity`
instead. The radius ball is therefore a change of *selection*, not of values — and it needed the
`sep` relocation to be reachable at all, since `phylogeny_matrix` receives only the estimator.

**The radius ball barely re-ranks, and that is stated in the docstrings.** Against the
equal-cardinality prefix of the path-length ordering, a hop shell is identical at every shell on a
triangulated graph (0 of 54, 121, 165, 186) and differs by 1, 1, 3 and 2 pairs on a tree. Both
structures are selected by distance already, so a path length **refines** a hop count rather than
rivalling it. What it buys is the intermediate cardinalities: sweeping `dmax` over the same graph
reaches 36, 55, 100, 122, 151, 179. A reader who takes it for a conceptually different neighbourhood
would be wrong, so the documentation says so with the numbers.

**Centrality reads weights by declared polarity, and the polarity belongs to the algorithm.**
`centrality_polarity` declares `DistancePolarity` for the four shortest-path algorithms,
`SimilarityPolarity` for `EigenvectorCentrality`, and `nothing` for the three `Graphs.jl` cannot
weight. It is a fact about the algorithm's mathematics, not about the branch: on one and the same
triangulated graph, closeness wants `D` and eigenvector wants `S`. That killed the alternative of a
weighted *carrier* — the required weights are per-algorithm, so no single weighted object could
serve both. The graph is built at `centrality_graph`, the one site where the source and the algorithm
are both in scope.

**Polarity never decides whether the call succeeds.** Weightedness is a property of the **source**,
not of the request: there is no flag, so a caller names an algorithm and never asks for weights, and
an unweightable pairing has not been handed a request it cannot serve. Five cases therefore run on
the plain graph and none of them raises — any weightless source, `DegreeCentrality`, `Pagerank`,
`KatzCentrality`, and `EigenvectorCentrality` on a tree branch. What is traded away is an early
signal, not a computation: every improvement measured sits on a pair that still reads weights.

**The `args` passthrough is closed by refusal.** `assert_centrality_args` rejects an `AbstractMatrix`
anywhere in `args` on the three types that carry one. It was a second, undeclared weighting channel
answering the same question as the declared one, and it was never safe in either direction.

### 6. The split is a field, not a type, so no `Weighted*` noun was minted

`GradedNeighbourhood` is renamed **`Proximity`** and holds `decay` alone; `BinaryNeighbourhood` is
retired, being `Proximity(; decay = NoDecay())`. The hop-versus-path choice is `NetworkEstimator.sep`
and not a second feature-algorithm type, so the naming problem dissolved instead of being solved.

The seam that fixes where each field lives: `sep` decides **which pairs are related**, which every
consumer of a graph needs, while `decay` decides **how strongly, as a number**, which only the
feature producer wants. Moving `decay` onto the estimator would make it a dead field on four
consumers of five; leaving `sep` on the producer would hide it from the constraint path entirely.

## Consequences

- **Breaking, four times.** `NetworkEstimator.n` is *removed* in favour of `sep`
  (`NetworkEstimator(; sep = HopCount(; n = 2))`); `GradedNeighbourhood` and `BinaryNeighbourhood`
  are gone; a matrix in a centrality algorithm's `args` now throws a `ConflictingArgumentError`; and
  the default centrality answer **moves for four of eight algorithms** — closeness and radiality on
  both branches, betweenness and stress on the similarity branch. Two golden columns were
  regenerated to record that; the other six were checked and left alone. No deprecation shims: `src`
  has never carried an `@deprecate`.
- **A silent numerical change is documented rather than guarded, in four places.** `PathLength()`
  bare is the *maximal* ball — every reachable pair, measured 190 of 190 — which is benign through
  `Proximity`, where the decay does the falling off, and a trap through `phylogeny_matrix`, which
  *selects*: `SemiDefinitePhylogenyEstimator(; pl = NetworkEstimator(; sep = PathLength()))` forbids
  all pairwise co-movement. It is the opposite end of the dial from `HopCount()`'s default `n = 1`.
  A complete ball is the honest reading of an unstated budget, and nothing distinguishes it from a
  forgotten one, so prose is the only place this can live.
- **`NetworkEstimator.sep` is inert on the weighted centrality routes and live on the unweighted
  ones.** They read the structure itself, because a closure is a sum of matrix powers and a power of
  a weighted matrix sums *products* of distances. So `HopCount(; n = 2)` moves a `DegreeCentrality`
  and leaves a `ClosenessCentrality` where it was. Invisible at the default `n = 1`.
- **Two invariances are theorems, and are asserted as such.** Betweenness and stress are exactly
  unchanged by weights on a tree — a tree has one path between any two vertices, so the
  shortest-path set is weight-independent — which is why they are *allowed* rather than refused:
  failing for a reason intrinsic to the graph is categorically unlike failing for a reason intrinsic
  to the algorithm, and a provably-equal answer is not a lie. Both are asserted beside the
  algorithms that do move, so the invariance is evidence the weights arrived.
- **`Proximity`'s `f(0)` is data-dependent under `PathLength()`'s default budget**, and the
  dependence is *additive*: a diameter that moves across cross-validation folds shifts every row of
  `Z` rather than rescaling it. Recorded as a live hazard in ADR
  [0045](0045-a-feature-matrix-is-data-not-estimator-configuration.md)'s transform question, not
  solved here. A fixed `dmax` buys back the stability; the decays that pin `f(0) = 1` never had the
  exposure.
- **Two docstring claims were found wrong en route and corrected.** `calc_adjacency` promised
  `Matrix{Int}` while returning a `SparseMatrixCSC`, and `LogDistance` was floored at zero at four
  sites — both pre-existing, neither introduced here.
- **`Graphs.eigenvector_centrality` is not deterministic**, differing from itself by about `6e-16`
  between two runs on one and the same graph, so its regression test compares within a tolerance
  rather than exactly.
- **Left open.** The `(nte, X)` entry point of `calc_weighted_adjacency` still has no consumer
  outside the tests, and the chain now stands at four names at two arities each — whether that is the
  right shape is a live question rather than a settled one. Also open: a similarity-polarity
  separation member (a widest path, or a max-product path through `-log`), which is what would
  un-refuse `EigenvectorCentrality` on the tree branch; and whether the decay and separation families
  are offered beyond the phylogeny producers — a taxonomy depth is a separation, which is why the
  family name was left unqualified and its kernels take `(sep, nte, ·)`.
- **Ruled out of scope.** `AngularSimilarity`'s `PMFG_T2s` failure is pre-existing and has nothing to
  do with weights — filed as [#239](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/239)
  and chartered as its own map,
  [#241](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/241). Adopting
  `Graphs.planar_maximally_filtered_graph` (new in Graphs 1.14) is a separate effort: `PMFG_T2s` also
  returns the triangle and clique structure DBHT consumes.
