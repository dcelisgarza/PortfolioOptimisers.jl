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

| consumer                           | reads the weights? | resolution                                                                            |
|:---------------------------------- |:------------------ |:------------------------------------------------------------------------------------- |
| `phylogeny_matrix` — **values**    | no                 | `PhylogenyResult` stays `Int`                                                         |
| `phylogeny_matrix` — **selection** | yes                | new radius ball, `weighted_dist <= dmax`, still binary                                |
| `SemiDefinitePhylogeny`            | no                 | weight-inert: `A ⊙ W == 0` is the same constraint at any magnitude                    |
| `IntegerPhylogeny`                 | no                 | *broken* by a number — `B` is an integer cardinality                                  |
| both `clusterise` methods          | yes                | read them before this map too; folded onto `calc_weighted_adjacency`, `HopCount` only |
| `Proximity`                        | yes                | takes either separation through the same contract                                     |
| centrality                         | yes                | by **declared polarity**, per algorithm                                               |

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

## Amendment (2026-08-10): the `clusterise` row said "no", and both methods read the weights

Section 5's table shipped with a wrong cell. The row for `clusterise` answered **no** to "reads the
weights?". Both methods read them, and did so before this map as well.

The cell is corrected in place rather than by a note here, because it was never true. The
amend-never-rewrite convention protects *superseded decisions* — an ADR describing behaviour that
has since changed is correct history. This was a transcription slip about code that never behaved
the way the cell claimed, and the table is where a reader looks first.

### What the code does

Both methods enter the weighted chain at its middle tier, `calc_weighted_adjacency`, and accumulate
against it. The tree method:

```julia
A = calc_weighted_adjacency(nte.nte.alg, D)
for i in 0:(nte.nte.sep.n)
    P .+= D^i - A^i
end
```

The triangulated method is the same shape with the branch's own selecting quantity:

```julia
Rpm = calc_weighted_adjacency(nte.nte.alg, S)
for i in 0:(nte.nte.sep.n)
    P .+= S^i - Rpm^i
end
```

`calc_weighted_adjacency` carries the branch's selecting quantity as edge values, so both sums are
weighted minus weighted. Nothing binarises.

### Why the cell was wrong rather than stale

Three records in place when this ADR was written say the opposite:

 1. Section 3 of this ADR, on the chain: "a caller may already hold that matrix: `clusterise` does,
    and re-deriving it there costs **98% of `clusterise`'s runtime** under `VariationInfoDistance`."
    That two-argument entry point exists *for* `clusterise`.
 2. [#199](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/199) corrected the map's
    charting-time ground truth to "neither `clusterise` method binarises", measured. The pre-map
    belief that weights were discarded in two places was already retired.
 3. [#207](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/207) shipped both methods
    onto the two-argument entry point and verified `P` bit-for-bit against the deleted bodies.

The cell's own justification argues the fold and the `HopCount` narrowing. Those are the answers to
"what changed here?" and "which separations are admitted?", not to "does it read the weights?" —
which is how a right answer to one question came to sit in the column for another.

### Consequence

The load-bearing noes are three, not four: `phylogeny_matrix`'s **values**,
`SemiDefinitePhylogeny`, and `IntegerPhylogeny`. All three refuse a *number*, which is the argument
against a weighted `PhylogenyResult` and is untouched by this correction. `clusterise` was never
part of that argument — it consumes a weighted matrix and always has.

Found by [#209](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/209), the final
verification of map [#195](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/195), and
filed as [#262](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/262).

## Amendment (2026-08-10): the tree-branch eigenvector item names a refusal that no longer exists

The "Left open" list says that a similarity-polarity separation member "is what would **un-refuse**
`EigenvectorCentrality` on the tree branch". Two words there are stale, and one is wrong.

**There is no refusal to undo.** The polarity bullet three items above already records the settled
position: five cases run on the plain graph and none of them raises, and
`EigenvectorCentrality` on a tree branch is the fifth. The refusal was withdrawn before this ADR
shipped, on the ground that weightedness is a property of the source rather than of the request, so
a caller who names an algorithm has not asked for weights and cannot be told no. What the open item
is really about is **weighting** that pairing, not permitting it.

**A separation member is the wrong mechanism for it.** `sep` selects which pairs count as related,
and this ADR records that it is **inert** on every weighted centrality route, because those routes
read the structure rather than the separation closure. A widest-path or max-product separation would
change which pairs `phylogeny_matrix` relates; it would not put a similarity on a tree's edges.

**What weighting that pairing actually needs is a similarity for the tree branch**, and the tree
branch names no similarity algorithm at all — `nte.alg` is an `AbstractTreeType` there. ADR
[0049](0049-a-similarity-reaches-the-pmfg-only-if-it-cannot-go-negative.md) does not supply one: it
narrows three existing similarity fields and adds none. It does remove the obstacle that the
original refusal cited, `AngularSimilarity` throwing inside `PMFG_T2s`, and it removes it by
**excluding** that member, which is not a route to a tree-branch similarity either.

The item stays open, restated: **whether the tree branch should name a similarity, and which one.**
The arithmetic permits it. The argument this ADR uses to legitimise `calc_distance_weighted_graph`
— re-weighting a PMFG with `D` is legitimate because every similarity algorithm is a strictly
decreasing function of `D`, so `D` is the selecting quantity's monotone preimage — is **symmetric**.
A tree re-weighted with `S = f(D)` is the selecting quantity's monotone image by the same reasoning.
So this is a design decision about which member a tree would name and where that field lives, not an
arithmetic obstacle and not something an interface unblocks.

Found by [#245](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/245) while discharging
map [#241](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/241)'s report to map
[#195](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/195).

## Amendment (2026-08-10): a caller may decline the weights, and the moved-answer figure was wrong

Map [#252](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/252) asked whether a caller
may override an algorithm's declared polarity to get the centrality over the network's topology
alone. The answer is yes, and it ships as `TopologyOnly` in an `ov` field on the five algorithms
that declare a polarity. This amendment records what that does to two claims of this ADR: one is
**limited**, and one was **wrong when written**.

### The rule is limited, not contradicted

"A network relates by its separation, and weights by what selected it" is untouched. The override
never supplies weights and never names a quantity. It withdraws the declaration, so
`centrality_polarity` answers `nothing` and `centrality_graph` takes the plain-graph route it
already builds for `DegreeCentrality`, `Pagerank` and `KatzCentrality`. Nothing new is weighted by
anything, so the selecting-quantity rule has no new case to serve.

What is limited is the sentence in section 5 that reads: *there is no flag, so a caller names an
algorithm and never asks for weights*. A caller now names a **configured** algorithm, and there is
exactly one request to make. Its codomain is closed at two states and it runs **one way**:

- **Away from weights is a request, and every source honours it.** The topology-only answer is what
  a partition source, a precomputed `PhylogenyResult` and the tree branch under `SimilarityPolarity`
  already compute, so the request is satisfied before it is made. It adds no case to the five that
  run on the plain graph, and no warning.
- **Toward weights is not offered at all.** Forcing a polarity onto an algorithm would **succeed**
  rather than raise — `calc_distance_weighted_graph` carries distances on both branches — and the
  algorithm would read a distance where it needs a similarity, reversing its own ordering in
  silence. Polarity correctness is not a runtime property, so nothing could catch it. The field is
  therefore not typed over `AbstractCentralityPolarity`, not even bounded to one member of it.

So this ADR's *reason* stands and its *scope* narrows: weightedness is still not the caller's to
assert, because the one thing a caller may say is "do not weight it", and that is a request no
source can fail. The rejected spelling is a `Bool`. `unweighted::Bool` makes a claim about the
graph, which this ADR says is not the caller's to make; a named request asks what to read, which
is. Same two states, different referent.

The five that carry `ov` are `BetweennessCentrality`, `ClosenessCentrality`, `StressCentrality`,
`RadialityCentrality` and `EigenvectorCentrality`. The other three carry no field, so
`DegreeCentrality(; ov = TopologyOnly())` is a `MethodError` with no check written. Capability is
type-level, which is why the refusal is free. `CentralityEstimator` gains nothing and stays a pure
bundle: `ct` is positional on every public surface, so the override reaches all of them with zero
signature changes.

### The moved-answer figure was wrong, and it counted a different thing

The Consequences say that the default centrality answer **moves for four of eight algorithms** —
"closeness and radiality on both branches, betweenness and stress on the similarity branch". That
enumeration is missing `EigenvectorCentrality`, which also moves on the similarity branch, and the
count is not four on either branch.

[#257](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/257) re-measured it over 7
windows, 9 universes, 7 distance estimators and 7 network algorithms — 2608 measured cells. The
count of algorithms whose answer changes between the weighted and the unweighted graph is **2 on
every tree cell and 5 on every graph cell**. Not one cut moves it, and **four never occurs**.

The four is real and counts the **`sep` split**: four algorithms take a weighted route on a tree
(betweenness, closeness, radiality, stress) and are therefore `sep`-inert. Only two of them answer
differently for it, because betweenness and stress are invariant on a tree by the theorem this ADR
already asserts. **Taking a weighted route is not the same as the answer moving**, and the two
coincide only on a graph, where both are five. The golden columns this ADR regenerated are
unaffected: they record values, not a count.

Found by [#257](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/257) and
[#259](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/259) while working map
[#252](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/252).

## Amendment (2026-08-11): a budget may be a rule, resolved by a third kernel

Decision 2 gives the separation **two** kernels, and says the budget lives on the member because it
is the only place that has a unit to state it in. Both still hold. What was missing is that a
caller sometimes has the unit and still cannot state the number: a cross-validation fold and a
subproblem of a meta optimiser each refit the graph, so a `dmax` tuned once is applied to graphs it
was never tuned for. That is a real cost of this ADR's own design — the budget is configuration,
and configuration is fixed before the data arrives.

[#248](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/248) widens both budget fields
to admit a **rule** in place of a number, and adds `resolve_separation` as a third kernel.

### The rule is a deferred number, not a second notion of a budget

`HopCount.n` takes a `HopCountAlgorithm` and `PathLength.dmax` takes a `PathLengthAlgorithm`, each
a callable struct invoked as `rule(nte, X; dims, kwargs...)`; a bare `Function` is admitted in
either field under the same obligation. `resolve_separation(sep, nte, X)` calls it and rebuilds the
member around the answer, so everything downstream sees an ordinary `HopCount{<:Integer}` or
`PathLength{<:Number}` and no reader changed.

Two families rather than one, because the return obligations differ and the split is what lets one
of them be enforced. A hop count must be an `Integer` — this ADR's own reason, that three readers
index a matrix power by it — and a path length may be any `Number`. `nothing` is deliberately
outside `PathLengthValue` and reached only through the `Option`: it means the observed diameter,
which is a budget the caller *states*, not one a rule computes.

**The check is at run time and cannot be anywhere else.** A functor's return type is not part of
its signature, so the `Integer` obligation is unstatable in the type system. `resolve_separation`
checks the value and then feeds it back through the ordinary constructor, so a rule's answer meets
exactly the validation a stated budget meets.

### Why the kernel is third rather than folded into `separation_budget`

Decision 2 makes `separation_budget`'s third argument the separation **matrix**, so that `HopCount`
never pays for a diameter reduction it ignores. That choice is what forbids resolving there: a rule
needs `X`, and `X` is the one thing that kernel does not have. So `separation_budget` **refuses** an
unresolved member rather than returning a function, and the consumer resolves first —
`phylogeny_matrix`, both `clusterise` methods, and `phylogeny_features` for `Proximity`.

The fallback on `AbstractSeparationAlgorithm` is an identity, so an extension inherits the kernel
and a stated budget costs nothing. The family's extension contract is still two methods, not three.

### What a rule buys, and what it does not

It changes **which quantity stays put**, and nothing more. A stated `dmax` holds the *radius* still
and lets the related-pair count move with the graph; `PathLengthQuantile(; q)` holds the *count*
still and lets the radius move. Neither is fold-stable in both senses, because the graph is
refitted either way — the same shape as the amendment above, where an override trades one exposure
for another rather than removing one.

Measured over four 63-day folds of one year on twenty assets: a `dmax` fixed at the whole-sample
quarter-quantile relates `84`, `110`, `96` and `110` of the `380` pairs, while
`PathLengthQuantile(; q = 0.25)` relates `96` in every fold and moves the radius between `1.2055`
and `0.8574`.

**The two quantile rules are not equally good at it, and the unit is why.** `q` is continuous and a
hop count is an integer, so `HopCountQuantile` rounds to a shell: on the same graph `q = 0.1`,
`0.2` and `0.25` all resolve to `n = 2`. `PathLengthQuantile` lands on `q` to within a pair. This is
the one place where Decision 2's "not comparable as values, only interchangeable" has a practical
consequence a caller can act on — the radius ball's intermediate cardinalities, which the
Consequences record as its whole gain, become reachable **by name**.

## Amendment (2026-08-17): the tree family carries the same two splat fields, and there every channel is a weight

Decision 5 closed the `args` passthrough on the three centrality types that carry one. It stopped
there, and the ninth security pass read the remainder as one gap. It is two, they are not the same
gap, and only one of them is real.

**`kwargs` never needed a guard, on any family.** A keyword binds by *name*, so a matrix in `kwargs`
cannot reach the positional slot `assert_centrality_args` was written for. None of
`betweenness_centrality`, `closeness_centrality`, `stress_centrality` or `_degree_centrality`
declares a matrix-valued keyword, and `normalize`, `endpoints`, `rng` and `seed` each refuse one on
their own. Probed live, all five pairings fail closed with a `MethodError` or a `TypeError`. The
report's reasoning — that the `args` rationale "applies verbatim to `kwargs`" — does not hold, and
no guard was added there.

**The tree family is the real gap, and it is worse than the one already closed.** `KruskalTree`,
`BoruvkaTree` and `PrimTree` carry the identical `args`/`kwargs` pair, and unlike the centrality
case the calls **succeed**. Three channels, all silent:

| Channel | Binds to | Effect |
| --- | --- | --- |
| `AbstractMatrix` in `args` | `distmx`, the **second** positional of all three | replaces the estimator's distances |
| `AbstractVector` in `args` | `kruskal_mst`'s `weight_vector` | the same override in another shape |
| `minimize` in `kwargs` | `kruskal_mst`/`boruvka_mst`'s `minimize` | a **maximum** spanning tree |

`calc_weighted_adjacency_graph` hands `calc_mst` a graph already weighted by `de` and `ce`, and
`Graphs.jl` defaults `distmx` to exactly those weights, so a filled `args` answers a question that
was already answered. Measured on eight assets over three hundred observations, a bogus matrix in
`KruskalTree.args` moved `20` of the `64` entries of the phylogeny matrix while leaving the edge
count at `14` — nothing downstream can notice. Only a *wrong-sized* matrix fails, with a
`BoundsError`. That makes this a silent wrong result where Decision 5's case was a crash, so it is
the more serious of the two and was rated the other way round.

**The refusal is therefore wider than the centrality one, and the asymmetry is deliberate.**
`assert_tree_args` refuses a matrix *or a vector* in `args`, and `minimize` in `kwargs`. Centrality
keeps its narrower rule because `vs` and `k` are genuine non-weight positionals there; the three
spanning-tree functions declare **no** positional argument that is not a weight, so nothing
legitimate is being turned away. Both guards now share one kernel,
`assert_no_weight_channel_args`, which differs only in the shape it rejects and the declared channel
it names.

- **Not breaking in practice.** No caller in `src`, `test`, `examples` or `user_guide` fills either
  field on a tree type, so the refusal removes no working configuration. It is breaking in
  principle, on the same footing as Decision 5's.
- **`minimize` is refused rather than honoured.** A maximum spanning tree is a coherent object, but
  it is not the one this branch is defined to build, and the branch's identity is read by
  `SimilarityPolarity` and by `calc_weighted_adjacency_graph`'s docstring. Making it reachable is a
  separate decision about the *estimator*, not a keyword to leave open on the algorithm.
- **The stack-overflow claim in Decision 5 was overstated.** Reproduced in a fresh subprocess,
  `betweenness_centrality(g, M)` raises a catchable `StackOverflowError` and the process survives.
  The docstring said it "takes the session with it"; it takes the call. The guard is unaffected —
  the second-channel argument was always the load-bearing one.

## Amendment (2026-08-17): the kernels take the structure, and the sentinel test belongs to the family

Two candidates of the 2026-08-16 architecture review, D and E, are about the same seam from opposite
ends: what the separation kernels are handed, and what a consumer does with an entry once it has
one. Both are answered here, because both change the family's extension contract.

### D. The kernels take the structure, not the producer that makes one

Decision 2 gave the separation two kernels and the amendment above added a third, and every one of
them took `(sep, nte, X)` — a producer plus the data — although a separation's behaviour depends on
the **structure** alone. Each kernel therefore derived its own, privately. That is fine for one
kernel and wrong for two: `phylogeny_matrix` and `phylogeny_features` call `resolve_separation` and
then `separation_matrix`, so a budget **rule** made them derive the same structure twice.

`clusterise` was the sharpest case. Decision 3's `calc_weighted_adjacency_graph` has a two-argument
entry point precisely so that `clusterise` — which already holds `D` and `S` — does not re-derive
the correlation, `98%` of its runtime under `VariationInfoDistance`. With a rule in the budget field
the second full derivation happened anyway, inside the very function that avoided it.

**`separation_graph` is now the family's first kernel, and the measuring kernels take its output.**

| kernel | interface | wrapper |
| :------ | :--------- | :------- |
| `separation_graph` | `(sep, nte, X)`, plus `(sep, G)` for `HopCount` | — |
| `separation_matrix` | `(sep, g)` | `(sep, nte, X)` |
| `resolve_separation` | `(sep, nte, X, g)` | `(sep, nte, X)` |
| `separation_budget` | `(sep, nte, d)` | — |

- **A consumer builds one structure per call** and hands it to both readers. All four do:
  `phylogeny_matrix`, `phylogeny_features`, and both `clusterise` methods, the last through
  `separation_graph(sep, G)` off the graph they already build from the selecting quantity.
- **The rule contract gains the graph**: `rule(nte, X, g; dims, kwargs...)`. This is **breaking** for
  an extension rule and for a bare `Function` in a budget field, and it is the point — a rule that
  wanted the structure had to build one, and the contract said so in as many words. `nte` and `X`
  stay, inert for what ships, as the channel to what a graph does not carry.
- **`resolve_separation` keeps a wrapper per case, not one generic wrapper.** The resolved case must
  build *nothing*, so the identity fallback is reached without touching `separation_graph`; the
  rule-carrying parameterisations get the wrapper that builds.
- **Only `HopCount` takes a bare graph.** A graph carries no polarity tag — Decision 3 — so `G` is
  distance-weighted on the tree branch and similarity-weighted on the PMFG branch, and a shortest
  path over similarities returns an answer instead of raising. The hop count is exempt because it
  discards the weights, and it is handed a **binarised** graph because `_phylogeny_matrix`'s power
  sum reads `adjacency_matrix` off it, where a weight would sum products of distances.
- **`separation_budget` is unchanged.** It already took the matrix rather than the data, which is
  the reason it cannot resolve a rule, and that reason is untouched.
- **`calc_weighted_adjacency` gains the same one-argument entry point**, `(G)`, because
  `clusterise` now keeps the graph it builds and reads the matrix off it. Without that method the
  read would be a bare `Graphs.adjacency_matrix` call and the polarity contract — *weights, not
  `0`/`1`* — would sit one function away from the site that depends on it. It also means the
  function keeps a caller in `src/`: the report's note that `calc_weighted_adjacency(nte, X)` has
  none is now true of the two-argument form as well, and whether that surface should shrink is a
  separate decision about public API.

The extension contract is therefore **three** methods where Decision 2 said two:
`separation_graph`, `separation_matrix`, `separation_budget`. `resolve_separation` and the two
predicates below still have working fallbacks, so an extension still writes none of them.

**Measured.** A `CountingDistance` fixture counts derivations per consumer call. Over both
branches and all four budgets — stated hops, `HopCountQuantile`, unstated radius,
`PathLengthQuantile` — `phylogeny_matrix` and `phylogeny_features` derive **once**, and so does
`clusterise` with a rule. Before, the three rule cases derived twice.

**Two of three test doubles are gone, not three.** `FixedAdjacency` and `FixedWeights` in
`test_12d_phylogeny_features.jl` subtyped `AbstractNetworkEstimator` and overrode `calc_adjacency`
and `calc_distance_weighted_graph` only to choose a graph; they are now two helpers and a
`SimpleWeightedGraph`. `FixedDistanceGraph` in `test_13_phylogeny.jl` **stays**: two
centrality testsets drive `centrality_vector` and `phylogeny_matrix` through it, and those verbs
take an estimator, so no graph argument can replace it. The review's count was one too high.

`_proximity_features` is split out of `phylogeny_features` for the same reason — the scoring loop is
a function of the separations alone, so the unreachable branch is now driven by handing it a matrix
rather than by an estimator that lies about its graph.

### E. The sentinel test belongs to the family

The rule *not the sentinel, and no further than the budget* was written out at four sites in four
spellings, and two of them disagreed:

| site | spelling |
| :---- | :-------- |
| `separation_budget`'s diameter reduction | `isfinite(dij)` |
| `separation_quantile`'s population | `isfinite(d) && d != typemax(T)` |
| `_phylogeny_matrix`'s radius ball | `d .<= dmax` |
| `phylogeny_features`'s loop | `duv <= dmax` |

`isfinite` is `true` for every `Integer`, so the first admits `HopCount`'s `typemax(Int)`. It was
**latent, not live**: `separation_budget`'s `PathLength` method is the only caller and a `PathLength`
matrix carries `Inf`, so the two travel together today. The file already said which spelling was
wrong, forty-six lines below the wrong one.

`is_reachable(sep, d)` and `is_related(sep, d, dmax)` are now the two spellings, both single generic
methods on `AbstractSeparationAlgorithm`:

- **One body covers both sentinels.** `isfinite(d) && d != typemax(typeof(d))` — the second clause
  catches `typemax(Int)`, and it *is* `Inf` for a `Float64`, so the first is left only to reject a
  `NaN` no shipped path produces.
- **Reachability is tested before the budget**, so the predicate is right independently of what the
  budget happens to be. `d <= dmax` alone rejects both shipped sentinels only because a `PathLength`
  budget is clamped to the observed finite diameter and a hop budget is capped far below
  `typemax(Int)`; that is a property of the budgets, not of the comparison.
- **A generic method rather than one per member.** A member whose routine reports an exotic sentinel
  overrides `is_reachable` alone and inherits `is_related`.
- **The predicate owns the rule; the call site still owns the laziness.** `phylogeny_features` keeps
  its short-circuiting `?:`, because what must not be evaluated is the *decay* —
  `ReciprocalDecay` overflows `1 + d` at `typemax(Int)`, and a fractional `power` turns that into a
  `DomainError`. A predicate cannot make a caller lazy; it can only stop the caller from spelling
  the test itself.

`separation_quantile` gains the separation as its first argument, since it is the population's
sentinel test that it needed. Both predicates are unexported, like `assert_separation_decay` and
like every graph builder in this family; the api pages document them under their qualified names.
