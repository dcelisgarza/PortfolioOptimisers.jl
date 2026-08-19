---
status: accepted
---

# A similarity reaches the PMFG only if it cannot go negative

## Context

`NetworkEstimator(; alg = AngularSimilarity())` threw. The error came from
[`PMFG_T2s`](../../src/11_Phylogeny/04_DBHT.jl)'s own non-negativity check, one transformation
after the mistake was made, and it named `W` rather than the configuration that produced it. That
was [#239](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/239).

The blast radius is wider than the report. Four call sites hand `distance_to_similarity` output
straight to `PMFG_T2s`: [`calc_adjacency`](../../src/11_Phylogeny/06_Phylogeny.jl),
[`clusterise`](../../src/11_Phylogeny/06_Phylogeny.jl),
[`DBHTs`](../../src/11_Phylogeny/04_DBHT.jl) and `logo!`. None of them guarded the sign, so
`DBHT(; sim = AngularSimilarity())` and `LoGo(; sim = AngularSimilarity())` failed the same way.
Three estimators, one defect.

The negative number is a symptom. The disease is a **mismatched pairing**.
[`AngularSimilarity`](../../src/09_Distance/04_Similarity.jl) is `cos(pi * D)`, the honest inverse
of an angular distance and of nothing else. [`SimpleDistance`](../../src/09_Distance/02_Distance.jl)
is `sqrt((1 - rho) / 2)`, and it shares `AngularDist`'s `[0, 1]` range exactly. The two are
**indistinguishable by range**, so `cos(pi * D)` type-checks for both while being correct for only
one. Under `SimpleDistance` the similarity turns negative wherever `rho < 0.5`, not where
`rho < 0`. The measured `D = 0.706` in #239 is a correlation of `0.003`, and `cos(pi * 0.706)` is
`-0.618`.

[`default_similarity`](../../src/09_Distance/04_Similarity.jl) already pairs a metric with its
inverse, and [`FeatureDistance`](../../src/09_Distance/05_FeatureDistance.jl) uses it.
`NetworkEstimator.alg`, `DBHT.sim` and `LoGo.sim` took any member, with no reference to the distance
estimator that produced `D`.

No distance in this library can be negative. All four `_dist_from_cor` branches `clamp!` into
`[0, 1]`, the two `LogDistance` ones are `max(-log(.), 0)`, and `AngularDist` is `arccos(rho)/pi`.
The negative quantity is only ever the **similarity**.

## Decision

**A similarity matrix algorithm reaches the PMFG path if and only if it declares that it cannot
return a negative entry, and the declaration is a type bound rather than a runtime check.**

`AbstractNonNegativeSimilarityMatrixAlgorithm` is a new abstract subtype of
`AbstractSimilarityMatrixAlgorithm`. `Tree_SimMat`, `DBHT.sim` and `LoGo.sim` are bounded by it, so
`NetworkEstimator(; alg = AngularSimilarity())` is a `MethodError` at **construction**. Four of the
five shipped members join; `AngularSimilarity` does not.

Both are **unexported**, per the repository convention that abstract types stay unexported unless
asked for. An extension names either through the module prefix, and both keep their docstring and
their `docs/src/api/09_Distance/04_Similarity.md` entry, so unexported is not undocumented.
`test/test_43_exported_abstract_type_census.jl` gates the convention.

### The requirement is DBHT's, not the PMFG's, and the bound is deliberately wider

`PMFG_T2s` does not need a non-negative weight arithmetically. Its gain argmax compares sums of
exactly three weights, and its seed strength `sum(W ⊙ (W .> mean(W)))` is shift-invariant. Both run
on signed input and pick the least bad edge.

The requirement comes from **downstream DBHT**, at two of its three weight aggregations. Both
failures are **silent**: the caller gets wrong clusters and no error.

- **`DirectHb` (`04_DBHT.jl:942-948`) is unsafe.** `left` and `right` are unnormalised sums over
  different-sized vertex sets, so a large bubble accumulates negative mass and loses for a reason
  unrelated to attachment. The winner is written into `Hc` as a weight, and
  `Sep = iszero.(sum(Hc; dims = 2))` — so cancelling signs sum to an exact zero and **manufacture a
  separating bubble**.
- **`BubbleMember` (`04_DBHT.jl:1084-1086`) is unsafe.** `all_bub = diag(Mv' * Rpm * Mv) / 2` is a
  total **weight**, not a count. Signed cancellation drives the denominator near zero (`Inf`), to
  exactly zero (`NaN`), or negative — in which case `argmax` picks the **worst** bubble.
- **`BubbleCluster8s` (`04_DBHT.jl:1022-1024`) is safe.** `all_cont = 3 * (size - 2)` is an edge
  **count**, so the ratio is a mean and stays comparable across bubble sizes.

Neither unsafe aggregation can see a negative in the shipped library, and that is worth stating
plainly. `DBHTs` is the only caller of the bubble machinery, its `Rpm` is `PMFG_T2s(S)[1]`, and
`PMFG_T2s` refuses a negative and a `NaN` before it returns. So `PMFG_T2s`'s own check is what makes
the aggregations safe, and the type bound with `assert_similarity_domain` buys the **message**. That
is the same division of labour as *Open by declaration, not by proof* below, seen from the other end.
`BubbleMember` does fail on zeros alone — an all-zero bubble makes `0/0`, and `argmax` selects the
`NaN` — which is one more reason the zero is refused in the next section.

The type bound binds **all four** `PMFG_T2s` callers, which is wider than that provenance.
`calc_adjacency` binarises the graph, `clusterise` takes matrix powers of it, and `logo!` reads
separators and cliques — structure only. None of them reaches the bubble machinery.

They are bound anyway. Binding DBHT alone would be more honest about provenance, but it means
relaxing the one guard that presently makes the other three safe by accident, and ADR
[0048](0048-a-network-relates-by-its-separation-and-weights-by-what-selected-it.md) independently
established that matrix powers are incoherent on weights — a live reason to distrust `clusterise`
with signed input specifically. Widening the three non-DBHT callers back out is a later effort with
its own destination.

A future reader who finds only the type bound will reasonably conclude that a PMFG needs
non-negative weights. It does not. That is why this section exists.

### Non-negative reaches the check, positive reaches the graph

`exp(-Inf)` is `0` exactly, and that route is live: `LogDistance` maps an exactly zero correlation
to an infinite distance. A zero similarity is an admissible **value**, so `PMFG_T2s`'s input check
stays `>= 0` and the interface keeps the name **non-negative**.

It is not an admissible **edge weight**. `PMFG_T2s` returns the structure and the weights in one
matrix — `A = W ⊙ ((A + A') .== 1)` — so a zero weight is an *absent* edge rather than a weak one.
Nothing upstream declines it either: every remaining vertex is inserted whatever the gain, so the
algorithm selects an edge and then the weight deletes it. A maximal planar graph on `N` vertices has
`3N - 6` edges, and each zero on a selected edge silently removes one.

Measured on 12 assets built from a Hadamard matrix, so that many sample correlations are exactly
zero: `LogDistance` with `ExponentialSimilarity` leaves **21 of the 30** edges, and the run then dies
inside `turn_into_Hclust_merges` with a `BoundsError` about a matrix index, because
`HierarchyConstruct4s` builds fewer merges than the dendrogram needs. The zero is the cause and the
infinity is not: replacing the infinite distances with `20.0` still crashes, and replacing the zero
similarities with `exp(-20)` runs clean.

`assert_pmfg_weights` counts the stored edges and refuses the shortfall. It runs at the three sites
that consume the **weighted** structure — `DBHTs`, `calc_weighted_adjacency_graph` and
`calc_distance_weighted_graph`. `logo!` is the fourth `PMFG_T2s` caller and is deliberately **not**
guarded: it reads separators and cliques, which `PMFG_T2s` derives from the insertion order rather
than from `A`, so a zero weight does not shrink them and refusing it would refuse a configuration
that works.

**This is breaking, a fourth time, and the configuration it breaks was returning an empty network.**
`NetworkEstimator(; ce = PortfolioOptimisersCovariance(; mp = MatrixProcessing(; dn = Denoise())),
de = Distance(; alg = LogDistance()), alg = ExponentialSimilarity())` on noise denoises to the
identity correlation, so every off-diagonal similarity is exactly zero and the PMFG came back with
**0 of its 54** edges. That empty structure was returned as an answer, and `test_13_phylogeny.jl`
asserted it. It is now a `DomainError` naming the missing edges. The old behaviour was the silent
wrong answer this ADR exists to remove, one level lower down.

### The guarantee is per member, so it carries a domain

The type says *which algorithm*. It cannot say *which data*, and two of the four admitted members
are non-negative only over part of the distances a caller can produce.

| Member                         | Holds when     |
|:------------------------------ |:-------------- |
| `ExponentialSimilarity`        | always         |
| `GeneralExponentialSimilarity` | always         |
| `MaximumDistanceSimilarity`    | `D` is finite  |
| `ComplementSimilarity`         | `all(D .<= 1)` |

`MaximumDistanceSimilarity` was believed unconditionally non-negative while this was being charted.
It is not: `ceil(Inf^2) - Inf^2` is `NaN`, and every other entry becomes `Inf`. The matrix is
degenerate as well as invalid, the `LogDistance` route above reaches it, and this member is the
**default** of both `DBHT` and `LoGo`. The old argcheck caught it only by accident — `0 <= NaN` is
`false` — and then blamed non-negativity for a `NaN`. That check is now split, so a `NaN` is named
as a `NaN`.

`AngularSimilarity` is excluded **permanently**, and not for want of a precondition. Even paired
correctly it returns a negative wherever `rho < 0`, which is ordinary data.

### The rule is `D <= 1`, not "the metric is unbounded"

`ComplementSimilarity`'s precondition was framed while charting as a fix for unbounded metrics. That
understates the refusal set by a wide margin, and the documentation says `D <= 1` everywhere for
that reason.

`Distances.CosineDist` and `Distances.CorrDist` are **bounded — by `2`, not by `1`** — and both are
refused whenever they exceed `1`, which `CorrDist` does at every negative correlation. Measured over
8 assets and 60 signed features: `CosineDist` reaches `1.88` with 1824 entries above `1`, `CorrDist`
reaches `1.92` with 1832. So `FeatureDistance(; metric = CorrDist())` on the PMFG path is refused
whenever any correlation is negative. That is **correct** — an honest inverse returns a negative and
DBHT cannot take one — and it breaks nothing, because `S = 1 - D < 0` already threw at `PMFG_T2s`.

Four in-library sources exceed `1`, not the two that were charted, and one of them is a default:
`LogDistance` (max `9.08`, 95.0% of entries above `1`), `DistanceDistance` — whose
`Distances.Euclidean` **default** puts 58.5% of entries above `1` at 20 assets —
`VariationInfoDistance(; normalise = false)`, and any `Distances.SemiMetric` on a `FeatureDistance`.

### Normalising the distance was considered and rejected

The charted decision was to keep `ComplementSimilarity` inside the interface **and normalise an
unbounded distance so that it fits**. The normalisation half is withdrawn. Nothing is normalised;
the distance is refused rather than repaired.

The stated reason for normalising was that `ComplementSimilarity` is `default_similarity`'s fallback
for every `SemiMetric` and so cannot be excluded. That is true, and it argues for **keeping the
member**. It does not argue for putting the member **inside the interface**. The premise then fell
outright: every PMFG entry point recomputes the similarity from `nte.alg`, `cle.alg.sim` or
`je.sim`, and discards the one `cor_and_dist` returned, so a `FeatureDistance.sim` **never reaches
`PMFG_T2s`**. The fallback role lives entirely on a path the type bound does not touch.
`AngularSimilarity` is the proof by example: kept, and outside the interface.

Two independent reasons finish it.

**Normalisation is a new feature, not a repair.** Every pairing the precondition refuses already
threw. `Distance(; alg = LogDistance())` and `DistanceDistance()` against `ComplementSimilarity`
both fail today; `SimpleDistance` against `ComplementSimilarity` and `LogDistance` against
`ExponentialSimilarity` both work today and are untouched. No shipped configuration that works
depends on an unbounded distance reaching `ComplementSimilarity`. To normalise would **add**
configurations that presently throw.

**No normalisation shape is both faithful on `[0, 1]` and defined on an `Inf`.**
`ComplementSimilarity` is the honest inverse of `CosineDist`, `Jaccard`, `BrayCurtis` and
`CorrDist`, so a shape that is not the identity on `[0, 1]` breaks the cases that work.

| Shape                    | Identity on `[0, 1]` | Defined on an `Inf`        |
|:------------------------ |:-------------------- |:-------------------------- |
| `D / maximum(D)`         | no                   | no — `Inf/Inf` is `NaN`    |
| `D / max(1, maximum(D))` | yes                  | no — the same `NaN`        |
| `min(D, 1)`              | yes                  | yes, but `S = 0` above `1` |
| `D / (1 + D)`            | no                   | yes                        |
| `tanh(D)`                | no                   | yes                        |
| `1 - exp(-D)`            | no                   | yes                        |

The last row is worse than merely unfaithful. `1 - exp(-D)` makes `ComplementSimilarity`
**identical** to `ExponentialSimilarity`, because `1 - (1 - exp(-D))` is `exp(-D)`.

Divide-by-max carries a second hazard on top: ADR
[0048](0048-a-network-relates-by-its-separation-and-weights-by-what-selected-it.md) recorded a
data-dependent divisor biting `PathLength`'s `dmax = nothing`, where a diameter that moves across
cross-validation folds shifts every row of the feature matrix.

### The domain precondition is interface-scoped, not member-wide

[`assert_similarity_domain(sim, de, D)`](../../src/09_Distance/04_Similarity.jl) runs at the five
PMFG entry points and nowhere else. It is deliberately **not** called inside
`distance_to_similarity`, which stays a pure transformation with no domain of its own.

The scope follows the failure, which is path-dependent rather than member-wide. `ComplementSimilarity`
against an unbounded distance is **documented, tested behaviour** on the `FeatureDistance` path —
`test/test_13_phylogeny.jl` drives it with a `Euclidean` distance and asserts that nothing throws,
and `ComplementSimilarity`'s own docstring gives `D = 7` yielding `S = -6`. Only on the PMFG path is
the same value unusable.

This **moved** a decision made one ticket earlier, which had put each member's domain check inside
`distance_to_similarity`. That reasoning survives — each member still declares its own domain — and
only the site of enforcement changed, to one uniform mechanism. It is recorded here because a reader
who finds the earlier ticket and the shipped code will otherwise see them disagree.

The error is a `DomainError`, which is what was thrown before, so no test and no user expectation
moves. It names **both halves** — the distance estimator that produced the offending maximum and the
similarity that refused it — which is why `de` is passed in and read for nothing else. The checks
are written `all(<=(one(eltype(D))), D)` and `all(isfinite, D)`, never the broadcast form, which
allocates a `BitArray`.

### Open by declaration, not by proof

Membership is a **claim a subtype makes**, not one the library verifies. The claim is made by
writing `<: PortfolioOptimisers.AbstractNonNegativeSimilarityMatrixAlgorithm`, which is the ordinary
route to every other open family in this library. A probe cannot check the claim:
the contract quantifies over every admissible distance matrix, so a probe that passes
`ComplementSimilarity` at `D = 0.5` still misses its failure at `D = 7`. `PMFG_T2s`'s own
non-negativity check is therefore **kept** as the backstop against an extension that claims
membership and does not keep it.

### The alternatives that were rejected

- **Bind DBHT alone.** More honest about provenance; it relaxes the one guard that makes the other
  three callers safe. Treated as a later effort, above.
- **Delete or unexport `AngularSimilarity`.** It is *correct* on the one path where
  `default_similarity` pairs it with `AngularDist`. Removing it deletes working behaviour to fix
  broken behaviour elsewhere.
- **Exclude `ComplementSimilarity` from the interface.** It is `default_similarity`'s fallback for
  every `SemiMetric` and cannot be deleted; the precondition costs one method and keeps the member
  usable wherever it is honest.
- **Shift the matrix by a constant.** Inert for `PMFG_T2s`'s gain argmax, which compares three
  weights at a time. **Not** inert for the seed strength, whose masked set varies per row, and not
  inert for `DirectHb`'s different-sized sums. A shift is not a transparent repair.
- **Check the output range, `all(>=(0), S)`.** Total, and needs no per-member method. It names the
  **symptom**, which is exactly the complaint against the old argcheck, one level earlier. The two
  are complementary rather than competing: the per-member assert gives the message, and `PMFG_T2s`'s
  check stays the backstop.

## Consequences

**Breaking, three times at construction.** `NetworkEstimator(; alg = AngularSimilarity())`,
`DBHT(; sim = ...)` and `LoGo(; sim = ...)` now fail at construction. `@concrete`'s bounded-field
syntax `alg <: Tree_SimMat` puts the bound on the generated type parameter, so **every** construction
route refuses. The keyword route raises a `TypeError` naming the bound and the positional route a
`MethodError`; neither is reachable.

**Breaking a fourth time, at run time.** `assert_pmfg_weights` refuses an exactly zero edge weight,
which takes one configuration that returned an empty network into a `DomainError`. *Non-negative
reaches the check, positive reaches the graph* above carries the measurement and the reason.

**One inverse-claiming member stays inside the interface, and the residue is documented rather than
fixed.** `ComplementSimilarity` against `SimpleDistance` type-checks, satisfies `D <= 1`, is
non-negative on every entry, and returns `0.29` where the correlation is `0.003` — #239's root cause
with the throw removed. **No check placed anywhere can catch it**, because `0.706` is a perfectly
legal bounded distance. The destination of this effort stops at non-negativity, not at a correct
pairing: excluding the inverse-claiming members leaves monotone transforms on the PMFG path, and a
monotone transform cannot be mispaired. `ComplementSimilarity` is the exception, and it carries a
`!!! warning "The pairing is not checked"` admonition saying so in the words above.

**The general fix is not taken here.** Widening `default_similarity` from `Distances.SemiMetric` to
`AbstractDistanceAlgorithm`, so the network path defaults its similarity from its distance the way
`FeatureDistance` does, is a design in its own right. All five entry points already hold `de` in
scope and `assert_similarity_domain` already takes it, so the wiring exists. What is undecided is
whether that widening is right at all, and whether a warning or nothing is the right answer for a
value that is in domain and wrong.

**`FeatureDistance` keeps all five members.** The PMFG entry points recompute the similarity from
the estimator's own field, so narrowing the three estimator fields is *sufficient*. Nothing on the
`FeatureDistance` path changed.

**The tree branch gains nothing, and this interface is not what would give it something.**
`EigenvectorCentrality` declares `SimilarityPolarity` and runs unweighted on a tree branch, which
ADR [0048](0048-a-network-relates-by-its-separation-and-weights-by-what-selected-it.md) left open.
It is not *refused* there — that refusal was withdrawn before ADR 0048 shipped, on the ground that
weightedness is a property of the source rather than of the request — so there is no permission for
this interface to grant.

What weighting that pairing needs is a **similarity for the tree branch**, and this interface
supplies none: it narrows three existing fields and adds no field, so `nte.alg` on a tree is still
an `AbstractTreeType` with nothing for `distance_to_similarity` to dispatch on. The one member that
would have manufactured a signed correlation from a distance, `AngularSimilarity`, is the member
this interface **excludes**.

The arithmetic is nonetheless open, by ADR 0048's own argument. It legitimises
`calc_distance_weighted_graph` — re-weighting a PMFG with `D` — on the ground that every similarity
algorithm is strictly decreasing in `D`, so `D` is the selecting quantity's monotone **preimage**
rather than a foreign quantity. That reasoning is symmetric: a tree re-weighted with `S = f(D)` is
the selecting quantity's monotone **image** by the same step. What blocks it is a design decision —
which member a tree would name, and where that field lives — not arithmetic. ADR 0048 is amended
accordingly; its *Left open* item said a similarity-polarity **separation** member is what would
un-refuse the pairing, and `sep` is inert on every weighted centrality route, so that is the wrong
mechanism for a question that no longer concerns a refusal.

**`Denoise()` alone does not reach the `Inf` route on the repo's own fixture.** The finiteness
precondition was justified by `LogDistance` mapping an exactly zero correlation to `Inf`, and that
arithmetic holds. Reproducing it needs a noise matrix: `Denoise()` on the shipped `SP500` fixture
produces **no** exact zero.
