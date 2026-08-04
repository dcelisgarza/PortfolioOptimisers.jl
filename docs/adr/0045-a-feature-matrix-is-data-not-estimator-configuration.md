---
status: accepted
---

# A feature matrix is data, not estimator configuration

## Context

Every distance the clustering and network stack consumes is derived from the returns themselves.
[`Distance`](../../src/09_Distance/02_Distance.jl) turns a correlation into a distance;
[`DistanceDistance`](../../src/09_Distance/03_DistanceDistance.jl) turns that distance into a
distance of distances. A hierarchy built this way can only ever express relationships that the
return series already encode.

Structure that returns do not encode is routinely available and routinely wanted: a sector and
industry taxonomy, an ESG or fundamentals panel, a supply-chain graph, the loadings of a fitted
factor model, an adjacency matrix from a minimum spanning tree. All of these are naturally shaped
**assets × features** — one row per asset, one column per measured quantity — and a distance between
rows of that matrix is a perfectly good input to `hclust`, DBHT or a network estimator.

The library had no way to express such a matrix. Adding one raises a first question that decides
everything after it: *where does the matrix live?*

The obvious answer — a field on the distance estimator, the same way `Distance` holds its `alg` — is
wrong here, and wrong **silently**. The clustering stack subsets assets constantly:
[`NestedClustered`](../../src/20_Optimisation/17_NestedClustered.jl) optimises each cluster as its
own subproblem, `SubsetResampling` draws asset subsets, and every cross-validation fold slices both
observations and assets. All of that subsetting happens through `port_opt_view`, which walks the
**data** — `ReturnsResult`, prior results — and slices the asset axis, while treating estimators as
configuration and passing them through unchanged. That asymmetry is deliberate and correct: an
estimator is *how* to compute, and how-to-compute does not shrink when the universe does.

A matrix held on the estimator therefore inherits the estimator's treatment. It would survive a
three-asset subproblem still describing the full eight-asset universe, and the resulting distance
matrix would be the wrong size — or, worse, the right size and the wrong numbers. `ClustersEstimator`
tags `ce` and `de` `@fprop` only, and `HierarchicalOptimiser` does not `@vprop` its `cle`, so
`@propagatable` emits no `port_opt_view` for either: nothing exists that *could* have sliced it.
This is not hypothetical. Issue
[#184](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/184) found the same hole live
elsewhere in the library — `SemiDefinitePhylogenyEstimator` holds a precomputed `PhylogenyResult`
with no `@vprop`, so a three-asset subproblem got an `8 × 8` constraint matrix without complaint.
That issue also marks the boundary of "data is sliced": a phylogeny is data, but it is not
**separable** over the asset universe, so slicing it is not a restriction of the same quantity. Its
resolution is a refusal rather than a view — see the last consequence below.

Data, by contrast, is already sliced everywhere by machinery that exists and is tested.

## Decision

**The feature matrix is data.** It is carried on the result types alongside the returns, travels the
paths the returns travel, and is sliced by the views that slice the returns. The estimator holds only
configuration: which metric, how to collapse a window, which similarity counterpart.

Seven decisions follow from that one, in the order they constrain each other.

### 1. Vocabulary: **feature**, and the symbol is `Z`

The noun is **feature**. *Characteristic* was the first candidate and is unavailable: `CONTEXT.md`
already pins **Characteristic Vector** as the per-asset quantity an ℓ1 uncertainty set is built
around, with `CharacteristicUncertaintySet` naming it in code (ADR
[0032](0032-quintile-portfolios-are-an-uncertainty-set.md)). *Attribute* was rejected as vaguer than
either, with no established meaning in the surrounding literature. The two nouns are near-synonyms
held deliberately apart, and the glossary polices the pair.

The symbol is `Z`. `F` and `nf` are taken by factor returns; `Z` follows the existing `X`/`F`/`B`
single-capital rhythm and is the asset-pricing literature's own symbol for a matrix of asset
characteristics. Feature names are `nz`, matching `nx`/`nf`/`nb`.

The estimator is [`FeatureDistance`](../../src/09_Distance/05_FeatureDistance.jl) `<:
AbstractDistanceEstimator`. Not `AdjacencyDistance`: every sibling algorithm is named for what it
computes *from*, and "adjacency" implies a square `0`/`1` graph while the type accepts rectangular,
unbounded, signed reals.

### 2. Both shapes, distinguished by `ndims`

A feature can be static (a sector membership) or time-varying (a rolling fundamental). Both are
admitted, with no wrapper type:

- **static** — `Z :: assets × features`
- **time-varying** — `Z :: obs × assets × features`, observation axis always leading

Dispatch is on a new `Arr3Num` alias rather than a runtime `ndims` branch, so the two entry points
stay apart the way the rest of the library keeps shapes apart. Wrapper types (`StaticFeatures`,
`TimeVaryingFeatures`) were rejected: they add two names and a constructor to express what `ndims`
already expresses, and every consumer would have to unwrap before it could compute.

### 3. Orientation is declared, not configured — and a *carried* matrix has no choice

The raw-matrix entry point `distance(de, Z; dims)` reuses the **existing `dims` keyword**, which
*retargets* from `X` to `Z` for this algorithm. That retarget is only coherent because decision 5
makes the similarity counterpart feature-derived too, so `X` is genuinely unused by this estimator.

A `dims` **field on the algorithm** was rejected. Orientation is a property of the matrix in hand,
not of the estimator, and a field would have to agree with the ambient keyword at every call site
that has both.

A **carried** matrix is stricter still: it is canonically assets-major and the constructor rejects
anything else. The driving fact is that `port_opt_view` has no `dims` and hardcodes
`view(rd.X, :, i)` — the whole view and cross-validation layer is `dims = 1`-only. Accordingly the
routed three-argument `distance`/`cor_and_dist` methods hardcode `dims = 1` and ignore the ambient
value; forwarding it would transpose a correct `Z` for every `dims = 2` caller, and the result would
still be square, finite and symmetric.

### 4. When features *are* assets, a view slices both axes

If `nz == nx` — compared by **name**, not by axis length, since equal counts are not a claim that the
axes mean the same thing — the feature axis *is* the asset axis, and an asset view must slice both:
`view(Z, j, j)`, or `view(Z, i, j, j)` in three dimensions. Without this, a square adjacency survives
an `NestedClustered` cluster with its column axis still pointing at the full universe.

The prior carrier cannot make that comparison — it holds no names at all (decision 6) — so it carried
an explicit `z_sq::Bool` stated by whoever built the matrix, validated to actually be square. **That
flag has since been deleted** and this decision now applies to the data carrier alone; see the
amendment at the end.

This is the one carrier shape where **`distance` does not commute with an asset view**. Slicing the
feature axis truncates every row's feature vector, so a subproblem is measured against its own
neighbourhood structure rather than against the whole universe. That is the intended reading, and it
is a real semantic difference from the rectangular case, which does commute. On the prior carrier the
same reading survives the flag's deletion by a different mechanism — a refit rather than a slice.

### 5. An open metric family, and a similarity counterpart derived from the distance

`FeatureDistance(; metric = …)` accepts **any `Distances.SemiMetric`**, including user-defined ones.
`Distances` is already a direct dependency. The default is `AngularDist`, defined here in about
twenty lines: `acos(clamp(1 - CosineDist, -1, 1))/π`, a true metric where `CosineDist = 1 - cos` is
not. It delegates its pairwise path to the `CosineDist` gemm kernel already in `Distances`, which
beat a naive loop at every size measured, with no crossover — so there is no fast/slow split to tune
and no cost ceiling to express.

`cor_and_dist`'s `S` slot is filled by the **existing `AbstractSimilarityMatrixAlgorithm` family**,
which gains two members: `ComplementSimilarity` (`1 .- D`) and `AngularSimilarity` (`cos.(π .* D)`).
`FeatureDistance.sim` is non-optional and defaulted from the metric by `default_similarity`, so the
resolved choice is visible on the printed object rather than hidden in the kernel.

Two alternatives were rejected here.

**`S` from the returns correlation** was rejected because it would give `S` and `D` different
provenance — a similarity measured on returns paired with a distance measured on features — and
because it would resurrect the `dims` incoherence decision 3 avoids by making `X` load-bearing again.

**A closed metric family** — a blessed list with a per-metric `feature_similarity(metric, Z)` hook
that throws for anything else — was designed and then discarded. It collapsed on inspection: four of
its five methods were literally `1 .- D`, which `ComplementSimilarity` absorbs while still giving
each metric its named counterpart (cosine, Ruzicka, Sørensen–Dice, Pearson), and the fifth was a gram
matrix that `cos.(π .* D)` recovers exactly with no `Z` at all. The decisive fact against throwing is
that **every consumer calls `cor_and_dist`, not `distance`** — all six clustering and network sites.
A hook that threw would have made every unblessed metric unusable in the entire stack, not merely
unable to produce a similarity.

The cost of accepting every metric is that `ComplementSimilarity` is unbounded below when `D` is: a
`Euclidean` distance of 7 gives `S = -6`, breaching the `[-1, 1]` convention. Symmetry and the unit
diagonal survive; the visible failure is silent clipping in `plot_clusters`' colour scale, which is
documented rather than guarded. Negatives never reach `PMFG_T2s`, which does require non-negative
weights, because DBHT overwrites `S` first.

Validation splits along the same line the design does. **Construction** checks structure —
non-empty, shapes against `nx`/`nz`, `allunique(nz)`, no `NaN` or `±Inf`. **The kernel** checks
metric-dependent domains, because the metric is unknown when a `ReturnsResult` is built: the Ruzicka
`Jaccard`, `BrayCurtis` and `ChiSqDist` are defined only on non-negative reals. A blanket
non-negativity check at construction would have rejected signed factor loadings *and* the default
metric. `Jaccard` in particular fails silently — it returns values up to `2` on signed input, with no
error, straight into `hclust`.

### 6. Two carriers, chosen by a named selector

The matrix lives on `ReturnsResult` (and `PricesResult`) as user-supplied data, **and** on
`LowOrderPrior` as derived data. Provenance is chosen by `z_src::Symbol = :data` — a field on
`JuMPOptimiser`, `HierarchicalOptimiser` and `NestedClustered`, next to and matching `x_src` (ADR
[0044](0044-matrix-sources-are-named-not-flagged.md)).

A **single carrier** was rejected in both directions. Data-only cannot express a matrix that must be
recomputed per fold from the fold's own returns — factor loadings are the obvious case. Prior-only
would force a user with a static sector matrix to write a prior estimator to hold it.

Two carriers only risk disagreeing if the same matrix can reach both. It cannot: provenance is
strictly derived, `prior(pe, rd)` deliberately drops `rd.Z`, and no `Z` keyword threads through the
prior estimators. So `z_src` never selects between two copies of one matrix — it selects between two
genuinely different matrices, and the guarantee is structural rather than validated.

The two selectors' **defaults differ deliberately**, which is the argument for naming them rather
than flagging them: `x_src = :prior` because the prior is what the optimisation is defined on, and
`z_src = :data` because between two real sources, one of them hand-typed, explicit outranks derived.

The prior carrier is `Z` and **no `nz`** — originally `Z` plus a `z_sq::Bool`, now the matrix alone
(see the amendment). A prior result has never carried names —
asset names for `X` live on the `ReturnsResult` or an `AssetSets` — and a producer runs inside
`prior(pe, X, F; …)` where no names exist to record.

Load order decided how the matrix reaches the kernel. `09_Distance` loads before `13_Prior`, so the
kernel can dispatch on `ReturnsResult` but cannot name `AbstractPriorResult`, and therefore cannot
resolve `z_src` itself. **The prior-layer bridge resolves the selector and the raw matrix crosses as
a keyword argument** into new three-argument methods. Consumers needed no edits — they already forward
`kwargs...`.

The selector rides the wire alongside the matrix as a diagnostic, so an absent `Z` can say which of
three distinct mistakes produced it rather than reporting a bare `nothing`. Asking for a feature
matrix that is not there throws `IsNothingError`; carrying one that nothing asks for is **silent**,
matching how `iv`, `ivpa`, `F` and `B` already behave. The asymmetry is principled: "asked for it,
absent" is an error, "have it, did not ask" is not.

### 7. A window of features collapses before, or as, the distance

The three-dimensional shape needs a rule for turning a window into one matrix. It is an open family,
`AbstractFeatureCollapseAlgorithm`, on the estimator's `alg` field — named `alg` and not `collapse`
because `prices_to_returns(; collapse_args)` already owns that word, on the same observation axis, in
the same file as `ReturnsResult`:

- **`LastObservation`** (default) — the most recent slice
- **`AggregateFeatures`** — aggregate the window into one feature matrix, then measure
- **`AggregateDistances`** — measure each slice, then aggregate the distance matrices
- **`StackObservations`** — reshape `(T, N, K) → (N, T·K)`, one gemm, no aggregation choice

The aggregating two take a second tiny family, `MeanCollapse` (default) or `MedianCollapse`, and
optional observation weights. `AggregateDistances` **rejects `MedianCollapse` at construction**: a
convex combination of metrics is a metric, an elementwise median of distance matrices is not.

The two-dimensional path never reads `alg` — inert, not an error — because `z_src` legitimately
switches between two-dimensional sources and three-dimensional ones. At `T = 1` all four rules agree
exactly; that is a real degeneracy, not a special case.

### 8. Four producers

A **producer** turns something the library already computes into a feature matrix, and runs inside
the wrapping `FeaturePrior`, which attaches its output to the prior it wraps.

1. **A literal matrix.** `FeaturePrior(; ze = Z)` attaches a matrix the user already has. The
   identity producer, and the reason `FeaturePrior` exists as a wrapper rather than as a keyword on
   every prior estimator.
2. **`RegressionFeatures`** — factor loadings. It reads `pr.rr.L`, the reduced-dimension coordinate
   system the asset lives in, *not* `pr.rr.M`, the reconstructed full-factor loadings. `L` always
   resolves, falling back to `M` through `Regression`'s `swap(L, M)` forwarding, so there is no
   branch. Exposed as an explicit producer rather than populated directly, which keeps `pr.Z` from
   becoming a second live spelling of `pr.rr.L`.
3. **`AssetSetsFeatures`** — exogenous taxonomy memberships, built from the `AssetSets` machinery
   already behind group constraints. Rectangular: its feature axis indexes groups, never assets.
4. **`PhylogenyFeatures`** — a square neighbourhood matrix from a network estimator (a graph, under
   `BinaryNeighbourhood` or `GradedNeighbourhood` decay) or a clustering estimator (a partition, for
   which the decay is inert). The only producer
   whose feature axis *is* the asset axis. Both sources are estimators and both refit, so this
   producer is endogenous; a partition source is admitted but coarse (see the type's docstring).

Producer three was originally specified as a non-square **phylogeny** routine, and that framing was
wrong. Phylogeny routines are endogenous to the returns by construction, and the value of a feature
matrix is precisely that it is exogenous. The proof is a round trip: multi-resolution cluster
memberships look ideal — concatenate indicator blocks from several dendrogram cuts and every row has
equal norm, so cosine needs no standardisation — but `cutree` gives **nested** partitions, so the
shared-membership count between two assets depends on nothing but the finest cut at which they still
co-cluster, which is monotone in cophenetic merge height. The resulting distance is a monotone
discretisation of the cophenetic distance of the hierarchy that produced it, and clustering it
returns that hierarchy. **Any producer that clusters and then re-encodes the clustering is a
recoding.** A single partition is worse: every row is one-hot, every `Distances` semimetric is a
coordinate-permutation-invariant sum, so the distance matrix takes at most two distinct values.

Taxonomies escape this because they never touch the returns, and nested taxonomies (sector ⊃ industry
⊃ sub-industry) make cosine similarity a literal count of shared classification levels.

Producer four is endogenous and kept anyway, because it is the only square route and because what it
buys was measured rather than asserted: against a plain correlation distance it gives a different
merge order and different weights, but identical cuts at `k = 2` and `k = 3`, diverging only from
`k = 4`. A genuinely exogenous graph through the same producer disagrees at every cut. Endogeneity is
a spectrum, and that measurement is the yardstick any future endogenous producer has to beat.

Its diagonal is a **choice between two algorithms, not a convention**. Excluding self, the two
non-adjacent endpoints of a three-node path come out identical and the adjacent pair maximally
distant — structural equivalence, the opposite of proximity. Self is therefore always included, and
needs no flag because the value is just `f(0)`. There is a second, independent argument: an asset view
of a spanning tree routinely isolates every selected vertex, and zero-diagonal rows are then zero
rows, which the zero-vector convention declares mutually identical — a measured case gave an
all-zero distance matrix for three unrelated assets inside a cluster.

## Consequences

- **`ReturnsResult` and `PricesResult` gain `nz` and `Z`, appended last**, so
  `returns_result_picker`'s partially-applied dispatch keeps working untouched. Nine pretty-print
  doctests gain two rows.
- **`LowOrderPrior` gains `Z`, appended last** (originally `Z` and `z_sq`; see the amendment).
  `HighOrderPrior` needs no edit — its `@forward_properties` branch forwards any property. Every
  `LowOrderPrior(; …)` construction site forwards it, so nesting order does not matter.
- **A price-level `Z` is not on the master clock and cannot be** — no three-dimensional `TimeArray`
  exists — so it is held positionally parallel to `X`, and three sites re-establish that alignment by
  hand: `port_opt_view`, `MissingDataFilter`'s preprocessing, and `prices_to_returns`.
- **A meta-optimiser's outer problem has no features, deliberately and loudly** (superseded; see the
  second amendment). All three synthetic-universe builders drop `Z`, because their assets are
  sub-portfolios, clusters or predictions rather than the universe. An outer estimator asking for a
  `FeatureDistance` gets an `IsNothingError`, never a stale full-universe matrix. Whether `Z` should
  instead be *collapsed* onto the synthetic assets is
  [#179](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/179).
- **`z_src = :data` slices, `z_src = :prior` refits.** Under a fold or a cluster the carried matrix is
  subselected while the derived one is recomputed on the subproblem's own returns — so the derived
  one's feature axis stays the full factor set while the carried square one's does not. The two
  selectors pick between two semantics, not merely two sources.
- **A time-varying literal `ze` cannot survive an observation fold** and throws at construction. Only
  a producer derives a per-fold time-varying `Z`.
- **A phylogeny is why "data is sliced" needed a boundary at all.** A `PhylogenyResult` is data, but
  it is not *separable* over the asset universe: every entry states something about the whole graph,
  so no slice of it is the phylogeny of the slice. The subgraph of a spanning tree is not the
  spanning tree of the subgraph, and is routinely disconnected — an eight-asset minimum spanning tree
  restricted to three assets keeps two of its fourteen edge entries and can leave a selected asset
  attached to nothing. Worse, the slice re-validates (symmetric, zero diagonal), so nothing
  downstream could have caught it. Rather than teach the view layer to refuse, the shape was removed:
  no estimator holds a `PhylogenyResult`, so nothing ever presents one to a view (next consequence).
- **No estimator holds a precomputed result any more, and that is the root-cause half of
  the same fix.** `SemiDefinitePhylogenyEstimator`, `IntegerPhylogenyEstimator`,
  `CentralityEstimator` and `PhylogenyFeatures` all took a `PhylogenyResult` or a `Clusters` — an estimator
  presenting as configuration while holding data one level down, invisible to every guard aimed at
  results, which is exactly how #184 got through. All four `pl` slots are now bounded by `NwE_ClE`
  (sources only), so the shape is rejected by the **type** rather than policed at runtime, and
  nothing is lost: precomputed structure belongs in the constraint *result* (`SemiDefinitePhylogeny`
  / `IntegerPhylogeny`, whose `A` takes a `PhylogenyResult` or a matrix), which is precisely what
  `phylogeny_constraints(estimator, X)` returns. Removing the shape removed the checks with it —
  there is no predicate walking estimators for embedded data, because there is nothing to walk. The
  single runtime guard left is `assert_external_optimiser` rejecting a precomputed constraint result
  in `ple`, which no type bound can take over because `ple` legitimately accepts a result outside a
  meta-optimiser. That guard also had a latent bug: `||` binds looser than `&&`, so its vector branch
  was unreachable and a result inside a vector passed in exactly the case the branch was written for.
  This generalises beyond phylogeny and is recorded as a library-wide rule in `CONTEXT.md` §1: **an
  Estimator never holds a Result.** One consequence is worth stating precisely — with
  `PhylogenyFeatures` narrowed, every square-producing *producer* now refits from the returns, so the
  **prior** carrier has no exogenous route to square structure. The **data** carrier still does, and
  it is the default: a hand-supplied adjacency goes on `ReturnsResult` as `nz`/`Z` with `nz == nx`,
  where squareness is derived from the names rather than declared, so it cannot be stated wrongly.
  What a caller must not do is pass a square matrix as a literal `ze` — the literal path never
  declares its axes, and its columns then keep pointing at the full universe inside a subproblem.
- **Breaking, twice.** `cle_pr` became `x_src` (ADR
  [0044](0044-matrix-sources-are-named-not-flagged.md)), and `dbht_similarity` became
  `distance_to_similarity` when the similarity family moved from `11_Phylogeny/04_DBHT.jl` into
  `09_Distance/04_Similarity.jl` — required, because annotating `FeatureDistance.sim` with a type
  defined in a later-loading file is an `UndefVarError` at load. Neither has a deprecation shim; `src`
  has never carried an `@deprecate`.
- **"Feature" becomes domain language, which makes its existing loose uses wrong.** The word was
  already in the codebase with two other referents — *factors* in the regression docstrings, *assets*
  in the shared `field_dict`/`ret_dict` templates and hence across the moments layer. Both are
  corrected, and the glossary records the pair *Feature* / *Characteristic Vector* explicitly so they
  stay apart.
- **Left open.** Where a `Z` transform lives — inverse-document-frequency reweighting for a dense hub
  column, standardisation for heterogeneous feature scales, and `StackObservations`' magnitude
  domination are one question in four costumes, and probably want one answer. Also open: whether an
  *endogenous* producer is wanted beyond producer four, with centrality features the one candidate the
  round-trip proof above does not reach; and whether a `Clusters` result should record which matrix
  produced it, which is harder now that the prior carrier is nameless by decision.
- **Ruled out of scope.** Consuming `Z` directly in JuMP constraints (feature-based budgets) is a
  separate feature — this design turns `Z` into a *distance*, and the constraint-generation layer has
  its own vocabulary for asset attributes. Feeding a `FeatureDistance` to `LoGo` is unreachable by
  construction: `logo!` is called from matrix processing with only `sigma` and `X` in hand, so no
  carrier can reach it, and it throws rather than silently measuring returns.

## Amendment (2026-07-29): the prior carrier's `z_sq` is deleted

Decision 4 gave the prior carrier an explicit `z_sq::Bool`, and decision 6 justified it: the carrier
is nameless, so squareness could not be *derived* there the way `features_are_assets` derives it on
the data carrier, and had to be *stated* by the producer instead. The flag's whole job was one
branch — `port_opt_view(::LowOrderPrior, i)` slicing the feature axis as well as the asset axis when
the two coincided.

**That branch lost its last reachable consumer**, and the change that emptied it is recorded three
paragraphs above: with `PhylogenyFeatures` narrowed to an estimator source (issue
[#184](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/184) and the library-wide *an
Estimator never holds a Result* rule), every producer that builds a square feature matrix now
**refits** — the one path that never reaches `feature_matrix_view`. The asset-subsetting
meta-optimisers view the estimator and the `ReturnsResult`, not the fitted prior; the single site that
does view a fitted prior discards the slice; and all three of them refuse a user-supplied precomputed
prior outright. Issue
[#192](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/192) measured the emptiness rather
than arguing it: with the branch and the flag both neutered, every failing assertion was one that
hand-constructed a `LowOrderPrior` and called `port_opt_view` on it directly, while every test driving
a real fold, cluster or resample stayed green. Not one optimiser, distance or weight changes.

So `z_sq`, `assert_square_feature_axis` and the `(Z, z_sq)` producer return are deleted:
`feature_matrix(ze, pr, X, F, sets)` returns `Z` alone. **Breaking on a documented extension point** —
a user-written `AbstractFeatureMatrixEstimator` fails on tuple arity, which is the intended failure
mode and the same reasoning as `x_src` and `prepare_outer_rd` — with no deprecation shim, as above.

Three things are worth stating precisely.

- **The semantics survive the flag, by a better route.** A subproblem is still measured on its own
  neighbourhood structure; it gets there by recomputing the graph on its own universe instead of
  slicing a matrix that describes a larger one. That is #184's non-separability argument arriving where
  it was headed: no slice of a phylogeny is the phylogeny of the slice.
- **It makes this ADR internally consistent.** The consequence above already declared that the prior
  carrier has no exogenous route to square structure and that the data carrier is the only one. The
  flag was a squareness vocabulary on the carrier that has no squareness to state; deleting it leaves
  the *derived* carrier saying only "these are features" and the *data* carrier the sole place where
  the features can be the assets — and there squareness is derived, so it cannot be stated wrongly.
- **It retires a silent-wrongness hazard that guarded nothing.** `assert_square_feature_axis` existed
  because a lying `z_sq` surfaced as a `BoundsError` deep inside a cluster fold rather than at
  construction. A flag whose only failure mode is silent, protecting a branch with no consumer, is a
  bad trade in both directions.

Unaffected: `feature_matrix_view`'s `sq` parameter stays — `ReturnsResult` and `PricesResult` still
derive it from names and still need it, and only the prior-carrier call site changes, which now passes
a literal `false` as two sites already did. The collapse onto synthetic assets
([#188](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/188)) detects squareness with
`features_are_assets` on the data carrier, so it is unchanged either way. And the literal-`ze` gap in
the consequence above is neither opened nor closed: the literal path never declared its axes in the
first place.

## Amendment (2026-07-29): the feature matrix collapses onto a meta-optimiser's synthetic assets

The consequence above — "a meta-optimiser's outer problem has no features, deliberately and loudly" —
recorded a *boundary*, not a decision: `NestedClustered` and `Stacking` build an outer
`ReturnsResult` whose assets are clusters or sub-portfolios, so a matrix indexed by real assets had no
axis to bind to and was dropped. Issue
[#179](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/179) settled the question the
bullet deferred: **the feature matrix is aggregated onto the synthetic assets**, and that collapse is
the primary route, since `z_src` defaults to `:data`. Refit and collapse remain two different claims —
re-estimating a synthetic asset's feature in its own right versus aggregating its members' — and
`z_src` still selects between them. The source is `rd.Z` only: writing a collapsed `pr.Z` onto the
outer result would land a *derived* matrix on the *data* carrier, undoing the strictness decision 5
exists for.

**Features are intensive, so the collapse is a convex combination**, sharing `synthetic_asset_weights`
with `iv` and `ivpa` rather than restating the normalisation. Under the default `AngularDist` this is
a mathematical no-op for a rectangular matrix — it scales one row of the result — and it is *not* one
in the square case, where the two-sided product rescales feature columns too. It is also what keeps
the collapse bounded for any gross exposure, so only the exact zero needs a guard, and a degenerate
synthetic asset lands on the zero-feature-vector convention rather than throwing. An *extensive*
feature wanting a weighted sum is unsupported, and a caller cannot pre-scale their way to one: the
divisor depends on the inner solve. The per-column intensive/extensive tag belongs to the `Z`-transform
question the map still carries.

Two things this ADR should state, because the collapse is not one operation but two.

- **Squareness is preserved where the weights allow it.** With the whole weight matrix in hand —
  `prepare_outer_rd`, the fold-less path — a square feature matrix collapses two-sided, and its
  feature axis is renamed after the synthetic assets so `features_are_assets` stays true one level up.
  The diagonal is left exactly as computed: re-zeroing it would turn a cluster with no cross-cluster
  edges into a zero row, and zero rows are declared mutually identical — the same fold argument that
  made self-inclusion load-bearing one level down.
- **Inside a cross-validation fold there is only one weight vector, and that changes the answer.**
  `reconstruct_rd` collapses onto a single synthetic asset, so the second contraction of the square
  case is unavailable; performing it one-sidedly with the vector in hand would leave *one number* per
  synthetic asset, a feature space in which every asset is trivially identical. The fold path
  therefore leaves the feature axis alone, and a square matrix reads there as the synthetic asset's
  weighted-average neighbourhood over the real assets. The kernel says so by shape: its vector arity
  takes no `sq` argument at all.

**The fold path is time-varying by construction**, because each fold collapses with its own weights.
`reconstruct_rd` gives its result an observation axis — a static source is genuinely constant within a
fold and different in the next — folds stack down that axis, and `rebuild_returns_result` lays the sub-
portfolios out along the asset axis, giving `observations × assets × features` with the same row count
as `X`. The outer optimiser's default `LastObservation` then reads the most recent fold, and the rest
of the collapse family reads as many as it is asked to. That path needs a **fourth carrier**, `nz`/`Z`
on `PredictionReturnsResult`: pure transport, and write-only by construction, since a
`PredictionReturnsResult` can never reach the `Pr_RR` bridge. (The amendment below tested that claim
against a second, non-bridge reader and it held — for a reason the bridge argument did not anticipate.)

One combination still drops the matrix, and it is the intersection of the two paragraphs above:
`NestedClustered` **with** cross-validation **and** a square feature matrix. Its folds see
cluster-sliced returns, so each cluster's feature axis is its own asset subset and there is nothing to
stack them against. The matrix is dropped and an outer estimator asking for features gets the error it
would get had none been supplied — never a matrix assembled from mismatched axes.

> **Superseded in part by the fifth amendment below.** The cross-validated collapse no longer happens
> inside the fold: it moved to the assembly seam, where every synthetic asset's weights are in scope
> at once. So the fold-path paragraphs above describe history — the one-sided square collapse, the
> fourth carrier, and this drop are all gone — while the fold-less path and the intensive-collapse
> argument are unchanged.

`prepare_outer_rd` returning `nz`/`Z` is **breaking on a documented overload point**, same reasoning as
`x_src` and the producer return above. It returns them **before** the returns buffer `X`, not after,
and that ordering is the whole point: Julia's destructuring discards trailing values without
complaint, so appending the pair would have let a stale `predict_outer_*` overload keep building a
feature-less result in silence — decision 1's failure class arriving through the fix for it.

## Amendment (2026-07-29): preselection reads the data carrier, and carries no `z_src`

Decision 5 gave the clustering and network estimators a `z_src` selector on the optimiser, choosing
between the data carrier and the prior carrier. Asset preselection is the first consumer that reaches
`clusterise` from *outside* an optimiser, and issue
[#180](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/180) asked what `z_src` means
there. It means nothing — but the useful finding is stronger than that: **at a pre-prior site `z_src`
is not merely meaningless, it is unreferenceable**, because there is no prior for it to name.

A prior is unreachable from an asset selector by three independent mechanisms, each sufficient on its
own. `run_step(::AbstractReturnsPreprocessingEstimator, ctx)` passes `ctx.returns` and never the
context, so a selector cannot see a prior even when one exists. `maybe_inject_step` has methods for an
`OptimisationEstimator`, a result and a `PipelineStep`, so a preprocessing estimator hits the identity
fallback and nothing can be injected into it. And a selector writes `:returns`, which
`PIPELINE_INVALIDATES` declares invalidates `:prior` — the pipeline's own lattice forbids the very
ordering a `z_src = :prior` would need. So `ClusterGroups` gains **no field**: a knob with one legal
position is not a knob, and a throw needs something to throw on. **The absence of the flag is the
statement**, the same shape as the fold-path kernel above whose vector arity takes no `sq` argument.
The same fact settles `x_src`.

`ClusterGroups` therefore reads `rd.Z` directly, calling the raw `clusterise(cle, rd.X; Z = rd.Z,
z_src = :data_only)` rather than going through the `Pr_RR` bridge. That **widens an implicit contract**
— an `AbstractReturnsResult` reaching an asset selector must now supply `{nx, X, Z}`, not `{nx, X}` —
which is the right side to widen: `Pr_RR`'s concreteness is load-bearing at nine routing sites, while
this contract is local to the selector family and documented on `AbstractReturnsResult` and
`select_assets`.

The diagnostic family gains a **fifth** member, `:data_only`. It is needed because `:neither`'s remedy —
"use a `FeaturePrior` to derive it" — is *actively wrong* here and would send the user to debug the
unreachable half of a decision. It is named for the **situation**, not for the caller, so any future
pre-prior site inherits it; `ClusterGroups` is simply the one that exists. It is an explicit branch
rather than a widened `else`, so an unrecognised symbol still falls through to `:neither`.

Two things this amendment settles that #180 got the other way round.

- **`PredictionReturnsResult.Z` stays write-only.** #180 expected this site to make it readable for
  the first time, by a direct `fit_preprocessing(sel, prediction_result)` call. It does not, and the
  reason has nothing to do with `Z` or with the bridge: `PredictionReturnsResult.X` is a *portfolio*
  return vector — the asset axis is exactly what the collapse removed — while `nx` still names the
  real assets, so `size(X, 2) == 1 ≠ length(nx)`. The type satisfies neither the old contract nor the
  widened one, and every entry point refuses it loudly: the selectors on `X`'s shape, and the replay
  half on the `port_opt_view` tripwire. Even `CompleteAssetSelector`, which touches no feature matrix,
  is refused. The transport carrier is therefore write-only for a second, independent reason, and the
  claim above is stronger than the argument that first established it.
- **The replay half needed nothing.** `apply_preprocessing(::AssetSelectorResult, rd)` already
  delegates to `port_opt_view`, so a selected universe propagates to `Z` — one axis for a rectangular
  matrix, both for a square one. The ordering is right rather than lucky: the selection is decided on
  the **full** universe and sliced only afterwards, which is the same semantics as a square producer
  one layer down, where a subproblem is measured on its own neighbourhood structure only after the
  structure over everything has been computed.

## Amendment (2026-08-04): the taxonomy producer takes a graded program over a declared axis

`AssetSetsFeatures` produced *membership*: a list of `UniverseSets` keys, each stacked as a one-hot
block, every written cell at `1.0`. Its `vals` now admits a second reading — an ordered
**edge-authoring program** of `Pair`s over a feature axis the sets *declares* — with the key list
surviving as the degenerate case by dispatch on element type. The producer stops being "taxonomy
memberships, optionally weighted" and becomes "a user-authored biadjacency matrix that resolves
against a `UniverseSets` and refits under a fold".

The grammar:

```text
entry  := rowsel => targets                    # row scope, then explicit columns
        | taxkey [=> group] => value           # diagonal: those rows, their own membership
rowsel := asset | group | taxkey [=> group]
target := taxkey [=> group] => value | asset => value | group => value
value  := Number                               # sets, absolutely
        | <:AbstractFeatureValue               # Scale(x): x × the key's natural value
```

Entries apply in order and every write is a pure overwrite, so **last wins**.

### What is new, and why each shape is what it is

- **`UniverseSets` gains `zkey`** (default `"nz"`), the **third** declared axis after ADR 0047's
  `fkey`. It is the fifth *name* field: the inner constructor goes 5 → 6 positionals, breaking on a
  documented public type, with one positional call site in the repo — its own keyword constructor.
  `zkey` joins the mutual-prefix loop, taking it from 12 ordered checks to 20.

  It gets **no** prefix convention and **no** unique-entry sibling, which is the asymmetry worth
  recording: `xkey` and `fkey` each have one because each names an axis that partitions are written
  *over*. Nothing is written over the feature axis — the taxonomy keys are `xkey`-prefixed and
  asset-length, and the columns are named straight out of the flat list. Its only rule is
  `allunique(dict[zkey])`, so `ReturnsResult`'s own `nz` check cannot be reached with a duplicate.
  The entry is **optional**, like the factor axis, and diagnosed at the point of need by
  `feature_universe` — `factor_universe`'s sibling, minus the arity reconciliation, because the
  feature axis has no matrix to agree with: it *defines* the width.

- **The axis is declared, not derived**, and that buys two things a `Dict` cannot give: a column
  order, and a node that nothing writes to. It also makes the axis **fold-invariant** —
  `port_opt_view(::UniverseSets, i, args...)` passes `zkey` through, gaining no branch. That is the
  *opposite* of what this ADR's original text records for the key path, where the viewed producer
  rebuilds the axis from the viewed taxonomy and a group with no members left disappears. Both are
  now true, on different paths, and the docs say which is which.

  The exemption has a different cause from the factor axis's. Factors pass through because an asset
  index is meaningless on them; the feature axis passes through even though some of its nodes *are*
  assets, because it is **authored** rather than summarised. The accepted cost: an asset node whose
  asset a view dropped survives as an all-zero column — benign for every blessed metric (a zero
  column adds nothing to a dot product or a row norm) except `CorrDist`, which centres each row.

- **A bare number sets; `Scale` scales; the marker is an open family.** Letting *nesting depth*
  decide instead was rejected by arithmetic: on a one-hot key the natural value is always `1.0`, so
  the two readings coincide and the divergence appears only once a taxonomy carries numbers — which
  is exactly the feature being added. Two properties of `Scale` are **forced**, not chosen. It scales
  the key's own *datum* and never the accumulated cell, because at the top level there is no
  accumulated value; that is what keeps the program a pure overwrite and lets last-wins survive the
  marker. And consequently **scaling a cross edge gives zero**, since the natural value of "is C in
  the US?" is `0` when C is UK. A documented hazard, not a defect.

- **Targets are always fully qualified.** There is no ambient scope and no fallback chain, because
  the chain would have to be four levels deep and would answer "is `UK` the country or the ticker?"
  by proximity rather than by what the caller wrote. Uniformity costs two entries in the worked
  example and gains one: a bare `Number` right-hand side can then always mean the diagonal write, so
  `"nx_country" => ["UK" => 0.5]` collapses to `"nx_country" => "UK" => 0.5`.

  Row-selector precedence needs no new rule. `UniverseSets` already guarantees every `xkey`-prefixed
  key is asset-parallel, so the prefix decides, and what is left is `estimator_to_val`'s existing
  asset-then-group order. A **factor-axis** key is refused by name between the two, rather than
  falling through to the plain-group branch and failing later on a length mismatch that names neither
  the axis nor the cause. That refusal comes *before* the grammar check in target position, since a
  factor key wearing a taxonomy key's two-level shape would otherwise be reported as a syntax error
  and send the caller hunting for a missing bracket.

- **Node names are bare.** Qualified-first-with-bare-fallback was declined, with the cost stated: a
  nested taxonomy with a **repeated value** is inexpressible in graded mode, because both levels land
  on the one node and the later entry overwrites the earlier — harmless under one-hot, where both
  wrote `1.0`, silently lossy under grading. The key path still qualifies and stays the tool for it.

- **`strict` governs names only, and is a field.** Unknown names warn with a `did_you_mean`
  suggestion over a pool widened to assets + group keys + taxonomy keys + taxonomy values + declared
  nodes, and throw under `strict`. Three structural refusals were offered and all declined: an
  all-zero row is legal, a one-column matrix is legal, and `allunique(vals)` is dropped because
  repeating a key is the whole point of last-wins. Only non-emptiness is unconditional — and a
  *malformed entry* throws regardless of `strict`, because a syntax error has no reading to fall back
  to. It is a field rather than a keyword because the producer interface is
  `feature_matrix(ze, pr, X, F, sets)`, with nowhere to pass one through; it appears as a keyword on
  the public `asset_sets_features`, which matters because that function is the only route to the
  default `z_src = :data` carrier, so a grammar landing on the producer alone would be reachable from
  one carrier and not the other.

### Consequences of the changes

- **One type now carries two contracts**, and the equal-row-norm identity holds on only one of them.
  Accepted, and chosen over a fifth producer, because the grammar strictly *subsumes* the key list:
  `"nx_sector" => 2.0` emits the same columns as `"nx_sector"`, only scaled, and an all-`1.0` program
  is bit-identical to stacking the same keys. The identity is not recoverable on the graded path by
  any knob — row-normalising is an exact no-op under the cosine family, and the identity needs `0`/`1`
  entries rather than merely equal norms.

- **The all-zero row becomes reachable for the first time.** Today every key is a partition, so every
  row has exactly `L` ones; an asset no entry touches is now all-zero, and this ADR's own zero-norm
  convention declares zero rows *mutually identical* — so forgotten assets cluster together at
  distance `0`. Opened deliberately, and documented rather than refused.

- **The transform question loses its *reweighting* instance, for this producer only.** Per-key and
  per-column weighting is now authored in `vals`, so "sector counts double" needs no transform at all.
  It does not generalise, and says so by its own shape: `Scale` resolves per cell against a natural
  value, whereas `idf` is data-dependent across a whole column and cannot be an `AbstractFeatureValue`
  at all, and an intensive/extensive tag is a per-column carrier schema no producer can supply.

- **`nz ⊃ nx` becomes reachable** — a **mixed** axis, part asset node and part taxonomy node, which
  neither the asset-keyed square producers nor the refitting ones had. `features_are_assets` is strict
  equality, so a mixed axis takes the *rectangular* path everywhere: a view slices rows only, and
  `collapse_feature_matrix` applies the one-sided `Wᵀ Z`, making an asset-node column read as "this
  cluster's weighted-average edge to asset A" while A may itself be in that cluster. Left alone, on
  the precedent that a synthetic asset's self-reference stays exactly as computed.

- **The pre-prior site is unaffected but asymmetric**: the third amendment's `ClusterGroups` reads
  `rd.Z` off the data carrier and carries no `z_src`, so a graded program reaches preselection only
  by way of the public `asset_sets_features` — never by way of the producer.

- Every `UniverseSets` doctest gains exactly one line and nothing re-indents: `zkey` is four
  characters and `uxkey`/`ufkey` already set the right-alignment width.

## Amendment (2026-08-04): the cross-validated collapse moves to the assembly seam

The second amendment recorded two things it treated as facts of life: that inside a fold there is only
one weight vector, so a square feature matrix keeps the real assets as its feature axis; and that one
combination — `NestedClustered` **with** cross-validation **and** a square matrix — drops the matrix
outright. Issue [#194](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/194) asked whether
that drop should be closed, and answered **yes — but the criterion that closes it is *path-consistency*,
not "features must survive folds"**.

The deciding fact is not that a matrix is lost. It is that **`cv` is execution control** (ADR 0030),
and toggling it silently changed the data the outer problem saw:

| | non-`cv` (`prepare_outer_rd`) | `cv` (`rebuild_returns_result`) |
| --- | --- | --- |
| `Stacking`, square `Z` | `N × N`, `nz` = synthetic names, two-sided | `T × N × nassets`, `nz` = **real** assets, one-sided |
| `NestedClustered`, square `Z` | `k × k`, `nz` = synthetic names, two-sided | **dropped** |

Same carrier, same shape, same producer, two outer semantics selected by an execution knob. So the
drop was never the whole defect — `Stacking` had the same defect in a milder form, and the second
amendment's "one combination" undercounted it.

**The collapse moves out of the fold and into the assembly seam.** `rebuild_returns_result` now makes
the *same* `collapse_feature_matrix(rd.Z, sq, W_f)` call `prepare_outer_rd` makes, once per fold, from
the original **unsliced** `rd.Z` — with `sq` from `features_are_assets` flowing through unchanged and
**no square branch**, since square indexes both trailing axes precisely because they are the same axis.
`W_f` is the `assets × sub-portfolios` weight matrix, zero-padded onto `cls[i]` for `NestedClustered`
and used directly for `Stacking`; the padding invents nothing, because `cls` partitions the universe.
The inner solves are untouched — each still sees its own cluster-sliced matrix, so the second
amendment's "measured on the subproblem's own neighbourhood structure" survives intact one level down.

Consequences worth stating.

- **The fourth carrier is deleted.** `nz`/`Z` leave `PredictionReturnsResult`, along with
  `fold_feature_matrix`'s vector arity, `mapreduce_FeatMtx`, and the `Z` computation in both
  `reconstruct_rd` arities. The argument is not "unused" but **"it is the one site where square
  cannot be treated like non-square"** — structurally, not by choice, since one weight vector cannot
  index both axes. Keeping the carrier meant keeping the sole square special case in the collapse.
  The third amendment's finding that it was write-only for a second, independent reason stands, and
  is now moot: there is nothing left to write.

- **Re-expanding each cluster's collapsed vector onto the full asset axis is dead by arithmetic**, not
  by the zero-row convention. `cls` is a disjoint partition, so cluster *i*'s re-expanded row is
  supported only on `cls[i]`; disjoint supports ⇒ every off-diagonal inner product is exactly `0` ⇒
  under `AngularDist` **every pair of synthetic assets sits at exactly `0.5`**. A constant distance
  matrix carries no information at all — this ADR's `P·Pᵀ - I` degeneracy arriving one level up, and
  worse, since there it was two distinct values and here it is one.

- **The fold's rows are recovered, not stored.** A time-varying `rd.Z` must be row-sliced per fold,
  and cumulative counts cannot stand in: `IndexWalkForward` and `KFold` test blocks do not start at
  row 1. Rather than add fold provenance to `PredictionResult`, the rows are recovered from the fold's
  **timestamps** — `port_opt_view` slices `ts` with the very `test_idx` the fold was built from, so a
  fold's `ts` *is* its slice of the clock, and `feature_row_indices` (already the price-level mechanism
  for exactly this) matches it back. Recovering beats storing on the combinatorial path in particular,
  where `sort_predictions!` assembles a path's folds in split order rather than chronologically: the
  timestamps carry whatever order actually happened, while re-deriving the split would have to
  reproduce it.

- **`ts` therefore *keys* the observation axis, and `ReturnsResult` now requires `allunique(ts)`.**
  A repeated timestamp resolves to its first occurrence and would pair an asset with another period's
  features. This is a **breaking** validation change, and library-wide rather than local to this seam,
  because the axis either is uniquely keyed or it is not. The clock requirement itself stays narrow:
  only a *time-varying* matrix needs rows at all, so a static one runs on fold sizes alone and a
  missing `ts` is refused only where it is actually load-bearing.

- **Two invariants the seam had always assumed are now asserted.** That every sub-portfolio's fold `f`
  covers the same observations — `reshape(X, :, N)` has depended on it since before this map, and
  `W_f` is only well defined if it holds; it matters most on the combinatorial path, where each
  sub-portfolio's `scorer` picks a path independently. And that the stacked rows really are the fold
  rows. Neither is introduced here; they are made visible.

- **`rebuild_returns_result` stacks into copies.** It used to `append!` into `predictions[1].mrd`'s own
  buffers, mutating the predictions and silently doubling the stacked height on a second call. Found
  by the row-count assertion above, and fixed at the source rather than worked around.

- **`cls` is a positional third argument, deliberately.** A keyword with a full-universe default would
  let a stale two-argument call keep working — correct for `Stacking`, and for `NestedClustered`
  silently writing every cluster's weights to the wrong rows, which by the arithmetic above produces
  not an error but a *plausible-looking* matrix. Same reasoning as `prepare_outer_rd` returning
  `nz`/`Z` *before* `X`: the break has to be arranged to be loud, and `Stacking`'s explicit `nothing`
  reads as a statement rather than an omission.

Two candidates #194 weighed and rejected, recorded because their bases were real. **Refusing the
combination at construction** is structurally impossible — `NestedClustered` construction never sees
`rd`, and squareness is `nz == nx`, knowable only once data arrives. **Accepting the drop** rested on a
verified fact: `z_src = :prior` on the outer optimiser refits `PhylogenyFeatures` on the synthetic
returns and yields a genuine `k × k` needing no carrier, so the only thing actually lost was the
*exogenous* square data carrier. That narrowness is why it was a fair candidate — but it answers "is
anything unreachable", not "does an execution knob change the data", and the latter is the criterion.
