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
[#184](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/184) records the same hole live
elsewhere in the library — `SemiDefinitePhylogenyEstimator` holds a precomputed `PhylogenyResult`
with no `@vprop`, so a three-asset subproblem gets an `8 × 8` constraint matrix without complaint.

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

The prior carrier cannot make that comparison — it holds no names at all (decision 6) — so it carries
an explicit `z_sq::Bool` stated by whoever built the matrix, validated to actually be square.

This is the one carrier shape where **`distance` does not commute with an asset view**. Slicing the
feature axis truncates every row's feature vector, so a subproblem is measured against its own
neighbourhood structure rather than against the whole universe. That is the intended reading, and it
is a real semantic difference from the rectangular case, which does commute.

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

The prior carrier is `Z` plus `z_sq::Bool` and **no `nz`**. A prior result has never carried names —
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
   already behind group constraints. Rectangular, `z_sq = false`.
4. **`PhylogenyFeatures`** — a square neighbourhood matrix from a network estimator or a precomputed
   `PhylogenyResult`, under `BinaryNeighbourhood` or `GradedNeighbourhood` decay. The only producer
   with `z_sq = true`, and the only exogenous route for *square* structure.

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
- **`LowOrderPrior` gains `Z` and `z_sq`, appended last.** `HighOrderPrior` needs no edit — its
  `@forward_properties` branch forwards any property. Every `LowOrderPrior(; …)` construction site
  forwards both, so nesting order does not matter.
- **A price-level `Z` is not on the master clock and cannot be** — no three-dimensional `TimeArray`
  exists — so it is held positionally parallel to `X`, and three sites re-establish that alignment by
  hand: `port_opt_view`, `MissingDataFilter`'s preprocessing, and `prices_to_returns`.
- **A meta-optimiser's outer problem has no features, deliberately and loudly.** All three
  synthetic-universe builders drop `Z`, because their assets are sub-portfolios, clusters or
  predictions rather than the universe. An outer estimator asking for a `FeatureDistance` gets an
  `IsNothingError`, never a stale full-universe matrix. Whether `Z` should instead be *collapsed* onto
  the synthetic assets is [#179](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/179).
- **`z_src = :data` slices, `z_src = :prior` refits.** Under a fold or a cluster the carried matrix is
  subselected while the derived one is recomputed on the subproblem's own returns — so the derived
  one's feature axis stays the full factor set while the carried square one's does not. The two
  selectors pick between two semantics, not merely two sources.
- **A time-varying literal `ze` cannot survive an observation fold** and throws at construction. Only
  a producer derives a per-fold time-varying `Z`.
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
