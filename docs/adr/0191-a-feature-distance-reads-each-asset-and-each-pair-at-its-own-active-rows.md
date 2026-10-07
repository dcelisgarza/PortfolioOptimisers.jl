---
status: accepted
---

# A Feature Distance reads each asset and each pair at its own active rows

## Context

A `FeatureDistance` measures the Feature Matrix that it stacks from an Asset Panel. A numeric
Panel Field keeps a finite value in every cell, the inactive cells included (ADR 0102), and that
value is arbitrary: a fill value, a zero, or the last value before a delisting. The census of
issue #1411 wrote `1e6` into every inactive cell and found that every collapse of a
`FeatureDistance` read it. `LastObservation` read the last row, where a delisted asset holds its stored value,
and the two aggregates and `StackObservations` read every row of the window.

Issue #1450 measured the reach. A default prior drops every asset that is not active over its
whole window, so no default optimisation reads an inactive cell. A Coverage Policy admits a
partial asset, and then the poison moved the distance that an HRP clustering reads by up to
0.18 (`StackObservations`), 0.76 (`AggregateFeatures`), 0.27 (`AggregateDistances`) and 0.91
(`LastObservation` under `ExpireCoverage`). `Clustering.hclust` refuses a `NaN` entry with a
message that names no asset, so a `NaN` for a missing value is no remedy.

## Decision

**A `FeatureDistance` reads each asset, or each pair of assets, only at its own readable rows.**
A row is readable for an asset where the asset is active and every value column of the Feature
Selector was observed (#1508), because the panel stores a placeholder in an unobserved cell, not
data. This is available-case estimation, as a Coverage Policy fits a covariance cell on the rows
at which both of its assets are active. The maintainer decided each part in a grilling session on
issue #1450, and #1454 built it.

| Part | Rule |
| --- | --- |
| `LastObservation` | A field `alg` holds a rule. `LastRow()`, the default, reads the last row, and an asset that is not readable there has no value to read. `LastActiveRow()` reads each asset at its last readable row of the window. |
| `AggregateFeatures` | Each asset over its readable rows. `w` is resolved once over the window, restricted to the asset's readable rows and divided by their sum. `MedianCollapse` takes the same weights. |
| `AggregateDistances` | Each pair over the rows at which both assets are readable, the weights restricted and divided by their sum. |
| `StackObservations` | Each pair stacks its `n` shared readable rows, and `stack_rescale` rescales the distance to the `T` rows of the window, one method per metric (below). |
| An asset with no value to read | Inside a fit, the entry composes the Investable Mask of the prior with the assets that each `FeatureDistance` can read (`feature_readable_mask`), before `investable_reduction`. The asset departs as a non-investable asset: announced, dropped, weight zero, and listed on the Non-Investable Axis. A direct `distance`, `cor_and_dist` or `clusterise` call refuses, and names the asset and the remedy. |
| A pair with no shared readable row | A field `pair` on `AggregateDistances` and `StackObservations`. `RefusePair()`, the default, refuses and names the pair. `DropFewerRows()` drops, at the entry of a fit, the asset of the first empty pair with fewer readable rows (the later asset on a tie), until no pair is empty. `FeatureFallback(; alg = MeanCollapse(), w = nothing)` measures the pair by each asset's features collapsed over its own readable rows. |

**The rescale of the stack.** A metric that sums over the coordinates sums over fewer of them
when a pair shares fewer rows, so the stack scales the sum back to the whole window, the rule of
R's `stats::dist` for a missing coordinate:

| Metric | Rescale |
| --- | --- |
| `Cityblock`, `TotalVariation`, `SqEuclidean`, `ChiSqDist` | `d · T / n` |
| `Euclidean` | `d · sqrt(T / n)` |
| `Minkowski(p)` | `d · (T / n)^(1 / p)` |
| `WeightedEuclidean`, `WeightedSqEuclidean`, `WeightedCityblock`, `WeightedMinkowski` | the ratio of the weight sums in place of `T / n` |
| `AngularDist`, `CosineDist`, `CorrDist`, `Jaccard`, `BrayCurtis`, the four mean deviations | none |

Any other metric refuses a pair with `n < T`, and the message names the method to add. #1450
simulated the stack against complete latent data (200 draws, AR(1) features, the gaps of the
fixture): the shared rows with no rescale were biased by −16.7 % (Euclidean) and −29.5 %
(Cityblock), and with the rescale by −0.2 % and +0.2 %.

**The fallback keeps one scale.** Under `FeatureFallback` each asset of an empty pair holds its
collapsed features at every row of the window. `AggregateDistances` then reads the metric of the
two collapsed vectors, the value that every row gives, and `StackObservations` reads the stack
of `T` equal rows, which a sum metric scales by `T` as its rescale at `n = 1` does. So the
fallback needs no rescale method, and a weighted metric keeps the length of its weights.

**A window with no unreadable cell takes the old path.** The kernel receives the mask of the
readable cells of the stacked rows, and a mask with no `false` entry reads as `nothing`. So every default result
is unchanged bit for bit. #1454 compared 40 results of the old and the new code (direct
distances of each collapse under three metrics, HRP and HERC fits under the default prior).

**Where the entry composes the mask.** `HierarchicalRiskParity` (two sites),
`HierarchicalEqualRiskContribution`, `SchurComplementHierarchicalRiskParity` (two sites),
`NestedClustered` for its own clustering, and a `JuMPOptimiser` through its phylogeny and
centrality estimators. `Stacking` and `SubsetResampling` cluster only inside their inner
optimisers, and each inner optimiser composes the mask at its own entry. So a member of a
stack that reads no feature keeps the asset that a clustering member drops.

## Consequences

- A prior whose covariance needs a shared row refuses an empty pair before the entry sees it, so
  `DropFewerRows` and `FeatureFallback` act in a fit only under a prior that does not need one,
  for example a precomputed prior result.
- `LastObservation`, `AggregateDistances` and `StackObservations` print one more field.
- The panel collapse of a meta-optimiser is its own decision, #1451.
