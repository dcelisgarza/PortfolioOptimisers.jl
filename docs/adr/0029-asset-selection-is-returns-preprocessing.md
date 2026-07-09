---
status: accepted
---

# Asset selection is a returns-preprocessing subfamily whose universe is fitted state

## Context

Callers want to narrow the asset universe from the data before a prior or an optimiser sees
it: drop constant columns, keep the best or worst *k* assets by some risk measure, prune
highly correlated assets. These are decoupled concerns that should compose with everything
else the way estimators already do.

Three facts shaped the design more than any preference.

**The seam already existed and was empty.** `AbstractReturnsPreprocessingEstimator` and
`AbstractReturnsPreprocessingResult` were declared in `03_Preprocessing.jl` with *no
concrete subtypes*. The pipeline already routed them (`run_step`, `pipe_reads`,
`pipe_writes`) and `predict` already replayed them (`apply_fitted_step`). A returns-level
selector therefore needs **zero new pipeline code** to be steppable, cross-validatable, and
tunable.

**The risk-measure family already answers "score one asset".** `supports_precomputed_returns`
(ADR 0018) declares whether a measure's functor accepts a bare return series; scoring asset
`i` is exactly `r(X[:, i])`. `bigger_is_better` already gives `:best`/`:worst` a
precise, orientation-aware meaning, and is already used that way when ranking
cross-validation predictions.

**The pre-filtering free functions were staging, not API.** `find_complete_indices` and
`find_uncorrelated_indices` were exported but called from nowhere in `src/`;
`select_k_extremes` was an unimplemented stub. They were written to be integrated.

## Decisions

1. **Asset selection is a returns-preprocessing subfamily, not a new stage.**
   `AbstractAssetSelector <: AbstractReturnsPreprocessingEstimator`. One shared
   `AssetSelectorResult` carries the fitted universe `nx`; one shared `fit_preprocessing`
   and one shared `apply_preprocessing` serve the family. A concrete selector implements a
   single method, `select_assets(sel, rd) -> BitVector`.

2. **Selectors restrict columns only.** Observation filtering stays at the price level
   (`MissingDataFilter.row_thr`). A fitted selector cannot coherently decide *which rows* of
   an unseen window to drop, and doing so would break the weights/returns alignment
   `assert_universe_aligned` enforces. This is why `find_complete_indices`'s `dims = 2`
   (complete-row) mode has no estimator form.

3. **`apply_preprocessing` is strict and order-preserving.** The surviving columns are
   emitted in *fitted* order, because `assert_universe_aligned` compares `rd.nx == train.nx`
   elementwise. A fitted asset absent from the window throws rather than silently shrinking
   the universe.

4. **Scoring reuses the risk-measure family; no new score taxonomy.**
   `ScoreSelector(; score::AbstractBaseRiskMeasure, rule::AbstractSelectionRule)`. The
   constructor rejects a `score` whose `supports_precomputed_returns` is false. The cost is
   real and accepted: `Variance` is a `WeightsInput` measure, so zero-variance filtering
   must be spelled `SCM()`. The rejection message says so, and `ZeroVarianceFilter` is an
   alias that spells it for the caller.

5. **Rules split literal from ordinal.** `ThresholdRule(; lo, hi)` is an absolute band and
   ignores `bigger_is_better` — a trait-aware "keep the better ones" would invert a
   zero-variance filter, whose whole point is to drop the *low*-variance assets.
   `RankRule(; best, worst, action)` and `QuantileRule(; best, worst, action)` consult
   `bigger_is_better` and take **counts (or fractions) from each end**, not positions. Counts
   from each end make "drop the worst 5" expressible without knowing `n`, and make both tails
   selectable at once; positions would not. `RankRule` and `QuantileRule` are separate types
   because `best = 1` (one asset) and `best = 1.0` (the whole universe) must not differ only
   by a literal's type.

6. **Counts saturate; empty selections throw.** `best = 50` on a 30-asset window keeps all
   30 rather than erroring, so a hyperparameter search over `best` is not killed by its
   largest point. Every other degenerate case fails closed: an unscoreable measure, a rule
   that selects nothing, a non-finite score, an empty keep-set, a fitted name missing at
   apply time.

7. **Ties are excluded, not split — "if we cannot tell them apart, trust neither".**
   `find_uncorrelated_indices` already removed *both* assets of an exactly-tied correlated
   pair; that stance is adopted library-wide rather than reverted. A tied block straddling a
   rank cut is dropped entirely, so `RankRule(; best = 20)` may return fewer than 20 assets,
   and a window of identical scores selects nothing (and therefore throws). The alternative —
   breaking ties by column index — would make the result depend on column order, which is not
   a property of the data.

8. **The free functions become the estimators' unexported internals.** `select_k_extremes`
   is deleted (it becomes `ScoreSelector` + `RankRule`). `find_complete_indices` and
   `find_uncorrelated_indices` lose their exports and keep their implementations, wrapped by
   `CompleteAssetSelector` and `RedundancySelector` respectively. This is a breaking change
   to the exported surface; neither name was reachable from any other part of the library.

9. **Redundancy is one selector with a pluggable grouping algorithm.**
   `RedundancySelector(; alg, score)`. The seam is `redundancy_keep(alg, rd, scores, bib) ->
   BitVector`, a *keep-mask* rather than a partition, because greedy pairwise pruning drops one
   asset at a time and may keep two members of the same correlated blob — which
   partition-then-argbest cannot express. `groups_argbest` is a helper the two partitioning
   algorithms share, not the seam.

   `PairwiseCorrelation` delegates to the existing `find_uncorrelated_indices`, which grows an
   optional `scores` keyword (higher means "drop me"); absent it, today's mean-|ρ| survivor rule
   is unchanged, so the existing behaviour and its test are preserved verbatim.
   `CorrelationComponents` reads the same correlations *transitively* and gives a different,
   stronger answer: a chain `A ~ B ~ C` with `A ⊥ C` is one blob keeping one survivor, where
   greedy keeps both `A` and `C`. Greedy is the default because it honours the literal promise
   the threshold makes. `ClusterGroups` has no fallback survivor rule, so `requires_score`
   is true for it and the constructor rejects a `nothing` score.

10. **Concrete selectors live in `src/24_AssetSelection.jl`; the seam stays in
    `src/03_Preprocessing.jl`.** The seam depends on nothing but `ReturnsResult`. The
    concretes need covariance estimators (include #75), clustering (#116), and risk measures
    (#149), and the library's `@concrete struct` idiom evaluates its inner constructor's type
    annotations at definition time — so they physically cannot sit beside `MissingDataFilter`
    and `Imputer` at include #3. Nothing forces them to precede `23_Pipeline/`, since
    `run_step` dispatches on the abstract type. This is the exception to ADR 0028's post-M6
    amendment, which placed preprocessing in `03_Preprocessing.jl`.

## Alternatives considered

**A `port_opt_view`-free `subselect_assets` helper.** Unnecessary: `returns_result_view` —
now `port_opt_view(rd, i)` after ADR 0010's amendment — already sub-selects `nx`, `X`, `nb`,
`B`, `iv`, and `ivpa` correctly.

**Standalone estimators per filter.** `ZeroVarianceFilter` is a special case of
`ScoreSelector`, and the correlation and cluster filters share "partition, then keep the
survivor". Separate types would re-encode the score-and-rank logic three times.

**A new `AbstractAssetScore` taxonomy.** `RiskMeasureScore(rm)` would be a pure wrapper
carrying no information, and the risk-measure family already spans variance, semi-variance,
VaR, CVaR, drawdowns, and mean return.

**Groups as the shared seam for redundancy.** Pairwise-greedy pruning cannot be expressed as
partition-then-argbest: it drops one asset at a time and may keep two members of the same
connected component. The seam is the *keep-set*; "partition into groups, keep the argbest of
each" is a helper two of the three redundancy algorithms happen to use.

## Consequences

- `Pipeline` needed a companion fix (ADR 0028 amendment): a step that rewrites `:returns`
  after a prior/phylogeny/uncertainty/constraint step leaves that slot asset-misdimensioned.
  Selectors made that latent hazard reachable, so construction now rejects it.
- `RankRule(; best = k)` returning fewer than `k` assets is documented behaviour, not a bug.
- New concepts `Asset Selector`, `Selection Rule`, and the trust-neither tie policy are in
  `CONTEXT.md`.
- `PairwiseCorrelation` and `CorrelationComponents` give *different answers on the same input*
  by design. The docstrings state the chaining difference at the call site rather than burying
  it, because a caller who sets `thr = 0.95` expecting a pairwise guarantee would read
  component semantics as a bug.
- `find_complete_indices` and `find_uncorrelated_indices` are no longer exported.
- Worked example at `examples/1_foundations/03_Asset_Selection.jl`, which demonstrates the
  greedy/transitive divergence on the real SP500 slice rather than on a contrived matrix.
- `absolute_drawdown_vec` and `relative_drawdown_vec` seeded their running peak by mutating
  the caller's vector (`pushfirst!` / compute / `popfirst!`). Scoring an asset column
  surfaced it: a view cannot be resized. They now carry the peak in a scalar, so they read
  `x` and never write it, and any `AbstractVector` works. Fixed as a prerequisite rather
  than worked around in `asset_scores`.
