---
status: accepted
---

# Each reader of the returns states its own source

## Context

Two pickers choose the returns that a reader fits on. `x_src::Symbol` (ADR
[0044](0044-matrix-sources-are-named-not-flagged.md)) is a field of `JuMPOptimiser`,
`HierarchicalOptimiser` and `NestedClustered`. It picks `pr.X` or `rd.X` for a clustering, phylogeny
or centrality estimator. `brt::Bool` (ADR
[0204](0204-a-returns-result-drops-no-field-that-the-mathematics-lets-it-carry.md)) is a field of
eight owners. It makes each reader of the returns read `X - B`. Both pickers live on a head, and each
head forwards them as keywords to the readers that it calls: `prior(pe, rd; brt)`,
`returns_matrix_picker(pr, rd, x_src, brt)`, `ucs(uc, rd; brt)`, `resolve_factor_regression(re, rd,
pr; brt)` and the graph verbs.

The forwarding has a cost that ADR 0204 states: a reader that a later change adds must take `brt`, or
it reads the raw `X` under `brt = true`. #1651, #1659 and #1670 each found a reader that did not.
The map "`x_src` takes over `brt`" (#1678) moves the frame into `x_src`, so that each reader states
which returns it reads, and the excess returns `X - B` are one of the values. This ADR decides where
`x_src` lives (#1680). The values, their type and their names are #1681, in ADR
[0212](0212-a-returns-source-is-a-singleton-type.md). The check of a mix of frames
is #1682. The online step, the outer problem of a meta-optimiser and the covariance forecast
evaluation are #1683.

The census at `89e8d37a72` found these facts.

- About 30 concrete reader types: 13 priors, 7 graph readers, 4 uncertainty sets that hold a `pe`,
  and 2 regression estimators.
- The readers nest. `FactorPrior` holds `pe` and `re`, an uncertainty set holds `pe`, and a
  `CentralityConstraint` holds a `CentralityEstimator` that holds a network estimator. Only the
  outermost reader of a slot receives `rd`. Each inner reader receives a matrix from it. This is
  true of the `pl` of a `PhylogenyPanel` too: it reads the `X` that the outer reader picked.
- `resolve_factor_regression` takes `pr.rr` when the prior holds loadings, and refits `re` on `rd`
  only when it holds none.
- Under a `FactorPrior`, `pr.X` is the reconstruction `F * M' + b`, not the raw returns. A scenario
  window and the drop of a Shared Series gap also cut the rows of `pr.X`.
- The step of `OnlinePortfolioSelection` reads the price relative itself. Its 21 rules are
  algorithms, not estimators, and each one receives a matrix of rows from the step.

## Decision

The maintainer decided on 2026-10-13.

1. **Each reader estimator holds an `x_src` field, and the default is `nothing`.** A verb reads the
   field of its estimator, so `prior(pe, rd)` reads `pe.x_src`. No head forwards a source, and no
   verb takes a `brt` or an `x_src` keyword.
2. **The holders.** The 13 prior estimators. The 7 graph readers: `ClustersEstimator`,
   `NetworkClustersEstimator`, `NetworkEstimator`, `CentralityEstimator`,
   `SemiDefinitePhylogenyEstimator`, `IntegerPhylogenyEstimator` and `CentralityConstraint`. The 2
   regression estimators, `StepwiseRegression` and `DimensionReductionRegression`. And
   `OnlinePortfolioSelection`, for its own step (decision 7).
3. **An inner reader keeps `nothing`.** An estimator that holds a reader which receives only a matrix
   refuses a non-`nothing` `x_src` on that reader in its constructor. The error names the field and
   tells the caller to state the source on the outer reader. Examples: the `pe` and `re` of a prior,
   the `A` of a `CentralityConstraint` and its `pl`, the `nte` of a `NetworkClustersEstimator`, the
   `pl` of a `PhylogenyPanel`. The default `nothing` means "not stated", so the refusal has no
   ambiguous case. The configuration alone decides the refusal, so it goes in the constructor.
4. **A holder of one prior passes the returns to it.** An uncertainty set, a `RiskLoss` and a
   `ProgrammeAllocationSet` hold no field. Each one gives `rd`, or its rows, to its own `pe`, and that
   prior states the source. A set with `pe = nothing` reads the Prior Result, so it has no field to
   misplace. `RiskLoss` and `ProgrammeAllocationSet` lose their `brt` field. Each graph slot and each
   exposure slot of a `ProgrammeAllocationSet` states its own source, as the same slot of a
   `JuMPOptimiser` does.
5. **The meaning of `nothing` on the outermost reader.** A prior and a graph reader read `rd.X`, and a
   graph reader reads `pr.X` only when no `rd` is present. A regression reader takes `pr.rr` when the
   prior holds loadings, and otherwise refits on `rd.X`. The two families differ for a reason.
   `pr.rr` is the model that `pr.sigma` comes from, so a factor risk contribution and an exposure
   constraint decompose the covariance that the head optimises. `pr.X` under a factor prior is a
   reconstruction of rank `nf`, so a clustering on it sees only the factors.
6. **No head keeps a source field.** `JuMPOptimiser`, `HierarchicalOptimiser` and `NestedClustered`
   lose `x_src`. Every owner of `brt` loses it: `JuMPOptimiser`, `HierarchicalOptimiser`,
   `InverseVolatility`, `NestedClustered`, `Stacking`, `SubsetResampling` and `PipelineStep`. A head
   gives no default to the readers whose field is `nothing`. The caller states the source on each
   reader. #1682 decides whether a reader that the caller forgot, which reads `rd.X` beside a reader
   of `X - B`, is refused. The record of `CovarianceForecastEvaluationResult` is #1683.
7. **`OnlinePortfolioSelection` keeps a source for its own step.** Its `brt` becomes `x_src`. The
   head is the estimator in the selection that receives `rd`, and its step reads the price relative.
   Each inner reader of its rule (a forecaster, the window of `AntiCorrelation`, the row selector of
   `FollowTheLeader`, the scale of `ForecastReversion`) reads the frame of the step through
   `head_frame_returns`, as today. A rule holds no field, so a rule and its sub-rules always read one
   frame. The experts of an `ExpertMixture` therefore earn comparable returns. This is not a default
   for other readers: a leader, a `RiskLoss` and a `ProgrammeAllocationSet` read the rows as returns
   data, so their priors state their own source.

## Consequences

- The obligation of ADR 0204 goes away: a reader that a later change adds reads its own field, so no
  head has to remember to forward a keyword.
- About 30 types gain a field, and each docstring and each printed doctest gains a row.
- About 20 nesting fields gain a check through one shared helper.
- A tracking-error `MeanRisk` with a prior, an uncertainty set and a centrality constraint states
  `ExcessSource()` in three places, where `brt = true` was one.
- **The default of a graph reader changes from `pr.X` to `rd.X`.** The clusters of a graph reader
  move under a factor prior, a scenario window or a gap. The parity rule of #1678 says that a
  migration build closes only when the pins of `test/test_78_brt_pinned_outputs.jl` pass unchanged.
  So this change is its own build with new pins, and it lands before the migration builds.
- The estimators of a covariance forecast evaluation receive a matrix, so they get no field. #1683
  decides where the verb gets the frame.
- ADR 0204 describes the head-level `brt` that this decision removes. The map supersedes it when the
  migration builds land.

## Alternatives considered

- **A wrapper per family**, for example `SourcedPrior(pe; x_src = ExcessSource())`. The reader types stay
  unchanged, but each family gets one more type, each bridge unwraps it, and a wrapper can sit in an
  inner slot where it does nothing.
- **One source field per reader slot on the head**, for example `pe_src` and `cte_src`. An uncertainty
  set inside a risk measure or a constraint has no slot on the head, the direct verbs keep a keyword,
  and the obligation of ADR 0204 stays.
- **Ignore the field of an inner reader and document it.** No code, but
  `FactorPrior(; pe = EmpiricalPrior(; x_src = ExcessSource()))` fits the raw returns with no error.
- **Let the inner value govern.** The outer reader passes `rd` down. One prior can then fit its
  regression on `X` and its covariance on `X - B`, the defect class of #1651 inside one estimator.
- **The field on the uncertainty set.** The source is then stated in two places in the type tree, and
  one of them must always stay empty.
- **`nothing` keeps `pr.X` on a graph reader.** No output changes, but the default reads a
  reconstruction under a factor prior.
- **`nothing` refits a regression on `rd.X`.** The loadings can disagree with `pr.sigma`.
- **A default on the head**, `JuMPOptimiser(; x_src = ExcessSource())`, that fills each reader whose field is
  `nothing`. It is the head-level `brt` under a new name, it needs a rule of precedence, and each head
  forwards the value again.
- **The field on each online rule.** 21 rule types get it, a rule inside a rule over rules must keep
  `nothing`, and the source is stated where it cannot differ from its sub-rules.
- **The field on each inner reader of an online rule.** These readers receive a matrix, so `rd` must
  pass down, and the step still needs a frame of its own.

## Related

- ADR 0044: the rule that named `x_src`. Amended on 2026-10-13 to point here.
- ADR 0204: the current rule for `brt`.
- ADR 0208: `DataTracking`, which refuses the Excess Frame.
- ADR [0213](0213-brt-is-removed-in-one-break.md): `brt` is removed in one break, with no
  deprecation.
- ADR [0214](0214-the-readers-of-one-head-can-read-different-frames.md): the readers of one head
  can read different frames, and a forgotten raw reader warns (#1682).
- ADR [0215](0215-the-paths-that-read-the-frame-of-a-head-read-the-source-of-a-reader.md): the online
  fold, the meta outer problem and the forecast evaluation (#1683). `covariance_forecast_evaluation`
  takes an `x_src` keyword, the one exception to decision 1.
- `returns_matrix_picker`, `benchmark_relative_returns`, `resolve_factor_regression`,
  `head_frame_returns`.
