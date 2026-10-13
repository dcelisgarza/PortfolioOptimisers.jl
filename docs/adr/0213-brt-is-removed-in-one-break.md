---
status: accepted
---

# `brt` is removed in one break

## Context

ADR [0211](0211-each-reader-of-the-returns-states-its-own-source.md) removes `brt` from every owner
and puts `x_src` on each reader of the returns. ADR
[0212](0212-a-returns-source-is-a-singleton-type.md) gives `x_src` the values `nothing`,
`DataSource()`, `ExcessSource()` and `PriorSource()`. This ADR decides how the removal reaches a
caller: in one break, or through one release in which `brt` is a deprecated keyword that maps to
`x_src` on the readers (#1684).

The facts at `e7085bce58`:

- `brt` is a public keyword and a field of `JuMPOptimiser`, `HierarchicalOptimiser`,
  `InverseVolatility`, `NestedClustered`, `Stacking`, `SubsetResampling` and `PipelineStep`. It is a
  keyword of `covariance_forecast_evaluation`. The docstring keys `:brt` and `:brt_read` document it.
- `RiskLoss`, `ProgrammeAllocationSet` and `OnlinePortfolioSelection` hold a `brt` field too.
  `prior`, `clusterise`, `sigma_ucs` and other verbs take a `brt` keyword.
- The keyword constructors of the owners and `covariance_forecast_evaluation` take no `kwargs...`
  sink. A keyword that they do not declare is a `MethodError` that names the keyword. Some verbs
  that take `brt`, for example `clusterise` and `sigma_ucs`, also take a `kwargs...` sink and forward
  it to an inner method. This ADR did not measure where each sink ends.
- `CovarianceForecastEvaluationResult` takes `brt` by position, as its fifth argument. The struct is
  `@concrete` and the field has no declared type, so the generic positional constructor accepts any
  value in that position.
- `src/` holds no `@deprecate` and no `depwarn`. ADR
  [0044](0044-matrix-sources-are-named-not-flagged.md) replaced `cle_pr` by `x_src` as a clean break,
  with no deprecation path.
- `docs/src/migration.md` lists each break of a release, with the code to write in its place. It has
  one section per version.
- A head under `brt = true` reads `X - B` on every reader. ADR 0211 gives a head no source of its
  own, and an inner reader must keep `nothing`. So the replacement of one `brt = true` is one
  `x_src = ExcessSource()` on each outermost reader of the returns.

## Decision

The maintainer decided on 2026-10-13.

1. **`brt` goes in one break.** No release keeps `brt` as a deprecated keyword, and no code maps
   `brt = true` to `x_src` on the readers. Every owner loses the field and the keyword in the same
   release: the seven owners that ADR 0211 lists, `RiskLoss`, `ProgrammeAllocationSet`, every verb
   that takes the keyword, and the internal helpers. `OnlinePortfolioSelection` renames its `brt` to
   `x_src` (ADR 0211).
2. **The error of the language is the refusal.** A call that passes `brt` gets the `MethodError` of
   an unsupported keyword. No code adds a custom refusal that names `x_src`. A `kwargs...` sink must
   not swallow `brt`: each public verb that takes `brt` today gets a test that `brt = true` throws
   after the removal. Where a sink swallows it, the build makes the sink refuse an unknown keyword.
3. **The migration guide gives the replacement.** The section of `docs/src/migration.md` for the
   release that removes `brt` lists `brt` in its table of removed names. It shows the replacement on
   each kind of reader: a prior, a graph reader, a regression reader, the prior of an uncertainty
   set, and the step of `OnlinePortfolioSelection`. It also lists, under results that move, the
   change of the default of a graph reader from `pr.X` to `rd.X` (ADR 0211).
4. **A positional `Bool` does not build a Result.** #1683 decides how
   `CovarianceForecastEvaluationResult` records the frame. A caller that builds the Result by position
   with a `Bool` in the old position must get an error, not a Result. So the new record either has a
   type that refuses a `Bool`, or a different position or arity.

## Consequences

- The migration builds can land one at a time on `dev`. The release that ships the first of them
  ships all of them, so a caller meets the break once. No release is cut from `dev` while it holds
  some migration builds and not all.
- The docstring keys `:brt` and `:brt_read` go in the build that removes their last reader.
- No deprecation code is written, tested or removed later.
- A caller who upgrades gets an error at each `brt` and fixes it with the migration guide. A
  tracking-error `MeanRisk` with a prior, an uncertainty set and a centrality constraint changes one
  keyword into three (ADR 0211).
- The pins of `test/test_78_brt_pinned_outputs.jl` stay the oracle of each migration build.

## Alternatives considered

- **Deprecate `brt` for one release.** A head under `brt = true` warns, then rebuilds each outermost
  reader whose `x_src` is `nothing` with `ExcessSource()`. This is the default on the head that ADR
  0211 refused. It needs a rule of precedence over a reader that the caller stated, and a walk that
  finds the outermost reader of each slot, because an inner reader must keep `nothing`. A head
  under today's `x_src = :prior` also maps to `PriorSource()` on its graph readers, so the map
  reads two fields. The positional Result needs a second method beside the one that `@concrete`
  writes. All this code lives for one release, and the library has no deprecation code today.
- **Keep `brt` as an alias with no warning.** Two spellings of one frame then exist, and the frame
  check of #1682 must reconcile them.
- **A custom refusal of `brt` that names `x_src`.** The message helps, but a removed name keeps code
  and a test in each owner. The migration guide gives the same text in one place.

## Related

- ADR 0044: the clean break of `cle_pr`.
- ADR 0204: the current rule for `brt`.
- ADR 0211: the owners that lose `brt`.
- ADR 0212: the values of `x_src`.
- ADR [0215](0215-the-paths-that-read-the-frame-of-a-head-read-the-source-of-a-reader.md): the
  record of `CovarianceForecastEvaluationResult`, which decision 4 waits on (#1683).
- `docs/src/migration.md`, `CovarianceForecastEvaluationResult`, `covariance_forecast_evaluation`,
  `clusterise`, `sigma_ucs`.
