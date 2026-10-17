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

The maintainer decided on 2026-10-13. Decision 5 came on 2026-10-14 (#1693), when the migration
was split into builds.

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
   each kind of reader: a prior, a graph reader, a regression reader and the prior of an uncertainty
   set. It also lists, under results that move, the change of the default of a graph reader from
   `pr.X` to `rd.X` (ADR 0211). The guide lists the breaks against the last release. `v0.32.0` had
   `brt` on six heads and on `returns_result_picker` alone, so a `brt` that came after it and goes
   before the next release has no row: `OnlinePortfolioSelection`, `RiskLoss`,
   `ProgrammeAllocationSet`, `PipelineStep` and the keyword of each verb. A test of decision 2 still
   covers each of them.
4. **A positional `Bool` does not build a Result.** #1683 decides how
   `CovarianceForecastEvaluationResult` records the frame. A caller that builds the Result by position
   with a `Bool` in the old position must get an error, not a Result. So the new record either has a
   type that refuses a `Bool`, or a different position or arity.
5. **The migration reaches `dev` in one merge (#1693).** The pieces of the migration meet on one
   integration branch, `migration-1678`, which starts from `dev`. Each piece is a session in its own
   worktree. It owns whole files, the test files of its area included, so no two parallel pieces edit
   one file. A piece rebases onto the integration branch and fast-forwards into it, and the branch
   can be red between two pieces. A foundation piece goes first and alone. It adds the values of ADR
   0212, the shared assertion, the refusal of an inner reader, the verb of the frame walk of ADR
   [0214](0214-the-readers-of-one-head-can-read-different-frames.md) with its default method, and
   every new docstring key. Each later piece adds the walk methods of the types that it owns. A last
   piece rebases the integration branch onto `dev`, runs the pins and the census, and fast-forwards
   `dev`. Only that piece rebases the integration branch.

## Consequences

- `dev` never holds a part of the migration, so a release cut from `dev` has all of it or none of
  it, and a caller meets the break once.
- Inside a piece, no head forwards `brt` to a migrated reader, and no rule of precedence joins the
  two. No transitional code is written.
- A piece cannot run the tests of a head until the pieces that it calls are in. The last piece runs
  the tests and repairs a failure, and its report names the piece at fault.
- The docstring keys `:brt` and `:brt_read` go in the last piece, which removes their last reader.
- No deprecation code is written, tested or removed later.
- A caller who upgrades gets an error at each `brt` and fixes it with the migration guide. A
  tracking-error `MeanRisk` with a prior, an uncertainty set and a centrality constraint changes one
  keyword into three (ADR 0211).
- The pins of `test/test_78_brt_pinned_outputs.jl` stay the oracle of the migration. The last piece
  closes only when they pass unchanged.

## Alternatives considered

- **Builds that land one at a time on `dev`, each green.** A migrated reader reads only its own
  `x_src`, and the head keeps `brt` for the readers that are not migrated yet. Each pin run tests
  the new spelling of one family. But every build moves the test sites of its family while the
  head still holds `brt`, and the head paths of ADR 0215 read two frames until the last build.
- **A migrated reader at `nothing` reads `X - B` under a head with `brt = true`, on `dev` only.**
  No helper changes until the last build, so the pins test the new spelling only once. It is the
  rule of precedence that this ADR refused for a release.
- **One integrator.** Each piece is a branch from `dev` that nobody merges, and one session merges
  them all. The pieces do not see each other's work, so two pieces can write two forms of one
  helper.
- **The frame walk in one file**, as `feature_readable_mask` is. One piece must then name the inner
  fields of readers that other pieces own.
- **The last piece moves every old test site.** That piece then carries 24 test files, `test_78`,
  the docs and the run, which is more than one session.

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
