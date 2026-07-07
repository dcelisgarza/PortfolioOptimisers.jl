# Implementation plan — Pipeline workflow estimator

Companion to [ADR 0028](../adr/0028-pipeline-workflow-estimator.md), which records the agreed
design and rationale. Glossary terms: `CONTEXT.md` → Pipeline, Pipeline Context, Preprocessing
Estimator, PricesResult. All code lives in `src/25_Pipeline/` (included after `24_Aliases.jl`
or renumbered as appropriate); new estimators/results follow the house conventions in
`.github/prompts/add-estimator.prompt.md` / `add-result.prompt.md` (field_dict docstrings,
`@argcheck` validation, `@concrete`, pretty-printing via the abstract trees).

Milestones are ordered so each is independently mergeable and testable.

## M1 — Foundations: types, `PricesResult`, context, slot traits

`src/25_Pipeline/01_Base_Pipeline.jl`

- `abstract type AbstractPipelineEstimator <: AbstractEstimator`,
  `AbstractPipelineResult <: AbstractResult`,
  `AbstractPreprocessingEstimator <: AbstractEstimator`.
- `PricesResult <: AbstractResult`: bundles the `prices_to_returns` inputs — asset `TimeArray`
  plus optional factor/benchmark/iv `TimeArray`s and `ivpa` — with timestamp-alignment
  validation (mirror `ReturnsResult`'s checks at price level).
- Timestamp-window slicing: `prices_view(pr::PricesResult, window)` (contiguous
  timestamp/index range → sliced `PricesResult`). This is the pipeline-CV analogue of
  `returns_result_view`.
- `PipelineContext`: internal struct with `Option` slots — `prices`, `returns`, `prior`,
  `phylogeny`, `uncertainty` (collection), `constraints` (collection), `opt`
  (terminal optimisation result). Not exported, not user-facing.
- Slot traits by dispatch on the existing taxonomy:
  `writes_slot(::Type{<:AbstractPriorEstimator}) = :prior` (reads `:returns`), phylogeny
  estimators → `:phylogeny`, uncertainty-set estimators → `:uncertainty`, constraint
  estimators → `:constraints`, preprocessing estimators → `:prices`/`:returns`,
  `OptimisationEstimator` → `:opt`.
- `PipelineStep` explicit wrapper (escape hatch): user-supplied reads/writes plus a routing
  annotation for cases dispatch cannot disambiguate (notably mu-vs-sigma uncertainty targets).

Tests: `PricesResult` validation; `prices_view` window semantics (inclusive bounds, iv/ivpa
alignment); slot-trait coverage for every steppable family.

## M2 — Preprocessing estimators + fit/apply contract

`src/25_Pipeline/02_Preprocessing_Steps.jl` (split into files per estimator if they grow)

- The step execution contract: `run_step(est, ctx) -> (fitted_result, ctx′)` dispatching to
  each family's native verb (`prior`, `clusters`, constraint generation, `optimise`, …), and
  the apply contract for prep results: `apply_step(fitted_result, data) -> data′`.
- `PricesToReturns` estimator (ret_method simple/log, padding, collapse args) wrapping the
  existing `prices_to_returns` internals ([src/03_Preprocessing.jl:586]). Stateless: its
  fitted Result is the config; apply = run. The existing function stays unchanged.
- `MissingDataFilter`: the `missing_col_percent` / `missing_row_percent` logic extracted from
  `prices_to_returns` kwargs into a tunable estimator. Fitted Result **carries the surviving
  asset universe**; apply subsets unseen windows to that universe (train decides, test
  follows — this is the step that proves universe-as-fitted-state).
- `Imputer`: wraps Impute.jl (already a dependency). Fitted Result carries per-column
  imputation parameters computed on the train window; apply imputes unseen windows with the
  *train* parameters (the leakage-prevention exemplar).

Tests: each estimator standalone; a leakage regression test — impute a test window and assert
the fill values come from train statistics, not test statistics.

## M3 — `Pipeline` struct + `fit`

`src/25_Pipeline/03_Pipeline.jl`

- `Pipeline` struct: tuple of steps with optional names via Pair syntax
  (`"impute" => Imputer(…)`); auto-names derived from the written slot, suffixed on repeats
  (`prices_1`, `prices_2`, per ADR 0015's suffix convention).
- Construction-time validation: every step's read slots must be writable by an earlier step
  or fillable from the declared input; duplicate-name rejection; injection type checks where
  static (e.g. high-order prior vs optimiser requirements). A terminal optimiser is **not**
  required (a prior-only pipeline is legal; `predict` is what needs weights).
- `StatsAPI.fit(pipe::Pipeline, data) -> PipelineResult`: fill the context slot matching the
  input type (`PricesResult` → `:prices`, `ReturnsResult` → `:returns`), walk steps
  left-to-right via `run_step`, record each fitted Result under its step name.
- **Injection** (immediately before the optimisation step runs): computed slots override the
  optimiser's internal config, implemented with Accessors/factory on known field paths —
  `:prior` → `pe` (identity pass-through `prior(::AbstractPriorResult)` already exists),
  `:phylogeny` → `cle` on `HierarchicalOptimiser` / phylogeny-constraint fields on
  `JuMPOptimiser`, `:uncertainty` elements routed by type: mu sets → `ArithmeticReturn.ucs`,
  sigma sets → `UncertaintySetVariance.ucs` (both fields already accept pre-built sets; reuse
  the existing `ucs_selector`/factory threading), `:constraints` elements routed by Result
  type to `JuMPOptimiser` fields (`lcse`, `wb`, `sets`, cardinality/threshold fields).
  Unroutable elements are a construction- or fit-time error naming the offending type.
- `PipelineResult <: AbstractPipelineResult`: named per-step fitted Results, final context,
  terminal optimisation result, `weights` accessor.
- Nesting: a `Pipeline` is itself a steppable estimator (writes its terminal slot);
  meta-optimisers are valid optimisation steps. Wrapping a `Pipeline` *in* a meta-optimiser
  must fail with a clear error message (future expansion — see ADR 0028).

Tests: end-to-end fit at prices level and at returns level (skipping price stages); override
semantics (pipeline prior beats optimiser default `pe`; absent step falls back); shared prior
computed once (count invocations); nested pipeline; meta-optimiser-as-step.

## M4 — `predict` + input-window cross-validation splits

`src/25_Pipeline/04_Prediction_CV.jl`

- `StatsAPI.predict(pr::PipelineResult, data, window)`: apply the fitted prep Results in step
  order to the test window (universe subset → imputation → returns conversion), then delegate
  to the existing weights-level prediction so the return type is the same `PredictionResult`
  family the scorers already consume. Errors clearly if the pipeline produced no weights.
- Window-based splitting for `PricesResult`: adapt `KFold` (contiguous folds + purging /
  embargo re-expressed in input rows) and `WalkForward` (`IndexWalkForward` /
  `DateWalkForward`; date alignment via the existing `DateAdjusterEstimator`). Splits are
  contiguous timestamp windows; account for `prices_to_returns` consuming one leading
  observation per window when sizing train/test.
- `CombinatorialCrossValidation` on pipelines: explicit unsupported error (stays
  returns-level).

Tests: predict-vs-manual equivalence (fit on train slice by hand, compare); fold arithmetic
(window boundaries, purge/embargo, the T→T−1 row contraction); date-based walk-forward on a
`PricesResult` with irregular calendars.

## M5 — Search-CV integration (the driver)

`src/25_Pipeline/05_SearchCrossValidation.jl`

- `search_cross_validation(pipe::Pipeline, gscv::GridSearchCrossValidation, data)` plus the
  pipeline `fit_and_score`: split the input into windows, per candidate apply the lens grid
  to the pipeline, `fit` on train window, `predict` on test window, score with the existing
  risk-measure/scorer machinery (`expected_risk`, `bigger_is_better`, `HighestMeanScore`,
  train-score option) — no scorer changes.
- Lens addressing: extend `parse_lens` so a leading step name resolves to the step's position
  (name → index → property path); bare integers keep working. Structural search (estimator
  objects as grid values) needs no work — verify with a test swapping `EmpiricalPrior()` vs a
  factor prior.
- `RandomisedSearchCrossValidation` delegates to the grid form exactly as today.

Tests: tune a prep hyperparameter (imputation method) jointly with an optimiser
hyperparameter; name-addressed vs index-addressed lenses give identical results; a
leakage-sensitive tuning case where full-sample preprocessing would pick a different winner.

## M6 — Exports, docs, precompilation, polish

- Exports + `24_Aliases.jl` entries; `public` for internal-but-documented names
  (`PipelineContext` stays private).
- Docs: `docs/src/api/25_Pipeline/…` pages (grouped per ADR 0014, docs-by-pipeline-stage), a
  user-guide example: prices → filter → impute → returns → prior → phylogeny → constraints →
  MeanRisk, tuned end-to-end with walk-forward CV.
- Precompilation workload addition (`23_Precompilation.jl`) once the API settles.
- Sweep: error-message quality for every "deliberately unsupported" boundary (meta-wrapping,
  combinatorial CV, predict-without-weights, unroutable slot elements).

## Known risks / open questions (to resolve during implementation, not design)

- `split` for existing CV types is written against `ReturnsResult` row counts; factor out a
  shared "number of observations + index vocabulary" seam rather than duplicating fold logic.
- `prices_to_returns` padding mode changes the T→T−1 contraction; `PricesToReturns` should
  either forbid padding inside pipelines or make the row bookkeeping explicit.
- Constraint-generation estimators need `AssetSets` at fit time; decide whether `sets` lives
  in the context (a step output/input) or stays optimiser config in v1.
- Mu/sigma routing when a single uncertainty step produces both: return a small composite
  Result that the router splits, or require two steps; pick when writing M3.
