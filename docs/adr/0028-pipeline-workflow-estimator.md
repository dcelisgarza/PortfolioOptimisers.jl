# Pipeline is a new root workflow Estimator, not an OptimisationEstimator

We are adding a reified end-to-end workflow object, `Pipeline` (code under `src/25_Pipeline/`),
whose driving purpose is to **widen the tuning/cross-validation boundary**: today the tunable
unit is a `NonFiniteAllocationOptimisationEstimator` fitted on a `ReturnsResult`, so everything
upstream of the returns matrix (price cleaning, `prices_to_returns`, returns filtering) sits
outside the CV/tuning loop and its hyperparameters can only be chosen by hand — or worse, tuned
with full-sample leakage. A `Pipeline` makes the whole workflow the unit that is fitted per
fold and searched by the lens-based grid/randomised search machinery.

## Decision cluster

1. **Shape: linear step list over an accumulating context (blackboard), not a DAG.** Steps run
   in user order; what flows between them is a `PipelineContext` with coarse typed slots
   (prices, returns, prior, phylogeny, uncertainty, constraints, weights). Parallel branches
   (phylogeny ~ constraints ~ uncertainty sets) are ordinary steps writing different slots.
   A true DAG was rejected as a workflow engine we don't need; strict sklearn-style linear
   piping was rejected because the fan-in at optimisation would force a mega-step.
2. **Steps are ordinary Estimators, mapped to slots by their existing abstract-type taxonomy**
   (`AbstractPriorEstimator` → prior, phylogeny estimators → phylogeny, …). No per-step wrapper
   ceremony; an explicit wrapper type is the escape hatch where dispatch cannot infer the slot
   (e.g. a mu-vs-sigma-ambiguous uncertainty set). New first-class Preprocessing Estimators are
   created for the data-prep stages (v1: `PricesToReturns`, missing-data filter, imputation).
3. **Computed slots override the optimiser's internal config via `factory` injection.** If the
   pipeline has a prior step, its `PriorResult` replaces the optimisation step's `pe` (the
   existing result-in-place-of-estimator idiom, cf. `prior(::AbstractPriorResult)` identity
   pass-through); absent steps fall back to the optimiser computing them internally, which is
   what makes every stage optional. Erroring on conflict was rejected because every optimiser
   carries a *default* `pe`, so intent is indistinguishable from default. Injection is
   type-checked at pipeline construction where possible. A shared prior is computed once per
   fold instead of once per consumer.
4. **Slot granularity is coarse, with type-driven routing.** The `uncertainty` and
   `constraints` slots hold heterogeneous collections whose elements route to their
   `JuMPOptimiser` fields by Result type. One-slot-per-optimiser-field was rejected as a
   maintenance mirror of `JuMPOptimiser` that turns the domain vocabulary into plumbing.
5. **CV splits on the pipeline's *input* rows as contiguous timestamp windows**, restricted to
   the sequential schemes (purged KFold, WalkForward); `CombinatorialCrossValidation` stays
   returns-level for now. Splitting on returns rows with a run-once "pre" segment was rejected
   because stateful prep (imputation stats, filter thresholds) leaks test information —
   precisely the leakage class the pipeline exists to remove. Note `prices_to_returns` maps
   T prices to T−1 returns, so index arithmetic must be window/timestamp-based.
6. **Prep steps have a fit/apply contract.** Fitting on the train window yields a Result
   carrying fitted state — imputation parameters, thresholds, and the selected asset universe —
   which is *applied* to the test window so train weights and test returns stay aligned.
   Stateless-only was rejected because retrofitting fit/apply later would break the step
   interface. Universe selection is explicitly fitted state.
7. **`Pipeline <: AbstractEstimator` as a new root concept, verbs `StatsAPI.fit`/`predict`** —
   *not* `OptimisationEstimator`/`optimise`. Chosen for conceptual honesty (a workflow, not an
   optimiser; its input is prices-or-returns, not `ReturnsResult`; its terminal step need not
   be an optimiser forever). Accepted cost: search-CV/predict/scoring need explicit new methods
   for `Pipeline` rather than inheriting the optimiser surface. `predict` on a pipeline result
   applies the fitted prep Results to the test window, then delegates to the existing
   weights-level prediction, returning the same `PredictionResult` types — scorers, risk
   measures, and `HighestMeanScore` carry over untouched.
8. **Steps are optionally named** (`"impute" => ImputePrices(…)`), auto-named from their slot
   otherwise (`prices_1`, `prices_2` on repeats); tuning lens paths accept name or index.
   Because lens values are arbitrary objects, structural search (swapping whole estimators as
   grid values) needs no extra design.
9. **Input container is `PricesResult`** — the prices-level mirror of `ReturnsResult` (asset +
   optional factor/benchmark/implied-vol `TimeArray`s). `fit(pipe, data)` fills whichever slot
   matches the input type, so passing a `ReturnsResult` skips the price stages. The filtering
   kwargs currently embedded in `prices_to_returns` (`missing_col_percent`, …) migrate into
   step-estimator configs; the existing function is unchanged.
10. **Nesting: pipelines may be steps of pipelines; meta-optimisers may be the optimisation
    step; wrapping a Pipeline *in* a meta-optimiser is unsupported in v1** (clear error, not a
    silent gap), because it requires `port_opt_view` semantics for a pipeline whose fitted prep
    steps themselves decide the asset universe.

## Future expansion

Pipelines inside meta-optimisers (NCO/Stacking/SubsetResampling wrapping a `Pipeline`) is a
good idea deliberately deferred, not rejected. Doing it requires answering: what
`port_opt_view` (asset sub-selection) means for a pipeline whose universe is fitted state, and
how a prices-level pipeline consumes the `ReturnsResult` views meta-optimisers produce. Revisit
once the v1 fit/apply contract has settled.

## Consequences

- New concepts `Pipeline`, `Pipeline Context`, `Preprocessing Estimator`, `PricesResult` are in
  `CONTEXT.md`; the glossary intro now says "workflow" so "Pipeline" unambiguously means the
  estimator.
- `PricesResult` follows the `ReturnsResult` precedent (data under the Result tree);
  `FiniteAllocationInput` (ADR 0017) remains the only data-as-Estimator deviation.
- v1 prep estimators are exactly one exemplar per contract — stateless (`PricesToReturns`),
  universe-mutating (missing-data filter), parameter-fitting (imputation) — to prove the
  machinery; further prep steps are additive.
- Existing entry points (`optimise(opt, rd)`, `search_cross_validation(opt, …, rd)`,
  `prices_to_returns`) are unchanged; the pipeline is a purely additive layer.

## Amendment (implementation, M4/M6): `PricesToReturns` is stateless but not universe-neutral

Implementing `predict` surfaced a wrinkle in decision 6. `PricesToReturns` is stateless as
designed — its fitted object is its config — but the `prices_to_returns` it wraps *drops
assets that are entirely missing in the window being converted*. A training window in which
an asset has no history therefore yields a smaller universe than a clean test window, so
weights fitted on train would silently misalign with test returns (in practice, a
`DimensionMismatch` deep inside the risk calculation).

Rather than make `PricesToReturns` fitted — which would destroy the one-exemplar-per-contract
property that motivated it — the universe contract is enforced at the boundary:
`assert_universe_aligned` compares the replayed test universe against the training universe
and rejects a mismatch with an error naming both. The remedy is the design's own answer: pin
the universe with `MissingDataFilter` (universe as fitted state) and fill the remaining gaps
with `Imputer` (parameters as fitted state) *before* converting. Both the `PricesToReturns`
docstring and the user-facing example say so.

The takeaway is that "stateless" in decision 6 means *carries no fitted state*, not *cannot
change the asset universe*. Only a universe-mutating step may define the universe.

## Amendment (post-M6): preprocessing estimators are decoupled from the pipeline

Decision 2 says steps are ordinary Estimators, but the v1 prep estimators were defined inside
`src/24_Pipeline/` and their fit/apply verbs were named `fit_step`/`apply_step` — naming and
placement that implied a dependency on the pipeline that never actually existed. The verbs
take `(estimator, data)` and `(fitted, data)`; they never saw a `PipelineContext`.

`PricesResult`, the preprocessing type hierarchy, `PricesToReturns`, `MissingDataFilter`, and
`Imputer` now live in `src/03_Preprocessing.jl` beside `ReturnsResult` and
`prices_to_returns`, and the verbs are `fit_preprocessing`/`apply_preprocessing`. They are
exported and usable on their own.

This makes preprocessing symmetric with every other family: the pipeline's only pipeline-aware
layer is `run_step`, which reads the context slots a step needs, dispatches to that family's
**native verb** (`prior`, `clusterise`, `optimise`, `ucs`, `fit_preprocessing`), and writes the
slot the family produces. Nothing outside `src/24_Pipeline/` knows the pipeline exists.

## Amendment (post-M6): uncertainty steps accept `target = :both`

The open question "mu/sigma routing when a single uncertainty step produces both" is resolved
in favour of *both mechanisms*. `PipelineUncertaintySets` is the composite the router splits,
and a `PipelineStep` declares `target = :mu`, `:sigma`, or `:both`. `:both` calls `ucs`, which
derives the two halves from a single fit — sharing the prior and, for the sampling algorithms,
the simulation draws — making it strictly cheaper than two narrowed calls.

A bare, unwrapped uncertainty estimator still throws. Although `ucs` makes the written *slot*
unambiguous, which parameters the user wants bounded remains an intent to declare, not one to
guess; and injection stays strict, so a populated half that cannot reach the optimiser is an
error rather than a silent drop. `:both` therefore requires an `ArithmeticReturn` return
estimator *and* an `UncertaintySetVariance` risk measure.
