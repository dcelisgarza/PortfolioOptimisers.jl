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
   **Amended by [ADR 0038](0038-optimisers-own-pipeline-routing.md):** as implemented, "the
   optimisation step's `pe`" meant only an optimiser holding a `JuMPOptimiser`/
   `HierarchicalOptimiser` configuration, so the naive and meta-optimisers — which carry a
   `pe` of their own, and have the most consumers — silently recomputed it. They now receive
   it. The construction-time claim was also aspirational: routability is now checked, but
   only for the `uncertainty` slot, whose targets are the only ones knowable before fitting.
4. **Slot granularity is coarse, with type-driven routing.** The `uncertainty` and
   `constraints` slots hold heterogeneous collections whose elements route to their
   `JuMPOptimiser` fields by Result type. One-slot-per-optimiser-field was rejected as a
   maintenance mirror of `JuMPOptimiser` that turns the domain vocabulary into plumbing.
   **Scoped by [ADR 0038](0038-optimisers-own-pipeline-routing.md):** this governs
   `PIPELINE_SLOTS`, the pipeline-author vocabulary, which remains coarse and unchanged.
   0038 adds a finer, per-field *Routing Target* vocabulary addressing optimiser authors,
   and keeps it internal precisely so this decision is not circumvented.
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

## Amendment (post-M6): a step may not invalidate a slot an earlier step wrote

Construction-time validation checked only that a step's *reads* were already available: the
`avail` set grew monotonically and never noticed that a step's *write* could make an
earlier slot stale. `inject_context` then injected `ctx.prior` into the optimiser without a
dimension check. So this constructed, ran, and injected an N-asset prior into an M-asset
problem:

```julia
Pipeline(steps = [PricesToReturns(), EmpiricalPrior(), ZeroVarianceFilter(), MeanRisk()])
```

The hazard was latent while `PricesToReturns` was the only step writing `:returns` — nobody
puts it after a prior. Returns-level asset selectors (ADR 0029) make "a step that shrinks
`:returns`" an ordinary, easily-misordered thing, and the same applies to `:phylogeny`,
`:uncertainty`, and `:constraints`, all asset-dimensioned and all derived from `:returns`.

`PIPELINE_INVALIDATES` names, per written slot, the slots that write invalidates.
`Pipeline(; steps)` tracks which slots were written *by a step* — distinct from `avail`,
which is seeded with `:prices` and `:returns` because `fit` may be handed either — and
throws when a step writes a slot that invalidates an already-written one, naming both
steps. It also catches `PipelineStep(est = callable, writes = :returns)`, since the check
reads declared writes rather than inspecting the estimator.

Failing at construction rather than dimension-checking at injection was chosen because it
costs nothing at runtime, reports before any data is touched, and names the offending pair
of steps rather than the symptom. It also catches a pre-existing latent bug for free:
`PricesToReturns → MissingDataFilter` writes `:prices` after `:returns` was derived from
it, so the optimiser silently consumed the pre-filter returns.

The rule forbids a deliberate "prior-informed selection, then re-prior" ordering. The
remedy is to say so: add a second prior step *after* the selector, which the rule permits
and which is what the user meant.

## Amendment: file renumbering

`src/23_Precompilation.jl` was removed and `src/24_Pipeline/` became `src/22_Pipeline/`
(mirrored by `docs/src/api/22_Pipeline/`), freeing `24` for `src/23_AssetSelection.jl`.
References to the old paths above are retained as written at the time of the decision.
