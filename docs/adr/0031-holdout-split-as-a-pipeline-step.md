---
status: accepted
---

# The holdout split is a pipeline step, pinned to the first position, and excludes cross-validation

## Context

`train_test_split(rd; train_size, test_size)` cut price- or returns-level data into a train/test
pair, and was the only way to evaluate a workflow out-of-sample without invoking the whole
cross-validation machinery. It composed with `Pipeline` only by hand:

```julia
train, test = train_test_split(pr; test_size = 0.2)
res  = fit(pipe, train)
pred = predict(res, test)
```

Three facts shaped the design.

**The manual composition is correct but easy to get wrong.** Nothing stops a caller from
fitting the pipeline on the *whole* dataset and predicting on a slice of it — the same call
shape, silently in-sample. The library fails closed everywhere else; here it did not.

**`safe_index` was broken in two of its five branches.** With no sizes given, or with
`train_size` alone, it returned `(1:N_l, (N_l+1):N_h)` — and since `N_h = N - N_l` is a
*count*, not an index, the test range was empty whenever `N_l > N/2`. Only the `test_size`
branch, which the sole in-repo caller (`docs/paper/paper.jl`) happened to use, worked. The
estimator could not be built on top of a function whose default was an empty test set.

**A holdout is a statement about rows, not about a data level.** Every other preprocessing
estimator is pinned to prices or returns. A holdout is meaningful at either, and pinning it to
one would force a prices-fed pipeline to convert before it could split.

## Decisions

1. **`TrainTestSplit` is a preprocessing estimator, and the protocol lives inside the
   pipeline.** `TrainTestSplit(; train_size, test_size)` (alias `TTS`) subtypes
   `AbstractPreprocessingEstimator` *directly* — neither of the two data-level subtypes — and
   `run_step` narrows whichever data slot the pipeline input filled. `train_test_split` remains
   the free-function form and the estimator's implementation.

   The alternative, keeping the protocol outside the pipeline (a `Holdout` cross-validation
   estimator plugging into the `Base.split`/`n_splits` seam), was rejected: it would make the
   holdout a *scoring* mechanism competing with the real cross-validators, when what callers
   actually want is for the workflow itself to carry its evaluation split — a `fit` that trains
   on the training rows, whoever calls it.

2. **The split must be the first step, and may not nest.** `assert_split_position` rejects a
   `TrainTestSplit` at any other index, and `has_split` recursively rejects one inside a nested
   `Pipeline` or `PipelineStep`. This is the whole safety argument: a *stateful* step fitted
   before the split — a `MissingDataFilter` choosing the universe, an `Imputer` computing fill
   values from column statistics — would have read the held-out rows, and its fitted state would
   carry test information into the training workflow. Position one is the only position where
   that cannot happen.

   The cost is real: a prices-fed pipeline cannot split at the returns level. The two differ by
   exactly one boundary return, the same approximation price-level cross-validation splitting
   already accepted (ADR 0028). The alternative — permitting the split after provably *stateless*
   steps — needs a statelessness trait on preprocessing estimators that does not exist, to buy
   one row.

3. **The fitted result carries both windows; replay is a pass-through.**
   `TrainTestSplitResult(train, test)` holds both as `port_opt_view`s. `fit_predict(pipe, data)`
   predicts on the stored `test` window — the one-line holdout evaluation — and stays in-sample
   for a pipeline without a split.

   Crucially, `apply_preprocessing(::TrainTestSplitResult, data) = data`. A fitted holdout's rows
   are a fact about the *fitting* window, so replaying them on an unseen window is meaningless;
   passing through keeps `predict(res, future_data)` working on genuinely new observations, which
   a row-selecting replay would have broken. This is why the result is a plain `AbstractResult`
   and not an `AbstractReturnsPreprocessingResult`: the returns-level `apply_fitted_step` methods
   must not claim it.

4. **A pipeline carrying a holdout cannot be cross-validated.** `assert_no_holdout` throws from
   `search_cross_validation` and `fit_and_score`. Cross-validation already defines the train and
   test window of every fold; a split left in the pipeline would shave a second holdout off each
   fold's training data and stash a test window nobody reads. Two evaluation protocols on one
   workflow is user error, and the library says so rather than silently losing training rows.

5. **`:split` is a sentinel, not a context slot.** `pipe_writes(::TrainTestSplit) = :split`,
   which is deliberately *not* in `PIPELINE_SLOTS`: it names the step (`pipe.names` reads
   `"split"`), invalidates nothing, and satisfies nothing, so the generic `Pipeline` constructor
   needed no special case beyond the position check. This is sound only because the split is
   pinned to position one, where both data slots are already available from the input and no
   derived slot exists to invalidate. Declaring `:prices` or `:returns` instead would have been a
   lie about the other level, and would have made a later prices step spuriously trip the
   `PIPELINE_INVALIDATES` check.

6. **Sizing: complement, then embargo.** One size given makes the other its complement; neither
   given splits 75/25. Both given and summing to less than `N` puts train at the head, test at the
   tail, and leaves the rows between them **embargoed** — belonging to neither window, which is
   how a gap between train and test is expressed. Overlapping windows throw, as does a split that
   would empty either side. Counts saturate at `N` (consistent with ADR 0029's rule for `RankRule`),
   but a saturating `train_size` still throws, because an empty test window defeats the purpose.

7. **The Integer/Float duality is kept.** `train_size = 250` is a row count, `train_size = 0.25` a
   fraction — the duality ADR 0029 deliberately *split* into two types for `RankRule` and
   `QuantileRule`. The asymmetry is intentional: there, `best = 1` versus `best = 1.0` silently
   produced one asset versus the whole universe, and both were legal. Here the dangerous case,
   `test_size = 1.0`, empties the training window and dies loudly on the non-empty guard, so one
   estimator with two field types costs nothing.

## Consequences

- **Behaviour change**: `train_test_split(rd)` and `train_test_split(rd; train_size = k)`
  previously returned an *empty* test window for most inputs. They now return the complement.
  A caller relying on the empty result was relying on a bug, but it is a visible change.
- Floats are now restricted to `(0, 1)` exclusive; `1.0` was previously accepted and produced an
  empty test set.
- `fit_predict(pipe, data)` changes meaning for a split-bearing pipeline: out-of-sample rather
  than in-sample. Pipelines without a split are unaffected.
- New concepts `Holdout Split` and `Embargo` are in `CONTEXT.md`.
- `docs/paper/paper.jl` continues to work unchanged: it uses the one `safe_index` branch that
  was already correct.
