---
status: accepted
---

# The cross-validation fold loop is one seam

## Context

Every cross-validation entry point ran its own fold loop. There were seven of them: the
non-sequential scheme (`01_Base_CrossValidation.jl`), the combinatorial scheme
(`03_Combinatorial.jl`), the walk-forward scheme (`04_WalkForward.jl`), the multiple-randomised
path (`05_MultipleRandomised.jl`), and three in `23_Pipeline/04_PredictionCV.jl` — the
combinatorial pipeline, the pipeline path, and the contiguous single-path pipeline.

The seven were the same loop. Each read the fold count off `train_idx`, called
`is_time_dependent`, called `assert_time_dependent_fold_count` under that flag, built a
`TimeDependentContext` per fold, swapped the schedules through
`update_time_dependent_estimator`, and — at the five time-ordered sites — threaded the previous
fold's weights through `factory`. Only the tail differed: what the fold does with its resolved
estimator and its two index vectors.

The guards did not follow the loop. Two of the seven refused a scheme that declares a set
`shuffle` field. One of the seven also refused non-increasing training indices. A guard landed
where its author was reading, not where the shape was.

The monotonic guard was also wrong. It was written

```julia
all(map(x -> x > zero(x), map(x -> diff(x), train_idx)))
```

`x` there is a whole `diff` vector, so `x > zero(x)` is `Base.isless` on two vectors, which is
lexicographic. A fold whose training indices are `[1, 5, 3, 9]` passes it: only the first gap is
read. The guard tested the first index gap of each fold, not the fold.

Two sites recomputed `needs_previous_weights` inside the fold body, once per fold, on top of the
call `run_folds` already makes.

## Decision

One `fold_loop`, in `20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl` beside
`parallel_folds` and `run_folds`:

```julia
fold_loop(fit_fold, est, n, ex; rd, train_idx, test_idx, path_id = nothing,
          ElT = PredictionResult, time_ordered = true, fold_view = nothing)
```

Per fold it takes the fold's view of `(est, rd)`, swaps the time-dependent entries, threads the
previous weights, and calls `fit_fold(i, esti, rdi, train_idx[i], test_idx[i])`. Each of the
seven sites is now that callback and nothing else.

Three parameters carry the differences the seven had.

- `time_ordered` states whether the fold enumeration of the scheme is a timeline. `true` routes
  through `run_folds`, so an optimiser that needs the previous weights runs sequentially.
  `false` routes through `parallel_folds`: a k-fold or combinatorial enumeration is not a
  timeline, so it has no previous fold to thread. Two sites pass `false`.
- `fold_view(i)` returns the fold's `(est, rd)`. The two asset-resampling paths give one; the
  other five do not.
- `ElT` is the per-fold element type. The two combinatorial sites pass
  `Vector{PredictionResult}`.

`needs_previous_weights` is called once, in `fold_loop`, and handed to `run_folds` as the new
`prev_w_flag` keyword. Its default is the call it replaces, so `run_folds` keeps its signature
for a direct caller.

Both guards become one `assert_unshuffled_folds(cv, train_idx)`, called once per `split` at
every entry point — the seven loops and the two predict-only twins. The monotonic half is
rewritten as `all(x -> all(>(zero(eltype(x))), diff(x)), train_idx)`, which reads every gap.

## Consequences

The monotonic guard now holds for all schemes, not one. Every scheme this package ships
enumerates strictly increasing training indices — checked on `KFold`, `KFold` with purging and
embargoing, `CombinatorialCrossValidation`, `IndexWalkForward` and `MultipleRandomised` — so no
shipped path reaches the guard. Gaps stay legal: purging, embargoing and the combinatorial
splits all leave them, and a gap is an increase.

The correction is a real widening. A user-supplied scheme, or a `CrossValidationResult` built by
hand, whose training indices are reordered after the first gap was accepted before and is
refused now. That is the case the guard was written for.

The shuffle-field guard now runs after `split` rather than before it, because it is one call
with the index guard. No scheme in this package has a `shuffle` field, so this only reorders
which error a user-defined shuffled scheme raises.

`parallel_folds` and `run_folds` keep their published signatures and stay in use: the two
path-level fan-outs and the two predict-only twins call `parallel_folds` directly, and they have
no estimator to resolve.

ADR 0030 (time-dependent constraints) is untouched — the swap-then-inject order it fixes is now
stated once instead of seven times. ADR 0001 (the parallel test suite) is untouched.
