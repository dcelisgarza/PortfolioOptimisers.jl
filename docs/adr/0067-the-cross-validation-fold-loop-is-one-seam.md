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

## Amendment (2026-08-19)

A measured comparison of the seam against its two predecessors — the seven hand-written loops of
v0.27.0 and the fully explicit `@floop` loops of v0.23.2 — found the seam's run-time cost to be
nil and its complexity the lowest of the three. It also found one real defect and one real
readability cost. Both are corrected here. The decision of this ADR stands.

### `ElT` becomes a positional `::Type{ElT}` argument

`fold_loop` took the per-fold element type as a keyword value and forwarded it to `run_folds`,
which forwarded it again to `parallel_folds`. A keyword *value* of type `Type` only reaches the
allocation as a constant through constant propagation, and one forwarding hop is enough to lose
it. JET recorded the loss exactly:

```text
Core.kwcall(::@NamedTuple{ElT::UnionAll, prev_w_flag::Bool}, ::typeof(run_folds), …)
  ↳ #run_folds#1375(::Type{PredictionResult}, ::Bool, …)   # still concrete
    ↳ Core.kwcall(::NamedTuple{(:ElT,), <:Tuple{Type}}, ::typeof(parallel_folds), …)
      ↳ #parallel_folds#1370(::Type, …)                    # constant lost
        ↳ runtime dispatch detected
```

`Vector{ElT}(undef, n)` was therefore a runtime dispatch. All three functions now take the
element type as a positional `::Type{ElT}` argument, so a method always specialises on it and the
constant cannot be lost. The seven `fold_loop` sites and the four direct `parallel_folds` sites
pass it positionally.

The cost this removes is small and honest: each lost dispatch ran **once per cross-validation
call**, not once per fold. The gain is a clean JET report, not a faster backtest.

### The callback takes one `Fold`

The callback took five positional parameters, `fit_fold(i, esti, rdi, train_idx[i],
test_idx[i])`. A reader at a call site saw `do i, opti, rdi, tr, te` and had to open `fold_loop`
to learn what they were and in what order. Five positional parameters is the point at which
position stops being a readable contract.

`fold_loop` now builds one immutable `Fold` record — `i`, `n`, `est`, `rd`, `train`, `test` — and
the callback reads it by name. `Fold` is constructed at exactly one site, so it has no keyword
constructor, and every field is concretely typed there, so the record is free.

`Fold` is deliberately *not* `TimeDependentContext`. The context is what a schedule entry is
resolved *against*, one step earlier and inside the loop; the `Fold` is what the loop hands out
once resolution is finished. Merging them would put `w_prev` and `path_id` — which a callback
must not read, because they are already applied — back on the callback's surface.

### The previous-weights flag becomes a `Val`

Fixing `ElT` left two runtime dispatches inside the loop machinery, one each on the KFold and
walk-forward paths. JET named the line: the *sequential* branch of `run_folds`,

```julia
predictions[i] = fit_fold(i, i > 1 ? predictions[i - 1] : nothing)
```

`ElT` defaults to `PredictionResult`, which is a `UnionAll` and therefore abstract, so
`predictions[i - 1]` is abstractly typed and that call cannot resolve statically. The branch is
dead for any optimiser that does not need the previous weights — but `prev_w_flag` was a
run-time `Bool`, so inference analysed the branch anyway and reported the dispatch.

The flag is now a type parameter:

```julia
function run_folds(fit_fold, opt, n::Integer, ex::FLoops.Transducers.Executor,
                   ::Type{ElT} = PredictionResult,
                   ::Val{PW} = Val(needs_previous_weights(opt))) where {ElT, PW}
    if PW   # compile-time constant: the dead branch is eliminated, not inferred
```

The default is safe: `Val(needs_previous_weights(opt))` is decided by ordinary type inference,
not by constant propagation, and infers to `Val{false}` for a plain optimiser. It therefore
survives the extra call layer that `fold_loop` adds, which constant propagation of the `Bool` did
not — an explicitly forwarded `prev_w_flag = prev_w_flag` was measured and did *not* restore the
elimination.

This supersedes the note in the original decision that `run_folds` keeps `prev_w_flag` as a
keyword for a direct caller. It has no direct caller other than `fold_loop`.

Runtime dispatch inside the loop machinery is now **zero** on all four schemes, matching the
pre-seam v0.27.0 code, which reached zero by accident: its call sites passed neither the element
type nor the flag, so both folded one hop from their use.

### What did not change

`parallel_folds` and `run_folds` keep their roles and their default element type. The four
predict-only and path-level sites still call `parallel_folds` directly. The `time_ordered` and
`fold_view` parameters are unchanged. ADR 0030 is untouched.
