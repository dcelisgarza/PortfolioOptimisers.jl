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

## Amendment (2026-09-04)

Issue #759. The original decision wrote `time_ordered` as a literal at each call site, and said
"Two sites pass `false`". That count was a fact about the call sites, not a rule, and the
`Pipeline` seam broke it. The `Pipeline` method for `cv::CVER` serves `KFold` and both
walk-forwards through one body, so it could not restate the literal, and it took the `true`
default for all three. A `Pipeline` over a `KFold` therefore carried the previous fold's weights,
where the optimiser-level `KFold` path did not.

The scheme now states the answer, and the call sites read it:

```julia
folds_are_time_ordered(::Any) = true
folds_are_time_ordered(::NonSeqCVER) = false
```

Five sites hold a scheme, and each passes `time_ordered = folds_are_time_ordered(cv)`: the
optimiser-level non-sequential, combinatorial and walk-forward methods, and the `Pipeline`
combinatorial and `CVER` methods. Two path-level sites enumerate an inner walk-forward and hold
no scheme, so they keep the `true` default.

The default is `true`, so a user-supplied scheme threads history unless it declares that it does
not. Threading history into a scheme that has none is a wrong reading; dropping it from a
timeline is a wrong optimisation. The conservative default is the one that keeps the timeline.

`fold_loop` keeps the `time_ordered` keyword and its `true` default. The change is which value
each call site sends, not what the loop does with it.

### Behaviour this moves

A `Pipeline` over a `KFold` or a `KFoldResult` whose steps need the previous weights now
runs its folds in parallel with `w_prev == nothing`, and emits no `cv_sequential_info` message. A
turnover budget, a turnover fee or a tracking term in such a pipeline reads its own reference
weights instead of the previous fold's. Every other scheme is unchanged.

## Amendment (2026-09-04, second)

Issue #762. The rule is a conjunction, and no site computed it. A run is sequential only when
the fold enumeration is a timeline **and** the estimator needs the previous fold's weights. The
first amendment put the timeline half in `fold_loop`'s `time_ordered::Bool` keyword and the
previous-weights half in `run_folds`'s `::Val{PW}` type parameter. `fold_loop` sent one case to
`run_folds`, which re-decided and sent its own `else` arm back to `parallel_folds`. So
`parallel_folds` was reached by two routes for one reason, and `run_folds` did not always run the
folds the way its name says.

`fold_loop` now computes the conjunction, and `run_folds` is the sequential loop:

```julia
# fold_loop
return if folds_are_time_ordered(cv) && prev_w_flag
    run_folds(fold, n, ElT)
else
    parallel_folds(i -> fold(i, nothing), n, ex, ElT)
end
```

### The keyword becomes the scheme, not a `Bool`

`time_ordered::Bool` is gone, and `fold_loop` takes `cv = nothing` in its place. This is what
makes the conjunction fold. A keyword *value* survives only by constant propagation, which one
call hop loses — the same mechanism the first amendment measured for `ElT`. A keyword *type* does
not. `folds_are_time_ordered(cv)` and `needs_previous_weights(est)` are both per-type methods over
concretely-typed arguments, so inference decides the conjunction from types alone and eliminates
the arm that cannot run. `folds_are_time_ordered(::Any)` already answers `nothing`, so the two
path-level sites that hold no scheme omit the keyword.

`run_folds` also loses `opt` and `ex`. `opt` existed for the `Val` default, and `ex` for the
`else` arm; with no branch, both are dead. `cv_sequential_info` loses its `prev_w_flag` argument
for the same reason: the only caller is the sequential loop, so the value was always `true`. The
message now states both facts rather than quoting a constant back.

### What the measurement says

The claim of the first amendment — "runtime dispatch inside the loop machinery is now **zero** on
all four schemes" — was too strong as written, and this shape improves on what was actually there.
Runtime-dispatch reports attributed to `fold_loop`, `run_folds` or `parallel_folds`, from
`JET.report_opt` over `cross_val_predict`, on an `EqualWeighted` with and without a
`PreviousWeightsFunction` schedule:

| scheme | plain, before | plain, after | previous weights, before | previous weights, after |
| :--- | ---: | ---: | ---: | ---: |
| `KFold` | 4 | 2 | 3 | 2 |
| `CombinatorialCrossValidation` | 5 | 3 | 3 | 3 |
| `IndexWalkForward` | 4 | 2 | 3 | 1 |
| `MultipleRandomised` | 26 | 17 | 18 | 2 |

Every case falls or holds. The reports that remain sit on the `@floop` line of `parallel_folds`,
which is FLoops spawning its tasks, and one on the sequential loop's abstractly-typed
`predictions[i - 1]`.

That last one is the defect this amendment removes. Before, it was reported on the `KFold` and
combinatorial paths too, where the sequential loop can never run: `time_ordered` was a run-time
`Bool`, so the `run_folds` arm was inferred even for a scheme that never reaches it. It is now
reported only on `IndexWalkForward` and `MultipleRandomised` with a previous-weights estimator —
the two cases that do run the sequential loop.

### What did not change

`folds_are_time_ordered` keeps its two methods and its conservative `true` default, so a
user-supplied scheme still threads history unless it declares that it does not. `parallel_folds`
keeps its signature and its four direct callers. `Fold`, `fold_view` and `assert_unshuffled_folds`
are untouched. ADR 0030 is untouched.

`test_37_time_dependent_constraints.jl` gains a testset that names the conjunction and covers all
three reachable combinations at the optimiser level: both halves (sequential, and fold 2 reads
fold 1), the previous-weights half alone over a `KFold` (parallel, silent, no history), and the
timeline half alone (parallel, silent).
