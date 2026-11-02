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

## Amendment (2026-09-11)

Issue #871. The seven loops this ADR folded into `fold_loop` were not all of them. The search's
`search_cross_validation`, for the optimiser and for the `Pipeline`, ran an eighth: per candidate
it looped over the folds and refitted each one from its training window through `fit_and_score`,
building no `TimeDependentContext` and threading no previous weights. The search was outside the
seam, so a walk-forward search scored a walk-forward it did not declare, and it could not take the
online arm ADR 0140 added.

[ADR 0141](0141-a-search-scores-every-candidate-through-the-one-fold-loop-online-and-batch-alike.md)
puts it inside: a search scores every candidate through `fit_and_predict(opt_i, rd, cv; ex =
SequentialEx())`, and the `fit_and_score` families are deleted. Every cross-validation entry
point, the searches included, now goes through the one loop.

## Amendment (2026-11-02)

Issue #1938. The folds of a prediction Result stay a `Vector{PredictionResult}`. The loop does
not narrow them to the concrete type of the folds. This is a decision, not an omission.

`ElT = PredictionResult` is a `UnionAll`, as the first amendment states. So a reader of the folds,
for example `stacked_fold_weights`, `held_path_turnover` or `comparator_path_length`, infers its
result abstractly. `log_wealth_regret(a, b)` with a multi-period comparator infers every field of
its Result except `path_length`. `performance_summary` of a multi-period prediction infers every
field except `turnover`. The values are correct, and each value has a concrete type at run time.

### What the narrowing costs

A narrowing to the concrete fold type, for example `concrete_typed_array(pred)` in the inner
constructor of `MultiPeriodPredictionResult`, makes these readers infer. It also puts the type of
the optimiser Result into the type of the prediction Result. Every verb that takes a
`MultiPeriodPredictionResult` then compiles one time for each optimiser configuration, and not one
time for all of them. The measure is `performance_summary(p)` followed by `log_wealth_regret(p, p)`
on an `IndexWalkForward(30, 10)` over 120 rows and 4 assets, which gives 9 folds. The figures are
for the first call on a `MeanRisk` with a risk measure that no earlier call used, over three such
risk measures, and for a warm call. The tree is `dev` at `009482721d`, with 16 threads:

| Folds | First call on a new optimiser type | Of which compile | Warm call |
| :--- | ---: | ---: | ---: |
| `Vector{PredictionResult}` | 24 ms | 23 ms | 58 µs |
| narrowed to the fold type | 590 ms | 590 ms | 14 µs |

The narrowing saves 44 µs on each warm call, and it costs about 0.57 s more compile time on each
new optimiser type. The break-even is about 13 000 calls for each type. A warm `cross_val_predict`
of the `MeanRisk` over the same folds takes about 12.6 ms. So the saved time is small next to the
work that makes the Result. The compile time is spread over the verbs:

| Function, narrowed | Compile on a new optimiser type |
| :--- | ---: |
| `stacked_fold_weights(p, fold_target)` | 0.29 s |
| `held_path_turnover`, which adds `fold_held` | 0.13 s |
| the body of `log_wealth_regret` | 0.16 s |
| `performance_summary` and `comparator_path_length` | 0.01 s each |

The cost does not stop at the fold readers. The body of `log_wealth_regret` reads no fold, and it
compiles again because the type of its argument changes. A plot, a scorer and `expected_risk` of a
multi-period prediction take the same kind of cost. This amendment did not measure them.

### Where a narrowed element type pays

The vector of folds is a type-erasure boundary. Under a batch scheme, every optimiser gives one
`MultiPeriodPredictionResult` type for one returns data, so each verb that reads the Result
compiles one time. Under an Online Scheme the Result holds the threaded estimator in `opt`, but
since #1939 the type of `opt` is not a type parameter of the Result, so the same holds. Each fold
reader dispatches on the fold at run time, and that dispatch compiles one small method for each
new fold type, which is the 23 ms above. Since #1939 a new risk measure gives no new fold type.

`concrete_typed_array` and `concrete_typed_array_if_abstract` narrow a vector that an estimator or
a one-shot configuration holds. The library already compiles on the type of that holder, so the
narrowing adds no new axis of compilation. Each library site is of this kind:

| Site | Vector | Why the narrowing stays |
| :--- | :--- | :--- |
| `factory` and `port_opt_view` of `VecTnE_Tn` | the turnover constraints of an estimator | The estimator type holds each element type already. The narrowing changes only a mixed vector. |
| `port_opt_view` of a vector of asset sets matrices | the membership matrices of an estimator | As for the turnover constraints. |
| `rebuild_asset_panel` | the panel fields of the folds of one meta-optimisation | A comprehension gives a concrete element type when the folds share one type. The call changes only a mixed vector. |
| `make_p_grid` of a randomised search | the trials of one search | Dispatch needs it. A `Vector{Pair}` does not match the `AbstractVector{<:Pair{<:Any, <:AbstractVector}}` of `lens_val_grid`. The call runs one time for each search. |

The rule: narrow the element type of a vector whose holder the library compiles on already. Do not
narrow a vector of a Result that many verbs read, because each new element type then compiles
every one of those verbs again.

### The type of an estimator in a holder (#1939)

Issue #1939 measured each holder whose type parameters carried the type of an estimator, or of a
part of one. Each figure is the first call on a new type, after a warm-up that compiled every part
that the two forms share, and then a warm call. Each form ran in a fresh process with one thread,
on `dev` at `e641c54bba`, with 200 rows and 6 assets. Each range covers five risk measures:
`ConditionalValueatRisk`, `StandardDeviation`, `Range`, `WorstRealisation` and `MaximumDrawdown`.

| Holder and measured call | Form | Compile on a new type | Warm call |
| :--- | :--- | ---: | ---: |
| `MultiPeriodPredictionResult.opt` of an online run: `performance_summary` and `log_wealth_regret` | type parameter | 326–344 ms | 118–211 µs |
| | erased | 31–33 ms | 110–189 µs |
| `MeanRiskResult.r`: `expected_risk` two times and `predict` | type parameter | 89–97 ms | 78–93 µs |
| | erased | 0 ms | 31–68 µs |
| `Stacking.opti`: `optimise` | narrowed to the union of the element types | 530–620 ms | 18.6–30.1 ms |
| | `Vector{OptE_Opt_TD}` | 107–119 ms | 18.6–25.5 ms |

The erased form of `opt` and of `r` is emulated: the Result is rebuilt with `opt = nothing`, or
with the `r` of the warm-up. No verb reads either field, so the emulation compiles what the
erased field compiles. No warm call is slower in an erased form, so no call count reaches a
break-even. A compile of `optimise` on a new `MeanRisk` takes 0.7 to 3 s, and the erasure of `r`
removes about 90 ms after it.

The decisions:

- `MultiPeriodPredictionResult` binds `opt` to `Union{Nothing, AbstractEstimator}`, and the
  field leaves the type parameters. Only `Resume` reads `opt`. `Resume` now holds the same object in
  a second field, `opt`, so its second type parameter is the type of the estimator.
  `OptimiserResume`, `PipelineResume` and `Pipeline_OnlPipe` dispatch on that parameter.
- `MeanRiskResult`, `RiskBudgetingResult`, `FactorRiskContributionResult`,
  `NearOptimalCenteringResult`, `HierarchicalRiskParityResult` and
  `SchurComplementHierarchicalRiskParityResult` keep the bound of `r` on the field, and the field
  leaves the type parameters. `r` was the field that gave each risk measure its own fold type in
  #1938.
- `Stacking` stores a vector `opti` as a `Vector{OptE_Opt_TD}`, through `erase_optimiser_vector`
  in both constructors. The loop over `opti` solves one optimisation for each element, so a
  dispatch at run time costs nothing next to it. A lens write such as `@set st.opti[2] = x`
  already gave an abstract element type, so the narrow type did not hold after a search wrote a
  candidate. `StackingResult.resi` was a `Vector{NonFiniteAllocationOptimisationResult}` already.
- `SearchCrossValidationResult.opt` keeps its type parameter. No verb reads that Result, so the
  parameter compiles nothing.
- `ProcessedJuMPOptimiserAttributes`, the `pa` of a `JuMPOptimisationResult`, holds `sca` and
  `ret`, and `HierarchicalRiskParityResult` holds `sca`. A new risk measure did not change their
  type, and #1939 did not measure them. The next section measures them.
- The vectors of constraints that `factory` and `port_opt_view` narrow keep the narrowing, because
  dispatch needs it. A `Vector{TnE_Tn}` of two `Turnover` values has no method of
  `turnover_constraints` without sets, and the narrowed `Vector{Union{Turnover{…}, Turnover{…}}}`
  takes the method for a `VecTn`. A user had to narrow such a literal by hand, because a mixed
  literal such as `[Turnover(…), TurnoverEstimator(…)]` infers `Any`, and the bound of the field
  refused it. The keyword constructors of `JuMPOptimiser` and `ProgrammeAllocationSet` (`lcse`,
  `sgcarde`, `smtx`, `sgmtx`, `slt`, `sst`, `sglt`, `sgst`, `tn` and `ple`) and of
  `FactorRiskContribution` (`frc_ple`) now narrow it through `narrow_mixed_vector`. The positional
  constructors keep their bounds.
- A vector of search grids whose types differ infers the element type `Vector`, and the bounds
  of `GridSearchCrossValidation`, `RandomisedSearchCrossValidation` and `WeightMassScorer`
  refused it. The search runs one time, so the narrow type pays nothing at run time, but
  `lens_val_grid` dispatches on it. The outer constructors now narrow such a literal through
  `narrow_mixed_vector`, and `make_p_grid` keeps its own narrowing of the trials for the same
  dispatch.

The rule becomes: a holder that many verbs read, or a meta-estimator over a vector of estimators,
does not carry the type of an estimator in a type parameter, unless a measured verb pays for it.
Narrow the element type of a vector only where dispatch needs it.

### The parts of an estimator in a holder (#1941)

Issue #1941 measured the holders that #1939 left: `ProcessedJuMPOptimiserAttributes.sca` and
`.ret`, which every JuMP optimisation Result holds in `jr.pa`, and
`HierarchicalRiskParityResult.sca`. `HierarchicalEqualRiskContributionResult` held `ri`, `ro`,
`scai` and `scao` in type parameters by the same mechanism, so #1941 measured it too. The method
is the method of #1939, on `dev` at `5d54b94ade` with the commits of #1939, two rounds for each
form. The measured calls are `expected_risk` two times and `predict` on the Result, and
`performance_summary` and `log_wealth_regret` on a `cross_val_predict` Result under
`IndexWalkForward(40, 20)`.

| Holder and new parts | Form | Compile, Result | Warm call, Result | Compile, cross-validation | Warm call, cross-validation |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `pa` of a `MeanRiskResult`: `MaxScalariser`, `LogSumExpScalariser`, `LogarithmicReturn`, `NoReturn` | type parameter | 81–85 ms | 66–93 µs | 30–31 ms | 103–187 µs |
| | erased | 0 ms | 29–69 µs | 0 ms | 75–190 µs |
| `HierarchicalRiskParityResult.sca`: `MaxScalariser`, `LogSumExpScalariser`, `MinScalariser` | type parameter | 79–84 ms | 48–74 µs | 31–32 ms | 113–224 µs |
| | erased | 0 ms | 13–57 µs | 0 ms | 78–199 µs |
| `HierarchicalEqualRiskContributionResult`: a new `ri`, `ro`, `scai` or `scao` | type parameter | 77–86 ms | 44–64 µs | 31–38 ms | 111–147 µs |
| | erased | 0 ms | 14–45 µs | 0 ms | 79–113 µs |

The erased form is emulated as in #1939: the Result is rebuilt with the parts of the warm-up. The
erased row of `pa` leaves out `LogSumExpScalariser`. Its solve failed, so the `retcode` of its
Result had a new type, and the Result compiled 88–115 ms for that reason alone (#1942). A new
`mu` of an `ArithmeticReturn` gave no new type, because the processing fills `mu` from the prior.

No warm call is slower in an erased form, so no call count reaches a break-even. No verb reads
the fields of the Result. The model builders read `attrs.ret` once for each `optimise` call, and
once for each point of a frontier, so a dispatch at run time costs nothing next to the solve.

The decisions:

- `ProcessedJuMPOptimiserAttributes` binds `ret` to `JRE_VecJRE` and `sca` to `Scalariser` on the
  field, and both fields leave the type parameters. So the Results of `MeanRisk`,
  `RiskBudgeting`, `FactorRiskContribution` and `NearOptimalCentering` have one type for every
  scalariser and every returns estimator.
- `HierarchicalRiskParityResult` binds `sca` to `Scalariser`, and
  `HierarchicalEqualRiskContributionResult` binds `ri` and `ro` to `BaseRM_VecBaseRM` and `scai`
  and `scao` to `Scalariser`. These fields leave the type parameters.
- The processed constraint Results in `pa` are data, and they keep their type parameters.

### Consequences

- `@inferred` is not a contract for a value that a reader takes from the folds of a prediction
  Result. A test checks the type of that value at run time.
- A user who calls a verb many times on one Result type, for example in a loop over many
  scenarios, can narrow the folds with
  `MultiPeriodPredictionResult(; pred = concrete_typed_array(p.pred), id = p.id, opt = p.opt)`.
  The verbs accept both forms.
- `fold_loop`, `parallel_folds` and `run_folds` do not change.
- #1939 removes a type parameter from `MultiPeriodPredictionResult` and from the six optimisation
  Results above, and adds one to `Resume`. Code that names the parameters of these types by
  position breaks. A method that must dispatch on the type of the estimator of a Result takes
  the field as an argument, or a wrapper holds the estimator in a field of its own, as `Resume`
  does.
- #1941 removes two type parameters from `ProcessedJuMPOptimiserAttributes`, one from
  `HierarchicalRiskParityResult` and four from `HierarchicalEqualRiskContributionResult`. Code that
  names the parameters of these types by position breaks.
