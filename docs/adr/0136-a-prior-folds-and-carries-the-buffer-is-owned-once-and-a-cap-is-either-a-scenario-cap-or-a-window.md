---
status: accepted
---

# A prior folds and carries, the buffer is owned once at the bottom, and a cap is either a scenario cap or a window

## Context

[Map #861](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/861) lifts every layer above
the moments onto the online seam. [ADR 0106](0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md)
fixed where a Partial Fit State lives and
[ADR 0107](0107-the-update-seam-has-two-verbs-and-a-view-slices-by-asset-and-drops-by-observation.md)
fixed its two verbs.
[Issue #865](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/865) then decided that a
member with no exact recursion refits from a **Sample Buffer** seeded by an `Online` wrapper, and
that **the state's type is the route**: a `SampleBufferState` means carry-or-refit, a family state
means exact fold.

The prior family is the first layer above the moments, and it is the layer where that rule meets a
result that carries its own sample. `LowOrderPrior` holds `X` beside `mu` and `sigma`
(`13_Prior/01_Base_Prior.jl`) because the scenario-based risk measures — CVaR, EVaR, RLVaR, CDaR and
their kin — price it. So an incremental prior saves the moment recomputation and **not** the memory,
and it cannot decline to hold observations the way a moment estimator can.

### What #997 changed under this ticket

[Issue #997](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/997) established that a
wrapper **replaces** an exact fold rather than adding to one: every family narrows the `cache` type
parameter of its own `partial_fit!`, so a wrapped estimator buffers and answers every read-out verb
with the batch verb over the buffer's rows. **The cap is therefore the window**, and a rolling
window is a composition rather than a mode. That ruling is right for a member whose estimate comes
*from* the buffer. A prior's does not.

### What the reference does

Its empirical prior forwards `partial_fit` to the mean and covariance sub-estimators, reads their
fitted moments back, and separately appends the observations to a growing buffer that the fitted
result carries. Its cap on that buffer is a **field of the estimator**, shared by the batch and the
incremental code path, and it truncates the carried scenarios alone: the moments stay fitted over
every observation, which its own documentation states outright. It advertises that cap as the
remedy for a large zero-filled share, and it names each zero-filled asset once per fitted state so
that streaming does not repeat the warning.

### What the library already holds

- **The ledger.** [Issue #862](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/862)
  measured every prior: three are exact by composition and nine refit, because they run a
  time-series regression, an optimisation over the observation weights, or a per-observation
  cross-sectional regression.
- **The fill.** `scenario_fill` writes zero at every missing entry of an investable asset's column
  and reports above the fitting estimator's own `fill_limit`
  ([ADR 0118](0118-a-fold-zeroes-a-held-gap-once-and-a-value-level-verb-reduces-to-the-investable-mask.md),
  [#975](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/975),
  [#976](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/976)). `strict_diagnostic`
  throws under `strict` and otherwise warns, and it has no memory.
- **The composite.** `PortfolioOptimisersCovariance` is the default covariance of `EmpiricalPrior`
  and has no `partial_fit!` at all, so the **default** prior cannot fold.

## Decision

### A prior folds and carries; it does not refit

`EmpiricalPrior.partial_fit!` forwards the observation to `me` and `ce`, which fold exactly, and
appends the row to a carry state of its own. The read-out takes `mu` and `sigma` off the folded
inner states and `X` off the buffer, so the step is `O(N²)` and never `O(t·N²)`. The buffer is
**memory, not arithmetic**: it exists because `LowOrderPrior` carries `X`, not because anything
needs refitting.

Reading the moments back out of the buffer is the shape to refuse in review. It passes every parity
test and throws away both exact folds, which is the arithmetic the seam exists to avoid.

The horizon arm feeds `log1p.(X)` to the inner estimators and puts the **arithmetic** `X` into the
result, so the carry buffer holds the linear rows and the fold transforms on the way in.

### The buffer is owned once, at the bottom of the chain

Only `EmpiricalPrior` puts the caller's own matrix into `X`. `BlackLittermanPrior` returns
`forward_prior(prior_model; …)` and `HighOrderPriorEstimator` builds `HighOrderPrior(; pr = pr, …)`,
both reusing the inner prior's result, so both fold by forwarding to their inner `pe` and own **no**
buffer, reading `pr.X` when they need the sample. This is the argument #975 used to make
`fill_limit` an `EmpiricalPrior` field: the cost is paid once, at the bottom, and rides up through
the result.

Nothing at the prior layer therefore refuses the step. A carrying host always has rows to refit
from, so a refusal would protect nothing, and no type bound and no runtime check is added.

### A host folds what folds and refits the rest from its own rows

A host that carries the observations folds every member that folds and runs the batch verb over its
own rows for every member that does not. `HighOrderPriorEstimator` with a `SemiMoment` `ske` folds
`pe` and `kte` and refits `ske` alone, from `pr.X`.

The caller writes the estimator they would write in batch. They need no wrapper and no knowledge of
the ledger, and no member carries a second copy of `X`. A member is foldable **as a whole**, not per
component: a composite whose transform reads the sample does not fold, and a host refits it
entirely.

This is a second clause on "the state's type is the route" — a stateless member owned by a carrying
host is refit from that host's rows — and it is bounded to a host that must carry the observations
anyway.

### A cap is either a Scenario Cap or a window, and the two have different names

| Written | Meaning | Equals |
| --- | --- | --- |
| `EmpiricalPrior(; max_scenarios = w)` | `mu` and `sigma` folded over all `t`; `X` is the last `w` rows. Batch and online alike. | No batch fit. Documented, not tested. |
| `Online(EmpiricalPrior(); max_history = w)` | The whole fit windowed: one buffer, every read-out over the last `w` rows. | `prior(pe, X[end - w + 1:end, :])`, exactly. |

`max_scenarios` is the reference's knob. `Online`'s `max_history` keeps #997's meaning unbent, and
it is also the cheap route to a rolling-window prior: wrapping `me` and `ce` separately costs three
copies of `X` and three caps to keep in sync. Both set, they nest — moments over `max_history`,
scenarios over `max_scenarios` — and `max_scenarios >= max_history` is a no-op.

The two routes need two state types. A `SampleBufferState` means refit, everywhere in the library. A
fold-and-carry prior seeds a carry state of its own, and its `partial_fit!` binds that type, so a
**wrapped** prior falls to the refit route and the table above holds.

`max_scenarios` is structurally an `EmpiricalPrior` field. A refitting prior needs none: its
estimate comes from the buffer, so capping the buffer windows the estimate, which is what
`max_history` already means.

### A composite covariance folds on a count

`matrix_processing_step!` reads `X` in one step and reads only its shape: `dn` takes the `T / N`
ratio, while `pdm` and `dt` never read it and the default `mp` sets neither `dn` nor `dt` nor `alg`.
So `PortfolioOptimisersCovariance` forwards `partial_fit!` to its inner estimator and its
one-argument `cov` applies `mp` with the folded observation count in place of `size(X, 1)`. The
substitution is exact, because the inner state's count **is** `size(X, 1)`, non-finite entries
included.

A `mp.alg` holding a caller's own algorithm is the one case with no shape substitute, and it is
refused by name at the fold: the algorithm reads the sample, so the composite must be wrapped in
`Online` or hosted by a prior that carries rows for it.

### The fill is unchanged, and it is named once per asset

The buffer holds observations verbatim, non-finite entries included, so the online step does nothing
new to a missing row: no per-step fill, no freeze rule and no second convention. Coverage Universe
parity with batch holds by construction, because `coverage_mask` is a pure function of rows the
buffer holds exactly.

The cadence changes instead. A walk-forward reading out at every step names the same fill at every
step, because `strict_diagnostic` has no memory. The carry state therefore remembers which assets it
has named, and a read-out names only the assets new to that set, so an asset that lists part-way
through a run is still named when it happens. `merge_states` unions the sets and `copy` copies them.
Under `strict` the first fill still throws, as it does in batch.

[Issue #976](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/976) bounds how often that
bites without removing it. It made the fill's share **per asset** and tied `fill_limit` to
`min_coverage` as one number, so `nothing` derives `1 - maximum(min_coverage)` over the arms that
state a floor and never fires there. Two cases still name a fill on every read-out: an estimator
family that states no floor — the exponentially weighted one, which gates on a count — where
`nothing` keeps its meaning and names every fill, and an explicit `fill_limit` tighter than
admission. In both, a long run repeats one notice per step, which is what the named-asset set stops.

### A factor matrix rides in the one buffer, and the estimator tree says whether it is read

Whether a fit reads a factor matrix is a fact of the estimator **tree**, not of the host's type. No
prior whose factor argument is optional reads it: `BlackLittermanPrior`, `HighOrderPriorEstimator`,
`EntropyPoolingPrior`, `MeucciEntropyPoolingPrior` and `OpinionPoolingPrior` hand `F` to the prior
they embed, `EmpiricalPrior` declares it and never reads it, and the cross-sectional prior builds its
own factors off the panel. Only the five members that declare `F::MatNum` read it. So a "half-empty
factor buffer" has no case that reads it, and the seam asks the tree instead
([#1009](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1009)).

`needs_factor_returns(pe)` answers by source shape, three ways: a member that requires factor
returns answers `true`, a member that never reads them answers `false`, and a member whose factor
argument is optional answers `nothing` — *the type does not say; take what the fold is given*, which
is what its batch verb does. Each of the library's optional-argument members recurses into the prior
it embeds, so it answers the leaf's value. A caller's own optional-argument subtype with no method
behaves as its batch verb does.

The factor observations ride **inside** the one Sample Buffer, as an optional backing beside the
masks that share its rows, its offset and its cap; whether the buffer records them is fixed by the
first append, and a mixture is refused in both directions, as it is for the masks. There is no
second buffer type: a refit reads `prior(pe, X, F)` off one state when the state holds factor rows
and `prior(pe, X)` otherwise, and the contemporaneity of the `t`-th rows is structural rather than
asserted.

The fold mirrors the batch verb's arity, `partial_fit!(pe, x, f = nothing)`, and the tree's answer
decides what the fold does with `f`: `true` and no `f` is refused by name with the door's own
refusal, before any row is appended; `false` drops `f`, as the batch verb drops it; `nothing`
records it when it is given. The same answer replaces the shallow `isa` test at the five doors that
check for a missing factor matrix — the prior's `ReturnsResult` door, the optimiser's step, and the
three uncertainty-set doors — which also closes a batch defect: a factor leaf nested under an
optional-argument host met a `MethodError` at the leaf rather than the named refusal written for
it.

`F` is owned once. The optimiser's Fold Context keeps the factor column only when the prior's tree
answers `false`; otherwise the prior's buffer holds it and the read-out reads it back through the
prior, as it reads the returns. A wrapper's cap applies to the factor rows as to the rest, because
it windows the fit. The `AssetPanel` is not buffered: it is fold context, not sample.

## Considered options

**One rule, one knob: a capped prior is a rolling-window prior.** Rejected. It is the simplest rule
to state and the map's oracle would hold with no exception, but the reference's capability becomes
inexpressible: a caller capping to bound memory, or to cut the zero-filled share, would silently
shorten their covariance estimation window, and those are different concerns.

**The reference's semantics on the wrapper's knob.** Rejected. It needs no new field and no second
state type, but there is then no windowed-prior form at all, and `Online`'s ruling would gain an
exception rather than a sibling.

**The fold loop hands `X` at read-out.** Rejected. It costs no field and holds one copy of the
sample, but a prior would only ever be online *inside* a fold loop and could not stream, which is
half of what the map's destination asks for.

**The caller wraps each member that cannot fold.** Rejected for mixed hosts. It leaves "the type is
the route" unbent, but the caller must know the ledger to write a working estimator and each wrapped
member carries its own copy of `X`.

## Consequences

- `EmpiricalPrior` gains `cache` and `max_scenarios`. `show_fields` renders the cap only where it is
  set, so no doctest moves.
- One state type joins `SampleBufferState`: the carry state. The factor rows ride inside the
  Sample Buffer as an optional backing, and `needs_factor_returns` is a per-type predicate of the
  prior family, recursive through an embedded prior.
- `PortfolioOptimisersCovariance` gains `partial_fit!`, a one-argument `cov` and one named refusal.
  Without them the default `EmpiricalPrior()` does not fold.
- A capped fold-and-carry prior equals no batch fit. That divergence is documented, never tested,
  and it is the one hole in the map's batch-parity oracle.
- The build is [#968](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/968), whose
  specification this ADR and the resolution of
  [#704](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/704) carry together. #968
  built the factor rows as a second state type, a pair of buffers; the section above replaces
  that on the ruling of [#1009](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1009),
  and its build folds the pair into the one buffer.
