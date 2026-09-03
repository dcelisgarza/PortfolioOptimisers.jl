---
status: accepted
---

# The update seam has two verbs, and a view slices a state by asset and drops it by observation

## Context

[ADR 0106](0106-a-partial-fit-state-is-the-one-result-an-estimator-holds.md) put the running state
of an incremental fit in an optional field on the estimator, and named the field's type bound as
the enforcement. It settled *where* a state lives. It did not settle what the verb that writes one
promises, nor what a propagation channel does with one, and issue #712 found that the shipped
families had answered both questions differently.

Three facts shaped this decision. Each was read in the code on 2026-09-03, on `dev` at
`916af9aa06` plus the two draft branches of #700 and #703.

- **The shipped aliasing promise was not one a caller could use.** The `partial_fit!` docstring of
  #701 stated that the returned estimator shares the state object with the one it was built from,
  so two estimators returned by successive calls read the same running quantities. The methods of
  #700 and #701 mutate the array fields and rebind the scalar fields with `Accessors.@reset`, so an
  estimator that was not rebound holds moved arrays and an old count. It reads neither the old
  sample nor the new one, and the sentence was false in its detail.

- **The two families diverged.** The higher-moment methods of #703 build a fresh state and rebind
  the whole `cache` field, because the merge of two blocks is one derivation shared with
  `merge_states`, which is asserted associative over three disjoint blocks. So an estimator a
  caller kept from an earlier call still read the sample it was shown. The second-order methods
  aliased. Both were correct against their own docstrings, and no reader could hold one rule.

- **The reason #308 gave for `Accessors.@reset` held only without aliasing.** That reason is that
  two folds of a cross-validation sharing one estimator cannot contaminate each other. Under
  aliasing, two folds that start from one warm estimator write into the same arrays, which is
  exactly that contamination.

The propagation question is the second half. `cache` carried no tag, so `factory`, `port_opt_view`
and `obs_weights_view` all passed it through unchanged. A state describes the sample it was fitted
on, so a carried state answers over assets a view excluded, and over observations a window never
selected. The behaviour was written down in the field text, which is a note rather than an
enforcement.

## Decision

### The seam has two verbs, and neither promises aliasing

**`partial_fit!` is the method each family writes.** It is the family's cheapest exact fold. It
writes into the arrays of the state where it can, rebinds the scalar fields with
`Accessors.@reset`, and returns the estimator. It promises **nothing** about an estimator the
caller kept from before the call, and its docstring names three pitfalls in place of the promise
it made before.

- A kept estimator holds a state whose arrays moved and whose count did not, so it reads neither
  the old sample nor the new one.
- Two folds that start from one warm estimator write into the same arrays, so they contaminate
  each other.
- A family whose fold builds a fresh state leaves the kept estimator valid by accident, and no
  caller may rely on that.

**`partial_fit` is one generic method on the seam, and it has value semantics.** It copies the
state, and calls `partial_fit!` on the estimator that holds the copy. This is the pair
`matrix_processing` and `matrix_processing!` make. The estimator handed over is untouched, so the
no-contamination reason of #308 holds for this verb. A family writes no method for it.

**The higher-moment family is the one exception, and it overrides `partial_fit`.** Its fold builds
a fresh state in either verb, so the generic copy buys nothing and costs a copy of `M4`, which is
`assets² × assets²`: 800 MB at a hundred assets, per call.

**Both verbs are exported.** The seam's public surface is two names and two Capability Catalogue
entries. The warning about the pitfalls is the docstring of `partial_fit!`. No runtime warning is
emitted, because the fold runs once per observation.

**Each state family owes one `Base.copy` method that names its constructor**, the way the prior
carriers name theirs. The `AbstractPartialFitState` interface therefore lists two methods,
`merge_states` and `copy`.

### A view slices the state on the asset axis, and drops it on the observation axis

| Channel | What it does to `cache` | Mechanism |
| --- | --- | --- |
| `factory` | carries it unchanged | `@fprop`, through `factory_child`, which reaches the identity `factory(::AbstractResult)` |
| `port_opt_view` | slices it to the selected assets, **by index copy** | `@vprop`, and one `port_opt_view` method per state family |
| `obs_weights_view` | drops it, so the viewed estimator carries `nothing` | `@fprop`, and one root method `obs_weights_view(::AbstractPartialFitState, ::Any) = nothing` |

So the tags are `@fprop @vprop cache`, on every propagatable estimator of the seam, and **no new
tag is needed**. Issue #712 said that no tag expresses a drop, and that was false: `@fprop` routes
the field through `obs_weights_view`, and a root method that returns `nothing` is the drop.

**The slice copies by index and does not `view`.** Under the first ruling a `partial_fit!` on a
viewed estimator writes into the arrays it was handed, so a `view` would write through into the
arrays of the estimator the view was taken from. The copy is small: a cluster's sub-tensor of `M4`
is quartic in the cluster size, so ten clusters of ten assets copy 80 KB each out of 800 MB.

**The slice is exact where it exists.** A Welford mean of one asset reads that asset's
observations alone, and an accumulator of one pair reads those two assets' observations alone, so
the slice of a second-order state is the state of the sliced universe, entry for entry. The same
holds one and two orders further up. For the higher moments the column of a pair `(p, q)` is
`(p - 1) N + q`, so the slice of `M3` keeps the rows `i` and the columns of every pair drawn from
`i`, and the slice of `M4` keeps those columns on both axes.

**No slice of a state exists on the observation axis.** Removing an observation from a running
accumulator has no numerically stable inverse, which is the argument #308 used to refuse the five
windowed estimators. So that channel drops, whatever the asset channel does.

**A family whose state has no exact asset slice returns `nothing` and names the reason**, as it
does for the merge. `RegimeAdjustedVarianceCache` is that family: its regime state reads the
standardised innovations of every asset in the universe, so the regime state of a subset is not a
function of the state. Its estimator is not `@propagatable` today, so no channel reaches it, and
the method is owed the day it becomes one.

### A read-out refuses a state its estimator no longer matches

`factory` carries the state, and `@wprop` replaces `w` when an `ObsWeights` is threaded. The
estimator then says weighted and holds a state fitted unweighted. **Every state read-out runs the
same configuration guard the fold runs**, and throws a named `ArgumentError` when `w` or `me` does
not match the state. The state stays on the estimator, so a caller who restores `w = nothing`
reads it again.

## Consequences

- `partial_fit` joins `partial_fit!` on the seam's public surface, and the Capability Catalogue
  gains a second entry.
- A new state struct owes two methods, `merge_states` and `copy`, and the interface section of the
  `AbstractPartialFitState` docstring states both. ADR 0106 is amended to match.
- The `cache` field of every propagatable estimator of the seam carries `@fprop @vprop`. A new
  estimator that forgets the tags carries a state through a view, which is the defect this decision
  removes, so the tags are what a reviewer looks for.
- A family that adds a state and no `port_opt_view` method for it reaches the universal
  `port_opt_view` fallback and carries the state unchanged. The rule is that such a family writes
  the method or writes the `nothing` refusal, and the field text states it.
- `arg_dict[:pfcache]` states the channel rule for every estimator of the seam, in one place. It no
  longer says that a restricted estimator must be fitted again, because a restricted estimator is
  now either sliced exactly or refused.

## Alternatives considered

- **Narrow the general `partial_fit!` docstring, and let each family alias or not.** Refused. It is
  the cheapest option and it leaves the caller with no verb whose promise is usable: every call site
  would have to know which family it holds.
- **Hold the aliasing promise, and write the merged accumulators back with `copyto!`.** Refused. It
  costs one `assets² × assets²` copy per call for the higher-moment family, and it buys a promise
  that #308's own reason argues against. Under full sharing two folds from one warm estimator
  contaminate each other, and the scalar count would have to be shared as well, which `@reset`
  refuses.
- **Leave the propagation channels carrying the state, and keep the note.** Refused. The note is
  written where a reader of the field text meets it, and a read-out that answers over the wrong
  assets is silent, not loud.
- **A new propagation tag that means *drop*.** Refused, because `@fprop` plus a root method that
  returns `nothing` already is the drop, and a new tag would be a second way to say one thing.
- **Slice on the observation axis too.** Refused. No numerically stable inverse of a Welford update
  exists, which is the same reason the seam refuses a windowed estimator.
