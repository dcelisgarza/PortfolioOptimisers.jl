---
status: accepted
---

# A partial-fit state is the one Result an estimator holds

## Context

`CLAUDE.md` states one rule without exception: *Estimators never hold Results internally.
Enforce it with the field's type bound, not a runtime check; precomputed structure belongs on the
Result type.* The rule keeps an Estimator a pure configuration, so the same Estimator answers any
input it is given, and two callers that share one Estimator cannot contaminate each other.

Issue #308 decided where the running state of an incremental fit lives. An incremental fit folds
one observation into an estimate without reading the sample again, and it needs a place to keep
the running observation count, the running mean and the running second-moment accumulator between
calls. The decision put that state in an optional field on the Estimator, bound to
`Union{Nothing, <:AbstractPartialFitState}` and defaulting to `nothing`.

The state is a Result under the library's own definition, so the decision breaks the rule as it
stands. Three facts shaped the exception rather than a rewrite of the rule.

- The library already ships the shape. `RegimeAdjustedVarianceCache` held the running state of
  the regime-adjusted online recursion, and it subtyped `AbstractResult`. Its own docstring calls
  it an implementation detail that is not intended for direct use, so the general Result root read
  as the wrong parent for it.
- The alternatives each cost more than the exception. A state threaded by the caller puts the
  bookkeeping in every call site. A `Dict` payload puts a hash lookup and a boxed load in the
  innermost loop, which is the cost the seam exists to remove.
- The bound is narrow. `Union{Nothing, <:AbstractPartialFitState}` admits one small family, so a
  field carrying it still refuses an arbitrary Result.

## Decision

**A partial-fit state is the single named exception to the rule that an Estimator holds no
Result, and the exception is enforced by the field's type bound.**

The exception rests on one property: **a state is not consumable.** The rest of the library reads
an ordinary Result, and no consumer of a moment, a prior or a risk measure reads a state. A
read-out verb turns a state into the ordinary Result first, so a state never reaches a consumer
in place of the answer it stands for. That is what separates it from the Result the rule refuses,
which *is* the answer and would let an Estimator serve a cached answer for an input it was never
given.

**The state takes its own root.** `AbstractPartialFitState <: AbstractResult` is unexported, and
every state struct subtypes it. It stays under `AbstractResult`, so the length-1 iteration
protocol and the pretty `show` come free and neither `Union` needs an edit.

**A field that holds a state is bound to `Union{Nothing, <:AbstractPartialFitState}`, and defaults
to `nothing`.** The bound is the enforcement the rule already asks for. A wider bound would admit
the Result the rule refuses, so the bound is the rule, not a comment about it.

**Every state answers `merge_states`, with the fold or with a refusal.** Two states fitted on
disjoint blocks of observations merge into the state of the concatenated block whenever the
statistic is a sum of blocks, which is what makes an incremental fit parallel and associative.
`assert_mergeable_states` refuses a pair of different types and a pair over different numbers of
assets, and a family whose merge needs more adds a method of its own that calls it first.
`chan_merge` carries the mathematics of the merge, once, for every family.

**A family whose state is not a sufficient statistic for its block refuses the pair instead, and
names the reason.** `RegimeAdjustedVarianceCache` is that case, and issue #701 measured it. Its
exponentially weighted accumulator does fold, as `decay^n_B * v_A + v_B`, but the regime state that
scales the answer reads each observation's standardised squared innovation, gated by the running
observation count. A block fitted from a cold start therefore skips its own first `min_obs`
observations of the regime state, and no function of the two block states can put back what neither
of them recorded. The family's route is a sequential `partial_fit!`, which is exact.

**The verb is `merge_states`, not `Base.merge`.** `Base.merge` on a `Dict` and on a `NamedTuple`
means that the right operand wins a key conflict. This merge is a sum, so the borrowed name would
state the wrong contract. The repository's own idiom settles it too: `merge_linear_constraints`
and `merge_partial_linear_constraints` are the only other `merge_*` verbs, and both are
unexported.

**`RegimeAdjustedVarianceCache` is re-parented to the new root.** It is the shipped instance of
the shape, so it moves under the root that names what it is.

## Consequences

- `CLAUDE.md` names this ADR beside the rule, so a reader who meets the rule meets its one
  exception at the same time.
- `CONTEXT.md` gains the two terms the seam introduces: Partial Fit State, and the merge of two
  states.
- A new state struct owes exactly one method, `merge_states`, and the interface section of the
  `AbstractPartialFitState` docstring states it. The method is the fold when the statistic is a sum
  of blocks, and a refusal naming the reason when it is not.
- The exception is closed. A field of any other Result type on an Estimator is still refused, and
  the type bound is what refuses it. Widening the bound is what a reviewer looks for.
- `RegimeAdjustedVarianceCache` changes supertype. It is unexported and no doctest renders its
  supertype, so no expected output moves.

## Alternatives considered

- **Thread the state through the caller.** Refused. It is the shape the library shipped, and it is
  why no state crossed a call: every call site would carry the bookkeeping, and a cross-validation
  fold would have to route the state by hand.
- **A `Dict{Symbol, Any}` payload on the Estimator.** Refused. The update runs once per
  observation, so a hash lookup and a boxed load land in the innermost loop. The immutability the
  `Dict` was wanted for is already there: a state is an immutable struct whose array fields are
  mutated in place.
- **Drop the rule instead of naming an exception.** Refused. The rule holds for every other Result,
  and it is the reason an Estimator is safe to share. One named exception with a narrow type bound
  keeps the rule readable and keeps the enforcement mechanical.
- **A mono state struct for every estimator.** Refused. A struct that fits one estimator and leaves
  a field unused in another carries a dead field, which no reader can interpret.
- **`Base.merge` for the merge.** Refused, for the contract its `Dict` and `NamedTuple` methods
  already state.
