---
status: accepted
---

# One `RESOURCE_LIMITS` cap per sink, named after the field it guards

## Context

A recurring class of config→allocation weakness runs through the library: an untrusted sizing
integer (config file, tuning grid, UI) whose own constructor bounds it only from *below*, so an
absurd value — a stray extra digit, a mis-scaled sweep — is accepted and the process is killed by
the OOM killer rather than told what went wrong. The seventh security pass introduced
[`RESOURCE_LIMITS`](../../src/01_Base/04_ScopedConfig.jl) (a
[`ScopedConfig`](../../src/01_Base/04_ScopedConfig.jl) holding a
`ResourceLimits`) with two such caps — `max_samples` (Monte-Carlo draws `n_sim`) and `max_subsets`
(resampled subsets `n_subsets`) — enforced by [`assert_resource_cap`](../../src/01_Base/10_Assertions.jl), which
fails closed with a typed `DomainError` naming both the rejected field and the knob that raises it.
This mirrored [ADR 0027](0027-cap-equation-parser-recursion.md) but was never itself recorded.

The eighth security pass (`docs/reports/security-review-20260723-005959.html`) found two more sinks
of the same class that the mechanism had not reached: `Frontier.N` (each frontier point runs a full
inner `optimise_JuMP_model!` solve) and `bins` on the mutual-information estimators (the joint
histogram is a `bins × bins` weights matrix per asset pair). A grill of that report surfaced a third
— `VariationInfoDistance.bins` — the exact same histogram sink the report had missed, and showed the
report's proposed fix for `bins` was itself wrong: it reused the linear `max_samples` cap, but the
sink is *quadratic* (`bins²`), so `bins <= max_samples` fails to bound the allocation at all
(`bins = 50_000` passes the `≤ 1_000_000` check yet allocates `2.5×10⁹` cells).

## Decision

**Every distinct sizing sink gets its own `ResourceLimits` field; caps are never reused across
sinks; and each cap is named to mirror the field it guards.** `ResourceLimits` now holds four caps —
`max_n_sim`, `max_n_subsets`, `max_frontier`, `max_bins` — the first two being a **breaking rename**
of `max_samples`/`max_subsets` (kwargs of `set_resource_limits!`/`with_resource_limits` and the
`LocalPreferences.toml` keys change with them). Construction goes through a keyword constructor
`ResourceLimits(; …)`, since the four fields are same-typed and two share the value `100_000`, making
positional construction error-prone.

The no-reuse rule is load-bearing, not cosmetic: **a linear cap cannot bound a quadratic sink.** A
compute sink (one unit → one solve: `n_subsets`, `Frontier.N`) is bounded by a linear cap on the
count; a `bins × bins` memory sink is not, so `bins` gets a distinct `max_bins` whose default
(`10_000`, ≈ 800 MB per histogram) is chosen against the *squared* footprint. Naming each cap after
its field makes the `assert_resource_cap` message (`"$sym exceeds RESOURCE_LIMITS[].$knob"`)
self-documenting: a caller who set `n_sim` and hit `max_n_sim` sees the link immediately.

## Consequences

- Existing `LocalPreferences.toml` entries and code using `max_samples`/`max_subsets` break and must
  migrate to `max_n_sim`/`max_n_subsets`. Preferences.jl cannot enumerate set keys, so a stale key is
  silently ignored (the shipped default applies) rather than erroring — the rename is silent at load.
- Every future sizing sink of this class is expected to add its own field rather than borrow an
  existing one; borrowing is the specific mistake this ADR exists to prevent.

## Amendment (2026-08-16)

**A cap must be measured against the sink, not against the field that names it.** The decision
above classified `Frontier.N` as a compute sink of the linear kind — "one unit → one solve" — and
placed the check in `Frontier`'s constructor, where a single `N` is in hand. That classification was
wrong about the sink, in the same way the report's `bins` proposal was wrong about `bins`.

The efficient-frontier sweep is an `Iterators.product`. Each swept risk measure pushes its own entry
onto `:risk_frontier`, and since [ADR 0052](0052-a-return-expression-is-a-weighted-sum-of-terms.md)'s return
multiplicity each swept return term pushes its own onto `:ret_frontier`; the solve loops range over
the product of all of them. So `k` bounds of `N` points each cost `N^k` full solves, not `k · N`.
`Frontier`'s constructor sees one `N` and can never see the product, so **two risk measures at the
`100_000` ceiling asked for `10^10` solves and no guard fired** — a hole in the same trust boundary
this ADR was written to close, present since the risk frontier first accepted several measures.

The rule is therefore two checks against one ceiling, not two ceilings:

```julia
Frontier(N)   assert N <= max_frontier                 # the cheap early check, unchanged
assembly      assert prod(Nᵢ) * prod(Nⱼ) <= max_frontier
```

This is **not** the cap reuse the decision forbids. Reuse is one ceiling standing in for two
*different* sinks, as `max_samples` would have for `bins`. Here there is one sink — total solves —
and the constructor check was only ever a lower bound on it; the assembly check measures the same
sink correctly. Adding a second field would ask a caller to reason about two numbers for one cost.

`assert_frontier_sweep_cap` (`src/20_Optimisation/08_Base_JuMPOptimisation.jl`) is the second check,
called as the last statement of `assemble_jump_model!`. Model Assembly is the earliest point at
which the whole sweep is in hand — both frontier registries are complete there, and no sweep solve
has started. A `Frontier` is *not* yet resolved into its range at that point (resolution happens in
`compute_ret_lbs` / `compute_risk_ubs`, which pay corner solves to do it), so the count is read off
the shape: a `Frontier`'s `N`, or a stated bound vector's `length`. The product is accumulated as a
`BigInt`, because four bounds at the shipped ceiling is `10^20` and an `Int64` product would wrap
past the very check it is feeding.

The generalisation for future sinks: **when a sink is fed by several instances of a config object,
the cap belongs where the instances meet, and a per-instance check is at best an early exit.**

Behaviour-changing, not API-breaking: a configuration whose product exceeds the ceiling now raises a
`DomainError` naming the product, the bounds that made it, and the knob. The ceiling is raised
deliberately as before — `set_resource_limits!(; max_frontier)`, `with_resource_limits` for a single
scope, or the `"max_frontier"` preference for a project.

## Amendment (2026-08-17)

The ninth security pass (`docs/reports/security-review-20260816-230218.html`) found two more sinks of
this class, one of each shape the ADR already distinguishes. `ResourceLimits` therefore holds **six**
caps: `max_hop_count` and `max_search_grid` join the four above, with the same naming rule and the
same no-reuse rule.

`max_hop_count` (default `100_000`) bounds `HopCount.n`. Three readers sum `A^i` over `i in 0:n`, so
the sink is linear in `n` at `N³` flops a power: a compute sink of exactly the `max_n_subsets` shape,
and it takes the linear cap that shape calls for. `HopCount`'s constructor was bounded only from
below (`n >= 1`), so `HopCount(; n = 10^9)` was accepted while the structurally identical
`get_n_subsets(10^9)` failed closed.

The check goes in the constructor **alone**, and that is the whole of it: `resolve_separation` builds
its answer with `HopCount(n)`, so a hop count computed by a caller-authored rule meets the same cap
as a stated one. Where the frontier needed two checks because the constructor could not see the
product, here one constructor sees every value that ever reaches the sink.

`max_search_grid` (default `100_000`) bounds the search cross-validation grid, and it is the
frontier's lesson applied a second time. The grid is an `Iterators.product` materialised by
`collect`, so `k` tuned parameters of `N` values cost `N^k` candidates and `N^k` full cross-validated
fits; `RandomisedSearchCrossValidation` checked `n_iter` as a scalar, which is a linear check on a
product sink. `assert_search_grid_cap`
(`src/20_Optimisation/02_CrossValidation/09_Base_SearchCrossValidation.jl`) is called by
`lens_val_grid` and `pipeline_lens_val_grid` before the `collect`, accumulates the product as a
`BigInt`, and names each parameter's value count in the message. Concatenated parameter sets are a
**sum** of products rather than a product, so each set is capped as it is built and the concatenated
total is capped again on the way out — a per-set check alone would let `k` sets at the ceiling
through, which is the hole this ADR exists to prevent.

Behaviour-changing, not API-breaking, on both counts. `ResourceLimits`'s positional constructor takes
six arguments now; the keyword constructor, which the decision above already made the recommended
path, is unaffected.

## Amendment (2026-08-17) — a preference file may raise a cap, but not in silence

The tenth security pass (`docs/reports/security-review-20260817.html`, finding 1) asked a question
this ADR had left open. The decision above accepts that a *caller* resizes a cap:
`set_resource_limits!` for the session, `with_resource_limits` for one scope. It never decided
whether a **file** may, at load, with no code run. `__init__` reads eleven Preferences.jl keys of the
active project and applies them through the same setters, and the only value check is
`Integer && !Bool && > 0` — so one `LocalPreferences.toml` raises every cap the library has, silently,
before any user code. That channel is the one an attacker reaches without running code: the file is
data, it travels with a cloned project or a template, and it is often untracked.

**A load-time preference keeps its full range, and a value that widens a guard is announced with a
warning.** The ceiling was the alternative and it is rejected. The caps are deliberately far above
legitimate use — they exist to convert an OOM kill into a typed error, not to second-guess a sizing
choice — so a project on a machine sized for a genuinely large run has a legitimate reason to raise
one, and that reason does not disappear because the value lives in a file rather than in a call. A
ceiling would also need a second number per cap, which is exactly the "two numbers for one cost" the
frontier amendment above refuses. What the channel actually lacked was not a bound but a **record**:
nothing distinguished a project that chose a wide cap from a project that inherited one.

Widening is direction-typed, not "any change": a raised `ResourceLimits` or `EquationLimits` cap, and
a *lowered* `STRING_DISTANCE.min_score` — the info-leak direction of
[ADR 0026](0026-lenient-constraint-names-with-suggestions.md), since a lower threshold admits more
candidates. A value that tightens a guard, or that matches the default it replaces, stays silent: a
project that hardens itself must not be trained to ignore a warning at every `using`. The comparison
is against the default *in effect when the preference is applied*, which at load is the shipped
default; `apply_preferences!` therefore needs no second copy of the shipped numbers, which could
drift from the constructors that own them.

The warning is one message for the whole load, built by `relaxed_preferences_msg`
(`src/01_Base.jl`), naming each widened guard as `key: default → value` and the file the values came
from. It names the keys it reports and nothing else — the same info-leak-safe message discipline as
`unknown_variable_msg`.

The fail-closed behaviour for an *invalid* value is unchanged, and the docstrings that described it
were corrected: they claimed the load "fails closed rather than silently running with a weaker cap",
which covered invalid values alone and read as a promise about weak caps in general.
