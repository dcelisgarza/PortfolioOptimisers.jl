---
status: accepted
---

# One `RESOURCE_LIMITS` cap per sink, named after the field it guards

## Context

A recurring class of config→allocation weakness runs through the library: an untrusted sizing
integer (config file, tuning grid, UI) whose own constructor bounds it only from *below*, so an
absurd value — a stray extra digit, a mis-scaled sweep — is accepted and the process is killed by
the OOM killer rather than told what went wrong. The seventh security pass introduced
[`RESOURCE_LIMITS`](../../src/01_Base.jl) (a [`ScopedConfig`](../../src/01_Base.jl) holding a
`ResourceLimits`) with two such caps — `max_samples` (Monte-Carlo draws `n_sim`) and `max_subsets`
(resampled subsets `n_subsets`) — enforced by [`assert_resource_cap`](../../src/01_Base.jl), which
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

```
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
