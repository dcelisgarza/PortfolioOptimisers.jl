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
