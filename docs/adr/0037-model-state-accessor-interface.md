---
status: accepted
amends: 0004 §2 and §6.5, 0005 (final bullet — lock scope; `preg!` renamed)
---

# Model State is reached only through typed accessors, and the seam-lock fails closed

## Context

[ADR 0004](0004-typed-jump-model-state.md) gave the JuMP model state a named accessor
interface and accepted, in §2, that this is "a discipline seam, not a compiler-enforced
one" — enforcement is a test (§6.5), not the type system. [ADR 0005](0005-prefix-namespaced-risk-state.md)
then namespaced per-build risk state by key prefix, and scoped that test to **literal**
`model[:` reads of an enumerated list of prefix-managed keys.

Both ADRs named the resulting hole and accepted it: a sloppy `Symbol(prefix, :newCatA)`
looks like Category-B scratch and slips through. Measured on `dev` before this change, the
cost had two parts, only one of which the earlier ADRs named:

- **52 hand-built key sites** across 15 files outside the interface, each spelling the
  namespacing convention itself — `haskey(model, Symbol(prefix, k)) → model[Symbol(prefix, k)]`
  … `preg!(model, prefix, k, …)`. The convention was interface, not implementation.
- **The lock's allowlist enumerated the *managed* keys** (40 symbols, hand-synced, with a
  standing "keep this list in sync" comment). That polarity is the real defect: forgetting
  to add a newly-managed key *silently opens a hole*. The list could only ever describe
  keys someone had already thought about.

A bare read of a prefix-managed entry is not hypothetical — it is the regression class that
broke `IndependentVariableTracking` (the OWA / PowerNorm / Turnover / DR-CDaR gaps) and made
`set_iplg_constraints!` throw `KeyError(:ib)` on a short + threshold/fee/xbgt model
(ADRs [0033](0033-split-mip-file-into-indicator-layer-and-emitters.md)/[0034](0034-mip-indicator-bundle-lives-in-model-state.md)).

## Decision

### 1. One memoise combinator, not per-key accessors

Model State grows five operations in
[08_Base_JuMPOptimisation.jl](../../src/20_Optimisation/08_Base_JuMPOptimisation.jl):
`state_key`, `state_set!`, `state_has`, `state_get`, `state_build!`, plus `nested_prefix`.

`state_build!(f, model, prefix, name)` is the load-bearing one: it is the
memoise-on-prefixed-key idiom every emitter shared, with the key resolved *inside* the
interface. Because resolution moved, **an entry added in future participates in the prefix
discipline with no further work** — which is precisely what closes ADR 0004 §2's hole.

Per-key named accessors (`get_W`, `get_Au`, …) were rejected as the primary mechanism: they
cannot close the hole by construction, since key #21 has no accessor until someone writes
one. A `@prefixed` macro was rejected for hiding control flow in codegen, and because it
would not touch nested-prefix construction at all.

### 2. Named accessors coexist; they are not subsumed

The existing `get_X`/`has_dd`/… pairs stay and are reimplemented over the new primitives.
They carry something `state_get` structurally cannot: a *specific* diagnostic ("portfolio
returns have not been registered; call `set_portfolio_returns!` first"). Losing that for the
six entries with a known producer would make out-of-order reads harder to diagnose, not
easier. The cost accepted: two legitimate spellings exist for those six entries.

### 3. A nested prefix is not a key

`nested_prefix(prefix, tag, i)` names what `Symbol(prefix, :tr_iv_, i, :_)` was doing:
composing a **namespace**, not an entry name. Folding it into the key machinery is what made
the old lock's exemption logic murky. Separating them lets the lock state one rule per
concept.

### 4. `preg!` becomes `state_set!`

`preg!` ("prefix-register") left the glossary term *Model State* with no fingerprint in the
code — a reader could not get from the concept to the mechanism. All five operations now
share the `state_*` prefix, so the concept is greppable. 57 call sites, mechanical.

### 5. Bare entries are reached through `shared_*`, validated at run time

Deliberately-unprefixed entries go through `shared_get` / `shared_has` / `shared_set!`,
which assert the name is on `SHARED_STATE` — the enumerated complement of Per-Build Risk
State, defined next to the interface it guards.

Routing them through an accessor is only worth anything *because* of that validation. An
unvalidated `shared_get(model, name)` would accept `:W` and merely rename the hole. With the
assertion, reaching for a per-build entry without a prefix **throws at the call site**, so
the classification is enforced by the code and not only by CI.

Two things deliberately stay as they are. Entries registered by *named* JuMP macros
(`JuMP.@variable(model, bucs_w[1:N])`, `JuMP.@expression(model, Gkt, G)`) keep that
registration: converting them to anonymous values plus an explicit register would strip the
names from the printed model and from solver diagnostics. Only their reads and probes move.
And the OWA weight-fitting solve and the discrete-allocation MIP build a **different**
`JuMP.Model`, so they are exempt wholesale — Model State vocabulary does not describe them.

### 6. The seam-lock inverts its polarity — the key decision

[test_28_seam_lock.jl](../../test/test_28_seam_lock.jl) now enforces two rules:

1. **Construction** — no `Symbol(prefix` outside the interface. This rule names *no keys*,
   so it is closed: it covers entries that do not exist yet.
2. **Bare access** — no literal `model[:key]` / `haskey(model, :key)` outside the interface
   **at all**. Shared entries route through `shared_*`, which validates against
   `SHARED_STATE` at run time (§5).

Rule 1 alone is *not* sufficient, and believing otherwise was an error made partway through
this work: it stops key **construction** but not a fresh bare literal `model[:W]`. Rule 2
catches that. Crucially, once shared entries go through `shared_*`, **neither rule names a
key** — the test carries no allowlist at all, and the one list that remains (`SHARED_STATE`)
lives in the source beside the interface it guards.

The polarity is the point. The old lint enumerated the *managed* keys, so forgetting to list
a newly-managed key silently opened a hole. `SHARED_STATE` enumerates the deliberately-
**shared** entries, so forgetting to classify a new key fails — at run time, and in CI. That
inversion, not the accessors, is what converts the invariant from "remember to register your
key" into one the tooling tells you about.

One entry is called out specially. `:variance_flag` is written **prefixed** and read
**bare**, and that asymmetry is deliberate (ADR 0005): the inner write is prefixed so a
nested variance cannot leak its presence outward, while the only readers are outer-level
phylogeny builders that add a `p·tr(W)` penalty when no variance is present and so must see
the *outer* flag. Prefixing that read would silently re-add the penalty under tracking.

## Consequences

- The 52 hand-built key sites are gone; the construction rule reports **zero** violations,
  so the migration is provably complete rather than believed complete.
- The seam-lock's hand-synced managed-key list is deleted, and **the test now contains no
  key list of any kind**. The one remaining list, `SHARED_STATE`, sits in the source next to
  the interface it guards, fails closed, and is enforced at run time as well as in CI.
- The ~101 bare `model[:key]` / `haskey(model, :key)` sites across 14 files were migrated to
  `shared_get`/`shared_has`/`shared_set!`. Their classification was audited first; no bug was
  found, which is the expected outcome — the value is that the classification is now
  *enforced* rather than merely true.
- A dead `#=` block (~130 lines, lines ~197–330 of
  [02_Returns_and_ObjectiveFunctions.jl](../../src/20_Optimisation/09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl))
  was deleted. It targeted an older `port` API (`Sharpe`, `AKelly`, `calc_variance_risk` —
  none of which exist), and it was the only source of `:variance_risk`, `:dev`, `:scale_obj`
  and `:scale_constr`. Keeping it would have meant carrying four dead names in a
  runtime-validated set, implying they were live shared state.
- The lock now also skips `#=` block comments. It previously did not, which mattered once
  bare reads were policed: `09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl` carries
  a dead legacy `port`-API block (lines ~197–330) whose `model[:variance_risk]`,
  `model[:scale_obj]`, `model[:scale_constr]` and `model[:dev]` reads are not live code.
- **Behaviour is unchanged and was verified as such.** The oracle was a byte-for-byte diff of
  `sprint(print, model)` over 38 models covering all 15 migrated files, with a coverage
  proof that every one of the 39 migrated entry names is actually registered by some case.
  Solve-value tests were *not* used as the primary oracle: this refactor's risk surface is
  constraint and variable **names**, and weights assertions are nearly blind to those, while
  `:risk_frontier` keys off `(bound_var_key, bound_key)` pairs — so silent name drift would
  corrupt frontier results without failing a weights test.
- `set_risk_bounds_and_expression!` now takes a bare name plus a `prefix` keyword and
  resolves the key itself. This keeps key symbols from escaping emitters, so the lock stays
  one closed pattern rather than growing an escape-hatch exemption.
- **Still out of scope**: `SHARED_STATE` records and enforces the classification of the
  shared entries; it does not restructure them. Several are plausibly *per-optimiser* rather
  than model-wide (`:noc_rk`, `:noc_rt`, `:psi`, `:w_obj`) and could earn their own
  narrower home later. That is a separate question from whether they may be read bare, which
  is the one this ADR answers.
