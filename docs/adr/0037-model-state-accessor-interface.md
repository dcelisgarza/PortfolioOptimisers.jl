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

## Amendment (2026-08-17) — the index is the second axis, and it is now inside the seam

The decision above policed **one** of the two ways a Model State key is disambiguated. A key
carries two independent axes:

- the **prefix**, which separates one *build* from another (ADR 0005), and
- the **index** `i`, which separates one *measure instance* from another inside a single
  build, so two `ConditionalValueatRisk` measures in one vector get their own scratch.

Rules 1 and 2 covered the prefix. The index was spelled by hand — `model[Symbol(:cvar_risk_, i)]`
— at ~200 sites across 21 files, carrying the same do-not-collide invariant and matching
neither rule. Inside `20_RiskMeasureConstraints/` alone, 6 files used the interface only, 9
used bare index keys only, and 5 used both in the same file.

### 1. `state_key` resolves both axes

`state_key(prefix, name, i)` joins `state_key(prefix, name)`, and `state_set!`, `state_get`,
`state_has`, `state_build!` and `mark_state!` each grow the matching indexed method. An
emitter now writes `state_set!(model, prefix, :cvar_risk_, i, expr)` and names neither
convention.

`nested_index(tag, i)` is the index axis's `nested_prefix`: a composite measure that builds
its parts **in the same build** — `set_range_risk_constraints!` over its `loss` and `gain`
tails — separates the parts by index, because they share the build's infrastructure entries
and must not each rebuild them.

### 2. The tracking path stopped carrying the same fact twice

`set_risk_tr_constraints!` took the composed tracking prefix and passed it as the `prefix`
keyword **and** as the seed of the measure index (`Symbol(tprefix, i)`), so a nested CVaR
landed at `:cvar_risk_tr_iv_2_1`. Separation across builds is the prefix's job alone; the
index now means what it means everywhere else, and the entry is `:tr_iv_2_cvar_risk_1`.
Nesting still composes — `nested_prefix` on one axis, `nested_index` on the other — so
tracking-nested-in-tracking is collision-free with each axis stating one thing.

Keys the caller may rely on are unchanged: with the default empty prefix,
`state_key(Symbol(""), :ret_, 1)` is `:ret_1`, exactly as before. Only the *nested* spellings
move, and those were never namable from outside.

### 3. The seam-lock grows a third rule, and it polices the access rather than the spelling

Rule 3 is that outside the interface `model[…]` and `haskey(model, …)` are simply not
reached for. Stating it on the *access* rather than on key construction is deliberate: a
construction-shaped rule stops `model[Symbol(:foo_, i)]` but not `key = Symbol(:foo_, i);
model[key] = v`, which is the same defect with one more line. Like rules 1 and 2, it names
no keys.

Two exemptions, both named by file or function and neither naming a key:

- `mip_key(sp, name)` — the MIP space's own resolver (ADR 0034). It is a *third* namespace,
  but it already has exactly one home, so it is not the defect this amendment fixes.
  Naming it is a claim: a second such resolver fails the lock until someone decides whether
  it belongs in the interface.
- `11_MeanRisk.jl` / `13_NearOptimalCentering.jl` — the frontier sweep loops read back a key
  `set_risk_upper_bound!` minted and handed them through the `:risk_frontier` registry. They
  compose nothing.

### 4. Two latent defects the scope change surfaced

- `set_tracking_error_constraints!` for `DependentVariableTracking` registered the risk
  difference at `Symbol(key, i)` — the *already composed* key with the index appended a
  second time — putting entry 1 at `:te_dr_11`, which is the key `te_dr` itself takes at
  entry 11. It is now its own name, `:te_dr_diff_`.
- `set_variance_risk!`'s `SquaredSOCRiskExpr` overload registered its SOC constraint at
  `Symbol(key_dev, :_soc)` (`:dev_1_soc`) while its `QuadRiskExpr` sibling used
  `:cdev_soc_1` for the same thing. Both now use `:cdev_soc_`.

Neither was reachable in the test suite; both are the class the rule exists to prevent.

### 5. Consequences

- `set_risk_bounds_and_expression!` and `set_variance_risk_bounds_and_expression!` take
  `(name, i)` and resolve the key themselves, so the bound keys (`<key>_ub`,
  `<key>_ub_var`) cannot drift from the key the emitter registered the expression under.
  Several helpers that took a composed `key::Symbol` across a function boundary
  (`set_kurtosis_risk!`, `set_negative_skewness_risk!`, `set_tracking_risk!`,
  `set_second_moment_risk!`, `set_variance_risk!`, `set_ucs_variance_risk!`) now take the
  index instead: a resolved key no longer escapes an emitter at all.
- `arg_dict[:key_sym]` lost its last user and is deleted.
- Rule 3 reports **zero** violations, so the migration is provably complete rather than
  believed complete — the same standard §6 set for rules 1 and 2.

## Amendment (2026-08-17, second) — registration is fresh, because composition is not injective

The amendment above brought the index axis inside the seam. It left one property of the
composition unstated, and the property is load-bearing: **`state_key` is not injective.**
Neither axis carries a delimiter, so a name that ends in a digit at a low index composes the
same `Symbol` as a shorter name at a higher index —

```julia
state_key(prefix, :te_dr_, 11) == state_key(prefix, :te_dr_1, 1) == :te_dr_11
```

That is not a spelling curiosity. §4 above records it as a defect that shipped:
`set_tracking_error_constraints!` put entry 1 at `:te_dr_11`, which is the key `te_dr` itself
takes at entry 11. **The three rules could not have caught it** — each key is spelled through
the interface, each access goes through an accessor, and the collision exists only in the
composed result. What caught it was a person reading the code during the index migration.

The consequence of a collision is a **wrong answer, not a crash**. `state_set!` wrote through
`model[key] = val`, so the second registration replaced the first, the object dictionary
carried one entry where the build needed two, and a later `state_get` handed a constraint the
wrong variable. Nothing warns.

### 1. `state_set!` refuses a key that is already registered

`assert_state_key_free(model, key)` guards both `state_set!` methods. A repeat throws an
`ArgumentError` that names the key, states the non-injectivity, and names the two verbs that
*do* accept a repeat.

Re-registration has no legitimate reading, which is what makes the guard affordable rather
than a behaviour change. The interface already carries both reuse cases in their own verbs:
`state_build!` builds an entry once and returns the existing one untouched, and `mark_state!`
is an idempotent presence flag built on it. `state_set!` is the *fresh registration* verb and
now says so at run time.

The guard closes more than the composition collision. Two emitters that each spell a
perfectly correct key, and happen to spell the *same* one, now fail where one silently
replaced the other's entry.

### 2. A delimiter was rejected as the fix

Making the composition injective — `Symbol(prefix, name, :_, i)` — was considered and
rejected on two grounds.

- It moves **every** top-level key spelling. `state_key(Symbol(""), :ret_, 1)` is `:ret_1`
  today, and that is the one caller-facing surface the amendment above went out of its way to
  hold still.
- It closes only half the class. Two emitters that pick the same *correct* key still collide,
  and a delimiter cannot see that. The guard sees both.

The spelling is therefore kept, the non-injectivity is documented on `state_key` itself, and
the collision is caught where it does harm: at registration.

### 3. The rule is a run-time rule, not a fourth lint

A fourth seam-lock rule was the cheaper alternative: run a census over the shipped name tags
and fail CI on a pair whose composition collides. It was rejected for the polarity §6 argues
for throughout. A census can only see the tags that exist when it runs, and a Model State key
is composed from a `prefix` and an `i` that are **values**, not literals — so a census would
have to enumerate the builds as well as the tags. The run-time guard names no tag and covers
the entry added tomorrow.

A census was still run, as evidence and not as the mechanism: over the **236** literal name
tags the emitters pass to the six `state_*` verbs, **no pair collides** — no tag is another
tag followed only by digits. The defect §4 records was the only instance, and it was already
fixed. The guard exists for the next one.

### 4. Verified

The guard is only affordable if no shipped build re-registers a key, and that is a claim about
**run time**, not about the source. It was measured by building models rather than by reading
code: 14 targeted test files, **1,287 passes and 0 failures**, covering the paths that register
the most Model State entries — `test_27_prefix_registration.jl` (251, every nested tracking
build), `test_18g`/`test_18h` (the two tracking algorithms, 137 + 94),
`test_16a_asset_risk_budgeting.jl` (235), `test_18j_formulations.jl` (113),
`test_18k_constraints.jl` (117), `test_18m` (97), `test_18o` (63), `test_20` (near-optimal
centering, 39), `test_03b` (35), `test_18n` (32), `test_19` (26), `test_22b` (10) and
`test_28_seam_lock.jl` (38). Not one build hit the guard.

`test_28_seam_lock.jl` gains the run-time half of the rule, next to the `SHARED_STATE` check
that plays the same part for rule 2: it asserts the two composed keys are equal, that the
second registration throws, that the first entry survives, and that `state_build!` and
`mark_state!` keep their idempotence.

Two reference-weight drifts surfaced while running the census and are **unrelated to the
guard** — they reproduce with it bypassed. `MeanRiskIT.csv.gz` columns 16
(`EntropicValueatRiskRange`) and 25 (`Kurtosis`) had drifted by 6.1e-6 and 1.6e-6 in absolute
weight, past their own `rtol`, against the range and index reworks recorded in ADR 0057 and in
the amendment above. Both columns were regenerated; the other 46 were left byte-identical.
