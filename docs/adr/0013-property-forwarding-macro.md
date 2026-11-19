---
status: accepted
---

# Generate property forwarding with a standalone `@forward_properties` macro

## Context

Architecture-review candidate 2 (`docs/architecture-review-20260615-025911.html`). The
report framed this as "~10–12 wrappers hand-copy the same delegate-to-child `getproperty`
template" and proposed a `@delegate_properties T => child` macro. Reading the 15 actual
`Base.getproperty` definitions in `src/13_Prior` and `src/20_Optimisation` shows the
family is not one template but **three distinct shapes**:

- **Shape A — specific-field aliasing (8 sites).** Map a fixed, small set of names onto a
  nested path, e.g. `:me → obj.pe.me`, `:ce → obj.pe.ce`. Six are byte-identical
  (`FactorPrior`, `BlackLittermanPrior`, `BayesianBlackLittermanPrior`,
  `FactorBlackLittermanPrior`, `EntropyPoolingPrior`, `HighOrderFactorPriorEstimator`,
  `HighOrderPriorEstimator`); `AugmentedBlackLittermanPrior` is a 4-way variant that also
  **renames** (`:f_me → obj.f_pe.me`). These are *prior estimators*, not results, and do
  **not** forward "everything" — they hoist named accessors.
- **Shape B — full delegation (4 sites).** "Own fields → else forward all of a named
  child's properties." `HighOrderPrior → pr`, `FactorRiskContributionResult → rr` then
  `jr`, `RiskBudgetingResult → prb` then `jr`, and the abstract default
  `RiskJuMPOptimisationResult → jr`. This is the shape the report's proposal actually
  describes.
- **Shape C — computed (2 sites).** `JuMPOptimisationResult.w` reaches a nested,
  receiver-rooted path (`sol.w`) with a scalar-or-broadcast branch; `MultiPeriodPredictionResult`
  broadcasts `getfield` over an inner vector. The report called these "genuine depth, leave
  alone."

Three structural facts ruled out the obvious designs:

- **A `@propagatable` field tag cannot express the targets.** `@propagatable` is the
  struct macro for *estimators*; it always emits `factory(x, args...)` (identity when
  untagged) and `port_opt_view` (ADR 0002, 0010, 0012). The delegation targets are
  `@concrete struct` *Result* types, which per the glossary are "never callable" and have
  no factory. Folding a tag into `@propagatable` would emit unwanted factory/view methods
  on results.
- **Two targets are not struct fields at all.** `RiskJuMPOptimisationResult` is an
  *abstract supertype* default (shared by every simple `jr`-embedding subtype), and
  `JuMPOptimisationResult.w` reaches a *nested path*, not a top-level field. Neither can be
  a field tag.
- **`propertynames` is overridden nowhere.** Every forwarded/aliased property is reachable
  via dot-access but invisible to `propertynames`, tab-completion, and introspection. ADR
  0012 already recorded this gap (`hasproperty(hop, :mu)` is `false` despite `hop.mu`
  working) and noted that `hasproperty` guards are therefore forwarding-blind.

## Decision

Introduce a **standalone** macro `@forward_properties T begin … end`, placed adjacent to
the struct (not a `@propagatable` field tag), that generates the `getproperty` +
`propertynames` pair from explicit-marker rules:

All names are written as **bare identifiers** (never quoted) — the macro reads them as
symbols from the unevaluated argument expressions. Every marker names its source via a
**locator** — a bare name `a` (the field `a` of the receiver) or a **dotted path**
`a.b.c` (the nested, receiver-rooted path `obj.a.b.c`, arbitrary depth). The dotted path is
a single argument, so it never collides with a marker's trailing names. **Nesting is simply
more dots.** Locators are the single point of nesting and are spelled identically in all
three markers. (This is the idiomatic Julia `@forward Foo.field` style, e.g. Lazy.jl.)

- `forward(loc)` — forward all properties of the value at `loc` (runtime
  `propertynames` of the walked value). `forward(pr)`; nested `forward(rr.jr)`.
- `forward(loc, names...)` — forward only the named subset. `forward(pe, me, ce)`;
  nested `forward(a.b, me, ce)`.
- `alias(exposed, loc)` — expose `exposed` as the value at `loc` (renaming).
  `alias(f_me, f_pe.me)`; nested `alias(x, a.b.me)`.
- `compute(exposed, …)` — expose `exposed` via arbitrary logic. Two forms:
  - **Path form** `compute(w, sol.w; broadcast)` — a dotted locator (depth ≥ 2; a depth-1
    "compute" is just `alias`); `broadcast` (path form only) maps the remainder of the path
    from the first vector-valued node onward (covers `JuMPOptimisationResult.w`, where
    `sol` may be a vector of results).
  - **Function form** `compute(w, obj -> …)` — an anonymous function is the general escape
    hatch; it owns its body entirely, so any scalar/vector handling lives inside it and
    `broadcast` does not apply. (A lambda is required rather than a bare/module-qualified
    function name, since the latter would be ambiguous with a dotted path.)
- `swap(field, …)` — *override the value of an existing field* (the one marker that may
  name a real field). Same two forms as `compute` plus a bare-name locator: `swap(L, M)`
  (locator), `swap(x, a.b.c)` (dotted path), `swap(L, obj -> …)` (function). Added when
  `Regression` — whose hand-written `getproperty` returns the loadings `M` for `:L` when
  `L` is unset — could not be expressed by `forward`/`alias`/`compute`: those resolve
  *after* the own-field check, so a rule on a real field (`:L`) is dead code. `swap` is the
  field-overriding counterpart and resolves *before* the field check. `T` may be a
  parametric/`UnionAll` signature so a `swap` can be specialised per type parameter:
  `@forward_properties Regression{<:Any, Nothing, <:Any} begin swap(L, M) end` (the
  matrix-`L` case needs no rule — default field access already returns the stored `L`).

**Nested-path traversal errors name the path.** Because a locator is known statically at
macro-expansion time, the generated walk for a depth-≥2 path checks each *intermediate* hop
and, when one is `nothing` (an optional field not yet populated — the common case), throws a
new `PropertyPathError <: PortfolioOptimisersError` (single `msg` field, like its siblings)
whose message names the receiver type, the full declared path, and the exact node that broke
— e.g.
``PropertyPathError: cannot descend path `sol.w` on `JuMPOptimisationResult`: intermediate `sol` is `nothing` ``.
The same walk (and the same error) backs both `getproperty` (fetching) and `propertynames`
(enumerating, for wildcard `forward`), so a half-built tree fails identically either way. A
genuinely wrong field name in a declared path still surfaces with the path string for
context; depth-1 locators have no intermediate and behave as today.

The three markers form a power hierarchy along two axes — *name mapping* (identity →
rename → arbitrary) and, for `forward`, *name selection* (enumerated subset → wildcard over
all of the located value's `propertynames`). `forward(loc, names...)` is exactly
`alias(n, loc.n)` repeated; `alias` adds renaming to the enumerated case but cannot
express `forward(loc)`'s runtime wildcard. The markers are kept separate rather than
collapsed into a single `forward` taking rename-pairs: one marker = one intent keeps
`forward` meaning "same-name pass-through", flags every rename with the word `alias`, and
fixes the `alias(exposed, loc)` argument direction positionally — avoiding a scoped
`=>` whose direction would have to be memorised (the same ambiguity that ruled out the
top-level arrow grammar). `forward` and `alias` stay purely structural; `compute` and `swap` are the
places arbitrary logic lives, keeping the other two trivially analyzable and
future-proofing the macro against computed properties that do not yet exist.

Block order is fall-through order, split into **two precedence tiers** keyed on whether a
rule adds a new name or overrides an existing field. The generated `getproperty` applies
`swap` rules **first** (they intentionally shadow the field check, since their job is to
override a field's value), then checks `fieldnames(T)` (not `propertynames`, which becomes
circular once overridden), then each `forward`/`alias`/`compute` rule with
**first-match-wins**, then a `getfield(obj, sym)` terminal (standard "no field" error on the
wrapper's own type). Within each tier, declaration order is preserved. The generated
`propertynames` unions own fieldnames with all forwarded/subset/alias/compute/swap exposed
names, deduped (a swapped name is already a field, so it dedups away).

The `swap` locator forms read through `getfield` and are recursion-safe. The `swap`
**function form** is a footgun in one narrow way: because the swap branch precedes the field
check, dot-access to the swapped field *inside the lambda* re-enters `getproperty` and
recurses (`StackOverflowError`) — the lambda must read the swapped field via
`getfield(obj, :field)` (the idiom the hand-written `Regression` body already used). Reading
*other* fields with dot-access is safe (they resolve via the field check). This is loud
(immediate overflow on first call), not silent, and is documented on the macro.

Covers Shapes A, B, and C (~13 sites). The abstract `RiskJuMPOptimisationResult` default
stays hand-written (no struct to attach to).

## Considered options

- **`@gprop` field tag inside `@propagatable`** — rejected: category mismatch (emits
  factory/view on results) and cannot express the abstract-supertype or nested-path
  targets.
- **The report's detached `@delegate_properties T => child` with a terse arrow DSL** —
  refined rather than adopted: a single overloaded `=>` would have to mean "from this
  child", "exposed ← path", and "rename" simultaneously, which is ambiguous. Explicit
  markers (`forward`/`alias`/`compute`) name each intent.
- **Delegation-only (3 concrete Shape-B structs) or aliasing-only (8 Shape-A sites)** —
  rejected in favour of one macro covering both, lifting leverage from 3 to ~13 sites; the
  8 byte-identical aliasing bodies are the strongest dedup case in the candidate.
- **`getproperty` only, no `propertynames`** — rejected: keeping the pair in sync is the
  macro's main correctness value, and it closes the ADR 0012 forwarding-blindness gap.

## Consequences

Emitting `propertynames` is the one **behavioural** change (everything else is
refactor-equivalent). Because nothing overrides `propertynames` today,
`propertynames ≡ fieldnames` everywhere, so the migration steps below are individually
low-risk — but they must be done, or the new overrides leak.

- **`@define_pretty_show` must iterate `fieldnames`, not `propertynames`**
  (`src/01_Base.jl:981`). This is a no-op for all current types (the two are identical
  today) and is the guard that keeps the flattened forwarded surface out of `show` — without
  it, each child would print both nested under its field and flattened at the top level. Its
  internal `hasproperty(obj, field)` guard (`src/01_Base.jl:993`) becomes always-true so
  should be changed to `hasfield`.
- **`hasproperty` call sites must be reclassified.** Field-existence tests
  (`!hasproperty(cv, :rng)` in NestedClustered/Stacking, `hasproperty(cv, :shuffle)` in
  CrossValidation) should become `hasfield` — they mean "does this struct really have this
  field". Capability/duck-typing checks on the delegation targets
  (`hasproperty(res, :fees|:pr|:jr)` in `src/20_Optimisation/01_Base_Optimisation.jl:1077,1124`)
  may start matching *forwarded* names once `propertynames` is emitted; their branch logic
  must be re-derived (and may collapse, since forwarding makes the property uniformly
  reachable), not mechanically swapped.
- **`setdiff(propertynames(x), (…))` reconstruction helpers must be audited**
  (`JuMPOptimiser.jl:1047`, `09_JuMPConstraints/01_Returns_and_ObjectiveFunctions.jl:345`,
  `19_RiskMeasures/27_RiskMeasureTools.jl` ×4). None operate on the 13 targets today; the
  rule is that no delegation target may flow into one, or it would try to rebuild using
  forwarded names as if they were fields. The 8 Shape-A types are estimators, so they are
  the set to watch.
- **Introspection/tab-completion now expose forwarded properties** on the converted types —
  the intended improvement, and the resolution of the ADR 0012 gap.

## Amendment (2026-08-24)

The `hasfield` guard inside `@define_pretty_show` is **removed**, not kept.

The Consequences above predicted that the guard "becomes always-true". It did, and an
always-true guard is dead code: `fields` is `fieldnames(typeof(obj))`, so
`hasfield(typeof(obj), field)` holds for every element of it. Its `continue` arm was
unreachable, and the `# Algorithm` of the macro stated a behaviour — a field the object does
not have is skipped — that no input can produce. Both are gone. The loop reads
`val = getproperty(obj, field)` directly.

The decision of this ADR is unchanged. `@define_pretty_show` still iterates `fieldnames`, so
the flattened forwarded surface still stays out of `show`. Reading the value through
`getproperty` rather than `getfield` is also unchanged, and it is what makes a `swap` rule
visible: `@forward_properties` emits its `swap` branches ahead of the own-field check, so a
rule that swaps a real field name prints the swapped value.

Found by #439, the condition 2 and 3 sweep of `src/01_Base.jl`, which needed every line of the
macro to be reachable from a test.

## Amendment (2026-10-07)

**The macro also generates `ConstructionBase.setproperties`.** The decision above names the
`getproperty` and `propertynames` pair alone.
[#1497](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1497) found that a search
could not tune a field of a prior that this macro decorates. `Accessors.set` rebuilds each struct
on a lens path through `ConstructionBase.setproperties`, and the default method refuses a type
whose `propertynames` differ from its `fieldnames`. It also reads every field through
`getproperty`, so a `swap(L, M)` rule would write `M` into an unset `L`.

The generated method calls `forward_setproperties`. It reads every field with `getfield`, merges
the patch over them, and calls the keyword constructor of `T`, so every guard of the constructor
runs on the new value. A patch that names a forwarded, aliased or computed property is refused,
because such a property is a view of a nested value with no storage of its own. The message names
the path to set instead, where the property has one, and the fields otherwise. The keyword constructor of a decorated type must take every field by its
field name.

## Amendment (2026-11-12)

**A `forward` rule can mark its names as settable.** The amendment of 2026-10-07 refuses a patch
that names a forwarded property. Item 4 of the ruling of Decision 3 of map
[#1213](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1213) needs an exception.
A search sets a key such as `"opt.wb"` through `Accessors.set`, and `wb` is a slot of a nested
part of `JuMPOptimiser`.
[#2037](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/2037) builds it.

`forward(loc; settable)` and `forward(loc, names...; settable)` forward as before. The patch of
the generated `ConstructionBase.setproperties` passes a name of such a rule to the keyword
constructor as a flat keyword, beside the fields. The type that marks a name takes that name as a
keyword. The keyword replaces that slot of the nested part that the constructor is given, and its
default leaves the slot as it is. `rebuild_struct` calls the constructor in the same way, so
the two give the same value. A rule without the option keeps the refusal, so no type that the
macro decorated before this amendment changes.

The rule that reads a name decides whether a patch can set it. Each rule other than `swap` gives
the pair of its path and its `settable` flag, in the order of the `getproperty` branches. A name
that an earlier rule reads stays refused, even when a later `settable` rule forwards it.

An option on a rule that it does not apply to is now refused. Before this amendment, the macro
ignored `broadcast` on `forward` and `alias` with no error. `broadcast` applies to `compute` and
`swap`, and `settable` applies to `forward`.

## Amendment (2026-11-16)

**The macro also generates `Base.hasproperty`, and a read through it allocates nothing.** The
decision above names the `getproperty` and `propertynames` pair, and the amendment of 2026-10-07
adds `ConstructionBase.setproperties`. `12058f0c14`
([#2101](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/2101)) adds the fourth
method. Before it, every read of a forwarded property allocated, for two causes:

- A `forward(loc)` rule tested `sym in propertynames(child)`. The `propertynames` of a decorated
  child removes its duplicate names with `unique` on each call, and allocates.
- The own-field check read `fieldnames(T)`. For a signature with a free parameter, such as
  `Regression{<:Any, Nothing, …}`, that expression builds a new `UnionAll` on each call.

The generated `hasproperty` gives `true` for the names of `propertynames`, from one branch for
each rule, and no other name. A `forward(loc)` rule asks `hasproperty` of the child, so a chain of
`forward` rules runs one generated method for each link. The three methods read the field names of
the bare name of `T`, a constant with the same field names. `propertynames` keeps its answer.

Measured on a `git archive` copy of the parent of `12058f0c14`, against `dev`: `retcode` of a
`RiskBudgetingResult` allocated 3 872 bytes, of a `FactorRiskContributionResult` 6 688 bytes,
`re.M` of a `Regression` 2 304 bytes, and `mu` of a `HighOrderPrior` 1 664 bytes. Each allocates
0 bytes after it.

The hand-written `getproperty` and `propertynames` of `RJR_NRJR`, the results of `MeanRisk` and
`NearOptimalCentering`, take the same `hasproperty` method by hand, because the macro does not
decorate a `Union`. Before, `hasproperty(res, :retcode)` on a `MeanRiskResult` allocated 5 648
bytes (R55 of [#1817](https://github.com/dcelisgarza/PortfolioOptimisers.jl/pull/1817)).

## Amendment (2026-11-16, second)

**The macro also generates a method of `forward_source`, and the refusal of a patch names the
path that stores the value.** The amendment of 2026-10-07 makes the refusal of a forwarded name
name the path of the rule. The `source` of that rule was a closure inside the generated
`setproperties`, so it gave the path as a string, and it could not ask the source of a nested
value. [#2129](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/2129) found that the
path can have no storage either. `HighOrderPrior` forwards every property of `pr`, and `f_mu` is a
computed property of the `LowOrderPrior` at `pr`. The refusal said "Set `pr.f_mu`", and that set
gave a second refusal.

The `source_branches` now make the method `forward_source(x::T, sym::Symbol)`, beside
`getproperty`, `propertynames`, `hasproperty` and `setproperties`. It gives the same pair as the
closure did, with the path as a tuple of symbols. A fallback method gives `nothing` for a type
that the macro does not decorate. `forward_storage_path` walks the path one hop at a time, and it
asks `forward_source` of the value at each hop. It puts the path of a rule in place of a hop that
the rule reads. The walk ends at a stored field, at a settable name, or at a computed property.
The refusal names the stored path, or the computed property with its type and the fields of that
type.

The fifth method replaces the closure, so the macro makes no more branches. The method that
`setproperties` calls drops its `source` argument. No public name changes, and every patch that
the macro let through before still goes through.

## Amendment (2026-11-16, third)

**Every JuMP optimisation Result takes its property methods from the macro, and no method is
written on `RJR_NRJR`.** The first amendment of 2026-11-16 kept the hand-written `getproperty`,
`propertynames` and `hasproperty` of `RJR_NRJR`, because the macro does not decorate a `Union`.
Those three methods had no `ConstructionBase.setproperties` beside them. So `Accessors.set` and
`Accessors.setproperties` refused every patch of a `MeanRiskResult` or a
`NearOptimalCenteringResult` with the generic error of ConstructionBase, also a patch of a field
([#2131](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/2131)).

A census of every concrete type of the library found 27 types that change their property access.
These two Results were the only types with no `setproperties` method. The three other subtypes of
the union, `RiskBudgetingResult`, `RelaxedRiskBudgetingResult` and
`FactorRiskContributionResult`, already used the macro.

The two Results now declare `@forward_properties T begin forward(jr) end`, and the three methods
on the union are gone. One mechanism serves all five subtypes. The macro gives the fifth and the
sixth methods too, so a patch of a field rebuilds the Result through its keyword constructor, and
the refusal of a forwarded name names the path that stores it. A new subtype of
`RiskJuMPOptimisationResult` or `NonRiskJuMPOptimisationResult` declares its own block. The
amendment of 2026-08-16 of ADR 0011 put the default on the union so that a new subtype forwards
without a declaration. That reason does not hold against a type that cannot take a patch, and a
subtype with no block fails at its first read of `res.w`.

No library code patches a Result through `Accessors`. The library rebuilds a Result through
`rebuild_struct`, and the search, the pipe routing and the pipeline context write estimators
alone.

## Amendment (2026-11-19)

**A `derived` rule marks a field that the inner constructor computes, and a patch leaves it
out.** The amendment of 2026-10-07 states the contract of `forward_setproperties`: the keyword
constructor takes every field by its name. Two decorated types broke it
([#2194](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/2194)), so
`Accessors.@set` on any of their fields, and `rebuild_struct`, raised a `MethodError`.
`PipelineResult` had no keyword constructor, and it now has one.
`MultiPeriodPredictionResult` computes `mrd` from `pred` in its inner constructor, so its
keyword constructor does not take `mrd`.

`derived(fields...)` names such fields. The macro makes one more method for each type that it
decorates, `forward_derived(::Type{<:T})`, which gives the names, or `()` when the block has no
`derived` rule. The fallback for a type that the macro does not decorate gives `()` too. `forward_setproperties`
and `rebuild_struct` leave a derived field out of the call of the keyword constructor, so the
constructor computes it again from the new values. `assert_not_derived` refuses a patch that names
a derived field, and the message names the fields to set. The rule adds no name and changes no
read.

The other form, a keyword `mrd = nothing` that the constructor ignores, would accept a patch of
`mrd` and drop it with no error. So that form loses. A type with no derived field gives the empty
tuple, the compiler folds the removal, and `rebuild_struct` allocates what it allocated before.
A census in `test/test_05_tools.jl` compares the fields of every type with a generated
`setproperties` method against the keyword names of its constructors.
