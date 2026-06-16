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
