---
status: accepted
---

# Unify the `*_view` family under `port_opt_view`; split propagation tags into `@fprop`/`@vprop`

## Context

The library performs two structurally identical kinds of propagation down a composed
struct tree:

- **Factory propagation** (ADR 0001/0002): push runtime *values* (observation weights,
  prior results, previous weights, solvers) into a struct tree, returning a new struct of
  the same type. Driven by `@propagatable` + `@prop` tags.
- **View propagation**: push an index *selection* (an asset subset, or — for returns data
  — an observation window) into a struct tree, returning a new struct of the same type
  with every data-bearing field and composed child consistently sub-selected. Used by
  Subset Resampling, Nested Clustered optimisation, Cross-Validation, and windowed moment
  estimators. (See the **View** entry in `CONTEXT.md`.)

View propagation was implemented as ~25 separate, domain-named function families —
`moment_view`, `opt_view`, `prior_view`, `risk_measure_view`, `tracking_view`,
`turnover_view`, `fees_view`, `regression_view`, … — totalling ~100+ hand-written
methods. Each recurses into composed children by calling the *correctly-named* sibling
family (`moment_view` into `moment_view`, but `opt_view` into `prior_view`,
`risk_measure_view`, etc.). This is the same pure-propagation shape `factory` had before
ADR 0001, but it never received a macro, so the bodies are hand-maintained and a forgotten
field is a silent wrong slice.

The obvious extension — "make `@propagatable` also emit a view method from the existing
`@prop` tags" — does not work, because the factory-relevant and view-relevant field sets
genuinely diverge. On `MeanRisk`: `fb` is factory-propagated but view-passthrough; `wi` is
view-sliced but factory-passthrough; `opt`/`r` are both; `obj` is neither. A single tag
cannot express this.

## Decisions

### 1. Two orthogonal propagation tags: `@fprop` and `@vprop`

**Decision:** replace the single `@prop` with two orthogonal tags inside a `@propagatable`
body — `@fprop` (factory propagation) and `@vprop` (view propagation). A field that
participates in both stacks them: `@fprop @vprop opt`. There is no combined `@fvprop`.

**Why:** the two field sets diverge in both directions (see `MeanRisk` above), so they must
be expressible independently. Stacking two tags reuses the parser's existing ability to
unwrap nested `:macrocall` nodes (it already handles `Core.@doc` wrapping `@prop`), and
avoids a combinatorial zoo of combined-tag macros if a third propagation axis ever appears.

**Consequence:** `@prop` is renamed to `@fprop` wholesale — no alias. The export list drops
`@prop` and adds `@fprop`, `@vprop`. A second error-stub macro (`@vprop`) is defined
alongside `@fprop` for use-outside-body diagnostics, mirroring decision 0002§5.

### 2. One unified view function: `port_opt_view(x, i, args...)`

**Decision:** collapse every `*_view` family into a single generic function
`port_opt_view(x, i, args...)`. The second positional argument `i` is the index selection;
the variadic tail carries optional threaded context (the returns matrix `X`, for the JuMP
families).

**Why:** the macro emits *one* call name per tagged field and trusts multiple dispatch on
the field's value type to choose the behaviour — recurse (composed child), slice (data
array), or pass through (scalar/`nothing`). This is only possible if the recursion target
has a single name; with the old per-family names the macro could not know whether a given
child needs `moment_view` or `risk_measure_view`.

**Consequence:** all `port_opt_view` methods must accept the variadic tail `(x, i, args...)`,
even the families (moments, priors) that never read `X`. Otherwise a macro-threaded
`port_opt_view(child, i, X)` would hit a fixed-arity `(x, i)` method and `MethodError`.
This is a forced, pervasive signature normalisation.

### 3. The macro stays dumb; complex structs opt out

**Decision:** `@vprop` emits exactly `port_opt_view(x.field, i, args...)` for each tagged
field and passes the rest through. No conditionals, no per-field slicing overrides, no
preamble hooks. A `port_opt_view` method is emitted only when ≥1 `@vprop` field is present
(`factory` emission is unchanged beyond the rename).

**Why:** several view bodies carry logic that cannot be mechanically generated —
`JuMPOptimiser` rebinds `X` from `opt.pe.X` and preserves field aliasing
(`if opt.smtx === opt.sgmtx …`); `Variance` has an `@argcheck` and slices `chol` on a
single dimension (`view(r.chol, :, i)`) while `sigma` is symmetric (`view(x, i, i)`);
`ReturnsResult` slices on three axes. Building a macro DSL to cover these would absorb
complexity the macro should not own.

**Consequence:** these structs opt out of `@vprop` and keep a hand-written `port_opt_view`
(renamed and signature-normalised) — exactly as `MeanRisk` already hand-writes its
`factory`. The macro's leverage is the large tail of pure-shape structs (the ~26 moment
estimators, simple priors, simple risk measures).

### 4. Leaf slicing stays a separate primitive; `port_opt_view` delegates

**Decision:** keep `nothing_scalar_array_view` as the "slice, never recurse" leaf primitive
(handling `AbstractVector`/`AbstractMatrix`/vector-of-arrays/passthrough). `port_opt_view`'s
universal fallback delegates to it: `port_opt_view(x, i, args...) = nothing_scalar_array_view(x, i)`.
`VecScalar` is lifted to its own first-class `port_opt_view(x::VecScalar, i, args...)` method.

**Why:** two clear roles — `port_opt_view` recurses into composed children and slices data;
`nothing_scalar_array_view` is the leaf that hand-written bodies call when they *know* a
field is data and must not recurse. The universal fallback also subsumes the per-family
identity fallbacks (`opt_view(::AbstractOptimisationEstimator,…)=opt`, etc.), which are
then deleted.

**Consequence:** the universal fallback must reach all abstract supertypes as passthrough;
the existing `AbstractBaseRiskMeasure` fixed-arity fallback must be normalised to
`(x, i, args...)` without colliding with the universal one.

## Alternatives considered

- **Extend `@prop` to emit both factory and view** (the architecture review's framing):
  rejected — the field sets diverge, so one tag cannot drive both.
- **A combined `@fvprop` tag:** rejected in favour of stacking — avoids a combined-tag
  combinatorial explosion.
- **A hook-driven macro** (per-field slicing dim, preamble for `X`-rebind): rejected —
  pushes bespoke complexity into the macro. Opt-out + hand-writing is more predictable.
- **Merge `nothing_scalar_array_view` into `port_opt_view` entirely:** rejected — loses the
  explicit "this is data, do not recurse" signal at hand-written call sites and risks
  behaviour changes across ~96 direct call sites.

## Rollout

Three staged PRs, each independently reviewable:

1. **Mechanical, behaviour-preserving** — rename every propagation-family `*_view` →
   `port_opt_view`, rename `@prop` → `@fprop` and update exports. Audit for any concrete
   type defining views in *two* families (which would collapse into duplicate
   `port_opt_view(::T, …)` definitions). Test suite must pass unchanged. Signature
   normalisation to `(x, i, args...)` is deferred to PR2 — in PR1 every call site keeps
   its exact arity, so no normalisation is needed yet.

   **Excluded from the rename** (kept under their own names):
   - Leaf/utility slicers that never recurse into a struct: `nothing_scalar_array_view`,
     `nothing_scalar_array_view_odd_order`, `nothing_scalar_array_getindex`,
     `dup_elim_sum_view`, `risk_measure_nothing_scalar_array_view`.
   - `returns_result_view` — a distinct user-facing multi-axis subsetting utility on
     `ReturnsResult` (`(rd, i)`, `(rd, i, j)`, `(rd, i, j, k=:)` for
     observation/asset/factor selection). It is never a recursion target inside another
     view body, and its positional `j`/`k` would clash with the normalised
     `(x, i, args...)` tail.
2. **Macro** — `@propagatable` gains `@vprop` emission; add the universal `port_opt_view`
   fallback, the `VecScalar` method, and leaf delegation. No struct converted yet.
3. **Dedup** — convert eligible pure-shape structs to `@vprop` and delete their
   hand-written `port_opt_view` bodies.

## Relationship to ADR 0002

Amends ADR 0002: decision 0002§1's `@prop` tag is renamed to `@fprop` and joined by the
orthogonal `@vprop`; the AST-handling and composition-order decisions (0002§2–§6) carry
over unchanged and now apply to both tags.
