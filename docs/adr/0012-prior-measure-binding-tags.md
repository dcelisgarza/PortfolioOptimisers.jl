---
status: proposed
---

# Deepen the prior→measure moment binding with `@pprop`/`@cprop` tags

## Context

Architecture-review candidate 1 (`docs/architecture-review-20260615-025911.html`). Each
risk measure hand-writes a `factory(r, pr::AbstractPriorResult)` body that re-lists which
moment fields it pulls from the prior — 46 such bodies across 13 files, driven by 162
`nothing_scalar_array_selector` calls. The field→prior map is invisible at the struct and
re-encoded in every body; a forgotten line silently feeds the wrong moment into an
optimiser. This is the same shallow copy-paste family ADR 0011 eliminated on the result
side, still hand-written on the moment side.

Unlike the result side, the natural fix is *not* a new bespoke macro: the library already
has `@propagatable` with the orthogonal `@fprop` (factory/value propagation) and `@vprop`
(view/index propagation) tags (ADR 0002, ADR 0010). Prior-binding is a third propagation
axis and belongs in the same machinery.

The investigation surfaced several structural facts that shape the design:

- **`@fprop` does not cover prior-binding.** `@fprop` emits `f = _factory_child(x.f, args...)`
  — it threads the whole runtime value down and dispatches on the *field's own value type*.
  Prior-binding must reach into a **second object `pr` by the field's name** (`sel(r.f, pr.f)`),
  which `_factory_child` cannot express.
- **Leaves select; composites recurse.** Leaf measures (`Variance`, the XatRisk family, …)
  do `f = sel(r.f, pr.f)`. Composites (`VarianceSkewKurtosis`) instead recurse
  `factory(r.child, pr)` — which *is* the `@fprop` shape, since risk measures are
  `AbstractEstimator`s and `_factory_child` already recurses them.
- **The prior method shadows the general method.** `factory(x, pr::AbstractPriorResult, args...)`
  is more specific than `factory(x, args...)`, so when called with a prior the general
  `@fprop` method never runs. Therefore the prior method must *itself* thread the `@fprop`
  children — prior-binding cannot be decoupled from `@fprop` knowledge, ruling out a
  cleanly separate macro.
- **`ObsWeights` factories initially rode on `@fprop`, but the two semantics collide.** A
  large existing `factory(x, w::ObsWeights)` family fills `nothing`-slots via
  `_factory_child(::Nothing, ::ObsWeights) = w`. Tagging a weights field `@fprop` reused
  this for free — but the same `_factory_child(::Nothing, ::ObsWeights)` method also fires
  on an `@fprop`-tagged **sub-estimator** field that happens to be `nothing` (e.g.
  `SimpleVariance(; me = nothing)`), wrongly setting it to the weights vector. Weights-
  replace (`nothing → w`) and estimator-recurse (`nothing → nothing`) are indistinguishable
  at runtime on that one dispatch, so they need separate tags (see `@wprop`, decision 0).
- **Solvers/uncertainty sets are a *third* source axis.** `slv`/`ucs` are selected against
  a **threaded optimiser value** (`factory(r, pr, slv)`), not against `pr` — there is no
  `pr.slv`. They also carry different selectors (`solver_selector`/`ucs_selector`) and,
  for solvers, a different both-`nothing` policy (throw, not `nothing`). And because
  `Solver <: AbstractEstimator` and `UcSE_UcS` are estimators/results, they already sit in
  `_factory_child`'s recursing union, so `@fprop` would *recurse* into them rather than
  select.
- **Prior results forward but do not expose `propertynames`.** `HighOrderPrior` embeds a
  `LowOrderPrior` and forwards `mu`/`sigma`/`w` via a custom `getproperty`, but
  `propertynames` is not overloaded, so `hasproperty(hop, :mu)` is `false` despite `hop.mu`
  working. A `hasproperty` guard is therefore forwarding-blind and unsafe.

## Decision

The `@propagatable` tag family grows from two axes to five. The existing tags name a
*mechanism* (`@fprop` = factory threading, `@vprop` = view slicing); `@pprop`/`@cprop` name
a *source* (`@pprop` = prior, `@cprop` = context); and `@wprop` splits the weights-replace
mechanism out of `@fprop` (decision 0), because reusing `@fprop`'s `nothing`-handling for
weights silently corrupts nothing-valued sub-estimator fields.

### 0. `@wprop` — observation-weights replacement (split from `@fprop`)

`@wprop f` declares that field `f` is an observation-weights slot: in the general (and
prior) factory it is **replaced** by an incoming `ObsWeights` argument via
`_wprop(getfield(x, :f), args...)` (`_wprop(::Any, w::ObsWeights, …) = w`, else the field
unchanged). `@fprop` keeps only the estimator-recursion role, and the
`_factory_child(::Nothing, w::ObsWeights) = w` shortcut is removed — so an `@fprop` sub-
estimator that is `nothing` now correctly stays `nothing`. Every weights field previously
tagged `@fprop w` is re-tagged `@wprop w`. `@wprop` may stack with `@pprop` (`@pprop @wprop w`
gives a measure both a prior factory and an `ObsWeights` factory; `@pprop` wins in the prior
method).

### 1. `@pprop` — prior selection (third tag)

`@pprop f` declares that field `f` is selected from the same-named field on the prior
result. Presence of any `@pprop` field emits a method
`factory(x, pr::AbstractPriorResult, args...)` that processes **both** tag families:

- `@pprop f` → `f = sel(getfield(x, :f), getproperty(pr, :f))`
- `@fprop`-only `g` → `g = _factory_child(getfield(x, :g), pr, args...)`
- `@wprop`-only `w` → `w = _wprop(getfield(x, :w), args...)`
- a field tagged both `@pprop @fprop`/`@pprop @wprop` → `@pprop` wins **in this method**.

The general `factory(x, args...)` is unchanged and remains the `ObsWeights`-direct factory
(now via `@wprop`). So `@pprop @wprop w` on a measure yields *both* a prior factory and an
`ObsWeights` factory from one declaration.

`@pprop` keys strictly by name (`r.f` ↔ `pr.f`); there is no `@pprop f => g` remap. Every
current measure is same-name; a future differing-name measure opts out.

### 2. `@cprop` — threaded-context selection (fourth tag)

`@cprop f` declares that field `f` is selected against a **threaded** optimiser value found
by type in the variadic tail: `f = sel(getfield(x, :f), _ctx(args...))`. `_ctx(args...)`
type-filters the `args` tuple for the lone context-typed argument (`Slv_VecSlv`); it is a
tuple scan the compiler unrolls, so it is type-stable and allocation-free. This is a
distinct axis from `@pprop` (source is `pr.f` by name) and `@fprop` (recurse into the
field's own value).

### 3. One selector wrapper `sel`, dispatching internally

`@pprop` and `@cprop` both emit the same `sel(rv, pv)` wrapper, which dispatches on operand
types to the existing leaf selectors (`nothing_scalar_array_selector`, `solver_selector`,
`ucs_selector`) and inlines to zero cost. The leaf selectors keep their names and call
sites; **nothing is renamed**. The two tags differ only in how operand 2 is *sourced*
(prior-by-name vs threaded-by-type), not in which selector ultimately runs.

### 4. Unconditional select; no `hasproperty` guard

`@pprop f` emits a plain `getproperty(pr, :f)` with no guard. This is safe because
`mu`/`sigma`/`chol`/`w` are provided by both concrete prior results (`LowOrderPrior`
directly; `HighOrderPrior` by forwarding). The forwarding-blind `hasproperty` landmine is
avoided entirely by opting out the few measures whose fields are not universal.

### 5. Opt-outs stay hand-written

The macro covers the ~29 (of ~36) pure-select and select-plus-`@fprop`-child measures
(`Variance`, `StandardDeviation`, the VaR/CVaR/EVaR/RVaR/PNVaR + Range/Drawdown families,
`MAD`, `ThirdCentralMoment`, `MeanReturn`, `ValueatRisk`/`…Range` via `@pprop w` + `@fprop alg`).
These opt out and keep hand-written `factory` methods:

- **Derived-`w` threading** (`alg`/`ve` rebuilt with the *locally-selected* `w`, not `pr`):
  `LowOrderMoment`, `HighOrderMoment`, `Skewness`.
- **Prior-type divergence** (`kt`/`sk` exist only on `HighOrderPrior`):
  `Kurtosis`, `NegativeSkewness`.
- **Two-context complexity / sole `ucs` user**: `UncertaintySetVariance`.

`VarianceSkewKurtosis` is a composite and uses `@fprop` (recurse `pr` into sub-measures),
not `@pprop`.

### 6. Tag invariants

- `@pprop` and `@cprop` are mutually exclusive on a field — a field's selected value comes
  from exactly one source (the prior, or the threaded context), never both.
- `@cprop` fields are not also `@fprop` (a solver field is selected, not recursed).
- `@fprop` (recurse sub-estimator) and `@wprop` (replace weights) are mutually exclusive on a
  field — a field is either a sub-estimator or a weights slot.
- The legal stacks are `@pprop @fprop` and `@pprop @wprop` (prior factory + recurse/weights
  factory), resolved `@pprop`-wins in the prior method (decision 1).

## Considered options

- **A new bespoke macro for prior-binding.** Rejected: the prior method shadows the general
  `@fprop` method, so it must thread `@fprop` children itself — a separate macro would just
  re-read `@propagatable`'s tags, gaining nothing and risking drift.
- **`@fprop` for `slv`/`ucs`.** Rejected: `Solver`/`UcS` are estimators already recursed by
  `_factory_child`; `@fprop` would rebuild them instead of selecting, and the prior method
  threads `pr` first, not the solver.
- **Renaming `nothing_scalar_array_selector` → `rm_pr_selector`.** Rejected as unnecessary:
  the wrapper (decision 3) gives a single emitted name without touching 162 call sites or
  the sibling selectors, mirroring ADR 0010's exclusion of leaf slicers from the
  `port_opt_view` rename.
- **`hasproperty(pr, :f)` guard to fold `Kurtosis` in.** Rejected for now: forwarding-blind
  until prior `propertynames` is made consistent (architecture-review candidate 2). Revisit
  then; until then `Kurtosis`/`NegativeSkewness` opt out.
- **Type-directed `@cprop` covering `ucs` too.** Deferred: `UncertaintySetVariance` is the
  sole `ucs` risk measure and the only case threading two context types, so it opts out
  rather than complicating `_ctx`.

## Consequences

- The field→prior map becomes one declared surface per measure; the silent-wrong-moment bug
  class disappears for the ~29 migrated measures, and the test surface shrinks to one
  binding instead of 46 bodies.
- **The solver both-`nothing` "cannot solve" throw is dropped, and this is safe.**
  `JuMPOptimiser` requires `slv` (a keyword with no default), so every JuMP optimiser
  carries a solver and the threaded `slv` is always present in model assembly — the
  `solver_selector(::Nothing, ::Nothing)` throw is unreachable via the pipeline. The shared
  `sel(::Nothing, ::Nothing)` returning `nothing` (the moment policy) therefore changes no
  reachable behaviour. A measure that genuinely needs the early error keeps a hand-written
  factory.
- `@pprop` silently depends on the same-name `r.f`↔`pr.f` convention and on
  `mu`/`sigma`/`chol`/`w` remaining universal across all `AbstractPriorResult`s; a new prior
  result lacking one, or a measure needing a differing-name binding, must opt out.
- Amends ADR 0002/0010: the `@propagatable` family grows from two tags to five
  (`@fprop`/`@wprop`/`@vprop`/`@pprop`/`@cprop`); exports and the use-outside-body error
  stubs extend accordingly. Splitting `@wprop` out of `@fprop` also fixes a latent bug where
  an `@fprop`-tagged sub-estimator field that was `nothing` (e.g. `SimpleVariance(; me =
  nothing)`, reachable via `factory(SecondMoment(), w)`) was wrongly replaced by the
  incoming weights and threw a constructor error.

## Verification

- `@inferred factory(r, pr)` is type-stable for every migrated measure — the generated
  prior method and the `sel` wrapper must infer concretely (mirrors ADR 0002 §Verification).
- A migrated measure's `factory(r, pr)` is observationally identical to its previous
  hand-written body (same fields selected, same fallbacks).
- `@pprop @wprop w` measures expose both `factory(r, pr)` and `factory(r, ::ObsWeights)`.
- `@fprop`-tagged sub-estimator fields that are `nothing` survive `factory(x, ::ObsWeights)`
  unchanged (regression test for the weights/estimator collision).
- Full test suite green before merge (mirrors ADR 0011).

## Rollout

Staged, each independently reviewable (mirrors ADR 0010):

1. **Machinery, no struct converted** — add the `sel` wrapper, `@pprop`/`@cprop` emission in
   `@propagatable`, the `_ctx` helper, exports, and the use-outside-body error stubs.
2. **Prior measures** — convert the ~29 `@pprop` (+ `@fprop` child) measures; delete their
   hand-written prior `factory` bodies.
3. **Context measures** — convert the `slv` families (Entropic/Relativistic/PowerNorm) to
   `@cprop`; drop the dead `slv`-first defensive factory methods.
4. **`@wprop` split** — fold the remaining eligible estimators into `@fprop`/`@vprop`, then
   add the `@wprop` tag and `_wprop` helper, remove the
   `_factory_child(::Nothing, ::ObsWeights)` shortcut, and re-tag every weights field
   `@fprop w` → `@wprop w`, fixing the nothing-sub-estimator collision.
